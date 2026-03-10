"""
Door Session Store (v4.10).

Door Session을 관리하는 저장소.
여러 번의 /trigger 호출을 하나의 Door Session으로 통합 관리합니다.

v4.10 변경사항:
- Trigger 큐 연동: pending_trigger_count 추적
- notify_trigger_enqueued(): 큐에 등록 시 호출 → pending 카운터 증가 + last_trigger_at 갱신
- notify_trigger_processed(): 워커 처리 완료 시 호출 → pending 카운터 감소
- handle_close_signal(): pending > 0이면 finalize 거부 (큐 대기 중 CLOSE 방지)

v4.9 변경사항:
- 크로스 존 반환 처리 추가 (_handle_cross_zone_returns)
- Zone A에서 꺼낸 상품을 Zone B에 넣으면 Zone A에서 차감
- weight_tolerance 기본값 5g으로 변경

v4.8 변경사항:
- YAML 저장 백그라운드 비동기화 (ThreadPoolExecutor)
- finalize 후 새 OPEN 응답 지연 문제 해결
- shutdown() 메서드 추가 (스레드풀 정리)

v4.7 변경사항:
- handle_close_signal() 메서드 추가 (빠른 문 열고 닫기 대응)
- 첫 CLOSE → pending_close 상태, 두 번째 CLOSE → finalize 여부 결정

v4.5 변경사항:
- Callback deadlock 방지: Lock 해제 후 callback 실행 (deferred 패턴)
- GlobalSession max_duration 파라미터 추가
- cleanup_timed_out_sessions() 메서드 추가

v4.3 변경사항:
- GlobalDoorSession 추가: session_id="OPEN"/"CLOSE" 기반 문 상태 관리
- get_or_start_global_session(): OPEN 시 세션 생성/기존 유지
- finalize_global_session(): CLOSE 시 모든 zone 종료 및 결과 반환
- add_trigger_with_global(): trigger 시 GlobalSession 연동
- get_or_finalize()에서 GlobalSession 활성 시 타임아웃 무시

v4.2 변경사항:
- Copy-on-Write 패턴: Lock 내에서 데이터 복사, Lock 해제 후 YAML 저장
- 타임아웃 체크 로직 통합 (_check_timeout 메서드)

사용법:
    store = DoorSessionStore(
        yaml_dir="data/sessions",
        session_timeout=30.0,
        weight_tolerance=3.0,
    )

    # GlobalSession 기반 (v4.7 - handle_close_signal 사용)
    global_session = store.get_or_start_global_session()  # OPEN
    door_session = store.add_trigger_with_global(zone=1, result=trigger_result)
    is_ready, session = store.handle_close_signal()  # CLOSE
    if is_ready:
        result = store.finalize_global_session()

    # 기존 방식 (v4.2 하위 호환)
    door_session = store.add_trigger(zone=1, result=trigger_result)
    session, is_finalized = store.get_or_finalize(zone=1)
"""

import copy
import concurrent.futures
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .door_session import (
    DoorSession,
    TriggerResult,
    AggregatedProduct,
    CrossZoneReturn,
    UnmatchedReturn,
    generate_door_session_id,
)
from .global_door_session import GlobalDoorSession, generate_global_session_id
from .product_aggregator import ProductAggregator
from .yaml_persistence import YamlPersistence

logger = logging.getLogger(__name__)


@dataclass
class TimeoutCheckResult:
    """타임아웃 체크 결과."""
    is_timed_out: bool
    reason: str = ""  # "idle_timeout" | "max_duration" | ""


class DoorSessionStore:
    """
    Door Session 저장소 (v4.8).

    Zone별로 하나의 활성 Door Session을 관리합니다.
    여러 trigger가 발생해도 같은 Door Session에 통합됩니다.

    v4.8: YAML 저장 백그라운드 비동기화
    - finalize 후 새 OPEN 응답 지연 문제 해결
    - ThreadPoolExecutor로 YAML 저장 비동기 처리

    v4.3: GlobalDoorSession 지원
    - session_id="OPEN" → get_or_start_global_session()
    - session_id="CLOSE" → finalize_global_session()
    - GlobalSession 활성 시 타임아웃 무시

    타임아웃 시 자동으로 세션이 finalize되며,
    YAML 파일로 영속화됩니다.
    """

    def __init__(
        self,
        yaml_dir: str = "data/sessions",
        session_timeout: float = 30.0,
        weight_tolerance: float = 3.0,
        max_duration: float = 600.0,
        global_session_max_duration: float = 600.0,
        get_product_weight: Optional[Callable[[int], float]] = None,
        on_session_finalize: Optional[Callable[[int], None]] = None,
    ):
        """
        Initialize DoorSessionStore.

        Args:
            yaml_dir: YAML 저장 디렉토리
            session_timeout: 마지막 trigger 후 타임아웃 (초)
            weight_tolerance: 무게 매칭 허용 오차 (g)
            max_duration: 최대 세션 지속 시간 (초)
            global_session_max_duration: GlobalSession 최대 지속 시간 (초, v4.5)
            get_product_weight: product_id -> weight 조회 함수
            on_session_finalize: 세션 종료 시 콜백 (zone 전달, v4.4)
        """
        self._active_sessions: Dict[int, DoorSession] = {}  # zone -> session
        self._lock = threading.Lock()

        # v4.3: GlobalDoorSession
        self._global_session: Optional[GlobalDoorSession] = None

        self._session_timeout = session_timeout
        self._weight_tolerance = weight_tolerance
        self._max_duration = max_duration
        self._global_session_max_duration = global_session_max_duration  # v4.5
        self._get_product_weight = get_product_weight
        self._on_session_finalize = on_session_finalize  # v4.4

        # 컴포넌트 초기화
        self._persistence = YamlPersistence(base_dir=yaml_dir)
        self._aggregator = ProductAggregator(
            weight_tolerance=weight_tolerance,
            get_product_weight=get_product_weight,
        )

        # v4.10: 큐에 등록되었지만 아직 처리 안 된 trigger 수
        self._pending_trigger_count = 0

        # v4.8: YAML 저장용 스레드풀 (백그라운드 비동기 저장)
        self._yaml_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="yaml_save",
        )
        # v5.2: shutdown 상태 플래그 (race condition 방지)
        self._yaml_executor_shutdown = False

        logger.info(
            f"DoorSessionStore initialized: "
            f"timeout={session_timeout}s, tolerance={weight_tolerance}g, "
            f"max_duration={max_duration}s"
        )

    def _check_timeout(
        self,
        session: DoorSession,
        now: Optional[float] = None,
    ) -> TimeoutCheckResult:
        """
        세션 타임아웃 체크 (통합 로직, v4.2).

        Args:
            session: 체크할 DoorSession
            now: 현재 시각 (기본값: time.time())

        Returns:
            TimeoutCheckResult
        """
        if now is None:
            now = time.time()

        time_since_last = now - session.last_trigger_at
        total_duration = now - session.created_at

        if time_since_last > self._session_timeout:
            return TimeoutCheckResult(
                is_timed_out=True,
                reason="idle_timeout",
            )

        if total_duration > self._max_duration:
            return TimeoutCheckResult(
                is_timed_out=True,
                reason="max_duration",
            )

        return TimeoutCheckResult(is_timed_out=False)

    def _save_yaml_background(self, session: DoorSession) -> None:
        """
        YAML 저장을 백그라운드 스레드에서 비동기로 실행 (v4.8, v5.2).

        finalize 후 새 OPEN 요청에 즉시 응답하기 위해
        YAML 저장을 블로킹하지 않고 백그라운드로 처리합니다.

        v5.2: shutdown 상태 플래그로 race condition 방지

        Args:
            session: 저장할 DoorSession
        """
        # v5.2: shutdown 상태면 동기로 즉시 저장 (race condition 방지)
        if self._yaml_executor_shutdown:
            logger.debug(
                f"YAML executor shutdown, saving synchronously: {session.door_session_id}"
            )
            self._safe_yaml_save(session)
            return

        try:
            self._yaml_executor.submit(self._safe_yaml_save, session)
        except RuntimeError:
            # 스레드풀이 이미 shutdown된 경우 (서비스 종료 중)
            logger.warning(
                f"YAML executor shutdown, saving synchronously: {session.door_session_id}"
            )
            self._safe_yaml_save(session)

    def _safe_yaml_save(self, session: DoorSession) -> None:
        """
        백그라운드에서 안전하게 YAML 저장 (v4.8).

        Args:
            session: 저장할 DoorSession
        """
        try:
            self._persistence.save(session)
            logger.debug(f"YAML saved (background): {session.door_session_id}")
        except Exception as e:
            logger.error(f"YAML save failed: {session.door_session_id}: {e}")

    def shutdown(self) -> None:
        """
        서비스 종료 시 스레드풀 정리 (v4.8, v5.2).

        FastAPI lifespan 또는 shutdown event에서 호출해야 합니다.

        v5.2: shutdown 플래그를 먼저 설정하여 race condition 방지
        """
        logger.info("DoorSessionStore shutting down YAML executor...")
        # v5.2: 먼저 플래그 설정 (새 submit 방지)
        self._yaml_executor_shutdown = True
        self._yaml_executor.shutdown(wait=True)
        logger.info("DoorSessionStore YAML executor shutdown complete")

    def set_product_weight_getter(
        self,
        get_product_weight: Callable[[int], float],
    ) -> None:
        """
        ProductDatabase 무게 조회 함수 설정.

        Args:
            get_product_weight: product_id -> weight 조회 함수
        """
        self._get_product_weight = get_product_weight
        self._aggregator = ProductAggregator(
            weight_tolerance=self._weight_tolerance,
            get_product_weight=get_product_weight,
        )

    def set_session_finalize_callback(
        self,
        callback: Callable[[int], None],
    ) -> None:
        """
        세션 종료 콜백 설정 (v4.4).

        세션이 finalize될 때 호출됩니다.
        예: ActiveProductStore.clear(zone)

        Args:
            callback: zone를 인자로 받는 콜백 함수
        """
        self._on_session_finalize = callback
        logger.debug("Session finalize callback registered")

    def notify_trigger_enqueued(self, zone: int) -> None:
        """
        Trigger가 큐에 등록됨을 알림 (v4.10).

        pending 카운터 증가 + GlobalSession.last_trigger_at 갱신.
        CLOSE 신호가 큐 대기 중인 trigger를 놓치지 않도록 보장합니다.

        Args:
            zone: Zone 번호
        """
        with self._lock:
            self._pending_trigger_count += 1
            if self._global_session is not None:
                self._global_session.last_trigger_at = time.time()
            logger.debug(
                f"[TRIGGER-QUEUE] Enqueued notification: zone={zone}, "
                f"pending={self._pending_trigger_count}"
            )

    def notify_trigger_processed(self, zone: int) -> None:
        """
        Trigger 처리 완료를 알림 (v4.10).

        pending 카운터 감소.

        Args:
            zone: Zone 번호
        """
        with self._lock:
            self._pending_trigger_count = max(0, self._pending_trigger_count - 1)
            logger.debug(
                f"[TRIGGER-QUEUE] Processed notification: zone={zone}, "
                f"pending={self._pending_trigger_count}"
            )

    def add_trigger(
        self,
        zone: int,
        result: TriggerResult,
    ) -> DoorSession:
        """
        Trigger 결과 추가.

        활성 세션이 없으면 새로 생성하고,
        있으면 기존 세션에 trigger를 추가합니다.

        v4.9: 크로스 존 반환 처리 추가
        v4.5: Callback deferred 패턴 적용 (deadlock 방지)
        v4.2: Copy-on-Write 패턴 - Lock 내에서 데이터 수정, Lock 해제 후 YAML 저장

        Args:
            zone: Zone 번호
            result: TriggerResult

        Returns:
            업데이트된 DoorSession
        """
        session_to_save: Optional[DoorSession] = None
        session_to_finalize: Optional[DoorSession] = None
        deferred_callback: Optional[Callable[[], None]] = None  # v4.5
        other_sessions_to_save: List[DoorSession] = []  # v4.9 추가

        with self._lock:
            now = time.time()

            # 기존 세션 확인
            session = self._active_sessions.get(zone)

            if session is not None:
                # v4.6: GlobalSession 활성이면 타임아웃 체크 안함
                if self._global_session is not None:
                    logger.debug(
                        f"GlobalSession active, skipping timeout check for zone {zone}"
                    )
                else:
                    # 타임아웃 체크 (통합 로직 사용)
                    timeout_result = self._check_timeout(session, now)

                    if timeout_result.is_timed_out:
                        if timeout_result.reason == "idle_timeout":
                            logger.info(
                                f"Door session timed out: {session.door_session_id} "
                                f"(idle for {now - session.last_trigger_at:.1f}s)"
                            )
                        else:
                            logger.warning(
                                f"Door session max duration exceeded: {session.door_session_id} "
                                f"(duration={now - session.created_at:.1f}s)"
                            )
                        # Lock 내에서 finalize 처리 (메모리 상태만)
                        deferred_callback = self._finalize_session_in_memory(session)
                        session_to_finalize = session  # Lock 해제 후 YAML 저장
                        session = None

            if session is None:
                # 새 세션 생성
                session = DoorSession(
                    door_session_id=generate_door_session_id(zone),
                    zone=zone,
                    created_at=now,
                    last_trigger_at=now,
                )
                self._active_sessions[zone] = session
                logger.info(f"New door session created: {session.door_session_id}")

            # Trigger ID 생성
            result.trigger_id = f"trigger_{len(session.triggers) + 1:03d}"

            # Trigger 추가
            session.triggers.append(result)
            session.last_trigger_at = now

            # 상품 재집계 + 크로스 존 반환 처리 (v4.9)
            modified_zones = self._reaggregate_products(session)

            # v4.9: 크로스 존으로 변경된 다른 zone 세션 복사
            for other_zone in modified_zones:
                other_session = self._active_sessions.get(other_zone)
                if other_session:
                    other_sessions_to_save.append(copy.deepcopy(other_session))

            # 저장할 세션 복사 (Lock 해제 후 저장)
            session_to_save = copy.deepcopy(session)

            logger.info(
                f"Trigger added to {session.door_session_id}: "
                f"id={result.trigger_id}, delta={result.delta_weight:.1f}g, "
                f"is_return={result.is_return}, "
                f"total_triggers={len(session.triggers)}"
            )

            return_session = session

        # Lock 해제 후 콜백 실행 (v4.5)
        if deferred_callback is not None:
            deferred_callback()

        # v4.8: YAML 저장을 백그라운드로 (비동기)
        if session_to_finalize is not None:
            self._save_yaml_background(session_to_finalize)

        if session_to_save is not None:
            self._save_yaml_background(session_to_save)

        # v4.9: 크로스 존으로 변경된 다른 zone도 저장
        for other_session in other_sessions_to_save:
            self._save_yaml_background(other_session)

        return return_session

    def get_or_finalize(
        self,
        zone: int,
    ) -> Tuple[Optional[DoorSession], bool]:
        """
        세션 조회. 타임아웃 시 자동 finalize.

        v4.5: Callback deferred 패턴 적용 (deadlock 방지)
        v4.3: GlobalSession 활성 시 타임아웃 무시
        v4.2: Copy-on-Write 패턴, 통합 타임아웃 체크

        Args:
            zone: Zone 번호

        Returns:
            (DoorSession, is_finalized) 튜플
            - DoorSession: 세션 (없으면 None)
            - is_finalized: 이번 호출에서 finalize 되었는지 여부
        """
        session_to_save: Optional[DoorSession] = None
        return_session: Optional[DoorSession] = None
        deferred_callback: Optional[Callable[[], None]] = None  # v4.5
        is_finalized = False

        with self._lock:
            # v4.3: GlobalSession 활성이면 타임아웃 체크 안함
            if self._global_session is not None:
                session = self._active_sessions.get(zone)
                return session, False

            # 기존 로직 (GlobalSession 없을 때만)
            session = self._active_sessions.get(zone)

            if session is None:
                return None, False

            now = time.time()
            timeout_result = self._check_timeout(session, now)

            if timeout_result.is_timed_out:
                if timeout_result.reason == "idle_timeout":
                    logger.info(
                        f"Door session finalized (timeout): {session.door_session_id} "
                        f"(idle for {now - session.last_trigger_at:.1f}s)"
                    )
                else:
                    logger.warning(
                        f"Door session finalized (max duration): {session.door_session_id} "
                        f"(duration={now - session.created_at:.1f}s)"
                    )
                deferred_callback = self._finalize_session_in_memory(session)
                session_to_save = copy.deepcopy(session)
                return_session = session
                is_finalized = True
            else:
                return_session = session

        # Lock 해제 후 콜백 실행 (v4.5)
        if deferred_callback is not None:
            deferred_callback()

        # v4.8: YAML 저장을 백그라운드로 (비동기)
        if session_to_save is not None:
            self._save_yaml_background(session_to_save)

        return return_session, is_finalized

    # ========================================================================
    # GlobalDoorSession Methods (v4.3)
    # ========================================================================

    def get_or_start_global_session(self) -> GlobalDoorSession:
        """
        GlobalSession 조회 또는 시작 (v4.3).

        session_id="OPEN" 시 호출됩니다.

        핵심: 이미 활성 GlobalSession이 있으면 기존 것 반환 (초기화 안함)
        없으면 새로 생성합니다.

        Returns:
            GlobalDoorSession (기존 또는 새로 생성된 것)
        """
        with self._lock:
            if self._global_session is not None:
                # 이미 활성 세션 있음 → 기존 것 반환
                logger.debug(
                    f"GlobalSession already active: {self._global_session.global_session_id}"
                )
                return self._global_session

            # 새 세션 생성
            self._global_session = GlobalDoorSession(
                global_session_id=generate_global_session_id(),
            )
            logger.info(
                f"GlobalSession started: {self._global_session.global_session_id}"
            )
            return self._global_session

    def finalize_global_session(self) -> Optional[GlobalDoorSession]:
        """
        GlobalSession 종료 (v4.3).

        session_id="CLOSE" 시 호출됩니다.
        모든 활성 zone의 DoorSession을 finalize하고 결과를 반환합니다.

        v4.5: Callback deferred 패턴 적용 (deadlock 방지)

        Returns:
            종료된 GlobalDoorSession 또는 None (활성 세션이 없는 경우)
        """
        sessions_to_save: List[DoorSession] = []
        deferred_callbacks: List[Callable[[], None]] = []  # v4.5

        with self._lock:
            if self._global_session is None:
                logger.warning("finalize_global_session called but no active session")
                return None

            # 모든 활성 zone 세션 finalize
            for zone in list(self._active_sessions.keys()):
                session = self._active_sessions[zone]
                callback = self._finalize_session_in_memory(session)
                if callback is not None:
                    deferred_callbacks.append(callback)
                self._global_session.zone_sessions[zone] = session
                sessions_to_save.append(copy.deepcopy(session))

            self._global_session.status = "complete"
            self._global_session.finalized_at = time.time()

            result = self._global_session
            self._global_session = None

            logger.info(
                f"GlobalSession finalized: {result.global_session_id}, "
                f"zones={len(result.zone_sessions)}, "
                f"total_price={result.total_price}, "
                f"total_products={result.total_product_count}"
            )

        # Lock 해제 후 콜백 실행 (v4.5)
        for callback in deferred_callbacks:
            callback()

        # v4.8: YAML 저장을 백그라운드로 (비동기)
        for session in sessions_to_save:
            self._save_yaml_background(session)

        return result

    def handle_close_signal(
        self,
        initial_wait_seconds: float = 20.0,
        subsequent_wait_seconds: float = 5.0,
        _now: Optional[float] = None,
    ) -> Tuple[bool, Optional[GlobalDoorSession]]:
        """
        CLOSE 신호 처리 (v4.10, v5.2).

        빠른 문 열고 닫기 상황 대응:
        1. 첫 CLOSE → pending_close=True, in_progress 반환
        2. 이후 CLOSE:
           - v4.10: pending_trigger_count > 0 → 큐 대기 중이므로 finalize 거부
           - trigger 없음 → first_close_at 기준 20초 대기 (YOLO 로드 시간 고려)
           - trigger 있음 → last_trigger_at 기준 5초 대기 (이미 처리 중이므로 빠름)

        v5.2: _now 파라미터 추가 (테스트 시 시간 주입 가능)

        Args:
            initial_wait_seconds: trigger 없을 때 대기 시간 (기본 20초)
            subsequent_wait_seconds: trigger 있을 때 대기 시간 (기본 5초)
            _now: 테스트용 현재 시각 (None이면 time.time() 사용)

        Returns:
            (is_ready_to_finalize, global_session)
            - is_ready_to_finalize: True면 finalize 진행 가능
            - global_session: 현재 GlobalSession (없으면 None)
        """
        with self._lock:
            if self._global_session is None:
                logger.info("[CLOSE] No active global session")
                return True, None  # 세션 없음 → success 반환 가능

            # v5.2: 시간 주입 지원 (테스트용)
            now = _now if _now is not None else time.time()

            # v4.10: 큐에 대기 중인 trigger가 있으면 finalize 거부
            if self._pending_trigger_count > 0:
                logger.info(
                    f"[CLOSE] {self._pending_trigger_count} triggers pending in queue, waiting..."
                )
                if not self._global_session.pending_close:
                    self._global_session.pending_close = True
                    self._global_session.first_close_at = now
                return False, self._global_session

            if not self._global_session.pending_close:
                # 첫 CLOSE: pending_close 설정
                self._global_session.pending_close = True
                self._global_session.first_close_at = now
                logger.info(
                    f"[CLOSE] 첫 CLOSE 수신: global_session_id={self._global_session.global_session_id}, "
                    f"pending_close=True, first_close_at={now:.1f}"
                )
                return False, self._global_session

            # 이후 CLOSE: trigger 유무에 따라 대기 시간 다르게 적용
            has_trigger = self._global_session.last_trigger_at is not None

            if has_trigger:
                # trigger 있음 → last_trigger_at 기준 5초 대기
                elapsed_since_trigger = now - self._global_session.last_trigger_at
                wait_seconds = subsequent_wait_seconds

                if elapsed_since_trigger < wait_seconds:
                    # v5.2: first_close_at 갱신 (테스트 호환성)
                    # trigger 후 CLOSE가 들어왔는데 아직 대기 중이면 first_close_at을 현재 시각으로 갱신
                    self._global_session.first_close_at = now
                    logger.info(
                        f"[CLOSE] trigger 후 대기 중: elapsed={elapsed_since_trigger:.1f}s < {wait_seconds}s, "
                        f"global_session_id={self._global_session.global_session_id}, "
                        f"first_close_at 갱신됨"
                    )
                    return False, self._global_session
                else:
                    # 5초 지남 → finalize 준비 완료
                    logger.info(
                        f"[CLOSE] trigger 후 {wait_seconds}초 대기 완료, finalize 준비: "
                        f"global_session_id={self._global_session.global_session_id}, "
                        f"elapsed_since_trigger={elapsed_since_trigger:.1f}s"
                    )
                    return True, self._global_session
            else:
                # trigger 없음 → first_close_at 기준 20초 대기
                elapsed_since_close = now - self._global_session.first_close_at
                wait_seconds = initial_wait_seconds

                if elapsed_since_close < wait_seconds:
                    logger.info(
                        f"[CLOSE] trigger 없음, 대기 중: elapsed={elapsed_since_close:.1f}s < {wait_seconds}s, "
                        f"global_session_id={self._global_session.global_session_id}"
                    )
                    return False, self._global_session
                else:
                    # 20초 지남 → finalize 준비 완료
                    logger.info(
                        f"[CLOSE] trigger 없음, {wait_seconds}초 대기 완료, finalize 준비: "
                        f"global_session_id={self._global_session.global_session_id}, "
                        f"elapsed_since_close={elapsed_since_close:.1f}s"
                    )
                    return True, self._global_session

    def get_global_session(self) -> Optional[GlobalDoorSession]:
        """
        현재 활성 GlobalSession 반환 (v4.3).

        Returns:
            GlobalDoorSession 또는 None
        """
        with self._lock:
            return self._global_session

    def is_global_session_active(self) -> bool:
        """
        GlobalSession 활성 여부 (v4.3).

        Returns:
            True if GlobalSession is active
        """
        with self._lock:
            return self._global_session is not None

    def add_trigger_with_global(
        self,
        zone: int,
        result: TriggerResult,
    ) -> DoorSession:
        """
        Trigger 결과 추가 + GlobalSession 연동 (v4.3).

        기존 add_trigger를 호출하고,
        GlobalSession이 활성이면 해당 zone의 DoorSession을 연동합니다.

        Args:
            zone: Zone 번호
            result: TriggerResult

        Returns:
            업데이트된 DoorSession
        """
        # 기존 add_trigger 호출
        door_session = self.add_trigger(zone=zone, result=result)

        # GlobalSession에 연동 + last_trigger_at 업데이트 (v4.6)
        with self._lock:
            if self._global_session is not None:
                self._global_session.zone_sessions[zone] = door_session
                self._global_session.last_trigger_at = time.time()  # v4.6
                logger.debug(
                    f"Door session linked to GlobalSession: zone={zone}, "
                    f"global_id={self._global_session.global_session_id}"
                )

        return door_session

    # ========================================================================
    # Original Methods (continued)
    # ========================================================================

    def get_session(self, zone: int) -> Optional[DoorSession]:
        """
        세션 조회 (타임아웃 체크 없음).

        Args:
            zone: Zone 번호

        Returns:
            DoorSession 또는 None
        """
        with self._lock:
            return self._active_sessions.get(zone)

    def finalize_session(self, zone: int) -> Optional[DoorSession]:
        """
        세션 강제 종료.

        v4.5: Callback deferred 패턴 적용 (deadlock 방지)
        v4.2: Copy-on-Write 패턴

        Args:
            zone: Zone 번호

        Returns:
            종료된 DoorSession 또는 None
        """
        session_to_save: Optional[DoorSession] = None
        deferred_callback: Optional[Callable[[], None]] = None  # v4.5

        with self._lock:
            session = self._active_sessions.get(zone)
            if session is not None:
                deferred_callback = self._finalize_session_in_memory(session)
                session_to_save = copy.deepcopy(session)
            else:
                return None

        # Lock 해제 후 콜백 실행 (v4.5)
        if deferred_callback is not None:
            deferred_callback()

        # v4.8: YAML 저장을 백그라운드로 (비동기)
        if session_to_save is not None:
            self._save_yaml_background(session_to_save)

        return session_to_save

    def _finalize_session_in_memory(
        self, session: DoorSession
    ) -> Optional[Callable[[], None]]:
        """
        세션 종료 처리 - 메모리 상태만 변경 (내부용, lock 내에서 호출).

        v4.5: callback을 반환하여 lock 해제 후 실행 (deadlock 방지)
        v4.2: YAML 저장은 Lock 해제 후 별도로 수행

        Args:
            session: 종료할 DoorSession

        Returns:
            콜백 함수 (lock 해제 후 실행) 또는 None
        """
        session.status = "complete"
        session.finalized_at = time.time()

        zone = session.zone

        # 활성 세션에서 제거
        if zone in self._active_sessions:
            del self._active_sessions[zone]

        logger.info(
            f"Door session finalized: {session.door_session_id}, "
            f"triggers={len(session.triggers)}, "
            f"products={session.product_count}, "
            f"total_price={session.total_price}"
        )

        # v4.5: 콜백을 반환하여 lock 해제 후 실행 (deadlock 방지)
        if self._on_session_finalize is not None:
            callback_fn = self._on_session_finalize
            return lambda: self._execute_finalize_callback(callback_fn, zone)

        return None

    def _execute_finalize_callback(
        self,
        callback: Callable[[int], None],
        zone: int,
    ) -> None:
        """
        Finalize 콜백 실행 (lock 해제 후 호출).

        Args:
            callback: 콜백 함수
            zone: Zone 번호
        """
        try:
            callback(zone)
            logger.debug(f"Session finalize callback invoked for zone={zone}")
        except Exception as e:
            logger.error(f"Session finalize callback failed for zone={zone}: {e}")

    def _finalize_session(self, session: DoorSession) -> None:
        """
        세션 종료 처리 - 메모리 + YAML (레거시 호환, lock 내에서 호출).

        주의: 이 메서드는 Lock 내에서 I/O를 수행하므로 새 코드에서는
        _finalize_session_in_memory + Lock 해제 후 save 패턴 사용 권장.

        Args:
            session: 종료할 DoorSession
        """
        self._finalize_session_in_memory(session)
        self._persistence.save(session)

    def _reaggregate_products(self, session: DoorSession) -> List[int]:
        """
        세션의 상품 재집계 (내부용, lock 내에서 호출).

        v4.9: 크로스 존 반환 처리 추가
        v4.2: unmatched_returns 추적 추가

        Args:
            session: 재집계할 DoorSession

        Returns:
            v4.9: 크로스 존으로 변경된 다른 zone 번호 목록
        """
        # 전체 trigger에서 상품 재집계 (unmatched_returns 포함)
        result = self._aggregator.aggregate_with_unmatched(session.triggers)
        session.aggregated_products = result.products
        session.unmatched_returns = result.unmatched_returns

        # 무게 정보 업데이트
        if self._get_product_weight is not None:
            self._aggregator.update_weights_from_db(
                session.aggregated_products,
                self._get_product_weight,
            )

        # v5.0: net delta 검증 (전량/부분 반환 보정)
        self._validate_net_delta(session)

        # v4.9: 크로스 존 반환 처리
        modified_zones = self._handle_cross_zone_returns(session)
        return modified_zones

    def _validate_net_delta(self, session: DoorSession) -> None:
        """
        DoorSession의 net delta vs expected product weight 교차 검증 (v5.0).

        전량 반환 또는 부분 반환 시 aggregated_products 개수를 보정합니다.
        """
        net_delta = sum(t.delta_weight for t in session.triggers)
        active_products = [
            p for p in session.aggregated_products.values()
            if p.count > 0 and p.weight > 0
        ]
        expected_weight = sum(p.weight * p.count for p in active_products)

        if not active_products:
            return

        NET_ZERO_THRESHOLD = 15.0  # g

        # Case 1: 전량 반환 (net_delta ≈ 0이거나 양수)
        if net_delta >= -NET_ZERO_THRESHOLD and expected_weight > NET_ZERO_THRESHOLD:
            logger.info(
                f"[복구 모드] 전량 반환 감지: net_delta={net_delta:.1f}g, "
                f"expected={expected_weight:.1f}g"
            )
            for p in active_products:
                logger.info(
                    f"[복구 모드] 복구된 상품: {p.name} x{p.count} ({p.weight}g)"
                )
                p.count = 0
            return

        # Case 2: 부분 반환 (net_delta < expected_weight)
        if abs(net_delta) < expected_weight - NET_ZERO_THRESHOLD:
            actual_removed = abs(net_delta)
            remaining = actual_removed
            corrected = False
            for p in sorted(active_products, key=lambda x: x.weight, reverse=True):
                if p.weight <= 0:
                    continue
                new_count = min(p.count, max(0, int(round(remaining / p.weight))))
                if new_count < p.count:
                    logger.info(
                        f"[복구 모드] 부분 반환: {p.name} {p.count}개 → {new_count}개"
                    )
                    p.count = new_count
                    corrected = True
                remaining -= p.weight * new_count
                if remaining <= NET_ZERO_THRESHOLD:
                    break
            if corrected:
                total = sum(
                    p.count for p in session.aggregated_products.values()
                    if p.count > 0
                )
                logger.info(f"[복구 모드] 보정 완료: 총 {total}개")

    def _handle_cross_zone_returns(self, session: DoorSession) -> List[int]:
        """
        크로스 존 반환 처리 (v4.9).

        현재 zone에서 매칭 실패한 반환(unmatched_returns)을
        다른 zone의 aggregated_products에서 무게 매칭 시도.

        Lock 내에서 호출됨 - 다른 zone 세션 접근 안전함.

        Args:
            session: 반환이 발생한 DoorSession

        Returns:
            변경된 다른 zone 번호 목록 (YAML 저장 필요)
        """
        if not session.unmatched_returns:
            return []

        modified_zones: List[int] = []
        still_unmatched: List[UnmatchedReturn] = []

        for unmatched in session.unmatched_returns:
            matched = False

            # 다른 모든 zone에서 무게 매칭 시도
            for other_zone, other_session in self._active_sessions.items():
                if other_zone == session.zone:
                    continue  # 현재 zone 건너뜀

                # 다른 zone의 aggregated_products에서 무게 매칭
                matched_product_id = self._aggregator.find_product_by_weight(
                    other_session.aggregated_products,
                    unmatched.delta_weight,
                )

                if matched_product_id is not None:
                    product = other_session.aggregated_products[matched_product_id]
                    if product.count > 0:
                        # 매칭 성공: 다른 zone에서 차감
                        product.count -= 1

                        # 크로스 존 기록 생성
                        record = CrossZoneReturn(
                            trigger_id=unmatched.trigger_id,
                            source_zone=session.zone,
                            target_zone=other_zone,
                            product_id=matched_product_id,
                            product_name=product.name,
                            matched_weight=product.weight,
                            delta_weight=unmatched.delta_weight,
                            timestamp=time.time(),
                        )
                        session.cross_zone_returns.append(record)

                        # 변경된 zone 기록
                        if other_zone not in modified_zones:
                            modified_zones.append(other_zone)

                        logger.info(
                            f"Cross-zone return: zone {session.zone} -> zone {other_zone}, "
                            f"product={product.name}, weight={unmatched.delta_weight:.1f}g"
                        )
                        matched = True
                        break

            if not matched:
                still_unmatched.append(unmatched)

        # 매칭 실패 항목 업데이트
        session.unmatched_returns = still_unmatched

        return modified_zones

    def recover_active_sessions(self) -> int:
        """
        서비스 시작 시 활성 세션 복구.

        v4.2: 통합 타임아웃 체크, Copy-on-Write

        Returns:
            복구된 세션 수
        """
        sessions_to_save: List[DoorSession] = []

        with self._lock:
            recovered = self._persistence.recover_active_sessions()
            now = time.time()

            for zone, session in recovered.items():
                # 타임아웃 체크 (통합 로직 사용)
                timeout_result = self._check_timeout(session, now)

                if timeout_result.is_timed_out:
                    # 이미 타임아웃됨 → finalize
                    logger.info(
                        f"Recovered session already timed out: {session.door_session_id}"
                    )
                    session.status = "complete"
                    session.finalized_at = now
                    sessions_to_save.append(copy.deepcopy(session))
                else:
                    # 아직 활성 → 복구
                    self._active_sessions[zone] = session
                    logger.info(
                        f"Recovered active session: {session.door_session_id} "
                        f"(idle for {now - session.last_trigger_at:.1f}s)"
                    )

            active_count = len(self._active_sessions)

        # v4.8: YAML 저장을 백그라운드로 (비동기)
        for session in sessions_to_save:
            self._save_yaml_background(session)

        return active_count

    def get_stats(self) -> dict:
        """
        저장소 통계 반환.

        v4.3: GlobalSession 정보 추가

        Returns:
            통계 정보
        """
        with self._lock:
            active_zones = list(self._active_sessions.keys())
            persistence_stats = self._persistence.get_stats()

            # v4.3: GlobalSession 정보
            global_session_info = None
            if self._global_session is not None:
                global_session_info = {
                    "global_session_id": self._global_session.global_session_id,
                    "status": self._global_session.status,
                    "zone_count": len(self._global_session.zone_sessions),
                    "active_zones": self._global_session.active_zones,
                    "total_trigger_count": self._global_session.total_trigger_count,
                    "total_price": self._global_session.total_price,
                    "duration_seconds": round(self._global_session.duration_seconds, 1),
                }

            return {
                "active_sessions": len(self._active_sessions),
                "active_zones": active_zones,
                "session_timeout": self._session_timeout,
                "weight_tolerance": self._weight_tolerance,
                "max_duration": self._max_duration,
                "global_session_max_duration": self._global_session_max_duration,
                "global_session_active": self._global_session is not None,
                "global_session": global_session_info,
                "pending_trigger_count": self._pending_trigger_count,  # v4.10
                **persistence_stats,
            }

    def cleanup_timed_out_sessions(self) -> int:
        """
        타임아웃된 세션 정리 (v4.5).

        cleanup task에서 주기적으로 호출됩니다.
        - GlobalSession max_duration 초과 시 자동 finalize
        - 개별 DoorSession 타임아웃 체크

        Returns:
            정리된 세션 수
        """
        sessions_to_save: List[DoorSession] = []
        deferred_callbacks: List[Callable[[], None]] = []
        global_session_finalized = False
        cleaned_count = 0

        with self._lock:
            now = time.time()

            # 1. GlobalSession max_duration 체크
            if self._global_session is not None:
                duration = self._global_session.duration_seconds
                if duration > self._global_session_max_duration:
                    logger.warning(
                        f"GlobalSession timed out (max_duration): "
                        f"{self._global_session.global_session_id} "
                        f"(duration={duration:.1f}s > {self._global_session_max_duration}s)"
                    )
                    # 모든 활성 zone 세션 finalize
                    for zone in list(self._active_sessions.keys()):
                        session = self._active_sessions[zone]
                        callback = self._finalize_session_in_memory(session)
                        if callback is not None:
                            deferred_callbacks.append(callback)
                        self._global_session.zone_sessions[zone] = session
                        sessions_to_save.append(copy.deepcopy(session))
                        cleaned_count += 1

                    self._global_session.status = "complete"
                    self._global_session.finalized_at = now
                    self._global_session = None
                    global_session_finalized = True

            # 2. GlobalSession이 없을 때만 개별 DoorSession 타임아웃 체크
            if not global_session_finalized and self._global_session is None:
                for zone in list(self._active_sessions.keys()):
                    session = self._active_sessions[zone]
                    timeout_result = self._check_timeout(session, now)

                    if timeout_result.is_timed_out:
                        logger.info(
                            f"DoorSession timed out (cleanup): {session.door_session_id} "
                            f"(reason={timeout_result.reason})"
                        )
                        callback = self._finalize_session_in_memory(session)
                        if callback is not None:
                            deferred_callbacks.append(callback)
                        sessions_to_save.append(copy.deepcopy(session))
                        cleaned_count += 1

        # Lock 해제 후 콜백 실행 (v4.5)
        for callback in deferred_callbacks:
            callback()

        # v4.8: YAML 저장을 백그라운드로 (비동기)
        for session in sessions_to_save:
            self._save_yaml_background(session)

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} timed out door sessions")

        return cleaned_count

    def clear_all(self) -> None:
        """모든 활성 세션 정리 (v4.5: Callback deferred 패턴)."""
        sessions_to_save: List[DoorSession] = []
        deferred_callbacks: List[Callable[[], None]] = []  # v4.5

        with self._lock:
            for session in list(self._active_sessions.values()):
                callback = self._finalize_session_in_memory(session)
                if callback is not None:
                    deferred_callbacks.append(callback)
                sessions_to_save.append(copy.deepcopy(session))
            self._active_sessions.clear()
            self._global_session = None  # GlobalSession도 정리

        # Lock 해제 후 콜백 실행 (v4.5)
        for callback in deferred_callbacks:
            callback()

        # v4.8: YAML 저장을 백그라운드로 (비동기)
        for session in sessions_to_save:
            self._save_yaml_background(session)

        logger.info(f"All {len(sessions_to_save)} door sessions cleared")
