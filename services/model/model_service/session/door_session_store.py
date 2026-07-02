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

import concurrent.futures
import copy
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from model_service.core.config import config
from model_service.core.logging_config import get_ops_logger

from .door_session import (
    AggregatedProduct,
    CrossZoneReturn,
    DeferredReturn,
    DoorSession,
    ReturnedPositionHint,
    TriggerResult,
    UnmatchedReturn,
    generate_door_session_id,
    trigger_effective_delta_weight,
    unmatched_return_delta_weight,
)
from .freezer_close_aggregate import FreezerCloseAggregateResolver
from .global_door_session import GlobalDoorSession, generate_global_session_id
from .product_aggregator import ProductAggregator
from .yaml_persistence import YamlPersistence

logger = logging.getLogger(__name__)
ops_logger = get_ops_logger()


@dataclass
class TimeoutCheckResult:
    """타임아웃 체크 결과."""
    is_timed_out: bool
    reason: str = ""  # "idle_timeout" | "max_duration" | ""


@dataclass(frozen=True)
class _CrossZoneReturnCandidate:
    """Single returned unit that can be matched against another zone."""

    target_zone: int
    product_id: int
    product_name: str
    unit_weight: float
    unit_index: int
    session_created_at: float
    channel_side: Optional[str] = None
    placement_index: Optional[int] = None
    placement_unit: Optional[Dict[str, object]] = None
    source_timestamp: float = 0.0

    @property
    def stable_key(self) -> tuple:
        return (
            self.session_created_at,
            self.target_zone,
            self.product_id,
            self.unit_index,
            self.channel_side or "",
        )


@dataclass
class _CrossZoneReturnMatch:
    """Candidate combination selected for a cross-zone return."""

    candidates: List[_CrossZoneReturnCandidate]
    total_weight: float
    weight_error: float


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
        self._pending_triggers: Dict[str, dict] = {}
        self._pending_trigger_seq = 0
        self._pending_trigger_count = 0
        self._skipped_balanced_trigger_count = 0

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

    def _legacy_notify_trigger_enqueued_count_only(self, zone: int) -> None:
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

    def _legacy_notify_trigger_processed_count_only(self, zone: int) -> None:
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

    def _next_legacy_pending_trigger_id_locked(self) -> str:
        self._pending_trigger_seq += 1
        return f"legacy:{self._pending_trigger_seq}"

    def _sync_pending_trigger_count_locked(self) -> None:
        self._pending_trigger_count = len(self._pending_triggers)

    def _resolve_pending_trigger_key_locked(
        self,
        zone: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        if isinstance(zone, str) and session_id is None:
            session_id = zone
            zone = None

        if session_id is not None and session_id in self._pending_triggers:
            return session_id

        if zone is None:
            return None

        for key, value in self._pending_triggers.items():
            if value.get("zone") == zone:
                return key
        return None

    def _pending_trigger_snapshot_locked(self) -> dict:
        pending = list(self._pending_triggers.values())
        chargeable_pending = [
            item for item in pending
            if item.get("chargeable_vision_required", True)
        ]
        session_ids = [
            str(item["session_id"])
            for item in pending
            if not str(item["session_id"]).startswith("legacy:")
        ]
        return {
            "pendingTriggerCount": len(pending),
            "pendingTriggerZones": sorted(
                {
                    int(item["zone"])
                    for item in pending
                    if item.get("zone") is not None
                }
            ),
            "pendingTriggerSessionIds": session_ids,
            "pendingTriggerStatuses": {
                str(item["session_id"]): item.get("status", "queued")
                for item in pending
                if not str(item["session_id"]).startswith("legacy:")
            },
            "pendingChargeableVisionCount": len(chargeable_pending),
            "pendingChargeableVisionSessionIds": [
                str(item["session_id"])
                for item in chargeable_pending
                if not str(item["session_id"]).startswith("legacy:")
            ],
            "skippedBalancedTriggerCount": self._skipped_balanced_trigger_count,
        }

    def get_pending_trigger_snapshot(self) -> dict:
        """Return current in-flight trigger details for CLOSE diagnostics."""
        with self._lock:
            return self._pending_trigger_snapshot_locked()

    def record_no_charge_diagnostic(
        self,
        *,
        zone: int,
        session_id: str,
        reason: str,
        delta_weight: float,
        processing_stage: str,
        payload_diagnostics: Optional[dict] = None,
        video_paths: Optional[dict] = None,
        message: Optional[str] = None,
    ) -> None:
        """Attach a no-charge trigger diagnostic to the active global session."""
        diagnostic = {
            "zone": int(zone),
            "sessionId": str(session_id),
            "reason": str(reason),
            "deltaWeight": round(float(delta_weight), 1),
            "processingStage": str(processing_stage),
            "timestamp": time.time(),
        }
        if message:
            diagnostic["message"] = str(message)
        if video_paths:
            diagnostic["videoPaths"] = dict(video_paths)
        if payload_diagnostics:
            field_map = {
                "payload_state": "payloadState",
                "raw_state": "rawState",
                "filtered_state": "filteredState",
                "first_raw_total": "firstRawTotal",
                "last_raw_total": "lastRawTotal",
                "first_filtered_total": "firstFilteredTotal",
                "last_filtered_total": "lastFilteredTotal",
                "filtered_channel_count": "filteredChannelCount",
                "filtered_zero_channel_count": "filteredZeroChannelCount",
            }
            for source, target in field_map.items():
                if source in payload_diagnostics:
                    diagnostic[target] = payload_diagnostics[source]

        with self._lock:
            if self._global_session is None:
                logger.debug(
                    "[NO-CHARGE-DIAGNOSTIC] ignored without active global session: "
                    f"zone={zone}, session_id={session_id}, reason={reason}"
                )
                return
            self._global_session.no_charge_diagnostics.append(diagnostic)

        logger.info(
            "[NO-CHARGE-DIAGNOSTIC] "
            f"zone={zone} session_id={session_id} reason={reason} "
            f"delta={delta_weight:.1f}g"
        )

    def notify_trigger_enqueued(
        self,
        zone: int,
        session_id: Optional[str] = None,
        chargeable_vision_required: bool = True,
    ) -> None:
        """Mark a trigger as queued and block CLOSE finalization."""
        with self._lock:
            key = session_id or self._next_legacy_pending_trigger_id_locked()
            now = time.time()
            self._pending_triggers[key] = {
                "session_id": key,
                "zone": zone,
                "status": "queued",
                "enqueued_at": now,
                "updated_at": now,
                "chargeable_vision_required": bool(chargeable_vision_required),
            }
            self._sync_pending_trigger_count_locked()
            if self._global_session is not None:
                self._global_session.last_trigger_at = now
            logger.debug(
                f"[TRIGGER-QUEUE] Enqueued notification: zone={zone}, "
                f"session_id={key}, pending={self._pending_trigger_count}"
            )

    def _mark_pending_trigger_status(
        self,
        status: str,
        zone: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> None:
        with self._lock:
            key = self._resolve_pending_trigger_key_locked(zone, session_id)
            if key is None:
                logger.debug(
                    f"[TRIGGER-QUEUE] Status notification ignored: "
                    f"zone={zone}, session_id={session_id}, status={status}, "
                    f"pending={self._pending_trigger_count}"
                )
                return
            self._pending_triggers[key]["status"] = status
            self._pending_triggers[key]["updated_at"] = time.time()
            logger.debug(
                f"[TRIGGER-QUEUE] Status notification: "
                f"zone={self._pending_triggers[key].get('zone')}, "
                f"session_id={key}, status={status}, "
                f"pending={self._pending_trigger_count}"
            )

    def notify_trigger_started(
        self,
        zone: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Mark a queued trigger as actively processing."""
        self._mark_pending_trigger_status("processing", zone, session_id)

    def notify_trigger_finalizing(
        self,
        zone: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Mark a trigger as past inference but still finalizing stores/traces."""
        self._mark_pending_trigger_status("finalizing", zone, session_id)

    def notify_trigger_processed(
        self,
        zone: Optional[int] = None,
        session_id: Optional[str] = None,
        status: str = "complete",
    ) -> None:
        """Remove a trigger from the in-flight registry."""
        with self._lock:
            key = self._resolve_pending_trigger_key_locked(zone, session_id)
            if key is None:
                logger.debug(
                    f"[TRIGGER-QUEUE] Processed notification ignored: "
                    f"zone={zone}, session_id={session_id}, status={status}, "
                    f"pending={self._pending_trigger_count}"
                )
                return
            pending = self._pending_triggers.pop(key)
            if status == "skipped_balanced":
                self._skipped_balanced_trigger_count += 1
            self._sync_pending_trigger_count_locked()
            logger.debug(
                f"[TRIGGER-QUEUE] Processed notification: zone={pending.get('zone')}, "
                f"session_id={key}, status={status}, pending={self._pending_trigger_count}"
            )

    @staticmethod
    def _ensure_trigger_placement_metadata(
        result: TriggerResult,
        *,
        zone: int,
    ) -> None:
        for product in result.products:
            units = [
                dict(unit)
                for unit in getattr(product, "placement_units", []) or []
                if isinstance(unit, dict)
            ]
            while len(units) < max(0, int(product.count)):
                units.append(
                    {
                        "product_id": int(product.product_id),
                        "product_idx": product.product_idx,
                        "name": product.name,
                        "channelSide": "unknown",
                        "source": "door_session_trigger_product",
                    }
                )
            product.placement_units = units[: max(0, int(product.count))]
            for unit in product.placement_units:
                unit.setdefault("zone", int(zone))
                unit.setdefault("sourceTriggerId", result.trigger_id)
                unit.setdefault("sourceSessionId", result.session_id)
                unit.setdefault("product_id", int(product.product_id))
                unit.setdefault("product_idx", product.product_idx)
                unit.setdefault("name", product.name)
                unit.setdefault("channelSide", "unknown")

    @staticmethod
    def _dedupe_returned_position_hints(
        hints: List[ReturnedPositionHint],
    ) -> List[ReturnedPositionHint]:
        deduped: List[ReturnedPositionHint] = []
        seen: set[tuple] = set()
        for hint in hints:
            key = (
                int(hint.product_id),
                hint.product_idx,
                int(hint.zone) if hint.zone is not None else None,
                hint.channel_side,
                hint.channel_index,
                hint.channel_position,
                hint.trigger_id,
                hint.source,
                hint.reason,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(hint)
        return deduped

    @staticmethod
    def _candidate_identity_passed(candidate: Dict[str, object]) -> bool:
        try:
            top_confidence = float(candidate.get("top_confidence", 0.0) or 0.0)
            side_confidence = float(candidate.get("side_confidence", 0.0) or 0.0)
            identity_confidence = float(
                candidate.get(
                    "identity_confidence",
                    candidate.get("confidence", 0.0),
                )
                or 0.0
            )
        except (TypeError, ValueError):
            return False

        top_detected = bool(candidate.get("top")) or top_confidence > 0.0
        side_detected = bool(candidate.get("side")) or side_confidence > 0.0
        if top_detected and top_confidence >= float(config.vision.top_confidence_threshold):
            return True
        if side_detected and side_confidence >= float(config.vision.side_confidence_threshold):
            return True
        if not top_detected and not side_detected:
            return identity_confidence >= min(
                float(config.vision.top_confidence_threshold),
                float(config.vision.side_confidence_threshold),
            )
        return False

    @staticmethod
    def _channel_target_has_position(target: Dict[str, object]) -> bool:
        return any(
            target.get(key) is not None
            for key in (
                "channel_side",
                "channelSide",
                "channel_index",
                "channelIndex",
                "channel_position",
                "channelPosition",
            )
        )

    @staticmethod
    def _positive_channel_targets(trigger: TriggerResult) -> List[Dict[str, object]]:
        diagnostics = getattr(trigger, "loadcell_diagnostics", {}) or {}
        if not isinstance(diagnostics, dict):
            return []
        targets: List[Dict[str, object]] = []
        for entry in diagnostics.get("channel_movement_targets") or []:
            if not isinstance(entry, dict):
                continue
            direction = str(entry.get("direction", "")).lower()
            try:
                delta = float(entry.get("delta", 0.0) or 0.0)
                weight = abs(float(entry.get("weight", delta) or delta))
            except (TypeError, ValueError):
                continue
            if weight <= 0:
                continue
            if delta > 0 or direction == "return":
                target = dict(entry)
                target["weight"] = round(weight, 1)
                targets.append(target)
        return targets

    def _collect_touch_return_hints(
        self,
        session: DoorSession,
    ) -> List[ReturnedPositionHint]:
        if str(config.machine.cabinet_type).strip().lower() != "freezer":
            return []

        hints: List[ReturnedPositionHint] = []
        recent_return_targets: List[Dict[str, object]] = []
        ordered_triggers = sorted(
            session.triggers,
            key=lambda trigger: (float(trigger.timestamp), trigger.trigger_id),
        )
        for trigger in ordered_triggers:
            positive_targets = [
                target
                for target in self._positive_channel_targets(trigger)
                if self._channel_target_has_position(target)
            ]
            if positive_targets:
                recent_return_targets = positive_targets

            hint = self._touch_return_hint_from_trigger(
                session,
                trigger,
                positive_targets or recent_return_targets,
            )
            if hint is not None:
                hints.append(hint)

            try:
                delta = float(trigger.delta_weight)
            except (TypeError, ValueError):
                delta = 0.0
            if delta < -float(config.trigger.min_weight_change_grams):
                diagnostics = getattr(trigger, "loadcell_diagnostics", {}) or {}
                if not isinstance(diagnostics, dict) or not diagnostics.get(
                    "mixed_sign_internal_segments"
                ):
                    recent_return_targets = []
        return hints

    def _touch_return_hint_from_trigger(
        self,
        session: DoorSession,
        trigger: TriggerResult,
        return_targets: List[Dict[str, object]],
    ) -> Optional[ReturnedPositionHint]:
        if not return_targets:
            return None
        if any(int(getattr(product, "count", 0) or 0) > 0 for product in trigger.products):
            return None

        try:
            delta = float(trigger.delta_weight)
        except (TypeError, ValueError):
            return None

        candidates = [
            candidate
            for candidate in getattr(trigger, "vision_candidates", []) or []
            if isinstance(candidate, dict)
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda item: int(item.get("rank", 999) or 999))
        candidate = candidates[0]
        if not self._candidate_identity_passed(candidate):
            return None

        try:
            product_id = int(candidate.get("product_id"))
            unit_weight = float(candidate.get("unit_weight", 0.0) or 0.0)
            stock_qty = int(candidate.get("stock_qty", 0) or 0)
        except (TypeError, ValueError):
            return None
        if product_id < 0 or unit_weight <= 0 or stock_qty <= 0:
            return None
        touch_limit = max(
            float(config.trigger.min_weight_change_grams),
            float(config.weight.freezer_weight_tolerance_grams),
            float(unit_weight) * 0.25,
        )
        if abs(delta) > touch_limit:
            return None
        if abs(abs(delta) - unit_weight) <= float(config.weight.freezer_weight_tolerance_grams):
            return None

        target = return_targets[0]
        channel_side = target.get("channel_side") or target.get("channelSide")
        channel_index = target.get("channel_index", target.get("channelIndex"))
        channel_position = target.get(
            "channel_position",
            target.get("channelPosition"),
        )
        try:
            parsed_channel_index = (
                int(channel_index) if channel_index is not None else None
            )
        except (TypeError, ValueError):
            parsed_channel_index = None
        try:
            parsed_channel_position = (
                int(channel_position) if channel_position is not None else None
            )
        except (TypeError, ValueError):
            parsed_channel_position = None
        confidence = float(
            candidate.get(
                "identity_confidence",
                candidate.get("confidence", 0.0),
            )
            or 0.0
        )
        return ReturnedPositionHint(
            product_id=product_id,
            product_idx=(
                str(candidate.get("product_idx"))
                if candidate.get("product_idx") is not None
                else None
            ),
            name=str(candidate.get("name") or product_id),
            unit_weight=round(unit_weight, 1),
            count=1,
            zone=int(session.zone),
            trigger_id=trigger.trigger_id,
            session_id=trigger.session_id,
            timestamp=float(trigger.timestamp),
            source="freezer_touch_return_candidate",
            confidence=confidence,
            reason="low_weight_touch_return_original_position",
            channel_side=str(channel_side) if channel_side is not None else None,
            channel_index=parsed_channel_index,
            channel_position=parsed_channel_position,
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
            self._ensure_trigger_placement_metadata(result, zone=zone)

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
            self._skipped_balanced_trigger_count = 0
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

            if self._active_sessions:
                first_session = next(iter(self._active_sessions.values()))
                self._reaggregate_products(first_session)
                self._apply_deferred_return_reconciliation()
                for active_session in self._active_sessions.values():
                    self._validate_net_delta(
                        active_session,
                        net_delta=self._calculate_effective_net_delta(active_session),
                    )
                self._apply_close_final_weight_validation()
                self._apply_freezer_close_aggregate()

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

    def _apply_freezer_close_aggregate(self) -> None:
        resolver = FreezerCloseAggregateResolver(
            get_product_weight=self._get_product_weight,
        )
        diagnostics = resolver.apply(self._active_sessions)
        if not diagnostics:
            return
        products_text = ", ".join(
            f"{product.get('name')}x{product.get('count')}"
            for product in diagnostics.get("selectedProducts", []) or []
            if isinstance(product, dict)
        ) or "none"
        suppressed_text = ", ".join(
            str(item.get("name") or item.get("productId"))
            for item in diagnostics.get("returnedPositionSuppressedCandidates", []) or []
            if isinstance(item, dict)
        ) or "none"
        ops_logger.info(
            "[OPS][FREEZER-CLOSE-AGGREGATE] accepted=%s reason=%s "
            "policy=%s output_zone=%s global_net_delta=%.1f "
            "final_target=%.1f selected_weight=%.1f residual=%.1f "
            "products=%s returned_suppressed=%s",
            diagnostics.get("accepted", False),
            diagnostics.get("reason", "unknown"),
            diagnostics.get("policy", "unknown"),
            diagnostics.get("outputZone", "n/a"),
            float(diagnostics.get("globalNetDelta", 0.0) or 0.0),
            float(diagnostics.get("finalTargetWeight", 0.0) or 0.0),
            float(diagnostics.get("selectedWeight", 0.0) or 0.0),
            float(diagnostics.get("residual", 0.0) or 0.0),
            products_text,
            suppressed_text,
        )

    def handle_close_signal(
        self,
        initial_wait_seconds: Optional[float] = None,
        subsequent_wait_seconds: Optional[float] = None,
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
        if initial_wait_seconds is None:
            initial_wait_seconds = config.door_session.close_initial_wait_seconds
        if subsequent_wait_seconds is None:
            subsequent_wait_seconds = config.door_session.close_subsequent_wait_seconds

        with self._lock:
            if self._global_session is None:
                logger.info("[CLOSE] No active global session")
                return True, None  # 세션 없음 → success 반환 가능

            # v5.2: 시간 주입 지원 (테스트용)
            now = _now if _now is not None else time.time()

            # v4.10: 큐에 대기 중인 trigger가 있으면 finalize 거부
            pending_snapshot = self._pending_trigger_snapshot_locked()
            pending_trigger_count = pending_snapshot["pendingTriggerCount"]
            pending_chargeable_vision_count = pending_snapshot[
                "pendingChargeableVisionCount"
            ]
            if pending_chargeable_vision_count > 0:
                logger.info(
                    f"[CLOSE][LATENCY] pending_triggers={pending_trigger_count}, "
                    f"pending_chargeable_vision={pending_chargeable_vision_count}, "
                    f"zones={pending_snapshot['pendingTriggerZones']}, "
                    f"session_ids={pending_snapshot['pendingTriggerSessionIds']}, "
                    "ready=false reason=pending_trigger"
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
                logger.info(
                    f"[CLOSE][LATENCY] trigger_wait_elapsed={elapsed_since_trigger:.1f}s "
                    f"wait_seconds={wait_seconds:.1f} "
                    f"ready={elapsed_since_trigger >= wait_seconds}"
                )

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
                logger.info(
                    f"[CLOSE][LATENCY] no_trigger_wait_elapsed={elapsed_since_close:.1f}s "
                    f"wait_seconds={wait_seconds:.1f} "
                    f"ready={elapsed_since_close >= wait_seconds}"
                )

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
        active door sessions의 상품 재집계 (내부용, lock 내에서 호출).

        v4.9: 크로스 존 반환 처리 추가
        v4.2: unmatched_returns 추적 추가

        Args:
            session: trigger가 추가된 DoorSession

        Returns:
            v4.9: 크로스 존으로 변경된 다른 zone 번호 목록
        """
        active_sessions = [
            self._active_sessions[zone]
            for zone in sorted(self._active_sessions.keys())
        ]
        defer_returns_until_close = self._global_session is not None

        # Cross-zone returns are recorded on the source session only, so any
        # reaggregation must rebuild every active session before replaying the
        # cross-zone repair pass.
        for active_session in active_sessions:
            result = self._aggregator.aggregate_with_unmatched(
                active_session.triggers,
                zone=active_session.zone,
            )
            active_session.aggregated_products = result.products
            active_session.returned_position_hints = (
                self._dedupe_returned_position_hints(
                    [
                        *result.returned_position_hints,
                        *self._collect_touch_return_hints(active_session),
                    ]
                )
            )
            if defer_returns_until_close:
                active_session.unmatched_returns = result.unmatched_returns
                active_session.deferred_returns = result.deferred_returns
            else:
                active_session.unmatched_returns = list(result.unmatched_returns)
                active_session.unmatched_returns.extend(
                    UnmatchedReturn(
                        trigger_id=deferred.trigger_id,
                        delta_weight=deferred.delta_weight,
                        timestamp=deferred.timestamp,
                        tolerance_used=deferred.tolerance_used,
                        channel_side=deferred.channel_side,
                        channel_index=deferred.channel_index,
                        channel_position=deferred.channel_position,
                        source_zone=deferred.source_zone,
                        source=deferred.source,
                    )
                    for deferred in result.deferred_returns
                )
                active_session.deferred_returns = []
            active_session.cross_zone_returns = []
            active_session.final_weight_validation = {}
            if result.location_return_diagnostics:
                active_session.final_weight_validation[
                    "freezerLocationReturnReconciliation"
                ] = {
                    "accepted": any(
                        bool(item.get("accepted"))
                        for item in result.location_return_diagnostics
                    ),
                    "source": "same_zone_reaggregation",
                    "items": result.location_return_diagnostics,
                }

            if self._get_product_weight is not None:
                self._aggregator.update_weights_from_db(
                    active_session.aggregated_products,
                    self._get_product_weight,
                )

        self._handle_cross_zone_returns()

        if not defer_returns_until_close:
            for active_session in active_sessions:
                self._validate_net_delta(
                    active_session,
                    net_delta=self._calculate_effective_net_delta(active_session),
                )

        return [
            active_session.zone
            for active_session in active_sessions
            if active_session.zone != session.zone
        ]

    def _calculate_effective_net_delta(self, session: DoorSession) -> float:
        """Return the basket-level delta after cross-zone return repair."""
        raw_delta = sum(trigger_effective_delta_weight(t) for t in session.triggers)
        unmatched_return_weight = unmatched_return_delta_weight(session)
        outgoing_cross_zone_weight = sum(
            record.delta_weight for record in session.cross_zone_returns
        )
        incoming_cross_zone_weight = sum(
            record.delta_weight
            for other_session in self._active_sessions.values()
            for record in other_session.cross_zone_returns
            if record.target_zone == session.zone
        )
        return (
            raw_delta
            - unmatched_return_weight
            - outgoing_cross_zone_weight
            + incoming_cross_zone_weight
        )

    def _validate_net_delta(
        self,
        session: DoorSession,
        net_delta: Optional[float] = None,
    ) -> None:
        """
        DoorSession의 net delta vs expected product weight 교차 검증 (v5.0).

        전량 반환 또는 부분 반환 시 aggregated_products 개수를 보정합니다.
        """
        # This pass is a safety net after trigger-level aggregation. It uses
        # the net door-session delta to repair counts when an item was removed
        # and later put back before the session closes.
        has_return_trigger = any(
            trigger.delta_weight > 0
            or trigger.is_return
            or bool(getattr(trigger, "return_weight_hints", None))
            for trigger in session.triggers
        )
        if not has_return_trigger:
            return

        if net_delta is None:
            net_delta = sum(t.delta_weight for t in session.triggers)

        active_products = [
            p for p in session.aggregated_products.values()
            if p.count > 0 and p.weight > 0
        ]
        expected_weight = sum(p.weight * p.count for p in active_products)

        if not active_products:
            return

        net_zero_threshold = min(15.0, max(3.0, expected_weight * 0.25))

        # Case 1: 전량 반환 (net_delta ≈ 0이거나 양수)
        if net_delta >= -net_zero_threshold and expected_weight > net_zero_threshold:
            logger.info(
                f"[복구 모드] 전량 반환 감지: net_delta={net_delta:.1f}g, "
                f"expected={expected_weight:.1f}g"
            )
            for p in active_products:
                logger.info(
                    f"[복구 모드] 복구된 상품: {p.name} x{p.count} ({p.weight}g)"
                )
                p.count = 0
                p.placement_units = []
            return

        # Case 2: 부분 반환 (net_delta < expected_weight)
        if abs(net_delta) < expected_weight - net_zero_threshold:
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
                    p.placement_units = list(p.placement_units)[:new_count]
                    corrected = True
                remaining -= p.weight * new_count
                if remaining <= net_zero_threshold:
                    break
            if corrected:
                total = sum(
                    p.count for p in session.aggregated_products.values()
                    if p.count > 0
                )
                logger.info(f"[복구 모드] 보정 완료: 총 {total}개")

    def _apply_deferred_return_reconciliation(self) -> None:
        """Replay deferred returns against the final close-time basket."""
        for zone in sorted(self._active_sessions.keys()):
            session = self._active_sessions[zone]
            deferred_returns = list(session.deferred_returns)
            diagnostics = self._build_deferred_return_diagnostics(session)
            if not deferred_returns:
                diagnostics["reason"] = "no_deferred_returns"
                session.final_weight_validation["deferredReturnReconciliation"] = (
                    diagnostics
                )
                continue

            remaining: List[DeferredReturn] = []
            target_weight = abs(self._calculate_effective_net_delta(session))
            for deferred in sorted(
                deferred_returns,
                key=lambda item: (float(item.timestamp), item.trigger_id),
            ):
                applied = self._apply_same_zone_deferred_return(
                    session,
                    deferred,
                    target_weight,
                    diagnostics,
                )
                if not applied:
                    remaining.append(deferred)

            still_remaining: List[DeferredReturn] = []
            for deferred in remaining:
                match = self._find_cross_zone_return_match(session.zone, deferred)
                if match is None:
                    still_remaining.append(deferred)
                    continue

                self._apply_cross_zone_return_match(session, deferred, match)
                diagnostics.setdefault("crossZoneApplied", []).append(
                    {
                        "triggerId": deferred.trigger_id,
                        "deltaWeight": round(float(deferred.delta_weight), 1),
                        "matchedWeight": round(float(match.total_weight), 1),
                        "matchedUnits": len(match.candidates),
                    }
                )

            target_weight = abs(self._calculate_effective_net_delta(session))
            current_weight = self._session_active_weight(session)
            current_residual = abs(target_weight - current_weight)
            diagnostics["currentWeightAfter"] = round(current_weight, 1)
            diagnostics["currentResidualAfter"] = round(current_residual, 1)
            diagnostics["accepted"] = bool(
                diagnostics.get("sameZoneApplied")
                or diagnostics.get("crossZoneApplied")
            )

            unresolved: List[DeferredReturn] = []
            for deferred in still_remaining:
                if current_residual <= self._weight_tolerance:
                    diagnostics.setdefault("notNeeded", []).append(
                        {
                            "triggerId": deferred.trigger_id,
                            "deltaWeight": round(float(deferred.delta_weight), 1),
                            "reason": "basket_matches_final_delta",
                        }
                    )
                    continue
                unresolved.append(deferred)
                session.unmatched_returns.append(
                    UnmatchedReturn(
                        trigger_id=deferred.trigger_id,
                        delta_weight=deferred.delta_weight,
                        timestamp=deferred.timestamp,
                        tolerance_used=deferred.tolerance_used,
                        channel_side=deferred.channel_side,
                        channel_index=deferred.channel_index,
                        channel_position=deferred.channel_position,
                        source_zone=deferred.source_zone,
                        source=deferred.source,
                    )
                )
                diagnostics.setdefault("unmatched", []).append(
                    {
                        "triggerId": deferred.trigger_id,
                        "deltaWeight": round(float(deferred.delta_weight), 1),
                        "source": deferred.source,
                    }
                )

            session.deferred_returns = unresolved
            if diagnostics["accepted"]:
                diagnostics["reason"] = "deferred_return_reconciled"
            elif diagnostics.get("unmatched"):
                diagnostics["reason"] = "deferred_return_unmatched"
            else:
                diagnostics["reason"] = diagnostics.get("reason", "not_needed")
            session.final_weight_validation["deferredReturnReconciliation"] = diagnostics

    def _build_deferred_return_diagnostics(
        self,
        session: DoorSession,
    ) -> Dict[str, object]:
        net_delta = self._calculate_effective_net_delta(session)
        current_weight = self._session_active_weight(session)
        target_weight = abs(net_delta)
        return {
            "accepted": False,
            "reason": "not_evaluated",
            "deferredCount": len(session.deferred_returns),
            "deferredWeight": round(session.deferred_returns_weight, 1),
            "targetWeight": round(target_weight, 1),
            "currentWeightBefore": round(current_weight, 1),
            "currentResidualBefore": round(abs(target_weight - current_weight), 1),
            "sameZoneApplied": [],
            "crossZoneApplied": [],
            "unmatched": [],
        }

    @staticmethod
    def _session_active_weight(session: DoorSession) -> float:
        return sum(
            product.weight * product.count
            for product in session.aggregated_products.values()
            if product.count > 0 and product.weight > 0
        )

    def _apply_same_zone_deferred_return(
        self,
        session: DoorSession,
        deferred: DeferredReturn,
        target_weight: float,
        diagnostics: Dict[str, object],
    ) -> bool:
        current_weight = self._session_active_weight(session)
        current_residual = abs(target_weight - current_weight)
        if current_residual <= self._weight_tolerance:
            diagnostics.setdefault("notNeeded", []).append(
                {
                    "triggerId": deferred.trigger_id,
                    "deltaWeight": round(float(deferred.delta_weight), 1),
                    "reason": "basket_matches_final_delta",
                }
            )
            return True
        if current_weight <= target_weight:
            return False

        combo = self._aggregator._weight_matcher.find_return_combination(
            session.aggregated_products,
            abs(float(deferred.delta_weight)),
        )
        if combo is None:
            return False

        combo_weight = 0.0
        combo_units = 0
        combo_names: List[str] = []
        for product_id, count in combo.items():
            product = session.aggregated_products.get(product_id)
            if product is None or product.count < count:
                return False
            combo_weight += product.weight * count
            combo_units += count
            combo_names.append(f"{product.name}x{count}")

        new_weight = current_weight - combo_weight
        new_residual = abs(target_weight - new_weight)
        if new_weight < -self._weight_tolerance:
            return False
        if new_residual >= current_residual:
            return False

        for product_id, count in combo.items():
            product = session.aggregated_products[product_id]
            selected_units = list(product.placement_units)[-count:]
            self._append_deferred_returned_position_hint(
                session,
                product=product,
                count=count,
                deferred=deferred,
                selected_units=selected_units,
            )
            product.count = max(0, product.count - count)
            product.placement_units = list(product.placement_units)[
                : max(0, product.count)
            ]

        diagnostics.setdefault("sameZoneApplied", []).append(
            {
                "triggerId": deferred.trigger_id,
                "deltaWeight": round(float(deferred.delta_weight), 1),
                "matchedWeight": round(combo_weight, 1),
                "matchedUnits": combo_units,
                "products": combo_names,
                "residualBefore": round(current_residual, 1),
                "residualAfter": round(new_residual, 1),
            }
        )
        return True

    def _append_deferred_returned_position_hint(
        self,
        session: DoorSession,
        *,
        product: AggregatedProduct,
        count: int,
        deferred: DeferredReturn,
        selected_units: List[Dict[str, object]],
    ) -> None:
        unit = selected_units[0] if selected_units else None
        channel_side: Optional[object] = deferred.channel_side
        channel_index: Optional[object] = deferred.channel_index
        channel_position: Optional[object] = deferred.channel_position
        if isinstance(unit, dict):
            channel_side = (
                channel_side
                or unit.get("channelSide")
                or unit.get("channel_side")
            )
            channel_index = (
                channel_index
                if channel_index is not None
                else unit.get("channelIndex", unit.get("channel_index"))
            )
            channel_position = (
                channel_position
                if channel_position is not None
                else unit.get("channelPosition", unit.get("channel_position"))
            )
        try:
            parsed_channel_index = (
                int(channel_index) if channel_index is not None else None
            )
        except (TypeError, ValueError):
            parsed_channel_index = None
        try:
            parsed_channel_position = (
                int(channel_position) if channel_position is not None else None
            )
        except (TypeError, ValueError):
            parsed_channel_position = None
        hint = ReturnedPositionHint(
            product_id=int(product.product_id),
            product_idx=product.product_idx,
            name=product.name,
            unit_weight=round(float(product.weight), 1),
            count=int(count),
            zone=deferred.source_zone
            if deferred.source_zone is not None
            else int(session.zone),
            trigger_id=deferred.trigger_id,
            session_id=None,
            timestamp=float(deferred.timestamp),
            source=deferred.source,
            confidence=float(product.average_confidence),
            reason="deferred_return_reconciled",
            channel_side=str(channel_side) if channel_side is not None else None,
            channel_index=parsed_channel_index,
            channel_position=parsed_channel_position,
        )
        session.returned_position_hints = self._dedupe_returned_position_hints(
            [*session.returned_position_hints, hint]
        )

    def _apply_close_final_weight_validation(self) -> None:
        """Repair over-fragmented close baskets with repeated candidate evidence."""
        for zone in sorted(self._active_sessions.keys()):
            session = self._active_sessions[zone]
            net_delta = self._calculate_effective_net_delta(session)
            replacement, diagnostics = self._select_close_final_weight_replacement(
                session,
                net_delta,
            )
            existing_deferred_diagnostics = session.final_weight_validation.get(
                "deferredReturnReconciliation"
            )
            if existing_deferred_diagnostics is not None:
                diagnostics["deferredReturnReconciliation"] = (
                    existing_deferred_diagnostics
                )
            existing_location_diagnostics = session.final_weight_validation.get(
                "freezerLocationReturnReconciliation"
            )
            if existing_location_diagnostics is not None:
                diagnostics["freezerLocationReturnReconciliation"] = (
                    existing_location_diagnostics
                )
            session.final_weight_validation = diagnostics
            if replacement is not None:
                session.aggregated_products = {replacement.product_id: replacement}
                logger.info(
                    "[CLOSE][FINAL_WEIGHT] corrected zone=%s product=%s count=%s "
                    "target=%.1fg current_residual=%.1fg replacement_residual=%.1fg",
                    zone,
                    replacement.name,
                    replacement.count,
                    diagnostics.get("targetWeight", 0.0),
                    diagnostics.get("currentResidual", 0.0),
                    diagnostics.get("replacementResidual", 0.0),
                )
            deferred_replacement, deferred_diagnostics = (
                self._select_deferred_close_candidate_repair(
                    session,
                    net_delta,
                )
            )
            if deferred_diagnostics:
                current_validation = dict(session.final_weight_validation or {})
                previous_reason = current_validation.get("reason")
                current_validation["deferredCandidateRepair"] = deferred_diagnostics
                if deferred_replacement is not None:
                    session.aggregated_products = {
                        deferred_replacement.product_id: deferred_replacement
                    }
                    current_validation.update(
                        {
                            "accepted": True,
                            "reason": "deferred_candidate_final_weight_correction",
                            "previousReason": previous_reason,
                            "targetWeight": deferred_diagnostics.get("targetWeight"),
                            "currentWeight": deferred_diagnostics.get("currentWeight"),
                            "currentResidual": deferred_diagnostics.get(
                                "currentResidual"
                            ),
                            "replacementWeight": deferred_diagnostics.get(
                                "replacementWeight"
                            ),
                            "replacementResidual": deferred_diagnostics.get(
                                "replacementResidual"
                            ),
                            "allowedResidual": deferred_diagnostics.get(
                                "allowedResidual"
                            ),
                            "selectedProduct": deferred_replacement.name,
                            "selectedProductId": int(deferred_replacement.product_id),
                            "selectedCount": int(deferred_replacement.count),
                            "candidateRank": deferred_diagnostics.get("candidateRank"),
                            "sourceZone": deferred_diagnostics.get("sourceZone"),
                            "sourceSessionId": deferred_diagnostics.get(
                                "sourceSessionId"
                            ),
                        }
                    )
                    logger.info(
                        "[CLOSE][CANDIDATE_REPAIR] corrected zone=%s product=%s "
                        "source_zone=%s target=%.1fg replacement_residual=%.1fg",
                        zone,
                        deferred_replacement.name,
                        deferred_diagnostics.get("sourceZone"),
                        deferred_diagnostics.get("targetWeight", 0.0),
                        deferred_diagnostics.get("replacementResidual", 0.0),
                    )
                session.final_weight_validation = current_validation
            self._apply_unresolved_close_weight_mismatch_guard(session, net_delta)

    def _apply_unresolved_close_weight_mismatch_guard(
        self,
        session: DoorSession,
        net_delta: float,
    ) -> None:
        """Exclude close baskets that still do not explain the full removal."""
        if net_delta >= 0:
            return
        blocker = self._close_final_weight_blocker(session)
        if blocker is not None:
            return

        active_products = [
            product
            for product in session.aggregated_products.values()
            if product.count > 0 and product.weight > 0
        ]
        if not active_products:
            return

        target_weight = abs(float(net_delta))
        current_weight = sum(
            float(product.weight) * int(product.count)
            for product in active_products
        )
        current_product_count = sum(int(product.count) for product in active_products)
        current_residual = abs(target_weight - current_weight)
        tolerance = self._close_full_delta_match_tolerance(current_product_count)
        if current_residual <= tolerance:
            return

        previous = dict(session.final_weight_validation or {})
        previous_reason = previous.get("reason")
        previous["accepted"] = False
        previous["reason"] = "unresolved_final_weight_mismatch"
        previous["previousReason"] = previous_reason
        previous["targetWeight"] = round(float(target_weight), 1)
        previous["currentWeight"] = round(float(current_weight), 1)
        previous["currentResidual"] = round(float(current_residual), 1)
        previous["allowedResidual"] = round(float(tolerance), 1)
        previous["currentProductCount"] = current_product_count
        product_payload = [
            {
                "productId": int(product.product_id),
                "name": product.name,
                "count": int(product.count),
                "unitWeight": round(float(product.weight), 1),
                "unitPrice": int(product.unit_price),
                "totalPrice": int(product.total_price),
            }
            for product in active_products
        ]
        preserve_products = str(config.machine.cabinet_type).lower() == "freezer"
        if preserve_products:
            previous["outputPolicy"] = "products_as_detected"
            previous["unresolvedProducts"] = product_payload
        else:
            previous["rejectedProducts"] = product_payload
        session.final_weight_validation = previous
        if not preserve_products:
            session.aggregated_products = {}
        logger.warning(
            "[CLOSE][FINAL_WEIGHT] unresolved mismatch zone=%s target=%.1fg "
            "current=%.1fg residual=%.1fg allowed=%.1fg",
            session.zone,
            target_weight,
            current_weight,
            current_residual,
            tolerance,
        )

    @staticmethod
    def _close_full_delta_match_tolerance(product_count: int) -> float:
        base_tolerance = max(0.0, float(config.weight.tolerance_grams))
        if product_count <= 1:
            return max(
                base_tolerance,
                float(config.weight.detected_single_fallback_tolerance_grams),
            )
        per_item_tolerance = max(
            0.0,
            float(config.weight.same_product_count_tolerance_grams),
        )
        return base_tolerance + per_item_tolerance * product_count

    def _select_close_final_weight_replacement(
        self,
        session: DoorSession,
        net_delta: float,
    ) -> Tuple[Optional[AggregatedProduct], Dict[str, object]]:
        base_tolerance = max(0.0, float(config.weight.tolerance_grams))
        per_item_tolerance = max(
            0.0,
            float(config.weight.same_product_count_tolerance_grams),
        )
        target_weight = abs(float(net_delta))
        active_products = [
            product
            for product in session.aggregated_products.values()
            if product.count > 0 and product.weight > 0
        ]
        current_weight = sum(product.weight * product.count for product in active_products)
        current_residual = abs(target_weight - current_weight)
        current_product_count = sum(product.count for product in active_products)
        removal_trigger_count = sum(
            1
            for trigger in session.triggers
            if trigger.delta_weight < 0 and not trigger.is_return
        )
        segment_count_cap = max(1, removal_trigger_count) * max(
            1,
            int(config.weight.max_items_per_segment),
        )
        diagnostics: Dict[str, object] = {
            "accepted": False,
            "reason": "not_evaluated",
            "targetWeight": round(target_weight, 1),
            "currentWeight": round(current_weight, 1),
            "currentResidual": round(current_residual, 1),
            "currentProductCount": current_product_count,
            "currentKindCount": len(active_products),
            "removalTriggerCount": removal_trigger_count,
            "segmentCountCap": segment_count_cap,
        }

        blocker = self._close_final_weight_blocker(session)
        if blocker is not None:
            diagnostics["reason"] = blocker
            return None, diagnostics

        if net_delta >= 0:
            diagnostics["reason"] = "non_removal_delta"
            return None, diagnostics
        if target_weight <= base_tolerance:
            diagnostics["reason"] = "target_within_tolerance"
            return None, diagnostics
        if not active_products:
            diagnostics["reason"] = "no_active_products"
            return None, diagnostics
        if len(active_products) <= 1 and current_product_count < 4:
            diagnostics["reason"] = "current_basket_not_overfragmented"
            return None, diagnostics

        candidate_groups = self._close_candidate_groups(session)
        diagnostics["candidateGroupCount"] = len(candidate_groups)
        if not candidate_groups:
            diagnostics["reason"] = "no_candidate_snapshot"
            return None, diagnostics

        viable = []
        rejected_candidates: List[Dict[str, object]] = []
        strong_candidate_ids = {
            product_id
            for product_id, group in candidate_groups.items()
            if self._close_group_has_strong_repeat_evidence(group)
        }
        current_ids = {product.product_id for product in active_products}
        current_has_unsupported_fragments = (
            self._close_current_has_unsupported_fragments(
                active_products,
                strong_candidate_ids,
            )
        )
        residual_gap_allowed = (
            base_tolerance + per_item_tolerance
            if current_has_unsupported_fragments
            else base_tolerance
        )
        diagnostics["currentHasUnsupportedFragments"] = (
            current_has_unsupported_fragments
        )
        diagnostics["residualGapAllowed"] = round(residual_gap_allowed, 1)

        for product_id, group in candidate_groups.items():
            unit_weight = float(group["unit_weight"])
            unit_price = int(group["unit_price"])
            stock_qty = int(group["stock_qty"])
            if unit_weight <= 0 or unit_price <= 0:
                continue

            estimated_count = int(round(target_weight / unit_weight))
            if estimated_count < 2:
                continue
            count_cap_values = [
                max(1, int(config.weight.same_product_max_count)),
                max(1, int(config.weight.max_count_per_item)),
                segment_count_cap,
            ]
            if stock_qty > 0:
                count_cap_values.append(stock_qty)
            close_repeat_count_cap = min(count_cap_values)
            if estimated_count > close_repeat_count_cap:
                rejected_candidates.append(
                    {
                        "productId": int(product_id),
                        "name": str(group.get("name", product_id)),
                        "estimatedCount": estimated_count,
                        "closeRepeatCountCap": close_repeat_count_cap,
                        "reason": "count_exceeds_close_repeat_cap",
                    }
                )
                continue

            replacement_weight = unit_weight * estimated_count
            replacement_residual = abs(target_weight - replacement_weight)
            allowed_residual = base_tolerance + (
                per_item_tolerance * estimated_count
            )
            if replacement_residual > allowed_residual:
                continue
            residual_gap = replacement_residual - current_residual
            if residual_gap > residual_gap_allowed:
                continue
            if not self._close_group_has_strong_repeat_evidence(group):
                continue

            viable.append(
                {
                    **group,
                    "product_id": product_id,
                    "count": estimated_count,
                    "replacement_weight": replacement_weight,
                    "replacement_residual": replacement_residual,
                    "allowed_residual": allowed_residual,
                    "residual_gap": residual_gap,
                    "residual_gap_allowed": residual_gap_allowed,
                    "close_repeat_count_cap": close_repeat_count_cap,
                }
            )

        if not viable:
            diagnostics["reason"] = "no_viable_repeat_candidate"
            if rejected_candidates:
                diagnostics["rejectedCandidates"] = rejected_candidates
            return None, diagnostics

        viable.sort(
            key=lambda item: (
                -int(item["distinct_trigger_count"]),
                -int(item["regular_count"]),
                int(item["best_regular_rank"] or item["best_rank"] or 999),
                -float(item["best_confidence"]),
                float(item["replacement_residual"]),
            )
        )
        selected = viable[0]
        selected_product_id = int(selected["product_id"])

        if (
            current_ids
            and current_ids.issubset(strong_candidate_ids)
            and current_ids != {selected_product_id}
        ):
            diagnostics.update(
                {
                    "reason": "clean_supported_basket_preferred",
                    "identitySwapBlocked": True,
                    "selectedProductId": selected_product_id,
                    "currentProductIds": sorted(int(product_id) for product_id in current_ids),
                    "replacementWeight": round(
                        float(selected["replacement_weight"]),
                        1,
                    ),
                    "replacementResidual": round(
                        float(selected["replacement_residual"]),
                        1,
                    ),
                }
            )
            return None, diagnostics

        if (
            current_ids
            and current_ids.issubset(strong_candidate_ids)
            and current_residual <= float(selected["replacement_residual"])
        ):
            diagnostics.update(
                {
                    "reason": "clean_supported_basket_preferred",
                    "identitySwapBlocked": False,
                    "replacementWeight": round(
                        float(selected["replacement_weight"]),
                        1,
                    ),
                    "replacementResidual": round(
                        float(selected["replacement_residual"]),
                        1,
                    ),
                }
            )
            return None, diagnostics

        replacement = AggregatedProduct(
            product_id=int(selected["product_id"]),
            product_idx=selected.get("product_idx"),
            name=str(selected["name"]),
            count=int(selected["count"]),
            unit_price=int(selected["unit_price"]),
            weight=float(selected["unit_weight"]),
            total_confidence=float(selected["average_confidence"])
            * int(selected["count"]),
            detection_count=int(selected["count"]),
        )
        diagnostics.update(
            {
                "accepted": True,
                "reason": "candidate_repeat_final_weight_correction",
                "replacementWeight": round(float(selected["replacement_weight"]), 1),
                "replacementResidual": round(
                    float(selected["replacement_residual"]),
                    1,
                ),
                "allowedResidual": round(float(selected["allowed_residual"]), 1),
                "residualGap": round(float(selected["residual_gap"]), 1),
                "residualGapAllowed": round(
                    float(selected["residual_gap_allowed"]),
                    1,
                ),
                "currentHasUnsupportedFragments": current_has_unsupported_fragments,
                "selectedProduct": selected["name"],
                "selectedProductId": int(selected["product_id"]),
                "selectedCount": int(selected["count"]),
                "closeRepeatCountCap": int(selected["close_repeat_count_cap"]),
                "candidateRank": int(selected["best_rank"]),
                "regularCandidateCount": int(selected["regular_count"]),
                "distinctTriggerCount": int(selected["distinct_trigger_count"]),
            }
        )
        return replacement, diagnostics

    def _select_deferred_close_candidate_repair(
        self,
        session: DoorSession,
        net_delta: float,
    ) -> Tuple[Optional[AggregatedProduct], Dict[str, object]]:
        """Use later unused freezer candidates to repair no-charge/mismatch closes."""
        if str(config.machine.cabinet_type).lower() != "freezer":
            return None, {}
        if net_delta >= 0:
            return None, {}

        blocker = self._close_final_weight_blocker(session)
        if blocker is not None:
            return None, {}

        target_weight = abs(float(net_delta))
        base_tolerance = max(0.0, float(config.weight.tolerance_grams))
        if target_weight <= base_tolerance:
            return None, {}

        removal_triggers = [
            trigger
            for trigger in session.triggers
            if trigger_effective_delta_weight(trigger) < 0 and not trigger.is_return
        ]
        active_products = [
            product
            for product in session.aggregated_products.values()
            if product.count > 0 and product.weight > 0
        ]
        current_product_count = sum(int(product.count) for product in active_products)
        current_weight = sum(
            float(product.weight) * int(product.count)
            for product in active_products
        )
        current_residual = abs(target_weight - current_weight)
        allowed_residual = self._close_full_delta_match_tolerance(
            max(1, current_product_count)
        )

        mismatch_eligible = not active_products or current_residual > allowed_residual
        if not mismatch_eligible:
            return None, {}

        diagnostics: Dict[str, object] = {
            "applied": False,
            "reason": "not_evaluated",
            "targetWeight": round(target_weight, 1),
            "currentWeight": round(current_weight, 1),
            "currentResidual": round(current_residual, 1),
            "allowedResidual": round(float(allowed_residual), 1),
            "currentProductCount": current_product_count,
            "currentProducts": [
                {
                    "productId": int(product.product_id),
                    "name": product.name,
                    "count": int(product.count),
                    "unitWeight": round(float(product.weight), 1),
                }
                for product in active_products
            ],
        }

        if len(removal_triggers) != 1:
            diagnostics["reason"] = "requires_single_removal_trigger"
            diagnostics["removalTriggerCount"] = len(removal_triggers)
            return None, diagnostics

        target_trigger = removal_triggers[0]
        diagnostics["targetTriggerId"] = target_trigger.trigger_id
        diagnostics["targetSessionId"] = target_trigger.session_id
        diagnostics["eligibility"] = (
            "no_active_products" if not active_products else "final_weight_mismatch"
        )

        candidates, rejected = self._deferred_close_candidate_repair_candidates(
            target_session=session,
            target_trigger=target_trigger,
            target_weight=target_weight,
        )
        diagnostics["candidateCount"] = len(candidates)
        if rejected:
            diagnostics["rejectedCandidates"] = rejected[:10]
        if not candidates:
            diagnostics["reason"] = "no_later_unused_weight_match"
            return None, diagnostics

        candidates.sort(
            key=lambda item: (
                float(item["replacement_residual"]),
                int(item["rank"]),
                -float(item["confidence"]),
                float(item["source_timestamp"]),
                int(item["source_zone"]),
                int(item["product_id"]),
            )
        )
        selected = candidates[0]
        replacement = AggregatedProduct(
            product_id=int(selected["product_id"]),
            product_idx=selected.get("product_idx"),
            name=str(selected["name"]),
            count=1,
            unit_price=int(selected["unit_price"]),
            weight=float(selected["unit_weight"]),
            total_confidence=float(selected["confidence"]),
            detection_count=1,
        )
        diagnostics.update(
            {
                "applied": True,
                "reason": "later_unused_candidate_weight_match",
                "replacementWeight": round(float(selected["unit_weight"]), 1),
                "replacementResidual": round(
                    float(selected["replacement_residual"]),
                    1,
                ),
                "selectedProduct": selected["name"],
                "selectedProductId": int(selected["product_id"]),
                "selectedCount": 1,
                "candidateRank": int(selected["rank"]),
                "candidateConfidence": round(float(selected["confidence"]), 4),
                "candidateSource": str(selected["source"]),
                "sourceZone": int(selected["source_zone"]),
                "sourceTriggerId": selected["source_trigger_id"],
                "sourceSessionId": selected["source_session_id"],
            }
        )
        return replacement, diagnostics

    def _deferred_close_candidate_repair_candidates(
        self,
        *,
        target_session: DoorSession,
        target_trigger: TriggerResult,
        target_weight: float,
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        candidates: List[Dict[str, object]] = []
        rejected: List[Dict[str, object]] = []
        allowed_residual = self._close_full_delta_match_tolerance(1)
        target_timestamp = float(target_trigger.timestamp)

        for source_zone in sorted(self._active_sessions.keys()):
            source_session = self._active_sessions[source_zone]
            for source_trigger in source_session.triggers:
                if source_trigger is target_trigger:
                    continue
                if float(source_trigger.timestamp) <= target_timestamp:
                    continue
                if trigger_effective_delta_weight(source_trigger) >= 0:
                    continue
                if source_trigger.is_return:
                    continue

                consumed_ids = self._close_consumed_product_ids_for_trigger(
                    source_trigger
                )
                for candidate in getattr(source_trigger, "vision_candidates", []) or []:
                    if not isinstance(candidate, dict):
                        continue
                    candidate_entry = self._deferred_close_candidate_entry(
                        candidate=candidate,
                        source_zone=source_zone,
                        source_trigger=source_trigger,
                        target_weight=target_weight,
                        allowed_residual=allowed_residual,
                        consumed_ids=consumed_ids,
                    )
                    if candidate_entry.get("accepted"):
                        candidates.append(candidate_entry["candidate"])
                    else:
                        rejected.append(candidate_entry["rejection"])

        return candidates, rejected

    def _deferred_close_candidate_entry(
        self,
        *,
        candidate: Dict[str, object],
        source_zone: int,
        source_trigger: TriggerResult,
        target_weight: float,
        allowed_residual: float,
        consumed_ids: set[int],
    ) -> Dict[str, Any]:
        try:
            product_id = int(candidate.get("product_id"))
            unit_weight = float(candidate.get("unit_weight", 0.0) or 0.0)
        except (TypeError, ValueError):
            return {
                "accepted": False,
                "rejection": {"reason": "invalid_candidate_identity"},
            }

        name = str(candidate.get("name") or product_id)
        base_rejection = {
            "productId": product_id,
            "name": name,
            "sourceZone": int(source_zone),
            "sourceSessionId": source_trigger.session_id,
            "rank": int(candidate.get("rank", 999) or 999),
        }
        if product_id in consumed_ids:
            return {
                "accepted": False,
                "rejection": {**base_rejection, "reason": "consumed_by_source_result"},
            }
        try:
            unit_price = int(candidate.get("unit_price", 0) or 0)
            stock_qty = int(candidate.get("stock_qty", 0) or 0)
            confidence = float(candidate.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            return {
                "accepted": False,
                "rejection": {**base_rejection, "reason": "invalid_candidate_metadata"},
            }
        if unit_weight <= 0 or unit_price <= 0 or stock_qty <= 0:
            return {
                "accepted": False,
                "rejection": {**base_rejection, "reason": "inactive_candidate_product"},
            }

        residual = abs(target_weight - unit_weight)
        if residual > allowed_residual:
            return {
                "accepted": False,
                "rejection": {
                    **base_rejection,
                    "reason": "weight_residual_exceeds_tolerance",
                    "unitWeight": round(unit_weight, 1),
                    "weightResidual": round(residual, 1),
                    "allowedResidual": round(float(allowed_residual), 1),
                },
            }

        return {
            "accepted": True,
            "candidate": {
                "product_id": product_id,
                "product_idx": candidate.get("product_idx"),
                "name": name,
                "unit_weight": unit_weight,
                "unit_price": unit_price,
                "stock_qty": stock_qty,
                "confidence": confidence,
                "rank": int(candidate.get("rank", 999) or 999),
                "source": str(candidate.get("source", "vision") or "vision"),
                "replacement_residual": residual,
                "source_zone": int(source_zone),
                "source_trigger_id": source_trigger.trigger_id,
                "source_session_id": source_trigger.session_id,
                "source_timestamp": float(source_trigger.timestamp),
            },
        }

    def _close_consumed_product_ids_for_trigger(
        self,
        trigger: TriggerResult,
    ) -> set[int]:
        products = [product for product in trigger.products if product.count > 0]
        if not products:
            return set()

        candidate_weights: Dict[int, float] = {}
        for candidate in getattr(trigger, "vision_candidates", []) or []:
            if not isinstance(candidate, dict):
                continue
            try:
                product_id = int(candidate.get("product_id"))
                unit_weight = float(candidate.get("unit_weight", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if product_id >= 0 and unit_weight > 0:
                candidate_weights[product_id] = unit_weight

        total_weight = 0.0
        product_count = 0
        consumed_ids: set[int] = set()
        for product in products:
            product_id = int(product.product_id)
            unit_weight = candidate_weights.get(product_id)
            if unit_weight is None and self._get_product_weight is not None:
                try:
                    unit_weight = float(self._get_product_weight(product_id))
                except (TypeError, ValueError):
                    unit_weight = None
            if unit_weight is None or unit_weight <= 0:
                return set()
            count = int(product.count)
            total_weight += unit_weight * count
            product_count += count
            consumed_ids.add(product_id)

        target_weight = abs(float(trigger_effective_delta_weight(trigger)))
        tolerance = self._close_full_delta_match_tolerance(product_count)
        if abs(target_weight - total_weight) <= tolerance:
            return consumed_ids
        return set()

    def _close_final_weight_blocker(self, session: DoorSession) -> Optional[str]:
        if session.unmatched_returns:
            return "unmatched_return_present"
        if session.cross_zone_returns:
            return "outgoing_cross_zone_return_present"
        for other_session in self._active_sessions.values():
            if any(
                record.target_zone == session.zone
                for record in other_session.cross_zone_returns
            ):
                return "incoming_cross_zone_return_present"
        return None

    def _close_candidate_groups(
        self,
        session: DoorSession,
    ) -> Dict[int, Dict[str, object]]:
        groups: Dict[int, Dict[str, object]] = {}
        for trigger_index, trigger in enumerate(session.triggers):
            if trigger.delta_weight >= 0 or trigger.is_return:
                continue
            for candidate in getattr(trigger, "vision_candidates", []) or []:
                if not isinstance(candidate, dict):
                    continue
                try:
                    product_id = int(candidate.get("product_id"))
                    unit_weight = float(candidate.get("unit_weight", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                if product_id < 0 or unit_weight <= 0:
                    continue

                group = groups.setdefault(
                    product_id,
                    {
                        "name": str(candidate.get("name") or product_id),
                        "product_idx": candidate.get("product_idx"),
                        "unit_weight": unit_weight,
                        "unit_price": int(candidate.get("unit_price", 0) or 0),
                        "stock_qty": int(candidate.get("stock_qty", 0) or 0),
                        "best_rank": 999,
                        "best_regular_rank": None,
                        "best_confidence": 0.0,
                        "confidence_sum": 0.0,
                        "candidate_count": 0,
                        "regular_count": 0,
                        "trigger_indexes": set(),
                    },
                )
                rank = int(candidate.get("rank", 999) or 999)
                confidence = float(candidate.get("confidence", 0.0) or 0.0)
                source = str(candidate.get("source", "vision") or "vision")
                group["best_rank"] = min(int(group["best_rank"]), rank)
                group["best_confidence"] = max(
                    float(group["best_confidence"]),
                    confidence,
                )
                group["confidence_sum"] = float(group["confidence_sum"]) + confidence
                group["candidate_count"] = int(group["candidate_count"]) + 1
                group["trigger_indexes"].add(trigger_index)
                if source == "vision":
                    group["regular_count"] = int(group["regular_count"]) + 1
                    current_best = group["best_regular_rank"]
                    group["best_regular_rank"] = (
                        rank
                        if current_best is None
                        else min(int(current_best), rank)
                    )

        for group in groups.values():
            candidate_count = max(1, int(group["candidate_count"]))
            group["average_confidence"] = (
                float(group["confidence_sum"]) / candidate_count
            )
            group["distinct_trigger_count"] = len(group["trigger_indexes"])
            group.pop("trigger_indexes", None)
        return groups

    @staticmethod
    def _close_group_has_strong_repeat_evidence(group: Dict[str, object]) -> bool:
        if int(group.get("regular_count", 0) or 0) <= 0:
            return False
        if int(group.get("distinct_trigger_count", 0) or 0) >= 2:
            return True
        best_regular_rank = group.get("best_regular_rank")
        if best_regular_rank is None:
            return False
        return (
            int(best_regular_rank) <= 3
            and float(group.get("best_confidence", 0.0) or 0.0) >= 0.25
        )

    @staticmethod
    def _close_current_has_unsupported_fragments(
        active_products: List[AggregatedProduct],
        strong_candidate_ids: set[int],
    ) -> bool:
        if len(active_products) <= 1:
            return False
        for product in active_products:
            if int(product.product_id) in strong_candidate_ids:
                continue
            if float(product.weight) < 200.0 or int(product.count) > 1:
                return True
        return False

    def _handle_cross_zone_returns(self) -> None:
        """
        크로스 존 반환 처리 (v4.9).

        각 zone에서 매칭 실패한 반환(unmatched_returns)을 다른 zone들의
        aggregated_products 조합으로 무게 매칭 시도.

        Lock 내에서 호출됨 - 다른 zone 세션 접근 안전함.
        """
        # Same-zone recovery runs first inside ProductAggregator. Only returns
        # that are still unmatched reach this cross-zone reconciliation pass.
        for zone in sorted(self._active_sessions.keys()):
            session = self._active_sessions[zone]
            if not session.unmatched_returns:
                continue

            still_unmatched: List[UnmatchedReturn] = []

            for unmatched in session.unmatched_returns:
                match = self._find_cross_zone_return_match(session.zone, unmatched)
                if match is None:
                    still_unmatched.append(unmatched)
                    continue

                self._apply_cross_zone_return_match(session, unmatched, match)
                diagnostics = session.final_weight_validation.setdefault(
                    "freezerLocationReturnReconciliation",
                    {
                        "accepted": False,
                        "source": "cross_zone_reconciliation",
                        "items": [],
                    },
                )
                if isinstance(diagnostics, dict):
                    diagnostics["accepted"] = True
                    diagnostics.setdefault("items", []).append(
                        {
                            "triggerId": unmatched.trigger_id,
                            "sourceZone": session.zone,
                            "channelSide": getattr(unmatched, "channel_side", None),
                            "targetWeight": round(
                                abs(float(unmatched.delta_weight)),
                                1,
                            ),
                            "matchTier": (
                                "other_zone_same_side"
                                if getattr(unmatched, "channel_side", None)
                                and all(
                                    candidate.channel_side == unmatched.channel_side
                                    for candidate in match.candidates
                                )
                                else "other_zone_any_side"
                            ),
                            "matchedProducts": [
                                {
                                    "zone": candidate.target_zone,
                                    "productId": candidate.product_id,
                                    "name": candidate.product_name,
                                    "channelSide": candidate.channel_side,
                                    "weight": round(candidate.unit_weight, 1),
                                }
                                for candidate in match.candidates
                            ],
                            "matchedWeight": round(float(match.total_weight), 1),
                            "residual": round(float(match.weight_error), 1),
                        }
                    )

            session.unmatched_returns = still_unmatched

    @staticmethod
    def _placement_channel_side(unit: object) -> Optional[str]:
        if not isinstance(unit, dict):
            return None
        raw_side = unit.get("channelSide") or unit.get("channel_side")
        return str(raw_side) if raw_side is not None else None

    @staticmethod
    def _placement_source_timestamp(unit: object) -> float:
        if not isinstance(unit, dict):
            return 0.0
        try:
            return float(unit.get("sourceTimestamp", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _build_cross_zone_return_candidates(
        self,
        source_zone: int,
        *,
        channel_side: Optional[str] = None,
    ) -> List[_CrossZoneReturnCandidate]:
        candidates: List[_CrossZoneReturnCandidate] = []
        max_count_per_item = max(1, int(config.weight.max_count_per_item))

        for target_zone in sorted(self._active_sessions.keys()):
            if target_zone == source_zone:
                continue
            target_session = self._active_sessions[target_zone]
            for product_id in sorted(target_session.aggregated_products.keys()):
                product = target_session.aggregated_products[product_id]
                if product.count <= 0 or product.weight <= 0:
                    continue
                placement_units = [
                    unit
                    for unit in product.placement_units[: product.count]
                    if isinstance(unit, dict)
                ]
                if placement_units:
                    for placement_index, unit in enumerate(
                        placement_units[:max_count_per_item]
                    ):
                        unit_side = self._placement_channel_side(unit)
                        if channel_side and unit_side != channel_side:
                            continue
                        candidates.append(
                            _CrossZoneReturnCandidate(
                                target_zone=target_zone,
                                product_id=product_id,
                                product_name=product.name,
                                unit_weight=product.weight,
                                unit_index=placement_index,
                                session_created_at=target_session.created_at,
                                channel_side=unit_side,
                                placement_index=placement_index,
                                placement_unit=unit,
                                source_timestamp=self._placement_source_timestamp(unit),
                            )
                        )
                    continue

                for unit_index in range(min(product.count, max_count_per_item)):
                    candidates.append(
                        _CrossZoneReturnCandidate(
                            target_zone=target_zone,
                            product_id=product_id,
                            product_name=product.name,
                            unit_weight=product.weight,
                            unit_index=unit_index,
                            session_created_at=target_session.created_at,
                            channel_side=None,
                            placement_index=None,
                            placement_unit=None,
                        )
                    )

        return sorted(
            candidates,
            key=lambda candidate: (
                -candidate.unit_weight,
                -candidate.source_timestamp,
                candidate.session_created_at,
                candidate.target_zone,
                candidate.product_id,
                candidate.unit_index,
            ),
        )

    def _cross_zone_match_allowed_error(self, unit_count: int) -> float:
        return min(15.0, max(self._weight_tolerance, unit_count * self._weight_tolerance))

    def _cross_zone_match_key(self, match: _CrossZoneReturnMatch) -> tuple:
        stable_keys = tuple(
            sorted(candidate.stable_key for candidate in match.candidates)
        )
        kind_count = len(
            {
                (candidate.target_zone, candidate.product_id)
                for candidate in match.candidates
            }
        )
        latest_timestamp = max(
            (candidate.source_timestamp for candidate in match.candidates),
            default=0.0,
        )
        return (
            len(match.candidates),
            match.weight_error,
            -latest_timestamp,
            kind_count,
            stable_keys,
        )

    def _find_cross_zone_return_match(
        self,
        source_zone: int,
        unmatched: UnmatchedReturn,
    ) -> Optional[_CrossZoneReturnMatch]:
        target_weight = abs(unmatched.delta_weight)
        if target_weight <= 0:
            return None

        max_units = max(
            1,
            int(config.weight.max_count_per_item),
            int(config.weight.max_combination_size),
            int(config.weight.max_combination_items),
        )
        max_kinds = max(1, int(config.weight.max_combination_kinds))
        max_visits = max(1000, int(config.weight.max_combinations) * 50)
        max_possible_error = self._cross_zone_match_allowed_error(max_units)
        attempted_candidate_count = 0

        def search_candidates(
            candidates: List[_CrossZoneReturnCandidate],
        ) -> Optional[_CrossZoneReturnMatch]:
            best_match: Optional[_CrossZoneReturnMatch] = None
            visits = 0

            def consider(
                chosen: List[_CrossZoneReturnCandidate],
                total: float,
            ) -> None:
                nonlocal best_match
                if not chosen:
                    return
                error = abs(total - target_weight)
                if error > self._cross_zone_match_allowed_error(len(chosen)):
                    return
                match = _CrossZoneReturnMatch(
                    candidates=list(chosen),
                    total_weight=total,
                    weight_error=error,
                )
                if (
                    best_match is None
                    or self._cross_zone_match_key(match)
                    < self._cross_zone_match_key(best_match)
                ):
                    best_match = match

            def search(
                start_idx: int,
                chosen: List[_CrossZoneReturnCandidate],
                total: float,
            ) -> None:
                nonlocal visits
                if visits >= max_visits:
                    return
                visits += 1

                consider(chosen, total)

                if len(chosen) >= max_units:
                    return
                if total > target_weight + max_possible_error:
                    return

                current_kinds = {
                    (candidate.target_zone, candidate.product_id)
                    for candidate in chosen
                }
                for index in range(start_idx, len(candidates)):
                    candidate = candidates[index]
                    next_kinds = set(current_kinds)
                    next_kinds.add((candidate.target_zone, candidate.product_id))
                    if len(next_kinds) > max_kinds:
                        continue
                    chosen.append(candidate)
                    search(index + 1, chosen, total + candidate.unit_weight)
                    chosen.pop()

            search(0, [], 0.0)
            return best_match

        source_side = getattr(unmatched, "channel_side", None)
        tiers: List[tuple[str, Optional[str]]] = []
        if source_side:
            tiers.append(("other_zone_same_side", str(source_side)))
            tiers.append(("other_zone_any_side", None))
        else:
            tiers.append(("other_zone_any_side", None))

        best_match: Optional[_CrossZoneReturnMatch] = None
        for _tier_name, side in tiers:
            candidates = self._build_cross_zone_return_candidates(
                source_zone,
                channel_side=side,
            )
            attempted_candidate_count += len(candidates)
            if not candidates:
                continue
            best_match = search_candidates(candidates)
            if best_match is not None:
                break

        if best_match is None:
            logger.warning(
                f"Cross-zone return matching failed: zone={source_zone}, "
                f"delta={unmatched.delta_weight:.1f}g, "
                f"candidates={attempted_candidate_count}"
            )
            return None

        logger.info(
            f"Cross-zone return combination matched: zone={source_zone}, "
            f"delta={unmatched.delta_weight:.1f}g, "
            f"matched={best_match.total_weight:.1f}g, "
            f"error={best_match.weight_error:.1f}g, "
            f"units={len(best_match.candidates)}"
        )
        return best_match

    def _apply_cross_zone_return_match(
        self,
        source_session: DoorSession,
        unmatched: UnmatchedReturn,
        match: _CrossZoneReturnMatch,
    ) -> None:
        total_weight = sum(candidate.unit_weight for candidate in match.candidates)
        if total_weight <= 0:
            return

        allocated_delta = 0.0
        sorted_candidates = sorted(
            match.candidates,
            key=lambda candidate: candidate.stable_key,
        )
        for index, candidate in enumerate(sorted_candidates):
            target_session = self._active_sessions.get(candidate.target_zone)
            if target_session is None:
                continue
            product = target_session.aggregated_products.get(candidate.product_id)
            if product is None or product.count <= 0:
                continue

            product.count -= 1
            if candidate.placement_unit is not None:
                remaining_units = list(product.placement_units)
                for unit_index, unit in enumerate(remaining_units):
                    if unit == candidate.placement_unit:
                        remaining_units.pop(unit_index)
                        break
                product.placement_units = remaining_units
            else:
                product.placement_units = list(product.placement_units)[
                    : max(0, product.count)
                ]
            if index == len(sorted_candidates) - 1:
                component_delta = unmatched.delta_weight - allocated_delta
            else:
                component_delta = unmatched.delta_weight * (
                    candidate.unit_weight / total_weight
                )
                allocated_delta += component_delta

            record = CrossZoneReturn(
                trigger_id=unmatched.trigger_id,
                source_zone=source_session.zone,
                target_zone=candidate.target_zone,
                product_id=candidate.product_id,
                product_name=candidate.product_name,
                matched_weight=candidate.unit_weight,
                delta_weight=component_delta,
                timestamp=time.time(),
            )
            source_session.cross_zone_returns.append(record)

        product_summary: Dict[tuple[int, int, str], int] = {}
        for candidate in sorted_candidates:
            key = (
                candidate.target_zone,
                candidate.product_id,
                candidate.product_name,
            )
            product_summary[key] = product_summary.get(key, 0) + 1
        summary_text = ", ".join(
            f"zone {zone}:{name}x{count}"
            for (zone, _product_id, name), count in sorted(product_summary.items())
        )
        logger.info(
            f"Cross-zone return applied: zone {source_session.zone} -> "
            f"{summary_text}, delta={unmatched.delta_weight:.1f}g"
        )

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

            pending_snapshot = self._pending_trigger_snapshot_locked()
            return {
                "active_sessions": len(self._active_sessions),
                "active_zones": active_zones,
                "session_timeout": self._session_timeout,
                "weight_tolerance": self._weight_tolerance,
                "max_duration": self._max_duration,
                "global_session_max_duration": self._global_session_max_duration,
                "global_session_active": self._global_session is not None,
                "global_session": global_session_info,
                "pending_trigger_count": pending_snapshot["pendingTriggerCount"],
                "pending_trigger_zones": pending_snapshot["pendingTriggerZones"],
                "pending_trigger_session_ids": pending_snapshot["pendingTriggerSessionIds"],
                "pending_trigger_statuses": pending_snapshot["pendingTriggerStatuses"],
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
