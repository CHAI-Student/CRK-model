from __future__ import annotations

"""
Trigger Service (v4.10).

트리거 비즈니스 로직 - YOLO 추론 및 세션 저장.
라우터에서 분리된 핵심 비즈니스 로직.

v4.10 변경사항:
- asyncio.Queue 기반 순차 처리 워커 추가
- enqueue_trigger(): 즉시 "queued" 응답 반환, 백그라운드 워커에서 순차 처리
- CLOSE 신호 race condition 방지 (notify_trigger_enqueued/processed)
- Jetson 4GB 단일 GPU에서 TensorRT 동시 추론 충돌 방지

v4.7 변경사항:
- ActiveProductStore 상품 정보를 engine.judge()에 전달
- count_calculator가 active_products를 우선 사용하여 stock 필터링 문제 해결
- DEFAULT_PRODUCTS fallback 방지

v4.6 변경사항:
- 무게 변화 5g 이하면 비디오 처리 스킵 (불필요한 YOLO 추론 방지)
- Node.js 상품 리스트에 없는 상품 제거 (최종 필터링)
- product_weights 전달하여 로그 개선

v4.5 변경사항:
- Idempotency key 기반 중복 체크 (5초 이내 동일 요청 스킵)
- PendingTriggerStore 관련 기능 제거
- 전역 상품 리스트로 YOLO 필터링 (zone별 관리 제거)
"""

import asyncio
import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Dict, List, Optional, Tuple

import model_service.core.loadcell_stats as loadcell_stats
from model_service.video import VideoProcessor, VoteResult
from model_service.video.frame_trace import TriggerTraceContext
from model_service.engine import ProductDecisionEngine, EnsembleResult
from model_service.session import SessionStore, SessionData, ProductResult, DoorSessionStore, TriggerResult
from model_service.session.session_store import generate_session_id
from model_service.session.active_product_store import ActiveProductStore
from model_service.core.config import config

logger = logging.getLogger(__name__)


@dataclass
class LoadcellReading:
    """로드셀 읽기 데이터."""
    timestamp: str
    raw_value: List[str]
    filtered_value: List[str]
    filter_method: str = "none"


@dataclass
class TriggerTimingMetadata:
    """Camera-side timing metadata attached to a trigger payload."""

    capture_started_at: Optional[str] = None
    capture_ended_at: Optional[str] = None
    loadcell_started_at: Optional[str] = None
    loadcell_ended_at: Optional[str] = None
    trigger_started_at: Optional[str] = None
    trigger_end_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            key: value
            for key, value in {
                "capture_started_at": self.capture_started_at,
                "capture_ended_at": self.capture_ended_at,
                "loadcell_started_at": self.loadcell_started_at,
                "loadcell_ended_at": self.loadcell_ended_at,
                "trigger_started_at": self.trigger_started_at,
                "trigger_end_reason": self.trigger_end_reason,
            }.items()
            if value is not None
        }


@dataclass
class TriggerInput:
    """트리거 입력 데이터."""
    zone: int
    loadcells: List[LoadcellReading]
    top_video_path: Optional[str]
    side_video_path: Optional[str]
    timing: Optional[TriggerTimingMetadata] = None


@dataclass
class TriggerOutput:
    """트리거 출력 데이터."""
    success: bool
    session_id: str
    door_session_id: Optional[str]
    message: str
    error_code: Optional[str] = None
    status: str = "complete"  # "complete", "duplicate", "skipped", "queued"
    waiting_for: Optional[str] = None  # "products" if pending


@dataclass
class QueueItem:
    """큐 아이템 (v4.10)."""
    input_data: TriggerInput
    session_id: str
    idempotency_key: str
    delta_weight: float
    enqueued_at: float
    allowed_class_ids: Optional[List[int]] = None
    cached_active_products: Optional[List] = None
    product_weights: Optional[Dict[int, float]] = None
    trace_context: Optional[TriggerTraceContext] = None


class TriggerService:
    """
    트리거 비즈니스 로직 서비스.

    YOLO 추론, 무게 계산, 상품 판단, 세션 저장을 담당.

    v5.2: Deduplication 캐시 크기 제한 추가 (메모리 누수 방지)
    v4.10: asyncio.Queue 기반 순차 처리 워커
    v4.6: 무게 변화 5g 이하 스킵, Node.js 필터링 추가
    v4.5: Idempotency key 기반 중복 체크 추가
    """

    # v5.2: 설정값 (config.trigger에서 가져옴, 클래스 상수는 기본값으로 유지)
    # v4.5: 중복 체크 TTL (초)
    DEDUP_TTL_SECONDS = config.trigger.dedup_ttl_seconds
    # v5.2: 최대 캐시 크기 제한 (메모리 누수 방지)
    DEDUP_MAX_SIZE = config.trigger.dedup_max_size
    # v4.6: 최소 무게 변화량 (이하면 비디오 처리 스킵)
    MIN_WEIGHT_CHANGE_GRAMS = config.trigger.min_weight_change_grams
    # v4.10: 큐 최대 크기
    QUEUE_MAX_SIZE = config.trigger.queue_max_size

    def __init__(
        self,
        video_processor: VideoProcessor,
        engine: ProductDecisionEngine,
        session_store: SessionStore,
        door_session_store: Optional[DoorSessionStore] = None,
        active_product_store: Optional[ActiveProductStore] = None,
    ):
        """
        Initialize trigger service.

        Args:
            video_processor: VideoProcessor 인스턴스
            engine: ProductDecisionEngine 인스턴스
            session_store: SessionStore 인스턴스
            door_session_store: DoorSessionStore 인스턴스 (선택)
            active_product_store: ActiveProductStore 인스턴스 (v4.5, 전역 상품 관리)
        """
        self._video_processor = video_processor
        self._engine = engine
        self._session_store = session_store
        self._door_session_store = door_session_store
        self._active_product_store = active_product_store

        # v4.5: Deduplication 캐시 (idempotency_key -> (timestamp, session_id))
        self._dedup_cache: Dict[str, Tuple[float, str]] = {}
        self._dedup_lock = threading.Lock()

        # v4.10: 순차 처리 큐
        self._queue: Optional[asyncio.Queue] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._stop_event: Optional[asyncio.Event] = None

    def _generate_idempotency_key(self, input_data: TriggerInput) -> str:
        """
        Idempotency key 생성 (v4.5).

        zone + video paths를 기반으로 고유 키 생성.

        Args:
            input_data: TriggerInput

        Returns:
            Idempotency key (MD5 hash)
        """
        def _file_sig(path: str) -> str:
            """파일 mtime+size로 고유 서명 생성. 파일 없으면 'missing'."""
            try:
                st = os.stat(path)
                return f"{st.st_mtime_ns}_{st.st_size}"
            except OSError:
                return "missing"

        key_parts = [
            str(input_data.zone),
            input_data.top_video_path or "",
            _file_sig(input_data.top_video_path or ""),
            input_data.side_video_path or "",
            _file_sig(input_data.side_video_path or ""),
        ]
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _check_duplicate(self, idempotency_key: str) -> Optional[str]:
        """
        중복 요청 체크 (v4.5, v5.2).

        v5.2: 캐시 크기 제한 추가 (메모리 누수 방지)
        - DEDUP_MAX_SIZE 초과 시 가장 오래된 항목 제거

        Args:
            idempotency_key: Idempotency key

        Returns:
            이전 session_id (중복인 경우) 또는 None
        """
        now = time.time()

        with self._dedup_lock:
            # 만료된 항목 정리
            expired_keys = [
                k for k, (ts, _) in self._dedup_cache.items()
                if now - ts > self.DEDUP_TTL_SECONDS
            ]
            for k in expired_keys:
                del self._dedup_cache[k]

            # v5.2: 캐시 크기 제한 (메모리 누수 방지)
            while len(self._dedup_cache) >= self.DEDUP_MAX_SIZE:
                # 가장 오래된 항목 제거
                if self._dedup_cache:
                    oldest_key = min(
                        self._dedup_cache.keys(),
                        key=lambda k: self._dedup_cache[k][0]
                    )
                    del self._dedup_cache[oldest_key]
                    logger.debug(
                        f"[DEDUP] v5.2: Cache size limit reached, removed oldest entry: "
                        f"{oldest_key[:8]}..."
                    )
                else:
                    break

            # 중복 체크
            if idempotency_key in self._dedup_cache:
                ts, session_id = self._dedup_cache[idempotency_key]
                if now - ts <= self.DEDUP_TTL_SECONDS:
                    return session_id

            return None

    def _register_request(self, idempotency_key: str, session_id: str) -> None:
        """
        요청 등록 (v4.5).

        Args:
            idempotency_key: Idempotency key
            session_id: Session ID
        """
        with self._dedup_lock:
            self._dedup_cache[idempotency_key] = (time.time(), session_id)

    def _new_trace_context(
        self,
        session_id: str,
        input_data: TriggerInput,
    ) -> TriggerTraceContext:
        return TriggerTraceContext(
            session_id=session_id,
            zone=input_data.zone,
            top_path=input_data.top_video_path,
            side_path=input_data.side_video_path,
        )

    # ========================================================================
    # Queue Worker (v4.10)
    # ========================================================================

    async def start_worker(self) -> None:
        """
        큐 워커 시작 (v4.10).

        lifespan startup에서 호출됩니다.
        """
        self._queue = asyncio.Queue(maxsize=self.QUEUE_MAX_SIZE)
        self._stop_event = asyncio.Event()
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info(
            f"[TRIGGER-WORKER] Started (max_queue_size={self.QUEUE_MAX_SIZE})"
        )

    async def stop_worker(self) -> None:
        """
        큐 워커 중지 (v4.10).

        lifespan shutdown에서 호출됩니다.
        잔여 큐 항목을 30초간 drain 시도 후 타임아웃 시 cancel합니다.
        """
        if self._worker_task is None:
            return

        self._stop_event.set()
        try:
            await asyncio.wait_for(self._worker_task, timeout=30.0)
            logger.info("[TRIGGER-WORKER] Stopped gracefully")
        except asyncio.TimeoutError:
            logger.warning("[TRIGGER-WORKER] Timeout waiting for worker, cancelling...")
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        except asyncio.CancelledError:
            pass
        self._worker_task = None

    async def enqueue_trigger(self, input_data: TriggerInput) -> TriggerOutput:
        """
        트리거를 큐에 등록 (v4.10).

        중복 체크, 비디오 검증, 무게 < 5g 체크를 즉시 수행하고,
        YOLO 추론이 필요한 경우 큐에 등록하여 즉시 "queued" 응답을 반환합니다.

        워커가 시작되지 않은 경우 (테스트 등) 기존 process_trigger로 fallback합니다.

        Args:
            input_data: 트리거 입력 데이터

        Returns:
            TriggerOutput: status="queued" (큐 등록 성공)
        """
        # 워커가 시작되지 않은 경우 기존 방식으로 처리
        if self._queue is None:
            return await self.process_trigger(input_data)
        # v4.5: 중복 요청 체크
        idempotency_key = self._generate_idempotency_key(input_data)
        duplicate_session_id = self._check_duplicate(idempotency_key)
        if duplicate_session_id is not None:
            self._new_trace_context(duplicate_session_id, input_data).finalize(
                status="duplicate",
                error="duplicate request",
            )
            logger.warning(
                f"[TRIGGER] Duplicate request detected: "
                f"zone={input_data.zone}, idempotency_key={idempotency_key[:8]}..., "
                f"returning previous session_id={duplicate_session_id}"
            )
            return TriggerOutput(
                success=True,
                session_id=duplicate_session_id,
                door_session_id=None,
                message="중복 요청 (이전 결과 반환)",
                status="duplicate",
            )

        session_id = generate_session_id(input_data.zone)
        trace_context = self._new_trace_context(session_id, input_data)

        logger.info(f"[TRIGGER] ========== 큐 등록 시작 ==========")
        logger.info(f"[TRIGGER] zone={input_data.zone}, session_id={session_id}")
        logger.info(f"[TRIGGER] videos: top={input_data.top_video_path}, side={input_data.side_video_path}")
        logger.info(f"[TRIGGER] loadcells: {len(input_data.loadcells)}개")

        # 1. 비디오 파일 검증
        validation_error = self._validate_video_paths(
            input_data.top_video_path,
            input_data.side_video_path,
        )
        if validation_error:
            trace_context.finalize(status="error", error=validation_error)
            return TriggerOutput(
                success=False,
                session_id=session_id,
                door_session_id=None,
                message=validation_error,
                error_code="VIDEO_VALIDATION_ERROR",
            )

        # 2. 전역 상품 정보 확인 (v4.5 사전 필터링)
        allowed_class_ids = None
        if self._active_product_store is not None:
            if self._active_product_store.has_products():
                allowed_class_ids = self._active_product_store.get_allowed_class_ids()
                if allowed_class_ids is not None:
                    logger.info(
                        f"[TRIGGER] hybrid_recall active_class_hints={len(allowed_class_ids)} "
                        "inference_classes=all ids="
                        f"{allowed_class_ids[:10]}{'...' if len(allowed_class_ids) > 10 else ''}"
                    )
            else:
                logger.warning("[TRIGGER] No products available, YOLO will detect all classes")

        # 3. 무게 변화량 조기 계산
        delta_analysis = self._analyze_weight_delta(input_data.loadcells)
        delta_weight = delta_analysis.delta
        logger.info(
            f"[TRIGGER][loadcell] sample_count={delta_analysis.sample_count}, "
            f"span_s={delta_analysis.sample_span_seconds:.3f}, "
            f"window={delta_analysis.window_size}, "
            f"threshold={delta_analysis.stability_threshold:.1f}, "
            f"start_idx={delta_analysis.start_stable_idx}, "
            f"end_idx={delta_analysis.end_stable_idx}, "
            f"fallback={delta_analysis.used_simple_fallback}, "
            f"reason={delta_analysis.reason}"
        )
        logger.info(f"[TRIGGER] 조기 무게 계산: delta_weight={delta_weight:.1f}g")

        # 4. 무게 < 5g이면 YOLO 불필요 → 즉시 처리 (큐 거치지 않음)
        if abs(delta_weight) <= self.MIN_WEIGHT_CHANGE_GRAMS:
            return self._handle_low_weight_skip(
                input_data,
                session_id,
                idempotency_key,
                delta_weight,
                trace_context=trace_context,
            )

        # 5. active_products 캐시 (v4.11: 조회 시점 통일)
        product_weights: Dict[int, float] = {}
        cached_active_products: List = []
        if self._active_product_store is not None:
            cached_active_products = self._active_product_store.get_all_products()
            for product_info in cached_active_products:
                if product_info.yolo_class_id is not None:
                    product_weights[product_info.yolo_class_id] = product_info.product_weight
            if cached_active_products:
                logger.info(f"[TRIGGER] v4.11: active_products {len(cached_active_products)}개 캐시됨")

        # 6. SessionStore에 "queued" 상태 저장
        initial_session = SessionData(
            session_id=session_id,
            zone=input_data.zone,
            status="processing",
            processing_stage="queued",
            processing_stage_detail="큐에서 대기 중",
        )
        self._session_store.save(session_id, initial_session)

        # 7. DoorSessionStore에 pending 알림 (CLOSE 안전장치)
        if self._door_session_store is not None:
            self._door_session_store.notify_trigger_enqueued(input_data.zone)

        # 8. 큐에 등록
        item = QueueItem(
            input_data=input_data,
            session_id=session_id,
            idempotency_key=idempotency_key,
            delta_weight=delta_weight,
            enqueued_at=time.time(),
            allowed_class_ids=allowed_class_ids,
            cached_active_products=cached_active_products,
            product_weights=product_weights,
            trace_context=trace_context,
        )

        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            logger.error(
                f"[TRIGGER] Queue full ({self.QUEUE_MAX_SIZE}), rejecting trigger: "
                f"zone={input_data.zone}, session_id={session_id}"
            )
            trace_context.finalize(
                status="error",
                error="QUEUE_FULL",
            )
            # 큐 등록 실패 → pending 알림 취소
            if self._door_session_store is not None:
                self._door_session_store.notify_trigger_processed(input_data.zone)
            self._session_store.update_stage(
                session_id,
                processing_stage="error",
                processing_stage_detail="큐 가득 참",
                status="error",
            )
            return TriggerOutput(
                success=False,
                session_id=session_id,
                door_session_id=None,
                message="처리 큐가 가득 찼습니다. 잠시 후 다시 시도하세요.",
                error_code="QUEUE_FULL",
            )

        # 9. dedup 등록
        self._register_request(idempotency_key, session_id)

        logger.info(
            f"[TRIGGER] 큐 등록 완료: session_id={session_id}, "
            f"queue_size={self._queue.qsize()}"
        )

        return TriggerOutput(
            success=True,
            session_id=session_id,
            door_session_id=None,
            message="큐에 등록됨, 순차 처리 대기 중",
            status="queued",
        )

    def _handle_low_weight_skip(
        self,
        input_data: TriggerInput,
        session_id: str,
        idempotency_key: str,
        delta_weight: float,
        trace_context: Optional[TriggerTraceContext] = None,
    ) -> TriggerOutput:
        """
        무게 변화 미미 시 즉시 처리 (v4.10, 큐 거치지 않음).

        Args:
            input_data: 트리거 입력 데이터
            session_id: Session ID
            idempotency_key: Idempotency key
            delta_weight: 무게 변화량

        Returns:
            TriggerOutput: status="skipped"
        """
        logger.info(
            f"[TRIGGER] 무게 변화 미미: {abs(delta_weight):.1f}g <= "
            f"{self.MIN_WEIGHT_CHANGE_GRAMS}g, 비디오 처리 스킵"
        )
        session_data = SessionData(
            session_id=session_id,
            zone=input_data.zone,
            products=[],
            total_price=0,
            delta_weight=delta_weight,
            status="complete",
            processing_stage="skipped_low_weight",
            trigger_timing=input_data.timing.to_dict() if input_data.timing else None,
            processing_stage_detail=f"무게 변화 미미 ({abs(delta_weight):.1f}g)",
        )
        self._session_store.save(session_id, session_data)

        door_session_id = None
        if self._door_session_store is not None:
            lightweight_trigger = TriggerResult(
                trigger_id="",
                session_id=session_id,
                timestamp=time.time(),
                products=[],
                delta_weight=delta_weight,
                confidence=0.0,
                video_paths={
                    "top": str(input_data.top_video_path) if input_data.top_video_path else "",
                    "side": str(input_data.side_video_path) if input_data.side_video_path else "",
                },
                is_return=(delta_weight > 0),
                timing_metadata=input_data.timing.to_dict() if input_data.timing else None,
            )
            door_session = self._door_session_store.add_trigger_with_global(
                zone=input_data.zone,
                result=lightweight_trigger,
            )
            door_session_id = door_session.door_session_id
            logger.info(
                f"[TRIGGER] 스킵 trigger DoorSession에 추가: {door_session_id}, "
                f"delta={delta_weight:.1f}g"
            )

        self._register_request(idempotency_key, session_id)
        if trace_context is not None:
            trace_context.finalize(status="skipped")
        return TriggerOutput(
            success=True,
            session_id=session_id,
            door_session_id=door_session_id,
            message=f"무게 변화 미미 ({abs(delta_weight):.1f}g), 스킵",
            status="skipped",
        )

    async def _worker_loop(self) -> None:
        """
        백그라운드 워커 루프 (v4.10).

        큐에서 한 번에 1개씩 꺼내서 순차 처리합니다.
        GPU 독점을 보장하여 TensorRT 동시 추론 충돌을 방지합니다.
        """
        logger.info("[TRIGGER-WORKER] Worker loop started")
        while True:
            if self._stop_event.is_set() and self._queue.empty():
                logger.info("[TRIGGER-WORKER] Stop event set and queue empty, exiting")
                break
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                logger.info("[TRIGGER-WORKER] Worker cancelled")
                break

            try:
                await self._process_trigger_internal(item)
            except Exception as e:
                if item.trace_context is not None:
                    item.trace_context.finalize(status="error", error=str(e))
                logger.error(
                    f"[TRIGGER-WORKER] Error processing zone={item.input_data.zone}, "
                    f"session_id={item.session_id}: {e}",
                    exc_info=True,
                )
                self._session_store.update_stage(
                    item.session_id,
                    processing_stage="error",
                    processing_stage_detail=str(e)[:200],
                    status="error",
                )
                # pending 카운터 감소 (에러 시에도 반드시)
                if self._door_session_store is not None:
                    self._door_session_store.notify_trigger_processed(
                        item.input_data.zone
                    )
            finally:
                self._queue.task_done()

        logger.info("[TRIGGER-WORKER] Worker loop stopped")

    async def _process_trigger_internal(self, item: QueueItem) -> None:
        """
        큐에서 꺼낸 트리거를 실제 처리 (v4.10).

        기존 process_trigger의 비디오 처리~결과 저장 로직을 수행합니다.

        Args:
            item: QueueItem
        """
        start_time = time.time()
        input_data = item.input_data
        session_id = item.session_id
        trace_context = item.trace_context or self._new_trace_context(session_id, input_data)

        logger.info(
            f"[TRIGGER-WORKER] Processing: zone={input_data.zone}, "
            f"session_id={session_id}, "
            f"wait_time={start_time - item.enqueued_at:.1f}s"
        )

        # 1. SessionStore 상태 업데이트
        self._session_store.update_stage(
            session_id,
            processing_stage="extracting_frames",
            processing_stage_detail="비디오에서 프레임 추출 중",
        )

        # 2. 비디오 처리 (YOLO 추론) - GPU 독점
        # v5.3: feature flag 기반 async/sync 선택
        try:
            if config.async_streaming.enabled:
                logger.info("[TRIGGER-WORKER] v5.3: Async streaming 모드로 비디오 처리")
                processing_result = await self._video_processor.process_videos_async(
                    top_path=input_data.top_video_path,
                    side_path=input_data.side_video_path,
                    allowed_class_ids=item.allowed_class_ids,
                    product_weights=item.product_weights or {},
                    trace_context=trace_context,
                )
            else:
                processing_result = await asyncio.to_thread(
                    self._video_processor.process_videos,
                    top_path=input_data.top_video_path,
                    side_path=input_data.side_video_path,
                    allowed_class_ids=item.allowed_class_ids,
                    product_weights=item.product_weights or {},
                    trace_context=trace_context,
                )
        except Exception as exc:
            trace_context.finalize(status="error", error=str(exc))
            raise

        vote_results = processing_result.vote_results
        stats = processing_result.stats

        logger.info(
            f"[TRIGGER-WORKER] 비디오 처리 완료: zone={input_data.zone}, "
            f"프레임={stats.top_frames + stats.side_frames}, "
            f"후보={len(vote_results)}개, 시간={stats.processing_time_ms:.1f}ms"
        )

        # 3. 처리 단계 업데이트
        self._session_store.update_stage(
            session_id,
            processing_stage="calculating_count",
            processing_stage_detail=f"후보 {len(vote_results)}개 도출, 개수 판단 중",
        )

        # 4. 투표 결과를 EnsembleResult로 변환
        vision_candidates = self._vote_results_to_ensemble(vote_results)

        # 5. 최종 상품 판단
        delta_weight = item.delta_weight
        vision_only = delta_weight == 0.0 and len(input_data.loadcells) == 0

        active_products = item.cached_active_products or []
        if active_products:
            logger.info(
                f"[TRIGGER-WORKER] active_products {len(active_products)}개 → engine.judge()에 전달"
            )

        result = self._engine.judge(
            vision_candidates=vision_candidates,
            delta_weight=delta_weight,
            vision_only=vision_only,
            active_products=active_products,
        )

        # 6. Node.js 상품 리스트에 없는 상품 제거 (v4.6)
        filtered_engine_products = result.products
        if (
            self._active_product_store is not None
            and self._active_product_store.has_products()
        ):
            allowed_ids = self._active_product_store.get_allowed_class_ids()
            if allowed_ids is not None:
                allowed_ids_set = set(allowed_ids)
                original_count = len(filtered_engine_products)
                filtered_engine_products = [
                    p for p in filtered_engine_products
                    if p.product_id in allowed_ids_set
                ]
                removed_count = original_count - len(filtered_engine_products)
                if removed_count > 0:
                    logger.warning(
                        f"[TRIGGER-WORKER] v4.6: {removed_count}개 상품 제거 "
                        f"(Node.js 리스트에 없음)"
                    )

        # 7. SessionStore에 결과 저장
        products = [
            ProductResult(
                product_id=p.product_id,
                product_idx=self._get_product_idx(p.product_id),
                name=p.name,
                count=p.count,
                price=p.unit_price,
                confidence=p.confidence,
            )
            for p in filtered_engine_products
        ]

        final_total_price = sum(p.price * p.count for p in products)
        vision_candidates_dicts = [vc.to_dict() for vc in vision_candidates]

        session_data = SessionData(
            session_id=session_id,
            zone=input_data.zone,
            products=products,
            total_price=final_total_price,
            delta_weight=delta_weight,
            status="complete",
            processing_stage="complete",
            processing_stage_detail=f"상품 {len(products)}개 판단 완료",
            confidence=result.confidence,
            top_frames=stats.top_frames,
            side_frames=stats.side_frames,
            processing_time_ms=stats.processing_time_ms,
            vision_candidates=vision_candidates_dicts,
            trigger_timing=input_data.timing.to_dict() if input_data.timing else None,
        )
        self._session_store.save(session_id, session_data)

        # 8. DoorSessionStore에 추가
        door_session_id = None
        if self._door_session_store is not None:
            elapsed_ms = (time.time() - start_time) * 1000
            trigger_result = TriggerResult(
                trigger_id="",
                session_id=session_id,
                timestamp=time.time(),
                products=products,
                delta_weight=delta_weight,
                confidence=result.confidence,
                video_paths={
                    "top": str(input_data.top_video_path) if input_data.top_video_path else "",
                    "side": str(input_data.side_video_path) if input_data.side_video_path else "",
                },
                is_return=delta_weight > 0,
                processing_time_ms=elapsed_ms,
                timing_metadata=input_data.timing.to_dict() if input_data.timing else None,
            )
            door_session = self._door_session_store.add_trigger_with_global(
                zone=input_data.zone,
                result=trigger_result,
            )
            door_session_id = door_session.door_session_id

            # v4.10: pending 카운터 감소
            self._door_session_store.notify_trigger_processed(input_data.zone)

            logger.info(
                f"[TRIGGER-WORKER] Door session: {door_session_id}, "
                f"triggers={door_session.trigger_count}, "
                f"aggregated_products={len(door_session.aggregated_products)}"
            )

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"[TRIGGER-WORKER] ========== 판단 결과 ==========")
        logger.info(
            f"[TRIGGER-WORKER] zone={input_data.zone}, status={result.status.value}, "
            f"confidence={result.confidence:.3f}"
        )
        for p in filtered_engine_products:
            logger.info(f"  - {p.name} x{p.count}: {p.total_price}원")
        logger.info(
            f"[TRIGGER-WORKER] total_price={final_total_price}원, "
            f"elapsed={elapsed_ms:.1f}ms (wait={start_time - item.enqueued_at:.1f}s)"
        )
        trace_context.finalize(status="complete")

    def get_queue_stats(self) -> dict:
        """
        큐 상태 반환 (v4.10).

        Returns:
            큐 통계 정보
        """
        return {
            "worker_running": (
                self._worker_task is not None and not self._worker_task.done()
            ),
            "queue_size": self._queue.qsize() if self._queue else 0,
            "queue_max_size": self.QUEUE_MAX_SIZE,
        }

    async def process_trigger(self, input_data: TriggerInput) -> TriggerOutput:
        """
        트리거 요청 처리.

        v4.5: Idempotency key 기반 중복 체크 추가

        Args:
            input_data: 트리거 입력 데이터

        Returns:
            TriggerOutput: 처리 결과
        """
        start_time = time.time()

        # v4.5: 중복 요청 체크
        idempotency_key = self._generate_idempotency_key(input_data)
        duplicate_session_id = self._check_duplicate(idempotency_key)
        if duplicate_session_id is not None:
            self._new_trace_context(duplicate_session_id, input_data).finalize(
                status="duplicate",
                error="duplicate request",
            )
            logger.warning(
                f"[TRIGGER] Duplicate request detected: "
                f"zone={input_data.zone}, idempotency_key={idempotency_key[:8]}..., "
                f"returning previous session_id={duplicate_session_id}"
            )
            return TriggerOutput(
                success=True,
                session_id=duplicate_session_id,
                door_session_id=None,
                message="중복 요청 (이전 결과 반환)",
                status="duplicate",
            )

        session_id = generate_session_id(input_data.zone)
        trace_context = self._new_trace_context(session_id, input_data)

        logger.info(f"[TRIGGER] ========== 추론 시작 ==========")
        logger.info(f"[TRIGGER] zone={input_data.zone}, session_id={session_id}")
        logger.info(f"[TRIGGER] videos: top={input_data.top_video_path}, side={input_data.side_video_path}")
        logger.info(f"[TRIGGER] loadcells: {len(input_data.loadcells)}개")

        # 1. 비디오 파일 검증
        validation_error = self._validate_video_paths(
            input_data.top_video_path,
            input_data.side_video_path,
        )
        if validation_error:
            trace_context.finalize(status="error", error=validation_error)
            return TriggerOutput(
                success=False,
                session_id=session_id,
                door_session_id=None,
                message=validation_error,
                error_code="VIDEO_VALIDATION_ERROR",
            )

        # 1-B. 전역 상품 정보 확인 (v4.5 사전 필터링)
        allowed_class_ids = None
        if self._active_product_store is not None:
            if self._active_product_store.has_products():  # 전역 상품 확인 (zone 없음)
                allowed_class_ids = self._active_product_store.get_allowed_class_ids()
                if allowed_class_ids is not None:
                    logger.info(
                        f"[TRIGGER] hybrid_recall active_class_hints={len(allowed_class_ids)} "
                        "inference_classes=all ids="
                        f"{allowed_class_ids[:10]}{'...' if len(allowed_class_ids) > 10 else ''}"
                    )
            else:
                logger.warning("[TRIGGER] No products available, YOLO will detect all classes")

        # 1-C. 무게 변화량 조기 계산 (v4.6 - 비디오 처리 전 검증)
        delta_weight = self._calculate_weight_delta(input_data.loadcells)
        logger.info(f"[TRIGGER] 조기 무게 계산: delta_weight={delta_weight:.1f}g")

        if abs(delta_weight) <= self.MIN_WEIGHT_CHANGE_GRAMS:
            logger.info(
                f"[TRIGGER] 무게 변화 미미: {abs(delta_weight):.1f}g <= "
                f"{self.MIN_WEIGHT_CHANGE_GRAMS}g, 비디오 처리 스킵"
            )
            session_data = SessionData(
                session_id=session_id,
                zone=input_data.zone,
                products=[],
                total_price=0,
                delta_weight=delta_weight,
                status="complete",
                processing_stage="skipped_low_weight",
                processing_stage_detail=f"무게 변화 미미 ({abs(delta_weight):.1f}g)",
            )
            self._session_store.save(session_id, session_data)

            # v5.0: DoorSession이 활성이면 빈 trigger 추가 (net delta 추적)
            door_session_id = None
            if self._door_session_store is not None:
                lightweight_trigger = TriggerResult(
                    trigger_id="",
                    session_id=session_id,
                    timestamp=time.time(),
                    products=[],
                    delta_weight=delta_weight,
                    confidence=0.0,
                    video_paths={
                        "top": str(input_data.top_video_path) if input_data.top_video_path else "",
                        "side": str(input_data.side_video_path) if input_data.side_video_path else "",
                    },
                    is_return=(delta_weight > 0),
                )
                door_session = self._door_session_store.add_trigger_with_global(
                    zone=input_data.zone,
                    result=lightweight_trigger,
                )
                door_session_id = door_session.door_session_id
                logger.info(
                    f"[TRIGGER] 스킵 trigger DoorSession에 추가: {door_session_id}, "
                    f"delta={delta_weight:.1f}g"
                )

            self._register_request(idempotency_key, session_id)
            trace_context.finalize(status="skipped")
            return TriggerOutput(
                success=True,
                session_id=session_id,
                door_session_id=door_session_id,
                message=f"무게 변화 미미 ({abs(delta_weight):.1f}g), 스킵",
                status="skipped",
            )

        # 2. 초기 세션 저장 (processing 상태)
        initial_session = SessionData(
            session_id=session_id,
            zone=input_data.zone,
            status="processing",
            processing_stage="extracting_frames",
            processing_stage_detail="비디오 프레임 추출 준비 중",
        )
        self._session_store.save(session_id, initial_session)

        # 3. 비디오 처리 (비동기) - allowed_class_ids, product_weights 전달 (v4.6)
        self._session_store.update_stage(
            session_id,
            processing_stage="extracting_frames",
            processing_stage_detail="비디오에서 프레임 추출 중",
        )

        # v4.11: 비디오 처리 전에 active_products 캐시 (조회 시점 통일)
        # 문제: 비디오 처리 중(~22초) Door session finalize 콜백이 active_product_store.clear() 호출
        # 해결: 비디오 처리 전에 캐시하여 처리 후에도 동일한 데이터 사용
        product_weights: Dict[int, float] = {}
        cached_active_products: List = []  # v4.11: 캐시
        if self._active_product_store is not None:
            cached_active_products = self._active_product_store.get_all_products()
            for product_info in cached_active_products:
                if product_info.yolo_class_id is not None:
                    product_weights[product_info.yolo_class_id] = product_info.product_weight
            if cached_active_products:
                logger.info(f"[TRIGGER] v4.11: active_products {len(cached_active_products)}개 캐시됨")

        # v5.3: feature flag 기반 async/sync 선택
        try:
            if config.async_streaming.enabled:
                logger.info("[TRIGGER] v5.3: Async streaming 모드로 비디오 처리")
                processing_result = await self._video_processor.process_videos_async(
                    top_path=input_data.top_video_path,
                    side_path=input_data.side_video_path,
                    allowed_class_ids=allowed_class_ids,
                    product_weights=product_weights,
                    trace_context=trace_context,
                )
            else:
                processing_result = await asyncio.to_thread(
                    self._video_processor.process_videos,
                    top_path=input_data.top_video_path,
                    side_path=input_data.side_video_path,
                    allowed_class_ids=allowed_class_ids,
                    product_weights=product_weights,
                    trace_context=trace_context,
                )
        except Exception as exc:
            trace_context.finalize(status="error", error=str(exc))
            raise

        vote_results = processing_result.vote_results
        stats = processing_result.stats

        logger.info(f"[TRIGGER] ========== 비디오 처리 완료 ==========")
        logger.info(
            f"[TRIGGER] 총 프레임: {stats.top_frames + stats.side_frames}, "
            f"후보: {len(vote_results)}개, 처리시간: {stats.processing_time_ms:.1f}ms"
        )

        # 4. 처리 단계 업데이트
        self._session_store.update_stage(
            session_id,
            processing_stage="calculating_count",
            processing_stage_detail=f"후보 {len(vote_results)}개 도출, 개수 판단 중",
        )

        # 5. 무게 변화량 (이미 1-C에서 계산됨, v4.6)
        logger.info(f"[TRIGGER] ========== 무게 확인 ==========")
        logger.info(f"[TRIGGER] delta_weight={delta_weight:.1f}g (조기 계산값 사용)")

        # 6. 투표 결과를 EnsembleResult로 변환
        vision_candidates = self._vote_results_to_ensemble(vote_results)

        # 7. 최종 상품 판단 - v4.7: active_products 전달
        vision_only = delta_weight == 0.0 and len(input_data.loadcells) == 0

        # v4.11: 캐시된 active_products 사용 (비디오 처리 중 store가 clear될 수 있음)
        active_products = cached_active_products  # v4.11: 캐시된 값 사용
        if active_products:
            logger.info(f"[TRIGGER] v4.7: active_products {len(active_products)}개 → engine.judge()에 전달")

        result = self._engine.judge(
            vision_candidates=vision_candidates,
            delta_weight=delta_weight,
            vision_only=vision_only,
            active_products=active_products,  # v4.7: 신규 파라미터
        )

        # 7-B. Node.js 상품 리스트에 없는 상품 제거 (v4.6)
        filtered_engine_products = result.products
        if (
            self._active_product_store is not None
            and self._active_product_store.has_products()
        ):
            allowed_ids = self._active_product_store.get_allowed_class_ids()
            if allowed_ids is not None:
                allowed_ids_set = set(allowed_ids)
                original_count = len(filtered_engine_products)
                filtered_engine_products = [
                    p for p in filtered_engine_products
                    if p.product_id in allowed_ids_set
                ]
                removed_count = original_count - len(filtered_engine_products)
                if removed_count > 0:
                    logger.warning(
                        f"[TRIGGER] v4.6: {removed_count}개 상품 제거 "
                        f"(Node.js 리스트에 없음)"
                    )

        # 8. SessionStore에 결과 저장
        products = [
            ProductResult(
                product_id=p.product_id,
                product_idx=self._get_product_idx(p.product_id),
                name=p.name,
                count=p.count,
                price=p.unit_price,
                confidence=p.confidence,
            )
            for p in filtered_engine_products
        ]

        # v4.6: 필터링 후 총 가격 재계산
        final_total_price = sum(p.price * p.count for p in products)

        vision_candidates_dicts = [vc.to_dict() for vc in vision_candidates]

        session_data = SessionData(
            session_id=session_id,
            zone=input_data.zone,
            products=products,
            total_price=final_total_price,
            delta_weight=delta_weight,
            status="complete",
            processing_stage="complete",
            processing_stage_detail=f"상품 {len(products)}개 판단 완료",
            confidence=result.confidence,
            top_frames=stats.top_frames,
            side_frames=stats.side_frames,
            processing_time_ms=stats.processing_time_ms,
            vision_candidates=vision_candidates_dicts,
        )
        self._session_store.save(session_id, session_data)

        # 9. DoorSessionStore에 추가 (v4.1)
        door_session_id = None
        if self._door_session_store is not None:
            elapsed_ms = (time.time() - start_time) * 1000
            trigger_result = TriggerResult(
                trigger_id="",
                session_id=session_id,
                timestamp=time.time(),
                products=products,
                delta_weight=delta_weight,
                confidence=result.confidence,
                video_paths={
                    "top": str(input_data.top_video_path) if input_data.top_video_path else "",
                    "side": str(input_data.side_video_path) if input_data.side_video_path else "",
                },
                is_return=delta_weight > 0,
                processing_time_ms=elapsed_ms,
            )
            door_session = self._door_session_store.add_trigger_with_global(
                zone=input_data.zone,
                result=trigger_result,
            )
            door_session_id = door_session.door_session_id
            logger.info(
                f"[TRIGGER] Door session: {door_session_id}, "
                f"triggers={door_session.trigger_count}, "
                f"aggregated_products={len(door_session.aggregated_products)}"
            )

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"[TRIGGER] ========== 판단 결과 ==========")
        logger.info(f"[TRIGGER] status={result.status.value}, confidence={result.confidence:.3f}")
        for p in filtered_engine_products:
            logger.info(f"  - {p.name} x{p.count}: {p.total_price}원")
        logger.info(f"[TRIGGER] total_price={final_total_price}원, elapsed={elapsed_ms:.1f}ms")

        # v4.5: 요청 등록 (중복 방지용)
        self._register_request(idempotency_key, session_id)
        trace_context.finalize(status="complete")

        return TriggerOutput(
            success=True,
            session_id=session_id,
            door_session_id=door_session_id,
            message="추론 완료",
        )

    def _validate_video_paths(
        self,
        top_path: Optional[str],
        side_path: Optional[str],
    ) -> Optional[str]:
        """비디오 파일 경로 검증."""
        if top_path and not Path(top_path).exists():
            return f"Top video file not found: {top_path}"
        if side_path and not Path(side_path).exists():
            return f"Side video file not found: {side_path}"
        if not top_path and not side_path:
            return "At least one video path (top or side) is required"
        return None

    def _get_product_idx(self, product_id: int) -> Optional[str]:
        """YOLO class_id로 IF11 product_idx 조회."""
        if self._active_product_store is not None:
            product_info = self._active_product_store.get_by_yolo_class_id(product_id)
            if product_info and product_info.product_idx:
                return product_info.product_idx
        return None

    def _calculate_weight_delta(self, loadcells: List[LoadcellReading]) -> float:
        """로드셀 데이터에서 무게 변화량 계산."""
        if not loadcells or len(loadcells) < 2:
            return 0.0

        start_avg, end_avg, is_valid = self._detect_stable_regions(loadcells)

        if not is_valid:
            logger.warning("Could not detect stable regions, using simple delta")
            try:
                start = self._avg_loadcell_channels(loadcells[0].filtered_value) if loadcells[0].filtered_value else 0.0
                end = self._avg_loadcell_channels(loadcells[-1].filtered_value) if loadcells[-1].filtered_value else 0.0
                return end - start
            except (IndexError, AttributeError) as e:
                logger.warning(f"Failed to calculate weight delta: {e}")
                return 0.0

        delta = end_avg - start_avg
        logger.info(
            f"Weight delta (stable regions): "
            f"start={start_avg:.1f}g -> end={end_avg:.1f}g = {delta:.1f}g"
        )
        return delta

    @staticmethod
    def _filter_peaks(
        values: List[float],
        context_window: int = 5,
        threshold_factor: float = 1.5,
        min_diff_grams: float = 50.0,
    ) -> List[float]:
        """
        로드셀 데이터에서 급격한 피크(방해)를 제거.

        각 포인트에 대해 전후 context_window 이웃의 median을 계산하고,
        median과의 차이가 임계값을 초과하면 피크로 판단하여 제거.

        Args:
            values: 로드셀 값 리스트
            context_window: 전후 이웃 포인트 수 (기본 5)
            threshold_factor: 피크 판단 계수 (기본 1.5)
            min_diff_grams: 최소 차이 임계값 (기본 50g)

        Returns:
            피크가 제거된 값 리스트. 데이터 부족하면 원본 반환.
        """
        if len(values) < context_window * 2 + 1:
            return values

        filtered = []
        for i, val in enumerate(values):
            start = max(0, i - context_window)
            end = min(len(values), i + context_window + 1)
            neighbors = [values[j] for j in range(start, end) if j != i]

            if not neighbors:
                filtered.append(val)
                continue

            neighbor_median = float(median(neighbors))
            diff = abs(val - neighbor_median)
            threshold = max(abs(neighbor_median) * (threshold_factor - 1), min_diff_grams)

            if diff <= threshold:
                filtered.append(val)

        return filtered

    def _detect_stable_regions(
        self,
        loadcells: List[LoadcellReading],
        window_size: int = 5,
        stability_threshold: float = 15.0,
    ) -> Tuple[float, float, bool]:
        """안정 구간 감지하여 시작/종료 평균값 계산."""
        if len(loadcells) < window_size * 2:
            return self._simple_delta_values(loadcells)

        values = []
        for lc in loadcells:
            if lc.filtered_value:
                val = self._avg_loadcell_channels(lc.filtered_value)
                values.append(val)

        if len(values) < window_size * 2:
            return self._simple_delta_values(loadcells)

        # Phase 3: 피크 필터링 적용
        filtered_values = self._filter_peaks(values)
        working_values = filtered_values if len(filtered_values) >= window_size * 2 else values
        values_arr = working_values

        # 시작 안정 구간 찾기
        start_stable_idx = 0
        for i in range(len(values_arr) - window_size):
            window = values_arr[i:i + window_size]
            if pstdev(window) < stability_threshold:
                start_stable_idx = i
                break

        start_region = values_arr[start_stable_idx:start_stable_idx + window_size]
        start_avg = float(fmean(start_region))

        # 종료 안정 구간 찾기
        end_stable_idx = len(values_arr) - window_size
        for i in range(len(values_arr) - 1, window_size - 1, -1):
            window = values_arr[i - window_size + 1:i + 1]
            if pstdev(window) < stability_threshold:
                end_stable_idx = i - window_size + 1
                break

        end_region = values_arr[end_stable_idx:end_stable_idx + window_size]
        end_avg = float(fmean(end_region))

        is_valid = (end_stable_idx > start_stable_idx + window_size)

        logger.debug(
            f"Stable regions: start_idx={start_stable_idx}, end_idx={end_stable_idx}, "
            f"start_avg={start_avg:.1f}, end_avg={end_avg:.1f}, valid={is_valid}"
        )

        return start_avg, end_avg, is_valid

    def _simple_delta_values(
        self, loadcells: List[LoadcellReading]
    ) -> Tuple[float, float, bool]:
        """단순 첫/마지막 값 비교 (fallback)."""
        if not loadcells:
            return 0.0, 0.0, False

        try:
            start = self._avg_loadcell_channels(loadcells[0].filtered_value) if loadcells[0].filtered_value else 0.0
            end = self._avg_loadcell_channels(loadcells[-1].filtered_value) if loadcells[-1].filtered_value else 0.0
            return start, end, True
        except (IndexError, AttributeError):
            return 0.0, 0.0, False

    def _parse_loadcell_value(self, value: str) -> float:
        """로드셀 값 파싱 (+12345 형식)."""
        try:
            cleaned = value.strip()
            if cleaned.startswith("+"):
                return float(cleaned[1:])
            return float(cleaned)
        except (ValueError, AttributeError):
            return 0.0

    def _avg_loadcell_channels(self, values: list) -> float:
        """filtered_value 배열 전체 채널 평균. 비거나 파싱 실패 시 0.0."""
        parsed = []
        for v in values:
            try:
                parsed.append(self._parse_loadcell_value(str(v)))
            except Exception:
                pass
        return sum(parsed) if parsed else 0.0

    def _analyze_weight_delta(
        self, loadcells: List[LoadcellReading]
    ) -> loadcell_stats.LoadcellDeltaAnalysis:
        analysis = loadcell_stats.analyze_weight_delta(loadcells)
        logger.info(
            f"Weight delta calculated: {analysis.delta:.1f}g "
            f"(sample_count={analysis.sample_count}, span_s={analysis.sample_span_seconds:.3f}, "
            f"window={analysis.window_size}, threshold={analysis.stability_threshold:.1f}, "
            f"start_idx={analysis.start_stable_idx}, end_idx={analysis.end_stable_idx}, "
            f"fallback={analysis.used_simple_fallback}, reason={analysis.reason})"
        )
        return analysis

    def _calculate_weight_delta(self, loadcells: List[LoadcellReading]) -> float:
        return self._analyze_weight_delta(loadcells).delta

    @staticmethod
    def _filter_peaks(
        values: List[float],
        context_window: int = 5,
        threshold_factor: float = 1.5,
        min_diff_grams: float = 50.0,
    ) -> List[float]:
        return loadcell_stats.filter_peaks(
            values,
            context_window=context_window,
            threshold_factor=threshold_factor,
            min_diff_grams=min_diff_grams,
        )

    def _detect_stable_regions(
        self,
        loadcells: List[LoadcellReading],
        window_size: int = 5,
        stability_threshold: float = 15.0,
    ) -> Tuple[float, float, bool]:
        return loadcell_stats.detect_stable_regions(
            loadcells,
            window_size=window_size,
            stability_threshold=stability_threshold,
        )

    def _simple_delta_values(
        self, loadcells: List[LoadcellReading]
    ) -> Tuple[float, float, bool]:
        return loadcell_stats.simple_delta_values(loadcells)

    def _parse_loadcell_value(self, value: str) -> float:
        return loadcell_stats.parse_loadcell_value(value)

    def _avg_loadcell_channels(self, values: list) -> float:
        return loadcell_stats.avg_loadcell_channels(values)

    def _vote_results_to_ensemble(
        self, vote_results: List[VoteResult]
    ) -> List[EnsembleResult]:
        """VoteResult를 EnsembleResult로 변환."""
        ensemble_results = []
        for vote in vote_results:
            ensemble = EnsembleResult(
                class_id=vote.class_id,
                class_name=vote.class_name,
                top_confidence=vote.top_max_confidence,
                side_confidence=vote.side_max_confidence,
                combined_confidence=vote.weighted_confidence,
                vote_count=2 if (vote.top_detected and vote.side_detected) else 1,
            )
            ensemble_results.append(ensemble)
        return ensemble_results

