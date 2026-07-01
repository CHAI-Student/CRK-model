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
- 정적 상품 fallback 방지

v4.6 변경사항:
- 무게 변화 5g 이하면 비디오 처리 스킵 (불필요한 YOLO 추론 방지)
- Node.js 상품 리스트에 없는 상품 제거 (최종 필터링)
- product_weights 전달하여 로그 개선

v4.5 변경사항:
- Idempotency key 기반 중복 체크 (5초 이내 동일 요청 스킵)
- PendingTriggerStore 관련 기능 제거
- 전역 상품 리스트로 YOLO 필터링 (zone별 관리 제거)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import model_service.core.loadcell_stats as loadcell_stats
from model_service.core.config import config
from model_service.core.logging_config import get_ops_logger
from model_service.engine import EnsembleResult, ProductDecisionEngine
from model_service.session import (
    DoorSessionStore,
    ProductResult,
    SessionData,
    SessionStore,
    TriggerResult,
    build_trigger_candidate_snapshot,
)
from model_service.session.active_product_store import ActiveProductStore
from model_service.session.session_store import generate_session_id
from model_service.video import VideoProcessor, VoteResult
from model_service.video.frame_trace import TriggerTraceContext

logger = logging.getLogger(__name__)
ops_logger = get_ops_logger()


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
    cabinet_type: str = "refrigerated"
    loadcell_scope: str = "zone"
    loadcell_source: str = "loadcells"
    requested_zone: Optional[int] = None
    effective_channel_count: int = 0
    loadcell_validation_reason: Optional[str] = None


@dataclass
class TriggerOutput:
    """트리거 출력 데이터."""
    success: bool
    session_id: str
    door_session_id: Optional[str]
    message: str
    error_code: Optional[str] = None
    status: str = "complete"  # "complete", "duplicate", "skipped", "queued", "waiting"
    waiting_for: Optional[str] = None  # "products" or "stable_loadcell" if pending


@dataclass
class LoadcellEvent:
    """Loadcell-first relevance event for trigger scheduling."""

    event_id: str
    session_id: str
    zone: int
    delta_weight: float
    abs_weight: float
    sign: int
    state: str
    matched_event_ids: List[str]
    created_at: float


@dataclass
class QueueItem:
    """큐 아이템 (v4.10)."""
    input_data: TriggerInput
    session_id: str
    idempotency_key: str
    delta_weight: float
    delta_analysis: Optional[loadcell_stats.LoadcellDeltaAnalysis]
    enqueued_at: float
    event_id: Optional[str] = None
    chargeable_vision_required: bool = True
    allowed_class_ids: Optional[List[int]] = None
    cached_active_products: Optional[List] = None
    active_product_snapshot_metadata: Optional[dict] = None
    product_weights: Optional[Dict[int, float]] = None
    product_stocks: Optional[Dict[int, int]] = None
    trace_context: Optional[TriggerTraceContext] = None
    return_weight_hints: List[Dict[str, object]] = field(default_factory=list)


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
        self._loadcell_events: Dict[str, LoadcellEvent] = {}
        self._event_seq = 0
        self._event_lock = threading.Lock()

    @staticmethod
    def _has_video_path(input_data: TriggerInput) -> bool:
        return bool(input_data.top_video_path or input_data.side_video_path)

    def _is_low_weight_delta(self, delta_weight: float) -> bool:
        return abs(delta_weight) <= self.MIN_WEIGHT_CHANGE_GRAMS

    @staticmethod
    def _is_reliable_low_weight_analysis(
        delta_analysis: loadcell_stats.LoadcellDeltaAnalysis,
    ) -> bool:
        return (
            delta_analysis.stable_region_valid
            and not delta_analysis.used_simple_fallback
            and delta_analysis.reason == "stable_regions"
        )

    def _should_skip_low_weight(
        self,
        input_data: TriggerInput,
        delta_analysis: loadcell_stats.LoadcellDeltaAnalysis,
        delta_weight: Optional[float] = None,
    ) -> bool:
        selected_delta = delta_analysis.delta if delta_weight is None else delta_weight
        return self._is_low_weight_delta(selected_delta)

    def _should_force_vision_only(
        self,
        input_data: TriggerInput,
        delta_analysis: loadcell_stats.LoadcellDeltaAnalysis,
    ) -> bool:
        return False

    @staticmethod
    def _analysis_positive_trend(
        delta_analysis: loadcell_stats.LoadcellDeltaAnalysis,
    ) -> float:
        trend = delta_analysis.end_avg - delta_analysis.start_avg
        return max(float(delta_analysis.delta), float(trend))

    @staticmethod
    def _analysis_negative_trend(
        delta_analysis: loadcell_stats.LoadcellDeltaAnalysis,
    ) -> float:
        trend = delta_analysis.end_avg - delta_analysis.start_avg
        return min(
            float(delta_analysis.delta),
            float(delta_analysis.decision_delta),
            float(trend),
        )

    @staticmethod
    def _is_return_stable_enough(
        delta_analysis: loadcell_stats.LoadcellDeltaAnalysis,
    ) -> bool:
        if not config.trigger.return_stabilization_require_stable_regions:
            return delta_analysis.delta > 0
        return (
            delta_analysis.delta > 0
            and delta_analysis.stable_region_valid
            and not delta_analysis.used_simple_fallback
            and delta_analysis.reason == "stable_regions"
        )

    def _is_return_stabilization_candidate(
        self,
        delta_analysis: loadcell_stats.LoadcellDeltaAnalysis,
    ) -> bool:
        return self._analysis_positive_trend(delta_analysis) > self.MIN_WEIGHT_CHANGE_GRAMS

    @staticmethod
    def _is_removal_stable_enough(
        delta_analysis: loadcell_stats.LoadcellDeltaAnalysis,
    ) -> bool:
        return (
            delta_analysis.decision_delta < 0
            and delta_analysis.stable_region_valid
            and not delta_analysis.used_simple_fallback
            and delta_analysis.reason == "stable_regions"
        )

    def _removal_stabilization_from_analysis(
        self,
        delta_analysis: loadcell_stats.LoadcellDeltaAnalysis,
    ) -> Optional[dict]:
        observed_delta = self._analysis_negative_trend(delta_analysis)
        if observed_delta >= -self.MIN_WEIGHT_CHANGE_GRAMS:
            return None
        if delta_analysis.decision_delta_reliable:
            return None
        if self._is_removal_stable_enough(delta_analysis):
            return None

        reason = (
            "simple_fallback_not_chargeable"
            if delta_analysis.used_simple_fallback
            else "unstable_or_unconfirmed_removal_loadcell"
        )
        return {
            "accepted": True,
            "reason": reason,
            "observed_delta": round(float(observed_delta), 1),
            "net_delta": round(float(delta_analysis.delta), 1),
            "decision_delta": round(float(delta_analysis.decision_delta), 1),
            "analysis_reason": delta_analysis.reason,
            "stable_delta_source": delta_analysis.stable_delta_source,
            "stable_region_valid": bool(delta_analysis.stable_region_valid),
            "used_simple_fallback": bool(delta_analysis.used_simple_fallback),
            "baseline_stable_avg": round(float(delta_analysis.baseline_stable_avg), 1),
            "final_stable_avg": round(float(delta_analysis.final_stable_avg), 1),
            "trailing_unstable_sample_count": int(
                delta_analysis.trailing_unstable_sample_count
            ),
            "raw_simple_delta": round(float(delta_analysis.raw_simple_delta), 1),
            "raw_extreme_delta": round(float(delta_analysis.raw_extreme_delta), 1),
            "engine_skipped": True,
        }

    def _record_return_stabilization(
        self,
        trace_context: Optional[TriggerTraceContext],
        diagnostics: dict,
    ) -> None:
        if trace_context is None:
            return
        existing = dict(getattr(trace_context, "weight_diagnostics", {}) or {})
        existing["return_stabilization"] = dict(diagnostics)
        trace_context.record_weight_diagnostics(existing)

    async def _stabilize_return_delta(
        self,
        input_data: TriggerInput,
        delta_analysis: loadcell_stats.LoadcellDeltaAnalysis,
        trace_context: Optional[TriggerTraceContext],
    ) -> loadcell_stats.LoadcellDeltaAnalysis:
        if not config.trigger.return_video_skip_enabled:
            return delta_analysis
        if not self._is_return_stabilization_candidate(delta_analysis):
            return delta_analysis
        if self._is_return_stable_enough(delta_analysis):
            self._record_return_stabilization(
                trace_context,
                {
                    "enabled": True,
                    "waited_seconds": 0.0,
                    "ready": True,
                    "selected_delta": round(float(delta_analysis.delta), 1),
                    "selected_reason": delta_analysis.reason,
                    "selected_stable_region_valid": delta_analysis.stable_region_valid,
                    "selected_used_simple_fallback": delta_analysis.used_simple_fallback,
                },
            )
            return delta_analysis

        wait_seconds = max(
            0.0,
            float(config.trigger.return_stabilization_wait_seconds),
        )
        if wait_seconds > 0:
            logger.info(
                "[TRIGGER][RETURN-STABILIZE] waiting %.3fs before committing "
                "return delta (initial_delta=%.1fg, reason=%s, stable=%s, fallback=%s)",
                wait_seconds,
                delta_analysis.delta,
                delta_analysis.reason,
                delta_analysis.stable_region_valid,
                delta_analysis.used_simple_fallback,
            )
            await asyncio.sleep(wait_seconds)

        refreshed = self._analyze_weight_delta(
            input_data.loadcells,
            cabinet_type=input_data.cabinet_type,
        )
        selected = refreshed
        if refreshed.delta <= 0 and delta_analysis.delta > 0:
            selected = delta_analysis
        elif (
            not self._is_return_stable_enough(refreshed)
            and refreshed.delta < delta_analysis.delta
        ):
            selected = delta_analysis

        self._record_return_stabilization(
            trace_context,
            {
                "enabled": True,
                "waited_seconds": round(wait_seconds, 3),
                "initial_delta": round(float(delta_analysis.delta), 1),
                "initial_reason": delta_analysis.reason,
                "initial_stable_region_valid": delta_analysis.stable_region_valid,
                "initial_used_simple_fallback": delta_analysis.used_simple_fallback,
                "refreshed_delta": round(float(refreshed.delta), 1),
                "refreshed_reason": refreshed.reason,
                "refreshed_stable_region_valid": refreshed.stable_region_valid,
                "refreshed_used_simple_fallback": refreshed.used_simple_fallback,
                "ready": self._is_return_stable_enough(selected),
                "selected_delta": round(float(selected.delta), 1),
                "selected_reason": selected.reason,
                "selected_stable_region_valid": selected.stable_region_valid,
                "selected_used_simple_fallback": selected.used_simple_fallback,
                "require_stable_regions": bool(
                    config.trigger.return_stabilization_require_stable_regions
                ),
            },
        )
        return selected

    @staticmethod
    def _is_500ml_bottle_weight(unit_weight: float) -> bool:
        return 450.0 <= float(unit_weight) <= 560.0

    @staticmethod
    def _candidate_votes(candidate: EnsembleResult) -> int:
        return max(
            int(getattr(candidate, "raw_vote_count", 0) or 0),
            int(getattr(candidate, "vote_count", 0) or 0),
        )

    def _removal_stabilization_conflict(
        self,
        *,
        vision_candidates: List[EnsembleResult],
        delta_weight: float,
        active_products: Optional[List],
    ) -> Optional[dict]:
        """Detect strong x2 bottle evidence whose loadcell delta is still short."""
        if delta_weight >= 0 or not vision_candidates or not active_products:
            return None

        candidate = vision_candidates[0]
        if str(getattr(candidate, "source", "vision") or "vision") != "vision":
            return None

        active_map = {
            int(product.yolo_class_id): product
            for product in active_products
            if getattr(product, "yolo_class_id", None) is not None
        }
        product = active_map.get(int(candidate.class_id))
        if product is None:
            return None

        try:
            unit_weight = float(getattr(product, "product_weight", 0.0) or 0.0)
            stock_qty = int(getattr(product, "stock_qty", 0) or 0)
        except (TypeError, ValueError):
            return None
        if unit_weight <= 0 or stock_qty < 2:
            return None
        if not self._is_500ml_bottle_weight(unit_weight):
            return None

        confidence_threshold = getattr(self._engine, "confidence_threshold", 0.3)
        try:
            confidence_floor = max(0.45, float(confidence_threshold))
        except (TypeError, ValueError):
            confidence_floor = 0.45
        confidence = float(getattr(candidate, "combined_confidence", 0.0) or 0.0)
        votes = self._candidate_votes(candidate)
        min_votes = max(0, int(config.weight.detected_single_fallback_min_votes))
        if confidence < confidence_floor or votes < min_votes:
            return None

        target_weight = abs(float(delta_weight))
        nearest_count = int(round(target_weight / unit_weight))
        if nearest_count != 2:
            return None

        ratio = target_weight / unit_weight
        if ratio < 1.65 or ratio >= 2.0:
            return None

        expected_weight = unit_weight * 2
        undercount = expected_weight - target_weight
        base_x2_allowance = float(config.weight.tolerance_grams) + (
            float(config.weight.same_product_count_tolerance_grams) * 2
        )
        max_undercount = unit_weight * 0.20
        if undercount <= base_x2_allowance or undercount > max_undercount:
            return None

        return {
            "accepted": True,
            "reason": "strong_regular_bottle_x2_loadcell_undercount",
            "target_weight": round(target_weight, 1),
            "expected_weight": round(expected_weight, 1),
            "undercount": round(undercount, 1),
            "base_x2_allowance": round(base_x2_allowance, 1),
            "max_undercount": round(max_undercount, 1),
            "ratio": round(ratio, 3),
            "selected": {
                "class_id": int(candidate.class_id),
                "name": str(candidate.class_name),
                "source": "vision",
                "confidence": round(confidence, 4),
                "votes": int(votes),
                "unit_weight": round(unit_weight, 1),
                "expected_count": 2,
            },
        }

    def _handle_unstable_removal_loadcell(
        self,
        *,
        input_data: TriggerInput,
        session_id: str,
        delta_weight: float,
        diagnostics: dict,
        trace_context: Optional[TriggerTraceContext],
        stats: object,
        vision_candidates: List[EnsembleResult],
        event_id: Optional[str] = None,
        failure_reason: Optional[str] = None,
    ) -> None:
        detail = (
            "removal loadcell is not stable yet; retry trigger after stable tail"
        )
        top_frames = int(getattr(stats, "top_frames", 0) or 0)
        side_frames = int(getattr(stats, "side_frames", 0) or 0)
        processing_time_ms = float(getattr(stats, "processing_time_ms", 0.0) or 0.0)
        session_data = SessionData(
            session_id=session_id,
            zone=input_data.zone,
            products=[],
            total_price=0,
            delta_weight=delta_weight,
            status="waiting",
            processing_stage="removal_waiting_for_stable_loadcell",
            processing_stage_detail=detail,
            confidence=0.0,
            top_frames=top_frames,
            side_frames=side_frames,
            processing_time_ms=processing_time_ms,
            vision_candidates=[candidate.to_dict() for candidate in vision_candidates],
            trigger_timing=input_data.timing.to_dict() if input_data.timing else None,
            failure_reason=failure_reason,
        )
        self._session_store.save(session_id, session_data)
        if trace_context is not None:
            existing = dict(getattr(trace_context, "weight_diagnostics", {}) or {})
            diagnostic_update = {
                "decision_branch": "removal_stabilization",
                "removal_stabilization": diagnostics,
                "engine_skipped": True,
                "excluded_from_close_summary": True,
            }
            if failure_reason:
                diagnostic_update["active_product_failure_reason"] = failure_reason
            existing.update(diagnostic_update)
            trace_context.record_weight_diagnostics(existing)
            trace_context.record_final_result(
                products=[],
                total_price=0,
                status="removal_waiting_for_stable_loadcell",
                confidence=0.0,
            )
            if failure_reason:
                trace_context.final_result["failure_reason"] = failure_reason
            trace_context.record_storage_result(products=[], total_price=0)
            trace_context.finalize(status="waiting")

        self._mark_loadcell_event_state(event_id, "waiting_for_stable_loadcell")
        if self._door_session_store is not None:
            self._door_session_store.notify_trigger_processed(
                input_data.zone,
                session_id=session_id,
                status="waiting_for_stable_loadcell",
            )
        ops_logger.info(
            f"[OPS][RESULT] zone={input_data.zone} "
            "status=removal_waiting_for_stable_loadcell "
            f"delta={delta_weight:.1f}g products=none total_price=0"
            f"{f' failure_reason={failure_reason}' if failure_reason else ''}"
        )

    def _return_unstable_removal_loadcell(
        self,
        *,
        input_data: TriggerInput,
        session_id: str,
        idempotency_key: Optional[str],
        delta_weight: float,
        diagnostics: dict,
        trace_context: Optional[TriggerTraceContext],
        stats: Optional[object] = None,
        vision_candidates: Optional[List[EnsembleResult]] = None,
        event_id: Optional[str] = None,
        failure_reason: Optional[str] = None,
    ) -> TriggerOutput:
        self._handle_unstable_removal_loadcell(
            input_data=input_data,
            session_id=session_id,
            delta_weight=delta_weight,
            diagnostics=diagnostics,
            trace_context=trace_context,
            stats=stats
            or SimpleNamespace(
                top_frames=0,
                side_frames=0,
                processing_time_ms=0.0,
            ),
            vision_candidates=vision_candidates or [],
            event_id=event_id,
            failure_reason=failure_reason,
        )
        if idempotency_key:
            self._register_request(idempotency_key, session_id)
        return TriggerOutput(
            success=True,
            session_id=session_id,
            door_session_id=None,
            message="removal loadcell is not stable yet; retry trigger after stable tail",
            status="waiting",
            waiting_for="stable_loadcell",
        )

    @staticmethod
    def _loadcell_channel_count(loadcells: Sequence[LoadcellReading]) -> int:
        max_count = 0
        for loadcell in loadcells:
            for values in (loadcell.raw_value, loadcell.filtered_value):
                if isinstance(values, (list, tuple)):
                    max_count = max(max_count, len(values))
        return max_count

    def _input_loadcell_metadata(self, input_data: TriggerInput) -> dict:
        return {
            "cabinet_type": input_data.cabinet_type,
            "loadcell_scope": input_data.loadcell_scope,
            "loadcell_source": input_data.loadcell_source,
            "requested_zone": (
                input_data.zone
                if input_data.requested_zone is None
                else input_data.requested_zone
            ),
            "effective_channel_count": (
                input_data.effective_channel_count
                or self._loadcell_channel_count(input_data.loadcells)
            ),
            "loadcell_validation_reason": input_data.loadcell_validation_reason,
        }

    def _loadcell_trace_metadata(
        self,
        loadcells: List[LoadcellReading],
        delta_analysis: Optional[loadcell_stats.LoadcellDeltaAnalysis],
        *,
        zone: Optional[int] = None,
        event_id: Optional[str] = None,
        input_data: Optional[TriggerInput] = None,
    ) -> dict:
        metadata = loadcell_stats.summarize_loadcell_payload(loadcells)
        if input_data is not None:
            metadata.update(self._input_loadcell_metadata(input_data))
        if delta_analysis is None:
            metadata.update(
                {
                    "sample_count": len(loadcells),
                    "parsed_sample_count": metadata["filtered_parsed_channel_count"],
                    "working_sample_count": 0,
                    "reason": None,
                    "net_delta_weight": None,
                    "decision_delta_weight": None,
                    "decision_delta_reliable": False,
                    "stable_delta_source": None,
                    "baseline_stable_avg": None,
                    "final_stable_avg": None,
                    "trailing_unstable_sample_count": 0,
                    "raw_simple_delta": None,
                    "raw_extreme_delta": None,
                    "endpoint_delta_weight": None,
                    "endpoint_fallback_applied": False,
                    "endpoint_fallback_reason": None,
                    "stable_plateaus": [],
                    "purchase_delta_candidates": [],
                    "removal_segment_targets": [],
                    "channel_removal_segment_targets": [],
                    "channel_delta_diagnostics": {},
                    "return_segment_targets": [],
                    "vision_required_segment_targets": [],
                    "paired_loadcell_movements": [],
                    "ignored_loadcell_movements": [],
                    "mixed_sign_net_masking_guard": {},
                    "pressure_like_event": False,
                }
            )
            if zone is not None:
                recent_events = self._recent_same_zone_event_summaries(
                    zone=zone,
                    exclude_event_id=event_id,
                )
                metadata["recent_same_zone_window_seconds"] = round(
                    float(config.trigger.rapid_same_zone_window_seconds),
                    3,
                )
                metadata["recent_same_zone_events"] = recent_events
                metadata["recent_return_weights_g"] = [
                    event["abs_weight"]
                    for event in recent_events
                    if int(event.get("sign", 0) or 0) > 0
                ]
            return metadata
        segments = list(delta_analysis.segments)
        positive_segments = [segment for segment in segments if segment.sign > 0]
        negative_segments = [segment for segment in segments if segment.sign < 0]
        metadata.update(
            {
                "sample_count": delta_analysis.sample_count,
                "parsed_sample_count": delta_analysis.parsed_sample_count,
                "working_sample_count": delta_analysis.working_sample_count,
                "reason": delta_analysis.reason,
                "net_delta_weight": round(float(delta_analysis.delta), 1),
                "decision_delta_weight": round(float(delta_analysis.decision_delta), 1),
                "decision_delta_reliable": bool(delta_analysis.decision_delta_reliable),
                "stable_delta_source": delta_analysis.stable_delta_source,
                "baseline_stable_avg": round(
                    float(delta_analysis.baseline_stable_avg),
                    1,
                ),
                "final_stable_avg": round(float(delta_analysis.final_stable_avg), 1),
                "trailing_unstable_sample_count": int(
                    delta_analysis.trailing_unstable_sample_count
                ),
                "raw_simple_delta": round(float(delta_analysis.raw_simple_delta), 1),
                "raw_extreme_delta": round(float(delta_analysis.raw_extreme_delta), 1),
                "endpoint_delta_weight": round(float(delta_analysis.endpoint_delta_weight), 1),
                "endpoint_fallback_applied": bool(
                    delta_analysis.endpoint_fallback_applied
                ),
                "endpoint_fallback_reason": delta_analysis.endpoint_fallback_reason,
                "stable_plateaus": [
                    plateau.to_dict()
                    for plateau in delta_analysis.stable_plateaus
                ],
                "purchase_delta_candidates": list(
                    delta_analysis.purchase_delta_candidates
                ),
                "removal_segment_targets": list(delta_analysis.removal_segment_targets),
                "channel_removal_segment_targets": list(
                    delta_analysis.channel_removal_segment_targets
                ),
                "channel_delta_diagnostics": dict(
                    delta_analysis.channel_delta_diagnostics
                ),
                "return_segment_targets": list(delta_analysis.return_segment_targets),
                "vision_required_segment_targets": list(
                    delta_analysis.vision_required_segment_targets
                ),
                "paired_loadcell_movements": list(
                    delta_analysis.paired_loadcell_movements
                ),
                "ignored_loadcell_movements": list(
                    delta_analysis.ignored_loadcell_movements
                ),
                "mixed_sign_net_masking_guard": dict(
                    delta_analysis.mixed_sign_net_masking_guard
                ),
                "pressure_like_event": bool(delta_analysis.pressure_like_event),
                "compound_segments": [segment.to_dict() for segment in segments],
                "compound_segment_count": len(segments),
                "compound_event": len(segments) >= 2,
                "compound_positive_segment_count": len(positive_segments),
                "compound_negative_segment_count": len(negative_segments),
                "compound_positive_weights_g": [
                    round(abs(float(segment.delta)), 1)
                    for segment in positive_segments
                ],
                "compound_negative_weights_g": [
                    round(abs(float(segment.delta)), 1)
                    for segment in negative_segments
                ],
            }
        )
        if zone is not None:
            recent_events = self._recent_same_zone_event_summaries(
                zone=zone,
                exclude_event_id=event_id,
            )
            metadata["recent_same_zone_window_seconds"] = round(
                float(config.trigger.rapid_same_zone_window_seconds),
                3,
            )
            metadata["recent_same_zone_events"] = recent_events
            metadata["recent_return_weights_g"] = [
                event["abs_weight"]
                for event in recent_events
                if int(event.get("sign", 0) or 0) > 0
            ]
        return metadata

    @staticmethod
    def _loadcell_payload_issue_reason(payload_diagnostics: Optional[dict]) -> Optional[str]:
        if not payload_diagnostics:
            return None

        payload_state = payload_diagnostics.get("payload_state")
        raw_state = payload_diagnostics.get("raw_state")
        if payload_state == "all_zero" and raw_state == "nonzero":
            return "filtered_all_zero_raw_nonzero"
        if payload_state in {"empty_payload", "invalid_only", "all_zero"}:
            return f"loadcell_payload_{payload_state}"
        return None

    def _record_no_charge_diagnostic(
        self,
        *,
        input_data: TriggerInput,
        session_id: str,
        reason: str,
        delta_weight: float,
        processing_stage: str,
        payload_diagnostics: Optional[dict],
        message: str,
    ) -> None:
        if self._door_session_store is None:
            return
        self._door_session_store.record_no_charge_diagnostic(
            zone=input_data.zone,
            session_id=session_id,
            reason=reason,
            delta_weight=delta_weight,
            processing_stage=processing_stage,
            payload_diagnostics=payload_diagnostics,
            video_paths={
                "top": str(input_data.top_video_path) if input_data.top_video_path else "",
                "side": str(input_data.side_video_path) if input_data.side_video_path else "",
            },
            message=message,
        )

    @staticmethod
    def _products_for_storage(result, log_prefix: str = "[TRIGGER]") -> list:
        if result.is_success:
            return result.products
        if result.products:
            status = getattr(result.status, "value", result.status)
            logger.warning(
                f"{log_prefix} dropping {len(result.products)} product(s) "
                f"from storage because judgment_status={status}"
            )
        return []

    @staticmethod
    def _snapshot_allowed_class_ids(active_products: List) -> Optional[set[int]]:
        if not active_products:
            return None

        allowed_ids: set[int] = set()
        for product_info in active_products:
            class_id = getattr(product_info, "yolo_class_id", None)
            stock_qty = getattr(product_info, "stock_qty", 0)
            if class_id is not None and (stock_qty is None or stock_qty > 0):
                allowed_ids.add(class_id)
        return allowed_ids

    @staticmethod
    def _snapshot_product_info(active_products: Optional[List], product_id: int):
        if not active_products:
            return None
        for product_info in active_products:
            if getattr(product_info, "yolo_class_id", None) == product_id:
                return product_info
        return None

    def _filter_products_by_snapshot(
        self,
        products: List,
        active_products: Optional[List],
        log_prefix: str,
    ) -> List:
        allowed_ids = self._snapshot_allowed_class_ids(active_products or [])
        if allowed_ids is None:
            return products

        filtered = []
        for product in products:
            if product.product_id in allowed_ids:
                filtered.append(product)
            else:
                logger.warning(
                    f"{log_prefix} dropping product outside active snapshot: "
                    f"product_id={product.product_id}, name={product.name}"
                )
        return filtered

    @staticmethod
    def _format_products_for_ops(products: List) -> str:
        if not products:
            return "none"
        return ", ".join(
            f"{getattr(product, 'name', 'unknown')}x{getattr(product, 'count', 0)}"
            for product in products
        )

    @staticmethod
    def _log_candidate_ops(
        zone: int,
        vote_results: List[VoteResult],
        product_weights: Optional[Dict[int, float]] = None,
        prefix: str = "[OPS][CANDIDATES]",
    ) -> None:
        candidate_limit = max(1, int(config.vision.top_k))
        if not vote_results:
            ops_logger.info(f"{prefix} zone={zone} none")
            return

        product_weights = product_weights or {}
        for index, vote in enumerate(vote_results[:candidate_limit], start=1):
            product_weight = product_weights.get(vote.class_id)
            weight_text = (
                f"{float(product_weight):.1f}g"
                if product_weight is not None
                else "unknown"
            )
            source = getattr(vote, "source", "vision")
            ops_logger.info(
                f"{prefix} zone={zone} rank={index} name={vote.class_name} "
                f"weight={weight_text} "
                f"confidence={vote.weighted_confidence:.3f} "
                f"top={vote.top_detected} side={vote.side_detected} "
                f"source={source} "
                f"count_hint={getattr(vote, 'instance_count_hint', 1)} "
                f"freezer_exit_votes={getattr(vote, 'freezer_exit_path_votes', 0)}"
            )

    @staticmethod
    def _active_product_diagnostics(
        active_products: Optional[List],
        allowed_class_ids: Optional[List[int]],
        store_stats: Optional[dict] = None,
        snapshot_metadata: Optional[dict] = None,
    ) -> dict:
        active_products = active_products or []
        stock_positive_count = 0
        stock_positive_weight_count = 0
        zero_stock_count = 0
        zero_weight_count = 0

        for product in active_products:
            stock = int(getattr(product, "stock_qty", 0) or 0)
            weight = float(getattr(product, "product_weight", 0.0) or 0.0)
            if stock > 0:
                stock_positive_count += 1
            else:
                zero_stock_count += 1
            if weight <= 0:
                zero_weight_count += 1
            if stock > 0 and weight > 0:
                stock_positive_weight_count += 1

        allowed_class_ids_count = (
            len(allowed_class_ids) if allowed_class_ids is not None else 0
        )
        fail_closed_reason = None
        if allowed_class_ids is None:
            fail_closed_reason = "missing_active_product_snapshot_fail_closed"
        elif len(allowed_class_ids) == 0:
            fail_closed_reason = "empty_allowlist_fail_closed"

        diagnostics = {
            "active_products_count": len(active_products),
            "allowed_class_ids_count": allowed_class_ids_count,
            "allowed_class_ids": list(allowed_class_ids or []),
            "stock_positive_products": stock_positive_count,
            "stock_positive_weight_products": stock_positive_weight_count,
            "zero_stock_products": zero_stock_count,
            "zero_weight_products": zero_weight_count,
            "inference_fail_closed_reason": fail_closed_reason,
        }
        if snapshot_metadata:
            diagnostics.update(snapshot_metadata)
        if store_stats:
            diagnostics["store_stats"] = dict(store_stats)
        return diagnostics

    def _record_active_product_diagnostics(
        self,
        *,
        trace_context: Optional[TriggerTraceContext],
        active_products: Optional[List],
        allowed_class_ids: Optional[List[int]],
        snapshot_metadata: Optional[dict] = None,
        log_prefix: str,
    ) -> dict:
        store_stats = (
            self._active_product_store.get_stats()
            if self._active_product_store is not None
            and hasattr(self._active_product_store, "get_stats")
            else None
        )
        diagnostics = self._active_product_diagnostics(
            active_products,
            allowed_class_ids,
            store_stats,
            snapshot_metadata,
        )
        if (
            trace_context is not None
            and hasattr(trace_context, "record_active_product_diagnostics")
        ):
            trace_context.record_active_product_diagnostics(diagnostics)

        logger.info(
            f"{log_prefix}[ACTIVE-PRODUCTS] "
            f"active={diagnostics['active_products_count']} "
            f"allowed={diagnostics['allowed_class_ids_count']} "
            f"stock_positive={diagnostics['stock_positive_products']} "
            f"stock_positive_weight={diagnostics['stock_positive_weight_products']} "
            f"zero_stock={diagnostics['zero_stock_products']} "
            f"zero_weight={diagnostics['zero_weight_products']} "
            f"snapshot_source={diagnostics.get('snapshot_source', 'unknown')}"
        )
        if diagnostics["inference_fail_closed_reason"]:
            logger.warning(
                f"{log_prefix}[ACTIVE-PRODUCTS] "
                f"reason={diagnostics['inference_fail_closed_reason']}"
            )
        return diagnostics

    def _capture_active_product_snapshot(
        self,
    ) -> tuple[List, Optional[List[int]], dict]:
        if self._active_product_store is None:
            return [], None, {
                "snapshot_source": "missing",
                "used_last_valid_snapshot": False,
                "current_snapshot_present": False,
                "last_valid_snapshot_present": False,
                "last_valid_snapshot_expired": False,
            }

        if hasattr(self._active_product_store, "get_effective_snapshot"):
            snapshot = self._active_product_store.get_effective_snapshot()
            return (
                list(snapshot.products),
                (
                    list(snapshot.allowed_class_ids)
                    if snapshot.allowed_class_ids is not None
                    else None
                ),
                snapshot.diagnostics(),
            )

        if self._active_product_store.has_products():
            return (
                self._active_product_store.get_all_products(),
                self._active_product_store.get_allowed_class_ids(),
                {
                    "snapshot_source": "current",
                    "used_last_valid_snapshot": False,
                    "current_snapshot_present": True,
                    "last_valid_snapshot_present": False,
                    "last_valid_snapshot_expired": False,
                },
            )

        return [], None, {
            "snapshot_source": "missing",
            "used_last_valid_snapshot": False,
            "current_snapshot_present": False,
            "last_valid_snapshot_present": False,
            "last_valid_snapshot_expired": False,
        }

    @staticmethod
    def _product_weights_from_snapshot(active_products: Optional[List]) -> Dict[int, float]:
        product_weights: Dict[int, float] = {}
        for product_info in active_products or []:
            class_id = getattr(product_info, "yolo_class_id", None)
            product_weight = getattr(product_info, "product_weight", None)
            if class_id is None or product_weight is None:
                continue
            try:
                product_weights[int(class_id)] = float(product_weight)
            except (TypeError, ValueError):
                continue
        return product_weights

    @staticmethod
    def _product_stocks_from_snapshot(active_products: Optional[List]) -> Dict[int, int]:
        product_stocks: Dict[int, int] = {}
        for product_info in active_products or []:
            class_id = getattr(product_info, "yolo_class_id", None)
            stock_qty = getattr(product_info, "stock_qty", None)
            if class_id is None or stock_qty is None:
                continue
            try:
                product_stocks[int(class_id)] = int(stock_qty)
            except (TypeError, ValueError):
                continue
        return product_stocks

    def _register_loadcell_event(
        self,
        *,
        session_id: str,
        zone: int,
        delta_weight: float,
        state: str,
    ) -> LoadcellEvent:
        with self._event_lock:
            self._event_seq += 1
            event = LoadcellEvent(
                event_id=f"loadcell_event_{self._event_seq:06d}",
                session_id=session_id,
                zone=zone,
                delta_weight=delta_weight,
                abs_weight=abs(delta_weight),
                sign=1 if delta_weight > 0 else -1 if delta_weight < 0 else 0,
                state=state,
                matched_event_ids=[],
                created_at=time.time(),
            )
            self._loadcell_events[event.event_id] = event
            return event

    def _recent_same_zone_event_summaries(
        self,
        *,
        zone: Optional[int],
        exclude_event_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> List[dict]:
        if zone is None:
            return []
        window_seconds = max(0.0, float(config.trigger.rapid_same_zone_window_seconds))
        if window_seconds <= 0:
            return []

        current_time = time.time() if now is None else now
        with self._event_lock:
            events = [
                event
                for event in self._loadcell_events.values()
                if event.zone == zone
                and event.event_id != exclude_event_id
                and 0.0 <= current_time - event.created_at <= window_seconds
            ]

        events.sort(key=lambda event: event.created_at, reverse=True)
        return [
            {
                "event_id": event.event_id,
                "session_id": event.session_id,
                "delta_weight": round(float(event.delta_weight), 1),
                "abs_weight": round(float(event.abs_weight), 1),
                "sign": int(event.sign),
                "state": event.state,
                "age_seconds": round(float(current_time - event.created_at), 3),
                "matched_event_ids": list(event.matched_event_ids),
            }
            for event in events[:8]
        ]

    def _mark_loadcell_event_state(
        self,
        event_id: Optional[str],
        state: str,
        *,
        matched_event_ids: Optional[List[str]] = None,
    ) -> None:
        if event_id is None:
            return
        with self._event_lock:
            event = self._loadcell_events.get(event_id)
            if event is None:
                return
            if event.state in {"cancelled_by_return", "balanced_out"} and state == "processing":
                return
            event.state = state
            if matched_event_ids:
                for matched_id in matched_event_ids:
                    if matched_id not in event.matched_event_ids:
                        event.matched_event_ids.append(matched_id)

    def _loadcell_event_is_cancelled(self, event_id: Optional[str]) -> bool:
        if event_id is None:
            return False
        with self._event_lock:
            event = self._loadcell_events.get(event_id)
            return event is not None and event.state == "cancelled_by_return"

    @staticmethod
    def _balanced_match_allowed_error(unit_count: int) -> float:
        tolerance = float(config.door_session.weight_tolerance_grams)
        return min(15.0, max(tolerance, unit_count * tolerance))

    def _match_balanced_pending_removals(
        self,
        return_event: LoadcellEvent,
    ) -> List[str]:
        if not config.trigger.balanced_event_cancel_enabled:
            return []
        if return_event.delta_weight <= 0:
            return []

        with self._event_lock:
            candidates = [
                event
                for event in self._loadcell_events.values()
                if event.sign < 0 and event.state == "queued"
            ]

        candidates.sort(key=lambda event: (-event.abs_weight, event.created_at, event.zone))
        target_weight = return_event.abs_weight
        max_units = max(
            1,
            int(config.weight.max_count_per_item),
            int(config.weight.max_combination_size),
            int(config.weight.max_combination_items),
        )
        max_visits = max(1000, int(config.weight.max_combinations) * 50)
        best_match: Optional[List[LoadcellEvent]] = None
        best_key: Optional[tuple] = None
        visits = 0

        def consider(chosen: List[LoadcellEvent], total: float) -> None:
            nonlocal best_match, best_key
            if not chosen:
                return
            error = abs(total - target_weight)
            if error > self._balanced_match_allowed_error(len(chosen)):
                return
            key = (
                error,
                len(chosen),
                tuple((event.created_at, event.zone, event.session_id) for event in chosen),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_match = list(chosen)

        def search(start_idx: int, chosen: List[LoadcellEvent], total: float) -> None:
            nonlocal visits
            if visits >= max_visits:
                return
            visits += 1
            consider(chosen, total)
            if len(chosen) >= max_units:
                return
            if total > target_weight + self._balanced_match_allowed_error(max_units):
                return
            for index in range(start_idx, len(candidates)):
                event = candidates[index]
                chosen.append(event)
                search(index + 1, chosen, total + event.abs_weight)
                chosen.pop()

        search(0, [], 0.0)
        if not best_match:
            return []

        matched_ids = [event.event_id for event in best_match]
        with self._event_lock:
            for event_id in matched_ids:
                event = self._loadcell_events.get(event_id)
                if event is None or event.state != "queued":
                    return []
            for event_id in matched_ids:
                event = self._loadcell_events[event_id]
                event.state = "cancelled_by_return"
                event.matched_event_ids.append(return_event.event_id)
            current_return = self._loadcell_events.get(return_event.event_id)
            if current_return is not None:
                current_return.state = "balanced_out"
                current_return.matched_event_ids.extend(matched_ids)

        if self._door_session_store is not None:
            for event in best_match:
                self._door_session_store.notify_trigger_processed(
                    event.zone,
                    session_id=event.session_id,
                    status="skipped_balanced",
                )

        logger.info(
            "[TRIGGER][RELEVANCE] balanced return cancels queued removal(s): "
            f"return_session={return_event.session_id}, matched_sessions="
            f"{[event.session_id for event in best_match]}, "
            f"delta={return_event.delta_weight:.1f}g"
        )
        return matched_ids

    def _record_trigger_relevance(
        self,
        trace_context: Optional[TriggerTraceContext],
        diagnostics: dict,
    ) -> None:
        if trace_context is None:
            return
        existing = dict(getattr(trace_context, "weight_diagnostics", {}) or {})
        existing["trigger_relevance"] = dict(diagnostics)
        trace_context.record_weight_diagnostics(existing)

    @staticmethod
    def _mixed_return_hints_from_analysis(
        delta_analysis: Optional[loadcell_stats.LoadcellDeltaAnalysis],
        *,
        decision_delta: float,
    ) -> List[Dict[str, object]]:
        return loadcell_stats.mixed_return_hints_from_analysis(
            delta_analysis,
            decision_delta=decision_delta,
        )

    def _record_mixed_return_segment_diagnostics(
        self,
        trace_context: Optional[TriggerTraceContext],
        *,
        delta_analysis: Optional[loadcell_stats.LoadcellDeltaAnalysis],
        return_weight_hints: List[Dict[str, object]],
        decision_delta: float,
    ) -> None:
        if trace_context is None:
            return

        existing = dict(getattr(trace_context, "weight_diagnostics", {}) or {})
        existing["mixed_return_segments"] = (
            loadcell_stats.mixed_return_segment_diagnostics(
                delta_analysis,
                return_weight_hints=return_weight_hints,
                decision_delta=decision_delta,
            )
        )
        trace_context.record_weight_diagnostics(existing)

    @staticmethod
    def _record_effective_count_guard_diagnostics(
        trace_context: Optional[TriggerTraceContext],
        *,
        products: Sequence[object],
        delta_weight: float,
        return_weight_hints: Sequence[object],
    ) -> None:
        if trace_context is None:
            return
        existing = dict(getattr(trace_context, "weight_diagnostics", {}) or {})
        existing["effective_count_guard"] = (
            loadcell_stats.effective_count_guard_diagnostics(
                products,
                delta_weight=delta_weight,
                return_weight_hints=return_weight_hints,
            )
        )
        trace_context.record_weight_diagnostics(existing)

    @staticmethod
    def _trigger_failure_reason(
        *,
        products: List[ProductResult],
        active_product_diagnostics: Optional[dict],
    ) -> Optional[str]:
        if products:
            return None

        fail_closed_reason = (active_product_diagnostics or {}).get(
            "inference_fail_closed_reason"
        )
        if fail_closed_reason == "missing_active_product_snapshot_fail_closed":
            return "missing_active_products"
        if fail_closed_reason == "empty_allowlist_fail_closed":
            return "empty_active_product_allowlist"
        return None

    @staticmethod
    def _merge_rescue_votes(
        vote_results: List[VoteResult],
        rescue_votes: List[VoteResult],
    ) -> List[VoteResult]:
        return VideoProcessor.merge_rescue_votes(vote_results, rescue_votes)

    @staticmethod
    def _record_raw_and_filter_handled_candidates(
        *,
        vote_results: List[VoteResult],
        delta_weight: Optional[float],
        product_weights: Optional[Dict[int, float]],
        trace_context: Optional[TriggerTraceContext],
        log_prefix: str,
        zone: Optional[int] = None,
        product_stocks: Optional[Dict[int, int]] = None,
    ) -> List[VoteResult]:
        raw_count = len(vote_results or [])
        if trace_context is not None:
            trace_context.record_raw_vision_candidates(
                vote_results,
                product_weights or {},
            )
        filtered = VideoProcessor.filter_freezer_handled_candidates(
            vote_results,
            delta_weight=delta_weight,
            product_weights=product_weights or {},
            product_stocks=product_stocks or {},
            trace_context=trace_context,
            log_prefix=log_prefix,
        )
        TriggerService._log_freezer_candidate_filter_ops(
            zone=zone,
            raw_count=raw_count,
            handled_count=len(filtered),
            delta_weight=delta_weight,
            trace_context=trace_context,
        )
        return filtered

    @staticmethod
    def _log_freezer_candidate_filter_ops(
        *,
        zone: Optional[int],
        raw_count: int,
        handled_count: int,
        delta_weight: Optional[float],
        trace_context: Optional[TriggerTraceContext],
    ) -> None:
        if zone is None or str(config.machine.cabinet_type).lower() != "freezer":
            return
        diagnostics = {}
        if trace_context is not None:
            diagnostics = dict(
                (getattr(trace_context, "weight_diagnostics", {}) or {}).get(
                    "freezer_candidate_filter",
                    {},
                )
            )
        camera_layout = str(config.vision.camera_layout).lower()
        enabled = diagnostics.get(
            "freezer_handled_filter_enabled",
            camera_layout == "dual_top_proxy"
            and delta_weight is not None
            and float(delta_weight) < 0.0,
        )
        reason = diagnostics.get("reason", "not_recorded")
        selected = diagnostics.get("selected")
        selected = selected if isinstance(selected, dict) else {}
        repeat_reject = "none"
        for item in diagnostics.get("rejectedSameProductRepeatCandidates", []) or []:
            if not isinstance(item, dict):
                continue
            repeat_reject = str(
                item.get("sameProductRepeatRejectedReason")
                or item.get("repeatSelectionRejectedReason")
                or item.get("reason")
                or "rejected"
            )
            break
        ops_logger.info(
            "[OPS][FREEZER-CANDIDATE-FILTER] zone=%s camera_layout=%s "
            "enabled=%s raw=%s handled=%s reason=%s top_k=%s "
            "freezer_min_votes=%s freezer_min_ratio=%.3f "
            "freezer_motion_min_px=%.1f freezer_exit_votes=%s "
            "selected_count=%s expected_weight=%s count_residual=%s "
            "repeat_mode=%s repeat_reject=%s",
            zone,
            camera_layout,
            enabled,
            diagnostics.get("raw_candidate_count", raw_count),
            diagnostics.get("handled_candidate_count", handled_count),
            reason,
            int(config.vision.top_k),
            int(config.vision.freezer_min_vote_count),
            float(config.vision.freezer_min_vote_ratio),
            float(config.vision.freezer_motion_min_displacement_px),
            int(config.vision.freezer_min_exit_path_votes),
            selected.get("count", "n/a"),
            selected.get("expectedWeight", "n/a"),
            selected.get("countWeightResidual", "n/a"),
            selected.get("repeatEvidenceMode", "n/a"),
            repeat_reject,
        )
        if camera_layout != "dual_top_proxy":
            ops_logger.warning(
                "[OPS][CONFIG] cabinet_type=freezer camera_layout=%s "
                "expected=dual_top_proxy freezer_candidate_filter=disabled",
                camera_layout,
            )

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

        logger.info("[TRIGGER] ========== 큐 등록 시작 ==========")
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
        (
            cached_active_products,
            allowed_class_ids,
            snapshot_metadata,
        ) = self._capture_active_product_snapshot()
        if allowed_class_ids is not None:
            logger.info(
                f"[TRIGGER] strict_active_products allowed_classes={len(allowed_class_ids)} "
                "inference_classes=allowed ids="
                f"{allowed_class_ids[:10]}{'...' if len(allowed_class_ids) > 10 else ''} "
                f"snapshot_source={snapshot_metadata.get('snapshot_source', 'unknown')}"
            )
        else:
            logger.warning(
                "[TRIGGER] No active products available, YOLO candidates will be empty"
            )

        # 3. 무게 변화량 조기 계산
        delta_analysis = self._analyze_weight_delta(
            input_data.loadcells,
            cabinet_type=input_data.cabinet_type,
        )
        delta_weight = delta_analysis.decision_delta
        logger.info(
            f"[TRIGGER][loadcell] sample_count={delta_analysis.sample_count}, "
            f"parsed={delta_analysis.parsed_sample_count}, "
            f"working={delta_analysis.working_sample_count}, "
            f"span_s={delta_analysis.sample_span_seconds:.3f}, "
            f"window={delta_analysis.window_size}, "
            f"threshold={delta_analysis.stability_threshold:.1f}, "
            f"start_avg={delta_analysis.start_avg:.1f}, "
            f"end_avg={delta_analysis.end_avg:.1f}, "
            f"net_delta={delta_analysis.delta:.1f}, "
            f"decision_delta={delta_weight:.1f}, "
            f"start_idx={delta_analysis.start_stable_idx}, "
            f"end_idx={delta_analysis.end_stable_idx}, "
            f"fallback={delta_analysis.used_simple_fallback}, "
            f"reason={delta_analysis.reason}"
        )
        logger.info(f"[TRIGGER] 조기 무게 계산: delta_weight={delta_weight:.1f}g")
        payload_diagnostics = self._loadcell_trace_metadata(
            input_data.loadcells,
            delta_analysis,
            zone=input_data.zone,
            input_data=input_data,
        )
        trace_context.record_loadcell_delta(
            delta_weight=delta_weight,
            **payload_diagnostics,
        )
        ops_logger.info(
            f"[OPS][TRIGGER] zone={input_data.zone} "
            f"delta_weight={delta_weight:.1f}g "
            f"payload_state={payload_diagnostics['payload_state']} "
            f"filtered_channels={payload_diagnostics['filtered_channel_count']} "
            f"filtered_valid={payload_diagnostics['filtered_parsed_channel_count']} "
            f"filtered_zero={payload_diagnostics['filtered_zero_channel_count']} "
            f"first_filtered_total={payload_diagnostics['first_filtered_total']} "
            f"last_filtered_total={payload_diagnostics['last_filtered_total']} "
            f"analysis_reason={delta_analysis.reason} "
            f"cabinet_type={payload_diagnostics.get('cabinet_type')} "
            f"camera_layout={config.vision.camera_layout} "
            f"loadcell_scope={payload_diagnostics.get('loadcell_scope')} "
            f"loadcell_source={payload_diagnostics.get('loadcell_source')} "
            f"effective_channels={payload_diagnostics.get('effective_channel_count')} "
            f"top_video={input_data.top_video_path or 'none'} "
            f"side_video={input_data.side_video_path or 'none'}"
        )

        # 4. 무게 < 5g이면 YOLO 불필요 → 즉시 처리 (큐 거치지 않음)
        if (
            config.trigger.return_video_skip_enabled
            and self._is_return_stabilization_candidate(delta_analysis)
        ):
            delta_analysis = await self._stabilize_return_delta(
                input_data,
                delta_analysis,
                trace_context,
            )
            delta_weight = (
                delta_analysis.decision_delta
                if self._is_return_stable_enough(delta_analysis)
                else self._analysis_positive_trend(delta_analysis)
            )
            payload_diagnostics = self._loadcell_trace_metadata(
                input_data.loadcells,
                delta_analysis,
                zone=input_data.zone,
                input_data=input_data,
            )
            trace_context.record_loadcell_delta(
                delta_weight=delta_weight,
                **payload_diagnostics,
            )
            logger.info(
                f"[TRIGGER][RETURN-STABILIZE] selected_delta={delta_weight:.1f}g "
                f"stable={delta_analysis.stable_region_valid} "
                f"fallback={delta_analysis.used_simple_fallback} "
                f"reason={delta_analysis.reason}"
            )

        if (
            config.trigger.return_video_skip_enabled
            and self._is_return_stabilization_candidate(delta_analysis)
            and not self._is_return_stable_enough(delta_analysis)
        ):
            return self._handle_unstable_return_loadcell(
                input_data=input_data,
                session_id=session_id,
                delta_weight=delta_weight,
                delta_analysis=delta_analysis,
                payload_diagnostics=payload_diagnostics,
                trace_context=trace_context,
            )

        removal_stabilization = self._removal_stabilization_from_analysis(
            delta_analysis
        )
        if removal_stabilization is not None:
            active_product_diagnostics = self._record_active_product_diagnostics(
                trace_context=trace_context,
                active_products=cached_active_products,
                allowed_class_ids=allowed_class_ids,
                snapshot_metadata=snapshot_metadata,
                log_prefix="[TRIGGER]",
            )
            failure_reason = self._trigger_failure_reason(
                products=[],
                active_product_diagnostics=active_product_diagnostics,
            )
            waiting_delta = float(removal_stabilization["observed_delta"])
            trace_context.record_loadcell_delta(
                delta_weight=waiting_delta,
                **payload_diagnostics,
            )
            return self._return_unstable_removal_loadcell(
                input_data=input_data,
                session_id=session_id,
                idempotency_key=idempotency_key,
                delta_weight=waiting_delta,
                diagnostics=removal_stabilization,
                trace_context=trace_context,
                failure_reason=failure_reason,
            )

        if self._should_skip_low_weight(input_data, delta_analysis, delta_weight):
            if (
                config.trigger.low_weight_vision_fallback
                and self._has_video_path(input_data)
            ):
                return await self._handle_low_weight_video_diagnostic(
                    input_data=input_data,
                    session_id=session_id,
                    idempotency_key=idempotency_key,
                    delta_weight=delta_weight,
                    payload_diagnostics=payload_diagnostics,
                    trace_context=trace_context,
                    cached_active_products=cached_active_products,
                    allowed_class_ids=allowed_class_ids,
                    snapshot_metadata=snapshot_metadata,
                )
            return self._handle_low_weight_skip(
                input_data,
                session_id,
                idempotency_key,
                delta_weight,
                payload_diagnostics=payload_diagnostics,
                trace_context=trace_context,
            )

        if delta_weight > 0 and config.trigger.return_video_skip_enabled:
            if not self._is_return_stable_enough(delta_analysis):
                return self._handle_unstable_return_loadcell(
                    input_data=input_data,
                    session_id=session_id,
                    delta_weight=delta_weight,
                    delta_analysis=delta_analysis,
                    payload_diagnostics=payload_diagnostics,
                    trace_context=trace_context,
                )
            event = self._register_loadcell_event(
                session_id=session_id,
                zone=input_data.zone,
                delta_weight=delta_weight,
                state="return_only",
            )
            matched_event_ids = self._match_balanced_pending_removals(event)
            return self._handle_loadcell_return_only(
                input_data=input_data,
                session_id=session_id,
                idempotency_key=idempotency_key,
                delta_weight=delta_weight,
                trace_context=trace_context,
                event=event,
                matched_event_ids=matched_event_ids,
            )

        # 5. active_products 캐시 (v4.11: 조회 시점 통일)
        product_weights = self._product_weights_from_snapshot(cached_active_products)
        product_stocks = self._product_stocks_from_snapshot(cached_active_products)
        if cached_active_products:
            logger.info(
                f"[TRIGGER] v4.11: active_products {len(cached_active_products)}개 캐시됨 "
                f"(snapshot_source={snapshot_metadata.get('snapshot_source', 'unknown')})"
            )

        # 6. SessionStore에 "queued" 상태 저장
        self._record_active_product_diagnostics(
            trace_context=trace_context,
            active_products=cached_active_products,
            allowed_class_ids=allowed_class_ids,
            snapshot_metadata=snapshot_metadata,
            log_prefix="[TRIGGER]",
        )

        initial_session = SessionData(
            session_id=session_id,
            zone=input_data.zone,
            status="processing",
            processing_stage="queued",
            processing_stage_detail="큐에서 대기 중",
        )
        self._session_store.save(session_id, initial_session)
        event = self._register_loadcell_event(
            session_id=session_id,
            zone=input_data.zone,
            delta_weight=delta_weight,
            state="queued",
        )

        # 7. DoorSessionStore에 pending 알림 (CLOSE 안전장치)
        if self._door_session_store is not None:
            self._door_session_store.notify_trigger_enqueued(
                input_data.zone,
                session_id=session_id,
                chargeable_vision_required=True,
            )

        # 8. 큐에 등록
        return_weight_hints = self._mixed_return_hints_from_analysis(
            delta_analysis,
            decision_delta=delta_weight,
        )
        self._record_mixed_return_segment_diagnostics(
            trace_context,
            delta_analysis=delta_analysis,
            return_weight_hints=return_weight_hints,
            decision_delta=delta_weight,
        )

        item = QueueItem(
            input_data=input_data,
            session_id=session_id,
            idempotency_key=idempotency_key,
            delta_weight=delta_weight,
            delta_analysis=delta_analysis,
            enqueued_at=time.time(),
            event_id=event.event_id,
            chargeable_vision_required=True,
            allowed_class_ids=allowed_class_ids,
            cached_active_products=cached_active_products,
            active_product_snapshot_metadata=snapshot_metadata,
            product_weights=product_weights,
            product_stocks=product_stocks,
            trace_context=trace_context,
            return_weight_hints=return_weight_hints,
        )

        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            logger.error(
                f"[TRIGGER] Queue full ({self.QUEUE_MAX_SIZE}), rejecting trigger: "
                f"zone={input_data.zone}, session_id={session_id}"
            )
            self._mark_loadcell_event_state(event.event_id, "error")
            trace_context.finalize(
                status="error",
                error="QUEUE_FULL",
            )
            # 큐 등록 실패 → pending 알림 취소
            if self._door_session_store is not None:
                self._door_session_store.notify_trigger_processed(
                    input_data.zone,
                    session_id=session_id,
                    status="error",
                )
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

    async def _handle_low_weight_video_diagnostic(
        self,
        *,
        input_data: TriggerInput,
        session_id: str,
        idempotency_key: str,
        delta_weight: float,
        payload_diagnostics: Optional[dict],
        trace_context: Optional[TriggerTraceContext],
        cached_active_products: Optional[List],
        allowed_class_ids: Optional[List[int]],
        snapshot_metadata: Optional[dict],
    ) -> TriggerOutput:
        """Run video diagnostics for low-weight tails without charging them."""
        logger.info(
            "[TRIGGER] low-weight video diagnostic: "
            f"zone={input_data.zone}, session_id={session_id}, "
            f"delta={delta_weight:.1f}g"
        )
        active_products = cached_active_products or []
        product_weights = self._product_weights_from_snapshot(active_products)
        product_stocks = self._product_stocks_from_snapshot(active_products)
        self._record_active_product_diagnostics(
            trace_context=trace_context,
            active_products=active_products,
            allowed_class_ids=allowed_class_ids,
            snapshot_metadata=snapshot_metadata,
            log_prefix="[TRIGGER]",
        )
        self._session_store.save(
            session_id,
            SessionData(
                session_id=session_id,
                zone=input_data.zone,
                products=[],
                total_price=0,
                delta_weight=delta_weight,
                status="processing",
                processing_stage="low_weight_video_diagnostic",
                processing_stage_detail="low-weight video diagnostic only",
                trigger_timing=(
                    input_data.timing.to_dict() if input_data.timing else None
                ),
            ),
        )
        self._session_store.update_stage(
            session_id,
            processing_stage="low_weight_video_diagnostic",
            processing_stage_detail="processing video diagnostics without engine judge",
        )
        ops_logger.info(
            f"[OPS][FRAMES] zone={input_data.zone} "
            f"top_video={input_data.top_video_path or 'none'} "
            f"side_video={input_data.side_video_path or 'none'}"
        )

        video_started = time.perf_counter()
        try:
            if config.async_streaming.enabled:
                processing_result = await self._video_processor.process_videos_async(
                    top_path=input_data.top_video_path,
                    side_path=input_data.side_video_path,
                    allowed_class_ids=allowed_class_ids,
                    product_weights=product_weights,
                    trace_context=trace_context,
                    delta_weight=delta_weight,
                )
            else:
                processing_result = await asyncio.to_thread(
                    self._video_processor.process_videos,
                    top_path=input_data.top_video_path,
                    side_path=input_data.side_video_path,
                    allowed_class_ids=allowed_class_ids,
                    product_weights=product_weights,
                    trace_context=trace_context,
                    delta_weight=delta_weight,
                )
        except Exception as exc:
            if trace_context is not None:
                trace_context.finalize(status="error", error=str(exc))
            self._session_store.update_stage(
                session_id,
                processing_stage="error",
                processing_stage_detail=str(exc),
                status="error",
            )
            raise

        elapsed_ms = (time.perf_counter() - video_started) * 1000
        vote_results = list(getattr(processing_result, "vote_results", []) or [])
        vote_results = self._record_raw_and_filter_handled_candidates(
            vote_results=vote_results,
            delta_weight=delta_weight,
            product_weights=product_weights,
            product_stocks=product_stocks,
            trace_context=trace_context,
            log_prefix="TRIGGER-LOW-WEIGHT",
            zone=input_data.zone,
        )
        stats = getattr(processing_result, "stats", None)
        if trace_context is not None:
            if stats is not None:
                trace_context.record_video_stats(stats)
            trace_context.record_active_product_snapshot(
                active_products,
                delta_weight=delta_weight,
            )
            trace_context.record_candidates(vote_results, product_weights)
            diagnostics = dict(
                getattr(trace_context, "weight_diagnostics", {}) or {}
            )
            diagnostics.update(
                {
                    "decision_branch": "low_weight_video_diagnostic",
                    "ignored_low_weight_delta": True,
                    "threshold_grams": float(self.MIN_WEIGHT_CHANGE_GRAMS),
                    "excluded_from_close_summary": True,
                    "engine_skipped": True,
                    "delta_weight": round(float(delta_weight), 1),
                    "video_diagnostic_candidate_count": len(vote_results),
                    "video_diagnostic_elapsed_ms": round(float(elapsed_ms), 1),
                }
            )
            if payload_diagnostics:
                diagnostics.update(
                    {
                        "payload_state": payload_diagnostics.get("payload_state"),
                        "raw_state": payload_diagnostics.get("raw_state"),
                        "filtered_state": payload_diagnostics.get("filtered_state"),
                        "first_raw_total": payload_diagnostics.get("first_raw_total"),
                        "last_raw_total": payload_diagnostics.get("last_raw_total"),
                        "first_filtered_total": payload_diagnostics.get(
                            "first_filtered_total"
                        ),
                        "last_filtered_total": payload_diagnostics.get(
                            "last_filtered_total"
                        ),
                    }
                )
            trace_context.record_weight_diagnostics(diagnostics)
            trace_context.record_final_result(
                products=[],
                total_price=0,
                status="complete",
                confidence=0.0,
            )
            trace_context.record_storage_result(products=[], total_price=0)
            trace_context.finalize(status="complete")

        top_frames = int(getattr(stats, "top_frames", 0) or 0) if stats else 0
        side_frames = int(getattr(stats, "side_frames", 0) or 0) if stats else 0
        processing_time_ms = (
            float(getattr(stats, "processing_time_ms", elapsed_ms) or elapsed_ms)
            if stats
            else elapsed_ms
        )
        session_data = SessionData(
            session_id=session_id,
            zone=input_data.zone,
            products=[],
            total_price=0,
            delta_weight=delta_weight,
            status="complete",
            processing_stage="low_weight_video_diagnostic",
            processing_stage_detail="engine skipped; excluded from close summary",
            confidence=0.0,
            top_frames=top_frames,
            side_frames=side_frames,
            processing_time_ms=processing_time_ms,
            vision_candidates=[],
            trigger_timing=input_data.timing.to_dict() if input_data.timing else None,
        )
        self._session_store.save(session_id, session_data)
        diagnostic_reason = (
            self._loadcell_payload_issue_reason(payload_diagnostics)
            or "low_weight_video_diagnostic"
        )
        self._record_no_charge_diagnostic(
            input_data=input_data,
            session_id=session_id,
            reason=diagnostic_reason,
            delta_weight=delta_weight,
            processing_stage="low_weight_video_diagnostic",
            payload_diagnostics=payload_diagnostics,
            message="engine skipped; excluded from close summary",
        )
        self._register_request(idempotency_key, session_id)
        ops_logger.info(
            f"[OPS][RESULT] zone={input_data.zone} "
            "status=low_weight_video_diagnostic products=none total_price=0"
        )
        return TriggerOutput(
            success=True,
            session_id=session_id,
            door_session_id=None,
            message=(
                f"low weight delta ({abs(delta_weight):.1f}g), "
                "video diagnostic only"
            ),
            status="complete",
        )

    def _handle_low_weight_skip(
        self,
        input_data: TriggerInput,
        session_id: str,
        idempotency_key: str,
        delta_weight: float,
        payload_diagnostics: Optional[dict] = None,
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
        payload_issue_reason = self._loadcell_payload_issue_reason(payload_diagnostics)
        processing_stage = (
            f"skipped_{payload_issue_reason}"
            if payload_issue_reason
            else "skipped_low_weight"
        )
        processing_detail = (
            f"loadcell payload diagnostic: {payload_issue_reason}"
            if payload_issue_reason
            else f"low weight delta ({abs(delta_weight):.1f}g)"
        )
        session_data = SessionData(
            session_id=session_id,
            zone=input_data.zone,
            products=[],
            total_price=0,
            delta_weight=delta_weight,
            status="complete",
            processing_stage=processing_stage,
            trigger_timing=input_data.timing.to_dict() if input_data.timing else None,
            processing_stage_detail=processing_detail,
        )
        self._session_store.save(session_id, session_data)

        door_session_id = None
        if self._door_session_store is not None:
            logger.info(
                "[TRIGGER] ignored low-weight trigger excluded from DoorSession "
                f"summary: zone={input_data.zone}, session_id={session_id}, "
                f"delta={delta_weight:.1f}g"
            )
            self._record_no_charge_diagnostic(
                input_data=input_data,
                session_id=session_id,
                reason=payload_issue_reason or "low_weight_ignored",
                delta_weight=delta_weight,
                processing_stage=processing_stage,
                payload_diagnostics=payload_diagnostics,
                message=processing_detail,
            )

        self._register_request(idempotency_key, session_id)
        if trace_context is not None:
            diagnostics = {
                "decision_branch": (
                    "loadcell_payload_diagnostic"
                    if payload_issue_reason
                    else "low_weight_ignored"
                ),
                "ignored_low_weight_delta": True,
                "threshold_grams": float(self.MIN_WEIGHT_CHANGE_GRAMS),
                "excluded_from_close_summary": True,
                "delta_weight": round(float(delta_weight), 1),
                "loadcell_payload_issue": bool(payload_issue_reason),
                "loadcell_payload_reason": payload_issue_reason,
            }
            if payload_diagnostics:
                diagnostics.update(
                    {
                        "payload_state": payload_diagnostics.get("payload_state"),
                        "raw_state": payload_diagnostics.get("raw_state"),
                        "filtered_state": payload_diagnostics.get("filtered_state"),
                        "first_raw_total": payload_diagnostics.get("first_raw_total"),
                        "last_raw_total": payload_diagnostics.get("last_raw_total"),
                        "first_filtered_total": payload_diagnostics.get(
                            "first_filtered_total"
                        ),
                        "last_filtered_total": payload_diagnostics.get(
                            "last_filtered_total"
                        ),
                    }
                )
            trace_context.record_weight_diagnostics(diagnostics)
            trace_context.finalize(status="skipped")
        return TriggerOutput(
            success=True,
            session_id=session_id,
            door_session_id=door_session_id,
            message=f"무게 변화 미미 ({abs(delta_weight):.1f}g), 스킵",
            status="skipped",
        )

    def _handle_unstable_return_loadcell(
        self,
        input_data: TriggerInput,
        session_id: str,
        delta_weight: float,
        delta_analysis: loadcell_stats.LoadcellDeltaAnalysis,
        payload_diagnostics: dict,
        trace_context: Optional[TriggerTraceContext],
    ) -> TriggerOutput:
        detail = (
            "return loadcell is not stable yet; retry trigger after stable tail"
        )
        session_data = SessionData(
            session_id=session_id,
            zone=input_data.zone,
            products=[],
            total_price=0,
            delta_weight=delta_weight,
            status="waiting",
            processing_stage="return_waiting_for_stable_loadcell",
            processing_stage_detail=detail,
            confidence=0.0,
            top_frames=0,
            side_frames=0,
            processing_time_ms=0.0,
            vision_candidates=[],
            trigger_timing=input_data.timing.to_dict() if input_data.timing else None,
        )
        self._session_store.save(session_id, session_data)
        self._record_trigger_relevance(
            trace_context,
            {
                "chargeable_vision_required": False,
                "balanced_out": False,
                "skip_reason": "return_waiting_for_stable_loadcell",
                "delta_weight": round(float(delta_weight), 1),
                "analysis_reason": delta_analysis.reason,
                "stable_region_valid": delta_analysis.stable_region_valid,
                "used_simple_fallback": delta_analysis.used_simple_fallback,
                "payload_state": payload_diagnostics.get("payload_state"),
            },
        )
        if trace_context is not None:
            trace_context.record_final_result(
                products=[],
                total_price=0,
                status="return_waiting_for_stable_loadcell",
                confidence=0.0,
            )
            trace_context.record_storage_result(products=[], total_price=0)
            trace_context.finalize(status="waiting")

        ops_logger.info(
            f"[OPS][RESULT] zone={input_data.zone} "
            "status=return_waiting_for_stable_loadcell "
            f"delta={delta_weight:.1f}g reason={delta_analysis.reason} "
            f"stable={delta_analysis.stable_region_valid} "
            f"fallback={delta_analysis.used_simple_fallback}"
        )
        return TriggerOutput(
            success=True,
            session_id=session_id,
            door_session_id=None,
            message=detail,
            status="waiting",
            waiting_for="stable_loadcell",
        )

    def _handle_loadcell_return_only(
        self,
        input_data: TriggerInput,
        session_id: str,
        idempotency_key: str,
        delta_weight: float,
        trace_context: Optional[TriggerTraceContext],
        event: LoadcellEvent,
        matched_event_ids: List[str],
    ) -> TriggerOutput:
        balanced_out = bool(matched_event_ids)
        stage = "balanced_out" if balanced_out else "loadcell_return_only"
        detail = (
            "반환이 대기 중 제거 trigger와 상쇄되어 비디오 처리 생략"
            if balanced_out
            else "반환 trigger는 loadcell-only로 처리"
        )
        session_data = SessionData(
            session_id=session_id,
            zone=input_data.zone,
            products=[],
            total_price=0,
            delta_weight=delta_weight,
            status="complete",
            processing_stage=stage,
            processing_stage_detail=detail,
            confidence=1.0 if balanced_out else 0.0,
            top_frames=0,
            side_frames=0,
            processing_time_ms=0.0,
            vision_candidates=[],
            trigger_timing=input_data.timing.to_dict() if input_data.timing else None,
        )
        self._session_store.save(session_id, session_data)

        door_session_id = None
        if self._door_session_store is not None and not balanced_out:
            trigger_result = TriggerResult(
                trigger_id="",
                session_id=session_id,
                timestamp=event.created_at,
                products=[],
                delta_weight=delta_weight,
                confidence=0.0,
                video_paths={
                    "top": str(input_data.top_video_path) if input_data.top_video_path else "",
                    "side": str(input_data.side_video_path) if input_data.side_video_path else "",
                },
                is_return=True,
                processing_time_ms=0.0,
                timing_metadata=input_data.timing.to_dict() if input_data.timing else None,
            )
            door_session = self._door_session_store.add_trigger_with_global(
                zone=input_data.zone,
                result=trigger_result,
            )
            door_session_id = door_session.door_session_id

        self._register_request(idempotency_key, session_id)
        self._record_trigger_relevance(
            trace_context,
            {
                "chargeable_vision_required": False,
                "balanced_out": balanced_out,
                "matched_event_ids": list(matched_event_ids),
                "skip_reason": "balanced_out" if balanced_out else "return_loadcell_only",
                "event_id": event.event_id,
            },
        )
        if trace_context is not None:
            trace_context.record_final_result(
                products=[],
                total_price=0,
                status=stage,
                confidence=1.0 if balanced_out else 0.0,
            )
            trace_context.record_storage_result(products=[], total_price=0)
            trace_context.finalize(status="skipped" if balanced_out else "complete")

        ops_logger.info(
            f"[OPS][RESULT] zone={input_data.zone} status={stage} "
            f"products=none total_price=0"
        )
        return TriggerOutput(
            success=True,
            session_id=session_id,
            door_session_id=door_session_id,
            message=detail,
            status="complete",
        )

    def _handle_cancelled_queue_item(
        self,
        item: QueueItem,
        trace_context: TriggerTraceContext,
    ) -> None:
        input_data = item.input_data
        session_data = SessionData(
            session_id=item.session_id,
            zone=input_data.zone,
            products=[],
            total_price=0,
            delta_weight=item.delta_weight,
            status="complete",
            processing_stage="skipped_balanced",
            processing_stage_detail="대기 중 제거 trigger가 이후 반환과 상쇄됨",
            confidence=1.0,
            top_frames=0,
            side_frames=0,
            processing_time_ms=0.0,
            vision_candidates=[],
            trigger_timing=input_data.timing.to_dict() if input_data.timing else None,
        )
        self._session_store.save(item.session_id, session_data)
        self._record_trigger_relevance(
            trace_context,
            {
                "chargeable_vision_required": False,
                "balanced_out": True,
                "matched_event_ids": [],
                "skip_reason": "cancelled_by_return",
                "event_id": item.event_id,
            },
        )
        trace_context.record_loadcell_delta(
            delta_weight=item.delta_weight,
            **self._loadcell_trace_metadata(
                input_data.loadcells,
                item.delta_analysis,
                zone=input_data.zone,
                event_id=item.event_id,
                input_data=input_data,
            ),
        )
        trace_context.record_final_result(
            products=[],
            total_price=0,
            status="skipped_balanced",
            confidence=1.0,
        )
        trace_context.record_storage_result(products=[], total_price=0)
        trace_context.finalize(status="skipped")
        if self._door_session_store is not None:
            self._door_session_store.notify_trigger_processed(
                input_data.zone,
                session_id=item.session_id,
                status="skipped_balanced",
            )
        ops_logger.info(
            f"[OPS][RESULT] zone={input_data.zone} status=skipped_balanced "
            "products=none total_price=0"
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
                self._mark_loadcell_event_state(item.event_id, "error")
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
                        item.input_data.zone,
                        session_id=item.session_id,
                        status="error",
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
        queue_wait_ms = (start_time - item.enqueued_at) * 1000

        logger.info(
            f"[TRIGGER-WORKER] Processing: zone={input_data.zone}, "
            f"session_id={session_id}, "
            f"wait_time={start_time - item.enqueued_at:.1f}s"
        )
        if (
            config.trigger.cooperative_cancel_enabled
            and self._loadcell_event_is_cancelled(item.event_id)
        ):
            self._handle_cancelled_queue_item(item, trace_context)
            return

        self._mark_loadcell_event_state(item.event_id, "processing")
        if self._door_session_store is not None:
            self._door_session_store.notify_trigger_started(
                input_data.zone,
                session_id=session_id,
            )
        trace_context.record_loadcell_delta(
            delta_weight=item.delta_weight,
            **self._loadcell_trace_metadata(
                input_data.loadcells,
                item.delta_analysis,
                zone=input_data.zone,
                event_id=item.event_id,
                input_data=input_data,
            ),
        )
        active_product_diagnostics = self._record_active_product_diagnostics(
            trace_context=trace_context,
            active_products=item.cached_active_products or [],
            allowed_class_ids=item.allowed_class_ids,
            snapshot_metadata=item.active_product_snapshot_metadata,
            log_prefix="[TRIGGER-WORKER]",
        )
        ops_logger.info(
            f"[OPS][FRAMES] zone={input_data.zone} "
            f"top_video={input_data.top_video_path or 'none'} "
            f"side_video={input_data.side_video_path or 'none'}"
        )

        # 1. SessionStore 상태 업데이트
        self._session_store.update_stage(
            session_id,
            processing_stage="extracting_frames",
            processing_stage_detail="비디오에서 프레임 추출 중",
        )

        # 2. 비디오 처리 (YOLO 추론) - GPU 독점
        # v5.3: feature flag 기반 async/sync 선택
        video_started = time.perf_counter()
        try:
            if (
                config.trigger.cooperative_cancel_enabled
                and self._loadcell_event_is_cancelled(item.event_id)
            ):
                self._handle_cancelled_queue_item(item, trace_context)
                return
            if config.async_streaming.enabled:
                logger.info("[TRIGGER-WORKER] v5.3: Async streaming 모드로 비디오 처리")
                processing_result = await self._video_processor.process_videos_async(
                    top_path=input_data.top_video_path,
                    side_path=input_data.side_video_path,
                    allowed_class_ids=item.allowed_class_ids,
                    product_weights=item.product_weights or {},
                    trace_context=trace_context,
                    delta_weight=item.delta_weight,
                )
            else:
                processing_result = await asyncio.to_thread(
                    self._video_processor.process_videos,
                    top_path=input_data.top_video_path,
                    side_path=input_data.side_video_path,
                    allowed_class_ids=item.allowed_class_ids,
                    product_weights=item.product_weights or {},
                    trace_context=trace_context,
                    delta_weight=item.delta_weight,
                )
        except Exception as exc:
            trace_context.finalize(status="error", error=str(exc))
            raise
        video_elapsed_ms = (time.perf_counter() - video_started) * 1000

        active_products = item.cached_active_products or []
        existing_class_ids = {vote.class_id for vote in processing_result.vote_results}
        threshold_rescue_diagnostics: dict = {}
        rescue_votes = VideoProcessor.build_weight_gated_rescue_votes(
            getattr(processing_result, "threshold_rescue_candidates", []),
            active_products,
            item.delta_weight,
            diagnostics=threshold_rescue_diagnostics,
            existing_class_ids=existing_class_ids,
        )
        roi_rescue_diagnostics: dict = {}
        roi_rescue_votes = VideoProcessor.build_weight_gated_rescue_votes(
            getattr(processing_result, "roi_rescue_candidates", []),
            active_products,
            item.delta_weight,
            diagnostics=roi_rescue_diagnostics,
            existing_class_ids=existing_class_ids,
        )
        if rescue_votes:
            logger.info(
                f"[TRIGGER-WORKER][THRESHOLD-RESCUE] weight_gated={len(rescue_votes)}"
            )
        if roi_rescue_votes:
            logger.info(
                f"[TRIGGER-WORKER][ROI-RESCUE] weight_gated={len(roi_rescue_votes)}"
            )
        vote_results = self._merge_rescue_votes(
            processing_result.vote_results,
            rescue_votes + roi_rescue_votes,
        )
        vote_results = self._record_raw_and_filter_handled_candidates(
            vote_results=vote_results,
            delta_weight=item.delta_weight,
            product_weights=item.product_weights or {},
            product_stocks=item.product_stocks or {},
            trace_context=trace_context,
            log_prefix="TRIGGER-WORKER",
            zone=input_data.zone,
        )
        stats = processing_result.stats
        trace_context.record_video_stats(stats)
        trace_context.record_active_product_snapshot(
            active_products,
            delta_weight=item.delta_weight,
        )
        trace_context.record_rescue_diagnostics(
            "threshold_rescue",
            threshold_rescue_diagnostics,
        )
        trace_context.record_rescue_diagnostics(
            "roi_rescue",
            roi_rescue_diagnostics,
        )
        trace_context.record_candidates(vote_results, item.product_weights or {})

        logger.info(
            f"[TRIGGER-WORKER] 비디오 처리 완료: zone={input_data.zone}, "
            f"프레임={stats.top_frames + stats.side_frames}, "
            f"후보={len(vote_results)}개, 시간={stats.processing_time_ms:.1f}ms"
        )
        self._log_candidate_ops(
            input_data.zone,
            vote_results,
            item.product_weights or {},
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
        delta_analysis = item.delta_analysis or loadcell_stats.LoadcellDeltaAnalysis(
            delta=delta_weight,
        )
        vision_only = (
            self._should_force_vision_only(input_data, delta_analysis)
            or (delta_weight == 0.0 and len(input_data.loadcells) == 0)
        )
        removal_stabilization = self._removal_stabilization_conflict(
            vision_candidates=vision_candidates,
            delta_weight=delta_weight,
            active_products=active_products,
        )
        if removal_stabilization is not None:
            failure_reason = self._trigger_failure_reason(
                products=[],
                active_product_diagnostics=active_product_diagnostics,
            )
            self._handle_unstable_removal_loadcell(
                input_data=input_data,
                session_id=session_id,
                delta_weight=delta_weight,
                diagnostics=removal_stabilization,
                trace_context=trace_context,
                stats=stats,
                vision_candidates=vision_candidates,
                event_id=item.event_id,
                failure_reason=failure_reason,
            )
            return

        engine_started = time.perf_counter()
        if active_products:
            logger.info(
                f"[TRIGGER-WORKER] active_products {len(active_products)}개 → engine.judge()에 전달"
            )

        result = self._engine.judge(
            vision_candidates=vision_candidates,
            delta_weight=delta_weight,
            vision_only=vision_only,
            active_products=active_products,
            trace_context=trace_context,
        )

        # 6. Node.js 상품 리스트에 없는 상품 제거 (v4.6)
        engine_elapsed_ms = (time.perf_counter() - engine_started) * 1000
        filtered_engine_products = self._products_for_storage(
            result,
            log_prefix="[TRIGGER-WORKER]",
        )
        filtered_engine_products = self._filter_products_by_snapshot(
            filtered_engine_products,
            active_products,
            log_prefix="[TRIGGER-WORKER]",
        )
        trace_context.record_final_result(
            products=filtered_engine_products,
            total_price=sum(p.total_price for p in filtered_engine_products),
            status=result.status.value,
            confidence=result.confidence,
        )
        # 7. SessionStore에 결과 저장
        products = [
            ProductResult(
                product_id=p.product_id,
                product_idx=self._get_product_idx(p.product_id, active_products),
                name=p.name,
                count=p.count,
                price=p.unit_price,
                confidence=p.confidence,
            )
            for p in filtered_engine_products
        ]

        final_total_price = sum(p.price * p.count for p in products)
        trace_context.record_storage_result(
            products=products,
            total_price=final_total_price,
        )
        vision_candidates_dicts = [vc.to_dict() for vc in vision_candidates]
        close_candidate_snapshot = build_trigger_candidate_snapshot(
            vision_candidates,
            active_products,
        )
        failure_reason = self._trigger_failure_reason(
            products=products,
            active_product_diagnostics=active_product_diagnostics,
        )
        self._record_mixed_return_segment_diagnostics(
            trace_context,
            delta_analysis=item.delta_analysis,
            return_weight_hints=list(item.return_weight_hints),
            decision_delta=delta_weight,
        )
        self._record_effective_count_guard_diagnostics(
            trace_context,
            products=filtered_engine_products,
            delta_weight=delta_weight,
            return_weight_hints=list(item.return_weight_hints),
        )
        if self._door_session_store is not None:
            self._door_session_store.notify_trigger_finalizing(
                input_data.zone,
                session_id=session_id,
            )

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
        door_session_elapsed_ms = 0.0
        if self._door_session_store is not None:
            door_session_started = time.perf_counter()
            elapsed_ms = (time.time() - start_time) * 1000
            trigger_result = TriggerResult(
                trigger_id="",
                session_id=session_id,
                timestamp=item.enqueued_at,
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
                failure_reason=failure_reason,
                return_weight_hints=list(item.return_weight_hints),
                vision_candidates=close_candidate_snapshot,
            )
            door_session = self._door_session_store.add_trigger_with_global(
                zone=input_data.zone,
                result=trigger_result,
            )
            door_session_id = door_session.door_session_id

            # Keep pending until trace finalization below completes.
            door_session_elapsed_ms = (
                time.perf_counter() - door_session_started
            ) * 1000

            logger.info(
                f"[TRIGGER-WORKER] Door session: {door_session_id}, "
                f"triggers={door_session.trigger_count}, "
                f"aggregated_products={len(door_session.aggregated_products)}"
            )

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info("[TRIGGER-WORKER] ========== 판단 결과 ==========")
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
        logger.info(
            f"[TRIGGER-WORKER][LATENCY] session_id={session_id} "
            f"zone={input_data.zone} queue_wait_ms={queue_wait_ms:.1f} "
            f"video_ms={video_elapsed_ms:.1f} "
            f"video_stats_ms={getattr(stats, 'processing_time_ms', 0.0):.1f} "
            f"frame_stride={getattr(stats, 'frame_stride', 2)} "
            f"original_frames={getattr(stats, 'original_frames', 0)} "
            f"processed_frames="
            f"{getattr(stats, 'processed_frames', getattr(stats, 'top_frames', 0) + getattr(stats, 'side_frames', 0))} "
            f"skipped_frames={getattr(stats, 'skipped_frames', 0)} "
            f"yolo_total_ms={getattr(stats, 'yolo_total_time_ms', 0.0):.1f} "
            f"yolo_avg_ms={getattr(stats, 'yolo_avg_time_ms', 0.0):.1f} "
            f"yolo_count={getattr(stats, 'yolo_inference_count', 0)} "
            f"engine_ms={engine_elapsed_ms:.1f} "
            f"door_session_ms={door_session_elapsed_ms:.1f} "
            f"total_ms={elapsed_ms:.1f}"
        )
        ops_logger.info(
            f"[OPS][RESULT] zone={input_data.zone} "
            f"status={result.status.value} "
            f"products={self._format_products_for_ops(products)} "
            f"product_count={sum(p.count for p in products)} "
            f"total_price={final_total_price}"
        )
        trace_context.finalize(status="complete")
        self._mark_loadcell_event_state(item.event_id, "complete")
        if self._door_session_store is not None:
            self._door_session_store.notify_trigger_processed(
                input_data.zone,
                session_id=session_id,
                status="complete",
            )

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

        logger.info("[TRIGGER] ========== 추론 시작 ==========")
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
        (
            cached_active_products,
            allowed_class_ids,
            snapshot_metadata,
        ) = self._capture_active_product_snapshot()
        if allowed_class_ids is not None:
            logger.info(
                f"[TRIGGER] strict_active_products allowed_classes={len(allowed_class_ids)} "
                "inference_classes=allowed ids="
                f"{allowed_class_ids[:10]}{'...' if len(allowed_class_ids) > 10 else ''} "
                f"snapshot_source={snapshot_metadata.get('snapshot_source', 'unknown')}"
            )
        else:
            logger.warning(
                "[TRIGGER] No active products available, YOLO candidates will be empty"
            )

        # 1-C. 무게 변화량 조기 계산 (v4.6 - 비디오 처리 전 검증)
        delta_analysis = self._analyze_weight_delta(
            input_data.loadcells,
            cabinet_type=input_data.cabinet_type,
        )
        delta_weight = delta_analysis.decision_delta
        logger.info(f"[TRIGGER] 조기 무게 계산: delta_weight={delta_weight:.1f}g")
        payload_diagnostics = self._loadcell_trace_metadata(
            input_data.loadcells,
            delta_analysis,
            zone=input_data.zone,
            input_data=input_data,
        )
        trace_context.record_loadcell_delta(
            delta_weight=delta_weight,
            **payload_diagnostics,
        )
        ops_logger.info(
            f"[OPS][TRIGGER] zone={input_data.zone} "
            f"delta_weight={delta_weight:.1f}g "
            f"payload_state={payload_diagnostics['payload_state']} "
            f"filtered_channels={payload_diagnostics['filtered_channel_count']} "
            f"filtered_valid={payload_diagnostics['filtered_parsed_channel_count']} "
            f"filtered_zero={payload_diagnostics['filtered_zero_channel_count']} "
            f"first_filtered_total={payload_diagnostics['first_filtered_total']} "
            f"last_filtered_total={payload_diagnostics['last_filtered_total']} "
            f"analysis_reason={delta_analysis.reason} "
            f"cabinet_type={payload_diagnostics.get('cabinet_type')} "
            f"camera_layout={config.vision.camera_layout} "
            f"loadcell_scope={payload_diagnostics.get('loadcell_scope')} "
            f"loadcell_source={payload_diagnostics.get('loadcell_source')} "
            f"effective_channels={payload_diagnostics.get('effective_channel_count')} "
            f"top_video={input_data.top_video_path or 'none'} "
            f"side_video={input_data.side_video_path or 'none'}"
        )

        if (
            config.trigger.return_video_skip_enabled
            and self._is_return_stabilization_candidate(delta_analysis)
        ):
            delta_analysis = await self._stabilize_return_delta(
                input_data,
                delta_analysis,
                trace_context,
            )
            delta_weight = (
                delta_analysis.decision_delta
                if self._is_return_stable_enough(delta_analysis)
                else self._analysis_positive_trend(delta_analysis)
            )
            payload_diagnostics = self._loadcell_trace_metadata(
                input_data.loadcells,
                delta_analysis,
                zone=input_data.zone,
                input_data=input_data,
            )
            trace_context.record_loadcell_delta(
                delta_weight=delta_weight,
                **payload_diagnostics,
            )

        if (
            config.trigger.return_video_skip_enabled
            and self._is_return_stabilization_candidate(delta_analysis)
            and not self._is_return_stable_enough(delta_analysis)
        ):
            return self._handle_unstable_return_loadcell(
                input_data=input_data,
                session_id=session_id,
                delta_weight=delta_weight,
                delta_analysis=delta_analysis,
                payload_diagnostics=payload_diagnostics,
                trace_context=trace_context,
            )

        removal_stabilization = self._removal_stabilization_from_analysis(
            delta_analysis
        )
        if removal_stabilization is not None:
            active_product_diagnostics = self._record_active_product_diagnostics(
                trace_context=trace_context,
                active_products=cached_active_products,
                allowed_class_ids=allowed_class_ids,
                snapshot_metadata=snapshot_metadata,
                log_prefix="[TRIGGER]",
            )
            failure_reason = self._trigger_failure_reason(
                products=[],
                active_product_diagnostics=active_product_diagnostics,
            )
            waiting_delta = float(removal_stabilization["observed_delta"])
            trace_context.record_loadcell_delta(
                delta_weight=waiting_delta,
                **payload_diagnostics,
            )
            return self._return_unstable_removal_loadcell(
                input_data=input_data,
                session_id=session_id,
                idempotency_key=idempotency_key,
                delta_weight=waiting_delta,
                diagnostics=removal_stabilization,
                trace_context=trace_context,
                failure_reason=failure_reason,
            )

        if self._should_skip_low_weight(input_data, delta_analysis, delta_weight):
            if (
                config.trigger.low_weight_vision_fallback
                and self._has_video_path(input_data)
            ):
                return await self._handle_low_weight_video_diagnostic(
                    input_data=input_data,
                    session_id=session_id,
                    idempotency_key=idempotency_key,
                    delta_weight=delta_weight,
                    payload_diagnostics=payload_diagnostics,
                    trace_context=trace_context,
                    cached_active_products=cached_active_products,
                    allowed_class_ids=allowed_class_ids,
                    snapshot_metadata=snapshot_metadata,
                )
            return self._handle_low_weight_skip(
                input_data,
                session_id,
                idempotency_key,
                delta_weight,
                payload_diagnostics=payload_diagnostics,
                trace_context=trace_context,
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
        product_weights = self._product_weights_from_snapshot(cached_active_products)
        product_stocks = self._product_stocks_from_snapshot(cached_active_products)
        if cached_active_products:
            logger.info(
                f"[TRIGGER] v4.11: active_products {len(cached_active_products)}개 캐시됨 "
                f"(snapshot_source={snapshot_metadata.get('snapshot_source', 'unknown')})"
            )

        # v5.3: feature flag 기반 async/sync 선택
        active_product_diagnostics = self._record_active_product_diagnostics(
            trace_context=trace_context,
            active_products=cached_active_products,
            allowed_class_ids=allowed_class_ids,
            snapshot_metadata=snapshot_metadata,
            log_prefix="[TRIGGER]",
        )

        try:
            ops_logger.info(
                f"[OPS][FRAMES] zone={input_data.zone} "
                f"top_video={input_data.top_video_path or 'none'} "
                f"side_video={input_data.side_video_path or 'none'}"
            )
            if config.async_streaming.enabled:
                logger.info("[TRIGGER] v5.3: Async streaming 모드로 비디오 처리")
                processing_result = await self._video_processor.process_videos_async(
                    top_path=input_data.top_video_path,
                    side_path=input_data.side_video_path,
                    allowed_class_ids=allowed_class_ids,
                    product_weights=product_weights,
                    trace_context=trace_context,
                    delta_weight=delta_weight,
                )
            else:
                processing_result = await asyncio.to_thread(
                    self._video_processor.process_videos,
                    top_path=input_data.top_video_path,
                    side_path=input_data.side_video_path,
                    allowed_class_ids=allowed_class_ids,
                    product_weights=product_weights,
                    trace_context=trace_context,
                    delta_weight=delta_weight,
                )
        except Exception as exc:
            trace_context.finalize(status="error", error=str(exc))
            raise

        existing_class_ids = {vote.class_id for vote in processing_result.vote_results}
        threshold_rescue_diagnostics: dict = {}
        rescue_votes = VideoProcessor.build_weight_gated_rescue_votes(
            getattr(processing_result, "threshold_rescue_candidates", []),
            cached_active_products,
            delta_weight,
            diagnostics=threshold_rescue_diagnostics,
            existing_class_ids=existing_class_ids,
        )
        roi_rescue_diagnostics: dict = {}
        roi_rescue_votes = VideoProcessor.build_weight_gated_rescue_votes(
            getattr(processing_result, "roi_rescue_candidates", []),
            cached_active_products,
            delta_weight,
            diagnostics=roi_rescue_diagnostics,
            existing_class_ids=existing_class_ids,
        )
        if rescue_votes:
            logger.info(f"[TRIGGER][THRESHOLD-RESCUE] weight_gated={len(rescue_votes)}")
        if roi_rescue_votes:
            logger.info(f"[TRIGGER][ROI-RESCUE] weight_gated={len(roi_rescue_votes)}")
        vote_results = self._merge_rescue_votes(
            processing_result.vote_results,
            rescue_votes + roi_rescue_votes,
        )
        vote_results = self._record_raw_and_filter_handled_candidates(
            vote_results=vote_results,
            delta_weight=delta_weight,
            product_weights=product_weights,
            product_stocks=product_stocks,
            trace_context=trace_context,
            log_prefix="TRIGGER",
            zone=input_data.zone,
        )
        stats = processing_result.stats
        trace_context.record_video_stats(stats)
        trace_context.record_active_product_snapshot(
            cached_active_products,
            delta_weight=delta_weight,
        )
        trace_context.record_rescue_diagnostics(
            "threshold_rescue",
            threshold_rescue_diagnostics,
        )
        trace_context.record_rescue_diagnostics(
            "roi_rescue",
            roi_rescue_diagnostics,
        )
        trace_context.record_candidates(vote_results, product_weights)

        logger.info("[TRIGGER] ========== 비디오 처리 완료 ==========")
        logger.info(
            f"[TRIGGER] 총 프레임: {stats.top_frames + stats.side_frames}, "
            f"후보: {len(vote_results)}개, 처리시간: {stats.processing_time_ms:.1f}ms"
        )
        self._log_candidate_ops(input_data.zone, vote_results, product_weights)

        # 4. 처리 단계 업데이트
        self._session_store.update_stage(
            session_id,
            processing_stage="calculating_count",
            processing_stage_detail=f"후보 {len(vote_results)}개 도출, 개수 판단 중",
        )

        # 5. 무게 변화량 (이미 1-C에서 계산됨, v4.6)
        logger.info("[TRIGGER] ========== 무게 확인 ==========")
        logger.info(f"[TRIGGER] delta_weight={delta_weight:.1f}g (조기 계산값 사용)")

        # 6. 투표 결과를 EnsembleResult로 변환
        vision_candidates = self._vote_results_to_ensemble(vote_results)

        # 7. 최종 상품 판단 - v4.7: active_products 전달
        vision_only = (
            self._should_force_vision_only(input_data, delta_analysis)
            or (delta_weight == 0.0 and len(input_data.loadcells) == 0)
        )

        # v4.11: 캐시된 active_products 사용 (비디오 처리 중 store가 clear될 수 있음)
        active_products = cached_active_products  # v4.11: 캐시된 값 사용
        if active_products:
            logger.info(f"[TRIGGER] v4.7: active_products {len(active_products)}개 → engine.judge()에 전달")

        removal_stabilization = self._removal_stabilization_conflict(
            vision_candidates=vision_candidates,
            delta_weight=delta_weight,
            active_products=active_products,
        )
        if removal_stabilization is not None:
            failure_reason = self._trigger_failure_reason(
                products=[],
                active_product_diagnostics=active_product_diagnostics,
            )
            self._handle_unstable_removal_loadcell(
                input_data=input_data,
                session_id=session_id,
                delta_weight=delta_weight,
                diagnostics=removal_stabilization,
                trace_context=trace_context,
                stats=stats,
                vision_candidates=vision_candidates,
                failure_reason=failure_reason,
            )
            return TriggerOutput(
                success=True,
                session_id=session_id,
                door_session_id=None,
                message="removal loadcell is not stable yet; retry trigger after stable tail",
                status="waiting",
                waiting_for="stable_loadcell",
            )

        result = self._engine.judge(
            vision_candidates=vision_candidates,
            delta_weight=delta_weight,
            vision_only=vision_only,
            active_products=active_products,  # v4.7: 신규 파라미터
            trace_context=trace_context,
        )

        # 7-B. Node.js 상품 리스트에 없는 상품 제거 (v4.6)
        filtered_engine_products = self._products_for_storage(result)
        filtered_engine_products = self._filter_products_by_snapshot(
            filtered_engine_products,
            active_products,
            log_prefix="[TRIGGER]",
        )
        trace_context.record_final_result(
            products=filtered_engine_products,
            total_price=sum(p.total_price for p in filtered_engine_products),
            status=result.status.value,
            confidence=result.confidence,
        )
        # 8. SessionStore에 결과 저장
        products = [
            ProductResult(
                product_id=p.product_id,
                product_idx=self._get_product_idx(p.product_id, active_products),
                name=p.name,
                count=p.count,
                price=p.unit_price,
                confidence=p.confidence,
            )
            for p in filtered_engine_products
        ]

        # v4.6: 필터링 후 총 가격 재계산
        final_total_price = sum(p.price * p.count for p in products)
        trace_context.record_storage_result(
            products=products,
            total_price=final_total_price,
        )
        failure_reason = self._trigger_failure_reason(
            products=products,
            active_product_diagnostics=active_product_diagnostics,
        )

        vision_candidates_dicts = [vc.to_dict() for vc in vision_candidates]
        close_candidate_snapshot = build_trigger_candidate_snapshot(
            vision_candidates,
            active_products,
        )

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
            return_weight_hints = self._mixed_return_hints_from_analysis(
                delta_analysis,
                decision_delta=delta_weight,
            )
            self._record_mixed_return_segment_diagnostics(
                trace_context,
                delta_analysis=delta_analysis,
                return_weight_hints=return_weight_hints,
                decision_delta=delta_weight,
            )
            self._record_effective_count_guard_diagnostics(
                trace_context,
                products=filtered_engine_products,
                delta_weight=delta_weight,
                return_weight_hints=return_weight_hints,
            )
            trigger_result = TriggerResult(
                trigger_id="",
                session_id=session_id,
                timestamp=start_time,
                products=products,
                delta_weight=delta_weight,
                confidence=result.confidence,
                video_paths={
                    "top": str(input_data.top_video_path) if input_data.top_video_path else "",
                    "side": str(input_data.side_video_path) if input_data.side_video_path else "",
                },
                is_return=delta_weight > 0,
                processing_time_ms=elapsed_ms,
                failure_reason=failure_reason,
                return_weight_hints=return_weight_hints,
                vision_candidates=close_candidate_snapshot,
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
        logger.info("[TRIGGER] ========== 판단 결과 ==========")
        logger.info(f"[TRIGGER] status={result.status.value}, confidence={result.confidence:.3f}")
        for p in filtered_engine_products:
            logger.info(f"  - {p.name} x{p.count}: {p.total_price}원")
        logger.info(f"[TRIGGER] total_price={final_total_price}원, elapsed={elapsed_ms:.1f}ms")
        ops_logger.info(
            f"[OPS][RESULT] zone={input_data.zone} "
            f"status={result.status.value} "
            f"products={self._format_products_for_ops(products)} "
            f"total_price={final_total_price}"
        )

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

    def _get_product_idx(
        self,
        product_id: int,
        active_products: Optional[List] = None,
    ) -> Optional[str]:
        """YOLO class_id로 IF11 product_idx 조회."""
        snapshot_product = self._snapshot_product_info(active_products, product_id)
        if snapshot_product is not None and snapshot_product.product_idx:
            return snapshot_product.product_idx

        if self._active_product_store is not None:
            product_info = self._active_product_store.get_by_yolo_class_id(product_id)
            if product_info and product_info.product_idx:
                return product_info.product_idx
        return None

    def _analyze_weight_delta(
        self,
        loadcells: List[LoadcellReading],
        *,
        cabinet_type: Optional[str] = None,
    ) -> loadcell_stats.LoadcellDeltaAnalysis:
        analysis = loadcell_stats.analyze_weight_delta(
            loadcells,
            endpoint_fallback_enabled=loadcell_stats.endpoint_fallback_enabled_for_cabinet(
                cabinet_type
            ),
            prefer_mixed_sign_removal_delta=(
                loadcell_stats.prefer_mixed_sign_removal_delta_for_cabinet(
                    cabinet_type
                )
            ),
        )
        mixed_sign_guard = dict(analysis.mixed_sign_net_masking_guard or {})
        if mixed_sign_guard.get("accepted"):
            ops_logger.info(
                "[OPS][LOADCELL] mixed_sign_net_masking_guard "
                "cabinet_type=%s net_delta=%.1fg return_total=%.1fg "
                "removal_total=%.1fg selected_delta=%.1fg",
                (cabinet_type or config.machine.cabinet_type),
                float(mixed_sign_guard.get("net_delta", 0.0) or 0.0),
                float(mixed_sign_guard.get("return_total", 0.0) or 0.0),
                float(mixed_sign_guard.get("removal_total", 0.0) or 0.0),
                float(mixed_sign_guard.get("selected_decision_delta", 0.0) or 0.0),
            )
        logger.info(
            f"Weight delta calculated: {analysis.delta:.1f}g "
            f"(sample_count={analysis.sample_count}, parsed={analysis.parsed_sample_count}, "
            f"working={analysis.working_sample_count}, span_s={analysis.sample_span_seconds:.3f}, "
            f"window={analysis.window_size}, threshold={analysis.stability_threshold:.1f}, "
            f"start_avg={analysis.start_avg:.1f}, end_avg={analysis.end_avg:.1f}, "
            f"start_idx={analysis.start_stable_idx}, end_idx={analysis.end_stable_idx}, "
            f"fallback={analysis.used_simple_fallback}, reason={analysis.reason}, "
            f"endpoint_fallback={analysis.endpoint_fallback_applied}, "
            f"endpoint_reason={analysis.endpoint_fallback_reason})"
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
            raw_vote_count = max(
                int(getattr(vote, "raw_vote_count", 0) or 0),
                int(getattr(vote, "vote_count", 0) or 0),
            )
            ensemble = EnsembleResult(
                class_id=vote.class_id,
                class_name=vote.class_name,
                top_confidence=vote.top_max_confidence,
                side_confidence=vote.side_max_confidence,
                combined_confidence=vote.weighted_confidence,
                vote_count=2 if (vote.top_detected and vote.side_detected) else 1,
                source=getattr(vote, "source", "vision"),
                raw_vote_count=raw_vote_count,
                top_motion_passed=getattr(vote, "top_motion_passed", False),
                side_motion_passed=getattr(vote, "side_motion_passed", False),
                motion_gate_passed=getattr(vote, "motion_gate_passed", True),
                weight_gate_passed=getattr(vote, "weight_gate_passed", None),
                rescue_tolerance_g=getattr(vote, "rescue_tolerance_g", None),
                rescue_weight_residual_g=getattr(
                    vote,
                    "rescue_weight_residual_g",
                    None,
                ),
                instance_count_hint=getattr(vote, "instance_count_hint", 1),
                freezer_exit_path_votes=getattr(
                    vote,
                    "freezer_exit_path_votes",
                    0,
                ),
            )
            ensemble_results.append(ensemble)
        return ensemble_results
