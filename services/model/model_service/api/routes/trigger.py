"""Trigger API routes."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import model_service.core.loadcell_stats as loadcell_stats
from fastapi import APIRouter, Depends, HTTPException
from model_service.api.deps import (
    get_active_product_store_optional,
    get_decision_engine,
    get_door_session_store_optional,
    get_session_store,
    get_trigger_service_optional,
    get_video_processor,
)
from model_service.core.config import config
from model_service.core.exceptions import (
    FFmpegError,
    VideoCorruptedError,
    VideoProcessingError,
    YOLOGPUError,
    YOLOInferenceError,
    YOLOModelNotLoadedError,
)
from model_service.core.logging_config import get_ops_logger
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
from model_service.video.frame_trace import TriggerTraceContext
from pydantic import AliasChoices, BaseModel, Field

logger = logging.getLogger(__name__)
ops_logger = get_ops_logger()

router = APIRouter(tags=["trigger"])


class LoadcellData(BaseModel):
    timestamp: str = Field(..., description="Timestamp (ISO format)")
    raw_value: List[str] = Field(..., description="Raw loadcell values")
    filtered_value: List[str] = Field(..., description="Filtered loadcell values")
    filter_method: str = Field(default="none", description="Filter method")


class VideosPaths(BaseModel):
    top: Optional[str] = Field(None, description="Top camera AVI path")
    side: Optional[str] = Field(None, description="Side camera AVI path")


class TriggerTimingMetadataModel(BaseModel):
    capture_started_at: Optional[str] = None
    capture_ended_at: Optional[str] = None
    loadcell_started_at: Optional[str] = None
    loadcell_ended_at: Optional[str] = None
    trigger_started_at: Optional[str] = None
    trigger_end_reason: Optional[str] = None


class TriggerRequest(BaseModel):
    zone: int = Field(..., ge=0, description="Zone number")
    loadcells: List[LoadcellData] = Field(default_factory=list, description="Loadcell data")
    global_loadcells: List[LoadcellData] = Field(
        default_factory=list,
        validation_alias=AliasChoices("global_loadcells", "globalLoadcells"),
        description="Deprecated compatibility field; model uses zone loadcells",
    )
    videos: VideosPaths = Field(..., description="Recorded AVI paths")
    timing: Optional[TriggerTimingMetadataModel] = Field(
        default=None,
        description="Optional trigger timing metadata from the camera service",
    )


class TriggerResponse(BaseModel):
    success: bool
    session_id: str
    door_session_id: Optional[str] = None
    message: str
    status: str = "complete"
    waiting_for: Optional[str] = None


def _parse_loadcell_value(value: str) -> float:
    return loadcell_stats.parse_loadcell_value(value)


def _avg_loadcell_channels(values: list) -> float:
    return loadcell_stats.avg_loadcell_channels(values)


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
    loadcells: List[LoadcellData],
    window_size: int = 5,
    stability_threshold: float = 15.0,
) -> Tuple[float, float, bool]:
    return loadcell_stats.detect_stable_regions(
        loadcells,
        window_size=window_size,
        stability_threshold=stability_threshold,
    )


def _simple_delta_values(loadcells: List[LoadcellData]) -> Tuple[float, float, bool]:
    return loadcell_stats.simple_delta_values(loadcells)


def _calculate_weight_delta(loadcells: List[LoadcellData]) -> float:
    return loadcell_stats.calculate_weight_delta(loadcells)


def _analyze_weight_delta(
    loadcells: List[LoadcellData],
    *,
    cabinet_type: Optional[str] = None,
) -> loadcell_stats.LoadcellDeltaAnalysis:
    analysis = loadcell_stats.analyze_weight_delta(
        loadcells,
        endpoint_fallback_enabled=loadcell_stats.endpoint_fallback_enabled_for_cabinet(
            cabinet_type
        ),
        prefer_mixed_sign_removal_delta=(
            loadcell_stats.prefer_mixed_sign_removal_delta_for_cabinet(cabinet_type)
        ),
        stable_net_delta_only=(
            loadcell_stats.stable_net_delta_only_for_cabinet(cabinet_type)
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
    return analysis


def _loadcell_channel_count(loadcells: Sequence[Any]) -> int:
    max_count = 0
    for loadcell in loadcells:
        raw_value = getattr(loadcell, "raw_value", None)
        filtered_value = getattr(loadcell, "filtered_value", None)
        if isinstance(loadcell, dict):
            raw_value = loadcell.get("raw_value")
            filtered_value = loadcell.get("filtered_value")
        for values in (raw_value, filtered_value):
            if isinstance(values, (list, tuple)):
                max_count = max(max_count, len(values))
    return max_count


def _select_effective_loadcells(request: TriggerRequest) -> tuple[List[LoadcellData], dict]:
    cabinet_type = config.machine.cabinet_type
    requested_channel_count = _loadcell_channel_count(request.loadcells)
    global_channel_count = _loadcell_channel_count(request.global_loadcells)

    return request.loadcells, {
        "cabinet_type": cabinet_type,
        "loadcell_scope": "zone",
        "loadcell_source": "loadcells",
        "requested_zone": request.zone,
        "effective_channel_count": requested_channel_count,
        "requested_channel_count": requested_channel_count,
        "global_channel_count": global_channel_count,
        "loadcell_validation_reason": None,
    }


def _has_video_path(videos: VideosPaths) -> bool:
    return bool(videos.top or videos.side)


def _is_low_weight_delta(delta_weight: float) -> bool:
    return abs(delta_weight) <= config.trigger.min_weight_change_grams


def _is_reliable_low_weight_analysis(
    delta_analysis: loadcell_stats.LoadcellDeltaAnalysis,
) -> bool:
    return (
        delta_analysis.stable_region_valid
        and not delta_analysis.used_simple_fallback
        and delta_analysis.reason == "stable_regions"
    )


def _should_skip_low_weight(
    videos: VideosPaths,
    delta_analysis: loadcell_stats.LoadcellDeltaAnalysis,
    delta_weight: float | None = None,
) -> bool:
    selected_delta = delta_analysis.delta if delta_weight is None else delta_weight
    return _is_low_weight_delta(selected_delta)


def _should_force_vision_only(
    videos: VideosPaths,
    delta_analysis: loadcell_stats.LoadcellDeltaAnalysis,
) -> bool:
    return False


def _mixed_return_hints_from_analysis(
    delta_analysis: loadcell_stats.LoadcellDeltaAnalysis | None,
    *,
    decision_delta: float,
) -> List[dict[str, object]]:
    return loadcell_stats.mixed_return_hints_from_analysis(
        delta_analysis,
        decision_delta=decision_delta,
    )


def _record_mixed_return_segment_diagnostics(
    trace_context: TriggerTraceContext | None,
    *,
    delta_analysis: loadcell_stats.LoadcellDeltaAnalysis | None,
    return_weight_hints: List[dict[str, object]],
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


def _record_effective_count_guard_diagnostics(
    trace_context: TriggerTraceContext | None,
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


def _loadcell_trace_metadata(
    loadcells: List[LoadcellData],
    delta_analysis: loadcell_stats.LoadcellDeltaAnalysis,
    loadcell_metadata: Optional[dict] = None,
) -> dict:
    metadata = loadcell_stats.summarize_loadcell_payload(loadcells)
    if loadcell_metadata:
        metadata.update(loadcell_metadata)
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
            "baseline_stable_avg": round(float(delta_analysis.baseline_stable_avg), 1),
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
            "purchase_delta_candidates": list(delta_analysis.purchase_delta_candidates),
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
    return metadata


def _snapshot_product_info(active_products_snapshot: List[Any], product_id: int) -> Any | None:
    for product_info in active_products_snapshot:
        if getattr(product_info, "yolo_class_id", None) == product_id:
            return product_info
    return None


def _snapshot_allowed_class_ids(active_products_snapshot: List[Any]) -> set[int] | None:
    if not active_products_snapshot:
        return None
    allowed_ids: set[int] = set()
    for product_info in active_products_snapshot:
        class_id = getattr(product_info, "yolo_class_id", None)
        stock_qty = getattr(product_info, "stock_qty", 0)
        if class_id is not None and (stock_qty is None or stock_qty > 0):
            allowed_ids.add(class_id)
    return allowed_ids


def _effective_active_product_snapshot(
    active_product_store: ActiveProductStore | None,
) -> tuple[List[Any], Optional[List[int]], dict]:
    if active_product_store is None:
        return [], None, {
            "snapshot_source": "missing",
            "used_last_valid_snapshot": False,
        }

    if hasattr(active_product_store, "get_effective_snapshot"):
        snapshot = active_product_store.get_effective_snapshot()
        return (
            list(snapshot.products),
            (
                list(snapshot.allowed_class_ids)
                if snapshot.allowed_class_ids is not None
                else None
            ),
            snapshot.diagnostics(),
        )

    products = active_product_store.get_all_products()
    has_products = (
        active_product_store.has_products()
        if hasattr(active_product_store, "has_products")
        else bool(products)
    )
    allowed_class_ids = (
        active_product_store.get_allowed_class_ids()
        if has_products and hasattr(active_product_store, "get_allowed_class_ids")
        else None
    )
    return (
        products,
        allowed_class_ids,
        {
            "snapshot_source": "current" if products else "missing",
            "used_last_valid_snapshot": False,
        },
    )


def _active_product_diagnostics(
    active_products: List[Any],
    allowed_class_ids: Optional[List[int]],
    snapshot_metadata: dict,
) -> dict:
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

    fail_closed_reason = None
    if allowed_class_ids is None:
        fail_closed_reason = "missing_active_product_snapshot_fail_closed"
    elif len(allowed_class_ids) == 0:
        fail_closed_reason = "empty_allowlist_fail_closed"

    diagnostics = {
        "active_products_count": len(active_products),
        "allowed_class_ids_count": (
            len(allowed_class_ids) if allowed_class_ids is not None else 0
        ),
        "allowed_class_ids": list(allowed_class_ids or []),
        "stock_positive_products": stock_positive_count,
        "stock_positive_weight_products": stock_positive_weight_count,
        "zero_stock_products": zero_stock_count,
        "zero_weight_products": zero_weight_count,
        "inference_fail_closed_reason": fail_closed_reason,
    }
    diagnostics.update(snapshot_metadata)
    return diagnostics


def _vote_results_to_ensemble(vote_results: List[Any]) -> List[Any]:
    from model_service.engine import EnsembleResult

    ensemble_results = []
    for vote in vote_results:
        raw_vote_count = max(
            int(getattr(vote, "raw_vote_count", 0) or 0),
            int(getattr(vote, "vote_count", 0) or 0),
        )
        ensemble_results.append(
            EnsembleResult(
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
        )
    return ensemble_results


def _candidate_ops_confidence_fields(vote: Any) -> Tuple[float, float, float, float]:
    from model_service.video import VideoProcessor

    _, identity_confidence, identity_threshold, top_confidence, side_confidence = (
        VideoProcessor._freezer_identity_confidence_gate(vote)
    )
    return (
        float(identity_confidence),
        float(identity_threshold),
        float(top_confidence),
        float(side_confidence),
    )


def _record_raw_and_filter_handled_candidates(
    *,
    vote_results: List[Any],
    delta_weight: Optional[float],
    product_weights: Optional[dict[int, float]],
    trace_context: Optional[TriggerTraceContext],
    log_prefix: str,
    zone: Optional[int] = None,
    product_stocks: Optional[dict[int, int]] = None,
) -> List[Any]:
    from model_service.video import VideoProcessor

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
    _log_freezer_candidate_filter_ops(
        zone=zone,
        raw_count=raw_count,
        handled_count=len(filtered),
        delta_weight=delta_weight,
        trace_context=trace_context,
    )
    return filtered


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
    ops_logger.info(
        "[OPS][FREEZER-CANDIDATE-FILTER] zone=%s camera_layout=%s "
        "enabled=%s raw=%s handled=%s reason=%s top_k=%s "
        "freezer_min_votes=%s freezer_min_ratio=%.3f "
        "freezer_motion_min_px=%.1f freezer_exit_votes=%s",
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
    )
    if camera_layout != "dual_top_proxy":
        ops_logger.warning(
            "[OPS][CONFIG] cabinet_type=freezer camera_layout=%s "
            "expected=dual_top_proxy freezer_candidate_filter=disabled",
            camera_layout,
        )


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


def _freezer_prior_selected_product_idxs(
    door_session_store: DoorSessionStore | None,
    delta_weight: float,
) -> set[str]:
    if not bool(config.weight.freezer_prior_trigger_dedupe_enabled):
        return set()
    if str(config.machine.cabinet_type).strip().lower() != "freezer":
        return set()
    if delta_weight >= 0:
        return set()
    if door_session_store is None:
        return set()

    global_session = door_session_store.get_global_session()
    if global_session is None:
        return set()

    selected: set[str] = set()
    for session in global_session.zone_sessions.values():
        for trigger in getattr(session, "triggers", []) or []:
            try:
                if float(getattr(trigger, "delta_weight", 0.0) or 0.0) >= 0:
                    continue
            except (TypeError, ValueError):
                continue
            for product in getattr(trigger, "products", []) or []:
                try:
                    if int(getattr(product, "count", 0) or 0) <= 0:
                        continue
                except (TypeError, ValueError):
                    continue
                product_idx = getattr(product, "product_idx", None)
                if product_idx is not None and str(product_idx).strip():
                    selected.add(str(product_idx).strip())
                try:
                    selected.add(f"id:{int(getattr(product, 'product_id'))}")
                except (TypeError, ValueError):
                    pass
    return selected


def _record_no_charge_diagnostic(
    *,
    door_session_store: DoorSessionStore | None,
    request: TriggerRequest,
    session_id: str,
    reason: str,
    delta_weight: float,
    processing_stage: str,
    payload_diagnostics: Optional[dict],
    message: str,
) -> None:
    if door_session_store is None:
        return
    door_session_store.record_no_charge_diagnostic(
        zone=request.zone,
        session_id=session_id,
        reason=reason,
        delta_weight=delta_weight,
        processing_stage=processing_stage,
        payload_diagnostics=payload_diagnostics,
        video_paths={
            "top": str(request.videos.top) if request.videos.top else "",
            "side": str(request.videos.side) if request.videos.side else "",
        },
        message=message,
    )


def _validate_video_paths(videos: VideosPaths) -> None:
    if videos.top and not Path(videos.top).exists():
        raise HTTPException(status_code=400, detail=f"Top video file not found: {videos.top}")

    if videos.side and not Path(videos.side).exists():
        raise HTTPException(status_code=400, detail=f"Side video file not found: {videos.side}")

    if not videos.top and not videos.side:
        raise HTTPException(
            status_code=400,
            detail="At least one video path (top or side) is required",
        )


@router.post("/trigger", response_model=TriggerResponse)
async def trigger_judgment(
    request: TriggerRequest,
    video_processor: Any = Depends(get_video_processor),
    engine: Any = Depends(get_decision_engine),
    session_store: SessionStore = Depends(get_session_store),
    active_product_store: ActiveProductStore | None = Depends(get_active_product_store_optional),
    door_session_store: DoorSessionStore | None = Depends(get_door_session_store_optional),
    trigger_service: Any = Depends(get_trigger_service_optional),
):
    """Handle a completed camera recording."""
    start_time = time.time()
    session_id = generate_session_id(request.zone)
    # Capture the active-product snapshot once per request so the service path
    # and the fallback path evaluate the same live machine state.
    (
        active_products_snapshot,
        snapshot_allowed_class_ids,
        snapshot_metadata,
    ) = _effective_active_product_snapshot(active_product_store)
    product_weights: dict[int, float] = {}
    product_stocks: dict[int, int] = {}
    for product_info in active_products_snapshot:
        class_id = getattr(product_info, "yolo_class_id", None)
        product_weight = getattr(product_info, "product_weight", None)
        if class_id is None:
            continue
        try:
            class_id_int = int(class_id)
        except (TypeError, ValueError):
            continue
        if product_weight is not None:
            try:
                product_weights[class_id_int] = float(product_weight)
            except (TypeError, ValueError):
                pass
        stock_qty = getattr(product_info, "stock_qty", None)
        if stock_qty is not None:
            try:
                product_stocks[class_id_int] = int(stock_qty)
            except (TypeError, ValueError):
                pass
    effective_loadcells, loadcell_metadata = _select_effective_loadcells(request)

    logger.info("[TRIGGER] ========== inference start ==========")
    logger.info(f"[TRIGGER] zone={request.zone}, session_id={session_id}")
    logger.info(f"[TRIGGER] videos: top={request.videos.top}, side={request.videos.side}")
    logger.info(
        f"[TRIGGER] loadcells: {len(effective_loadcells)} "
        f"cabinet_type={loadcell_metadata['cabinet_type']} "
        f"scope={loadcell_metadata['loadcell_scope']} "
        f"source={loadcell_metadata['loadcell_source']} "
        f"channels={loadcell_metadata['effective_channel_count']}"
    )

    trace_context = None

    try:
        if trigger_service is not None:
            from model_service.service.trigger_service import (
                LoadcellReading,
                TriggerInput,
                TriggerTimingMetadata,
            )

            trigger_input = TriggerInput(
                zone=request.zone,
                loadcells=[
                    LoadcellReading(
                        timestamp=loadcell.timestamp,
                        raw_value=loadcell.raw_value,
                        filtered_value=loadcell.filtered_value,
                        filter_method=loadcell.filter_method,
                    )
                    for loadcell in effective_loadcells
                ],
                top_video_path=request.videos.top,
                side_video_path=request.videos.side,
                timing=(
                    TriggerTimingMetadata(**request.timing.model_dump(exclude_none=True))
                    if request.timing is not None
                    else None
                ),
                cabinet_type=loadcell_metadata["cabinet_type"],
                loadcell_scope=loadcell_metadata["loadcell_scope"],
                loadcell_source=loadcell_metadata["loadcell_source"],
                requested_zone=loadcell_metadata["requested_zone"],
                effective_channel_count=loadcell_metadata["effective_channel_count"],
                loadcell_validation_reason=loadcell_metadata[
                    "loadcell_validation_reason"
                ],
            )
            logger.info(
                f"[TRIGGER][path=service] active_products_snapshot={len(active_products_snapshot)}"
                f" snapshot_source={snapshot_metadata.get('snapshot_source', 'unknown')}"
            )
            if request.timing is not None:
                logger.info(
                    f"[TRIGGER][timing] capture_started_at={request.timing.capture_started_at}, "
                    f"capture_ended_at={request.timing.capture_ended_at}, "
                    f"trigger_end_reason={request.timing.trigger_end_reason}"
                )

            output = await trigger_service.enqueue_trigger(trigger_input)

            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"[TRIGGER] TriggerService complete: elapsed={elapsed_ms:.1f}ms")

            if not output.success:
                logger.error(
                    f"[TRIGGER ERROR] session_id={output.session_id}, "
                    f"error_code={output.error_code}, message={output.message}"
                )
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error_code": output.error_code or "TRIGGER_ERROR",
                        "message": output.message,
                        "session_id": output.session_id,
                    },
                )

            return TriggerResponse(
                success=output.success,
                session_id=output.session_id,
                door_session_id=output.door_session_id,
                message=output.message,
                status=output.status,
                waiting_for=output.waiting_for,
            )

        # The fallback route exists for compatibility. Keep it behaviorally
        # aligned with TriggerService, especially around active-product context.
        logger.warning(
            f"[TRIGGER][path=fallback] TriggerService not available, "
            f"active_products_snapshot={len(active_products_snapshot)}"
        )
        trace_context = TriggerTraceContext(
            session_id=session_id,
            zone=request.zone,
            top_path=request.videos.top,
            side_path=request.videos.side,
        )

        _validate_video_paths(request.videos)

        delta_analysis = _analyze_weight_delta(
            effective_loadcells,
            cabinet_type=loadcell_metadata["cabinet_type"],
        )
        delta_weight = delta_analysis.decision_delta
        payload_diagnostics = _loadcell_trace_metadata(
            effective_loadcells,
            delta_analysis,
            loadcell_metadata,
        )
        trace_context.record_loadcell_delta(
            delta_weight=delta_weight,
            **payload_diagnostics,
        )
        return_weight_hints = _mixed_return_hints_from_analysis(
            delta_analysis,
            decision_delta=delta_weight,
        )
        _record_mixed_return_segment_diagnostics(
            trace_context,
            delta_analysis=delta_analysis,
            return_weight_hints=return_weight_hints,
            decision_delta=delta_weight,
        )
        ops_logger.info(
            f"[OPS][TRIGGER] zone={request.zone} "
            f"delta_weight={delta_weight:.1f}g "
            f"payload_state={payload_diagnostics['payload_state']} "
            f"filtered_channels={payload_diagnostics['filtered_channel_count']} "
            f"filtered_valid={payload_diagnostics['filtered_parsed_channel_count']} "
            f"filtered_zero={payload_diagnostics['filtered_zero_channel_count']} "
            f"first_filtered_total={payload_diagnostics['first_filtered_total']} "
            f"last_filtered_total={payload_diagnostics['last_filtered_total']} "
            f"analysis_reason={delta_analysis.reason} "
            f"cabinet_type={payload_diagnostics['cabinet_type']} "
            f"camera_layout={config.vision.camera_layout} "
            f"loadcell_scope={payload_diagnostics['loadcell_scope']} "
            f"loadcell_source={payload_diagnostics['loadcell_source']} "
            f"effective_channels={payload_diagnostics['effective_channel_count']} "
            f"top_video={request.videos.top or 'none'} "
            f"side_video={request.videos.side or 'none'}"
        )
        logger.info(
            f"[TRIGGER][fallback][loadcell] sample_count={delta_analysis.sample_count}, "
            f"parsed={delta_analysis.parsed_sample_count}, "
            f"working={delta_analysis.working_sample_count}, "
            f"span_s={delta_analysis.sample_span_seconds:.3f}, "
            f"delta={delta_analysis.delta:.1f}, "
            f"fallback={delta_analysis.used_simple_fallback}, "
            f"reason={delta_analysis.reason}"
        )

        if _should_skip_low_weight(request.videos, delta_analysis, delta_weight):
            if config.trigger.low_weight_vision_fallback and _has_video_path(
                request.videos
            ):
                allowed_ids = (
                    set(snapshot_allowed_class_ids)
                    if snapshot_allowed_class_ids is not None
                    else _snapshot_allowed_class_ids(active_products_snapshot)
                )
                trace_context.record_active_product_diagnostics(
                    _active_product_diagnostics(
                        active_products_snapshot,
                        list(allowed_ids) if allowed_ids is not None else None,
                        snapshot_metadata,
                    )
                )
                session_store.save(
                    session_id,
                    SessionData(
                        session_id=session_id,
                        zone=request.zone,
                        products=[],
                        total_price=0,
                        delta_weight=delta_weight,
                        status="processing",
                        processing_stage="low_weight_video_diagnostic",
                        processing_stage_detail="Low weight video diagnostic only",
                        trigger_timing=(
                            request.timing.model_dump(exclude_none=True)
                            if request.timing
                            else None
                        ),
                    ),
                )
                processing_started = time.perf_counter()
                processing_result = await asyncio.to_thread(
                    video_processor.process_videos,
                    top_path=request.videos.top,
                    side_path=request.videos.side,
                    allowed_class_ids=(
                        list(allowed_ids) if allowed_ids is not None else None
                    ),
                    product_weights=product_weights,
                    trace_context=trace_context,
                    delta_weight=delta_weight,
                )
                elapsed_ms = (time.perf_counter() - processing_started) * 1000
                vote_results = list(
                    getattr(processing_result, "vote_results", []) or []
                )
                vote_results = _record_raw_and_filter_handled_candidates(
                    vote_results=vote_results,
                    delta_weight=delta_weight,
                    product_weights=product_weights,
                    product_stocks=product_stocks,
                    trace_context=trace_context,
                    log_prefix="TRIGGER-LOW-WEIGHT",
                    zone=request.zone,
                )
                stats = getattr(processing_result, "stats", None)
                if stats is not None:
                    trace_context.record_video_stats(stats)
                trace_context.record_active_product_snapshot(
                    active_products_snapshot,
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
                        "threshold_grams": float(
                            config.trigger.min_weight_change_grams
                        ),
                        "excluded_from_close_summary": True,
                        "engine_skipped": True,
                        "delta_weight": round(float(delta_weight), 1),
                        "video_diagnostic_candidate_count": len(vote_results),
                        "video_diagnostic_elapsed_ms": round(float(elapsed_ms), 1),
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

                top_frames = (
                    int(getattr(stats, "top_frames", 0) or 0) if stats else 0
                )
                side_frames = (
                    int(getattr(stats, "side_frames", 0) or 0) if stats else 0
                )
                processing_time_ms = (
                    float(
                        getattr(stats, "processing_time_ms", elapsed_ms)
                        or elapsed_ms
                    )
                    if stats
                    else elapsed_ms
                )
                session_store.save(
                    session_id,
                    SessionData(
                        session_id=session_id,
                        zone=request.zone,
                        products=[],
                        total_price=0,
                        delta_weight=delta_weight,
                        status="complete",
                        processing_stage="low_weight_video_diagnostic",
                        processing_stage_detail=(
                            "engine skipped; excluded from close summary"
                        ),
                        confidence=0.0,
                        top_frames=top_frames,
                        side_frames=side_frames,
                        processing_time_ms=processing_time_ms,
                        vision_candidates=[],
                        trigger_timing=(
                            request.timing.model_dump(exclude_none=True)
                            if request.timing
                            else None
                        ),
                    ),
                )
                diagnostic_reason = (
                    _loadcell_payload_issue_reason(payload_diagnostics)
                    or "low_weight_video_diagnostic"
                )
                _record_no_charge_diagnostic(
                    door_session_store=door_session_store,
                    request=request,
                    session_id=session_id,
                    reason=diagnostic_reason,
                    delta_weight=delta_weight,
                    processing_stage="low_weight_video_diagnostic",
                    payload_diagnostics=payload_diagnostics,
                    message="engine skipped; excluded from close summary",
                )
                trace_context.finalize(status="complete")
                return TriggerResponse(
                    success=True,
                    session_id=session_id,
                    door_session_id=None,
                    message=(
                        f"Low weight change ({abs(delta_weight):.1f}g), "
                        "video diagnostic only"
                    ),
                    status="complete",
                )

            session_data = SessionData(
                session_id=session_id,
                zone=request.zone,
                products=[],
                total_price=0,
                delta_weight=delta_weight,
                status="complete",
                processing_stage="skipped_low_weight",
                processing_stage_detail=(
                    f"Low weight change ({abs(delta_weight):.1f}g)"
                ),
                trigger_timing=request.timing.model_dump(exclude_none=True) if request.timing else None,
            )
            session_store.save(session_id, session_data)

            door_session = None
            if door_session_store is not None:
                logger.info(
                    "[TRIGGER] ignored low-weight trigger excluded from DoorSession "
                    f"summary: zone={request.zone}, session_id={session_id}, "
                    f"delta={delta_weight:.1f}g"
                )
                _record_no_charge_diagnostic(
                    door_session_store=door_session_store,
                    request=request,
                    session_id=session_id,
                    reason=(
                        _loadcell_payload_issue_reason(payload_diagnostics)
                        or "low_weight_ignored"
                    ),
                    delta_weight=delta_weight,
                    processing_stage="skipped_low_weight",
                    payload_diagnostics=payload_diagnostics,
                    message=f"Low weight change ({abs(delta_weight):.1f}g)",
                )

            trace_context.record_weight_diagnostics(
                {
                    "decision_branch": "low_weight_ignored",
                    "ignored_low_weight_delta": True,
                    "threshold_grams": float(config.trigger.min_weight_change_grams),
                    "excluded_from_close_summary": True,
                    "delta_weight": round(float(delta_weight), 1),
                }
            )
            trace_context.finalize(status="skipped")
            return TriggerResponse(
                success=True,
                session_id=session_id,
                door_session_id=door_session.door_session_id if door_session else None,
                message=f"Low weight change ({abs(delta_weight):.1f}g), skipped",
                status="skipped",
            )

        allowed_ids = (
            set(snapshot_allowed_class_ids)
            if snapshot_allowed_class_ids is not None
            else _snapshot_allowed_class_ids(active_products_snapshot)
        )
        trace_context.record_active_product_diagnostics(
            _active_product_diagnostics(
                active_products_snapshot,
                list(allowed_ids) if allowed_ids is not None else None,
                snapshot_metadata,
            )
        )
        initial_session = SessionData(
            session_id=session_id,
            zone=request.zone,
            status="processing",
            processing_stage="extracting_frames",
            processing_stage_detail="Preparing frame extraction",
        )
        session_store.save(session_id, initial_session)

        session_store.update_stage(
            session_id,
            processing_stage="extracting_frames",
            processing_stage_detail="Extracting frames from recorded video",
        )

        ops_logger.info(
            f"[OPS][FRAMES] zone={request.zone} "
            f"top_video={request.videos.top or 'none'} "
            f"side_video={request.videos.side or 'none'}"
        )
        processing_result = await asyncio.to_thread(
            video_processor.process_videos,
            top_path=request.videos.top,
            side_path=request.videos.side,
            allowed_class_ids=list(allowed_ids) if allowed_ids is not None else None,
            product_weights=product_weights,
            trace_context=trace_context,
            delta_weight=delta_weight,
        )

        from model_service.video.video_processor import VideoProcessor

        existing_class_ids = {vote.class_id for vote in processing_result.vote_results}
        threshold_rescue_diagnostics: dict = {}
        rescue_votes = VideoProcessor.build_weight_gated_rescue_votes(
            getattr(processing_result, "threshold_rescue_candidates", []),
            active_products_snapshot,
            delta_weight,
            diagnostics=threshold_rescue_diagnostics,
            existing_class_ids=existing_class_ids,
        )
        roi_rescue_diagnostics: dict = {}
        roi_rescue_votes = VideoProcessor.build_weight_gated_rescue_votes(
            getattr(processing_result, "roi_rescue_candidates", []),
            active_products_snapshot,
            delta_weight,
            diagnostics=roi_rescue_diagnostics,
            existing_class_ids=existing_class_ids,
        )
        vote_results = processing_result.vote_results
        all_rescue_votes = rescue_votes + roi_rescue_votes
        if all_rescue_votes:
            vote_results = VideoProcessor.merge_rescue_votes(
                vote_results,
                all_rescue_votes,
            )
        vote_results = _record_raw_and_filter_handled_candidates(
            vote_results=vote_results,
            delta_weight=delta_weight,
            product_weights=product_weights,
            product_stocks=product_stocks,
            trace_context=trace_context,
            log_prefix="TRIGGER",
            zone=request.zone,
        )
        stats = processing_result.stats
        trace_context.record_video_stats(stats)
        trace_context.record_active_product_snapshot(
            active_products_snapshot,
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

        logger.info("[TRIGGER] ========== video processing complete ==========")
        logger.info(
            f"[TRIGGER] total_frames={stats.top_frames + stats.side_frames}, "
            f"candidates={len(vote_results)}, processing_time_ms={stats.processing_time_ms:.1f}"
        )
        for index, vote in enumerate(vote_results[:config.vision.top_k], start=1):
            product_weight = product_weights.get(vote.class_id)
            weight_text = (
                f"{float(product_weight):.1f}g"
                if product_weight is not None
                else "unknown"
            )
            logger.info(
                f"[TRIGGER] top{index} {vote.class_name}: votes={vote.vote_count}, "
                f"weighted_conf={vote.weighted_confidence:.3f}, "
                f"top={vote.top_detected}, side={vote.side_detected}"
            )
            (
                identity_confidence,
                identity_threshold,
                top_confidence,
                side_confidence,
            ) = _candidate_ops_confidence_fields(vote)
            ops_logger.info(
                f"[OPS][CANDIDATES] zone={request.zone} rank={index} "
                f"name={vote.class_name} weight={weight_text} "
                f"confidence={identity_confidence:.3f} "
                f"weighted_confidence={vote.weighted_confidence:.3f} "
                f"top_confidence={top_confidence:.3f} "
                f"side_confidence={side_confidence:.3f} "
                f"identity_threshold={identity_threshold:.3f} "
                f"top={vote.top_detected} side={vote.side_detected} "
                f"source={getattr(vote, 'source', 'vision')} "
                f"count_hint={getattr(vote, 'instance_count_hint', 1)} "
                f"freezer_exit_votes={getattr(vote, 'freezer_exit_path_votes', 0)}"
            )
        if not vote_results:
            ops_logger.info(f"[OPS][CANDIDATES] zone={request.zone} none")

        session_store.update_stage(
            session_id,
            processing_stage="calculating_count",
            processing_stage_detail=f"Derived {len(vote_results)} candidates, judging counts",
        )

        delta_analysis = _analyze_weight_delta(
            effective_loadcells,
            cabinet_type=loadcell_metadata["cabinet_type"],
        )
        delta_weight = delta_analysis.decision_delta
        logger.info(f"[TRIGGER] delta_weight={delta_weight:.1f}g")
        logger.info(
            f"[TRIGGER][loadcell] sample_count={delta_analysis.sample_count}, "
            f"parsed={delta_analysis.parsed_sample_count}, "
            f"working={delta_analysis.working_sample_count}, "
            f"span_s={delta_analysis.sample_span_seconds:.3f}, "
            f"stable_window={delta_analysis.window_size}, "
            f"threshold={delta_analysis.stability_threshold:.1f}, "
            f"start_avg={delta_analysis.start_avg:.1f}, "
            f"end_avg={delta_analysis.end_avg:.1f}, "
            f"delta={delta_analysis.delta:.1f}, "
            f"start_idx={delta_analysis.start_stable_idx}, "
            f"end_idx={delta_analysis.end_stable_idx}, "
            f"fallback={delta_analysis.used_simple_fallback}, "
            f"reason={delta_analysis.reason}"
        )

        vision_candidates = _vote_results_to_ensemble(vote_results)
        vision_only = (
            _should_force_vision_only(request.videos, delta_analysis)
            or (delta_weight == 0.0 and len(effective_loadcells) == 0)
        )
        prior_selected_product_idxs = _freezer_prior_selected_product_idxs(
            door_session_store,
            delta_weight,
        )
        if prior_selected_product_idxs:
            logger.info(
                "[TRIGGER][FREEZER-DEDUPE] prior_selected_product_idxs=%s",
                sorted(prior_selected_product_idxs),
            )

        result = engine.judge(
            vision_candidates=vision_candidates,
            delta_weight=delta_weight,
            vision_only=vision_only,
            active_products=active_products_snapshot,
            trace_context=trace_context,
            prior_selected_product_idxs=prior_selected_product_idxs,
        )

        def get_product_idx(product_id: int) -> str | None:
            product_info = _snapshot_product_info(active_products_snapshot, product_id)
            if product_info and product_info.product_idx:
                return product_info.product_idx
            if active_product_store is None:
                return None
            product_info = active_product_store.get_by_yolo_class_id(product_id)
            if product_info and product_info.product_idx:
                return product_info.product_idx
            return None

        if result.is_success:
            result_products = result.products
        else:
            if result.products:
                status = getattr(result.status, "value", result.status)
                logger.warning(
                    f"[TRIGGER][fallback] dropping {len(result.products)} product(s) "
                    f"from storage because judgment_status={status}"
                )
            result_products = []

        if allowed_ids is not None:
            filtered_result_products = []
            for product in result_products:
                if product.product_id in allowed_ids:
                    filtered_result_products.append(product)
                else:
                    logger.warning(
                        "[TRIGGER][fallback] dropping product outside active snapshot: "
                        f"product_id={product.product_id}, name={product.name}"
                    )
            result_products = filtered_result_products
        _record_effective_count_guard_diagnostics(
            trace_context,
            products=result_products,
            delta_weight=delta_weight,
            return_weight_hints=return_weight_hints,
        )
        trace_context.record_final_result(
            products=result_products,
            total_price=sum(product.total_price for product in result_products),
            status=result.status.value,
            confidence=result.confidence,
        )

        products = [
            ProductResult(
                product_id=product.product_id,
                product_idx=get_product_idx(product.product_id),
                name=product.name,
                count=product.count,
                price=product.unit_price,
                confidence=product.confidence,
            )
            for product in result_products
        ]
        final_total_price = sum(product.price * product.count for product in products)
        trace_context.record_storage_result(
            products=products,
            total_price=final_total_price,
        )

        session_data = SessionData(
            session_id=session_id,
            zone=request.zone,
            products=products,
            total_price=final_total_price,
            delta_weight=delta_weight,
            status="complete",
            processing_stage="complete",
            processing_stage_detail=f"Judged {len(products)} products",
            confidence=result.confidence,
            top_frames=stats.top_frames,
            side_frames=stats.side_frames,
            processing_time_ms=stats.processing_time_ms,
            vision_candidates=[candidate.to_dict() for candidate in vision_candidates],
            trigger_timing=request.timing.model_dump(exclude_none=True) if request.timing else None,
        )
        session_store.save(session_id, session_data)

        door_session = None
        if door_session_store is not None:
            close_candidate_snapshot = build_trigger_candidate_snapshot(
                vision_candidates,
                active_products_snapshot,
            )
            elapsed_ms_for_trigger = (time.time() - start_time) * 1000
            trigger_result = TriggerResult(
                trigger_id="",
                session_id=session_id,
                timestamp=start_time,
                products=products,
                delta_weight=delta_weight,
                confidence=result.confidence,
                video_paths={
                    "top": str(request.videos.top) if request.videos.top else "",
                    "side": str(request.videos.side) if request.videos.side else "",
                },
                is_return=delta_weight > 0,
                processing_time_ms=elapsed_ms_for_trigger,
                timing_metadata=request.timing.model_dump(exclude_none=True) if request.timing else None,
                return_weight_hints=return_weight_hints,
                vision_candidates=close_candidate_snapshot,
                loadcell_diagnostics=(
                    loadcell_stats.close_trigger_loadcell_diagnostics(
                        delta_analysis
                    )
                ),
            )
            door_session = door_session_store.add_trigger_with_global(
                zone=request.zone,
                result=trigger_result,
            )
            logger.info(
                f"[TRIGGER] Door session: {door_session.door_session_id}, "
                f"triggers={door_session.trigger_count}, "
                f"aggregated_products={len(door_session.aggregated_products)}, "
                f"global_session_active={door_session_store.is_global_session_active()}"
            )
        product_text = ", ".join(
            f"{product.name}x{product.count}" for product in products
        ) or "none"
        ops_logger.info(
            f"[OPS][RESULT] zone={request.zone} "
            f"status={result.status.value} "
            f"products={product_text} "
            f"product_count={sum(product.count for product in products)} "
            f"total_price={final_total_price}"
        )

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"[TRIGGER] final_status={result.status.value}, "
            f"confidence={result.confidence:.3f}, "
            f"total_price={final_total_price}, elapsed_ms={elapsed_ms:.1f}"
        )
        trace_context.finalize(status="complete")

        return TriggerResponse(
            success=True,
            session_id=session_id,
            door_session_id=door_session.door_session_id if door_session else None,
            message="Inference complete",
            status="complete",
        )

    except HTTPException as exc:
        if trace_context is not None:
            trace_context.finalize(status="error", error=str(exc.detail))
        logger.error(
            f"[TRIGGER ERROR] session_id={session_id}, "
            f"status={exc.status_code}, detail={exc.detail}"
        )
        session_store.update_stage(
            session_id,
            processing_stage="error",
            processing_stage_detail=f"HTTP error: {exc.status_code}",
            status="error",
        )
        raise

    except FileNotFoundError as exc:
        if trace_context is not None:
            trace_context.finalize(status="error", error=str(exc))
        logger.error(f"[TRIGGER ERROR] session_id={session_id}, video file not found: {exc}")
        session_store.update_stage(
            session_id,
            processing_stage="error",
            processing_stage_detail=f"Video file not found: {exc}",
            status="error",
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "VIDEO_FILE_NOT_FOUND",
                "message": f"Video file not found: {exc}",
            },
        )

    except (VideoCorruptedError, FFmpegError) as exc:
        if trace_context is not None:
            trace_context.finalize(status="error", error=str(exc))
        logger.error(f"[TRIGGER ERROR] session_id={session_id}, video processing: {exc}", exc_info=True)
        session_store.update_stage(
            session_id,
            processing_stage="error",
            processing_stage_detail=f"Video processing error: {exc.error_code}",
            status="error",
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": exc.error_code,
                "message": str(exc),
                "video_path": getattr(exc, "video_path", None),
            },
        )

    except VideoProcessingError as exc:
        if trace_context is not None:
            trace_context.finalize(status="error", error=str(exc))
        logger.error(f"[TRIGGER ERROR] session_id={session_id}, video processing: {exc}", exc_info=True)
        session_store.update_stage(
            session_id,
            processing_stage="error",
            processing_stage_detail=f"Video processing error: {exc.error_code}",
            status="error",
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": exc.error_code,
                "message": str(exc),
            },
        )

    except YOLOModelNotLoadedError as exc:
        if trace_context is not None:
            trace_context.finalize(status="error", error=str(exc))
        logger.error(f"[TRIGGER ERROR] session_id={session_id}, YOLO model not loaded: {exc}")
        session_store.update_stage(
            session_id,
            processing_stage="error",
            processing_stage_detail="YOLO model not loaded",
            status="error",
        )
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": exc.error_code,
                "message": "YOLO model not loaded. Service is not ready.",
            },
        )

    except YOLOGPUError as exc:
        if trace_context is not None:
            trace_context.finalize(status="error", error=str(exc))
        logger.error(f"[TRIGGER ERROR] session_id={session_id}, YOLO GPU: {exc}", exc_info=True)
        session_store.update_stage(
            session_id,
            processing_stage="error",
            processing_stage_detail=f"YOLO GPU error: {exc.error_code}",
            status="error",
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": exc.error_code,
                "message": str(exc),
            },
        )

    except YOLOInferenceError as exc:
        if trace_context is not None:
            trace_context.finalize(status="error", error=str(exc))
        logger.error(f"[TRIGGER ERROR] session_id={session_id}, YOLO inference: {exc}", exc_info=True)
        session_store.update_stage(
            session_id,
            processing_stage="error",
            processing_stage_detail=f"YOLO inference error: {exc.error_code}",
            status="error",
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": exc.error_code,
                "message": str(exc),
            },
        )

    except Exception as exc:
        if trace_context is not None:
            trace_context.finalize(status="error", error=str(exc))
        logger.error(f"[TRIGGER ERROR] session_id={session_id}, unexpected: {exc}", exc_info=True)
        session_store.update_stage(
            session_id,
            processing_stage="error",
            processing_stage_detail=f"Internal error: {type(exc).__name__}",
            status="error",
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "INTERNAL_ERROR",
                "message": f"Internal server error: {type(exc).__name__}: {exc}",
            },
        )


@router.get("/trigger/stats")
async def get_trigger_stats(
    video_processor: Any = Depends(get_video_processor),
    session_store: SessionStore = Depends(get_session_store),
):
    """Return processor and session store status."""
    return {
        "video_processor": {
            "min_vote_ratio": video_processor.min_vote_ratio,
            "confidence_threshold": video_processor.confidence_threshold,
            "top_confidence_threshold": video_processor.top_confidence_threshold,
            "side_confidence_threshold": video_processor.side_confidence_threshold,
            "candidate_limit": config.vision.top_k,
            "yolo_loaded": video_processor.yolo.is_loaded,
        },
        "session_store": session_store.get_stats(),
        "timestamp": time.time(),
    }
