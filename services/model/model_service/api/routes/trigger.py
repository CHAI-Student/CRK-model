from __future__ import annotations

"""Trigger API routes."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import model_service.core.loadcell_stats as loadcell_stats
from model_service.api.deps import (
    get_active_product_store_optional,
    get_decision_engine,
    get_door_session_store_optional,
    get_session_store,
    get_trigger_service_optional,
    get_video_processor,
)
from model_service.core.exceptions import (
    FFmpegError,
    VideoCorruptedError,
    VideoProcessingError,
    YOLOGPUError,
    YOLOInferenceError,
    YOLOModelNotLoadedError,
)
from model_service.session import (
    DoorSessionStore,
    ProductResult,
    SessionData,
    SessionStore,
    TriggerResult,
)
from model_service.session.active_product_store import ActiveProductStore
from model_service.session.session_store import generate_session_id
from model_service.video.frame_trace import TriggerTraceContext

logger = logging.getLogger(__name__)

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


def _analyze_weight_delta(loadcells: List[LoadcellData]) -> loadcell_stats.LoadcellDeltaAnalysis:
    return loadcell_stats.analyze_weight_delta(loadcells)


def _vote_results_to_ensemble(vote_results: List[Any]) -> List[Any]:
    from model_service.engine import EnsembleResult

    ensemble_results = []
    for vote in vote_results:
        ensemble_results.append(
            EnsembleResult(
                class_id=vote.class_id,
                class_name=vote.class_name,
                top_confidence=vote.top_max_confidence,
                side_confidence=vote.side_max_confidence,
                combined_confidence=vote.weighted_confidence,
                vote_count=2 if (vote.top_detected and vote.side_detected) else 1,
            )
        )
    return ensemble_results


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
    active_products_snapshot = (
        active_product_store.get_all_products()
        if active_product_store is not None
        else []
    )

    logger.info("[TRIGGER] ========== inference start ==========")
    logger.info(f"[TRIGGER] zone={request.zone}, session_id={session_id}")
    logger.info(f"[TRIGGER] videos: top={request.videos.top}, side={request.videos.side}")
    logger.info(f"[TRIGGER] loadcells: {len(request.loadcells)}")

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
                    for loadcell in request.loadcells
                ],
                top_video_path=request.videos.top,
                side_video_path=request.videos.side,
                timing=(
                    TriggerTimingMetadata(**request.timing.model_dump(exclude_none=True))
                    if request.timing is not None
                    else None
                ),
            )
            logger.info(
                f"[TRIGGER][path=service] active_products_snapshot={len(active_products_snapshot)}"
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

        processing_result = await asyncio.to_thread(
            video_processor.process_videos,
            top_path=request.videos.top,
            side_path=request.videos.side,
            trace_context=trace_context,
        )

        vote_results = processing_result.vote_results
        stats = processing_result.stats

        logger.info("[TRIGGER] ========== video processing complete ==========")
        logger.info(
            f"[TRIGGER] total_frames={stats.top_frames + stats.side_frames}, "
            f"candidates={len(vote_results)}, processing_time_ms={stats.processing_time_ms:.1f}"
        )
        for index, vote in enumerate(vote_results[:5], start=1):
            logger.info(
                f"[TRIGGER] top{index} {vote.class_name}: votes={vote.vote_count}, "
                f"weighted_conf={vote.weighted_confidence:.3f}, "
                f"top={vote.top_detected}, side={vote.side_detected}"
            )

        session_store.update_stage(
            session_id,
            processing_stage="calculating_count",
            processing_stage_detail=f"Derived {len(vote_results)} candidates, judging counts",
        )

        delta_analysis = _analyze_weight_delta(request.loadcells)
        delta_weight = delta_analysis.delta
        logger.info(f"[TRIGGER] delta_weight={delta_weight:.1f}g")
        logger.info(
            f"[TRIGGER][loadcell] samples={delta_analysis.sample_count}, "
            f"span_s={delta_analysis.sample_span_seconds:.3f}, "
            f"stable_window={delta_analysis.window_size}, "
            f"threshold={delta_analysis.stability_threshold:.1f}, "
            f"start_idx={delta_analysis.start_stable_idx}, "
            f"end_idx={delta_analysis.end_stable_idx}, "
            f"fallback={delta_analysis.used_simple_fallback}, "
            f"reason={delta_analysis.reason}"
        )

        vision_candidates = _vote_results_to_ensemble(vote_results)
        vision_only = delta_weight == 0.0 and len(request.loadcells) == 0

        result = engine.judge(
            vision_candidates=vision_candidates,
            delta_weight=delta_weight,
            vision_only=vision_only,
            active_products=active_products_snapshot,
        )

        def get_product_idx(product_id: int) -> str | None:
            if active_product_store is None:
                return None
            product_info = active_product_store.get_by_yolo_class_id(product_id)
            if product_info and product_info.product_idx:
                return product_info.product_idx
            return None

        products = [
            ProductResult(
                product_id=product.product_id,
                product_idx=get_product_idx(product.product_id),
                name=product.name,
                count=product.count,
                price=product.unit_price,
                confidence=product.confidence,
            )
            for product in result.products
        ]

        session_data = SessionData(
            session_id=session_id,
            zone=request.zone,
            products=products,
            total_price=result.total_price,
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
            elapsed_ms_for_trigger = (time.time() - start_time) * 1000
            trigger_result = TriggerResult(
                trigger_id="",
                session_id=session_id,
                timestamp=time.time(),
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

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"[TRIGGER] final_status={result.status.value}, "
            f"confidence={result.confidence:.3f}, "
            f"total_price={result.total_price}, elapsed_ms={elapsed_ms:.1f}"
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
            "yolo_loaded": video_processor.yolo.is_loaded,
        },
        "session_store": session_store.get_stats(),
        "timestamp": time.time(),
    }
