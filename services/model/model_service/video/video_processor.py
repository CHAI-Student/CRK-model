"""
Video Processor for AVI-based YOLO Inference.

Processes entire AVI videos frame-by-frame with YOLO inference
and aggregates results using voting-based ensemble.

Memory-efficient design for Jetson Orin Nano:
- FFmpeg subprocess with NVDEC hardware decoding
- Streaming frame extraction (one frame at a time)
- Immediate memory release after inference
- Only vote counts are accumulated (not images)

v5.3 추가:
- Async streaming video processing (process_videos_async)
- Top/Side 프레임 인터리빙으로 I/O 병렬화
- 단일 YOLO 인스턴스로 순차 추론 (GPU 메모리 제약)

v4.6 추가:
- HandPathTracker: 손 경로 추적 기반 상품 필터링
- product_weights 파라미터 추가 (로그용)

v4.1 추가:
- Bounding box 중심점 이동 추적 (Motion Tracking)
- 이동이 감지된 객체만 후보에 포함

Usage:
    processor = VideoProcessor(yolo=yolo_wrapper)
    results = processor.process_videos(
        top_path="/path/to/top.avi",
        side_path="/path/to/side.avi"
    )

    # Async streaming (v5.3)
    results = await processor.process_videos_async(
        top_path="/path/to/top.avi",
        side_path="/path/to/side.avi"
    )
"""

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

from model_service.core.config import config
from model_service.core.exceptions import ModelServiceError, VideoProcessingError
from model_service.vision import YOLODetection, YOLOWrapper
from model_service.vision.hand_path_tracker import HandPathTracker

from .frame_extractor import create_frame_extractor
from .frame_trace import TriggerTraceContext
from .voting_ensemble import VoteResult, VotingEnsemble

logger = logging.getLogger(__name__)


def _async_zero_frame_failure_reason(extractor: object) -> Optional[str]:
    diagnostics = getattr(extractor, "last_diagnostics", None)
    if diagnostics is None:
        return None

    expected_frames = int(getattr(diagnostics, "expected_frames", 0) or 0)
    decoded_frames = int(getattr(diagnostics, "decoded_frames", 0) or 0)
    if expected_frames <= 0 or decoded_frames > 0:
        return None

    method = getattr(diagnostics, "method", None)
    final_branch = getattr(diagnostics, "final_branch", None)
    stderr_tail = getattr(diagnostics, "stderr_tail", None)
    reason = (
        f"ffmpeg decoded zero frames after async retry "
        f"(expected_frames={expected_frames}, method={method}, final_branch={final_branch})"
    )
    if stderr_tail:
        reason += f", stderr_tail={stderr_tail!r}"
    return reason


def _raise_async_task_error(task_name: str, exc: BaseException) -> None:
    error_msg = f"Task error in {task_name}: {type(exc).__name__}: {exc}"
    logger.error(
        "[VIDEO-ASYNC] Critical failure during streaming. "
        f"Propagation triggered: {error_msg}"
    )
    if isinstance(exc, ModelServiceError):
        raise exc
    raise VideoProcessingError(f"Async video streaming failed: {error_msg}")


@dataclass
class BboxTracker:
    """
    Bounding box 중심점 이동 추적.

    각 class_id별로 첫 번째/마지막 bbox 중심점과 최대 이동 거리를 추적.

    Attributes:
        first_center: 첫 번째 감지 시 중심점 (x, y)
        last_center: 마지막 감지 시 중심점 (x, y)
        max_distance: 관찰된 최대 이동 거리 (픽셀)
        detection_count: 총 감지 횟수
        frame_indices: 감지된 프레임 인덱스 목록
        dynamic_threshold: bbox 크기 기반 동적 임계값 (픽셀)
    """
    first_center: Optional[Tuple[float, float]] = None
    last_center: Optional[Tuple[float, float]] = None
    max_distance: float = 0.0
    detection_count: int = 0
    frame_indices: List[int] = field(default_factory=list)
    min_x: Optional[float] = None
    max_x: Optional[float] = None
    min_y: Optional[float] = None
    max_y: Optional[float] = None
    dynamic_threshold: float = 0.0  # bbox 크기 기반 동적 임계값

    def update(self, center: Tuple[float, float], frame_idx: int) -> None:
        """bbox 중심점 업데이트."""
        if self.first_center is None:
            self.first_center = center

        # 이전 중심점과의 거리 계산
        if self.last_center is not None:
            distance = math.sqrt(
                (center[0] - self.last_center[0]) ** 2 +
                (center[1] - self.last_center[1]) ** 2
            )
            self.max_distance = max(self.max_distance, distance)

        self.last_center = center
        self.detection_count += 1
        self.frame_indices.append(frame_idx)
        center_x = float(center[0])
        center_y = float(center[1])
        self.min_x = center_x if self.min_x is None else min(self.min_x, center_x)
        self.max_x = center_x if self.max_x is None else max(self.max_x, center_x)
        self.min_y = center_y if self.min_y is None else min(self.min_y, center_y)
        self.max_y = center_y if self.max_y is None else max(self.max_y, center_y)

    @property
    def total_displacement(self) -> float:
        """첫 번째와 마지막 위치 간 총 이동 거리."""
        if self.first_center is None or self.last_center is None:
            return 0.0
        return math.sqrt(
            (self.last_center[0] - self.first_center[0]) ** 2 +
            (self.last_center[1] - self.first_center[1]) ** 2
        )

    @property
    def center_span_x(self) -> float:
        if self.min_x is None or self.max_x is None:
            return 0.0
        return max(0.0, self.max_x - self.min_x)

    @property
    def center_span_y(self) -> float:
        if self.min_y is None or self.max_y is None:
            return 0.0
        return max(0.0, self.max_y - self.min_y)

    def has_motion(self, min_displacement: float = 30.0) -> bool:
        """
        이동이 있었는지 여부.

        Args:
            min_displacement: 최소 이동 거리 임계값 (픽셀)

        Returns:
            이동이 감지되었으면 True
        """
        # 동적 임계값이 설정되어 있으면 사용, 아니면 기본값 사용
        threshold = self.dynamic_threshold if self.dynamic_threshold > 0 else min_displacement
        return self.total_displacement >= threshold or self.max_distance >= threshold


@dataclass
class VideoProcessingStats:
    """
    Video processing statistics.

    Attributes:
        top_frames: Number of frames processed from top camera
        side_frames: Number of frames processed from side camera
        top_detections: Total detections from top camera
        side_detections: Total detections from side camera
        processing_time_ms: Total processing time in milliseconds
        motion_filtered_classes: Number of classes filtered out due to no motion
        hand_path_filtered_classes: Number of classes filtered out due to hand path (v4.6)
    """
    top_frames: int = 0
    side_frames: int = 0
    top_original_frames: int = 0
    side_original_frames: int = 0
    frame_stride: int = 2
    top_raw_detections: int = 0
    side_raw_detections: int = 0
    top_threshold_filtered: int = 0
    side_threshold_filtered: int = 0
    top_detections: int = 0
    side_detections: int = 0
    processing_time_ms: float = 0.0
    yolo_inference_count: int = 0
    yolo_total_time_ms: float = 0.0
    yolo_avg_time_ms: float = 0.0
    roi_filtered_detections: int = 0
    side_roi_soft_passed_detections: int = 0
    side_roi_soft_filtered_detections: int = 0
    motion_filtered_classes: int = 0
    hand_path_filtered_classes: int = 0

    @property
    def original_frames(self) -> int:
        top_original = self.top_original_frames or self.top_frames
        side_original = self.side_original_frames or self.side_frames
        return top_original + side_original

    @property
    def processed_frames(self) -> int:
        return self.top_frames + self.side_frames

    @property
    def skipped_frames(self) -> int:
        return max(0, self.original_frames - self.processed_frames)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "top_frames": self.top_frames,
            "side_frames": self.side_frames,
            "top_original_frames": self.top_original_frames or self.top_frames,
            "side_original_frames": self.side_original_frames or self.side_frames,
            "original_frames": self.original_frames,
            "processed_frames": self.processed_frames,
            "skipped_frames": self.skipped_frames,
            "frame_stride": self.frame_stride,
            "top_raw_detections": self.top_raw_detections,
            "side_raw_detections": self.side_raw_detections,
            "top_threshold_filtered": self.top_threshold_filtered,
            "side_threshold_filtered": self.side_threshold_filtered,
            "top_detections": self.top_detections,
            "side_detections": self.side_detections,
            "total_frames": self.top_frames + self.side_frames,
            "total_detections": self.top_detections + self.side_detections,
            "processing_time_ms": round(self.processing_time_ms, 1),
            "yolo_inference_count": self.yolo_inference_count,
            "yolo_total_time_ms": round(self.yolo_total_time_ms, 1),
            "yolo_avg_time_ms": round(self.yolo_avg_time_ms, 1),
            "roi_filtered_detections": self.roi_filtered_detections,
            "side_roi_soft_passed_detections": self.side_roi_soft_passed_detections,
            "side_roi_soft_filtered_detections": self.side_roi_soft_filtered_detections,
            "motion_filtered_classes": self.motion_filtered_classes,
            "hand_path_filtered_classes": self.hand_path_filtered_classes,
        }


@dataclass
class ThresholdRescueCandidate:
    """Low-confidence class that can be considered only through weight gating."""

    class_id: int
    class_name: str
    vote_count: int
    max_confidence: float
    avg_confidence: float
    top_detected: bool = False
    side_detected: bool = False
    top_vote_count: int = 0
    side_vote_count: int = 0
    top_max_confidence: float = 0.0
    side_max_confidence: float = 0.0
    top_motion_passed: bool = False
    side_motion_passed: bool = False
    top_total_displacement: float = 0.0
    side_total_displacement: float = 0.0
    top_max_distance: float = 0.0
    side_max_distance: float = 0.0
    roi_x_min: Optional[float] = None
    roi_x_max: Optional[float] = None
    roi_x_avg: Optional[float] = None
    roi_x_limit: Optional[float] = None
    sample_frames: List[dict] = field(default_factory=list)
    source: str = "threshold_rescue"
    motion_gate_passed: bool = True
    motion_gate_reason: Optional[str] = None
    weight_gate_passed: Optional[bool] = None
    rescue_weight_residual_g: Optional[float] = None
    rescue_tolerance_g: Optional[float] = None
    roi_conflict: bool = False
    roi_conflict_reason: Optional[str] = None
    roi_conflict_side_vote_count: int = 0
    roi_conflict_side_max_confidence: float = 0.0
    roi_conflict_side_roi_x_avg: Optional[float] = None
    roi_conflict_side_roi_x_limit: Optional[float] = None

    def to_dict(self) -> dict:
        payload = {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "vote_count": self.vote_count,
            "raw_vote_count": self.vote_count,
            "max_confidence": round(self.max_confidence, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "confidence": round(self.max_confidence, 4),
            "top_detected": self.top_detected,
            "side_detected": self.side_detected,
            "top_vote_count": self.top_vote_count,
            "side_vote_count": self.side_vote_count,
            "top_max_confidence": round(self.top_max_confidence, 4),
            "side_max_confidence": round(self.side_max_confidence, 4),
            "top_motion_passed": self.top_motion_passed,
            "side_motion_passed": self.side_motion_passed,
            "top_total_displacement": round(self.top_total_displacement, 1),
            "side_total_displacement": round(self.side_total_displacement, 1),
            "top_max_distance": round(self.top_max_distance, 1),
            "side_max_distance": round(self.side_max_distance, 1),
            "source": self.source,
            "motion_gate_passed": self.motion_gate_passed,
            "motion_gate_reason": self.motion_gate_reason,
            "weight_gate_passed": self.weight_gate_passed,
            "rescue_weight_residual_g": (
                round(float(self.rescue_weight_residual_g), 1)
                if self.rescue_weight_residual_g is not None
                else None
            ),
            "rescue_tolerance_g": self.rescue_tolerance_g,
            "roi_conflict": self.roi_conflict,
            "roi_conflict_reason": self.roi_conflict_reason,
            "roi_conflict_side_vote_count": self.roi_conflict_side_vote_count,
            "roi_conflict_side_max_confidence": round(
                float(self.roi_conflict_side_max_confidence),
                4,
            ),
            "roi_conflict_side_roi_x_avg": (
                round(float(self.roi_conflict_side_roi_x_avg), 1)
                if self.roi_conflict_side_roi_x_avg is not None
                else None
            ),
            "roi_conflict_side_roi_x_limit": (
                round(float(self.roi_conflict_side_roi_x_limit), 1)
                if self.roi_conflict_side_roi_x_limit is not None
                else None
            ),
        }
        if self.roi_x_min is not None:
            payload.update(
                {
                    "roi_x_min": round(float(self.roi_x_min), 1),
                    "roi_x_max": round(float(self.roi_x_max or 0.0), 1),
                    "roi_x_avg": round(float(self.roi_x_avg or 0.0), 1),
                    "roi_x_limit": round(float(self.roi_x_limit or 0.0), 1),
                    "sample_frames": list(self.sample_frames),
                }
            )
        return payload


@dataclass
class _LowConfidenceClassStats:
    class_id: int
    class_name: str = ""
    top_vote_count: int = 0
    side_vote_count: int = 0
    top_conf_sum: float = 0.0
    side_conf_sum: float = 0.0
    top_max_confidence: float = 0.0
    side_max_confidence: float = 0.0
    top_tracker: BboxTracker = field(default_factory=BboxTracker)
    side_tracker: BboxTracker = field(default_factory=BboxTracker)

    def add(self, camera: str, confidence: float, class_name: str, center: Tuple[float, float], frame_idx: int, bbox_size: float) -> None:
        if class_name and not self.class_name:
            self.class_name = class_name
        if camera == "side":
            self.side_vote_count += 1
            self.side_conf_sum += confidence
            self.side_max_confidence = max(self.side_max_confidence, confidence)
            tracker = self.side_tracker
        else:
            self.top_vote_count += 1
            self.top_conf_sum += confidence
            self.top_max_confidence = max(self.top_max_confidence, confidence)
            tracker = self.top_tracker

        tracker.update(center, frame_idx)
        tracker.dynamic_threshold = max(
            tracker.dynamic_threshold,
            max(float(config.vision.motion_min_displacement_px), bbox_size * 0.10),
        )

    @property
    def vote_count(self) -> int:
        return self.top_vote_count + self.side_vote_count

    @property
    def confidence_sum(self) -> float:
        return self.top_conf_sum + self.side_conf_sum

    @property
    def max_confidence(self) -> float:
        return max(self.top_max_confidence, self.side_max_confidence)

    def to_rescue_candidate(
        self,
        min_motion_displacement: float,
        *,
        allow_no_motion: bool = False,
        no_motion_min_votes: int = 0,
    ) -> ThresholdRescueCandidate | None:
        if self.vote_count <= 0:
            return None
        top_motion = (
            self.top_vote_count > 0
            and self.top_tracker.has_motion(min_motion_displacement)
        )
        side_motion = (
            self.side_vote_count > 0
            and self.side_tracker.has_motion(min_motion_displacement)
        )
        motion_passed = top_motion or side_motion
        motion_gate_reason = None
        if config.vision.threshold_rescue_require_motion and not motion_passed:
            if not allow_no_motion or self.vote_count < max(0, no_motion_min_votes):
                return None
            motion_gate_reason = "weight_gated_no_motion_candidate"

        return ThresholdRescueCandidate(
            class_id=self.class_id,
            class_name=self.class_name,
            vote_count=self.vote_count,
            max_confidence=self.max_confidence,
            avg_confidence=self.confidence_sum / self.vote_count,
            top_detected=self.top_vote_count > 0,
            side_detected=self.side_vote_count > 0,
            top_vote_count=self.top_vote_count,
            side_vote_count=self.side_vote_count,
            top_max_confidence=self.top_max_confidence,
            side_max_confidence=self.side_max_confidence,
            top_motion_passed=top_motion,
            side_motion_passed=side_motion,
            top_total_displacement=self.top_tracker.total_displacement,
            side_total_displacement=self.side_tracker.total_displacement,
            top_max_distance=self.top_tracker.max_distance,
            side_max_distance=self.side_tracker.max_distance,
            motion_gate_passed=motion_passed,
            motion_gate_reason=motion_gate_reason,
        )


@dataclass
class _RoiFilteredClassStats:
    class_id: int
    class_name: str = ""
    vote_count: int = 0
    conf_sum: float = 0.0
    max_confidence: float = 0.0
    side_tracker: BboxTracker = field(default_factory=BboxTracker)
    center_x_sum: float = 0.0
    center_x_min: Optional[float] = None
    center_x_max: Optional[float] = None
    roi_x_limit: float = 0.0
    sample_frames: List[dict] = field(default_factory=list)

    def add(
        self,
        *,
        confidence: float,
        class_name: str,
        center: Tuple[float, float],
        frame_idx: int,
        bbox_size: float,
        roi_x_limit: float,
    ) -> None:
        if class_name and not self.class_name:
            self.class_name = class_name
        center_x = float(center[0])
        center_y = float(center[1])
        self.vote_count += 1
        self.conf_sum += confidence
        self.max_confidence = max(self.max_confidence, confidence)
        self.center_x_sum += center_x
        self.center_x_min = center_x if self.center_x_min is None else min(self.center_x_min, center_x)
        self.center_x_max = center_x if self.center_x_max is None else max(self.center_x_max, center_x)
        self.roi_x_limit = float(roi_x_limit)
        self.side_tracker.update(center, frame_idx)
        self.side_tracker.dynamic_threshold = max(
            self.side_tracker.dynamic_threshold,
            max(float(config.vision.motion_min_displacement_px), bbox_size * 0.10),
        )
        if len(self.sample_frames) < 5:
            self.sample_frames.append(
                {
                    "frame_index": int(frame_idx),
                    "confidence": round(float(confidence), 4),
                    "center_x": round(center_x, 1),
                    "center_y": round(center_y, 1),
                }
            )

    @property
    def avg_confidence(self) -> float:
        return self.conf_sum / self.vote_count if self.vote_count > 0 else 0.0

    @property
    def avg_center_x(self) -> float:
        return self.center_x_sum / self.vote_count if self.vote_count > 0 else 0.0

    def to_rescue_candidate(self, min_motion_displacement: float) -> ThresholdRescueCandidate | None:
        if self.vote_count <= 0:
            return None
        side_motion = self.side_tracker.has_motion(min_motion_displacement)
        return ThresholdRescueCandidate(
            class_id=self.class_id,
            class_name=self.class_name,
            vote_count=self.vote_count,
            max_confidence=self.max_confidence,
            avg_confidence=self.avg_confidence,
            side_detected=True,
            side_vote_count=self.vote_count,
            side_max_confidence=self.max_confidence,
            side_motion_passed=side_motion,
            side_total_displacement=self.side_tracker.total_displacement,
            side_max_distance=self.side_tracker.max_distance,
            roi_x_min=self.center_x_min,
            roi_x_max=self.center_x_max,
            roi_x_avg=self.avg_center_x,
            roi_x_limit=self.roi_x_limit,
            sample_frames=self.sample_frames,
            source="roi_rescue",
            motion_gate_passed=side_motion,
        )


@dataclass
class VideoProcessingResult:
    """
    Video processing result.

    Attributes:
        vote_results: Combined voting results from both cameras
        top_ensemble: Top camera voting ensemble
        side_ensemble: Side camera voting ensemble
        stats: Processing statistics
    """
    vote_results: List[VoteResult]
    top_ensemble: VotingEnsemble
    side_ensemble: VotingEnsemble
    stats: VideoProcessingStats
    threshold_rescue_candidates: List[ThresholdRescueCandidate] = field(default_factory=list)
    roi_rescue_candidates: List[ThresholdRescueCandidate] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "vote_results": [r.to_dict() for r in self.vote_results],
            "threshold_rescue_candidates": [
                candidate.to_dict() for candidate in self.threshold_rescue_candidates
            ],
            "roi_rescue_candidates": [
                candidate.to_dict() for candidate in self.roi_rescue_candidates
            ],
            "stats": self.stats.to_dict(),
        }


class VideoProcessor:
    """
    AVI video processor with YOLO inference and voting ensemble.

    Processes videos frame-by-frame to minimize memory usage,
    suitable for Jetson Orin Nano deployment.

    Uses FFmpeg with NVDEC hardware acceleration when available.
    """
    _field_tuning_warning_emitted = False

    def __init__(
        self,
        yolo: YOLOWrapper,
        min_vote_ratio: Optional[float] = None,
        confidence_threshold: Optional[float] = None,
        top_confidence_threshold: Optional[float] = None,
        side_confidence_threshold: Optional[float] = None,
        use_hwaccel: bool = True,
        motion_filter_enabled: bool = True,
        min_motion_displacement: Optional[float] = None,
        side_roi_x_max: Optional[float] = None,
        top_roi_enabled: Optional[bool] = None,
        top_roi_y_split: Optional[float] = None,
        hand_path_filter_enabled: bool = True,
        min_vote_count: Optional[int] = None,
    ):
        """
        Initialize video processor.

        Args:
            yolo: YOLOWrapper instance for inference
            min_vote_ratio: Minimum vote ratio to include in results (default: 5%)
            confidence_threshold: Backward-compatible shared confidence threshold
            top_confidence_threshold: Minimum confidence for top camera detections
            side_confidence_threshold: Minimum confidence for side camera detections
            use_hwaccel: Use hardware acceleration for video decoding (default: True)
            motion_filter_enabled: Enable motion-based filtering (default: True)
            min_motion_displacement: Minimum bbox center displacement to consider as motion (default: 10 pixels)
            side_roi_x_max: Side camera ROI max X coordinate (default: 400px)
            top_roi_enabled: Enable top camera vertical ROI filtering when trigger direction is known
            top_roi_y_split: Top camera Y split line; top of frame is 0
            hand_path_filter_enabled: Enable hand path-based filtering (v4.6, default: True)
        """
        self.yolo = yolo
        self.min_vote_ratio = (
            config.vision.min_vote_ratio
            if min_vote_ratio is None
            else min_vote_ratio
        )
        if confidence_threshold is not None:
            default_top_threshold = confidence_threshold
            default_side_threshold = confidence_threshold
        else:
            default_top_threshold = config.vision.top_confidence_threshold
            default_side_threshold = config.vision.side_confidence_threshold
        self.top_confidence_threshold = (
            top_confidence_threshold
            if top_confidence_threshold is not None
            else default_top_threshold
        )
        self.side_confidence_threshold = (
            side_confidence_threshold
            if side_confidence_threshold is not None
            else default_side_threshold
        )
        self.confidence_threshold = min(
            self.top_confidence_threshold,
            self.side_confidence_threshold,
        )
        self.use_hwaccel = use_hwaccel
        self.motion_filter_enabled = motion_filter_enabled
        self.min_motion_displacement = (
            config.vision.motion_min_displacement_px
            if min_motion_displacement is None
            else min_motion_displacement
        )
        self.side_roi_x_max = (
            config.vision.side_roi_x_max
            if side_roi_x_max is None
            else side_roi_x_max
        )
        self.side_roi_soft_margin_px = max(
            0.0,
            float(config.vision.side_roi_soft_margin_px),
        )
        self.top_roi_enabled = (
            config.vision.top_roi_enabled
            if top_roi_enabled is None
            else top_roi_enabled
        )
        self.top_roi_y_split = (
            config.vision.top_roi_y_split
            if top_roi_y_split is None
            else top_roi_y_split
        )
        self.hand_path_filter_enabled = hand_path_filter_enabled
        self.min_vote_count = (
            config.vision.min_vote_count
            if min_vote_count is None
            else min_vote_count
        )
        self._warn_if_field_tuning_not_loaded()

    @classmethod
    def _warn_if_field_tuning_not_loaded(cls) -> None:
        if cls._field_tuning_warning_emitted:
            return
        cls._field_tuning_warning_emitted = True
        if (
            config.vision.top_crop_policy != "left"
            or config.vision.side_crop_policy != "left"
            or not config.vision.diagnostic_all_class_trace
        ):
            logger.warning(
                "[VIDEO][config] field tuning may not be loaded: "
                f"top_crop_policy={config.vision.top_crop_policy}, "
                f"side_crop_policy={config.vision.side_crop_policy}, "
                f"diagnostic_all_class_trace={config.vision.diagnostic_all_class_trace}"
            )

    def _threshold_for_camera(self, camera_type: str) -> float:
        if self._uses_freezer_dual_top_profile(camera_type):
            return self.top_confidence_threshold
        if camera_type == "side":
            return self.side_confidence_threshold
        return self.top_confidence_threshold

    @staticmethod
    def _is_freezer_mode() -> bool:
        return str(config.machine.cabinet_type).lower() == "freezer"

    def _uses_freezer_dual_top_profile(self, camera_type: str) -> bool:
        return (
            self._is_freezer_mode()
            and str(config.vision.camera_layout).lower() == "dual_top_proxy"
            and camera_type in {"top", "side"}
        )

    @staticmethod
    def _freezer_handled_filter_enabled(delta_weight: Optional[float]) -> bool:
        return (
            str(config.machine.cabinet_type).lower() == "freezer"
            and str(config.vision.camera_layout).lower() == "dual_top_proxy"
            and delta_weight is not None
            and float(delta_weight) < 0.0
        )

    @classmethod
    def _freezer_candidate_filter_config_payload(
        cls,
        *,
        delta_weight: Optional[float],
        raw_candidate_count: int,
        handled_candidate_count: int,
    ) -> dict[str, Any]:
        target_weight: Optional[float]
        try:
            target_weight = (
                round(abs(float(delta_weight)), 1)
                if delta_weight is not None
                else None
            )
        except (TypeError, ValueError):
            target_weight = None
        return {
            "cabinet_type": str(config.machine.cabinet_type).lower(),
            "camera_layout": str(config.vision.camera_layout).lower(),
            "freezer_handled_filter_enabled": cls._freezer_handled_filter_enabled(
                delta_weight
            ),
            "raw_candidate_count": int(raw_candidate_count),
            "handled_candidate_count": int(handled_candidate_count),
            "target_weight": target_weight,
            "top_k": int(config.vision.top_k),
            "freezer_min_vote_ratio": float(config.vision.freezer_min_vote_ratio),
            "freezer_min_vote_count": int(config.vision.freezer_min_vote_count),
            "freezer_motion_min_displacement_px": float(
                config.vision.freezer_motion_min_displacement_px
            ),
            "freezer_roi_vertical_region": str(
                config.vision.freezer_roi_vertical_region
            ).lower(),
            "freezer_roi_y_split": float(cls._freezer_configured_roi_y_split()),
            "freezer_lower_roi_y_split_legacy": float(
                config.vision.freezer_lower_roi_y_split
            ),
            "freezer_min_exit_path_votes": int(config.vision.freezer_min_exit_path_votes),
            "freezer_vision_multi_without_weight_enabled": bool(
                config.weight.freezer_vision_multi_without_weight_enabled
            ),
            "freezer_multi_min_confidence": float(
                config.weight.freezer_multi_min_confidence
            ),
        }

    @classmethod
    def _freezer_candidate_filter_passthrough_reason(
        cls,
        *,
        delta_weight: Optional[float],
        raw_candidate_count: int,
        has_stage_evidence: bool,
    ) -> str:
        if str(config.vision.camera_layout).lower() != "dual_top_proxy":
            return "disabled_camera_layout"
        if delta_weight is None:
            return "missing_delta_weight"
        try:
            if float(delta_weight) >= 0.0:
                return "non_removal_delta"
        except (TypeError, ValueError):
            return "invalid_delta_weight"
        if raw_candidate_count <= 1 and not has_stage_evidence:
            return "single_candidate_passthrough"
        return "filter_disabled"

    @staticmethod
    def _freezer_trace_has_multi_item_evidence(
        trace_context: Optional[TriggerTraceContext],
    ) -> bool:
        loadcell = getattr(trace_context, "loadcell", None)
        if not isinstance(loadcell, dict):
            return False

        removal_targets = loadcell.get("removal_segment_targets") or []
        if isinstance(removal_targets, list) and len(removal_targets) >= 2:
            return True
        if bool(loadcell.get("compound_event")):
            return True
        try:
            return int(loadcell.get("compound_negative_segment_count", 0) or 0) >= 2
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _freezer_candidate_unit_weight(
        vote: VoteResult,
        product_weights: Optional[Dict[int, float]],
    ) -> Optional[float]:
        if not product_weights:
            return None
        try:
            weight = product_weights.get(int(vote.class_id))
        except (TypeError, ValueError):
            return None
        try:
            weight_value = float(weight)
        except (TypeError, ValueError):
            return None
        return weight_value if weight_value > 0.0 else None

    @staticmethod
    def _freezer_candidate_stock(
        vote: VoteResult,
        product_stocks: Optional[Dict[int, int]],
    ) -> Optional[int]:
        if not product_stocks:
            return None
        try:
            stock = product_stocks.get(int(vote.class_id))
        except (TypeError, ValueError):
            return None
        if stock is None:
            return None
        try:
            return int(stock)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _freezer_weight_tolerance_grams() -> float:
        return max(0.0, float(config.weight.freezer_weight_tolerance_grams))

    @classmethod
    def _freezer_count_tolerance(cls, count: int) -> float:
        return cls._freezer_weight_tolerance_grams()

    @classmethod
    def _freezer_supported_instance_count(
        cls,
        vote: VoteResult,
        *,
        target_weight: float,
        product_weights: Optional[Dict[int, float]],
    ) -> int:
        unit_weight = cls._freezer_candidate_unit_weight(vote, product_weights)
        if unit_weight is None:
            return 1
        try:
            hint = int(getattr(vote, "instance_count_hint", 1) or 1)
        except (TypeError, ValueError):
            hint = 1
        hint = max(1, min(hint, int(config.weight.max_count_per_item)))
        if hint <= 1:
            return 1

        best_count = 1
        best_residual = abs(target_weight - unit_weight)
        for count in range(2, hint + 1):
            residual = abs(target_weight - unit_weight * count)
            if residual <= cls._freezer_count_tolerance(count) and residual < best_residual:
                best_count = count
                best_residual = residual
        return best_count

    @classmethod
    def _freezer_count_allowed_residual(cls, count: int) -> float:
        count_scaled = float(config.weight.tolerance_grams) + (
            float(config.weight.same_product_count_tolerance_grams) * max(1, int(count))
        )
        return min(cls._freezer_weight_tolerance_grams(), count_scaled)

    @classmethod
    def _freezer_same_product_repeat_diagnostic(
        cls,
        vote: VoteResult,
        *,
        target_weight: float,
        unit_weight: Optional[float],
        single_residual: float,
        confidence: float,
        exit_path_votes: int,
        source: str,
        product_stocks: Optional[Dict[int, int]],
    ) -> Optional[dict[str, Any]]:
        if unit_weight is None or unit_weight <= 0.0:
            return None
        nearest_count = int(round(target_weight / unit_weight))
        if nearest_count < 2:
            return None

        stock = cls._freezer_candidate_stock(vote, product_stocks)
        caps = [
            max(1, int(config.weight.max_items_per_segment)),
            max(1, int(config.weight.same_product_max_count)),
            max(1, int(config.weight.max_count_per_item)),
        ]
        if stock is not None:
            caps.append(max(0, stock))
        max_count = min(caps)
        base = {
            "class_id": int(vote.class_id),
            "name": vote.class_name,
            "nearestCount": nearest_count,
            "maxCount": int(max_count),
            "unitWeight": round(float(unit_weight), 1),
            "stock": stock,
        }
        if max_count < 2:
            return {**base, "accepted": False, "reason": "count_cap_below_repeat"}

        best_count = 1
        best_residual = float(single_residual)
        for count in range(2, max_count + 1):
            residual = abs(target_weight - (unit_weight * count))
            if residual < best_residual:
                best_count = count
                best_residual = residual

        if best_count < 2:
            return None

        expected_weight = unit_weight * best_count
        allowed_residual = cls._freezer_count_allowed_residual(best_count)
        vote_count = max(
            int(getattr(vote, "vote_count", 0) or 0),
            int(getattr(vote, "raw_vote_count", 0) or 0),
        )
        min_votes = max(
            int(config.vision.freezer_min_vote_count),
            int(config.weight.detected_single_fallback_min_votes),
        )
        diagnostic = {
            **base,
            "count": int(best_count),
            "expectedWeight": round(float(expected_weight), 1),
            "countWeightResidual": round(float(best_residual), 1),
            "countAllowedResidual": round(float(allowed_residual), 1),
            "confidence": round(float(confidence), 4),
            "freezerExitPathVotes": int(exit_path_votes),
            "voteCount": int(vote_count),
            "accepted": False,
        }
        if str(source) != "vision":
            diagnostic["reason"] = "not_regular_vision_candidate"
        elif confidence < float(config.weight.freezer_multi_min_confidence):
            diagnostic["reason"] = "confidence_below_repeat_floor"
        elif exit_path_votes < int(config.vision.freezer_min_exit_path_votes):
            diagnostic["reason"] = "insufficient_exit_path_votes"
        elif vote_count < min_votes:
            diagnostic["reason"] = "insufficient_repeat_votes"
        elif best_residual > allowed_residual:
            diagnostic["reason"] = "repeat_residual_exceeds_tolerance"
        else:
            diagnostic["accepted"] = True
            diagnostic["reason"] = "same_product_repeat_weight_gate"
        return diagnostic

    @staticmethod
    def _freezer_stage_entry(
        trace_context: Optional[TriggerTraceContext],
        class_id: int,
    ) -> dict[str, Any]:
        stage_counts = getattr(trace_context, "stage_counts_by_class", {}) or {}
        if not isinstance(stage_counts, dict):
            return {}
        entry = stage_counts.get(str(class_id)) or stage_counts.get(class_id)
        return entry if isinstance(entry, dict) else {}

    _FREEZER_AMBIGUOUS_PRODUCT_CLASSES = frozenset({30, 42, 44})

    @staticmethod
    def _freezer_stage_int(entry: dict[str, Any], *keys: str) -> int:
        values: list[int] = []
        for key in keys:
            try:
                values.append(int(entry.get(key, 0) or 0))
            except (TypeError, ValueError):
                pass
        return max(values or [0])

    @staticmethod
    def _freezer_stage_float(entry: dict[str, Any], *keys: str) -> float:
        values: list[float] = []
        for key in keys:
            try:
                values.append(float(entry.get(key, 0.0) or 0.0))
            except (TypeError, ValueError):
                pass
        return max(values or [0.0])

    @staticmethod
    def _freezer_stage_bool(entry: dict[str, Any], *keys: str) -> bool:
        for key in keys:
            value = entry.get(key)
            if isinstance(value, bool):
                if value:
                    return True
                continue
            if isinstance(value, (int, float)) and value:
                return True
            if isinstance(value, str) and value.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                return True
        return False

    @classmethod
    def _freezer_candidate_interaction_evidence(
        cls,
        vote: VoteResult,
        *,
        stage_entry: dict[str, Any],
        dual_camera_exit_path: bool,
    ) -> dict[str, Any]:
        path_displacement = max(
            cls._freezer_stage_float(
                stage_entry,
                "pathDisplacementPx",
                "path_displacement_px",
            ),
            float(getattr(vote, "top_total_displacement", 0.0) or 0.0),
            float(getattr(vote, "side_total_displacement", 0.0) or 0.0),
        )
        max_distance = max(
            cls._freezer_stage_float(
                stage_entry,
                "maxDistancePx",
                "max_distance_px",
            ),
            float(getattr(vote, "top_max_distance", 0.0) or 0.0),
            float(getattr(vote, "side_max_distance", 0.0) or 0.0),
        )
        center_span_x = cls._freezer_stage_float(
            stage_entry,
            "centerSpanX",
            "center_span_x",
        )
        center_span_y = cls._freezer_stage_float(
            stage_entry,
            "centerSpanY",
            "center_span_y",
        )
        motion_threshold = cls._freezer_stage_float(
            stage_entry,
            "motionThresholdPx",
            "motion_threshold_px",
        )
        if motion_threshold <= 0.0:
            motion_threshold = float(config.vision.freezer_motion_min_displacement_px)

        has_motion_evidence = any(
            key in stage_entry
            for key in (
                "pathDisplacementPx",
                "path_displacement_px",
                "maxDistancePx",
                "max_distance_px",
                "centerSpanX",
                "center_span_x",
                "centerSpanY",
                "center_span_y",
            )
        ) or path_displacement > 0.0 or max_distance > 0.0
        trajectory_passed = cls._freezer_stage_bool(
            stage_entry,
            "trajectoryExitPathPassed",
            "trajectory_exit_path_passed",
        )
        if not trajectory_passed and motion_threshold > 0.0:
            trajectory_passed = path_displacement >= motion_threshold

        hand_path_valid = cls._freezer_stage_bool(
            stage_entry,
            "handPathValid",
            "hand_path_valid",
            "handPathValidUpperRoi",
            "hand_path_valid_upper_roi",
        )
        hand_path_valid_upper_roi = cls._freezer_stage_bool(
            stage_entry,
            "handPathValidUpperRoi",
            "hand_path_valid_upper_roi",
        )
        hand_interaction_passed = cls._freezer_stage_bool(
            stage_entry,
            "handInteractionPassed",
            "hand_interaction_passed",
        )
        hand_path_passed = cls._freezer_stage_bool(
            stage_entry,
            "handPathPassed",
            "hand_path_passed",
        ) or hand_interaction_passed
        hand_path_blocked = cls._freezer_stage_bool(
            stage_entry,
            "handPathBlocked",
            "hand_path_blocked",
        )
        hand_near_frame_count = cls._freezer_stage_int(
            stage_entry,
            "handNearFrameCount",
            "hand_near_frame_count",
        )
        hand_near_vote_ratio = cls._freezer_stage_float(
            stage_entry,
            "handNearVoteRatio",
            "hand_near_vote_ratio",
        )
        min_hand_distance_values = [
            cls._freezer_stage_float(
                stage_entry,
                "minHandDistancePx",
                "min_hand_distance_px",
            )
        ]
        min_hand_distance = min(
            [value for value in min_hand_distance_values if value > 0.0] or [0.0]
        )
        top_only = (
            not bool(dual_camera_exit_path)
            and (
                bool(getattr(vote, "top_detected", False))
                or bool(getattr(vote, "side_detected", False))
            )
        )
        static_likely = cls._freezer_stage_bool(
            stage_entry,
            "staticShelfLikely",
            "static_shelf_likely",
        )
        if (
            not static_likely
            and top_only
            and has_motion_evidence
            and motion_threshold > 0.0
            and not trajectory_passed
            and path_displacement < motion_threshold
        ):
            static_likely = True
        interaction_penalty = bool(
            top_only
            and static_likely
            and not trajectory_passed
            and not hand_path_passed
        )
        hand_path_hard_reject = bool(
            top_only and hand_path_valid and hand_path_blocked and not hand_path_passed
        )
        return {
            "pathDisplacementPx": round(float(path_displacement), 1),
            "maxDistancePx": round(float(max_distance), 1),
            "centerSpanX": round(float(center_span_x), 1),
            "centerSpanY": round(float(center_span_y), 1),
            "motionThresholdPx": round(float(motion_threshold), 1),
            "trajectoryExitPathPassed": bool(trajectory_passed),
            "staticShelfLikely": bool(static_likely),
            "handPathValid": bool(hand_path_valid),
            "handPathValidUpperRoi": bool(hand_path_valid_upper_roi),
            "handPathPassed": bool(hand_path_passed),
            "handPathBlocked": bool(hand_path_blocked),
            "handInteractionPassed": bool(hand_interaction_passed),
            "handNearFrameCount": int(hand_near_frame_count),
            "handNearVoteRatio": round(float(hand_near_vote_ratio), 4),
            "minHandDistancePx": (
                round(float(min_hand_distance), 1)
                if min_hand_distance > 0.0
                else None
            ),
            "interactionPenalty": interaction_penalty,
            "handPathHardReject": hand_path_hard_reject,
        }

    @classmethod
    def _freezer_stage_camera_exit_counts(
        cls,
        entry: dict[str, Any],
    ) -> dict[str, int]:
        cameras = entry.get("cameras") or {}
        if not isinstance(cameras, dict):
            return {}

        exit_counts: dict[str, int] = {}
        for camera, camera_entry in cameras.items():
            if not isinstance(camera_entry, dict):
                continue
            count = cls._freezer_stage_int(
                camera_entry,
                "freezerExitPathVotes",
                "freezer_exit_path_votes",
                "freezer_roi_passed",
            )
            if count > 0:
                exit_counts[str(camera)] = count
        return exit_counts

    @staticmethod
    def _freezer_stage_center(
        entry: dict[str, Any],
    ) -> tuple[Optional[float], Optional[float]]:
        try:
            x_avg = (
                float(entry["roi_x_avg"])
                if entry.get("roi_x_avg") is not None
                else None
            )
        except (TypeError, ValueError):
            x_avg = None
        try:
            y_avg = (
                float(entry["roi_y_avg"])
                if entry.get("roi_y_avg") is not None
                else None
            )
        except (TypeError, ValueError):
            y_avg = None
        return x_avg, y_avg

    @staticmethod
    def _freezer_spatially_clustered(
        first: dict[str, Any],
        second: dict[str, Any],
    ) -> bool:
        if (
            first.get("roi_x_avg") is None
            or first.get("roi_y_avg") is None
            or second.get("roi_x_avg") is None
            or second.get("roi_y_avg") is None
        ):
            return False
        try:
            return (
                abs(float(first["roi_x_avg"]) - float(second["roi_x_avg"])) <= 130.0
                and abs(float(first["roi_y_avg"]) - float(second["roi_y_avg"])) <= 130.0
            )
        except (TypeError, ValueError):
            return False

    @classmethod
    def _freezer_stage_vote_result(
        cls,
        *,
        class_id: int,
        entry: dict[str, Any],
        product_weights: Optional[Dict[int, float]],
        target_weight: float,
        near_residual_limit: float,
        min_exit_path_votes: int,
    ) -> Optional[VoteResult]:
        exit_path_votes = cls._freezer_stage_int(
            entry,
            "freezerExitPathVotes",
            "freezer_exit_path_votes",
            "freezer_roi_passed",
        )
        threshold_votes = cls._freezer_stage_int(
            entry,
            "threshold_passed",
            "motion_passed",
            "raw",
        )
        min_stage_votes = max(1, int(config.weight.detected_single_fallback_min_votes))
        if exit_path_votes < min_exit_path_votes or threshold_votes < min_stage_votes:
            return None

        unit_weight = cls._freezer_stage_float(entry, "unit_weight_g")
        if unit_weight <= 0.0 and product_weights is not None:
            try:
                unit_weight = float(product_weights.get(class_id, 0.0) or 0.0)
            except (TypeError, ValueError):
                unit_weight = 0.0
        if unit_weight <= 0.0:
            return None

        residual = abs(target_weight - unit_weight)
        if residual > near_residual_limit:
            return None

        stage_confidence = cls._freezer_stage_float(
            entry,
            "freezer_roi_passed_max_confidence",
            "freezer_roi_filtered_max_confidence",
            "threshold_passed_max_confidence",
            "raw_max_confidence",
        )
        if stage_confidence < float(config.weight.multi_kind_min_confidence):
            return None

        cameras = entry.get("cameras") if isinstance(entry.get("cameras"), dict) else {}
        top_entry = cameras.get("top", {}) if isinstance(cameras, dict) else {}
        side_entry = cameras.get("side", {}) if isinstance(cameras, dict) else {}
        if not isinstance(top_entry, dict):
            top_entry = {}
        if not isinstance(side_entry, dict):
            side_entry = {}

        top_confidence = cls._freezer_stage_float(
            top_entry,
            "freezer_roi_passed_max_confidence",
            "freezer_roi_filtered_max_confidence",
            "threshold_passed_max_confidence",
            "raw_max_confidence",
        )
        side_confidence = cls._freezer_stage_float(
            side_entry,
            "freezer_roi_passed_max_confidence",
            "freezer_roi_filtered_max_confidence",
            "threshold_passed_max_confidence",
            "raw_max_confidence",
        )
        top_votes = cls._freezer_stage_int(
            top_entry,
            "freezerExitPathVotes",
            "freezer_roi_passed",
            "threshold_passed",
            "raw",
        )
        side_votes = cls._freezer_stage_int(
            side_entry,
            "freezerExitPathVotes",
            "freezer_roi_passed",
            "threshold_passed",
            "raw",
        )

        return VoteResult(
            class_id=class_id,
            class_name=str(entry.get("name") or f"class_{class_id}"),
            vote_count=max(1, threshold_votes),
            max_confidence=stage_confidence,
            avg_confidence=stage_confidence,
            top_detected=top_votes > 0,
            side_detected=side_votes > 0,
            top_vote_count=top_votes,
            side_vote_count=side_votes,
            top_max_confidence=top_confidence,
            side_max_confidence=side_confidence,
            weighted_confidence=stage_confidence,
            source="freezer_stage_exit_path",
            raw_vote_count=cls._freezer_stage_int(entry, "raw", "threshold_passed"),
            top_motion_passed=bool(top_entry.get("motion_passed") or top_entry.get("motion_filtered")),
            side_motion_passed=bool(side_entry.get("motion_passed") or side_entry.get("motion_filtered")),
            motion_gate_passed=True,
            weight_gate_passed=residual
            <= float(config.weight.detected_single_fallback_tolerance_grams),
            rescue_tolerance_g=near_residual_limit,
            rescue_weight_residual_g=residual,
            freezer_exit_path_votes=exit_path_votes,
        )

    @classmethod
    def _freezer_stage_only_candidates(
        cls,
        *,
        existing_class_ids: set[int],
        target_weight: float,
        near_residual_limit: float,
        min_exit_path_votes: int,
        product_weights: Optional[Dict[int, float]],
        trace_context: Optional[TriggerTraceContext],
    ) -> list[VoteResult]:
        stage_counts = getattr(trace_context, "stage_counts_by_class", {}) or {}
        if not isinstance(stage_counts, dict):
            return []

        candidates: list[VoteResult] = []
        for raw_class_id, entry in stage_counts.items():
            if not isinstance(entry, dict):
                continue
            try:
                class_id = int(entry.get("class_id", raw_class_id))
            except (TypeError, ValueError):
                continue
            if class_id in existing_class_ids:
                continue
            candidate = cls._freezer_stage_vote_result(
                class_id=class_id,
                entry=entry,
                product_weights=product_weights,
                target_weight=target_weight,
                near_residual_limit=near_residual_limit,
                min_exit_path_votes=min_exit_path_votes,
            )
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    @classmethod
    def _freezer_exit_path_votes(
        cls,
        vote: VoteResult,
        trace_context: Optional[TriggerTraceContext],
    ) -> int:
        values: list[int] = []
        for value in (
            getattr(vote, "freezer_exit_path_votes", 0),
            getattr(vote, "freezerExitPathVotes", 0),
        ):
            try:
                values.append(int(value or 0))
            except (TypeError, ValueError):
                pass

        entry = cls._freezer_stage_entry(trace_context, int(vote.class_id))
        for key in ("freezerExitPathVotes", "freezer_exit_path_votes", "freezer_roi_passed"):
            try:
                values.append(int(entry.get(key, 0) or 0))
            except (TypeError, ValueError):
                pass
        return max(values or [0])

    @classmethod
    def filter_freezer_handled_candidates(
        cls,
        vote_results: List[VoteResult],
        *,
        delta_weight: Optional[float],
        product_weights: Optional[Dict[int, float]] = None,
        product_stocks: Optional[Dict[int, int]] = None,
        trace_context: Optional[TriggerTraceContext] = None,
        log_prefix: str = "VIDEO",
    ) -> List[VoteResult]:
        """Reduce freezer dual-top removal candidates to handled-item evidence."""
        votes = list(vote_results or [])
        stage_counts = getattr(trace_context, "stage_counts_by_class", {}) or {}
        has_stage_evidence = isinstance(stage_counts, dict) and bool(stage_counts)
        if not cls._freezer_handled_filter_enabled(delta_weight) or (
            len(votes) <= 1 and not has_stage_evidence
        ):
            if cls._is_freezer_mode():
                reason = cls._freezer_candidate_filter_passthrough_reason(
                    delta_weight=delta_weight,
                    raw_candidate_count=len(votes),
                    has_stage_evidence=has_stage_evidence,
                )
                cls._record_freezer_candidate_filter_diagnostics(
                    trace_context,
                    {
                        **cls._freezer_candidate_filter_config_payload(
                            delta_weight=delta_weight,
                            raw_candidate_count=len(votes),
                            handled_candidate_count=len(votes),
                        ),
                        "accepted": False,
                        "reason": reason,
                    },
                )
                if reason == "disabled_camera_layout":
                    logger.warning(
                        "[%s][FREEZER-CANDIDATE-FILTER] disabled "
                        "cabinet_type=freezer camera_layout=%s expected=dual_top_proxy",
                        log_prefix,
                        config.vision.camera_layout,
                    )
            return votes

        target_weight = abs(float(delta_weight))
        if cls._freezer_trace_has_multi_item_evidence(trace_context):
            cls._record_freezer_candidate_filter_diagnostics(
                trace_context,
                {
                    **cls._freezer_candidate_filter_config_payload(
                        delta_weight=delta_weight,
                        raw_candidate_count=len(votes),
                        handled_candidate_count=len(votes),
                    ),
                    "accepted": False,
                    "reason": "multi_item_trace_evidence_passthrough",
                },
            )
            return votes

        candidate_limit = max(1, int(config.vision.top_k))
        ranked_votes = votes[:candidate_limit]
        confidence_band = max(0.0, float(config.weight.freezer_confidence_tie_band))
        min_exit_path_votes = max(
            0,
            int(getattr(config.vision, "freezer_min_exit_path_votes", 3)),
        )
        strict_residual_limit = max(
            0.0,
            float(config.weight.detected_single_fallback_tolerance_grams),
        )
        near_residual_limit = max(
            strict_residual_limit,
            cls._freezer_weight_tolerance_grams(),
        )

        existing_class_ids = {int(vote.class_id) for vote in ranked_votes}
        stage_only_votes = cls._freezer_stage_only_candidates(
            existing_class_ids=existing_class_ids,
            target_weight=target_weight,
            near_residual_limit=near_residual_limit,
            min_exit_path_votes=min_exit_path_votes,
            product_weights=product_weights,
            trace_context=trace_context,
        )

        scored: list[dict[str, Any]] = []
        for index, vote in enumerate([*ranked_votes, *stage_only_votes]):
            unit_weight = cls._freezer_candidate_unit_weight(vote, product_weights)
            residual = (
                abs(target_weight - unit_weight)
                if unit_weight is not None
                else target_weight
            )
            confidence = float(getattr(vote, "weighted_confidence", 0.0) or 0.0)
            exit_path_votes = cls._freezer_exit_path_votes(vote, trace_context)
            stage_entry = cls._freezer_stage_entry(trace_context, int(vote.class_id))
            camera_exit_counts = cls._freezer_stage_camera_exit_counts(stage_entry)
            source = str(getattr(vote, "source", "vision") or "vision")
            stage_only = source == "freezer_stage_exit_path"
            identity_supported = source != "vision" or confidence >= 0.3
            dual_camera_exit_path = len(camera_exit_counts) >= 2
            roi_x_avg, roi_y_avg = cls._freezer_stage_center(stage_entry)
            interaction_evidence = cls._freezer_candidate_interaction_evidence(
                vote,
                stage_entry=stage_entry,
                dual_camera_exit_path=dual_camera_exit_path,
            )
            repeat_diagnostic = cls._freezer_same_product_repeat_diagnostic(
                vote,
                target_weight=target_weight,
                unit_weight=unit_weight,
                single_residual=residual,
                confidence=confidence,
                exit_path_votes=exit_path_votes,
                source=source,
                product_stocks=product_stocks,
            )
            if (
                stage_only
                and int(vote.class_id) in cls._FREEZER_AMBIGUOUS_PRODUCT_CLASSES
                and dual_camera_exit_path
                and unit_weight is not None
                and residual <= near_residual_limit
                and exit_path_votes >= min_exit_path_votes
            ):
                tier = -1
                reason = "ambiguous_dual_camera_stage_exit_path"
            elif (
                unit_weight is not None
                and residual <= strict_residual_limit
                and exit_path_votes >= min_exit_path_votes
            ):
                tier = 0
                reason = "weight_gate_exit_path"
            elif (
                unit_weight is not None
                and residual <= near_residual_limit
                and exit_path_votes >= min_exit_path_votes
            ):
                tier = 1
                reason = "near_weight_exit_path"
            else:
                tier = 2
                reason = "confidence_weight_tiebreak"
            scored.append(
                {
                    "index": index,
                    "vote": vote,
                    "unit_weight": unit_weight,
                    "residual": residual,
                    "confidence": confidence,
                    "freezer_exit_path_votes": exit_path_votes,
                    "camera_exit_counts": camera_exit_counts,
                    "dual_camera_exit_path": dual_camera_exit_path,
                    "roi_x_avg": roi_x_avg,
                    "roi_y_avg": roi_y_avg,
                    "interaction": interaction_evidence,
                    "interaction_penalty": bool(
                        interaction_evidence["interactionPenalty"]
                    ),
                    "hand_path_hard_reject": bool(
                        interaction_evidence["handPathHardReject"]
                    ),
                    "source": source,
                    "stage_only": stage_only,
                    "identity_supported": identity_supported,
                    "source_priority": 1 if stage_only else 0,
                    "tier": tier,
                    "reason": reason,
                    "same_product_repeat": repeat_diagnostic
                    if repeat_diagnostic and repeat_diagnostic.get("accepted")
                    else None,
                    "same_product_repeat_rejection": repeat_diagnostic
                    if repeat_diagnostic and not repeat_diagnostic.get("accepted")
                    else None,
                }
            )

        def considered_entry(item: dict[str, Any]) -> dict[str, Any]:
            unit_weight = item["unit_weight"]
            entry = {
                "rank": int(item["index"]) + 1,
                "class_id": int(item["vote"].class_id),
                "name": item["vote"].class_name,
                "confidence": round(float(item["confidence"]), 4),
                "unitWeight": round(float(unit_weight), 1)
                if unit_weight is not None
                else None,
                "weightResidual": round(float(item["residual"]), 1),
                "freezerExitPathVotes": int(item["freezer_exit_path_votes"]),
                "source": str(item["source"]),
                "stageOnly": bool(item["stage_only"]),
                "identitySupported": bool(item["identity_supported"]),
                "cameraExitCounts": dict(item["camera_exit_counts"]),
                "dualCameraExitPath": bool(item["dual_camera_exit_path"]),
                "roiXAvg": (
                    round(float(item["roi_x_avg"]), 1)
                    if item["roi_x_avg"] is not None
                    else None
                ),
                "roiYAvg": (
                    round(float(item["roi_y_avg"]), 1)
                    if item["roi_y_avg"] is not None
                    else None
                ),
                "selectionTier": str(item["reason"]),
            }
            interaction = item.get("interaction") or {}
            entry.update(
                {
                    "pathDisplacementPx": interaction.get("pathDisplacementPx"),
                    "maxDistancePx": interaction.get("maxDistancePx"),
                    "centerSpanX": interaction.get("centerSpanX"),
                    "centerSpanY": interaction.get("centerSpanY"),
                    "trajectoryExitPathPassed": bool(
                        interaction.get("trajectoryExitPathPassed")
                    ),
                    "staticShelfLikely": bool(
                        interaction.get("staticShelfLikely")
                    ),
                    "handPathValid": bool(interaction.get("handPathValid")),
                    "handPathValidUpperRoi": bool(
                        interaction.get("handPathValidUpperRoi")
                    ),
                    "handPathPassed": bool(interaction.get("handPathPassed")),
                    "handPathBlocked": bool(interaction.get("handPathBlocked")),
                    "handInteractionPassed": bool(
                        interaction.get("handInteractionPassed")
                    ),
                    "handNearFrameCount": interaction.get("handNearFrameCount"),
                    "handNearVoteRatio": interaction.get("handNearVoteRatio"),
                    "minHandDistancePx": interaction.get("minHandDistancePx"),
                    "interactionPenalty": bool(item.get("interaction_penalty")),
                }
            )
            if item.get("interaction_rejected_reason"):
                entry["interactionRejectedReason"] = str(
                    item["interaction_rejected_reason"]
                )
            repeat = item.get("same_product_repeat")
            if repeat:
                entry.update(
                    {
                        "sameProductRepeatCandidate": True,
                        "count": int(repeat["count"]),
                        "expectedWeight": round(float(repeat["expectedWeight"]), 1),
                        "countWeightResidual": round(
                            float(repeat["countWeightResidual"]), 1
                        ),
                        "countAllowedResidual": round(
                            float(repeat["countAllowedResidual"]), 1
                        ),
                    }
                )
            rejection = item.get("same_product_repeat_rejection")
            if rejection:
                entry["sameProductRepeatRejectedReason"] = str(
                    rejection.get("reason", "rejected")
                )
                entry["nearestRepeatCount"] = int(rejection.get("count", 0) or 0)
            return entry

        selectable_scored = list(scored)
        interaction_rejected_items: list[dict[str, Any]] = []
        hand_blocked_items = [
            item for item in scored if bool(item.get("hand_path_hard_reject"))
        ]
        if hand_blocked_items and len(hand_blocked_items) < len(scored):
            for item in hand_blocked_items:
                item["interaction_rejected_reason"] = "hand_path_blocked"
            interaction_rejected_items = hand_blocked_items
            selectable_scored = [
                item for item in scored if not bool(item.get("hand_path_hard_reject"))
            ]

        rejected_multi_diagnostics: Optional[dict[str, Any]] = None
        if bool(config.weight.freezer_vision_multi_without_weight_enabled):
            multi_min_confidence = float(config.weight.freezer_multi_min_confidence)
            strong_multi_items = [
                item
                for item in selectable_scored
                if float(item["confidence"]) >= multi_min_confidence
                and int(item["freezer_exit_path_votes"]) >= min_exit_path_votes
                and bool(item["dual_camera_exit_path"])
                and bool(item["identity_supported"])
            ]
            if len(strong_multi_items) >= 2:
                strong_multi_items.sort(
                    key=lambda item: (int(item["index"]), -float(item["confidence"]))
                )
                max_kinds = max(2, int(config.weight.max_combination_kinds))
                allowed_residual = cls._freezer_weight_tolerance_grams()
                viable_multi: list[dict[str, Any]] = []
                for size in range(2, min(max_kinds, len(strong_multi_items)) + 1):
                    for combo in combinations(strong_multi_items, size):
                        selected = list(combo)
                        expected_weight = 0.0
                        selected_counts: dict[int, int] = {}
                        weight_known = True
                        for item in selected:
                            unit_weight = item["unit_weight"]
                            if unit_weight is None:
                                weight_known = False
                                break
                            count = cls._freezer_supported_instance_count(
                                item["vote"],
                                target_weight=target_weight,
                                product_weights=product_weights,
                            )
                            selected_counts[int(item["vote"].class_id)] = count
                            expected_weight += float(unit_weight) * count
                        if not weight_known:
                            continue
                        residual = abs(target_weight - expected_weight)
                        if residual <= allowed_residual:
                            viable_multi.append(
                                {
                                    "selected": selected,
                                    "selected_counts": selected_counts,
                                    "expected_weight": expected_weight,
                                    "residual": residual,
                                    "rank_sum": sum(
                                        int(item["index"]) for item in selected
                                    ),
                                    "confidence_sum": sum(
                                        float(item["confidence"]) for item in selected
                                    ),
                                }
                            )
                if not viable_multi:
                    rejected_multi_diagnostics = {
                        "reason": "freezer_multi_kind_weight_mismatch",
                        "targetWeight": round(target_weight, 1),
                        "allowedResidual": round(allowed_residual, 1),
                        "selectedClassIds": [
                            int(item["vote"].class_id) for item in strong_multi_items
                        ],
                        "considered": [
                            considered_entry(item) for item in strong_multi_items
                        ],
                    }
                else:
                    viable_multi.sort(
                        key=lambda item: (
                            len(item["selected"]),
                            float(item["residual"]),
                            int(item["rank_sum"]),
                            -float(item["confidence_sum"]),
                        )
                    )
                    selected_multi = viable_multi[0]
                    selected_items = selected_multi["selected"]
                    selected_counts = selected_multi["selected_counts"]
                    selected_votes = [
                        replace(
                            item["vote"],
                            instance_count_hint=selected_counts[
                                int(item["vote"].class_id)
                            ],
                            freezer_exit_path_votes=int(item["freezer_exit_path_votes"]),
                            hand_path_valid=bool(
                                (item.get("interaction") or {}).get("handPathValid")
                            ),
                            hand_path_valid_upper_roi=bool(
                                (item.get("interaction") or {}).get(
                                    "handPathValidUpperRoi"
                                )
                            ),
                            hand_path_passed=bool(
                                (item.get("interaction") or {}).get("handPathPassed")
                            ),
                            hand_path_blocked=bool(
                                (item.get("interaction") or {}).get("handPathBlocked")
                            ),
                            hand_interaction_passed=bool(
                                (item.get("interaction") or {}).get(
                                    "handInteractionPassed"
                                )
                            ),
                            hand_near_frame_count=int(
                                (item.get("interaction") or {}).get(
                                    "handNearFrameCount",
                                    0,
                                )
                                or 0
                            ),
                            hand_near_vote_ratio=float(
                                (item.get("interaction") or {}).get(
                                    "handNearVoteRatio",
                                    0.0,
                                )
                                or 0.0
                            ),
                            min_hand_distance_px=(item.get("interaction") or {}).get(
                                "minHandDistancePx"
                            ),
                        )
                        for item in selected_items
                    ]
                    diagnostics = {
                        **cls._freezer_candidate_filter_config_payload(
                            delta_weight=delta_weight,
                            raw_candidate_count=len(votes),
                            handled_candidate_count=len(selected_votes),
                        ),
                        "accepted": False,
                        "reason": "freezer_multi_kind_weight_fit",
                        "stage_only_candidate_count": len(stage_only_votes),
                        "minFreezerExitPathVotes": min_exit_path_votes,
                        "multiMinConfidence": round(multi_min_confidence, 4),
                        "expectedWeight": round(
                            float(selected_multi["expected_weight"]),
                            1,
                        ),
                        "weightResidual": round(float(selected_multi["residual"]), 1),
                        "allowedResidual": round(allowed_residual, 1),
                        "selectedClassIds": [
                            int(item["vote"].class_id) for item in selected_items
                        ],
                        "selected": [
                            {
                                **considered_entry(item),
                                "instance_count_hint": selected_counts[
                                    int(item["vote"].class_id)
                                ],
                            }
                            for item in selected_items
                        ],
                        "considered": [considered_entry(item) for item in scored],
                        "rejectedInteractionCandidates": [
                            considered_entry(item)
                            for item in interaction_rejected_items
                        ],
                    }
                    cls._record_freezer_candidate_filter_diagnostics(
                        trace_context,
                        diagnostics,
                    )
                    logger.info(
                        "[%s][FREEZER-CANDIDATE-FILTER] raw=%s handled=%s "
                        "reason=freezer_multi_kind_weight_fit residual=%.1fg",
                        log_prefix,
                        len(votes),
                        len(selected_votes),
                        float(selected_multi["residual"]),
                    )
                    return selected_votes

        handled_pool = [item for item in selectable_scored if int(item["tier"]) < 2]
        if handled_pool:
            selected_item = min(
                handled_pool,
                key=lambda item: (
                    int(item["tier"]) + int(item.get("interaction_penalty", False)),
                    int(item.get("interaction_penalty", False)),
                    int(not item["identity_supported"]),
                    int(item["source_priority"]),
                    -int(item["freezer_exit_path_votes"]),
                    float(item["residual"]),
                    -float(item["confidence"]),
                    int(item["index"]),
                ),
            )
            reason = str(selected_item["reason"])
        else:
            selectable_best_confidence = max(
                float(item["confidence"]) for item in selectable_scored
            )
            tie_pool = [
                item
                for item in selectable_scored
                if float(item["confidence"])
                >= selectable_best_confidence - confidence_band
            ]
            selected_item = min(
                tie_pool,
                key=lambda item: (
                    int(item.get("interaction_penalty", False)),
                    item["unit_weight"] is None,
                    float(item["residual"]),
                    int(item["index"]),
                    -float(item["confidence"]),
                ),
            )
            reason = "single_removal_weight_tiebreak"

        repeat_candidates = [
            item
            for item in selectable_scored
            if item.get("same_product_repeat") is not None
        ]
        rejected_repeat_selection: list[dict[str, Any]] = []
        if repeat_candidates:
            repeat_item = min(
                repeat_candidates,
                key=lambda item: (
                    float(item["same_product_repeat"]["countWeightResidual"]),
                    int(item["index"]),
                    -float(item["confidence"]),
                    -int(item["freezer_exit_path_votes"]),
                ),
            )
            repeat_residual = float(
                repeat_item["same_product_repeat"]["countWeightResidual"]
            )
            selected_single_residual = float(selected_item["residual"])
            residual_grace = float(config.weight.same_product_count_tolerance_grams)
            if bool(selected_item["dual_camera_exit_path"]):
                rejected_repeat_selection.append(
                    {
                        **considered_entry(repeat_item),
                        "reason": "dual_camera_single_preferred",
                    }
                )
            elif bool(repeat_item.get("interaction_penalty")) and not bool(
                selected_item.get("interaction_penalty")
            ):
                rejected_repeat_selection.append(
                    {
                        **considered_entry(repeat_item),
                        "reason": "interaction_supported_single_preferred",
                    }
                )
            elif repeat_residual > selected_single_residual + residual_grace:
                rejected_repeat_selection.append(
                    {
                        **considered_entry(repeat_item),
                        "reason": "single_residual_gap_preferred",
                    }
                )
            else:
                selected_item = repeat_item
                reason = "same_product_repeat_weight_gate"

        selected_index = int(selected_item["index"])
        selected_vote = selected_item["vote"]

        selected_repeat = selected_item.get("same_product_repeat")
        if reason == "same_product_repeat_weight_gate" and selected_repeat:
            supported_count = int(selected_repeat["count"])
        else:
            supported_count = cls._freezer_supported_instance_count(
                selected_vote,
                target_weight=target_weight,
                product_weights=product_weights,
            )
        selected = replace(
            selected_vote,
            instance_count_hint=supported_count,
            freezer_exit_path_votes=int(selected_item["freezer_exit_path_votes"]),
            hand_path_valid=bool(
                (selected_item.get("interaction") or {}).get("handPathValid")
            ),
            hand_path_valid_upper_roi=bool(
                (selected_item.get("interaction") or {}).get("handPathValidUpperRoi")
            ),
            hand_path_passed=bool(
                (selected_item.get("interaction") or {}).get("handPathPassed")
            ),
            hand_path_blocked=bool(
                (selected_item.get("interaction") or {}).get("handPathBlocked")
            ),
            hand_interaction_passed=bool(
                (selected_item.get("interaction") or {}).get("handInteractionPassed")
            ),
            hand_near_frame_count=int(
                (selected_item.get("interaction") or {}).get(
                    "handNearFrameCount",
                    0,
                )
                or 0
            ),
            hand_near_vote_ratio=float(
                (selected_item.get("interaction") or {}).get(
                    "handNearVoteRatio",
                    0.0,
                )
                or 0.0
            ),
            min_hand_distance_px=(selected_item.get("interaction") or {}).get(
                "minHandDistancePx"
            ),
        )
        unit_weight = cls._freezer_candidate_unit_weight(selected_vote, product_weights)
        residual = (
            abs(target_weight - (unit_weight * supported_count))
            if unit_weight is not None
            else target_weight
        )
        ambiguous_candidates = [
            {
                **considered_entry(item),
                "spatialClusterWithSelected": cls._freezer_spatially_clustered(
                    item,
                    selected_item,
                ),
            }
            for item in scored
            if int(item["vote"].class_id) in cls._FREEZER_AMBIGUOUS_PRODUCT_CLASSES
            and float(item["residual"]) <= near_residual_limit
            and int(item["freezer_exit_path_votes"]) >= min_exit_path_votes
        ]
        diagnostics = {
            **cls._freezer_candidate_filter_config_payload(
                delta_weight=delta_weight,
                raw_candidate_count=len(votes),
                handled_candidate_count=1,
            ),
            "accepted": True,
            "reason": reason,
            "stage_only_candidate_count": len(stage_only_votes),
            "confidence_band": round(confidence_band, 4),
            "minFreezerExitPathVotes": min_exit_path_votes,
            "strictResidualLimit": round(strict_residual_limit, 1),
            "nearResidualLimit": round(near_residual_limit, 1),
            "selected": {
                "rank": selected_index + 1,
                "class_id": int(selected.class_id),
                "name": selected.class_name,
                "confidence": round(
                    float(getattr(selected, "weighted_confidence", 0.0) or 0.0),
                    4,
                ),
                "source": getattr(selected, "source", "vision"),
                "stageOnly": bool(selected_item["stage_only"]),
                "identitySupported": bool(selected_item["identity_supported"]),
                "unit_weight": round(unit_weight, 1) if unit_weight is not None else None,
                "weight_residual": round(residual, 1),
                "count": int(supported_count),
                "expectedWeight": round(float(unit_weight or 0.0) * supported_count, 1),
                "countWeightResidual": round(residual, 1),
                "instance_count_hint": supported_count,
                "freezerExitPathVotes": int(selected.freezer_exit_path_votes),
                "cameraExitCounts": dict(selected_item["camera_exit_counts"]),
                "dualCameraExitPath": bool(selected_item["dual_camera_exit_path"]),
                "pathDisplacementPx": (
                    selected_item.get("interaction") or {}
                ).get("pathDisplacementPx"),
                "maxDistancePx": (
                    selected_item.get("interaction") or {}
                ).get("maxDistancePx"),
                "centerSpanX": (
                    selected_item.get("interaction") or {}
                ).get("centerSpanX"),
                "centerSpanY": (
                    selected_item.get("interaction") or {}
                ).get("centerSpanY"),
                "trajectoryExitPathPassed": bool(
                    (selected_item.get("interaction") or {}).get(
                        "trajectoryExitPathPassed"
                    )
                ),
                "staticShelfLikely": bool(
                    (selected_item.get("interaction") or {}).get("staticShelfLikely")
                ),
                "handPathValid": bool(
                    (selected_item.get("interaction") or {}).get("handPathValid")
                ),
                "handPathValidUpperRoi": bool(
                    (selected_item.get("interaction") or {}).get(
                        "handPathValidUpperRoi"
                    )
                ),
                "handPathPassed": bool(
                    (selected_item.get("interaction") or {}).get("handPathPassed")
                ),
                "handPathBlocked": bool(
                    (selected_item.get("interaction") or {}).get("handPathBlocked")
                ),
                "handInteractionPassed": bool(
                    (selected_item.get("interaction") or {}).get(
                        "handInteractionPassed"
                    )
                ),
                "handNearFrameCount": (
                    selected_item.get("interaction") or {}
                ).get("handNearFrameCount"),
                "handNearVoteRatio": (
                    selected_item.get("interaction") or {}
                ).get("handNearVoteRatio"),
                "minHandDistancePx": (
                    selected_item.get("interaction") or {}
                ).get("minHandDistancePx"),
                "interactionPenalty": bool(selected_item.get("interaction_penalty")),
                "selectionTier": reason,
            },
            "considered": [considered_entry(item) for item in scored],
            "rejectedInteractionCandidates": [
                considered_entry(item) for item in interaction_rejected_items
            ],
            "sameProductRepeatCandidates": [
                considered_entry(item)
                for item in scored
                if item.get("same_product_repeat") is not None
            ],
            "rejectedSameProductRepeatCandidates": [
                considered_entry(item)
                for item in scored
                if item.get("same_product_repeat_rejection") is not None
            ]
            + rejected_repeat_selection,
            "ambiguousCandidates": ambiguous_candidates,
            "hardNegativeCandidates": ambiguous_candidates,
        }
        if rejected_multi_diagnostics is not None:
            diagnostics["rejectedMultiCandidate"] = rejected_multi_diagnostics
        cls._record_freezer_candidate_filter_diagnostics(trace_context, diagnostics)
        logger.info(
            "[%s][FREEZER-CANDIDATE-FILTER] raw=%s handled=1 selected=%s "
            "target=%.1fg residual=%.1fg",
            log_prefix,
            len(votes),
            selected.class_name,
            target_weight,
            residual,
        )
        return [selected]

    @staticmethod
    def _record_freezer_candidate_filter_diagnostics(
        trace_context: Optional[TriggerTraceContext],
        diagnostics: dict,
    ) -> None:
        if trace_context is None or not hasattr(trace_context, "record_weight_diagnostics"):
            return
        existing = dict(getattr(trace_context, "weight_diagnostics", {}) or {})
        existing["freezer_candidate_filter"] = diagnostics
        trace_context.record_weight_diagnostics(existing)

    @staticmethod
    def _freezer_configured_roi_y_split() -> float:
        configured = getattr(config.vision, "freezer_roi_y_split", None)
        if configured is None:
            return float(config.vision.freezer_lower_roi_y_split)
        return float(configured)

    def _freezer_roi_y_split(self) -> float:
        return self._freezer_configured_roi_y_split()

    @staticmethod
    def _freezer_roi_vertical_region() -> str:
        region = str(
            getattr(config.vision, "freezer_roi_vertical_region", "upper")
        ).strip().lower()
        return region if region in {"upper", "lower"} else "upper"

    def _freezer_roi_direction(self) -> str:
        return f"freezer_{self._freezer_roi_vertical_region()}_half"

    def _new_hand_path_tracker(self) -> HandPathTracker:
        if (
            self._is_freezer_mode()
            and str(config.vision.camera_layout).lower() == "dual_top_proxy"
        ):
            return HandPathTracker(
                roi_y_split=self._freezer_roi_y_split(),
                roi_vertical_region=self._freezer_roi_vertical_region(),
                max_distance_px=float(config.vision.max_distance_px),
            )
        return HandPathTracker(max_distance_px=float(config.vision.max_distance_px))

    def _effective_min_vote_ratio(self) -> float:
        if self._is_freezer_mode():
            return float(config.vision.freezer_min_vote_ratio)
        return float(self.min_vote_ratio)

    def _effective_min_vote_count(self) -> int:
        if self._is_freezer_mode():
            return int(config.vision.freezer_min_vote_count)
        return int(self.min_vote_count)

    def _top_roi_direction(self, delta_weight: Optional[float]) -> Optional[str]:
        if not self.top_roi_enabled or delta_weight is None or delta_weight == 0.0:
            return None
        if delta_weight < 0.0:
            return "removal"
        return "return"

    def _top_roi_accepts(
        self,
        detection: YOLODetection,
        delta_weight: Optional[float],
    ) -> tuple[bool, Optional[str]]:
        direction = self._top_roi_direction(delta_weight)
        if direction is None:
            return True, None

        center_y = detection.center[1]
        return center_y >= self.top_roi_y_split, direction

    def _freezer_roi_accepts(self, detection: YOLODetection) -> bool:
        if not self.top_roi_enabled:
            return True
        center_y = float(detection.center[1])
        split = self._freezer_roi_y_split()
        if self._freezer_roi_vertical_region() == "lower":
            return center_y >= split
        return center_y <= split

    def _freezer_lower_roi_accepts(self, detection: YOLODetection) -> bool:
        return self._freezer_roi_accepts(detection)

    def _freezer_motion_floor(self) -> float:
        if self._is_freezer_mode():
            return float(config.vision.freezer_motion_min_displacement_px)
        return float(self.min_motion_displacement)

    def _motion_threshold_for_detection(
        self,
        camera_type: str,
        bbox_size: float,
    ) -> float:
        min_floor = (
            self._freezer_motion_floor()
            if self._uses_freezer_dual_top_profile(camera_type)
            else float(self.min_motion_displacement)
        )
        return max(min_floor, bbox_size * 0.10)

    def _low_confidence_roi_eligible(
        self,
        detection: YOLODetection,
        camera_type: str,
        delta_weight: Optional[float],
    ) -> bool:
        if self._uses_freezer_dual_top_profile(camera_type):
            return self._freezer_roi_accepts(detection)
        if camera_type == "top":
            roi_eligible, _ = self._top_roi_accepts(detection, delta_weight)
            return roi_eligible
        if camera_type == "side":
            return detection.center[0] <= self.side_roi_x_max
        return True

    def _side_roi_soft_limit(self) -> float:
        return float(self.side_roi_x_max) + float(self.side_roi_soft_margin_px)

    def _side_roi_accepts(self, center_x: float) -> tuple[bool, bool]:
        if center_x <= self.side_roi_x_max:
            return True, False
        if center_x <= self._side_roi_soft_limit():
            return True, True
        return False, False

    def _detect_frame(
        self,
        frame,
        allowed_class_ids: Optional[List[int]],
        camera_type: str,
    ):
        try:
            return self.yolo.detect(
                frame,
                allowed_class_ids=allowed_class_ids,
                camera_type=camera_type,
            )
        except TypeError as exc:
            if "camera_type" not in str(exc):
                raise
            return self.yolo.detect(frame, allowed_class_ids=allowed_class_ids)

    @staticmethod
    def _candidate_source_rank(vote: VoteResult) -> int:
        source = getattr(vote, "source", "vision") or "vision"
        if source == "vision":
            return 0
        if source in {"roi_rescue", "threshold_rescue"}:
            return 1
        return 2

    @classmethod
    def rank_candidates_by_source_priority(
        cls,
        candidates: List[VoteResult],
    ) -> List[VoteResult]:
        return sorted(
            candidates,
            key=lambda vote: (
                cls._candidate_source_rank(vote),
                -float(getattr(vote, "weighted_confidence", 0.0) or 0.0),
                -int(getattr(vote, "vote_count", 0) or 0),
                int(getattr(vote, "class_id", 0) or 0),
            ),
        )

    @classmethod
    def merge_rescue_votes(
        cls,
        vote_results: List[VoteResult],
        rescue_votes: List[VoteResult],
        candidate_limit: Optional[int] = None,
    ) -> List[VoteResult]:
        if not rescue_votes:
            return vote_results

        seen_class_ids = {vote.class_id for vote in vote_results}
        merged = list(vote_results)
        for rescue_vote in rescue_votes:
            if rescue_vote.class_id in seen_class_ids:
                continue
            merged.append(rescue_vote)
            seen_class_ids.add(rescue_vote.class_id)

        limit = max(
            1,
            int(config.vision.top_k if candidate_limit is None else candidate_limit),
        )
        return cls.rank_candidates_by_source_priority(merged)[:limit]

    @staticmethod
    def _roi_rescue_rejection_reason(
        candidate: ThresholdRescueCandidate,
    ) -> Optional[str]:
        if candidate.source != "roi_rescue":
            return None

        if config.vision.roi_rescue_require_motion and not candidate.side_motion_passed:
            return "roi_rescue_no_motion"

        if candidate.roi_x_avg is None or candidate.roi_x_limit is None:
            return "roi_rescue_missing_roi"

        max_over_limit_px = float(config.vision.roi_rescue_max_over_limit_px)
        if max_over_limit_px >= 0.0:
            max_allowed_x = float(candidate.roi_x_limit) + max_over_limit_px
            if float(candidate.roi_x_avg) > max_allowed_x:
                return "roi_rescue_too_far_right"

        return None

    @staticmethod
    def _record_stage(
        trace_context: Optional[TriggerTraceContext],
        *,
        class_id: int,
        class_name: str,
        stage: str,
        camera: str,
        amount: int = 1,
        confidence: Optional[float] = None,
        center: Optional[Tuple[float, float]] = None,
        roi_x_limit: Optional[float] = None,
        roi_y_limit: Optional[float] = None,
        roi_direction: Optional[str] = None,
    ) -> None:
        if trace_context is not None:
            trace_context.record_stage_count(
                class_id=class_id,
                class_name=class_name,
                stage=stage,
                camera=camera,
                amount=amount,
                confidence=confidence,
                center=center,
                roi_x_limit=roi_x_limit,
                roi_y_limit=roi_y_limit,
                roi_direction=roi_direction,
            )

    def _record_motion_evidence(
        self,
        trace_context: Optional[TriggerTraceContext],
        *,
        class_id: int,
        class_name: str,
        camera: str,
        tracker: Optional[BboxTracker],
        motion_passed: bool,
    ) -> None:
        if trace_context is None or tracker is None:
            return
        threshold = (
            tracker.dynamic_threshold
            if tracker.dynamic_threshold > 0
            else self._freezer_motion_floor()
        )
        trajectory_passed = bool(motion_passed and tracker.total_displacement >= threshold)
        static_likely = bool(
            motion_passed
            and threshold > 0.0
            and tracker.detection_count >= self._effective_min_vote_count()
            and tracker.total_displacement < threshold
        )
        trace_context.record_motion_evidence(
            class_id=class_id,
            class_name=class_name,
            camera=camera,
            path_displacement_px=tracker.total_displacement,
            max_distance_px=tracker.max_distance,
            center_span_x=tracker.center_span_x,
            center_span_y=tracker.center_span_y,
            motion_threshold_px=threshold,
            trajectory_exit_path_passed=trajectory_passed,
            static_shelf_likely=static_likely,
        )

    @staticmethod
    def _record_hand_path_evidence(
        trace_context: Optional[TriggerTraceContext],
        *,
        result: VoteResult,
        hand_path_valid: bool,
        hand_path_passed: bool,
        hand_path_blocked: bool,
        hand_metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        if trace_context is None:
            return
        metrics = hand_metrics or {}
        trace_context.record_hand_path_evidence(
            class_id=result.class_id,
            class_name=result.class_name,
            camera="top",
            hand_path_valid=hand_path_valid,
            hand_path_passed=hand_path_passed,
            hand_path_blocked=hand_path_blocked,
            hand_interaction_passed=bool(
                metrics.get("handInteractionPassed", hand_path_passed)
            ),
            hand_near_frame_count=int(metrics.get("handNearFrameCount", 0) or 0),
            hand_near_vote_ratio=float(metrics.get("handNearVoteRatio", 0.0) or 0.0),
            min_hand_distance_px=metrics.get("minHandDistancePx"),
            hand_path_valid_upper_roi=metrics.get(
                "handPathValidUpperRoi",
                hand_path_valid,
            ),
        )

    @staticmethod
    def _record_low_confidence_detection(
        low_confidence_stats: Dict[int, _LowConfidenceClassStats],
        *,
        class_id: int,
        class_name: str,
        confidence: float,
        camera: str,
        center: Tuple[float, float],
        frame_idx: int,
        bbox_size: float,
        roi_eligible: bool = True,
    ) -> None:
        if not config.vision.threshold_rescue_enabled or not roi_eligible:
            return
        stats = low_confidence_stats.setdefault(
            class_id,
            _LowConfidenceClassStats(class_id=class_id, class_name=class_name),
        )
        stats.add(camera, confidence, class_name, center, frame_idx, bbox_size)

    @staticmethod
    def _record_roi_filtered_detection(
        roi_filtered_stats: Dict[int, _RoiFilteredClassStats],
        *,
        class_id: int,
        class_name: str,
        confidence: float,
        center: Tuple[float, float],
        frame_idx: int,
        bbox_size: float,
        roi_x_limit: float,
    ) -> None:
        if not config.vision.threshold_rescue_enabled:
            return
        stats = roi_filtered_stats.setdefault(
            class_id,
            _RoiFilteredClassStats(class_id=class_id, class_name=class_name),
        )
        stats.add(
            confidence=confidence,
            class_name=class_name,
            center=center,
            frame_idx=frame_idx,
            bbox_size=bbox_size,
            roi_x_limit=roi_x_limit,
        )

    def _build_threshold_rescue_candidates(
        self,
        low_confidence_stats: Dict[int, _LowConfidenceClassStats],
        roi_filtered_stats: Dict[int, _RoiFilteredClassStats],
        allowed_class_ids: Optional[List[int]],
        log_prefix: str,
    ) -> List[ThresholdRescueCandidate]:
        if self._is_freezer_mode() or not config.vision.threshold_rescue_enabled:
            return []
        allowed_set = set(allowed_class_ids) if allowed_class_ids is not None else None
        candidates: List[ThresholdRescueCandidate] = []
        for class_id, stats in low_confidence_stats.items():
            if allowed_set is not None and class_id not in allowed_set:
                continue
            candidate = stats.to_rescue_candidate(
                self.min_motion_displacement,
                allow_no_motion=config.vision.weight_rescue_no_motion_enabled,
                no_motion_min_votes=config.vision.weight_rescue_no_motion_min_raw_votes,
            )
            if candidate is not None:
                self._mark_threshold_rescue_roi_conflict(
                    candidate,
                    roi_filtered_stats.get(class_id),
                )
                candidates.append(candidate)

        candidates.sort(
            key=lambda candidate: (
                int(candidate.top_detected) + int(candidate.side_detected),
                candidate.vote_count,
                candidate.max_confidence,
            ),
            reverse=True,
        )
        limit = max(0, int(config.vision.threshold_rescue_max_candidates))
        limited = candidates[:limit]
        if limited:
            logger.info(
                f"[{log_prefix}][THRESHOLD-RESCUE] candidates={len(limited)} "
                f"before_limit={len(candidates)}"
            )
        return limited

    @staticmethod
    def _mark_threshold_rescue_roi_conflict(
        candidate: ThresholdRescueCandidate,
        roi_stats: Optional[_RoiFilteredClassStats],
    ) -> None:
        if roi_stats is None or roi_stats.vote_count <= 0:
            return
        if roi_stats.max_confidence < float(config.vision.side_confidence_threshold):
            return
        candidate.roi_conflict = True
        candidate.roi_conflict_reason = "side_roi_filtered_stronger_evidence"
        candidate.roi_conflict_side_vote_count = int(roi_stats.vote_count)
        candidate.roi_conflict_side_max_confidence = float(roi_stats.max_confidence)
        candidate.roi_conflict_side_roi_x_avg = roi_stats.avg_center_x
        candidate.roi_conflict_side_roi_x_limit = float(roi_stats.roi_x_limit)

    @staticmethod
    def _threshold_rescue_roi_conflict_rejection_reason(
        candidate: ThresholdRescueCandidate,
    ) -> Optional[str]:
        if candidate.source != "threshold_rescue" or not candidate.roi_conflict:
            return None
        strong_inside_evidence = (
            candidate.max_confidence >= float(config.weight.multi_kind_min_confidence)
            and candidate.vote_count >= max(
                1,
                int(config.weight.detected_single_fallback_min_votes),
            )
        )
        if strong_inside_evidence:
            return None
        return candidate.roi_conflict_reason or "side_roi_filtered_stronger_evidence"

    def _build_roi_rescue_candidates(
        self,
        roi_filtered_stats: Dict[int, _RoiFilteredClassStats],
        allowed_class_ids: Optional[List[int]],
        log_prefix: str,
    ) -> List[ThresholdRescueCandidate]:
        if self._is_freezer_mode() or not config.vision.threshold_rescue_enabled:
            return []
        allowed_set = set(allowed_class_ids) if allowed_class_ids is not None else None
        candidates: List[ThresholdRescueCandidate] = []
        for class_id, stats in roi_filtered_stats.items():
            if allowed_set is not None and class_id not in allowed_set:
                continue
            candidate = stats.to_rescue_candidate(self.min_motion_displacement)
            if candidate is None:
                continue
            reason = self._roi_rescue_rejection_reason(candidate)
            if reason is not None:
                logger.info(
                    f"[{log_prefix}][ROI-RESCUE] rejected class={candidate.class_id} "
                    f"reason={reason} roi_x_avg={candidate.roi_x_avg} "
                    f"roi_x_limit={candidate.roi_x_limit} "
                    f"side_motion_passed={candidate.side_motion_passed}"
                )
                continue
            candidates.append(candidate)

        candidates.sort(
            key=lambda candidate: (
                candidate.vote_count,
                candidate.max_confidence,
                candidate.side_max_distance,
            ),
            reverse=True,
        )
        limit = max(0, int(config.vision.threshold_rescue_max_candidates))
        limited = candidates[:limit]
        if limited:
            logger.info(
                f"[{log_prefix}][ROI-RESCUE] candidates={len(limited)} "
                f"before_limit={len(candidates)}"
            )
        return limited

    @staticmethod
    def build_weight_gated_rescue_votes(
        rescue_candidates: List[ThresholdRescueCandidate],
        active_products: Optional[List[object]],
        delta_weight: float,
        *,
        diagnostics: Optional[dict] = None,
        existing_class_ids: Optional[set[int]] = None,
    ) -> List[VoteResult]:
        if not rescue_candidates or not active_products:
            if diagnostics is not None:
                diagnostics.update(
                    {
                        "target_weight": round(abs(float(delta_weight)), 1),
                        "considered": len(rescue_candidates or []),
                        "accepted": 0,
                        "rejections": {"no_roi_votes": 0},
                        "candidates": [],
                    }
                )
            return []

        active_map = {
            int(product.yolo_class_id): product
            for product in active_products
            if getattr(product, "yolo_class_id", None) is not None
        }
        target_weight = abs(delta_weight)
        rescue_tolerance = float(config.weight.rescue_tolerance_grams)
        no_motion_tolerance = float(
            config.vision.weight_rescue_no_motion_max_residual_grams
        )
        cap = float(config.vision.threshold_rescue_confidence_cap)
        rescue_votes: List[VoteResult] = []
        existing_class_ids = existing_class_ids or set()
        rejections: Dict[str, int] = {
            "duplicate_vision_candidate": 0,
            "not_active": 0,
            "invalid_weight": 0,
            "zero_stock": 0,
            "weight_mismatch": 0,
            "rescue_weight_mismatch": 0,
            "motion_rejected_but_weight_matched": 0,
            "roi_rescue_no_motion": 0,
            "roi_rescue_missing_roi": 0,
            "roi_rescue_too_far_right": 0,
            "threshold_rescue_roi_conflict": 0,
        }
        diagnostic_candidates: List[dict] = []

        for candidate in rescue_candidates:
            residual: Optional[float] = None
            reason: Optional[str] = None
            product_weight = 0.0
            stock_qty = 0
            motion_gate_passed = bool(
                getattr(
                    candidate,
                    "motion_gate_passed",
                    candidate.top_motion_passed or candidate.side_motion_passed,
                )
            )
            product = active_map.get(candidate.class_id)
            if candidate.class_id in existing_class_ids:
                reason = "duplicate_vision_candidate"
            if reason is None and candidate.source == "roi_rescue":
                reason = VideoProcessor._roi_rescue_rejection_reason(candidate)
            if reason is None and product is None:
                reason = "not_active"
            if reason is None and product is not None:
                product_weight = float(getattr(product, "product_weight", 0.0) or 0.0)
                stock_qty = int(getattr(product, "stock_qty", 0) or 0)
                residual = abs(target_weight - product_weight) if product_weight > 0 else None
                if product_weight <= 0:
                    reason = "invalid_weight"
                elif stock_qty <= 0:
                    reason = "zero_stock"
                elif residual is None or residual > rescue_tolerance:
                    reason = "rescue_weight_mismatch"
                elif (
                    candidate.source == "threshold_rescue"
                    and not motion_gate_passed
                    and (
                        not config.vision.weight_rescue_no_motion_enabled
                        or candidate.vote_count < config.vision.weight_rescue_no_motion_min_raw_votes
                        or residual > no_motion_tolerance
                    )
                ):
                    reason = "motion_rejected_but_weight_matched"
                elif (
                    threshold_roi_conflict_reason
                    := VideoProcessor._threshold_rescue_roi_conflict_rejection_reason(
                        candidate,
                    )
                ) is not None:
                    reason = "threshold_rescue_roi_conflict"

            candidate.weight_gate_passed = reason is None and residual is not None
            candidate.rescue_weight_residual_g = residual
            candidate.rescue_tolerance_g = rescue_tolerance

            if reason is not None:
                rejections[reason] = rejections.get(reason, 0) + 1
                if reason == "rescue_weight_mismatch":
                    rejections["weight_mismatch"] = rejections.get("weight_mismatch", 0) + 1
                diagnostic_candidates.append(
                    {
                        "class_id": candidate.class_id,
                        "name": candidate.class_name,
                        "source": candidate.source,
                        "raw_vote_count": candidate.vote_count,
                        "max_confidence": round(candidate.max_confidence, 4),
                        "unit_weight_g": round(product_weight, 1) if product_weight > 0 else None,
                        "stock_qty": stock_qty,
                        "weight_residual_g": (
                            round(float(residual), 1) if residual is not None else None
                        ),
                        "rescue_weight_residual_g": (
                            round(float(residual), 1) if residual is not None else None
                        ),
                        "rescue_tolerance_g": rescue_tolerance,
                        "weight_gate_passed": False,
                        "motion_gate_passed": motion_gate_passed,
                        "side_motion_passed": candidate.side_motion_passed,
                        "roi_x_avg": (
                            round(float(candidate.roi_x_avg), 1)
                            if candidate.roi_x_avg is not None
                            else None
                        ),
                        "roi_x_limit": (
                            round(float(candidate.roi_x_limit), 1)
                            if candidate.roi_x_limit is not None
                            else None
                        ),
                        "roi_conflict": candidate.roi_conflict,
                        "roi_conflict_reason": candidate.roi_conflict_reason,
                        "roi_conflict_side_vote_count": candidate.roi_conflict_side_vote_count,
                        "roi_conflict_side_max_confidence": round(
                            float(candidate.roi_conflict_side_max_confidence),
                            4,
                        ),
                        "roi_conflict_side_roi_x_avg": (
                            round(float(candidate.roi_conflict_side_roi_x_avg), 1)
                            if candidate.roi_conflict_side_roi_x_avg is not None
                            else None
                        ),
                        "roi_conflict_side_roi_x_limit": (
                            round(float(candidate.roi_conflict_side_roi_x_limit), 1)
                            if candidate.roi_conflict_side_roi_x_limit is not None
                            else None
                        ),
                        "threshold_rescue_rejected_reason": (
                            threshold_roi_conflict_reason
                            if reason == "threshold_rescue_roi_conflict"
                            else None
                        ),
                        "reason": reason,
                    }
                )
                continue

            confidence = min(max(candidate.max_confidence, 0.01), cap)
            rescue_votes.append(
                VoteResult(
                    class_id=candidate.class_id,
                    class_name=candidate.class_name,
                    vote_count=candidate.vote_count,
                    max_confidence=candidate.max_confidence,
                    avg_confidence=candidate.avg_confidence,
                    vote_ratio=0.0,
                    top_detected=candidate.top_detected,
                    side_detected=candidate.side_detected,
                    top_vote_count=candidate.top_vote_count,
                    side_vote_count=candidate.side_vote_count,
                    top_max_confidence=candidate.top_max_confidence,
                    side_max_confidence=candidate.side_max_confidence,
                    weighted_confidence=confidence,
                    source=candidate.source,
                    raw_vote_count=candidate.vote_count,
                    top_motion_passed=candidate.top_motion_passed,
                    side_motion_passed=candidate.side_motion_passed,
                    top_total_displacement=candidate.top_total_displacement,
                    side_total_displacement=candidate.side_total_displacement,
                    top_max_distance=candidate.top_max_distance,
                    side_max_distance=candidate.side_max_distance,
                    roi_x_min=candidate.roi_x_min,
                    roi_x_max=candidate.roi_x_max,
                    roi_x_avg=candidate.roi_x_avg,
                    roi_x_limit=candidate.roi_x_limit,
                    weight_residual_g=residual,
                    motion_gate_passed=motion_gate_passed,
                    weight_gate_passed=True,
                    rescue_tolerance_g=rescue_tolerance,
                    rescue_weight_residual_g=residual,
                )
            )
            diagnostic_candidates.append(
                {
                    "class_id": candidate.class_id,
                    "name": candidate.class_name,
                    "source": candidate.source,
                    "raw_vote_count": candidate.vote_count,
                    "max_confidence": round(candidate.max_confidence, 4),
                    "unit_weight_g": round(product_weight, 1),
                    "stock_qty": stock_qty,
                    "weight_residual_g": round(float(residual or 0.0), 1),
                    "rescue_weight_residual_g": round(float(residual or 0.0), 1),
                    "rescue_tolerance_g": rescue_tolerance,
                    "weight_gate_passed": True,
                    "motion_gate_passed": motion_gate_passed,
                    "side_motion_passed": candidate.side_motion_passed,
                    "roi_x_avg": (
                        round(float(candidate.roi_x_avg), 1)
                        if candidate.roi_x_avg is not None
                        else None
                    ),
                    "roi_x_limit": (
                        round(float(candidate.roi_x_limit), 1)
                        if candidate.roi_x_limit is not None
                        else None
                    ),
                    "roi_conflict": candidate.roi_conflict,
                    "roi_conflict_reason": candidate.roi_conflict_reason,
                    "roi_conflict_side_vote_count": candidate.roi_conflict_side_vote_count,
                    "roi_conflict_side_max_confidence": round(
                        float(candidate.roi_conflict_side_max_confidence),
                        4,
                    ),
                    "roi_conflict_side_roi_x_avg": (
                        round(float(candidate.roi_conflict_side_roi_x_avg), 1)
                        if candidate.roi_conflict_side_roi_x_avg is not None
                        else None
                    ),
                    "roi_conflict_side_roi_x_limit": (
                        round(float(candidate.roi_conflict_side_roi_x_limit), 1)
                        if candidate.roi_conflict_side_roi_x_limit is not None
                        else None
                    ),
                    "threshold_rescue_rejected_reason": None,
                    "reason": "accepted",
                }
            )

        if diagnostics is not None:
            diagnostics.update(
                {
                    "target_weight": round(target_weight, 1),
                    "tolerance": rescue_tolerance,
                    "rescue_tolerance_g": rescue_tolerance,
                    "no_motion_rescue_tolerance_g": no_motion_tolerance,
                    "considered": len(rescue_candidates),
                    "accepted": len(rescue_votes),
                    "rejections": rejections,
                    "candidates": diagnostic_candidates,
                }
            )
        return rescue_votes

    def _record_preprocess(
        self,
        trace_context: Optional[TriggerTraceContext],
        camera_type: str,
    ) -> None:
        if trace_context is None:
            return
        preprocess = getattr(self.yolo, "last_preprocess", None)
        if preprocess:
            trace_context.record_preprocess(camera_type, preprocess)

    def _record_diagnostic_detections(
        self,
        trace_context: Optional[TriggerTraceContext],
        frame,
        camera_type: str,
        frame_index: int,
    ) -> None:
        if trace_context is None or not config.vision.diagnostic_all_class_trace:
            return
        if frame_index >= max(0, int(config.vision.diagnostic_trace_max_frames)):
            return
        for det in self._detect_frame(frame, None, camera_type):
            if det.is_hand:
                continue
            trace_context.record_diagnostic_detection(
                camera=camera_type,
                frame_index=frame_index,
                class_id=det.cls,
                class_name=det.name,
                confidence=det.conf,
            )

    def _limit_candidates(
        self,
        results: List[VoteResult],
        log_prefix: str,
    ) -> List[VoteResult]:
        candidate_limit = max(1, int(config.vision.top_k))
        limited = results[:candidate_limit]
        logger.info(
            f"[{log_prefix}] candidate_limit={candidate_limit}, "
            f"before_limit={len(results)}, after_limit={len(limited)}"
        )
        return limited

    def _log_candidate_trace(
        self,
        log_prefix: str,
        stats: VideoProcessingStats,
        results: List[VoteResult],
    ) -> None:
        """Write detailed candidate pipeline diagnostics to the file log."""
        candidate_limit = max(1, int(config.vision.top_k))
        logger.info(
            f"[{log_prefix}][CANDIDATE-TRACE] "
            f"raw_top={stats.top_raw_detections}, raw_side={stats.side_raw_detections}, "
            f"threshold_filtered_top={stats.top_threshold_filtered}, "
            f"threshold_filtered_side={stats.side_threshold_filtered}, "
            f"roi_filtered={stats.roi_filtered_detections}, "
            f"motion_filtered={stats.motion_filtered_classes}, "
            f"hand_path_filtered={stats.hand_path_filtered_classes}, "
            f"final_candidates={len(results)}, candidate_limit={candidate_limit}"
        )
        for index, result in enumerate(results[:candidate_limit], start=1):
            logger.info(
                f"[{log_prefix}][CANDIDATE-TRACE] rank={index}, "
                f"class_id={result.class_id}, name={result.class_name}, "
                f"confidence={result.weighted_confidence:.3f}, "
                f"top={result.top_detected}, side={result.side_detected}, "
                f"votes={result.vote_count}"
            )

    def _inference_allowed_class_ids(
        self,
        allowed_class_ids: Optional[List[int]],
        log_prefix: str,
    ) -> Optional[List[int]]:
        if allowed_class_ids is None:
            logger.warning(
                f"[{log_prefix}] active_products snapshot missing; "
                "inference_classes=none fail_closed=true"
            )
            return []

        normalized_ids = list(dict.fromkeys(allowed_class_ids))
        if normalized_ids:
            logger.info(
                f"[{log_prefix}] strict_active_products allowed_classes={len(normalized_ids)} "
                "inference_classes=allowed"
            )
        else:
            logger.warning(
                f"[{log_prefix}] active_products has no stock-positive classes; "
                "inference_classes=none fail_closed=true"
            )
        return normalized_ids

    @staticmethod
    def _filter_results_by_allowed_class_ids(
        results: List[VoteResult],
        allowed_class_ids: Optional[List[int]],
        log_prefix: str,
    ) -> List[VoteResult]:
        if allowed_class_ids is None:
            return results

        allowed_set = set(allowed_class_ids)
        filtered = [result for result in results if result.class_id in allowed_set]
        removed_count = len(results) - len(filtered)
        if removed_count > 0:
            logger.warning(
                f"[{log_prefix}] active_product_filtered={removed_count} "
                f"allowed_classes={len(allowed_set)}"
            )
        return filtered

    def _apply_hand_path_filter(
        self,
        results: List[VoteResult],
        hand_path_tracker: HandPathTracker,
        log_prefix: str,
    ) -> Tuple[List[VoteResult], int]:
        candidate_class_ids = [r.class_id for r in results]
        valid_class_ids = hand_path_tracker.filter_products_by_path(candidate_class_ids)
        valid_class_ids_set = set(valid_class_ids)

        filtered_results = [r for r in results if r.class_id in valid_class_ids_set]
        removed_count = len(results) - len(filtered_results)
        if removed_count > 0:
            logger.info(f"[{log_prefix}] hand_path_filtered={removed_count}")
        if results and not filtered_results:
            logger.warning(
                f"[{log_prefix}] fallback=kept_candidates "
                "reason=hand_path_removed_all"
            )
            return results, 0
        return filtered_results, removed_count

    def _apply_hand_path_filter_with_trace(
        self,
        results: List[VoteResult],
        hand_path_tracker: HandPathTracker,
        trace_context: Optional[TriggerTraceContext],
        stats: VideoProcessingStats,
        log_prefix: str,
    ) -> List[VoteResult]:
        candidate_class_ids = [int(result.class_id) for result in results]
        hand_path_valid = hand_path_tracker.has_valid_hand_path()
        metrics_by_class = hand_path_tracker.hand_interaction_metrics(
            candidate_class_ids
        )
        valid_class_ids_set = {
            int(class_id)
            for class_id, metrics in metrics_by_class.items()
            if bool(metrics.get("handInteractionPassed"))
        }
        has_hand_near_candidate = bool(hand_path_valid and valid_class_ids_set)

        for result in results:
            metrics = metrics_by_class.get(
                int(result.class_id),
                {
                    "handPathValid": hand_path_valid,
                    "handPathValidUpperRoi": hand_path_valid,
                    "handInteractionPassed": False,
                    "handNearFrameCount": 0,
                    "handNearVoteRatio": 0.0,
                    "minHandDistancePx": None,
                },
            )
            hand_path_passed = bool(
                hand_path_valid and metrics.get("handInteractionPassed")
            )
            hand_path_blocked = bool(
                has_hand_near_candidate and not hand_path_passed
            )
            self._record_hand_path_evidence(
                trace_context,
                result=result,
                hand_path_valid=hand_path_valid,
                hand_path_passed=hand_path_passed,
                hand_path_blocked=hand_path_blocked,
                hand_metrics=metrics,
            )

        before_count = len(results)
        if not has_hand_near_candidate:
            stats.hand_path_filtered_classes = 0
            if hand_path_valid and before_count > 0:
                logger.warning(
                    "[%s] fallback=kept_candidates reason=hand_path_no_near_candidate",
                    log_prefix,
                )
            return results

        filtered_results = [
            result for result in results if int(result.class_id) in valid_class_ids_set
        ]
        removed_results = [
            result
            for result in results
            if int(result.class_id) not in valid_class_ids_set
        ]
        if not filtered_results:
            stats.hand_path_filtered_classes = 0
            logger.warning(
                "[%s] fallback=kept_candidates reason=hand_path_removed_all",
                log_prefix,
            )
            return results

        stats.hand_path_filtered_classes = before_count - len(filtered_results)
        for result in filtered_results:
            self._record_stage(
                trace_context,
                class_id=result.class_id,
                class_name=result.class_name,
                stage="hand_path_passed",
                camera="top",
            )
        for result in removed_results:
            self._record_stage(
                trace_context,
                class_id=result.class_id,
                class_name=result.class_name,
                stage="hand_path_filtered",
                camera="top",
            )
        if stats.hand_path_filtered_classes > 0:
            logger.info(
                "[%s] hand_path_filtered=%s",
                log_prefix,
                stats.hand_path_filtered_classes,
            )
        return filtered_results

    def process_videos(
        self,
        top_path: Optional[str] = None,
        side_path: Optional[str] = None,
        allowed_class_ids: Optional[List[int]] = None,
        product_weights: Optional[Dict[int, float]] = None,
        trace_context: Optional[TriggerTraceContext] = None,
        delta_weight: Optional[float] = None,
    ) -> VideoProcessingResult:
        """
        Process top and side camera videos.

        Args:
            top_path: Path to top camera AVI file (optional)
            side_path: Path to side camera AVI file (optional)
            allowed_class_ids: 허용된 YOLO 클래스 ID 리스트 (v4.4)
                               None이면 모든 클래스 탐지
                               리스트가 있으면 해당 클래스만 탐지
            product_weights: {class_id: weight_in_grams} for logging (v4.6)

        Returns:
            VideoProcessingResult with combined voting results
        """
        start_time = time.time()
        stats = VideoProcessingStats()

        logger.info("[VIDEO] ========== 비디오 처리 시작 ==========")
        logger.info(f"[VIDEO] top_path={top_path}")
        logger.info(f"[VIDEO] side_path={side_path}")
        logger.info(
            f"[VIDEO] thresholds: top={self.top_confidence_threshold:.2f}, "
            f"side={self.side_confidence_threshold:.2f}"
        )
        inference_allowed_class_ids = self._inference_allowed_class_ids(
            allowed_class_ids,
            "VIDEO",
        )

        top_ensemble = VotingEnsemble(min_vote_ratio=self.min_vote_ratio)
        side_ensemble = VotingEnsemble(min_vote_ratio=self.min_vote_ratio)
        low_confidence_stats: Dict[int, _LowConfidenceClassStats] = {}
        roi_filtered_stats: Dict[int, _RoiFilteredClassStats] = {}

        # v4.6: 손 경로 추적기 생성 (Top 카메라에서만 사용)
        top_hand_tracker: Optional[HandPathTracker] = None
        if self.hand_path_filter_enabled:
            top_hand_tracker = self._new_hand_path_tracker()

        # Process top camera video
        if top_path:
            logger.info("[VIDEO] Top 카메라 처리 시작...")
            top_stats = self._process_single_video(
                top_path, top_ensemble, "top", inference_allowed_class_ids,
                hand_path_tracker=top_hand_tracker,
                trace_context=trace_context,
                low_confidence_stats=low_confidence_stats,
                roi_filtered_stats=roi_filtered_stats,
                delta_weight=delta_weight,
            )
            stats.top_frames = top_stats["frames"]
            stats.top_raw_detections = top_stats.get("raw_detections", 0)
            stats.top_threshold_filtered = top_stats.get("threshold_filtered", 0)
            stats.top_detections = top_stats["detections"]
            stats.yolo_inference_count += top_stats.get("yolo_inference_count", 0)
            stats.yolo_total_time_ms += top_stats.get("yolo_total_time_ms", 0.0)
            stats.roi_filtered_detections += top_stats.get("roi_filtered", 0)
            stats.side_roi_soft_passed_detections += top_stats.get(
                "side_roi_soft_passed",
                0,
            )
            stats.side_roi_soft_filtered_detections += top_stats.get(
                "side_roi_soft_filtered",
                0,
            )
            stats.motion_filtered_classes += top_stats.get("motion_filtered", 0)
            logger.info(
                f"[VIDEO] Top 완료: 총 {stats.top_frames}프레임, "
                f"탐지={stats.top_detections}개, 고유클래스={len(top_ensemble.votes)}개"
            )

        # Process side camera video
        if side_path:
            logger.info("[VIDEO] Side 카메라 처리 시작...")
            side_stats = self._process_single_video(
                side_path, side_ensemble, "side", inference_allowed_class_ids,
                hand_path_tracker=(
                    top_hand_tracker
                    if self._uses_freezer_dual_top_profile("side")
                    else None
                ),
                trace_context=trace_context,
                low_confidence_stats=low_confidence_stats,
                roi_filtered_stats=roi_filtered_stats,
                delta_weight=delta_weight,
            )
            stats.side_frames = side_stats["frames"]
            stats.side_raw_detections = side_stats.get("raw_detections", 0)
            stats.side_threshold_filtered = side_stats.get("threshold_filtered", 0)
            stats.side_detections = side_stats["detections"]
            stats.yolo_inference_count += side_stats.get("yolo_inference_count", 0)
            stats.yolo_total_time_ms += side_stats.get("yolo_total_time_ms", 0.0)
            stats.roi_filtered_detections += side_stats.get("roi_filtered", 0)
            stats.side_roi_soft_passed_detections += side_stats.get(
                "side_roi_soft_passed",
                0,
            )
            stats.side_roi_soft_filtered_detections += side_stats.get(
                "side_roi_soft_filtered",
                0,
            )
            stats.motion_filtered_classes += side_stats.get("motion_filtered", 0)
            logger.info(
                f"[VIDEO] Side 완료: 총 {stats.side_frames}프레임, "
                f"탐지={stats.side_detections}개, 고유클래스={len(side_ensemble.votes)}개"
            )

        # Combine results with config weights (v4.6: product_weights 전달)
        combined_results = VotingEnsemble.combine(
            top_ensemble=top_ensemble,
            side_ensemble=side_ensemble,
            top_weight=config.top_weight,
            side_weight=config.side_weight,
            common_class_bonus=config.common_class_bonus,
            product_weights=product_weights,
            top_only_weight=config.top_only_weight,
            side_only_weight=config.side_only_weight,
        )
        combined_results = self._filter_results_by_allowed_class_ids(
            combined_results,
            inference_allowed_class_ids,
            "VIDEO",
        )

        # v4.6: 손 경로 필터링 적용 (Top 카메라 기준)
        if top_hand_tracker is not None and self.hand_path_filter_enabled:
            combined_results = self._apply_hand_path_filter_with_trace(
                combined_results,
                top_hand_tracker,
                trace_context,
                stats,
                "VIDEO",
            )

        # Deprecated average-center hand filter path kept unreachable for context.
        if False and top_hand_tracker is not None and self.hand_path_filter_enabled:
            candidate_class_ids = [r.class_id for r in combined_results]
            hand_path_valid = top_hand_tracker.has_valid_hand_path()
            valid_class_ids = top_hand_tracker.filter_products_by_path(candidate_class_ids)
            valid_class_ids_set = set(valid_class_ids)

            before_count = len(combined_results)
            filtered_by_hand_path = [
                r for r in combined_results if r.class_id in valid_class_ids_set
            ]
            for result in combined_results:
                hand_path_passed = bool(
                    hand_path_valid and result.class_id in valid_class_ids_set
                )
                hand_path_blocked = bool(
                    hand_path_valid and result.class_id not in valid_class_ids_set
                )
                self._record_hand_path_evidence(
                    trace_context,
                    result=result,
                    hand_path_valid=hand_path_valid,
                    hand_path_passed=hand_path_passed,
                    hand_path_blocked=hand_path_blocked,
                )
            if before_count > 0 and not filtered_by_hand_path:
                logger.warning(
                    "[VIDEO] fallback=kept_candidates reason=hand_path_removed_all"
                )
                stats.hand_path_filtered_classes = 0
            else:
                removed_results = [
                    result
                    for result in combined_results
                    if result.class_id not in valid_class_ids_set
                ]
                combined_results = filtered_by_hand_path
                stats.hand_path_filtered_classes = before_count - len(combined_results)
                if hand_path_valid:
                    for result in filtered_by_hand_path:
                        self._record_stage(
                            trace_context,
                            class_id=result.class_id,
                            class_name=result.class_name,
                            stage="hand_path_passed",
                            camera="top",
                        )
                    for result in removed_results:
                        self._record_stage(
                            trace_context,
                            class_id=result.class_id,
                            class_name=result.class_name,
                            stage="hand_path_filtered",
                            camera="top",
                        )

            if stats.hand_path_filtered_classes > 0:
                logger.info(
                    f"[VIDEO] 손 경로 필터링: {stats.hand_path_filtered_classes}개 제외"
                )

        # Filter by minimum vote ratio OR minimum vote count
        # 조건 1: vote_ratio >= 5% (기존)
        # 조건 2: vote_count >= min_vote_count (짧은 비디오 대응)
        min_vote_count = self._effective_min_vote_count()
        min_vote_ratio = self._effective_min_vote_ratio()
        filtered_results = [
            r for r in combined_results
            if r.vote_ratio >= min_vote_ratio or r.vote_count >= min_vote_count
        ]
        filtered_results = self._limit_candidates(filtered_results, "VIDEO")
        threshold_rescue_candidates = self._build_threshold_rescue_candidates(
            low_confidence_stats,
            roi_filtered_stats,
            inference_allowed_class_ids,
            "VIDEO",
        )
        roi_rescue_candidates = self._build_roi_rescue_candidates(
            roi_filtered_stats,
            inference_allowed_class_ids,
            "VIDEO",
        )
        if trace_context is not None:
            for rank, result in enumerate(filtered_results, start=1):
                trace_context.record_final_candidate_rank(result.class_id, rank)
            trace_context.record_threshold_rescue_candidates(
                threshold_rescue_candidates,
                product_weights,
            )
            trace_context.record_roi_rescue_candidates(
                roi_rescue_candidates,
                product_weights,
            )

        # 필터링 로그
        filtered_by_ratio = sum(1 for r in combined_results if r.vote_ratio >= min_vote_ratio)
        filtered_by_count = sum(1 for r in combined_results if r.vote_count >= min_vote_count and r.vote_ratio < min_vote_ratio)
        logger.info(
            f"[VIDEO] 필터링: vote_ratio >= {min_vote_ratio*100:.0f}%: {filtered_by_ratio}개, "
            f"vote_count >= {min_vote_count}: 추가 {filtered_by_count}개"
        )

        stats.processing_time_ms = (time.time() - start_time) * 1000
        if stats.yolo_inference_count > 0:
            stats.yolo_avg_time_ms = (
                stats.yolo_total_time_ms / stats.yolo_inference_count
            )

        logger.info(
            f"[VIDEO][LATENCY] total_ms={stats.processing_time_ms:.1f} "
            f"yolo_total_ms={stats.yolo_total_time_ms:.1f} "
            f"yolo_avg_ms={stats.yolo_avg_time_ms:.1f} "
            f"yolo_count={stats.yolo_inference_count}"
        )
        self._log_candidate_trace("VIDEO", stats, filtered_results)
        if trace_context is not None:
            trace_context.record_video_stats(stats)
            trace_context.record_candidates(filtered_results, product_weights)

        logger.info(f"[VIDEO] 앙상블 결합 완료: {len(filtered_results)}개 후보")

        return VideoProcessingResult(
            vote_results=filtered_results,
            top_ensemble=top_ensemble,
            side_ensemble=side_ensemble,
            stats=stats,
            threshold_rescue_candidates=threshold_rescue_candidates,
            roi_rescue_candidates=roi_rescue_candidates,
        )

    async def process_videos_async(
        self,
        top_path: Optional[str] = None,
        side_path: Optional[str] = None,
        allowed_class_ids: Optional[List[int]] = None,
        product_weights: Optional[Dict[int, float]] = None,
        trace_context: Optional[TriggerTraceContext] = None,
        delta_weight: Optional[float] = None,
    ) -> VideoProcessingResult:
        """
        Async streaming video processing (v5.3).

        Top과 Side 카메라의 프레임 추출을 병렬로 수행하고,
        단일 YOLO 인스턴스에서 인터리빙 추론합니다.

        I/O 병렬화로 처리 시간 20-30% 개선 예상:
        - 현재: 12-20초/트리거
        - 목표: 8-14초/트리거

        Args:
            top_path: Path to top camera AVI file (optional)
            side_path: Path to side camera AVI file (optional)
            allowed_class_ids: 허용된 YOLO 클래스 ID 리스트
            product_weights: {class_id: weight_in_grams} for logging

        Returns:
            VideoProcessingResult with combined voting results
        """
        start_time = time.time()
        stats = VideoProcessingStats()

        logger.info("[VIDEO-ASYNC] ========== 비동기 스트리밍 처리 시작 ==========")
        logger.info(f"[VIDEO-ASYNC] top_path={top_path}")
        logger.info(f"[VIDEO-ASYNC] side_path={side_path}")
        logger.info(
            f"[VIDEO-ASYNC] thresholds: top={self.top_confidence_threshold:.2f}, "
            f"side={self.side_confidence_threshold:.2f}"
        )
        inference_allowed_class_ids = self._inference_allowed_class_ids(
            allowed_class_ids,
            "VIDEO-ASYNC",
        )
        frame_stride = 2

        top_ensemble = VotingEnsemble(min_vote_ratio=self.min_vote_ratio)
        side_ensemble = VotingEnsemble(min_vote_ratio=self.min_vote_ratio)

        # v5.3: 손 경로 추적기 (Top 카메라에서만 사용)
        top_hand_tracker: Optional[HandPathTracker] = None
        if self.hand_path_filter_enabled:
            top_hand_tracker = self._new_hand_path_tracker()

        # 프레임 큐: (camera_type, frame_idx, frame, extractor_done)
        # None frame = EOF marker
        frame_queue: asyncio.Queue[Tuple[str, int, Optional[Any]]] = asyncio.Queue(
            maxsize=config.async_streaming.frame_queue_size
        )

        # Motion tracking
        top_bbox_trackers: Dict[int, BboxTracker] = {}
        side_bbox_trackers: Dict[int, BboxTracker] = {}
        top_pending_votes: Dict[int, List[Tuple[float, str, int]]] = {}
        side_pending_votes: Dict[int, List[Tuple[float, str, int]]] = {}
        low_confidence_stats: Dict[int, _LowConfidenceClassStats] = {}
        roi_filtered_stats: Dict[int, _RoiFilteredClassStats] = {}

        # Frame counters
        top_frame_count = 0
        side_frame_count = 0
        top_original_frame_count = 0
        side_original_frame_count = 0
        top_raw_detection_count = 0
        side_raw_detection_count = 0
        top_threshold_filtered_count = 0
        side_threshold_filtered_count = 0
        top_detection_count = 0
        side_detection_count = 0
        roi_filtered_count = 0
        side_roi_soft_passed_count = 0
        side_roi_soft_filtered_count = 0
        extractor_diagnostics: Dict[str, object] = {}

        # Active extractors count
        active_extractors = 0
        if top_path:
            active_extractors += 1
        if side_path:
            active_extractors += 1

        if active_extractors == 0:
            logger.warning("[VIDEO-ASYNC] No video paths provided")
            return VideoProcessingResult(
                vote_results=[],
                top_ensemble=top_ensemble,
                side_ensemble=side_ensemble,
                stats=stats,
            )

        async def extract_frames(path: str, camera_type: str) -> None:
            """프레임 추출 태스크 (비동기)."""
            nonlocal top_frame_count, side_frame_count
            nonlocal top_original_frame_count, side_original_frame_count

            extractor = create_frame_extractor(
                path,
                prefer_ffmpeg=True,
                use_hwaccel=self.use_hwaccel,
                camera_type=camera_type,
            )
            if trace_context is not None:
                trace_context.plan_camera(
                    camera_type,
                    int(getattr(extractor, "total_frames", 0) or 0),
                )

            frame_idx = 0
            # ffmpeg 미존재 시 CV2FrameExtractor가 반환될 수 있음(__aiter__ 미지원)
            if not hasattr(extractor, '__aiter__'):
                message = (
                    f"[VIDEO-ASYNC] {camera_type}: async streaming requires ffmpeg "
                    "but extractor does not support async iteration (ffmpeg not available?). "
                    "Video extraction cannot continue safely."
                )
                logger.error(message)
                raise VideoProcessingError(message)

            try:
                async for frame in extractor:
                    original_frame_idx = frame_idx
                    frame_idx += 1

                    if camera_type == "top":
                        top_original_frame_count = frame_idx
                    else:
                        side_original_frame_count = frame_idx

                    if original_frame_idx % frame_stride != 0:
                        continue

                    if trace_context is not None:
                        trace_context.record_frame(camera_type, original_frame_idx, frame)
                    await frame_queue.put((camera_type, original_frame_idx, frame))

                    # Update frame count
                    if camera_type == "top":
                        top_frame_count += 1
                    else:
                        side_frame_count += 1

                zero_frame_reason = _async_zero_frame_failure_reason(extractor)
                if zero_frame_reason is not None:
                    raise VideoProcessingError(
                        f"{camera_type} video extraction failed: {zero_frame_reason}",
                        video_path=path,
                    )

            except asyncio.CancelledError:
                logger.warning(f"[VIDEO-ASYNC] {camera_type} extraction cancelled at frame {frame_idx}")
                raise
            finally:
                extractor_diagnostics[camera_type] = getattr(
                    extractor,
                    "last_diagnostics",
                    None,
                )
                # EOF marker
                await frame_queue.put((camera_type, -1, None))
                logger.info(f"[VIDEO-ASYNC] {camera_type} 추출 완료: {frame_idx}개 프레임")

        async def yolo_inference_loop() -> None:
            """YOLO 추론 루프 (단일 인스턴스)."""
            nonlocal top_raw_detection_count, side_raw_detection_count
            nonlocal top_threshold_filtered_count, side_threshold_filtered_count
            nonlocal top_detection_count, side_detection_count, roi_filtered_count
            nonlocal side_roi_soft_passed_count, side_roi_soft_filtered_count

            eof_received = 0
            expected_eofs = active_extractors

            while eof_received < expected_eofs:
                try:
                    camera_type, frame_idx, frame = await asyncio.wait_for(
                        frame_queue.get(),
                        timeout=60.0  # 60초 타임아웃
                    )
                except asyncio.TimeoutError:
                    logger.error("[VIDEO-ASYNC] Frame queue timeout")
                    raise VideoProcessingError(
                        "Async video frame queue timeout before all extractors finished"
                    )

                # EOF marker
                if frame is None:
                    eof_received += 1
                    logger.debug(f"[VIDEO-ASYNC] EOF received from {camera_type} ({eof_received}/{expected_eofs})")
                    continue

                # YOLO 추론 (to_thread로 CPU 양보)
                self._record_diagnostic_detections(
                    trace_context,
                    frame,
                    camera_type,
                    frame_idx,
                )
                yolo_started = time.perf_counter()
                detections = await asyncio.to_thread(
                    self._detect_frame, frame, inference_allowed_class_ids, camera_type
                )
                stats.yolo_total_time_ms += (
                    time.perf_counter() - yolo_started
                ) * 1000
                stats.yolo_inference_count += 1
                self._record_preprocess(trace_context, camera_type)

                if top_hand_tracker is not None and self._uses_freezer_dual_top_profile(
                    camera_type
                ):
                    top_hand_tracker.update_frame(detections, frame_idx)

                # 카메라별 처리
                if camera_type == "top":
                    # 손 경로 추적 업데이트
                    if (
                        top_hand_tracker is not None
                        and not self._uses_freezer_dual_top_profile("top")
                    ):
                        top_hand_tracker.update_frame(detections, frame_idx)

                    for det in detections:
                        if det.is_hand:
                            continue
                        top_raw_detection_count += 1
                        self._record_stage(
                            trace_context,
                            class_id=det.cls,
                            class_name=det.name,
                            stage="raw",
                            camera="top",
                            confidence=det.conf,
                        )
                        if det.conf < self._threshold_for_camera("top"):
                            top_threshold_filtered_count += 1
                            bbox_size = max(det.x2 - det.x1, det.y2 - det.y1)
                            roi_eligible = self._low_confidence_roi_eligible(
                                det,
                                "top",
                                delta_weight,
                            )
                            self._record_low_confidence_detection(
                                low_confidence_stats,
                                class_id=det.cls,
                                class_name=det.name,
                                confidence=det.conf,
                                camera="top",
                                center=det.center,
                                frame_idx=frame_idx,
                                bbox_size=bbox_size,
                                roi_eligible=roi_eligible,
                            )
                            self._record_stage(
                                trace_context,
                                class_id=det.cls,
                                class_name=det.name,
                                stage="threshold_filtered",
                                camera="top",
                                confidence=det.conf,
                            )
                            continue
                        self._record_stage(
                            trace_context,
                            class_id=det.cls,
                            class_name=det.name,
                            stage="threshold_passed",
                            camera="top",
                            confidence=det.conf,
                        )
                        freezer_dual_top = self._uses_freezer_dual_top_profile("top")
                        if freezer_dual_top:
                            top_roi_passed = self._freezer_roi_accepts(det)
                            top_roi_direction = self._freezer_roi_direction()
                            top_roi_y_limit = self._freezer_roi_y_split()
                        else:
                            top_roi_passed, top_roi_direction = self._top_roi_accepts(
                                det,
                                delta_weight,
                            )
                            top_roi_y_limit = self.top_roi_y_split
                        if not top_roi_passed:
                            roi_filtered_count += 1
                            self._record_stage(
                                trace_context,
                                class_id=det.cls,
                                class_name=det.name,
                                stage=(
                                    "freezer_roi_filtered"
                                    if freezer_dual_top
                                    else "roi_filtered"
                                ),
                                camera="top",
                                confidence=det.conf,
                                center=det.center,
                                roi_y_limit=top_roi_y_limit,
                                roi_direction=top_roi_direction,
                            )
                            continue
                        self._record_stage(
                            trace_context,
                            class_id=det.cls,
                            class_name=det.name,
                            stage=(
                                "freezer_roi_passed"
                                if freezer_dual_top
                                else "roi_passed"
                            ),
                            camera="top",
                            confidence=det.conf if freezer_dual_top else None,
                            center=det.center,
                            roi_y_limit=top_roi_y_limit if freezer_dual_top else None,
                            roi_direction=top_roi_direction if freezer_dual_top else None,
                        )

                        class_id = det.cls
                        center = det.center

                        # 동적 임계값 계산
                        bbox_size = max(det.x2 - det.x1, det.y2 - det.y1)
                        dynamic_threshold = self._motion_threshold_for_detection(
                            "top",
                            bbox_size,
                        )

                        if class_id not in top_bbox_trackers:
                            top_bbox_trackers[class_id] = BboxTracker()
                        top_bbox_trackers[class_id].update(center, frame_idx)
                        top_bbox_trackers[class_id].dynamic_threshold = max(
                            top_bbox_trackers[class_id].dynamic_threshold,
                            dynamic_threshold
                        )

                        if class_id not in top_pending_votes:
                            top_pending_votes[class_id] = []
                        top_pending_votes[class_id].append((det.conf, det.name, frame_idx))
                        top_detection_count += 1

                else:  # side
                    for det in detections:
                        if det.is_hand:
                            continue
                        side_raw_detection_count += 1
                        self._record_stage(
                            trace_context,
                            class_id=det.cls,
                            class_name=det.name,
                            stage="raw",
                            camera="side",
                            confidence=det.conf,
                        )
                        if det.conf < self._threshold_for_camera("side"):
                            side_threshold_filtered_count += 1
                            bbox_size = max(det.x2 - det.x1, det.y2 - det.y1)
                            self._record_low_confidence_detection(
                                low_confidence_stats,
                                class_id=det.cls,
                                class_name=det.name,
                                confidence=det.conf,
                                camera="side",
                                center=det.center,
                                frame_idx=frame_idx,
                                bbox_size=bbox_size,
                                roi_eligible=self._low_confidence_roi_eligible(
                                    det,
                                    "side",
                                    delta_weight,
                                ),
                            )
                            self._record_stage(
                                trace_context,
                                class_id=det.cls,
                                class_name=det.name,
                                stage="threshold_filtered",
                                camera="side",
                                confidence=det.conf,
                            )
                            continue
                        self._record_stage(
                            trace_context,
                            class_id=det.cls,
                            class_name=det.name,
                            stage="threshold_passed",
                            camera="side",
                            confidence=det.conf,
                        )

                        # Side ROI 필터
                        center_x = det.center[0]
                        freezer_dual_top_side = self._uses_freezer_dual_top_profile("side")
                        side_roi_soft_passed = False
                        if freezer_dual_top_side:
                            side_roi_passed = self._freezer_roi_accepts(det)
                        else:
                            side_roi_passed, side_roi_soft_passed = self._side_roi_accepts(
                                center_x
                            )
                        if not side_roi_passed:
                            roi_filtered_count += 1
                            if not freezer_dual_top_side and center_x > self._side_roi_soft_limit():
                                side_roi_soft_filtered_count += 1
                            bbox_size = max(det.x2 - det.x1, det.y2 - det.y1)
                            if not freezer_dual_top_side:
                                self._record_roi_filtered_detection(
                                    roi_filtered_stats,
                                    class_id=det.cls,
                                    class_name=det.name,
                                    confidence=det.conf,
                                    center=det.center,
                                    frame_idx=frame_idx,
                                    bbox_size=bbox_size,
                                    roi_x_limit=self.side_roi_x_max,
                                )
                            self._record_stage(
                                trace_context,
                                class_id=det.cls,
                                class_name=det.name,
                                stage=(
                                    "freezer_roi_filtered"
                                    if freezer_dual_top_side
                                    else "roi_filtered"
                                ),
                                camera="side",
                                confidence=det.conf,
                                center=det.center,
                                roi_x_limit=None if freezer_dual_top_side else self.side_roi_x_max,
                                roi_y_limit=(
                                    self._freezer_roi_y_split()
                                    if freezer_dual_top_side
                                    else None
                                ),
                                roi_direction=(
                                    self._freezer_roi_direction()
                                    if freezer_dual_top_side
                                    else None
                                ),
                            )
                            if not freezer_dual_top_side and center_x > self._side_roi_soft_limit():
                                self._record_stage(
                                    trace_context,
                                    class_id=det.cls,
                                    class_name=det.name,
                                    stage="soft_margin_filtered",
                                    camera="side",
                                    confidence=det.conf,
                                )
                            continue
                        if side_roi_soft_passed:
                            side_roi_soft_passed_count += 1
                            self._record_stage(
                                trace_context,
                                class_id=det.cls,
                                class_name=det.name,
                                stage="side_roi_soft_passed",
                                camera="side",
                                confidence=det.conf,
                                center=det.center,
                                roi_x_limit=self._side_roi_soft_limit(),
                            )
                        self._record_stage(
                            trace_context,
                            class_id=det.cls,
                            class_name=det.name,
                            stage=(
                                "freezer_roi_passed"
                                if freezer_dual_top_side
                                else "roi_passed"
                            ),
                            camera="side",
                            confidence=det.conf if freezer_dual_top_side else None,
                            center=det.center,
                            roi_y_limit=(
                                self._freezer_roi_y_split()
                                if freezer_dual_top_side
                                else None
                            ),
                            roi_direction=(
                                self._freezer_roi_direction()
                                if freezer_dual_top_side
                                else None
                            ),
                        )

                        class_id = det.cls
                        center = det.center

                        bbox_size = max(det.x2 - det.x1, det.y2 - det.y1)
                        dynamic_threshold = self._motion_threshold_for_detection(
                            "side",
                            bbox_size,
                        )

                        if class_id not in side_bbox_trackers:
                            side_bbox_trackers[class_id] = BboxTracker()
                        side_bbox_trackers[class_id].update(center, frame_idx)
                        side_bbox_trackers[class_id].dynamic_threshold = max(
                            side_bbox_trackers[class_id].dynamic_threshold,
                            dynamic_threshold
                        )

                        if class_id not in side_pending_votes:
                            side_pending_votes[class_id] = []
                        side_pending_votes[class_id].append((det.conf, det.name, frame_idx))
                        side_detection_count += 1

                # 진행 로그 (50프레임마다)
                total_frames = top_frame_count + side_frame_count
                if total_frames > 0 and total_frames % 50 == 0:
                    logger.info(
                        f"[VIDEO-ASYNC] 처리 중: top={top_frame_count}, side={side_frame_count}, "
                        f"탐지={top_detection_count + side_detection_count}"
                    )

        # Python 3.10 does not support asyncio.TaskGroup / except*.
        tasks: dict[asyncio.Task[None], str] = {}
        if top_path:
            tasks[asyncio.create_task(extract_frames(top_path, "top"))] = "top-extractor"
        if side_path:
            tasks[asyncio.create_task(extract_frames(side_path, "side"))] = "side-extractor"
        tasks[asyncio.create_task(yolo_inference_loop())] = "yolo-inference"

        done: set[asyncio.Task[None]] = set()
        pending: set[asyncio.Task[None]] = set(tasks)

        try:
            first_error: BaseException | None = None

            while pending and first_error is None:
                current_done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_EXCEPTION,
                )
                done.update(current_done)

                for task in current_done:
                    if task.cancelled():
                        continue

                    exc = task.exception()
                    if exc is not None:
                        first_error = exc
                        break

            if first_error is not None and pending:
                for task in pending:
                    task.cancel()

            if pending:
                current_done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.ALL_COMPLETED,
                )
                done.update(current_done)

        except asyncio.CancelledError:
            for task in pending:
                task.cancel()

            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
                done.update(pending)
                pending.clear()

            cancelled_count = sum(1 for task in done if task.cancelled())
            logger.warning(
                f"[VIDEO-ASYNC] Tasks cancelled: {cancelled_count} task(s), "
                f"processed frames: top={top_frame_count}, side={side_frame_count}"
            )
            raise
        else:
            cancelled_names: list[str] = []
            task_errors: list[tuple[str, BaseException]] = []

            for task, task_name in tasks.items():
                if task.cancelled():
                    cancelled_names.append(task_name)
                    continue

                exc = task.exception()
                if exc is not None:
                    task_errors.append((task_name, exc))

            if cancelled_names:
                logger.warning(
                    f"[VIDEO-ASYNC] Tasks cancelled: {', '.join(cancelled_names)}, "
                    f"processed frames: top={top_frame_count}, side={side_frame_count}"
                )

            for task_name, exc in task_errors:
                logger.error(f"[VIDEO-ASYNC] Task error in {task_name}: {type(exc).__name__}: {exc}")

            if task_errors:
                primary_task_name, primary_exc = task_errors[0]
                _raise_async_task_error(primary_task_name, primary_exc)

        # Frame counts 설정
        top_ensemble.set_frame_count(top_frame_count)
        side_ensemble.set_frame_count(side_frame_count)

        # Motion 필터링 및 투표 적용 (Top)
        top_motion_filtered = self._apply_motion_filter_and_votes(
            "top", top_pending_votes, top_bbox_trackers, top_ensemble, trace_context
        )

        # Motion 필터링 및 투표 적용 (Side)
        side_motion_filtered = self._apply_motion_filter_and_votes(
            "side", side_pending_votes, side_bbox_trackers, side_ensemble, trace_context
        )

        stats.top_frames = top_frame_count
        stats.side_frames = side_frame_count
        stats.top_original_frames = top_original_frame_count
        stats.side_original_frames = side_original_frame_count
        stats.frame_stride = frame_stride
        stats.top_raw_detections = top_raw_detection_count
        stats.side_raw_detections = side_raw_detection_count
        stats.top_threshold_filtered = top_threshold_filtered_count
        stats.side_threshold_filtered = side_threshold_filtered_count
        stats.top_detections = top_detection_count
        stats.side_detections = side_detection_count
        stats.roi_filtered_detections = roi_filtered_count
        stats.side_roi_soft_passed_detections = side_roi_soft_passed_count
        stats.side_roi_soft_filtered_detections = side_roi_soft_filtered_count
        stats.motion_filtered_classes = top_motion_filtered + side_motion_filtered

        # Side ROI 필터링 로그
        if roi_filtered_count > 0:
            logger.info(
                f"[VIDEO-ASYNC] ROI 필터링: {roi_filtered_count}개 탐지 제외 "
                f"(center_x > {self.side_roi_x_max}px, "
                f"soft_limit={self._side_roi_soft_limit()}px)"
            )

        # 앙상블 결합
        combined_results = VotingEnsemble.combine(
            top_ensemble=top_ensemble,
            side_ensemble=side_ensemble,
            top_weight=config.top_weight,
            side_weight=config.side_weight,
            common_class_bonus=config.common_class_bonus,
            product_weights=product_weights,
            top_only_weight=config.top_only_weight,
            side_only_weight=config.side_only_weight,
        )
        combined_results = self._filter_results_by_allowed_class_ids(
            combined_results,
            inference_allowed_class_ids,
            "VIDEO-ASYNC",
        )

        # 손 경로 필터링
        if top_hand_tracker is not None and self.hand_path_filter_enabled:
            combined_results = self._apply_hand_path_filter_with_trace(
                combined_results,
                top_hand_tracker,
                trace_context,
                stats,
                "VIDEO-ASYNC",
            )

        if False and top_hand_tracker is not None and self.hand_path_filter_enabled:
            candidate_class_ids = [r.class_id for r in combined_results]
            hand_path_valid = top_hand_tracker.has_valid_hand_path()
            valid_class_ids = top_hand_tracker.filter_products_by_path(candidate_class_ids)
            valid_class_ids_set = set(valid_class_ids)

            before_count = len(combined_results)
            filtered_by_hand_path = [
                r for r in combined_results if r.class_id in valid_class_ids_set
            ]
            for result in combined_results:
                hand_path_passed = bool(
                    hand_path_valid and result.class_id in valid_class_ids_set
                )
                hand_path_blocked = bool(
                    hand_path_valid and result.class_id not in valid_class_ids_set
                )
                self._record_hand_path_evidence(
                    trace_context,
                    result=result,
                    hand_path_valid=hand_path_valid,
                    hand_path_passed=hand_path_passed,
                    hand_path_blocked=hand_path_blocked,
                )
            if before_count > 0 and not filtered_by_hand_path:
                logger.warning(
                    "[VIDEO-ASYNC] fallback=kept_candidates reason=hand_path_removed_all"
                )
                stats.hand_path_filtered_classes = 0
            else:
                removed_results = [
                    result
                    for result in combined_results
                    if result.class_id not in valid_class_ids_set
                ]
                combined_results = filtered_by_hand_path
                stats.hand_path_filtered_classes = before_count - len(combined_results)
                if hand_path_valid:
                    for result in filtered_by_hand_path:
                        self._record_stage(
                            trace_context,
                            class_id=result.class_id,
                            class_name=result.class_name,
                            stage="hand_path_passed",
                            camera="top",
                        )
                    for result in removed_results:
                        self._record_stage(
                            trace_context,
                            class_id=result.class_id,
                            class_name=result.class_name,
                            stage="hand_path_filtered",
                            camera="top",
                        )

            if stats.hand_path_filtered_classes > 0:
                logger.info(
                    f"[VIDEO-ASYNC] 손 경로 필터링: {stats.hand_path_filtered_classes}개 제외"
                )

        # 최소 투표 필터링
        min_vote_count = self._effective_min_vote_count()
        min_vote_ratio = self._effective_min_vote_ratio()
        filtered_results = [
            r for r in combined_results
            if r.vote_ratio >= min_vote_ratio or r.vote_count >= min_vote_count
        ]
        filtered_results = self._limit_candidates(filtered_results, "VIDEO-ASYNC")
        threshold_rescue_candidates = self._build_threshold_rescue_candidates(
            low_confidence_stats,
            roi_filtered_stats,
            inference_allowed_class_ids,
            "VIDEO-ASYNC",
        )
        roi_rescue_candidates = self._build_roi_rescue_candidates(
            roi_filtered_stats,
            inference_allowed_class_ids,
            "VIDEO-ASYNC",
        )
        if trace_context is not None:
            for rank, result in enumerate(filtered_results, start=1):
                trace_context.record_final_candidate_rank(result.class_id, rank)
            for camera_type, diagnostics in extractor_diagnostics.items():
                trace_context.record_extractor_diagnostics(camera_type, diagnostics)
            trace_context.record_threshold_rescue_candidates(
                threshold_rescue_candidates,
                product_weights,
            )
            trace_context.record_roi_rescue_candidates(
                roi_rescue_candidates,
                product_weights,
            )

        stats.processing_time_ms = (time.time() - start_time) * 1000
        if stats.yolo_inference_count > 0:
            stats.yolo_avg_time_ms = (
                stats.yolo_total_time_ms / stats.yolo_inference_count
            )

        logger.info(
            f"[VIDEO-ASYNC][LATENCY] total_ms={stats.processing_time_ms:.1f} "
            f"frame_stride={stats.frame_stride} "
            f"original_frames={stats.original_frames} "
            f"processed_frames={stats.processed_frames} "
            f"skipped_frames={stats.skipped_frames} "
            f"yolo_total_ms={stats.yolo_total_time_ms:.1f} "
            f"yolo_avg_ms={stats.yolo_avg_time_ms:.1f} "
            f"yolo_count={stats.yolo_inference_count}"
        )
        self._log_candidate_trace("VIDEO-ASYNC", stats, filtered_results)
        if trace_context is not None:
            trace_context.record_video_stats(stats)
            trace_context.record_candidates(filtered_results, product_weights)

        logger.info("[VIDEO-ASYNC] ========== 비동기 처리 완료 ==========")
        logger.info(
            f"[VIDEO-ASYNC] 프레임: top={top_frame_count}, side={side_frame_count}, "
            f"후보={len(filtered_results)}개, 시간={stats.processing_time_ms:.1f}ms"
        )

        return VideoProcessingResult(
            vote_results=filtered_results,
            top_ensemble=top_ensemble,
            side_ensemble=side_ensemble,
            stats=stats,
            threshold_rescue_candidates=threshold_rescue_candidates,
            roi_rescue_candidates=roi_rescue_candidates,
        )

    def _apply_motion_filter_and_votes(
        self,
        camera_type: str,
        pending_votes: Dict[int, List[Tuple[float, str, int]]],
        bbox_trackers: Dict[int, BboxTracker],
        ensemble: VotingEnsemble,
        trace_context: Optional[TriggerTraceContext] = None,
    ) -> int:
        """
        Motion 필터링 적용 및 투표 등록 (v5.3).

        Args:
            camera_type: "top" or "side"
            pending_votes: 대기 중인 투표 (class_id -> [(conf, name), ...])
            bbox_trackers: BboxTracker 딕셔너리
            ensemble: 투표를 등록할 VotingEnsemble

        Returns:
            필터링된 클래스 수
        """
        motion_filtered_count = 0
        motion_passed_count = 0

        for class_id, votes in pending_votes.items():
            tracker = bbox_trackers.get(class_id)
            normalized_votes: List[Tuple[float, str, int]] = []
            for idx, vote in enumerate(votes):
                if len(vote) >= 3:
                    conf, class_name, frame_idx = vote[0], vote[1], vote[2]
                else:
                    conf, class_name = vote[0], vote[1]
                    frame_idx = idx
                normalized_votes.append((conf, class_name, frame_idx))

            has_motion = True
            motion_required = self.motion_filter_enabled or self._is_freezer_mode()
            if motion_required and tracker is not None:
                has_motion = tracker.has_motion(self._freezer_motion_floor())

            if has_motion:
                frame_instance_counts: Dict[int, int] = {}
                for _, _, frame_idx in normalized_votes:
                    frame_instance_counts[frame_idx] = (
                        frame_instance_counts.get(frame_idx, 0) + 1
                    )
                instance_count_hint = max(frame_instance_counts.values(), default=1)
                for conf, class_name, _ in normalized_votes:
                    ensemble.add_vote(
                        class_id=class_id,
                        confidence=conf,
                        class_name=class_name,
                        instance_count=instance_count_hint,
                    )
                self._record_stage(
                    trace_context,
                    class_id=class_id,
                    class_name=normalized_votes[0][1] if normalized_votes else "",
                    stage="motion_passed",
                    camera=camera_type,
                    amount=len(normalized_votes),
                )
                self._record_motion_evidence(
                    trace_context,
                    class_id=class_id,
                    class_name=normalized_votes[0][1] if normalized_votes else "",
                    camera=camera_type,
                    tracker=tracker,
                    motion_passed=True,
                )
                motion_passed_count += 1

                if tracker:
                    threshold_used = tracker.dynamic_threshold if tracker.dynamic_threshold > 0 else self.min_motion_displacement
                    logger.debug(
                        f"[MOTION-ASYNC] {camera_type} class {class_id}: PASSED "
                        f"(displacement={tracker.total_displacement:.1f}px, "
                        f"threshold={threshold_used:.1f}px)"
                    )
            else:
                motion_filtered_count += 1
                self._record_stage(
                    trace_context,
                    class_id=class_id,
                    class_name=normalized_votes[0][1] if normalized_votes else "",
                    stage="motion_filtered",
                    camera=camera_type,
                    amount=len(normalized_votes),
                )
                self._record_motion_evidence(
                    trace_context,
                    class_id=class_id,
                    class_name=normalized_votes[0][1] if normalized_votes else "",
                    camera=camera_type,
                    tracker=tracker,
                    motion_passed=False,
                )
                if tracker:
                    threshold_used = tracker.dynamic_threshold if tracker.dynamic_threshold > 0 else self.min_motion_displacement
                    logger.info(
                        f"[MOTION-ASYNC] {camera_type} class {class_id}: FILTERED "
                        f"(displacement={tracker.total_displacement:.1f}px < threshold={threshold_used:.1f}px)"
                    )

        logger.info(
            f"[MOTION-ASYNC] {camera_type} 필터링: 통과={motion_passed_count}, 제외={motion_filtered_count}"
        )

        return motion_filtered_count

    def _process_single_video(
        self,
        video_path: str,
        ensemble: VotingEnsemble,
        camera_type: str = "unknown",
        allowed_class_ids: Optional[List[int]] = None,
        hand_path_tracker: Optional[HandPathTracker] = None,
        trace_context: Optional[TriggerTraceContext] = None,
        low_confidence_stats: Optional[Dict[int, _LowConfidenceClassStats]] = None,
        roi_filtered_stats: Optional[Dict[int, _RoiFilteredClassStats]] = None,
        delta_weight: Optional[float] = None,
    ) -> dict:
        """
        Process a single video file with motion-based filtering.

        Uses FFmpeg for hardware-accelerated decoding (NVDEC on Jetson).
        Streams frames one at a time to minimize memory usage.
        Each frame is immediately released after YOLO inference.

        Motion Tracking (v4.1):
        - Tracks bbox center points for each class across frames
        - Only includes classes with significant center movement in final results
        - Filters out stationary background objects

        Hand Path Tracking (v4.6):
        - Tracks hand movement trajectory
        - Filters products that don't intersect with hand path

        Args:
            video_path: Path to video file
            ensemble: VotingEnsemble to accumulate votes
            camera_type: Camera type for logging ("top" or "side")
            allowed_class_ids: 허용된 YOLO 클래스 ID 리스트 (v4.4)
            hand_path_tracker: HandPathTracker for hand path filtering (v4.6)

        Returns:
            Statistics dict with frames, detections, and motion_filtered count
        """
        frame_count = 0
        detection_count = 0
        raw_detection_count = 0
        threshold_filtered_count = 0
        yolo_inference_count = 0
        yolo_total_time_ms = 0.0

        # Motion tracking: class_id -> BboxTracker
        bbox_trackers: Dict[int, BboxTracker] = {}

        # Buffer votes until motion filtering decides whether to keep them.
        # class_id -> list of (confidence, class_name) tuples
        pending_votes: Dict[int, List[Tuple[float, str, int]]] = {}

        # Use factory to get appropriate extractor (ffmpeg or cv2 fallback)
        # v4.6: camera_type 전달하여 카메라별 gamma/contrast 적용
        extractor = create_frame_extractor(
            video_path,
            prefer_ffmpeg=True,
            use_hwaccel=self.use_hwaccel,
            camera_type=camera_type,
        )
        if trace_context is not None:
            trace_context.plan_camera(
                camera_type,
                int(getattr(extractor, "total_frames", 0) or 0),
            )

        # ROI 필터링 통계
        roi_filtered_count = 0
        side_roi_soft_passed_count = 0
        side_roi_soft_filtered_count = 0

        for frame in extractor:
            frame_count += 1
            if trace_context is not None:
                trace_context.record_frame(camera_type, frame_count - 1, frame)

            # YOLO inference (single frame) - v4.4: allowed_class_ids 전달
            self._record_diagnostic_detections(
                trace_context,
                frame,
                camera_type,
                frame_count - 1,
            )
            yolo_started = time.perf_counter()
            detections = self._detect_frame(frame, allowed_class_ids, camera_type)
            yolo_total_time_ms += (time.perf_counter() - yolo_started) * 1000
            yolo_inference_count += 1
            self._record_preprocess(trace_context, camera_type)

            # v4.6: 손 경로 추적기에 모든 탐지 결과 전달 (손 포함)
            if hand_path_tracker is not None:
                hand_path_tracker.update_frame(detections, frame_count)

            # Process detections
            for det in detections:
                # Filter out hands and low confidence
                if det.is_hand:
                    continue
                raw_detection_count += 1
                self._record_stage(
                    trace_context,
                    class_id=det.cls,
                    class_name=det.name,
                    stage="raw",
                    camera=camera_type,
                    confidence=det.conf,
                )
                if det.conf < self._threshold_for_camera(camera_type):
                    threshold_filtered_count += 1
                    bbox_size = max(det.x2 - det.x1, det.y2 - det.y1)
                    roi_eligible = self._low_confidence_roi_eligible(
                        det,
                        camera_type,
                        delta_weight,
                    )
                    if low_confidence_stats is not None:
                        self._record_low_confidence_detection(
                            low_confidence_stats,
                            class_id=det.cls,
                            class_name=det.name,
                            confidence=det.conf,
                            camera=camera_type,
                            center=det.center,
                            frame_idx=frame_count - 1,
                            bbox_size=bbox_size,
                            roi_eligible=roi_eligible,
                        )
                    self._record_stage(
                        trace_context,
                        class_id=det.cls,
                        class_name=det.name,
                        stage="threshold_filtered",
                        camera=camera_type,
                        confidence=det.conf,
                    )
                    continue
                self._record_stage(
                    trace_context,
                    class_id=det.cls,
                    class_name=det.name,
                    stage="threshold_passed",
                    camera=camera_type,
                    confidence=det.conf,
                )

                # Top camera ROI changes by weight direction.
                if camera_type == "top" or self._uses_freezer_dual_top_profile(camera_type):
                    freezer_dual_top = self._uses_freezer_dual_top_profile(camera_type)
                    if freezer_dual_top:
                        top_roi_passed = self._freezer_roi_accepts(det)
                        top_roi_direction = self._freezer_roi_direction()
                        top_roi_y_limit = self._freezer_roi_y_split()
                    else:
                        top_roi_passed, top_roi_direction = self._top_roi_accepts(
                            det,
                            delta_weight,
                        )
                        top_roi_y_limit = self.top_roi_y_split
                    if not top_roi_passed:
                        roi_filtered_count += 1
                        self._record_stage(
                            trace_context,
                            class_id=det.cls,
                            class_name=det.name,
                            stage=(
                                "freezer_roi_filtered"
                                if freezer_dual_top
                                else "roi_filtered"
                            ),
                            camera=camera_type,
                            confidence=det.conf,
                            center=det.center,
                            roi_y_limit=top_roi_y_limit,
                            roi_direction=top_roi_direction,
                        )
                        continue

                # Side 카메라 ROI 필터: 왼쪽 영역만 허용
                # bbox 중심점이 오른쪽 영역(> side_roi_x_max)에 있으면 제외
                if camera_type == "side" and not self._uses_freezer_dual_top_profile(camera_type):
                    center_x = det.center[0]
                    side_roi_passed, side_roi_soft_passed = self._side_roi_accepts(
                        center_x
                    )
                    if not side_roi_passed:
                        roi_filtered_count += 1
                        if center_x > self._side_roi_soft_limit():
                            side_roi_soft_filtered_count += 1
                        bbox_size = max(det.x2 - det.x1, det.y2 - det.y1)
                        if roi_filtered_stats is not None:
                            self._record_roi_filtered_detection(
                                roi_filtered_stats,
                                class_id=det.cls,
                                class_name=det.name,
                                confidence=det.conf,
                                center=det.center,
                                frame_idx=frame_count - 1,
                                bbox_size=bbox_size,
                                roi_x_limit=self.side_roi_x_max,
                            )
                        self._record_stage(
                            trace_context,
                            class_id=det.cls,
                            class_name=det.name,
                            stage="roi_filtered",
                            camera=camera_type,
                            confidence=det.conf,
                            center=det.center,
                            roi_x_limit=self.side_roi_x_max,
                        )
                        if center_x > self._side_roi_soft_limit():
                            self._record_stage(
                                trace_context,
                                class_id=det.cls,
                                class_name=det.name,
                                stage="soft_margin_filtered",
                                camera=camera_type,
                                confidence=det.conf,
                            )
                        continue
                    if side_roi_soft_passed:
                        side_roi_soft_passed_count += 1
                        self._record_stage(
                            trace_context,
                            class_id=det.cls,
                            class_name=det.name,
                            stage="side_roi_soft_passed",
                            camera=camera_type,
                            confidence=det.conf,
                            center=det.center,
                            roi_x_limit=self._side_roi_soft_limit(),
                        )
                self._record_stage(
                    trace_context,
                    class_id=det.cls,
                    class_name=det.name,
                    stage=(
                        "freezer_roi_passed"
                        if self._uses_freezer_dual_top_profile(camera_type)
                        else "roi_passed"
                    ),
                    camera=camera_type,
                    confidence=det.conf
                    if self._uses_freezer_dual_top_profile(camera_type)
                    else None,
                    center=det.center,
                    roi_y_limit=self._freezer_roi_y_split()
                    if self._uses_freezer_dual_top_profile(camera_type)
                    else None,
                    roi_direction=self._freezer_roi_direction()
                    if self._uses_freezer_dual_top_profile(camera_type)
                    else None,
                )

                class_id = det.cls

                # Use YOLODetection's center property
                center = det.center

                # bbox 크기 기반 동적 임계값 계산
                bbox_width = det.x2 - det.x1
                bbox_height = det.y2 - det.y1
                bbox_size = max(bbox_width, bbox_height)
                # Dynamic threshold: 10% of bbox size, floored by motion config.
                dynamic_threshold = self._motion_threshold_for_detection(
                    camera_type,
                    bbox_size,
                )

                # Update bbox tracker
                if class_id not in bbox_trackers:
                    bbox_trackers[class_id] = BboxTracker()
                bbox_trackers[class_id].update(center, frame_count)
                # 동적 임계값 업데이트 (최대값 유지)
                bbox_trackers[class_id].dynamic_threshold = max(
                    bbox_trackers[class_id].dynamic_threshold,
                    dynamic_threshold
                )

                # Store vote for later (will be applied after motion filtering)
                if class_id not in pending_votes:
                    pending_votes[class_id] = []
                pending_votes[class_id].append((det.conf, det.name, frame_count - 1))

                detection_count += 1

            # Log progress every 50 frames
            if frame_count % 50 == 0:
                logger.info(
                    f"[VIDEO] {camera_type} 처리 중: {frame_count}프레임, "
                    f"탐지={detection_count}개"
                )

        # Set frame count
        ensemble.set_frame_count(frame_count)

        # Apply motion filtering and add votes to ensemble
        motion_filtered_count = 0
        motion_passed_count = 0

        for class_id, votes in pending_votes.items():
            tracker = bbox_trackers.get(class_id)

            has_motion = True
            motion_required = self.motion_filter_enabled or self._is_freezer_mode()
            if motion_required and tracker is not None:
                has_motion = tracker.has_motion(self._freezer_motion_floor())

            if has_motion:
                # Add all votes for this class
                frame_instance_counts: Dict[int, int] = {}
                for _, _, frame_idx in votes:
                    frame_instance_counts[frame_idx] = (
                        frame_instance_counts.get(frame_idx, 0) + 1
                    )
                instance_count_hint = max(frame_instance_counts.values(), default=1)
                for conf, class_name, _ in votes:
                    ensemble.add_vote(
                        class_id=class_id,
                        confidence=conf,
                        class_name=class_name,
                        instance_count=instance_count_hint,
                    )
                self._record_stage(
                    trace_context,
                    class_id=class_id,
                    class_name=votes[0][1] if votes else "",
                    stage="motion_passed",
                    camera=camera_type,
                    amount=len(votes),
                )
                self._record_motion_evidence(
                    trace_context,
                    class_id=class_id,
                    class_name=votes[0][1] if votes else "",
                    camera=camera_type,
                    tracker=tracker,
                    motion_passed=True,
                )
                motion_passed_count += 1

                if tracker:
                    threshold_used = tracker.dynamic_threshold if tracker.dynamic_threshold > 0 else self.min_motion_displacement
                    logger.debug(
                        f"[MOTION] {camera_type} class {class_id}: PASSED "
                        f"(displacement={tracker.total_displacement:.1f}px, "
                        f"max_dist={tracker.max_distance:.1f}px, "
                        f"threshold={threshold_used:.1f}px, "
                        f"detections={tracker.detection_count})"
                    )
            else:
                motion_filtered_count += 1
                self._record_stage(
                    trace_context,
                    class_id=class_id,
                    class_name=votes[0][1] if votes else "",
                    stage="motion_filtered",
                    camera=camera_type,
                    amount=len(votes),
                )
                self._record_motion_evidence(
                    trace_context,
                    class_id=class_id,
                    class_name=votes[0][1] if votes else "",
                    camera=camera_type,
                    tracker=tracker,
                    motion_passed=False,
                )
                if tracker:
                    threshold_used = tracker.dynamic_threshold if tracker.dynamic_threshold > 0 else self.min_motion_displacement
                    logger.info(
                        f"[MOTION] {camera_type} class {class_id}: FILTERED "
                        f"(displacement={tracker.total_displacement:.1f}px < threshold={threshold_used:.1f}px, "
                        f"detections={tracker.detection_count})"
                    )

        logger.info(
            f"[MOTION] {camera_type} 필터링 결과: "
            f"통과={motion_passed_count}개, 제외={motion_filtered_count}개 "
            f"(기본 임계값={self.min_motion_displacement}px, 동적 임계값 적용)"
        )

        # Side 카메라 ROI 필터링 결과 로그
        if camera_type == "side" and roi_filtered_count > 0:
            logger.info(
                f"[ROI] {camera_type} ROI 필터링: "
                f"{roi_filtered_count}개 탐지 제외 (center_x > {self.side_roi_x_max}px)"
            )

        return {
            "frames": frame_count,
            "raw_detections": raw_detection_count,
            "threshold_filtered": threshold_filtered_count,
            "detections": detection_count,
            "yolo_inference_count": yolo_inference_count,
            "yolo_total_time_ms": yolo_total_time_ms,
            "motion_filtered": motion_filtered_count,
            "roi_filtered": roi_filtered_count,
            "side_roi_soft_passed": side_roi_soft_passed_count,
            "side_roi_soft_filtered": side_roi_soft_filtered_count,
        }

    def process_single_video_file(
        self,
        video_path: str,
    ) -> VideoProcessingResult:
        """
        Process a single video file (for testing or single-camera setups).

        Args:
            video_path: Path to video file

        Returns:
            VideoProcessingResult with voting results
        """
        return self.process_videos(top_path=video_path)
