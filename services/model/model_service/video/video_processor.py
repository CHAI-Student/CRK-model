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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .frame_extractor import create_frame_extractor
from .voting_ensemble import VotingEnsemble, VoteResult
from model_service.vision import YOLOWrapper
from model_service.vision.hand_path_tracker import HandPathTracker
from model_service.core.config import config

logger = logging.getLogger(__name__)


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

    @property
    def total_displacement(self) -> float:
        """첫 번째와 마지막 위치 간 총 이동 거리."""
        if self.first_center is None or self.last_center is None:
            return 0.0
        return math.sqrt(
            (self.last_center[0] - self.first_center[0]) ** 2 +
            (self.last_center[1] - self.first_center[1]) ** 2
        )

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
    top_detections: int = 0
    side_detections: int = 0
    processing_time_ms: float = 0.0
    motion_filtered_classes: int = 0
    hand_path_filtered_classes: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "top_frames": self.top_frames,
            "side_frames": self.side_frames,
            "top_detections": self.top_detections,
            "side_detections": self.side_detections,
            "total_frames": self.top_frames + self.side_frames,
            "total_detections": self.top_detections + self.side_detections,
            "processing_time_ms": round(self.processing_time_ms, 1),
            "motion_filtered_classes": self.motion_filtered_classes,
            "hand_path_filtered_classes": self.hand_path_filtered_classes,
        }


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

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "vote_results": [r.to_dict() for r in self.vote_results],
            "stats": self.stats.to_dict(),
        }


class VideoProcessor:
    """
    AVI video processor with YOLO inference and voting ensemble.

    Processes videos frame-by-frame to minimize memory usage,
    suitable for Jetson Orin Nano deployment.

    Uses FFmpeg with NVDEC hardware acceleration when available.
    """

    def __init__(
        self,
        yolo: YOLOWrapper,
        min_vote_ratio: float = 0.05,
        confidence_threshold: float = 0.4,
        use_hwaccel: bool = True,
        motion_filter_enabled: bool = True,
        min_motion_displacement: float = 30.0,
        side_roi_x_max: float = 240.0,
        hand_path_filter_enabled: bool = True,
    ):
        """
        Initialize video processor.

        Args:
            yolo: YOLOWrapper instance for inference
            min_vote_ratio: Minimum vote ratio to include in results (default: 5%)
            confidence_threshold: Minimum confidence for YOLO detection (default: 0.6)
            use_hwaccel: Use hardware acceleration for video decoding (default: True)
            motion_filter_enabled: Enable motion-based filtering (default: True)
            min_motion_displacement: Minimum bbox center displacement to consider as motion (default: 30 pixels)
            side_roi_x_max: Side 카메라 ROI 최대 X 좌표 (기본 240px, 왼쪽 절반만 허용)
            hand_path_filter_enabled: Enable hand path-based filtering (v4.6, default: True)
        """
        self.yolo = yolo
        self.min_vote_ratio = min_vote_ratio
        self.confidence_threshold = confidence_threshold
        self.use_hwaccel = use_hwaccel
        self.motion_filter_enabled = motion_filter_enabled
        self.min_motion_displacement = min_motion_displacement
        self.side_roi_x_max = side_roi_x_max
        self.hand_path_filter_enabled = hand_path_filter_enabled

    def process_videos(
        self,
        top_path: Optional[str] = None,
        side_path: Optional[str] = None,
        allowed_class_ids: Optional[List[int]] = None,
        product_weights: Optional[Dict[int, float]] = None,
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

        logger.info(f"[VIDEO] ========== 비디오 처리 시작 ==========")
        logger.info(f"[VIDEO] top_path={top_path}")
        logger.info(f"[VIDEO] side_path={side_path}")
        if allowed_class_ids is not None:
            logger.info(f"[VIDEO] allowed_class_ids={len(allowed_class_ids)} classes")

        top_ensemble = VotingEnsemble(min_vote_ratio=self.min_vote_ratio)
        side_ensemble = VotingEnsemble(min_vote_ratio=self.min_vote_ratio)

        # v4.6: 손 경로 추적기 생성 (Top 카메라에서만 사용)
        top_hand_tracker: Optional[HandPathTracker] = None
        if self.hand_path_filter_enabled:
            top_hand_tracker = HandPathTracker()

        # Process top camera video
        if top_path:
            logger.info(f"[VIDEO] Top 카메라 처리 시작...")
            top_stats = self._process_single_video(
                top_path, top_ensemble, "top", allowed_class_ids,
                hand_path_tracker=top_hand_tracker,
            )
            stats.top_frames = top_stats["frames"]
            stats.top_detections = top_stats["detections"]
            stats.motion_filtered_classes += top_stats.get("motion_filtered", 0)
            logger.info(
                f"[VIDEO] Top 완료: 총 {stats.top_frames}프레임, "
                f"탐지={stats.top_detections}개, 고유클래스={len(top_ensemble.votes)}개"
            )

        # Process side camera video
        if side_path:
            logger.info(f"[VIDEO] Side 카메라 처리 시작...")
            side_stats = self._process_single_video(
                side_path, side_ensemble, "side", allowed_class_ids,
                hand_path_tracker=None,  # Side 카메라에서는 손 경로 필터링 안 함
            )
            stats.side_frames = side_stats["frames"]
            stats.side_detections = side_stats["detections"]
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

        # v4.6: 손 경로 필터링 적용 (Top 카메라 기준)
        if top_hand_tracker is not None and self.hand_path_filter_enabled:
            candidate_class_ids = [r.class_id for r in combined_results]
            valid_class_ids = top_hand_tracker.filter_products_by_path(candidate_class_ids)
            valid_class_ids_set = set(valid_class_ids)

            before_count = len(combined_results)
            combined_results = [r for r in combined_results if r.class_id in valid_class_ids_set]
            stats.hand_path_filtered_classes = before_count - len(combined_results)

            if stats.hand_path_filtered_classes > 0:
                logger.info(
                    f"[VIDEO] 손 경로 필터링: {stats.hand_path_filtered_classes}개 제외"
                )

        # Filter by minimum vote ratio OR minimum vote count
        # 조건 1: vote_ratio >= 5% (기존)
        # 조건 2: vote_count >= 3 (절대값 3프레임 이상이면 포함 - 짧은 비디오 대응)
        min_vote_count = 3
        filtered_results = [
            r for r in combined_results
            if r.vote_ratio >= self.min_vote_ratio or r.vote_count >= min_vote_count
        ]

        # 필터링 로그
        filtered_by_ratio = sum(1 for r in combined_results if r.vote_ratio >= self.min_vote_ratio)
        filtered_by_count = sum(1 for r in combined_results if r.vote_count >= min_vote_count and r.vote_ratio < self.min_vote_ratio)
        logger.info(
            f"[VIDEO] 필터링: vote_ratio >= {self.min_vote_ratio*100:.0f}%: {filtered_by_ratio}개, "
            f"vote_count >= {min_vote_count}: 추가 {filtered_by_count}개"
        )

        stats.processing_time_ms = (time.time() - start_time) * 1000

        logger.info(f"[VIDEO] 앙상블 결합 완료: {len(filtered_results)}개 후보")

        return VideoProcessingResult(
            vote_results=filtered_results,
            top_ensemble=top_ensemble,
            side_ensemble=side_ensemble,
            stats=stats,
        )

    async def process_videos_async(
        self,
        top_path: Optional[str] = None,
        side_path: Optional[str] = None,
        allowed_class_ids: Optional[List[int]] = None,
        product_weights: Optional[Dict[int, float]] = None,
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

        logger.info(f"[VIDEO-ASYNC] ========== 비동기 스트리밍 처리 시작 ==========")
        logger.info(f"[VIDEO-ASYNC] top_path={top_path}")
        logger.info(f"[VIDEO-ASYNC] side_path={side_path}")
        if allowed_class_ids is not None:
            logger.info(f"[VIDEO-ASYNC] allowed_class_ids={len(allowed_class_ids)} classes")

        top_ensemble = VotingEnsemble(min_vote_ratio=self.min_vote_ratio)
        side_ensemble = VotingEnsemble(min_vote_ratio=self.min_vote_ratio)

        # v5.3: 손 경로 추적기 (Top 카메라에서만 사용)
        top_hand_tracker: Optional[HandPathTracker] = None
        if self.hand_path_filter_enabled:
            top_hand_tracker = HandPathTracker()

        # 프레임 큐: (camera_type, frame_idx, frame, extractor_done)
        # None frame = EOF marker
        frame_queue: asyncio.Queue[Tuple[str, int, Optional["np.ndarray"]]] = asyncio.Queue(
            maxsize=config.async_streaming.frame_queue_size
        )

        # Motion tracking
        top_bbox_trackers: Dict[int, BboxTracker] = {}
        side_bbox_trackers: Dict[int, BboxTracker] = {}
        top_pending_votes: Dict[int, List[Tuple[float, str]]] = {}
        side_pending_votes: Dict[int, List[Tuple[float, str]]] = {}

        # Frame counters
        top_frame_count = 0
        side_frame_count = 0
        top_detection_count = 0
        side_detection_count = 0
        roi_filtered_count = 0

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

            extractor = create_frame_extractor(
                path,
                prefer_ffmpeg=True,
                use_hwaccel=self.use_hwaccel,
                camera_type=camera_type,
            )

            frame_idx = 0
            # ffmpeg 미존재 시 CV2FrameExtractor가 반환될 수 있음(__aiter__ 미지원)
            if not hasattr(extractor, '__aiter__'):
                logger.error(
                    f"[VIDEO-ASYNC] {camera_type}: async streaming requires ffmpeg "
                    "but extractor does not support async iteration (ffmpeg not available?). "
                    "Skipping video extraction."
                )
                await frame_queue.put((camera_type, -1, None))
                return

            try:
                async for frame in extractor:
                    await frame_queue.put((camera_type, frame_idx, frame))
                    frame_idx += 1

                    # Update frame count
                    if camera_type == "top":
                        top_frame_count = frame_idx
                    else:
                        side_frame_count = frame_idx

            except asyncio.CancelledError:
                logger.warning(f"[VIDEO-ASYNC] {camera_type} extraction cancelled at frame {frame_idx}")
                raise
            finally:
                # EOF marker
                await frame_queue.put((camera_type, -1, None))
                logger.info(f"[VIDEO-ASYNC] {camera_type} 추출 완료: {frame_idx}개 프레임")

        async def yolo_inference_loop() -> None:
            """YOLO 추론 루프 (단일 인스턴스)."""
            nonlocal top_detection_count, side_detection_count, roi_filtered_count

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
                    break

                # EOF marker
                if frame is None:
                    eof_received += 1
                    logger.debug(f"[VIDEO-ASYNC] EOF received from {camera_type} ({eof_received}/{expected_eofs})")
                    continue

                # YOLO 추론 (to_thread로 CPU 양보)
                detections = await asyncio.to_thread(
                    self.yolo.detect, frame, allowed_class_ids
                )

                # 카메라별 처리
                if camera_type == "top":
                    # 손 경로 추적 업데이트
                    if top_hand_tracker is not None:
                        top_hand_tracker.update_frame(detections, frame_idx)

                    for det in detections:
                        if det.is_hand or det.conf < self.confidence_threshold:
                            continue

                        class_id = det.cls
                        center = det.center

                        # 동적 임계값 계산
                        bbox_size = max(det.x2 - det.x1, det.y2 - det.y1)
                        dynamic_threshold = max(15.0, bbox_size * 0.10)

                        if class_id not in top_bbox_trackers:
                            top_bbox_trackers[class_id] = BboxTracker()
                        top_bbox_trackers[class_id].update(center, frame_idx)
                        top_bbox_trackers[class_id].dynamic_threshold = max(
                            top_bbox_trackers[class_id].dynamic_threshold,
                            dynamic_threshold
                        )

                        if class_id not in top_pending_votes:
                            top_pending_votes[class_id] = []
                        top_pending_votes[class_id].append((det.conf, det.name))
                        top_detection_count += 1

                else:  # side
                    for det in detections:
                        if det.is_hand or det.conf < self.confidence_threshold:
                            continue

                        # Side ROI 필터
                        center_x = det.center[0]
                        if center_x > self.side_roi_x_max:
                            roi_filtered_count += 1
                            continue

                        class_id = det.cls
                        center = det.center

                        bbox_size = max(det.x2 - det.x1, det.y2 - det.y1)
                        dynamic_threshold = max(15.0, bbox_size * 0.10)

                        if class_id not in side_bbox_trackers:
                            side_bbox_trackers[class_id] = BboxTracker()
                        side_bbox_trackers[class_id].update(center, frame_idx)
                        side_bbox_trackers[class_id].dynamic_threshold = max(
                            side_bbox_trackers[class_id].dynamic_threshold,
                            dynamic_threshold
                        )

                        if class_id not in side_pending_votes:
                            side_pending_votes[class_id] = []
                        side_pending_votes[class_id].append((det.conf, det.name))
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
            first_error: Exception | None = None

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

        # Frame counts 설정
        top_ensemble.set_frame_count(top_frame_count)
        side_ensemble.set_frame_count(side_frame_count)

        # Motion 필터링 및 투표 적용 (Top)
        top_motion_filtered = self._apply_motion_filter_and_votes(
            "top", top_pending_votes, top_bbox_trackers, top_ensemble
        )

        # Motion 필터링 및 투표 적용 (Side)
        side_motion_filtered = self._apply_motion_filter_and_votes(
            "side", side_pending_votes, side_bbox_trackers, side_ensemble
        )

        stats.top_frames = top_frame_count
        stats.side_frames = side_frame_count
        stats.top_detections = top_detection_count
        stats.side_detections = side_detection_count
        stats.motion_filtered_classes = top_motion_filtered + side_motion_filtered

        # Side ROI 필터링 로그
        if roi_filtered_count > 0:
            logger.info(
                f"[VIDEO-ASYNC] ROI 필터링: {roi_filtered_count}개 탐지 제외 "
                f"(center_x > {self.side_roi_x_max}px)"
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

        # 손 경로 필터링
        if top_hand_tracker is not None and self.hand_path_filter_enabled:
            candidate_class_ids = [r.class_id for r in combined_results]
            valid_class_ids = top_hand_tracker.filter_products_by_path(candidate_class_ids)
            valid_class_ids_set = set(valid_class_ids)

            before_count = len(combined_results)
            combined_results = [r for r in combined_results if r.class_id in valid_class_ids_set]
            stats.hand_path_filtered_classes = before_count - len(combined_results)

            if stats.hand_path_filtered_classes > 0:
                logger.info(
                    f"[VIDEO-ASYNC] 손 경로 필터링: {stats.hand_path_filtered_classes}개 제외"
                )

        # 최소 투표 필터링
        min_vote_count = 3
        filtered_results = [
            r for r in combined_results
            if r.vote_ratio >= self.min_vote_ratio or r.vote_count >= min_vote_count
        ]

        stats.processing_time_ms = (time.time() - start_time) * 1000

        logger.info(f"[VIDEO-ASYNC] ========== 비동기 처리 완료 ==========")
        logger.info(
            f"[VIDEO-ASYNC] 프레임: top={top_frame_count}, side={side_frame_count}, "
            f"후보={len(filtered_results)}개, 시간={stats.processing_time_ms:.1f}ms"
        )

        return VideoProcessingResult(
            vote_results=filtered_results,
            top_ensemble=top_ensemble,
            side_ensemble=side_ensemble,
            stats=stats,
        )

    def _apply_motion_filter_and_votes(
        self,
        camera_type: str,
        pending_votes: Dict[int, List[Tuple[float, str]]],
        bbox_trackers: Dict[int, BboxTracker],
        ensemble: VotingEnsemble,
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

            has_motion = True
            if self.motion_filter_enabled and tracker is not None:
                has_motion = tracker.has_motion(self.min_motion_displacement)

            if has_motion:
                for conf, class_name in votes:
                    ensemble.add_vote(
                        class_id=class_id,
                        confidence=conf,
                        class_name=class_name,
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

        # Motion tracking: class_id -> BboxTracker
        bbox_trackers: Dict[int, BboxTracker] = {}

        # Temporary storage for votes (applied after motion filtering)
        # class_id -> list of (confidence, class_name) tuples
        pending_votes: Dict[int, List[Tuple[float, str]]] = {}

        # Use factory to get appropriate extractor (ffmpeg or cv2 fallback)
        # v4.6: camera_type 전달하여 카메라별 gamma/contrast 적용
        extractor = create_frame_extractor(
            video_path,
            prefer_ffmpeg=True,
            use_hwaccel=self.use_hwaccel,
            camera_type=camera_type,
        )

        # ROI 필터링 통계
        roi_filtered_count = 0

        for frame in extractor:
            frame_count += 1

            # YOLO inference (single frame) - v4.4: allowed_class_ids 전달
            detections = self.yolo.detect(frame, allowed_class_ids=allowed_class_ids)

            # v4.6: 손 경로 추적기에 모든 탐지 결과 전달 (손 포함)
            if hand_path_tracker is not None:
                hand_path_tracker.update_frame(detections, frame_count)

            # Process detections
            for det in detections:
                # Filter out hands and low confidence
                if det.is_hand:
                    continue
                if det.conf < self.confidence_threshold:
                    continue

                # Side 카메라 ROI 필터: 왼쪽 영역만 허용
                # bbox 중심점이 오른쪽 영역(> side_roi_x_max)에 있으면 제외
                if camera_type == "side":
                    center_x = det.center[0]
                    if center_x > self.side_roi_x_max:
                        roi_filtered_count += 1
                        continue

                class_id = det.cls

                # Use YOLODetection's center property
                center = det.center

                # bbox 크기 기반 동적 임계값 계산
                bbox_width = det.x2 - det.x1
                bbox_height = det.y2 - det.y1
                bbox_size = max(bbox_width, bbox_height)
                # 동적 임계값: bbox 크기의 10%, 최소 15px
                dynamic_threshold = max(15.0, bbox_size * 0.10)

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
                pending_votes[class_id].append((det.conf, det.name))

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

            # Check motion
            has_motion = True
            if self.motion_filter_enabled and tracker is not None:
                has_motion = tracker.has_motion(self.min_motion_displacement)

            if has_motion:
                # Add all votes for this class
                for conf, class_name in votes:
                    ensemble.add_vote(
                        class_id=class_id,
                        confidence=conf,
                        class_name=class_name,
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
            "detections": detection_count,
            "motion_filtered": motion_filtered_count,
            "roi_filtered": roi_filtered_count,
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
