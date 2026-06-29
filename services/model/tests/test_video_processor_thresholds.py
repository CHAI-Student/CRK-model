import json
import logging
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from model_service.vision.yolo_wrapper import YOLODetection


def _patch_single_frame_extractor(monkeypatch: pytest.MonkeyPatch) -> None:
    import model_service.video.video_processor as video_processor_module

    class FakeExtractor:
        total_frames = 1
        last_diagnostics = None

        def __iter__(self) -> Iterator[int]:
            return iter([0])

    def create_fake_extractor(*_args: object, **_kwargs: object) -> FakeExtractor:
        return FakeExtractor()

    monkeypatch.setattr(
        video_processor_module,
        "create_frame_extractor",
        create_fake_extractor,
    )


class TopRoiFakeYolo:
    last_preprocess: dict[str, object] = {}

    def detect(
        self,
        frame: object,
        allowed_class_ids: list[int] | None = None,
        camera_type: str | None = None,
    ) -> list[YOLODetection]:
        return [
            YOLODetection(
                xyxy=(10.0, 100.0, 40.0, 140.0),
                cls=52,
                conf=0.91,
                name="PRODUCT_UPPER_HALF",
            ),
            YOLODetection(
                xyxy=(10.0, 220.0, 40.0, 260.0),
                cls=53,
                conf=0.92,
                name="PRODUCT_LOWER_HALF",
            ),
        ]


def _process_side_detections(
    monkeypatch: pytest.MonkeyPatch,
    centers: list[float],
    *,
    class_id: int = 42,
    confidence: float = 0.75,
):
    import model_service.video.video_processor as video_processor_module
    from model_service.video import VideoProcessor

    class FakeExtractor:
        last_diagnostics = None

        def __init__(self) -> None:
            self.total_frames = len(centers)

        def __iter__(self) -> Iterator[int]:
            return iter(range(len(centers)))

    class FakeYolo:
        last_preprocess = {}

        def detect(
            self,
            frame: object,
            allowed_class_ids: list[int] | None = None,
            camera_type: str | None = None,
        ) -> list[YOLODetection]:
            center_x = centers[int(frame)]
            return [
                YOLODetection(
                    xyxy=(center_x - 10.0, 210.0, center_x + 10.0, 250.0),
                    cls=class_id,
                    conf=confidence,
                    name=f"PRODUCT_{class_id}",
                )
            ]

    monkeypatch.setattr(
        video_processor_module,
        "create_frame_extractor",
        lambda *args, **kwargs: FakeExtractor(),
    )
    processor = VideoProcessor(
        yolo=FakeYolo(),
        motion_filter_enabled=False,
        hand_path_filter_enabled=False,
        min_vote_count=1,
    )

    return processor.process_videos(
        side_path="/tmp/side.avi",
        allowed_class_ids=[class_id],
    )


def test_video_processor_uses_split_camera_thresholds():
    from model_service.video.video_processor import VideoProcessor

    processor = VideoProcessor(
        yolo=MagicMock(),
        top_confidence_threshold=0.5,
        side_confidence_threshold=0.25,
    )

    assert processor._threshold_for_camera("top") == 0.5
    assert processor._threshold_for_camera("side") == 0.25
    assert processor.confidence_threshold == 0.25


def test_video_processor_shared_threshold_remains_backward_compatible():
    from model_service.video.video_processor import VideoProcessor

    processor = VideoProcessor(yolo=MagicMock(), confidence_threshold=0.4)

    assert processor._threshold_for_camera("top") == 0.4
    assert processor._threshold_for_camera("side") == 0.4
    assert processor.confidence_threshold == 0.4


def test_video_processor_operational_roi_defaults_are_480_left_crop_aligned():
    from model_service.core.config import VisionModel

    vision = VisionModel()

    assert vision.top_crop_policy == "left"
    assert vision.side_crop_policy == "left"
    assert vision.crop_width == 480
    assert vision.side_roi_x_max == 400.0
    assert vision.side_roi_soft_margin_px == 5.0
    assert vision.top_roi_y_split == 240.0
    assert vision.roi_rescue_max_over_limit_px == 0.0


def test_video_processor_default_side_roi_keeps_hard_boundary_and_soft_margin(monkeypatch):
    result = _process_side_detections(
        monkeypatch,
        [399.9, 402.5, 405.1],
        class_id=201,
        confidence=0.76,
    )

    assert [candidate.class_id for candidate in result.vote_results] == [201]
    assert result.vote_results[0].vote_count == 2
    assert result.stats.side_roi_soft_passed_detections == 1
    assert result.stats.roi_filtered_detections == 1


def test_video_processor_soft_side_roi_promotes_pepsi_boundary_trace_shape(monkeypatch):
    result = _process_side_detections(
        monkeypatch,
        [402.1, 403.8, 404.0],
        class_id=75,
        confidence=0.646,
    )

    assert [candidate.class_id for candidate in result.vote_results] == [75]
    assert result.vote_results[0].side_vote_count == 3
    assert result.stats.side_roi_soft_passed_detections == 3
    assert result.stats.roi_filtered_detections == 0


def test_video_processor_soft_side_roi_still_filters_trevi_far_right_shape(monkeypatch):
    result = _process_side_detections(
        monkeypatch,
        [409.8, 410.2],
        class_id=54,
        confidence=0.894,
    )

    assert result.vote_results == []
    assert result.stats.side_roi_soft_passed_detections == 0
    assert result.stats.roi_filtered_detections == 2


def test_video_processor_default_side_roi_promotes_bibigo_trace_shape(monkeypatch):
    result = _process_side_detections(
        monkeypatch,
        [330.2, 339.4],
        class_id=120,
        confidence=0.7535,
    )

    assert [candidate.class_id for candidate in result.vote_results] == [120]
    assert result.vote_results[0].side_vote_count == 2
    assert result.stats.roi_filtered_detections == 0


def test_video_processor_default_side_roi_promotes_pepero_trace_shape(monkeypatch):
    result = _process_side_detections(
        monkeypatch,
        [424.4, 412.7, 374.0, 366.8],
        class_id=114,
        confidence=0.7677,
    )

    assert [candidate.class_id for candidate in result.vote_results] == [114]
    assert result.vote_results[0].side_vote_count == 2
    assert result.stats.roi_filtered_detections == 2


def test_video_processor_side_threshold_promotes_letsbe_trace_confidence(monkeypatch):
    result = _process_side_detections(
        monkeypatch,
        [220.0, 225.0],
        class_id=12,
        confidence=0.2926,
    )

    assert [candidate.class_id for candidate in result.vote_results] == [12]
    assert result.vote_results[0].side_vote_count == 2
    assert result.stats.side_threshold_filtered == 0


def test_video_processor_limits_candidates_to_configured_top_k(monkeypatch):
    from model_service.core.config import config
    from model_service.video.video_processor import VideoProcessor
    from model_service.video.voting_ensemble import VoteResult

    monkeypatch.setattr(config.vision, "top_k", 10)
    processor = VideoProcessor(yolo=MagicMock())
    results = [
        VoteResult(
            class_id=index,
            class_name=f"product-{index}",
            vote_count=1,
            max_confidence=0.9,
            avg_confidence=0.9,
            weighted_confidence=0.9,
        )
        for index in range(12)
    ]

    limited = processor._limit_candidates(results, "TEST")

    assert len(limited) == 10
    assert [result.class_id for result in limited] == list(range(10))


def test_video_processor_merge_rescue_votes_keeps_vision_source_first(monkeypatch):
    from model_service.core.config import config
    from model_service.video.video_processor import VideoProcessor
    from model_service.video.voting_ensemble import VoteResult

    def candidate(class_id: int, source: str, confidence: float) -> VoteResult:
        return VoteResult(
            class_id=class_id,
            class_name=f"product-{class_id}",
            vote_count=1,
            max_confidence=confidence,
            avg_confidence=confidence,
            weighted_confidence=confidence,
            source=source,
        )

    vision = candidate(1, "vision", 0.2)
    roi_rescue = candidate(2, "roi_rescue", 0.99)
    threshold_rescue = candidate(3, "threshold_rescue", 0.98)

    ranked = VideoProcessor.rank_candidates_by_source_priority(
        [roi_rescue, threshold_rescue, vision]
    )

    assert [result.class_id for result in ranked] == [1, 2, 3]

    monkeypatch.setattr(config.vision, "top_k", 1)

    merged = VideoProcessor.merge_rescue_votes(
        [vision],
        [roi_rescue, threshold_rescue],
    )

    assert [result.class_id for result in merged] == [1]


def test_video_processor_defaults_use_configurable_filter_settings(monkeypatch):
    from model_service.core.config import config
    from model_service.video.video_processor import VideoProcessor

    monkeypatch.setattr(config.vision, "min_vote_ratio", 0.02, raising=False)
    monkeypatch.setattr(config.vision, "motion_min_displacement_px", 8.0, raising=False)
    monkeypatch.setattr(config.vision, "side_roi_x_max", 275.0, raising=False)
    monkeypatch.setattr(config.vision, "min_vote_count", 1, raising=False)

    processor = VideoProcessor(yolo=MagicMock())

    assert processor.min_vote_ratio == 0.02
    assert processor.min_motion_displacement == 8.0
    assert processor.side_roi_x_max == 275.0
    assert processor.min_vote_count == 1


@pytest.mark.asyncio
async def test_async_video_processor_frame_stride_skips_frames_and_records_stats(
    monkeypatch,
    tmp_path,
):
    import json

    import model_service.video.video_processor as video_processor_module
    from model_service.video import VideoProcessor
    from model_service.video.frame_trace import TriggerTraceContext
    from model_service.vision.yolo_wrapper import YOLODetection

    class FakeAsyncExtractor:
        last_diagnostics = None

        def __init__(self, frames):
            self.frames = frames
            self.total_frames = len(frames)

        async def __aiter__(self):
            for frame in self.frames:
                yield frame

    class FakeYolo:
        last_preprocess = {}

        def __init__(self):
            self.calls = []

        def detect(self, frame, allowed_class_ids=None, camera_type=None):
            self.calls.append((camera_type, frame))
            return [
                YOLODetection(
                    xyxy=(10.0, 10.0, 30.0, 30.0),
                    cls=42,
                    conf=0.91,
                    name="PRODUCT_STRIDE",
                )
            ]

    def extractor_factory(path, *args, **kwargs):
        camera_type = kwargs["camera_type"]
        base = 0 if camera_type == "top" else 100
        return FakeAsyncExtractor([base + index for index in range(6)])

    monkeypatch.setattr(
        video_processor_module,
        "create_frame_extractor",
        extractor_factory,
    )

    trace_context = TriggerTraceContext(
        session_id="frame-stride",
        zone=3,
        top_path="/tmp/top.avi",
        side_path="/tmp/side.avi",
        log_dir=tmp_path / "logs",
        sample_export_dir=tmp_path / "samples",
        sample_export_enabled=False,
    )
    yolo = FakeYolo()
    processor = VideoProcessor(
        yolo=yolo,
        motion_filter_enabled=False,
        hand_path_filter_enabled=False,
        min_vote_count=1,
    )

    result = await processor.process_videos_async(
        top_path="/tmp/top.avi",
        side_path="/tmp/side.avi",
        trace_context=trace_context,
    )
    trace_context.finalize(status="complete")

    assert [(camera, frame) for camera, frame in yolo.calls] == [
        ("top", 0),
        ("top", 2),
        ("top", 4),
        ("side", 100),
        ("side", 102),
        ("side", 104),
    ]
    assert result.stats.frame_stride == 2
    assert result.stats.original_frames == 12
    assert result.stats.processed_frames == 6
    assert result.stats.skipped_frames == 6
    assert result.stats.yolo_inference_count == 6

    detail_file = next((tmp_path / "logs" / "triggers").glob("*/*.json"))
    detail = json.loads(detail_file.read_text(encoding="utf-8"))
    assert detail["cameras"]["top"]["total_frames"] == 6
    assert detail["cameras"]["top"]["processed_frames"] == 3
    assert detail["cameras"]["side"]["total_frames"] == 6
    assert detail["cameras"]["side"]["processed_frames"] == 3
    assert detail["video_stats"]["frame_stride"] == 2
    assert detail["video_stats"]["original_frames"] == 12
    assert detail["video_stats"]["processed_frames"] == 6
    assert detail["video_stats"]["skipped_frames"] == 6


@pytest.mark.asyncio
async def test_async_video_processor_ignores_non_two_stride_override(
    monkeypatch,
):
    import model_service.video.video_processor as video_processor_module
    from model_service.core.config import config
    from model_service.video import VideoProcessor
    from model_service.vision.yolo_wrapper import YOLODetection

    class FakeAsyncExtractor:
        last_diagnostics = None
        total_frames = 3

        async def __aiter__(self):
            for frame in range(3):
                yield frame

    class FakeYolo:
        last_preprocess = {}

        def __init__(self):
            self.calls = []

        def detect(self, frame, allowed_class_ids=None, camera_type=None):
            self.calls.append(frame)
            return [
                YOLODetection(
                    xyxy=(10.0, 10.0, 30.0, 30.0),
                    cls=42,
                    conf=0.91,
                    name="PRODUCT_STRIDE",
                )
            ]

    monkeypatch.setattr(config.async_streaming, "frame_stride", 1, raising=False)
    monkeypatch.setattr(
        video_processor_module,
        "create_frame_extractor",
        lambda *args, **kwargs: FakeAsyncExtractor(),
    )

    yolo = FakeYolo()
    processor = VideoProcessor(
        yolo=yolo,
        motion_filter_enabled=False,
        hand_path_filter_enabled=False,
        min_vote_count=1,
    )

    result = await processor.process_videos_async(top_path="/tmp/top.avi")

    assert yolo.calls == [0, 2]
    assert result.stats.frame_stride == 2
    assert result.stats.original_frames == 3
    assert result.stats.processed_frames == 2
    assert result.stats.skipped_frames == 1


def test_async_streaming_frame_stride_is_fixed_to_two():
    from model_service.core.config import AsyncStreamingModel

    assert AsyncStreamingModel().frame_stride == 2

    with pytest.raises(ValueError, match="frame_stride is fixed at 2"):
        AsyncStreamingModel(frame_stride=1)

    with pytest.raises(ValueError, match="frame_stride is fixed at 2"):
        AsyncStreamingModel(frame_stride=3)


@pytest.mark.asyncio
async def test_async_video_processor_reraises_model_service_task_error(monkeypatch):
    import model_service.video.video_processor as video_processor_module
    from model_service.core.exceptions import YOLOGPUError
    from model_service.video import VideoProcessor

    class FakeAsyncExtractor:
        total_frames = 1
        last_diagnostics = None

        async def __aiter__(self):
            yield 0

    class FailingYolo:
        last_preprocess = {}

        def detect(self, frame, allowed_class_ids=None, camera_type=None):
            raise YOLOGPUError("cuda out of memory")

    monkeypatch.setattr(
        video_processor_module,
        "create_frame_extractor",
        lambda *args, **kwargs: FakeAsyncExtractor(),
    )

    processor = VideoProcessor(
        yolo=FailingYolo(),
        motion_filter_enabled=False,
        hand_path_filter_enabled=False,
        min_vote_count=1,
    )

    with pytest.raises(YOLOGPUError, match="cuda out of memory"):
        await processor.process_videos_async(top_path="/tmp/top.avi")


@pytest.mark.asyncio
async def test_async_video_processor_wraps_unknown_task_error(monkeypatch):
    import model_service.video.video_processor as video_processor_module
    from model_service.core.exceptions import VideoProcessingError
    from model_service.video import VideoProcessor

    class FakeAsyncExtractor:
        total_frames = 1
        last_diagnostics = None

        async def __aiter__(self):
            yield 0

    class FailingYolo:
        last_preprocess = {}

        def detect(self, frame, allowed_class_ids=None, camera_type=None):
            raise RuntimeError("unexpected detector crash")

    monkeypatch.setattr(
        video_processor_module,
        "create_frame_extractor",
        lambda *args, **kwargs: FakeAsyncExtractor(),
    )

    processor = VideoProcessor(
        yolo=FailingYolo(),
        motion_filter_enabled=False,
        hand_path_filter_enabled=False,
        min_vote_count=1,
    )

    with pytest.raises(VideoProcessingError, match="Task error in yolo-inference"):
        await processor.process_videos_async(top_path="/tmp/top.avi")


@pytest.mark.asyncio
async def test_async_video_processor_fails_without_async_extractor(monkeypatch):
    import model_service.video.video_processor as video_processor_module
    from model_service.core.exceptions import VideoProcessingError
    from model_service.video import VideoProcessor

    class SyncOnlyExtractor:
        total_frames = 1
        last_diagnostics = None

        def __iter__(self):
            return iter([0])

    monkeypatch.setattr(
        video_processor_module,
        "create_frame_extractor",
        lambda *args, **kwargs: SyncOnlyExtractor(),
    )

    processor = VideoProcessor(
        yolo=MagicMock(),
        motion_filter_enabled=False,
        hand_path_filter_enabled=False,
        min_vote_count=1,
    )

    with pytest.raises(VideoProcessingError, match="does not support async iteration"):
        await processor.process_videos_async(top_path="/tmp/top.avi")


@pytest.mark.asyncio
async def test_async_video_processor_fails_on_zero_frames_after_retry(monkeypatch):
    import model_service.video.video_processor as video_processor_module
    from model_service.core.exceptions import VideoProcessingError
    from model_service.video import VideoProcessor

    class Diagnostics:
        expected_frames = 3
        decoded_frames = 0
        method = "sync_mjpeg_retry"
        final_branch = "sync_mjpeg_retry"
        stderr_tail = "corrupt input"

    class ZeroFrameExtractor:
        total_frames = 3
        last_diagnostics = Diagnostics()

        async def __aiter__(self):
            if False:
                yield 0

    monkeypatch.setattr(
        video_processor_module,
        "create_frame_extractor",
        lambda *args, **kwargs: ZeroFrameExtractor(),
    )

    processor = VideoProcessor(
        yolo=MagicMock(),
        motion_filter_enabled=False,
        hand_path_filter_enabled=False,
        min_vote_count=1,
    )

    with pytest.raises(VideoProcessingError, match="decoded zero frames"):
        await processor.process_videos_async(top_path="/tmp/top.avi")


@pytest.mark.asyncio
async def test_async_video_processor_fails_on_frame_queue_timeout(monkeypatch):
    import asyncio

    import model_service.video.video_processor as video_processor_module
    from model_service.core.exceptions import VideoProcessingError
    from model_service.video import VideoProcessor

    original_wait_for = video_processor_module.asyncio.wait_for

    async def fake_wait_for(awaitable, timeout=None):
        if timeout == 60.0:
            close = getattr(awaitable, "close", None)
            if close is not None:
                close()
            raise asyncio.TimeoutError
        return await original_wait_for(awaitable, timeout=timeout)

    class NoFrameExtractor:
        total_frames = 0
        last_diagnostics = None

        async def __aiter__(self):
            if False:
                yield 0

    monkeypatch.setattr(video_processor_module.asyncio, "wait_for", fake_wait_for)
    monkeypatch.setattr(
        video_processor_module,
        "create_frame_extractor",
        lambda *args, **kwargs: NoFrameExtractor(),
    )

    processor = VideoProcessor(
        yolo=MagicMock(),
        motion_filter_enabled=False,
        hand_path_filter_enabled=False,
        min_vote_count=1,
    )

    with pytest.raises(VideoProcessingError, match="frame queue timeout"):
        await processor.process_videos_async(top_path="/tmp/top.avi")


def test_video_processor_records_preprocess_and_stage_counts(monkeypatch, tmp_path):
    import model_service.video.video_processor as video_processor_module
    from model_service.video import VideoProcessor
    from model_service.video.frame_trace import TriggerTraceContext
    from model_service.vision.yolo_wrapper import YOLODetection

    class FakeExtractor:
        total_frames = 1
        last_diagnostics = None

        def __iter__(self):
            return iter([[[0]]])

    class FakeYolo:
        last_preprocess = {
            "camera_type": "side",
            "original_width": 640,
            "original_height": 480,
            "processed_width": 480,
            "processed_height": 480,
            "crop_policy": "left",
            "crop_box": {"x1": 0, "y1": 0, "x2": 480, "y2": 480},
        }

        def detect(self, frame, allowed_class_ids=None, camera_type=None):
            return [
                YOLODetection(
                    xyxy=(400.0, 10.0, 430.0, 40.0),
                    cls=42,
                    conf=0.91,
                    name="PRODUCT_ROI_FILTERED",
                ),
                YOLODetection(
                    xyxy=(10.0, 10.0, 40.0, 40.0),
                    cls=43,
                    conf=0.92,
                    name="PRODUCT_KEPT",
                ),
            ]

    monkeypatch.setattr(
        video_processor_module,
        "create_frame_extractor",
        lambda *args, **kwargs: FakeExtractor(),
    )

    trace_context = TriggerTraceContext(
        session_id="stage-session",
        zone=1,
        top_path=None,
        side_path="/tmp/side.avi",
        log_dir=tmp_path / "logs",
        sample_export_dir=tmp_path / "samples",
    )
    processor = VideoProcessor(
        yolo=FakeYolo(),
        motion_filter_enabled=False,
        hand_path_filter_enabled=False,
        side_roi_x_max=360.0,
        min_vote_count=1,
    )

    result = processor.process_videos(
        side_path="/tmp/side.avi",
        allowed_class_ids=[42, 43],
        trace_context=trace_context,
    )
    trace_context.finalize(status="complete")

    assert [candidate.class_id for candidate in result.vote_results] == [43]
    detail_file = next((tmp_path / "logs" / "triggers").glob("*/*.json"))
    detail = __import__("json").loads(detail_file.read_text(encoding="utf-8"))
    assert detail["preprocess"]["side"]["crop_policy"] == "left"
    assert detail["stage_counts_by_class"]["42"]["roi_filtered"] == 1
    assert detail["stage_counts_by_class"]["43"]["final_rank"] == 1


def test_video_processor_top_roi_removal_keeps_lower_region(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from model_service.video import VideoProcessor
    from model_service.video.frame_trace import TriggerTraceContext

    _patch_single_frame_extractor(monkeypatch)
    processor = VideoProcessor(
        yolo=TopRoiFakeYolo(),
        motion_filter_enabled=False,
        hand_path_filter_enabled=False,
        min_vote_count=1,
    )
    trace_context = TriggerTraceContext(
        session_id="top-roi-removal",
        zone=1,
        top_path="/tmp/top.avi",
        side_path=None,
        log_dir=tmp_path / "logs",
        sample_export_dir=tmp_path / "samples",
        sample_export_enabled=False,
    )

    result = processor.process_videos(
        top_path="/tmp/top.avi",
        allowed_class_ids=[52, 53],
        delta_weight=-50.0,
        trace_context=trace_context,
    )
    trace_context.finalize(status="complete")

    assert [candidate.class_id for candidate in result.vote_results] == [53]
    assert result.stats.roi_filtered_detections == 1
    assert result.roi_rescue_candidates == []

    detail_file = next((tmp_path / "logs" / "triggers").glob("*/*.json"))
    detail = json.loads(detail_file.read_text(encoding="utf-8"))
    assert detail["stage_counts_by_class"]["52"]["roi_filtered"] == 1
    assert detail["stage_counts_by_class"]["52"]["roi_y_limit"] == 240.0
    assert detail["stage_counts_by_class"]["52"]["roi_direction"] == "removal"
    assert detail["vision_config"]["top_roi_enabled"] is True
    assert detail["vision_config"]["top_roi_y_split"] == 240.0


def test_video_processor_top_roi_return_keeps_lower_region(monkeypatch: pytest.MonkeyPatch) -> None:
    from model_service.video import VideoProcessor

    _patch_single_frame_extractor(monkeypatch)
    processor = VideoProcessor(
        yolo=TopRoiFakeYolo(),
        motion_filter_enabled=False,
        hand_path_filter_enabled=False,
        min_vote_count=1,
    )

    result = processor.process_videos(
        top_path="/tmp/top.avi",
        allowed_class_ids=[52, 53],
        delta_weight=50.0,
    )

    assert [candidate.class_id for candidate in result.vote_results] == [53]
    assert result.stats.roi_filtered_detections == 1
    assert result.roi_rescue_candidates == []


def test_video_processor_top_roi_skips_zero_delta(monkeypatch: pytest.MonkeyPatch) -> None:
    from model_service.video import VideoProcessor

    _patch_single_frame_extractor(monkeypatch)
    processor = VideoProcessor(
        yolo=TopRoiFakeYolo(),
        motion_filter_enabled=False,
        hand_path_filter_enabled=False,
        min_vote_count=1,
    )

    result = processor.process_videos(
        top_path="/tmp/top.avi",
        allowed_class_ids=[52, 53],
        delta_weight=0.0,
    )

    assert [candidate.class_id for candidate in result.vote_results] == [53, 52]
    assert result.stats.roi_filtered_detections == 0
    assert result.roi_rescue_candidates == []


def test_video_processor_freezer_dual_top_side_uses_lower_y_roi(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import model_service.video.video_processor as video_processor_module
    from model_service.core.config import config
    from model_service.video import VideoProcessor
    from model_service.video.frame_trace import TriggerTraceContext

    class FakeExtractor:
        last_diagnostics = None
        total_frames = 2

        def __iter__(self) -> Iterator[int]:
            return iter([0, 1])

    class FreezerRoiYolo:
        last_preprocess: dict[str, object] = {}

        def detect(
            self,
            frame: object,
            allowed_class_ids: list[int] | None = None,
            camera_type: str | None = None,
        ) -> list[YOLODetection]:
            offset = float(frame) * 12.0
            return [
                YOLODetection(
                    xyxy=(10.0, 100.0, 40.0, 140.0),
                    cls=52,
                    conf=0.91,
                    name="PRODUCT_UPPER_HALF",
                ),
                YOLODetection(
                    xyxy=(10.0 + offset, 220.0, 40.0 + offset, 260.0),
                    cls=53,
                    conf=0.92,
                    name="PRODUCT_LOWER_HALF",
                ),
            ]

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")
    monkeypatch.setattr(config.vision, "freezer_motion_min_displacement_px", 0.0)
    monkeypatch.setattr(config.vision, "freezer_min_vote_count", 1)
    monkeypatch.setattr(config.vision, "freezer_min_vote_ratio", 0.0)
    monkeypatch.setattr(
        video_processor_module,
        "create_frame_extractor",
        lambda *args, **kwargs: FakeExtractor(),
    )

    processor = VideoProcessor(
        yolo=FreezerRoiYolo(),
        motion_filter_enabled=False,
        hand_path_filter_enabled=False,
        min_vote_count=1,
    )
    trace_context = TriggerTraceContext(
        session_id="freezer-dual-top-side-roi",
        zone=1,
        top_path=None,
        side_path="/tmp/top-side.avi",
        log_dir=tmp_path / "logs",
        sample_export_dir=tmp_path / "samples",
        sample_export_enabled=False,
    )

    result = processor.process_videos(
        side_path="/tmp/top-side.avi",
        allowed_class_ids=[52, 53],
        delta_weight=0.0,
        trace_context=trace_context,
    )
    trace_context.finalize(status="complete")

    assert [candidate.class_id for candidate in result.vote_results] == [53]
    assert result.stats.roi_filtered_detections == 2
    assert result.stats.side_roi_soft_filtered_detections == 0
    assert result.roi_rescue_candidates == []

    detail_file = next((tmp_path / "logs" / "triggers").glob("*/*.json"))
    detail = json.loads(detail_file.read_text(encoding="utf-8"))
    assert detail["stage_counts_by_class"]["52"]["freezer_roi_filtered"] == 2
    assert detail["stage_counts_by_class"]["52"]["roi_y_limit"] == 240.0
    assert detail["stage_counts_by_class"]["52"]["roi_direction"] == (
        "freezer_lower_half"
    )


def test_video_processor_records_same_frame_instance_count_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from model_service.core.config import config
    from model_service.video import VideoProcessor

    class MultiBoxYolo:
        last_preprocess: dict[str, object] = {}

        def detect(
            self,
            frame: object,
            allowed_class_ids: list[int] | None = None,
            camera_type: str | None = None,
        ) -> list[YOLODetection]:
            return [
                YOLODetection(
                    xyxy=(10.0, 250.0, 40.0, 290.0),
                    cls=53,
                    conf=0.92,
                    name="PRODUCT_LOWER_HALF",
                ),
                YOLODetection(
                    xyxy=(60.0, 250.0, 90.0, 290.0),
                    cls=53,
                    conf=0.91,
                    name="PRODUCT_LOWER_HALF",
                ),
            ]

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")
    monkeypatch.setattr(config.vision, "freezer_motion_min_displacement_px", 0.0)
    monkeypatch.setattr(config.vision, "freezer_min_vote_count", 1)
    monkeypatch.setattr(config.vision, "freezer_min_vote_ratio", 0.0)
    _patch_single_frame_extractor(monkeypatch)

    processor = VideoProcessor(
        yolo=MultiBoxYolo(),
        motion_filter_enabled=False,
        hand_path_filter_enabled=False,
        min_vote_count=1,
    )

    result = processor.process_videos(
        top_path="/tmp/top.avi",
        allowed_class_ids=[53],
        delta_weight=-50.0,
    )

    assert len(result.vote_results) == 1
    assert result.vote_results[0].class_id == 53
    assert result.vote_results[0].instance_count_hint == 2


def test_video_processor_freezer_handled_filter_keeps_weight_tiebreak_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from model_service.core.config import config
    from model_service.video import VideoProcessor, VoteResult
    from model_service.video.frame_trace import TriggerTraceContext

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")

    trace_context = TriggerTraceContext(
        session_id="freezer-handled-filter",
        zone=2,
        top_path="/tmp/top.avi",
        side_path="/tmp/side.avi",
        log_dir=tmp_path / "logs",
        sample_export_dir=tmp_path / "samples",
        sample_export_enabled=False,
    )
    vote_results = [
        VoteResult(
            class_id=13,
            class_name="BAG_COOZROCK_JUICY_MEAT_DUMPLING_168G",
            vote_count=79,
            max_confidence=0.9251,
            avg_confidence=0.9,
            weighted_confidence=1.0,
            top_detected=True,
            side_detected=True,
            top_vote_count=39,
            side_vote_count=40,
            instance_count_hint=3,
        ),
        VoteResult(
            class_id=44,
            class_name="STICK_BINGGRAE_MELONA_75ML",
            vote_count=12,
            max_confidence=0.8891,
            avg_confidence=0.86,
            weighted_confidence=1.0,
            top_detected=True,
            side_detected=True,
            top_vote_count=8,
            side_vote_count=4,
            instance_count_hint=2,
        ),
        VoteResult(
            class_id=37,
            class_name="BOX_SAJO_OLD_LUNCHBOX_JAJANGBAP_250G",
            vote_count=68,
            max_confidence=0.8851,
            avg_confidence=0.48,
            weighted_confidence=0.4868,
            top_detected=True,
            top_vote_count=68,
            instance_count_hint=3,
        ),
    ]

    handled = VideoProcessor.filter_freezer_handled_candidates(
        vote_results,
        delta_weight=-84.2,
        product_weights={13: 185.0, 44: 93.0, 37: 307.0},
        trace_context=trace_context,
        log_prefix="TEST",
    )

    assert [candidate.class_id for candidate in handled] == [44]
    assert handled[0].instance_count_hint == 1
    assert trace_context.weight_diagnostics["freezer_candidate_filter"][
        "raw_candidate_count"
    ] == 3
    assert trace_context.weight_diagnostics["freezer_candidate_filter"]["selected"][
        "class_id"
    ] == 44
    assert (
        trace_context.weight_diagnostics["freezer_candidate_filter"]["reason"]
        == "single_removal_weight_tiebreak"
    )


def test_video_processor_freezer_filter_warns_when_dual_top_layout_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from model_service.core.config import config
    from model_service.video import VideoProcessor, VoteResult
    from model_service.video.frame_trace import TriggerTraceContext

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "legacy_top_side")
    caplog.set_level(logging.WARNING)

    trace_context = TriggerTraceContext(
        session_id="freezer-layout-warning",
        zone=2,
        top_path="/tmp/top.avi",
        side_path="/tmp/side.avi",
        log_dir=tmp_path / "logs",
        sample_export_dir=tmp_path / "samples",
        sample_export_enabled=False,
    )
    vote_results = [
        VoteResult(
            class_id=30,
            class_name="BAG_BIBIGO_CHEONGYANG_MEAT_DUMPLINGS_200G",
            vote_count=20,
            max_confidence=0.9,
            avg_confidence=0.8,
            weighted_confidence=0.9,
            top_detected=True,
            side_detected=True,
        ),
        VoteResult(
            class_id=44,
            class_name="STICK_BINGGRAE_MELONA_75ML",
            vote_count=8,
            max_confidence=0.7,
            avg_confidence=0.6,
            weighted_confidence=0.7,
            top_detected=True,
        ),
    ]

    handled = VideoProcessor.filter_freezer_handled_candidates(
        vote_results,
        delta_weight=-75.9,
        product_weights={30: 224.0, 44: 79.0},
        trace_context=trace_context,
        log_prefix="TEST",
    )

    assert handled == vote_results
    diagnostics = trace_context.weight_diagnostics["freezer_candidate_filter"]
    assert diagnostics["reason"] == "disabled_camera_layout"
    assert diagnostics["camera_layout"] == "legacy_top_side"
    assert diagnostics["freezer_handled_filter_enabled"] is False
    assert diagnostics["raw_candidate_count"] == 2
    assert diagnostics["handled_candidate_count"] == 2
    assert "expected=dual_top_proxy" in caplog.text


def test_freezer_candidate_filter_ops_logs_layout_counts_and_reason(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from model_service.core.config import config
    from model_service.service.trigger_service import TriggerService
    from model_service.video.frame_trace import TriggerTraceContext

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")
    caplog.set_level(logging.INFO)

    trace_context = TriggerTraceContext(
        session_id="freezer-filter-ops",
        zone=4,
        top_path="/tmp/top.avi",
        side_path="/tmp/side.avi",
        log_dir=tmp_path / "logs",
        sample_export_dir=tmp_path / "samples",
        sample_export_enabled=False,
    )
    trace_context.record_weight_diagnostics(
        {
            "freezer_candidate_filter": {
                "freezer_handled_filter_enabled": True,
                "raw_candidate_count": 4,
                "handled_candidate_count": 1,
                "reason": "single_removal_weight_tiebreak",
            }
        }
    )

    TriggerService._log_freezer_candidate_filter_ops(
        zone=4,
        raw_count=4,
        handled_count=1,
        delta_weight=-75.9,
        trace_context=trace_context,
    )

    assert "[OPS][FREEZER-CANDIDATE-FILTER] zone=4" in caplog.text
    assert "camera_layout=dual_top_proxy" in caplog.text
    assert "raw=4 handled=1" in caplog.text
    assert "reason=single_removal_weight_tiebreak" in caplog.text


def test_video_processor_freezer_handled_filter_preserves_strong_multi_vision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from model_service.core.config import config
    from model_service.video import VideoProcessor, VoteResult
    from model_service.video.frame_trace import TriggerTraceContext

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")
    monkeypatch.setattr(
        config.weight,
        "freezer_vision_multi_without_weight_enabled",
        True,
    )

    trace_context = TriggerTraceContext(
        session_id="freezer-vision-multi",
        zone=1,
        top_path="/tmp/top.avi",
        side_path="/tmp/side.avi",
        log_dir=tmp_path / "logs",
        sample_export_dir=tmp_path / "samples",
        sample_export_enabled=False,
    )
    trace_context.stage_counts_by_class = {
        "41": {
            "class_id": 41,
            "name": "FREEZER_A",
            "freezerExitPathVotes": 8,
            "freezer_roi_filtered_max_confidence": 0.94,
            "cameras": {
                "top": {"freezerExitPathVotes": 4},
                "side": {"freezerExitPathVotes": 4},
            },
        },
        "42": {
            "class_id": 42,
            "name": "FREEZER_B",
            "freezerExitPathVotes": 6,
            "freezer_roi_filtered_max_confidence": 0.9,
            "cameras": {
                "top": {"freezerExitPathVotes": 3},
                "side": {"freezerExitPathVotes": 3},
            },
        },
    }
    vote_results = [
        VoteResult(
            class_id=41,
            class_name="FREEZER_A",
            vote_count=12,
            max_confidence=0.94,
            avg_confidence=0.9,
            weighted_confidence=0.94,
            top_detected=True,
            side_detected=True,
            top_vote_count=6,
            side_vote_count=6,
        ),
        VoteResult(
            class_id=42,
            class_name="FREEZER_B",
            vote_count=10,
            max_confidence=0.9,
            avg_confidence=0.86,
            weighted_confidence=0.9,
            top_detected=True,
            side_detected=True,
            top_vote_count=5,
            side_vote_count=5,
        ),
        VoteResult(
            class_id=99,
            class_name="STATIC_NOISE",
            vote_count=9,
            max_confidence=0.88,
            avg_confidence=0.8,
            weighted_confidence=0.88,
            top_detected=True,
            top_vote_count=9,
        ),
    ]

    handled = VideoProcessor.filter_freezer_handled_candidates(
        vote_results,
        delta_weight=-999.0,
        product_weights={41: 100.0, 42: 110.0, 99: 999.0},
        trace_context=trace_context,
        log_prefix="TEST",
    )

    assert [candidate.class_id for candidate in handled] == [41, 42]
    diagnostics = trace_context.weight_diagnostics["freezer_candidate_filter"]
    assert diagnostics["reason"] == "freezer_multi_kind_vision_passthrough"
    assert diagnostics["handled_candidate_count"] == 2
    assert diagnostics["selectedClassIds"] == [41, 42]


def test_video_processor_freezer_exit_path_prefers_melona_over_static_lala(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from model_service.core.config import config
    from model_service.video import VideoProcessor, VoteResult
    from model_service.video.frame_trace import TriggerTraceContext

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")

    trace_context = TriggerTraceContext(
        session_id="freezer-melona-exit-path",
        zone=2,
        top_path="/tmp/top.avi",
        side_path="/tmp/side.avi",
        log_dir=tmp_path / "logs",
        sample_export_dir=tmp_path / "samples",
        sample_export_enabled=False,
    )
    trace_context.stage_counts_by_class = {
        "46": {"class_id": 46, "name": "STICK_LALA", "freezer_roi_filtered": 1},
        "44": {
            "class_id": 44,
            "name": "STICK_BINGGRAE_MELONA_75ML",
            "freezer_roi_filtered": 19,
        },
    }

    handled = VideoProcessor.filter_freezer_handled_candidates(
        [
            VoteResult(
                class_id=46,
                class_name="STICK_LALA_SWEET_GRAPE_ZERO_70ML",
                vote_count=153,
                max_confidence=0.928,
                avg_confidence=0.82,
                weighted_confidence=1.0,
                top_detected=True,
                side_detected=True,
                instance_count_hint=4,
            ),
            VoteResult(
                class_id=44,
                class_name="STICK_BINGGRAE_MELONA_75ML",
                vote_count=20,
                max_confidence=0.9164,
                avg_confidence=0.88,
                weighted_confidence=1.0,
                top_detected=True,
                side_detected=True,
                instance_count_hint=2,
            ),
        ],
        delta_weight=-81.0,
        product_weights={46: 71.0, 44: 93.0},
        trace_context=trace_context,
        log_prefix="TEST",
    )

    diagnostics = trace_context.weight_diagnostics["freezer_candidate_filter"]
    assert [candidate.class_id for candidate in handled] == [44]
    assert handled[0].instance_count_hint == 1
    assert diagnostics["reason"] == "near_weight_exit_path"
    assert diagnostics["selected"]["freezerExitPathVotes"] == 19


def test_video_processor_freezer_exit_path_prefers_cup_weight_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from model_service.core.config import config
    from model_service.video import VideoProcessor, VoteResult
    from model_service.video.frame_trace import TriggerTraceContext

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")
    monkeypatch.setattr(config.vision, "top_k", 6)

    trace_context = TriggerTraceContext(
        session_id="freezer-cup-exit-path",
        zone=1,
        top_path="/tmp/top.avi",
        side_path="/tmp/side.avi",
        log_dir=tmp_path / "logs",
        sample_export_dir=tmp_path / "samples",
        sample_export_enabled=False,
    )
    trace_context.stage_counts_by_class = {
        "30": {
            "class_id": 30,
            "name": "BOX_BINGGRAE_YOMAMTE_150ML",
            "raw": 31,
            "raw_max_confidence": 0.5891,
            "threshold_passed": 8,
            "threshold_passed_max_confidence": 0.5891,
            "freezer_roi_filtered": 8,
            "freezerExitPathVotes": 8,
            "freezer_roi_filtered_max_confidence": 0.5891,
            "unit_weight_g": 87.0,
            "roi_x_avg": 463.0,
            "roi_y_avg": 128.7,
            "cameras": {
                "top": {
                    "threshold_passed": 8,
                    "freezer_roi_filtered": 8,
                    "freezerExitPathVotes": 8,
                    "raw_max_confidence": 0.5891,
                },
                "side": {"raw": 6, "raw_max_confidence": 0.1214},
            },
        },
        "46": {"class_id": 46, "name": "STICK_LALA", "freezer_roi_filtered": 0},
        "44": {
            "class_id": 44,
            "name": "STICK_BINGGRAE_MELONA_75ML",
            "freezer_roi_filtered": 4,
        },
        "42": {
            "class_id": 42,
            "name": "CUP_MAEIL_SANGHAFARM_MILK_ICE_CREAMG_100G",
            "freezer_roi_filtered": 13,
        },
    }

    handled = VideoProcessor.filter_freezer_handled_candidates(
        [
            VoteResult(46, "STICK_LALA_SWEET_GRAPE_ZERO_70ML", 137, 0.8803, 0.74, weighted_confidence=1.0),
            VoteResult(37, "BOX_SAJO_OLD_LUNCHBOX_JAJANGBAP_250G", 82, 0.8601, 0.7, weighted_confidence=1.0),
            VoteResult(13, "BAG_COOZROCK_JUICY_MEAT_DUMPLING_168G", 36, 0.878, 0.72, weighted_confidence=1.0),
            VoteResult(44, "STICK_BINGGRAE_MELONA_75ML", 4, 0.8736, 0.52, weighted_confidence=0.5242),
            VoteResult(24, "BAG_JACKSONVILLE_BIG_HOT_DOG_115G", 44, 0.7839, 0.45, weighted_confidence=0.4311),
            VoteResult(42, "CUP_MAEIL_SANGHAFARM_MILK_ICE_CREAMG_100G", 3, 0.8253, 0.4, weighted_confidence=0.4011),
        ],
        delta_weight=-97.9,
        product_weights={46: 71.0, 37: 307.0, 13: 185.0, 44: 93.0, 24: 154.0, 42: 93.0, 30: 87.0},
        trace_context=trace_context,
        log_prefix="TEST",
    )

    diagnostics = trace_context.weight_diagnostics["freezer_candidate_filter"]
    assert [candidate.class_id for candidate in handled] == [42]
    assert diagnostics["reason"] == "weight_gate_exit_path"
    assert diagnostics["selected"]["freezerExitPathVotes"] == 13


def test_video_processor_freezer_stage_only_rescues_yomamte_dual_camera(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from model_service.core.config import config
    from model_service.video import VideoProcessor, VoteResult
    from model_service.video.frame_trace import TriggerTraceContext

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")
    monkeypatch.setattr(config.vision, "top_k", 5)

    trace_context = TriggerTraceContext(
        session_id="freezer-yomamte-stage-only",
        zone=2,
        top_path="/tmp/top.avi",
        side_path="/tmp/side.avi",
        log_dir=tmp_path / "logs",
        sample_export_dir=tmp_path / "samples",
        sample_export_enabled=False,
    )
    trace_context.stage_counts_by_class = {
        "30": {
            "class_id": 30,
            "name": "BOX_BINGGRAE_YOMAMTE_150ML",
            "raw": 47,
            "raw_max_confidence": 0.5761,
            "threshold_passed": 11,
            "threshold_passed_max_confidence": 0.5761,
            "freezer_roi_filtered": 11,
            "freezerExitPathVotes": 11,
            "freezer_roi_filtered_max_confidence": 0.5761,
            "unit_weight_g": 87.0,
            "roi_x_avg": 369.8,
            "roi_y_avg": 73.0,
            "cameras": {
                "top": {
                    "raw": 34,
                    "threshold_passed": 10,
                    "freezer_roi_filtered": 10,
                    "freezerExitPathVotes": 10,
                    "raw_max_confidence": 0.5761,
                },
                "side": {
                    "raw": 13,
                    "threshold_passed": 1,
                    "freezer_roi_filtered": 1,
                    "freezerExitPathVotes": 1,
                    "raw_max_confidence": 0.5291,
                },
            },
        },
        "42": {
            "class_id": 42,
            "name": "CUP_MAEIL_SANGHAFARM_MILK_ICE_CREAMG_100G",
            "raw": 57,
            "raw_max_confidence": 0.6808,
            "threshold_passed": 15,
            "threshold_passed_max_confidence": 0.6808,
            "freezer_roi_filtered": 14,
            "freezerExitPathVotes": 14,
            "freezer_roi_filtered_max_confidence": 0.6808,
            "unit_weight_g": 93.0,
            "roi_x_avg": 380.1,
            "roi_y_avg": 66.9,
            "cameras": {
                "top": {
                    "raw": 50,
                    "threshold_passed": 14,
                    "freezer_roi_filtered": 14,
                    "freezerExitPathVotes": 14,
                    "raw_max_confidence": 0.6808,
                },
                "side": {"raw": 7, "threshold_passed": 1, "raw_max_confidence": 0.335},
            },
        },
        "44": {
            "class_id": 44,
            "name": "STICK_BINGGRAE_MELONA_75ML",
            "raw": 26,
            "raw_max_confidence": 0.666,
            "threshold_passed": 8,
            "freezer_roi_filtered": 5,
            "freezerExitPathVotes": 5,
            "unit_weight_g": 93.0,
            "roi_x_avg": 438.0,
            "roi_y_avg": 92.5,
            "cameras": {
                "top": {
                    "threshold_passed": 8,
                    "freezer_roi_filtered": 5,
                    "freezerExitPathVotes": 5,
                    "raw_max_confidence": 0.666,
                }
            },
        },
    }

    handled = VideoProcessor.filter_freezer_handled_candidates(
        [
            VoteResult(46, "STICK_LALA_SWEET_GRAPE_ZERO_70ML", 184, 0.8941, 0.82, weighted_confidence=1.0),
            VoteResult(13, "BAG_COOZROCK_JUICY_MEAT_DUMPLING_168G", 44, 0.878, 0.75, weighted_confidence=0.7469),
            VoteResult(37, "BOX_SAJO_OLD_LUNCHBOX_JAJANGBAP_250G", 118, 0.7849, 0.62, weighted_confidence=0.4317),
            VoteResult(24, "BAG_JACKSONVILLE_BIG_HOT_DOG_115G", 44, 0.7839, 0.45, weighted_confidence=0.341),
            VoteResult(44, "STICK_BINGGRAE_MELONA_75ML", 3, 0.666, 0.153, weighted_confidence=0.153),
        ],
        delta_weight=-96.3,
        product_weights={46: 71.0, 13: 185.0, 37: 307.0, 24: 154.0, 44: 93.0, 30: 87.0, 42: 93.0},
        trace_context=trace_context,
        log_prefix="TEST",
    )

    diagnostics = trace_context.weight_diagnostics["freezer_candidate_filter"]
    assert [candidate.class_id for candidate in handled] == [30]
    assert handled[0].source == "freezer_stage_exit_path"
    assert diagnostics["reason"] == "ambiguous_dual_camera_stage_exit_path"
    assert diagnostics["selected"]["stageOnly"] is True
    assert diagnostics["selected"]["dualCameraExitPath"] is True
    assert {item["class_id"] for item in diagnostics["ambiguousCandidates"]} == {
        30,
        42,
        44,
    }


def test_video_processor_freezer_stage_only_replaces_low_confidence_melona(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from model_service.core.config import config
    from model_service.video import VideoProcessor, VoteResult
    from model_service.video.frame_trace import TriggerTraceContext

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")
    monkeypatch.setattr(config.vision, "top_k", 5)

    trace_context = TriggerTraceContext(
        session_id="freezer-yomamte-low-confidence-melona",
        zone=2,
        top_path="/tmp/top.avi",
        side_path="/tmp/side.avi",
        log_dir=tmp_path / "logs",
        sample_export_dir=tmp_path / "samples",
        sample_export_enabled=False,
    )
    trace_context.stage_counts_by_class = {
        "30": {
            "class_id": 30,
            "name": "BOX_BINGGRAE_YOMAMTE_150ML",
            "raw": 60,
            "raw_max_confidence": 0.8223,
            "threshold_passed": 13,
            "threshold_passed_max_confidence": 0.8223,
            "freezer_roi_filtered": 10,
            "freezerExitPathVotes": 10,
            "freezer_roi_filtered_max_confidence": 0.7808,
            "unit_weight_g": 87.0,
            "roi_x_avg": 420.9,
            "roi_y_avg": 69.5,
            "cameras": {
                "top": {
                    "raw": 40,
                    "threshold_passed": 11,
                    "freezer_roi_filtered": 10,
                    "freezerExitPathVotes": 10,
                    "raw_max_confidence": 0.8223,
                },
                "side": {"raw": 20, "threshold_passed": 2, "raw_max_confidence": 0.7047},
            },
        },
        "44": {
            "class_id": 44,
            "name": "STICK_BINGGRAE_MELONA_75ML",
            "raw": 78,
            "raw_max_confidence": 0.7111,
            "threshold_passed": 13,
            "freezer_roi_filtered": 9,
            "freezerExitPathVotes": 9,
            "unit_weight_g": 93.0,
            "roi_x_avg": 441.8,
            "roi_y_avg": 119.4,
            "cameras": {
                "top": {"freezer_roi_filtered": 7, "freezerExitPathVotes": 7},
                "side": {"freezer_roi_filtered": 2, "freezerExitPathVotes": 2},
            },
        },
    }

    handled = VideoProcessor.filter_freezer_handled_candidates(
        [
            VoteResult(
                44,
                "STICK_BINGGRAE_MELONA_75ML",
                4,
                0.5983,
                0.2411,
                weighted_confidence=0.2411,
                top_detected=True,
                top_vote_count=4,
                freezer_exit_path_votes=9,
            )
        ],
        delta_weight=-94.7,
        product_weights={44: 93.0, 30: 87.0},
        trace_context=trace_context,
        log_prefix="TEST",
    )

    diagnostics = trace_context.weight_diagnostics["freezer_candidate_filter"]
    assert [candidate.class_id for candidate in handled] == [30]
    assert handled[0].source == "freezer_stage_exit_path"
    assert diagnostics["selected"]["source"] == "freezer_stage_exit_path"
    assert diagnostics["selected"]["identitySupported"] is True
    melona = next(
        item for item in diagnostics["considered"] if item["class_id"] == 44
    )
    assert melona["identitySupported"] is False


@pytest.mark.asyncio
async def test_async_video_processor_top_roi_return_keeps_lower_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import model_service.video.video_processor as video_processor_module
    from model_service.video import VideoProcessor

    class FakeAsyncExtractor:
        total_frames = 1
        last_diagnostics = None

        async def __aiter__(self) -> AsyncIterator[int]:
            yield 0

    def create_fake_extractor(*_args: object, **_kwargs: object) -> FakeAsyncExtractor:
        return FakeAsyncExtractor()

    monkeypatch.setattr(
        video_processor_module,
        "create_frame_extractor",
        create_fake_extractor,
    )
    processor = VideoProcessor(
        yolo=TopRoiFakeYolo(),
        motion_filter_enabled=False,
        hand_path_filter_enabled=False,
        min_vote_count=1,
    )

    result = await processor.process_videos_async(
        top_path="/tmp/top.avi",
        allowed_class_ids=[52, 53],
        delta_weight=50.0,
    )

    assert [candidate.class_id for candidate in result.vote_results] == [53]
    assert result.stats.roi_filtered_detections == 1
    assert result.roi_rescue_candidates == []


def test_video_processor_records_optional_all_class_diagnostic_trace(monkeypatch, tmp_path):
    import json

    import model_service.video.video_processor as video_processor_module
    from model_service.core.config import config
    from model_service.video import VideoProcessor
    from model_service.video.frame_trace import TriggerTraceContext
    from model_service.vision.yolo_wrapper import YOLODetection

    class FakeExtractor:
        total_frames = 1
        last_diagnostics = None

        def __iter__(self):
            return iter([[[0]]])

    class FakeYolo:
        last_preprocess = {}

        def detect(self, frame, allowed_class_ids=None, camera_type=None):
            if allowed_class_ids == []:
                return []
            return [
                YOLODetection(
                    xyxy=(10.0, 10.0, 30.0, 30.0),
                    cls=96,
                    conf=0.64,
                    name="BAG_JAYEONLU_MOIST_SWEET_CHESTNUT_80G",
                )
            ]

    monkeypatch.setattr(config.vision, "diagnostic_all_class_trace", True, raising=False)
    monkeypatch.setattr(config.vision, "diagnostic_trace_max_frames", 1, raising=False)
    monkeypatch.setattr(
        video_processor_module,
        "create_frame_extractor",
        lambda *args, **kwargs: FakeExtractor(),
    )

    trace_context = TriggerTraceContext(
        session_id="diagnostic-all-class",
        zone=1,
        top_path="/tmp/top.avi",
        side_path=None,
        log_dir=tmp_path / "logs",
        sample_export_dir=tmp_path / "samples",
    )
    processor = VideoProcessor(
        yolo=FakeYolo(),
        motion_filter_enabled=False,
        hand_path_filter_enabled=False,
    )

    result = processor.process_videos(
        top_path="/tmp/top.avi",
        allowed_class_ids=[],
        trace_context=trace_context,
    )
    trace_context.finalize(status="complete")

    assert result.vote_results == []
    detail_file = next((tmp_path / "logs" / "triggers").glob("*/*.json"))
    detail = json.loads(detail_file.read_text(encoding="utf-8"))
    assert detail["diagnostic_candidates"][0]["name"] == "BAG_JAYEONLU_MOIST_SWEET_CHESTNUT_80G"
    assert detail["candidates"] == []


def test_video_processor_records_moving_threshold_rescue_candidate(monkeypatch, tmp_path):
    import json

    import model_service.video.video_processor as video_processor_module
    from model_service.core.config import config
    from model_service.video import VideoProcessor
    from model_service.video.frame_trace import TriggerTraceContext
    from model_service.vision.yolo_wrapper import YOLODetection

    class FakeExtractor:
        total_frames = 2
        last_diagnostics = None

        def __iter__(self):
            return iter([0, 1])

    class FakeYolo:
        last_preprocess = {
            "crop_policy": "left",
            "crop_box": {"x1": 0, "y1": 0, "x2": 480, "y2": 480},
        }

        def detect(self, frame, allowed_class_ids=None, camera_type=None):
            shift = int(frame) * 40.0
            confidence = 0.19 if camera_type == "top" else 0.16
            return [
                YOLODetection(
                    xyxy=(10.0 + shift, 10.0, 40.0 + shift, 40.0),
                    cls=75,
                    conf=confidence,
                    name="BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
                )
            ]

    monkeypatch.setattr(config.vision, "threshold_rescue_enabled", True, raising=False)
    monkeypatch.setattr(config.vision, "threshold_rescue_require_motion", True, raising=False)
    monkeypatch.setattr(config.vision, "threshold_rescue_max_candidates", 20, raising=False)
    monkeypatch.setattr(
        video_processor_module,
        "create_frame_extractor",
        lambda *args, **kwargs: FakeExtractor(),
    )

    trace_context = TriggerTraceContext(
        session_id="threshold-rescue",
        zone=4,
        top_path="/tmp/top.avi",
        side_path="/tmp/side.avi",
        log_dir=tmp_path / "logs",
        sample_export_dir=tmp_path / "samples",
        sample_export_enabled=False,
    )
    processor = VideoProcessor(
        yolo=FakeYolo(),
        top_confidence_threshold=0.5,
        side_confidence_threshold=0.5,
        motion_filter_enabled=False,
        hand_path_filter_enabled=False,
        min_motion_displacement=5.0,
        min_vote_count=1,
    )

    result = processor.process_videos(
        top_path="/tmp/top.avi",
        side_path="/tmp/side.avi",
        allowed_class_ids=[75],
        product_weights={75: 516.8},
        trace_context=trace_context,
    )
    trace_context.finalize(status="complete")

    assert result.vote_results == []
    assert len(result.threshold_rescue_candidates) == 1
    rescue = result.threshold_rescue_candidates[0]
    assert rescue.class_id == 75
    assert rescue.top_detected is True
    assert rescue.side_detected is True
    assert rescue.top_motion_passed is True
    assert rescue.side_motion_passed is True

    detail_file = next((tmp_path / "logs" / "triggers").glob("*/*.json"))
    detail = json.loads(detail_file.read_text(encoding="utf-8"))
    assert detail["vision_config"]["threshold_rescue_enabled"] is True
    assert detail["threshold_rescue_candidates"][0]["class_id"] == 75
    assert detail["threshold_rescue_candidates"][0]["unit_weight_g"] == 516.8
    assert detail["stage_counts_by_class"]["75"]["threshold_filtered"] == 4
    assert detail["stage_counts_by_class"]["75"]["threshold_filtered_max_confidence"] == 0.19
    assert detail["stage_counts_by_class"]["75"]["threshold_rescue_motion"]["top"]["passed"] is True


def test_threshold_rescue_rejects_roi_conflicted_low_confidence_class(monkeypatch):
    from types import SimpleNamespace

    import model_service.video.video_processor as video_processor_module
    from model_service.core.config import config
    from model_service.video import VideoProcessor

    class FakeExtractor:
        total_frames = 3
        last_diagnostics = None

        def __iter__(self):
            return iter([0, 1, 2])

    class FakeYolo:
        last_preprocess = {
            "crop_policy": "left",
            "crop_box": {"x1": 0, "y1": 0, "x2": 480, "y2": 480},
        }

        def detect(self, frame, allowed_class_ids=None, camera_type=None):
            if camera_type != "side":
                return []
            shift = int(frame) * 55.0
            return [
                YOLODetection(
                    xyxy=(390.0, 250.0, 430.0, 310.0),
                    cls=54,
                    conf=0.89,
                    name="BOTTLE_LOTTE_TREVI_LEMON_500ML",
                ),
                YOLODetection(
                    xyxy=(20.0 + shift, 260.0, 50.0 + shift, 300.0),
                    cls=54,
                    conf=0.14,
                    name="BOTTLE_LOTTE_TREVI_LEMON_500ML",
                ),
            ]

    monkeypatch.setattr(config.vision, "threshold_rescue_enabled", True, raising=False)
    monkeypatch.setattr(config.vision, "threshold_rescue_require_motion", True, raising=False)
    monkeypatch.setattr(config.vision, "side_roi_x_max", 400.0, raising=False)
    monkeypatch.setattr(
        video_processor_module,
        "create_frame_extractor",
        lambda *args, **kwargs: FakeExtractor(),
    )

    processor = VideoProcessor(
        yolo=FakeYolo(),
        top_confidence_threshold=0.25,
        side_confidence_threshold=0.25,
        motion_filter_enabled=False,
        hand_path_filter_enabled=False,
        min_motion_displacement=5.0,
        min_vote_count=1,
    )

    result = processor.process_videos(
        side_path="/tmp/side.avi",
        allowed_class_ids=[54],
        product_weights={54: 530.0},
        delta_weight=-529.0,
    )

    assert result.vote_results == []
    assert len(result.threshold_rescue_candidates) == 1
    rescue = result.threshold_rescue_candidates[0]
    assert rescue.class_id == 54
    assert rescue.roi_conflict is True
    assert rescue.roi_conflict_reason == "side_roi_filtered_stronger_evidence"

    diagnostics = {}
    rescue_votes = VideoProcessor.build_weight_gated_rescue_votes(
        result.threshold_rescue_candidates,
        [
            SimpleNamespace(
                yolo_class_id=54,
                product_name="BOTTLE_LOTTE_TREVI_LEMON_500ML",
                product_weight=530.0,
                stock_qty=10,
            )
        ],
        -529.0,
        diagnostics=diagnostics,
    )

    assert rescue_votes == []
    assert diagnostics["rejections"]["threshold_rescue_roi_conflict"] == 1
    assert diagnostics["candidates"][0]["threshold_rescue_rejected_reason"] == (
        "side_roi_filtered_stronger_evidence"
    )


def test_video_processor_does_not_rescue_static_threshold_filtered_class(monkeypatch):
    import model_service.video.video_processor as video_processor_module
    from model_service.core.config import config
    from model_service.video import VideoProcessor
    from model_service.vision.yolo_wrapper import YOLODetection

    class FakeExtractor:
        total_frames = 2
        last_diagnostics = None

        def __iter__(self):
            return iter([0, 1])

    class FakeYolo:
        last_preprocess = {}

        def detect(self, frame, allowed_class_ids=None, camera_type=None):
            return [
                YOLODetection(
                    xyxy=(10.0, 10.0, 40.0, 40.0),
                    cls=75,
                    conf=0.19,
                    name="BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
                )
            ]

    monkeypatch.setattr(config.vision, "threshold_rescue_enabled", True, raising=False)
    monkeypatch.setattr(config.vision, "threshold_rescue_require_motion", True, raising=False)
    monkeypatch.setattr(
        video_processor_module,
        "create_frame_extractor",
        lambda *args, **kwargs: FakeExtractor(),
    )

    processor = VideoProcessor(
        yolo=FakeYolo(),
        top_confidence_threshold=0.5,
        side_confidence_threshold=0.5,
        motion_filter_enabled=False,
        hand_path_filter_enabled=False,
        min_motion_displacement=5.0,
        min_vote_count=1,
    )

    result = processor.process_videos(
        top_path="/tmp/top.avi",
        allowed_class_ids=[75],
        product_weights={75: 516.8},
    )

    assert result.vote_results == []
    assert result.threshold_rescue_candidates == []


def test_weight_gated_no_motion_threshold_rescue_can_recover_tight_weight_match(
    monkeypatch,
):
    from types import SimpleNamespace

    import model_service.video.video_processor as video_processor_module
    from model_service.core.config import config
    from model_service.video import VideoProcessor
    from model_service.vision.yolo_wrapper import YOLODetection

    class FakeExtractor:
        total_frames = 9
        last_diagnostics = None

        def __iter__(self):
            return iter(range(9))

    class FakeYolo:
        last_preprocess = {}

        def detect(self, frame, allowed_class_ids=None, camera_type=None):
            return [
                YOLODetection(
                    xyxy=(10.0, 10.0, 40.0, 40.0),
                    cls=95,
                    conf=0.13,
                    name="BOTTLE_HY_HALUYACHE_PURPLE_200ML_V2",
                )
            ]

    monkeypatch.setattr(config.vision, "threshold_rescue_enabled", True, raising=False)
    monkeypatch.setattr(config.vision, "threshold_rescue_require_motion", True, raising=False)
    monkeypatch.setattr(config.vision, "weight_rescue_no_motion_enabled", True, raising=False)
    monkeypatch.setattr(config.vision, "weight_rescue_no_motion_min_raw_votes", 8, raising=False)
    monkeypatch.setattr(
        config.vision,
        "weight_rescue_no_motion_max_residual_grams",
        2.0,
        raising=False,
    )
    monkeypatch.setattr(config.weight, "rescue_tolerance_grams", 5.0, raising=False)
    monkeypatch.setattr(
        video_processor_module,
        "create_frame_extractor",
        lambda *args, **kwargs: FakeExtractor(),
    )

    processor = VideoProcessor(
        yolo=FakeYolo(),
        top_confidence_threshold=0.5,
        side_confidence_threshold=0.5,
        motion_filter_enabled=False,
        hand_path_filter_enabled=False,
        min_motion_displacement=5.0,
        min_vote_count=1,
    )

    result = processor.process_videos(
        top_path="/tmp/top.avi",
        allowed_class_ids=[95],
        product_weights={95: 220.0},
    )

    assert result.vote_results == []
    assert len(result.threshold_rescue_candidates) == 1
    rescue = result.threshold_rescue_candidates[0]
    assert rescue.class_id == 95
    assert rescue.vote_count == 9
    assert rescue.motion_gate_passed is False

    diagnostics = {}
    rescue_votes = VideoProcessor.build_weight_gated_rescue_votes(
        result.threshold_rescue_candidates,
        [
            SimpleNamespace(
                yolo_class_id=95,
                product_name="BOTTLE_HY_HALUYACHE_PURPLE_200ML_V2",
                stock_qty=1,
                product_weight=220.0,
                sale_price=1800,
            )
        ],
        delta_weight=-221.0,
        diagnostics=diagnostics,
    )

    assert len(rescue_votes) == 1
    assert rescue_votes[0].class_id == 95
    assert rescue_votes[0].source == "threshold_rescue"
    assert rescue_votes[0].motion_gate_passed is False
    assert rescue_votes[0].weight_gate_passed is True
    assert rescue_votes[0].rescue_tolerance_g == 5.0
    assert diagnostics["accepted"] == 1
    assert diagnostics["candidates"][0]["weight_gate_passed"] is True
    assert diagnostics["candidates"][0]["motion_gate_passed"] is False


def test_weight_gated_no_motion_threshold_rescue_rejects_loose_weight_match(
    monkeypatch,
):
    from types import SimpleNamespace

    import model_service.video.video_processor as video_processor_module
    from model_service.core.config import config
    from model_service.video import VideoProcessor
    from model_service.vision.yolo_wrapper import YOLODetection

    class FakeExtractor:
        total_frames = 9
        last_diagnostics = None

        def __iter__(self):
            return iter(range(9))

    class FakeYolo:
        last_preprocess = {}

        def detect(self, frame, allowed_class_ids=None, camera_type=None):
            return [
                YOLODetection(
                    xyxy=(10.0, 10.0, 40.0, 40.0),
                    cls=95,
                    conf=0.13,
                    name="BOTTLE_HY_HALUYACHE_PURPLE_200ML_V2",
                )
            ]

    monkeypatch.setattr(config.vision, "threshold_rescue_enabled", True, raising=False)
    monkeypatch.setattr(config.vision, "threshold_rescue_require_motion", True, raising=False)
    monkeypatch.setattr(config.vision, "weight_rescue_no_motion_enabled", True, raising=False)
    monkeypatch.setattr(config.vision, "weight_rescue_no_motion_min_raw_votes", 8, raising=False)
    monkeypatch.setattr(
        config.vision,
        "weight_rescue_no_motion_max_residual_grams",
        2.0,
        raising=False,
    )
    monkeypatch.setattr(config.weight, "rescue_tolerance_grams", 5.0, raising=False)
    monkeypatch.setattr(
        video_processor_module,
        "create_frame_extractor",
        lambda *args, **kwargs: FakeExtractor(),
    )

    processor = VideoProcessor(
        yolo=FakeYolo(),
        top_confidence_threshold=0.5,
        side_confidence_threshold=0.5,
        motion_filter_enabled=False,
        hand_path_filter_enabled=False,
        min_motion_displacement=5.0,
        min_vote_count=1,
    )

    result = processor.process_videos(
        top_path="/tmp/top.avi",
        allowed_class_ids=[95],
        product_weights={95: 218.0},
    )
    diagnostics = {}
    rescue_votes = VideoProcessor.build_weight_gated_rescue_votes(
        result.threshold_rescue_candidates,
        [
            SimpleNamespace(
                yolo_class_id=95,
                product_name="BOTTLE_HY_HALUYACHE_PURPLE_200ML_V2",
                stock_qty=1,
                product_weight=218.0,
                sale_price=1800,
            )
        ],
        delta_weight=-221.0,
        diagnostics=diagnostics,
    )

    assert rescue_votes == []
    assert diagnostics["accepted"] == 0
    assert diagnostics["candidates"][0]["reason"] == "motion_rejected_but_weight_matched"
    assert diagnostics["candidates"][0]["rescue_weight_residual_g"] == 3.0


def test_video_processor_records_roi_rescue_candidate_for_weight_gated_side_detection(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace

    import model_service.video.video_processor as video_processor_module
    from model_service.core.config import config
    from model_service.video import VideoProcessor
    from model_service.video.frame_trace import TriggerTraceContext
    from model_service.vision.yolo_wrapper import YOLODetection

    class FakeExtractor:
        total_frames = 2
        last_diagnostics = None

        def __iter__(self):
            return iter([0, 1])

    class FakeYolo:
        last_preprocess = {}

        def detect(self, frame, allowed_class_ids=None, camera_type=None):
            shift = int(frame) * 16.0
            return [
                YOLODetection(
                    xyxy=(348.0 + shift, 40.0, 388.0 + shift, 100.0),
                    cls=8,
                    conf=0.64,
                    name="CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
                )
            ]

    monkeypatch.setattr(config.vision, "threshold_rescue_enabled", True, raising=False)
    monkeypatch.setattr(config.vision, "threshold_rescue_max_candidates", 20, raising=False)
    monkeypatch.setattr(config.vision, "roi_rescue_max_over_limit_px", 20.0, raising=False)
    monkeypatch.setattr(
        video_processor_module,
        "create_frame_extractor",
        lambda *args, **kwargs: FakeExtractor(),
    )

    trace_context = TriggerTraceContext(
        session_id="roi-rescue",
        zone=4,
        top_path=None,
        side_path="/tmp/side.avi",
        log_dir=tmp_path / "logs",
        sample_export_dir=tmp_path / "samples",
        sample_export_enabled=False,
    )
    processor = VideoProcessor(
        yolo=FakeYolo(),
        side_confidence_threshold=0.25,
        side_roi_x_max=360.0,
        motion_filter_enabled=False,
        hand_path_filter_enabled=False,
        min_vote_count=1,
    )

    result = processor.process_videos(
        side_path="/tmp/side.avi",
        allowed_class_ids=[8],
        product_weights={8: 369.0},
        trace_context=trace_context,
    )

    active_products = [
        SimpleNamespace(
            yolo_class_id=8,
            product_name="CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
            stock_qty=1,
            product_weight=369.0,
            sale_price=2000,
        )
    ]
    rescue_votes = VideoProcessor.build_weight_gated_rescue_votes(
        result.roi_rescue_candidates,
        active_products,
        delta_weight=-369.0,
    )
    trace_context.record_roi_rescue_candidates(result.roi_rescue_candidates, {8: 369.0})
    trace_context.finalize(status="complete")

    assert result.vote_results == []
    assert len(result.roi_rescue_candidates) == 1
    rescue = result.roi_rescue_candidates[0]
    assert rescue.class_id == 8
    assert rescue.source == "roi_rescue"
    assert rescue.side_detected is True
    assert rescue.roi_x_min > 360.0
    assert rescue.roi_x_avg <= rescue.roi_x_limit + 20.0
    assert rescue.roi_x_limit == 360.0

    assert len(rescue_votes) == 1
    assert rescue_votes[0].class_id == 8
    assert rescue_votes[0].source == "roi_rescue"

    detail_file = next((tmp_path / "logs" / "triggers").glob("*/*.json"))
    detail = json.loads(detail_file.read_text(encoding="utf-8"))
    assert detail["roi_rescue_candidates"][0]["class_id"] == 8
    assert detail["roi_rescue_candidates"][0]["source"] == "roi_rescue"
    assert detail["stage_counts_by_class"]["8"]["roi_filtered"] == 2
    assert detail["stage_counts_by_class"]["8"]["roi_x_max"] > 360.0


def test_video_processor_drops_static_far_right_roi_rescue_candidate(monkeypatch, tmp_path):
    import json

    import model_service.video.video_processor as video_processor_module
    from model_service.core.config import config
    from model_service.video import VideoProcessor
    from model_service.video.frame_trace import TriggerTraceContext
    from model_service.vision.yolo_wrapper import YOLODetection

    class FakeExtractor:
        total_frames = 2
        last_diagnostics = None

        def __iter__(self):
            return iter([0, 1])

    class FakeYolo:
        last_preprocess = {}

        def detect(self, frame, allowed_class_ids=None, camera_type=None):
            return [
                YOLODetection(
                    xyxy=(410.0, 40.0, 450.0, 100.0),
                    cls=8,
                    conf=0.64,
                    name="CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
                )
            ]

    monkeypatch.setattr(config.vision, "threshold_rescue_enabled", True, raising=False)
    monkeypatch.setattr(config.vision, "roi_rescue_require_motion", True, raising=False)
    monkeypatch.setattr(config.vision, "roi_rescue_max_over_limit_px", 20.0, raising=False)
    monkeypatch.setattr(
        video_processor_module,
        "create_frame_extractor",
        lambda *args, **kwargs: FakeExtractor(),
    )

    trace_context = TriggerTraceContext(
        session_id="roi-rescue-static-right",
        zone=4,
        top_path=None,
        side_path="/tmp/side.avi",
        log_dir=tmp_path / "logs",
        sample_export_dir=tmp_path / "samples",
        sample_export_enabled=False,
    )
    processor = VideoProcessor(
        yolo=FakeYolo(),
        side_confidence_threshold=0.30,
        side_roi_x_max=360.0,
        motion_filter_enabled=False,
        hand_path_filter_enabled=False,
        min_vote_count=1,
    )

    result = processor.process_videos(
        side_path="/tmp/side.avi",
        allowed_class_ids=[8],
        product_weights={8: 369.0},
        trace_context=trace_context,
    )
    trace_context.record_roi_rescue_candidates(result.roi_rescue_candidates, {8: 369.0})
    trace_context.finalize(status="complete")

    assert result.vote_results == []
    assert result.roi_rescue_candidates == []
    assert result.stats.roi_filtered_detections == 2

    detail_file = next((tmp_path / "logs" / "triggers").glob("*/*.json"))
    detail = json.loads(detail_file.read_text(encoding="utf-8"))
    assert detail["roi_rescue_candidates"] == []
    assert detail["stage_counts_by_class"]["8"]["roi_filtered"] == 2
    assert detail["stage_counts_by_class"]["8"]["roi_x_avg"] == 430.0
    assert detail["stage_counts_by_class"]["8"]["roi_x_limit"] == 360.0


def test_weight_gated_roi_rescue_rejects_no_motion_and_far_right(monkeypatch):
    from types import SimpleNamespace

    from model_service.core.config import config
    from model_service.video.video_processor import ThresholdRescueCandidate, VideoProcessor

    monkeypatch.setattr(config.vision, "roi_rescue_require_motion", True, raising=False)
    monkeypatch.setattr(config.vision, "roi_rescue_max_over_limit_px", 20.0, raising=False)

    active_products = [
        SimpleNamespace(
            yolo_class_id=8,
            product_name="CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
            stock_qty=1,
            product_weight=369.0,
            sale_price=2000,
        )
    ]
    static_candidate = ThresholdRescueCandidate(
        class_id=8,
        class_name="CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
        vote_count=2,
        max_confidence=0.64,
        avg_confidence=0.64,
        side_detected=True,
        side_vote_count=2,
        side_max_confidence=0.64,
        side_motion_passed=False,
        roi_x_avg=370.0,
        roi_x_limit=360.0,
        source="roi_rescue",
        motion_gate_passed=False,
    )
    far_right_candidate = ThresholdRescueCandidate(
        class_id=8,
        class_name="CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
        vote_count=2,
        max_confidence=0.64,
        avg_confidence=0.64,
        side_detected=True,
        side_vote_count=2,
        side_max_confidence=0.64,
        side_motion_passed=True,
        roi_x_avg=390.1,
        roi_x_limit=360.0,
        source="roi_rescue",
        motion_gate_passed=True,
    )
    diagnostics: dict = {}

    rescue_votes = VideoProcessor.build_weight_gated_rescue_votes(
        [static_candidate, far_right_candidate],
        active_products,
        delta_weight=-369.0,
        diagnostics=diagnostics,
    )

    assert rescue_votes == []
    assert diagnostics["accepted"] == 0
    assert diagnostics["rejections"]["roi_rescue_no_motion"] == 1
    assert diagnostics["rejections"]["roi_rescue_too_far_right"] == 1
    assert [entry["reason"] for entry in diagnostics["candidates"]] == [
        "roi_rescue_no_motion",
        "roi_rescue_too_far_right",
    ]


def test_weight_gated_roi_rescue_default_is_strict_at_side_roi_limit(monkeypatch):
    from types import SimpleNamespace

    from model_service.core.config import config
    from model_service.video.video_processor import ThresholdRescueCandidate, VideoProcessor

    monkeypatch.setattr(config.vision, "roi_rescue_require_motion", True, raising=False)
    monkeypatch.setattr(config.vision, "roi_rescue_max_over_limit_px", 0.0, raising=False)

    active_products = [
        SimpleNamespace(
            yolo_class_id=8,
            product_name="CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
            stock_qty=1,
            product_weight=369.0,
            sale_price=2000,
        )
    ]
    just_outside_candidate = ThresholdRescueCandidate(
        class_id=8,
        class_name="CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
        vote_count=2,
        max_confidence=0.64,
        avg_confidence=0.64,
        side_detected=True,
        side_vote_count=2,
        side_max_confidence=0.64,
        side_motion_passed=True,
        roi_x_avg=400.1,
        roi_x_limit=400.0,
        source="roi_rescue",
        motion_gate_passed=True,
    )
    diagnostics: dict = {}

    rescue_votes = VideoProcessor.build_weight_gated_rescue_votes(
        [just_outside_candidate],
        active_products,
        delta_weight=-369.0,
        diagnostics=diagnostics,
    )

    assert rescue_votes == []
    assert diagnostics["accepted"] == 0
    assert diagnostics["rejections"]["roi_rescue_too_far_right"] == 1
    assert diagnostics["candidates"][0]["reason"] == "roi_rescue_too_far_right"


def test_stage_threshold_evidence_can_drive_detected_single_fallback(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace

    import model_service.engine.decision_engine as decision_engine_module
    from model_service.engine import EnsembleResult, JudgmentStatus, ProductDecisionEngine
    from model_service.video.frame_trace import TriggerTraceContext

    monkeypatch.setattr(
        decision_engine_module.config.weight,
        "strict_mode_fallback",
        True,
    )
    trace_context = TriggerTraceContext(
        session_id="stage-fallback",
        zone=3,
        top_path=None,
        side_path="/tmp/side.avi",
        log_dir=tmp_path / "logs",
        sample_export_dir=tmp_path / "samples",
        sample_export_enabled=False,
    )
    trace_context.record_stage_count(
        class_id=114,
        class_name="BOX_LOTTE_PEPERO_ORIGINAL_46G",
        stage="threshold_filtered",
        camera="side",
        amount=10,
        confidence=0.1854,
    )

    result = ProductDecisionEngine(strict_mode=True).judge(
        vision_candidates=[
            EnsembleResult(
                class_id=115,
                class_name="BOX_LOTTE_PEPERO_ALMOND_37G",
                top_confidence=0.2303,
                side_confidence=0.0,
                combined_confidence=0.2303,
                vote_count=1,
            )
        ],
        delta_weight=-71.6,
        active_products=[
            SimpleNamespace(
                yolo_class_id=114,
                product_name="BOX_LOTTE_PEPERO_ORIGINAL_46G",
                product_weight=66.0,
                stock_qty=25,
                sale_price=2500,
                product_idx="P114",
                has_loadcell="true",
            ),
            SimpleNamespace(
                yolo_class_id=115,
                product_name="BOX_LOTTE_PEPERO_ALMOND_37G",
                product_weight=58.0,
                stock_qty=21,
                sale_price=1400,
                product_idx="P115",
                has_loadcell="true",
            ),
        ],
        trace_context=trace_context,
    )

    trace_context.record_final_result(
        products=result.products,
        total_price=result.total_price,
        status=result.status.value,
        confidence=result.confidence,
    )
    trace_context.finalize(status="complete")

    assert result.status == JudgmentStatus.COMPLETE
    assert result.products[0].product_id == 114
    assert result.weight_residual == 5.6

    detail_file = next((tmp_path / "logs" / "triggers").glob("*/*.json"))
    detail = json.loads(detail_file.read_text(encoding="utf-8"))
    assert detail["weight_diagnostics"]["fallback_reason"] == "detected_single_item_fallback"
    assert (
        detail["weight_diagnostics"]["detected_single_item_fallback"]["accepted"]["class_id"]
        == 114
    )
