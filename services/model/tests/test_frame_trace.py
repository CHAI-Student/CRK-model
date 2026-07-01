import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def read_trace_entries(log_dir: Path) -> list[dict]:
    files = sorted(log_dir.glob("frame_split_*.jsonl"))
    assert files, f"no trace files found in {log_dir}"
    return [
        json.loads(line)
        for line in files[-1].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_trigger_detail_files(log_dir: Path) -> list[Path]:
    files = sorted((log_dir / "triggers").glob("*/*.json"))
    assert files, f"no trigger detail files found in {log_dir}"
    return files


def make_trace_factory(tmp_path: Path):
    from model_service.video.frame_trace import TriggerTraceContext as RealTriggerTraceContext

    def factory(*args, **kwargs):
        return RealTriggerTraceContext(
            *args,
            log_dir=tmp_path / "logs",
            sample_export_dir=tmp_path / "samples",
            **kwargs,
        )

    return factory


def sample_frame(value: int) -> np.ndarray:
    return np.full((4, 4, 3), value, dtype=np.uint8)


def low_weight_loadcells() -> list:
    return [
        {
            "timestamp": "2026-03-20T10:00:00.000Z",
            "raw_value": ["+1000"],
            "filtered_value": ["+1000"],
            "filter_method": "none",
        },
        {
            "timestamp": "2026-03-20T10:00:01.000Z",
            "raw_value": ["+1001"],
            "filtered_value": ["+1000"],
            "filter_method": "none",
        },
    ]


def threshold_weight_loadcells() -> list:
    return [
        {
            "timestamp": "2026-03-20T10:00:00.000Z",
            "raw_value": ["+1000"],
            "filtered_value": ["+1000"],
            "filter_method": "none",
        },
        {
            "timestamp": "2026-03-20T10:00:01.000Z",
            "raw_value": ["+1005"],
            "filtered_value": ["+1005"],
            "filter_method": "none",
        },
    ]


def stable_low_weight_loadcells() -> list:
    values = [1000.0] * 6 + [1001.0] * 6
    return [
        {
            "timestamp": f"2026-03-20T10:00:{index:02d}.000Z",
            "raw_value": [f"+{value:.1f}"],
            "filtered_value": [f"+{value:.1f}"],
            "filter_method": "none",
        }
        for index, value in enumerate(values)
    ]


def changed_weight_loadcells() -> list:
    return [
        {
            "timestamp": "2026-03-20T10:00:00.000Z",
            "raw_value": ["+1000"],
            "filtered_value": ["+1000"],
            "filter_method": "none",
        },
        {
            "timestamp": "2026-03-20T10:00:01.000Z",
            "raw_value": ["+1015"],
            "filtered_value": ["+1015"],
            "filter_method": "none",
        },
    ]


def create_input(video_path: Path, loadcells: list):
    from model_service.service.trigger_service import LoadcellReading, TriggerInput

    return TriggerInput(
        zone=1,
        loadcells=[LoadcellReading(**item) for item in loadcells],
        top_video_path=str(video_path),
        side_video_path=None,
    )


def loadcells_for_delta(delta: float) -> list:
    start = 1000.0
    end = start + delta
    values = [start] * 5 + [start + (delta * 0.35), start + (delta * 0.7)] + [end] * 5
    return [
        {
            "timestamp": f"2026-03-20T10:00:{index:02d}.000Z",
            "raw_value": [f"{value:+.1f}"],
            "filtered_value": [f"{value:+.1f}"],
            "filter_method": "none",
        }
        for index, value in enumerate(values)
    ]


def whole_machine_loadcells_for_zone_delta(delta: float, zone: int = 5) -> list:
    start_channels = [1000.0] * 10
    end_channels = list(start_channels)
    zone_start = (zone - 1) * 2
    end_channels[zone_start] += delta / 2.0
    end_channels[zone_start + 1] += delta / 2.0
    mid_channels = [
        start + ((end - start) * 0.5)
        for start, end in zip(start_channels, end_channels)
    ]
    samples = [start_channels] * 5 + [mid_channels] * 2 + [end_channels] * 5
    return [
        {
            "timestamp": f"2026-03-20T10:00:{index:02d}.000Z",
            "raw_value": [f"{value:+.1f}" for value in values],
            "filtered_value": [f"{value:+.1f}" for value in values],
            "filter_method": "none",
        }
        for index, values in enumerate(samples)
    ]


def two_channel_loadcells_for_delta(delta: float) -> list:
    start_channels = [1000.0, 1000.0]
    end_channels = [1000.0 + delta / 2.0, 1000.0 + delta / 2.0]
    mid_channels = [
        start + ((end - start) * 0.5)
        for start, end in zip(start_channels, end_channels)
    ]
    samples = [start_channels] * 5 + [mid_channels] * 2 + [end_channels] * 5
    return [
        {
            "timestamp": f"2026-03-20T10:00:{index:02d}.000Z",
            "raw_value": [f"{value:+.1f}" for value in values],
            "filtered_value": [f"{value:+.1f}" for value in values],
            "filter_method": "none",
        }
        for index, values in enumerate(samples)
    ]


def unstable_removal_loadcells() -> list:
    values = [1000.0] * 5 + [900.0, 800.0, 700.0, 620.0, 540.0]
    return [
        {
            "timestamp": f"2026-03-20T10:00:{index:02d}.000Z",
            "raw_value": [f"{value:+.1f}"],
            "filtered_value": [f"{value:+.1f}"],
            "filter_method": "none",
        }
        for index, value in enumerate(values)
    ]


def short_loadcells_for_delta(delta: float) -> list:
    start = 1000.0
    end = start + delta
    return [
        {
            "timestamp": "2026-03-20T10:00:00.000Z",
            "raw_value": [f"{start:+.1f}"],
            "filtered_value": [f"{start:+.1f}"],
            "filter_method": "none",
        },
        {
            "timestamp": "2026-03-20T10:00:01.000Z",
            "raw_value": [f"{end:+.1f}"],
            "filtered_value": [f"{end:+.1f}"],
            "filter_method": "none",
        },
    ]


def create_input_for_delta(video_path: Path, delta: float, zone: int = 1):
    from model_service.service.trigger_service import LoadcellReading, TriggerInput

    return TriggerInput(
        zone=zone,
        loadcells=[LoadcellReading(**item) for item in loadcells_for_delta(delta)],
        top_video_path=str(video_path),
        side_video_path=None,
    )


def complete_judgment_result():
    from model_service.engine.models import JudgmentResult, JudgmentStatus, ProductJudgment

    return JudgmentResult(
        products=[
            ProductJudgment(
                product_id=26,
                name="Chicken Mayo",
                count=1,
                unit_price=3500,
                total_price=3500,
                confidence=0.91,
                unit_weight=365.0,
            )
        ],
        total_price=3500,
        confidence=0.91,
        status=JudgmentStatus.COMPLETE,
        weight_delta=-365.0,
        weight_explained=-365.0,
        weight_residual=0.0,
    )


def no_detection_judgment_result():
    from model_service.engine.models import JudgmentResult, JudgmentStatus

    return JudgmentResult(
        products=[],
        total_price=0,
        confidence=0.0,
        status=JudgmentStatus.NO_DETECTION,
        weight_delta=-52.0,
        weight_explained=0.0,
        weight_residual=-52.0,
    )


def uncertain_judgment_result_with_product():
    from model_service.engine.models import JudgmentResult, JudgmentStatus, ProductJudgment

    return JudgmentResult(
        products=[
            ProductJudgment(
                product_id=113,
                name="STICK_INNON_CONDITION_STICK_18G",
                count=1,
                unit_price=3000,
                total_price=3000,
                confidence=0.2,
                unit_weight=19.0,
            )
        ],
        total_price=3000,
        confidence=0.2,
        status=JudgmentStatus.UNCERTAIN,
        weight_delta=-7.6,
        weight_explained=19.0,
        weight_residual=11.4,
    )


def test_trigger_trace_context_exports_selected_samples_and_writes_jsonl(tmp_path):
    from model_service.video.frame_trace import TriggerTraceContext

    context = TriggerTraceContext(
        session_id="session-trace",
        zone=1,
        top_path="/tmp/top.avi",
        side_path=None,
        log_dir=tmp_path / "logs",
        sample_export_dir=tmp_path / "samples",
        sample_export_enabled=True,
        sample_count_per_camera=3,
    )

    context.plan_camera("top", 5)
    for index in range(5):
        context.record_frame("top", index, sample_frame(index))
    context.finalize(status="complete")

    entries = read_trace_entries(tmp_path / "logs")
    assert len(entries) == 1
    entry = entries[0]
    assert entry["processing_mode"] == "avi_to_frames"
    assert entry["inference_unit"] == "image_frame"
    assert entry["status"] == "complete"
    assert entry["cameras"]["top"]["total_frames"] == 5
    assert entry["cameras"]["top"]["processed_frames"] == 5
    assert entry["cameras"]["top"]["sample_indices"] == [0, 2, 4]
    assert len(entry["cameras"]["top"]["sample_files"]) == 3
    for exported in entry["cameras"]["top"]["sample_files"]:
        assert Path(exported).exists()


def test_video_processor_passes_allowed_ids_to_yolo(monkeypatch):
    import model_service.video.video_processor as video_processor_module
    from model_service.video import VideoProcessor

    class FakeExtractor:
        total_frames = 1

        def __iter__(self):
            return iter([sample_frame(1)])

    monkeypatch.setattr(
        video_processor_module,
        "create_frame_extractor",
        lambda *args, **kwargs: FakeExtractor(),
    )

    fake_yolo = MagicMock()
    fake_yolo.detect.return_value = []
    processor = VideoProcessor(
        yolo=fake_yolo,
        motion_filter_enabled=False,
        hand_path_filter_enabled=False,
    )

    processor.process_videos(top_path="/tmp/top.avi", allowed_class_ids=[1, 7, 9])

    assert fake_yolo.detect.call_count == 1
    assert fake_yolo.detect.call_args.kwargs["allowed_class_ids"] == [1, 7, 9, 0]


def test_video_processor_empty_allowed_ids_fail_closed(monkeypatch):
    import model_service.video.video_processor as video_processor_module
    from model_service.video import VideoProcessor
    from model_service.vision.yolo_wrapper import YOLODetection

    class FakeExtractor:
        total_frames = 1

        def __iter__(self):
            return iter([sample_frame(1)])

    monkeypatch.setattr(
        video_processor_module,
        "create_frame_extractor",
        lambda *args, **kwargs: FakeExtractor(),
    )

    fake_yolo = MagicMock()

    def fake_detect(frame, allowed_class_ids=None):
        if allowed_class_ids == []:
            return []
        return [
            YOLODetection(
                xyxy=(0.0, 0.0, 10.0, 10.0),
                cls=26,
                conf=0.99,
                name="Chicken Mayo",
            )
        ]

    fake_yolo.detect.side_effect = fake_detect
    processor = VideoProcessor(
        yolo=fake_yolo,
        motion_filter_enabled=False,
        hand_path_filter_enabled=False,
    )

    result = processor.process_videos(top_path="/tmp/top.avi", allowed_class_ids=[])

    assert fake_yolo.detect.call_args.kwargs["allowed_class_ids"] == []
    assert result.vote_results == []


def test_vote_result_conversion_preserves_frame_vote_count():
    from model_service.api.routes.trigger import (
        _vote_results_to_ensemble as route_vote_results_to_ensemble,
    )
    from model_service.service.trigger_service import TriggerService
    from model_service.video.voting_ensemble import VoteResult

    vote = VoteResult(
        class_id=27,
        class_name="BAG_NULLDAM_BAGEL_140G",
        vote_count=4,
        max_confidence=0.91,
        avg_confidence=0.88,
        weighted_confidence=0.80,
        top_detected=True,
        top_vote_count=4,
        instance_count_hint=1,
        freezer_exit_path_votes=9,
    )

    service_candidate = TriggerService._vote_results_to_ensemble(object(), [vote])[0]
    route_candidate = route_vote_results_to_ensemble([vote])[0]

    assert service_candidate.vote_count == 1
    assert service_candidate.raw_vote_count == 4
    assert route_candidate.vote_count == 1
    assert route_candidate.raw_vote_count == 4


def test_video_processor_motion_filter_rejects_static_candidates():
    from model_service.video import VideoProcessor
    from model_service.video.voting_ensemble import VotingEnsemble

    class NoMotionTracker:
        dynamic_threshold = 10.0
        total_displacement = 0.0

        def has_motion(self, threshold):
            return False

    processor = VideoProcessor(yolo=MagicMock())
    ensemble = VotingEnsemble()

    filtered = processor._apply_motion_filter_and_votes(
        "top",
        {42: [(0.8, "PRODUCT_42")]},
        {42: NoMotionTracker()},
        ensemble,
    )

    assert filtered == 1
    assert 42 not in ensemble.votes


def test_video_processor_motion_filter_accepts_ten_pixel_motion():
    from model_service.video import VideoProcessor
    from model_service.video.video_processor import BboxTracker
    from model_service.video.voting_ensemble import VotingEnsemble

    tracker = BboxTracker()
    tracker.update((0.0, 0.0), 0)
    tracker.update((10.0, 0.0), 1)
    tracker.dynamic_threshold = 10.0
    processor = VideoProcessor(yolo=MagicMock(), min_motion_displacement=10.0)
    ensemble = VotingEnsemble()

    filtered = processor._apply_motion_filter_and_votes(
        "top",
        {42: [(0.8, "PRODUCT_42")]},
        {42: tracker},
        ensemble,
    )

    assert filtered == 0
    assert ensemble.votes[42].count == 1


def test_video_processor_hand_path_fail_open_keeps_candidates():
    from model_service.video import VideoProcessor
    from model_service.video.voting_ensemble import VoteResult

    class RejectingHandPath:
        def filter_products_by_path(self, candidate_class_ids):
            return []

    processor = VideoProcessor(yolo=MagicMock())
    results = [
        VoteResult(
            class_id=42,
            class_name="PRODUCT_42",
            vote_count=2,
            max_confidence=0.8,
            avg_confidence=0.75,
        )
    ]

    kept, removed = processor._apply_hand_path_filter(
        results,
        RejectingHandPath(),
        "TEST",
    )

    assert kept == results
    assert removed == 0


def test_frame_trace_records_freezer_interaction_evidence(tmp_path):
    from model_service.video.frame_trace import TriggerTraceContext

    trace_context = TriggerTraceContext(
        session_id="interaction-evidence",
        zone=1,
        top_path="/tmp/top.avi",
        side_path=None,
        log_dir=tmp_path / "logs",
        sample_export_dir=tmp_path / "samples",
        sample_export_enabled=False,
    )
    trace_context.record_motion_evidence(
        class_id=101,
        class_name="STATIC_TIGHT_SINGLE",
        camera="top",
        path_displacement_px=2.0,
        max_distance_px=14.0,
        center_span_x=3.0,
        center_span_y=3.0,
        motion_threshold_px=12.0,
        trajectory_exit_path_passed=False,
        static_shelf_likely=True,
    )
    trace_context.record_hand_path_evidence(
        class_id=101,
        class_name="STATIC_TIGHT_SINGLE",
        camera="top",
        hand_path_valid=True,
        hand_path_passed=False,
        hand_path_blocked=True,
    )
    trace_context.finalize(status="complete")

    detail = json.loads(read_trigger_detail_files(tmp_path / "logs")[0].read_text())
    entry = detail["stage_counts_by_class"]["101"]
    assert entry["pathDisplacementPx"] == 2.0
    assert entry["maxDistancePx"] == 14.0
    assert entry["centerSpanX"] == 3.0
    assert entry["centerSpanY"] == 3.0
    assert entry["trajectoryExitPathPassed"] is False
    assert entry["staticShelfLikely"] is True
    assert entry["handPathValid"] is True
    assert entry["handPathPassed"] is False
    assert entry["handPathBlocked"] is True


def test_video_processor_process_videos_records_trace_samples(monkeypatch, tmp_path):
    import model_service.video.video_processor as video_processor_module
    from model_service.video import VideoProcessor
    from model_service.video.frame_trace import TriggerTraceContext

    class FakeExtractor:
        def __init__(self, frames):
            self.frames = frames
            self.total_frames = len(frames)

        def __iter__(self):
            return iter(self.frames)

    frames = [sample_frame(index) for index in range(5)]
    monkeypatch.setattr(
        video_processor_module,
        "create_frame_extractor",
        lambda *args, **kwargs: FakeExtractor(frames),
    )

    fake_yolo = MagicMock()
    fake_yolo.detect.side_effect = lambda frame, allowed_class_ids=None: []

    processor = VideoProcessor(
        yolo=fake_yolo,
        motion_filter_enabled=False,
        hand_path_filter_enabled=False,
    )
    trace_context = TriggerTraceContext(
        session_id="processor-session",
        zone=1,
        top_path="/tmp/top.avi",
        side_path=None,
        log_dir=tmp_path / "logs",
        sample_export_dir=tmp_path / "samples",
        sample_export_enabled=True,
        sample_count_per_camera=3,
    )

    result = processor.process_videos(
        top_path="/tmp/top.avi",
        trace_context=trace_context,
    )
    trace_context.finalize(status="complete")

    assert result.stats.top_frames == 5
    entries = read_trace_entries(tmp_path / "logs")
    assert entries[0]["cameras"]["top"]["processed_frames"] == 5
    assert entries[0]["cameras"]["top"]["sample_indices"] == [0, 2, 4]


def test_trigger_trace_context_writes_per_trigger_detail_json_on_finalize(tmp_path):
    from model_service.video.frame_trace import TriggerTraceContext

    context = TriggerTraceContext(
        session_id="detail-session",
        zone=3,
        top_path="/tmp/top.avi",
        side_path="/tmp/side.avi",
        log_dir=tmp_path / "logs",
        sample_export_dir=tmp_path / "samples",
        sample_export_enabled=False,
    )
    context.record_loadcell_delta(delta_weight=-113.5, sample_count=12)
    context.record_video_stats(
        {
            "top_frames": 10,
            "side_frames": 9,
            "top_raw_detections": 6,
            "side_raw_detections": 5,
            "top_threshold_filtered": 1,
            "side_threshold_filtered": 2,
            "roi_filtered_detections": 1,
            "motion_filtered_classes": 0,
            "hand_path_filtered_classes": 1,
            "processing_time_ms": 42.5,
        }
    )
    context.record_raw_vision_candidates(
        [
            {
                "rank": 1,
                "class_id": 26,
                "name": "Chicken Mayo",
                "confidence": 0.91,
                "top": True,
                "side": False,
            },
            {
                "rank": 2,
                "class_id": 27,
                "name": "Raw Only",
                "confidence": 0.89,
                "top": True,
                "side": True,
            },
        ],
        {26: 365.0, 27: 120.0},
    )
    context.record_candidates(
        [
            {
                "rank": 1,
                "class_id": 26,
                "name": "Chicken Mayo",
                "confidence": 0.91,
                "top": True,
                "side": False,
            }
        ],
        {26: 365.0},
    )
    context.record_final_result(
        products=[
            {
                "name": "Chicken Mayo",
                "count": 1,
                "total_price": 3500,
            }
        ],
        total_price=3500,
        status="complete",
        confidence=0.91,
    )

    assert list((tmp_path / "logs").glob("triggers/*/*.json")) == []

    context.finalize(status="complete")

    detail_files = read_trigger_detail_files(tmp_path / "logs")
    assert len(detail_files) == 1
    detail = json.loads(detail_files[0].read_text(encoding="utf-8"))
    assert detail["session_id"] == "detail-session"
    assert detail["zone"] == 3
    assert detail["loadcell"]["delta_weight"] == -113.5
    assert detail["video_stats"]["top_raw_detections"] == 6
    assert [candidate["name"] for candidate in detail["raw_vision_candidates"]] == [
        "Chicken Mayo",
        "Raw Only",
    ]
    assert detail["candidates"][0]["name"] == "Chicken Mayo"
    assert detail["candidates"][0]["unit_weight_g"] == 365.0
    assert detail["final_result"]["products"][0]["name"] == "Chicken Mayo"


def test_trigger_trace_records_dual_top_proxy_camera_roles(monkeypatch, tmp_path):
    from model_service.core.config import config
    from model_service.video.frame_trace import TriggerTraceContext

    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")

    context = TriggerTraceContext(
        session_id="dual-top-session",
        zone=2,
        top_path="/tmp/top-center.avi",
        side_path="/tmp/top-left.avi",
        log_dir=tmp_path / "logs",
        sample_export_dir=tmp_path / "samples",
        sample_export_enabled=False,
    )
    context.plan_camera("top", 6)
    context.plan_camera("side", 6)
    context.record_frame("top", 0)
    context.record_frame("side", 0)

    entry = context.finalize(status="complete")
    detail_file = read_trigger_detail_files(tmp_path / "logs")[0]
    detail = json.loads(detail_file.read_text(encoding="utf-8"))

    assert entry["camera_layout"] == "dual_top_proxy"
    assert entry["camera_roles"]["top"] == {
        "logical_role": "top",
        "physical_role": "top_middle",
        "processing_profile": "top",
    }
    assert entry["camera_roles"]["side"] == {
        "logical_role": "side",
        "physical_role": "top_side",
        "processing_profile": "top",
    }
    assert detail["camera_layout"] == "dual_top_proxy"
    assert detail["camera_roles"] == entry["camera_roles"]


def test_trigger_trace_context_candidate_weight_defaults_to_null(tmp_path):
    from model_service.video.frame_trace import TriggerTraceContext

    context = TriggerTraceContext(
        session_id="unknown-weight-session",
        zone=1,
        top_path="/tmp/top.avi",
        side_path=None,
        log_dir=tmp_path / "logs",
        sample_export_dir=tmp_path / "samples",
        sample_export_enabled=False,
    )
    context.record_candidates(
        [
            {
                "rank": 1,
                "class_id": 99,
                "name": "Unknown Weight Candidate",
                "confidence": 0.52,
            }
        ]
    )

    context.finalize(status="complete")

    detail_files = read_trigger_detail_files(tmp_path / "logs")
    detail = json.loads(detail_files[0].read_text(encoding="utf-8"))
    assert detail["candidates"][0]["unit_weight_g"] is None


def test_trigger_service_candidate_ops_log_includes_weight(caplog):
    from model_service.service.trigger_service import TriggerService
    from model_service.video.voting_ensemble import VoteResult

    caplog.set_level(logging.INFO, logger="model_service.ops")

    TriggerService._log_candidate_ops(
        3,
        [
            VoteResult(
                class_id=114,
                class_name="BOX_LOTTE_PEPERO_ORIGINAL_46G",
                vote_count=13,
                max_confidence=0.386,
                avg_confidence=0.28,
                top_detected=True,
                side_detected=False,
                top_vote_count=13,
                top_max_confidence=0.386,
                weighted_confidence=0.174,
            )
        ],
        {114: 66.0},
    )

    assert "name=BOX_LOTTE_PEPERO_ORIGINAL_46G weight=66.0g" in caplog.text


def test_trigger_trace_context_writes_diagnostics_sections(tmp_path):
    from model_service.video.frame_trace import TriggerTraceContext

    context = TriggerTraceContext(
        session_id="diagnostics-session",
        zone=2,
        top_path="/tmp/top.avi",
        side_path=None,
        log_dir=tmp_path / "logs",
        sample_export_dir=tmp_path / "samples",
    )

    context.record_preprocess(
        "top",
        {
            "camera_type": "top",
            "original_width": 640,
            "original_height": 480,
            "processed_width": 480,
            "processed_height": 480,
            "crop_policy": "center",
            "crop_box": {"x1": 80, "y1": 0, "x2": 560, "y2": 480},
        },
    )
    context.record_stage_count(
        class_id=96,
        class_name="BAG_JAYEONLU_MOIST_SWEET_CHESTNUT_80G",
        stage="raw",
        camera="top",
    )
    context.record_stage_count(
        class_id=96,
        class_name="BAG_JAYEONLU_MOIST_SWEET_CHESTNUT_80G",
        stage="threshold_passed",
        camera="top",
    )
    context.record_extractor_diagnostics(
        "top",
        {
            "expected_frames": 12,
            "decoded_frames": 10,
            "partial_reads": 1,
            "final_branch": "sync_retry",
        },
    )
    context.record_diagnostic_detection(
        camera="top",
        frame_index=0,
        class_id=96,
        class_name="BAG_JAYEONLU_MOIST_SWEET_CHESTNUT_80G",
        confidence=0.44,
    )
    context.record_weight_diagnostics(
        {
            "target_weight": 80.0,
            "tolerance": 3.0,
            "candidate_products": [
                {
                    "class_id": 96,
                    "name": "BAG_JAYEONLU_MOIST_SWEET_CHESTNUT_80G",
                    "weight": 80.0,
                    "stock": 1,
                }
            ],
        }
    )

    context.finalize(status="complete")

    detail_file = read_trigger_detail_files(tmp_path / "logs")[0]
    detail = json.loads(detail_file.read_text(encoding="utf-8"))
    assert detail["preprocess"]["top"]["crop_box"]["x1"] == 80
    assert detail["stage_counts_by_class"]["96"]["raw"] == 1
    assert detail["extractor_diagnostics"]["top"]["final_branch"] == "sync_retry"
    assert detail["diagnostic_candidates"][0]["name"] == "BAG_JAYEONLU_MOIST_SWEET_CHESTNUT_80G"
    assert detail["weight_diagnostics"]["candidate_products"][0]["weight"] == 80.0


@pytest.mark.asyncio
async def test_trigger_service_worker_complete_writes_trace_entry(monkeypatch, tmp_path, session_store):
    import model_service.service.trigger_service as trigger_service_module
    from model_service.service.trigger_service import TriggerService
    from model_service.session import DoorSessionStore

    trace_factory = make_trace_factory(tmp_path)
    monkeypatch.setattr(trigger_service_module, "TriggerTraceContext", trace_factory)
    monkeypatch.setattr(
        trigger_service_module,
        "generate_session_id",
        lambda zone: "worker-session",
    )
    monkeypatch.setattr(trigger_service_module.config.async_streaming, "enabled", True)

    class FakeVideoProcessor:
        async def process_videos_async(self, **kwargs):
            trace_context = kwargs["trace_context"]
            trace_context.plan_camera("top", 4)
            for index in range(4):
                trace_context.record_frame("top", index, sample_frame(index))
            return SimpleNamespace(
                vote_results=[],
                stats=SimpleNamespace(top_frames=4, side_frames=0, processing_time_ms=12.5),
            )

    engine = MagicMock()
    engine.judge.return_value = complete_judgment_result()

    door_store = DoorSessionStore(yaml_dir=str(tmp_path / "sessions"))
    try:
        door_store.get_or_start_global_session()
        service = TriggerService(
            video_processor=FakeVideoProcessor(),
            engine=engine,
            session_store=session_store,
            door_session_store=door_store,
        )
        service._queue = asyncio.Queue(maxsize=service.QUEUE_MAX_SIZE)

        video_path = tmp_path / "top.avi"
        video_path.write_bytes(b"avi")
        output = await service.enqueue_trigger(create_input_for_delta(video_path, -365.0))
        assert output.status == "queued"
        assert door_store.get_pending_trigger_snapshot()["pendingTriggerCount"] == 1

        item = service._queue.get_nowait()
        await service._process_trigger_internal(item)
        assert door_store.get_pending_trigger_snapshot()["pendingTriggerCount"] == 0
        assert door_store.get_global_session().total_trigger_count == 1

        entries = read_trace_entries(tmp_path / "logs")
        assert len(entries) == 1
        assert entries[0]["session_id"] == "worker-session"
        assert entries[0]["status"] == "complete"
        assert entries[0]["cameras"]["top"]["processed_frames"] == 4

        detail_files = read_trigger_detail_files(tmp_path / "logs")
        detail = json.loads(detail_files[0].read_text(encoding="utf-8"))
        assert detail["loadcell"]["delta_weight"] == -365.0
        assert detail["video_stats"]["top_frames"] == 4
        assert detail["final_result"]["products"][0]["name"] == "Chicken Mayo"
    finally:
        door_store.shutdown()


@pytest.mark.asyncio
async def test_trigger_service_freezer_single_bagel_candidate_counts_repeat_and_closes(
    monkeypatch,
    tmp_path,
    session_store,
):
    import model_service.service.trigger_service as trigger_service_module
    from model_service.core.config import config
    from model_service.engine.decision_engine import ProductDecisionEngine
    from model_service.service.trigger_service import (
        LoadcellReading,
        TriggerInput,
        TriggerService,
    )
    from model_service.session import DoorSessionStore
    from model_service.video import VoteResult

    trace_factory = make_trace_factory(tmp_path)
    monkeypatch.setattr(trigger_service_module, "TriggerTraceContext", trace_factory)
    monkeypatch.setattr(
        trigger_service_module,
        "generate_session_id",
        lambda zone: "freezer-bagel-repeat-session",
    )
    monkeypatch.setattr(trigger_service_module.config.async_streaming, "enabled", True)
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")
    monkeypatch.setattr(config.weight, "freezer_multi_min_confidence", 0.45)
    monkeypatch.setattr(config.weight, "freezer_weight_tolerance_grams", 15.0)

    class FakeVideoProcessor:
        async def process_videos_async(self, **kwargs):
            return SimpleNamespace(
                vote_results=[
                    VoteResult(
                            class_id=27,
                            class_name="BAG_NULLDAM_BAGEL_140G",
                            vote_count=1,
                            max_confidence=0.728,
                            avg_confidence=0.728,
                            vote_ratio=1.0,
                            top_detected=False,
                            side_detected=True,
                            top_vote_count=0,
                            side_vote_count=1,
                            top_max_confidence=0.0,
                            side_max_confidence=0.728,
                            weighted_confidence=0.728,
                        raw_vote_count=1,
                        side_motion_passed=True,
                        motion_gate_passed=True,
                        instance_count_hint=1,
                    )
                ],
                stats=SimpleNamespace(
                    top_frames=0,
                    side_frames=3,
                    processing_time_ms=7.0,
                ),
            )

    active_product = SimpleNamespace(
        yolo_class_id=27,
        product_name="BAG_NULLDAM_BAGEL_140G",
        product_eng_name="BAG_NULLDAM_BAGEL_140G",
        product_weight=156.0,
        stock_qty=10,
        sale_price=2800,
        product_idx="P_BAGEL",
        has_loadcell="true",
    )

    class FakeActiveProductStore:
        def has_products(self):
            return True

        def get_all_products(self):
            return [active_product]

        def get_allowed_class_ids(self):
            return [27]

        def get_by_yolo_class_id(self, product_id):
            return active_product if int(product_id) == 27 else None

        def get_stats(self):
            return {"source": "test"}

    door_store = DoorSessionStore(
        yaml_dir=str(tmp_path / "sessions"),
        get_product_weight=lambda product_id: {27: 156.0}.get(product_id, 0.0),
    )
    try:
        door_store.get_or_start_global_session()
        service = TriggerService(
            video_processor=FakeVideoProcessor(),
            engine=ProductDecisionEngine(strict_mode=True),
            session_store=session_store,
            door_session_store=door_store,
            active_product_store=FakeActiveProductStore(),
        )
        service._queue = asyncio.Queue(maxsize=service.QUEUE_MAX_SIZE)

        top_path = tmp_path / "top.avi"
        side_path = tmp_path / "side.avi"
        top_path.write_bytes(b"top")
        side_path.write_bytes(b"side")
        input_data = TriggerInput(
            zone=1,
            loadcells=[
                LoadcellReading(**item)
                for item in two_channel_loadcells_for_delta(-309.5)
            ],
            top_video_path=str(top_path),
            side_video_path=str(side_path),
        )

        queued = await service.enqueue_trigger(input_data)
        assert queued.status == "queued"
        item = service._queue.get_nowait()
        await service._process_trigger_internal(item)

        session_data = session_store.get("freezer-bagel-repeat-session")
        assert session_data is not None
        assert session_data.status == "complete"
        assert [(product.name, product.count) for product in session_data.products] == [
            ("BAG_NULLDAM_BAGEL_140G", 2)
        ]
        door_session = door_store.get_session(zone=1)
        assert door_session is not None
        assert [
            (product.name, product.count)
            for product in door_session.get_active_products()
        ] == [("BAG_NULLDAM_BAGEL_140G", 2)]

        global_session = door_store.finalize_global_session()
        zone_session = global_session.zone_sessions[1]
        assert [
            (product.name, product.count)
            for product in zone_session.get_active_products()
        ] == [("BAG_NULLDAM_BAGEL_140G", 2)]
        assert (
            zone_session.final_weight_validation is None
            or zone_session.final_weight_validation.get("reason")
            != "unresolved_final_weight_mismatch"
        )

        detail = json.loads(read_trigger_detail_files(tmp_path / "logs")[0].read_text())
        diagnostics = detail["weight_diagnostics"]["freezer_vision_first"]
        assert diagnostics["reason"] == "freezer_ordered_vision_candidate_pool"
        assert diagnostics["selected"][0]["count"] == 2
        assert diagnostics["selected"][0]["combinationResidual"] == pytest.approx(
            2.5,
            abs=0.2,
        )
    finally:
        door_store.shutdown()


@pytest.mark.asyncio
async def test_trigger_service_worker_marks_video_processing_error(
    monkeypatch,
    tmp_path,
    session_store,
):
    import model_service.service.trigger_service as trigger_service_module
    from model_service.core.exceptions import VideoProcessingError
    from model_service.service.trigger_service import TriggerService
    from model_service.session import DoorSessionStore

    trace_factory = make_trace_factory(tmp_path)
    monkeypatch.setattr(trigger_service_module, "TriggerTraceContext", trace_factory)
    monkeypatch.setattr(
        trigger_service_module,
        "generate_session_id",
        lambda zone: "worker-error-session",
    )
    monkeypatch.setattr(trigger_service_module.config.async_streaming, "enabled", True)

    class FailingVideoProcessor:
        async def process_videos_async(self, **kwargs):
            raise VideoProcessingError("async video decode failed")

    engine = MagicMock()
    door_store = DoorSessionStore(yaml_dir=str(tmp_path / "sessions"))
    try:
        door_store.get_or_start_global_session()
        service = TriggerService(
            video_processor=FailingVideoProcessor(),
            engine=engine,
            session_store=session_store,
            door_session_store=door_store,
        )
        await service.start_worker()

        video_path = tmp_path / "top.avi"
        video_path.write_bytes(b"avi")
        output = await service.enqueue_trigger(create_input_for_delta(video_path, -365.0))
        assert output.status == "queued"

        await asyncio.wait_for(service._queue.join(), timeout=1.0)
        await service.stop_worker()

        session_data = session_store.get("worker-error-session")
        assert session_data is not None
        assert session_data.status == "error"
        assert session_data.processing_stage == "error"
        assert "async video decode failed" in session_data.processing_stage_detail
        assert door_store.get_pending_trigger_snapshot()["pendingTriggerCount"] == 0
        engine.judge.assert_not_called()

        entries = read_trace_entries(tmp_path / "logs")
        assert entries[0]["session_id"] == "worker-error-session"
        assert entries[0]["status"] == "error"
        assert "async video decode failed" in entries[0]["error"]
    finally:
        await service.stop_worker()
        door_store.shutdown()


@pytest.mark.asyncio
async def test_trigger_service_waits_before_queue_on_unstable_removal_tail(
    monkeypatch,
    tmp_path,
    session_store,
):
    import asyncio

    import model_service.service.trigger_service as trigger_service_module
    from model_service.service.trigger_service import LoadcellReading, TriggerInput, TriggerService
    from model_service.session import DoorSessionStore

    trace_factory = make_trace_factory(tmp_path)
    monkeypatch.setattr(trigger_service_module, "TriggerTraceContext", trace_factory)
    monkeypatch.setattr(
        trigger_service_module,
        "generate_session_id",
        lambda zone: "unstable-removal-waiting-session",
    )
    monkeypatch.setattr(trigger_service_module.config.async_streaming, "enabled", True)

    class FakeVideoProcessor:
        def __init__(self):
            self.calls = 0

        async def process_videos_async(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                vote_results=[],
                stats=SimpleNamespace(top_frames=0, side_frames=0, processing_time_ms=0.0),
            )

    door_store = DoorSessionStore(yaml_dir=str(tmp_path / "sessions"))
    video_processor = FakeVideoProcessor()
    engine = MagicMock()
    try:
        door_store.get_or_start_global_session()
        service = TriggerService(
            video_processor=video_processor,
            engine=engine,
            session_store=session_store,
            door_session_store=door_store,
        )
        service._queue = asyncio.Queue(maxsize=service.QUEUE_MAX_SIZE)

        video_path = tmp_path / "unstable-removal.avi"
        video_path.write_bytes(b"avi")
        input_data = TriggerInput(
            zone=1,
            loadcells=[LoadcellReading(**item) for item in unstable_removal_loadcells()],
            top_video_path=str(video_path),
            side_video_path=None,
        )

        output = await service.enqueue_trigger(input_data)

        assert output.status == "waiting"
        assert output.waiting_for == "stable_loadcell"
        assert video_processor.calls == 0
        engine.judge.assert_not_called()
        assert service._queue.empty()

        session_data = session_store.get("unstable-removal-waiting-session")
        assert session_data is not None
        assert session_data.status == "waiting"
        assert session_data.processing_stage == "removal_waiting_for_stable_loadcell"
        assert session_data.failure_reason == "missing_active_products"
        assert session_data.products == []
        assert session_data.total_price == 0
        assert door_store.get_session(zone=1) is None
        assert door_store.get_pending_trigger_snapshot()["pendingTriggerCount"] == 0

        detail = json.loads(
            read_trigger_detail_files(tmp_path / "logs")[0].read_text(encoding="utf-8")
        )
        diagnostics = detail["weight_diagnostics"]["removal_stabilization"]
        assert detail["final_result"]["status"] == "removal_waiting_for_stable_loadcell"
        assert detail["final_result"]["failure_reason"] == "missing_active_products"
        assert (
            detail["active_product_diagnostics"]["inference_fail_closed_reason"]
            == "missing_active_product_snapshot_fail_closed"
        )
        assert (
            detail["weight_diagnostics"]["active_product_failure_reason"]
            == "missing_active_products"
        )
        assert diagnostics["accepted"] is True
        assert diagnostics["reason"] == "unstable_or_unconfirmed_removal_loadcell"
        assert diagnostics["stable_region_valid"] is False
        assert diagnostics["engine_skipped"] is True
    finally:
        door_store.shutdown()


@pytest.mark.asyncio
async def test_trigger_service_waits_for_stable_removal_delta_on_pepsi_x2_undercount(
    monkeypatch,
    tmp_path,
    session_store,
):
    import asyncio

    import model_service.service.trigger_service as trigger_service_module
    from model_service.service.trigger_service import TriggerService
    from model_service.session import DoorSessionStore

    trace_factory = make_trace_factory(tmp_path)
    monkeypatch.setattr(trigger_service_module, "TriggerTraceContext", trace_factory)
    monkeypatch.setattr(
        trigger_service_module,
        "generate_session_id",
        lambda zone: "pepsi-x2-waiting-session",
    )
    monkeypatch.setattr(trigger_service_module.config.async_streaming, "enabled", True)

    pepsi_product = SimpleNamespace(
        yolo_class_id=75,
        product_idx="PEPSI",
        product_name="BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
        stock_qty=10,
        product_weight=520.0,
        sale_price=2300,
        has_loadcell="true",
    )

    class FakeActiveProductStore:
        def has_products(self):
            return True

        def get_all_products(self):
            return [pepsi_product]

        def get_allowed_class_ids(self):
            return [75]

    class FakeVideoProcessor:
        async def process_videos_async(self, **kwargs):
            trace_context = kwargs["trace_context"]
            trace_context.plan_camera("side", 2)
            for index in range(2):
                trace_context.record_frame("side", index, sample_frame(index))
            return SimpleNamespace(
                vote_results=[
                    SimpleNamespace(
                        class_id=75,
                        class_name="BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
                        top_detected=False,
                        side_detected=True,
                        top_max_confidence=0.0,
                        side_max_confidence=0.525,
                        weighted_confidence=0.525,
                        raw_vote_count=150,
                        source="vision",
                        top_motion_passed=False,
                        side_motion_passed=True,
                        motion_gate_passed=True,
                        weight_gate_passed=None,
                    )
                ],
                stats=SimpleNamespace(top_frames=0, side_frames=2, processing_time_ms=8.0),
            )

    engine = MagicMock()
    engine.confidence_threshold = 0.3

    door_store = DoorSessionStore(yaml_dir=str(tmp_path / "sessions"))
    try:
        door_store.get_or_start_global_session()
        service = TriggerService(
            video_processor=FakeVideoProcessor(),
            engine=engine,
            session_store=session_store,
            door_session_store=door_store,
            active_product_store=FakeActiveProductStore(),
        )
        service._queue = asyncio.Queue(maxsize=service.QUEUE_MAX_SIZE)

        video_path = tmp_path / "pepsi-x2.avi"
        video_path.write_bytes(b"avi")
        output = await service.enqueue_trigger(create_input_for_delta(video_path, -964.0))
        assert output.status == "queued"
        assert door_store.get_pending_trigger_snapshot()["pendingTriggerCount"] == 1

        item = service._queue.get_nowait()
        await service._process_trigger_internal(item)

        engine.judge.assert_not_called()
        session_data = session_store.get("pepsi-x2-waiting-session")
        assert session_data is not None
        assert session_data.status == "waiting"
        assert session_data.processing_stage == "removal_waiting_for_stable_loadcell"
        assert session_data.products == []
        assert session_data.total_price == 0
        assert door_store.get_session(zone=1) is None
        assert door_store.get_pending_trigger_snapshot()["pendingTriggerCount"] == 0

        detail = json.loads(
            read_trigger_detail_files(tmp_path / "logs")[0].read_text(encoding="utf-8")
        )
        diagnostics = detail["weight_diagnostics"]["removal_stabilization"]
        assert detail["final_result"]["status"] == "removal_waiting_for_stable_loadcell"
        assert diagnostics["accepted"] is True
        assert diagnostics["selected"]["class_id"] == 75
        assert diagnostics["undercount"] == 76.0
    finally:
        door_store.shutdown()


def test_removal_stabilization_conflict_requires_strong_x2_bottle_evidence(session_store):
    from model_service.engine.models import EnsembleResult
    from model_service.service.trigger_service import TriggerService

    engine = MagicMock()
    engine.confidence_threshold = 0.3
    service = TriggerService(
        video_processor=MagicMock(),
        engine=engine,
        session_store=session_store,
    )
    active_products = [
        SimpleNamespace(
            yolo_class_id=75,
            product_name="BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
            product_weight=520.0,
            stock_qty=10,
            sale_price=2300,
            has_loadcell="true",
        )
    ]
    weak_candidate = EnsembleResult(
        class_id=75,
        class_name="BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
        top_confidence=0.0,
        side_confidence=0.30,
        combined_confidence=0.30,
        vote_count=2,
        raw_vote_count=150,
        source="vision",
    )
    strong_candidate = EnsembleResult(
        class_id=75,
        class_name="BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
        top_confidence=0.0,
        side_confidence=0.525,
        combined_confidence=0.525,
        vote_count=2,
        raw_vote_count=150,
        source="vision",
    )

    assert service._removal_stabilization_conflict(
        vision_candidates=[weak_candidate],
        delta_weight=-964.0,
        active_products=active_products,
    ) is None
    assert service._removal_stabilization_conflict(
        vision_candidates=[strong_candidate],
        delta_weight=-524.0,
        active_products=active_products,
    ) is None


@pytest.mark.asyncio
async def test_trigger_service_duplicate_writes_trace_entry(monkeypatch, tmp_path, session_store):
    import asyncio

    import model_service.service.trigger_service as trigger_service_module
    from model_service.service.trigger_service import TriggerService

    trace_factory = make_trace_factory(tmp_path)
    monkeypatch.setattr(trigger_service_module, "TriggerTraceContext", trace_factory)
    session_ids = iter(["queued-session", "duplicate-session"])
    monkeypatch.setattr(
        trigger_service_module,
        "generate_session_id",
        lambda zone: next(session_ids),
    )

    service = TriggerService(
        video_processor=MagicMock(),
        engine=MagicMock(),
        session_store=session_store,
    )
    service._queue = asyncio.Queue(maxsize=service.QUEUE_MAX_SIZE)

    video_path = tmp_path / "top.avi"
    video_path.write_bytes(b"avi")
    input_data = create_input_for_delta(video_path, -365.0)

    first = await service.enqueue_trigger(input_data)
    second = await service.enqueue_trigger(input_data)

    assert first.status == "queued"
    assert second.status == "duplicate"
    assert second.session_id == "queued-session"

    entries = read_trace_entries(tmp_path / "logs")
    assert len(entries) == 1
    assert entries[0]["status"] == "duplicate"
    assert entries[0]["session_id"] == "queued-session"


@pytest.mark.asyncio
async def test_trigger_service_skipped_writes_trace_entry(monkeypatch, tmp_path, session_store):
    import asyncio

    import model_service.service.trigger_service as trigger_service_module
    from model_service.service.trigger_service import TriggerService

    trace_factory = make_trace_factory(tmp_path)
    monkeypatch.setattr(trigger_service_module, "TriggerTraceContext", trace_factory)
    monkeypatch.setattr(
        trigger_service_module.config.trigger,
        "low_weight_vision_fallback",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        trigger_service_module,
        "generate_session_id",
        lambda zone: "skipped-session",
    )

    service = TriggerService(
        video_processor=MagicMock(),
        engine=MagicMock(),
        session_store=session_store,
    )
    service._queue = asyncio.Queue(maxsize=service.QUEUE_MAX_SIZE)

    video_path = tmp_path / "top.avi"
    video_path.write_bytes(b"avi")
    output = await service.enqueue_trigger(create_input(video_path, low_weight_loadcells()))

    assert output.status == "skipped"

    entries = read_trace_entries(tmp_path / "logs")
    assert len(entries) == 1
    assert entries[0]["status"] == "skipped"
    assert entries[0]["session_id"] == "skipped-session"
    assert entries[0]["cameras"]["top"]["processed_frames"] == 0


@pytest.mark.asyncio
async def test_trigger_service_low_weight_video_is_diagnostic_only(
    monkeypatch,
    tmp_path,
    session_store,
):
    import asyncio

    import model_service.service.trigger_service as trigger_service_module
    from model_service.service.trigger_service import TriggerService

    trace_factory = make_trace_factory(tmp_path)
    monkeypatch.setattr(trigger_service_module, "TriggerTraceContext", trace_factory)
    monkeypatch.setattr(
        trigger_service_module,
        "generate_session_id",
        lambda zone: "low-weight-vision-session",
    )
    monkeypatch.setattr(trigger_service_module.config.async_streaming, "enabled", True)
    monkeypatch.setattr(
        trigger_service_module.config.trigger,
        "low_weight_vision_fallback",
        True,
        raising=False,
    )

    class FakeVideoProcessor:
        called = False

        async def process_videos_async(self, **kwargs):
            self.called = True
            trace_context = kwargs["trace_context"]
            trace_context.plan_camera("top", 2)
            for index in range(2):
                trace_context.record_frame("top", index, sample_frame(index))
            trace_context.record_candidates(
                [
                    SimpleNamespace(
                        class_id=119,
                        class_name="BOTTLE_FANTA_ORANGE_600ML",
                        top_confidence=0.0,
                        side_confidence=0.46,
                        combined_confidence=0.46,
                        vote_count=5,
                        source="vision",
                        raw_vote_count=5,
                        top_motion_passed=False,
                        side_motion_passed=True,
                        motion_gate_passed=True,
                        weight_gate_passed=None,
                    )
                ],
                {119: 634.0},
            )
            return SimpleNamespace(
                vote_results=[
                    SimpleNamespace(
                        class_id=119,
                        class_name="BOTTLE_FANTA_ORANGE_600ML",
                        top_confidence=0.0,
                        side_confidence=0.46,
                        combined_confidence=0.46,
                        vote_count=5,
                        source="vision",
                        raw_vote_count=5,
                        top_motion_passed=False,
                        side_motion_passed=True,
                        motion_gate_passed=True,
                        weight_gate_passed=None,
                    )
                ],
                stats=SimpleNamespace(top_frames=2, side_frames=0, processing_time_ms=6.0),
            )

    engine = MagicMock()
    engine.judge.return_value = complete_judgment_result()
    video_processor = FakeVideoProcessor()

    service = TriggerService(
        video_processor=video_processor,
        engine=engine,
        session_store=session_store,
    )
    service._queue = asyncio.Queue(maxsize=service.QUEUE_MAX_SIZE)

    video_path = tmp_path / "top.avi"
    video_path.write_bytes(b"avi")
    output = await service.enqueue_trigger(create_input(video_path, low_weight_loadcells()))

    assert output.status == "complete"
    assert service._queue.empty()
    assert video_processor.called is True
    engine.judge.assert_not_called()
    session_data = session_store.get("low-weight-vision-session")
    assert session_data.processing_stage == "low_weight_video_diagnostic"
    assert session_data.products == []
    entries = read_trace_entries(tmp_path / "logs")
    assert entries[0]["status"] == "complete"
    assert entries[0]["cameras"]["top"]["processed_frames"] == 2
    assert entries[0]["candidates"][0]["class_id"] == 119
    assert entries[0]["loadcell"]["payload_state"] == "nonzero"
    assert entries[0]["loadcell"]["filtered_channel_count"] == 2
    assert entries[0]["loadcell"]["filtered_zero_channel_count"] == 0
    assert entries[0]["loadcell"]["filtered_nonzero_channel_count"] == 2
    assert entries[0]["loadcell"]["first_filtered_total"] == 1000.0
    assert entries[0]["loadcell"]["last_filtered_total"] == 1000.0
    assert entries[0]["weight_diagnostics"]["decision_branch"] == (
        "low_weight_video_diagnostic"
    )
    assert entries[0]["weight_diagnostics"]["excluded_from_close_summary"] is True
    assert entries[0]["weight_diagnostics"]["engine_skipped"] is True


@pytest.mark.asyncio
async def test_trigger_service_stable_low_weight_video_is_skipped(monkeypatch, tmp_path, session_store):
    import asyncio

    import model_service.service.trigger_service as trigger_service_module
    from model_service.service.trigger_service import TriggerService

    trace_factory = make_trace_factory(tmp_path)
    monkeypatch.setattr(trigger_service_module, "TriggerTraceContext", trace_factory)
    monkeypatch.setattr(
        trigger_service_module,
        "generate_session_id",
        lambda zone: "stable-low-weight-session",
    )
    monkeypatch.setattr(
        trigger_service_module.config.trigger,
        "low_weight_vision_fallback",
        False,
        raising=False,
    )

    video_processor = MagicMock()
    engine = MagicMock()

    service = TriggerService(
        video_processor=video_processor,
        engine=engine,
        session_store=session_store,
    )
    service._queue = asyncio.Queue(maxsize=service.QUEUE_MAX_SIZE)

    video_path = tmp_path / "top.avi"
    video_path.write_bytes(b"avi")
    output = await service.enqueue_trigger(create_input(video_path, stable_low_weight_loadcells()))

    assert output.status == "skipped"
    assert service._queue.empty()
    video_processor.process_videos.assert_not_called()
    engine.judge.assert_not_called()

    session_data = session_store.get("stable-low-weight-session")
    assert session_data is not None
    assert session_data.processing_stage == "skipped_low_weight"


@pytest.mark.asyncio
async def test_trigger_service_threshold_weight_is_excluded_from_door_session(
    monkeypatch,
    tmp_path,
    session_store,
    door_session_store,
):
    import asyncio

    import model_service.service.trigger_service as trigger_service_module
    from model_service.service.trigger_service import TriggerService

    trace_factory = make_trace_factory(tmp_path)
    monkeypatch.setattr(trigger_service_module, "TriggerTraceContext", trace_factory)
    monkeypatch.setattr(
        trigger_service_module,
        "generate_session_id",
        lambda zone: "threshold-low-weight-session",
    )
    monkeypatch.setattr(
        trigger_service_module.config.trigger,
        "low_weight_vision_fallback",
        False,
        raising=False,
    )

    door_session_store.get_or_start_global_session()
    service = TriggerService(
        video_processor=MagicMock(),
        engine=MagicMock(),
        session_store=session_store,
        door_session_store=door_session_store,
    )
    service._queue = asyncio.Queue(maxsize=service.QUEUE_MAX_SIZE)

    video_path = tmp_path / "top.avi"
    video_path.write_bytes(b"avi")
    output = await service.enqueue_trigger(create_input(video_path, threshold_weight_loadcells()))

    assert output.status == "skipped"
    assert output.door_session_id is None
    global_session = door_session_store.finalize_global_session()
    assert global_session is not None
    assert global_session.total_trigger_count == 0
    assert global_session.active_zones == []


@pytest.mark.asyncio
async def test_trigger_service_does_not_store_uncertain_products(monkeypatch, tmp_path, session_store):
    import asyncio

    import model_service.service.trigger_service as trigger_service_module
    from model_service.service.trigger_service import TriggerService

    trace_factory = make_trace_factory(tmp_path)
    monkeypatch.setattr(trigger_service_module, "TriggerTraceContext", trace_factory)
    monkeypatch.setattr(
        trigger_service_module,
        "generate_session_id",
        lambda zone: "uncertain-session",
    )
    monkeypatch.setattr(trigger_service_module.config.async_streaming, "enabled", True)

    class FakeVideoProcessor:
        async def process_videos_async(self, **kwargs):
            return SimpleNamespace(
                vote_results=[],
                stats=SimpleNamespace(top_frames=1, side_frames=0, processing_time_ms=5.0),
            )

    engine = MagicMock()
    engine.judge.return_value = uncertain_judgment_result_with_product()

    service = TriggerService(
        video_processor=FakeVideoProcessor(),
        engine=engine,
        session_store=session_store,
    )
    service._queue = asyncio.Queue(maxsize=service.QUEUE_MAX_SIZE)

    video_path = tmp_path / "top.avi"
    video_path.write_bytes(b"avi")
    output = await service.enqueue_trigger(create_input_for_delta(video_path, -365.0))

    assert output.status == "queued"
    item = service._queue.get_nowait()
    await service._process_trigger_internal(item)

    session_data = session_store.get("uncertain-session")
    assert session_data is not None
    assert session_data.products == []
    assert session_data.total_price == 0


@pytest.mark.asyncio
async def test_trigger_service_uses_active_product_snapshot_after_store_clear(monkeypatch, tmp_path, session_store):
    import asyncio

    import model_service.service.trigger_service as trigger_service_module
    from model_service.service.trigger_service import TriggerService

    trace_factory = make_trace_factory(tmp_path)
    monkeypatch.setattr(trigger_service_module, "TriggerTraceContext", trace_factory)
    monkeypatch.setattr(
        trigger_service_module,
        "generate_session_id",
        lambda zone: "snapshot-session",
    )
    monkeypatch.setattr(trigger_service_module.config.async_streaming, "enabled", True)

    class FakeVideoProcessor:
        async def process_videos_async(self, **kwargs):
            return SimpleNamespace(
                vote_results=[],
                stats=SimpleNamespace(top_frames=1, side_frames=0, processing_time_ms=5.0),
            )

    class FakeActiveProductStore:
        def __init__(self):
            self._products = [
                SimpleNamespace(
                    yolo_class_id=26,
                    product_idx="P26",
                    product_name="Chicken Mayo",
                    stock_qty=1,
                    product_weight=365.0,
                    sale_price=3500,
                )
            ]

        def has_products(self):
            return bool(self._products)

        def get_allowed_class_ids(self):
            return [product.yolo_class_id for product in self._products if product.stock_qty > 0]

        def get_all_products(self):
            return list(self._products)

        def get_by_yolo_class_id(self, product_id):
            for product in self._products:
                if product.yolo_class_id == product_id:
                    return product
            return None

        def clear(self):
            self._products = []

    active_product_store = FakeActiveProductStore()
    engine = MagicMock()
    engine.judge.return_value = complete_judgment_result()

    service = TriggerService(
        video_processor=FakeVideoProcessor(),
        engine=engine,
        session_store=session_store,
        active_product_store=active_product_store,
    )
    service._queue = asyncio.Queue(maxsize=service.QUEUE_MAX_SIZE)

    video_path = tmp_path / "top.avi"
    video_path.write_bytes(b"avi")
    output = await service.enqueue_trigger(create_input_for_delta(video_path, -365.0))
    assert output.status == "queued"

    item = service._queue.get_nowait()
    active_product_store.clear()
    await service._process_trigger_internal(item)

    session_data = session_store.get("snapshot-session")
    assert session_data is not None
    assert len(session_data.products) == 1
    assert session_data.products[0].product_idx == "P26"


@pytest.mark.asyncio
async def test_trigger_service_cancels_balanced_pending_removal_before_video(
    monkeypatch,
    tmp_path,
    session_store,
):
    import asyncio

    import model_service.service.trigger_service as trigger_service_module
    from model_service.service.trigger_service import TriggerService

    session_ids = iter(["remove-a", "return-a", "remove-a-again"])
    trace_factory = make_trace_factory(tmp_path)
    monkeypatch.setattr(trigger_service_module, "TriggerTraceContext", trace_factory)
    monkeypatch.setattr(
        trigger_service_module,
        "generate_session_id",
        lambda zone: next(session_ids),
    )
    monkeypatch.setattr(trigger_service_module.config.async_streaming, "enabled", True)

    class FakeVideoProcessor:
        def __init__(self):
            self.calls = 0

        async def process_videos_async(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                vote_results=[],
                stats=SimpleNamespace(top_frames=1, side_frames=0, processing_time_ms=5.0),
            )

    video_processor = FakeVideoProcessor()
    engine = MagicMock()
    engine.judge.return_value = complete_judgment_result()
    service = TriggerService(
        video_processor=video_processor,
        engine=engine,
        session_store=session_store,
    )
    service._queue = asyncio.Queue(maxsize=service.QUEUE_MAX_SIZE)

    first_video = tmp_path / "first.avi"
    return_video = tmp_path / "return.avi"
    final_video = tmp_path / "final.avi"
    first_video.write_bytes(b"avi")
    return_video.write_bytes(b"avi")
    final_video.write_bytes(b"avi")

    first = await service.enqueue_trigger(create_input_for_delta(first_video, -365.0))
    returned = await service.enqueue_trigger(create_input_for_delta(return_video, 365.0))
    final = await service.enqueue_trigger(create_input_for_delta(final_video, -365.0))

    assert first.status == "queued"
    assert returned.status == "complete"
    assert final.status == "queued"

    first_item = service._queue.get_nowait()
    await service._process_trigger_internal(first_item)
    final_item = service._queue.get_nowait()
    await service._process_trigger_internal(final_item)

    assert video_processor.calls == 1
    assert session_store.get("remove-a").processing_stage == "skipped_balanced"
    assert session_store.get("return-a").processing_stage == "balanced_out"
    assert session_store.get("remove-a-again").products[0].product_id == 26


@pytest.mark.asyncio
async def test_trigger_service_loadcell_return_only_updates_door_session_without_video(
    monkeypatch,
    tmp_path,
    session_store,
):
    import asyncio

    import model_service.service.trigger_service as trigger_service_module
    from model_service.service.trigger_service import TriggerService
    from model_service.session import DoorSessionStore

    trace_factory = make_trace_factory(tmp_path)
    monkeypatch.setattr(trigger_service_module, "TriggerTraceContext", trace_factory)
    monkeypatch.setattr(
        trigger_service_module,
        "generate_session_id",
        lambda zone: "return-only-session",
    )

    class FakeVideoProcessor:
        def __init__(self):
            self.calls = 0

        async def process_videos_async(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                vote_results=[],
                stats=SimpleNamespace(top_frames=1, side_frames=0, processing_time_ms=5.0),
            )

    video_processor = FakeVideoProcessor()
    engine = MagicMock()
    door_store = DoorSessionStore(yaml_dir=str(tmp_path / "sessions"))
    try:
        door_store.get_or_start_global_session()
        service = TriggerService(
            video_processor=video_processor,
            engine=engine,
            session_store=session_store,
            door_session_store=door_store,
        )
        service._queue = asyncio.Queue(maxsize=service.QUEUE_MAX_SIZE)

        video_path = tmp_path / "return.avi"
        video_path.write_bytes(b"avi")
        output = await service.enqueue_trigger(create_input_for_delta(video_path, 365.0))

        assert output.status == "complete"
        assert video_processor.calls == 0
        assert session_store.get("return-only-session").processing_stage == (
            "loadcell_return_only"
        )
        door_session = door_store.get_session(zone=1)
        assert door_session is not None
        assert door_session.triggers[0].is_return is True
        assert door_session.triggers[0].delta_weight == 365.0
    finally:
        door_store.shutdown()


@pytest.mark.asyncio
async def test_trigger_service_waits_for_stable_return_delta_before_commit(
    monkeypatch,
    tmp_path,
    session_store,
):
    import asyncio

    import model_service.service.trigger_service as trigger_service_module
    from model_service.service.trigger_service import LoadcellReading, TriggerService
    from model_service.session import DoorSessionStore

    trace_factory = make_trace_factory(tmp_path)
    monkeypatch.setattr(trigger_service_module, "TriggerTraceContext", trace_factory)
    monkeypatch.setattr(
        trigger_service_module,
        "generate_session_id",
        lambda zone: "return-stabilized-session",
    )
    monkeypatch.setattr(
        trigger_service_module.config.trigger,
        "return_stabilization_wait_seconds",
        0.25,
    )

    class FakeVideoProcessor:
        def __init__(self):
            self.calls = 0

        async def process_videos_async(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                vote_results=[],
                stats=SimpleNamespace(top_frames=1, side_frames=0, processing_time_ms=5.0),
            )

    video_processor = FakeVideoProcessor()
    door_store = DoorSessionStore(yaml_dir=str(tmp_path / "sessions"))
    try:
        door_store.get_or_start_global_session()
        service = TriggerService(
            video_processor=video_processor,
            engine=MagicMock(),
            session_store=session_store,
            door_session_store=door_store,
        )
        service._queue = asyncio.Queue(maxsize=service.QUEUE_MAX_SIZE)

        video_path = tmp_path / "return-stabilized.avi"
        video_path.write_bytes(b"avi")
        input_data = create_input(video_path, short_loadcells_for_delta(180.0))
        sleep_calls = []

        async def fake_sleep(seconds):
            sleep_calls.append(seconds)
            input_data.loadcells = [
                LoadcellReading(**item) for item in loadcells_for_delta(365.0)
            ]

        monkeypatch.setattr(trigger_service_module.asyncio, "sleep", fake_sleep)

        output = await service.enqueue_trigger(input_data)

        assert output.status == "complete"
        assert sleep_calls == [0.25]
        assert video_processor.calls == 0
        assert session_store.get("return-stabilized-session").delta_weight == 365.0
        door_session = door_store.get_session(zone=1)
        assert door_session is not None
        assert door_session.triggers[0].delta_weight == 365.0
    finally:
        door_store.shutdown()


@pytest.mark.asyncio
async def test_trigger_service_waits_for_stable_return_delta_instead_of_partial_commit(
    monkeypatch,
    tmp_path,
    session_store,
):
    import asyncio

    import model_service.service.trigger_service as trigger_service_module
    from model_service.service.trigger_service import TriggerService
    from model_service.session import DoorSessionStore

    trace_factory = make_trace_factory(tmp_path)
    monkeypatch.setattr(trigger_service_module, "TriggerTraceContext", trace_factory)
    monkeypatch.setattr(
        trigger_service_module,
        "generate_session_id",
        lambda zone: "return-waiting-session",
    )
    monkeypatch.setattr(
        trigger_service_module.config.trigger,
        "return_stabilization_wait_seconds",
        0.0,
    )

    class FakeVideoProcessor:
        def __init__(self):
            self.calls = 0

        async def process_videos_async(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                vote_results=[],
                stats=SimpleNamespace(top_frames=1, side_frames=0, processing_time_ms=5.0),
            )

    video_processor = FakeVideoProcessor()
    door_store = DoorSessionStore(yaml_dir=str(tmp_path / "sessions"))
    try:
        door_store.get_or_start_global_session()
        service = TriggerService(
            video_processor=video_processor,
            engine=MagicMock(),
            session_store=session_store,
            door_session_store=door_store,
        )
        service._queue = asyncio.Queue(maxsize=service.QUEUE_MAX_SIZE)

        video_path = tmp_path / "return-partial.avi"
        video_path.write_bytes(b"avi")
        output = await service.enqueue_trigger(
            create_input(video_path, short_loadcells_for_delta(180.0))
        )

        assert output.status == "waiting"
        assert output.waiting_for == "stable_loadcell"
        assert video_processor.calls == 0
        session_data = session_store.get("return-waiting-session")
        assert session_data.processing_stage == "return_waiting_for_stable_loadcell"
        assert session_data.delta_weight == 180.0
        assert door_store.get_session(zone=1) is None
    finally:
        door_store.shutdown()


@pytest.mark.asyncio
async def test_trigger_service_cancels_multi_remove_combo_with_single_return(
    monkeypatch,
    tmp_path,
    session_store,
):
    import asyncio

    import model_service.service.trigger_service as trigger_service_module
    from model_service.service.trigger_service import TriggerService

    session_ids = iter(["remove-a", "remove-b", "return-combo"])
    trace_factory = make_trace_factory(tmp_path)
    monkeypatch.setattr(trigger_service_module, "TriggerTraceContext", trace_factory)
    monkeypatch.setattr(
        trigger_service_module,
        "generate_session_id",
        lambda zone: next(session_ids),
    )
    monkeypatch.setattr(trigger_service_module.config.async_streaming, "enabled", True)

    class FakeVideoProcessor:
        def __init__(self):
            self.calls = 0

        async def process_videos_async(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                vote_results=[],
                stats=SimpleNamespace(top_frames=1, side_frames=0, processing_time_ms=5.0),
            )

    video_processor = FakeVideoProcessor()
    service = TriggerService(
        video_processor=video_processor,
        engine=MagicMock(),
        session_store=session_store,
    )
    service._queue = asyncio.Queue(maxsize=service.QUEUE_MAX_SIZE)

    paths = [tmp_path / f"trigger-{index}.avi" for index in range(3)]
    for path in paths:
        path.write_bytes(b"avi")

    await service.enqueue_trigger(create_input_for_delta(paths[0], -365.0, zone=1))
    await service.enqueue_trigger(create_input_for_delta(paths[1], -250.0, zone=2))
    output = await service.enqueue_trigger(create_input_for_delta(paths[2], 615.0, zone=3))

    assert output.status == "complete"
    first_item = service._queue.get_nowait()
    second_item = service._queue.get_nowait()
    await service._process_trigger_internal(first_item)
    await service._process_trigger_internal(second_item)

    assert video_processor.calls == 0
    assert session_store.get("remove-a").processing_stage == "skipped_balanced"
    assert session_store.get("remove-b").processing_stage == "skipped_balanced"
    assert session_store.get("return-combo").processing_stage == "balanced_out"


@pytest.mark.asyncio
async def test_trigger_service_uses_last_valid_active_product_snapshot_after_close_clear(
    monkeypatch,
    tmp_path,
    session_store,
):
    import asyncio

    import model_service.service.trigger_service as trigger_service_module
    from model_service.service.trigger_service import TriggerService
    from model_service.session.active_product_store import ActiveProductStore

    trace_factory = make_trace_factory(tmp_path)
    monkeypatch.setattr(trigger_service_module, "TriggerTraceContext", trace_factory)
    monkeypatch.setattr(
        trigger_service_module,
        "generate_session_id",
        lambda zone: "last-valid-snapshot-session",
    )
    monkeypatch.setattr(trigger_service_module.config.async_streaming, "enabled", True)

    class FakeVideoProcessor:
        async def process_videos_async(self, **kwargs):
            assert kwargs["allowed_class_ids"] == [26]
            return SimpleNamespace(
                vote_results=[],
                stats=SimpleNamespace(top_frames=1, side_frames=0, processing_time_ms=5.0),
            )

    active_product_store = ActiveProductStore({"BOWL_CHICKEN_MAYO": 26})
    active_product_store.set_products(
        [
            {
                "product_idx": "P26",
                "product_name": "Chicken Mayo",
                "product_eng_name": "BOWL_CHICKEN_MAYO",
                "yolo_class_id": 26,
                "sale_price": 3500,
                "product_weight": "365",
                "stock_qty": 1,
            }
        ]
    )
    active_product_store.clear()

    engine = MagicMock()
    engine.judge.return_value = complete_judgment_result()

    service = TriggerService(
        video_processor=FakeVideoProcessor(),
        engine=engine,
        session_store=session_store,
        active_product_store=active_product_store,
    )
    service._queue = asyncio.Queue(maxsize=service.QUEUE_MAX_SIZE)

    video_path = tmp_path / "top.avi"
    video_path.write_bytes(b"avi")
    output = await service.enqueue_trigger(create_input_for_delta(video_path, -365.0))
    assert output.status == "queued"

    item = service._queue.get_nowait()
    await service._process_trigger_internal(item)

    session_data = session_store.get("last-valid-snapshot-session")
    assert session_data is not None
    assert len(session_data.products) == 1
    assert session_data.products[0].product_idx == "P26"

    detail_file = read_trigger_detail_files(tmp_path / "logs")[0]
    detail = json.loads(detail_file.read_text(encoding="utf-8"))
    diagnostics = detail["active_product_diagnostics"]
    assert diagnostics["snapshot_source"] == "last_valid"
    assert diagnostics["used_last_valid_snapshot"] is True
    assert diagnostics["inference_fail_closed_reason"] is None


@pytest.mark.asyncio
async def test_trigger_service_traces_empty_active_product_allowlist(
    monkeypatch,
    tmp_path,
    session_store,
    caplog,
):
    import asyncio

    import model_service.service.trigger_service as trigger_service_module
    from model_service.service.trigger_service import TriggerService

    trace_factory = make_trace_factory(tmp_path)
    monkeypatch.setattr(trigger_service_module, "TriggerTraceContext", trace_factory)
    monkeypatch.setattr(
        trigger_service_module,
        "generate_session_id",
        lambda zone: "empty-allowlist-session",
    )
    monkeypatch.setattr(trigger_service_module.config.async_streaming, "enabled", True)

    class FakeVideoProcessor:
        async def process_videos_async(self, **kwargs):
            assert kwargs["allowed_class_ids"] == []
            return SimpleNamespace(
                vote_results=[],
                stats=SimpleNamespace(top_frames=0, side_frames=0, processing_time_ms=2.0),
            )

    class FakeActiveProductStore:
        def __init__(self):
            self._products = [
                SimpleNamespace(
                    yolo_class_id=76,
                    product_idx="DIGET",
                    product_name="BOX_ORION_DIGET_SSIN_84G",
                    stock_qty=0,
                    product_weight=0.0,
                    sale_price=2000,
                )
            ]

        def has_products(self):
            return True

        def get_allowed_class_ids(self):
            return []

        def get_all_products(self):
            return list(self._products)

        def get_by_yolo_class_id(self, product_id):
            return None

        def get_stats(self):
            return {
                "products_count": 1,
                "allowed_classes_count": 0,
                "zero_stock_products": 1,
                "stock_positive_weight_products": 0,
            }

    engine = MagicMock()
    engine.judge.return_value = complete_judgment_result()

    service = TriggerService(
        video_processor=FakeVideoProcessor(),
        engine=engine,
        session_store=session_store,
        active_product_store=FakeActiveProductStore(),
    )
    service._queue = asyncio.Queue(maxsize=service.QUEUE_MAX_SIZE)

    video_path = tmp_path / "top.avi"
    video_path.write_bytes(b"avi")

    caplog.set_level(logging.WARNING)
    output = await service.enqueue_trigger(create_input_for_delta(video_path, -365.0))
    assert output.status == "queued"

    item = service._queue.get_nowait()
    await service._process_trigger_internal(item)

    detail_file = read_trigger_detail_files(tmp_path / "logs")[0]
    detail = json.loads(detail_file.read_text(encoding="utf-8"))
    diagnostics = detail["active_product_diagnostics"]

    assert diagnostics["active_products_count"] == 1
    assert diagnostics["allowed_class_ids_count"] == 0
    assert diagnostics["stock_positive_weight_products"] == 0
    assert diagnostics["zero_stock_products"] == 1
    assert diagnostics["zero_weight_products"] == 1
    assert diagnostics["inference_fail_closed_reason"] == "empty_allowlist_fail_closed"
    assert diagnostics["store_stats"]["allowed_classes_count"] == 0
    assert "empty_allowlist_fail_closed" in caplog.text


@pytest.mark.asyncio
async def test_trigger_service_marks_missing_active_products_failure_reason(
    monkeypatch,
    tmp_path,
    session_store,
):
    import asyncio

    import model_service.service.trigger_service as trigger_service_module
    from model_service.service.trigger_service import TriggerService
    from model_service.session import DoorSessionStore

    trace_factory = make_trace_factory(tmp_path)
    monkeypatch.setattr(trigger_service_module, "TriggerTraceContext", trace_factory)
    monkeypatch.setattr(
        trigger_service_module,
        "generate_session_id",
        lambda zone: "missing-active-session",
    )
    monkeypatch.setattr(trigger_service_module.config.async_streaming, "enabled", True)

    class FakeVideoProcessor:
        async def process_videos_async(self, **kwargs):
            assert kwargs["allowed_class_ids"] is None
            return SimpleNamespace(
                vote_results=[],
                stats=SimpleNamespace(top_frames=1, side_frames=0, processing_time_ms=2.0),
            )

    class EmptyActiveProductStore:
        def has_products(self):
            return False

        def get_all_products(self):
            return []

        def get_stats(self):
            return {
                "products_count": 0,
                "allowed_classes_count": 0,
                "stock_positive_weight_products": 0,
            }

    engine = MagicMock()
    engine.judge.return_value = no_detection_judgment_result()
    door_store = DoorSessionStore(yaml_dir=str(tmp_path / "sessions"))
    try:
        door_store.get_or_start_global_session()
        service = TriggerService(
            video_processor=FakeVideoProcessor(),
            engine=engine,
            session_store=session_store,
            active_product_store=EmptyActiveProductStore(),
            door_session_store=door_store,
        )
        service._queue = asyncio.Queue(maxsize=service.QUEUE_MAX_SIZE)

        video_path = tmp_path / "top.avi"
        video_path.write_bytes(b"avi")
        output = await service.enqueue_trigger(create_input_for_delta(video_path, -365.0))
        assert output.status == "queued"

        item = service._queue.get_nowait()
        await service._process_trigger_internal(item)

        door_session = door_store.get_global_session().zone_sessions[1]
        assert door_session.triggers[0].failure_reason == "missing_active_products"

        detail_file = read_trigger_detail_files(tmp_path / "logs")[0]
        detail = json.loads(detail_file.read_text(encoding="utf-8"))
        assert (
            detail["active_product_diagnostics"]["inference_fail_closed_reason"]
            == "missing_active_product_snapshot_fail_closed"
        )
    finally:
        door_store.shutdown()


def test_trigger_trace_records_active_product_snapshot_and_rescue_diagnostics(tmp_path):
    from types import SimpleNamespace

    from model_service.video.frame_trace import TriggerTraceContext

    trace_context = TriggerTraceContext(
        session_id="field-debug",
        zone=4,
        top_path=None,
        side_path="/tmp/side.avi",
        log_dir=tmp_path / "logs",
        sample_export_dir=tmp_path / "samples",
        sample_export_enabled=False,
    )
    trace_context.record_loadcell_delta(delta_weight=-369.0, sample_count=42)
    trace_context.record_stage_count(
        class_id=8,
        class_name="CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
        stage="roi_filtered",
        camera="side",
        confidence=0.6424,
        center=(418.0, 70.0),
        roi_x_limit=360.0,
    )
    trace_context.record_active_product_snapshot(
        [
            SimpleNamespace(
                yolo_class_id=8,
                product_name="CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
                stock_qty=1,
                product_weight=369.0,
                sale_price=2000,
            )
        ],
        delta_weight=-369.0,
    )
    trace_context.record_rescue_diagnostics(
        "roi_rescue",
        {
            "accepted": 1,
            "rejections": {"weight_mismatch": 0},
            "candidates": [{"class_id": 8, "reason": "accepted"}],
        },
    )
    trace_context.finalize(status="complete")

    detail_file = next((tmp_path / "logs" / "triggers").glob("*/*.json"))
    detail = json.loads(detail_file.read_text(encoding="utf-8"))

    assert detail["loadcell"]["target_weight_abs"] == 369.0
    assert detail["loadcell"]["strict_tolerance_g"] == 5.0
    assert detail["active_product_snapshot"][0]["class_id"] == 8
    assert detail["active_product_snapshot"][0]["unit_weight_g"] == 369.0
    assert detail["stage_counts_by_class"]["8"]["unit_weight_g"] == 369.0
    assert detail["stage_counts_by_class"]["8"]["stock_qty"] == 1
    assert detail["stage_counts_by_class"]["8"]["weight_residual_g"] == 0.0
    assert detail["stage_counts_by_class"]["8"]["roi_x_max"] == 418.0
    assert detail["rescue_diagnostics"]["roi_rescue"]["accepted"] == 1


def test_trigger_trace_records_runtime_vision_config(monkeypatch, tmp_path):
    from model_service.core.config import config
    from model_service.video.frame_trace import TriggerTraceContext

    monkeypatch.setattr(
        config.vision,
        "yolo_model_path",
        "models/0204_morning.engine",
        raising=False,
    )
    monkeypatch.setattr(config.vision, "yolo_internal_conf_threshold", 0.01, raising=False)
    monkeypatch.setattr(config.vision, "top_confidence_threshold", 0.70, raising=False)
    monkeypatch.setattr(config.vision, "side_confidence_threshold", 0.70, raising=False)
    monkeypatch.setattr(config.vision, "hand_class_id", 0, raising=False)
    monkeypatch.setattr(config.vision, "hand_confidence_threshold", 0.40, raising=False)
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer", raising=False)
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy", raising=False)
    monkeypatch.setattr(config.vision, "top_k", 7, raising=False)
    monkeypatch.setattr(config.vision, "freezer_min_vote_ratio", 0.08, raising=False)
    monkeypatch.setattr(config.vision, "freezer_min_vote_count", 3, raising=False)
    monkeypatch.setattr(
        config.vision,
        "freezer_motion_min_displacement_px",
        12.0,
        raising=False,
    )
    monkeypatch.setattr(config.vision, "freezer_roi_vertical_region", "upper", raising=False)
    monkeypatch.setattr(config.vision, "freezer_roi_y_split", 240.0, raising=False)
    monkeypatch.setattr(config.vision, "freezer_lower_roi_y_split", 240.0, raising=False)
    monkeypatch.setattr(config.vision, "freezer_min_exit_path_votes", 3, raising=False)
    monkeypatch.setattr(config.weight, "freezer_confidence_tie_band", 0.08, raising=False)
    monkeypatch.setattr(config.weight, "freezer_multi_min_confidence", 0.45, raising=False)
    monkeypatch.setattr(
        config.weight,
        "freezer_vision_multi_without_weight_enabled",
        True,
        raising=False,
    )
    monkeypatch.setattr(config.vision, "side_roi_x_max", 400.0, raising=False)
    monkeypatch.setattr(config.vision, "side_roi_soft_margin_px", 5.0, raising=False)
    monkeypatch.setattr(config.vision, "ffmpeg_top_gamma", 1.2, raising=False)
    monkeypatch.setattr(config.vision, "ffmpeg_top_contrast", 1.2, raising=False)
    monkeypatch.setattr(config.vision, "ffmpeg_side_gamma", 1.0, raising=False)
    monkeypatch.setattr(config.vision, "ffmpeg_side_contrast", 1.0, raising=False)
    monkeypatch.setattr(config.async_streaming, "frame_stride", 2, raising=False)

    trace_context = TriggerTraceContext(
        session_id="vision-config",
        zone=4,
        top_path="/tmp/top.avi",
        side_path="/tmp/side.avi",
        log_dir=tmp_path / "logs",
        sample_export_dir=tmp_path / "samples",
        sample_export_enabled=False,
    )
    trace_context.finalize(status="complete")

    detail_file = next((tmp_path / "logs" / "triggers").glob("*/*.json"))
    detail = json.loads(detail_file.read_text(encoding="utf-8"))
    vision_config = detail["vision_config"]

    assert vision_config["yolo_model_path"] == "models/0204_morning.engine"
    assert vision_config["yolo_internal_conf_threshold"] == 0.01
    assert vision_config["cabinet_type"] == "freezer"
    assert vision_config["camera_layout"] == "dual_top_proxy"
    assert vision_config["freezer_handled_filter_enabled"] is True
    assert vision_config["top_k"] == 7
    assert vision_config["freezer_min_vote_ratio"] == 0.08
    assert vision_config["freezer_min_vote_count"] == 3
    assert vision_config["freezer_motion_min_displacement_px"] == 12.0
    assert vision_config["freezer_roi_vertical_region"] == "upper"
    assert vision_config["freezer_roi_y_split"] == 240.0
    assert vision_config["freezer_lower_roi_y_split_legacy"] == 240.0
    assert vision_config["freezer_min_exit_path_votes"] == 3
    assert vision_config["freezer_confidence_tie_band"] == 0.08
    assert vision_config["freezer_multi_min_confidence"] == 0.45
    assert vision_config["freezer_vision_multi_without_weight_enabled"] is True
    assert vision_config["top_confidence_threshold"] == 0.70
    assert vision_config["side_confidence_threshold"] == 0.70
    assert vision_config["hand_class_id"] == 0
    assert vision_config["hand_confidence_threshold"] == 0.40
    assert vision_config["regular_threshold"] == {"top": 0.70, "side": 0.70}
    assert vision_config["side_roi_x_max"] == 400.0
    assert vision_config["side_roi_soft_margin_px"] == 5.0
    assert vision_config["side_roi_soft_x_max"] == 405.0
    assert vision_config["ffmpeg_top_gamma"] == 1.2
    assert vision_config["ffmpeg_top_contrast"] == 1.2
    assert vision_config["ffmpeg_side_gamma"] == 1.0
    assert vision_config["ffmpeg_side_contrast"] == 1.0
    assert vision_config["async_frame_stride"] == 2


def test_trigger_trace_records_side_roi_soft_passed_stage(tmp_path):
    from model_service.video.frame_trace import TriggerTraceContext

    trace_context = TriggerTraceContext(
        session_id="side-soft-roi",
        zone=4,
        top_path=None,
        side_path="/tmp/side.avi",
        log_dir=tmp_path / "logs",
        sample_export_dir=tmp_path / "samples",
        sample_export_enabled=False,
    )
    trace_context.record_stage_count(
        class_id=75,
        class_name="BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
        stage="side_roi_soft_passed",
        camera="side",
        confidence=0.646,
        center=(402.5, 240.0),
        roi_x_limit=405.0,
    )
    trace_context.finalize(status="complete")

    detail_file = next((tmp_path / "logs" / "triggers").glob("*/*.json"))
    detail = json.loads(detail_file.read_text(encoding="utf-8"))
    stage_entry = detail["stage_counts_by_class"]["75"]

    assert stage_entry["side_roi_soft_passed"] == 1
    assert stage_entry["roi_x_avg"] == 402.5
    assert stage_entry["roi_x_limit"] == 405.0
    assert stage_entry["side_roi_soft_passed_samples"] == [
        {"camera": "side", "center_x": 402.5, "center_y": 240.0}
    ]


def test_trigger_trace_marks_visually_seen_weight_rejected_class(tmp_path):
    from types import SimpleNamespace

    from model_service.video.frame_trace import TriggerTraceContext

    trace_context = TriggerTraceContext(
        session_id="weight-rejected",
        zone=1,
        top_path="/tmp/top.avi",
        side_path=None,
        log_dir=tmp_path / "logs",
        sample_export_dir=tmp_path / "samples",
        sample_export_enabled=False,
    )
    trace_context.record_loadcell_delta(delta_weight=-70.0, sample_count=21)
    trace_context.record_stage_count(
        class_id=110,
        class_name="BOX_SEYANG_BAKED_EGGS_70G",
        stage="raw",
        camera="top",
        confidence=0.025,
        center=(210.0, 240.0),
    )
    trace_context.record_stage_count(
        class_id=110,
        class_name="BOX_SEYANG_BAKED_EGGS_70G",
        stage="threshold_filtered",
        camera="top",
        confidence=0.025,
        center=(210.0, 240.0),
    )
    trace_context.record_threshold_rescue_candidates(
        [
            {
                "class_id": 110,
                "class_name": "BOX_SEYANG_BAKED_EGGS_70G",
                "vote_count": 5,
                "max_confidence": 0.025,
                "avg_confidence": 0.02,
                "source": "threshold_rescue",
                "top_motion_passed": True,
                "side_motion_passed": False,
            }
        ],
        {110: 95.0},
    )
    trace_context.record_active_product_snapshot(
        [
            SimpleNamespace(
                yolo_class_id=110,
                product_name="BOX_SEYANG_BAKED_EGGS_70G",
                stock_qty=1,
                product_weight=95.0,
                sale_price=2500,
            )
        ],
        delta_weight=-70.0,
    )
    trace_context.finalize(status="complete")

    detail_file = next((tmp_path / "logs" / "triggers").glob("*/*.json"))
    detail = json.loads(detail_file.read_text(encoding="utf-8"))
    stage_entry = detail["stage_counts_by_class"]["110"]

    assert detail["loadcell"]["target_weight_abs"] == 70.0
    assert detail["loadcell"]["rescue_tolerance_g"] == 5.0
    assert stage_entry["unit_weight_g"] == 95.0
    assert stage_entry["delta_abs_g"] == 70.0
    assert stage_entry["weight_residual_g"] == 25.0
    assert stage_entry["rescue_weight_residual_g"] == 25.0
    assert stage_entry["vision_seen_weight_rejected"] is True
    assert stage_entry["strict_candidate_weight_mismatch"] is True
    assert stage_entry["weight_gate_passed"] is False
    assert stage_entry["rescue_tolerance_g"] == 5.0


def _fallback_trigger_client(
    monkeypatch,
    tmp_path,
    session_store,
    engine,
    video_processor,
    *,
    session_id: str,
    active_products: list | None = None,
):
    import model_service.api.routes.trigger as trigger_route_module
    from model_service.api.routes.trigger import (
        get_active_product_store_optional,
        get_decision_engine,
        get_door_session_store_optional,
        get_session_store,
        get_trigger_service_optional,
        get_video_processor,
        router,
    )

    trace_factory = make_trace_factory(tmp_path)
    monkeypatch.setattr(trigger_route_module, "TriggerTraceContext", trace_factory)
    monkeypatch.setattr(
        trigger_route_module,
        "generate_session_id",
        lambda zone: session_id,
    )

    if active_products is None:
        active_products = [
            SimpleNamespace(
                product_name="Chicken Mayo",
                product_idx="P26",
                stock_qty=5,
                yolo_class_id=26,
                product_weight=365.0,
                sale_price=3500,
                has_loadcell="true",
            )
        ]

    class FakeActiveProductStore:
        def get_all_products(self):
            return active_products

        def get_by_yolo_class_id(self, product_id):
            for product in active_products:
                if getattr(product, "yolo_class_id", None) == product_id:
                    return product
            return None

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_video_processor] = lambda: video_processor
    app.dependency_overrides[get_decision_engine] = lambda: engine
    app.dependency_overrides[get_session_store] = lambda: session_store
    app.dependency_overrides[get_active_product_store_optional] = (
        lambda: FakeActiveProductStore()
    )
    app.dependency_overrides[get_door_session_store_optional] = lambda: None
    app.dependency_overrides[get_trigger_service_optional] = lambda: None
    return TestClient(app)


def test_trigger_route_fallback_writes_trace_entry(monkeypatch, tmp_path, session_store):
    import model_service.api.routes.trigger as trigger_route_module
    from model_service.api.routes.trigger import (
        get_active_product_store_optional,
        get_decision_engine,
        get_door_session_store_optional,
        get_session_store,
        get_trigger_service_optional,
        get_video_processor,
        router,
    )

    trace_factory = make_trace_factory(tmp_path)
    monkeypatch.setattr(trigger_route_module, "TriggerTraceContext", trace_factory)
    monkeypatch.setattr(trigger_route_module, "generate_session_id", lambda zone: "fallback-session")

    class FakeVideoProcessor:
        def process_videos(self, **kwargs):
            trace_context = kwargs["trace_context"]
            trace_context.plan_camera("top", 3)
            for index in range(3):
                trace_context.record_frame("top", index, sample_frame(index))
            return SimpleNamespace(
                vote_results=[],
                stats=SimpleNamespace(top_frames=3, side_frames=0, processing_time_ms=8.0),
            )

    engine = MagicMock()
    engine.judge.return_value = complete_judgment_result()
    active_products = [SimpleNamespace(product_name="Chicken Mayo", stock_qty=5)]

    class FakeActiveProductStore:
        def get_all_products(self):
            return active_products

        def get_by_yolo_class_id(self, product_id):
            return None

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_video_processor] = lambda: FakeVideoProcessor()
    app.dependency_overrides[get_decision_engine] = lambda: engine
    app.dependency_overrides[get_session_store] = lambda: session_store
    app.dependency_overrides[get_active_product_store_optional] = lambda: FakeActiveProductStore()
    app.dependency_overrides[get_door_session_store_optional] = lambda: None
    app.dependency_overrides[get_trigger_service_optional] = lambda: None

    video_path = tmp_path / "fallback.avi"
    video_path.write_bytes(b"avi")

    client = TestClient(app)
    response = client.post(
        "/trigger",
        json={
            "zone": 1,
            "loadcells": loadcells_for_delta(-365.0),
            "videos": {"top": str(video_path), "side": None},
        },
    )

    assert response.status_code == 200
    assert engine.judge.call_args.kwargs["active_products"] is active_products
    entries = read_trace_entries(tmp_path / "logs")
    assert len(entries) == 1
    assert entries[0]["status"] == "complete"
    assert entries[0]["session_id"] == "fallback-session"


def test_trigger_route_refrigerated_ignores_global_loadcells(
    monkeypatch,
    tmp_path,
    session_store,
):
    from model_service.core.config import config

    monkeypatch.setattr(config.machine, "cabinet_type", "refrigerated")

    class FakeVideoProcessor:
        def process_videos(self, **kwargs):
            return SimpleNamespace(
                vote_results=[],
                stats=SimpleNamespace(
                    top_frames=1,
                    side_frames=0,
                    processing_time_ms=1.0,
                ),
            )

    engine = MagicMock()
    engine.judge.return_value = complete_judgment_result()
    client = _fallback_trigger_client(
        monkeypatch,
        tmp_path,
        session_store,
        engine,
        FakeVideoProcessor(),
        session_id="refrigerated-session",
    )
    video_path = tmp_path / "refrigerated.avi"
    video_path.write_bytes(b"avi")

    response = client.post(
        "/trigger",
        json={
            "zone": 1,
            "loadcells": loadcells_for_delta(-365.0),
            "global_loadcells": whole_machine_loadcells_for_zone_delta(-500.0),
            "videos": {"top": str(video_path), "side": None},
        },
    )

    assert response.status_code == 200
    assert engine.judge.call_args.kwargs["delta_weight"] == -365.0
    entries = read_trace_entries(tmp_path / "logs")
    assert entries[0]["loadcell"]["cabinet_type"] == "refrigerated"
    assert entries[0]["loadcell"]["loadcell_scope"] == "zone"


def test_trigger_route_freezer_prefers_zone_loadcells_when_compat_payload_present(
    monkeypatch,
    tmp_path,
    session_store,
):
    from model_service.core.config import config

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")

    class FakeVideoProcessor:
        def process_videos(self, **kwargs):
            return SimpleNamespace(
                vote_results=[],
                stats=SimpleNamespace(
                    top_frames=1,
                    side_frames=0,
                    processing_time_ms=1.0,
                ),
            )

    engine = MagicMock()
    engine.judge.return_value = complete_judgment_result()
    client = _fallback_trigger_client(
        monkeypatch,
        tmp_path,
        session_store,
        engine,
        FakeVideoProcessor(),
        session_id="freezer-zone-session",
    )
    video_path = tmp_path / "freezer-zone.avi"
    video_path.write_bytes(b"avi")

    response = client.post(
        "/trigger",
        json={
            "zone": 1,
            "loadcells": two_channel_loadcells_for_delta(-365.0),
            "global_loadcells": whole_machine_loadcells_for_zone_delta(-500.0),
            "videos": {"top": str(video_path), "side": None},
        },
    )

    assert response.status_code == 200
    assert engine.judge.call_args.kwargs["delta_weight"] == -365.0
    loadcell = read_trace_entries(tmp_path / "logs")[0]["loadcell"]
    assert loadcell["cabinet_type"] == "freezer"
    assert loadcell["loadcell_scope"] == "zone"
    assert loadcell["loadcell_source"] == "loadcells"
    assert loadcell["requested_zone"] == 1
    assert loadcell["effective_channel_count"] == 2


def test_trigger_route_freezer_keeps_full_channel_loadcells_as_zone_scope(
    monkeypatch,
    tmp_path,
    session_store,
):
    from model_service.core.config import config

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")

    class FakeVideoProcessor:
        def process_videos(self, **kwargs):
            return SimpleNamespace(
                vote_results=[],
                stats=SimpleNamespace(
                    top_frames=1,
                    side_frames=0,
                    processing_time_ms=1.0,
                ),
            )

    engine = MagicMock()
    engine.judge.return_value = complete_judgment_result()
    client = _fallback_trigger_client(
        monkeypatch,
        tmp_path,
        session_store,
        engine,
        FakeVideoProcessor(),
        session_id="freezer-full-loadcells-session",
    )
    video_path = tmp_path / "freezer-full.avi"
    video_path.write_bytes(b"avi")

    response = client.post(
        "/trigger",
        json={
            "zone": 1,
            "loadcells": whole_machine_loadcells_for_zone_delta(-500.0),
            "videos": {"top": str(video_path), "side": None},
        },
    )

    assert response.status_code == 200
    assert engine.judge.call_args.kwargs["delta_weight"] == -500.0
    loadcell = read_trace_entries(tmp_path / "logs")[0]["loadcell"]
    assert loadcell["loadcell_scope"] == "zone"
    assert loadcell["loadcell_source"] == "loadcells"
    assert loadcell["effective_channel_count"] == 10


def test_trigger_route_freezer_processes_zone_loadcells_without_compat_payload(
    monkeypatch,
    tmp_path,
    session_store,
):
    from model_service.core.config import config

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    video_processor = MagicMock()
    video_processor.process_videos.return_value = SimpleNamespace(
        vote_results=[],
        threshold_rescue_candidates=[],
        roi_rescue_candidates=[],
        stats=SimpleNamespace(
            top_frames=1,
            side_frames=0,
            processing_time_ms=1.0,
        ),
    )
    engine = MagicMock()
    engine.judge.return_value = complete_judgment_result()
    client = _fallback_trigger_client(
        monkeypatch,
        tmp_path,
        session_store,
        engine,
        video_processor,
        session_id="freezer-zone-loadcells-session",
    )
    video_path = tmp_path / "freezer-zone-loadcells.avi"
    video_path.write_bytes(b"avi")

    response = client.post(
        "/trigger",
        json={
            "zone": 1,
            "loadcells": two_channel_loadcells_for_delta(-365.0),
            "videos": {"top": str(video_path), "side": None},
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "complete"
    engine.judge.assert_called_once()
    video_processor.process_videos.assert_called_once()
    saved = session_store.get("freezer-zone-loadcells-session")
    assert saved.processing_stage == "complete"
    entry = read_trace_entries(tmp_path / "logs")[0]
    assert entry["status"] == "complete"
    assert entry["loadcell"]["cabinet_type"] == "freezer"
    assert entry["loadcell"]["loadcell_scope"] == "zone"
    assert entry["loadcell"]["loadcell_validation_reason"] is None
