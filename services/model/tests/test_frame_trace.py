import asyncio
import json
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


@pytest.mark.asyncio
async def test_trigger_service_worker_complete_writes_trace_entry(monkeypatch, tmp_path, session_store):
    import model_service.service.trigger_service as trigger_service_module
    from model_service.service.trigger_service import TriggerService

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

    service = TriggerService(
        video_processor=FakeVideoProcessor(),
        engine=engine,
        session_store=session_store,
    )
    service._queue = asyncio.Queue(maxsize=service.QUEUE_MAX_SIZE)

    video_path = tmp_path / "top.avi"
    video_path.write_bytes(b"avi")
    output = await service.enqueue_trigger(create_input(video_path, changed_weight_loadcells()))
    assert output.status == "queued"

    item = service._queue.get_nowait()
    await service._process_trigger_internal(item)

    entries = read_trace_entries(tmp_path / "logs")
    assert len(entries) == 1
    assert entries[0]["session_id"] == "worker-session"
    assert entries[0]["status"] == "complete"
    assert entries[0]["cameras"]["top"]["processed_frames"] == 4


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
    input_data = create_input(video_path, changed_weight_loadcells())

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

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_video_processor] = lambda: FakeVideoProcessor()
    app.dependency_overrides[get_decision_engine] = lambda: engine
    app.dependency_overrides[get_session_store] = lambda: session_store
    app.dependency_overrides[get_active_product_store_optional] = lambda: None
    app.dependency_overrides[get_door_session_store_optional] = lambda: None
    app.dependency_overrides[get_trigger_service_optional] = lambda: None

    video_path = tmp_path / "fallback.avi"
    video_path.write_bytes(b"avi")

    client = TestClient(app)
    response = client.post(
        "/trigger",
        json={
            "zone": 1,
            "loadcells": changed_weight_loadcells(),
            "videos": {"top": str(video_path), "side": None},
        },
    )

    assert response.status_code == 200
    entries = read_trace_entries(tmp_path / "logs")
    assert len(entries) == 1
    assert entries[0]["status"] == "complete"
    assert entries[0]["session_id"] == "fallback-session"
