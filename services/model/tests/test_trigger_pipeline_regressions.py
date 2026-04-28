import asyncio
from collections import deque

import numpy as np
import pytest


def _make_loadcell(timestamp: str, value: float) -> dict:
    formatted = f"{value:+.1f}"
    return {
        "timestamp": timestamp,
        "raw_value": [formatted, formatted],
        "filtered_value": [formatted, formatted],
        "filter_method": "exponential",
    }


def _make_weight_series(values: list[float]) -> list[dict]:
    return [
        _make_loadcell(f"2026-04-02T15:00:{index:02d}.000Z", value)
        for index, value in enumerate(values)
    ]


class _FakeAsyncStdout:
    def __init__(self, chunks: list[bytes]):
        self._chunks = deque(chunks)

    async def readexactly(self, size: int) -> bytes:
        payload = bytearray()
        while len(payload) < size:
            if not self._chunks:
                raise asyncio.IncompleteReadError(bytes(payload), size)
            payload.extend(self._chunks.popleft())
        if len(payload) > size:
            overflow = bytes(payload[size:])
            self._chunks.appendleft(overflow)
            del payload[size:]
        return bytes(payload)

    async def read(self) -> bytes:
        chunks = list(self._chunks)
        self._chunks.clear()
        return b"".join(chunks)


class _FakeAsyncStderr:
    def __init__(self, text: str = ""):
        self._payload = text.encode("utf-8")

    async def read(self) -> bytes:
        payload = self._payload
        self._payload = b""
        return payload


class _FakeAsyncProcess:
    def __init__(self, stdout_chunks: list[bytes], stderr_text: str = ""):
        self.stdout = _FakeAsyncStdout(stdout_chunks)
        self.stderr = _FakeAsyncStderr(stderr_text)
        self.returncode = 0
        self._terminated = False

    def terminate(self) -> None:
        self._terminated = True

    def kill(self) -> None:
        self._terminated = True
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode


@pytest.mark.asyncio
async def test_async_frame_extractor_reassembles_partial_reads(monkeypatch):
    import model_service.video.frame_extractor as frame_extractor_module
    from model_service.video.frame_extractor import StreamingFrameExtractor

    extractor = StreamingFrameExtractor("partial.avi", use_hwaccel=False)
    extractor._initialized = True
    extractor._width = 2
    extractor._height = 1
    extractor._fps = 25.0
    extractor._total_frames = 2

    frame_size = extractor.frame_size
    raw_frame_a = bytes(range(frame_size))
    raw_frame_b = bytes(range(frame_size, frame_size * 2))
    chunks = [
        raw_frame_a[:2],
        raw_frame_a[2:5],
        raw_frame_a[5:],
        raw_frame_b[:1],
        raw_frame_b[1:4],
        raw_frame_b[4:],
    ]

    monkeypatch.setattr(extractor, "_probe_video", lambda: True)
    monkeypatch.setattr(
        extractor,
        "_build_ffmpeg_cmd",
        lambda decoder="auto": ["ffmpeg", "-i", "partial.avi", decoder],
    )
    async def fake_create_subprocess_exec(*args, **kwargs):
        return _FakeAsyncProcess(chunks)

    monkeypatch.setattr(
        frame_extractor_module.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    frames = []
    async for frame in extractor:
        frames.append(frame.copy())

    assert len(frames) == 2
    assert frames[0].shape == (1, 2, 3)
    assert frames[1].shape == (1, 2, 3)
    assert extractor.last_diagnostics.decoded_frames == 2
    assert extractor.last_diagnostics.bytes_read == frame_size * 2
    assert extractor.last_diagnostics.final_branch == "async"


@pytest.mark.asyncio
async def test_async_frame_extractor_retries_sync_when_async_returns_zero(monkeypatch):
    import model_service.video.frame_extractor as frame_extractor_module
    from model_service.core.config import config
    from model_service.video.frame_extractor import StreamingFrameExtractor

    extractor = StreamingFrameExtractor("retry.avi", use_hwaccel=False)
    extractor._initialized = True
    extractor._width = 2
    extractor._height = 1
    extractor._fps = 25.0
    extractor._total_frames = 3

    fallback_frame = np.full((1, 2, 3), 7, dtype=np.uint8)

    async def fake_async_attempt():
        raise asyncio.IncompleteReadError(b"", extractor.frame_size)

    async def fake_create_subprocess_exec(*args, **kwargs):
        return _FakeAsyncProcess([])

    def fake_sync_frames(diagnostics):
        diagnostics.decoded_frames = 1
        diagnostics.bytes_read = extractor.frame_size
        yield fallback_frame

    monkeypatch.setattr(config.async_streaming, "zero_frame_retry_enabled", True)
    monkeypatch.setattr(extractor, "_probe_video", lambda: True)
    monkeypatch.setattr(
        extractor,
        "_build_ffmpeg_cmd",
        lambda decoder="auto": ["ffmpeg", "-i", "retry.avi", decoder],
    )
    monkeypatch.setattr(extractor, "_read_exact_frame_async", lambda process, diagnostics: fake_async_attempt())
    monkeypatch.setattr(extractor, "_iter_sync_frames", fake_sync_frames)
    monkeypatch.setattr(
        frame_extractor_module.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    frames = []
    async for frame in extractor:
        frames.append(frame.copy())

    assert len(frames) == 1
    assert np.array_equal(frames[0], fallback_frame)
    assert extractor.last_diagnostics.retry_used is True
    assert extractor.last_diagnostics.final_branch == "sync_retry"


def test_ffmpeg_primary_decode_uses_auto_codec_then_mjpeg_fallback():
    from model_service.video.frame_extractor import StreamingFrameExtractor

    extractor = StreamingFrameExtractor("sample.avi", use_hwaccel=False)

    primary = extractor._build_ffmpeg_cmd()
    fallback = extractor._build_ffmpeg_cmd(decoder="mjpeg")

    assert "-i" in primary
    assert "-c:v" not in primary
    assert fallback[fallback.index("-c:v") + 1] == "mjpeg"


def test_ffmpeg_zero_frame_diagnostics_include_decoder_and_counts(caplog):
    from model_service.video.frame_extractor import FrameExtractionDiagnostics, StreamingFrameExtractor

    diagnostics = FrameExtractionDiagnostics(
        method="sync",
        expected_frames=4,
        decoded_frames=0,
        bytes_read=17,
        partial_reads=1,
        stderr_tail="Invalid data found",
        final_branch="sync",
        decoder="auto",
    )

    StreamingFrameExtractor._log_diagnostics("[FFMPEG]", diagnostics)

    assert "expected_frames=4" in caplog.text
    assert "decoded_frames=0" in caplog.text
    assert "bytes_read=17" in caplog.text
    assert "partial_reads=1" in caplog.text
    assert "decoder=auto" in caplog.text
    assert "stderr_tail='Invalid data found'" in caplog.text


def test_loadcell_analysis_requires_confirmed_stable_tail():
    from model_service.core import loadcell_stats

    full_series = _make_weight_series(
        [1000.0, 1000.0, 1000.0, 1000.0, 900.0, 800.0, 700.0, 600.0, 500.0, 400.0, 400.0, 400.0, 400.0]
    )
    truncated_series = _make_weight_series(
        [1000.0, 1000.0, 1000.0, 1000.0, 900.0, 800.0, 700.0]
    )

    full = loadcell_stats.analyze_weight_delta(
        full_series,
        window_size=3,
        stability_threshold=1.0,
    )
    truncated = loadcell_stats.analyze_weight_delta(
        truncated_series,
        window_size=3,
        stability_threshold=1.0,
    )

    assert full.delta == pytest.approx(-600.0)
    assert full.used_simple_fallback is False
    assert full.reason == "stable_regions"
    assert truncated.delta == 0.0
    assert truncated.used_simple_fallback is False
    assert truncated.reason == "unstable_or_truncated_loadcell"
    assert truncated.sample_count < full.sample_count
