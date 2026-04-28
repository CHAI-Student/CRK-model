from __future__ import annotations

"""Streaming frame extraction helpers for AVI trigger processing."""

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Iterator, Optional

import numpy as np

from model_service.core.config import config

logger = logging.getLogger(__name__)


@dataclass
class FrameExtractionDiagnostics:
    """Per-attempt frame extraction diagnostics."""

    method: str
    expected_frames: int
    decoded_frames: int = 0
    bytes_read: int = 0
    partial_reads: int = 0
    truncated_bytes: int = 0
    ffmpeg_returncode: int | None = None
    stderr_tail: str = ""
    retry_used: bool = False
    final_branch: str = ""
    decoder: str = "auto"


def _next_iterator_item(iterator: Iterator[np.ndarray]) -> tuple[bool, Optional[np.ndarray]]:
    """Pull one item from a blocking iterator inside asyncio.to_thread."""

    try:
        return False, next(iterator)
    except StopIteration:
        return True, None


class StreamingFrameExtractor:
    """Memory-efficient ffmpeg-backed frame extractor."""

    def __init__(
        self,
        video_path: str,
        use_hwaccel: bool = True,
        pixel_format: str = "bgr24",
        camera_type: str = "top",
    ):
        self.video_path = str(video_path)
        self.use_hwaccel = use_hwaccel
        self.pixel_format = pixel_format
        self.camera_type = camera_type

        self._width: int = 0
        self._height: int = 0
        self._fps: float = 0.0
        self._total_frames: int = 0
        self._duration: float = 0.0
        self._initialized = False
        self._hwaccel_available: Optional[bool] = None
        self.last_diagnostics: Optional[FrameExtractionDiagnostics] = None

    def _probe_video(self) -> bool:
        """Probe video metadata using ffprobe."""
        if self._initialized:
            return True

        max_retries = 3
        retry_interval = 10

        for attempt in range(1, max_retries + 1):
            video_file = Path(self.video_path)
            if not video_file.exists():
                logger.error(f"Video file not found: {self.video_path}")
                return False

            logger.info(
                f"[FFPROBE] attempt={attempt}/{max_retries}, "
                f"path={self.video_path}, size={video_file.stat().st_size} bytes"
            )

            try:
                result = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-print_format",
                        "json",
                        "-show_format",
                        "-show_streams",
                        "-select_streams",
                        "v:0",
                        self.video_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except subprocess.TimeoutExpired:
                logger.error("ffprobe timeout")
                return False
            except Exception as exc:
                logger.error(f"ffprobe failed: {exc}")
                return False

            if result.returncode != 0:
                stderr_msg = result.stderr.strip()
                if "Invalid data" in stderr_msg and attempt < max_retries:
                    logger.warning(
                        f"[FFPROBE] File may still be writing, retrying in {retry_interval}s... "
                        f"(attempt {attempt}/{max_retries})"
                    )
                    time.sleep(retry_interval)
                    continue

                logger.error(
                    f"ffprobe failed: returncode={result.returncode}, "
                    f"stderr={result.stderr!r}, stdout={result.stdout!r}"
                )
                return False

            try:
                info = json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                logger.error(f"ffprobe JSON parse error: {exc}")
                return False

            streams = info.get("streams") or []
            if not streams:
                logger.error("No video stream found")
                return False

            stream = streams[0]
            self._width = int(stream.get("width", 0))
            self._height = int(stream.get("height", 0))

            fps_str = stream.get("r_frame_rate", "0/1")
            if "/" in fps_str:
                numerator, denominator = fps_str.split("/", maxsplit=1)
                denominator_value = float(denominator)
                self._fps = float(numerator) / denominator_value if denominator_value > 0 else 0.0
            else:
                self._fps = float(fps_str)

            self._total_frames = int(stream.get("nb_frames", 0))
            if "format" in info:
                self._duration = float(info["format"].get("duration", 0))
            if self._total_frames == 0 and self._duration > 0 and self._fps > 0:
                self._total_frames = int(self._duration * self._fps)

            self._initialized = True
            return True

        return False

    def _check_hwaccel(self) -> bool:
        """Check whether ffmpeg exposes CUDA decoding."""
        if self._hwaccel_available is not None:
            return self._hwaccel_available

        try:
            result = subprocess.run(
                ["ffmpeg", "-hwaccels"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            self._hwaccel_available = "cuda" in result.stdout.lower()
        except Exception as exc:
            logger.debug(f"hwaccel check failed: {exc}")
            self._hwaccel_available = False

        return self._hwaccel_available

    @property
    def total_frames(self) -> int:
        if not self._initialized:
            self._probe_video()
        return self._total_frames

    @property
    def fps(self) -> float:
        if not self._initialized:
            self._probe_video()
        return self._fps

    @property
    def width(self) -> int:
        if not self._initialized:
            self._probe_video()
        return self._width

    @property
    def height(self) -> int:
        if not self._initialized:
            self._probe_video()
        return self._height

    @property
    def duration_seconds(self) -> float:
        if not self._initialized:
            self._probe_video()
        return self._duration

    @property
    def frame_size(self) -> int:
        return self._width * self._height * 3

    def _build_ffmpeg_cmd(self, decoder: str = "auto") -> list[str]:
        cmd = ["ffmpeg"]
        if self.use_hwaccel and self._check_hwaccel():
            cmd.extend(["-hwaccel", "cuda"])

        if decoder == "mjpeg":
            cmd.extend(["-c:v", "mjpeg"])

        cmd.extend(["-i", self.video_path])

        if self.camera_type == "side":
            gamma = config.vision.ffmpeg_side_gamma
            contrast = config.vision.ffmpeg_side_contrast
        else:
            gamma = config.vision.ffmpeg_top_gamma
            contrast = config.vision.ffmpeg_top_contrast

        cmd.extend(["-vf", f"eq=gamma={gamma}:contrast={contrast}"])
        cmd.extend(
            [
                "-f",
                "rawvideo",
                "-pix_fmt",
                self.pixel_format,
                "-an",
                "-sn",
                "-v",
                "error",
                "pipe:1",
            ]
        )
        return cmd

    def _new_diagnostics(
        self,
        method: str,
        final_branch: str,
        decoder: str = "auto",
    ) -> FrameExtractionDiagnostics:
        return FrameExtractionDiagnostics(
            method=method,
            expected_frames=self._total_frames,
            final_branch=final_branch,
            decoder=decoder,
        )

    @staticmethod
    def _summarize_stderr(stderr: bytes | str, limit: int = 240) -> str:
        text = stderr.decode("utf-8", errors="ignore") if isinstance(stderr, bytes) else stderr
        text = text.strip()
        if len(text) <= limit:
            return text
        return text[-limit:]

    def _log_diagnostics(self, prefix: str, diagnostics: FrameExtractionDiagnostics) -> None:
        message = (
            f"{prefix} final_branch={diagnostics.final_branch} "
            f"expected_frames={diagnostics.expected_frames} "
            f"decoded_frames={diagnostics.decoded_frames} "
            f"bytes_read={diagnostics.bytes_read} "
            f"partial_reads={diagnostics.partial_reads} "
            f"decoder={diagnostics.decoder}"
        )
        if diagnostics.stderr_tail:
            message += f" stderr_tail={diagnostics.stderr_tail!r}"

        if diagnostics.expected_frames > 0 and diagnostics.decoded_frames == 0:
            logger.warning(message)
        else:
            logger.info(message)

    def _read_exact_frame_sync(self, stream, diagnostics: FrameExtractionDiagnostics) -> bytes:
        payload = bytearray()
        while len(payload) < self.frame_size:
            chunk = stream.read(self.frame_size - len(payload))
            if not chunk:
                break
            payload.extend(chunk)
            diagnostics.bytes_read += len(chunk)

        if not payload:
            return b""

        if len(payload) < self.frame_size:
            diagnostics.partial_reads += 1
            diagnostics.truncated_bytes += len(payload)
            return b""

        return bytes(payload)

    async def _read_exact_frame_async(
        self,
        process: asyncio.subprocess.Process,
        diagnostics: FrameExtractionDiagnostics,
    ) -> bytes:
        assert process.stdout is not None
        try:
            payload = await process.stdout.readexactly(self.frame_size)
            diagnostics.bytes_read += len(payload)
            return payload
        except asyncio.IncompleteReadError as exc:
            if exc.partial:
                diagnostics.bytes_read += len(exc.partial)
                diagnostics.partial_reads += 1
                diagnostics.truncated_bytes += len(exc.partial)
            return b""

    def _iter_sync_frames(self, diagnostics: FrameExtractionDiagnostics) -> Iterator[np.ndarray]:
        logger.info("[FFMPEG] ========== 프레임 추출 시작 ==========")
        logger.info(f"[FFMPEG] 비디오: {self.video_path}")
        logger.info(f"[FFMPEG] 해상도: {self._width}x{self._height}, FPS: {self._fps:.1f}")
        logger.info(f"[FFMPEG] 예상 프레임: {self._total_frames}개")
        hwaccel_status = "NVDEC" if self.use_hwaccel and self._hwaccel_available else "CPU"
        logger.info(f"[FFMPEG] HWACCEL: {hwaccel_status}")

        process: subprocess.Popen | None = None
        try:
            process = subprocess.Popen(
                self._build_ffmpeg_cmd(diagnostics.decoder),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self.frame_size,
            )

            while True:
                assert process.stdout is not None
                raw_frame = self._read_exact_frame_sync(process.stdout, diagnostics)
                if not raw_frame:
                    break

                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                frame = frame.reshape((self._height, self._width, 3))
                diagnostics.decoded_frames += 1
                yield frame

        finally:
            if process is not None:
                try:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("[FFMPEG] Process kill timeout, forcing kill")
                    process.kill()
                    process.wait(timeout=5)
                finally:
                    diagnostics.ffmpeg_returncode = process.returncode
                    if process.stderr is not None:
                        diagnostics.stderr_tail = self._summarize_stderr(process.stderr.read())
                    if process.stdout is not None:
                        process.stdout.close()
                    if process.stderr is not None:
                        process.stderr.close()

    async def _finalize_async_process(
        self,
        process: asyncio.subprocess.Process,
        diagnostics: FrameExtractionDiagnostics,
    ) -> None:
        stderr_payload = b""
        try:
            await asyncio.wait_for(process.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            try:
                process.terminate()
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("[FFMPEG-ASYNC] Process kill timeout, forcing kill")
                process.kill()
                await process.wait()

        if process.stderr is not None:
            stderr_payload = await process.stderr.read()

        diagnostics.ffmpeg_returncode = process.returncode
        diagnostics.stderr_tail = self._summarize_stderr(stderr_payload)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over frames synchronously."""
        if not self._probe_video():
            return

        diagnostics = self._new_diagnostics(method="sync", final_branch="sync")
        self.last_diagnostics = diagnostics
        try:
            yield from self._iter_sync_frames(diagnostics)
            if diagnostics.decoded_frames == 0 and diagnostics.expected_frames > 0:
                self._log_diagnostics("[FFMPEG]", diagnostics)
                retry_diagnostics = self._new_diagnostics(
                    method="sync_mjpeg_retry",
                    final_branch="sync_mjpeg_retry",
                    decoder="mjpeg",
                )
                retry_diagnostics.retry_used = True
                self.last_diagnostics = retry_diagnostics
                logger.warning(
                    "[FFMPEG] Zero frames decoded with auto decoder; retrying mjpeg decoder"
                )
                yield from self._iter_sync_frames(retry_diagnostics)
                diagnostics = retry_diagnostics
        except Exception as exc:
            logger.error(f"[FFMPEG] Frame extraction failed: {exc}")
        finally:
            self.last_diagnostics = diagnostics
            self._log_diagnostics("[FFMPEG]", diagnostics)

    async def __aiter__(self) -> AsyncIterator[np.ndarray]:
        """Iterate over frames asynchronously."""
        loop = asyncio.get_running_loop()
        probe_ok = await loop.run_in_executor(None, self._probe_video)
        if not probe_ok:
            return

        logger.info("[FFMPEG-ASYNC] ========== 비동기 프레임 추출 시작 ==========")
        logger.info(f"[FFMPEG-ASYNC] 비디오: {self.video_path}")
        logger.info(f"[FFMPEG-ASYNC] 해상도: {self._width}x{self._height}, FPS: {self._fps:.1f}")
        logger.info(f"[FFMPEG-ASYNC] 예상 프레임: {self._total_frames}개")
        hwaccel_status = "NVDEC" if self.use_hwaccel and self._hwaccel_available else "CPU"
        logger.info(f"[FFMPEG-ASYNC] HWACCEL: {hwaccel_status}")

        diagnostics = self._new_diagnostics(method="async", final_branch="async")
        self.last_diagnostics = diagnostics

        process: asyncio.subprocess.Process | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                *self._build_ffmpeg_cmd(diagnostics.decoder),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            while True:
                raw_frame = await self._read_exact_frame_async(process, diagnostics)
                if not raw_frame:
                    break

                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                frame = frame.reshape((self._height, self._width, 3))
                diagnostics.decoded_frames += 1
                yield frame

        except asyncio.CancelledError:
            logger.warning("[FFMPEG-ASYNC] Frame extraction cancelled")
            raise
        except Exception as exc:
            logger.error(f"[FFMPEG-ASYNC] Frame extraction failed: {exc}")
        finally:
            if process is not None:
                try:
                    await self._finalize_async_process(process, diagnostics)
                except Exception as exc:
                    logger.warning(f"[FFMPEG-ASYNC] Cleanup error: {exc}")

        if (
            diagnostics.decoded_frames == 0
            and diagnostics.expected_frames > 0
            and config.async_streaming.zero_frame_retry_enabled
        ):
            self._log_diagnostics("[FFMPEG-ASYNC]", diagnostics)
            diagnostics.retry_used = True
            logger.warning(
                "[FFMPEG-ASYNC] Zero frames decoded despite ffprobe metadata; retrying sync decode"
            )

            retry_diagnostics = self._new_diagnostics(
                method="sync_retry",
                final_branch="sync_retry",
            )
            retry_diagnostics.retry_used = True
            self.last_diagnostics = retry_diagnostics

            iterator = self._iter_sync_frames(retry_diagnostics)
            while True:
                done, frame = await asyncio.to_thread(_next_iterator_item, iterator)
                if done:
                    break
                assert frame is not None
                yield frame

            diagnostics = retry_diagnostics

            if diagnostics.decoded_frames == 0:
                self._log_diagnostics("[FFMPEG-ASYNC]", diagnostics)
                mjpeg_diagnostics = self._new_diagnostics(
                    method="sync_mjpeg_retry",
                    final_branch="sync_mjpeg_retry",
                    decoder="mjpeg",
                )
                mjpeg_diagnostics.retry_used = True
                self.last_diagnostics = mjpeg_diagnostics
                logger.warning(
                    "[FFMPEG-ASYNC] Sync retry decoded zero frames; retrying mjpeg decoder"
                )

                iterator = self._iter_sync_frames(mjpeg_diagnostics)
                while True:
                    done, frame = await asyncio.to_thread(_next_iterator_item, iterator)
                    if done:
                        break
                    assert frame is not None
                    yield frame

                diagnostics = mjpeg_diagnostics

        self.last_diagnostics = diagnostics
        self._log_diagnostics("[FFMPEG-ASYNC]", diagnostics)

    def read_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Read a specific frame by number."""
        if not self._probe_video():
            return None
        if frame_number < 0 or frame_number >= self._total_frames:
            logger.warning(
                f"Frame number out of range: {frame_number} (total: {self._total_frames})"
            )
            return None

        timestamp = frame_number / self._fps if self._fps > 0 else 0
        cmd = ["ffmpeg"]
        if self.use_hwaccel and self._check_hwaccel():
            cmd.extend(["-hwaccel", "cuda"])
        cmd.extend(["-ss", f"{timestamp:.6f}", "-i", self.video_path])
        cmd.extend(
            [
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                self.pixel_format,
                "-v",
                "error",
                "pipe:1",
            ]
        )

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=10)
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout reading frame {frame_number}")
            return None
        except Exception as exc:
            logger.error(f"Failed to read frame {frame_number}: {exc}")
            return None

        if len(result.stdout) < self.frame_size:
            logger.warning(f"Failed to read frame {frame_number}")
            return None

        frame = np.frombuffer(result.stdout[: self.frame_size], dtype=np.uint8)
        return frame.reshape((self._height, self._width, 3))

    def __enter__(self) -> "StreamingFrameExtractor":
        self._probe_video()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


class CV2FrameExtractor:
    """Fallback extractor using OpenCV only."""

    def __init__(self, video_path: str):
        import cv2

        self.video_path = str(video_path)
        self._cap: Optional[cv2.VideoCapture] = None
        self._initialized = False

    def _init_capture(self) -> bool:
        import cv2

        if self._initialized:
            return True
        if not Path(self.video_path).exists():
            logger.error(f"Video file not found: {self.video_path}")
            return False

        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            logger.error(f"Failed to open video: {self.video_path}")
            return False

        self._initialized = True
        return True

    @property
    def total_frames(self) -> int:
        import cv2

        if not self._initialized:
            self._init_capture()
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self._cap else 0

    @property
    def fps(self) -> float:
        import cv2

        if not self._initialized:
            self._init_capture()
        return self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 0.0

    @property
    def width(self) -> int:
        import cv2

        if not self._initialized:
            self._init_capture()
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._cap else 0

    @property
    def height(self) -> int:
        import cv2

        if not self._initialized:
            self._init_capture()
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._cap else 0

    def __iter__(self) -> Iterator[np.ndarray]:
        if not self._init_capture():
            return

        while self._cap is not None and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame

        self.release()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self):
        self.release()


def create_frame_extractor(
    video_path: str,
    prefer_ffmpeg: bool = True,
    use_hwaccel: bool = True,
    camera_type: str = "top",
) -> StreamingFrameExtractor | CV2FrameExtractor:
    """Create the preferred extractor implementation."""
    if prefer_ffmpeg:
        try:
            result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
            if result.returncode == 0:
                return StreamingFrameExtractor(
                    video_path,
                    use_hwaccel=use_hwaccel,
                    camera_type=camera_type,
                )
        except Exception:
            pass

        logger.warning("ffmpeg not available, falling back to cv2")

    return CV2FrameExtractor(video_path)
