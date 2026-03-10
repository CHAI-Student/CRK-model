from __future__ import annotations

"""
Streaming Frame Extractor using FFmpeg.

Memory-efficient frame extraction from video files using ffmpeg subprocess.
Supports hardware-accelerated decoding (NVDEC) on Jetson.

Usage:
    extractor = StreamingFrameExtractor("video.avi")
    for frame in extractor:
        # Process frame (numpy array BGR)
        pass
"""

import asyncio
import logging
import subprocess
import json
import time
from pathlib import Path
from typing import AsyncIterator, Iterator, Optional, Tuple

import numpy as np

from model_service.core.config import config

logger = logging.getLogger(__name__)


class StreamingFrameExtractor:
    """
    Memory-efficient streaming frame extractor using FFmpeg.

    Uses ffmpeg subprocess with pipe output for minimal memory footprint.
    Supports NVDEC hardware acceleration on Jetson Orin Nano.

    Attributes:
        video_path: Path to the video file
        total_frames: Total number of frames in the video
        fps: Frames per second
        width: Frame width
        height: Frame height
    """

    def __init__(
        self,
        video_path: str,
        use_hwaccel: bool = True,
        pixel_format: str = "bgr24",
        camera_type: str = "top",
    ):
        """
        Initialize frame extractor.

        Args:
            video_path: Path to the video file (AVI, MP4, etc.)
            use_hwaccel: Use hardware acceleration (NVDEC) if available
            pixel_format: Output pixel format (bgr24 for OpenCV compatibility)
            camera_type: Camera type ("top" or "side") for gamma/contrast settings (v4.6)
        """
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

    def _probe_video(self) -> bool:
        """
        Probe video metadata using ffprobe.

        AVI 파일이 아직 저장 중일 수 있으므로 10초 간격으로 최대 6번 재시도.
        총 1분 후에도 실패하면 timeout error.

        Returns:
            True if successful, False otherwise
        """
        if self._initialized:
            return True

        MAX_RETRIES = 3
        RETRY_INTERVAL = 10  # seconds

        for attempt in range(1, MAX_RETRIES + 1):
            video_file = Path(self.video_path)
            if not video_file.exists():
                logger.error(f"Video file not found: {self.video_path}")
                return False

            # 파일 크기 로깅
            file_size = video_file.stat().st_size
            logger.info(
                f"[FFPROBE] attempt={attempt}/{MAX_RETRIES}, "
                f"path={self.video_path}, size={file_size} bytes"
            )

            try:
                cmd = [
                    "ffprobe",
                    "-v", "error",  # 에러 메시지만 stderr로 출력
                    "-print_format", "json",
                    "-show_format",
                    "-show_streams",
                    "-select_streams", "v:0",
                    self.video_path,
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode != 0:
                    stderr_msg = result.stderr.strip()

                    # "Invalid data" 에러는 파일이 아직 저장 중일 수 있음 → 재시도
                    if "Invalid data" in stderr_msg and attempt < MAX_RETRIES:
                        logger.warning(
                            f"[FFPROBE] File may still be writing, retrying in {RETRY_INTERVAL}s... "
                            f"(attempt {attempt}/{MAX_RETRIES})"
                        )
                        time.sleep(RETRY_INTERVAL)
                        continue

                    # 마지막 시도 실패 또는 다른 에러
                    if attempt == MAX_RETRIES and "Invalid data" in stderr_msg:
                        logger.error(
                            f"[FFPROBE] Timeout after {MAX_RETRIES} attempts "
                            f"({MAX_RETRIES * RETRY_INTERVAL}s): path={self.video_path}"
                        )
                    else:
                        logger.error(
                            f"ffprobe failed: returncode={result.returncode}, "
                            f"stderr={result.stderr!r}, stdout={result.stdout!r}"
                        )
                    return False

                info = json.loads(result.stdout)

                # Extract video stream info
                if "streams" not in info or len(info["streams"]) == 0:
                    logger.error("No video stream found")
                    return False

                stream = info["streams"][0]
                self._width = int(stream.get("width", 0))
                self._height = int(stream.get("height", 0))

                # FPS parsing (can be "30/1" or "29.97")
                fps_str = stream.get("r_frame_rate", "0/1")
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    self._fps = float(num) / float(den) if float(den) > 0 else 0.0
                else:
                    self._fps = float(fps_str)

                # Total frames
                self._total_frames = int(stream.get("nb_frames", 0))

                # Duration from format
                if "format" in info:
                    self._duration = float(info["format"].get("duration", 0))

                # If nb_frames is missing, estimate from duration
                if self._total_frames == 0 and self._duration > 0 and self._fps > 0:
                    self._total_frames = int(self._duration * self._fps)

                self._initialized = True

                logger.debug(
                    f"Video probed: {self.video_path}, "
                    f"{self._width}x{self._height}, "
                    f"{self._fps:.2f}fps, "
                    f"{self._total_frames} frames, "
                    f"{self._duration:.2f}s"
                )
                return True

            except subprocess.TimeoutExpired:
                logger.error("ffprobe timeout")
                return False
            except json.JSONDecodeError as e:
                logger.error(f"ffprobe JSON parse error: {e}")
                return False
            except Exception as e:
                logger.error(f"ffprobe failed: {e}")
                return False

        # 모든 재시도 실패 (이론상 도달하지 않음)
        return False

    def _check_hwaccel(self) -> bool:
        """
        Check if NVDEC hardware acceleration is available.

        Returns:
            True if NVDEC is available
        """
        if self._hwaccel_available is not None:
            return self._hwaccel_available

        try:
            cmd = ["ffmpeg", "-hwaccels"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
            )
            self._hwaccel_available = "cuda" in result.stdout.lower()

            if self._hwaccel_available:
                logger.info("NVDEC hardware acceleration available")
            else:
                logger.debug("NVDEC not available, using CPU decoding")

            return self._hwaccel_available

        except Exception as e:
            logger.debug(f"hwaccel check failed: {e}")
            self._hwaccel_available = False
            return False

    @property
    def total_frames(self) -> int:
        """Total number of frames in the video."""
        if not self._initialized:
            self._probe_video()
        return self._total_frames

    @property
    def fps(self) -> float:
        """Frames per second."""
        if not self._initialized:
            self._probe_video()
        return self._fps

    @property
    def width(self) -> int:
        """Frame width."""
        if not self._initialized:
            self._probe_video()
        return self._width

    @property
    def height(self) -> int:
        """Frame height."""
        if not self._initialized:
            self._probe_video()
        return self._height

    @property
    def duration_seconds(self) -> float:
        """Video duration in seconds."""
        if not self._initialized:
            self._probe_video()
        return self._duration

    @property
    def frame_size(self) -> int:
        """Size of one frame in bytes (BGR24)."""
        return self._width * self._height * 3

    def _build_ffmpeg_cmd(self) -> list:
        """
        Build ffmpeg command for frame extraction.

        Jetson Orin Nano 최적화:
        - MJPEG 코덱 명시 (-c:v mjpeg)
        - NVDEC 하드웨어 가속 (가능한 경우)
        - Gamma/Contrast 보정 필터 (v4.4)
        """
        cmd = ["ffmpeg"]

        # Hardware acceleration (Jetson NVDEC)
        if self.use_hwaccel and self._check_hwaccel():
            cmd.extend(["-hwaccel", "cuda"])

        # MJPEG 코덱 명시 (AVI 파일용)
        cmd.extend(["-c:v", "mjpeg"])

        # Input
        cmd.extend(["-i", self.video_path])

        # Gamma/Contrast 보정 필터 - 카메라별 설정 (v4.6)
        if self.camera_type == "side":
            gamma = config.vision.ffmpeg_side_gamma
            contrast = config.vision.ffmpeg_side_contrast
        else:  # "top" 또는 기본값
            gamma = config.vision.ffmpeg_top_gamma
            contrast = config.vision.ffmpeg_top_contrast
        cmd.extend(["-vf", f"eq=gamma={gamma}:contrast={contrast}"])
        logger.debug(f"[FFMPEG] {self.camera_type} camera: gamma={gamma}, contrast={contrast}")

        # Output format: raw video to pipe
        cmd.extend([
            "-f", "rawvideo",
            "-pix_fmt", self.pixel_format,
            "-an",  # No audio
            "-sn",  # No subtitles
            "-v", "error",  # Only show errors
            "pipe:1",  # Output to stdout
        ])

        return cmd

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Iterate over frames.

        Yields:
            Frame as numpy array (BGR format, H x W x 3)
        """
        if not self._probe_video():
            return

        logger.info(f"[FFMPEG] ========== 프레임 추출 시작 ==========")
        logger.info(f"[FFMPEG] 비디오: {self.video_path}")
        logger.info(f"[FFMPEG] 해상도: {self._width}x{self._height}, FPS: {self._fps:.1f}")
        logger.info(f"[FFMPEG] 예상 프레임: {self._total_frames}개")
        hwaccel_status = 'NVDEC' if self.use_hwaccel and self._hwaccel_available else 'CPU'
        logger.info(f"[FFMPEG] HWACCEL: {hwaccel_status}")

        cmd = self._build_ffmpeg_cmd()
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        process = None
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self.frame_size,
            )

            frame_count = 0
            while True:
                # Read exactly one frame
                raw_frame = process.stdout.read(self.frame_size)

                if len(raw_frame) < self.frame_size:
                    # End of stream or error
                    break

                # Convert to numpy array
                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                frame = frame.reshape((self._height, self._width, 3))

                frame_count += 1
                yield frame

            logger.info(f"[FFMPEG] 프레임 추출 완료: {frame_count}개")

        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")

        finally:
            if process is not None:
                process.stdout.close()
                process.stderr.close()
                process.terminate()
                process.wait(timeout=5)

    async def __aiter__(self) -> AsyncIterator[np.ndarray]:
        """
        Async iterate over frames (v5.3).

        비동기 프레임 스트리밍. asyncio.create_subprocess_exec()를 사용하여
        I/O 대기 중 다른 태스크가 실행될 수 있도록 합니다.

        Yields:
            Frame as numpy array (BGR format, H x W x 3)
        """
        # _probe_video()는 time.sleep을 포함한 동기 함수이므로
        # 이벤트 루프 블로킹 방지를 위해 스레드풀에서 실행
        loop = asyncio.get_running_loop()
        probe_ok = await loop.run_in_executor(None, self._probe_video)
        if not probe_ok:
            return

        logger.info(f"[FFMPEG-ASYNC] ========== 비동기 프레임 추출 시작 ==========")
        logger.info(f"[FFMPEG-ASYNC] 비디오: {self.video_path}")
        logger.info(f"[FFMPEG-ASYNC] 해상도: {self._width}x{self._height}, FPS: {self._fps:.1f}")
        logger.info(f"[FFMPEG-ASYNC] 예상 프레임: {self._total_frames}개")
        hwaccel_status = 'NVDEC' if self.use_hwaccel and self._hwaccel_available else 'CPU'
        logger.info(f"[FFMPEG-ASYNC] HWACCEL: {hwaccel_status}")

        cmd = self._build_ffmpeg_cmd()
        logger.debug(f"FFmpeg async command: {' '.join(cmd)}")

        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            frame_count = 0
            while True:
                # Read exactly one frame asynchronously
                raw_frame = await process.stdout.read(self.frame_size)

                if len(raw_frame) < self.frame_size:
                    # End of stream or error
                    break

                # Convert to numpy array
                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                frame = frame.reshape((self._height, self._width, 3))

                frame_count += 1
                yield frame

            logger.info(f"[FFMPEG-ASYNC] 프레임 추출 완료: {frame_count}개")

        except asyncio.CancelledError:
            logger.warning("[FFMPEG-ASYNC] Frame extraction cancelled")
            raise

        except Exception as e:
            logger.error(f"[FFMPEG-ASYNC] Frame extraction failed: {e}")

        finally:
            if process is not None:
                # Cleanup: terminate process and close streams
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("[FFMPEG-ASYNC] Process kill timeout, forcing kill")
                    process.kill()
                    await process.wait()
                except Exception as e:
                    logger.warning(f"[FFMPEG-ASYNC] Cleanup error: {e}")

    def read_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Read a specific frame by number.

        Uses ffmpeg seek for efficient random access.

        Args:
            frame_number: Frame number (0-indexed)

        Returns:
            Frame as numpy array (BGR) or None if failed
        """
        if not self._probe_video():
            return None

        if frame_number < 0 or frame_number >= self._total_frames:
            logger.warning(
                f"Frame number out of range: {frame_number} "
                f"(total: {self._total_frames})"
            )
            return None

        # Calculate timestamp for seeking
        timestamp = frame_number / self._fps if self._fps > 0 else 0

        cmd = ["ffmpeg"]

        # Hardware acceleration
        if self.use_hwaccel and self._check_hwaccel():
            cmd.extend(["-hwaccel", "cuda"])

        # Seek before input (fast seek)
        cmd.extend(["-ss", f"{timestamp:.6f}"])

        # Input
        cmd.extend(["-i", self.video_path])

        # Output: single frame
        cmd.extend([
            "-frames:v", "1",
            "-f", "rawvideo",
            "-pix_fmt", self.pixel_format,
            "-v", "error",
            "pipe:1",
        ])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=10,
            )

            if len(result.stdout) < self.frame_size:
                logger.warning(f"Failed to read frame {frame_number}")
                return None

            frame = np.frombuffer(result.stdout[:self.frame_size], dtype=np.uint8)
            frame = frame.reshape((self._height, self._width, 3))
            return frame

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout reading frame {frame_number}")
            return None
        except Exception as e:
            logger.error(f"Failed to read frame {frame_number}: {e}")
            return None

    def __enter__(self) -> "StreamingFrameExtractor":
        """Context manager entry."""
        self._probe_video()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        pass  # No persistent resources to clean up


class CV2FrameExtractor:
    """
    Fallback frame extractor using OpenCV.

    Used when ffmpeg is not available.
    """

    def __init__(self, video_path: str):
        """Initialize with video path."""
        import cv2
        self.video_path = str(video_path)
        self._cap: Optional[cv2.VideoCapture] = None
        self._initialized = False

    def _init_capture(self) -> bool:
        """Initialize video capture."""
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
        """Iterate over frames."""
        if not self._init_capture():
            return

        while self._cap is not None and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame

        self.release()

    def release(self) -> None:
        """Release video capture."""
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
) -> "StreamingFrameExtractor | CV2FrameExtractor":
    """
    Factory function to create appropriate frame extractor.

    Args:
        video_path: Path to video file
        prefer_ffmpeg: Prefer ffmpeg over cv2 (default: True)
        use_hwaccel: Use hardware acceleration if available
        camera_type: Camera type ("top" or "side") for gamma/contrast settings (v4.6)

    Returns:
        StreamingFrameExtractor (ffmpeg) or CV2FrameExtractor (fallback)
    """
    if prefer_ffmpeg:
        # Check if ffmpeg is available
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                timeout=5,
            )
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
