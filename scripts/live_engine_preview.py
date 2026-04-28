#!/usr/bin/env python3
from __future__ import annotations

"""Live TensorRT engine preview for Jetson camera validation.

This utility is intentionally separate from the FastAPI service. Run it on the
Jetson Orin Nano Ubuntu 22.04 device to visually verify the camera stream and
the `.engine` model output with real-time bounding boxes and labels.
"""

import argparse
import logging
import os
from pathlib import Path
import sys
import time
from typing import Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PACKAGE_DIR = ROOT_DIR / "services" / "model"
if str(MODEL_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_PACKAGE_DIR))


def _bootstrap_jetson_runtime_paths() -> None:
    """Re-exec this script with the same Jetson runtime paths as model-service."""
    try:
        from model_service.core.runtime_env import (
            JETSON_ENV_SENTINEL,
            build_jetson_runtime_environment,
            is_jetson_environment,
        )
    except Exception:
        return

    if not is_jetson_environment():
        return
    if os.environ.get(JETSON_ENV_SENTINEL) == "1":
        return

    env = build_jetson_runtime_environment()
    args = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    os.execvpe(sys.executable, args, env)


_bootstrap_jetson_runtime_paths()

import cv2  # noqa: E402
from ultralytics import YOLO  # noqa: E402


LOGGER = logging.getLogger("live_engine_preview")


def _parse_source(value: str) -> int | str:
    text = value.strip()
    if text.isdigit():
        return int(text)
    return text


def _parse_classes(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    classes: list[int] = []
    for part in value.split(","):
        stripped = part.strip()
        if stripped:
            classes.append(int(stripped))
    return classes


def _open_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    source = _parse_source(args.source)
    backend = args.backend

    if backend == "gstreamer":
        capture = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
    elif backend == "v4l2":
        capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
    elif backend == "ffmpeg":
        capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    elif isinstance(source, int):
        capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
    else:
        capture = cv2.VideoCapture(source)

    if isinstance(source, int):
        if args.width > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        if args.height > 0:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if args.fps > 0:
            capture.set(cv2.CAP_PROP_FPS, args.fps)

    return capture


def _draw_overlay(frame, lines: Iterable[str]) -> None:
    y = 24
    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 24


def _create_writer(path: str, fps: float, frame_shape) -> cv2.VideoWriter:
    height, width = frame_shape[:2]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output}")
    return writer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview a TensorRT .engine model on a live Jetson camera feed.",
    )
    parser.add_argument("--model", default="models/siyeon_best.engine", help="Path to TensorRT .engine file.")
    parser.add_argument("--source", default="0", help="Camera index, video path, RTSP URL, or GStreamer pipeline.")
    parser.add_argument("--backend", choices=["auto", "v4l2", "gstreamer", "ffmpeg"], default="auto")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--classes", default=None, help="Optional comma-separated YOLO class IDs.")
    parser.add_argument("--crop-width", type=int, default=480, help="Match service inference crop width; 0 disables crop.")
    parser.add_argument("--window-name", default="CRK TensorRT Preview")
    parser.add_argument("--record", default=None, help="Optional MP4 output path for annotated preview.")
    parser.add_argument("--no-display", action="store_true", help="Run without cv2.imshow; useful with --record.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT_DIR / model_path
    if model_path.suffix != ".engine":
        LOGGER.error("This preview tool is for TensorRT .engine files, got: %s", model_path)
        return 2
    if not model_path.exists():
        LOGGER.error("TensorRT engine not found: %s", model_path)
        return 2

    LOGGER.info("loading TensorRT engine path=%s device=%s imgsz=%s", model_path, args.device, args.imgsz)
    model = YOLO(str(model_path))
    LOGGER.info("engine loaded classes=%s", len(getattr(model, "names", {}) or {}))

    capture = _open_capture(args)
    if not capture.isOpened():
        LOGGER.error("camera/video source could not be opened source=%s backend=%s", args.source, args.backend)
        return 2

    class_filter = _parse_classes(args.classes)
    writer = None
    fps_ema = 0.0
    frame_count = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                LOGGER.warning("source returned no frame after %s frames", frame_count)
                break

            if args.crop_width > 0 and frame.shape[1] > args.crop_width:
                infer_frame = frame[:, : args.crop_width]
            else:
                infer_frame = frame

            start = time.perf_counter()
            results = model.predict(
                infer_frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                half=True,
                max_det=args.max_det,
                classes=class_filter,
                verbose=False,
            )
            elapsed = max(time.perf_counter() - start, 1e-6)
            current_fps = 1.0 / elapsed
            fps_ema = current_fps if fps_ema == 0.0 else (fps_ema * 0.9 + current_fps * 0.1)

            result = results[0]
            annotated = result.plot()
            detections = len(result.boxes) if getattr(result, "boxes", None) is not None else 0
            _draw_overlay(
                annotated,
                [
                    f"model={model_path.name}",
                    f"source={args.source} detections={detections} fps={fps_ema:.1f}",
                    "press q or ESC to quit",
                ],
            )

            if writer is None and args.record:
                writer = _create_writer(args.record, max(args.fps, 1.0), annotated.shape)
            if writer is not None:
                writer.write(annotated)

            if not args.no_display:
                cv2.imshow(args.window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            frame_count += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    LOGGER.info("preview stopped frames=%s", frame_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
