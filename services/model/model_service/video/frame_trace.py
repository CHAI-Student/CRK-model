from __future__ import annotations

"""Trigger frame trace logging and optional sample export."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from model_service.core.config import config

logger = logging.getLogger(__name__)


def _model_service_dir() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class CameraTraceSummary:
    video_path: Optional[str]
    total_frames: int = 0
    processed_frames: int = 0
    sample_indices: list[int] = field(default_factory=list)
    sample_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "video_path": self.video_path,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "sample_indices": list(self.sample_indices),
            "sample_files": list(self.sample_files),
        }


class TriggerTraceContext:
    """Collect per-trigger frame split trace and persist it once."""

    def __init__(
        self,
        session_id: str,
        zone: int,
        top_path: Optional[str],
        side_path: Optional[str],
        *,
        log_dir: Optional[Path] = None,
        sample_export_dir: Optional[Path] = None,
        sample_export_enabled: Optional[bool] = None,
        sample_count_per_camera: Optional[int] = None,
    ) -> None:
        self.session_id = session_id
        self.zone = zone
        self.video_paths = {
            "top": top_path or "",
            "side": side_path or "",
        }
        self.log_dir = Path(log_dir) if log_dir is not None else (_model_service_dir() / "logs")
        configured_sample_dir = (
            Path(sample_export_dir)
            if sample_export_dir is not None
            else Path(config.trace.sample_export_dir)
        )
        if not configured_sample_dir.is_absolute():
            configured_sample_dir = _model_service_dir() / configured_sample_dir
        self.sample_export_dir = configured_sample_dir
        self.sample_export_enabled = (
            config.trace.sample_export_enabled
            if sample_export_enabled is None
            else sample_export_enabled
        )
        self.sample_count_per_camera = max(
            0,
            config.trace.sample_count_per_camera
            if sample_count_per_camera is None
            else sample_count_per_camera,
        )
        self.cameras = {
            "top": CameraTraceSummary(video_path=top_path),
            "side": CameraTraceSummary(video_path=side_path),
        }
        self._finalized = False

    @staticmethod
    def build_sample_indices(total_frames: int, sample_count: int) -> list[int]:
        if total_frames <= 0 or sample_count <= 0:
            return []
        if total_frames <= sample_count:
            return list(range(total_frames))
        if sample_count == 1:
            return [0]

        positions = []
        last_index = total_frames - 1
        for slot in range(sample_count):
            candidate = round(slot * last_index / (sample_count - 1))
            if not positions or candidate != positions[-1]:
                positions.append(candidate)
        return positions

    def plan_camera(self, camera: str, total_frames: int) -> list[int]:
        summary = self.cameras[camera]
        summary.total_frames = max(0, total_frames)
        if self.sample_export_enabled:
            summary.sample_indices = self.build_sample_indices(
                summary.total_frames,
                self.sample_count_per_camera,
            )
        else:
            summary.sample_indices = []
        return list(summary.sample_indices)

    def record_frame(self, camera: str, frame_index: int, frame: Optional[Any] = None) -> None:
        summary = self.cameras[camera]
        summary.processed_frames += 1
        if (
            not self.sample_export_enabled
            or frame is None
            or frame_index not in summary.sample_indices
            or any(Path(path).stem == f"frame_{frame_index}" for path in summary.sample_files)
        ):
            return

        try:
            exported = self._export_frame(camera, frame_index, frame)
        except Exception as exc:
            logger.warning(
                "[FRAME-TRACE] sample export failed: session_id=%s camera=%s frame=%s error=%s",
                self.session_id,
                camera,
                frame_index,
                exc,
            )
            return

        summary.sample_files.append(str(exported))

    def finalize(self, status: str, error: Optional[str] = None) -> dict:
        if self._finalized:
            return {}
        self._finalized = True

        entry = {
            "timestamp": datetime.now().isoformat(),
            "processing_mode": "avi_to_frames",
            "inference_unit": "image_frame",
            "session_id": self.session_id,
            "zone": self.zone,
            "status": status,
            "video_paths": dict(self.video_paths),
            "cameras": {
                camera: summary.to_dict()
                for camera, summary in self.cameras.items()
            },
        }
        if error:
            entry["error"] = error

        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.log_dir / f"frame_split_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info(
            "[FRAME-TRACE] session_id=%s zone=%s status=%s mode=avi_to_frames unit=image_frame "
            "top=%s/%s side=%s/%s samples_top=%s samples_side=%s",
            self.session_id,
            self.zone,
            status,
            self.cameras["top"].processed_frames,
            self.cameras["top"].total_frames,
            self.cameras["side"].processed_frames,
            self.cameras["side"].total_frames,
            len(self.cameras["top"].sample_files),
            len(self.cameras["side"].sample_files),
        )
        if error:
            logger.warning(
                "[FRAME-TRACE] session_id=%s error=%s",
                self.session_id,
                error,
            )

        return entry

    def _export_frame(self, camera: str, frame_index: int, frame: Any) -> Path:
        array = np.asarray(frame, dtype=np.uint8)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError(f"unsupported frame shape: {array.shape}")

        if array.shape[2] == 4:
            rgb = array[:, :, [2, 1, 0, 3]]
            image = Image.fromarray(rgb, mode="RGBA")
        else:
            rgb = array[:, :, ::-1]
            image = Image.fromarray(rgb, mode="RGB")

        camera_dir = self.sample_export_dir / self.session_id / camera
        camera_dir.mkdir(parents=True, exist_ok=True)
        output_path = camera_dir / f"frame_{frame_index}.jpg"
        image.save(output_path, format="JPEG")
        return output_path
