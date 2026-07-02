"""Trigger frame trace logging and optional sample export."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from model_service.core.config import config
from PIL import Image

from .camera_roles import camera_roles_payload, normalize_camera_layout

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
        self.camera_layout = normalize_camera_layout(config.vision.camera_layout)
        self.camera_roles = camera_roles_payload(self.camera_layout)
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
        self.loadcell: dict[str, Any] = {}
        self.video_stats: dict[str, Any] = {}
        self.candidates: list[dict[str, Any]] = []
        self.raw_vision_candidates: list[dict[str, Any]] = []
        self.preprocess: dict[str, Any] = {}
        self.stage_counts_by_class: dict[str, dict[str, Any]] = {}
        self.extractor_diagnostics: dict[str, Any] = {}
        self.diagnostic_detections: list[dict[str, Any]] = []
        self.threshold_rescue_candidates: list[dict[str, Any]] = []
        self.roi_rescue_candidates: list[dict[str, Any]] = []
        self.rescue_diagnostics: dict[str, Any] = {}
        self.active_product_snapshot: list[dict[str, Any]] = []
        self.active_product_diagnostics: dict[str, Any] = {}
        self.weight_diagnostics: dict[str, Any] = {}
        self.final_result: dict[str, Any] = {}
        self.storage_result: dict[str, Any] = {}
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

    def record_loadcell_delta(self, delta_weight: float, **metadata: Any) -> None:
        self.loadcell = {
            "delta_weight": round(float(delta_weight), 1),
            **metadata,
        }

    def record_video_stats(self, stats: Any) -> None:
        if hasattr(stats, "to_dict"):
            stats_dict = stats.to_dict()
        elif isinstance(stats, dict):
            stats_dict = dict(stats)
        else:
            stats_dict = {
                key: value
                for key, value in vars(stats).items()
                if not key.startswith("_")
            }
        self.video_stats = stats_dict

    def record_candidates(
        self,
        candidates: list[Any],
        product_weights: Optional[dict[int, float]] = None,
    ) -> None:
        candidate_limit = max(1, int(config.vision.top_k))
        self.candidates = [
            self._candidate_to_dict(candidate, rank, product_weights)
            for rank, candidate in enumerate(candidates[:candidate_limit], start=1)
        ]

    def record_raw_vision_candidates(
        self,
        candidates: list[Any],
        product_weights: Optional[dict[int, float]] = None,
    ) -> None:
        candidate_limit = max(1, int(config.vision.top_k))
        self.raw_vision_candidates = [
            self._candidate_to_dict(candidate, rank, product_weights)
            for rank, candidate in enumerate(candidates[:candidate_limit], start=1)
        ]

    def record_preprocess(self, camera: str, preprocess: dict[str, Any]) -> None:
        self.preprocess[camera] = dict(preprocess)

    def record_stage_count(
        self,
        *,
        class_id: int,
        class_name: str,
        stage: str,
        camera: str,
        amount: int = 1,
        confidence: Optional[float] = None,
        center: Optional[tuple[float, float]] = None,
        roi_x_limit: Optional[float] = None,
        roi_y_limit: Optional[float] = None,
        roi_direction: Optional[str] = None,
    ) -> None:
        key = str(class_id)
        entry = self.stage_counts_by_class.setdefault(
            key,
            {
                "class_id": int(class_id),
                "name": class_name,
                "cameras": {},
            },
        )
        if class_name and not entry.get("name"):
            entry["name"] = class_name
        entry[stage] = int(entry.get(stage, 0)) + int(amount)
        if stage == "freezer_roi_passed":
            entry["freezerExitPathVotes"] = int(entry.get("freezerExitPathVotes", 0)) + int(amount)
        if stage == "freezer_roi_filtered":
            entry["freezerRoiFilteredVotes"] = (
                int(entry.get("freezerRoiFilteredVotes", 0)) + int(amount)
            )
        camera_counts = entry["cameras"].setdefault(camera, {})
        camera_counts[stage] = int(camera_counts.get(stage, 0)) + int(amount)
        if stage == "freezer_roi_passed":
            camera_counts["freezerExitPathVotes"] = (
                int(camera_counts.get("freezerExitPathVotes", 0)) + int(amount)
            )
        if stage == "freezer_roi_filtered":
            camera_counts["freezerRoiFilteredVotes"] = (
                int(camera_counts.get("freezerRoiFilteredVotes", 0)) + int(amount)
            )
        if confidence is not None:
            confidence_key = f"{stage}_max_confidence"
            rounded_confidence = round(float(confidence), 4)
            entry[confidence_key] = max(
                float(entry.get(confidence_key, 0.0)),
                rounded_confidence,
            )
            camera_counts[confidence_key] = max(
                float(camera_counts.get(confidence_key, 0.0)),
                rounded_confidence,
            )
        if center is not None:
            center_x = round(float(center[0]), 1)
            center_y = round(float(center[1]), 1)
            entry["roi_x_min"] = min(float(entry.get("roi_x_min", center_x)), center_x)
            entry["roi_x_max"] = max(float(entry.get("roi_x_max", center_x)), center_x)
            entry["roi_x_sum"] = round(float(entry.get("roi_x_sum", 0.0)) + center_x, 1)
            entry["roi_x_count"] = int(entry.get("roi_x_count", 0)) + 1
            entry["roi_x_avg"] = round(entry["roi_x_sum"] / entry["roi_x_count"], 1)
            if roi_x_limit is not None:
                entry["roi_x_limit"] = round(float(roi_x_limit), 1)
            entry["roi_y_min"] = min(float(entry.get("roi_y_min", center_y)), center_y)
            entry["roi_y_max"] = max(float(entry.get("roi_y_max", center_y)), center_y)
            entry["roi_y_sum"] = round(float(entry.get("roi_y_sum", 0.0)) + center_y, 1)
            entry["roi_y_count"] = int(entry.get("roi_y_count", 0)) + 1
            entry["roi_y_avg"] = round(entry["roi_y_sum"] / entry["roi_y_count"], 1)
            if roi_y_limit is not None:
                entry["roi_y_limit"] = round(float(roi_y_limit), 1)
            if roi_direction is not None:
                entry["roi_direction"] = roi_direction
            camera_counts["roi_x_min"] = min(
                float(camera_counts.get("roi_x_min", center_x)),
                center_x,
            )
            camera_counts["roi_x_max"] = max(
                float(camera_counts.get("roi_x_max", center_x)),
                center_x,
            )
            camera_counts["roi_x_sum"] = round(
                float(camera_counts.get("roi_x_sum", 0.0)) + center_x,
                1,
            )
            camera_counts["roi_x_count"] = int(camera_counts.get("roi_x_count", 0)) + 1
            camera_counts["roi_x_avg"] = round(
                camera_counts["roi_x_sum"] / camera_counts["roi_x_count"],
                1,
            )
            if roi_x_limit is not None:
                camera_counts["roi_x_limit"] = round(float(roi_x_limit), 1)
            camera_counts["roi_y_min"] = min(
                float(camera_counts.get("roi_y_min", center_y)),
                center_y,
            )
            camera_counts["roi_y_max"] = max(
                float(camera_counts.get("roi_y_max", center_y)),
                center_y,
            )
            camera_counts["roi_y_sum"] = round(
                float(camera_counts.get("roi_y_sum", 0.0)) + center_y,
                1,
            )
            camera_counts["roi_y_count"] = int(camera_counts.get("roi_y_count", 0)) + 1
            camera_counts["roi_y_avg"] = round(
                camera_counts["roi_y_sum"] / camera_counts["roi_y_count"],
                1,
            )
            if roi_y_limit is not None:
                camera_counts["roi_y_limit"] = round(float(roi_y_limit), 1)
            if roi_direction is not None:
                camera_counts["roi_direction"] = roi_direction
            sample_key = (
                "roi_filtered_samples"
                if stage in {"roi_filtered", "soft_margin_filtered"}
                else f"{stage}_samples"
            )
            samples = entry.setdefault(sample_key, [])
            if len(samples) < 5:
                samples.append({"camera": camera, "center_x": center_x, "center_y": center_y})

    def record_final_candidate_rank(self, class_id: int, rank: int) -> None:
        key = str(class_id)
        entry = self.stage_counts_by_class.setdefault(
            key,
            {
                "class_id": int(class_id),
                "name": "",
                "cameras": {},
            },
        )
        entry["final_rank"] = int(rank)

    def record_motion_evidence(
        self,
        *,
        class_id: int,
        class_name: str,
        camera: str,
        path_displacement_px: float,
        max_distance_px: float,
        center_span_x: float,
        center_span_y: float,
        motion_threshold_px: float,
        trajectory_exit_path_passed: bool,
        static_shelf_likely: bool,
    ) -> None:
        key = str(class_id)
        entry = self.stage_counts_by_class.setdefault(
            key,
            {
                "class_id": int(class_id),
                "name": class_name,
                "cameras": {},
            },
        )
        if class_name and not entry.get("name"):
            entry["name"] = class_name
        camera_counts = entry["cameras"].setdefault(camera, {})

        motion_payload = {
            "pathDisplacementPx": round(float(path_displacement_px), 1),
            "maxDistancePx": round(float(max_distance_px), 1),
            "centerSpanX": round(float(center_span_x), 1),
            "centerSpanY": round(float(center_span_y), 1),
            "motionThresholdPx": round(float(motion_threshold_px), 1),
            "trajectoryExitPathPassed": bool(trajectory_exit_path_passed),
            "staticShelfLikely": bool(static_shelf_likely),
        }
        camera_counts.update(motion_payload)

        for key_name in (
            "pathDisplacementPx",
            "maxDistancePx",
            "centerSpanX",
            "centerSpanY",
            "motionThresholdPx",
        ):
            entry[key_name] = max(
                float(entry.get(key_name, 0.0) or 0.0),
                float(motion_payload[key_name]),
            )

        entry["trajectoryExitPathPassed"] = bool(
            entry.get("trajectoryExitPathPassed")
        ) or bool(trajectory_exit_path_passed)
        if entry["trajectoryExitPathPassed"]:
            entry["staticShelfLikely"] = False
        else:
            entry["staticShelfLikely"] = bool(entry.get("staticShelfLikely")) or bool(
                static_shelf_likely
            )

    def record_hand_path_evidence(
        self,
        *,
        class_id: int,
        class_name: str,
        camera: str,
        hand_path_valid: bool,
        hand_path_passed: bool,
        hand_path_blocked: bool,
        hand_interaction_passed: bool = False,
        hand_near_frame_count: int = 0,
        hand_near_vote_ratio: float = 0.0,
        min_hand_distance_px: Optional[float] = None,
        hand_path_valid_upper_roi: Optional[bool] = None,
        hand_track_id: Optional[int] = None,
        hand_track_count: int = 0,
        valid_hand_track_count: int = 0,
        hand_track_near_frame_count: int = 0,
    ) -> None:
        key = str(class_id)
        entry = self.stage_counts_by_class.setdefault(
            key,
            {
                "class_id": int(class_id),
                "name": class_name,
                "cameras": {},
            },
        )
        if class_name and not entry.get("name"):
            entry["name"] = class_name
        camera_counts = entry["cameras"].setdefault(camera, {})
        payload = {
            "handPathValid": bool(hand_path_valid),
            "handPathPassed": bool(hand_path_passed),
            "handPathBlocked": bool(hand_path_blocked),
            "handInteractionPassed": bool(hand_interaction_passed),
            "handNearFrameCount": int(hand_near_frame_count),
            "handNearVoteRatio": round(float(hand_near_vote_ratio), 4),
            "minHandDistancePx": (
                round(float(min_hand_distance_px), 1)
                if min_hand_distance_px is not None
                else None
            ),
            "handPathValidUpperRoi": (
                bool(hand_path_valid_upper_roi)
                if hand_path_valid_upper_roi is not None
                else bool(hand_path_valid)
            ),
            "handTrackId": int(hand_track_id) if hand_track_id is not None else None,
            "handTrackCount": int(hand_track_count),
            "validHandTrackCount": int(valid_hand_track_count),
            "handTrackNearFrameCount": int(hand_track_near_frame_count),
        }
        camera_counts.update(payload)
        entry["handPathValid"] = bool(entry.get("handPathValid")) or bool(
            hand_path_valid
        )
        entry["handPathPassed"] = bool(entry.get("handPathPassed")) or bool(
            hand_path_passed
        )
        entry["handInteractionPassed"] = bool(
            entry.get("handInteractionPassed")
        ) or bool(hand_interaction_passed)
        entry["handPathValidUpperRoi"] = bool(
            entry.get("handPathValidUpperRoi")
        ) or bool(payload["handPathValidUpperRoi"])
        entry["handNearFrameCount"] = max(
            int(entry.get("handNearFrameCount", 0) or 0),
            int(hand_near_frame_count),
        )
        entry["handNearVoteRatio"] = max(
            float(entry.get("handNearVoteRatio", 0.0) or 0.0),
            float(hand_near_vote_ratio),
        )
        entry["handTrackCount"] = max(
            int(entry.get("handTrackCount", 0) or 0),
            int(hand_track_count),
        )
        entry["validHandTrackCount"] = max(
            int(entry.get("validHandTrackCount", 0) or 0),
            int(valid_hand_track_count),
        )
        entry["handTrackNearFrameCount"] = max(
            int(entry.get("handTrackNearFrameCount", 0) or 0),
            int(hand_track_near_frame_count),
        )
        if hand_track_id is not None:
            entry["handTrackId"] = int(hand_track_id)
        if min_hand_distance_px is not None:
            current_distance = entry.get("minHandDistancePx")
            entry["minHandDistancePx"] = (
                round(float(min_hand_distance_px), 1)
                if current_distance is None
                else min(float(current_distance), round(float(min_hand_distance_px), 1))
            )
        if entry["handPathPassed"]:
            entry["handPathBlocked"] = False
        else:
            entry["handPathBlocked"] = bool(entry.get("handPathBlocked")) or bool(
                hand_path_blocked
            )

    def record_extractor_diagnostics(self, camera: str, diagnostics: Any) -> None:
        if diagnostics is None:
            return
        if hasattr(diagnostics, "__dataclass_fields__"):
            payload = {
                key: getattr(diagnostics, key)
                for key in diagnostics.__dataclass_fields__
            }
        elif isinstance(diagnostics, dict):
            payload = dict(diagnostics)
        else:
            payload = {
                key: value
                for key, value in vars(diagnostics).items()
                if not key.startswith("_")
            }
        self.extractor_diagnostics[camera] = payload

    def record_diagnostic_detection(
        self,
        *,
        camera: str,
        frame_index: int,
        class_id: int,
        class_name: str,
        confidence: float,
    ) -> None:
        self.diagnostic_detections.append(
            {
                "camera": camera,
                "frame_index": int(frame_index),
                "class_id": int(class_id),
                "name": class_name,
                "confidence": round(float(confidence), 4),
            }
        )

    def record_threshold_rescue_candidates(
        self,
        candidates: list[Any],
        product_weights: Optional[dict[int, float]] = None,
    ) -> None:
        limit = max(0, int(config.vision.threshold_rescue_max_candidates))
        self.threshold_rescue_candidates = [
            self._candidate_to_dict(candidate, rank, product_weights)
            for rank, candidate in enumerate(candidates[:limit], start=1)
        ]
        for candidate in self.threshold_rescue_candidates:
            class_id = candidate.get("class_id")
            if class_id is None:
                continue
            entry = self.stage_counts_by_class.setdefault(
                str(class_id),
                {
                    "class_id": int(class_id),
                    "name": candidate.get("name", ""),
                    "cameras": {},
                },
            )
            entry["threshold_rescue_candidate"] = True
            entry["threshold_rescue_source"] = candidate.get("source")
            entry["motion_gate_passed"] = candidate.get("motion_gate_passed")
            if candidate.get("motion_gate_reason"):
                entry["motion_gate_reason"] = candidate.get("motion_gate_reason")
            entry["threshold_rescue_motion"] = {
                "top": {
                    "passed": bool(candidate.get("top_motion_passed")),
                    "total_displacement": candidate.get("top_total_displacement"),
                    "max_distance": candidate.get("top_max_distance"),
                },
                "side": {
                    "passed": bool(candidate.get("side_motion_passed")),
                    "total_displacement": candidate.get("side_total_displacement"),
                    "max_distance": candidate.get("side_max_distance"),
                },
            }

    def record_roi_rescue_candidates(
        self,
        candidates: list[Any],
        product_weights: Optional[dict[int, float]] = None,
    ) -> None:
        limit = max(0, int(config.vision.threshold_rescue_max_candidates))
        self.roi_rescue_candidates = [
            self._candidate_to_dict(candidate, rank, product_weights)
            for rank, candidate in enumerate(candidates[:limit], start=1)
        ]
        for candidate in self.roi_rescue_candidates:
            class_id = candidate.get("class_id")
            if class_id is None:
                continue
            entry = self.stage_counts_by_class.setdefault(
                str(class_id),
                {
                    "class_id": int(class_id),
                    "name": candidate.get("name", ""),
                    "cameras": {},
                },
            )
            entry["roi_rescue_candidate"] = True
            entry["roi_rescue_source"] = candidate.get("source")
            entry["motion_gate_passed"] = candidate.get("motion_gate_passed")
            for key in ("roi_x_min", "roi_x_max", "roi_x_avg", "roi_x_limit"):
                if candidate.get(key) is not None:
                    entry[key] = candidate.get(key)

    def record_active_product_snapshot(
        self,
        active_products: list[Any],
        *,
        delta_weight: Optional[float] = None,
    ) -> None:
        target_weight = abs(float(delta_weight)) if delta_weight is not None else None
        snapshot: list[dict[str, Any]] = []
        by_class: dict[int, dict[str, Any]] = {}
        for product in active_products or []:
            class_id = getattr(product, "yolo_class_id", None)
            if class_id is None:
                continue
            try:
                class_id_int = int(class_id)
            except (TypeError, ValueError):
                continue
            weight = float(getattr(product, "product_weight", 0.0) or 0.0)
            stock = int(getattr(product, "stock_qty", 0) or 0)
            residual = (
                round(abs(target_weight - weight), 1)
                if target_weight is not None and weight > 0
                else None
            )
            entry = {
                "class_id": class_id_int,
                "name": getattr(product, "product_name", "") or getattr(product, "name", ""),
                "stock_qty": stock,
                "unit_weight_g": round(weight, 1),
                "unit_price": int(getattr(product, "sale_price", 0) or 0),
                "weight_residual_g": residual,
            }
            snapshot.append(entry)
            by_class[class_id_int] = entry

        self.active_product_snapshot = sorted(snapshot, key=lambda item: item["class_id"])
        if target_weight is not None:
            self.loadcell["target_weight_abs"] = round(target_weight, 1)
            self.loadcell["strict_tolerance_g"] = float(config.weight.tolerance_grams)
            self.loadcell["rescue_tolerance_g"] = float(config.weight.rescue_tolerance_grams)

        for class_key, stage_entry in self.stage_counts_by_class.items():
            try:
                class_id_int = int(class_key)
            except (TypeError, ValueError):
                continue
            product_entry = by_class.get(class_id_int)
            if product_entry is None:
                continue
            stage_entry["unit_weight_g"] = product_entry["unit_weight_g"]
            stage_entry["stock_qty"] = product_entry["stock_qty"]
            stage_entry["weight_residual_g"] = product_entry["weight_residual_g"]
            if target_weight is not None:
                residual = product_entry["weight_residual_g"]
                strict_tolerance = float(config.weight.tolerance_grams)
                rescue_tolerance = float(config.weight.rescue_tolerance_grams)
                stage_entry["delta_abs_g"] = round(target_weight, 1)
                stage_entry["strict_tolerance_g"] = strict_tolerance
                stage_entry["rescue_tolerance_g"] = rescue_tolerance
                stage_entry["rescue_weight_residual_g"] = residual
                stage_entry["strict_candidate_weight_mismatch"] = (
                    residual is not None and residual > strict_tolerance
                )
                stage_entry["vision_seen_weight_rejected"] = (
                    residual is not None and residual > rescue_tolerance
                )
                stage_entry["weight_gate_passed"] = (
                    residual is not None
                    and product_entry["stock_qty"] > 0
                    and product_entry["unit_weight_g"] > 0
                    and residual <= rescue_tolerance
                )

    def record_active_product_diagnostics(self, diagnostics: dict[str, Any]) -> None:
        self.active_product_diagnostics.update(dict(diagnostics))

    def record_rescue_diagnostics(self, name: str, diagnostics: dict[str, Any]) -> None:
        self.rescue_diagnostics[name] = dict(diagnostics)
        for candidate in diagnostics.get("candidates", []):
            class_id = candidate.get("class_id")
            if class_id is None:
                continue
            entry = self.stage_counts_by_class.setdefault(
                str(class_id),
                {
                    "class_id": int(class_id),
                    "name": candidate.get("name", ""),
                    "cameras": {},
                },
            )
            reason = candidate.get("reason")
            if reason:
                entry.setdefault("rescue_rejections", {})[name] = reason
            for key in (
                "rescue_weight_residual_g",
                "rescue_tolerance_g",
                "weight_gate_passed",
                "motion_gate_passed",
            ):
                if candidate.get(key) is not None:
                    entry[key] = candidate.get(key)

    def record_weight_diagnostics(self, diagnostics: dict[str, Any]) -> None:
        self.weight_diagnostics = dict(diagnostics)

    def record_detected_single_fallback(self, diagnostics: dict[str, Any]) -> None:
        self.weight_diagnostics.setdefault(
            "detected_single_item_fallback",
            dict(diagnostics),
        )
        if diagnostics.get("accepted"):
            self.weight_diagnostics["fallback_reason"] = "detected_single_item_fallback"

    def record_final_result(
        self,
        *,
        products: list[Any],
        total_price: int,
        status: str,
        confidence: float,
    ) -> None:
        self.final_result = {
            "status": status,
            "confidence": round(float(confidence), 4),
            "total_price": int(total_price),
            "products": [self._product_to_dict(product) for product in products],
        }

    def record_storage_result(
        self,
        *,
        products: list[Any],
        total_price: int,
    ) -> None:
        self.storage_result = {
            "total_price": int(total_price),
            "products": [self._product_to_dict(product) for product in products],
        }

    def finalize(self, status: str, error: Optional[str] = None) -> dict:
        if self._finalized:
            return {}
        self._finalized = True
        now = datetime.now()

        entry = {
            "timestamp": now.isoformat(),
            "processing_mode": "avi_to_frames",
            "inference_unit": "image_frame",
            "session_id": self.session_id,
            "zone": self.zone,
            "status": status,
            "video_paths": dict(self.video_paths),
            "camera_layout": self.camera_layout,
            "camera_roles": dict(self.camera_roles),
            "cameras": {
                camera: summary.to_dict()
                for camera, summary in self.cameras.items()
            },
        }
        if error:
            entry["error"] = error
        if self.loadcell:
            entry["loadcell"] = dict(self.loadcell)
        if self.weight_diagnostics:
            entry["weight_diagnostics"] = dict(self.weight_diagnostics)
        if self.raw_vision_candidates:
            entry["raw_vision_candidates"] = list(self.raw_vision_candidates)
        if self.candidates:
            entry["candidates"] = list(self.candidates)
        if self.active_product_diagnostics:
            entry["active_product_diagnostics"] = dict(self.active_product_diagnostics)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        detail_log_path = self._write_detail_log(now, status, error)
        entry["detail_log_path"] = str(detail_log_path)

        log_file = self.log_dir / f"frame_split_{now.strftime('%Y%m%d')}.jsonl"
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

    def _write_detail_log(
        self,
        timestamp: datetime,
        status: str,
        error: Optional[str],
    ) -> Path:
        day_dir = self.log_dir / "triggers" / timestamp.strftime("%Y%m%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        safe_session_id = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_"
            for char in self.session_id
        )
        output_path = (
            day_dir
            / f"{timestamp.strftime('%H%M%S')}_zone{self.zone}_{safe_session_id}.json"
        )
        payload = {
            "timestamp": timestamp.isoformat(),
            "session_id": self.session_id,
            "zone": self.zone,
            "status": status,
            "video_paths": dict(self.video_paths),
            "camera_layout": self.camera_layout,
            "camera_roles": dict(self.camera_roles),
            "loadcell": dict(self.loadcell),
            "cameras": {
                camera: summary.to_dict()
                for camera, summary in self.cameras.items()
            },
            "video_stats": dict(self.video_stats),
            "vision_config": self._vision_config_snapshot(),
            "raw_vision_candidates": list(self.raw_vision_candidates),
            "candidates": list(self.candidates),
            "preprocess": dict(self.preprocess),
            "stage_counts_by_class": dict(self.stage_counts_by_class),
            "extractor_diagnostics": dict(self.extractor_diagnostics),
            "diagnostic_candidates": self._diagnostic_candidates(),
            "threshold_rescue_candidates": list(self.threshold_rescue_candidates),
            "roi_rescue_candidates": list(self.roi_rescue_candidates),
            "rescue_diagnostics": dict(self.rescue_diagnostics),
            "active_product_snapshot": list(self.active_product_snapshot),
            "active_product_diagnostics": dict(self.active_product_diagnostics),
            "weight_diagnostics": dict(self.weight_diagnostics),
            "final_result": dict(self.final_result),
            "storage_result": dict(self.storage_result),
        }
        if error:
            payload["error"] = error

        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path

    def _diagnostic_candidates(self) -> list[dict[str, Any]]:
        by_class: dict[int, dict[str, Any]] = {}
        for detection in self.diagnostic_detections:
            class_id = int(detection["class_id"])
            entry = by_class.setdefault(
                class_id,
                {
                    "class_id": class_id,
                    "name": detection["name"],
                    "confidence": 0.0,
                    "votes": 0,
                    "cameras": set(),
                    "sample_frames": [],
                },
            )
            entry["confidence"] = max(
                float(entry["confidence"]),
                float(detection["confidence"]),
            )
            entry["votes"] += 1
            entry["cameras"].add(detection["camera"])
            if len(entry["sample_frames"]) < 5:
                entry["sample_frames"].append(
                    {
                        "camera": detection["camera"],
                        "frame_index": detection["frame_index"],
                        "confidence": detection["confidence"],
                    }
                )

        candidates = sorted(
            by_class.values(),
            key=lambda item: (item["votes"], item["confidence"]),
            reverse=True,
        )
        limit = max(1, int(config.vision.top_k))
        result = []
        for rank, candidate in enumerate(candidates[:limit], start=1):
            result.append(
                {
                    "rank": rank,
                    "class_id": candidate["class_id"],
                    "name": candidate["name"],
                    "confidence": round(float(candidate["confidence"]), 4),
                    "votes": candidate["votes"],
                    "cameras": sorted(candidate["cameras"]),
                    "sample_frames": candidate["sample_frames"],
                }
            )
        return result

    @staticmethod
    def _candidate_to_dict(
        candidate: Any,
        rank: int,
        product_weights: Optional[dict[int, float]] = None,
    ) -> dict[str, Any]:
        if hasattr(candidate, "to_dict"):
            raw = candidate.to_dict()
        elif isinstance(candidate, dict):
            raw = dict(candidate)
        else:
            raw = {
                key: getattr(candidate, key)
                for key in dir(candidate)
                if not key.startswith("_") and not callable(getattr(candidate, key))
            }

        class_id = raw.get("class_id", raw.get("classId"))
        unit_weight = raw.get(
            "unit_weight_g",
            raw.get(
                "unitWeightG",
                raw.get(
                    "unit_weight",
                    raw.get(
                        "unitWeight",
                        raw.get("product_weight", raw.get("productWeight")),
                    ),
                ),
            ),
        )
        if unit_weight is None and product_weights:
            try:
                unit_weight = product_weights.get(int(class_id))
            except (TypeError, ValueError):
                unit_weight = None

        try:
            normalized_unit_weight = (
                round(float(unit_weight), 1) if unit_weight is not None else None
            )
        except (TypeError, ValueError):
            normalized_unit_weight = None

        return {
            "rank": int(raw.get("rank", rank)),
            "class_id": class_id,
            "name": raw.get("name", raw.get("class_name", raw.get("className"))),
            "unit_weight_g": normalized_unit_weight,
            "confidence": raw.get(
                "confidence",
                raw.get("weighted_confidence", raw.get("combined_confidence")),
            ),
            "top": raw.get("top", raw.get("top_detected", raw.get("topDetected"))),
            "side": raw.get("side", raw.get("side_detected", raw.get("sideDetected"))),
            "votes": raw.get("votes", raw.get("vote_count", raw.get("voteCount"))),
            "source": raw.get("source", "vision"),
            "raw_vote_count": raw.get("raw_vote_count", raw.get("rawVoteCount")),
            "top_vote_count": raw.get("top_vote_count", raw.get("topVoteCount")),
            "side_vote_count": raw.get("side_vote_count", raw.get("sideVoteCount")),
            "top_motion_passed": raw.get(
                "top_motion_passed",
                raw.get("topMotionPassed"),
            ),
            "side_motion_passed": raw.get(
                "side_motion_passed",
                raw.get("sideMotionPassed"),
            ),
            "top_total_displacement": raw.get(
                "top_total_displacement",
                raw.get("topTotalDisplacement"),
            ),
            "side_total_displacement": raw.get(
                "side_total_displacement",
                raw.get("sideTotalDisplacement"),
            ),
            "top_max_distance": raw.get("top_max_distance", raw.get("topMaxDistance")),
            "side_max_distance": raw.get("side_max_distance", raw.get("sideMaxDistance")),
            "roi_x_min": raw.get("roi_x_min", raw.get("roiXMin")),
            "roi_x_max": raw.get("roi_x_max", raw.get("roiXMax")),
            "roi_x_avg": raw.get("roi_x_avg", raw.get("roiXAvg")),
            "roi_x_limit": raw.get("roi_x_limit", raw.get("roiXLimit")),
            "weight_residual_g": raw.get("weight_residual_g", raw.get("weightResidualG")),
            "rescue_weight_residual_g": raw.get(
                "rescue_weight_residual_g",
                raw.get("rescueWeightResidualG"),
            ),
            "rescue_tolerance_g": raw.get("rescue_tolerance_g", raw.get("rescueToleranceG")),
            "weight_gate_passed": raw.get("weight_gate_passed", raw.get("weightGatePassed")),
            "motion_gate_passed": raw.get("motion_gate_passed", raw.get("motionGatePassed")),
            "motion_gate_reason": raw.get("motion_gate_reason", raw.get("motionGateReason")),
            "freezerExitPathVotes": raw.get(
                "freezer_exit_path_votes",
                raw.get("freezerExitPathVotes"),
            ),
            "pathDisplacementPx": raw.get(
                "pathDisplacementPx",
                raw.get("path_displacement_px"),
            ),
            "maxDistancePx": raw.get("maxDistancePx", raw.get("max_distance_px")),
            "centerSpanX": raw.get("centerSpanX", raw.get("center_span_x")),
            "centerSpanY": raw.get("centerSpanY", raw.get("center_span_y")),
            "trajectoryExitPathPassed": raw.get(
                "trajectoryExitPathPassed",
                raw.get("trajectory_exit_path_passed"),
            ),
            "staticShelfLikely": raw.get(
                "staticShelfLikely",
                raw.get("static_shelf_likely"),
            ),
            "handPathValid": raw.get("handPathValid", raw.get("hand_path_valid")),
            "handPathPassed": raw.get("handPathPassed", raw.get("hand_path_passed")),
            "handPathBlocked": raw.get(
                "handPathBlocked",
                raw.get("hand_path_blocked"),
            ),
            "handInteractionPassed": raw.get(
                "handInteractionPassed",
                raw.get("hand_interaction_passed"),
            ),
            "handNearFrameCount": raw.get(
                "handNearFrameCount",
                raw.get("hand_near_frame_count"),
            ),
            "handNearVoteRatio": raw.get(
                "handNearVoteRatio",
                raw.get("hand_near_vote_ratio"),
            ),
            "minHandDistancePx": raw.get(
                "minHandDistancePx",
                raw.get("min_hand_distance_px"),
            ),
            "handPathValidUpperRoi": raw.get(
                "handPathValidUpperRoi",
                raw.get("hand_path_valid_upper_roi"),
            ),
            "handTrackId": raw.get("handTrackId", raw.get("hand_track_id")),
            "handTrackCount": raw.get(
                "handTrackCount",
                raw.get("hand_track_count"),
            ),
            "validHandTrackCount": raw.get(
                "validHandTrackCount",
                raw.get("valid_hand_track_count"),
            ),
            "handTrackNearFrameCount": raw.get(
                "handTrackNearFrameCount",
                raw.get("hand_track_near_frame_count"),
            ),
            "instance_count_hint": raw.get(
                "instance_count_hint",
                raw.get("instanceCountHint"),
            ),
            "sample_frames": raw.get("sample_frames", raw.get("sampleFrames")),
            "roi_conflict": raw.get("roi_conflict", raw.get("roiConflict")),
            "roi_conflict_reason": raw.get(
                "roi_conflict_reason",
                raw.get("roiConflictReason"),
            ),
            "threshold_rescue_rejected_reason": raw.get(
                "threshold_rescue_rejected_reason",
                raw.get("thresholdRescueRejectedReason"),
            ),
            "roi_conflict_side_vote_count": raw.get(
                "roi_conflict_side_vote_count",
                raw.get("roiConflictSideVoteCount"),
            ),
            "roi_conflict_side_max_confidence": raw.get(
                "roi_conflict_side_max_confidence",
                raw.get("roiConflictSideMaxConfidence"),
            ),
            "roi_conflict_side_roi_x_avg": raw.get(
                "roi_conflict_side_roi_x_avg",
                raw.get("roiConflictSideRoiXAvg"),
            ),
            "roi_conflict_side_roi_x_limit": raw.get(
                "roi_conflict_side_roi_x_limit",
                raw.get("roiConflictSideRoiXLimit"),
            ),
        }

    @staticmethod
    def _vision_config_snapshot() -> dict[str, Any]:
        return {
            "yolo_model_path": config.vision.yolo_model_path,
            "yolo_internal_conf_threshold": config.vision.yolo_internal_conf_threshold,
            "cabinet_type": config.machine.cabinet_type,
            "camera_layout": config.vision.camera_layout,
            "camera_roles": camera_roles_payload(config.vision.camera_layout),
            "freezer_handled_filter_enabled": (
                str(config.machine.cabinet_type).lower() == "freezer"
                and str(config.vision.camera_layout).lower() == "dual_top_proxy"
            ),
            "top_confidence_threshold": config.vision.top_confidence_threshold,
            "side_confidence_threshold": config.vision.side_confidence_threshold,
            "hand_confidence_threshold": config.vision.hand_confidence_threshold,
            "hand_class_id": config.vision.hand_class_id,
            "top_k": config.vision.top_k,
            "freezer_min_vote_ratio": config.vision.freezer_min_vote_ratio,
            "freezer_min_vote_count": config.vision.freezer_min_vote_count,
            "freezer_motion_min_displacement_px": (
                config.vision.freezer_motion_min_displacement_px
            ),
            "freezer_roi_vertical_region": config.vision.freezer_roi_vertical_region,
            "freezer_roi_y_split": (
                config.vision.freezer_roi_y_split
                if config.vision.freezer_roi_y_split is not None
                else config.vision.freezer_lower_roi_y_split
            ),
            "freezer_lower_roi_y_split_legacy": (
                config.vision.freezer_lower_roi_y_split
            ),
            "freezer_min_exit_path_votes": config.vision.freezer_min_exit_path_votes,
            "freezer_confidence_tie_band": config.weight.freezer_confidence_tie_band,
            "freezer_multi_min_confidence": config.weight.freezer_multi_min_confidence,
            "freezer_vision_multi_without_weight_enabled": (
                config.weight.freezer_vision_multi_without_weight_enabled
            ),
            "regular_threshold": {
                "top": config.vision.top_confidence_threshold,
                "side": config.vision.side_confidence_threshold,
            },
            "top_crop_policy": config.vision.top_crop_policy,
            "side_crop_policy": config.vision.side_crop_policy,
            "top_roi_enabled": config.vision.top_roi_enabled,
            "top_roi_y_split": config.vision.top_roi_y_split,
            "side_roi_x_max": config.vision.side_roi_x_max,
            "side_roi_soft_margin_px": config.vision.side_roi_soft_margin_px,
            "side_roi_soft_x_max": (
                config.vision.side_roi_x_max
                + max(0.0, float(config.vision.side_roi_soft_margin_px))
            ),
            "ffmpeg_top_gamma": config.vision.ffmpeg_top_gamma,
            "ffmpeg_top_contrast": config.vision.ffmpeg_top_contrast,
            "ffmpeg_side_gamma": config.vision.ffmpeg_side_gamma,
            "ffmpeg_side_contrast": config.vision.ffmpeg_side_contrast,
            "async_frame_stride": config.async_streaming.frame_stride,
            "min_vote_ratio": config.vision.min_vote_ratio,
            "min_vote_count": config.vision.min_vote_count,
            "motion_min_displacement_px": config.vision.motion_min_displacement_px,
            "top_weight": config.vision.top_weight,
            "side_weight": config.vision.side_weight,
            "top_only_weight": config.vision.top_only_weight,
            "side_only_weight": config.vision.side_only_weight,
            "common_class_bonus": config.vision.common_class_bonus,
            "diagnostic_all_class_trace": config.vision.diagnostic_all_class_trace,
            "threshold_rescue_enabled": config.vision.threshold_rescue_enabled,
            "threshold_rescue_require_motion": config.vision.threshold_rescue_require_motion,
            "weight_rescue_no_motion_enabled": config.vision.weight_rescue_no_motion_enabled,
            "weight_rescue_no_motion_min_raw_votes": config.vision.weight_rescue_no_motion_min_raw_votes,
            "weight_rescue_no_motion_max_residual_grams": config.vision.weight_rescue_no_motion_max_residual_grams,
            "rescue_tolerance_g": config.weight.rescue_tolerance_grams,
            "detected_single_fallback_enabled": config.weight.detected_single_fallback_enabled,
            "detected_single_fallback_tolerance_g": config.weight.detected_single_fallback_tolerance_grams,
            "detected_single_fallback_min_votes": config.weight.detected_single_fallback_min_votes,
        }

    @staticmethod
    def _product_to_dict(product: Any) -> dict[str, Any]:
        if hasattr(product, "to_dict"):
            raw = product.to_dict()
        elif isinstance(product, dict):
            raw = dict(product)
        else:
            raw = {
                key: value
                for key, value in vars(product).items()
                if not key.startswith("_")
            }
        count = raw.get("count", 0) or 0
        unit_price = raw.get("unit_price", raw.get("unitPrice", raw.get("price", 0))) or 0
        total_price = raw.get("total_price", raw.get("totalPrice"))
        if total_price is None:
            total_price = unit_price * count
        return {
            "product_id": raw.get("product_id", raw.get("productId")),
            "product_idx": raw.get("product_idx", raw.get("productIdx")),
            "name": raw.get("name", raw.get("product_name", raw.get("productName"))),
            "count": count,
            "unit_price": unit_price,
            "total_price": total_price,
            "confidence": raw.get("confidence", 0.0),
        }

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
