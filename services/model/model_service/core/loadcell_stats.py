"""Stdlib-only helpers for loadcell parsing and weight delta calculation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import fsum
from statistics import fmean, median, pstdev
from typing import Protocol, Sequence

from model_service.core.config import config


class SupportsFilteredValue(Protocol):
    """Duck-typed protocol for trigger loadcell data."""

    filtered_value: Sequence[object] | None
    timestamp: str


def _read_field(loadcell: object, name: str) -> object:
    if isinstance(loadcell, dict):
        return loadcell.get(name)
    return getattr(loadcell, name, None)


@dataclass
class LoadcellStablePlateau:
    """Collapsed stable loadcell region used for movement history."""

    start_index: int
    end_index: int
    start_timestamp: str | None
    end_timestamp: str | None
    avg: float
    samples: int

    def to_dict(self) -> dict[str, object]:
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "avg": round(float(self.avg), 1),
            "samples": int(self.samples),
        }


@dataclass
class LoadcellDeltaSegment:
    """Stable plateau-to-plateau movement inside one trigger payload."""

    start_index: int
    end_index: int
    start_timestamp: str | None
    end_timestamp: str | None
    start_avg: float
    end_avg: float
    delta: float
    sign: int
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "start_avg": round(float(self.start_avg), 1),
            "end_avg": round(float(self.end_avg), 1),
            "delta": round(float(self.delta), 1),
            "abs_delta": round(abs(float(self.delta)), 1),
            "sign": int(self.sign),
            "duration_seconds": round(float(self.duration_seconds), 3),
        }


@dataclass
class LoadcellDeltaAnalysis:
    """Detailed trigger-level loadcell delta analysis."""

    delta: float = 0.0
    sample_count: int = 0
    parsed_sample_count: int = 0
    working_sample_count: int = 0
    sample_span_seconds: float = 0.0
    window_size: int = 0
    stability_threshold: float = 0.0
    start_avg: float = 0.0
    end_avg: float = 0.0
    start_stable_idx: int | None = None
    end_stable_idx: int | None = None
    stable_region_valid: bool = False
    used_simple_fallback: bool = False
    reason: str = ""
    segments: list[LoadcellDeltaSegment] = field(default_factory=list)
    stable_plateaus: list[LoadcellStablePlateau] = field(default_factory=list)
    purchase_delta_candidates: list[dict[str, object]] = field(default_factory=list)
    removal_segment_targets: list[dict[str, object]] = field(default_factory=list)
    channel_removal_segment_targets: list[dict[str, object]] = field(default_factory=list)
    channel_delta_diagnostics: dict[str, object] = field(default_factory=dict)
    return_segment_targets: list[dict[str, object]] = field(default_factory=list)
    vision_required_segment_targets: list[dict[str, object]] = field(default_factory=list)
    paired_loadcell_movements: list[dict[str, object]] = field(default_factory=list)
    ignored_loadcell_movements: list[dict[str, object]] = field(default_factory=list)
    mixed_sign_net_masking_guard: dict[str, object] = field(default_factory=dict)
    pressure_like_event: bool = False
    decision_delta: float = 0.0
    decision_delta_reliable: bool = False
    stable_delta_source: str = ""
    baseline_stable_avg: float = 0.0
    final_stable_avg: float = 0.0
    trailing_unstable_sample_count: int = 0
    raw_simple_delta: float = 0.0
    raw_extreme_delta: float = 0.0
    endpoint_delta_weight: float = 0.0
    endpoint_fallback_applied: bool = False
    endpoint_fallback_reason: str = "not_evaluated"


def mixed_return_hints_from_analysis(
    delta_analysis: LoadcellDeltaAnalysis | None,
    *,
    decision_delta: float,
) -> list[dict[str, object]]:
    """Build internal return hints for negative triggers with positive segments."""
    if delta_analysis is None or decision_delta >= 0:
        return []

    return_targets = list(getattr(delta_analysis, "return_segment_targets", []) or [])
    if not return_targets:
        return []

    removal_indices: list[int] = []
    for target in getattr(delta_analysis, "removal_segment_targets", []) or []:
        try:
            removal_indices.append(int(target.get("segment_index")))  # type: ignore[union-attr]
        except (AttributeError, TypeError, ValueError):
            continue

    first_removal_index = min(removal_indices) if removal_indices else None
    hints: list[dict[str, object]] = []
    for target in return_targets:
        if not isinstance(target, dict):
            continue
        try:
            weight = abs(float(target.get("weight", target.get("delta", 0.0))))
            segment_index = int(target.get("segment_index", -1))
        except (TypeError, ValueError):
            continue
        if weight <= 0:
            continue
        replay_position = (
            "before_removal"
            if first_removal_index is None or segment_index < first_removal_index
            else "after_removal"
        )
        hints.append(
            {
                "source": target.get("source", "unpaired_positive_segment"),
                "weight": round(weight, 1),
                "delta": round(weight, 1),
                "segment_index": segment_index,
                "segment_indices": list(target.get("segment_indices", [segment_index])),
                "start_timestamp": target.get("start_timestamp"),
                "end_timestamp": target.get("end_timestamp"),
                "duration_seconds": target.get("duration_seconds"),
                "reason": target.get("reason", "unpaired_return_segment"),
                "replay_position": replay_position,
            }
        )
    return hints


def mixed_return_segment_diagnostics(
    delta_analysis: LoadcellDeltaAnalysis | None,
    *,
    return_weight_hints: list[dict[str, object]],
    decision_delta: float,
) -> dict[str, object]:
    """Return trace diagnostics for mixed return/removal segment replay."""
    return_targets = list(getattr(delta_analysis, "return_segment_targets", []) or [])
    mixed_sign_guard = dict(
        getattr(delta_analysis, "mixed_sign_net_masking_guard", {}) or {}
    )
    return {
        "accepted": bool(return_weight_hints),
        "decision_delta": round(float(decision_delta), 1),
        "return_segment_targets": return_targets,
        "return_weight_hints": list(return_weight_hints),
        "mixed_sign_net_masking_guard": mixed_sign_guard,
        "reason": (
            "return_hints_attached_to_negative_trigger"
            if return_weight_hints
            else "no_unpaired_positive_segments_for_negative_trigger"
        ),
    }


def effective_count_guard_diagnostics(
    products: Sequence[object],
    *,
    delta_weight: float,
    return_weight_hints: Sequence[object],
) -> dict[str, object]:
    """Describe count reductions expected after mixed-return effective delta repair."""
    hint_total = 0.0
    for hint in return_weight_hints or []:
        try:
            if isinstance(hint, dict):
                raw_value = hint.get("delta", hint.get("weight", 0.0))
            else:
                raw_value = hint
            hint_total += abs(float(raw_value))
        except (TypeError, ValueError):
            continue

    effective_delta = float(delta_weight) + hint_total
    expected_weight = 0.0
    corrections: list[dict[str, object]] = []
    tolerance = float(config.weight.tolerance_grams)
    tolerance_per_item = float(config.weight.same_product_count_tolerance_grams)

    for product in products or []:
        try:
            count = int(getattr(product, "count", 0) or 0)
            unit_weight = float(getattr(product, "unit_weight", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if count <= 0 or unit_weight <= 0:
            continue
        expected_weight += unit_weight * count
        if count < 2:
            continue

        effective_abs = abs(effective_delta)
        corrected_count = min(count, max(0, int(round(effective_abs / unit_weight))))
        if corrected_count >= count:
            continue
        corrected_expected = unit_weight * corrected_count
        residual = abs(effective_abs - corrected_expected)
        allowed_residual = tolerance + tolerance_per_item * max(corrected_count, 1)
        corrections.append(
            {
                "class_id": int(getattr(product, "product_id", -1) or -1),
                "name": str(getattr(product, "name", "")),
                "original_count": count,
                "corrected_count": corrected_count,
                "unit_weight": round(unit_weight, 1),
                "residual": round(residual, 1),
                "allowed_residual": round(allowed_residual, 1),
                "accepted": residual <= allowed_residual,
            }
        )

    accepted = any(bool(entry["accepted"]) for entry in corrections)
    if not return_weight_hints:
        reason = "no_return_weight_hints"
    elif accepted:
        reason = "effective_delta_supports_lower_count"
    else:
        reason = "no_count_reduction_supported"

    return {
        "accepted": accepted,
        "raw_delta": round(float(delta_weight), 1),
        "return_hint_total": round(hint_total, 1),
        "effective_delta": round(effective_delta, 1),
        "expected_weight": round(expected_weight, 1),
        "corrections": corrections,
        "reason": reason,
    }


def parse_loadcell_value(value: object) -> float:
    """Parse a loadcell value string such as '+12345'."""
    try:
        cleaned = str(value).strip()
        if cleaned.startswith("+"):
            cleaned = cleaned[1:]
        return float(cleaned)
    except (ValueError, TypeError):
        return 0.0


def _parse_loadcell_value_optional(value: object) -> float | None:
    try:
        cleaned = str(value).strip()
        if cleaned.startswith("+"):
            cleaned = cleaned[1:]
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _values_as_strings(values: object) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    try:
        return [str(value) for value in values]
    except TypeError:
        return [str(values)]


def _summed_values(values: object) -> float | None:
    parsed = [
        parsed
        for value in _values_as_strings(values)
        if (parsed := _parse_loadcell_value_optional(value)) is not None
    ]
    return float(fsum(parsed)) if parsed else None


def _payload_state(
    *,
    sample_count: int,
    parsed_channel_count: int,
    nonzero_channel_count: int,
) -> str:
    if sample_count == 0:
        return "empty_payload"
    if parsed_channel_count == 0:
        return "invalid_only"
    if nonzero_channel_count == 0:
        return "all_zero"
    return "nonzero"


def _summarize_channel_field(
    loadcells: Sequence[SupportsFilteredValue],
    field_name: str,
) -> dict[str, object]:
    prefix = "raw" if field_name == "raw_value" else "filtered"
    channel_count = 0
    parsed_channel_count = 0
    invalid_channel_count = 0
    zero_channel_count = 0
    nonzero_channel_count = 0

    for loadcell in loadcells:
        for value in _values_as_strings(_read_field(loadcell, field_name)):
            channel_count += 1
            parsed = _parse_loadcell_value_optional(value)
            if parsed is None:
                invalid_channel_count += 1
                continue
            parsed_channel_count += 1
            if parsed == 0.0:
                zero_channel_count += 1
            else:
                nonzero_channel_count += 1

    first_values = (
        _values_as_strings(_read_field(loadcells[0], field_name))
        if loadcells
        else []
    )
    last_values = (
        _values_as_strings(_read_field(loadcells[-1], field_name))
        if loadcells
        else []
    )
    first_total = _summed_values(first_values)
    last_total = _summed_values(last_values)

    return {
        f"{prefix}_state": _payload_state(
            sample_count=len(loadcells),
            parsed_channel_count=parsed_channel_count,
            nonzero_channel_count=nonzero_channel_count,
        ),
        f"{prefix}_channel_count": channel_count,
        f"{prefix}_parsed_channel_count": parsed_channel_count,
        f"{prefix}_invalid_channel_count": invalid_channel_count,
        f"{prefix}_zero_channel_count": zero_channel_count,
        f"{prefix}_nonzero_channel_count": nonzero_channel_count,
        f"first_{prefix}_values": first_values,
        f"last_{prefix}_values": last_values,
        f"first_{prefix}_total": (
            round(first_total, 1) if first_total is not None else None
        ),
        f"last_{prefix}_total": (
            round(last_total, 1) if last_total is not None else None
        ),
    }


def summarize_loadcell_payload(
    loadcells: Sequence[SupportsFilteredValue],
) -> dict[str, object]:
    """Summarize loadcell payload shape without changing delta behavior."""
    diagnostics: dict[str, object] = {
        "sample_count": len(loadcells),
    }
    diagnostics.update(_summarize_channel_field(loadcells, "raw_value"))
    diagnostics.update(_summarize_channel_field(loadcells, "filtered_value"))
    diagnostics["payload_state"] = diagnostics["filtered_state"]
    return diagnostics


def avg_loadcell_channels(values: Sequence[object]) -> float:
    """Sum parsed loadcell channels for one zone, ignoring invalid values.

    Camera sends two physical loadcell channels for a vending zone. The service
    must use their combined zone load; averaging the pair reports exactly half
    of the real weight.
    """
    parsed: list[float] = []
    for value in values:
        try:
            cleaned = str(value).strip()
            if cleaned.startswith("+"):
                cleaned = cleaned[1:]
            parsed.append(float(cleaned))
        except (ValueError, TypeError):
            continue
    return fsum(parsed) if parsed else 0.0


def filter_peaks(
    values: Sequence[float],
    context_window: int = 5,
    threshold_factor: float = 1.5,
    min_diff_grams: float = 50.0,
) -> list[float]:
    """Remove peak noise using neighbor median comparison."""
    if len(values) < context_window * 2 + 1:
        return list(values)

    filtered: list[float] = []
    for index, value in enumerate(values):
        start = max(0, index - context_window)
        end = min(len(values), index + context_window + 1)
        neighbors = [values[pos] for pos in range(start, end) if pos != index]

        if not neighbors:
            filtered.append(value)
            continue

        neighbor_median = float(median(neighbors))
        diff = abs(value - neighbor_median)
        threshold = max(abs(neighbor_median) * (threshold_factor - 1), min_diff_grams)

        if diff <= threshold:
            filtered.append(value)

    return filtered


def simple_delta_values(loadcells: Sequence[SupportsFilteredValue]) -> tuple[float, float, bool]:
    """Fallback delta calculation based on the first and last samples."""
    if not loadcells:
        return 0.0, 0.0, False

    try:
        first_values = _read_field(loadcells[0], "filtered_value") or []
        last_values = _read_field(loadcells[-1], "filtered_value") or []
        start = avg_loadcell_channels(first_values)
        end = avg_loadcell_channels(last_values)
        return start, end, True
    except (IndexError, AttributeError, TypeError):
        return 0.0, 0.0, False


def _parse_iso_timestamp(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _sample_span_seconds(loadcells: Sequence[SupportsFilteredValue]) -> float:
    if len(loadcells) < 2:
        return 0.0

    first = _parse_iso_timestamp(_read_field(loadcells[0], "timestamp"))
    last = _parse_iso_timestamp(_read_field(loadcells[-1], "timestamp"))
    if first is None or last is None:
        return 0.0
    return max(0.0, last - first)


def _filtered_total_records(
    loadcells: Sequence[SupportsFilteredValue],
) -> list[tuple[int, str | None, float]]:
    records: list[tuple[int, str | None, float]] = []
    for index, loadcell in enumerate(loadcells):
        filtered_value = _read_field(loadcell, "filtered_value")
        if filtered_value:
            timestamp = _read_field(loadcell, "timestamp")
            records.append(
                (
                    index,
                    str(timestamp) if timestamp is not None else None,
                    avg_loadcell_channels(filtered_value),
                )
            )
    return records


def _plateau_channel_means(
    loadcells: Sequence[SupportsFilteredValue],
    plateau: LoadcellStablePlateau,
) -> dict[int, float]:
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    start = max(0, int(plateau.start_index))
    end = min(len(loadcells) - 1, int(plateau.end_index))
    if end < start:
        return {}

    for loadcell in loadcells[start:end + 1]:
        values = _values_as_strings(_read_field(loadcell, "filtered_value"))
        for channel_index, value in enumerate(values):
            parsed = _parse_loadcell_value_optional(value)
            if parsed is None:
                continue
            sums[channel_index] = sums.get(channel_index, 0.0) + parsed
            counts[channel_index] = counts.get(channel_index, 0) + 1

    return {
        channel_index: sums[channel_index] / counts[channel_index]
        for channel_index in sorted(sums)
        if counts.get(channel_index, 0) > 0
    }


def _channel_delta_targets_from_plateaus(
    loadcells: Sequence[SupportsFilteredValue],
    *,
    stable_plateaus: Sequence[LoadcellStablePlateau],
    net_delta: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    min_delta = float(config.trigger.min_weight_change_grams)
    tolerance = max(
        float(config.weight.tolerance_grams),
        float(config.door_session.weight_tolerance_grams),
    )
    diagnostics: dict[str, object] = {
        "accepted": False,
        "reason": "not_evaluated",
        "net_delta": round(float(net_delta), 1),
        "tolerance": round(float(tolerance), 1),
        "negative_channel_count": 0,
        "positive_channel_count": 0,
        "channels": [],
    }

    if net_delta >= -min_delta:
        diagnostics["reason"] = "not_removal_delta"
        return [], diagnostics
    if len(stable_plateaus) < 2:
        diagnostics["reason"] = "insufficient_stable_plateaus"
        return [], diagnostics

    start_plateau = stable_plateaus[0]
    end_plateau = stable_plateaus[-1]
    start_means = _plateau_channel_means(loadcells, start_plateau)
    end_means = _plateau_channel_means(loadcells, end_plateau)
    channel_indices = sorted(set(start_means) & set(end_means))
    if len(channel_indices) < 2:
        diagnostics["reason"] = "insufficient_channels"
        return [], diagnostics

    channels: list[dict[str, object]] = []
    negative_channels: list[dict[str, object]] = []
    positive_channels: list[dict[str, object]] = []
    for channel_index in channel_indices:
        start_value = float(start_means[channel_index])
        end_value = float(end_means[channel_index])
        delta = end_value - start_value
        channel = {
            "channel_index": int(channel_index),
            "start": round(start_value, 1),
            "end": round(end_value, 1),
            "delta": round(delta, 1),
            "abs_delta": round(abs(delta), 1),
        }
        channels.append(channel)
        if delta <= -min_delta:
            negative_channels.append(channel)
        elif delta >= min_delta:
            positive_channels.append(channel)

    diagnostics["channels"] = channels
    diagnostics["negative_channel_count"] = len(negative_channels)
    diagnostics["positive_channel_count"] = len(positive_channels)

    if positive_channels:
        diagnostics["reason"] = "positive_channel_delta_present"
        return [], diagnostics
    if len(negative_channels) < 2:
        diagnostics["reason"] = "insufficient_negative_channels"
        return [], diagnostics

    negative_total = sum(float(channel["abs_delta"]) for channel in negative_channels)
    residual = abs(negative_total - abs(float(net_delta)))
    diagnostics["negative_total"] = round(float(negative_total), 1)
    diagnostics["residual"] = round(float(residual), 1)
    if residual > tolerance:
        diagnostics["reason"] = "channel_total_mismatch"
        return [], diagnostics

    targets: list[dict[str, object]] = []
    for target_index, channel in enumerate(negative_channels):
        weight = float(channel["abs_delta"])
        delta = float(channel["delta"])
        targets.append(
            {
                "source": "simultaneous_channel_delta",
                "weight": round(weight, 1),
                "delta": round(delta, 1),
                "segment_index": int(target_index),
                "segment_indices": [int(target_index)],
                "channel_index": int(channel["channel_index"]),
                "reason": "simultaneous_channel_removal",
                "start_timestamp": start_plateau.end_timestamp,
                "end_timestamp": end_plateau.start_timestamp,
                "duration_seconds": 0.0,
                "evidence_required": True,
            }
        )

    diagnostics["accepted"] = True
    diagnostics["reason"] = "simultaneous_channel_removal_targets"
    diagnostics["targets"] = targets
    return targets, diagnostics


def _resolve_window_size(window_size: int | None) -> int:
    return max(1, window_size or config.loadcell.stable_window_size)


def _resolve_stability_threshold(stability_threshold: float | None) -> float:
    return max(0.0, stability_threshold or config.loadcell.stability_threshold_grams)


def detect_delta_segments(
    loadcells: Sequence[SupportsFilteredValue],
    window_size: int | None = None,
    stability_threshold: float | None = None,
    min_delta_grams: float | None = None,
) -> list[LoadcellDeltaSegment]:
    """Detect compound movements while preserving net-delta behavior."""

    plateaus = detect_stable_plateaus(
        loadcells,
        window_size=window_size,
        stability_threshold=stability_threshold,
        min_delta_grams=min_delta_grams,
    )
    return _segments_from_plateaus(plateaus, min_delta_grams=min_delta_grams)


def detect_stable_plateaus(
    loadcells: Sequence[SupportsFilteredValue],
    window_size: int | None = None,
    stability_threshold: float | None = None,
    min_delta_grams: float | None = None,
) -> list[LoadcellStablePlateau]:
    """Detect collapsed stable plateaus from loadcell history."""

    resolved_window_size = _resolve_window_size(window_size)
    resolved_threshold = _resolve_stability_threshold(stability_threshold)
    min_delta = max(
        0.0,
        float(
            config.trigger.min_weight_change_grams
            if min_delta_grams is None
            else min_delta_grams
        ),
    )
    records = _filtered_total_records(loadcells)
    if len(records) < resolved_window_size * 2:
        return []

    stable_windows: list[dict[str, object]] = []
    for index in range(len(records) - resolved_window_size + 1):
        window_records = records[index:index + resolved_window_size]
        window_values = [record[2] for record in window_records]
        if pstdev(window_values) < resolved_threshold:
            stable_windows.append(
                {
                    "window_start": index,
                    "window_end": index + resolved_window_size - 1,
                    "record_start_index": window_records[0][0],
                    "record_end_index": window_records[-1][0],
                    "start_timestamp": window_records[0][1],
                    "end_timestamp": window_records[-1][1],
                    "mean": float(fmean(window_values)),
                    "samples": len(window_values),
                }
            )

    if len(stable_windows) < 2:
        return []

    plateaus: list[dict[str, object]] = []
    for window in stable_windows:
        if not plateaus:
            plateaus.append(dict(window))
            continue

        previous = plateaus[-1]
        previous_mean = float(previous["mean"])
        current_mean = float(window["mean"])
        adjacent = int(window["window_start"]) <= int(previous["window_end"]) + 1
        similar = abs(current_mean - previous_mean) <= max(resolved_threshold, min_delta)

        if adjacent and similar:
            previous_samples = int(previous["samples"])
            current_samples = int(window["samples"])
            previous["mean"] = (
                previous_mean * previous_samples + current_mean * current_samples
            ) / (previous_samples + current_samples)
            previous["samples"] = previous_samples + current_samples
            previous["window_end"] = window["window_end"]
            previous["record_end_index"] = window["record_end_index"]
            previous["end_timestamp"] = window["end_timestamp"]
            continue

        if abs(current_mean - previous_mean) < min_delta:
            previous["window_end"] = window["window_end"]
            previous["record_end_index"] = window["record_end_index"]
            previous["end_timestamp"] = window["end_timestamp"]
            continue

        plateaus.append(dict(window))

    return [
        LoadcellStablePlateau(
            start_index=int(plateau["record_start_index"]),
            end_index=int(plateau["record_end_index"]),
            start_timestamp=(
                str(plateau["start_timestamp"])
                if plateau["start_timestamp"] is not None
                else None
            ),
            end_timestamp=(
                str(plateau["end_timestamp"])
                if plateau["end_timestamp"] is not None
                else None
            ),
            avg=float(plateau["mean"]),
            samples=int(plateau["samples"]),
        )
        for plateau in plateaus
    ]


def _segments_from_plateaus(
    plateaus: Sequence[LoadcellStablePlateau],
    min_delta_grams: float | None = None,
) -> list[LoadcellDeltaSegment]:
    min_delta = max(
        0.0,
        float(
            config.trigger.min_weight_change_grams
            if min_delta_grams is None
            else min_delta_grams
        ),
    )
    segments: list[LoadcellDeltaSegment] = []
    for previous, current in zip(plateaus, plateaus[1:]):
        start_avg = float(previous.avg)
        end_avg = float(current.avg)
        delta = end_avg - start_avg
        if abs(delta) < min_delta:
            continue
        start_timestamp = previous.end_timestamp
        end_timestamp = current.start_timestamp
        start_seconds = _parse_iso_timestamp(start_timestamp)
        end_seconds = _parse_iso_timestamp(end_timestamp)
        duration = (
            max(0.0, end_seconds - start_seconds)
            if start_seconds is not None and end_seconds is not None
            else 0.0
        )
        segments.append(
            LoadcellDeltaSegment(
                start_index=int(previous.end_index),
                end_index=int(current.start_index),
                start_timestamp=str(start_timestamp) if start_timestamp is not None else None,
                end_timestamp=str(end_timestamp) if end_timestamp is not None else None,
                start_avg=start_avg,
                end_avg=end_avg,
                delta=delta,
                sign=1 if delta > 0 else -1,
                duration_seconds=duration,
            )
        )

    return segments


def _movement_dict(
    *,
    index: int,
    segment: LoadcellDeltaSegment,
    reason: str,
) -> dict[str, object]:
    data = segment.to_dict()
    data["segment_index"] = int(index)
    data["reason"] = reason
    return data


def _candidate_dict(
    *,
    source: str,
    weight: float,
    delta: float,
    segment_indices: Sequence[int],
    reason: str,
) -> dict[str, object]:
    return {
        "source": source,
        "weight": round(abs(float(weight)), 1),
        "delta": round(float(delta), 1),
        "segment_indices": [int(index) for index in segment_indices],
        "reason": reason,
    }


def _segment_target_dict(
    *,
    source: str,
    index: int,
    segment: LoadcellDeltaSegment,
    reason: str,
    evidence_required: bool = False,
) -> dict[str, object]:
    return {
        "source": source,
        "weight": round(abs(float(segment.delta)), 1),
        "delta": round(float(segment.delta), 1),
        "segment_index": int(index),
        "segment_indices": [int(index)],
        "reason": reason,
        "start_timestamp": segment.start_timestamp,
        "end_timestamp": segment.end_timestamp,
        "duration_seconds": round(float(segment.duration_seconds), 3),
        "evidence_required": bool(evidence_required),
    }


def analyze_movement_history(
    *,
    net_delta: float,
    stable_region_valid: bool,
    stable_plateaus: Sequence[LoadcellStablePlateau],
    segments: Sequence[LoadcellDeltaSegment],
    prefer_mixed_sign_removal_delta: bool = False,
) -> dict[str, object]:
    """Summarize purchase-like movements from stable loadcell history."""

    min_delta = float(config.trigger.min_weight_change_grams)
    pair_tolerance = max(
        float(config.weight.tolerance_grams),
        float(config.door_session.weight_tolerance_grams),
    )
    paired_indices: set[int] = set()
    paired_movements: list[dict[str, object]] = []
    ignored_movements: list[dict[str, object]] = []
    vision_required_segment_targets: list[dict[str, object]] = []

    for index, segment in enumerate(segments):
        if index in paired_indices:
            continue
        for other_index in range(index + 1, len(segments)):
            if other_index in paired_indices:
                continue
            other = segments[other_index]
            if segment.sign == other.sign:
                continue
            residual = abs(abs(segment.delta) - abs(other.delta))
            if residual > pair_tolerance:
                continue
            paired_indices.update({index, other_index})
            reason = (
                "pressure_release_pair"
                if segment.sign > 0 and other.sign < 0
                else "removal_return_pair"
            )
            paired_movements.append(
                {
                    "first_segment_index": int(index),
                    "second_segment_index": int(other_index),
                    "first_delta": round(float(segment.delta), 1),
                    "second_delta": round(float(other.delta), 1),
                    "residual": round(float(residual), 1),
                    "reason": reason,
                }
            )
            if reason == "pressure_release_pair" and abs(other.delta) >= min_delta:
                vision_required_segment_targets.append(
                    _segment_target_dict(
                        source="pressure_release_negative_segment",
                        index=other_index,
                        segment=other,
                        reason="paired_positive_then_negative_requires_vision",
                        evidence_required=True,
                    )
                )
            break

    for index, segment in enumerate(segments):
        if abs(segment.delta) < min_delta:
            ignored_movements.append(
                _movement_dict(
                    index=index,
                    segment=segment,
                    reason="below_min_weight_change",
                )
            )
        elif index in paired_indices:
            ignored_movements.append(
                _movement_dict(
                    index=index,
                    segment=segment,
                    reason="paired_opposite_movement",
                )
            )

    if abs(net_delta) < min_delta and not segments and len(stable_plateaus) >= 2:
        ignored_movements.append(
            {
                "reason": "net_below_min_weight_change",
                "delta": round(float(net_delta), 1),
                "threshold": round(float(min_delta), 1),
            }
        )

    unpaired_negative_indices = [
        index
        for index, segment in enumerate(segments)
        if segment.sign < 0 and index not in paired_indices
    ]
    unpaired_positive_indices = [
        index
        for index, segment in enumerate(segments)
        if segment.sign > 0 and index not in paired_indices
    ]
    removal_segment_targets = [
        _segment_target_dict(
            source="unpaired_negative_segment",
            index=index,
            segment=segments[index],
            reason="unpaired_removal_segment",
        )
        for index in unpaired_negative_indices
        if abs(segments[index].delta) >= min_delta
    ]
    return_segment_targets = [
        _segment_target_dict(
            source="unpaired_positive_segment",
            index=index,
            segment=segments[index],
            reason="unpaired_return_segment",
        )
        for index in unpaired_positive_indices
        if abs(segments[index].delta) >= min_delta
    ]
    candidates: list[dict[str, object]] = []

    def add_candidate(candidate: dict[str, object]) -> None:
        weight = float(candidate["weight"])
        if weight < min_delta:
            return
        for existing in candidates:
            if abs(float(existing["weight"]) - weight) < 0.1:
                return
        candidates.append(candidate)

    total_unpaired_negative = (
        sum(abs(float(segments[index].delta)) for index in unpaired_negative_indices)
        if unpaired_negative_indices
        else 0.0
    )
    chargeable_negative_indices = [
        index
        for index in unpaired_negative_indices
        if abs(float(segments[index].delta)) >= min_delta
    ]
    chargeable_positive_indices = [
        index
        for index in unpaired_positive_indices
        if abs(float(segments[index].delta)) >= min_delta
    ]
    mixed_sign_negative_total = (
        sum(abs(float(segments[index].delta)) for index in chargeable_negative_indices)
        if chargeable_negative_indices
        else 0.0
    )
    mixed_sign_positive_total = (
        sum(abs(float(segments[index].delta)) for index in chargeable_positive_indices)
        if chargeable_positive_indices
        else 0.0
    )
    prefer_unpaired_negative_total = bool(
        prefer_mixed_sign_removal_delta
        and chargeable_negative_indices
        and chargeable_positive_indices
    )

    if prefer_unpaired_negative_total:
        add_candidate(
            _candidate_dict(
                source="unpaired_negative_total",
                weight=mixed_sign_negative_total,
                delta=-mixed_sign_negative_total,
                segment_indices=chargeable_negative_indices,
                reason="freezer_mixed_sign_removal_total_preferred",
            )
        )

    if stable_region_valid and net_delta < -min_delta:
        add_candidate(
            _candidate_dict(
                source="net_stable_delta",
                weight=abs(net_delta),
                delta=net_delta,
                segment_indices=[],
                reason="stable_start_end_net_removal",
            )
        )

    if unpaired_negative_indices:
        total_unpaired = total_unpaired_negative
        if not prefer_unpaired_negative_total:
            add_candidate(
                _candidate_dict(
                    source="unpaired_negative_total",
                    weight=total_unpaired,
                    delta=-total_unpaired,
                    segment_indices=unpaired_negative_indices,
                    reason="sum_of_unpaired_removal_segments",
                )
            )
        last_index = unpaired_negative_indices[-1]
        add_candidate(
            _candidate_dict(
                source="last_unpaired_negative_segment",
                weight=abs(float(segments[last_index].delta)),
                delta=float(segments[last_index].delta),
                segment_indices=[last_index],
                reason="latest_unpaired_removal_segment",
            )
        )
        for index in unpaired_negative_indices:
            add_candidate(
                _candidate_dict(
                    source="unpaired_negative_segment",
                    weight=abs(float(segments[index].delta)),
                    delta=float(segments[index].delta),
                    segment_indices=[index],
                    reason="unpaired_removal_segment",
                )
            )

    pressure_like_event = (
        bool(paired_movements)
        and not unpaired_negative_indices
        and abs(net_delta) < min_delta
    )
    decision_delta = (
        -float(candidates[0]["weight"])
        if candidates
        else float(net_delta)
    )
    mixed_sign_guard: dict[str, object] = {}
    if prefer_unpaired_negative_total:
        selected = candidates[0] if candidates else {}
        mixed_sign_guard = {
            "accepted": True,
            "reason": "mixed_sign_net_masking_guard",
            "net_delta": round(float(net_delta), 1),
            "return_total": round(float(mixed_sign_positive_total), 1),
            "removal_total": round(float(mixed_sign_negative_total), 1),
            "selected_source": selected.get("source"),
            "selected_decision_delta": round(float(decision_delta), 1),
        }
    return {
        "purchase_delta_candidates": candidates,
        "removal_segment_targets": removal_segment_targets,
        "return_segment_targets": return_segment_targets,
        "vision_required_segment_targets": vision_required_segment_targets,
        "paired_loadcell_movements": paired_movements,
        "ignored_loadcell_movements": ignored_movements,
        "mixed_sign_net_masking_guard": mixed_sign_guard,
        "pressure_like_event": pressure_like_event,
        "decision_delta": decision_delta,
    }


def _apply_movement_history(
    analysis: LoadcellDeltaAnalysis,
    *,
    net_delta: float,
    stable_region_valid: bool,
    prefer_mixed_sign_removal_delta: bool = False,
) -> None:
    movement_history = analyze_movement_history(
        net_delta=net_delta,
        stable_region_valid=stable_region_valid,
        stable_plateaus=analysis.stable_plateaus,
        segments=analysis.segments,
        prefer_mixed_sign_removal_delta=prefer_mixed_sign_removal_delta,
    )
    analysis.purchase_delta_candidates = list(
        movement_history["purchase_delta_candidates"]
    )
    analysis.removal_segment_targets = list(
        movement_history["removal_segment_targets"]
    )
    analysis.return_segment_targets = list(
        movement_history["return_segment_targets"]
    )
    analysis.vision_required_segment_targets = list(
        movement_history["vision_required_segment_targets"]
    )
    analysis.paired_loadcell_movements = list(
        movement_history["paired_loadcell_movements"]
    )
    analysis.ignored_loadcell_movements = list(
        movement_history["ignored_loadcell_movements"]
    )
    analysis.mixed_sign_net_masking_guard = dict(
        movement_history["mixed_sign_net_masking_guard"]
    )
    analysis.pressure_like_event = bool(movement_history["pressure_like_event"])
    analysis.decision_delta = float(movement_history["decision_delta"])


def _raw_extreme_delta(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    simple_direction = float(values[-1]) - float(values[0])
    if simple_direction < 0:
        return float(min(values)) - float(max(values))
    return float(max(values)) - float(min(values))


def _apply_endpoint_fallback_if_eligible(
    analysis: LoadcellDeltaAnalysis,
    *,
    loadcells: Sequence[SupportsFilteredValue],
    records: Sequence[tuple[int, str | None, float]],
    endpoint_fallback_enabled: bool,
) -> None:
    """Use a conservative endpoint removal delta when stable history is absent."""

    min_delta = float(config.trigger.min_weight_change_grams)
    if analysis.stable_region_valid and not analysis.used_simple_fallback:
        analysis.decision_delta_reliable = True
        analysis.endpoint_fallback_reason = (
            "already_has_chargeable_decision_delta"
            if analysis.decision_delta < -min_delta
            else "confirmed_stable_plateaus"
        )
        return

    if not endpoint_fallback_enabled:
        analysis.endpoint_fallback_reason = "disabled"
        return

    if analysis.decision_delta < -min_delta:
        analysis.endpoint_fallback_reason = "already_has_chargeable_decision_delta"
        return

    min_samples = max(2, int(config.loadcell.endpoint_fallback_min_samples))
    if len(records) < min_samples:
        analysis.endpoint_fallback_reason = "insufficient_samples"
        return

    min_span_seconds = max(0.0, float(config.loadcell.endpoint_fallback_min_span_seconds))
    if analysis.sample_span_seconds < min_span_seconds:
        analysis.endpoint_fallback_reason = "insufficient_sample_span"
        return

    filtered_summary = _summarize_channel_field(loadcells, "filtered_value")
    if filtered_summary["filtered_state"] != "nonzero":
        analysis.endpoint_fallback_reason = "filtered_payload_not_nonzero"
        return

    endpoint_delta = float(analysis.raw_simple_delta)
    analysis.endpoint_delta_weight = endpoint_delta
    if endpoint_delta >= -min_delta:
        analysis.endpoint_fallback_reason = "not_negative_endpoint_delta"
        return

    values = [float(record[2]) for record in records]
    if len(values) < 2:
        analysis.endpoint_fallback_reason = "insufficient_parsed_records"
        return

    start_value = float(values[0])
    end_value = float(values[-1])
    high_value = max(values)
    low_value = min(values)
    endpoint_margin = max(
        float(config.loadcell.stability_threshold_grams),
        float(config.weight.tolerance_grams),
        float(config.door_session.weight_tolerance_grams),
        min_delta,
    )
    if abs(start_value - high_value) > endpoint_margin:
        analysis.endpoint_fallback_reason = "start_not_near_payload_high"
        return
    if abs(end_value - low_value) > endpoint_margin:
        analysis.endpoint_fallback_reason = "end_not_near_payload_low"
        return

    analysis.decision_delta = endpoint_delta
    analysis.decision_delta_reliable = True
    analysis.endpoint_fallback_applied = True
    analysis.endpoint_fallback_reason = "freezer_endpoint_delta"
    analysis.stable_delta_source = "freezer_endpoint_fallback"
    analysis.purchase_delta_candidates = [
        _candidate_dict(
            source="freezer_endpoint_delta",
            weight=abs(endpoint_delta),
            delta=endpoint_delta,
            segment_indices=[],
            reason="freezer_endpoint_fallback",
        )
    ]


def resolved_cabinet_type(cabinet_type: str | None = None) -> str:
    return (cabinet_type or config.machine.cabinet_type).strip().lower()


def endpoint_fallback_enabled_for_cabinet(cabinet_type: str | None = None) -> bool:
    return (
        resolved_cabinet_type(cabinet_type) == "freezer"
        and config.loadcell.freezer_endpoint_fallback_enabled
    )


def prefer_mixed_sign_removal_delta_for_cabinet(
    cabinet_type: str | None = None,
) -> bool:
    return resolved_cabinet_type(cabinet_type) == "freezer"


def analyze_weight_delta(
    loadcells: Sequence[SupportsFilteredValue],
    window_size: int | None = None,
    stability_threshold: float | None = None,
    endpoint_fallback_enabled: bool = False,
    prefer_mixed_sign_removal_delta: bool = False,
) -> LoadcellDeltaAnalysis:
    """Analyze trigger-level loadcell movement with stable-window diagnostics."""

    resolved_window_size = _resolve_window_size(window_size)
    resolved_threshold = _resolve_stability_threshold(stability_threshold)

    analysis = LoadcellDeltaAnalysis(
        sample_count=len(loadcells),
        sample_span_seconds=_sample_span_seconds(loadcells),
        window_size=resolved_window_size,
        stability_threshold=resolved_threshold,
    )

    if len(loadcells) < 2:
        analysis.reason = "insufficient_samples"
        return analysis

    records = _filtered_total_records(loadcells)
    values = [record[2] for record in records]

    analysis.parsed_sample_count = len(values)
    simple_start, simple_end, simple_valid = simple_delta_values(loadcells)
    if simple_valid:
        analysis.start_avg = simple_start
        analysis.end_avg = simple_end
        analysis.raw_simple_delta = simple_end - simple_start
        analysis.endpoint_delta_weight = analysis.raw_simple_delta
    analysis.raw_extreme_delta = _raw_extreme_delta(values)
    analysis.stable_plateaus = detect_stable_plateaus(
        loadcells,
        window_size=resolved_window_size,
        stability_threshold=resolved_threshold,
    )
    analysis.segments = _segments_from_plateaus(analysis.stable_plateaus)

    if len(values) < resolved_window_size * 2:
        if simple_valid:
            analysis.delta = analysis.raw_simple_delta
            analysis.used_simple_fallback = True
            analysis.stable_delta_source = "simple_fallback_diagnostic"
        analysis.reason = (
            "insufficient_stable_samples" if simple_valid else "invalid_loadcell_samples"
        )
        _apply_endpoint_fallback_if_eligible(
            analysis,
            loadcells=loadcells,
            records=records,
            endpoint_fallback_enabled=endpoint_fallback_enabled,
        )
        return analysis

    filtered_values = filter_peaks(values)
    analysis.working_sample_count = (
        len(filtered_values) if len(filtered_values) >= resolved_window_size * 2 else len(values)
    )

    if len(analysis.stable_plateaus) < 2:
        analysis.reason = "unstable_or_truncated_loadcell"
        analysis.stable_delta_source = "insufficient_stable_plateaus"
        _apply_endpoint_fallback_if_eligible(
            analysis,
            loadcells=loadcells,
            records=records,
            endpoint_fallback_enabled=endpoint_fallback_enabled,
        )
        return analysis

    baseline = analysis.stable_plateaus[0]
    final = analysis.stable_plateaus[-1]
    analysis.start_stable_idx = int(baseline.start_index)
    analysis.end_stable_idx = int(final.start_index)
    analysis.start_avg = float(baseline.avg)
    analysis.end_avg = float(final.avg)
    analysis.baseline_stable_avg = float(baseline.avg)
    analysis.final_stable_avg = float(final.avg)
    analysis.trailing_unstable_sample_count = sum(
        1 for record in records if int(record[0]) > int(final.end_index)
    )

    max_trailing_unstable = max(0, resolved_window_size - 1)
    if analysis.trailing_unstable_sample_count > max_trailing_unstable:
        analysis.delta = 0.0
        analysis.decision_delta = 0.0
        analysis.reason = "unstable_or_truncated_loadcell"
        analysis.stable_delta_source = "stable_tail_not_confirmed"
        _apply_endpoint_fallback_if_eligible(
            analysis,
            loadcells=loadcells,
            records=records,
            endpoint_fallback_enabled=endpoint_fallback_enabled,
        )
        return analysis

    analysis.delta = float(final.avg) - float(baseline.avg)
    analysis.reason = "stable_regions"
    analysis.stable_region_valid = True
    analysis.stable_delta_source = "confirmed_stable_plateaus"
    _apply_movement_history(
        analysis,
        net_delta=analysis.delta,
        stable_region_valid=True,
        prefer_mixed_sign_removal_delta=prefer_mixed_sign_removal_delta,
    )
    _apply_endpoint_fallback_if_eligible(
        analysis,
        loadcells=loadcells,
        records=records,
        endpoint_fallback_enabled=endpoint_fallback_enabled,
    )
    (
        analysis.channel_removal_segment_targets,
        analysis.channel_delta_diagnostics,
    ) = _channel_delta_targets_from_plateaus(
        loadcells,
        stable_plateaus=analysis.stable_plateaus,
        net_delta=analysis.delta,
    )
    return analysis


def detect_stable_regions(
    loadcells: Sequence[SupportsFilteredValue],
    window_size: int | None = None,
    stability_threshold: float | None = None,
) -> tuple[float, float, bool]:
    """Detect stable windows at the start and end of a loadcell sequence."""

    analysis = analyze_weight_delta(
        loadcells,
        window_size=window_size,
        stability_threshold=stability_threshold,
    )

    return analysis.start_avg, analysis.end_avg, analysis.stable_region_valid


def calculate_weight_delta(loadcells: Sequence[SupportsFilteredValue]) -> float:
    """Calculate the overall weight delta for a trigger."""

    return analyze_weight_delta(loadcells).delta
