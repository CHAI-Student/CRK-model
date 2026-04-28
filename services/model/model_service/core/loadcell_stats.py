from __future__ import annotations

"""Stdlib-only helpers for loadcell parsing and weight delta calculation."""

from dataclasses import dataclass
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


def parse_loadcell_value(value: object) -> float:
    """Parse a loadcell value string such as '+12345'."""
    try:
        cleaned = str(value).strip()
        if cleaned.startswith("+"):
            cleaned = cleaned[1:]
        return float(cleaned)
    except (ValueError, TypeError):
        return 0.0


def avg_loadcell_channels(values: Sequence[object]) -> float:
    """Average parsed loadcell channels, ignoring invalid values."""
    parsed: list[float] = []
    for value in values:
        try:
            cleaned = str(value).strip()
            if cleaned.startswith("+"):
                cleaned = cleaned[1:]
            parsed.append(float(cleaned))
        except (ValueError, TypeError):
            continue
    return fsum(parsed) / len(parsed) if parsed else 0.0


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


def _resolve_window_size(window_size: int | None) -> int:
    return max(1, window_size or config.loadcell.stable_window_size)


def _resolve_stability_threshold(stability_threshold: float | None) -> float:
    return max(0.0, stability_threshold or config.loadcell.stability_threshold_grams)


def analyze_weight_delta(
    loadcells: Sequence[SupportsFilteredValue],
    window_size: int | None = None,
    stability_threshold: float | None = None,
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

    values: list[float] = []
    for loadcell in loadcells:
        filtered_value = _read_field(loadcell, "filtered_value")
        if filtered_value:
            values.append(avg_loadcell_channels(filtered_value))

    analysis.parsed_sample_count = len(values)
    if len(values) < resolved_window_size * 2:
        start_avg, end_avg, is_valid = simple_delta_values(loadcells)
        analysis.start_avg = start_avg
        analysis.end_avg = end_avg
        analysis.reason = (
            "insufficient_stable_samples" if is_valid else "invalid_loadcell_samples"
        )
        return analysis

    filtered_values = filter_peaks(values)
    working_values = filtered_values if len(filtered_values) >= resolved_window_size * 2 else values
    analysis.working_sample_count = len(working_values)

    start_stable_idx: int | None = None
    for index in range(len(working_values) - resolved_window_size + 1):
        window = working_values[index:index + resolved_window_size]
        if pstdev(window) < resolved_threshold:
            start_stable_idx = index
            break

    if start_stable_idx is None:
        fallback_start_avg, fallback_end_avg, _ = simple_delta_values(loadcells)
        analysis.start_avg = fallback_start_avg
        analysis.end_avg = fallback_end_avg
        analysis.reason = "unstable_or_truncated_loadcell"
        return analysis

    start_region = working_values[start_stable_idx:start_stable_idx + resolved_window_size]
    start_avg = float(fmean(start_region))

    end_stable_idx: int | None = None
    for index in range(len(working_values) - 1, resolved_window_size - 2, -1):
        window = working_values[index - resolved_window_size + 1:index + 1]
        if pstdev(window) < resolved_threshold:
            end_stable_idx = index - resolved_window_size + 1
            break

    if end_stable_idx is None:
        fallback_start_avg, fallback_end_avg, _ = simple_delta_values(loadcells)
        analysis.start_stable_idx = start_stable_idx
        analysis.start_avg = fallback_start_avg
        analysis.end_avg = fallback_end_avg
        analysis.reason = "unstable_or_truncated_loadcell"
        return analysis

    end_region = working_values[end_stable_idx:end_stable_idx + resolved_window_size]
    end_avg = float(fmean(end_region))
    stable_region_valid = end_stable_idx > start_stable_idx + resolved_window_size

    analysis.start_stable_idx = start_stable_idx
    analysis.end_stable_idx = end_stable_idx
    analysis.start_avg = start_avg
    analysis.end_avg = end_avg
    analysis.stable_region_valid = stable_region_valid

    if stable_region_valid:
        analysis.delta = end_avg - start_avg
        analysis.reason = "stable_regions"
        return analysis

    fallback_start_avg, fallback_end_avg, _ = simple_delta_values(loadcells)
    analysis.start_avg = fallback_start_avg
    analysis.end_avg = fallback_end_avg
    analysis.reason = "unstable_or_truncated_loadcell"
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
