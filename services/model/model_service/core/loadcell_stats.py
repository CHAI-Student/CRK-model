from __future__ import annotations

"""Stdlib-only helpers for loadcell parsing and weight delta calculation."""

from math import fsum
from statistics import fmean, median, pstdev
from typing import Protocol, Sequence


class SupportsFilteredValue(Protocol):
    """Duck-typed protocol for trigger loadcell data."""

    filtered_value: Sequence[object] | None


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
        first_values = getattr(loadcells[0], "filtered_value", None) or []
        last_values = getattr(loadcells[-1], "filtered_value", None) or []
        start = avg_loadcell_channels(first_values)
        end = avg_loadcell_channels(last_values)
        return start, end, True
    except (IndexError, AttributeError, TypeError):
        return 0.0, 0.0, False


def detect_stable_regions(
    loadcells: Sequence[SupportsFilteredValue],
    window_size: int = 5,
    stability_threshold: float = 15.0,
) -> tuple[float, float, bool]:
    """Detect stable windows at the start and end of a loadcell sequence."""
    if len(loadcells) < window_size * 2:
        return simple_delta_values(loadcells)

    values: list[float] = []
    for loadcell in loadcells:
        filtered_value = getattr(loadcell, "filtered_value", None)
        if filtered_value:
            values.append(avg_loadcell_channels(filtered_value))

    if len(values) < window_size * 2:
        return simple_delta_values(loadcells)

    filtered_values = filter_peaks(values)
    working_values = filtered_values if len(filtered_values) >= window_size * 2 else values

    start_stable_idx = 0
    for index in range(len(working_values) - window_size):
        window = working_values[index:index + window_size]
        if pstdev(window) < stability_threshold:
            start_stable_idx = index
            break

    start_region = working_values[start_stable_idx:start_stable_idx + window_size]
    start_avg = float(fmean(start_region))

    end_stable_idx = len(working_values) - window_size
    for index in range(len(working_values) - 1, window_size - 1, -1):
        window = working_values[index - window_size + 1:index + 1]
        if pstdev(window) < stability_threshold:
            end_stable_idx = index - window_size + 1
            break

    end_region = working_values[end_stable_idx:end_stable_idx + window_size]
    end_avg = float(fmean(end_region))
    is_valid = end_stable_idx > start_stable_idx + window_size

    return start_avg, end_avg, is_valid


def calculate_weight_delta(loadcells: Sequence[SupportsFilteredValue]) -> float:
    """Calculate the overall weight delta for a trigger."""
    if not loadcells or len(loadcells) < 2:
        return 0.0

    start_avg, end_avg, is_valid = detect_stable_regions(loadcells)
    if not is_valid:
        start_avg, end_avg, is_valid = simple_delta_values(loadcells)
        if not is_valid:
            return 0.0

    return end_avg - start_avg
