"""Model/catalog consistency checks used at Jetson startup."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModelClassValidationResult:
    ok: bool
    engine_class_count: int
    dataset_class_count: int
    mapping_class_count: int
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    missing_in_engine: list[int] = field(default_factory=list)
    missing_in_dataset: list[int] = field(default_factory=list)
    missing_in_mapping: list[int] = field(default_factory=list)


def _normalize_engine_names(engine_class_names: dict[Any, Any]) -> dict[int, str]:
    normalized: dict[int, str] = {}
    for key, value in (engine_class_names or {}).items():
        try:
            class_id = int(key)
        except (TypeError, ValueError):
            continue
        normalized[class_id] = str(value)
    return normalized


def _read_dataset_names(dataset_path: Path) -> dict[int, str]:
    if not dataset_path.exists():
        return {}

    names: dict[int, str] = {}
    pattern = re.compile(r"^\s*(\d+):\s*(.+?)\s*$")
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            names[int(match.group(1))] = match.group(2).strip().strip("'\"")
    return names


def _read_mapping_names(mapping_path: Path) -> dict[int, str]:
    if not mapping_path.exists():
        return {}

    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    names: dict[int, str] = {}
    for item in payload.get("mappings", []):
        class_id = item.get("yolo_class_id")
        class_name = item.get("yolo_class_name")
        if class_id is None or not class_name:
            continue
        names[int(class_id)] = str(class_name)
    return names


def validate_model_class_mapping(
    *,
    engine_class_names: dict[Any, Any],
    dataset_path: Path,
    mapping_path: Path,
) -> ModelClassValidationResult:
    """Compare engine class names against dataset.yaml and yolo mapping."""

    engine_names = _normalize_engine_names(engine_class_names)
    dataset_names = _read_dataset_names(Path(dataset_path))
    mapping_names = _read_mapping_names(Path(mapping_path))

    relevant_ids = set(engine_names) | set(dataset_names) | set(mapping_names)
    mismatches: list[dict[str, Any]] = []
    missing_in_engine: list[int] = []
    missing_in_dataset: list[int] = []
    missing_in_mapping: list[int] = []

    for class_id in sorted(relevant_ids):
        engine = engine_names.get(class_id)
        dataset = dataset_names.get(class_id)
        mapping = mapping_names.get(class_id)

        if engine is None:
            missing_in_engine.append(class_id)
        if dataset is None:
            missing_in_dataset.append(class_id)
        if class_id != 0 and mapping is None:
            missing_in_mapping.append(class_id)

        present_values = {value for value in (engine, dataset, mapping) if value is not None}
        if len(present_values) > 1:
            mismatches.append(
                {
                    "class_id": class_id,
                    "dataset": dataset,
                    "mapping": mapping,
                    "engine": engine,
                }
            )

    result = ModelClassValidationResult(
        ok=not mismatches and not missing_in_engine and not missing_in_dataset and not missing_in_mapping,
        engine_class_count=len(engine_names),
        dataset_class_count=len(dataset_names),
        mapping_class_count=len(mapping_names),
        mismatches=mismatches,
        missing_in_engine=missing_in_engine,
        missing_in_dataset=missing_in_dataset,
        missing_in_mapping=missing_in_mapping,
    )

    if result.ok:
        logger.info(
            "[MODEL-VALIDATION] class mapping ok: engine=%s dataset=%s mapping=%s",
            result.engine_class_count,
            result.dataset_class_count,
            result.mapping_class_count,
        )
    else:
        logger.warning(
            "[MODEL-VALIDATION] class mapping issues: mismatches=%s missing_engine=%s "
            "missing_dataset=%s missing_mapping=%s",
            len(result.mismatches),
            result.missing_in_engine[:10],
            result.missing_in_dataset[:10],
            result.missing_in_mapping[:10],
        )

    return result
