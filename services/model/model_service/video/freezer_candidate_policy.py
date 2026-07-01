"""Shared freezer candidate repeat/count policy helpers."""

from __future__ import annotations

from typing import Any

from model_service.core.config import config


def freezer_weight_tolerance_grams() -> float:
    return max(0.0, float(config.weight.freezer_weight_tolerance_grams))


def supported_instance_count(
    *,
    unit_weight: float | None,
    target_weight: float,
    instance_count_hint: int,
    stock: int | None = None,
) -> int:
    """Return the supported repeated count from explicit instance evidence."""
    if unit_weight is None or unit_weight <= 0.0:
        return 1
    caps = [
        max(1, int(instance_count_hint)),
        max(1, int(config.weight.max_count_per_item)),
    ]
    if stock is not None:
        caps.append(max(0, int(stock)))
    max_hint = max(1, min(caps))
    if max_hint <= 1:
        return 1

    best_count = 1
    best_residual = abs(float(target_weight) - float(unit_weight))
    for count in range(2, max_hint + 1):
        residual = abs(float(target_weight) - float(unit_weight) * count)
        if residual <= freezer_weight_tolerance_grams() and residual < best_residual:
            best_count = count
            best_residual = residual
    return best_count


def same_product_repeat_diagnostic(
    *,
    class_id: int,
    name: str,
    target_weight: float,
    unit_weight: float | None,
    stock: int | None,
    single_residual: float,
    confidence: float,
    exit_path_votes: int,
    vote_count: int,
    source: str,
    single_regular_vision_identity: bool = False,
) -> dict[str, Any] | None:
    """Build repeat-count diagnostics for a freezer same-product candidate."""
    if unit_weight is None or unit_weight <= 0.0:
        return None
    nearest_count = int(round(float(target_weight) / float(unit_weight)))
    if nearest_count < 2:
        return None

    caps = [
        max(1, int(config.weight.max_items_per_segment)),
        max(1, int(config.weight.same_product_max_count)),
        max(1, int(config.weight.max_count_per_item)),
    ]
    if stock is not None:
        caps.append(max(0, int(stock)))
    max_count = min(caps)
    base = {
        "class_id": int(class_id),
        "name": name,
        "nearestCount": nearest_count,
        "maxCount": int(max_count),
        "unitWeight": round(float(unit_weight), 1),
        "stock": stock,
    }
    if max_count < 2:
        return {**base, "accepted": False, "reason": "count_cap_below_repeat"}

    if nearest_count > max_count:
        return {
            **base,
            "count": int(nearest_count),
            "accepted": False,
            "reason": "nearest_repeat_count_exceeds_cap",
        }

    expected_weight = float(unit_weight) * nearest_count
    repeat_residual = abs(float(target_weight) - expected_weight)
    allowed_residual = freezer_weight_tolerance_grams()
    min_votes = max(
        int(config.vision.freezer_min_vote_count),
        int(config.weight.detected_single_fallback_min_votes),
    )
    diagnostic: dict[str, Any] = {
        **base,
        "count": int(nearest_count),
        "expectedWeight": round(float(expected_weight), 1),
        "countWeightResidual": round(float(repeat_residual), 1),
        "countAllowedResidual": round(float(allowed_residual), 1),
        "confidence": round(float(confidence), 4),
        "freezerExitPathVotes": int(exit_path_votes),
        "voteCount": int(vote_count),
        "minRepeatVotes": int(min_votes),
        "singleRegularVisionIdentity": bool(single_regular_vision_identity),
        "accepted": False,
    }
    if str(source) != "vision":
        diagnostic["reason"] = "not_regular_vision_candidate"
    elif confidence < float(config.weight.freezer_multi_min_confidence):
        diagnostic["reason"] = "confidence_below_repeat_floor"
    elif repeat_residual > allowed_residual:
        diagnostic["reason"] = "repeat_residual_exceeds_tolerance"
    elif repeat_residual >= float(single_residual):
        diagnostic["reason"] = "single_residual_not_worse"
    elif bool(single_regular_vision_identity) and vote_count > 0:
        diagnostic["accepted"] = True
        diagnostic["reason"] = "same_product_repeat_weight_gate"
        diagnostic["repeatEvidenceMode"] = "single_regular_vision_identity"
    elif exit_path_votes < int(config.vision.freezer_min_exit_path_votes):
        diagnostic["reason"] = "insufficient_exit_path_votes"
    elif vote_count < min_votes:
        diagnostic["reason"] = "insufficient_repeat_votes"
    else:
        diagnostic["accepted"] = True
        diagnostic["reason"] = "same_product_repeat_weight_gate"
        diagnostic["repeatEvidenceMode"] = "exit_path_votes"
    return diagnostic
