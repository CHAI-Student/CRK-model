from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from model_service.core.config import WeightModel, config
from model_service.engine.decision_engine import ProductDecisionEngine
from model_service.engine.models import EnsembleResult, JudgmentStatus


@dataclass
class MockActiveProduct:
    yolo_class_id: int
    product_name: str
    product_weight: float
    stock_qty: int
    sale_price: int = 3500
    product_idx: str = "P1"
    has_loadcell: str = "true"


@pytest.fixture(autouse=True)
def reset_weight_identity_policy(monkeypatch):
    monkeypatch.setattr(config.weight, "identity_policy", "vision_first")
    monkeypatch.setattr(config.machine, "cabinet_type", "refrigerated")
    monkeypatch.setattr(
        config.weight,
        "freezer_vision_multi_without_weight_enabled",
        True,
    )
    monkeypatch.setattr(config.weight, "freezer_weight_tolerance_grams", 15.0)
    monkeypatch.setattr(
        config.weight,
        "freezer_distinct_mixed_preference_enabled",
        True,
    )
    monkeypatch.setattr(
        config.weight,
        "freezer_distinct_mixed_max_extra_residual_grams",
        5.0,
    )
    monkeypatch.setattr(config.weight, "freezer_prior_trigger_dedupe_enabled", True)


def test_low_confidence_trace_evidence_does_not_create_fallback_identity(
    monkeypatch,
):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")
    monkeypatch.setattr(config.vision, "top_confidence_threshold", 0.70)
    monkeypatch.setattr(config.vision, "side_confidence_threshold", 0.70)

    trace_context = SimpleNamespace(
        stage_counts_by_class={
            "42": {
                "class_id": 42,
                "name": "LOW_STAGE_PRODUCT",
                "raw": 25,
                "raw_max_confidence": 0.69,
                "motion_gate_passed": True,
            }
        },
        diagnostic_detections=[
            {
                "class_id": 43,
                "name": "LOW_DIAGNOSTIC_PRODUCT",
                "confidence": 0.69,
            }
            for _ in range(6)
        ],
    )
    rescue_candidate = EnsembleResult(
        class_id=44,
        class_name="LOW_RESCUE_PRODUCT",
        top_confidence=0.0,
        side_confidence=0.69,
        combined_confidence=0.69,
        vote_count=10,
        source="threshold_rescue",
        raw_vote_count=10,
        weight_gate_passed=True,
        side_motion_passed=True,
    )

    evidence = ProductDecisionEngine(strict_mode=True)._collect_detected_single_evidence(
        [rescue_candidate],
        trace_context,
    )

    assert evidence == {}


def use_weight_aware_identity(monkeypatch):
    monkeypatch.setattr(config.weight, "identity_policy", "weight_aware")


def test_weight_model_default_segment_grip_limit_is_three():
    assert WeightModel().max_items_per_segment == 3


def test_weight_model_defaults_to_vision_first_identity_policy():
    settings = WeightModel()

    assert settings.identity_policy == "vision_first"
    assert settings.fusion_vision_weight == 0.65
    assert settings.fusion_loadcell_weight == 0.25
    assert settings.fusion_count_weight == 0.10
    assert settings.freezer_weight_tolerance_grams == 15.0
    assert settings.freezer_vision_multi_without_weight_enabled is True
    assert settings.freezer_distinct_mixed_preference_enabled is True
    assert settings.freezer_distinct_mixed_max_extra_residual_grams == 5.0
    assert settings.freezer_prior_trigger_dedupe_enabled is True


def make_candidate(
    class_id: int = 26,
    name: str = "치킨마요",
    confidence: float = 1.0,
    raw_vote_count: int = 6,
    instance_count_hint: int = 1,
) -> EnsembleResult:
    return EnsembleResult(
        class_id=class_id,
        class_name=name,
        top_confidence=confidence,
        side_confidence=confidence,
        combined_confidence=confidence,
        vote_count=2,
        raw_vote_count=raw_vote_count,
        top_motion_passed=True,
        side_motion_passed=True,
        motion_gate_passed=True,
        instance_count_hint=instance_count_hint,
    )


def make_freezer_candidate(
    *,
    class_id: int,
    name: str,
    combined: float,
    top: float = 0.0,
    side: float = 0.0,
    raw_vote_count: int = 6,
    motion_gate_passed: bool = True,
    source: str = "vision",
) -> EnsembleResult:
    return EnsembleResult(
        class_id=class_id,
        class_name=name,
        top_confidence=top,
        side_confidence=side,
        combined_confidence=combined,
        vote_count=2 if top > 0.0 and side > 0.0 else 1,
        source=source,
        raw_vote_count=raw_vote_count,
        top_motion_passed=motion_gate_passed and top > 0.0,
        side_motion_passed=motion_gate_passed and side > 0.0,
        motion_gate_passed=motion_gate_passed,
    )


def make_active_product(
    class_id: int = 26,
    name: str = "치킨마요",
    weight: float = 365.0,
    stock: int = 5,
    price: int = 3500,
    product_idx: str = "P1",
) -> MockActiveProduct:
    return MockActiveProduct(
        yolo_class_id=class_id,
        product_name=name,
        product_weight=weight,
        stock_qty=stock,
        sale_price=price,
        product_idx=product_idx,
    )


class FakeStageTrace:
    def __init__(self, entries):
        self.weight_diagnostics = {}
        self.stage_counts_by_class = {
            str(class_id): self._stage_entry(class_id, name, confidence)
            for class_id, name, confidence in entries
        }

    @staticmethod
    def _stage_entry(class_id, name, confidence=0.92):
        return {
            "class_id": class_id,
            "name": name,
            "raw": 4,
            "raw_max_confidence": confidence,
            "motion_passed": True,
        }

    def record_weight_diagnostics(self, diagnostics):
        self.weight_diagnostics.update(diagnostics)


class FakeLoadcellTrace:
    def __init__(self, loadcell):
        self.loadcell = loadcell
        self.weight_diagnostics = {}
        self.stage_counts_by_class = {}

    def record_weight_diagnostics(self, diagnostics):
        self.weight_diagnostics.update(diagnostics)


def make_stage_count_entry(
    class_id: int,
    name: str,
    *,
    confidence: float,
    raw: int,
    threshold_passed: int = 0,
    motion_gate_passed: bool = True,
    camera: str = "side",
) -> dict:
    camera_entry = {
        "raw": raw,
        "raw_max_confidence": confidence,
    }
    if threshold_passed > 0:
        camera_entry.update(
            {
                "threshold_passed": threshold_passed,
                "threshold_passed_max_confidence": confidence,
                "motion_filtered": threshold_passed if motion_gate_passed else 0,
            }
        )
    entry = {
        "class_id": class_id,
        "name": name,
        "raw": raw,
        "raw_max_confidence": confidence,
        "threshold_passed": threshold_passed,
        "threshold_passed_max_confidence": confidence if threshold_passed > 0 else 0.0,
        "motion_gate_passed": motion_gate_passed,
        "cameras": {camera: camera_entry},
    }
    if motion_gate_passed:
        entry["motion_passed"] = max(raw, threshold_passed)
    return entry


def make_freezer_channel_trace(
    *,
    weight: float,
    side: str = "left",
    index: int = 0,
) -> FakeLoadcellTrace:
    return FakeLoadcellTrace(
        {
            "channel_movement_targets": [
                {
                    "source": "stable_channel_delta",
                    "direction": "removal",
                    "weight": weight,
                    "delta": -abs(weight),
                    "channel_index": index,
                    "channel_position": index,
                    "channel_side": side,
                }
            ]
        }
    )


def make_freezer_channel_stage_evidence(
    *,
    class_id: int,
    name: str,
    raw: int,
    confidence: float,
    threshold_passed: int,
    exit_path_votes: int,
    top_confidence: float = 0.0,
    side_confidence: float = 0.0,
    hand_path_passed: bool = True,
    hand_path_blocked: bool = False,
    trajectory_passed: bool = True,
    static_shelf: bool = False,
    path_displacement: float | None = 60.0,
    motion_threshold: float = 12.0,
) -> dict:
    top_confidence = top_confidence or confidence
    side_confidence = side_confidence or 0.0
    top_votes = max(0, exit_path_votes - (exit_path_votes // 2))
    side_votes = max(0, exit_path_votes - top_votes)
    cameras = {
        "top": {
            "raw": max(raw - (raw // 3), top_votes),
            "raw_max_confidence": top_confidence,
            "threshold_passed": max(0, threshold_passed - (threshold_passed // 2)),
            "threshold_passed_max_confidence": top_confidence,
        }
    }
    if top_votes > 0:
        cameras["top"].update(
            {
                "freezer_roi_passed": top_votes,
                "freezerExitPathVotes": top_votes,
                "freezer_roi_passed_max_confidence": top_confidence,
            }
        )
    if side_confidence > 0.0:
        cameras["side"] = {
            "raw": max(raw // 3, side_votes),
            "raw_max_confidence": side_confidence,
            "threshold_passed": threshold_passed // 2,
            "threshold_passed_max_confidence": side_confidence,
        }
        if side_votes > 0:
            cameras["side"].update(
                {
                    "freezer_roi_passed": side_votes,
                    "freezerExitPathVotes": side_votes,
                    "freezer_roi_passed_max_confidence": side_confidence,
                }
            )

    entry = {
        "class_id": class_id,
        "name": name,
        "raw": raw,
        "raw_max_confidence": confidence,
        "threshold_passed": threshold_passed,
        "threshold_passed_max_confidence": confidence,
        "freezer_roi_passed": exit_path_votes,
        "freezerExitPathVotes": exit_path_votes,
        "freezer_roi_passed_max_confidence": confidence,
        "trajectoryExitPathPassed": trajectory_passed,
        "staticShelfLikely": static_shelf,
        "handPathValid": hand_path_passed or hand_path_blocked,
        "handPathPassed": hand_path_passed,
        "handPathBlocked": hand_path_blocked,
        "handInteractionPassed": hand_path_passed,
        "handPathValidUpperRoi": hand_path_passed or hand_path_blocked,
        "cameras": cameras,
    }
    if path_displacement is not None:
        entry.update(
            {
                "pathDisplacementPx": path_displacement,
                "maxDistancePx": path_displacement,
                "motionThresholdPx": motion_threshold,
                "motion_passed": threshold_passed if trajectory_passed else 0,
            }
        )
    return entry


def test_vision_only_without_product_db_returns_partial():
    engine = ProductDecisionEngine(product_db=None, strict_mode=True)
    candidate = make_candidate(confidence=0.95)

    result = engine.judge(
        vision_candidates=[candidate],
        delta_weight=0.0,
        vision_only=True,
    )

    assert result.status == JudgmentStatus.PARTIAL
    assert result.products[0].product_id == 26
    assert result.products[0].name == candidate.class_name


def test_loadcell_only_skips_boolean_no_loadcell_products():
    engine = ProductDecisionEngine(strict_mode=True)
    active_product = make_active_product()
    active_product.has_loadcell = False

    result = engine.judge_by_weight_only(
        delta_weight=-365.0,
        active_products=[active_product],
    )

    assert result.status == JudgmentStatus.NO_DETECTION
    assert result.products == []


def test_loadcell_only_does_not_return_closest_match_outside_tolerance():
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge_by_weight_only(
        delta_weight=-7.6,
        active_products=[
            make_active_product(
                class_id=113,
                name="STICK_INNON_CONDITION_STICK_18G",
                weight=19.0,
                stock=1,
            )
        ],
    )

    assert result.status in {JudgmentStatus.UNCERTAIN, JudgmentStatus.NO_DETECTION}
    assert result.products == []


def test_vision_first_no_vision_loadcell_match_does_not_create_identity():
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[],
        delta_weight=-365.0,
        active_products=[make_active_product(weight=365.0)],
    )

    assert result.status == JudgmentStatus.NO_DETECTION
    assert result.products == []


def test_vision_first_weight_conflict_preserves_strong_vision_identity_as_partial():
    class Trace:
        def __init__(self):
            self.weight_diagnostics = {}
            self.stage_counts_by_class = {}

        def record_weight_diagnostics(self, diagnostics):
            self.weight_diagnostics.update(diagnostics)

    trace = Trace()
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[make_candidate(class_id=26, name="Vision Product", confidence=0.94)],
        delta_weight=-430.0,
        active_products=[
            make_active_product(26, "Vision Product", weight=365.0, stock=5),
            make_active_product(99, "Weight Nearest Active", weight=430.0, stock=5),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.PARTIAL
    assert [(product.product_id, product.count) for product in result.products] == [(26, 1)]
    assert result.weight_explained == 365.0
    assert result.weight_residual == 65.0
    diagnostics = trace.weight_diagnostics["vision_first_identity_validation"]
    assert diagnostics["accepted"] is True
    assert diagnostics["weight_validation_passed"] is False
    assert diagnostics["reason"] == "vision_identity_preserved_weight_mismatch"
    assert diagnostics["selected"]["class_id"] == 26


def test_vision_first_missing_weight_preserves_strong_vision_identity_as_partial():
    class Trace:
        def __init__(self):
            self.weight_diagnostics = {}
            self.stage_counts_by_class = {}

        def record_weight_diagnostics(self, diagnostics):
            self.weight_diagnostics.update(diagnostics)

    trace = Trace()
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(class_id=26, name="Vision Product", confidence=0.94)
        ],
        delta_weight=-430.0,
        active_products=[
            make_active_product(26, "Vision Product", weight=0.0, stock=5),
            make_active_product(99, "Weight Nearest Active", weight=430.0, stock=5),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.PARTIAL
    assert [(product.product_id, product.count) for product in result.products] == [(26, 1)]
    assert result.weight_explained == 0.0
    assert result.weight_residual == 430.0
    diagnostics = trace.weight_diagnostics["vision_first_identity_validation"]
    assert diagnostics["accepted"] is True
    assert diagnostics["weight_validation_passed"] is False
    assert diagnostics["reason"] == "vision_identity_preserved_weight_unavailable"
    assert diagnostics["selected"]["weight_status"] == "unavailable"


def test_freezer_vision_first_accepts_candidate_outside_refrigerated_tolerance(
    monkeypatch,
):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    trace = FakeLoadcellTrace({})
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(class_id=26, name="FREEZER_PRODUCT_A", confidence=0.94)
        ],
        delta_weight=-375.0,
        active_products=[
            make_active_product(26, "FREEZER_PRODUCT_A", weight=365.0, stock=5),
            make_active_product(99, "WEIGHT_NEAREST_ONLY", weight=375.0, stock=5),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [(26, 1)]
    assert result.weight_explained == 365.0
    assert result.weight_residual == 10.0
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["accepted"] is True
    assert diagnostics["weight_used_as"] == "combination_validation"
    assert diagnostics["weight_reliable"] is True
    assert diagnostics["freezer_weight_tolerance"] == 15.0


def test_freezer_vision_first_keeps_rank_order_when_first_candidate_fits(monkeypatch):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.weight, "freezer_multi_min_confidence", 0.99)
    trace = FakeLoadcellTrace({})
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(class_id=1, name="HIGHER_CONF", confidence=0.95),
            make_candidate(class_id=2, name="CLOSER_WEIGHT", confidence=0.90),
        ],
        delta_weight=-112.0,
        active_products=[
            make_active_product(1, "HIGHER_CONF", weight=100.0, stock=5),
            make_active_product(2, "CLOSER_WEIGHT", weight=112.0, stock=5),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [(1, 1)]
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["reason"] == "freezer_ordered_vision_candidate_pool"
    assert diagnostics["selected"][0]["name"] == "HIGHER_CONF"
    assert diagnostics["orderedCombinationSearch"]["attempts"][0]["classIds"] == [1]


def test_freezer_vision_first_does_not_pick_non_candidate_weight_match(monkeypatch):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    trace = FakeLoadcellTrace({})
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(class_id=1, name="VISION_ONLY_CANDIDATE", confidence=0.94)
        ],
        delta_weight=-200.0,
        active_products=[
            make_active_product(1, "VISION_ONLY_CANDIDATE", weight=100.0, stock=5),
            make_active_product(2, "NON_CANDIDATE_WEIGHT_MATCH", weight=200.0, stock=5),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [(1, 2)]
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["selected"][0]["class_id"] == 1
    assert diagnostics["selected"][0]["name"] == "VISION_ONLY_CANDIDATE"
    assert diagnostics["selected"][0]["count"] == 2
    assert diagnostics["orderedCombinationSearch"]["attempts"][1]["counts"] == [2]


def test_freezer_vision_first_outputs_multi_kind_when_vision_is_strong(monkeypatch):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    trace = FakeLoadcellTrace({})
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(class_id=1, name="FREEZER_A", confidence=0.92),
            make_candidate(class_id=2, name="FREEZER_B", confidence=0.88),
        ],
        delta_weight=-210.0,
        active_products=[
            make_active_product(1, "FREEZER_A", weight=80.0, stock=5),
            make_active_product(2, "FREEZER_B", weight=130.0, stock=5),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (1, 1),
        (2, 1),
    ]
    assert trace.weight_diagnostics["freezer_vision_first"]["reason"] == (
        "freezer_ordered_vision_candidate_pool"
    )


def test_freezer_vision_first_rejects_multi_kind_without_weight_fit(
    monkeypatch,
):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.weight, "freezer_multi_min_confidence", 0.45)
    trace = FakeLoadcellTrace({})
    trace.stage_counts_by_class = {
        "1": {
            "class_id": 1,
            "name": "FREEZER_A",
            "freezerExitPathVotes": 7,
            "freezer_roi_filtered_max_confidence": 0.93,
            "cameras": {
                "top": {"freezerExitPathVotes": 4},
                "side": {"freezerExitPathVotes": 3},
            },
        },
        "2": {
            "class_id": 2,
            "name": "FREEZER_B",
            "freezerExitPathVotes": 6,
            "freezer_roi_filtered_max_confidence": 0.89,
            "cameras": {
                "top": {"freezerExitPathVotes": 3},
                "side": {"freezerExitPathVotes": 3},
            },
        },
    }
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(class_id=1, name="FREEZER_A", confidence=0.93),
            make_candidate(class_id=2, name="FREEZER_B", confidence=0.89),
        ],
        delta_weight=-999.0,
        active_products=[
            make_active_product(1, "FREEZER_A", weight=100.0, stock=5),
            make_active_product(2, "FREEZER_B", weight=110.0, stock=5),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.UNCERTAIN
    assert result.products == []
    assert result.weight_explained == 0.0
    assert result.weight_residual == 999.0
    assert trace.weight_diagnostics["freezer_vision_first"]["reason"] == (
        "no_weight_fit_for_vision_candidate_pool"
    )
    search = trace.weight_diagnostics["freezer_vision_first"][
        "orderedCombinationSearch"
    ]
    assert search["accepted"] is False
    assert search["reason"] == "no_weight_fit_for_vision_candidate_pool"


def test_freezer_vision_first_selects_single_178g_candidate_not_top_three(
    monkeypatch,
):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.weight, "freezer_multi_min_confidence", 0.45)
    trace = FakeLoadcellTrace({})
    trace.stage_counts_by_class = {
        str(class_id): {
            "class_id": class_id,
            "name": name,
            "freezerExitPathVotes": 6,
            "freezer_roi_filtered_max_confidence": confidence,
            "cameras": {
                "top": {"freezerExitPathVotes": 3},
                "side": {"freezerExitPathVotes": 3},
            },
        }
        for class_id, name, confidence in [
            (101, "FREEZER_172G", 0.93),
            (102, "FREEZER_170G", 0.91),
            (103, "FREEZER_166G", 0.89),
        ]
    }
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(class_id=101, name="FREEZER_172G", confidence=0.93),
            make_candidate(class_id=102, name="FREEZER_170G", confidence=0.91),
            make_candidate(class_id=103, name="FREEZER_166G", confidence=0.89),
        ],
        delta_weight=-178.0,
        active_products=[
            make_active_product(101, "FREEZER_172G", weight=172.0, stock=10),
            make_active_product(102, "FREEZER_170G", weight=170.0, stock=10),
            make_active_product(103, "FREEZER_166G", weight=166.0, stock=10),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (101, 1)
    ]
    assert result.weight_explained == 172.0
    assert result.weight_residual == 6.0
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["reason"] == "freezer_ordered_vision_candidate_pool"
    assert diagnostics["orderedCombinationSearch"]["attempts"][0]["classIds"] == [101]


def test_freezer_vision_first_prefers_mixed_dumplings_over_baskin_repeat(
    monkeypatch,
):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")
    monkeypatch.setattr(config.vision, "top_confidence_threshold", 0.70)
    monkeypatch.setattr(config.vision, "side_confidence_threshold", 0.70)
    monkeypatch.setattr(config.vision, "freezer_min_vote_count", 3)
    trace = FakeLoadcellTrace({"removal_segment_targets": [224.0, 189.0]})
    trace.stage_counts_by_class = {
        "3": {
            "class_id": 3,
            "name": "BAG_BIBIGO_CHEONGYANG_MEAT_DUMPLINGS_200G",
            "freezerExitPathVotes": 15,
            "pathDisplacementPx": 214.6,
            "motionThresholdPx": 12.0,
            "trajectoryExitPathPassed": True,
            "cameras": {
                "top": {"freezerExitPathVotes": 6},
                "side": {"freezerExitPathVotes": 9},
            },
        },
        "40": {
            "class_id": 40,
            "name": "CUP_BASKIN_CHERRIES_JUBILEE_170ML",
            "freezerExitPathVotes": 8,
            "pathDisplacementPx": 33.6,
            "motionThresholdPx": 12.0,
            "trajectoryExitPathPassed": True,
            "cameras": {"top": {"freezerExitPathVotes": 8}},
        },
        "13": {
            "class_id": 13,
            "name": "BAG_COOZROCK_JUICY_MEAT_DUMPLING_168G",
            "freezerExitPathVotes": 10,
            "pathDisplacementPx": 109.3,
            "motionThresholdPx": 12.0,
            "trajectoryExitPathPassed": True,
            "cameras": {
                "top": {"freezerExitPathVotes": 6},
                "side": {"freezerExitPathVotes": 4},
            },
        },
    }
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_freezer_candidate(
                class_id=3,
                name="BAG_BIBIGO_CHEONGYANG_MEAT_DUMPLINGS_200G",
                combined=1.0,
                top=0.8520,
                side=0.8524,
                raw_vote_count=15,
            ),
            make_freezer_candidate(
                class_id=40,
                name="CUP_BASKIN_CHERRIES_JUBILEE_170ML",
                combined=0.5471,
                top=0.9118,
                raw_vote_count=8,
            ),
            make_freezer_candidate(
                class_id=13,
                name="BAG_COOZROCK_JUICY_MEAT_DUMPLING_168G",
                combined=0.4664,
                top=0.7773,
                raw_vote_count=10,
            ),
        ],
        delta_weight=-407.4,
        active_products=[
            make_active_product(
                3,
                "BAG_BIBIGO_CHEONGYANG_MEAT_DUMPLINGS_200G",
                weight=224.0,
                stock=64,
            ),
            make_active_product(
                40,
                "CUP_BASKIN_CHERRIES_JUBILEE_170ML",
                weight=131.0,
                stock=30,
            ),
            make_active_product(
                13,
                "BAG_COOZROCK_JUICY_MEAT_DUMPLING_168G",
                weight=189.0,
                stock=30,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (3, 1),
        (13, 1),
    ]
    assert result.weight_explained == 413.0
    assert result.weight_residual == pytest.approx(5.6)
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["orderedCombinationSearch"]["selectedOrder"] > 0
    assert any(
        attempt["classIds"] == [40] and attempt["counts"] == [3]
        for attempt in diagnostics["orderedCombinationSearch"]["attempts"]
    )


def test_freezer_vision_first_prefers_all_single_mixed_over_exact_repeat(
    monkeypatch,
):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")
    trace = FakeLoadcellTrace({})
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_freezer_candidate(
                class_id=101,
                name="FREEZER_A",
                combined=0.95,
                top=0.95,
            ),
            make_freezer_candidate(
                class_id=102,
                name="FREEZER_B",
                combined=0.93,
                top=0.93,
            ),
            make_freezer_candidate(
                class_id=103,
                name="FREEZER_C",
                combined=0.91,
                top=0.91,
            ),
        ],
        delta_weight=-300.0,
        active_products=[
            make_active_product(
                101,
                "FREEZER_A",
                weight=100.0,
                stock=10,
                product_idx="P101",
            ),
            make_active_product(
                102,
                "FREEZER_B",
                weight=102.0,
                stock=10,
                product_idx="P102",
            ),
            make_active_product(
                103,
                "FREEZER_C",
                weight=103.0,
                stock=10,
                product_idx="P103",
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (101, 1),
        (102, 1),
        (103, 1),
    ]
    assert result.weight_explained == 305.0
    assert result.weight_residual == 5.0
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["distinctMixedPreferred"] is True
    assert any(
        attempt["classIds"] == [101] and attempt["counts"] == [3]
        for attempt in diagnostics["orderedCombinationSearch"]["attempts"]
    )


def test_freezer_vision_first_keeps_repeat_when_no_mixed_candidate(
    monkeypatch,
):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")
    trace = FakeLoadcellTrace({})
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_freezer_candidate(
                class_id=101,
                name="FREEZER_A",
                combined=0.95,
                top=0.95,
            )
        ],
        delta_weight=-300.0,
        active_products=[
            make_active_product(
                101,
                "FREEZER_A",
                weight=100.0,
                stock=10,
                product_idx="P101",
            )
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (101, 3)
    ]
    assert trace.weight_diagnostics["freezer_vision_first"][
        "distinctMixedPreferred"
    ] is False


def test_freezer_vision_first_keeps_repeat_when_mixed_is_too_much_worse(
    monkeypatch,
):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")
    trace = FakeLoadcellTrace({})
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_freezer_candidate(
                class_id=101,
                name="FREEZER_A",
                combined=0.95,
                top=0.95,
            ),
            make_freezer_candidate(
                class_id=102,
                name="FREEZER_B",
                combined=0.93,
                top=0.93,
            ),
            make_freezer_candidate(
                class_id=103,
                name="FREEZER_C",
                combined=0.91,
                top=0.91,
            ),
        ],
        delta_weight=-300.0,
        active_products=[
            make_active_product(
                101,
                "FREEZER_A",
                weight=100.0,
                stock=10,
                product_idx="P101",
            ),
            make_active_product(
                102,
                "FREEZER_B",
                weight=107.0,
                stock=10,
                product_idx="P102",
            ),
            make_active_product(
                103,
                "FREEZER_C",
                weight=108.0,
                stock=10,
                product_idx="P103",
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (101, 3)
    ]
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["distinctMixedPreferred"] is False


def test_freezer_vision_first_prior_trigger_dedupe_selects_next_candidate(
    monkeypatch,
):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")
    trace = FakeLoadcellTrace({})
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_freezer_candidate(
                class_id=101,
                name="FREEZER_A",
                combined=0.95,
                top=0.95,
            ),
            make_freezer_candidate(
                class_id=102,
                name="FREEZER_B",
                combined=0.93,
                top=0.93,
            ),
        ],
        delta_weight=-100.0,
        active_products=[
            make_active_product(
                101,
                "FREEZER_A",
                weight=100.0,
                stock=10,
                product_idx="P101",
            ),
            make_active_product(
                102,
                "FREEZER_B",
                weight=100.0,
                stock=10,
                product_idx="P102",
            ),
        ],
        trace_context=trace,
        prior_selected_product_idxs={"P101"},
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (102, 1)
    ]
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["priorExclusionApplied"] is True
    assert diagnostics["priorExclusionFallback"] is False


def test_freezer_vision_first_prior_trigger_dedupe_advances_to_third_candidate(
    monkeypatch,
):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")
    trace = FakeLoadcellTrace({})
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_freezer_candidate(
                class_id=101,
                name="FREEZER_A",
                combined=0.95,
                top=0.95,
            ),
            make_freezer_candidate(
                class_id=102,
                name="FREEZER_B",
                combined=0.93,
                top=0.93,
            ),
            make_freezer_candidate(
                class_id=103,
                name="FREEZER_C",
                combined=0.91,
                top=0.91,
            ),
        ],
        delta_weight=-100.0,
        active_products=[
            make_active_product(
                101,
                "FREEZER_A",
                weight=100.0,
                stock=10,
                product_idx="P101",
            ),
            make_active_product(
                102,
                "FREEZER_B",
                weight=100.0,
                stock=10,
                product_idx="P102",
            ),
            make_active_product(
                103,
                "FREEZER_C",
                weight=100.0,
                stock=10,
                product_idx="P103",
            ),
        ],
        trace_context=trace,
        prior_selected_product_idxs={"P101", "P102"},
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (103, 1)
    ]
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["priorExclusionApplied"] is True
    assert diagnostics["priorExclusionFallback"] is False


def test_freezer_channel_targets_lock_single_then_solve_remaining_repeat(
    monkeypatch,
):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")
    trace = FakeLoadcellTrace(
        {
            "channel_removal_segment_targets": [
                {
                    "source": "simultaneous_channel_delta",
                    "weight": 100.0,
                    "delta": -100.0,
                    "segment_index": 0,
                    "channel_index": 0,
                    "channel_position": 0,
                    "channel_side": "left",
                    "evidence_required": True,
                },
                {
                    "source": "simultaneous_channel_delta",
                    "weight": 120.0,
                    "delta": -120.0,
                    "segment_index": 1,
                    "channel_index": 1,
                    "channel_position": 1,
                    "channel_side": "right",
                    "evidence_required": True,
                },
            ]
        }
    )
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_freezer_candidate(
                class_id=101,
                name="FREEZER_LEFT_50G",
                combined=0.95,
                top=0.95,
            ),
            make_freezer_candidate(
                class_id=102,
                name="FREEZER_RIGHT_120G",
                combined=0.93,
                top=0.93,
            ),
        ],
        delta_weight=-220.0,
        active_products=[
            make_active_product(
                101,
                "FREEZER_LEFT_50G",
                weight=50.0,
                stock=10,
                product_idx="P101",
            ),
            make_active_product(
                102,
                "FREEZER_RIGHT_120G",
                weight=120.0,
                stock=10,
                product_idx="P102",
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (101, 2),
        (102, 1),
    ]
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["reason"] == "freezer_channel_target_product_groups"
    search = diagnostics["orderedCombinationSearch"]
    assert search["policy"] == "loadcell_channel_product_group_ordered_weight_validation"
    assert search["accepted"] is True
    selected_by_side = {
        selected["channelSide"]: selected for selected in diagnostics["selected"]
    }
    assert selected_by_side["right"]["class_id"] == 102
    assert selected_by_side["right"]["count"] == 1
    assert selected_by_side["left"]["class_id"] == 101
    assert selected_by_side["left"]["count"] == 2


def test_freezer_single_channel_target_solves_side_repeat(monkeypatch):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")
    trace = FakeLoadcellTrace(
        {
            "channel_movement_targets": [
                {
                    "source": "stable_channel_delta",
                    "direction": "removal",
                    "weight": 100.0,
                    "delta": -100.0,
                    "segment_index": 0,
                    "channel_index": 0,
                    "channel_position": 0,
                    "channel_side": "left",
                }
            ]
        }
    )
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_freezer_candidate(
                class_id=101,
                name="FREEZER_LEFT_50G",
                combined=0.95,
                top=0.95,
            ),
            make_freezer_candidate(
                class_id=102,
                name="FREEZER_RIGHT_120G",
                combined=0.93,
                top=0.93,
            ),
        ],
        delta_weight=-100.0,
        active_products=[
            make_active_product(
                101,
                "FREEZER_LEFT_50G",
                weight=50.0,
                stock=10,
                product_idx="P101",
            ),
            make_active_product(
                102,
                "FREEZER_RIGHT_120G",
                weight=120.0,
                stock=10,
                product_idx="P102",
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (101, 2)
    ]
    assert [
        unit["channelSide"]
        for unit in result.products[0].placement_units
    ] == ["left", "left"]
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["reason"] == "freezer_channel_target_product_groups"
    search = diagnostics["orderedCombinationSearch"]
    assert search["accepted"] is True
    assert search["channelTargets"][0]["channelSide"] == "left"


def test_freezer_vision_first_prior_trigger_dedupe_fails_closed(
    monkeypatch,
):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")
    trace = FakeLoadcellTrace({})
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_freezer_candidate(
                class_id=101,
                name="FREEZER_A",
                combined=0.95,
                top=0.95,
            )
        ],
        delta_weight=-100.0,
        active_products=[
            make_active_product(
                101,
                "FREEZER_A",
                weight=100.0,
                stock=10,
                product_idx="P101",
            )
        ],
        trace_context=trace,
        prior_selected_product_idxs={"P101"},
    )

    assert result.status == JudgmentStatus.UNCERTAIN
    assert result.products == []
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["reason"] == "no_weight_fit_for_vision_candidate_pool"
    search = diagnostics["orderedCombinationSearch"]
    assert search["reason"] == "no_candidates_after_prior_trigger_exclusion"
    assert search["priorExclusionApplied"] is True
    assert search["priorExclusionFallback"] is False


def test_freezer_same_position_prior_repeat_overrides_prior_dedupe(monkeypatch):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")
    trace = FakeLoadcellTrace(
        {
            "channel_movement_targets": [
                {
                    "source": "stable_channel_delta",
                    "direction": "removal",
                    "weight": 146.5,
                    "delta": -146.5,
                    "segment_index": 0,
                    "channel_index": 1,
                    "channel_position": 1,
                    "channel_side": "right",
                }
            ]
        }
    )
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_freezer_candidate(
                class_id=44,
                name="STICK_BINGGRAE_MELONA_75ML",
                combined=0.96,
                top=0.96,
            ),
            make_freezer_candidate(
                class_id=27,
                name="BAG_NULLDAM_BAGEL_140G",
                combined=0.92,
                top=0.92,
            ),
        ],
        delta_weight=-146.5,
        active_products=[
            make_active_product(
                44,
                "STICK_BINGGRAE_MELONA_75ML",
                weight=75.0,
                stock=10,
                product_idx="P_MELONA",
            ),
            make_active_product(
                27,
                "BAG_NULLDAM_BAGEL_140G",
                weight=156.0,
                stock=10,
                product_idx="P_BAGEL",
            ),
        ],
        trace_context=trace,
        prior_selected_product_idxs={"P_BAGEL", "id:27"},
        prior_selected_position_product_idxs={
            "right|1|1": {"P_BAGEL", "id:27"},
        },
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (27, 1)
    ]
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    search = diagnostics["orderedCombinationSearch"]
    assert search["priorExclusionApplied"] is True
    assert search["samePositionRepeatApplied"] is True
    assert "P_BAGEL" in search["samePositionRepeatProductIdxs"]
    assert diagnostics["selected"][0]["samePositionRepeatApplied"] is True
    assert diagnostics["selected"][0]["channelSide"] == "right"


def test_freezer_vision_first_zone2_still_selects_baskin_single(monkeypatch):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")
    monkeypatch.setattr(config.vision, "top_confidence_threshold", 0.70)
    monkeypatch.setattr(config.vision, "side_confidence_threshold", 0.70)
    trace = FakeLoadcellTrace({})
    trace.stage_counts_by_class = {
        "40": {
            "class_id": 40,
            "name": "CUP_BASKIN_CHERRIES_JUBILEE_170ML",
            "freezerExitPathVotes": 10,
            "pathDisplacementPx": 259.1,
            "motionThresholdPx": 12.0,
            "trajectoryExitPathPassed": True,
            "cameras": {
                "top": {"freezerExitPathVotes": 4},
                "side": {"freezerExitPathVotes": 6},
            },
        }
    }
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_freezer_candidate(
                class_id=40,
                name="CUP_BASKIN_CHERRIES_JUBILEE_170ML",
                combined=1.0,
                top=0.9005,
                side=0.9404,
                raw_vote_count=10,
            )
        ],
        delta_weight=-129.9,
        active_products=[
            make_active_product(
                40,
                "CUP_BASKIN_CHERRIES_JUBILEE_170ML",
                weight=131.0,
                stock=27,
                price=4000,
            )
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (40, 1)
    ]
    assert result.weight_residual == pytest.approx(1.1)


def test_freezer_vision_first_roi_weight_rescues_worldcon_over_static_melona(
    monkeypatch,
):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")
    monkeypatch.setattr(config.vision, "top_confidence_threshold", 0.70)
    monkeypatch.setattr(config.vision, "side_confidence_threshold", 0.70)
    monkeypatch.setattr(config.vision, "freezer_min_vote_count", 3)
    trace = FakeLoadcellTrace({})
    trace.stage_counts_by_class = {
        "40": {
            "class_id": 40,
            "name": "CUP_BASKIN_CHERRIES_JUBILEE_170ML",
            "freezerExitPathVotes": 8,
            "pathDisplacementPx": 207.2,
            "motionThresholdPx": 12.0,
            "trajectoryExitPathPassed": True,
            "cameras": {
                "top": {"freezerExitPathVotes": 4},
                "side": {"freezerExitPathVotes": 4},
            },
        },
        "44": {
            "class_id": 44,
            "name": "STICK_BINGGRAE_MELONA_75ML",
            "freezerExitPathVotes": 3,
            "pathDisplacementPx": 3.2,
            "motionThresholdPx": 12.0,
            "trajectoryExitPathPassed": False,
            "staticShelfLikely": True,
            "cameras": {"top": {"freezerExitPathVotes": 3}},
        },
        "35": {
            "class_id": 35,
            "name": "BOX_LOTTE_WORLDCON_160ML",
            "threshold_passed": 25,
            "freezer_roi_filtered": 25,
            "freezer_roi_filtered_max_confidence": 0.7905,
            "cameras": {
                "top": {
                    "threshold_passed": 25,
                    "freezer_roi_filtered": 25,
                    "freezerRoiFilteredVotes": 25,
                    "freezer_roi_filtered_max_confidence": 0.7905,
                },
                "side": {"raw": 43, "raw_max_confidence": 0.6584},
            },
        },
    }
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_freezer_candidate(
                class_id=40,
                name="CUP_BASKIN_CHERRIES_JUBILEE_170ML",
                combined=1.0,
                top=0.9005,
                side=0.9320,
                raw_vote_count=8,
            ),
            make_freezer_candidate(
                class_id=44,
                name="STICK_BINGGRAE_MELONA_75ML",
                combined=0.4521,
                top=0.7535,
                raw_vote_count=3,
            ),
        ],
        delta_weight=-70.6,
        active_products=[
            make_active_product(
                40,
                "CUP_BASKIN_CHERRIES_JUBILEE_170ML",
                weight=131.0,
                stock=27,
            ),
            make_active_product(
                44,
                "STICK_BINGGRAE_MELONA_75ML",
                weight=79.0,
                stock=46,
            ),
            make_active_product(
                35,
                "BOX_LOTTE_WORLDCON_160ML",
                weight=70.0,
                stock=42,
                price=1400,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (35, 1)
    ]
    assert result.weight_residual == pytest.approx(0.6)
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    selected = diagnostics["selected"][0]
    assert selected["source"] == "freezer_roi_weight_rescue"
    assert selected["weight_residual"] == pytest.approx(0.6)
    rejected = diagnostics["rejectedInteractionCandidates"]
    assert rejected[0]["class_id"] == 44
    assert rejected[0]["interactionRejectedReason"] == (
        "static_low_vote_shelf_candidate"
    )


def test_freezer_channel_target_stage_rescue_selects_lala_without_final_candidates(
    monkeypatch,
):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")
    monkeypatch.setattr(config.vision, "top_confidence_threshold", 0.50)
    monkeypatch.setattr(config.vision, "side_confidence_threshold", 0.50)
    monkeypatch.setattr(config.vision, "freezer_min_vote_count", 4)
    monkeypatch.setattr(config.vision, "freezer_min_exit_path_votes", 3)
    trace = make_freezer_channel_trace(weight=74.8)
    trace.stage_counts_by_class = {
        "46": make_freezer_channel_stage_evidence(
            class_id=46,
            name="STICK_LALA_SWEET_GRAPE_ZERO_70ML",
            raw=49,
            confidence=0.697,
            top_confidence=0.697,
            side_confidence=0.6229,
            threshold_passed=4,
            exit_path_votes=2,
            hand_path_passed=True,
            trajectory_passed=True,
            path_displacement=137.4,
            motion_threshold=14.2,
        )
    }
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[],
        delta_weight=-71.5,
        active_products=[
            make_active_product(
                46,
                "STICK_LALA_SWEET_GRAPE_ZERO_70ML",
                weight=71.0,
                stock=82,
                price=500,
            ),
            make_active_product(
                35,
                "BOX_LOTTE_WORLDCON_160ML",
                weight=70.0,
                stock=41,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (46, 1)
    ]
    assert result.weight_residual == pytest.approx(0.5)
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    selected = diagnostics["selected"][0]
    assert selected["source"] == "freezer_channel_weight_stage_rescue"
    assert selected["relaxedExitPathPassed"] is True
    assert diagnostics["channelWeightStageRescueCandidates"][0]["class_id"] == 46


def test_freezer_channel_target_stage_rescue_prefers_dumpling_over_hotdog_candidate(
    monkeypatch,
):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")
    monkeypatch.setattr(config.vision, "top_confidence_threshold", 0.50)
    monkeypatch.setattr(config.vision, "side_confidence_threshold", 0.50)
    monkeypatch.setattr(config.vision, "freezer_min_vote_count", 4)
    monkeypatch.setattr(config.vision, "freezer_min_exit_path_votes", 3)
    trace = make_freezer_channel_trace(weight=182.3)
    trace.stage_counts_by_class = {
        "13": make_freezer_channel_stage_evidence(
            class_id=13,
            name="BAG_COOZROCK_JUICY_MEAT_DUMPLING_168G",
            raw=188,
            confidence=0.8269,
            top_confidence=0.6793,
            side_confidence=0.8269,
            threshold_passed=3,
            exit_path_votes=3,
            hand_path_passed=True,
            trajectory_passed=True,
            path_displacement=60.0,
            motion_threshold=36.2,
        ),
        "23": make_freezer_channel_stage_evidence(
            class_id=23,
            name="BAG_HANMAC_TRIPLE_CHEESE_HAMBURGER_155G",
            raw=16,
            confidence=0.2391,
            top_confidence=0.0742,
            side_confidence=0.2391,
            threshold_passed=0,
            exit_path_votes=0,
            hand_path_passed=False,
            trajectory_passed=False,
            path_displacement=None,
        ),
        "24": make_freezer_channel_stage_evidence(
            class_id=24,
            name="BAG_JACKSONVILLE_BIG_HOT_DOG_115G",
            raw=252,
            confidence=0.7616,
            top_confidence=0.6182,
            side_confidence=0.7616,
            threshold_passed=11,
            exit_path_votes=11,
            hand_path_passed=True,
            trajectory_passed=True,
            path_displacement=120.8,
        ),
    }
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_freezer_candidate(
                class_id=24,
                name="BAG_JACKSONVILLE_BIG_HOT_DOG_115G",
                combined=0.7992,
                top=0.6182,
                side=0.7616,
                raw_vote_count=11,
            )
        ],
        delta_weight=-178.9,
        active_products=[
            make_active_product(
                13,
                "BAG_COOZROCK_JUICY_MEAT_DUMPLING_168G",
                weight=189.0,
                stock=70,
                price=2100,
            ),
            make_active_product(
                23,
                "BAG_HANMAC_TRIPLE_CHEESE_HAMBURGER_155G",
                weight=176.0,
                stock=84,
            ),
            make_active_product(
                24,
                "BAG_JACKSONVILLE_BIG_HOT_DOG_115G",
                weight=165.0,
                stock=73,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (13, 1)
    ]
    assert result.weight_residual == pytest.approx(10.1)
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    selected = diagnostics["selected"][0]
    assert selected["source"] == "freezer_channel_weight_stage_rescue"
    assert selected["channelTargetWeight"] == pytest.approx(182.3)
    rejected = diagnostics["rejectedChannelWeightStageRescueCandidates"]
    assert any(
        item["class_id"] == 23
        and item["reason"] == "identity_confidence_below_threshold"
        for item in rejected
    )


def test_freezer_channel_target_stage_rescue_rejects_weight_only_stage_match(
    monkeypatch,
):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")
    monkeypatch.setattr(config.vision, "top_confidence_threshold", 0.50)
    monkeypatch.setattr(config.vision, "freezer_min_exit_path_votes", 3)
    trace = make_freezer_channel_trace(weight=74.8)
    trace.stage_counts_by_class = {
        "46": make_freezer_channel_stage_evidence(
            class_id=46,
            name="STICK_LALA_SWEET_GRAPE_ZERO_70ML",
            raw=20,
            confidence=0.90,
            top_confidence=0.90,
            threshold_passed=4,
            exit_path_votes=0,
            hand_path_passed=False,
            trajectory_passed=False,
            path_displacement=None,
        )
    }
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[],
        delta_weight=-71.5,
        active_products=[
            make_active_product(
                46,
                "STICK_LALA_SWEET_GRAPE_ZERO_70ML",
                weight=71.0,
                stock=82,
            )
        ],
        trace_context=trace,
    )

    assert result.products == []
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    rejected = diagnostics["rejectedChannelWeightStageRescueCandidates"]
    assert rejected[0]["reason"] == "insufficient_exit_path_evidence"


def test_freezer_channel_target_stage_rescue_rejects_blocked_or_static_stage(
    monkeypatch,
):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")
    monkeypatch.setattr(config.vision, "top_confidence_threshold", 0.50)
    monkeypatch.setattr(config.vision, "freezer_min_exit_path_votes", 3)
    trace = make_freezer_channel_trace(weight=74.8)
    trace.stage_counts_by_class = {
        "46": make_freezer_channel_stage_evidence(
            class_id=46,
            name="STICK_LALA_SWEET_GRAPE_ZERO_70ML",
            raw=20,
            confidence=0.90,
            top_confidence=0.90,
            threshold_passed=4,
            exit_path_votes=3,
            hand_path_passed=False,
            hand_path_blocked=True,
            trajectory_passed=True,
        ),
        "35": make_freezer_channel_stage_evidence(
            class_id=35,
            name="BOX_LOTTE_WORLDCON_160ML",
            raw=20,
            confidence=0.88,
            top_confidence=0.88,
            threshold_passed=4,
            exit_path_votes=3,
            hand_path_passed=True,
            trajectory_passed=False,
            static_shelf=True,
            path_displacement=1.0,
        ),
    }
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[],
        delta_weight=-71.5,
        active_products=[
            make_active_product(
                46,
                "STICK_LALA_SWEET_GRAPE_ZERO_70ML",
                weight=71.0,
                stock=82,
            ),
            make_active_product(
                35,
                "BOX_LOTTE_WORLDCON_160ML",
                weight=70.0,
                stock=41,
            ),
        ],
        trace_context=trace,
    )

    assert result.products == []
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    rejected_reasons = {
        item["class_id"]: item["reason"]
        for item in diagnostics["rejectedChannelWeightStageRescueCandidates"]
    }
    assert rejected_reasons[46] == "hand_path_blocked"
    assert rejected_reasons[35] == "static_shelf_likely"


def test_freezer_vision_first_multi_without_weight_flag_false_uses_single_path(
    monkeypatch,
):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(
        config.weight,
        "freezer_vision_multi_without_weight_enabled",
        False,
    )
    trace = FakeLoadcellTrace({})
    trace.stage_counts_by_class = {
        "1": {
            "class_id": 1,
            "name": "FREEZER_A",
            "freezerExitPathVotes": 7,
            "freezer_roi_filtered_max_confidence": 0.93,
            "cameras": {
                "top": {"freezerExitPathVotes": 4},
                "side": {"freezerExitPathVotes": 3},
            },
        },
        "2": {
            "class_id": 2,
            "name": "FREEZER_B",
            "freezerExitPathVotes": 6,
            "freezer_roi_filtered_max_confidence": 0.89,
            "cameras": {
                "top": {"freezerExitPathVotes": 3},
                "side": {"freezerExitPathVotes": 3},
            },
        },
    }
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(class_id=1, name="FREEZER_A", confidence=0.93),
            make_candidate(class_id=2, name="FREEZER_B", confidence=0.89),
        ],
        delta_weight=-999.0,
        active_products=[
            make_active_product(1, "FREEZER_A", weight=100.0, stock=5),
            make_active_product(2, "FREEZER_B", weight=110.0, stock=5),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.UNCERTAIN
    assert result.products == []
    assert trace.weight_diagnostics["freezer_vision_first"]["reason"] == (
        "no_weight_fit_for_vision_candidate_pool"
    )
    assert trace.weight_diagnostics["freezer_vision_first"][
        "orderedCombinationSearch"
    ]["accepted"] is False


def test_freezer_vision_first_selects_first_ranked_weight_fit(
    monkeypatch,
):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    trace = FakeLoadcellTrace({})
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                class_id=13,
                name="BAG_COOZROCK_JUICY_MEAT_DUMPLING_168G",
                confidence=1.0,
                instance_count_hint=3,
            ),
            make_candidate(
                class_id=44,
                name="STICK_BINGGRAE_MELONA_75ML",
                confidence=1.0,
                instance_count_hint=2,
            ),
            make_candidate(
                class_id=37,
                name="BOX_SAJO_OLD_LUNCHBOX_JAJANGBAP_250G",
                confidence=0.4868,
                instance_count_hint=3,
            ),
            make_candidate(
                class_id=46,
                name="STICK_LALA_SWEET_GRAPE_ZERO_70ML",
                confidence=0.4854,
                instance_count_hint=3,
            ),
        ],
        delta_weight=-84.2,
        active_products=[
            make_active_product(
                13,
                "BAG_COOZROCK_JUICY_MEAT_DUMPLING_168G",
                weight=185.0,
                stock=99,
            ),
            make_active_product(44, "STICK_BINGGRAE_MELONA_75ML", weight=93.0, stock=95),
            make_active_product(
                37,
                "BOX_SAJO_OLD_LUNCHBOX_JAJANGBAP_250G",
                weight=307.0,
                stock=100,
            ),
            make_active_product(
                46,
                "STICK_LALA_SWEET_GRAPE_ZERO_70ML",
                weight=71.0,
                stock=99,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (44, 1)
    ]
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["reason"] == "freezer_ordered_vision_candidate_pool"
    assert diagnostics["selected"][0]["class_id"] == 44
    assert diagnostics["selected"][0]["raw_instance_count_hint"] == 2
    assert diagnostics["selected"][0]["instance_count_hint"] == 1


def test_freezer_vision_first_prefers_melona_exit_path_over_static_lala(monkeypatch):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    trace = FakeLoadcellTrace({})
    trace.stage_counts_by_class = {
        "46": {
            "class_id": 46,
            "name": "STICK_LALA_SWEET_GRAPE_ZERO_70ML",
            "freezer_roi_passed": 1,
        },
        "44": {
            "class_id": 44,
            "name": "STICK_BINGGRAE_MELONA_75ML",
            "freezer_roi_passed": 19,
        },
    }
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                class_id=46,
                name="STICK_LALA_SWEET_GRAPE_ZERO_70ML",
                confidence=1.0,
                instance_count_hint=4,
            ),
            make_candidate(
                class_id=44,
                name="STICK_BINGGRAE_MELONA_75ML",
                confidence=1.0,
                instance_count_hint=2,
            ),
        ],
        delta_weight=-81.0,
        active_products=[
            make_active_product(46, "STICK_LALA_SWEET_GRAPE_ZERO_70ML", weight=71.0),
            make_active_product(44, "STICK_BINGGRAE_MELONA_75ML", weight=93.0),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (46, 1)
    ]
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["reason"] == "freezer_ordered_vision_candidate_pool"
    assert diagnostics["selected"][0]["class_id"] == 46
    assert diagnostics["orderedCombinationSearch"]["attempts"][0]["classIds"] == [46]


def test_freezer_vision_first_prefers_cup_exit_path_weight_gate(monkeypatch):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "top_k", 6)
    trace = FakeLoadcellTrace({})
    trace.stage_counts_by_class = {
        "30": {
            "class_id": 30,
            "name": "BOX_BINGGRAE_YOMAMTE_150ML",
            "raw": 31,
            "raw_max_confidence": 0.5891,
            "threshold_passed": 8,
            "threshold_passed_max_confidence": 0.5891,
            "freezer_roi_filtered": 8,
            "freezerExitPathVotes": 8,
            "freezer_roi_filtered_max_confidence": 0.5891,
            "roi_x_avg": 463.0,
            "roi_y_avg": 128.7,
            "cameras": {
                "top": {
                    "threshold_passed": 8,
                    "freezer_roi_filtered": 8,
                    "freezerExitPathVotes": 8,
                    "raw_max_confidence": 0.5891,
                },
                "side": {"raw": 6, "raw_max_confidence": 0.1214},
            },
        },
            "46": {
                "class_id": 46,
                "name": "STICK_LALA_SWEET_GRAPE_ZERO_70ML",
                "freezer_roi_passed": 0,
            },
            "44": {
                "class_id": 44,
                "name": "STICK_BINGGRAE_MELONA_75ML",
                "freezer_roi_passed": 4,
            },
            "42": {
                "class_id": 42,
                "name": "CUP_MAEIL_SANGHAFARM_MILK_ICE_CREAMG_100G",
                "freezer_roi_passed": 13,
            },
    }
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(46, "STICK_LALA_SWEET_GRAPE_ZERO_70ML", confidence=1.0),
            make_candidate(37, "BOX_SAJO_OLD_LUNCHBOX_JAJANGBAP_250G", confidence=1.0),
            make_candidate(13, "BAG_COOZROCK_JUICY_MEAT_DUMPLING_168G", confidence=1.0),
            make_candidate(44, "STICK_BINGGRAE_MELONA_75ML", confidence=0.7242),
            make_candidate(24, "BAG_JACKSONVILLE_BIG_HOT_DOG_115G", confidence=0.4311),
            make_candidate(
                42,
                "CUP_MAEIL_SANGHAFARM_MILK_ICE_CREAMG_100G",
                confidence=0.4011,
            ),
        ],
        delta_weight=-97.9,
        active_products=[
            make_active_product(46, "STICK_LALA_SWEET_GRAPE_ZERO_70ML", weight=71.0),
            make_active_product(37, "BOX_SAJO_OLD_LUNCHBOX_JAJANGBAP_250G", weight=307.0),
            make_active_product(13, "BAG_COOZROCK_JUICY_MEAT_DUMPLING_168G", weight=185.0),
            make_active_product(44, "STICK_BINGGRAE_MELONA_75ML", weight=93.0),
            make_active_product(24, "BAG_JACKSONVILLE_BIG_HOT_DOG_115G", weight=154.0),
            make_active_product(30, "BOX_BINGGRAE_YOMAMTE_150ML", weight=87.0),
            make_active_product(
                42,
                "CUP_MAEIL_SANGHAFARM_MILK_ICE_CREAMG_100G",
                weight=93.0,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (44, 1)
    ]
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["reason"] == "freezer_ordered_vision_candidate_pool"
    assert diagnostics["selected"][0]["class_id"] == 44


def test_freezer_vision_first_prefers_melona_residual_over_yomamte_exit_votes(
    monkeypatch,
):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "top_k", 10)
    trace = FakeLoadcellTrace(
        {
            "delta_weight": -78.8,
            "target_weight_abs": 78.8,
            "compound_event": True,
            "compound_negative_segment_count": 2,
            "removal_segment_targets": [
                {"weight": 55.8, "segment_index": 0},
                {"weight": 23.0, "segment_index": 1},
            ],
        }
    )
    trace.stage_counts_by_class = {
        "30": {
            "class_id": 30,
            "name": "BOX_BINGGRAE_YOMAMTE_150ML",
            "freezer_roi_passed": 52,
            "pathDisplacementPx": 66.7,
            "trajectoryExitPathPassed": True,
            "cameras": {"top": {"freezerExitPathVotes": 52}},
        },
        "44": {
            "class_id": 44,
            "name": "STICK_BINGGRAE_MELONA_75ML",
            "freezer_roi_passed": 36,
            "pathDisplacementPx": 422.4,
            "trajectoryExitPathPassed": True,
            "cameras": {
                "top": {"freezerExitPathVotes": 23},
                "side": {"freezerExitPathVotes": 13},
            },
        },
    }
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                30,
                "BOX_BINGGRAE_YOMAMTE_150ML",
                confidence=0.8075,
                raw_vote_count=103,
            ),
            make_candidate(
                44,
                "STICK_BINGGRAE_MELONA_75ML",
                confidence=1.0,
                raw_vote_count=44,
            ),
        ],
        delta_weight=-78.8,
        active_products=[
            make_active_product(30, "BOX_BINGGRAE_YOMAMTE_150ML", weight=82.0),
            make_active_product(44, "STICK_BINGGRAE_MELONA_75ML", weight=79.0),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (30, 1)
    ]
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    selected = diagnostics["selected"][0]
    assert diagnostics["reason"] == "freezer_ordered_vision_candidate_pool"
    assert diagnostics["multiItemTraceEvidence"] is True
    assert selected["class_id"] == 30
    assert selected["weight_residual"] == 3.2
    assert selected["selectionTier"] == "freezer_ordered_vision_candidate_pool"
    assert diagnostics["orderedCombinationSearch"]["attempts"][0]["classIds"] == [30]


def test_freezer_vision_first_keeps_video_handled_yomamte_over_stage_melona(
    monkeypatch,
):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "top_k", 10)
    trace = FakeLoadcellTrace(
        {
            "delta_weight": -87.1,
            "target_weight_abs": 87.1,
            "compound_event": True,
            "compound_positive_segment_count": 1,
            "compound_negative_segment_count": 1,
            "removal_segment_targets": [
                {"weight": 136.6, "segment_index": 1},
            ],
            "return_segment_targets": [
                {"weight": 49.4, "segment_index": 0},
            ],
        }
    )
    trace.record_weight_diagnostics(
        {
            "freezer_candidate_filter": {
                "accepted": True,
                "reason": "vision_identity_passthrough",
                "handled_candidate_count": 1,
                "selectedClassIds": [30],
                "considered": [
                    {"class_id": 30, "name": "BOX_BINGGRAE_YOMAMTE_150ML"},
                    {"class_id": 44, "name": "STICK_BINGGRAE_MELONA_75ML"},
                ],
            }
        }
    )
    trace.stage_counts_by_class = {
        "30": {
            "class_id": 30,
            "name": "BOX_BINGGRAE_YOMAMTE_150ML",
            "freezer_roi_passed": 24,
            "freezerExitPathVotes": 24,
            "pathDisplacementPx": 41.6,
            "trajectoryExitPathPassed": True,
            "cameras": {"top": {"freezerExitPathVotes": 24}},
        },
        "44": {
            "class_id": 44,
            "name": "STICK_BINGGRAE_MELONA_75ML",
            "raw": 108,
            "raw_max_confidence": 0.6842,
            "threshold_passed": 18,
            "threshold_passed_max_confidence": 0.6842,
            "freezer_roi_passed": 18,
            "freezerExitPathVotes": 18,
            "freezer_roi_passed_max_confidence": 0.6842,
            "pathDisplacementPx": 29.8,
            "trajectoryExitPathPassed": True,
            "cameras": {
                "top": {
                    "freezerExitPathVotes": 12,
                    "freezer_roi_passed": 12,
                    "raw_max_confidence": 0.6842,
                },
                "side": {
                    "freezerExitPathVotes": 6,
                    "freezer_roi_passed": 6,
                    "raw_max_confidence": 0.3324,
                },
            },
        },
    }
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                30,
                "BOX_BINGGRAE_YOMAMTE_150ML",
                confidence=0.7149,
            ),
        ],
        delta_weight=-87.1,
        active_products=[
            make_active_product(30, "BOX_BINGGRAE_YOMAMTE_150ML", weight=82.0),
            make_active_product(44, "STICK_BINGGRAE_MELONA_75ML", weight=79.0),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (30, 1)
    ]
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["reason"] == "freezer_ordered_vision_candidate_pool"
    assert diagnostics["selected"][0]["class_id"] == 30
    assert diagnostics["selected"][0]["weight_residual"] == 5.1
    assert "rejectedStageOnlyCandidates" not in diagnostics


def test_freezer_vision_first_strict_vision_candidate_blocks_stage_only_priority(
    monkeypatch,
):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "top_k", 10)
    trace = FakeLoadcellTrace({})
    trace.stage_counts_by_class = {
        "30": {
            "class_id": 30,
            "name": "BOX_BINGGRAE_YOMAMTE_150ML",
            "freezer_roi_passed": 24,
            "freezerExitPathVotes": 24,
            "cameras": {"top": {"freezerExitPathVotes": 24}},
        },
        "44": {
            "class_id": 44,
            "name": "STICK_BINGGRAE_MELONA_75ML",
            "raw": 108,
            "raw_max_confidence": 0.6842,
            "threshold_passed": 18,
            "threshold_passed_max_confidence": 0.6842,
            "freezer_roi_passed": 18,
            "freezerExitPathVotes": 18,
            "cameras": {
                "top": {"freezerExitPathVotes": 12, "raw_max_confidence": 0.6842},
                "side": {"freezerExitPathVotes": 6, "raw_max_confidence": 0.3324},
            },
        },
    }
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                30,
                "BOX_BINGGRAE_YOMAMTE_150ML",
                confidence=0.7149,
            ),
        ],
        delta_weight=-87.1,
        active_products=[
            make_active_product(30, "BOX_BINGGRAE_YOMAMTE_150ML", weight=82.0),
            make_active_product(44, "STICK_BINGGRAE_MELONA_75ML", weight=79.0),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (30, 1)
    ]
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["reason"] == "freezer_ordered_vision_candidate_pool"
    assert [item["class_id"] for item in diagnostics["considered"]] == [30]
    assert "rejectedStageOnlyCandidates" not in diagnostics


def test_freezer_vision_first_does_not_create_stage_only_identity(monkeypatch):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "top_k", 5)
    trace = FakeLoadcellTrace({})
    trace.stage_counts_by_class = {
        "30": {
            "class_id": 30,
            "name": "BOX_BINGGRAE_YOMAMTE_150ML",
            "raw": 47,
            "raw_max_confidence": 0.5761,
            "threshold_passed": 11,
            "threshold_passed_max_confidence": 0.5761,
            "freezer_roi_filtered": 11,
            "freezerExitPathVotes": 11,
            "freezer_roi_filtered_max_confidence": 0.5761,
            "roi_x_avg": 369.8,
            "roi_y_avg": 73.0,
            "cameras": {
                "top": {
                    "raw": 34,
                    "threshold_passed": 10,
                    "freezer_roi_filtered": 10,
                    "freezerExitPathVotes": 10,
                    "raw_max_confidence": 0.5761,
                },
                "side": {
                    "raw": 13,
                    "threshold_passed": 1,
                    "freezer_roi_filtered": 1,
                    "freezerExitPathVotes": 1,
                    "raw_max_confidence": 0.5291,
                },
            },
        },
        "42": {
            "class_id": 42,
            "name": "CUP_MAEIL_SANGHAFARM_MILK_ICE_CREAMG_100G",
            "raw": 57,
            "raw_max_confidence": 0.6808,
            "threshold_passed": 15,
            "threshold_passed_max_confidence": 0.6808,
            "freezer_roi_filtered": 14,
            "freezerExitPathVotes": 14,
            "freezer_roi_filtered_max_confidence": 0.6808,
            "roi_x_avg": 380.1,
            "roi_y_avg": 66.9,
            "cameras": {
                "top": {
                    "raw": 50,
                    "threshold_passed": 14,
                    "freezer_roi_filtered": 14,
                    "freezerExitPathVotes": 14,
                    "raw_max_confidence": 0.6808,
                },
                "side": {"raw": 7, "threshold_passed": 1, "raw_max_confidence": 0.335},
            },
        },
        "44": {
            "class_id": 44,
            "name": "STICK_BINGGRAE_MELONA_75ML",
            "raw": 26,
            "raw_max_confidence": 0.666,
            "threshold_passed": 8,
            "freezer_roi_filtered": 5,
            "freezerExitPathVotes": 5,
            "roi_x_avg": 438.0,
            "roi_y_avg": 92.5,
            "cameras": {
                "top": {
                    "threshold_passed": 8,
                    "freezer_roi_filtered": 5,
                    "freezerExitPathVotes": 5,
                    "raw_max_confidence": 0.666,
                }
            },
        },
    }
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(46, "STICK_LALA_SWEET_GRAPE_ZERO_70ML", confidence=1.0),
            make_candidate(13, "BAG_COOZROCK_JUICY_MEAT_DUMPLING_168G", confidence=0.7469),
            make_candidate(37, "BOX_SAJO_OLD_LUNCHBOX_JAJANGBAP_250G", confidence=0.4317),
            make_candidate(24, "BAG_JACKSONVILLE_BIG_HOT_DOG_115G", confidence=0.341),
            make_candidate(44, "STICK_BINGGRAE_MELONA_75ML", confidence=0.153),
        ],
        delta_weight=-96.3,
        active_products=[
            make_active_product(46, "STICK_LALA_SWEET_GRAPE_ZERO_70ML", weight=71.0),
            make_active_product(13, "BAG_COOZROCK_JUICY_MEAT_DUMPLING_168G", weight=185.0),
            make_active_product(37, "BOX_SAJO_OLD_LUNCHBOX_JAJANGBAP_250G", weight=307.0),
            make_active_product(24, "BAG_JACKSONVILLE_BIG_HOT_DOG_115G", weight=154.0),
            make_active_product(44, "STICK_BINGGRAE_MELONA_75ML", weight=93.0),
            make_active_product(30, "BOX_BINGGRAE_YOMAMTE_150ML", weight=87.0),
            make_active_product(
                42,
                "CUP_MAEIL_SANGHAFARM_MILK_ICE_CREAMG_100G",
                weight=93.0,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.UNCERTAIN
    assert result.products == []
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["reason"] == "no_weight_fit_for_vision_candidate_pool"
    melona = next(item for item in diagnostics["considered"] if item["class_id"] == 44)
    assert melona["reason"] == "insufficient_vision_identity_evidence"
    assert "ambiguousCandidates" not in diagnostics


def test_freezer_vision_first_uses_instance_hint_for_same_class_count(monkeypatch):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    trace = FakeLoadcellTrace({})
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                class_id=1,
                name="FREEZER_A",
                confidence=0.92,
                instance_count_hint=2,
            )
        ],
        delta_weight=-200.0,
        active_products=[
            make_active_product(1, "FREEZER_A", weight=100.0, stock=5),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [(1, 2)]
    assert trace.weight_diagnostics["freezer_vision_first"]["selected"][0][
        "instance_count_hint"
    ] == 2


def test_freezer_vision_first_selects_rank1_single_before_lower_rank_repeat(monkeypatch):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    trace = FakeLoadcellTrace({})
    trace.stage_counts_by_class = {
        "37": {
            "class_id": 37,
            "name": "BOX_SAJO_OLD_LUNCHBOX_JAJANGBAP_250G",
            "freezerExitPathVotes": 40,
            "freezer_roi_filtered": 40,
            "cameras": {"top": {"freezerExitPathVotes": 40}},
        },
        "27": {
            "class_id": 27,
            "name": "BAG_NULLDAM_BAGEL_140G",
            "freezerExitPathVotes": 9,
            "freezer_roi_filtered": 9,
            "cameras": {"top": {"freezerExitPathVotes": 9}},
        },
    }
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                class_id=37,
                name="BOX_SAJO_OLD_LUNCHBOX_JAJANGBAP_250G",
                confidence=1.0,
                raw_vote_count=147,
                instance_count_hint=3,
            ),
            make_candidate(
                class_id=27,
                name="BAG_NULLDAM_BAGEL_140G",
                confidence=0.52,
                raw_vote_count=4,
                instance_count_hint=1,
            ),
        ],
        delta_weight=-307.2,
        active_products=[
            make_active_product(
                37,
                "BOX_SAJO_OLD_LUNCHBOX_JAJANGBAP_250G",
                weight=309.0,
                stock=93,
            ),
            make_active_product(
                27,
                "BAG_NULLDAM_BAGEL_140G",
                weight=156.0,
                stock=97,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (37, 1)
    ]
    assert result.weight_explained == 309.0
    assert result.weight_residual == pytest.approx(1.8)
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["reason"] == "freezer_ordered_vision_candidate_pool"
    assert diagnostics["selected"][0]["class_id"] == 37
    assert diagnostics["selected"][0]["count"] == 1
    assert diagnostics["selected"][0]["expected_weight"] == 309.0
    assert diagnostics["selected"][0]["combinationResidual"] == pytest.approx(1.8)
    assert diagnostics["orderedCombinationSearch"]["attempts"][0]["classIds"] == [37]


@pytest.mark.parametrize(
    ("delta_weight", "expected_residual"),
    [
        (-303.0, 9.0),
        (-304.0, 8.0),
        (-305.0, 7.0),
        (-313.0, 1.0),
    ],
)
def test_freezer_vision_first_counts_bagel_repeat_from_weight(
    monkeypatch,
    delta_weight,
    expected_residual,
):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    trace = FakeLoadcellTrace({})
    engine = ProductDecisionEngine(strict_mode=True)
    candidate = make_candidate(
        class_id=27,
        name="BAG_NULLDAM_BAGEL_140G",
        confidence=0.80,
        raw_vote_count=4,
        instance_count_hint=1,
    )
    candidate.freezer_exit_path_votes = 9

    result = engine.judge(
        vision_candidates=[candidate],
        delta_weight=delta_weight,
        active_products=[
            make_active_product(
                27,
                "BAG_NULLDAM_BAGEL_140G",
                weight=156.0,
                stock=10,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (27, 2)
    ]
    assert result.weight_explained == 312.0
    assert result.weight_residual == pytest.approx(expected_residual)
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["reason"] == "freezer_ordered_vision_candidate_pool"
    assert diagnostics["selected"][0]["count"] == 2
    assert diagnostics["selected"][0]["combinationResidual"] == pytest.approx(
        expected_residual
    )
    assert diagnostics["sameProductRepeatCandidates"][0]["class_id"] == 27


def test_freezer_vision_first_counts_single_side_bagel_repeat_from_weight(
    monkeypatch,
):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    trace = FakeLoadcellTrace({})
    engine = ProductDecisionEngine(strict_mode=True)
    candidate = make_candidate(
        class_id=27,
        name="BAG_NULLDAM_BAGEL_140G",
        confidence=0.728,
        raw_vote_count=1,
        instance_count_hint=1,
    )
    candidate.vote_count = 1
    candidate.top_confidence = 0.0
    candidate.side_confidence = 0.728
    candidate.freezer_exit_path_votes = 0

    result = engine.judge(
        vision_candidates=[candidate],
        delta_weight=-309.5,
        active_products=[
            make_active_product(
                27,
                "BAG_NULLDAM_BAGEL_140G",
                weight=156.0,
                stock=10,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (27, 2)
    ]
    assert result.weight_explained == 312.0
    assert result.weight_residual == pytest.approx(2.5)
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["reason"] == "freezer_ordered_vision_candidate_pool"
    assert diagnostics["selected"][0]["count"] == 2
    assert diagnostics["selected"][0]["combinationResidual"] == pytest.approx(2.5)
    assert diagnostics["orderedCombinationSearch"]["accepted"] is True


def test_freezer_vision_first_rejects_low_raw_confidence_candidate(monkeypatch):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")
    monkeypatch.setattr(config.vision, "top_confidence_threshold", 0.70)
    monkeypatch.setattr(config.vision, "side_confidence_threshold", 0.70)
    trace = FakeLoadcellTrace({})
    engine = ProductDecisionEngine(strict_mode=True)
    candidate = make_candidate(
        class_id=44,
        name="STICK_BINGGRAE_MELONA_75ML",
        confidence=0.58,
        raw_vote_count=9,
    )
    candidate.top_confidence = 0.58
    candidate.side_confidence = 0.0

    result = engine.judge(
        vision_candidates=[candidate],
        delta_weight=-79.0,
        active_products=[
            make_active_product(44, "STICK_BINGGRAE_MELONA_75ML", weight=79.0),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.NO_DETECTION
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["reason"] == "no_supported_vision_candidates"
    considered = diagnostics["considered"][0]
    assert considered["reason"] == "insufficient_vision_identity_evidence"
    assert considered["identity_confidence"] == 0.58
    assert considered["identity_threshold"] == 0.70


def test_freezer_ordered_solver_selects_rank1_cheese_burger_before_dumpling(
    monkeypatch,
):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "camera_layout", "dual_top_proxy")
    trace = FakeLoadcellTrace({})
    engine = ProductDecisionEngine(strict_mode=True)

    cheese = make_candidate(
        class_id=23,
        name="BAG_HANMAC_TRIPLE_CHEESE_HAMBURGER_155G",
        confidence=1.0,
        raw_vote_count=16,
        instance_count_hint=2,
    )
    dumpling = make_candidate(
        class_id=13,
        name="BAG_COOZROCK_JUICY_MEAT_DUMPLING_168G",
        confidence=0.3101,
        raw_vote_count=19,
        instance_count_hint=1,
    )

    result = engine.judge(
        vision_candidates=[cheese, dumpling],
        delta_weight=-183.7,
        active_products=[
            make_active_product(
                23,
                "BAG_HANMAC_TRIPLE_CHEESE_HAMBURGER_155G",
                weight=176.0,
                stock=87,
            ),
            make_active_product(
                13,
                "BAG_COOZROCK_JUICY_MEAT_DUMPLING_168G",
                weight=189.0,
                stock=76,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (23, 1)
    ]
    assert result.weight_residual == pytest.approx(7.7)
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["reason"] == "freezer_ordered_vision_candidate_pool"
    assert diagnostics["selected"][0]["raw_instance_count_hint"] == 2
    assert diagnostics["selected"][0]["count"] == 1
    assert diagnostics["orderedCombinationSearch"]["attempts"][0]["classIds"] == [23]


def test_freezer_ordered_solver_selects_same_candidate_x2(monkeypatch):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    trace = FakeLoadcellTrace({})
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                class_id=27,
                name="BAG_NULLDAM_BAGEL_140G",
                confidence=0.8,
                raw_vote_count=4,
            )
        ],
        delta_weight=-309.5,
        active_products=[
            make_active_product(27, "BAG_NULLDAM_BAGEL_140G", weight=156.0, stock=10)
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (27, 2)
    ]
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["selected"][0]["count"] == 2
    assert diagnostics["orderedCombinationSearch"]["attempts"][1]["counts"] == [2]


def test_freezer_ordered_solver_selects_mixed_combo_after_singles_miss(monkeypatch):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    trace = FakeLoadcellTrace({})
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(101, "FREEZER_A", confidence=0.9),
            make_candidate(102, "FREEZER_B", confidence=0.8),
        ],
        delta_weight=-300.0,
        active_products=[
            make_active_product(101, "FREEZER_A", weight=120.0, stock=10),
            make_active_product(102, "FREEZER_B", weight=180.0, stock=10),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (101, 1),
        (102, 1),
    ]
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["orderedCombinationSearch"]["accepted"] is True
    assert diagnostics["selected"][0]["combinationExpectedWeight"] == 300.0


def test_freezer_ordered_solver_records_missing_weight_without_charge(monkeypatch):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    trace = FakeLoadcellTrace({})
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(101, "FREEZER_UNKNOWN_WEIGHT", confidence=0.9),
        ],
        delta_weight=-120.0,
        active_products=[
            make_active_product(101, "FREEZER_UNKNOWN_WEIGHT", weight=0.0, stock=10),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.NO_DETECTION
    assert result.products == []
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["reason"] == "no_supported_vision_candidates"
    assert diagnostics["considered"][0]["reason"] == "weight_unavailable"


def test_freezer_vision_first_single_bagel_repeat_requires_confidence_floor(
    monkeypatch,
):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    trace = FakeLoadcellTrace({})
    engine = ProductDecisionEngine(strict_mode=True)
    candidate = make_candidate(
        class_id=27,
        name="BAG_NULLDAM_BAGEL_140G",
        confidence=0.29,
        raw_vote_count=1,
        instance_count_hint=1,
    )
    candidate.vote_count = 1
    candidate.freezer_exit_path_votes = 0

    result = engine.judge(
        vision_candidates=[candidate],
        delta_weight=-309.5,
        active_products=[
            make_active_product(
                27,
                "BAG_NULLDAM_BAGEL_140G",
                weight=156.0,
                stock=10,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.NO_DETECTION
    assert result.products == []
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["reason"] == "no_supported_vision_candidates"
    assert diagnostics["considered"][0]["reason"] == (
        "insufficient_vision_identity_evidence"
    )


def test_freezer_vision_first_rejects_valid_weight_single_mismatch(
    monkeypatch,
):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    trace = FakeLoadcellTrace({})
    engine = ProductDecisionEngine(strict_mode=True)
    candidate = make_candidate(
        class_id=27,
        name="BAG_NULLDAM_BAGEL_140G",
        confidence=0.80,
        raw_vote_count=4,
        instance_count_hint=1,
    )
    candidate.freezer_exit_path_votes = 3

    result = engine.judge(
        vision_candidates=[candidate],
        delta_weight=-280.0,
        active_products=[
            make_active_product(
                27,
                "BAG_NULLDAM_BAGEL_140G",
                weight=156.0,
                stock=10,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.UNCERTAIN
    assert result.products == []
    assert result.weight_explained == pytest.approx(0.0)
    assert result.weight_residual == pytest.approx(280.0)
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["reason"] == "no_weight_fit_for_vision_candidate_pool"
    assert diagnostics["orderedCombinationSearch"]["accepted"] is False


@pytest.mark.parametrize(
    ("stock", "delta_weight", "raw_vote_count", "expected_reason"),
    [
        (1, -313.0, 4, "count_cap_below_repeat"),
        (10, -340.0, 4, "repeat_residual_exceeds_tolerance"),
    ],
)
def test_freezer_vision_first_rejects_unsafe_bagel_repeat(
    monkeypatch,
    stock,
    delta_weight,
    raw_vote_count,
    expected_reason,
):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    trace = FakeLoadcellTrace({})
    engine = ProductDecisionEngine(strict_mode=True)
    candidate = make_candidate(
        class_id=27,
        name="BAG_NULLDAM_BAGEL_140G",
        confidence=0.80,
        raw_vote_count=raw_vote_count,
        instance_count_hint=1,
    )
    candidate.freezer_exit_path_votes = 9

    result = engine.judge(
        vision_candidates=[candidate],
        delta_weight=delta_weight,
        active_products=[
            make_active_product(
                27,
                "BAG_NULLDAM_BAGEL_140G",
                weight=156.0,
                stock=stock,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.UNCERTAIN
    assert result.products == []
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["reason"] == "no_weight_fit_for_vision_candidate_pool"
    assert diagnostics["orderedCombinationSearch"]["accepted"] is False


def test_freezer_vision_first_keeps_single_when_bagel_x1_is_closer(monkeypatch):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    trace = FakeLoadcellTrace({})
    engine = ProductDecisionEngine(strict_mode=True)
    candidate = make_candidate(
        class_id=27,
        name="BAG_NULLDAM_BAGEL_140G",
        confidence=0.80,
        raw_vote_count=4,
        instance_count_hint=1,
    )
    candidate.freezer_exit_path_votes = 9

    result = engine.judge(
        vision_candidates=[candidate],
        delta_weight=-230.0,
        active_products=[
            make_active_product(
                27,
                "BAG_NULLDAM_BAGEL_140G",
                weight=156.0,
                stock=10,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.UNCERTAIN
    assert result.products == []
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["reason"] == "no_weight_fit_for_vision_candidate_pool"
    assert diagnostics["orderedCombinationSearch"]["accepted"] is False


def test_freezer_vision_first_ordered_solver_keeps_rank1_static_candidate(monkeypatch):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    trace = FakeLoadcellTrace({})
    trace.stage_counts_by_class = {
        "101": {
            "class_id": 101,
            "name": "STATIC_TIGHT_SINGLE",
            "freezerExitPathVotes": 40,
            "pathDisplacementPx": 2.0,
            "maxDistancePx": 14.0,
            "centerSpanX": 3.0,
            "centerSpanY": 3.0,
            "motionThresholdPx": 12.0,
            "trajectoryExitPathPassed": False,
            "staticShelfLikely": True,
            "cameras": {"top": {"freezerExitPathVotes": 40}},
        },
        "102": {
            "class_id": 102,
            "name": "TRAJECTORY_SUPPORTED_SINGLE",
            "freezerExitPathVotes": 8,
            "pathDisplacementPx": 18.0,
            "maxDistancePx": 18.0,
            "centerSpanX": 5.0,
            "centerSpanY": 18.0,
            "motionThresholdPx": 12.0,
            "trajectoryExitPathPassed": True,
            "staticShelfLikely": False,
            "cameras": {"top": {"freezerExitPathVotes": 8}},
        },
    }
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                class_id=101,
                name="STATIC_TIGHT_SINGLE",
                confidence=1.0,
                raw_vote_count=80,
            ),
            make_candidate(
                class_id=102,
                name="TRAJECTORY_SUPPORTED_SINGLE",
                confidence=0.7,
                raw_vote_count=8,
            ),
        ],
        delta_weight=-100.0,
        active_products=[
            make_active_product(101, "STATIC_TIGHT_SINGLE", weight=100.0, stock=10),
            make_active_product(
                102,
                "TRAJECTORY_SUPPORTED_SINGLE",
                weight=106.0,
                stock=10,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (101, 1)
    ]
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    static_candidate = next(
        item for item in diagnostics["considered"] if item["class_id"] == 101
    )
    assert static_candidate["interactionPenalty"] is True
    assert diagnostics["selected"][0]["class_id"] == 101


def test_freezer_vision_first_hand_far_static_single_is_rejected(monkeypatch):
    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    trace = FakeLoadcellTrace({})
    trace.stage_counts_by_class = {
        "101": {
            "class_id": 101,
            "name": "HAND_FAR_STATIC_TIGHT_SINGLE",
            "freezerExitPathVotes": 40,
            "handPathValid": True,
            "handPathValidUpperRoi": True,
            "handInteractionPassed": False,
            "handNearFrameCount": 0,
            "handPathPassed": False,
            "handPathBlocked": True,
            "staticShelfLikely": True,
            "cameras": {"top": {"freezerExitPathVotes": 40}},
        },
        "102": {
            "class_id": 102,
            "name": "HAND_NEAR_SINGLE",
            "freezerExitPathVotes": 6,
            "handPathValid": True,
            "handPathValidUpperRoi": True,
            "handInteractionPassed": True,
            "handNearFrameCount": 3,
            "handNearVoteRatio": 0.6,
            "minHandDistancePx": 12.0,
            "handPathPassed": True,
            "handPathBlocked": False,
            "cameras": {"top": {"freezerExitPathVotes": 6}},
        },
    }
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                class_id=101,
                name="HAND_FAR_STATIC_TIGHT_SINGLE",
                confidence=1.0,
                raw_vote_count=80,
            ),
            make_candidate(
                class_id=102,
                name="HAND_NEAR_SINGLE",
                confidence=0.7,
                raw_vote_count=8,
            ),
        ],
        delta_weight=-100.0,
        active_products=[
            make_active_product(101, "HAND_FAR_STATIC_TIGHT_SINGLE", weight=100.0),
            make_active_product(102, "HAND_NEAR_SINGLE", weight=106.0),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (102, 1)
    ]
    diagnostics = trace.weight_diagnostics["freezer_vision_first"]
    assert diagnostics["selected"][0]["handInteractionPassed"] is True
    assert [item["class_id"] for item in diagnostics["rejectedInteractionCandidates"]] == [
        101
    ]
    assert diagnostics["rejectedInteractionCandidates"][0][
        "interactionRejectedReason"
    ] == "hand_path_blocked"


def test_loadcell_only_returns_nearest_single_within_5g():
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge_by_weight_only(
        delta_weight=-57.0,
        active_products=[
            make_active_product(
                class_id=115,
                name="BOX_LOTTE_PEPERO_ALMOND_37G",
                weight=58.0,
                stock=1,
                price=1700,
            )
        ],
    )

    assert result.status == JudgmentStatus.PARTIAL
    assert result.products[0].product_id == 115
    assert result.products[0].count == 1
    assert result.weight_residual == 1.0


def test_no_vision_forced_final_fallback_rejects_nearest_active_mismatch(monkeypatch):
    use_weight_aware_identity(monkeypatch)
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace({})

    result = engine.judge(
        vision_candidates=[],
        delta_weight=-200.0,
        active_products=[
            make_active_product(101, "Nearest Active Product", weight=250.0, stock=3),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.UNCERTAIN
    assert result.products == []
    assert result.weight_residual == 50.0
    assert trace.weight_diagnostics["decision_branch"] == "forced_final_fallback"
    assert trace.weight_diagnostics["forced_final_fallback"]["inside_tolerance"] is False
    assert trace.weight_diagnostics["final_weight_mismatch_guard"]["accepted"] is False


def test_ranked_candidate_repeat_beats_same_weight_unseen_active_repeat():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace({})

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                31,
                "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML",
                0.402,
            ),
            make_candidate(
                44,
                "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                0.185,
            ),
        ],
        delta_weight=-1048.8,
        active_products=[
            make_active_product(
                31,
                "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML",
                weight=523.0,
                stock=10,
                price=1600,
            ),
            make_active_product(
                44,
                "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                weight=520.0,
                stock=10,
                price=2000,
            ),
            make_active_product(
                75,
                "BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
                weight=520.0,
                stock=10,
                price=2300,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (31, 2)
    ]
    assert result.weight_residual == pytest.approx(2.8)


def test_no_final_candidates_stage_combo_runs_before_active_forced_fallback():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace({})
    trace.stage_counts_by_class = {
        "31": make_stage_count_entry(
            31,
            "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML",
            confidence=0.402,
            raw=20,
            threshold_passed=8,
        ),
        "119": make_stage_count_entry(
            119,
            "BOTTLE_FANTA_ORANGE_600ML",
            confidence=0.4,
            raw=20,
            threshold_passed=8,
        ),
    }

    result = engine.judge(
        vision_candidates=[],
        delta_weight=-1157.0,
        active_products=[
            make_active_product(
                31,
                "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML",
                weight=523.0,
                stock=10,
                price=1600,
            ),
            make_active_product(
                119,
                "BOTTLE_FANTA_ORANGE_600ML",
                weight=634.0,
                stock=10,
                price=2000,
            ),
            make_active_product(
                44,
                "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                weight=520.0,
                stock=10,
                price=2000,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert {(product.product_id, product.count) for product in result.products} == {
        (31, 1),
        (119, 1),
    }
    diagnostics = trace.weight_diagnostics["stage_count_combination_match"]
    assert diagnostics["accepted"] is True
    assert diagnostics["stage_candidates_added"] == 2
    assert trace.weight_diagnostics["decision_branch"] == "stage_count_combination_match"
    assert "forced_final_fallback" not in trace.weight_diagnostics


def test_no_final_candidates_rescue_combo_runs_before_active_forced_fallback():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace({})
    trace.threshold_rescue_candidates = [
        {
            "class_id": 31,
            "name": "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML",
            "confidence": 0.402,
            "votes": 8,
            "side": True,
            "side_motion_passed": True,
            "motion_gate_passed": True,
            "weight_gate_passed": True,
        },
        {
            "class_id": 119,
            "name": "BOTTLE_FANTA_ORANGE_600ML",
            "confidence": 0.4,
            "votes": 8,
            "side": True,
            "side_motion_passed": True,
            "motion_gate_passed": True,
            "weight_gate_passed": True,
        },
    ]

    result = engine.judge(
        vision_candidates=[],
        delta_weight=-1157.0,
        active_products=[
            make_active_product(
                31,
                "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML",
                weight=523.0,
                stock=10,
                price=1600,
            ),
            make_active_product(
                119,
                "BOTTLE_FANTA_ORANGE_600ML",
                weight=634.0,
                stock=10,
                price=2000,
            ),
            make_active_product(
                44,
                "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                weight=520.0,
                stock=10,
                price=2000,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert {(product.product_id, product.count) for product in result.products} == {
        (31, 1),
        (119, 1),
    }
    diagnostics = trace.weight_diagnostics["stage_count_combination_match"]
    assert diagnostics["accepted"] is True
    assert diagnostics["rescue_candidates_added"] == 2
    assert trace.weight_diagnostics["decision_branch"] == "stage_count_combination_match"
    assert "forced_final_fallback" not in trace.weight_diagnostics


def test_no_final_candidates_diagnostic_combo_runs_before_active_forced_fallback():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace({})
    trace.diagnostic_detections = [
        {
            "class_id": class_id,
            "name": name,
            "confidence": confidence,
        }
        for class_id, name, confidence in (
            (31, "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML", 0.402),
            (119, "BOTTLE_FANTA_ORANGE_600ML", 0.4),
        )
        for _ in range(5)
    ]

    result = engine.judge(
        vision_candidates=[],
        delta_weight=-1157.0,
        active_products=[
            make_active_product(
                31,
                "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML",
                weight=523.0,
                stock=10,
                price=1600,
            ),
            make_active_product(
                119,
                "BOTTLE_FANTA_ORANGE_600ML",
                weight=634.0,
                stock=10,
                price=2000,
            ),
            make_active_product(
                44,
                "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                weight=520.0,
                stock=10,
                price=2000,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert {(product.product_id, product.count) for product in result.products} == {
        (31, 1),
        (119, 1),
    }
    diagnostics = trace.weight_diagnostics["stage_count_combination_match"]
    assert diagnostics["accepted"] is True
    assert diagnostics["diagnostic_candidates_added"] == 2
    assert trace.weight_diagnostics["decision_branch"] == "stage_count_combination_match"
    assert "forced_final_fallback" not in trace.weight_diagnostics


@pytest.mark.parametrize("delta_weight", [-6.0, -10.0])
def test_no_vision_low_weight_noise_does_not_force_condition_stick(
    delta_weight,
    monkeypatch,
):
    use_weight_aware_identity(monkeypatch)
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace({})

    result = engine.judge(
        vision_candidates=[],
        delta_weight=delta_weight,
        active_products=[
            make_active_product(
                113,
                "STICK_INNON_CONDITION_STICK_18G",
                weight=19.0,
                stock=10,
                price=3000,
            ),
            make_active_product(
                118,
                "BAG_CJ_CHICKEN_BREAST_STEAK_100G",
                weight=107.0,
                stock=10,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.UNCERTAIN
    assert result.products == []
    diagnostics = trace.weight_diagnostics["forced_final_fallback"]
    assert diagnostics["accepted"] is False
    assert diagnostics["reason"] == "active_only_low_weight_noise"


def test_no_vision_real_condition_stick_weight_still_matches_loadcell(monkeypatch):
    use_weight_aware_identity(monkeypatch)
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[],
        delta_weight=-18.0,
        active_products=[
            make_active_product(
                113,
                "STICK_INNON_CONDITION_STICK_18G",
                weight=19.0,
                stock=10,
                price=3000,
            )
        ],
    )

    assert result.status == JudgmentStatus.PARTIAL
    assert result.products[0].product_id == 113
    assert result.weight_residual == 1.0


def test_full_purchase_delta_candidate_matches_full_delta(monkeypatch):
    use_weight_aware_identity(monkeypatch)
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace(
        {
            "purchase_delta_candidates": [
                {"weight": 200.0, "source": "last_unpaired_negative_segment"}
            ]
        }
    )

    result = engine.judge(
        vision_candidates=[],
        delta_weight=-200.0,
        active_products=[
            make_active_product(101, "Aggregate Miss Candidate", weight=150.0, stock=3),
            make_active_product(102, "History Candidate", weight=200.0, stock=3),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.PARTIAL
    assert result.products[0].product_id == 102
    assert result.weight_residual == 0.0
    assert trace.weight_diagnostics["final_weight_mismatch_guard"]["accepted"] is True


def test_forced_final_fallback_rejects_partial_purchase_delta_target(monkeypatch):
    use_weight_aware_identity(monkeypatch)
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace(
        {
            "purchase_delta_candidates": [
                {"weight": 503.0, "source": "net_stable_delta"},
                {"weight": 51.9, "source": "last_unpaired_negative_segment"},
            ]
        }
    )

    result = engine.judge(
        vision_candidates=[],
        delta_weight=-503.0,
        active_products=[
            make_active_product(57, "BAG_HAITAI_JAGABEE_45G", weight=52.0, stock=3),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.UNCERTAIN
    assert result.products == []
    rejected = trace.weight_diagnostics["forced_fallback_rejected_partial_target"]
    assert rejected["accepted"] is False
    assert rejected["rejected"][0]["source"] == "last_unpaired_negative_segment"
    guard = trace.weight_diagnostics["final_weight_mismatch_guard"]
    assert guard["accepted"] is False
    assert guard["explained_weight"] == 52.0


def test_forced_final_fallback_tries_detected_plus_active_pair_after_single_miss(
    monkeypatch,
):
    use_weight_aware_identity(monkeypatch)
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace({})

    result = engine.judge(
        vision_candidates=[make_candidate(101, "Detected Candidate", confidence=0.65)],
        delta_weight=-250.0,
        active_products=[
            make_active_product(101, "Detected Candidate", weight=100.0, stock=3),
            make_active_product(102, "Unseen Active Product", weight=150.0, stock=3),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.PARTIAL
    assert {product.product_id: product.count for product in result.products} == {
        101: 1,
        102: 1,
    }
    assert result.weight_residual == 0.0
    assert trace.weight_diagnostics["forced_final_fallback"]["mode"] == (
        "detected_plus_active_pair"
    )


def test_forced_final_fallback_prefers_supported_detected_repeat_over_weak_pair(
    monkeypatch,
):
    use_weight_aware_identity(monkeypatch)
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace({"purchase_delta_candidates": [{"weight": 1047.0}]})
    trace.stage_counts_by_class = {
        "75": {
            "class_id": 75,
            "name": "BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
            "raw": 34,
            "raw_max_confidence": 0.2018,
            "threshold_passed": 3,
            "threshold_passed_max_confidence": 0.2018,
            "roi_filtered": 3,
            "roi_filtered_max_confidence": 0.2018,
            "cameras": {
                "side": {
                    "raw": 34,
                    "raw_max_confidence": 0.2018,
                    "threshold_passed": 3,
                    "threshold_passed_max_confidence": 0.2018,
                    "roi_filtered": 3,
                    "roi_filtered_max_confidence": 0.2018,
                }
            },
        },
        "54": {
            "class_id": 54,
            "name": "BOTTLE_LOTTE_TREVI_LEMON_500ML",
            "raw": 2,
            "raw_max_confidence": 0.0231,
            "threshold_filtered": 2,
            "threshold_filtered_max_confidence": 0.0231,
            "cameras": {
                "side": {
                    "raw": 2,
                    "raw_max_confidence": 0.0231,
                    "threshold_filtered": 2,
                    "threshold_filtered_max_confidence": 0.0231,
                }
            },
        },
    }

    result = engine.judge(
        vision_candidates=[],
        delta_weight=-1047.0,
        active_products=[
            make_active_product(
                75,
                "BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
                weight=520.0,
                stock=581,
                price=2300,
            ),
            make_active_product(
                54,
                "BOTTLE_LOTTE_TREVI_LEMON_500ML",
                weight=530.0,
                stock=584,
                price=1600,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.PARTIAL
    assert {product.product_id: product.count for product in result.products} == {75: 2}
    assert result.weight_explained == 1040.0
    assert result.weight_residual == 7.0
    diagnostics = trace.weight_diagnostics["forced_final_fallback"]
    assert diagnostics["mode"] == "detected_same_product_pair"
    assert diagnostics["pair_support_rank"] == 0


def test_stage_count_supported_mixed_pair_runs_before_forced_repeat():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace({"purchase_delta_candidates": [{"weight": 205.0}]})
    trace.stage_counts_by_class = {
        "101": {
            "class_id": 101,
            "name": "Detected Product",
            "raw": 12,
            "raw_max_confidence": 0.45,
            "cameras": {"side": {"raw": 12, "raw_max_confidence": 0.45}},
        },
        "102": {
            "class_id": 102,
            "name": "Also Supported Product",
            "raw": 10,
            "raw_max_confidence": 0.40,
            "cameras": {"side": {"raw": 10, "raw_max_confidence": 0.40}},
        },
    }

    result = engine.judge(
        vision_candidates=[],
        delta_weight=-205.0,
        active_products=[
            make_active_product(101, "Detected Product", weight=100.0, stock=3),
            make_active_product(102, "Also Supported Product", weight=105.0, stock=3),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert {product.product_id: product.count for product in result.products} == {
        101: 1,
        102: 1,
    }
    assert result.weight_residual == 0.0
    diagnostics = trace.weight_diagnostics["stage_count_combination_match"]
    assert diagnostics["accepted"] is True
    assert trace.weight_diagnostics["decision_branch"] == "stage_count_combination_match"
    assert "forced_final_fallback" not in trace.weight_diagnostics


def test_segment_first_matching_prefers_split_same_product_over_single_aggregate():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace(
        {
            "removal_segment_targets": [
                {"weight": 210.0, "segment_index": 0},
                {"weight": 105.0, "segment_index": 1},
                {"weight": 103.0, "segment_index": 2},
                {"weight": 107.0, "segment_index": 3},
            ]
        }
    )

    result = engine.judge(
        vision_candidates=[
            make_candidate(201, "Coca Cola 530", confidence=0.95),
            make_candidate(101, "Haneul Bori", confidence=0.82),
        ],
        delta_weight=-530.0,
        active_products=[
            make_active_product(101, "Haneul Bori", weight=105.0, stock=10),
            make_active_product(201, "Coca Cola 530", weight=530.0, stock=10),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (101, 5)
    ]
    assert result.weight_residual == 4.0
    assert trace.weight_diagnostics["decision_branch"] == "segment_weight_matching"


def test_aggregate_matching_still_uses_single_product_without_segment_targets():
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(201, "Coca Cola 530", confidence=0.95),
            make_candidate(101, "Haneul Bori", confidence=0.82),
        ],
        delta_weight=-530.0,
        active_products=[
            make_active_product(101, "Haneul Bori", weight=105.0, stock=10),
            make_active_product(201, "Coca Cola 530", weight=530.0, stock=10),
        ],
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (201, 1)
    ]


def test_no_vision_segment_match_returns_partial_instead_of_none():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace(
        {
            "removal_segment_targets": [
                {"weight": 210.0, "segment_index": 0},
                {"weight": 105.0, "segment_index": 1},
                {"weight": 103.0, "segment_index": 2},
                {"weight": 107.0, "segment_index": 3},
            ]
        }
    )

    result = engine.judge(
        vision_candidates=[],
        delta_weight=-530.0,
        active_products=[
            make_active_product(101, "Haneul Bori", weight=105.0, stock=10),
            make_active_product(201, "Coca Cola 530", weight=530.0, stock=10),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.PARTIAL
    assert result.products
    assert [(product.product_id, product.count) for product in result.products] == [
        (101, 5)
    ]


def test_segment_matching_prefers_candidate_supported_repeat_over_active_only_residual():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace(
        {
            "removal_segment_targets": [
                {"weight": 743.4, "segment_index": 0},
                {"weight": 373.6, "segment_index": 1},
            ]
        }
    )

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                8,
                "CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
                confidence=0.2371,
            )
        ],
        delta_weight=-1117.0,
        active_products=[
            make_active_product(
                8,
                "CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
                weight=367.0,
                stock=10,
                price=1300,
            ),
            make_active_product(
                26,
                "CAN_WELCHS_ZERO_GRAPE_355ML",
                weight=371.0,
                stock=10,
                price=1000,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (8, 3)
    ]
    assert result.weight_residual == 16.0
    diagnostics = trace.weight_diagnostics["segment_weight_matching"]
    assert diagnostics["candidate_supported_override"]["accepted"] is True
    assert diagnostics["candidate_supported_override"]["aggregate_residual"] == 16.0
    assert diagnostics["candidate_supported_override"]["allowed_residual"] == 20.0


def test_segment_matching_prefers_rank1_sky_repeat_over_supported_small_fragments():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace(
        {
            "removal_segment_targets": [
                {"weight": 38.0, "segment_index": 0},
                {"weight": 734.0, "segment_index": 1},
                {"weight": 265.4, "segment_index": 2},
            ]
        }
    )
    trace.stage_counts_by_class = {
        "113": {
            "class_id": 113,
            "name": "STICK_INNON_CONDITION_STICK_18G",
            "raw": 20,
            "raw_max_confidence": 0.45,
            "motion_passed": True,
            "final_rank": 5,
        },
        "40": {
            "class_id": 40,
            "name": "BOX_LOTTE_BINCH_102G",
            "raw": 20,
            "raw_max_confidence": 0.42,
            "motion_passed": True,
            "final_rank": 6,
        },
    }

    result = engine.judge(
        vision_candidates=[
            make_candidate(31, "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML", 1.0),
            make_candidate(35, "BAG_NONGSHIM_CHAPAGETTI_140G", 0.447),
            make_candidate(8, "CAN_LOTTE_HOT6_THE_KING_RUSH_355ML", 0.433),
            make_candidate(54, "BOTTLE_LOTTE_TREVI_LEMON_500ML", 0.417),
        ],
        delta_weight=-1037.4,
        active_products=[
            make_active_product(
                31,
                "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML",
                523.0,
                stock=10,
                price=1200,
            ),
            make_active_product(
                35,
                "BAG_NONGSHIM_CHAPAGETTI_140G",
                149.0,
                stock=10,
                price=4000,
            ),
            make_active_product(
                8,
                "CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
                367.0,
                stock=10,
                price=1300,
            ),
            make_active_product(
                54,
                "BOTTLE_LOTTE_TREVI_LEMON_500ML",
                530.0,
                stock=10,
                price=1600,
            ),
            make_active_product(
                113,
                "STICK_INNON_CONDITION_STICK_18G",
                19.0,
                stock=10,
                price=3000,
            ),
            make_active_product(
                40,
                "BOX_LOTTE_BINCH_102G",
                130.0,
                stock=10,
                price=1500,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (31, 2)
    ]
    assert result.weight_explained == 1046.0
    assert result.weight_residual == pytest.approx(8.6)
    diagnostics = trace.weight_diagnostics["segment_weight_matching"]
    override = diagnostics["aggregate_evidence_override"]
    assert override["accepted"] is True
    assert isinstance(override["selected_has_unsupported_small_fragments"], bool)
    assert override["selected"]["class_id"] == 31


def test_segment_matching_prefers_evidenced_trevi_and_king_rush_over_active_only_and_bibigo_override():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace(
        {
            "removal_segment_targets": [
                {"weight": 523.0, "segment_index": 0},
                {"weight": 371.8, "segment_index": 1},
            ]
        }
    )
    trace.stage_counts_by_class = {
        "54": {
            "class_id": 54,
            "name": "BOTTLE_LOTTE_TREVI_LEMON_500ML",
            "raw": 318,
            "raw_max_confidence": 0.8568,
            "threshold_passed": 224,
            "threshold_passed_max_confidence": 0.8568,
            "roi_passed": 3,
            "motion_gate_passed": True,
            "threshold_rescue_candidate": True,
        },
        "8": {
            "class_id": 8,
            "name": "CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
            "raw": 95,
            "raw_max_confidence": 0.295,
            "threshold_passed": 1,
            "threshold_passed_max_confidence": 0.295,
            "roi_passed": 1,
            "motion_gate_passed": True,
            "threshold_rescue_candidate": True,
        },
        "120": {
            "class_id": 120,
            "name": "CUP_BIBIGO_TTEOKBOKKI_110G",
            "raw": 7,
            "raw_max_confidence": 0.4177,
            "threshold_passed": 3,
            "threshold_passed_max_confidence": 0.4177,
            "roi_passed": 3,
            "motion_gate_passed": True,
            "threshold_rescue_candidate": True,
        },
    }

    result = engine.judge(
        vision_candidates=[],
        delta_weight=-894.0,
        active_products=[
            make_active_product(54, "BOTTLE_LOTTE_TREVI_LEMON_500ML", 530.0, 10),
            make_active_product(
                8,
                "CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
                367.0,
                10,
            ),
            make_active_product(120, "CUP_BIBIGO_TTEOKBOKKI_110G", 144.0, 10),
            make_active_product(31, "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML", 523.0, 10),
            make_active_product(26, "CAN_WELCHS_ZERO_GRAPE_355ML", 371.0, 10),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (54, 1),
        (8, 1),
    ]
    diagnostics = trace.weight_diagnostics["segment_weight_matching"]
    assert diagnostics["aggregate_evidence_override"]["accepted"] is False
    assert diagnostics["selections"][0]["class_id"] == 54
    assert diagnostics["selections"][1]["class_id"] == 8
    rejected_by_id = {
        candidate["class_id"]: candidate
        for candidate in diagnostics["aggregate_evidence_override"]["candidates"]
    }
    assert rejected_by_id[120]["reason"] == "insufficient_aggregate_evidence"


def test_segment_matching_keeps_clean_supported_segments_over_aggregate_repeat():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace(
        {
            "removal_segment_targets": [
                {"weight": 129.6, "segment_index": 0},
                {"weight": 219.3, "segment_index": 1},
                {"weight": 229.5, "segment_index": 2},
            ]
        }
    )
    trace.stage_counts_by_class = {
        "35": {
            "class_id": 35,
            "name": "BAG_NONGSHIM_CHAPAGETTI_140G",
            "raw": 142,
            "raw_max_confidence": 0.5889,
            "threshold_passed": 55,
            "threshold_passed_max_confidence": 0.5889,
            "roi_passed": 55,
            "motion_gate_passed": True,
            "cameras": {
                "side": {
                    "raw": 102,
                    "raw_max_confidence": 0.5889,
                    "threshold_passed": 55,
                    "threshold_passed_max_confidence": 0.5889,
                    "roi_passed": 55,
                    "motion_filtered": 55,
                },
                "top": {
                    "raw": 40,
                    "raw_max_confidence": 0.2516,
                },
            },
        },
        "40": {
            "class_id": 40,
            "name": "BOX_LOTTE_BINCH_102G",
            "raw": 179,
            "raw_max_confidence": 0.4668,
            "threshold_passed": 2,
            "threshold_passed_max_confidence": 0.4668,
            "roi_passed": 1,
            "motion_gate_passed": True,
            "cameras": {
                "top": {
                    "raw": 156,
                    "raw_max_confidence": 0.4668,
                    "threshold_passed": 1,
                    "threshold_passed_max_confidence": 0.4668,
                    "roi_filtered": 1,
                },
                "side": {
                    "raw": 23,
                    "raw_max_confidence": 0.3757,
                    "threshold_passed": 1,
                    "threshold_passed_max_confidence": 0.3757,
                    "roi_passed": 1,
                    "motion_filtered": 1,
                },
            },
        },
        "95": {
            "class_id": 95,
            "name": "BOTTLE_HY_HALUYACHE_PURPLE_200ML_V2",
            "raw": 258,
            "raw_max_confidence": 0.8661,
            "threshold_passed": 120,
            "threshold_passed_max_confidence": 0.8661,
            "roi_passed": 120,
            "motion_passed": 120,
            "motion_gate_passed": True,
            "cameras": {
                "side": {
                    "raw": 137,
                    "raw_max_confidence": 0.8661,
                    "threshold_passed": 120,
                    "threshold_passed_max_confidence": 0.8661,
                    "roi_passed": 120,
                    "motion_passed": 120,
                },
                "top": {
                    "raw": 121,
                    "raw_max_confidence": 0.2415,
                },
            },
        },
        "12": {
            "class_id": 12,
            "name": "CAN_LOTTE_LETSBE_MILD_COFFEE_175ML",
            "raw": 34,
            "raw_max_confidence": 0.3337,
            "threshold_passed": 1,
            "threshold_passed_max_confidence": 0.3337,
            "motion_gate_passed": True,
            "cameras": {
                "top": {
                    "raw": 25,
                    "raw_max_confidence": 0.3337,
                    "threshold_passed": 1,
                    "threshold_passed_max_confidence": 0.3337,
                    "roi_filtered": 1,
                },
                "side": {
                    "raw": 9,
                    "raw_max_confidence": 0.2024,
                },
            },
        },
    }

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                95,
                "BOTTLE_HY_HALUYACHE_PURPLE_200ML_V2",
                confidence=0.5197,
            )
        ],
        delta_weight=-580.0,
        active_products=[
            make_active_product(35, "BAG_NONGSHIM_CHAPAGETTI_140G", 149.0, 10, 4000),
            make_active_product(40, "BOX_LOTTE_BINCH_102G", 130.0, 10, 1500),
            make_active_product(
                95,
                "BOTTLE_HY_HALUYACHE_PURPLE_200ML_V2",
                220.0,
                10,
                2000,
            ),
            make_active_product(
                12,
                "CAN_LOTTE_LETSBE_MILD_COFFEE_175ML",
                228.0,
                10,
                1500,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (40, 1),
        (95, 1),
        (12, 1),
    ]
    assert result.weight_residual == 2.6
    diagnostics = trace.weight_diagnostics["segment_weight_matching"]
    assert diagnostics["aggregate_evidence_override"]["accepted"] is False
    assert (
        diagnostics["aggregate_evidence_override"]["reason"]
        == "clean_supported_segment_match_preferred"
    )
    assert diagnostics["aggregate_evidence_override"]["candidate_aggregate_residual"] == 17.6
    assert diagnostics["selected_segment_all_supported"] is True
    assert (
        diagnostics["aggregate_evidence_override"]["selected_segment_summary"][
            "all_segment_residuals_within_tolerance"
        ]
        is True
    )


def test_segment_stage_recovery_matches_full_haluyache_letsbe_jagabee_delta():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace(
        {
            "purchase_delta_candidates": [
                {"weight": 503.0, "source": "net_stable_delta"},
                {"weight": 502.8, "source": "unpaired_negative_total"},
                {"weight": 51.9, "source": "last_unpaired_negative_segment"},
            ],
            "removal_segment_targets": [
                {
                    "source": "unpaired_negative_segment",
                    "weight": 220.9,
                    "delta": -220.9,
                    "segment_index": 0,
                },
                {
                    "source": "unpaired_negative_segment",
                    "weight": 230.0,
                    "delta": -230.0,
                    "segment_index": 1,
                },
                {
                    "source": "unpaired_negative_segment",
                    "weight": 51.9,
                    "delta": -51.9,
                    "segment_index": 2,
                },
            ],
            "channel_removal_segment_targets": [
                {
                    "source": "simultaneous_channel_delta",
                    "weight": 220.0,
                    "delta": -220.0,
                    "segment_index": 0,
                    "channel_index": 0,
                    "evidence_required": True,
                },
                {
                    "source": "simultaneous_channel_delta",
                    "weight": 282.4,
                    "delta": -282.4,
                    "segment_index": 1,
                    "channel_index": 1,
                    "evidence_required": True,
                },
            ],
        }
    )
    trace.stage_counts_by_class = {
        "12": make_stage_count_entry(
            12,
            "CAN_LOTTE_LETSBE_MILD_COFFEE_175ML",
            confidence=0.3234,
            raw=8,
            threshold_passed=1,
            motion_gate_passed=True,
        ),
    }

    result = engine.judge(
        vision_candidates=[
            EnsembleResult(
                class_id=95,
                class_name="BOTTLE_HY_HALUYACHE_PURPLE_200ML_V2",
                top_confidence=0.0,
                side_confidence=0.5577,
                combined_confidence=0.5577,
                vote_count=3,
                source="vision",
                motion_gate_passed=True,
            ),
            EnsembleResult(
                class_id=40,
                class_name="BOX_LOTTE_BINCH_102G",
                top_confidence=0.0,
                side_confidence=0.3903,
                combined_confidence=0.3903,
                vote_count=2,
                source="vision",
                motion_gate_passed=True,
            ),
            EnsembleResult(
                class_id=57,
                class_name="BAG_HAITAI_JAGABEE_45G",
                top_confidence=0.1565,
                side_confidence=0.0,
                combined_confidence=0.1565,
                vote_count=2,
                source="vision",
                motion_gate_passed=True,
            ),
        ],
        delta_weight=-503.0,
        active_products=[
            make_active_product(
                95,
                "BOTTLE_HY_HALUYACHE_PURPLE_200ML_V2",
                220.0,
                10,
                2000,
            ),
            make_active_product(
                12,
                "CAN_LOTTE_LETSBE_MILD_COFFEE_175ML",
                228.0,
                10,
                1500,
            ),
            make_active_product(57, "BAG_HAITAI_JAGABEE_45G", 52.0, 10, 2500),
            make_active_product(40, "BOX_LOTTE_BINCH_102G", 130.0, 10, 1500),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert {product.product_id: product.count for product in result.products} == {
        95: 1,
        12: 1,
        57: 1,
    }
    assert result.weight_explained == 500.0
    assert result.weight_residual == 3.0
    diagnostics = trace.weight_diagnostics["segment_weight_matching"]
    assert diagnostics["target_source"] == "removal_segment_targets"
    assert trace.weight_diagnostics["channel_segment_weight_matching"]["reason"] == (
        "segment_without_valid_option"
    )
    assert trace.weight_diagnostics["final_weight_mismatch_guard"]["accepted"] is True
    assert "forced_final_fallback" not in trace.weight_diagnostics


def test_stage_count_prefers_letsbe_repeat_over_condition_fragment():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace({})
    trace.stage_counts_by_class = {
        "12": make_stage_count_entry(
            12,
            "CAN_LOTTE_LETSBE_MILD_COFFEE_175ML",
            confidence=0.3337,
            raw=72,
            threshold_passed=24,
            motion_gate_passed=True,
        ),
        "113": make_stage_count_entry(
            113,
            "STICK_INNON_CONDITION_STICK_18G",
            confidence=0.12,
            raw=8,
            motion_gate_passed=False,
        ),
    }

    result = engine._try_stage_count_combination_match(
        vision_candidates=[
            make_candidate(
                95,
                "BOTTLE_HY_HALUYACHE_PURPLE_200ML_V2",
                confidence=0.546,
            )
        ],
        delta_weight=-460.0,
        timestamp=123.0,
        active_products=[
            make_active_product(
                95,
                "BOTTLE_HY_HALUYACHE_PURPLE_200ML_V2",
                211.0,
                10,
                2000,
            ),
            make_active_product(
                12,
                "CAN_LOTTE_LETSBE_MILD_COFFEE_175ML",
                228.0,
                10,
                1500,
            ),
            make_active_product(
                113,
                "STICK_INNON_CONDITION_STICK_18G",
                19.0,
                10,
                3000,
            ),
        ],
        trace_context=trace,
    )

    assert result is not None
    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (12, 2)
    ]
    diagnostics = trace.weight_diagnostics["stage_count_combination_match"]
    assert diagnostics["accepted"] is True
    selected_items = diagnostics["selected"]["items"]
    assert len(selected_items) == 1
    assert selected_items[0]["class_id"] == 12
    assert selected_items[0]["count"] == 2
    assert selected_items[0]["source"] == "stage_counts"
    assert (
        diagnostics["strict_diagnostics"]["rejected_combination_counts"][
            "unsupported_small_repeat_fragment"
        ]
        >= 1
    )


def test_weak_condition_repeat_fragment_does_not_complete_basket():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace({})
    trace.stage_counts_by_class = {
        "113": make_stage_count_entry(
            113,
            "STICK_INNON_CONDITION_STICK_18G",
            confidence=0.12,
            raw=8,
            motion_gate_passed=False,
        ),
    }

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                95,
                "BOTTLE_HY_HALUYACHE_PURPLE_200ML_V2",
                confidence=0.546,
            )
        ],
        delta_weight=-460.0,
        active_products=[
            make_active_product(
                95,
                "BOTTLE_HY_HALUYACHE_PURPLE_200ML_V2",
                211.0,
                10,
                2000,
            ),
            make_active_product(
                113,
                "STICK_INNON_CONDITION_STICK_18G",
                19.0,
                10,
                3000,
            ),
        ],
        trace_context=trace,
    )

    assert all(product.product_id != 113 for product in result.products)
    assert not (
        result.status == JudgmentStatus.COMPLETE
        and {product.product_id for product in result.products} == {95, 113}
    )
    diagnostics = trace.weight_diagnostics["stage_count_combination_match"]
    assert diagnostics["accepted"] is False
    assert (
        diagnostics["strict_diagnostics"]["rejected_combination_counts"][
            "unsupported_small_repeat_fragment"
        ]
        >= 1
    )


def test_strong_stage_condition_repeat_is_still_allowed():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace({})
    trace.stage_counts_by_class = {
        "113": make_stage_count_entry(
            113,
            "STICK_INNON_CONDITION_STICK_18G",
            confidence=0.55,
            raw=50,
            threshold_passed=25,
            motion_gate_passed=True,
        ),
    }

    result = engine._try_stage_count_combination_match(
        vision_candidates=[],
        delta_weight=-38.0,
        timestamp=123.0,
        active_products=[
            make_active_product(
                113,
                "STICK_INNON_CONDITION_STICK_18G",
                19.0,
                10,
                3000,
            ),
        ],
        trace_context=trace,
    )

    assert result is not None
    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (113, 2)
    ]


def test_segment_matching_splits_merged_bottle_segment_before_small_repeats():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace(
        {
            "removal_segment_targets": [
                {"weight": 1163.9, "segment_index": 0},
                {"weight": 540.9, "segment_index": 1},
            ]
        }
    )
    trace.stage_counts_by_class = {
        "119": {
            "class_id": 119,
            "name": "BOTTLE_FANTA_ORANGE_600ML",
            "raw": 8,
            "raw_max_confidence": 0.7592,
            "threshold_passed": 5,
            "threshold_passed_max_confidence": 0.7592,
            "roi_passed": 5,
            "motion_passed": 5,
            "motion_gate_passed": True,
            "final_rank": 1,
            "cameras": {
                "side": {
                    "raw": 8,
                    "raw_max_confidence": 0.7592,
                    "threshold_passed": 5,
                    "threshold_passed_max_confidence": 0.7592,
                    "roi_passed": 5,
                    "motion_passed": 5,
                }
            },
        },
        "75": {
            "class_id": 75,
            "name": "BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
            "raw": 123,
            "raw_max_confidence": 0.1097,
            "cameras": {
                "side": {
                    "raw": 121,
                    "raw_max_confidence": 0.1097,
                    "threshold_filtered": 121,
                    "threshold_filtered_max_confidence": 0.1097,
                },
                "top": {
                    "raw": 2,
                    "raw_max_confidence": 0.0169,
                },
            },
        },
        "29": {
            "class_id": 29,
            "name": "BOTTLE_NONGSHIM_BAKSANSOO_500ML_V2",
            "raw": 8,
            "raw_max_confidence": 0.1132,
            "motion_gate_passed": True,
            "cameras": {
                "side": {
                    "raw": 7,
                    "raw_max_confidence": 0.1132,
                },
                "top": {
                    "raw": 1,
                    "raw_max_confidence": 0.0801,
                },
            },
        },
        "23": {
            "class_id": 23,
            "name": "BOTTLE_COCA_POWER_ADE_MOUNTAIN_BLAST_600ML",
            "raw": 1462,
            "raw_max_confidence": 0.5107,
            "threshold_passed": 16,
            "threshold_passed_max_confidence": 0.5107,
            "roi_passed": 5,
            "motion_passed": 5,
            "motion_gate_passed": True,
            "final_rank": 2,
            "cameras": {
                "side": {
                    "raw": 44,
                    "raw_max_confidence": 0.5107,
                    "threshold_passed": 5,
                    "threshold_passed_max_confidence": 0.5107,
                    "roi_passed": 5,
                    "motion_passed": 5,
                },
                "top": {
                    "raw": 1418,
                    "raw_max_confidence": 0.4766,
                    "threshold_passed": 11,
                    "threshold_passed_max_confidence": 0.4766,
                    "hand_path_passed": 1,
                },
            },
        },
        "114": {
            "class_id": 114,
            "name": "BOX_LOTTE_PEPERO_ORIGINAL_46G",
            "raw": 6,
            "raw_max_confidence": 0.7853,
            "threshold_passed": 4,
            "threshold_passed_max_confidence": 0.7853,
            "roi_passed": 2,
            "motion_passed": 2,
            "motion_gate_passed": True,
            "final_rank": 3,
            "cameras": {
                "side": {
                    "raw": 3,
                    "raw_max_confidence": 0.7853,
                    "threshold_passed": 2,
                    "roi_passed": 2,
                    "motion_passed": 2,
                }
            },
        },
        "35": {
            "class_id": 35,
            "name": "BAG_NONGSHIM_CHAPAGETTI_140G",
            "raw": 28,
            "raw_max_confidence": 0.4766,
            "threshold_passed": 9,
            "threshold_passed_max_confidence": 0.4766,
            "motion_gate_passed": True,
        },
    }

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                119,
                "BOTTLE_FANTA_ORANGE_600ML",
                confidence=0.4555,
            ),
            make_candidate(
                23,
                "BOTTLE_COCA_POWER_ADE_MOUNTAIN_BLAST_600ML",
                confidence=0.3064,
            ),
            make_candidate(
                114,
                "BOX_LOTTE_PEPERO_ORIGINAL_46G",
                confidence=0.2103,
            ),
        ],
        delta_weight=-1705.0,
        active_products=[
            make_active_product(119, "BOTTLE_FANTA_ORANGE_600ML", 634.0, 10, 2000),
            make_active_product(
                75,
                "BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
                520.0,
                10,
                2300,
            ),
            make_active_product(
                29,
                "BOTTLE_NONGSHIM_BAKSANSOO_500ML_V2",
                539.0,
                10,
                1000,
            ),
            make_active_product(
                23,
                "BOTTLE_COCA_POWER_ADE_MOUNTAIN_BLAST_600ML",
                639.0,
                10,
                6000,
            ),
            make_active_product(35, "BAG_NONGSHIM_CHAPAGETTI_140G", 149.0, 10, 4000),
            make_active_product(114, "BOX_LOTTE_PEPERO_ORIGINAL_46G", 66.0, 10, 2500),
        ],
        trace_context=trace,
    )

    assert [(product.product_id, product.count) for product in result.products] == [
        (119, 1),
        (75, 1),
        (29, 1),
    ]
    assert result.weight_residual == 11.8
    diagnostics = trace.weight_diagnostics["segment_weight_matching"]
    assert diagnostics["selections"][0]["option_kind"] == "compound"
    assert [
        (item["class_id"], item["count"])
        for item in diagnostics["selections"][0]["items"]
    ] == [(119, 1), (75, 1)]
    assert diagnostics["selections"][0]["selection_reason"] == (
        "trusted_compound_segment_split"
    )
    assert diagnostics["selections"][1]["class_id"] == 29
    rejected_small_repeats = [
        option
        for segment in diagnostics["segment_options"]
        for option in segment["top_options"] + segment.get("rejected_options", [])
        if option.get("rejected_reason")
        in {
            "trusted_or_single_item_segment_preferred",
            "count_exceeds_segment_grip_limit",
        }
    ]
    assert {option["class_id"] for option in rejected_small_repeats} >= {35, 114}


def test_segment_grip_limit_rejects_single_segment_pepero_x8_candidate():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace(
        {
            "removal_segment_targets": [
                {"weight": 528.0, "segment_index": 0},
            ]
        }
    )

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                114,
                "BOX_LOTTE_PEPERO_ORIGINAL_46G",
                confidence=0.72,
            )
        ],
        delta_weight=-528.0,
        active_products=[
            make_active_product(114, "BOX_LOTTE_PEPERO_ORIGINAL_46G", 66.0, 20, 2500),
            make_active_product(29, "BOTTLE_NONGSHIM_BAKSANSOO_500ML_V2", 530.0, 10, 1000),
        ],
        trace_context=trace,
    )

    assert [(product.product_id, product.count) for product in result.products] != [
        (114, 8)
    ]
    diagnostics = trace.weight_diagnostics["same_product_count_match"]
    assert diagnostics["segment_grip_limit"] == 3
    pepero_diag = next(
        candidate
        for candidate in diagnostics["candidates"]
        if candidate["class_id"] == 114
    )
    assert pepero_diag["nearest_count"] == 8
    assert pepero_diag["reason"] == "count_exceeds_segment_grip_limit"


def test_segment_grip_limit_allows_three_items_per_detected_segment():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace(
        {
            "removal_segment_targets": [
                {"weight": 198.0, "segment_index": 0},
                {"weight": 198.0, "segment_index": 1},
            ]
        }
    )

    result = engine.judge(
        vision_candidates=[],
        delta_weight=-396.0,
        active_products=[
            make_active_product(114, "BOX_LOTTE_PEPERO_ORIGINAL_46G", 66.0, 20, 2500),
        ],
        trace_context=trace,
    )

    assert [(product.product_id, product.count) for product in result.products] == [
        (114, 6)
    ]
    diagnostics = trace.weight_diagnostics["segment_weight_matching"]
    assert diagnostics["max_items_per_segment"] == 3
    assert [selection["count"] for selection in diagnostics["selections"]] == [3, 3]


def test_segment_aggregate_override_respects_total_grip_limit():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace(
        {
            "removal_segment_targets": [
                {"weight": 264.0, "segment_index": 0},
                {"weight": 264.0, "segment_index": 1},
            ]
        }
    )

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                114,
                "BOX_LOTTE_PEPERO_ORIGINAL_46G",
                confidence=0.72,
            )
        ],
        delta_weight=-528.0,
        active_products=[
            make_active_product(114, "BOX_LOTTE_PEPERO_ORIGINAL_46G", 66.0, 20, 2500),
            make_active_product(201, "BOTTLE_TEST_264G", 264.0, 10, 1000),
        ],
        trace_context=trace,
    )

    assert [(product.product_id, product.count) for product in result.products] == [
        (201, 2)
    ]
    diagnostics = trace.weight_diagnostics["segment_weight_matching"]
    override = diagnostics["aggregate_evidence_override"]
    assert override["accepted"] is False
    assert override["segment_grip_limit"] == 6
    pepero_diag = next(
        candidate
        for candidate in override["candidates"]
        if candidate["class_id"] == 114
    )
    assert pepero_diag["nearest_count"] == 8
    assert pepero_diag["reason"] == "count_exceeds_segment_grip_limit"


def test_segment_matching_uses_camera_aware_stage_score_for_welchs_over_top_raw_cupban():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace(
        {
            "removal_segment_targets": [
                {"weight": 370.7, "segment_index": 0},
                {"weight": 374.1, "segment_index": 1},
            ]
        }
    )
    trace.stage_counts_by_class = {
        "26": {
            "class_id": 26,
            "name": "CAN_WELCHS_ZERO_GRAPE_355ML",
            "raw": 10,
            "raw_max_confidence": 0.3748,
            "threshold_passed": 2,
            "threshold_passed_max_confidence": 0.3748,
            "roi_filtered": 1,
            "roi_filtered_max_confidence": 0.3007,
            "cameras": {
                "side": {
                    "raw": 10,
                    "raw_max_confidence": 0.3748,
                    "threshold_passed": 2,
                    "threshold_passed_max_confidence": 0.3748,
                    "roi_filtered": 1,
                    "roi_filtered_max_confidence": 0.3007,
                    "motion_filtered": 1,
                }
            },
        },
        "100": {
            "class_id": 100,
            "name": "CUP_CJ_HATBAN_CUPBAN_CHICKEN_MAYO_313G",
            "raw": 43,
            "raw_max_confidence": 0.0927,
            "threshold_filtered": 43,
            "threshold_filtered_max_confidence": 0.0927,
            "cameras": {
                "top": {
                    "raw": 43,
                    "raw_max_confidence": 0.0927,
                    "threshold_filtered": 43,
                    "threshold_filtered_max_confidence": 0.0927,
                }
            },
        },
    }

    result = engine.judge(
        vision_candidates=[],
        delta_weight=-744.4,
        active_products=[
            make_active_product(26, "CAN_WELCHS_ZERO_GRAPE_355ML", 371.0, 10),
            make_active_product(
                100,
                "CUP_CJ_HATBAN_CUPBAN_CHICKEN_MAYO_313G",
                365.0,
                10,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (26, 2)
    ]
    assert result.weight_residual == 3.4
    diagnostics = trace.weight_diagnostics["segment_weight_matching"]
    assert [selection["class_id"] for selection in diagnostics["selections"]] == [
        26,
        26,
    ]
    welchs_option = next(
        option
        for option in diagnostics["segment_options"][0]["top_options"]
        if option["class_id"] == 26
    )
    cupban_option = next(
        option
        for option in diagnostics["segment_options"][0]["top_options"]
        if option["class_id"] == 100
    )
    assert welchs_option["stage_score"] > cupban_option["stage_score"]
    assert welchs_option["side_confidence"] > cupban_option["side_confidence"]


def _fragmented_trevi_trace() -> FakeLoadcellTrace:
    trace = FakeLoadcellTrace(
        {
            "removal_segment_targets": [
                {"weight": 243.3, "segment_index": 0},
                {"weight": 276.5, "segment_index": 1},
                {"weight": 522.3, "segment_index": 2},
                {"weight": 533.9, "segment_index": 3},
                {"weight": 523.0, "segment_index": 4},
                {"weight": 184.7, "segment_index": 5},
                {"weight": 340.8, "segment_index": 6},
            ]
        }
    )
    trace.stage_counts_by_class = {
        "54": {
            "class_id": 54,
            "name": "BOTTLE_LOTTE_TREVI_LEMON_500ML",
            "raw": 203,
            "raw_max_confidence": 0.7295,
            "threshold_passed": 90,
            "threshold_passed_max_confidence": 0.7295,
            "roi_passed": 5,
            "roi_filtered": 85,
            "roi_filtered_max_confidence": 0.6689,
            "motion_gate_passed": True,
        },
        "75": {
            "class_id": 75,
            "name": "BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
            "raw": 54,
            "raw_max_confidence": 0.5651,
            "threshold_passed": 48,
            "threshold_passed_max_confidence": 0.5651,
            "roi_passed": 40,
            "motion_gate_passed": True,
        },
        "44": {
            "class_id": 44,
            "name": "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
            "raw": 155,
            "raw_max_confidence": 0.5458,
            "threshold_passed": 104,
            "threshold_passed_max_confidence": 0.5458,
            "threshold_rescue_candidate": True,
            "motion_gate_passed": False,
        },
    }
    trace.diagnostic_detections = [
        {
            "class_id": 54,
            "name": "BOTTLE_LOTTE_TREVI_LEMON_500ML",
            "confidence": 0.4388,
            "camera": "side",
            "frame_index": index,
        }
        for index in range(7)
    ] + [
        {
            "class_id": 75,
            "name": "BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
            "confidence": 0.5564,
            "camera": "side",
            "frame_index": index,
        }
        for index in range(3)
    ]
    return trace


def _fragmented_trevi_active_products() -> list[MockActiveProduct]:
    return [
        make_active_product(54, "BOTTLE_LOTTE_TREVI_LEMON_500ML", 530.0, 10, 1600),
        make_active_product(
            75,
            "BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
            520.0,
            10,
            2300,
        ),
        make_active_product(
            44,
            "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
            520.0,
            10,
            2000,
        ),
        make_active_product(31, "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML", 523.0, 10),
        make_active_product(38, "BAG_HAITAI_HOME_RUN_BALL_41G", 64.0, 10),
        make_active_product(40, "BOX_LOTTE_BINCH_102G", 130.0, 10),
        make_active_product(57, "BAG_HAITAI_JAGABEE_45G", 52.0, 10),
        make_active_product(113, "STICK_INNON_CONDITION_STICK_18G", 19.0, 10),
        make_active_product(114, "BOX_LOTTE_PEPERO_ORIGINAL_46G", 66.0, 10),
    ]


def test_segment_matching_prefers_strong_aggregate_evidence_for_collision_fragments():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = _fragmented_trevi_trace()

    result = engine.judge(
        vision_candidates=[],
        delta_weight=-2625.0,
        active_products=_fragmented_trevi_active_products(),
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (54, 5)
    ]
    assert result.weight_residual == 25.5
    diagnostics = trace.weight_diagnostics["segment_weight_matching"]
    assert diagnostics["aggregate_evidence_override"]["accepted"] is True
    assert diagnostics["aggregate_evidence_override"]["selected"]["class_id"] == 54
    assert (
        diagnostics["aggregate_evidence_override"]["selected"]["aggregate_residual"]
        == 25.5
    )
    assert (
        diagnostics["aggregate_evidence_override"]["selected"]["allowed_residual"]
        == 30.0
    )
    rejected_by_id = {
        candidate["class_id"]: candidate
        for candidate in diagnostics["aggregate_evidence_override"]["candidates"]
    }
    assert rejected_by_id[44]["reason"] == "insufficient_aggregate_evidence"


def test_segment_matching_does_not_use_low_diagnostic_evidence_for_aggregate_override():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = _fragmented_trevi_trace()
    trace.stage_counts_by_class = {}
    trace.diagnostic_detections = [
        {
            "class_id": 54,
            "name": "BOTTLE_LOTTE_TREVI_LEMON_500ML",
            "confidence": 0.29,
            "camera": "side",
            "frame_index": index,
        }
        for index in range(4)
    ]

    result = engine.judge(
        vision_candidates=[],
        delta_weight=-2625.0,
        active_products=_fragmented_trevi_active_products(),
        trace_context=trace,
    )

    assert [(product.product_id, product.count) for product in result.products] != [
        (54, 5)
    ]
    diagnostics = trace.weight_diagnostics["segment_weight_matching"]
    assert diagnostics["aggregate_evidence_override"]["accepted"] is False


def test_segment_matching_uses_strong_diagnostic_only_aggregate_as_partial():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = _fragmented_trevi_trace()
    trace.stage_counts_by_class = {}

    result = engine.judge(
        vision_candidates=[],
        delta_weight=-2625.0,
        active_products=_fragmented_trevi_active_products(),
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.PARTIAL
    assert [(product.product_id, product.count) for product in result.products] == [
        (54, 5)
    ]
    diagnostics = trace.weight_diagnostics["segment_weight_matching"]
    assert diagnostics["aggregate_evidence_override"]["accepted"] is True
    assert diagnostics["aggregate_evidence_override"]["selected"]["status"] == "partial"


def test_segment_matching_prefers_active_bottle_over_weak_small_item_repeats():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace(
        {
            "removal_segment_targets": [
                {"weight": 521.0, "segment_index": 0},
                {"weight": 521.0, "segment_index": 1},
            ]
        }
    )
    trace.stage_counts_by_class = {
        "114": {
            "class_id": 114,
            "name": "BOX_LOTTE_PEPERO_ORIGINAL_46G",
            "raw": 2,
            "raw_max_confidence": 0.019,
            "cameras": {"top": {"raw": 2, "raw_max_confidence": 0.019}},
        }
    }

    result = engine.judge(
        vision_candidates=[],
        delta_weight=-1042.0,
        active_products=[
            make_active_product(
                75,
                "BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
                520.0,
                stock=10,
                price=2300,
            ),
            make_active_product(
                114,
                "BOX_LOTTE_PEPERO_ORIGINAL_46G",
                66.0,
                stock=20,
                price=2500,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.PARTIAL
    assert [(product.product_id, product.count) for product in result.products] == [
        (75, 2)
    ]
    assert result.weight_explained == 1040.0
    assert result.weight_residual == 2.0
    diagnostics = trace.weight_diagnostics["segment_weight_matching"]
    assert diagnostics["products"] == [
        {
            "class_id": 75,
            "name": "BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
            "count": 2,
            "unit_weight": 520.0,
        }
    ]


def test_segment_matching_prefers_active_sky_barley_over_weak_binch_repeats():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace(
        {
            "removal_segment_targets": [
                {"weight": 522.7, "segment_index": 0},
                {"weight": 522.7, "segment_index": 1},
            ]
        }
    )
    trace.stage_counts_by_class = {
        "40": {
            "class_id": 40,
            "name": "BOX_LOTTE_BINCH_102G",
            "raw": 20,
            "raw_max_confidence": 0.1694,
            "cameras": {"top": {"raw": 20, "raw_max_confidence": 0.1694}},
        }
    }

    result = engine.judge(
        vision_candidates=[],
        delta_weight=-1045.4,
        active_products=[
            make_active_product(
                31,
                "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML",
                523.0,
                stock=10,
                price=2000,
            ),
            make_active_product(
                40,
                "BOX_LOTTE_BINCH_102G",
                130.0,
                stock=20,
                price=1500,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.PARTIAL
    assert [(product.product_id, product.count) for product in result.products] == [
        (31, 2)
    ]
    assert result.weight_explained == 1046.0
    assert result.weight_residual == 0.6


def test_strict_single_match_prefers_regular_rank1_over_rescue_residual_edge():
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(31, "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML", 0.417),
            make_candidate(23, "BOTTLE_COCA_POWER_ADE_MOUNTAIN_BLAST_600ML", 0.336),
            EnsembleResult(
                class_id=54,
                class_name="BOTTLE_LOTTE_TREVI_LEMON_500ML",
                top_confidence=0.0,
                side_confidence=0.082,
                combined_confidence=0.082,
                vote_count=3,
                source="threshold_rescue",
                raw_vote_count=3,
                side_motion_passed=True,
            ),
        ],
        delta_weight=-527.0,
        active_products=[
            make_active_product(
                31,
                "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML",
                523.0,
                10,
                1200,
            ),
            make_active_product(
                23,
                "BOTTLE_COCA_POWER_ADE_MOUNTAIN_BLAST_600ML",
                639.0,
                10,
                6000,
            ),
            make_active_product(
                54,
                "BOTTLE_LOTTE_TREVI_LEMON_500ML",
                530.0,
                10,
                1600,
            ),
        ],
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (31, 1)
    ]
    assert result.weight_residual == 4.0


def test_same_weight_candidate_guard_prefers_regular_pepsi_over_rescue_and_active_collision():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace({})

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                75,
                "BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
                0.235,
            ),
            EnsembleResult(
                class_id=31,
                class_name="BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML",
                top_confidence=0.0,
                side_confidence=0.068,
                combined_confidence=0.068,
                vote_count=2,
                source="threshold_rescue",
                raw_vote_count=2,
                side_motion_passed=True,
            ),
        ],
        delta_weight=-524.0,
        active_products=[
            make_active_product(
                75,
                "BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
                520.0,
                10,
                2300,
            ),
            make_active_product(
                44,
                "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                520.0,
                10,
                2000,
            ),
            make_active_product(
                31,
                "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML",
                523.0,
                10,
                1200,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (75, 1)
    ]
    assert result.weight_residual == 4.0
    diagnostics = trace.weight_diagnostics["same_weight_candidate_collision"]
    assert diagnostics["accepted"] is True
    assert diagnostics["selected"]["class_id"] == 75
    assert diagnostics["rejected_best_strict"]["class_id"] == 31


def test_single_regular_pepsi_outside_strict_does_not_override_trevi_weight_gate_rescue():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace({})

    result = engine.judge(
        vision_candidates=[
            EnsembleResult(
                class_id=75,
                class_name="BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
                top_confidence=0.0,
                side_confidence=0.32,
                combined_confidence=0.32,
                vote_count=12,
                source="vision",
                raw_vote_count=12,
            ),
            EnsembleResult(
                class_id=54,
                class_name="BOTTLE_LOTTE_TREVI_LEMON_500ML",
                top_confidence=0.1446,
                side_confidence=0.1446,
                combined_confidence=0.1446,
                vote_count=7,
                source="threshold_rescue",
                raw_vote_count=7,
                side_motion_passed=True,
                weight_gate_passed=True,
            ),
        ],
        delta_weight=-529.0,
        active_products=[
            make_active_product(
                75,
                "BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
                520.0,
                10,
                2300,
            ),
            make_active_product(
                54,
                "BOTTLE_LOTTE_TREVI_LEMON_500ML",
                530.0,
                10,
                1600,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (54, 1)
    ]
    assert result.weight_residual == 1.0
    diagnostics = trace.weight_diagnostics["same_weight_candidate_collision"]
    assert diagnostics["accepted"] is False
    pepsi_diag = next(
        candidate for candidate in diagnostics["candidates"]
        if candidate["class_id"] == 75
    )
    assert pepsi_diag["residual"] == 9.0
    assert pepsi_diag["allowed_residual"] == 5.0
    assert pepsi_diag["reason"] == "residual_exceeds_candidate_guard_tolerance"


def test_single_regular_trevi_outside_strict_does_not_override_corn_weight_gate_rescue():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace({})

    result = engine.judge(
        vision_candidates=[
            EnsembleResult(
                class_id=54,
                class_name="BOTTLE_LOTTE_TREVI_LEMON_500ML",
                top_confidence=0.0,
                side_confidence=0.527,
                combined_confidence=0.527,
                vote_count=14,
                source="vision",
                raw_vote_count=14,
                side_motion_passed=True,
            ),
            EnsembleResult(
                class_id=44,
                class_name="BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                top_confidence=0.0,
                side_confidence=0.097,
                combined_confidence=0.097,
                vote_count=4,
                source="threshold_rescue",
                raw_vote_count=4,
                side_motion_passed=True,
                weight_gate_passed=True,
            ),
        ],
        delta_weight=-521.0,
        active_products=[
            make_active_product(54, "BOTTLE_LOTTE_TREVI_LEMON_500ML", 530.0, 10),
            make_active_product(
                44,
                "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                520.0,
                10,
                2000,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (44, 1)
    ]
    assert result.weight_residual == 1.0
    diagnostics = trace.weight_diagnostics["same_weight_candidate_collision"]
    assert diagnostics["accepted"] is False
    trevi_diag = next(
        candidate for candidate in diagnostics["candidates"]
        if candidate["class_id"] == 54
    )
    assert trevi_diag["residual"] == 9.0
    assert trevi_diag["allowed_residual"] == 5.0


def test_same_weight_candidate_guard_keeps_true_pepsi_x2_without_return_hint():
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                75,
                "BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
                0.235,
            ),
            EnsembleResult(
                class_id=31,
                class_name="BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML",
                top_confidence=0.0,
                side_confidence=0.068,
                combined_confidence=0.068,
                vote_count=2,
                source="threshold_rescue",
                raw_vote_count=2,
                side_motion_passed=True,
            ),
        ],
        delta_weight=-1046.0,
        active_products=[
            make_active_product(
                75,
                "BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
                520.0,
                10,
                2300,
            ),
            make_active_product(
                44,
                "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                520.0,
                10,
                2000,
            ),
            make_active_product(
                31,
                "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML",
                523.0,
                10,
                1200,
            ),
        ],
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (75, 2)
    ]
    assert result.weight_residual == 6.0


def test_stage_weight_gate_does_not_override_higher_rank_strict_single_candidate():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace({})
    trace.stage_counts_by_class = {
        "75": {
            "class_id": 75,
            "name": "BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
            "raw": 211,
            "raw_max_confidence": 0.2173,
            "threshold_filtered": 211,
            "threshold_filtered_max_confidence": 0.2173,
            "cameras": {
                "side": {
                    "raw": 148,
                    "raw_max_confidence": 0.2173,
                    "threshold_filtered": 148,
                    "threshold_filtered_max_confidence": 0.2173,
                },
                "top": {
                    "raw": 63,
                    "raw_max_confidence": 0.0781,
                    "threshold_filtered": 63,
                    "threshold_filtered_max_confidence": 0.0781,
                },
            },
            "weight_gate_passed": True,
            "motion_gate_passed": True,
        }
    }

    result = engine.judge(
        vision_candidates=[
            EnsembleResult(
                class_id=31,
                class_name="BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML",
                top_confidence=0.0,
                side_confidence=0.18,
                combined_confidence=0.18,
                vote_count=75,
                source="threshold_rescue",
                raw_vote_count=75,
                side_motion_passed=True,
                weight_gate_passed=True,
            ),
            EnsembleResult(
                class_id=44,
                class_name="BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                top_confidence=0.0,
                side_confidence=0.0573,
                combined_confidence=0.0573,
                vote_count=7,
                source="threshold_rescue",
                raw_vote_count=7,
                side_motion_passed=True,
                weight_gate_passed=True,
            ),
        ],
        delta_weight=-523.0,
        active_products=[
            make_active_product(
                75,
                "BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
                520.0,
                10,
                2300,
            ),
            make_active_product(31, "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML", 523.0, 10),
            make_active_product(
                44,
                "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                520.0,
                10,
                2000,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (31, 1)
    ]
    assert result.weight_residual == 0.0
    diagnostics = trace.weight_diagnostics["stage_weight_gate_candidates"]
    assert diagnostics["accepted"] is True
    assert diagnostics["candidates"][0]["class_id"] == 75
    priority = trace.weight_diagnostics["strict_candidate_priority_selection"]
    assert priority["reason"] == "strict_match"
    assert priority["selected"]["items"][0]["class_id"] == 31


def test_stage_weight_gate_recovers_when_no_higher_rank_strict_single_candidate():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace({})
    trace.stage_counts_by_class = {
        "75": {
            "class_id": 75,
            "name": "BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
            "raw": 211,
            "raw_max_confidence": 0.2173,
            "threshold_filtered": 211,
            "threshold_filtered_max_confidence": 0.2173,
            "cameras": {
                "side": {
                    "raw": 148,
                    "raw_max_confidence": 0.2173,
                    "threshold_filtered": 148,
                    "threshold_filtered_max_confidence": 0.2173,
                },
                "top": {
                    "raw": 63,
                    "raw_max_confidence": 0.0781,
                    "threshold_filtered": 63,
                    "threshold_filtered_max_confidence": 0.0781,
                },
            },
            "weight_gate_passed": True,
            "motion_gate_passed": True,
        }
    }

    result = engine.judge(
        vision_candidates=[
            EnsembleResult(
                class_id=8,
                class_name="CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
                top_confidence=0.0,
                side_confidence=0.451,
                combined_confidence=0.451,
                vote_count=1,
                source="vision",
                raw_vote_count=212,
                side_motion_passed=True,
            )
        ],
        delta_weight=-523.0,
        active_products=[
            make_active_product(8, "CAN_LOTTE_HOT6_THE_KING_RUSH_355ML", 367.0, 10),
            make_active_product(
                75,
                "BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
                520.0,
                10,
                2300,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (75, 1)
    ]
    assert result.weight_residual == 3.0
    diagnostics = trace.weight_diagnostics["stage_weight_gate_candidates"]
    assert diagnostics["accepted"] is True
    priority = trace.weight_diagnostics["strict_candidate_priority_selection"]
    assert priority["selected"]["items"][0]["class_id"] == 75


def test_ranked_single_candidate_priority_keeps_sky_over_lower_rank_stage_trevi():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace({})
    trace.stage_counts_by_class = {
        "54": {
            "class_id": 54,
            "name": "BOTTLE_LOTTE_TREVI_LEMON_500ML",
            "raw": 12,
            "raw_max_confidence": 0.1624,
            "threshold_filtered": 12,
            "threshold_filtered_max_confidence": 0.1624,
            "weight_gate_passed": True,
            "motion_gate_passed": True,
        }
    }

    result = engine.judge(
        vision_candidates=[
            EnsembleResult(
                class_id=8,
                class_name="CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
                top_confidence=0.0,
                side_confidence=0.4512,
                combined_confidence=0.4512,
                vote_count=1,
                source="vision",
                raw_vote_count=212,
                side_motion_passed=True,
            ),
            EnsembleResult(
                class_id=31,
                class_name="BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML",
                top_confidence=0.0,
                side_confidence=0.1694,
                combined_confidence=0.1694,
                vote_count=1,
                source="threshold_rescue",
                raw_vote_count=2,
                side_motion_passed=True,
                weight_gate_passed=True,
            ),
            EnsembleResult(
                class_id=54,
                class_name="BOTTLE_LOTTE_TREVI_LEMON_500ML",
                top_confidence=0.1624,
                side_confidence=0.1624,
                combined_confidence=0.1624,
                vote_count=12,
                source="threshold_rescue",
                raw_vote_count=12,
                top_motion_passed=True,
                side_motion_passed=True,
                weight_gate_passed=True,
            ),
            EnsembleResult(
                class_id=44,
                class_name="BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                top_confidence=0.0,
                side_confidence=0.048,
                combined_confidence=0.048,
                vote_count=1,
                source="threshold_rescue",
                raw_vote_count=12,
                side_motion_passed=True,
                weight_gate_passed=True,
            ),
        ],
        delta_weight=-525.0,
        active_products=[
            make_active_product(8, "CAN_LOTTE_HOT6_THE_KING_RUSH_355ML", 367.0, 10),
            make_active_product(31, "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML", 523.0, 10),
            make_active_product(54, "BOTTLE_LOTTE_TREVI_LEMON_500ML", 530.0, 10),
            make_active_product(
                44,
                "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                520.0,
                10,
                2000,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (31, 1)
    ]
    assert result.weight_residual == 2.0
    diagnostics = trace.weight_diagnostics["stage_weight_gate_candidates"]
    trevi_diag = next(
        candidate for candidate in diagnostics["candidates"]
        if candidate["class_id"] == 54
    )
    assert trevi_diag["reason"] == "upgraded_existing_candidate"
    priority = trace.weight_diagnostics["strict_candidate_priority_selection"]
    assert priority["selected"]["items"][0]["class_id"] == 31


def test_regular_pepsi_candidate_beats_same_weight_stage_gate_corn_collision():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace({})
    trace.stage_counts_by_class = {
        "44": {
            "class_id": 44,
            "name": "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
            "raw": 18,
            "raw_max_confidence": 0.1476,
            "threshold_filtered": 18,
            "threshold_filtered_max_confidence": 0.1476,
            "cameras": {
                "top": {
                    "raw": 18,
                    "raw_max_confidence": 0.1476,
                    "threshold_filtered": 18,
                    "threshold_filtered_max_confidence": 0.1476,
                }
            },
            "weight_gate_passed": True,
            "motion_gate_passed": True,
        }
    }

    result = engine.judge(
        vision_candidates=[
            EnsembleResult(
                class_id=75,
                class_name="BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
                top_confidence=0.0,
                side_confidence=0.2691,
                combined_confidence=0.2691,
                vote_count=6,
                source="vision",
                raw_vote_count=653,
                side_motion_passed=True,
            )
        ],
        delta_weight=-520.0,
        active_products=[
            make_active_product(
                75,
                "BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
                520.0,
                10,
                2300,
            ),
            make_active_product(
                44,
                "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                520.0,
                10,
                2000,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (75, 1)
    ]
    diagnostics = trace.weight_diagnostics["stage_weight_gate_candidates"]
    assert diagnostics["accepted"] is True
    same_weight = trace.weight_diagnostics["same_weight_candidate_collision"]
    assert same_weight["accepted"] is True
    assert same_weight["selected"]["class_id"] == 75


def test_rank1_sky_plus_fanta_beats_lower_rank_corn_plus_fanta_residual_edge():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace({})

    result = engine.judge(
        vision_candidates=[
            EnsembleResult(
                class_id=31,
                class_name="BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML",
                top_confidence=0.406,
                side_confidence=0.0,
                combined_confidence=0.406,
                vote_count=5,
                source="vision",
                top_motion_passed=True,
            ),
            EnsembleResult(
                class_id=44,
                class_name="BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                top_confidence=0.0,
                side_confidence=0.369,
                combined_confidence=0.369,
                vote_count=5,
                source="vision",
                side_motion_passed=True,
            ),
            EnsembleResult(
                class_id=54,
                class_name="BOTTLE_LOTTE_TREVI_LEMON_500ML",
                top_confidence=0.368,
                side_confidence=0.0,
                combined_confidence=0.368,
                vote_count=5,
                source="vision",
                top_motion_passed=True,
            ),
            EnsembleResult(
                class_id=119,
                class_name="BOTTLE_FANTA_ORANGE_600ML",
                top_confidence=0.0,
                side_confidence=0.365,
                combined_confidence=0.365,
                vote_count=5,
                source="vision",
                side_motion_passed=True,
            ),
        ],
        delta_weight=-1151.0,
        active_products=[
            make_active_product(31, "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML", 523.0, 10),
            make_active_product(
                44,
                "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                520.0,
                10,
            ),
            make_active_product(54, "BOTTLE_LOTTE_TREVI_LEMON_500ML", 530.0, 10),
            make_active_product(119, "BOTTLE_FANTA_ORANGE_600ML", 634.0, 10),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (31, 1),
        (119, 1),
    ]
    assert result.weight_explained == 1157.0
    assert result.weight_residual == 6.0
    diagnostics = trace.weight_diagnostics["candidate_priority_combination_grace"]
    assert diagnostics["accepted"] is True
    assert diagnostics["selected"]["items"][0]["class_id"] == 31


def test_regular_pepsi_x2_beats_trevi_rescue_repeat_residual_edge():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace({})

    result = engine.judge(
        vision_candidates=[
            EnsembleResult(
                class_id=75,
                class_name="BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
                top_confidence=0.0,
                side_confidence=0.472,
                combined_confidence=0.472,
                vote_count=12,
                source="vision",
                raw_vote_count=12,
                side_motion_passed=True,
            ),
            EnsembleResult(
                class_id=54,
                class_name="BOTTLE_LOTTE_TREVI_LEMON_500ML",
                top_confidence=0.0,
                side_confidence=0.20,
                combined_confidence=0.20,
                vote_count=20,
                source="threshold_rescue",
                raw_vote_count=20,
                side_motion_passed=True,
                weight_gate_passed=True,
            ),
        ],
        delta_weight=-1058.0,
        active_products=[
            make_active_product(
                75,
                "BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
                520.0,
                10,
                2300,
            ),
            make_active_product(54, "BOTTLE_LOTTE_TREVI_LEMON_500ML", 530.0, 10),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (75, 2)
    ]
    assert result.weight_explained == 1040.0
    assert result.weight_residual == 18.0
    diagnostics = trace.weight_diagnostics["same_weight_candidate_collision"]
    assert diagnostics["accepted"] is True
    assert diagnostics["selected"]["class_id"] == 75
    assert diagnostics["selected"]["count"] == 2


def test_forced_final_fallback_prefers_regular_pepsi_repeat_over_active_trevi_pair(
    monkeypatch,
):
    use_weight_aware_identity(monkeypatch)
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace({})

    result = engine.judge(
        vision_candidates=[
            EnsembleResult(
                class_id=75,
                class_name="BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
                top_confidence=0.0,
                side_confidence=0.472,
                combined_confidence=0.472,
                vote_count=12,
                source="vision",
                raw_vote_count=12,
                side_motion_passed=True,
            )
        ],
        delta_weight=-1058.0,
        active_products=[
            make_active_product(
                75,
                "BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
                520.0,
                10,
                2300,
            ),
            make_active_product(54, "BOTTLE_LOTTE_TREVI_LEMON_500ML", 530.0, 10),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.PARTIAL
    assert [(product.product_id, product.count) for product in result.products] == [
        (75, 2)
    ]
    assert result.weight_explained == 1040.0
    assert result.weight_residual == 18.0
    diagnostics = trace.weight_diagnostics["forced_final_fallback"]
    assert diagnostics["mode"] == "detected_same_product_pair"
    assert diagnostics["pair_support_rank"] == 0


def test_segment_matching_rejects_top_only_trevi_reuse_for_pepsi_repeat():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace(
        {
            "removal_segment_targets": [
                {"weight": 519.5, "segment_index": 0},
                {"weight": 526.3, "segment_index": 1},
                {"weight": 521.9, "segment_index": 2},
                {"weight": 523.0, "segment_index": 3},
            ]
        }
    )
    trace.stage_counts_by_class = {
        "75": {
            "class_id": 75,
            "name": "BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
            "raw": 195,
            "raw_max_confidence": 0.148,
            "threshold_filtered": 195,
            "threshold_filtered_max_confidence": 0.148,
            "cameras": {
                "side": {
                    "raw": 19,
                    "raw_max_confidence": 0.0639,
                    "threshold_filtered": 19,
                    "threshold_filtered_max_confidence": 0.0639,
                },
                "top": {
                    "raw": 176,
                    "raw_max_confidence": 0.148,
                    "threshold_filtered": 176,
                    "threshold_filtered_max_confidence": 0.148,
                },
            },
            "motion_gate_passed": True,
        },
        "54": {
            "class_id": 54,
            "name": "BOTTLE_LOTTE_TREVI_LEMON_500ML",
            "raw": 15,
            "raw_max_confidence": 0.6514,
            "threshold_passed": 2,
            "threshold_passed_max_confidence": 0.6514,
            "roi_passed": 2,
            "cameras": {
                "side": {"raw": 2, "raw_max_confidence": 0.0759},
                "top": {
                    "raw": 13,
                    "raw_max_confidence": 0.6514,
                    "threshold_passed": 2,
                    "threshold_passed_max_confidence": 0.6514,
                    "roi_passed": 2,
                },
            },
            "motion_gate_passed": True,
            "final_rank": 1,
        },
        "31": {
            "class_id": 31,
            "name": "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML",
            "raw": 242,
            "raw_max_confidence": 0.1907,
            "threshold_filtered": 242,
            "threshold_filtered_max_confidence": 0.1907,
            "cameras": {
                "side": {
                    "raw": 242,
                    "raw_max_confidence": 0.1907,
                    "threshold_filtered": 242,
                    "threshold_filtered_max_confidence": 0.1907,
                }
            },
            "motion_gate_passed": True,
        },
        "44": {
            "class_id": 44,
            "name": "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
            "raw": 18,
            "raw_max_confidence": 0.0573,
            "threshold_filtered": 18,
            "threshold_filtered_max_confidence": 0.0573,
            "cameras": {
                "side": {
                    "raw": 18,
                    "raw_max_confidence": 0.0573,
                    "threshold_filtered": 18,
                    "threshold_filtered_max_confidence": 0.0573,
                }
            },
            "motion_gate_passed": True,
        },
    }

    result = engine.judge(
        vision_candidates=[
            EnsembleResult(
                class_id=54,
                class_name="BOTTLE_LOTTE_TREVI_LEMON_500ML",
                top_confidence=0.6514,
                side_confidence=0.0759,
                combined_confidence=0.3582,
                vote_count=2,
                source="vision",
                raw_vote_count=15,
                top_motion_passed=True,
            )
        ],
        delta_weight=-2091.0,
        active_products=[
            make_active_product(
                75,
                "BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
                520.0,
                10,
                2300,
            ),
            make_active_product(54, "BOTTLE_LOTTE_TREVI_LEMON_500ML", 530.0, 10, 1600),
            make_active_product(31, "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML", 523.0, 10),
            make_active_product(
                44,
                "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                520.0,
                10,
                2000,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.PARTIAL
    assert [(product.product_id, product.count) for product in result.products] == [
        (75, 4)
    ]
    assert result.weight_explained == 2080.0
    assert result.weight_residual == 11.7
    diagnostics = trace.weight_diagnostics["segment_weight_matching"]
    assert diagnostics["same_weight_bottle_collision"]["accepted"] is True
    assert diagnostics["same_weight_bottle_collision"]["selected"]["class_id"] == 75
    assert (
        diagnostics["repeated_segment_reuse_guard"]["rejected"][0]["reason"]
        == "repeated_segment_evidence_insufficient"
    )


def test_segment_matching_prefers_stage_supported_pepsi_repeat_over_mixed_fragments():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace(
        {
            "removal_segment_targets": [
                {"weight": 523.0, "segment_index": 0},
                {"weight": 523.0, "segment_index": 1},
                {"weight": 530.0, "segment_index": 2},
                {"weight": 523.0, "segment_index": 3},
            ]
        }
    )
    trace.stage_counts_by_class = {
        "75": {
            "class_id": 75,
            "name": "BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
            "raw": 80,
            "raw_max_confidence": 0.56,
            "threshold_passed": 32,
            "threshold_passed_max_confidence": 0.56,
            "motion_gate_passed": True,
        }
    }

    result = engine.judge(
        vision_candidates=[],
        delta_weight=-2099.0,
        active_products=[
            make_active_product(31, "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML", 523.0, 10),
            make_active_product(
                75,
                "BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
                520.0,
                10,
                2300,
            ),
            make_active_product(54, "BOTTLE_LOTTE_TREVI_LEMON_500ML", 530.0, 10),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (75, 4)
    ]
    diagnostics = trace.weight_diagnostics["segment_weight_matching"]
    assert diagnostics["aggregate_evidence_override"]["accepted"] is False
    assert (
        diagnostics["aggregate_evidence_override"]["reason"]
        == "clean_supported_segment_match_preferred"
    )
    assert diagnostics["products"] == [
        {
            "class_id": 75,
            "name": "BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
            "count": 4,
            "unit_weight": 520.0,
        }
    ]


def test_vision_required_segment_targets_do_not_use_active_only_fallback():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace(
        {
            "vision_required_segment_targets": [
                {"weight": 200.0, "segment_index": 1}
            ]
        }
    )

    result = engine.judge(
        vision_candidates=[],
        delta_weight=0.0,
        active_products=[
            make_active_product(101, "Pressure Candidate", weight=200.0, stock=10)
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.NO_DETECTION
    assert result.products == []


def test_same_product_count_accepts_dedicated_per_item_tolerance_for_pepero_x2():
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                class_id=115,
                name="BOX_LOTTE_PEPERO_ALMOND_37G",
                confidence=0.65,
            )
        ],
        delta_weight=-141.0,
        active_products=[
            make_active_product(
                class_id=115,
                name="BOX_LOTTE_PEPERO_ALMOND_37G",
                weight=66.0,
                stock=3,
                price=1700,
            )
        ],
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert result.products[0].name == "BOX_LOTTE_PEPERO_ALMOND_37G"
    assert result.products[0].count == 2
    assert result.weight_explained == 132.0
    assert result.weight_residual == 9.0


def test_compound_return_hint_downranks_previously_returned_weight_candidate():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace({"compound_positive_weights_g": [60.0]})

    result = engine.judge(
        vision_candidates=[
            make_candidate(101, "Returned Candidate A", confidence=0.95),
            make_candidate(102, "Next Extraction B", confidence=0.60),
        ],
        delta_weight=-64.0,
        active_products=[
            make_active_product(101, "Returned Candidate A", weight=60.0, stock=5),
            make_active_product(102, "Next Extraction B", weight=68.0, stock=5),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert result.products[0].product_id == 102
    assert result.products[0].name == "Next Extraction B"


def test_same_product_count_respects_stock_limit():
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                class_id=115,
                name="BOX_LOTTE_PEPERO_ALMOND_37G",
                confidence=0.65,
            )
        ],
        delta_weight=-138.0,
        active_products=[
            make_active_product(
                class_id=115,
                name="BOX_LOTTE_PEPERO_ALMOND_37G",
                weight=66.0,
                stock=1,
                price=1700,
            )
        ],
    )

    assert not (
        result.status == JudgmentStatus.COMPLETE
        and result.products
        and result.products[0].count == 2
    )


def test_same_product_count_accepts_scenario_a5():
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                class_id=101,
                name="Scenario Product A",
                confidence=0.8,
            )
        ],
        delta_weight=-505.0,
        active_products=[
            make_active_product(
                class_id=101,
                name="Scenario Product A",
                weight=101.0,
                stock=10,
                price=1000,
            )
        ],
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert result.products[0].product_id == 101
    assert result.products[0].count == 5
    assert result.total_price == 5000


def test_same_product_count_accepts_scenario_a8():
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                class_id=101,
                name="Scenario Product A",
                confidence=0.8,
            )
        ],
        delta_weight=-808.0,
        active_products=[
            make_active_product(
                class_id=101,
                name="Scenario Product A",
                weight=101.0,
                stock=10,
                price=1000,
            )
        ],
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert result.products[0].product_id == 101
    assert result.products[0].count == 8
    assert result.total_price == 8000


def test_same_product_count_rejects_scenario_a9_over_limit():
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                class_id=101,
                name="Scenario Product A",
                confidence=0.8,
            )
        ],
        delta_weight=-909.0,
        active_products=[
            make_active_product(
                class_id=101,
                name="Scenario Product A",
                weight=101.0,
                stock=10,
                price=1000,
            )
        ],
    )

    assert not (
        result.status == JudgmentStatus.COMPLETE
        and result.products
        and result.products[0].count == 9
    )


def test_high_confidence_three_kind_combo_accepts_scenario_abc():
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(101, "Scenario Product A", 0.9),
            make_candidate(102, "Scenario Product B", 0.88),
            make_candidate(103, "Scenario Product C", 0.86),
        ],
        delta_weight=-683.0,
        active_products=[
            make_active_product(101, "Scenario Product A", 101.0, stock=10, price=1000),
            make_active_product(102, "Scenario Product B", 223.0, stock=10, price=2000),
            make_active_product(103, "Scenario Product C", 359.0, stock=10, price=3000),
        ],
    )

    counts = {product.product_id: product.count for product in result.products}
    assert result.status == JudgmentStatus.COMPLETE
    assert counts == {101: 1, 102: 1, 103: 1}
    assert result.total_price == 6000


def test_high_confidence_two_kind_four_unit_combo_accepts_scenario_a2_b2():
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(101, "Scenario Product A", 0.9),
            make_candidate(102, "Scenario Product B", 0.88),
        ],
        delta_weight=-648.0,
        active_products=[
            make_active_product(101, "Scenario Product A", 101.0, stock=10, price=1000),
            make_active_product(102, "Scenario Product B", 223.0, stock=10, price=2000),
        ],
    )

    counts = {product.product_id: product.count for product in result.products}
    assert result.status == JudgmentStatus.COMPLETE
    assert counts == {101: 2, 102: 2}
    assert result.total_price == 6000


def test_compact_two_item_combo_beats_same_product_repeat():
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(101, "Scenario Product A", 0.23),
            make_candidate(102, "Scenario Product B", 0.22),
            make_candidate(103, "Scenario Product C", 0.24),
        ],
        delta_weight=-77.0,
        active_products=[
            make_active_product(101, "Scenario Product A", 50.0, stock=10, price=1000),
            make_active_product(102, "Scenario Product B", 26.0, stock=10, price=2000),
            make_active_product(103, "Scenario Product C", 19.0, stock=10, price=3000),
        ],
    )

    counts = {product.product_id: product.count for product in result.products}
    assert result.status == JudgmentStatus.COMPLETE
    assert counts == {101: 1, 102: 1}
    assert result.weight_residual == 1.0


def test_low_confidence_two_kind_combo_uses_relaxed_floor_and_5g_tolerance():
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(101, "Scenario Product A", 0.20),
            make_candidate(102, "Scenario Product B", 0.23),
        ],
        delta_weight=-81.0,
        active_products=[
            make_active_product(101, "Scenario Product A", 58.0, stock=10, price=1000),
            make_active_product(102, "Scenario Product B", 19.0, stock=10, price=2000),
        ],
    )

    counts = {product.product_id: product.count for product in result.products}
    assert result.status == JudgmentStatus.COMPLETE
    assert counts == {101: 1, 102: 1}
    assert result.weight_residual == 4.0


def test_same_product_repeat_can_win_when_compact_combo_is_outside_tolerance():
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(101, "Scenario Product A", 0.23),
            make_candidate(102, "Scenario Product B", 0.22),
            make_candidate(103, "Scenario Product C", 0.24),
        ],
        delta_weight=-88.0,
        active_products=[
            make_active_product(101, "Scenario Product A", 58.0, stock=10, price=1000),
            make_active_product(102, "Scenario Product B", 19.0, stock=10, price=2000),
            make_active_product(103, "Scenario Product C", 22.0, stock=10, price=3000),
        ],
    )

    counts = {product.product_id: product.count for product in result.products}
    assert result.status == JudgmentStatus.COMPLETE
    assert counts == {103: 4}
    assert result.weight_residual == 0.0


def test_candidate_same_product_repeat_preempts_stage_count_combo():
    trace = FakeStageTrace(
        [
            (113, "STICK_INNON_CONDITION_STICK_18G", 0.45),
            (100, "CUP_CJ_HATBAN_CUPBAN_CHICKEN_MAYO_313G", 0.42),
        ]
    )
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(26, "CAN_WELCHS_ZERO_GRAPE_355ML", 0.449),
        ],
        delta_weight=-753.0,
        active_products=[
            make_active_product(
                26,
                "CAN_WELCHS_ZERO_GRAPE_355ML",
                371.0,
                stock=10,
                price=1000,
            ),
            make_active_product(
                113,
                "STICK_INNON_CONDITION_STICK_18G",
                19.0,
                stock=10,
                price=3000,
            ),
            make_active_product(
                100,
                "CUP_CJ_HATBAN_CUPBAN_CHICKEN_MAYO_313G",
                365.0,
                stock=10,
                price=1100,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (26, 2)
    ]
    assert result.weight_residual == 11.0
    assert "stage_count_combination_match" not in trace.weight_diagnostics
    diagnostics = trace.weight_diagnostics["same_product_count_match"]
    assert diagnostics["accepted"] is True
    assert diagnostics["stage_count_preempted"] is True
    assert diagnostics["selected"]["allowed_residual"] == 15.0


def test_stage_count_combo_runs_when_candidate_repeat_is_outside_tolerance():
    trace = FakeStageTrace(
        [
            (113, "STICK_INNON_CONDITION_STICK_18G", 0.45),
            (100, "CUP_CJ_HATBAN_CUPBAN_CHICKEN_MAYO_313G", 0.42),
        ]
    )
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(26, "CAN_WELCHS_ZERO_GRAPE_355ML", 0.449),
        ],
        delta_weight=-760.0,
        active_products=[
            make_active_product(
                26,
                "CAN_WELCHS_ZERO_GRAPE_355ML",
                371.0,
                stock=10,
                price=1000,
            ),
            make_active_product(
                113,
                "STICK_INNON_CONDITION_STICK_18G",
                24.0,
                stock=10,
                price=3000,
            ),
            make_active_product(
                100,
                "CUP_CJ_HATBAN_CUPBAN_CHICKEN_MAYO_313G",
                365.0,
                stock=10,
                price=1100,
            ),
        ],
        trace_context=trace,
    )

    counts = {product.product_id: product.count for product in result.products}
    assert result.status == JudgmentStatus.COMPLETE
    assert counts == {26: 1, 113: 1, 100: 1}
    assert trace.weight_diagnostics["stage_count_combination_match"]["accepted"] is True


def test_channel_split_prefers_tteokbokki_and_welchs_over_aggregate_rescue():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace(
        {
            "removal_segment_targets": [
                {
                    "source": "unpaired_negative_segment",
                    "weight": 518.0,
                    "delta": -518.0,
                    "segment_index": 0,
                    "reason": "unpaired_removal_segment",
                }
            ],
            "channel_removal_segment_targets": [
                {
                    "source": "simultaneous_channel_delta",
                    "weight": 144.0,
                    "delta": -144.0,
                    "segment_index": 0,
                    "channel_index": 0,
                    "reason": "simultaneous_channel_removal",
                    "evidence_required": True,
                },
                {
                    "source": "simultaneous_channel_delta",
                    "weight": 375.0,
                    "delta": -375.0,
                    "segment_index": 1,
                    "channel_index": 1,
                    "reason": "simultaneous_channel_removal",
                    "evidence_required": True,
                },
            ],
        }
    )
    trace.stage_counts_by_class = {
        "26": {
            "class_id": 26,
            "name": "CAN_WELCHS_ZERO_GRAPE_355ML",
            "raw": 10,
            "raw_max_confidence": 0.7759,
            "threshold_filtered": 6,
            "threshold_filtered_max_confidence": 0.1813,
            "threshold_passed": 4,
            "threshold_passed_max_confidence": 0.7759,
            "cameras": {
                "top": {
                    "raw": 9,
                    "raw_max_confidence": 0.7759,
                    "threshold_passed": 4,
                    "threshold_passed_max_confidence": 0.7759,
                    "roi_filtered": 4,
                    "roi_filtered_max_confidence": 0.7759,
                },
                "side": {
                    "raw": 1,
                    "raw_max_confidence": 0.1813,
                },
            },
            "motion_gate_passed": True,
        }
    }

    result = engine.judge(
        vision_candidates=[
            EnsembleResult(
                class_id=120,
                class_name="CUP_BIBIGO_TTEOKBOKKI_110G",
                top_confidence=0.0,
                side_confidence=0.5328,
                combined_confidence=0.5328,
                vote_count=7,
                source="vision",
                motion_gate_passed=True,
            ),
            EnsembleResult(
                class_id=54,
                class_name="BOTTLE_LOTTE_TREVI_LEMON_500ML",
                top_confidence=0.3882,
                side_confidence=0.0,
                combined_confidence=0.3882,
                vote_count=3,
                source="vision",
                motion_gate_passed=True,
            ),
            EnsembleResult(
                class_id=113,
                class_name="STICK_INNON_CONDITION_STICK_18G",
                top_confidence=0.0,
                side_confidence=0.1736,
                combined_confidence=0.1736,
                vote_count=2,
                source="vision",
                motion_gate_passed=True,
            ),
            EnsembleResult(
                class_id=44,
                class_name="BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                top_confidence=0.028,
                side_confidence=0.0685,
                combined_confidence=0.028,
                vote_count=5,
                source="threshold_rescue",
                raw_vote_count=5,
                top_motion_passed=True,
                side_motion_passed=True,
                motion_gate_passed=True,
                weight_gate_passed=True,
            ),
        ],
        delta_weight=-519.0,
        active_products=[
            make_active_product(120, "CUP_BIBIGO_TTEOKBOKKI_110G", 144.0, 10, 5600),
            make_active_product(26, "CAN_WELCHS_ZERO_GRAPE_355ML", 371.0, 10, 1000),
            make_active_product(54, "BOTTLE_LOTTE_TREVI_LEMON_500ML", 530.0, 10, 1600),
            make_active_product(113, "STICK_INNON_CONDITION_STICK_18G", 19.0, 10, 3000),
            make_active_product(
                44,
                "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                520.0,
                10,
                2000,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (120, 1),
        (26, 1),
    ]
    assert result.weight_explained == 515.0
    assert result.weight_residual == 4.0
    diagnostics = trace.weight_diagnostics["segment_weight_matching"]
    assert diagnostics["target_source"] == "channel_removal_segment_targets"
    assert diagnostics["reason"] == "channel_supported_split_preferred"
    assert diagnostics["rejected_aggregate_rescue"]["class_id"] == 44


def test_channel_split_without_evidence_falls_back_to_aggregate_strict_match():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace(
        {
            "removal_segment_targets": [
                {
                    "source": "unpaired_negative_segment",
                    "weight": 520.0,
                    "delta": -520.0,
                    "segment_index": 0,
                }
            ],
            "channel_removal_segment_targets": [
                {
                    "source": "simultaneous_channel_delta",
                    "weight": 260.0,
                    "delta": -260.0,
                    "segment_index": 0,
                    "channel_index": 0,
                    "evidence_required": True,
                },
                {
                    "source": "simultaneous_channel_delta",
                    "weight": 260.0,
                    "delta": -260.0,
                    "segment_index": 1,
                    "channel_index": 1,
                    "evidence_required": True,
                },
            ],
        }
    )

    result = engine.judge(
        vision_candidates=[
            EnsembleResult(
                class_id=44,
                class_name="BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                top_confidence=0.2,
                side_confidence=0.2,
                combined_confidence=0.2,
                vote_count=5,
                source="threshold_rescue",
                raw_vote_count=5,
                top_motion_passed=True,
                side_motion_passed=True,
                motion_gate_passed=True,
                weight_gate_passed=True,
            )
        ],
        delta_weight=-520.0,
        active_products=[
            make_active_product(
                44,
                "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                520.0,
                10,
                2000,
            ),
            make_active_product(201, "ACTIVE_ONLY_HALF_WEIGHT_A", 260.0, 10, 1000),
            make_active_product(202, "ACTIVE_ONLY_HALF_WEIGHT_B", 260.0, 10, 1000),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (44, 1)
    ]
    diagnostics = trace.weight_diagnostics["segment_weight_matching"]
    assert diagnostics["accepted"] is False
    assert diagnostics["target_source"] == "channel_removal_segment_targets"
    assert diagnostics["reason"] == "segment_without_valid_option"


def test_regular_single_candidate_priority_beats_tiny_residual_gap():
    engine = ProductDecisionEngine(strict_mode=True)
    trace = FakeLoadcellTrace({})

    result = engine.judge(
        vision_candidates=[
            EnsembleResult(
                class_id=100,
                class_name="CUP_CJ_HATBAN_CUPBAN_CHICKEN_MAYO_313G",
                top_confidence=1.0,
                side_confidence=1.0,
                combined_confidence=1.0,
                vote_count=2,
                source="vision",
            ),
            EnsembleResult(
                class_id=38,
                class_name="BAG_HAITAI_HOME_RUN_BALL_41G",
                top_confidence=0.585,
                side_confidence=0.585,
                combined_confidence=0.585,
                vote_count=2,
                source="vision",
            ),
            EnsembleResult(
                class_id=8,
                class_name="CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
                top_confidence=0.0,
                side_confidence=0.287,
                combined_confidence=0.287,
                vote_count=1,
                source="vision",
            ),
        ],
        delta_weight=-368.0,
        active_products=[
            make_active_product(
                100,
                "CUP_CJ_HATBAN_CUPBAN_CHICKEN_MAYO_313G",
                365.0,
                stock=10,
                price=5600,
            ),
            make_active_product(
                38,
                "BAG_HAITAI_HOME_RUN_BALL_41G",
                64.0,
                stock=10,
                price=1200,
            ),
            make_active_product(
                8,
                "CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
                367.0,
                stock=10,
                price=1300,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (100, 1)
    ]
    assert result.weight_residual == 3.0

    diagnostics = trace.weight_diagnostics["strict_candidate_priority_selection"]
    assert diagnostics["reason"] == "regular_single_candidate_priority"
    assert diagnostics["selected"]["items"][0]["class_id"] == 100
    assert diagnostics["post_sort_top_combinations"][0]["items"][0]["class_id"] == 100
    assert trace.weight_diagnostics["valid_combinations"][0]["items"][0]["class_id"] == 8


def test_rank1_single_candidate_outside_tolerance_does_not_block_tighter_match():
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            EnsembleResult(
                class_id=100,
                class_name="CUP_CJ_HATBAN_CUPBAN_CHICKEN_MAYO_313G",
                top_confidence=1.0,
                side_confidence=1.0,
                combined_confidence=1.0,
                vote_count=2,
                source="vision",
            ),
            EnsembleResult(
                class_id=8,
                class_name="CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
                top_confidence=0.0,
                side_confidence=0.287,
                combined_confidence=0.287,
                vote_count=1,
                source="vision",
            ),
        ],
        delta_weight=-368.0,
        active_products=[
            make_active_product(
                100,
                "CUP_CJ_HATBAN_CUPBAN_CHICKEN_MAYO_313G",
                360.0,
                stock=10,
                price=5600,
            ),
            make_active_product(
                8,
                "CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
                367.0,
                stock=10,
                price=1300,
            ),
        ],
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert [(product.product_id, product.count) for product in result.products] == [
        (8, 1)
    ]
    assert result.weight_residual == 1.0


def test_stage_counts_combination_recovers_when_final_candidates_miss_combo():
    class FakeTrace:
        def __init__(self):
            self.weight_diagnostics = {}
            self.stage_counts_by_class = {
                "201": self._stage_entry(201, "Stage Filler 1"),
                "101": self._stage_entry(101, "Scenario Product A"),
                "202": self._stage_entry(202, "Stage Filler 2"),
                "203": self._stage_entry(203, "Stage Filler 3"),
                "204": self._stage_entry(204, "Stage Filler 4"),
                "205": self._stage_entry(205, "Stage Filler 5"),
                "206": self._stage_entry(206, "Stage Filler 6"),
                "102": self._stage_entry(102, "Scenario Product B", confidence=0.21),
                "207": self._stage_entry(207, "Stage Filler 7"),
            }

        @staticmethod
        def _stage_entry(class_id, name, confidence=0.22):
            return {
                "class_id": class_id,
                "name": name,
                "raw": 3,
                "raw_max_confidence": confidence,
                "cameras": {
                    "side": {
                        "raw": 3,
                        "raw_max_confidence": confidence,
                    }
                },
            }

        def record_weight_diagnostics(self, diagnostics):
            self.weight_diagnostics = diagnostics

    trace = FakeTrace()
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(103, "Scenario Product C", 0.92),
        ],
        delta_weight=-77.0,
        active_products=[
            make_active_product(101, "Scenario Product A", 58.0, stock=10, price=1000),
            make_active_product(102, "Scenario Product B", 19.0, stock=10, price=2000),
            make_active_product(103, "Scenario Product C", 200.0, stock=10, price=3000),
        ],
        trace_context=trace,
    )

    counts = {product.product_id: product.count for product in result.products}
    diagnostics = trace.weight_diagnostics["stage_count_combination_match"]
    assert result.status == JudgmentStatus.COMPLETE
    assert counts == {101: 1, 102: 1}
    assert result.weight_residual == 0.0
    assert diagnostics["accepted"] is True
    assert diagnostics["merged_candidate_count"] == 10
    assert diagnostics["stage_candidates_added"] == 9


def test_stage_count_expansion_uses_camera_aware_score_before_limit():
    trace = FakeLoadcellTrace({})
    for offset in range(9):
        class_id = 300 + offset
        trace.stage_counts_by_class[str(class_id)] = {
            "class_id": class_id,
            "name": f"Top Raw Filler {offset}",
            "raw": 50,
            "raw_max_confidence": 0.05,
            "cameras": {
                "top": {
                    "raw": 50,
                    "raw_max_confidence": 0.05,
                }
            },
        }
    trace.stage_counts_by_class["100"] = {
        "class_id": 100,
        "name": "CUP_CJ_HATBAN_CUPBAN_CHICKEN_MAYO_313G",
        "raw": 43,
        "raw_max_confidence": 0.0927,
        "cameras": {
            "top": {
                "raw": 43,
                "raw_max_confidence": 0.0927,
            }
        },
    }
    trace.stage_counts_by_class["26"] = {
        "class_id": 26,
        "name": "CAN_WELCHS_ZERO_GRAPE_355ML",
        "raw": 10,
        "raw_max_confidence": 0.3748,
        "threshold_passed": 2,
        "threshold_passed_max_confidence": 0.3748,
        "roi_filtered": 1,
        "roi_filtered_max_confidence": 0.3007,
        "cameras": {
            "side": {
                "raw": 10,
                "raw_max_confidence": 0.3748,
                "threshold_passed": 2,
                "threshold_passed_max_confidence": 0.3748,
                "roi_filtered": 1,
                "roi_filtered_max_confidence": 0.3007,
                "motion_filtered": 1,
            }
        },
    }
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine._try_stage_count_combination_match(
        vision_candidates=[make_candidate(999, "Irrelevant Candidate", 0.91)],
        delta_weight=-744.4,
        timestamp=123.0,
        active_products=[
            make_active_product(26, "CAN_WELCHS_ZERO_GRAPE_355ML", 371.0, 10),
            make_active_product(
                100,
                "CUP_CJ_HATBAN_CUPBAN_CHICKEN_MAYO_313G",
                365.0,
                10,
            ),
        ],
        trace_context=trace,
    )

    assert result is not None
    assert [(product.product_id, product.count) for product in result.products] == [
        (26, 2)
    ]
    diagnostics = trace.weight_diagnostics["stage_count_combination_match"]
    assert diagnostics["stage_candidates"][0]["class_id"] == 26
    assert any(
        candidate["class_id"] == 100
        and diagnostics["stage_candidates"][0]["stage_score"]
        > candidate["stage_score"]
        for candidate in diagnostics["stage_candidates"]
    )


def test_relaxed_combination_prefers_three_final_candidates_over_stage_count_pair():
    class FakeTrace:
        def __init__(self):
            self.weight_diagnostics = {}
            self.stage_counts_by_class = {
                "201": self._stage_entry(201, "Stage Product D"),
                "202": self._stage_entry(202, "Stage Product E"),
            }

        @staticmethod
        def _stage_entry(class_id, name, confidence=0.92):
            return {
                "class_id": class_id,
                "name": name,
                "raw": 4,
                "raw_max_confidence": confidence,
                "motion_passed": True,
            }

        def record_weight_diagnostics(self, diagnostics):
            self.weight_diagnostics.update(diagnostics)

    trace = FakeTrace()
    engine = ProductDecisionEngine(strict_mode=False)

    result = engine.judge(
        vision_candidates=[
            make_candidate(101, "Candidate Product A", 0.91),
            make_candidate(102, "Candidate Product B", 0.90),
            make_candidate(103, "Candidate Product C", 0.89),
        ],
        delta_weight=-340.0,
        active_products=[
            make_active_product(101, "Candidate Product A", 100.0, stock=10),
            make_active_product(102, "Candidate Product B", 110.0, stock=10),
            make_active_product(103, "Candidate Product C", 130.0, stock=10),
            make_active_product(201, "Stage Product D", 170.0, stock=10),
            make_active_product(202, "Stage Product E", 170.0, stock=10),
        ],
        trace_context=trace,
    )

    counts = {product.product_id: product.count for product in result.products}
    assert result.status == JudgmentStatus.COMPLETE
    assert counts == {101: 1, 102: 1, 103: 1}
    assert result.weight_residual == 0.0
    assert "relaxed_stage_count_combination_match" not in trace.weight_diagnostics
    assert (
        trace.weight_diagnostics[
            "relaxed_candidate_only_strict_combination_match"
        ]["selected"]["total_count"]
        == 3
    )


def test_stage_count_strict_fallback_prefers_final_candidate_combo_when_available():
    class FakeTrace:
        def __init__(self):
            self.weight_diagnostics = {}
            self.stage_counts_by_class = {
                "201": self._stage_entry(201, "Stage Product D"),
                "202": self._stage_entry(202, "Stage Product E"),
            }

        @staticmethod
        def _stage_entry(class_id, name, confidence=0.92):
            return {
                "class_id": class_id,
                "name": name,
                "raw": 4,
                "raw_max_confidence": confidence,
                "motion_passed": True,
            }

        def record_weight_diagnostics(self, diagnostics):
            self.weight_diagnostics.update(diagnostics)

    trace = FakeTrace()
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine._try_stage_count_combination_match(
        vision_candidates=[
            make_candidate(101, "Candidate Product A", 0.91),
            make_candidate(102, "Candidate Product B", 0.90),
            make_candidate(103, "Candidate Product C", 0.89),
        ],
        delta_weight=-340.0,
        timestamp=123.0,
        active_products=[
            make_active_product(101, "Candidate Product A", 100.0, stock=10),
            make_active_product(102, "Candidate Product B", 110.0, stock=10),
            make_active_product(103, "Candidate Product C", 130.0, stock=10),
            make_active_product(201, "Stage Product D", 170.0, stock=10),
            make_active_product(202, "Stage Product E", 170.0, stock=10),
        ],
        trace_context=trace,
    )

    assert result is not None
    counts = {product.product_id: product.count for product in result.products}
    diagnostics = trace.weight_diagnostics["stage_count_combination_match"]
    assert counts == {101: 1, 102: 1, 103: 1}
    assert diagnostics["selected"]["total_count"] == 3
    assert {
        item["class_id"] for item in diagnostics["selected"]["items"]
    } == {101, 102, 103}


def test_stage_count_combo_prefers_candidate_inclusive_over_all_stage_combo():
    trace = FakeStageTrace(
        [
            (102, "Stage Product B", 0.82),
            (201, "Stage Product C", 0.94),
            (202, "Stage Product D", 0.94),
        ]
    )
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine._try_stage_count_combination_match(
        vision_candidates=[
            make_candidate(101, "Candidate Product A", 0.61),
        ],
        delta_weight=-199.0,
        timestamp=123.0,
        active_products=[
            make_active_product(101, "Candidate Product A", 50.0, stock=1),
            make_active_product(102, "Stage Product B", 149.0, stock=1),
            make_active_product(201, "Stage Product C", 100.0, stock=1),
            make_active_product(202, "Stage Product D", 99.0, stock=1),
        ],
        trace_context=trace,
    )

    assert result is not None
    counts = {product.product_id: product.count for product in result.products}
    diagnostics = trace.weight_diagnostics["stage_count_combination_match"]
    assert counts == {101: 1, 102: 1}
    assert diagnostics["selected"]["weight_error"] == 0.0


def test_stage_count_combo_keeps_candidate_inclusive_with_slightly_higher_residual():
    trace = FakeStageTrace(
        [
            (102, "Stage Product B", 0.82),
            (201, "Stage Product C", 0.94),
            (202, "Stage Product D", 0.94),
        ]
    )
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine._try_stage_count_combination_match(
        vision_candidates=[
            make_candidate(101, "Candidate Product A", 0.61),
        ],
        delta_weight=-199.0,
        timestamp=123.0,
        active_products=[
            make_active_product(101, "Candidate Product A", 50.0, stock=1),
            make_active_product(102, "Stage Product B", 145.0, stock=1),
            make_active_product(201, "Stage Product C", 100.0, stock=1),
            make_active_product(202, "Stage Product D", 99.0, stock=1),
        ],
        trace_context=trace,
    )

    assert result is not None
    counts = {product.product_id: product.count for product in result.products}
    diagnostics = trace.weight_diagnostics["stage_count_combination_match"]
    assert counts == {101: 1, 102: 1}
    assert diagnostics["selected"]["weight_error"] == 4.0


def test_stage_count_combo_uses_all_stage_when_candidate_inclusive_invalid():
    trace = FakeStageTrace(
        [
            (102, "Stage Product B", 0.82),
            (201, "Stage Product C", 0.94),
            (202, "Stage Product D", 0.94),
        ]
    )
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine._try_stage_count_combination_match(
        vision_candidates=[
            make_candidate(101, "Candidate Product A", 0.61),
        ],
        delta_weight=-199.0,
        timestamp=123.0,
        active_products=[
            make_active_product(101, "Candidate Product A", 50.0, stock=1),
            make_active_product(102, "Stage Product B", 130.0, stock=1),
            make_active_product(201, "Stage Product C", 100.0, stock=1),
            make_active_product(202, "Stage Product D", 99.0, stock=1),
        ],
        trace_context=trace,
    )

    assert result is not None
    counts = {product.product_id: product.count for product in result.products}
    assert counts == {201: 1, 202: 1}


def test_low_confidence_three_item_weight_combo_is_not_completed():
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(113, "STICK_INNON_CONDITION_STICK_18G", 0.192),
            make_candidate(118, "BAG_CJ_CHICKEN_BREAST_STEAK_100G", 0.169),
            make_candidate(119, "BAG_NONGSHIM_SAEUKKANG_90G", 0.135),
        ],
        delta_weight=-220.4,
        active_products=[
            make_active_product(113, "STICK_INNON_CONDITION_STICK_18G", 19.0, stock=10),
            make_active_product(118, "BAG_CJ_CHICKEN_BREAST_STEAK_100G", 107.0, stock=10),
            make_active_product(119, "BAG_NONGSHIM_SAEUKKANG_90G", 97.0, stock=10),
        ],
    )

    assert result.status != JudgmentStatus.COMPLETE
    assert sum(product.count for product in result.products) != 3


def test_forced_final_fallback_rejects_active_product_weight_mismatch(monkeypatch):
    import model_service.engine.decision_engine as decision_engine_module

    use_weight_aware_identity(monkeypatch)
    monkeypatch.setattr(
        decision_engine_module.config.weight,
        "strict_mode_fallback",
        True,
    )

    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                class_id=115,
                name="BOX_LOTTE_PEPERO_ALMOND_37G",
                confidence=0.55,
            )
        ],
        delta_weight=-7.6,
        active_products=[
            make_active_product(
                class_id=115,
                name="BOX_LOTTE_PEPERO_ALMOND_37G",
                weight=58.0,
                stock=1,
            ),
            make_active_product(
                class_id=113,
                name="STICK_INNON_CONDITION_STICK_18G",
                weight=19.0,
                stock=1,
            ),
        ],
    )

    assert result.status == JudgmentStatus.UNCERTAIN
    assert result.products == []
    assert result.weight_residual == 11.4


def test_forced_final_fallback_uses_unseen_active_single_nearest(monkeypatch):
    import model_service.engine.decision_engine as decision_engine_module

    use_weight_aware_identity(monkeypatch)
    monkeypatch.setattr(
        decision_engine_module.config.weight,
        "strict_mode_fallback",
        True,
    )

    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                class_id=115,
                name="BOX_LOTTE_PEPERO_ALMOND_37G",
                confidence=0.55,
            )
        ],
        delta_weight=-18.0,
        active_products=[
            make_active_product(
                class_id=115,
                name="BOX_LOTTE_PEPERO_ALMOND_37G",
                weight=58.0,
                stock=1,
            ),
            make_active_product(
                class_id=113,
                name="STICK_INNON_CONDITION_STICK_18G",
                weight=19.0,
                stock=1,
                price=3000,
            ),
        ],
    )

    assert result.status == JudgmentStatus.PARTIAL
    assert result.products[0].product_id == 113
    assert result.weight_residual == 1.0


def test_loadcell_only_rejects_ambiguous_nearest_single():
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge_by_weight_only(
        delta_weight=-516.8,
        active_products=[
            make_active_product(
                class_id=75,
                name="BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
                weight=516.8,
                stock=1,
            ),
            make_active_product(
                class_id=112,
                name="BOTTLE_PULMUONE_SPRING_WATER_500ML",
                weight=518.0,
                stock=1,
            ),
        ],
    )

    assert result.status in {JudgmentStatus.UNCERTAIN, JudgmentStatus.NO_DETECTION}
    assert result.products == []


def test_threshold_rescue_candidate_can_complete_strict_match(monkeypatch):
    import model_service.engine.decision_engine as decision_engine_module

    monkeypatch.setattr(
        decision_engine_module.config.weight,
        "strict_mode_fallback",
        True,
    )

    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            EnsembleResult(
                class_id=75,
                class_name="BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
                top_confidence=0.19,
                side_confidence=0.16,
                combined_confidence=0.18,
                vote_count=2,
                source="threshold_rescue",
                raw_vote_count=9,
                top_motion_passed=True,
                side_motion_passed=True,
            )
        ],
        delta_weight=-516.8,
        active_products=[
            make_active_product(
                class_id=75,
                name="BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
                weight=516.8,
                stock=1,
                price=2000,
            ),
            make_active_product(
                class_id=112,
                name="BOTTLE_PULMUONE_SPRING_WATER_500ML",
                weight=518.0,
                stock=1,
                price=500,
            ),
        ],
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert result.products[0].product_id == 75
    assert result.products[0].name == "BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML"
    assert result.weight_residual == 0.0


def test_roi_rescue_candidate_can_complete_strict_match(monkeypatch):
    import model_service.engine.decision_engine as decision_engine_module

    monkeypatch.setattr(
        decision_engine_module.config.weight,
        "strict_mode_fallback",
        False,
    )

    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            EnsembleResult(
                class_id=38,
                class_name="BAG_HAITAI_HOME_RUN_BALL_41G",
                top_confidence=0.39,
                side_confidence=0.0,
                combined_confidence=0.39,
                vote_count=2,
                source="vision",
            ),
            EnsembleResult(
                class_id=8,
                class_name="CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
                top_confidence=0.0,
                side_confidence=0.18,
                combined_confidence=0.18,
                vote_count=9,
                source="roi_rescue",
                raw_vote_count=93,
            ),
        ],
        delta_weight=-369.0,
        active_products=[
            make_active_product(
                class_id=38,
                name="BAG_HAITAI_HOME_RUN_BALL_41G",
                weight=64.0,
                stock=5,
                price=1200,
            ),
            make_active_product(
                class_id=8,
                name="CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
                weight=369.0,
                stock=1,
                price=2000,
            ),
        ],
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert result.products[0].product_id == 8
    assert result.products[0].name == "CAN_LOTTE_HOT6_THE_KING_RUSH_355ML"
    assert result.weight_residual == 0.0


def test_roi_rescue_candidate_uses_rescue_only_tolerance(monkeypatch):
    import model_service.engine.decision_engine as decision_engine_module

    monkeypatch.setattr(
        decision_engine_module.config.weight,
        "strict_mode_fallback",
        True,
    )
    monkeypatch.setattr(
        decision_engine_module.config.weight,
        "tolerance_grams",
        3.0,
    )
    monkeypatch.setattr(
        decision_engine_module.config.weight,
        "rescue_tolerance_grams",
        5.0,
        raising=False,
    )

    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            EnsembleResult(
                class_id=38,
                class_name="BAG_HAITAI_HOME_RUN_BALL_41G",
                top_confidence=0.39,
                side_confidence=0.0,
                combined_confidence=0.39,
                vote_count=1,
                source="vision",
            ),
            EnsembleResult(
                class_id=26,
                class_name="CAN_WELCHS_ZERO_GRAPE_355ML",
                top_confidence=0.0,
                side_confidence=0.18,
                combined_confidence=0.18,
                vote_count=7,
                source="roi_rescue",
                raw_vote_count=57,
            ),
        ],
        delta_weight=-374.8,
        active_products=[
            make_active_product(
                class_id=38,
                name="BAG_HAITAI_HOME_RUN_BALL_41G",
                weight=64.0,
                stock=5,
                price=1200,
            ),
            make_active_product(
                class_id=26,
                name="CAN_WELCHS_ZERO_GRAPE_355ML",
                weight=371.0,
                stock=1,
                price=1800,
            ),
        ],
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert result.products[0].product_id == 26
    assert result.products[0].name == "CAN_WELCHS_ZERO_GRAPE_355ML"
    assert result.weight_residual == 3.8


def test_no_motion_threshold_rescue_uses_tight_weight_gate(monkeypatch):
    import model_service.engine.decision_engine as decision_engine_module

    monkeypatch.setattr(
        decision_engine_module.config.weight,
        "strict_mode_fallback",
        True,
    )
    monkeypatch.setattr(
        decision_engine_module.config.weight,
        "tolerance_grams",
        3.0,
    )
    monkeypatch.setattr(
        decision_engine_module.config.weight,
        "rescue_tolerance_grams",
        5.0,
        raising=False,
    )
    monkeypatch.setattr(
        decision_engine_module.config.vision,
        "weight_rescue_no_motion_enabled",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        decision_engine_module.config.vision,
        "weight_rescue_no_motion_min_raw_votes",
        8,
        raising=False,
    )
    monkeypatch.setattr(
        decision_engine_module.config.vision,
        "weight_rescue_no_motion_max_residual_grams",
        2.0,
        raising=False,
    )

    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            EnsembleResult(
                class_id=95,
                class_name="BOTTLE_HY_HALUYACHE_PURPLE_200ML_V2",
                top_confidence=0.13,
                side_confidence=0.0,
                combined_confidence=0.13,
                vote_count=1,
                source="threshold_rescue",
                raw_vote_count=18,
                top_motion_passed=False,
                side_motion_passed=False,
            )
        ],
        delta_weight=-221.0,
        active_products=[
            make_active_product(
                class_id=95,
                name="BOTTLE_HY_HALUYACHE_PURPLE_200ML_V2",
                weight=220.0,
                stock=1,
                price=1800,
            )
        ],
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert result.products[0].product_id == 95
    assert result.products[0].name == "BOTTLE_HY_HALUYACHE_PURPLE_200ML_V2"
    assert result.weight_residual == 1.0


def test_no_motion_threshold_rescue_rejects_weight_residual_above_tight_gate(monkeypatch):
    import model_service.engine.decision_engine as decision_engine_module

    monkeypatch.setattr(
        decision_engine_module.config.weight,
        "strict_mode_fallback",
        False,
    )
    monkeypatch.setattr(
        decision_engine_module.config.weight,
        "tolerance_grams",
        3.0,
    )
    monkeypatch.setattr(
        decision_engine_module.config.weight,
        "rescue_tolerance_grams",
        5.0,
        raising=False,
    )
    monkeypatch.setattr(
        decision_engine_module.config.vision,
        "weight_rescue_no_motion_enabled",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        decision_engine_module.config.vision,
        "weight_rescue_no_motion_min_raw_votes",
        8,
        raising=False,
    )
    monkeypatch.setattr(
        decision_engine_module.config.vision,
        "weight_rescue_no_motion_max_residual_grams",
        2.0,
        raising=False,
    )

    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            EnsembleResult(
                class_id=95,
                class_name="BOTTLE_HY_HALUYACHE_PURPLE_200ML_V2",
                top_confidence=0.13,
                side_confidence=0.0,
                combined_confidence=0.13,
                vote_count=1,
                source="threshold_rescue",
                raw_vote_count=18,
                top_motion_passed=False,
                side_motion_passed=False,
            )
        ],
        delta_weight=-221.0,
        active_products=[
            make_active_product(
                class_id=95,
                name="BOTTLE_HY_HALUYACHE_PURPLE_200ML_V2",
                weight=217.0,
                stock=1,
                price=1800,
            )
        ],
    )

    assert result.status in {JudgmentStatus.UNCERTAIN, JudgmentStatus.NO_DETECTION}
    assert result.products == []


def test_strict_mismatch_rejects_active_nearest_at_5g_or_more(monkeypatch):
    import model_service.engine.decision_engine as decision_engine_module

    use_weight_aware_identity(monkeypatch)
    monkeypatch.setattr(
        decision_engine_module.config.weight,
        "strict_mode_fallback",
        True,
    )

    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                class_id=115,
                name="BOX_LOTTE_PEPERO_ALMOND_37G",
                confidence=0.55,
            )
        ],
        delta_weight=-14.0,
        active_products=[
            make_active_product(
                class_id=115,
                name="BOX_LOTTE_PEPERO_ALMOND_37G",
                weight=58.0,
                stock=1,
            ),
            make_active_product(
                class_id=113,
                name="STICK_INNON_CONDITION_STICK_18G",
                weight=19.0,
                stock=1,
            ),
        ],
    )

    assert result.status == JudgmentStatus.PARTIAL
    assert result.products[0].product_id == 113
    assert result.weight_residual == 5.0


def test_strict_mismatch_partial_fallback_rejected_when_weight_mismatch(monkeypatch):
    import model_service.engine.decision_engine as decision_engine_module

    use_weight_aware_identity(monkeypatch)
    monkeypatch.setattr(
        decision_engine_module.config.weight,
        "strict_mode_fallback",
        True,
    )

    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[make_candidate()],
        delta_weight=-370.1,
        active_products=[make_active_product()],
    )

    assert result.status == JudgmentStatus.UNCERTAIN
    assert result.products == []
    assert result.weight_residual == 5.1


def test_strict_mismatch_default_vision_first_preserves_identity_as_partial():
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[make_candidate()],
        delta_weight=-370.1,
        active_products=[make_active_product()],
    )

    assert result.status == JudgmentStatus.PARTIAL
    assert [(product.product_id, product.count) for product in result.products] == [
        (26, 1)
    ]
    assert result.weight_residual == 5.1


def test_strict_mismatch_can_still_hard_fail(monkeypatch):
    import model_service.engine.decision_engine as decision_engine_module

    monkeypatch.setattr(
        decision_engine_module.config.weight,
        "strict_mode_fallback",
        False,
    )

    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[make_candidate()],
        delta_weight=-370.1,
        active_products=[make_active_product()],
    )

    assert result.status == JudgmentStatus.NO_DETECTION
    assert result.products == []


def test_decision_engine_records_strict_weight_diagnostics(monkeypatch):
    import model_service.engine.decision_engine as decision_engine_module

    class FakeTrace:
        def __init__(self):
            self.weight_diagnostics = None

        def record_weight_diagnostics(self, diagnostics):
            self.weight_diagnostics = diagnostics

    monkeypatch.setattr(
        decision_engine_module.config.weight,
        "strict_mode_fallback",
        True,
    )

    trace = FakeTrace()
    engine = ProductDecisionEngine(strict_mode=True)

    engine.judge(
        vision_candidates=[make_candidate()],
        delta_weight=-365.0,
        active_products=[make_active_product()],
        trace_context=trace,
    )

    assert trace.weight_diagnostics["target_weight"] == 365.0
    assert trace.weight_diagnostics["candidate_products"][0]["class_id"] == 26
    assert trace.weight_diagnostics["valid_combinations"][0]["total_weight"] == 365.0


def test_detected_single_fallback_uses_stage_evidence_within_8g(monkeypatch):
    import model_service.engine.decision_engine as decision_engine_module

    use_weight_aware_identity(monkeypatch)

    class FakeTrace:
        def __init__(self):
            self.stage_counts_by_class = {
                "11": {
                    "class_id": 11,
                    "name": "BOTTLE_DONGA_BACCHUS_F_120ML",
                    "threshold_filtered": 37,
                    "threshold_filtered_max_confidence": 0.1307,
                }
            }
            self.diagnostic_detections = []
            self.detected_single_fallback = None

        def record_weight_diagnostics(self, diagnostics):
            self.weight_diagnostics = diagnostics

        def record_detected_single_fallback(self, diagnostics):
            self.detected_single_fallback = diagnostics

    monkeypatch.setattr(
        decision_engine_module.config.weight,
        "strict_mode_fallback",
        True,
    )
    trace = FakeTrace()
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                class_id=57,
                name="BAG_HAITAI_JAGABEE_45G",
                confidence=0.92,
            )
        ],
        delta_weight=-264.0,
        active_products=[
            make_active_product(
                class_id=57,
                name="BAG_HAITAI_JAGABEE_45G",
                weight=80.0,
                stock=21,
            ),
            make_active_product(
                class_id=11,
                name="BOTTLE_DONGA_BACCHUS_F_120ML",
                weight=260.0,
                stock=20,
                price=500,
            ),
        ],
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert result.products[0].product_id == 11
    assert result.products[0].name == "BOTTLE_DONGA_BACCHUS_F_120ML"
    assert result.products[0].count == 1
    assert result.weight_residual == 4.0
    assert trace.detected_single_fallback["reason"] == "detected_single_item_fallback"


def test_detected_single_fallback_does_not_use_unseen_active_nearest(monkeypatch):
    import model_service.engine.decision_engine as decision_engine_module

    use_weight_aware_identity(monkeypatch)
    monkeypatch.setattr(
        decision_engine_module.config.weight,
        "strict_mode_fallback",
        True,
    )
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                class_id=57,
                name="BAG_HAITAI_JAGABEE_45G",
                confidence=0.92,
            )
        ],
        delta_weight=-264.0,
        active_products=[
            make_active_product(
                class_id=57,
                name="BAG_HAITAI_JAGABEE_45G",
                weight=80.0,
                stock=21,
            ),
            make_active_product(
                class_id=11,
                name="BOTTLE_DONGA_BACCHUS_F_120ML",
                weight=260.0,
                stock=20,
            ),
        ],
    )

    assert result.status == JudgmentStatus.PARTIAL
    assert result.products[0].product_id == 11
    assert result.weight_residual == 4.0


def test_vision_first_suppresses_low_confidence_mismatch_identity(monkeypatch):
    import model_service.engine.decision_engine as decision_engine_module

    class FakeTrace:
        stage_counts_by_class = {
            "114": {
                "class_id": 114,
                "name": "BOX_LOTTE_PEPERO_ORIGINAL_46G",
                "threshold_filtered": 10,
                "threshold_filtered_max_confidence": 0.1854,
            }
        }
        diagnostic_detections = []

        def record_weight_diagnostics(self, diagnostics):
            self.weight_diagnostics = diagnostics

    monkeypatch.setattr(
        decision_engine_module.config.weight,
        "strict_mode_fallback",
        True,
    )
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[
            make_candidate(
                class_id=115,
                name="BOX_LOTTE_PEPERO_ALMOND_37G",
                confidence=0.23,
            )
        ],
        delta_weight=-75.1,
        active_products=[
            make_active_product(
                class_id=114,
                name="BOX_LOTTE_PEPERO_ORIGINAL_46G",
                weight=66.0,
                stock=25,
            ),
            make_active_product(
                class_id=115,
                name="BOX_LOTTE_PEPERO_ALMOND_37G",
                weight=58.0,
                stock=21,
            ),
        ],
        trace_context=FakeTrace(),
    )

    assert result.status == JudgmentStatus.UNCERTAIN
    assert result.products == []
    assert result.weight_explained == 0.0
    assert result.weight_residual == 75.1


class SingleBottleFallbackTrace:
    def __init__(
        self,
        *,
        pepsi_side_confidence: float = 0.7311,
        pepsi_motion: bool = True,
        trevi_side_confidence: float = 0.0,
        trevi_side_votes: int = 0,
        trevi_weight_gate_passed: bool = True,
    ):
        pepsi_side = {
            "raw": 278,
            "raw_max_confidence": pepsi_side_confidence,
            "threshold_passed": 72,
            "roi_passed": 72,
        }
        if pepsi_motion:
            pepsi_side["motion_filtered"] = 72
        trevi_camera = {
            "top": {
                "raw": 18,
                "raw_max_confidence": 0.0447,
            }
        }
        trevi_raw = 18
        trevi_confidence = 0.0447
        if trevi_side_votes > 0:
            trevi_camera["side"] = {
                "raw": trevi_side_votes,
                "raw_max_confidence": trevi_side_confidence,
                "threshold_passed": trevi_side_votes,
                "motion_filtered": trevi_side_votes,
            }
            trevi_raw = max(trevi_raw, trevi_side_votes)
            trevi_confidence = max(trevi_confidence, trevi_side_confidence)

        self.stage_counts_by_class = {
            "75": {
                "class_id": 75,
                "name": "BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
                "raw": 470,
                "raw_max_confidence": pepsi_side_confidence,
                "threshold_passed": 72,
                "roi_passed": 72,
                "motion_gate_passed": pepsi_motion,
                "threshold_rescue_candidate": True,
                "weight_gate_passed": False,
                "weight_residual_g": 11.4,
                "cameras": {
                    "top": {
                        "raw": 192,
                        "raw_max_confidence": 0.2043,
                    },
                    "side": pepsi_side,
                },
            },
            "54": {
                "class_id": 54,
                "name": "BOTTLE_LOTTE_TREVI_LEMON_500ML",
                "raw": trevi_raw,
                "raw_max_confidence": trevi_confidence,
                "weight_gate_passed": trevi_weight_gate_passed,
                "weight_residual_g": 1.4,
                "motion_gate_passed": trevi_side_votes > 0,
                "cameras": trevi_camera,
            },
        }
        self.diagnostic_detections = []
        self.weight_diagnostics = {}
        self.detected_single_fallback = None

    def record_weight_diagnostics(self, diagnostics):
        self.weight_diagnostics.update(diagnostics)

    def record_detected_single_fallback(self, diagnostics):
        self.detected_single_fallback = diagnostics


def pepsi_trevi_single_bottle_products():
    return [
        make_active_product(
            class_id=75,
            name="BOTTLE_LOTTE_PEPSI_ZERO_SUGAR_LIME_500ML",
            weight=520.0,
            stock=6,
            price=2300,
        ),
        make_active_product(
            class_id=54,
            name="BOTTLE_LOTTE_TREVI_LEMON_500ML",
            weight=530.0,
            stock=6,
            price=1600,
        ),
        make_active_product(
            class_id=112,
            name="BOTTLE_PULMUONE_SPRING_WATER_500ML",
            weight=529.0,
            stock=6,
            price=900,
        ),
    ]


def test_detected_single_fallback_prefers_strong_pepsi_identity_over_weak_trevi():
    trace = SingleBottleFallbackTrace()
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[],
        delta_weight=-531.4,
        active_products=pepsi_trevi_single_bottle_products(),
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.UNCERTAIN
    assert result.products == []
    assert result.weight_residual == 11.4
    assert trace.weight_diagnostics["final_weight_mismatch_guard"]["accepted"] is False

    diagnostics = trace.detected_single_fallback
    assert diagnostics["accepted"]["single_bottle_identity_override"] is True
    override = diagnostics["single_bottle_identity_override"]
    assert override["accepted"] is True
    assert override["reason"] == "strong_identity_over_weak_weight_single"
    assert override["candidate"]["class_id"] == 75
    assert override["replaced"]["class_id"] == 54


def test_detected_single_fallback_keeps_trevi_when_pepsi_identity_is_weak():
    trace = SingleBottleFallbackTrace(pepsi_side_confidence=0.31)
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[],
        delta_weight=-531.4,
        active_products=pepsi_trevi_single_bottle_products(),
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert result.products[0].product_id == 54
    assert result.weight_residual == 1.4
    assert trace.detected_single_fallback["accepted"][
        "single_bottle_identity_override"
    ] is False
    assert trace.detected_single_fallback["single_bottle_identity_override"][
        "accepted"
    ] is False


def test_detected_single_fallback_keeps_trevi_when_trevi_identity_is_also_strong():
    trace = SingleBottleFallbackTrace(
        trevi_side_confidence=0.62,
        trevi_side_votes=28,
        trevi_weight_gate_passed=False,
    )
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[],
        delta_weight=-531.4,
        active_products=pepsi_trevi_single_bottle_products(),
        trace_context=trace,
    )

    assert result.status == JudgmentStatus.COMPLETE
    assert result.products[0].product_id == 54
    assert result.weight_residual == 1.4
    assert trace.detected_single_fallback["single_bottle_identity_override"][
        "reason"
    ] == "current_single_identity_strong"
