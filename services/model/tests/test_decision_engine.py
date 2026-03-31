from dataclasses import dataclass

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


def make_candidate(
    class_id: int = 26,
    name: str = "치킨마요",
    confidence: float = 1.0,
) -> EnsembleResult:
    return EnsembleResult(
        class_id=class_id,
        class_name=name,
        top_confidence=confidence,
        side_confidence=confidence,
        combined_confidence=confidence,
        vote_count=2,
    )


def make_active_product(
    class_id: int = 26,
    name: str = "치킨마요",
    weight: float = 365.0,
    stock: int = 5,
) -> MockActiveProduct:
    return MockActiveProduct(
        yolo_class_id=class_id,
        product_name=name,
        product_weight=weight,
        stock_qty=stock,
    )


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


def test_strict_mismatch_falls_back_to_partial(monkeypatch):
    import model_service.engine.decision_engine as decision_engine_module

    monkeypatch.setattr(
        decision_engine_module.config.weight,
        "strict_mode_fallback",
        True,
    )

    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[make_candidate()],
        delta_weight=-368.5,
        active_products=[make_active_product()],
    )

    assert result.status == JudgmentStatus.PARTIAL
    assert result.products[0].product_id == 26
    assert result.products[0].count == 1


def test_strict_mismatch_defaults_to_partial_fallback():
    engine = ProductDecisionEngine(strict_mode=True)

    result = engine.judge(
        vision_candidates=[make_candidate()],
        delta_weight=-368.5,
        active_products=[make_active_product()],
    )

    assert result.status == JudgmentStatus.PARTIAL


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
        delta_weight=-368.5,
        active_products=[make_active_product()],
    )

    assert result.status == JudgmentStatus.NO_DETECTION
    assert result.products == []
