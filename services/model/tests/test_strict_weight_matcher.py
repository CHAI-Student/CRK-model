"""
StrictWeightMatcher 테스트 (v5.2).

무게 우선 엄격 매칭기 테스트.

테스트 케이스:
1. test_single_product_exact_match - 단일 상품 정확 매칭
2. test_single_product_within_tolerance - 허용 오차 내 매칭
3. test_no_valid_combination - 유효한 조합 없음
4. test_multiple_products_combination - 다중 상품 조합
5. test_stock_limit_applied - 재고 제한 적용
6. test_max_combinations_limit - 최대 조합 수 제한
7. test_match_score_sorting - match_score 순 정렬
"""

from dataclasses import dataclass

import pytest
from model_service.engine.models import EnsembleResult
from model_service.weight.strict_weight_matcher import (
    CandidateProduct,
    StrictWeightMatcher,
    ValidCombination,
)


@dataclass
class MockActiveProduct:
    """테스트용 ActiveProduct mock."""
    yolo_class_id: int
    product_name: str
    product_weight: float
    stock_qty: int
    sale_price: int = 0


class TestStrictWeightMatcher:
    """StrictWeightMatcher 테스트."""

    @pytest.fixture
    def matcher(self):
        """기본 매칭기 픽스처."""
        return StrictWeightMatcher(
            tolerance=3.0,
            max_items=5,
            max_count_per_item=10,
            max_combinations=100,
        )

    def _make_candidate(
        self,
        class_id: int,
        name: str,
        confidence: float = 0.9,
        *,
        motion_gate_passed: bool = True,
        top_motion_passed: bool = False,
        side_motion_passed: bool = False,
    ) -> EnsembleResult:
        """테스트용 EnsembleResult 생성."""
        return EnsembleResult(
            class_id=class_id,
            class_name=name,
            top_confidence=confidence,
            side_confidence=confidence,
            combined_confidence=confidence,
            vote_count=2,
            motion_gate_passed=motion_gate_passed,
            top_motion_passed=top_motion_passed,
            side_motion_passed=side_motion_passed,
        )

    def _make_active_product(
        self,
        class_id: int,
        name: str,
        weight: float,
        stock: int = 10,
        price: int = 1000,
    ) -> MockActiveProduct:
        """테스트용 ActiveProduct 생성."""
        return MockActiveProduct(
            yolo_class_id=class_id,
            product_name=name,
            product_weight=weight,
            stock_qty=stock,
            sale_price=price,
        )

    def test_single_product_exact_match(self, matcher):
        """단일 상품 정확 매칭 테스트."""
        # Given: 치킨마요 365g 1개
        candidates = [self._make_candidate(1, "치킨마요", 0.95)]
        active_products = [self._make_active_product(1, "치킨마요", 365.0)]

        # When: 365g 제거
        combos = matcher.find_valid_combinations(
            candidates=candidates,
            delta_weight=-365.0,
            active_products=active_products,
        )

        # Then: 치킨마요 1개 조합 찾음
        assert len(combos) > 0
        best = combos[0]
        assert len(best.items) == 1
        assert best.items[0].candidate.name == "치킨마요"
        assert best.items[0].count == 1
        assert best.weight_error <= 3.0

    def test_single_product_within_tolerance(self, matcher):
        """허용 오차 내 매칭 테스트."""
        # Given: 치킨마요 365g
        candidates = [self._make_candidate(1, "치킨마요", 0.9)]
        active_products = [self._make_active_product(1, "치킨마요", 365.0)]

        # When: 367g 제거 (오차 2g)
        combos = matcher.find_valid_combinations(
            candidates=candidates,
            delta_weight=-367.0,
            active_products=active_products,
        )

        # Then: 치킨마요 1개 조합 찾음 (오차 2g <= tolerance 3g)
        assert len(combos) > 0
        best = combos[0]
        assert best.weight_error == 2.0

    def test_no_valid_combination(self, matcher):
        """유효한 조합 없음 테스트."""
        # Given: 치킨마요 365g
        candidates = [self._make_candidate(1, "치킨마요", 0.9)]
        active_products = [self._make_active_product(1, "치킨마요", 365.0)]

        # When: 100g 제거 (365의 배수 아님)
        combos = matcher.find_valid_combinations(
            candidates=candidates,
            delta_weight=-100.0,
            active_products=active_products,
        )

        # Then: 조합 없음
        assert len(combos) == 0

    def test_multiple_products_combination(self, matcher):
        """다중 상품 조합 테스트."""
        # Given: 치킨마요 365g, 참치마요 250g
        candidates = [
            self._make_candidate(1, "치킨마요", 0.9),
            self._make_candidate(2, "참치마요", 0.85),
        ]
        active_products = [
            self._make_active_product(1, "치킨마요", 365.0),
            self._make_active_product(2, "참치마요", 250.0),
        ]

        # When: 615g 제거 (365 + 250)
        combos = matcher.find_valid_combinations(
            candidates=candidates,
            delta_weight=-615.0,
            active_products=active_products,
        )

        # Then: 치킨마요 1 + 참치마요 1 조합 찾음
        assert len(combos) > 0
        # 2개 상품 조합이 있어야 함
        combo_with_two = None
        for c in combos:
            if len(c.items) == 2:
                combo_with_two = c
                break
        assert combo_with_two is not None
        assert combo_with_two.total_count == 2

    def test_stock_limit_applied(self, matcher):
        """재고 제한 적용 테스트."""
        # Given: 치킨마요 365g, 재고 2개
        candidates = [self._make_candidate(1, "치킨마요", 0.9)]
        active_products = [self._make_active_product(1, "치킨마요", 365.0, stock=2)]

        # When: 1095g 제거 (3개 필요하지만 재고 2개)
        combos = matcher.find_valid_combinations(
            candidates=candidates,
            delta_weight=-1095.0,  # 365 * 3
            active_products=active_products,
        )

        # Then: 조합 없음 (재고 부족)
        assert len(combos) == 0

        # When: 730g 제거 (2개 필요, 재고 2개)
        combos_valid = matcher.find_valid_combinations(
            candidates=candidates,
            delta_weight=-730.0,  # 365 * 2
            active_products=active_products,
        )

        # Then: 치킨마요 2개 조합 찾음
        assert len(combos_valid) > 0
        assert combos_valid[0].items[0].count == 2

    def test_zero_stock_candidate_is_excluded(self, matcher):
        """stock=0 상품은 strict 매칭 후보에서 제외한다."""
        candidates = [self._make_candidate(1, "치킨마요", 0.95)]
        active_products = [self._make_active_product(1, "치킨마요", 365.0, stock=0)]

        combos = matcher.find_valid_combinations(
            candidates=candidates,
            delta_weight=-365.0,
            active_products=active_products,
        )

        assert combos == []

    def test_low_confidence_three_kind_combo_is_rejected(self):
        matcher = StrictWeightMatcher(tolerance=3.0, max_items=3)
        candidates = [
            self._make_candidate(113, "STICK_INNON_CONDITION_STICK_18G", 0.192),
            self._make_candidate(118, "BAG_CJ_CHICKEN_BREAST_STEAK_100G", 0.169),
            self._make_candidate(119, "BAG_NONGSHIM_SAEUKKANG_90G", 0.135),
        ]
        active_products = [
            self._make_active_product(113, "STICK_INNON_CONDITION_STICK_18G", 19.0, stock=10),
            self._make_active_product(118, "BAG_CJ_CHICKEN_BREAST_STEAK_100G", 107.0, stock=10),
            self._make_active_product(119, "BAG_NONGSHIM_SAEUKKANG_90G", 97.0, stock=10),
        ]

        combos = matcher.find_valid_combinations(
            candidates=candidates,
            delta_weight=-223.0,
            active_products=active_products,
        )

        assert combos == []
        assert matcher.last_diagnostics["reason"] == "all_combinations_rejected"
        assert "low_confidence_combo" in matcher.last_diagnostics["rejected_combination_counts"]

    def test_high_confidence_three_kind_combo_is_accepted(self):
        matcher = StrictWeightMatcher(tolerance=3.0, max_items=5, max_kinds=3)
        candidates = [
            self._make_candidate(1, "A", 0.92),
            self._make_candidate(2, "B", 0.88),
            self._make_candidate(3, "C", 0.84),
        ]
        active_products = [
            self._make_active_product(1, "A", 101.0, stock=10),
            self._make_active_product(2, "B", 223.0, stock=10),
            self._make_active_product(3, "C", 359.0, stock=10),
        ]

        combos = matcher.find_valid_combinations(
            candidates=candidates,
            delta_weight=-683.0,
            active_products=active_products,
        )

        assert combos
        assert combos[0].kind_count == 3
        assert combos[0].total_count == 3

    def test_max_combinations_limit(self):
        """최대 조합 수 제한 테스트."""
        # Given: 제한된 max_combinations
        matcher = StrictWeightMatcher(
            tolerance=10.0,  # 넓은 tolerance
            max_combinations=3,
        )
        candidates = [
            self._make_candidate(1, "상품A", 0.9),
            self._make_candidate(2, "상품B", 0.85),
            self._make_candidate(3, "상품C", 0.8),
        ]
        active_products = [
            self._make_active_product(1, "상품A", 100.0, stock=5),
            self._make_active_product(2, "상품B", 100.0, stock=5),
            self._make_active_product(3, "상품C", 100.0, stock=5),
        ]

        # When: 300g 제거 (여러 조합 가능)
        combos = matcher.find_valid_combinations(
            candidates=candidates,
            delta_weight=-300.0,
            active_products=active_products,
        )

        # Then: 최대 3개까지만 반환
        assert len(combos) <= 3

    def test_match_score_sorting(self, matcher):
        """match_score 순 정렬 테스트."""
        # Given: 치킨마요 365g (신뢰도 높음), 참치마요 365g (신뢰도 낮음)
        candidates = [
            self._make_candidate(1, "치킨마요", 0.95),
            self._make_candidate(2, "참치마요", 0.60),
        ]
        active_products = [
            self._make_active_product(1, "치킨마요", 365.0),
            self._make_active_product(2, "참치마요", 365.0),
        ]

        # When: 365g 제거 (둘 다 매칭 가능)
        combos = matcher.find_valid_combinations(
            candidates=candidates,
            delta_weight=-365.0,
            active_products=active_products,
        )

        # Then: 신뢰도 높은 치킨마요가 먼저 (match_score 순)
        assert len(combos) >= 2
        # 첫 번째가 더 높은 match_score를 가져야 함
        assert combos[0].match_score >= combos[1].match_score
        # 신뢰도가 높은 치킨마요가 첫 번째여야 함
        assert combos[0].items[0].candidate.name == "치킨마요"

    def test_default_matcher_rejects_four_item_exact_combo_when_single_candidate_exists(self):
        """현장 로그처럼 4개 조합이 단일 비전 후보를 이기면 안 된다."""
        matcher = StrictWeightMatcher()
        candidates = [
            self._make_candidate(31, "BOTTLE_WOONGJIN_SKY_BARLEY_TEA_500ML", 0.20),
            self._make_candidate(35, "BAG_NONGSHIM_CHAPAGETTI_140G", 0.99),
            self._make_candidate(40, "BOX_LOTTE_BINCH_102G", 0.99),
            self._make_candidate(99, "BOTTLE_CJ_PROTEIN_SHAKE_NUTS_50G", 0.99),
            self._make_candidate(120, "CUP_BIBIGO_TTEOKBOKKI_110G", 0.99),
        ]
        active_products = [
            self._make_active_product(31, "BOTTLE_WOONGJIN_SKY_BARLEY_TEA_500ML", 523.0),
            self._make_active_product(35, "BAG_NONGSHIM_CHAPAGETTI_140G", 149.0),
            self._make_active_product(40, "BOX_LOTTE_BINCH_102G", 130.0),
            self._make_active_product(99, "BOTTLE_CJ_PROTEIN_SHAKE_NUTS_50G", 85.0),
            self._make_active_product(120, "CUP_BIBIGO_TTEOKBOKKI_110G", 159.0),
        ]

        combos = matcher.find_valid_combinations(
            candidates=candidates,
            delta_weight=-523.0,
            active_products=active_products,
        )

        assert combos
        assert combos[0].total_count == 1
        assert combos[0].items[0].candidate.name == "BOTTLE_WOONGJIN_SKY_BARLEY_TEA_500ML"

    def test_single_candidate_within_tolerance_beats_two_item_exact_combo(self):
        """떡볶이 단일 후보가 ±3g 안이면 더 복잡한 조합보다 우선한다."""
        matcher = StrictWeightMatcher(tolerance=3.0, max_items=2)
        candidates = [
            self._make_candidate(120, "CUP_BIBIGO_TTEOKBOKKI_110G", 0.20),
            self._make_candidate(115, "BOX_LOTTE_PEPERO_ALMOND_37G", 0.99),
            self._make_candidate(99, "BOTTLE_CJ_PROTEIN_SHAKE_NUTS_50G", 0.99),
        ]
        active_products = [
            self._make_active_product(120, "CUP_BIBIGO_TTEOKBOKKI_110G", 144.0),
            self._make_active_product(115, "BOX_LOTTE_PEPERO_ALMOND_37G", 58.0),
            self._make_active_product(99, "BOTTLE_CJ_PROTEIN_SHAKE_NUTS_50G", 85.0),
        ]

        combos = matcher.find_valid_combinations(
            candidates=candidates,
            delta_weight=-143.0,
            active_products=active_products,
        )

        assert combos
        assert combos[0].total_count == 1
        assert combos[0].items[0].candidate.name == "CUP_BIBIGO_TTEOKBOKKI_110G"

    def test_compact_two_item_combo_beats_four_item_exact_combo(self):
        matcher = StrictWeightMatcher(tolerance=5.0, max_items=5)
        candidates = [
            self._make_candidate(101, "Scenario Product A", 0.23),
            self._make_candidate(102, "Scenario Product B", 0.22),
            self._make_candidate(103, "Scenario Product C", 0.24),
        ]
        active_products = [
            self._make_active_product(101, "Scenario Product A", 50.0),
            self._make_active_product(102, "Scenario Product B", 26.0),
            self._make_active_product(103, "Scenario Product C", 19.0),
        ]

        combos = matcher.find_valid_combinations(
            candidates=candidates,
            delta_weight=-77.0,
            active_products=active_products,
        )

        assert combos
        assert combos[0].total_count == 2
        assert {item.candidate.class_id: item.count for item in combos[0].items} == {
            101: 1,
            102: 1,
        }
        assert combos[0].weight_error == 1.0

    def test_low_confidence_two_kind_combo_uses_configured_floor(self):
        matcher = StrictWeightMatcher(tolerance=5.0, max_items=2)
        candidates = [
            self._make_candidate(101, "Scenario Product A", 0.20),
            self._make_candidate(102, "Scenario Product B", 0.23),
        ]
        active_products = [
            self._make_active_product(101, "Scenario Product A", 58.0),
            self._make_active_product(102, "Scenario Product B", 19.0),
        ]

        combos = matcher.find_valid_combinations(
            candidates=candidates,
            delta_weight=-81.0,
            active_products=active_products,
        )

        assert combos
        assert combos[0].total_count == 2
        assert combos[0].weight_error == 4.0

    def test_motion_evidence_repeated_count_beats_static_single_same_weight(self):
        matcher = StrictWeightMatcher(tolerance=3.0, max_items=3)
        candidates = [
            self._make_candidate(
                101,
                "Moved Product A",
                0.92,
                motion_gate_passed=True,
                top_motion_passed=True,
            ),
            self._make_candidate(
                102,
                "Static Product B",
                0.98,
                motion_gate_passed=False,
            ),
        ]
        active_products = [
            self._make_active_product(101, "Moved Product A", 50.0, stock=5),
            self._make_active_product(102, "Static Product B", 100.0, stock=5),
        ]

        combos = matcher.find_valid_combinations(
            candidates=candidates,
            delta_weight=-100.0,
            active_products=active_products,
        )

        assert combos
        assert combos[0].total_count == 2
        assert combos[0].items[0].candidate.class_id == 101
        assert combos[0].items[0].count == 2

    def test_motion_evidence_two_kind_combo_beats_static_single_same_weight(self):
        matcher = StrictWeightMatcher(tolerance=3.0, max_items=3, max_kinds=2)
        candidates = [
            self._make_candidate(
                101,
                "Moved Product A",
                0.90,
                motion_gate_passed=True,
                top_motion_passed=True,
            ),
            self._make_candidate(
                102,
                "Moved Product B",
                0.88,
                motion_gate_passed=True,
                side_motion_passed=True,
            ),
            self._make_candidate(
                103,
                "Static Product C",
                0.99,
                motion_gate_passed=False,
            ),
        ]
        active_products = [
            self._make_active_product(101, "Moved Product A", 40.0, stock=5),
            self._make_active_product(102, "Moved Product B", 60.0, stock=5),
            self._make_active_product(103, "Static Product C", 100.0, stock=5),
        ]

        combos = matcher.find_valid_combinations(
            candidates=candidates,
            delta_weight=-100.0,
            active_products=active_products,
        )

        assert combos
        assert combos[0].total_count == 2
        assert {item.candidate.class_id for item in combos[0].items} == {101, 102}


class TestCandidateProduct:
    """CandidateProduct 모델 테스트."""

    def test_candidate_product_creation(self):
        """CandidateProduct 생성 테스트."""
        cp = CandidateProduct(
            class_id=1,
            name="치킨마요",
            weight=365.0,
            stock=10,
            vision_confidence=0.95,
            unit_price=3500,
        )

        assert cp.class_id == 1
        assert cp.name == "치킨마요"
        assert cp.weight == 365.0
        assert cp.stock == 10
        assert cp.vision_confidence == 0.95
        assert cp.unit_price == 3500


class TestValidCombination:
    """ValidCombination 모델 테스트."""

    def test_valid_combination_totals(self):
        """ValidCombination 총계 계산 테스트."""
        from model_service.weight.strict_weight_matcher import CombinationItem

        cp1 = CandidateProduct(
            class_id=1,
            name="치킨마요",
            weight=365.0,
            stock=10,
            vision_confidence=0.9,
            unit_price=3500,
        )
        cp2 = CandidateProduct(
            class_id=2,
            name="참치마요",
            weight=250.0,
            stock=10,
            vision_confidence=0.8,
            unit_price=3000,
        )

        combo = ValidCombination(
            items=[
                CombinationItem(candidate=cp1, count=2),
                CombinationItem(candidate=cp2, count=1),
            ],
            total_weight=980.0,  # 365*2 + 250
            target_weight=980.0,
            weight_error=0.0,
            avg_vision_confidence=0.85,
            match_score=0.9,
        )

        # total_price = 3500*2 + 3000*1 = 10000
        assert combo.total_price == 10000
        # total_count = 2 + 1 = 3
        assert combo.total_count == 3

    def test_valid_combination_to_dict(self):
        """ValidCombination.to_dict() 테스트."""
        from model_service.weight.strict_weight_matcher import CombinationItem

        cp = CandidateProduct(
            class_id=1,
            name="치킨마요",
            weight=365.0,
            stock=10,
            vision_confidence=0.9,
            unit_price=3500,
        )

        combo = ValidCombination(
            items=[CombinationItem(candidate=cp, count=2)],
            total_weight=730.0,
            target_weight=730.0,
            weight_error=0.0,
            avg_vision_confidence=0.9,
            match_score=0.95,
        )

        data = combo.to_dict()

        assert data["total_weight"] == 730.0
        assert data["target_weight"] == 730.0
        assert data["weight_error"] == 0.0
        assert data["total_count"] == 2
        assert data["total_price"] == 7000
        assert len(data["items"]) == 1
        assert data["items"][0]["name"] == "치킨마요"
        assert data["items"][0]["count"] == 2


class TestFindReturnCombination:
    """반품 조합 탐색 테스트 (Phase 1)."""

    def _make_aggregated(self, products):
        """Helper: {product_id: AggregatedProduct} 생성."""
        from model_service.session.door_session import AggregatedProduct
        return {
            pid: AggregatedProduct(
                product_id=pid, product_idx=str(pid), name=name,
                count=count, unit_price=3500, weight=weight,
                total_confidence=0.9, detection_count=count,
            )
            for pid, name, weight, count in products
        }

    def test_same_product_two_returned(self):
        """같은 상품 2개 동시 반납: delta=730g, 상품=365g."""
        matcher = StrictWeightMatcher(tolerance=3.0)
        agg = self._make_aggregated([(26, "치킨마요", 365.0, 3)])
        result = matcher.find_return_combination(agg, 730.0)
        assert result is not None
        assert result[26] == 2

    def test_different_products_combo(self):
        """다른 상품 조합 반납: delta=615g = 365+250."""
        matcher = StrictWeightMatcher(tolerance=3.0)
        agg = self._make_aggregated([
            (26, "치킨마요", 365.0, 2),
            (27, "김밥", 250.0, 2),
        ])
        result = matcher.find_return_combination(agg, 615.0)
        assert result is not None
        assert result.get(26, 0) == 1
        assert result.get(27, 0) == 1

    def test_three_item_combo(self):
        """3개 상품 조합: delta=980g = 365+365+250."""
        matcher = StrictWeightMatcher(tolerance=5.0)
        agg = self._make_aggregated([
            (26, "치킨마요", 365.0, 3),
            (27, "김밥", 250.0, 2),
        ])
        result = matcher.find_return_combination(agg, 980.0)
        assert result is not None
        assert result.get(26, 0) == 2
        assert result.get(27, 0) == 1

    def test_no_combination_found(self):
        """조합 실패 → None."""
        matcher = StrictWeightMatcher(tolerance=3.0)
        agg = self._make_aggregated([(26, "치킨마요", 365.0, 2)])
        result = matcher.find_return_combination(agg, 500.0)
        assert result is None

    def test_insufficient_stock(self):
        """재고 부족 → 재고 한도 내에서만 조합 (실패)."""
        matcher = StrictWeightMatcher(tolerance=3.0)
        agg = self._make_aggregated([(26, "치킨마요", 365.0, 1)])
        # count=1인데 730g(2개) 반납 시도 → 실패
        result = matcher.find_return_combination(agg, 730.0)
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
