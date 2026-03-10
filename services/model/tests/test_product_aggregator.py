"""
Product Aggregator Tests (v4.1).

Tests:
- Single/multiple product aggregation
- Return handling (weight matching)
- Weight tolerance matching
- DB weight updates
"""

import pytest
from unittest.mock import MagicMock


class TestProductAggregatorBasic:
    """ProductAggregator 기본 기능 테스트."""

    def test_aggregate_single_removal(self, product_aggregator, sample_trigger_result):
        """단일 상품 제거 테스트."""
        aggregated = product_aggregator.aggregate([sample_trigger_result])

        assert len(aggregated) == 1
        assert 26 in aggregated

        product = aggregated[26]
        assert product.name == "치킨마요"
        assert product.count == 1
        assert product.unit_price == 3500

    def test_aggregate_multiple_removals(self, product_aggregator):
        """다중 상품 제거 테스트."""
        from model_service.session import ProductResult
        from model_service.session.door_session import TriggerResult
        import time

        trigger1 = TriggerResult(
            trigger_id="trigger_001",
            session_id="zone_1_test_1",
            timestamp=time.time(),
            products=[
                ProductResult(
                    product_id=26, product_idx="26", name="치킨마요",
                    count=2, price=3500, confidence=0.9
                )
            ],
            delta_weight=-730.0,
            confidence=0.9,
            video_paths={},
            is_return=False,
        )

        trigger2 = TriggerResult(
            trigger_id="trigger_002",
            session_id="zone_1_test_2",
            timestamp=time.time(),
            products=[
                ProductResult(
                    product_id=26, product_idx="26", name="치킨마요",
                    count=1, price=3500, confidence=0.85
                ),
                ProductResult(
                    product_id=15, product_idx="15", name="참치마요",
                    count=1, price=3000, confidence=0.88
                ),
            ],
            delta_weight=-615.0,
            confidence=0.86,
            video_paths={},
            is_return=False,
        )

        aggregated = product_aggregator.aggregate([trigger1, trigger2])

        assert len(aggregated) == 2
        assert aggregated[26].count == 3  # 2 + 1
        assert aggregated[15].count == 1


class TestProductAggregatorReturn:
    """반환 처리 테스트."""

    def test_handle_return_weight_match(self, product_aggregator):
        """반환 무게 매칭 테스트."""
        from model_service.session import ProductResult
        from model_service.session.door_session import TriggerResult
        import time

        # First: Remove 2 chicken mayo
        trigger1 = TriggerResult(
            trigger_id="trigger_001",
            session_id="zone_1_test_1",
            timestamp=time.time(),
            products=[
                ProductResult(
                    product_id=26, product_idx="26", name="치킨마요",
                    count=2, price=3500, confidence=0.9
                )
            ],
            delta_weight=-730.0,
            confidence=0.9,
            video_paths={},
            is_return=False,
        )

        # Second: Return one (weight ~365g)
        trigger2 = TriggerResult(
            trigger_id="trigger_002",
            session_id="zone_1_test_2",
            timestamp=time.time(),
            products=[],
            delta_weight=363.0,  # Within tolerance of 365g
            confidence=0.0,
            video_paths={},
            is_return=True,
        )

        aggregated = product_aggregator.aggregate([trigger1, trigger2])

        # Should have 1 chicken mayo left (2 - 1 = 1)
        assert aggregated[26].count == 1

    def test_handle_return_no_match(self, product_aggregator):
        """반환 매칭 실패 테스트."""
        from model_service.session import ProductResult
        from model_service.session.door_session import TriggerResult
        import time

        # First: Remove chicken mayo (365g)
        trigger1 = TriggerResult(
            trigger_id="trigger_001",
            session_id="zone_1_test_1",
            timestamp=time.time(),
            products=[
                ProductResult(
                    product_id=26, product_idx="26", name="치킨마요",
                    count=1, price=3500, confidence=0.9
                )
            ],
            delta_weight=-365.0,
            confidence=0.9,
            video_paths={},
            is_return=False,
        )

        # Second: Return with unmatched weight (500g - way off from 365g)
        trigger2 = TriggerResult(
            trigger_id="trigger_002",
            session_id="zone_1_test_2",
            timestamp=time.time(),
            products=[],
            delta_weight=500.0,  # No product with this weight
            confidence=0.0,
            video_paths={},
            is_return=True,
        )

        aggregated = product_aggregator.aggregate([trigger1, trigger2])

        # Count should remain unchanged (no match)
        assert aggregated[26].count == 1


class TestProductAggregatorWeightMatching:
    """무게 매칭 테스트."""

    def test_find_product_by_weight_exact(self, product_aggregator, sample_aggregated_products):
        """정확한 무게 매칭 테스트."""
        product_id = product_aggregator.find_product_by_weight(
            sample_aggregated_products, 365.0
        )
        assert product_id == 26

    def test_find_product_by_weight_tolerance(self, product_aggregator, sample_aggregated_products):
        """허용 오차 내 무게 매칭 테스트."""
        # 365 ± 3 = 362~368 범위
        product_id = product_aggregator.find_product_by_weight(
            sample_aggregated_products, 363.0
        )
        assert product_id == 26

        product_id = product_aggregator.find_product_by_weight(
            sample_aggregated_products, 367.0
        )
        assert product_id == 26

    def test_find_product_by_weight_out_of_tolerance(self, product_aggregator, sample_aggregated_products):
        """허용 오차 외 무게 매칭 실패 테스트."""
        # 365 ± 3 범위 밖
        product_id = product_aggregator.find_product_by_weight(
            sample_aggregated_products, 350.0
        )
        assert product_id is None

    def test_find_product_by_weight_zero_count(self, product_aggregator):
        """count=0인 상품 매칭 제외 테스트."""
        from model_service.session.door_session import AggregatedProduct

        aggregated = {
            26: AggregatedProduct(
                product_id=26, product_idx="26", name="치킨마요",
                count=0,  # Zero count
                unit_price=3500, weight=365.0,
            ),
        }

        product_id = product_aggregator.find_product_by_weight(aggregated, 365.0)
        assert product_id is None


class TestProductAggregatorWeightUpdate:
    """무게 업데이트 테스트."""

    def test_update_weights_from_db(self, product_aggregator):
        """DB에서 무게 업데이트 테스트."""
        from model_service.session.door_session import AggregatedProduct

        # Weight가 0인 상품
        aggregated = {
            26: AggregatedProduct(
                product_id=26, product_idx="26", name="치킨마요",
                count=1, unit_price=3500, weight=0.0,  # No weight
            ),
            99: AggregatedProduct(
                product_id=99, product_idx="99", name="미등록상품",
                count=1, unit_price=1000, weight=0.0,
            ),
        }

        def mock_get_weight(product_id: int) -> float:
            if product_id == 26:
                return 365.0
            return 0.0

        updated = product_aggregator.update_weights_from_db(aggregated, mock_get_weight)

        # 1개만 업데이트됨 (product_id=26)
        assert updated == 1
        assert aggregated[26].weight == 365.0
        assert aggregated[99].weight == 0.0  # Not found in DB


class TestReturnCountEstimation:
    """반품 개수 추정 테스트 (Phase 0b)."""

    def _make_aggregated(self, count=3):
        from model_service.session.door_session import AggregatedProduct
        return {
            26: AggregatedProduct(
                product_id=26, product_idx="26", name="치킨마요",
                count=count, unit_price=3500, weight=365.0,
                total_confidence=2.7, detection_count=count,
            )
        }

    def test_single_return_count_1(self):
        """delta≈단위무게 → count 1 차감."""
        from model_service.session.product_aggregator import ProductAggregator
        aggregator = ProductAggregator(weight_tolerance=3.0)
        agg = self._make_aggregated(count=3)
        aggregator._handle_return(agg, 365.0)
        assert agg[26].count == 2  # 3 - 1

    def test_single_return_does_not_go_negative(self):
        """count=1인데 1개 반납 → count=0에서 멈춤."""
        from model_service.session.product_aggregator import ProductAggregator
        aggregator = ProductAggregator(weight_tolerance=3.0)
        agg = self._make_aggregated(count=1)
        aggregator._handle_return(agg, 365.0)
        assert agg[26].count == 0
        assert agg[26].count >= 0  # 음수 방지


    def test_safety_guard_abnormally_large_deducts_only_1(self):
        """Phase 0b 안전장치: estimated_count > agg.count → 1개만 차감."""
        from model_service.session.product_aggregator import ProductAggregator
        from model_service.session.door_session import AggregatedProduct
        # Wide tolerance (200g) so delta=290g matches unit_weight=100g
        aggregator = ProductAggregator(weight_tolerance=200.0)
        agg = {
            99: AggregatedProduct(
                product_id=99, product_idx="99", name="TestItem",
                count=2, unit_price=1000, weight=100.0,
                total_confidence=1.8, detection_count=2,
            )
        }
        # delta=290g, |100-290|=190 < 200 tolerance → match
        # estimated_count = round(290/100) = 3 > count=2 → safety guard → deduct 1
        aggregator._handle_return(agg, 290.0)
        assert agg[99].count == 1  # 2 - 1 (safety guard, not 2 - 2)

    def test_normal_multi_return_deducts_correctly(self):
        """정상 범위 내 multi-return: 2개 반납 추정 시 2개 차감."""
        from model_service.session.product_aggregator import ProductAggregator
        aggregator = ProductAggregator(weight_tolerance=3.0)
        agg = self._make_aggregated(count=3)
        # delta_weight=730g for 365g product → estimated_count=2 <= count=3 → deduct 2
        aggregator._handle_return(agg, 730.0)
        assert agg[26].count == 1  # 3 - 2


class TestBatchReturn:
    """동시 다중 반품 조합 매칭 테스트 (Phase 1)."""

    def _make_trigger(self, delta, is_return=True):
        from model_service.session.door_session import TriggerResult
        import time
        return TriggerResult(
            trigger_id=f"t_{delta}",
            session_id="test",
            timestamp=time.time(),
            products=[],
            delta_weight=delta,
            confidence=0.0,
            video_paths={},
            is_return=is_return,
        )

    def _make_removal_trigger(self, product_id, name, weight, count, price=3500):
        from model_service.session.door_session import TriggerResult
        from model_service.session import ProductResult
        import time
        return TriggerResult(
            trigger_id=f"removal_{product_id}",
            session_id="test",
            timestamp=time.time(),
            products=[ProductResult(
                product_id=product_id, product_idx=str(product_id),
                name=name, count=count, price=price, confidence=0.9,
            )],
            delta_weight=-weight * count,
            confidence=0.9,
            video_paths={},
            is_return=False,
        )

    def test_batch_return_same_product(self):
        """치킨마요 2개 취출 후 동시 반납 (delta=730g)."""
        from model_service.session.product_aggregator import ProductAggregator
        weights = {26: 365.0}
        aggregator = ProductAggregator(
            weight_tolerance=3.0,
            get_product_weight=lambda pid: weights.get(pid, 0.0),
        )

        removal = self._make_removal_trigger(26, "치킨마요", 365.0, 2)
        return_t = self._make_trigger(730.0, is_return=True)

        result = aggregator.aggregate([removal, return_t])
        assert result[26].count == 0

    def test_batch_return_different_products(self):
        """치킨마요 1개 + 김밥 1개 동시 반납 (delta=615g)."""
        from model_service.session.product_aggregator import ProductAggregator
        weights = {26: 365.0, 27: 250.0}
        aggregator = ProductAggregator(
            weight_tolerance=3.0,
            get_product_weight=lambda pid: weights.get(pid, 0.0),
        )
        removal1 = self._make_removal_trigger(26, "치킨마요", 365.0, 1)
        removal2 = self._make_removal_trigger(27, "김밥", 250.0, 1)
        return_t = self._make_trigger(615.0, is_return=True)  # 365+250

        result = aggregator.aggregate([removal1, removal2, return_t])
        assert result[26].count == 0
        assert result[27].count == 0

    def test_unmatched_return_recorded(self):
        """매칭 불가 반납은 unmatched_returns에 기록."""
        from model_service.session.product_aggregator import ProductAggregator
        aggregator = ProductAggregator(weight_tolerance=3.0)
        removal = self._make_removal_trigger(26, "치킨마요", 365.0, 1)
        return_t = self._make_trigger(999.0, is_return=True)  # 매칭 불가

        agg_result = aggregator.aggregate_with_unmatched([removal, return_t])
        assert len(agg_result.unmatched_returns) == 1
        assert agg_result.unmatched_returns[0].delta_weight == 999.0


class TestProductAggregatorComplexScenarios:
    """복합 시나리오 테스트."""

    def test_aggregate_removal_then_return(self, product_aggregator):
        """제거 후 반환 시나리오 테스트."""
        from model_service.session import ProductResult
        from model_service.session.door_session import TriggerResult
        import time

        triggers = [
            # 1. 치킨마요 2개 제거
            TriggerResult(
                trigger_id="001", session_id="test1", timestamp=time.time(),
                products=[
                    ProductResult(product_id=26, product_idx="26", name="치킨마요",
                                  count=2, price=3500, confidence=0.9)
                ],
                delta_weight=-730.0, confidence=0.9, video_paths={}, is_return=False,
            ),
            # 2. 참치마요 1개 제거
            TriggerResult(
                trigger_id="002", session_id="test2", timestamp=time.time(),
                products=[
                    ProductResult(product_id=15, product_idx="15", name="참치마요",
                                  count=1, price=3000, confidence=0.85)
                ],
                delta_weight=-250.0, confidence=0.85, video_paths={}, is_return=False,
            ),
            # 3. 치킨마요 1개 반환
            TriggerResult(
                trigger_id="003", session_id="test3", timestamp=time.time(),
                products=[],
                delta_weight=363.0,  # ~365g
                confidence=0.0, video_paths={}, is_return=True,
            ),
        ]

        aggregated = product_aggregator.aggregate(triggers)

        # 최종: 치킨마요 1개 (2-1), 참치마요 1개
        assert aggregated[26].count == 1
        assert aggregated[15].count == 1
        assert aggregated[26].total_price == 3500
        assert aggregated[15].total_price == 3000
