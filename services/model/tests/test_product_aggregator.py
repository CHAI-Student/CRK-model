"""
Product Aggregator Tests (v4.1).

Tests:
- Single/multiple product aggregation
- Return handling (weight matching)
- Weight tolerance matching
- DB weight updates
"""



class TestDoorSessionNetDeltaRecovery:
    """Regression coverage for DoorSession net-delta repair."""

    def _make_store(self, tmp_path, weights):
        from model_service.session import DoorSessionStore

        return DoorSessionStore(
            yaml_dir=str(tmp_path),
            session_timeout=5.0,
            weight_tolerance=3.0,
            max_duration=60.0,
            get_product_weight=lambda product_id: weights.get(product_id, 0.0),
        )

    def _removal_trigger(self, product_id, product_idx, name, delta, price=3000):
        import time

        from model_service.session import ProductResult
        from model_service.session.door_session import TriggerResult

        return TriggerResult(
            trigger_id="",
            session_id="test-removal",
            timestamp=time.time(),
            products=[
                ProductResult(
                    product_id=product_id,
                    product_idx=product_idx,
                    name=name,
                    count=1,
                    price=price,
                    confidence=0.9,
                )
            ],
            delta_weight=delta,
            confidence=0.9,
            video_paths={},
            is_return=False,
        )

    def _return_trigger(self, delta):
        import time

        from model_service.session.door_session import TriggerResult

        return TriggerResult(
            trigger_id="",
            session_id="test-return",
            timestamp=time.time(),
            products=[],
            delta_weight=delta,
            confidence=0.0,
            video_paths={},
            is_return=True,
        )

    def test_light_product_under_read_removal_is_excluded_on_close(self, tmp_path):
        store = self._make_store(tmp_path, {113: 19.0})
        store.get_or_start_global_session()

        door_session = store.add_trigger_with_global(
            zone=1,
            result=self._removal_trigger(
                product_id=113,
                product_idx="P_CONDITION",
                name="STICK_INNON_CONDITION_STICK_18G",
                delta=-10.0,
            ),
        )

        active_products = door_session.get_active_products()
        assert len(active_products) == 1
        assert active_products[0].product_idx == "P_CONDITION"
        assert active_products[0].count == 1

        global_session = store.finalize_global_session()
        assert global_session is not None
        zone_session = global_session.zone_sessions[1]
        assert zone_session.get_active_products() == []
        diagnostics = zone_session.final_weight_validation
        assert diagnostics["reason"] == "unresolved_final_weight_mismatch"
        assert diagnostics["currentResidual"] == 9.0
        store.clear_all()

    def test_net_delta_recovery_runs_when_return_trigger_exists(self, tmp_path):
        store = self._make_store(tmp_path, {99: 100.0})
        store.get_or_start_global_session()

        store.add_trigger_with_global(
            zone=1,
            result=self._removal_trigger(
                product_id=99,
                product_idx="P99",
                name="TEST_PRODUCT_100G",
                delta=-100.0,
            ),
        )
        door_session = store.add_trigger_with_global(
            zone=1,
            result=self._return_trigger(delta=100.0),
        )

        active_products = door_session.get_active_products()
        assert active_products == []
        assert door_session.deferred_returns == []

        global_session = store.finalize_global_session()
        finalized_products = global_session.zone_sessions[1].get_active_products()
        assert finalized_products == []
        store.clear_all()

    def test_return_hint_effective_delta_reduces_raw_x2_to_single_item(self, tmp_path):
        import time

        from model_service.session import ProductResult
        from model_service.session.door_session import TriggerResult

        store = self._make_store(tmp_path, {44: 520.0})
        store.get_or_start_global_session()

        door_session = store.add_trigger_with_global(
            zone=5,
            result=TriggerResult(
                trigger_id="",
                session_id="zone-5-kwangdong-x2",
                timestamp=time.time(),
                products=[
                    ProductResult(
                        product_id=44,
                        product_idx="P44",
                        name="BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                        count=2,
                        price=2000,
                        confidence=0.5,
                    )
                ],
                delta_weight=-1046.0,
                confidence=0.5,
                video_paths={},
                is_return=False,
                return_weight_hints=[
                    {
                        "weight": 520.0,
                        "delta": 520.0,
                        "segment_index": 0,
                        "replay_position": "after_removal",
                    }
                ],
            ),
        )

        active_products = door_session.get_active_products()
        assert len(active_products) == 1
        assert active_products[0].name == "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML"
        assert active_products[0].count == 2
        assert len(door_session.deferred_returns) == 1

        global_session = store.finalize_global_session()
        finalized_products = global_session.zone_sessions[5].get_active_products()
        assert len(finalized_products) == 1
        assert finalized_products[0].count == 1
        diagnostics = global_session.zone_sessions[5].final_weight_validation[
            "deferredReturnReconciliation"
        ]
        assert diagnostics["accepted"] is True
        assert diagnostics["sameZoneApplied"][0]["matchedUnits"] == 1
        store.clear_all()

    def test_combo_return_is_deferred_and_reconciled_on_close(self, tmp_path):
        import time

        from model_service.session import ProductResult
        from model_service.session.door_session import TriggerResult

        store = self._make_store(tmp_path, {31: 523.0, 44: 520.0})
        store.get_or_start_global_session()

        store.add_trigger_with_global(
            zone=4,
            result=TriggerResult(
                trigger_id="",
                session_id="remove-two-bottles",
                timestamp=time.time(),
                products=[
                    ProductResult(
                        product_id=31,
                        product_idx="P31",
                        name="BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML",
                        count=1,
                        price=1200,
                        confidence=0.8,
                    ),
                    ProductResult(
                        product_id=44,
                        product_idx="P44",
                        name="BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                        count=1,
                        price=2000,
                        confidence=0.8,
                    ),
                ],
                delta_weight=-1043.0,
                confidence=0.8,
                video_paths={},
                is_return=False,
            ),
        )
        door_session = store.add_trigger_with_global(
            zone=4,
            result=self._return_trigger(delta=1033.0),
        )

        assert {product.product_id for product in door_session.get_active_products()} == {
            31,
            44,
        }
        assert len(door_session.deferred_returns) == 1

        global_session = store.finalize_global_session()
        finalized = global_session.zone_sessions[4]
        assert finalized.get_active_products() == []
        diagnostics = finalized.final_weight_validation[
            "deferredReturnReconciliation"
        ]
        assert diagnostics["accepted"] is True
        assert diagnostics["sameZoneApplied"][0]["matchedUnits"] == 2
        store.clear_all()

    def test_mixed_return_hint_is_deferred_and_reconciled_on_close(self, tmp_path):
        import time

        from model_service.session import DoorSessionStore, ProductResult
        from model_service.session.door_session import TriggerResult

        store = DoorSessionStore(
            yaml_dir=str(tmp_path),
            session_timeout=5.0,
            weight_tolerance=5.0,
            max_duration=60.0,
            get_product_weight=lambda product_id: {95: 220.0, 113: 19.0}.get(
                product_id,
                0.0,
            ),
        )
        store.get_or_start_global_session()
        store.add_trigger_with_global(
            zone=2,
            result=self._removal_trigger(
                product_id=95,
                product_idx="P95",
                name="BOTTLE_HY_HALUYACHE_PURPLE_200ML_V2",
                delta=-220.0,
                price=2000,
            ),
        )
        door_session = store.add_trigger_with_global(
            zone=2,
            result=TriggerResult(
                trigger_id="",
                session_id="mixed-return-hint",
                timestamp=time.time(),
                products=[
                    ProductResult(
                        product_id=113,
                        product_idx="P113",
                        name="STICK_INNON_CONDITION_STICK_18G",
                        count=1,
                        price=3000,
                        confidence=0.8,
                    )
                ],
                delta_weight=-19.0,
                confidence=0.8,
                video_paths={},
                is_return=False,
                return_weight_hints=[
                    {
                        "delta": 220.0,
                        "weight": 220.0,
                        "replay_position": "before_removal",
                    }
                ],
            ),
        )

        assert {product.product_id for product in door_session.get_active_products()} == {
            95,
            113,
        }
        assert len(door_session.deferred_returns) == 1

        global_session = store.finalize_global_session()
        finalized_products = global_session.zone_sessions[2].get_active_products()
        assert len(finalized_products) == 1
        assert finalized_products[0].product_id == 113
        diagnostics = global_session.zone_sessions[2].final_weight_validation[
            "deferredReturnReconciliation"
        ]
        assert diagnostics["accepted"] is True
        assert diagnostics["sameZoneApplied"][0]["matchedUnits"] == 1
        store.clear_all()

    def test_freezer_mixed_sign_cross_zone_uses_signed_net_close_aggregate(
        self,
        monkeypatch,
        tmp_path,
    ):
        import time

        from model_service.core.config import config
        from model_service.session import ProductResult
        from model_service.session.door_session import TriggerResult

        monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
        store = self._make_store(tmp_path, {70: 70.0, 150: 150.0})
        store.get_or_start_global_session()
        store.add_trigger_with_global(
            zone=3,
            result=self._removal_trigger(
                product_id=70,
                product_idx="P70",
                name="FREEZER_70G_ITEM",
                delta=-70.0,
            ),
        )
        store.add_trigger_with_global(
            zone=2,
            result=TriggerResult(
                trigger_id="",
                session_id="zone-2-mixed-sign",
                timestamp=time.time(),
                products=[
                    ProductResult(
                        product_id=150,
                        product_idx="P150",
                        name="FREEZER_150G_ITEM",
                        count=1,
                        price=4500,
                        confidence=0.9,
                    )
                ],
                delta_weight=-80.0,
                confidence=0.9,
                video_paths={},
                is_return=False,
                loadcell_diagnostics={
                    "mixed_sign_internal_segments": True,
                    "compound_positive_segment_count": 1,
                    "compound_negative_segment_count": 1,
                    "net_delta_weight": -80.0,
                    "decision_delta_weight": -80.0,
                },
            ),
        )

        global_session = store.finalize_global_session()

        zone_3 = global_session.zone_sessions[3]
        zone_2 = global_session.zone_sessions[2]
        assert [(p.product_id, p.count) for p in zone_3.get_active_products()] == [
            (70, 1)
        ]
        finalized_products = zone_2.get_active_products()
        assert len(finalized_products) == 1
        assert finalized_products[0].product_id == 150
        assert finalized_products[0].count == 1
        aggregate = zone_2.final_weight_validation["freezerCloseAggregate"]
        assert aggregate["accepted"] is False
        assert aggregate["reason"] == "freezer_close_candidate_rebuild_disabled"
        assert aggregate["freezerCloseAggregateMode"] == "preserve_only"
        assert aggregate["policy"] == "signed_net_delta"
        assert aggregate["globalNetDelta"] == -150.0
        assert aggregate["selectedProducts"] == [
            {"productId": 150, "name": "FREEZER_150G_ITEM", "count": 1},
            {"productId": 70, "name": "FREEZER_70G_ITEM", "count": 1},
        ]
        assert zone_2.cross_zone_returns == []
        store.clear_all()

    def test_freezer_mixed_sign_no_signed_net_fit_clears_provisional_charge(
        self,
        monkeypatch,
        tmp_path,
    ):
        import time

        from model_service.core.config import config
        from model_service.session import ProductResult
        from model_service.session.door_session import TriggerResult

        monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
        store = self._make_store(tmp_path, {150: 150.0})
        store.get_or_start_global_session()
        store.add_trigger_with_global(
            zone=2,
            result=TriggerResult(
                trigger_id="",
                session_id="zone-2-unmatched-mixed-sign",
                timestamp=time.time(),
                products=[
                    ProductResult(
                        product_id=150,
                        product_idx="P150",
                        name="FREEZER_150G_ITEM",
                        count=1,
                        price=4500,
                        confidence=0.9,
                    )
                ],
                delta_weight=-80.0,
                confidence=0.9,
                video_paths={},
                is_return=False,
                loadcell_diagnostics={
                    "mixed_sign_internal_segments": True,
                    "compound_positive_segment_count": 1,
                    "compound_negative_segment_count": 1,
                    "net_delta_weight": -80.0,
                    "decision_delta_weight": -80.0,
                },
            ),
        )

        global_session = store.finalize_global_session()

        zone_2 = global_session.zone_sessions[2]
        finalized_products = zone_2.get_active_products()
        assert [(p.product_id, p.count) for p in finalized_products] == [(150, 1)]
        assert zone_2.unmatched_returns == []
        aggregate = zone_2.final_weight_validation["freezerCloseAggregate"]
        assert aggregate["accepted"] is False
        assert aggregate["reason"] == "freezer_close_candidate_rebuild_disabled"
        assert aggregate["freezerCloseAggregateMode"] == "preserve_only"
        assert aggregate["globalNetDelta"] == -80.0
        assert aggregate["selectedProducts"] == [
            {"productId": 150, "name": "FREEZER_150G_ITEM", "count": 1}
        ]
        store.clear_all()


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
        import time

        from model_service.session import ProductResult
        from model_service.session.door_session import TriggerResult

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
        import time

        from model_service.session import ProductResult
        from model_service.session.door_session import TriggerResult

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
        import time

        from model_service.session import ProductResult
        from model_service.session.door_session import TriggerResult

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
        from model_service.session.door_session import AggregatedProduct
        from model_service.session.product_aggregator import ProductAggregator
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

    def test_multi_return_is_not_deducted_immediately(self):
        """정상 범위 내 multi-return: 2개 반납 추정 시 2개 차감."""
        from model_service.session.product_aggregator import ProductAggregator
        aggregator = ProductAggregator(weight_tolerance=3.0)
        agg = self._make_aggregated(count=3)
        # delta_weight=730g for 365g product → estimated_count=2 <= count=3 → deduct 2
        assert aggregator._handle_return(agg, 730.0) is None
        assert agg[26].count == 3


class TestBatchReturn:
    """동시 다중 반품 조합 매칭 테스트 (Phase 1)."""

    def _make_trigger(self, delta, is_return=True):
        import time

        from model_service.session.door_session import TriggerResult
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
        import time

        from model_service.session import ProductResult
        from model_service.session.door_session import TriggerResult
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

    def _make_multi_removal_trigger(self, products, timestamp):
        from model_service.session import ProductResult
        from model_service.session.door_session import TriggerResult

        return TriggerResult(
            trigger_id="removal_multi",
            session_id="test",
            timestamp=timestamp,
            products=[
                ProductResult(
                    product_id=product_id,
                    product_idx=str(product_id),
                    name=name,
                    count=count,
                    price=price,
                    confidence=0.9,
                )
                for product_id, name, count, price in products
            ],
            delta_weight=-897.0,
            confidence=0.9,
            video_paths={},
            is_return=False,
        )

    def _make_timestamped_return_trigger(self, delta, timestamp):
        from model_service.session.door_session import TriggerResult

        return TriggerResult(
            trigger_id=f"return_{delta}",
            session_id="test",
            timestamp=timestamp,
            products=[],
            delta_weight=delta,
            confidence=0.0,
            video_paths={},
            is_return=True,
        )

    def _make_mixed_return_removal_trigger(
        self,
        *,
        timestamp,
        hint_weight,
        product_id,
        name,
        delta=-16.5,
        price=3000,
    ):
        from model_service.session import ProductResult
        from model_service.session.door_session import TriggerResult

        return TriggerResult(
            trigger_id="mixed_return_removal",
            session_id="test",
            timestamp=timestamp,
            products=[
                ProductResult(
                    product_id=product_id,
                    product_idx=str(product_id),
                    name=name,
                    count=1,
                    price=price,
                    confidence=0.9,
                )
            ],
            delta_weight=delta,
            confidence=0.9,
            video_paths={},
            is_return=False,
            return_weight_hints=[
                {
                    "weight": hint_weight,
                    "delta": hint_weight,
                    "segment_index": 0,
                    "replay_position": "before_removal",
                    "reason": "unpaired_return_segment",
                }
            ],
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

        result = aggregator.aggregate_with_unmatched([removal, return_t])
        assert result.products[26].count == 2
        assert result.unmatched_returns == []
        assert len(result.deferred_returns) == 1
        assert result.deferred_returns[0].delta_weight == 730.0

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

        result = aggregator.aggregate_with_unmatched([removal1, removal2, return_t])
        assert result.products[26].count == 1
        assert result.products[27].count == 1
        assert result.unmatched_returns == []
        assert len(result.deferred_returns) == 1

    def test_unmatched_return_recorded(self):
        """매칭 불가 반납은 unmatched_returns에 기록."""
        from model_service.session.product_aggregator import ProductAggregator
        aggregator = ProductAggregator(weight_tolerance=3.0)
        removal = self._make_removal_trigger(26, "치킨마요", 365.0, 1)
        return_t = self._make_trigger(999.0, is_return=True)  # 매칭 불가

        agg_result = aggregator.aggregate_with_unmatched([removal, return_t])
        assert agg_result.unmatched_returns == []
        assert len(agg_result.deferred_returns) == 1
        assert agg_result.deferred_returns[0].delta_weight == 999.0

    def test_trevi_return_relaxed_tolerance_leaves_king_rush_only(self):
        from model_service.session.product_aggregator import ProductAggregator

        weights = {54: 530.0, 8: 367.0}
        aggregator = ProductAggregator(
            weight_tolerance=5.0,
            get_product_weight=lambda pid: weights.get(pid, 0.0),
        )
        removal = self._make_multi_removal_trigger(
            [
                (54, "BOTTLE_LOTTE_TREVI_LEMON_500ML", 1, 1600),
                (8, "CAN_LOTTE_HOT6_THE_KING_RUSH_355ML", 1, 1300),
            ],
            timestamp=10.0,
        )
        return_t = self._make_timestamped_return_trigger(524.0, timestamp=20.0)

        result = aggregator.aggregate_with_unmatched([removal, return_t])

        assert result.products[54].count == 1
        assert result.products[8].count == 1
        assert len(result.deferred_returns) == 1

    def test_return_replays_by_event_timestamp_even_when_inserted_first(self):
        from model_service.session.product_aggregator import ProductAggregator

        weights = {54: 530.0, 8: 367.0}
        aggregator = ProductAggregator(
            weight_tolerance=5.0,
            get_product_weight=lambda pid: weights.get(pid, 0.0),
        )
        removal = self._make_multi_removal_trigger(
            [
                (54, "BOTTLE_LOTTE_TREVI_LEMON_500ML", 1, 1600),
                (8, "CAN_LOTTE_HOT6_THE_KING_RUSH_355ML", 1, 1300),
            ],
            timestamp=10.0,
        )
        return_t = self._make_timestamped_return_trigger(524.0, timestamp=20.0)

        result = aggregator.aggregate_with_unmatched([return_t, removal])

        assert result.products[54].count == 1
        assert result.products[8].count == 1
        assert len(result.deferred_returns) == 1

    def test_return_outside_relaxed_tolerance_stays_unmatched(self):
        from model_service.session.product_aggregator import ProductAggregator

        weights = {54: 530.0}
        aggregator = ProductAggregator(
            weight_tolerance=5.0,
            get_product_weight=lambda pid: weights.get(pid, 0.0),
        )
        removal = self._make_multi_removal_trigger(
            [(54, "BOTTLE_LOTTE_TREVI_LEMON_500ML", 1, 1600)],
            timestamp=10.0,
        )
        return_t = self._make_timestamped_return_trigger(515.0, timestamp=20.0)

        result = aggregator.aggregate_with_unmatched([removal, return_t])

        assert result.products[54].count == 1
        assert result.unmatched_returns == []
        assert len(result.deferred_returns) == 1

    def test_return_combo_uses_count_scaled_tolerance_for_two_500ml_bottles(self):
        from model_service.session.product_aggregator import ProductAggregator

        weights = {31: 523.0, 44: 520.0}
        aggregator = ProductAggregator(
            weight_tolerance=5.0,
            get_product_weight=lambda pid: weights.get(pid, 0.0),
        )
        removal = self._make_multi_removal_trigger(
            [
                (31, "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML", 1, 1200),
                (44, "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML", 1, 2000),
            ],
            timestamp=10.0,
        )
        return_t = self._make_timestamped_return_trigger(1033.0, timestamp=20.0)

        result = aggregator.aggregate_with_unmatched([removal, return_t])

        assert result.products[31].count == 1
        assert result.products[44].count == 1
        assert result.unmatched_returns == []
        assert len(result.deferred_returns) == 1
        assert result.deferred_returns[0].source == "positive_return"

    def test_mixed_return_hint_removes_previous_product_before_new_removal(self):
        from model_service.session.product_aggregator import ProductAggregator

        weights = {95: 220.0, 113: 19.0}
        aggregator = ProductAggregator(
            weight_tolerance=5.0,
            get_product_weight=lambda pid: weights.get(pid, 0.0),
        )
        haluyache = self._make_removal_trigger(
            95,
            "BOTTLE_HY_HALUYACHE_PURPLE_200ML_V2",
            220.0,
            1,
            price=2000,
        )
        haluyache.timestamp = 10.0
        condition = self._make_mixed_return_removal_trigger(
            timestamp=20.0,
            hint_weight=216.7,
            product_id=113,
            name="STICK_INNON_CONDITION_STICK_18G",
            delta=-16.5,
        )

        result = aggregator.aggregate_with_unmatched([haluyache, condition])

        assert result.products[95].count == 1
        assert result.products[113].count == 1
        assert result.unmatched_returns == []
        assert len(result.deferred_returns) == 1
        assert result.deferred_returns[0].source == "mixed_return_hint"

    def test_mixed_return_hint_replays_by_timestamp_when_inserted_first(self):
        from model_service.session.product_aggregator import ProductAggregator

        weights = {95: 220.0, 113: 19.0}
        aggregator = ProductAggregator(
            weight_tolerance=5.0,
            get_product_weight=lambda pid: weights.get(pid, 0.0),
        )
        haluyache = self._make_removal_trigger(
            95,
            "BOTTLE_HY_HALUYACHE_PURPLE_200ML_V2",
            220.0,
            1,
            price=2000,
        )
        haluyache.timestamp = 10.0
        condition = self._make_mixed_return_removal_trigger(
            timestamp=20.0,
            hint_weight=216.7,
            product_id=113,
            name="STICK_INNON_CONDITION_STICK_18G",
            delta=-16.5,
        )

        result = aggregator.aggregate_with_unmatched([condition, haluyache])

        assert result.products[95].count == 1
        assert result.products[113].count == 1
        assert result.unmatched_returns == []
        assert len(result.deferred_returns) == 1

    def test_unmatched_mixed_return_hint_keeps_new_removal(self):
        from model_service.session.product_aggregator import ProductAggregator

        weights = {113: 19.0}
        aggregator = ProductAggregator(
            weight_tolerance=5.0,
            get_product_weight=lambda pid: weights.get(pid, 0.0),
        )
        condition = self._make_mixed_return_removal_trigger(
            timestamp=20.0,
            hint_weight=216.7,
            product_id=113,
            name="STICK_INNON_CONDITION_STICK_18G",
            delta=-16.5,
        )

        result = aggregator.aggregate_with_unmatched([condition])

        assert result.products[113].count == 1
        assert result.unmatched_returns == []
        assert len(result.deferred_returns) == 1
        assert result.deferred_returns[0].delta_weight == 216.7


class TestProductAggregatorComplexScenarios:
    """복합 시나리오 테스트."""

    def test_aggregate_removal_then_return(self, product_aggregator):
        """제거 후 반환 시나리오 테스트."""
        import time

        from model_service.session import ProductResult
        from model_service.session.door_session import TriggerResult

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


class TestFreezerLocationReturnAggregation:
    def _removal_trigger(self, *, side: str, product_id: int, weight: float):
        from model_service.session import ProductResult
        from model_service.session.door_session import TriggerResult

        return TriggerResult(
            trigger_id="trigger_001",
            session_id="remove",
            timestamp=10.0,
            products=[
                ProductResult(
                    product_id=product_id,
                    product_idx=f"P{product_id}",
                    name=f"PRODUCT_{product_id}",
                    count=1,
                    price=1000,
                    confidence=0.9,
                    placement_units=[
                        {
                            "zone": 2,
                            "channelSide": side,
                            "channelIndex": 0 if side == "left" else 1,
                            "channelPosition": 0 if side == "left" else 1,
                            "sourceTriggerId": "trigger_001",
                            "sourceTimestamp": 10.0,
                        }
                    ],
                )
            ],
            delta_weight=-weight,
            confidence=0.9,
            video_paths={},
            is_return=False,
        )

    def _return_trigger(self, *, side: str, weight: float):
        from model_service.session.door_session import TriggerResult

        return TriggerResult(
            trigger_id="trigger_002",
            session_id="return",
            timestamp=20.0,
            products=[],
            delta_weight=weight,
            confidence=0.0,
            video_paths={},
            is_return=True,
            loadcell_diagnostics={
                "channel_movement_targets": [
                    {
                        "source": "stable_channel_delta",
                        "direction": "return",
                        "weight": weight,
                        "delta": weight,
                        "channel_index": 0 if side == "left" else 1,
                        "channel_position": 0 if side == "left" else 1,
                        "channel_side": side,
                    }
                ]
            },
        )

    def test_freezer_return_matches_same_zone_same_side_first(self):
        from model_service.session.product_aggregator import ProductAggregator

        aggregator = ProductAggregator(
            weight_tolerance=5.0,
            get_product_weight=lambda pid: {101: 70.0, 102: 70.0}.get(pid, 0.0),
        )

        result = aggregator.aggregate_with_unmatched(
            [
                self._removal_trigger(side="left", product_id=101, weight=70.0),
                self._removal_trigger(side="right", product_id=102, weight=70.0),
                self._return_trigger(side="left", weight=70.0),
            ],
            zone=2,
        )

        assert result.products[101].count == 0
        assert result.products[102].count == 1
        assert result.products[101].placement_units == []
        assert result.unmatched_returns == []
        diagnostics = result.location_return_diagnostics[0]
        assert diagnostics["accepted"] is True
        assert diagnostics["targets"][0]["matchTier"] == "same_zone_same_side"

    def test_freezer_return_falls_back_to_same_zone_other_side(self):
        from model_service.session.product_aggregator import ProductAggregator

        aggregator = ProductAggregator(
            weight_tolerance=5.0,
            get_product_weight=lambda pid: {101: 70.0}.get(pid, 0.0),
        )

        result = aggregator.aggregate_with_unmatched(
            [
                self._removal_trigger(side="right", product_id=101, weight=70.0),
                self._return_trigger(side="left", weight=70.0),
            ],
            zone=2,
        )

        assert result.products[101].count == 0
        assert result.unmatched_returns == []
        diagnostics = result.location_return_diagnostics[0]
        assert diagnostics["targets"][0]["matchTier"] == "same_zone_other_side"
