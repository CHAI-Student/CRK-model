import logging
import time

import pytest


def _make_global_session():
    from model_service.session.door_session import (
        AggregatedProduct,
        DoorSession,
        TriggerResult,
    )
    from model_service.session.global_door_session import GlobalDoorSession
    from model_service.session.session_store import ProductResult

    product = ProductResult(
        product_id=8,
        product_idx="P8",
        name="STICK_INNON_CONDITION_STICK_18G",
        count=1,
        price=3000,
        confidence=0.82,
    )
    trigger = TriggerResult(
        trigger_id="trigger-1",
        session_id="zone-1-session",
        timestamp=time.time(),
        products=[product],
        delta_weight=-10.0,
        confidence=0.82,
        video_paths={"top": "top.avi", "side": "side.avi"},
        is_return=False,
    )
    zone_session = DoorSession(
        door_session_id="door-zone-1",
        zone=1,
        status="complete",
        triggers=[trigger],
        aggregated_products={
            8: AggregatedProduct(
                product_id=8,
                product_idx="P8",
                name="STICK_INNON_CONDITION_STICK_18G",
                count=1,
                unit_price=3000,
                weight=19.0,
                total_confidence=0.82,
                detection_count=1,
            )
        },
        finalized_at=time.time(),
    )
    return GlobalDoorSession(
        global_session_id="global-test",
        status="complete",
        zone_sessions={1: zone_session},
        finalized_at=time.time(),
    )


class FakeCloseReadyStore:
    def __init__(self, global_session):
        self.global_session = global_session

    def handle_close_signal(self):
        return True, self.global_session

    def finalize_global_session(self):
        return self.global_session


def _candidate_snapshot(
    product_id,
    name,
    *,
    rank,
    weight,
    price=1200,
    confidence=0.4,
    identity_confidence=None,
    top_confidence=None,
    side_confidence=0.0,
    source="vision",
    stock=10,
):
    raw_identity = confidence if identity_confidence is None else identity_confidence
    top_raw = raw_identity if top_confidence is None else top_confidence
    side_raw = side_confidence
    return {
        "rank": rank,
        "product_id": product_id,
        "product_idx": f"P{product_id}",
        "name": name,
        "unit_weight": weight,
        "unit_price": price,
        "stock_qty": stock,
        "confidence": confidence,
        "identity_confidence": raw_identity,
        "source": source,
        "top": top_raw > 0,
        "side": side_raw > 0,
        "top_confidence": top_raw,
        "side_confidence": side_raw,
    }


def _product(product_id, name, count, price=1200, confidence=0.7):
    from model_service.session.session_store import ProductResult

    return ProductResult(
        product_id=product_id,
        product_idx=f"P{product_id}",
        name=name,
        count=count,
        price=price,
        confidence=confidence,
    )


def _trigger_result(
    *,
    product_id,
    name,
    delta,
    timestamp,
    count=1,
    weight=None,
    price=1200,
    return_weight_hints=None,
    trigger_id=None,
    loadcell_diagnostics=None,
):
    from model_service.session.door_session import TriggerResult

    return TriggerResult(
        trigger_id=trigger_id or f"trigger-{product_id}-{timestamp}",
        session_id=f"session-{product_id}-{timestamp}",
        timestamp=timestamp,
        products=[_product(product_id, name, count, price=price)],
        delta_weight=delta,
        confidence=0.9,
        video_paths={},
        is_return=False,
        return_weight_hints=list(return_weight_hints or []),
        loadcell_diagnostics=dict(loadcell_diagnostics or {}),
        vision_candidates=[
            _candidate_snapshot(
                product_id,
                name,
                rank=1,
                weight=weight if weight is not None else abs(delta),
                price=price,
                confidence=0.9,
            )
        ],
    )


def test_complete_close_response_includes_decision_summary(caplog):
    from model_service.api.routes.multi_zone import _handle_door_close

    caplog.set_level(logging.INFO)
    global_session = _make_global_session()
    response = _handle_door_close(FakeCloseReadyStore(global_session))

    summary = response["decisionSummary"]
    assert response["status"] == "success"
    assert summary["globalSessionId"] == "global-test"
    assert summary["totalWeightDelta"] == -10.0
    assert summary["totalProductCount"] == 1
    assert summary["totalPrice"] == 3000
    assert len(summary["zones"]) == 5

    zone_1 = summary["zones"][0]
    assert zone_1["zone"] == 1
    assert zone_1["weightDelta"] == -10.0
    assert zone_1["triggerCount"] == 1
    assert zone_1["productCount"] == 1
    assert zone_1["products"] == response["zones"][0]["products"]
    assert zone_1["products"][0]["name"] == "STICK_INNON_CONDITION_STICK_18G"

    empty_zone = summary["zones"][1]
    assert empty_zone["zone"] == 2
    assert empty_zone["weightDelta"] == 0.0
    assert empty_zone["triggerCount"] == 0
    assert empty_zone["productCount"] == 0
    assert empty_zone["products"] == []

    assert response["decisionSummaryText"] == (
        "zone=1 weight_delta=-10.0g products=STICK_INNON_CONDITION_STICK_18Gx1; "
        "total_weight_delta=-10.0g total_price=3000"
    )
    assert summary["zoneLines"][0] == (
        "zone=1 weight_delta=-10.0g products=STICK_INNON_CONDITION_STICK_18Gx1"
    )
    assert "STICK_INNON_CONDITION_STICK_18G" in summary["summaryText"]

    assert "[MULTI-ZONE CLOSE][SUMMARY]" in caplog.text
    assert "zone=1 weight_delta=-10.0g" in caplog.text


def test_freezer_close_aggregate_reroutes_mixed_sign_to_last_zone(
    monkeypatch,
    tmp_path,
):
    from model_service.api.routes.multi_zone import _handle_door_close
    from model_service.core.config import config
    from model_service.session import DoorSessionStore

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    store = DoorSessionStore(
        yaml_dir=str(tmp_path),
        get_product_weight=lambda product_id: {70: 70.0, 150: 150.0}.get(
            product_id,
            0.0,
        ),
    )
    try:
        store.get_or_start_global_session()
        store.add_trigger_with_global(
            zone=3,
            result=_trigger_result(
                product_id=70,
                name="FREEZER_70G_ITEM",
                delta=-70.0,
                weight=70.0,
                timestamp=100.0,
            ),
        )
        store.add_trigger_with_global(
            zone=2,
            result=_trigger_result(
                product_id=150,
                name="FREEZER_150G_ITEM",
                delta=-80.0,
                weight=150.0,
                timestamp=110.0,
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
        assert zone_3.get_active_products() == []
        assert [(p.product_id, p.count) for p in zone_2.get_active_products()] == [
            (150, 1)
        ]
        aggregate = zone_2.final_weight_validation["freezerCloseAggregate"]
        assert aggregate["accepted"] is True
        assert aggregate["policy"] == "signed_net_delta"
        assert aggregate["outputZone"] == 2
        assert aggregate["globalNetDelta"] == -150.0
        assert aggregate["finalTargetWeight"] == 150.0
        assert aggregate["selectedProducts"] == [
            {"productId": 150, "name": "FREEZER_150G_ITEM", "count": 1}
        ]

        response = _handle_door_close(FakeCloseReadyStore(global_session))
        zone_2_response = next(zone for zone in response["zones"] if zone["zone"] == 2)
        zone_3_response = next(zone for zone in response["zones"] if zone["zone"] == 3)
        assert zone_3_response["products"] == []
        assert zone_3_response["weightDelta"] == 0.0
        assert zone_2_response["products"][0]["name"] == "FREEZER_150G_ITEM"
        assert zone_2_response["weightDelta"] == -150.0
        assert response["decisionSummary"]["totalWeightDelta"] == -150.0
    finally:
        store.clear_all()


def test_freezer_close_aggregate_clears_mixed_sign_when_net_target_has_no_fit(
    monkeypatch,
    tmp_path,
):
    from model_service.api.routes.multi_zone import _handle_door_close
    from model_service.core.config import config
    from model_service.session import DoorSessionStore

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    store = DoorSessionStore(
        yaml_dir=str(tmp_path),
        get_product_weight=lambda product_id: {150: 150.0}.get(product_id, 0.0),
    )
    try:
        store.get_or_start_global_session()
        store.add_trigger_with_global(
            zone=2,
            result=_trigger_result(
                product_id=150,
                name="FREEZER_150G_ITEM",
                delta=-80.0,
                weight=150.0,
                timestamp=110.0,
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
        assert zone_2.get_active_products() == []
        aggregate = zone_2.final_weight_validation["freezerCloseAggregate"]
        assert aggregate["accepted"] is False
        assert aggregate["policy"] == "signed_net_delta"
        assert aggregate["globalNetDelta"] == -80.0
        assert aggregate["finalTargetWeight"] == 80.0
        assert aggregate["selectedProducts"] == []
        assert aggregate["noChargeReason"] == (
            "no_candidate_combination_for_signed_net_delta"
        )

        response = _handle_door_close(FakeCloseReadyStore(global_session))
        zone_2_response = next(zone for zone in response["zones"] if zone["zone"] == 2)
        assert zone_2_response["products"] == []
        assert zone_2_response["weightDelta"] == -80.0
        assert response["decisionSummary"]["totalPrice"] == 0
    finally:
        store.clear_all()


def test_freezer_close_aggregate_combines_multiple_zones_to_last_trigger_zone(
    monkeypatch,
    tmp_path,
):
    from model_service.api.routes.multi_zone import _handle_door_close
    from model_service.core.config import config
    from model_service.session import DoorSessionStore

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    store = DoorSessionStore(
        yaml_dir=str(tmp_path),
        get_product_weight=lambda product_id: {100: 100.0, 200: 200.0}.get(
            product_id,
            0.0,
        ),
    )
    try:
        store.get_or_start_global_session()
        store.add_trigger_with_global(
            zone=1,
            result=_trigger_result(
                product_id=100,
                name="FREEZER_100G_ITEM",
                delta=-100.0,
                weight=100.0,
                timestamp=100.0,
            ),
        )
        store.add_trigger_with_global(
            zone=4,
            result=_trigger_result(
                product_id=200,
                name="FREEZER_200G_ITEM",
                delta=-200.0,
                weight=200.0,
                timestamp=120.0,
            ),
        )

        global_session = store.finalize_global_session()
        response = _handle_door_close(FakeCloseReadyStore(global_session))
        zone_1 = next(zone for zone in response["zones"] if zone["zone"] == 1)
        zone_4 = next(zone for zone in response["zones"] if zone["zone"] == 4)
        assert [product["name"] for product in zone_1["products"]] == [
            "FREEZER_100G_ITEM"
        ]
        assert zone_1["weightDelta"] == -100.0
        assert [product["name"] for product in zone_4["products"]] == [
            "FREEZER_200G_ITEM"
        ]
        assert zone_4["weightDelta"] == -200.0
        assert response["decisionSummary"]["totalWeightDelta"] == -300.0
        aggregate = global_session.zone_sessions[4].final_weight_validation[
            "freezerCloseAggregate"
        ]
        assert aggregate["reason"] == (
            "freezer_close_aggregate_trigger_products_preserved"
        )
        assert aggregate["triggerSelectedWeight"] == 300.0
        assert aggregate["triggerSelectedResidual"] == 0.0
    finally:
        store.clear_all()


def test_freezer_close_aggregate_prefers_distinct_mixed_over_repeat(
    monkeypatch,
    tmp_path,
):
    from model_service.api.routes.multi_zone import _handle_door_close
    from model_service.core.config import config
    from model_service.session import DoorSessionStore
    from model_service.session.door_session import TriggerResult

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
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
    weights = {101: 100.0, 102: 102.0, 103: 103.0}
    store = DoorSessionStore(
        yaml_dir=str(tmp_path),
        get_product_weight=lambda product_id: weights.get(product_id, 0.0),
    )

    def trigger(seq, timestamp):
        return TriggerResult(
            trigger_id=f"trigger-{seq}",
            session_id=f"session-{seq}",
            timestamp=timestamp,
            products=[_product(101, "FREEZER_A", 1)],
            delta_weight=-150.0,
            confidence=0.9,
            video_paths={},
            is_return=False,
            vision_candidates=[
                _candidate_snapshot(
                    101,
                    "FREEZER_A",
                    rank=1,
                    weight=100.0,
                    confidence=0.95,
                ),
                _candidate_snapshot(
                    102,
                    "FREEZER_B",
                    rank=2,
                    weight=102.0,
                    confidence=0.93,
                ),
                _candidate_snapshot(
                    103,
                    "FREEZER_C",
                    rank=3,
                    weight=103.0,
                    confidence=0.91,
                ),
            ],
        )

    try:
        store.get_or_start_global_session()
        store.add_trigger_with_global(zone=1, result=trigger(1, 100.0))
        store.add_trigger_with_global(zone=1, result=trigger(2, 120.0))

        global_session = store.finalize_global_session()
        response = _handle_door_close(FakeCloseReadyStore(global_session))
        zone_1 = next(zone for zone in response["zones"] if zone["zone"] == 1)

        assert [(product["productId"], product["count"]) for product in zone_1["products"]] == [
            (101, 1),
            (102, 1),
            (103, 1),
        ]
        aggregate = global_session.zone_sessions[1].final_weight_validation[
            "freezerCloseAggregate"
        ]
        assert aggregate["selectedWeight"] == 305.0
        assert aggregate["residual"] == 5.0
        assert aggregate["selectedProducts"] == [
            {"productId": 101, "name": "FREEZER_A", "count": 1},
            {"productId": 102, "name": "FREEZER_B", "count": 1},
            {"productId": 103, "name": "FREEZER_C", "count": 1},
        ]
    finally:
        store.clear_all()


def test_freezer_close_aggregate_preserves_trigger_products_when_total_fits(
    monkeypatch,
    tmp_path,
):
    from model_service.api.routes.multi_zone import _handle_door_close
    from model_service.core.config import config
    from model_service.session import DoorSessionStore
    from model_service.session.door_session import TriggerResult

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "top_confidence_threshold", 0.70)
    monkeypatch.setattr(config.vision, "side_confidence_threshold", 0.70)
    weights = {13: 156.0, 23: 176.0, 44: 79.0}
    store = DoorSessionStore(
        yaml_dir=str(tmp_path),
        get_product_weight=lambda product_id: weights.get(product_id, 0.0),
    )
    try:
        store.get_or_start_global_session()
        store.add_trigger_with_global(
            zone=1,
            result=TriggerResult(
                trigger_id="zone1-bagel",
                session_id="zone1-bagel-session",
                timestamp=100.0,
                products=[
                    _product(
                        13,
                        "BAG_NULLDAM_BAGEL_140G",
                        1,
                        price=2800,
                        confidence=1.0,
                    )
                ],
                delta_weight=-147.0,
                confidence=1.0,
                video_paths={},
                vision_candidates=[
                    _candidate_snapshot(
                        13,
                        "BAG_NULLDAM_BAGEL_140G",
                        rank=1,
                        weight=156.0,
                        price=2800,
                        confidence=1.0,
                        identity_confidence=1.0,
                    ),
                    _candidate_snapshot(
                        44,
                        "STICK_BINGGRAE_MELONA_75ML",
                        rank=3,
                        weight=79.0,
                        price=800,
                        confidence=0.8,
                        identity_confidence=0.8,
                    ),
                ],
            ),
        )
        store.add_trigger_with_global(
            zone=4,
            result=TriggerResult(
                trigger_id="zone4-burger",
                session_id="zone4-burger-session",
                timestamp=120.0,
                products=[
                    _product(
                        23,
                        "BAG_HANMAC_TRIPLE_CHEESE_HAMBURGER_155G",
                        1,
                        price=2700,
                        confidence=1.0,
                    )
                ],
                delta_weight=-176.3,
                confidence=1.0,
                video_paths={},
                vision_candidates=[
                    _candidate_snapshot(
                        23,
                        "BAG_HANMAC_TRIPLE_CHEESE_HAMBURGER_155G",
                        rank=1,
                        weight=176.0,
                        price=2700,
                        confidence=1.0,
                        identity_confidence=1.0,
                    ),
                    _candidate_snapshot(
                        44,
                        "STICK_BINGGRAE_MELONA_75ML",
                        rank=3,
                        weight=79.0,
                        price=800,
                        confidence=0.8,
                        identity_confidence=0.8,
                    ),
                ],
            ),
        )

        global_session = store.finalize_global_session()
        response = _handle_door_close(FakeCloseReadyStore(global_session))

        zone_1 = next(zone for zone in response["zones"] if zone["zone"] == 1)
        zone_4 = next(zone for zone in response["zones"] if zone["zone"] == 4)
        assert [product["name"] for product in zone_1["products"]] == [
            "BAG_NULLDAM_BAGEL_140G"
        ]
        assert [product["name"] for product in zone_4["products"]] == [
            "BAG_HANMAC_TRIPLE_CHEESE_HAMBURGER_155G"
        ]
        assert response["decisionSummary"]["totalWeightDelta"] == -323.3
        assert response["decisionSummary"]["totalPrice"] == 5500
        aggregate = global_session.zone_sessions[4].final_weight_validation[
            "freezerCloseAggregate"
        ]
        assert aggregate["reason"] == (
            "freezer_close_aggregate_trigger_products_preserved"
        )
        assert aggregate["globalNetDelta"] == -323.3
        assert aggregate["triggerSelectedWeight"] == 332.0
        assert aggregate["triggerSelectedResidual"] == 8.7
        assert sorted(
            aggregate["selectedProducts"],
            key=lambda product: product["productId"],
        ) == [
            {"productId": 13, "name": "BAG_NULLDAM_BAGEL_140G", "count": 1},
            {
                "productId": 23,
                "name": "BAG_HANMAC_TRIPLE_CHEESE_HAMBURGER_155G",
                "count": 1,
            },
        ]
        assert {
            product.product_id: product.count
            for product in global_session.zone_sessions[1].get_active_products()
        } == {13: 1}
        assert {
            product.product_id: product.count
            for product in global_session.zone_sessions[4].get_active_products()
        } == {23: 1}
        selected_by_zone = {
            item["zone"]: {
                product["productId"]: product["count"]
                for product in item["products"]
            }
            for item in aggregate["selectedProductsByZone"]
        }
        assert selected_by_zone == {1: {13: 1}, 4: {23: 1}}
        assert all(
            product["name"] != "STICK_BINGGRAE_MELONA_75ML"
            for zone in response["zones"]
            for product in zone["products"]
        )
    finally:
        store.clear_all()


def test_freezer_close_aggregate_preserve_rewrites_stale_aggregated_products(
    monkeypatch,
):
    from model_service.core.config import config
    from model_service.session.door_session import (
        AggregatedProduct,
        DoorSession,
        TriggerResult,
    )
    from model_service.session.freezer_close_aggregate import (
        FreezerCloseAggregateResolver,
    )

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.weight, "freezer_weight_tolerance_grams", 15.0)
    weights = {24: 165.0, 44: 79.0, 46: 71.0, 77: 224.0}

    hotdog = _product(
        24,
        "BAG_JACKSONVILLE_BIG_HOT_DOG_115G",
        1,
        price=2200,
        confidence=0.9,
    )
    hotdog.placement_units = [{"channelSide": "left", "channelPosition": "left"}]
    melona = _product(
        44,
        "STICK_BINGGRAE_MELONA_75ML",
        1,
        price=800,
        confidence=0.9,
    )
    bibigo = _product(
        77,
        "BAG_BIBIGO_CHEONGYANG_MEAT_DUMPLINGS_200G",
        1,
        price=3900,
        confidence=0.9,
    )
    lala = _product(
        46,
        "STICK_LALA_SWEET_GRAPE_ZERO_70ML",
        1,
        price=500,
        confidence=0.9,
    )

    zone_4 = DoorSession(
        door_session_id="door-zone-4",
        zone=4,
        status="complete",
        triggers=[
            TriggerResult(
                trigger_id="zone4-hotdog-melona",
                session_id="zone4-hotdog-melona-session",
                timestamp=110.0,
                products=[hotdog, melona],
                delta_weight=-234.9,
                confidence=0.9,
                video_paths={},
            )
        ],
        aggregated_products={
            44: AggregatedProduct(
                product_id=44,
                product_idx="P44",
                name="STICK_BINGGRAE_MELONA_75ML",
                count=3,
                unit_price=800,
                weight=79.0,
                total_confidence=2.7,
                detection_count=3,
            )
        },
        last_trigger_at=110.0,
    )
    zone_2 = DoorSession(
        door_session_id="door-zone-2",
        zone=2,
        status="complete",
        triggers=[
            TriggerResult(
                trigger_id="zone2-bibigo-lala",
                session_id="zone2-bibigo-lala-session",
                timestamp=120.0,
                products=[bibigo, lala],
                delta_weight=-296.9,
                confidence=0.9,
                video_paths={},
            )
        ],
        aggregated_products={},
        last_trigger_at=120.0,
    )

    diagnostics = FreezerCloseAggregateResolver(
        get_product_weight=lambda product_id: weights.get(product_id, 0.0),
    ).apply({2: zone_2, 4: zone_4})

    assert diagnostics is not None
    assert diagnostics["reason"] == "freezer_close_aggregate_trigger_products_preserved"
    assert diagnostics["globalNetDelta"] == -531.8
    assert diagnostics["selectedWeight"] == 539.0
    assert diagnostics["residual"] == 7.2
    assert {
        product.product_id: product.count
        for product in zone_4.get_active_products()
    } == {24: 1, 44: 1}
    assert {
        product.product_id: product.count
        for product in zone_2.get_active_products()
    } == {77: 1, 46: 1}
    assert zone_4.aggregated_products[24].placement_units == [
        {"channelSide": "left", "channelPosition": "left"}
    ]
    assert zone_4.aggregated_products[44].count == 1
    selected_by_zone = {
        item["zone"]: {
            product["productId"]: product["count"]
            for product in item["products"]
        }
        for item in zone_2.final_weight_validation["freezerCloseAggregate"][
            "selectedProductsByZone"
        ]
    }
    assert selected_by_zone == {2: {46: 1, 77: 1}, 4: {24: 1, 44: 1}}


def test_freezer_close_aggregate_suppresses_returned_yomamte_candidate(
    monkeypatch,
    tmp_path,
):
    from model_service.api.routes.multi_zone import _handle_door_close
    from model_service.core.config import config
    from model_service.session import DoorSessionStore
    from model_service.session.door_session import TriggerResult

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.weight, "freezer_weight_tolerance_grams", 15.0)
    weights = {30: 82.0, 77: 224.0}
    store = DoorSessionStore(
        yaml_dir=str(tmp_path),
        weight_tolerance=5.0,
        get_product_weight=lambda product_id: weights.get(product_id, 0.0),
    )
    try:
        store.get_or_start_global_session()
        store.add_trigger_with_global(
            zone=3,
            result=TriggerResult(
                trigger_id="zone3-yomamte-return",
                session_id="zone3-yomamte-return-session",
                timestamp=110.0,
                products=[],
                delta_weight=77.7,
                confidence=0.0,
                video_paths={},
                is_return=True,
            ),
        )
        store.add_trigger_with_global(
            zone=3,
            result=TriggerResult(
                trigger_id="zone3-yomamte-removal",
                session_id="zone3-yomamte-removal-session",
                timestamp=100.0,
                products=[
                    _product(
                        30,
                        "BOX_BINGGRAE_YOMAMTE_150ML",
                        1,
                        price=1500,
                        confidence=0.797,
                    )
                ],
                delta_weight=-86.8,
                confidence=0.797,
                video_paths={},
                vision_candidates=[
                    _candidate_snapshot(
                        30,
                        "BOX_BINGGRAE_YOMAMTE_150ML",
                        rank=1,
                        weight=82.0,
                        price=1500,
                        confidence=0.797,
                        identity_confidence=0.797,
                    )
                ],
            ),
        )
        store.add_trigger_with_global(
            zone=1,
            result=TriggerResult(
                trigger_id="zone1-dumpling-removal",
                session_id="zone1-dumpling-removal-session",
                timestamp=120.0,
                products=[],
                delta_weight=-224.5,
                confidence=0.0,
                video_paths={},
                vision_candidates=[
                    _candidate_snapshot(
                        30,
                        "BOX_BINGGRAE_YOMAMTE_150ML",
                        rank=1,
                        weight=82.0,
                        price=1500,
                        confidence=0.89,
                        identity_confidence=0.89,
                    ),
                    _candidate_snapshot(
                        77,
                        "BAG_BIBIGO_DUMPLING_224G",
                        rank=2,
                        weight=224.0,
                        price=3900,
                        confidence=0.78,
                        identity_confidence=0.78,
                    ),
                ],
            ),
        )

        global_session = store.finalize_global_session()
        response = _handle_door_close(FakeCloseReadyStore(global_session))
        zone_1 = next(zone for zone in response["zones"] if zone["zone"] == 1)
        zone_3 = next(zone for zone in response["zones"] if zone["zone"] == 3)

        assert zone_3["products"] == []
        assert [product["name"] for product in zone_1["products"]] == [
            "BAG_BIBIGO_DUMPLING_224G"
        ]
        assert all(
            product["name"] != "BOX_BINGGRAE_YOMAMTE_150ML"
            for zone in response["zones"]
            for product in zone["products"]
        )
        aggregate = global_session.zone_sessions[1].final_weight_validation[
            "freezerCloseAggregate"
        ]
        assert aggregate["reason"] == "freezer_close_aggregate_applied"
        assert aggregate["selectedProducts"] == [
            {"productId": 77, "name": "BAG_BIBIGO_DUMPLING_224G", "count": 1}
        ]
        suppressed_names = {
            item["name"]
            for item in aggregate["returnedPositionSuppressedCandidates"]
        }
        assert "BOX_BINGGRAE_YOMAMTE_150ML" in suppressed_names
    finally:
        store.clear_all()


def test_freezer_close_aggregate_prefers_worldcon_after_lala_touch_return(
    monkeypatch,
    tmp_path,
):
    from model_service.api.routes.multi_zone import _handle_door_close
    from model_service.core.config import config
    from model_service.session import DoorSessionStore
    from model_service.session.door_session import TriggerResult

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.weight, "freezer_weight_tolerance_grams", 5.0)
    monkeypatch.setattr(config.vision, "top_confidence_threshold", 0.50)
    monkeypatch.setattr(config.vision, "side_confidence_threshold", 0.50)
    weights = {46: 71.0, 47: 70.0}
    store = DoorSessionStore(
        yaml_dir=str(tmp_path),
        get_product_weight=lambda product_id: weights.get(product_id, 0.0),
    )
    try:
        store.get_or_start_global_session()
        store.add_trigger_with_global(
            zone=2,
            result=TriggerResult(
                trigger_id="zone2-lala-touch-video",
                session_id="zone2-lala-touch-video-session",
                timestamp=110.0,
                products=[],
                delta_weight=-5.8,
                confidence=0.0,
                video_paths={},
                loadcell_diagnostics={
                    "channel_movement_targets": [
                        {
                            "channel_side": "left",
                            "channel_index": 0,
                            "channel_position": 0,
                            "delta": 6.0,
                            "weight": 6.0,
                            "direction": "return",
                        }
                    ]
                },
                vision_candidates=[
                    _candidate_snapshot(
                        46,
                        "STICK_LALA_SWEET_GRAPE_ZERO_70ML",
                        rank=1,
                        weight=71.0,
                        price=500,
                        confidence=0.856,
                        identity_confidence=0.856,
                        top_confidence=0.835,
                        side_confidence=0.856,
                    ),
                    _candidate_snapshot(
                        47,
                        "BOX_LOTTE_WORLDCON_160ML",
                        rank=2,
                        weight=70.0,
                        price=1200,
                        confidence=0.816,
                        identity_confidence=0.816,
                        top_confidence=0.654,
                        side_confidence=0.816,
                    ),
                ],
            ),
        )
        store.add_trigger_with_global(
            zone=4,
            result=TriggerResult(
                trigger_id="zone4-worldcon-removal",
                session_id="zone4-worldcon-removal-session",
                timestamp=120.0,
                products=[
                    _product(
                        46,
                        "STICK_LALA_SWEET_GRAPE_ZERO_70ML",
                        1,
                        price=500,
                        confidence=0.856,
                    )
                ],
                delta_weight=-65.9,
                confidence=0.856,
                video_paths={},
                loadcell_diagnostics={
                    "channel_removal_segment_targets": [
                        {
                            "channel_side": "right",
                            "channel_index": 1,
                            "channel_position": 0,
                            "delta": -69.0,
                            "weight": 69.0,
                            "direction": "removal",
                        }
                    ]
                },
                vision_candidates=[
                    _candidate_snapshot(
                        46,
                        "STICK_LALA_SWEET_GRAPE_ZERO_70ML",
                        rank=1,
                        weight=71.0,
                        price=500,
                        confidence=0.856,
                        identity_confidence=0.856,
                        top_confidence=0.835,
                        side_confidence=0.856,
                    ),
                    _candidate_snapshot(
                        47,
                        "BOX_LOTTE_WORLDCON_160ML",
                        rank=2,
                        weight=70.0,
                        price=1200,
                        confidence=0.816,
                        identity_confidence=0.816,
                        top_confidence=0.654,
                        side_confidence=0.816,
                    ),
                ],
            ),
        )

        global_session = store.finalize_global_session()
        response = _handle_door_close(FakeCloseReadyStore(global_session))
        zone_4 = next(zone for zone in response["zones"] if zone["zone"] == 4)

        assert [product["name"] for product in zone_4["products"]] == [
            "BOX_LOTTE_WORLDCON_160ML"
        ]
        aggregate = global_session.zone_sessions[4].final_weight_validation[
            "freezerCloseAggregate"
        ]
        assert aggregate["triggerProductsPreserveBlockedByReturnedPosition"] is True
        assert aggregate["selectedProducts"] == [
            {"productId": 47, "name": "BOX_LOTTE_WORLDCON_160ML", "count": 1}
        ]
        assert {
            item["name"]
            for item in aggregate["returnedPositionSuppressedCandidates"]
        } == {"STICK_LALA_SWEET_GRAPE_ZERO_70ML"}
    finally:
        store.clear_all()


def test_freezer_close_aggregate_allows_returned_product_same_position_again(
    monkeypatch,
    tmp_path,
):
    from model_service.api.routes.multi_zone import _handle_door_close
    from model_service.core.config import config
    from model_service.session import DoorSessionStore
    from model_service.session.door_session import TriggerResult

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.weight, "freezer_weight_tolerance_grams", 5.0)
    monkeypatch.setattr(config.vision, "top_confidence_threshold", 0.50)
    monkeypatch.setattr(config.vision, "side_confidence_threshold", 0.50)
    store = DoorSessionStore(
        yaml_dir=str(tmp_path),
        get_product_weight=lambda product_id: {46: 71.0}.get(product_id, 0.0),
    )
    try:
        store.get_or_start_global_session()
        store.add_trigger_with_global(
            zone=2,
            result=TriggerResult(
                trigger_id="zone2-lala-touch-video",
                session_id="zone2-lala-touch-video-session",
                timestamp=110.0,
                products=[],
                delta_weight=-1.0,
                confidence=0.0,
                video_paths={},
                loadcell_diagnostics={
                    "channel_movement_targets": [
                        {
                            "channel_side": "left",
                            "channel_index": 0,
                            "channel_position": 0,
                            "delta": 6.0,
                            "weight": 6.0,
                            "direction": "return",
                        }
                    ]
                },
                vision_candidates=[
                    _candidate_snapshot(
                        46,
                        "STICK_LALA_SWEET_GRAPE_ZERO_70ML",
                        rank=1,
                        weight=71.0,
                        price=500,
                        confidence=0.856,
                        identity_confidence=0.856,
                        top_confidence=0.835,
                        side_confidence=0.856,
                    )
                ],
            ),
        )
        store.add_trigger_with_global(
            zone=2,
            result=TriggerResult(
                trigger_id="zone2-lala-same-position-removal",
                session_id="zone2-lala-same-position-removal-session",
                timestamp=120.0,
                products=[
                    _product(
                        46,
                        "STICK_LALA_SWEET_GRAPE_ZERO_70ML",
                        1,
                        price=500,
                        confidence=0.856,
                    )
                ],
                delta_weight=-71.0,
                confidence=0.856,
                video_paths={},
                loadcell_diagnostics={
                    "mixed_sign_internal_segments": True,
                    "channel_removal_segment_targets": [
                        {
                            "channel_side": "left",
                            "channel_index": 0,
                            "channel_position": 0,
                            "delta": -71.0,
                            "weight": 71.0,
                            "direction": "removal",
                        }
                    ]
                },
                vision_candidates=[
                    _candidate_snapshot(
                        46,
                        "STICK_LALA_SWEET_GRAPE_ZERO_70ML",
                        rank=1,
                        weight=71.0,
                        price=500,
                        confidence=0.856,
                        identity_confidence=0.856,
                        top_confidence=0.835,
                        side_confidence=0.856,
                    )
                ],
            ),
        )

        global_session = store.finalize_global_session()
        response = _handle_door_close(FakeCloseReadyStore(global_session))
        zone_2 = next(zone for zone in response["zones"] if zone["zone"] == 2)

        assert [product["name"] for product in zone_2["products"]] == [
            "STICK_LALA_SWEET_GRAPE_ZERO_70ML"
        ]
        aggregate = global_session.zone_sessions[2].final_weight_validation[
            "freezerCloseAggregate"
        ]
        assert aggregate["reason"] == (
            "freezer_close_aggregate_trigger_products_preserved"
        )
        assert aggregate["samePositionReturnedProductAllowed"][0]["productId"] == 46
        assert "returnedPositionSuppressedCandidates" not in aggregate
    finally:
        store.clear_all()


def test_freezer_touch_return_hint_requires_channel_evidence(
    monkeypatch,
    tmp_path,
):
    from model_service.core.config import config
    from model_service.session import DoorSessionStore
    from model_service.session.door_session import TriggerResult

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.weight, "freezer_weight_tolerance_grams", 5.0)
    monkeypatch.setattr(config.vision, "top_confidence_threshold", 0.50)
    store = DoorSessionStore(
        yaml_dir=str(tmp_path),
        get_product_weight=lambda product_id: {46: 71.0}.get(product_id, 0.0),
    )
    try:
        store.get_or_start_global_session()
        session = store.add_trigger_with_global(
            zone=2,
            result=TriggerResult(
                trigger_id="zone2-lala-no-channel-touch",
                session_id="zone2-lala-no-channel-touch-session",
                timestamp=100.0,
                products=[],
                delta_weight=-1.0,
                confidence=0.0,
                video_paths={},
                vision_candidates=[
                    _candidate_snapshot(
                        46,
                        "STICK_LALA_SWEET_GRAPE_ZERO_70ML",
                        rank=1,
                        weight=71.0,
                        confidence=0.856,
                        identity_confidence=0.856,
                        top_confidence=0.835,
                    )
                ],
            ),
        )

        assert session.returned_position_hints == []
    finally:
        store.clear_all()


def test_freezer_close_aggregate_rejects_low_raw_confidence_candidate(
    monkeypatch,
    tmp_path,
):
    from model_service.core.config import config
    from model_service.session import DoorSessionStore
    from model_service.session.door_session import TriggerResult

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    monkeypatch.setattr(config.vision, "top_confidence_threshold", 0.70)
    monkeypatch.setattr(config.vision, "side_confidence_threshold", 0.70)
    store = DoorSessionStore(
        yaml_dir=str(tmp_path),
        get_product_weight=lambda product_id: {44: 79.0}.get(product_id, 0.0),
    )
    try:
        store.get_or_start_global_session()
        store.add_trigger_with_global(
            zone=4,
            result=TriggerResult(
                trigger_id="zone4-low-raw-melona",
                session_id="zone4-low-raw-melona-session",
                timestamp=120.0,
                products=[],
                delta_weight=-316.0,
                confidence=0.0,
                video_paths={},
                loadcell_diagnostics={
                    "mixed_sign_internal_segments": True,
                    "compound_positive_segment_count": 1,
                    "compound_negative_segment_count": 1,
                    "net_delta_weight": -316.0,
                    "decision_delta_weight": -316.0,
                },
                vision_candidates=[
                    _candidate_snapshot(
                        44,
                        "STICK_BINGGRAE_MELONA_75ML",
                        rank=1,
                        weight=79.0,
                        price=800,
                        confidence=0.58,
                        identity_confidence=0.58,
                    )
                ],
            ),
        )

        global_session = store.finalize_global_session()

        zone_4 = global_session.zone_sessions[4]
        assert zone_4.get_active_products() == []
        aggregate = zone_4.final_weight_validation["freezerCloseAggregate"]
        assert aggregate["accepted"] is False
        assert aggregate["candidateCount"] == 0
        assert aggregate["selectedProducts"] == []
        assert aggregate["noChargeReason"] == (
            "no_candidate_combination_for_signed_net_delta"
        )
    finally:
        store.clear_all()


def test_freezer_close_aggregate_net_zero_clears_participating_products(
    monkeypatch,
    tmp_path,
):
    from model_service.api.routes.multi_zone import _handle_door_close
    from model_service.core.config import config
    from model_service.session import DoorSessionStore
    from model_service.session.door_session import TriggerResult

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    store = DoorSessionStore(
        yaml_dir=str(tmp_path),
        get_product_weight=lambda product_id: {100: 100.0}.get(product_id, 0.0),
    )
    try:
        store.get_or_start_global_session()
        store.add_trigger_with_global(
            zone=1,
            result=_trigger_result(
                product_id=100,
                name="FREEZER_100G_ITEM",
                delta=-100.0,
                weight=100.0,
                timestamp=100.0,
            ),
        )
        store.add_trigger_with_global(
            zone=2,
            result=TriggerResult(
                trigger_id="return-trigger",
                session_id="return-session",
                timestamp=120.0,
                products=[],
                delta_weight=100.0,
                confidence=0.0,
                video_paths={},
                is_return=True,
            ),
        )

        global_session = store.finalize_global_session()
        response = _handle_door_close(FakeCloseReadyStore(global_session))
        zone_1 = next(zone for zone in response["zones"] if zone["zone"] == 1)
        zone_2 = next(zone for zone in response["zones"] if zone["zone"] == 2)
        assert zone_1["products"] == []
        assert zone_1["weightDelta"] == 0.0
        assert zone_2["products"] == []
        assert zone_2["weightDelta"] == 0.0
        assert response["decisionSummary"]["totalWeightDelta"] == 0.0
        assert response["decisionSummary"]["totalPrice"] == 0
        aggregate = global_session.zone_sessions[2].final_weight_validation[
            "freezerCloseAggregate"
        ]
        assert aggregate["accepted"] is True
        assert aggregate["reason"] == "freezer_close_aggregate_net_zero"
        assert aggregate["globalNetDelta"] == 0.0
    finally:
        store.clear_all()


def test_freezer_close_aggregate_leaves_single_simple_trigger_per_zone(
    monkeypatch,
    tmp_path,
):
    from model_service.core.config import config
    from model_service.session import DoorSessionStore

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    store = DoorSessionStore(
        yaml_dir=str(tmp_path),
        get_product_weight=lambda product_id: {150: 150.0}.get(product_id, 0.0),
    )
    try:
        store.get_or_start_global_session()
        store.add_trigger_with_global(
            zone=2,
            result=_trigger_result(
                product_id=150,
                name="FREEZER_150G_ITEM",
                delta=-150.0,
                weight=150.0,
                timestamp=110.0,
            ),
        )

        global_session = store.finalize_global_session()
        zone_2 = global_session.zone_sessions[2]
        assert [(p.product_id, p.count) for p in zone_2.get_active_products()] == [
            (150, 1)
        ]
        assert "freezerCloseAggregate" not in zone_2.final_weight_validation
    finally:
        store.clear_all()


def test_refrigerated_close_does_not_use_freezer_aggregate(
    monkeypatch,
    tmp_path,
):
    from model_service.core.config import config
    from model_service.session import DoorSessionStore

    monkeypatch.setattr(config.machine, "cabinet_type", "refrigerated")
    store = DoorSessionStore(
        yaml_dir=str(tmp_path),
        get_product_weight=lambda product_id: {100: 100.0, 200: 200.0}.get(
            product_id,
            0.0,
        ),
    )
    try:
        store.get_or_start_global_session()
        store.add_trigger_with_global(
            zone=1,
            result=_trigger_result(
                product_id=100,
                name="FRIDGE_100G_ITEM",
                delta=-100.0,
                weight=100.0,
                timestamp=100.0,
            ),
        )
        store.add_trigger_with_global(
            zone=4,
            result=_trigger_result(
                product_id=200,
                name="FRIDGE_200G_ITEM",
                delta=-200.0,
                weight=200.0,
                timestamp=120.0,
            ),
        )

        global_session = store.finalize_global_session()
        assert [(p.product_id, p.count) for p in global_session.zone_sessions[1].get_active_products()] == [
            (100, 1)
        ]
        assert [(p.product_id, p.count) for p in global_session.zone_sessions[4].get_active_products()] == [
            (200, 1)
        ]
        assert "freezerCloseAggregate" not in global_session.zone_sessions[1].final_weight_validation
        assert "freezerCloseAggregate" not in global_session.zone_sessions[4].final_weight_validation
    finally:
        store.clear_all()


def test_close_final_weight_validation_prefers_repeated_candidate_over_mix(tmp_path):
    from model_service.session.door_session import TriggerResult
    from model_service.session.door_session_store import DoorSessionStore

    weights = {8: 367.0, 31: 523.0, 35: 149.0, 54: 530.0, 64: 64.0}
    store = DoorSessionStore(
        yaml_dir=str(tmp_path),
        weight_tolerance=5.0,
        get_product_weight=lambda product_id: weights.get(product_id, 0.0),
    )
    store.get_or_start_global_session()
    store.add_trigger_with_global(
        4,
        TriggerResult(
            trigger_id="",
            session_id="zone-4-first",
            timestamp=10.0,
            products=[
                _product(8, "CAN_LOTTE_HOT6_THE_KING_RUSH_355ML", 2, 1300),
                _product(35, "BAG_NONGSHIM_CHAPAGETTI_140G", 2, 4000),
            ],
            delta_weight=-1037.4,
            confidence=0.5,
            video_paths={},
            vision_candidates=[
                _candidate_snapshot(
                    8,
                    "CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
                    rank=1,
                    weight=367.0,
                    price=1300,
                    confidence=0.506,
                ),
                _candidate_snapshot(
                    31,
                    "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML",
                    rank=2,
                    weight=523.0,
                    confidence=0.412,
                ),
                _candidate_snapshot(
                    54,
                    "BOTTLE_LOTTE_TREVI_LEMON_500ML",
                    rank=3,
                    weight=530.0,
                    price=1600,
                    confidence=0.322,
                ),
                _candidate_snapshot(
                    64,
                    "BAG_HAITAI_HOME_RUN_BALL_41G",
                    rank=4,
                    weight=64.0,
                    price=1500,
                    confidence=0.181,
                ),
            ],
        ),
    )
    store.add_trigger_with_global(
        4,
        TriggerResult(
            trigger_id="",
            session_id="zone-4-second",
            timestamp=20.0,
            products=[_product(31, "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML", 1)],
            delta_weight=-523.8,
            confidence=0.18,
            video_paths={},
            vision_candidates=[
                _candidate_snapshot(
                    8,
                    "CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
                    rank=1,
                    weight=367.0,
                    price=1300,
                    confidence=0.506,
                ),
                _candidate_snapshot(
                    31,
                    "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML",
                    rank=2,
                    weight=523.0,
                    confidence=0.180,
                    source="threshold_rescue",
                ),
            ],
        ),
    )

    global_session = store.finalize_global_session()

    zone_session = global_session.zone_sessions[4]
    active = zone_session.get_active_products()
    assert len(active) == 1
    assert active[0].product_id == 31
    assert active[0].count == 3
    assert zone_session.total_price == 3600
    diagnostics = zone_session.final_weight_validation
    assert diagnostics["accepted"] is True
    assert diagnostics["selectedProduct"] == "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML"
    assert diagnostics["selectedCount"] == 3
    assert diagnostics["currentResidual"] == 6.2
    assert diagnostics["replacementResidual"] == 7.8


def test_close_final_weight_validation_allows_strong_sky_repeat_residual_gap(tmp_path):
    from model_service.session.door_session import TriggerResult
    from model_service.session.door_session_store import DoorSessionStore

    weights = {8: 367.0, 31: 523.0, 35: 149.0, 40: 130.0, 54: 530.0, 113: 19.0}
    store = DoorSessionStore(
        yaml_dir=str(tmp_path),
        weight_tolerance=5.0,
        get_product_weight=lambda product_id: weights.get(product_id, 0.0),
    )
    store.get_or_start_global_session()
    store.add_trigger_with_global(
        4,
        TriggerResult(
            trigger_id="",
            session_id="zone-4-first",
            timestamp=10.0,
            products=[
                _product(113, "STICK_INNON_CONDITION_STICK_18G", 2, 3000),
                _product(8, "CAN_LOTTE_HOT6_THE_KING_RUSH_355ML", 2, 1300),
                _product(40, "BOX_LOTTE_BINCH_102G", 2, 1500),
            ],
            delta_weight=-1035.2,
            confidence=0.5,
            video_paths={},
            vision_candidates=[
                _candidate_snapshot(
                    31,
                    "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML",
                    rank=1,
                    weight=523.0,
                    confidence=1.0,
                ),
                _candidate_snapshot(
                    35,
                    "BAG_NONGSHIM_CHAPAGETTI_140G",
                    rank=2,
                    weight=149.0,
                    price=4000,
                    confidence=0.447,
                ),
                _candidate_snapshot(
                    8,
                    "CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
                    rank=3,
                    weight=367.0,
                    price=1300,
                    confidence=0.433,
                ),
                _candidate_snapshot(
                    54,
                    "BOTTLE_LOTTE_TREVI_LEMON_500ML",
                    rank=4,
                    weight=530.0,
                    price=1600,
                    confidence=0.417,
                ),
            ],
        ),
    )
    store.add_trigger_with_global(
        4,
        TriggerResult(
            trigger_id="",
            session_id="zone-4-second",
            timestamp=20.0,
            products=[_product(31, "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML", 1)],
            delta_weight=-523.8,
            confidence=0.36,
            video_paths={},
            vision_candidates=[
                _candidate_snapshot(
                    35,
                    "BAG_NONGSHIM_CHAPAGETTI_140G",
                    rank=1,
                    weight=149.0,
                    price=4000,
                    confidence=0.447,
                ),
                _candidate_snapshot(
                    54,
                    "BOTTLE_LOTTE_TREVI_LEMON_500ML",
                    rank=2,
                    weight=530.0,
                    price=1600,
                    confidence=0.369,
                ),
                _candidate_snapshot(
                    31,
                    "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML",
                    rank=3,
                    weight=523.0,
                    confidence=0.365,
                ),
            ],
        ),
    )

    global_session = store.finalize_global_session()

    zone_session = global_session.zone_sessions[4]
    active = zone_session.get_active_products()
    assert len(active) == 1
    assert active[0].product_id == 31
    assert active[0].count == 3
    assert zone_session.total_price == 3600
    diagnostics = zone_session.final_weight_validation
    assert diagnostics["accepted"] is True
    assert diagnostics["currentResidual"] == 4.0
    assert diagnostics["replacementResidual"] == 10.0
    assert diagnostics["residualGap"] == 6.0
    assert diagnostics["residualGapAllowed"] == 10.0
    assert diagnostics["currentHasUnsupportedFragments"] is True


def test_close_final_weight_validation_rejects_weak_threshold_only_candidate(tmp_path):
    from model_service.session.door_session import TriggerResult
    from model_service.session.door_session_store import DoorSessionStore

    weights = {8: 367.0, 31: 523.0, 35: 149.0}
    store = DoorSessionStore(
        yaml_dir=str(tmp_path),
        weight_tolerance=5.0,
        get_product_weight=lambda product_id: weights.get(product_id, 0.0),
    )
    store.get_or_start_global_session()
    store.add_trigger_with_global(
        4,
        TriggerResult(
            trigger_id="",
            session_id="weak-threshold-only",
            timestamp=10.0,
            products=[
                _product(8, "CAN_LOTTE_HOT6_THE_KING_RUSH_355ML", 2, 1300),
                _product(35, "BAG_NONGSHIM_CHAPAGETTI_140G", 2, 4000),
            ],
            delta_weight=-1037.4,
            confidence=0.5,
            video_paths={},
            vision_candidates=[
                _candidate_snapshot(
                    31,
                    "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML",
                    rank=2,
                    weight=523.0,
                    confidence=0.18,
                    source="threshold_rescue",
                )
            ],
        ),
    )

    global_session = store.finalize_global_session()

    zone_session = global_session.zone_sessions[4]
    active_ids = {product.product_id for product in zone_session.get_active_products()}
    assert active_ids == {8, 35}
    assert zone_session.final_weight_validation["accepted"] is False
    assert zone_session.final_weight_validation["reason"] == "no_viable_repeat_candidate"


def test_close_final_weight_validation_rejects_repeat_count_over_segment_cap(tmp_path):
    from model_service.session.door_session import TriggerResult
    from model_service.session.door_session_store import DoorSessionStore

    weights = {8: 367.0, 35: 149.0, 64: 64.0}
    store = DoorSessionStore(
        yaml_dir=str(tmp_path),
        weight_tolerance=5.0,
        get_product_weight=lambda product_id: weights.get(product_id, 0.0),
    )
    store.get_or_start_global_session()
    store.add_trigger_with_global(
        5,
        TriggerResult(
            trigger_id="",
            session_id="home-run-ball-cap",
            timestamp=10.0,
            products=[
                _product(8, "CAN_LOTTE_HOT6_THE_KING_RUSH_355ML", 5, 1300),
                _product(35, "BAG_NONGSHIM_CHAPAGETTI_140G", 1, 4000),
            ],
            delta_weight=-2096.0,
            confidence=0.6,
            video_paths={},
            vision_candidates=[
                _candidate_snapshot(
                    64,
                    "BAG_HAITAI_HOME_RUN_BALL_41G",
                    rank=1,
                    weight=64.0,
                    price=1200,
                    confidence=0.9,
                    stock=100,
                )
            ],
        ),
    )

    global_session = store.finalize_global_session()

    zone_session = global_session.zone_sessions[5]
    active = zone_session.get_active_products()
    assert active == []
    assert not (
        len(active) == 1
        and active[0].product_id == 64
        and active[0].count == 33
    )
    diagnostics = zone_session.final_weight_validation
    assert diagnostics["accepted"] is False
    assert diagnostics["reason"] == "unresolved_final_weight_mismatch"
    assert diagnostics["previousReason"] == "no_viable_repeat_candidate"
    assert diagnostics["rejectedCandidates"][0]["reason"] == (
        "count_exceeds_close_repeat_cap"
    )
    assert diagnostics["rejectedCandidates"][0]["estimatedCount"] == 33
    assert diagnostics["rejectedCandidates"][0]["closeRepeatCountCap"] == 3


def test_freezer_close_mismatch_preserves_detected_products_for_edge(
    monkeypatch,
    tmp_path,
    caplog,
):
    from model_service.api.routes.multi_zone import _handle_door_close
    from model_service.core.config import config
    from model_service.session.door_session import TriggerResult
    from model_service.session.door_session_store import DoorSessionStore

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    caplog.set_level(logging.INFO)
    store = DoorSessionStore(
        yaml_dir=str(tmp_path),
        weight_tolerance=5.0,
        get_product_weight=lambda product_id: {44: 93.0}.get(product_id, 0.0),
    )
    store.get_or_start_global_session()
    store.add_trigger_with_global(
        2,
        TriggerResult(
            trigger_id="",
            session_id="melona-mismatch",
            timestamp=10.0,
            products=[
                _product(
                    44,
                    "STICK_BINGGRAE_MELONA_75ML",
                    1,
                    price=1000,
                    confidence=1.0,
                ),
            ],
            delta_weight=-81.0,
            confidence=1.0,
            video_paths={},
        ),
    )

    global_session = store.finalize_global_session()
    zone_session = global_session.zone_sessions[2]
    active = zone_session.get_active_products()
    assert [(product.product_id, product.count) for product in active] == [(44, 1)]
    validation = zone_session.final_weight_validation
    assert validation["accepted"] is False
    assert validation["reason"] == "unresolved_final_weight_mismatch"
    assert validation["outputPolicy"] == "products_as_detected"
    assert validation["unresolvedProducts"] == [
        {
            "productId": 44,
            "name": "STICK_BINGGRAE_MELONA_75ML",
            "count": 1,
            "unitWeight": 93.0,
            "unitPrice": 1000,
            "totalPrice": 1000,
        }
    ]

    response = _handle_door_close(FakeCloseReadyStore(global_session))
    zone_2 = next(zone for zone in response["zones"] if zone["zone"] == 2)
    assert zone_2["products"] == [
        {
            "productIdx": "P44",
            "productId": 44,
            "name": "STICK_BINGGRAE_MELONA_75ML",
            "count": 1,
            "price": 1000,
            "confidence": 1.0,
        }
    ]
    assert zone_2["totalPrice"] == 1000
    assert "STICK_BINGGRAE_MELONA_75MLx1" in response["decisionSummaryText"]
    assert "[OPS][CLOSE]" in caplog.text
    assert "products=STICK_BINGGRAE_MELONA_75MLx1" in caplog.text


def test_freezer_close_aggregate_supersedes_deferred_candidate_repair(
    monkeypatch,
    tmp_path,
    caplog,
):
    from model_service.core.config import config
    from model_service.session.door_session import TriggerResult
    from model_service.session.door_session_store import DoorSessionStore

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    caplog.set_level(logging.INFO)
    weights = {30: 224.0, 44: 79.0}
    store = DoorSessionStore(
        yaml_dir=str(tmp_path),
        weight_tolerance=5.0,
        get_product_weight=lambda product_id: weights.get(product_id, 0.0),
    )
    store.get_or_start_global_session()
    store.add_trigger_with_global(
        3,
        TriggerResult(
            trigger_id="zone3-trigger",
            session_id="zone3-session",
            timestamp=10.0,
            products=[
                _product(
                    30,
                    "BAG_BIBIGO_CHEONGYANG_MEAT_DUMPLINGS_200G",
                    1,
                    price=3700,
                    confidence=1.0,
                ),
            ],
            delta_weight=-75.9,
            confidence=1.0,
            video_paths={},
            vision_candidates=[
                _candidate_snapshot(
                    30,
                    "BAG_BIBIGO_CHEONGYANG_MEAT_DUMPLINGS_200G",
                    rank=1,
                    weight=224.0,
                    price=3700,
                    confidence=1.0,
                ),
            ],
        ),
    )
    store.add_trigger_with_global(
        4,
        TriggerResult(
            trigger_id="zone4-trigger",
            session_id="zone4-session",
            timestamp=20.0,
            products=[
                _product(
                    30,
                    "BAG_BIBIGO_CHEONGYANG_MEAT_DUMPLINGS_200G",
                    1,
                    price=3700,
                    confidence=1.0,
                ),
            ],
            delta_weight=-224.1,
            confidence=1.0,
            video_paths={},
            vision_candidates=[
                _candidate_snapshot(
                    30,
                    "BAG_BIBIGO_CHEONGYANG_MEAT_DUMPLINGS_200G",
                    rank=2,
                    weight=224.0,
                    price=3700,
                    confidence=1.0,
                ),
                _candidate_snapshot(
                    44,
                    "STICK_BINGGRAE_MELONA_75ML",
                    rank=4,
                    weight=79.0,
                    price=1000,
                    confidence=0.462,
                    identity_confidence=0.9,
                ),
            ],
        ),
    )

    global_session = store.finalize_global_session()

    zone3_session = global_session.zone_sessions[3]
    zone4_session = global_session.zone_sessions[4]
    assert zone3_session.get_active_products() == []
    zone4_products = [
        (product.product_id, product.count)
        for product in zone4_session.get_active_products()
    ]
    assert zone4_products == [(30, 1), (44, 1)]
    zone3_aggregate = zone3_session.final_weight_validation["freezerCloseAggregate"]
    zone4_aggregate = zone4_session.final_weight_validation["freezerCloseAggregate"]
    assert zone3_aggregate["role"] == "rerouted"
    assert zone3_aggregate["weightDeltaOverride"] == 0.0
    assert zone4_aggregate["accepted"] is True
    assert zone4_aggregate["role"] == "output"
    assert zone4_aggregate["outputZone"] == 4
    assert zone4_aggregate["policy"] == "signed_net_delta"
    assert zone4_aggregate["globalNetDelta"] == -300.0
    assert zone4_aggregate["finalTargetWeight"] == 300.0
    assert zone4_aggregate["selectedWeight"] == 303.0
    assert zone4_aggregate["residual"] == 3.0
    assert zone4_aggregate["selectedProducts"] == [
        {
            "productId": 30,
            "name": "BAG_BIBIGO_CHEONGYANG_MEAT_DUMPLINGS_200G",
            "count": 1,
        },
        {"productId": 44, "name": "STICK_BINGGRAE_MELONA_75ML", "count": 1},
    ]
    assert "[CLOSE][CANDIDATE_REPAIR] corrected zone=3" in caplog.text


def test_freezer_close_aggregate_solves_full_target_from_later_candidates(
    monkeypatch,
    tmp_path,
):
    from model_service.core.config import config
    from model_service.session.door_session import TriggerResult
    from model_service.session.door_session_store import DoorSessionStore

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    weights = {30: 224.0, 44: 79.0}
    store = DoorSessionStore(
        yaml_dir=str(tmp_path),
        weight_tolerance=5.0,
        get_product_weight=lambda product_id: weights.get(product_id, 0.0),
    )
    store.get_or_start_global_session()
    store.add_trigger_with_global(
        3,
        TriggerResult(
            trigger_id="zone3-trigger",
            session_id="zone3-session",
            timestamp=10.0,
            products=[
                _product(44, "STICK_BINGGRAE_MELONA_75ML", 1, price=1000),
            ],
            delta_weight=-224.1,
            confidence=0.8,
            video_paths={},
        ),
    )
    store.add_trigger_with_global(
        4,
        TriggerResult(
            trigger_id="zone4-trigger",
            session_id="zone4-session",
            timestamp=20.0,
            products=[
                _product(
                    30,
                    "BAG_BIBIGO_CHEONGYANG_MEAT_DUMPLINGS_200G",
                    1,
                    price=3700,
                ),
            ],
            delta_weight=-224.1,
            confidence=1.0,
            video_paths={},
            vision_candidates=[
                _candidate_snapshot(
                    30,
                    "BAG_BIBIGO_CHEONGYANG_MEAT_DUMPLINGS_200G",
                    rank=1,
                    weight=224.0,
                    price=3700,
                    confidence=1.0,
                ),
            ],
        ),
    )

    global_session = store.finalize_global_session()

    zone3_session = global_session.zone_sessions[3]
    zone4_session = global_session.zone_sessions[4]
    assert zone3_session.get_active_products() == []
    zone4_products = [
        (product.product_id, product.count)
        for product in zone4_session.get_active_products()
    ]
    assert zone4_products == [(30, 2)]
    zone3_aggregate = zone3_session.final_weight_validation["freezerCloseAggregate"]
    zone4_aggregate = zone4_session.final_weight_validation["freezerCloseAggregate"]
    assert zone3_aggregate["role"] == "rerouted"
    assert zone3_aggregate["weightDeltaOverride"] == 0.0
    assert zone4_aggregate["accepted"] is True
    assert zone4_aggregate["role"] == "output"
    assert zone4_aggregate["outputZone"] == 4
    assert zone4_aggregate["policy"] == "signed_net_delta"
    assert zone4_aggregate["globalNetDelta"] == -448.2
    assert zone4_aggregate["finalTargetWeight"] == 448.2
    assert zone4_aggregate["selectedWeight"] == 448.0
    assert zone4_aggregate["residual"] == 0.2
    assert zone4_aggregate["selectedProducts"] == [
        {
            "productId": 30,
            "name": "BAG_BIBIGO_CHEONGYANG_MEAT_DUMPLINGS_200G",
            "count": 2,
        }
    ]


def test_freezer_close_deferred_candidate_rejects_residual_outside_tolerance(
    monkeypatch,
    tmp_path,
):
    from model_service.core.config import config
    from model_service.session.door_session import TriggerResult
    from model_service.session.door_session_store import DoorSessionStore

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    weights = {30: 224.0, 55: 100.0}
    store = DoorSessionStore(
        yaml_dir=str(tmp_path),
        weight_tolerance=5.0,
        get_product_weight=lambda product_id: weights.get(product_id, 0.0),
    )
    store.get_or_start_global_session()
    store.add_trigger_with_global(
        3,
        TriggerResult(
            trigger_id="zone3-trigger",
            session_id="zone3-session",
            timestamp=10.0,
            products=[
                _product(
                    30,
                    "BAG_BIBIGO_CHEONGYANG_MEAT_DUMPLINGS_200G",
                    1,
                    price=3700,
                ),
            ],
            delta_weight=-75.9,
            confidence=0.8,
            video_paths={},
        ),
    )
    store.add_trigger_with_global(
        4,
        TriggerResult(
            trigger_id="zone4-trigger",
            session_id="zone4-session",
            timestamp=20.0,
            products=[],
            delta_weight=-10.0,
            confidence=0.0,
            video_paths={},
            vision_candidates=[
                _candidate_snapshot(
                    55,
                    "UNUSED_100G_PRODUCT",
                    rank=1,
                    weight=100.0,
                    price=1000,
                    confidence=0.9,
                ),
            ],
        ),
    )

    global_session = store.finalize_global_session()

    zone3_session = global_session.zone_sessions[3]
    assert [(product.product_id, product.count) for product in zone3_session.get_active_products()] == [
        (30, 1)
    ]
    repair = zone3_session.final_weight_validation["deferredCandidateRepair"]
    assert repair["applied"] is False
    assert repair["reason"] == "no_later_unused_weight_match"
    assert repair["rejectedCandidates"][0]["reason"] == (
        "weight_residual_exceeds_tolerance"
    )


def test_freezer_close_deferred_candidate_keeps_complete_weight_match(
    monkeypatch,
    tmp_path,
):
    from model_service.core.config import config
    from model_service.session.door_session import TriggerResult
    from model_service.session.door_session_store import DoorSessionStore

    monkeypatch.setattr(config.machine, "cabinet_type", "freezer")
    weights = {44: 79.0, 55: 76.0}
    store = DoorSessionStore(
        yaml_dir=str(tmp_path),
        weight_tolerance=5.0,
        get_product_weight=lambda product_id: weights.get(product_id, 0.0),
    )
    store.get_or_start_global_session()
    store.add_trigger_with_global(
        3,
        TriggerResult(
            trigger_id="zone3-trigger",
            session_id="zone3-session",
            timestamp=10.0,
            products=[
                _product(44, "STICK_BINGGRAE_MELONA_75ML", 1, price=1000),
            ],
            delta_weight=-75.9,
            confidence=1.0,
            video_paths={},
        ),
    )
    store.add_trigger_with_global(
        4,
        TriggerResult(
            trigger_id="zone4-trigger",
            session_id="zone4-session",
            timestamp=20.0,
            products=[],
            delta_weight=-10.0,
            confidence=0.0,
            video_paths={},
            vision_candidates=[
                _candidate_snapshot(
                    55,
                    "UNUSED_76G_PRODUCT",
                    rank=1,
                    weight=76.0,
                    price=1000,
                    confidence=0.9,
                ),
            ],
        ),
    )

    global_session = store.finalize_global_session()

    zone3_session = global_session.zone_sessions[3]
    assert [(product.product_id, product.count) for product in zone3_session.get_active_products()] == [
        (44, 1)
    ]
    assert "deferredCandidateRepair" not in zone3_session.final_weight_validation


def test_close_final_weight_validation_keeps_clean_supported_mix(tmp_path):
    from model_service.session.door_session import TriggerResult
    from model_service.session.door_session_store import DoorSessionStore

    weights = {8: 367.0, 31: 523.0, 35: 149.0}
    store = DoorSessionStore(
        yaml_dir=str(tmp_path),
        weight_tolerance=5.0,
        get_product_weight=lambda product_id: weights.get(product_id, 0.0),
    )
    store.get_or_start_global_session()
    store.add_trigger_with_global(
        4,
        TriggerResult(
            trigger_id="",
            session_id="clean-supported-mix",
            timestamp=10.0,
            products=[
                _product(8, "CAN_LOTTE_HOT6_THE_KING_RUSH_355ML", 2, 1300),
                _product(35, "BAG_NONGSHIM_CHAPAGETTI_140G", 2, 4000),
            ],
            delta_weight=-1038.0,
            confidence=0.6,
            video_paths={},
            vision_candidates=[
                _candidate_snapshot(
                    8,
                    "CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
                    rank=1,
                    weight=367.0,
                    price=1300,
                    confidence=0.50,
                ),
                _candidate_snapshot(
                    31,
                    "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML",
                    rank=2,
                    weight=523.0,
                    confidence=0.40,
                ),
                _candidate_snapshot(
                    35,
                    "BAG_NONGSHIM_CHAPAGETTI_140G",
                    rank=3,
                    weight=149.0,
                    price=4000,
                    confidence=0.35,
                ),
            ],
        ),
    )

    global_session = store.finalize_global_session()

    zone_session = global_session.zone_sessions[4]
    active_ids = {product.product_id for product in zone_session.get_active_products()}
    assert active_ids == {8, 35}
    assert zone_session.final_weight_validation["accepted"] is False
    assert (
        zone_session.final_weight_validation["reason"]
        == "clean_supported_basket_preferred"
    )


def test_close_final_weight_validation_keeps_all_vision_supported_current_identity(
    tmp_path,
):
    from model_service.session.door_session import TriggerResult
    from model_service.session.door_session_store import DoorSessionStore

    weights = {101: 100.0, 102: 200.0, 103: 151.0}
    store = DoorSessionStore(
        yaml_dir=str(tmp_path),
        weight_tolerance=5.0,
        get_product_weight=lambda product_id: weights.get(product_id, 0.0),
    )
    store.get_or_start_global_session()
    store.add_trigger_with_global(
        3,
        TriggerResult(
            trigger_id="",
            session_id="vision-supported-current",
            timestamp=10.0,
            products=[
                _product(101, "VISION_PRODUCT_A_100G", 1, 1000),
                _product(102, "VISION_PRODUCT_B_200G", 1, 2000),
            ],
            delta_weight=-302.0,
            confidence=0.8,
            video_paths={},
            vision_candidates=[
                _candidate_snapshot(
                    103,
                    "LOADCELL_BETTER_BUT_DIFFERENT_151G",
                    rank=1,
                    weight=151.0,
                    price=1500,
                    confidence=0.90,
                ),
                _candidate_snapshot(
                    101,
                    "VISION_PRODUCT_A_100G",
                    rank=2,
                    weight=100.0,
                    price=1000,
                    confidence=0.80,
                ),
                _candidate_snapshot(
                    102,
                    "VISION_PRODUCT_B_200G",
                    rank=3,
                    weight=200.0,
                    price=2000,
                    confidence=0.70,
                ),
            ],
        ),
    )

    global_session = store.finalize_global_session()

    zone_session = global_session.zone_sessions[3]
    active = zone_session.get_active_products()
    assert {(product.product_id, product.count) for product in active} == {
        (101, 1),
        (102, 1),
    }
    assert zone_session.final_weight_validation["accepted"] is False
    assert (
        zone_session.final_weight_validation["reason"]
        == "clean_supported_basket_preferred"
    )
    assert zone_session.final_weight_validation["replacementResidual"] == 0.0


def test_close_final_weight_validation_allows_same_vision_product_count_adjustment(
    tmp_path,
):
    from model_service.session.door_session import TriggerResult
    from model_service.session.door_session_store import DoorSessionStore

    weights = {101: 100.0}
    store = DoorSessionStore(
        yaml_dir=str(tmp_path),
        weight_tolerance=5.0,
        get_product_weight=lambda product_id: weights.get(product_id, 0.0),
    )
    store.get_or_start_global_session()
    store.add_trigger_with_global(
        3,
        TriggerResult(
            trigger_id="",
            session_id="vision-supported-repeat",
            timestamp=10.0,
            products=[
                _product(101, "VISION_PRODUCT_A_100G", 4, 1000),
            ],
            delta_weight=-305.0,
            confidence=0.8,
            video_paths={},
            vision_candidates=[
                _candidate_snapshot(
                    101,
                    "VISION_PRODUCT_A_100G",
                    rank=1,
                    weight=100.0,
                    price=1000,
                    confidence=0.80,
                    stock=10,
                ),
            ],
        ),
    )

    global_session = store.finalize_global_session()

    zone_session = global_session.zone_sessions[3]
    active = zone_session.get_active_products()
    assert [(product.product_id, product.count) for product in active] == [(101, 3)]
    assert zone_session.final_weight_validation["accepted"] is True
    assert zone_session.final_weight_validation["selectedProductId"] == 101
    assert zone_session.final_weight_validation["selectedCount"] == 3


def test_trigger_result_preserves_candidate_snapshot():
    from model_service.session.door_session import TriggerResult

    trigger = TriggerResult(
        trigger_id="trigger-1",
        session_id="session-1",
        timestamp=10.0,
        products=[],
        delta_weight=-100.0,
        confidence=0.4,
        video_paths={},
        vision_candidates=[
            _candidate_snapshot(
                31,
                "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML",
                rank=1,
                weight=523.0,
                confidence=0.41,
            )
        ],
    )

    restored = TriggerResult.from_dict(trigger.to_dict())

    assert restored.vision_candidates == trigger.vision_candidates
    assert TriggerResult.from_dict(
        {
            "trigger_id": "old",
            "session_id": "old-session",
            "timestamp": 1.0,
            "products": [],
            "delta_weight": -1.0,
            "video_paths": {},
        }
    ).vision_candidates == []


def test_door_session_preserves_deferred_returns():
    from model_service.session.door_session import DeferredReturn, DoorSession

    session = DoorSession(
        door_session_id="door-zone-1",
        zone=1,
        deferred_returns=[
            DeferredReturn(
                trigger_id="return-1",
                delta_weight=1033.0,
                timestamp=10.0,
                source="return_combo_deferred",
                replay_position="return",
                tolerance_used=5.0,
            )
        ],
    )

    restored = DoorSession.from_dict(session.to_dict())

    assert len(restored.deferred_returns) == 1
    assert restored.deferred_returns[0].source == "return_combo_deferred"
    assert restored.deferred_returns[0].delta_weight == 1033.0
    assert DoorSession.from_dict(
        {
            "door_session_id": "old",
            "zone": 1,
        }
    ).deferred_returns == []


def test_close_summary_exposes_final_weight_validation():
    from model_service.api.routes.multi_zone import _handle_door_close

    global_session = _make_global_session()
    zone_session = global_session.zone_sessions[1]
    zone_session.final_weight_validation = {
        "accepted": True,
        "reason": "candidate_repeat_final_weight_correction",
        "targetWeight": 1561.2,
    }

    response = _handle_door_close(FakeCloseReadyStore(global_session))

    validation = response["decisionSummary"]["zones"][0]["finalWeightValidation"]
    assert validation["accepted"] is True
    assert response["zones"][0]["finalWeightValidation"]["targetWeight"] == 1561.2


def test_close_summary_counts_mixed_return_hints_in_effective_delta():
    from model_service.api.routes.multi_zone import _handle_door_close
    from model_service.session.door_session import (
        AggregatedProduct,
        DoorSession,
        TriggerResult,
    )
    from model_service.session.global_door_session import GlobalDoorSession
    from model_service.session.session_store import ProductResult

    haluyache = ProductResult(
        product_id=95,
        product_idx="P95",
        name="BOTTLE_HY_HALUYACHE_PURPLE_200ML_V2",
        count=1,
        price=2000,
        confidence=0.9,
    )
    condition = ProductResult(
        product_id=113,
        product_idx="P113",
        name="STICK_INNON_CONDITION_STICK_18G",
        count=1,
        price=3000,
        confidence=0.9,
    )
    zone_session = DoorSession(
        door_session_id="door-zone-2",
        zone=2,
        status="complete",
        triggers=[
            TriggerResult(
                trigger_id="trigger-1",
                session_id="zone-2-haluyache",
                timestamp=10.0,
                products=[haluyache],
                delta_weight=-219.0,
                confidence=0.9,
                video_paths={},
            ),
            TriggerResult(
                trigger_id="trigger-2",
                session_id="zone-2-condition",
                timestamp=20.0,
                products=[condition],
                delta_weight=-16.5,
                confidence=0.9,
                video_paths={},
                return_weight_hints=[
                    {
                        "weight": 216.7,
                        "delta": 216.7,
                        "segment_index": 0,
                        "replay_position": "before_removal",
                    }
                ],
            ),
        ],
        aggregated_products={
            113: AggregatedProduct(
                product_id=113,
                product_idx="P113",
                name="STICK_INNON_CONDITION_STICK_18G",
                count=1,
                unit_price=3000,
                weight=19.0,
                total_confidence=0.9,
                detection_count=1,
            )
        },
        finalized_at=time.time(),
    )
    global_session = GlobalDoorSession(
        global_session_id="global-mixed-return",
        status="complete",
        zone_sessions={2: zone_session},
        finalized_at=time.time(),
    )

    response = _handle_door_close(FakeCloseReadyStore(global_session))

    summary = response["decisionSummary"]
    assert summary["zones"][1]["weightDelta"] == -18.8
    assert summary["totalWeightDelta"] == -18.8
    assert (
        "zone=2 weight_delta=-18.8g products=STICK_INNON_CONDITION_STICK_18Gx1"
        in response["decisionSummaryText"]
    )


def test_close_summary_ignores_unmatched_mixed_return_hint_when_basket_is_empty():
    from model_service.api.routes.multi_zone import _handle_door_close
    from model_service.session.door_session import (
        DoorSession,
        TriggerResult,
        UnmatchedReturn,
    )
    from model_service.session.global_door_session import GlobalDoorSession
    from model_service.session.session_store import ProductResult

    chapagetti = ProductResult(
        product_id=11,
        product_idx="P11",
        name="BAG_NONGSHIM_CHAPAGETTI_140G",
        count=1,
        price=4000,
        confidence=0.9,
    )
    zone_session = DoorSession(
        door_session_id="door-zone-1",
        zone=1,
        status="complete",
        triggers=[
            TriggerResult(
                trigger_id="trigger-removal",
                session_id="zone-1-chapagetti",
                timestamp=10.0,
                products=[chapagetti],
                delta_weight=-147.0,
                confidence=0.9,
                video_paths={},
                return_weight_hints=[
                    {
                        "weight": 67.7,
                        "delta": 67.7,
                        "segment_index": 0,
                        "replay_position": "before_removal",
                    }
                ],
            ),
            TriggerResult(
                trigger_id="trigger-return",
                session_id="zone-1-return",
                timestamp=20.0,
                products=[],
                delta_weight=149.0,
                confidence=0.0,
                video_paths={},
                is_return=True,
            ),
        ],
        aggregated_products={},
        unmatched_returns=[
            UnmatchedReturn(
                trigger_id="trigger-removal",
                delta_weight=67.7,
                timestamp=10.0,
                tolerance_used=5.0,
            )
        ],
        finalized_at=time.time(),
    )
    global_session = GlobalDoorSession(
        global_session_id="global-unmatched-hint",
        status="complete",
        zone_sessions={1: zone_session},
        finalized_at=time.time(),
    )

    response = _handle_door_close(FakeCloseReadyStore(global_session))

    summary = response["decisionSummary"]
    assert summary["zones"][0]["weightDelta"] == 0.0
    assert summary["totalWeightDelta"] == 0.0
    assert "zone=1 weight_delta=0.0g products=none" in response["decisionSummaryText"]


def test_close_summary_uses_effective_delta_for_cross_zone_return(tmp_path):
    from model_service.api.routes.multi_zone import _handle_door_close
    from model_service.session import DoorSessionStore, ProductResult
    from model_service.session.door_session import TriggerResult

    def get_weight(product_id: int) -> float:
        return {26: 365.0, 27: 250.0}.get(product_id, 0.0)

    def removal_trigger(zone: int, delta: float, product_id: int, name: str):
        return TriggerResult(
            trigger_id="",
            session_id=f"zone-{zone}-removal",
            timestamp=time.time(),
            products=[
                ProductResult(
                    product_id=product_id,
                    product_idx=str(product_id),
                    name=name,
                    count=1,
                    price=3000,
                    confidence=0.9,
                )
            ],
            delta_weight=delta,
            confidence=0.9,
            video_paths={},
            is_return=False,
        )

    def return_trigger(zone: int, delta: float):
        return TriggerResult(
            trigger_id="",
            session_id=f"zone-{zone}-return",
            timestamp=time.time(),
            products=[],
            delta_weight=delta,
            confidence=0.0,
            video_paths={},
            is_return=True,
        )

    store = DoorSessionStore(
        yaml_dir=str(tmp_path),
        session_timeout=30.0,
        weight_tolerance=5.0,
        get_product_weight=get_weight,
    )
    try:
        store.get_or_start_global_session()
        store.add_trigger_with_global(
            zone=1,
            result=removal_trigger(1, -365.0, 26, "치킨마요"),
        )
        store.add_trigger_with_global(
            zone=2,
            result=removal_trigger(2, -250.0, 27, "참치마요"),
        )
        store.add_trigger_with_global(zone=2, result=return_trigger(2, 365.0))
        global_session = store.finalize_global_session()
        assert global_session is not None

        response = _handle_door_close(FakeCloseReadyStore(global_session))
    finally:
        store.shutdown()

    summary = response["decisionSummary"]
    zone_1 = summary["zones"][0]
    zone_2 = summary["zones"][1]

    assert zone_1["weightDelta"] == 0.0
    assert zone_1["products"] == []
    assert zone_2["weightDelta"] == -250.0
    assert zone_2["products"][0]["name"] == "참치마요"
    assert summary["totalWeightDelta"] == -250.0
    assert "zone=2 weight_delta=-250.0g products=참치마요x1" in summary["summaryText"]
    assert "weight_delta=115.0g" not in summary["summaryText"]
    assert "weight_delta=365.0g" not in summary["summaryText"]


def test_close_summary_uses_effective_delta_for_cross_zone_combo_return(tmp_path):
    from model_service.api.routes.multi_zone import _handle_door_close
    from model_service.session import DoorSessionStore, ProductResult
    from model_service.session.door_session import TriggerResult

    def get_weight(product_id: int) -> float:
        return {26: 365.0, 27: 250.0, 28: 400.0}.get(product_id, 0.0)

    def removal_trigger(zone: int, delta: float, product_id: int, name: str):
        return TriggerResult(
            trigger_id="",
            session_id=f"zone-{zone}-removal-{product_id}",
            timestamp=time.time(),
            products=[
                ProductResult(
                    product_id=product_id,
                    product_idx=str(product_id),
                    name=name,
                    count=1,
                    price=3000,
                    confidence=0.9,
                )
            ],
            delta_weight=delta,
            confidence=0.9,
            video_paths={},
            is_return=False,
        )

    def return_trigger(zone: int, delta: float):
        return TriggerResult(
            trigger_id="",
            session_id=f"zone-{zone}-return",
            timestamp=time.time(),
            products=[],
            delta_weight=delta,
            confidence=0.0,
            video_paths={},
            is_return=True,
        )

    store = DoorSessionStore(
        yaml_dir=str(tmp_path),
        session_timeout=30.0,
        weight_tolerance=5.0,
        get_product_weight=get_weight,
    )
    try:
        store.get_or_start_global_session()
        store.add_trigger_with_global(
            zone=1,
            result=removal_trigger(1, -365.0, 26, "치킨마요"),
        )
        store.add_trigger_with_global(
            zone=1,
            result=removal_trigger(1, -250.0, 27, "참치마요"),
        )
        store.add_trigger_with_global(
            zone=2,
            result=removal_trigger(2, -400.0, 28, "김치볶음밥"),
        )
        store.add_trigger_with_global(zone=2, result=return_trigger(2, 615.0))
        global_session = store.finalize_global_session()
        assert global_session is not None

        response = _handle_door_close(FakeCloseReadyStore(global_session))
    finally:
        store.shutdown()

    summary = response["decisionSummary"]
    zone_1 = summary["zones"][0]
    zone_2 = summary["zones"][1]

    assert zone_1["weightDelta"] == 0.0
    assert zone_1["products"] == []
    assert zone_2["weightDelta"] == -400.0
    assert [product["name"] for product in zone_2["products"]] == ["김치볶음밥"]
    assert summary["totalWeightDelta"] == -400.0
    assert "zone=2 weight_delta=-400.0g products=김치볶음밥x1" in summary["summaryText"]
    assert "weight_delta=615.0g" not in summary["summaryText"]
    assert "weight_delta=215.0g" not in summary["summaryText"]


def test_complete_close_response_exposes_missing_active_products_failure():
    from model_service.api.routes.multi_zone import _handle_door_close
    from model_service.session.door_session import DoorSession, TriggerResult
    from model_service.session.global_door_session import GlobalDoorSession

    trigger = TriggerResult(
        trigger_id="trigger-missing-active",
        session_id="zone-3-session",
        timestamp=time.time(),
        products=[],
        delta_weight=-396.6,
        confidence=0.0,
        video_paths={"top": "top.avi", "side": "side.avi"},
        failure_reason="missing_active_products",
    )
    zone_session = DoorSession(
        door_session_id="door-zone-3",
        zone=3,
        status="complete",
        triggers=[trigger],
        finalized_at=time.time(),
    )
    global_session = GlobalDoorSession(
        global_session_id="global-missing-active",
        status="complete",
        zone_sessions={3: zone_session},
        finalized_at=time.time(),
    )

    response = _handle_door_close(FakeCloseReadyStore(global_session))

    assert response["success"] is False
    assert response["status"] == "missing_active_products"
    assert response["failureReasons"] == ["missing_active_products"]
    assert response["zones"][2]["failureReasons"] == ["missing_active_products"]
    assert response["decisionSummary"]["zones"][2]["failureReasons"] == [
        "missing_active_products"
    ]
    assert "reason=missing_active_products" in response["decisionSummaryText"]


def test_complete_close_response_exposes_no_charge_loadcell_diagnostic():
    from model_service.api.routes.multi_zone import _handle_door_close
    from model_service.session.global_door_session import GlobalDoorSession

    global_session = GlobalDoorSession(
        global_session_id="global-zone-1-zero",
        status="complete",
        no_charge_diagnostics=[
            {
                "zone": 1,
                "sessionId": "zone-1-zero-session",
                "reason": "loadcell_payload_all_zero",
                "deltaWeight": 0.0,
                "processingStage": "skipped_loadcell_payload_all_zero",
                "payloadState": "all_zero",
            }
        ],
        finalized_at=time.time(),
    )

    response = _handle_door_close(FakeCloseReadyStore(global_session))
    summary = response["decisionSummary"]
    zone_1 = summary["zones"][0]

    assert response["status"] == "complete_no_products"
    assert zone_1["products"] == []
    assert zone_1["weightDelta"] == 0.0
    assert zone_1["triggerCount"] == 0
    assert zone_1["noChargeDiagnostics"][0]["reason"] == "loadcell_payload_all_zero"
    assert summary["diagnosticZoneLines"] == [
        "zone=1 weight_delta=0.0g diagnostic=loadcell_payload_all_zero"
    ]
    assert response["decisionSummaryText"] == (
        "zones=none; total_weight_delta=0.0g total_price=0"
    )


def test_close_signal_uses_short_default_initial_wait(tmp_path):
    from model_service.session import DoorSessionStore

    store = DoorSessionStore(yaml_dir=str(tmp_path))
    try:
        store.get_or_start_global_session()

        ready, session = store.handle_close_signal(_now=100.0)
        assert ready is False
        assert session is not None

        ready, _ = store.handle_close_signal(_now=102.9)
        assert ready is False

        ready, _ = store.handle_close_signal(_now=103.0)
        assert ready is True
    finally:
        store.shutdown()


def test_close_signal_uses_short_default_trigger_debounce_after_processing(tmp_path):
    from model_service.session import DoorSessionStore

    store = DoorSessionStore(yaml_dir=str(tmp_path))
    try:
        global_session = store.get_or_start_global_session()
        global_session.last_trigger_at = 200.0

        store.notify_trigger_enqueued(zone=1)
        ready, session = store.handle_close_signal(_now=200.0)
        assert ready is False
        assert session is not None

        store.notify_trigger_processed(zone=1)
        global_session.last_trigger_at = 200.0

        ready, _ = store.handle_close_signal(_now=200.9)
        assert ready is False

        ready, _ = store.handle_close_signal(_now=201.0)
        assert ready is True
    finally:
        store.shutdown()


def test_close_signal_tracks_session_id_pending_until_processed(tmp_path):
    from model_service.session import DoorSessionStore

    store = DoorSessionStore(yaml_dir=str(tmp_path))
    try:
        global_session = store.get_or_start_global_session()
        global_session.last_trigger_at = 300.0

        store.notify_trigger_enqueued(zone=2, session_id="zone-2-session")
        store.notify_trigger_started(session_id="zone-2-session")
        snapshot = store.get_pending_trigger_snapshot()
        assert snapshot["pendingTriggerCount"] == 1
        assert snapshot["pendingTriggerZones"] == [2]
        assert snapshot["pendingTriggerSessionIds"] == ["zone-2-session"]
        assert snapshot["pendingTriggerStatuses"]["zone-2-session"] == "processing"

        ready, session = store.handle_close_signal(_now=360.0)
        assert ready is False
        assert session is not None

        store.notify_trigger_processed(session_id="zone-2-session", status="error")
        assert store.get_pending_trigger_snapshot()["pendingTriggerCount"] == 0
        global_session.last_trigger_at = 300.0

        ready, _ = store.handle_close_signal(_now=301.0)
        assert ready is True
    finally:
        store.shutdown()


def test_handle_door_close_pending_response_includes_trigger_details(tmp_path):
    from model_service.api.routes.multi_zone import _handle_door_close
    from model_service.session import DoorSessionStore

    store = DoorSessionStore(yaml_dir=str(tmp_path))
    try:
        store.get_or_start_global_session()
        store.notify_trigger_enqueued(zone=3, session_id="zone-3-session")
        store.notify_trigger_started(session_id="zone-3-session")

        response = _handle_door_close(store)

        assert response["success"] is False
        assert response["status"] == "in_progress"
        assert response["reason"] == "pending_trigger"
        assert response["pendingTriggerCount"] == 1
        assert response["pendingTriggerZones"] == [3]
        assert response["pendingTriggerSessionIds"] == ["zone-3-session"]
        assert response["pendingTriggerStatuses"] == {"zone-3-session": "processing"}
    finally:
        store.notify_trigger_processed(session_id="zone-3-session", status="error")
        store.shutdown()


def test_close_signal_ignores_non_chargeable_pending_for_finalize(tmp_path):
    from model_service.session import DoorSessionStore

    store = DoorSessionStore(yaml_dir=str(tmp_path))
    try:
        global_session = store.get_or_start_global_session()
        global_session.last_trigger_at = 100.0
        store.notify_trigger_enqueued(
            zone=3,
            session_id="return-only-session",
            chargeable_vision_required=False,
        )
        global_session.last_trigger_at = 100.0

        ready, session = store.handle_close_signal(_now=100.0)
        assert ready is False
        assert session is not None

        snapshot = store.get_pending_trigger_snapshot()
        assert snapshot["pendingTriggerCount"] == 1
        assert snapshot["pendingChargeableVisionCount"] == 0

        ready, _ = store.handle_close_signal(_now=101.0)
        assert ready is True
    finally:
        store.notify_trigger_processed(session_id="return-only-session")
        store.shutdown()


def _inventory_product(
    *,
    product_idx: str,
    product_name: str,
    yolo_class_id: int,
    weight: str,
    stock: int,
) -> dict:
    return {
        "product_idx": product_idx,
        "product_name": product_name,
        "product_eng_name": product_name,
        "sale_price": 2000,
        "product_weight": weight,
        "stock_qty": stock,
        "has_loadcell": "true",
        "yolo_class_id": yolo_class_id,
    }


def _inventory_product_without_class_id(
    *,
    product_idx: str,
    product_name: str,
    product_eng_name: str | None = None,
    weight: str,
    stock: int,
) -> dict:
    return {
        "product_idx": product_idx,
        "product_name": product_name,
        "product_eng_name": product_eng_name or product_name,
        "sale_price": 2000,
        "product_weight": weight,
        "stock_qty": stock,
        "has_loadcell": "true",
    }


@pytest.mark.asyncio
async def test_multi_zone_waiting_session_returns_stable_loadcell_contract(
    monkeypatch,
    session_store,
):
    import model_service.api.routes.multi_zone as multi_zone_module
    from model_service.session import SessionData

    monkeypatch.setattr(multi_zone_module, "_log_request_to_file", lambda *args: None)
    session_store.save(
        "zone_waiting",
        SessionData(
            session_id="zone_waiting",
            zone=1,
            products=[],
            total_price=0,
            delta_weight=-964.0,
            status="waiting",
            processing_stage="removal_waiting_for_stable_loadcell",
            processing_stage_detail="removal loadcell is not stable yet",
            failure_reason="missing_active_products",
        ),
    )

    response = await multi_zone_module.judge_multi_zone(
        body={"session_id": "zone_waiting", "zone": 1, "products": []},
        session_store=session_store,
        door_session_store=None,
        active_product_store=None,
    )

    assert response["success"] is False
    assert response["status"] == "waiting"
    assert response["reason"] == "waiting_for_stable_loadcell"
    assert response["waiting_for"] == "stable_loadcell"
    assert response["processing_stage"] == "removal_waiting_for_stable_loadcell"
    assert response["failureReasons"] == ["missing_active_products"]


@pytest.mark.asyncio
async def test_multi_zone_valid_inventory_updates_active_product_store(
    monkeypatch,
    session_store,
):
    import model_service.api.routes.multi_zone as multi_zone_module
    from model_service.session.active_product_store import ActiveProductStore

    monkeypatch.setattr(multi_zone_module, "_log_request_to_file", lambda *args: None)

    active_store = ActiveProductStore({"POWER_ADE": 23})
    await multi_zone_module.judge_multi_zone(
        body={
            "session_id": None,
            "zone": 3,
            "products": [
                _inventory_product(
                    product_idx="POWER",
                    product_name="POWER_ADE",
                    yolo_class_id=23,
                    weight="639",
                    stock=6,
                )
            ],
        },
        session_store=session_store,
        door_session_store=None,
        active_product_store=active_store,
    )

    assert active_store.get_allowed_class_ids() == [23]
    assert active_store.has_stock_positive_weight_products() is True
    assert active_store.get_stats()["stock_positive_weight_products"] == 1


@pytest.mark.asyncio
async def test_multi_zone_product_eng_name_updates_active_product_store(
    monkeypatch,
    session_store,
):
    import model_service.api.routes.multi_zone as multi_zone_module
    from model_service.session.active_product_store import ActiveProductStore

    monkeypatch.setattr(multi_zone_module, "_log_request_to_file", lambda *args: None)

    active_store = ActiveProductStore(
        {"CAN_LOTTE_HOT6_THE_KING_RUSH_355ML": 8},
        source_policy="node_first",
    )
    await multi_zone_module.judge_multi_zone(
        body={
            "session_id": None,
            "zone": 3,
            "products": [
                _inventory_product_without_class_id(
                    product_idx="HOT6",
                    product_name="\ud56b\uc2dd\uc2a4",
                    product_eng_name="CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
                    weight="355",
                    stock=6,
                )
            ],
        },
        session_store=session_store,
        door_session_store=None,
        active_product_store=active_store,
    )

    product = active_store.get_by_yolo_class_id(8)
    assert active_store.get_allowed_class_ids() == [8]
    assert product is not None
    assert product.product_name == "\ud56b\uc2dd\uc2a4"
    assert product.product_eng_name == "CAN_LOTTE_HOT6_THE_KING_RUSH_355ML"
    assert product.class_id_source == "product_eng_name_engine"
    assert active_store.has_stock_positive_weight_products() is True


@pytest.mark.asyncio
async def test_multi_zone_edge_product_eng_name_payload_updates_active_product_store(
    monkeypatch,
    session_store,
):
    import model_service.api.routes.multi_zone as multi_zone_module
    from model_service.session.active_product_store import ActiveProductStore

    monkeypatch.setattr(multi_zone_module, "_log_request_to_file", lambda *args: None)

    active_store = ActiveProductStore(
        {
            "BAG_BINGGRAE_KKOTCHIGELANG_75G": 3,
            "BAG_BINGGRAE_KKOTCHIGELANG_GREEN_75G": 4,
            "BOX_ORION_DIGET_SSIN_84G": 76,
        },
        source_policy="node_first",
    )
    await multi_zone_module.judge_multi_zone(
        body={
            "session_id": None,
            "zone": 1,
            "products": [
                {
                    "division_idx": "DI17798460900133031",
                    "device_idx": "DE17798461293792881",
                    "product_idx": "P17790836506994281",
                    "product_name": "\ube59\uadf8\ub808 \uaf43\uac8c\ub791 \uc624\ub9ac\uc9c0\ub110",
                    "product_eng_name": "BAG_BINGGRAE_KKOTCHIGELANG_75G",
                    "sale_price": 500,
                    "stock_qty": 1,
                    "product_weight": "75",
                    "product_loadcell_weight": "83",
                    "has_loadcell": "Y",
                },
                {
                    "division_idx": "DI17798460900133031",
                    "device_idx": "DE17798461293792881",
                    "product_idx": "P17790836506994281",
                    "product_name": "\ube59\uadf8\ub808 \uaf43\uac8c\ub791 \uc624\ub9ac\uc9c0\ub110",
                    "product_eng_name": "BAG_BINGGRAE_KKOTCHIGELANG_GREEN_75G",
                    "sale_price": 500,
                    "stock_qty": 1,
                    "product_weight": "75",
                    "product_loadcell_weight": "79",
                    "has_loadcell": "Y",
                },
                {
                    "division_idx": "DI17798460900133031",
                    "device_idx": "DE17798461293792881",
                    "product_idx": "P17791537732012107",
                    "product_name": "\uc624\ub9ac\uc628 \ub2e4\uc774\uc81c \uc52c",
                    "product_eng_name": "BOX_ORION_DIGET_SSIN_84G",
                    "sale_price": 1000,
                    "stock_qty": 20,
                    "product_weight": "84",
                    "product_loadcell_weight": "110",
                    "has_loadcell": "Y",
                },
            ],
        },
        session_store=session_store,
        door_session_store=None,
        active_product_store=active_store,
    )

    assert active_store.get_allowed_class_ids() == [3, 4, 76]
    assert active_store.get_stats()["stock_positive_class_products"] == 3
    assert active_store.get_stats()["stock_positive_weight_products"] == 3


@pytest.mark.asyncio
async def test_multi_zone_legacy_product_name_class_payload_updates_active_product_store(
    monkeypatch,
    session_store,
):
    import model_service.api.routes.multi_zone as multi_zone_module
    from model_service.session.active_product_store import ActiveProductStore

    monkeypatch.setattr(multi_zone_module, "_log_request_to_file", lambda *args: None)

    active_store = ActiveProductStore(
        {
            "BAG_BIBIGO_CHEONGYANG_MEAT_DUMPLINGS_200G": 3,
            "BOX_BINGGRAE_YOMAMTE_150ML": 30,
            "STICK_LALA_SWEET_GRAPE_ZERO_70ML": 46,
        },
        source_policy="node_first",
    )
    await multi_zone_module.judge_multi_zone(
        body={
            "session_id": None,
            "zone": 1,
            "products": [
                {
                    "product_idx": "P17514184380842082",
                    "product_name": "BAG_BIBIGO_CHEONGYANG_MEAT_DUMPLINGS_200G",
                    "sale_price": 5000,
                    "stock_qty": 100,
                    "product_weight": "223",
                    "has_loadcell": "Y",
                },
                {
                    "product_idx": "P17437536515731485",
                    "product_name": "BOX_BINGGRAE_YOMAMTE_150ML",
                    "sale_price": 2500,
                    "stock_qty": 100,
                    "product_weight": "87",
                    "has_loadcell": "Y",
                },
                {
                    "product_idx": "P17815766992187371",
                    "product_name": "STICK_LALA_SWEET_GRAPE_ZERO_70ML",
                    "sale_price": 2000,
                    "stock_qty": 100,
                    "product_weight": "71",
                    "has_loadcell": "Y",
                },
            ],
        },
        session_store=session_store,
        door_session_store=None,
        active_product_store=active_store,
    )

    assert active_store.get_allowed_class_ids() == [3, 30, 46]
    assert active_store.get_stats()["stock_positive_class_products"] == 3
    assert active_store.get_by_yolo_class_id(3).class_id_source == (
        "product_name_engine_legacy"
    )


@pytest.mark.asyncio
async def test_multi_zone_name_compat_payload_updates_active_product_store(
    monkeypatch,
    session_store,
):
    import model_service.api.routes.multi_zone as multi_zone_module
    from model_service.session.active_product_store import ActiveProductStore

    monkeypatch.setattr(multi_zone_module, "_log_request_to_file", lambda *args: None)

    active_store = ActiveProductStore(
        {"BAG_BIBIGO_CHEONGYANG_MEAT_DUMPLINGS_200G": 3},
        source_policy="node_first",
    )
    await multi_zone_module.judge_multi_zone(
        body={
            "session_id": None,
            "zone": 1,
            "products": [
                {
                    "product_idx": "P17514184380842082",
                    "product_name": "비비고 청양고추 찐만두",
                    "name": "BAG_BIBIGO_CHEONGYANG_MEAT_DUMPLINGS_200G",
                    "sale_price": 5000,
                    "stock_qty": 100,
                    "product_weight": "223",
                    "has_loadcell": "Y",
                },
            ],
        },
        session_store=session_store,
        door_session_store=None,
        active_product_store=active_store,
    )

    product = active_store.get_by_yolo_class_id(3)
    assert active_store.get_allowed_class_ids() == [3]
    assert product.product_name == "비비고 청양고추 찐만두"
    assert product.class_id_source == "name_engine_compat"


@pytest.mark.asyncio
async def test_multi_zone_rejects_product_eng_name_camel_case_alias(
    monkeypatch,
    caplog,
    session_store,
):
    import model_service.api.routes.multi_zone as multi_zone_module
    from model_service.session.active_product_store import ActiveProductStore

    monkeypatch.setattr(multi_zone_module, "_log_request_to_file", lambda *args: None)

    active_store = ActiveProductStore(
        {"BAG_BINGGRAE_KKOTCHIGELANG_75G": 3},
        source_policy="node_first",
    )

    caplog.set_level(logging.WARNING)
    await multi_zone_module.judge_multi_zone(
        body={
            "session_id": None,
            "zone": 1,
            "products": [
                {
                    "product_idx": "P3",
                    "product_name": "Display name",
                    "productEngName": "BAG_BINGGRAE_KKOTCHIGELANG_75G",
                    "sale_price": 500,
                    "stock_qty": 1,
                    "product_weight": "75",
                }
            ],
        },
        session_store=session_store,
        door_session_store=None,
        active_product_store=active_store,
    )

    assert active_store.get_allowed_class_ids() == []
    assert "[MULTI-ZONE] unmapped products: ['Display name']" in caplog.text
    assert "product_eng_name=" in caplog.text
    assert "class_key_source=product_name" in caplog.text
    assert "'eng_name':" not in caplog.text


@pytest.mark.asyncio
async def test_multi_zone_zero_weight_valid_inventory_updates_vision_allowlist(
    monkeypatch,
    session_store,
):
    import model_service.api.routes.multi_zone as multi_zone_module
    from model_service.session.active_product_store import ActiveProductStore

    monkeypatch.setattr(multi_zone_module, "_log_request_to_file", lambda *args: None)

    active_store = ActiveProductStore({"CORN_TEA": 44})
    await multi_zone_module.judge_multi_zone(
        body={
            "session_id": None,
            "zone": 3,
            "products": [
                _inventory_product(
                    product_idx="CORN",
                    product_name="CORN_TEA",
                    yolo_class_id=44,
                    weight="0",
                    stock=6,
                )
            ],
        },
        session_store=session_store,
        door_session_store=None,
        active_product_store=active_store,
    )

    assert active_store.get_allowed_class_ids() == [44]
    assert active_store.has_stock_positive_class_products() is True
    assert active_store.has_stock_positive_weight_products() is False
    assert active_store.get_stats()["weight_unavailable_products"] == 1


@pytest.mark.asyncio
async def test_multi_zone_close_payload_does_not_overwrite_inventory_snapshot(
    monkeypatch,
    caplog,
    session_store,
):
    import model_service.api.routes.multi_zone as multi_zone_module
    from model_service.session.active_product_store import ActiveProductStore

    monkeypatch.setattr(multi_zone_module, "_log_request_to_file", lambda *args: None)

    active_store = ActiveProductStore(
        {
            "LABNOSH_PROTEIN": 45,
            "BOX_ORION_DIGET_SSIN_84G": 76,
        }
    )
    active_store.set_products(
        [
            _inventory_product(
                product_idx="LABNOSH",
                product_name="LABNOSH_PROTEIN",
                yolo_class_id=45,
                weight="396",
                stock=4,
            )
        ]
    )

    caplog.set_level(logging.INFO)
    await multi_zone_module.judge_multi_zone(
        body={
            "session_id": "CLOSE",
            "zone": 3,
            "products": [
                _inventory_product(
                    product_idx="DIGET",
                    product_name="BOX_ORION_DIGET_SSIN_84G",
                    yolo_class_id=76,
                    weight="0",
                    stock=0,
                )
            ],
        },
        session_store=session_store,
        door_session_store=None,
        active_product_store=active_store,
    )

    assert active_store.get_allowed_class_ids() == [45]
    assert active_store.get_by_yolo_class_id(76) is None
    assert "reason=close_request" in caplog.text


@pytest.mark.asyncio
async def test_multi_zone_zero_stock_only_payload_preserves_valid_snapshot(
    monkeypatch,
    caplog,
    session_store,
):
    import model_service.api.routes.multi_zone as multi_zone_module
    from model_service.session.active_product_store import ActiveProductStore

    monkeypatch.setattr(multi_zone_module, "_log_request_to_file", lambda *args: None)

    active_store = ActiveProductStore(
        {
            "LABNOSH_PROTEIN": 45,
            "BOX_ORION_DIGET_SSIN_84G": 76,
        }
    )
    active_store.set_products(
        [
            _inventory_product(
                product_idx="LABNOSH",
                product_name="LABNOSH_PROTEIN",
                yolo_class_id=45,
                weight="396",
                stock=4,
            )
        ]
    )

    caplog.set_level(logging.WARNING)
    await multi_zone_module.judge_multi_zone(
        body={
            "session_id": None,
            "zone": 3,
            "products": [
                _inventory_product(
                    product_idx="DIGET",
                    product_name="BOX_ORION_DIGET_SSIN_84G",
                    yolo_class_id=76,
                    weight="0",
                    stock=0,
                )
            ],
        },
        session_store=session_store,
        door_session_store=None,
        active_product_store=active_store,
    )

    assert active_store.get_allowed_class_ids() == [45]
    assert active_store.get_by_yolo_class_id(76) is None
    assert "reason=zero_stock_positive_class_products" in caplog.text
    assert "preserved_existing=true" in caplog.text
