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
    source="vision",
    stock=10,
):
    return {
        "rank": rank,
        "product_id": product_id,
        "product_idx": f"P{product_id}",
        "name": name,
        "unit_weight": weight,
        "unit_price": price,
        "stock_qty": stock,
        "confidence": confidence,
        "source": source,
        "top": False,
        "side": True,
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


def test_freezer_close_deferred_candidate_repairs_later_unused_weight_match(
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
                ),
            ],
        ),
    )

    global_session = store.finalize_global_session()

    zone3_session = global_session.zone_sessions[3]
    zone4_session = global_session.zone_sessions[4]
    assert [(product.product_id, product.count) for product in zone3_session.get_active_products()] == [
        (44, 1)
    ]
    assert [(product.product_id, product.count) for product in zone4_session.get_active_products()] == [
        (30, 1)
    ]
    validation = zone3_session.final_weight_validation
    assert validation["accepted"] is True
    assert validation["reason"] == "deferred_candidate_final_weight_correction"
    assert validation["selectedProductId"] == 44
    assert validation["sourceZone"] == 4
    assert validation["candidateRank"] == 4
    assert validation["deferredCandidateRepair"]["reason"] == (
        "later_unused_candidate_weight_match"
    )
    assert "[CLOSE][CANDIDATE_REPAIR] corrected zone=3" in caplog.text


def test_freezer_close_deferred_candidate_does_not_borrow_consumed_match(
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
    assert [(product.product_id, product.count) for product in zone3_session.get_active_products()] == [
        (44, 1)
    ]
    validation = zone3_session.final_weight_validation
    assert validation["reason"] == "unresolved_final_weight_mismatch"
    repair = validation["deferredCandidateRepair"]
    assert repair["applied"] is False
    assert repair["reason"] == "no_later_unused_weight_match"
    assert repair["rejectedCandidates"][0]["reason"] == "consumed_by_source_result"


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
