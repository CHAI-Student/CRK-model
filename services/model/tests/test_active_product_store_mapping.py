import json
from pathlib import Path


def test_active_product_store_uses_product_eng_name_and_ignores_direct_fields():
    from model_service.session.active_product_store import ActiveProductStore

    store = ActiveProductStore(
        yolo_name_to_id={
            "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML": 31,
            "BAG_JAYEONLU_MOIST_SWEET_CHESTNUT_80G": 96,
        }
    )

    result = store.set_products(
        [
            {
                "product_idx": "P31",
                "product_name": "Korean display name",
                "product_eng_name": "BOTTLE_WOONGIN_SKY_BARLEY_TEA_500ML",
                "yolo_class_id": 9999,
                "sale_price": 1800,
                "product_weight": "523",
                "stock_qty": 3,
            },
            {
                "product_idx": "P96",
                "product_name": "Another display name",
                "product_eng_name": "BAG_JAYEONLU_MOIST_SWEET_CHESTNUT_80G",
                "yolo_class_name": "STALE_CLASS_NAME",
                "sale_price": 2500,
                "product_weight": "80",
                "stock_qty": 0,
            },
        ]
    )

    assert result.total_products == 2
    assert result.mapped_products == 2
    assert result.allowed_products == 1
    assert result.zero_stock_products == 1
    assert result.unmapped_names == []
    assert result.invalid_class_id_total == 0
    assert store.get_allowed_class_ids() == [31]
    assert store.get_by_yolo_class_id(31).product_name == "Korean display name"
    assert store.get_by_yolo_class_id(96).stock_qty == 0


def test_active_product_store_rejects_trainingidx_without_product_eng_name():
    from model_service.session.active_product_store import ActiveProductStore

    store = ActiveProductStore(
        yolo_name_to_id={"ENGINE_CLASS_NAME": 44},
        source_policy="node_first",
    )

    result = store.set_products(
        [
            {
                "product_idx": "P44",
                "product_name": "Store display name",
                "trainingidx": 44,
                "sale_price": 1800,
                "product_weight": "520",
                "stock_qty": 3,
            }
        ]
    )

    assert result.mapped_products == 0
    assert result.unmapped_names == ["Store display name"]
    assert store.get_allowed_class_ids() == []


def test_active_product_store_uses_legacy_product_name_engine_match():
    from model_service.session.active_product_store import ActiveProductStore

    store = ActiveProductStore(
        yolo_name_to_id={"PRODUCT_A": 1},
        source_policy="node_first",
        product_name_fallback_enabled=False,
    )

    result = store.set_products(
        [
            {
                "product_idx": "P1",
                "product_name": "PRODUCT_A",
                "sale_price": 1000,
                "product_weight": "100",
                "stock_qty": 2,
            }
        ]
    )

    assert result.mapped_products == 1
    assert result.unmapped_names == []
    assert store.get_allowed_class_ids() == [1]
    assert store.get_by_yolo_class_id(1).class_id_source == "product_name_engine_legacy"


def test_active_product_store_node_first_uses_product_eng_name_engine_match():
    from model_service.session.active_product_store import ActiveProductStore

    store = ActiveProductStore(
        yolo_name_to_id={"CAN_LOTTE_HOT6_THE_KING_RUSH_355ML": 8},
        source_policy="node_first",
    )

    result = store.set_products(
        [
            {
                "product_idx": "P8",
                "product_name": "\ud55c\uae00 \ud45c\uc2dc\uba85",
                "product_eng_name": "CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
                "sale_price": 1800,
                "product_weight": "355",
                "stock_qty": 2,
            }
        ]
    )

    assert result.mapped_products == 1
    assert result.unmapped_names == []
    assert store.get_allowed_class_ids() == [8]
    product = store.get_by_yolo_class_id(8)
    assert product.product_name == "\ud55c\uae00 \ud45c\uc2dc\uba85"
    assert product.product_eng_name == "CAN_LOTTE_HOT6_THE_KING_RUSH_355ML"
    assert product.product_idx == "P8"
    assert product.class_id_source == "product_eng_name_engine"


def test_active_product_store_uses_name_compat_engine_match():
    from model_service.session.active_product_store import ActiveProductStore

    store = ActiveProductStore(
        yolo_name_to_id={"BAG_BIBIGO_CHEONGYANG_MEAT_DUMPLINGS_200G": 3},
        source_policy="node_first",
    )

    result = store.set_products(
        [
            {
                "product_idx": "P3",
                "product_name": "비비고 청양고추 찐만두",
                "name": "BAG_BIBIGO_CHEONGYANG_MEAT_DUMPLINGS_200G",
                "sale_price": 5000,
                "product_weight": "223",
                "stock_qty": 100,
            }
        ]
    )

    product = store.get_by_yolo_class_id(3)

    assert result.mapped_products == 1
    assert result.unmapped_names == []
    assert store.get_allowed_class_ids() == [3]
    assert product.product_name == "비비고 청양고추 찐만두"
    assert product.product_eng_name == ""
    assert product.class_id_source == "name_engine_compat"


def test_active_product_store_product_eng_name_wins_over_name_compat():
    from model_service.session.active_product_store import ActiveProductStore

    store = ActiveProductStore(
        yolo_name_to_id={"PRODUCT_A": 1, "PRODUCT_B": 2},
        source_policy="node_first",
    )

    result = store.set_products(
        [
            {
                "product_idx": "P1",
                "product_name": "Display name",
                "product_eng_name": "PRODUCT_A",
                "name": "PRODUCT_B",
                "sale_price": 1000,
                "product_weight": "100",
                "stock_qty": 2,
            }
        ]
    )

    assert result.mapped_products == 1
    assert store.get_allowed_class_ids() == [1]
    assert store.get_by_yolo_class_id(1).class_id_source == "product_eng_name_engine"
    assert store.get_by_yolo_class_id(2) is None


def test_active_product_store_product_eng_name_mapping_ignores_changed_product_idx():
    from model_service.session.active_product_store import ActiveProductStore

    store = ActiveProductStore(
        yolo_name_to_id={"BAG_NONGSHIM_POSTIC_84G": 70},
        source_policy="node_first",
    )

    first = store.set_products(
        [
            {
                "product_idx": "OLD_PRODUCT_IDX",
                "product_name": "\ub18d\uc2ec \ud3ec\uc2a4\ud2f1",
                "product_eng_name": "BAG_NONGSHIM_POSTIC_84G",
                "sale_price": 500,
                "product_weight": "84",
                "stock_qty": 100,
            }
        ]
    )
    second = store.set_products(
        [
            {
                "product_idx": "NEW_PRODUCT_IDX",
                "product_name": "\ub18d\uc2ec \ud3ec\uc2a4\ud2f1",
                "product_eng_name": "BAG_NONGSHIM_POSTIC_84G",
                "sale_price": 500,
                "product_weight": "84",
                "stock_qty": 100,
            }
        ]
    )

    product = store.get_by_yolo_class_id(70)

    assert first.mapped_products == 1
    assert second.mapped_products == 1
    assert store.get_allowed_class_ids() == [70]
    assert product.product_idx == "NEW_PRODUCT_IDX"
    assert product.product_name == "\ub18d\uc2ec \ud3ec\uc2a4\ud2f1"
    assert product.product_eng_name == "BAG_NONGSHIM_POSTIC_84G"


def test_active_product_store_rejects_eng_name_alias():
    from model_service.session.active_product_store import ActiveProductStore

    store = ActiveProductStore(
        yolo_name_to_id={"BAG_BINGGRAE_KKOTCHIGELANG_75G": 3},
        source_policy="node_first",
    )

    result = store.set_products(
        [
            {
                "product_idx": "P3",
                "product_name": "Display name",
                "eng_name": "BAG_BINGGRAE_KKOTCHIGELANG_75G",
                "sale_price": 1500,
                "product_weight": "75",
                "stock_qty": 2,
            }
        ]
    )

    assert result.mapped_products == 0
    assert result.unmapped_names == ["Display name"]
    assert store.get_allowed_class_ids() == []
    assert store.get_by_yolo_class_id(3) is None


def test_active_product_store_rejects_product_eng_name_camel_case_alias():
    from model_service.session.active_product_store import ActiveProductStore

    store = ActiveProductStore(
        yolo_name_to_id={"BAG_BINGGRAE_KKOTCHIGELANG_75G": 3},
        source_policy="node_first",
    )

    result = store.set_products(
        [
            {
                "product_idx": "P3",
                "product_name": "Display name",
                "productEngName": "BAG_BINGGRAE_KKOTCHIGELANG_75G",
                "sale_price": 1500,
                "product_weight": "75",
                "stock_qty": 2,
            }
        ]
    )

    assert result.mapped_products == 0
    assert result.unmapped_names == ["Display name"]
    assert store.get_allowed_class_ids() == []
    assert store.get_by_yolo_class_id(3) is None


def test_active_product_store_product_eng_name_overrides_invalid_direct_class_id():
    from model_service.session.active_product_store import ActiveProductStore

    store = ActiveProductStore(
        yolo_name_to_id={"CAN_LOTTE_HOT6_THE_KING_RUSH_355ML": 8},
        source_policy="node_first",
    )

    result = store.set_products(
        [
            {
                "product_idx": "P8",
                "product_name": "핫식스",
                "product_eng_name": "CAN_LOTTE_HOT6_THE_KING_RUSH_355ML",
                "trainingidx": "9999",
                "sale_price": 1800,
                "product_weight": "355",
                "stock_qty": 2,
            }
        ]
    )

    assert result.mapped_products == 1
    assert result.invalid_class_id_total == 0
    assert result.invalid_class_ids == []
    assert result.unmapped_names == []
    assert store.get_allowed_class_ids() == [8]
    assert store.get_by_yolo_class_id(8).class_id_source == "product_eng_name_engine"


def test_active_product_store_static_mapping_compat_still_uses_engine_product_eng_name():
    from model_service.session.active_product_store import ActiveProductStore

    store = ActiveProductStore(
        yolo_name_to_id={"PRODUCT_A": 1},
        source_policy="static_mapping_compat",
    )

    result = store.set_products(
        [
            {
                "product_idx": "P1",
                "product_name": "\ud55c\uae00 \uc0c1\ud488\uba85",
                "product_eng_name": "PRODUCT_A",
                "sale_price": 1000,
                "product_weight": "100",
                "stock_qty": 2,
            }
        ]
    )

    assert result.mapped_products == 1
    assert store.get_allowed_class_ids() == [1]


def test_active_product_store_does_not_use_korean_product_name_as_class_key():
    from model_service.session.active_product_store import ActiveProductStore

    store = ActiveProductStore(
        yolo_name_to_id={"PRODUCT_A": 1},
        source_policy="node_first",
    )

    result = store.set_products(
        [
            {
                "product_idx": "P1",
                "product_name": "\ud55c\uae00 \uc0c1\ud488\uba85",
                "sale_price": 1000,
                "product_weight": "100",
                "stock_qty": 2,
            }
        ]
    )

    assert result.mapped_products == 0
    assert result.unmapped_names == ["\ud55c\uae00 \uc0c1\ud488\uba85"]
    assert store.get_allowed_class_ids() == []


def test_active_product_store_static_mapping_does_not_override_engine_class_name():
    from model_service.session.active_product_store import ActiveProductStore

    store = ActiveProductStore(
        yolo_name_to_id={"BAG_BINGGRAE_KKOTCHIGELANG_75G": 3},
        source_policy="node_first",
    )
    loaded = store.load_yolo_mapping(
        {
            "mappings": [
                {
                    "yolo_class_id": 74,
                    "yolo_class_name": "BAG_BINGGRAE_KKOTCHIGELANG_75G",
                }
            ]
        }
    )

    result = store.set_products(
        [
            {
                "product_idx": "P-FRESH",
                "product_name": "\ube59\uadf8\ub808 \uaf43\uac8c\ub791",
                "product_eng_name": "BAG_BINGGRAE_KKOTCHIGELANG_75G",
                "sale_price": 500,
                "product_weight": "75",
                "stock_qty": 100,
            }
        ]
    )

    assert loaded == 1
    assert result.mapped_products == 1
    assert result.invalid_class_id_total == 0
    assert store.get_allowed_class_ids() == [3]
    assert store.get_by_yolo_class_id(3).class_id_source == "product_eng_name_engine"


def test_active_product_store_alias_maps_jayeonlu_jayeonru_variant():
    from model_service.session.active_product_store import ActiveProductStore

    store = ActiveProductStore(
        yolo_name_to_id={
            "BAG_JAYEONLU_MOIST_SWEET_CHESTNUT_80G": 96,
        },
        source_policy="static_mapping_compat",
    )

    result = store.set_products(
        [
            {
                "product_idx": "P96",
                "product_name": "\uc790\uc5f0\ub8e8 \ub9db\uad70\ubc24",
                "product_eng_name": "BAG_JAYEONRU_MOIST_SWEET_CHESTNUT_80G",
                "sale_price": 2500,
                "product_weight": "80",
                "stock_qty": 1,
            }
        ]
    )

    assert result.mapped_products == 1
    assert result.allowed_products == 1
    assert store.get_allowed_class_ids() == [96]


def test_yolo_mapping_file_matches_dataset_class_96_name():
    mapping_path = Path("services/config/yolo_product_mapping.json")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))

    names_by_id = {
        item["yolo_class_id"]: item["yolo_class_name"]
        for item in mapping["mappings"]
    }

    assert names_by_id[96] == "BAG_JAYEONLU_MOIST_SWEET_CHESTNUT_80G"


def test_active_product_store_stats_expose_mapping_diagnostics():
    from model_service.session.active_product_store import ActiveProductStore

    store = ActiveProductStore({"PRODUCT_A": 1}, source_policy="static_mapping_compat")

    result = store.set_products(
        [
            {
                "product_idx": "P1",
                "product_name": "\ud45c\uc2dc\uba85 A",
                "product_eng_name": "PRODUCT_A",
                "sale_price": 1000,
                "product_weight": "100",
                "stock_qty": 0,
            },
            {
                "product_idx": "P2",
                "product_name": "\uc54c \uc218 \uc5c6\ub294 \uc0c1\ud488",
                "product_eng_name": "UNKNOWN_PRODUCT",
                "sale_price": 1000,
                "product_weight": "100",
                "stock_qty": 5,
            },
        ]
    )

    stats = store.get_stats()

    assert result.zero_stock_products == 1
    assert result.unmapped_total == 1
    assert result.unmapped_names == ["UNKNOWN_PRODUCT"]
    assert stats["products_count"] == 1
    assert stats["allowed_classes_count"] == 0
    assert stats["zero_stock_products"] == 1
    assert stats["unmapped_products"] == 1


def test_active_product_store_preserves_valid_snapshot_on_invalid_overwrite(caplog):
    from model_service.session.active_product_store import ActiveProductStore

    store = ActiveProductStore({"PRODUCT_A": 1, "PRODUCT_B": 2})
    store.set_products(
        [
            {
                "product_idx": "P1",
                "product_name": "PRODUCT_A",
                "product_eng_name": "PRODUCT_A",
                "yolo_class_id": 1,
                "sale_price": 1000,
                "product_weight": "100",
                "stock_qty": 4,
            }
        ]
    )

    result = store.set_products(
        [
            {
                "product_idx": "P2",
                "product_name": "PRODUCT_B",
                "product_eng_name": "PRODUCT_B",
                "yolo_class_id": 2,
                "sale_price": 1200,
                "product_weight": "0",
                "stock_qty": 0,
            }
        ],
        preserve_on_invalid_existing=True,
    )

    stats = store.get_stats()

    assert result.preserved_existing is True
    assert result.stock_positive_weight_products == 0
    assert store.get_allowed_class_ids() == [1]
    assert store.get_by_yolo_class_id(1).product_name == "PRODUCT_A"
    assert store.get_by_yolo_class_id(2) is None
    assert stats["stock_positive_weight_products"] == 1
    assert "ignored invalid product snapshot" in caplog.text


def test_active_product_store_uses_last_valid_snapshot_after_clear():
    from model_service.session.active_product_store import ActiveProductStore

    store = ActiveProductStore({"PRODUCT_A": 1})
    store.set_products(
        [
            {
                "product_idx": "P1",
                "product_name": "PRODUCT_A",
                "product_eng_name": "PRODUCT_A",
                "yolo_class_id": 1,
                "sale_price": 1000,
                "product_weight": "100",
                "stock_qty": 4,
            }
        ]
    )

    assert store.clear() is True
    assert store.has_products() is False

    snapshot = store.get_effective_snapshot()

    assert snapshot.source == "last_valid"
    assert snapshot.used_last_valid_snapshot is True
    assert snapshot.allowed_class_ids == [1]
    assert snapshot.products[0].product_name == "PRODUCT_A"
    assert store.get_product_weight(1) == 100.0


def test_active_product_store_invalid_payload_after_clear_keeps_last_valid_fallback():
    from model_service.session.active_product_store import ActiveProductStore

    store = ActiveProductStore({"PRODUCT_A": 1, "PRODUCT_B": 2})
    store.set_products(
        [
            {
                "product_idx": "P1",
                "product_name": "PRODUCT_A",
                "product_eng_name": "PRODUCT_A",
                "yolo_class_id": 1,
                "sale_price": 1000,
                "product_weight": "100",
                "stock_qty": 4,
            }
        ]
    )
    store.clear()

    store.set_products(
        [
            {
                "product_idx": "P2",
                "product_name": "PRODUCT_B",
                "product_eng_name": "PRODUCT_B",
                "yolo_class_id": 2,
                "sale_price": 1200,
                "product_weight": "0",
                "stock_qty": 0,
            }
        ]
    )

    snapshot = store.get_effective_snapshot()

    assert snapshot.source == "last_valid"
    assert snapshot.current_snapshot_present is True
    assert snapshot.allowed_class_ids == [1]
    assert snapshot.products[0].product_name == "PRODUCT_A"


def test_active_product_store_last_valid_fallback_expires(monkeypatch):
    import model_service.session.active_product_store as active_product_store_module
    from model_service.session.active_product_store import ActiveProductStore

    now = 1000.0
    monkeypatch.setattr(active_product_store_module.time, "time", lambda: now)

    store = ActiveProductStore({"PRODUCT_A": 1}, last_valid_ttl_seconds=10.0)
    store.set_products(
        [
            {
                "product_idx": "P1",
                "product_name": "PRODUCT_A",
                "product_eng_name": "PRODUCT_A",
                "yolo_class_id": 1,
                "sale_price": 1000,
                "product_weight": "100",
                "stock_qty": 4,
            }
        ]
    )
    store.clear()

    now = 1011.0
    snapshot = store.get_effective_snapshot()

    assert snapshot.source == "missing"
    assert snapshot.allowed_class_ids is None
    assert snapshot.last_valid_snapshot_expired is True
    assert store.get_product_weight(1) == 0.0


def test_multi_zone_product_info_accepts_product_weight_alias():
    from model_service.api.routes.multi_zone import ProductInfo

    product = ProductInfo.model_validate(
        {
            "product_idx": "P44",
            "product_name": "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
            "sale_price": 1800,
            "productWeight": 520,
            "stock_qty": 2,
        }
    )

    assert float(product.product_weight) == 520.0

    product_with_trainingidx = ProductInfo.model_validate(
        {
            "product_idx": "P44",
            "product_name": "Store display name",
            "sale_price": 1800,
            "trainingidx": 44,
            "stock_qty": 2,
        }
    )

    assert product_with_trainingidx.yolo_class_id == 44


def test_active_product_store_accepts_product_weight_aliases():
    from model_service.session.active_product_store import ActiveProductStore

    store = ActiveProductStore({"BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML": 44})

    result = store.set_products(
        [
            {
                "product_idx": "P44",
                "product_name": "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                "product_eng_name": "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                "yolo_class_id": 44,
                "sale_price": 1800,
                "productWeight": 520,
                "stock_qty": 2,
            }
        ]
    )

    assert store.get_product_weight(44) == 520.0
    assert result.repaired_weight_products == 1
    assert result.repaired_weight_diagnostics[0]["source"] == "payload_alias"
    assert result.repaired_weight_diagnostics[0]["field"] == "productWeight"


def test_active_product_store_repairs_zero_weight_from_current_snapshot():
    from model_service.session.active_product_store import ActiveProductStore

    store = ActiveProductStore({"BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML": 44})
    store.set_products(
        [
            {
                "product_idx": "P44",
                "product_name": "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                "product_eng_name": "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                "yolo_class_id": 44,
                "sale_price": 1800,
                "product_weight": "520",
                "stock_qty": 2,
            }
        ]
    )

    result = store.set_products(
        [
            {
                "product_idx": "P44",
                "product_name": "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                "product_eng_name": "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                "yolo_class_id": 44,
                "sale_price": 1800,
                "product_weight": "0",
                "stock_qty": 2,
            }
        ]
    )

    assert store.get_product_weight(44) == 520.0
    assert result.repaired_weight_products == 1
    assert result.repaired_weight_diagnostics[0]["source"] == "current_snapshot"


def test_active_product_store_repairs_zero_weight_from_last_valid_snapshot():
    from model_service.session.active_product_store import ActiveProductStore

    store = ActiveProductStore({"BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML": 44})
    store.set_products(
        [
            {
                "product_idx": "P44",
                "product_name": "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                "product_eng_name": "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                "yolo_class_id": 44,
                "sale_price": 1800,
                "product_weight": "520",
                "stock_qty": 2,
            }
        ]
    )
    store.clear()

    result = store.set_products(
        [
            {
                "product_idx": "P44",
                "product_name": "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                "product_eng_name": "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                "yolo_class_id": 44,
                "sale_price": 1800,
                "product_weight": "0",
                "stock_qty": 2,
            }
        ]
    )

    assert store.get_product_weight(44) == 520.0
    assert result.repaired_weight_products == 1
    assert result.repaired_weight_diagnostics[0]["source"] == "last_valid_snapshot"


def test_active_product_store_current_zero_weight_snapshot_wins_for_vision_allowlist():
    from model_service.session.active_product_store import ActiveProductStore

    store = ActiveProductStore({"PRODUCT_A": 1, "PRODUCT_B": 2})
    store.set_products(
        [
            {
                "product_idx": "P1",
                "product_name": "PRODUCT_A",
                "product_eng_name": "PRODUCT_A",
                "yolo_class_id": 1,
                "sale_price": 1000,
                "product_weight": "100",
                "stock_qty": 4,
            }
        ]
    )

    store.set_products(
        [
            {
                "product_idx": "P2",
                "product_name": "PRODUCT_B",
                "product_eng_name": "PRODUCT_B",
                "yolo_class_id": 2,
                "sale_price": 1200,
                "product_weight": "0",
                "stock_qty": 3,
            }
        ]
    )

    snapshot = store.get_effective_snapshot()

    assert snapshot.source == "current"
    assert snapshot.allowed_class_ids == [2]
    assert snapshot.products[0].product_weight == 0.0
    assert store.get_stats()["weight_unavailable_products"] == 1


def test_active_product_store_node_first_does_not_use_known_weight_fallback():
    from model_service.session.active_product_store import ActiveProductStore

    store = ActiveProductStore({"BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML": 44})

    result = store.set_products(
        [
            {
                "product_idx": "P44",
                "product_name": "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                "product_eng_name": "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                "yolo_class_id": 44,
                "sale_price": 1800,
                "product_weight": "0",
                "stock_qty": 2,
            }
        ]
    )

    assert store.get_allowed_class_ids() == [44]
    assert store.get_product_weight(44) == 0.0
    assert result.repaired_weight_products == 0
    assert store.get_stats()["weight_unavailable_products"] == 1


def test_active_product_store_repairs_class_44_zero_weight_with_known_fallback():
    from model_service.session.active_product_store import ActiveProductStore

    store = ActiveProductStore(
        {"BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML": 44},
        source_policy="static_mapping_compat",
    )

    result = store.set_products(
        [
            {
                "product_idx": "P44",
                "product_name": "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                "product_eng_name": "BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML",
                "yolo_class_id": 44,
                "sale_price": 1800,
                "product_weight": "0",
                "stock_qty": 2,
            }
        ]
    )

    assert store.get_product_weight(44) == 520.0
    assert result.repaired_weight_products == 1
    assert result.repaired_weight_diagnostics[0]["source"] == (
        "known_product_weight_fallback"
    )


def test_active_product_store_does_not_repair_unknown_zero_weight_without_history():
    from model_service.session.active_product_store import ActiveProductStore

    store = ActiveProductStore({"PRODUCT_A": 1})

    result = store.set_products(
        [
            {
                "product_idx": "P1",
                "product_name": "PRODUCT_A",
                "product_eng_name": "PRODUCT_A",
                "yolo_class_id": 1,
                "sale_price": 1000,
                "product_weight": "0",
                "stock_qty": 2,
            }
        ]
    )

    assert store.get_product_weight(1) == 0.0
    assert result.repaired_weight_products == 0
