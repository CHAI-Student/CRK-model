from pathlib import Path
from types import SimpleNamespace


def test_validate_model_class_mapping_reports_dataset_mapping_mismatches(tmp_path):
    from model_service.core.model_validation import validate_model_class_mapping

    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(
        "\n".join(
            [
                "names:",
                "  0: hand",
                "  1: PRODUCT_A",
                "  2: PRODUCT_B",
            ]
        ),
        encoding="utf-8",
    )
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        """
{
  "mappings": [
    {"yolo_class_id": 1, "yolo_class_name": "PRODUCT_A"},
    {"yolo_class_id": 2, "yolo_class_name": "PRODUCT_C"}
  ]
}
""".strip(),
        encoding="utf-8",
    )

    result = validate_model_class_mapping(
        engine_class_names={0: "hand", 1: "PRODUCT_A", 2: "PRODUCT_B"},
        dataset_path=dataset,
        mapping_path=mapping,
    )

    assert result.ok is False
    assert result.engine_class_count == 3
    assert result.dataset_class_count == 3
    assert result.mapping_class_count == 2
    assert result.mismatches == [
        {
            "class_id": 2,
            "dataset": "PRODUCT_B",
            "mapping": "PRODUCT_C",
            "engine": "PRODUCT_B",
        }
    ]


def test_validate_model_class_mapping_passes_for_current_repo_mapping():
    from model_service.core.model_validation import validate_model_class_mapping

    dataset_path = Path("dataset.yaml")
    mapping_path = Path("services/config/yolo_product_mapping.json")
    engine_names = {
        0: "hand",
        96: "BAG_JAYEONLU_MOIST_SWEET_CHESTNUT_80G",
    }

    result = validate_model_class_mapping(
        engine_class_names=engine_names,
        dataset_path=dataset_path,
        mapping_path=mapping_path,
    )

    assert result.mismatches == []


def test_static_catalog_validation_is_disabled_by_default(tmp_path):
    from model_service.api.manager import maybe_validate_static_catalog
    from model_service.core.config import Settings

    calls = []

    def fake_validate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(ok=True)

    result = maybe_validate_static_catalog(
        settings=Settings(catalog={"static_validation_enabled": False}),
        engine_class_names={0: "hand", 1: "PRODUCT_A"},
        base_dir=tmp_path,
        validate_fn=fake_validate,
    )

    assert result is None
    assert calls == []


def test_static_catalog_validation_runs_when_enabled(tmp_path):
    from model_service.api.manager import maybe_validate_static_catalog
    from model_service.core.config import Settings

    calls = []
    validation = SimpleNamespace(ok=True)

    def fake_validate(**kwargs):
        calls.append(kwargs)
        return validation

    result = maybe_validate_static_catalog(
        settings=Settings(catalog={"static_validation_enabled": True}),
        engine_class_names={0: "hand", 1: "PRODUCT_A"},
        base_dir=tmp_path,
        validate_fn=fake_validate,
    )

    assert result is validation
    assert calls == [
        {
            "engine_class_names": {0: "hand", 1: "PRODUCT_A"},
            "dataset_path": tmp_path.parent / "dataset.yaml",
            "mapping_path": tmp_path / "config" / "yolo_product_mapping.json",
        }
    ]
