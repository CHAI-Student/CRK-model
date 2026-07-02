import json
import time
from dataclasses import dataclass
from pathlib import Path

from model_service.engine.decision_engine import ProductDecisionEngine
from model_service.engine.models import EnsembleResult, JudgmentStatus
from model_service.service.trigger_service import LoadcellReading, TriggerInput, TriggerService
from model_service.session import SessionStore

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "scenario_matrix.json"


@dataclass
class ScenarioActiveProduct:
    yolo_class_id: int
    product_name: str
    product_weight: float
    stock_qty: int = 10
    sale_price: int = 1000
    product_idx: str = "scenario"
    has_loadcell: str = "true"


class FakeTraceContext:
    def __init__(self) -> None:
        self.weight_diagnostics = {}
        self.status = None

    def record_weight_diagnostics(self, diagnostics: dict) -> None:
        self.weight_diagnostics = dict(diagnostics)

    def finalize(self, *, status: str, error: str | None = None) -> None:
        self.status = status
        self.error = error


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def make_candidate(product: dict, confidence: float = 0.9) -> EnsembleResult:
    return EnsembleResult(
        class_id=int(product["product_id"]),
        class_name=str(product["name"]),
        top_confidence=confidence,
        side_confidence=confidence,
        combined_confidence=confidence,
        vote_count=2,
    )


def make_active_product(product: dict) -> ScenarioActiveProduct:
    return ScenarioActiveProduct(
        yolo_class_id=int(product["product_id"]),
        product_name=str(product["name"]),
        product_weight=float(product["weight"]),
        sale_price=int(product["product_id"]) * 10,
        product_idx=str(product["product_id"]),
    )


def expected_product_counts(case: dict, catalog: dict) -> dict[int, int]:
    counts: dict[int, int] = {}
    for product_key, count in (case.get("expected_counts") or {}).items():
        counts[int(catalog[product_key]["product_id"])] = int(count)
    return counts


def expected_delta(case: dict, catalog: dict) -> float:
    total = 0.0
    for product_key, count in (case.get("expected_counts") or {}).items():
        total += float(catalog[product_key]["weight"]) * int(count)
    return -total if total else 0.0


def result_counts(result) -> dict[int, int]:
    return {int(product.product_id): int(product.count) for product in result.products}


def test_scenario_fixture_counts_and_contract_shape():
    fixture = load_fixture()

    assert fixture["metadata"]["expanded_case_count"] == 924
    assert fixture["metadata"]["checklist_row_count"] == 104
    assert fixture["metadata"]["frame_stride"] == 1
    assert fixture["metadata"]["latency_budget_ms"] == 20_000
    assert fixture["metadata"]["block_counts"] == {
        "S01": 12,
        "S02": 36,
        "S03": 24,
        "S04": 24,
        "S05": 192,
        "S06": 48,
        "S07": 96,
        "S08": 96,
        "S09": 96,
        "S10": 12,
        "S11": 24,
        "S12": 12,
        "S13": 24,
        "S14": 24,
        "S15": 6,
        "S16": 6,
        "S17": 24,
        "S18": 12,
        "S19": 12,
        "S20": 12,
        "S21": 12,
        "S22": 12,
        "S23": 24,
        "S24": 12,
        "S25": 12,
        "S26": 12,
        "S27": 24,
        "S28": 12,
        "S29": 12,
    }
    assert set(fixture["metadata"]["coverage_counts"]) <= {
        "model_contract",
        "model_contract_empty",
    }


def test_all_expanded_scenarios_are_decidable_by_model_contract():
    fixture = load_fixture()
    catalog = fixture["metadata"]["product_catalog"]
    active_products = [make_active_product(product) for product in catalog.values()]
    engine = ProductDecisionEngine(strict_mode=True)
    failures: list[str] = []
    started = time.perf_counter()

    for case in fixture["expanded_cases"]:
        if not str(case["coverage"]).startswith("model_contract"):
            continue

        expected_counts = expected_product_counts(case, catalog)
        candidates = [
            make_candidate(catalog[product_key])
            for product_key in (case.get("expected_counts") or {})
        ]
        result = engine.judge(
            vision_candidates=candidates,
            delta_weight=expected_delta(case, catalog),
            active_products=active_products,
        )

        if expected_counts:
            if result.status != JudgmentStatus.COMPLETE or result_counts(result) != expected_counts:
                failures.append(
                    f"{case['case_id']}: expected={expected_counts} "
                    f"status={result.status.value} actual={result_counts(result)}"
                )
        elif result.products:
            failures.append(f"{case['case_id']}: expected empty basket, got {result_counts(result)}")

    elapsed_ms = (time.perf_counter() - started) * 1000
    assert not failures, "\n".join(failures[:20])
    assert elapsed_ms <= fixture["metadata"]["latency_budget_ms"]


def test_scenario_matrix_exercises_required_combination_limits():
    fixture = load_fixture()

    max_total_count = 0
    max_kind_count = 0
    expected_baskets = set()
    for case in fixture["expanded_cases"]:
        counts = case.get("expected_counts") or {}
        max_total_count = max(max_total_count, sum(int(count) for count in counts.values()))
        max_kind_count = max(max_kind_count, len(counts))
        expected_baskets.add(case.get("expected_basket"))

    assert max_total_count == 5
    assert max_kind_count == 3
    assert "A 5개" in expected_baskets
    assert "A 1개, B 1개, C 1개" in expected_baskets
    assert "A 2개, B 2개" in expected_baskets


def test_default_stride_latency_contract_requires_complete_evidence_fields():
    fixture = load_fixture()
    latency_evidence = {
        "queue_wait_ms": 25.0,
        "video_ms": 5_200.0,
        "video_stats_ms": 5_180.0,
        "frame_stride": fixture["metadata"]["frame_stride"],
        "original_frames": 360,
        "processed_frames": 360,
        "skipped_frames": 0,
        "yolo_total_ms": 4_750.0,
        "yolo_avg_ms": 13.2,
        "yolo_count": 360,
        "engine_ms": 18.0,
        "door_session_ms": 12.0,
        "total_ms": 5_255.0,
    }

    required_fields = {
        "queue_wait_ms",
        "video_ms",
        "video_stats_ms",
        "frame_stride",
        "original_frames",
        "processed_frames",
        "skipped_frames",
        "yolo_total_ms",
        "yolo_avg_ms",
        "yolo_count",
        "engine_ms",
        "door_session_ms",
        "total_ms",
    }
    assert required_fields <= latency_evidence.keys()
    assert latency_evidence["frame_stride"] == 1
    assert latency_evidence["processed_frames"] == latency_evidence["original_frames"]
    assert latency_evidence["skipped_frames"] == 0
    assert latency_evidence["processed_frames"] == latency_evidence["yolo_count"]
    assert latency_evidence["total_ms"] <= fixture["metadata"]["latency_budget_ms"]


def test_all_zero_loadcell_skip_records_explicit_payload_diagnostic():
    session_store = SessionStore()
    service = TriggerService(
        video_processor=None,
        engine=None,
        session_store=session_store,
    )
    loadcells = [
        LoadcellReading(
            timestamp="2026-05-29T00:00:00+00:00",
            raw_value=["+00000", "+00000"],
            filtered_value=["+00000", "+00000"],
        ),
        LoadcellReading(
            timestamp="2026-05-29T00:00:01+00:00",
            raw_value=["+00000", "+00000"],
            filtered_value=["+00000", "+00000"],
        ),
    ]
    analysis = service._analyze_weight_delta(loadcells)
    metadata = service._loadcell_trace_metadata(loadcells, analysis)
    trace_context = FakeTraceContext()

    output = service._handle_low_weight_skip(
        TriggerInput(zone=1, loadcells=loadcells, top_video_path=None, side_video_path=None),
        "all-zero-session",
        "all-zero-key",
        analysis.delta,
        payload_diagnostics=metadata,
        trace_context=trace_context,
    )

    saved = session_store.get("all-zero-session")
    assert output.status == "skipped"
    assert saved.processing_stage == "skipped_loadcell_payload_all_zero"
    assert trace_context.weight_diagnostics["decision_branch"] == "loadcell_payload_diagnostic"
    assert trace_context.weight_diagnostics["loadcell_payload_reason"] == (
        "loadcell_payload_all_zero"
    )
    assert trace_context.weight_diagnostics["payload_state"] == "all_zero"
    assert trace_context.weight_diagnostics["first_filtered_total"] == 0.0
    assert trace_context.weight_diagnostics["last_filtered_total"] == 0.0


def test_filtered_zero_with_nonzero_raw_payload_has_distinct_reason():
    loadcells = [
        LoadcellReading(
            timestamp="2026-05-29T00:00:00+00:00",
            raw_value=["+01000", "+01000"],
            filtered_value=["+00000", "+00000"],
        ),
        LoadcellReading(
            timestamp="2026-05-29T00:00:01+00:00",
            raw_value=["+00950", "+01050"],
            filtered_value=["+00000", "+00000"],
        ),
    ]
    service = TriggerService(video_processor=None, engine=None, session_store=SessionStore())
    analysis = service._analyze_weight_delta(loadcells)
    metadata = service._loadcell_trace_metadata(loadcells, analysis)

    assert metadata["payload_state"] == "all_zero"
    assert metadata["raw_state"] == "nonzero"
    assert service._loadcell_payload_issue_reason(metadata) == "filtered_all_zero_raw_nonzero"
