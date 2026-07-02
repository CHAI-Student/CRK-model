"""Verify the committed scenario fixture against model-service contracts."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "services" / "model"
if str(MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(MODEL_PATH))

from model_service.engine.decision_engine import ProductDecisionEngine  # noqa: E402
from model_service.engine.models import EnsembleResult, JudgmentStatus  # noqa: E402

DEFAULT_FIXTURE = ROOT / "services/model/tests/fixtures/scenario_matrix.json"
DEFAULT_REPORT = ROOT / "docs/scenario-readiness/scenario_verification_report.md"


@dataclass
class ScenarioActiveProduct:
    yolo_class_id: int
    product_name: str
    product_weight: float
    stock_qty: int = 10
    sale_price: int = 1000
    product_idx: str = "scenario"
    has_loadcell: str = "true"


def _candidate(product: dict[str, Any], confidence: float = 0.9) -> EnsembleResult:
    return EnsembleResult(
        class_id=int(product["product_id"]),
        class_name=str(product["name"]),
        top_confidence=confidence,
        side_confidence=confidence,
        combined_confidence=confidence,
        vote_count=2,
    )


def _active_product(product: dict[str, Any]) -> ScenarioActiveProduct:
    product_id = int(product["product_id"])
    return ScenarioActiveProduct(
        yolo_class_id=product_id,
        product_name=str(product["name"]),
        product_weight=float(product["weight"]),
        sale_price=product_id * 10,
        product_idx=str(product_id),
    )


def _expected_counts(case: dict[str, Any], catalog: dict[str, Any]) -> dict[int, int]:
    return {
        int(catalog[product_key]["product_id"]): int(count)
        for product_key, count in (case.get("expected_counts") or {}).items()
    }


def _delta(case: dict[str, Any], catalog: dict[str, Any]) -> float:
    total = 0.0
    for product_key, count in (case.get("expected_counts") or {}).items():
        total += float(catalog[product_key]["weight"]) * int(count)
    return -total if total else 0.0


def _result_counts(result: Any) -> dict[int, int]:
    return {int(product.product_id): int(product.count) for product in result.products}


def verify_scenarios(fixture: dict[str, Any]) -> dict[str, Any]:
    catalog = fixture["metadata"]["product_catalog"]
    active_products = [_active_product(product) for product in catalog.values()]
    engine = ProductDecisionEngine(strict_mode=True)
    failures: list[str] = []
    verified = 0
    empty_verified = 0
    started = time.perf_counter()

    for case in fixture["expanded_cases"]:
        if not str(case["coverage"]).startswith("model_contract"):
            continue
        verified += 1
        expected_counts = _expected_counts(case, catalog)
        if not expected_counts:
            empty_verified += 1
        candidates = [
            _candidate(catalog[product_key])
            for product_key in (case.get("expected_counts") or {})
        ]
        result = engine.judge(
            vision_candidates=candidates,
            delta_weight=_delta(case, catalog),
            active_products=active_products,
        )
        actual = _result_counts(result)
        if expected_counts:
            if result.status != JudgmentStatus.COMPLETE or actual != expected_counts:
                failures.append(
                    f"{case['case_id']}: expected={expected_counts} "
                    f"status={result.status.value} actual={actual}"
                )
        elif actual:
            failures.append(f"{case['case_id']}: expected empty basket, got {actual}")

    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "verified_cases": verified,
        "empty_cases": empty_verified,
        "failures": failures,
        "elapsed_ms": round(elapsed_ms, 2),
        "latency_budget_ms": int(fixture["metadata"]["latency_budget_ms"]),
        "within_latency_budget": elapsed_ms <= int(fixture["metadata"]["latency_budget_ms"]),
    }


def summarize_trace_latency(
    trace_paths: list[Path],
    budget_ms: int,
    expected_frame_stride: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in trace_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stats = payload.get("video_stats") or {}
        if "processing_time_ms" not in stats:
            continue
        rows.append(
            {
                "file": path.name,
                "status": payload.get("status"),
                "frame_stride": stats.get("frame_stride"),
                "processing_time_ms": float(stats.get("processing_time_ms", 0.0)),
                "original_frames": stats.get("original_frames"),
                "processed_frames": stats.get(
                    "processed_frames",
                    stats.get("total_frames"),
                ),
                "skipped_frames": stats.get("skipped_frames"),
                "yolo_count": stats.get("yolo_inference_count"),
            }
        )

    matching_stride_rows = [
        row for row in rows if row["frame_stride"] == expected_frame_stride
    ]
    over_budget = [
        row for row in matching_stride_rows if row["processing_time_ms"] > budget_ms
    ]
    return {
        "trace_count": len(rows),
        "expected_frame_stride": expected_frame_stride,
        "matching_stride_trace_count": len(matching_stride_rows),
        "matching_stride_max_video_ms": (
            max(
                (row["processing_time_ms"] for row in matching_stride_rows),
                default=0.0,
            )
        ),
        "matching_stride_over_budget": over_budget,
        "rows": rows,
    }


def write_report(
    *,
    fixture: dict[str, Any],
    scenario_result: dict[str, Any],
    trace_result: dict[str, Any],
    output_path: Path,
) -> None:
    metadata = fixture["metadata"]
    lines = [
        "# Scenario Verification Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scenario Contract",
        "",
        f"- Expanded cases in fixture: {metadata['expanded_case_count']}",
        f"- Verified model-contract cases: {scenario_result['verified_cases']}",
        f"- Empty-basket cases: {scenario_result['empty_cases']}",
        f"- Failures: {len(scenario_result['failures'])}",
        f"- Engine decision elapsed: {scenario_result['elapsed_ms']} ms",
        f"- Decision-loop budget: {scenario_result['latency_budget_ms']} ms",
        f"- Within budget: {scenario_result['within_latency_budget']}",
        "",
        "## Trace Latency Evidence",
        "",
        f"- Trace files with video stats: {trace_result['trace_count']}",
        f"- Expected frame stride: {trace_result['expected_frame_stride']}",
        f"- Matching-stride trace files: {trace_result['matching_stride_trace_count']}",
        f"- Max matching-stride video processing time: "
        f"{trace_result['matching_stride_max_video_ms']:.1f} ms",
        f"- Matching-stride traces over 20s video budget: "
        f"{len(trace_result['matching_stride_over_budget'])}",
        "",
        "Trace JSON files do not include queue wait or total trigger-loop fields. Use",
        "`[TRIGGER-WORKER][LATENCY]` logs for full Jetson loop acceptance.",
        "",
    ]
    if scenario_result["failures"]:
        lines.extend(["## Failures", ""])
        lines.extend(f"- {failure}" for failure in scenario_result["failures"][:50])
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE, type=Path)
    parser.add_argument("--report", default=DEFAULT_REPORT, type=Path)
    parser.add_argument("--trace-glob", default="*.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.disable(logging.WARNING)
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    scenario_result = verify_scenarios(fixture)
    trace_result = summarize_trace_latency(
        list(ROOT.glob(args.trace_glob)),
        int(fixture["metadata"]["latency_budget_ms"]),
        int(fixture["metadata"]["frame_stride"]),
    )
    write_report(
        fixture=fixture,
        scenario_result=scenario_result,
        trace_result=trace_result,
        output_path=args.report,
    )
    print(
        f"verified={scenario_result['verified_cases']} "
        f"failures={len(scenario_result['failures'])} "
        f"elapsed_ms={scenario_result['elapsed_ms']} "
        f"matching_stride_traces={trace_result['matching_stride_trace_count']}"
    )
    if scenario_result["failures"] or not scenario_result["within_latency_budget"]:
        raise SystemExit(1)
    if trace_result["matching_stride_over_budget"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
