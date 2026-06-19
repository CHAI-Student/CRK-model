# Scenario Readiness And 0g Diagnostics

Status: current scenario contract and payment-path 0g investigation

## Current Thesis

The Excel scenario matrix is now represented by a repo-local JSON fixture so
local tests can prove model-service decision coverage without depending on the
operator download directory. This is still a contract/unit proof, not Jetson
TensorRT runtime proof.

## Scenario Fixture

- Source refresh script:
  [scripts/refresh_scenario_fixture.py](../../../scripts/refresh_scenario_fixture.py).
- Committed fixture:
  [services/model/tests/fixtures/scenario_matrix.json](../../../services/model/tests/fixtures/scenario_matrix.json).
- Human report:
  [docs/scenario-readiness/scenario_fixture_report.md](../../../docs/scenario-readiness/scenario_fixture_report.md).
- Verification script/report:
  [scripts/verify_scenario_readiness.py](../../../scripts/verify_scenario_readiness.py)
  and
  [docs/scenario-readiness/scenario_verification_report.md](../../../docs/scenario-readiness/scenario_verification_report.md).
- Fixture scope: 924 expanded scenario rows and 104 checklist rows.
- Test scope: model-service basket judgment for S01-S29, using synthetic A/B/C
  active products with non-ambiguous weights.
- External scope: product registration, card/payment/service-only checks remain
  cross-service/manual contracts.

## Model Coverage Changes

- Strict matching now distinguishes total unit limit from product-kind limit.
- Default strict search covers up to five total units and three distinct kinds.
- High-confidence A+B+C style three-kind combinations are allowed.
- Low-confidence multi-kind combinations remain rejected.
- Repeated same-product counts up to eight are allowed with a dedicated 5g/item
  tolerance only when they do not mask a plausible near single-product
  explanation or competing high-confidence multi-product evidence.

## Latency Contract

- Scenario fixture metadata records `frame_stride=2` and a 20,000 ms latency
  budget.
- `scenario_verification_report.md` currently records all 924 model-contract
  cases passing in 60.75 ms of engine decision time, plus 4 available stride-2
  trace JSONs with max video processing time 11,075.6 ms.
- Contract tests require latency evidence fields used by
  `[TRIGGER-WORKER][LATENCY]`: queue wait, video time, video stats time,
  frame stride, original/processed/skipped frames, YOLO total/average/count,
  engine time, door-session time, and total time.
- Local tests assert the contract shape. Real timing acceptance still requires
  Jetson traces with TensorRT engines and actual AVI payloads.

## 0g Diagnostic Branch

- Loadcell summaries already classify payloads as `empty_payload`,
  `invalid_only`, `all_zero`, or `nonzero`.
- Low-weight skips now copy that payload classification into weight diagnostics
  and use a diagnostic decision branch when the payload itself is suspect.
- A filtered-all-zero payload with nonzero raw channels gets the distinct
  reason `filtered_all_zero_raw_nonzero`.
- Code review of sibling services indicates the payment path may open the door
  before Camera `/recording/start`, which can leave model-service with missing
  or all-zero loadcell history. Model-service cannot reconstruct missing
  pre-trigger samples; it now reports that evidence explicitly.

## Related Wiki Pages

- [Decision and weight](../source-code/decision-and-weight.md)
- [Loadcell and trigger](../source-code/loadcell-and-trigger.md)
- [Latency and frame stride](latency-and-frame-stride.md)
- [System map](system-map.md)
