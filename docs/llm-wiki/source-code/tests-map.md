# Source Code Map: Tests

Source: [services/model/tests](../../../services/model/tests)

Status: current test map

## Current Thesis

Local tests are the main safe regression gate for this repo, but they do not
prove Jetson runtime readiness. Use focused suites when changing a subsystem,
then run `pytest services/model/tests -q` for docs/code delivery when feasible.

## Test Bootstrap

- [conftest.py](../../../services/model/tests/conftest.py): fixtures for
  sessions, loadcells, door sessions, product aggregation, and global container
  reset.
- [README.md](../../../services/model/tests/README.md): current test command
  notes and local verification history.

## Test Files By Behavior

- [test_active_product_store_mapping.py](../../../services/model/tests/test_active_product_store_mapping.py):
  YOLO class id/name mapping, mapping diagnostics, and active snapshot
  preservation on invalid overwrite, including TTL-bounded last-valid fallback.
- [test_fastapi_imports.py](../../../services/model/tests/test_fastapi_imports.py):
  import smoke, lazy exports, runtime settings, YOLO load failure surfacing,
  and best-effort `/api/health/detailed` runtime diagnostics.
- [test_logging_config.py](../../../services/model/tests/test_logging_config.py):
  logging setup and console filter behavior.
- [test_model_validation.py](../../../services/model/tests/test_model_validation.py):
  engine/dataset/mapping validation.
- [test_runtime_env.py](../../../services/model/tests/test_runtime_env.py):
  Jetson runtime path bootstrap and re-exec behavior.
- [test_setup_templates.py](../../../services/model/tests/test_setup_templates.py):
  `.env.example` freezer-template parsing, shell script syntax checks when
  Bash is available, and dry-style user launcher installation behavior.
- [test_terminal_restore.py](../../../services/model/tests/test_terminal_restore.py):
  terminal capture/restore around interruption.
- [test_trigger_helpers.py](../../../services/model/tests/test_trigger_helpers.py):
  loadcell parsing, delta helpers, stable regions, vote conversion.
- [test_loadcell_compound_segments.py](../../../services/model/tests/test_loadcell_compound_segments.py):
  compound loadcell segment detection, stable history decision deltas,
  removal and return segment target extraction, mixed return-hint construction,
  compatibility `/trigger` return-hint parity, pressure-like vision-required
  targets, pressure-like pair suppression, simultaneous physical-channel
  removal target extraction, rapid same-zone trace context, and freezer
  endpoint fallback trace metadata.
- [test_scenario_matrix_contract.py](../../../services/model/tests/test_scenario_matrix_contract.py):
  Excel-derived scenario fixture counts, all 924 expanded model-contract
  basket judgments, stride-2 latency evidence shape, and explicit 0g payload
  diagnostic branches.
- [test_trigger_pipeline_regressions.py](../../../services/model/tests/test_trigger_pipeline_regressions.py):
  async frame reads, zero-frame retry, ffprobe polling, loadcell channel
  behavior, stable-tail-only chargeable delta, raw transient diagnostics, and
  simple-fallback non-chargeable regressions, including freezer endpoint
  fallback acceptance/rejection boundaries.
- [test_frame_trace.py](../../../services/model/tests/test_frame_trace.py):
  trace export, samples, allowed class filtering, rescue diagnostics, active
  product diagnostics, trigger worker trace behavior, and last-valid active
  snapshot recovery after close cleanup. It also covers loadcell-first
  return-only handling, balanced queued-removal cancellation, regular
  motion-filter fail-closed behavior, low-weight video diagnostic-only traces,
  unstable removal waiting before queue/video/engine, and runtime vision config
  fields, including freezer camera layout, handled-filter enablement, and
  freezer threshold snapshots.
  Recent freezer coverage verifies that zone-sliced `loadcells` remain the
  effective payload, even when the deprecated compatibility field is present,
  and traces retain `loadcell_scope=zone`.
  It also verifies that propagated video processing failures mark worker
  sessions and traces as `error` instead of calling the decision engine.
- [test_video_processor_thresholds.py](../../../services/model/tests/test_video_processor_thresholds.py):
  thresholds, top-k, frame stride, 480-left-crop-aligned ROI defaults, top/side
  ROI filtering, side ROI hard `center_x <= 400` plus `+5px` soft-band Pepsi
  promotion, far-right Trevi filtering, Bibigo/Pepero/Letsbe candidate
  promotion regressions, stage counts, diagnostic trace, threshold/ROI/no-motion
  rescue, and side-ROI-conflicted threshold rescue rejection.
  Freezer tests cover dual-top upper-half ROI filtering, separation of
  `freezer_roi_passed` from rejected `freezer_roi_filtered` evidence, freezer
  rescue suppression, same-frame multi-bbox `instance_count_hint`,
  disabled-layout warnings, `15g` weight-fit multi selection, top-three
  mismatch narrowing, same-product repeat recovery for the zone5 bagel `x2`
  collision, repeat rejection when freezer exit-path evidence is too weak,
  dual-camera single preference over top-only repeats, static top-only ranking
  demotion, upper-ROI hand proximity, hand-path hard reject with alternative
  candidates, hand-path all-blocked fail-open behavior, and freezer filter OPS
  visibility.
  Async failure tests cover model-service exception re-raise, unknown task
  wrapping, missing async extractor support, frame queue timeout, and zero-frame
  failure after retry.
- [test_yolo_wrapper_geometry.py](../../../services/model/tests/test_yolo_wrapper_geometry.py):
  default left-crop and optional crop/letterbox geometry.
- [test_voting_ensemble.py](../../../services/model/tests/test_voting_ensemble.py):
  Top/Side voting weights and ordering.
- [test_decision_engine.py](../../../services/model/tests/test_decision_engine.py):
  strict mismatch, loadcell-only, rescue candidates, detected-single fallback,
  strict single-candidate source/rank priority, same-product repeated counts
  through x8, returned-weight hint ranking, and segment-first loadcell matching,
  including candidate-supported repeated-count priority over evidence-free
  active-only segment fits and aggregate-evidence recovery for collision-like
  segment fragmentation. It also covers clean supported segment priority over
  aggregate repeats for Binch + Haruyache + Letsbe versus Chapagetti x4,
  candidate-first same-product repeated recovery before stage-count expansion,
  preserving stage-count fallback when the candidate repeat is outside
  tolerance, Trevi plus King Rush evidence-aware segment recovery over Bibigo
  aggregate noise, camera-aware stage scoring for Welchs/Cupban segment and
  stage-count expansion cases, same-weight regular-candidate identity recovery
  for Pepsi/Kwangdong/Sky collisions, strict single-candidate priority for
  Hatban over lower-rank HOT6 residual differences, plus forced final fallback
  behavior that avoids `none` when active products exist while preferring supported
  same-product repeats over weak mixed active-product pairs. It also covers
  weak Pepero/Binch segment traces being treated as unsupported so active
  Pepsi/Sky Barley x2 selections are not replaced by small-item repeated
  counts, plus regular Pepsi selection over Trevi threshold-rescue residual
  collisions. It also covers merged bottle segment splitting, where a compound
  Fanta + Pepsi segment and a Baksansoo segment beat Chapagetti/Pepero
  small-item repeats, and segment grip limits where one segment rejects Pepero
  x8, two segments allow Pepero x6, and aggregate overrides cannot exceed the
  segment-derived total cap. Recent Pepsi coverage includes stage-weight-gate
  candidate promotion over same-weight threshold rescue, regular Pepsi
  candidate priority over same-weight Corn stage-gate evidence, plus same-weight
  500ml bottle segment collision handling that rejects top-only Trevi reuse
  across separated Pepsi-weight segments. It also covers simultaneous channel split
  matching where Tteokbokki + Welchs beats a Kwangdong aggregate rescue, while
  active-only channel splits fall back to aggregate strict matching. Recent
  coverage also proves ranked regular candidates beat same-weight unseen
  active-only repeats, and no-final-candidate stage-count combinations run
  before active-only forced fallback, including direct threshold/ROI rescue and
  diagnostic detection evidence. Full-delta regressions cover the
  Haluyache/Letsbe/Jagabee `-503g` segmented removal, rejection of
  `last_unpaired_negative_segment` as a standalone forced fallback target, and
  no-charge `UNCERTAIN` results when a successful branch does not explain the
  full stable removal delta.
  Freezer decision tests cover confidence-first single selection outside the
  normal weight tolerance, weight residual tie-breaks only inside the
  confidence band, candidate-only identity, freezer `15g` weight-gated
  multi-kind evidence, rejection of mismatched top-three freezer candidates,
  same-class counts gated by `instance_count_hint`, and count-aware
  same-product repeat inference for the zone5 bagel `x2` case. They also cover
  direct freezer static/trajectory interaction ranking so the engine and video
  handled-filter path stay aligned.
- [test_session_store_lifecycle.py](../../../services/model/tests/test_session_store_lifecycle.py):
  normal `processing -> complete/waiting/error` session saves do not emit
  `Session overwritten`, while replacing an already completed session still
  warns.
- [test_strict_weight_matcher.py](../../../services/model/tests/test_strict_weight_matcher.py):
  strict combinations, candidate extraction, motion-aware ambiguous
  combination ranking, and return combination search.
- [test_product_aggregator.py](../../../services/model/tests/test_product_aggregator.py):
  aggregation, returns, net-delta recovery, batch returns, chronological
  trigger replay, relaxed same-zone single return tolerance, mixed
  return/removal hint replay, return-hint count reduction, close-time
  unresolved final-weight mismatch exclusion, and complex scenarios.
- [test_cross_zone_return.py](../../../services/model/tests/test_cross_zone_return.py):
  cross-zone return repair, combination recovery scenarios, effective
  net-delta order, and edge cases.
- [test_multi_zone_summary.py](../../../services/model/tests/test_multi_zone_summary.py):
  close response decision summary, close latency defaults, and multi-zone active
  product snapshot guards, including effective cross-zone `weightDelta` output
  for single and combination returns. It also covers mixed return hints in
  effective `weightDelta` output, unmatched mixed-return hints being excluded
  from empty-basket CLOSE deltas, and non-chargeable pending trigger diagnostics
  for fast close finalization. Close final-weight validation also covers
  repeated-candidate correction followed by matched-only unresolved mismatch
  exclusion when no bounded correction can explain the net removal, plus
  freezer deferred repair from later unused candidate snapshots and rejection
  of consumed or out-of-tolerance later candidates.

## Which Tests To Run

- Static cleanup: `uv run --no-sync ruff check services/model scripts`.
  For a narrower safety gate while triaging old lint noise, require
  `uv run --no-sync ruff check services/model scripts --select F` to pass.
- Startup/API change: `test_fastapi_imports.py`,
  `test_multi_zone_summary.py`, `test_runtime_env.py`, and relevant route or
  trigger helper tests.
- Trigger/loadcell change: `test_trigger_helpers.py`,
  `test_loadcell_compound_segments.py`,
  `test_trigger_pipeline_regressions.py`, `test_frame_trace.py`,
  `test_active_product_store_mapping.py`.
- Video/vision change: `test_video_processor_thresholds.py`,
  `test_yolo_wrapper_geometry.py`, `test_voting_ensemble.py`.
- Async failure propagation change: `test_video_processor_thresholds.py` plus
  the worker error regression in `test_frame_trace.py`.
- Decision/weight change: `test_decision_engine.py`,
  `test_strict_weight_matcher.py`, `test_scenario_matrix_contract.py`.
- Freezer policy change: `test_frame_trace.py`,
  `test_video_processor_thresholds.py`, and `test_decision_engine.py`.
- Session/return change: `test_product_aggregator.py`,
  `test_cross_zone_return.py`, `test_multi_zone_summary.py`.
- Jetson bootstrap change: `test_runtime_env.py`.
- Health diagnostics change: `test_fastapi_imports.py`.

## Related Wiki Pages

- [Jetson and testing](../synthesis/jetson-and-testing.md)
- [File inventory](file-inventory.md)
- [Scenario readiness and 0g diagnostics](../synthesis/scenario-readiness-and-0g.md)
