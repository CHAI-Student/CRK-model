# Source Code Map: Observability And Traces

Source: [core/logging_config.py](../../../services/model/model_service/core/logging_config.py),
[video/frame_trace.py](../../../services/model/model_service/video/frame_trace.py),
[service/trigger_service.py](../../../services/model/model_service/service/trigger_service.py),
[video/frame_extractor.py](../../../services/model/model_service/video/frame_extractor.py),
[video/video_processor.py](../../../services/model/model_service/video/video_processor.py),
[engine/decision_engine.py](../../../services/model/model_service/engine/decision_engine.py)

Status: current observability map

## Current Thesis

The service is designed to explain field failures from logs and per-trigger
trace files. Do not rely on final status alone; compare OPS logs, latency logs,
trace sections, active product snapshots, and weight diagnostics.

## Logging

- `setup_logging()` configures structured logging and console filtering.
- `get_ops_logger()` emits operator-focused `[OPS]` events.
- Correlation id helpers exist for request-level context.
- `PerformanceLogger` is available for timed sections.

## OPS Events

Important patterns include:

- `[OPS][TRIGGER]`: zone, delta, loadcell payload state, filtered channel
  counts, first/last filtered totals, analysis reason, and top/side video
  paths. Cabinet runs also log `cabinet_type`, `loadcell_scope=zone`,
  `loadcell_source=loadcells`, `requested_zone`, and effective channel count.
- `[OPS][FRAMES]`: zone and video paths for frame processing.
- `[VIDEO-ASYNC]` fatal extractor, frame queue, and YOLO task errors propagate
  to the trigger worker and should end with trace/session `status=error`, not
  `status=complete` with empty candidates.
- `[OPS][CANDIDATES]`: ranked pre-engine handled candidates, weights,
  confidence, camera flags, source, `count_hint`, and `freezer_exit_votes`.
  These are not final chargeable counts.
- `[OPS][FREEZER-CANDIDATE-FILTER]`: freezer handled-filter enablement,
  raw/handled counts, and the filter reason. Normal enabled freezer removal
  paths should show `reason=vision_identity_passthrough`; selected counts and
  expected weights are engine-result diagnostics, not pre-engine filter output.
- `[OPS][RESULT]`: final engine status, products, `product_count`, and total
  price.
- `[OPS][FREEZER-CLOSE-AGGREGATE]`: freezer-only CLOSE aggregate resolver
  outcome. It records whether aggregate solving was accepted, the reason,
  `policy=signed_net_delta`, output zone, `global_net_delta`, final target,
  selected weight, residual, and selected product counts.
- `[OPS][CLOSE]`: close summary across zones.
- `[OPS][CLOSE_DIAGNOSTIC]`: no-charge skipped-trigger diagnostics by zone,
  such as `diagnostic=loadcell_payload_all_zero`, emitted without adding any
  products or price to the close summary.

## Latency Events

- `[TRIGGER-WORKER][LATENCY]`: session id, zone, queue wait, video, engine,
  door session, total time, YOLO total/count, `frame_stride`,
  `original_frames`, `processed_frames`, `skipped_frames`.
- `[VIDEO-ASYNC][LATENCY]`: video processing time and frame-stride counters.
- `[VIDEO][LATENCY]`: sync video processing stats.
- `[CLOSE][LATENCY]`: pending trigger count and close debounce elapsed time.
  Pending logs include in-flight zones and session ids when CLOSE is blocked
  by trigger work that has not fully finalized.

## Trace Files

`TriggerTraceContext` collects trace metadata by default. Frame image export is
opt-in: `.env.example` and the Jetson templates keep
`MODEL__TRACE__SAMPLE_EXPORT_ENABLED=false` so copied runtime env files do not
start writing sample images during inference.

`TriggerTraceContext` collects:

- camera video paths, total/processed frames, optional sample files
- loadcell delta, stable-region metadata, and payload-shape diagnostics such as
  `payload_state`, raw/filtered channel counts, invalid/zero/nonzero channel
  counts, and first/last raw or filtered zone totals
- loadcell compound movement metadata:
  `compound_segments`, `compound_positive_weights_g`,
  `compound_negative_weights_g`, and `compound_event`
- stable history and decision metadata:
  `stable_plateaus`, `purchase_delta_candidates`, `decision_delta_weight`,
  `net_delta_weight`, `removal_segment_targets`,
  `channel_removal_segment_targets`, `channel_delta_diagnostics`,
  `return_segment_targets`, `vision_required_segment_targets`,
  `paired_loadcell_movements`, `ignored_loadcell_movements`,
  `mixed_sign_net_masking_guard`, and `pressure_like_event`
- recent same-zone loadcell context:
  `recent_same_zone_window_seconds`, `recent_same_zone_events`, and
  `recent_return_weights_g`
- video stats
- candidates and product weights
- preprocessing metadata
- stage counts by class, including ROI-filtered counts and ROI center/limit
  metadata such as `roi_y_limit=240` and `roi_direction` for top-camera
  removal/return filtering. Freezer dual-top filtering records
  `freezer_roi_passed` for handled-exit evidence and `freezer_roi_filtered`
  for rejected ROI evidence. `freezerExitPathVotes` increments only from
  passed ROI evidence or explicit legacy fields; rejected ROI counts are kept
  as `freezerRoiFilteredVotes`. Freezer traces also expose motion,
  trajectory, upper-ROI hand proximity fields, `instance_count_hint` for
  multi-bbox same-class evidence, and `orderedCombinationSearch` diagnostics
  when the decision engine evaluates same-product counts or mixed candidate
  combinations.
- runtime vision config such as `yolo_model_path`,
  `yolo_internal_conf_threshold`, hand class/confidence settings, and the
  top/side `regular_threshold`
- extractor diagnostics
- diagnostic detections
- threshold and ROI rescue candidates
- active product snapshot
- active product diagnostics: active count, allowed class count, stock/weight
  counts, zero-stock/zero-weight counts, store stats, snapshot source, last
  valid fallback age, and fail-closed reason
- weight diagnostics, including `trigger_relevance` for `return_loadcell_only`,
  `balanced_out`, and `cancelled_by_return` paths, plus
  `mixed_return_segments` when a non-freezer/default negative trigger carries
  internal return hints for DoorSession replay. In current freezer mode,
  mixed-sign positive segments are diagnostics-only: traces still expose
  compound positive/negative segment weights and `TriggerResult` stores compact
  `loadcell_diagnostics`, but `mixed_sign_net_masking_guard` is no longer
  accepted in the default freezer path. `effective_count_guard` explains when
  return hints can reduce a raw repeated-count result, and
  `same_weight_candidate_collision` explains when regular candidate identity
  beats a same-weight active/rescue collision.
  Freezer decisions add `decision_branch=freezer_vision_first` with
  `weight_used_as=combination_validation` or `diagnostic`,
  `weight_reliable`, `weight_residual`, `orderedCombinationSearch`, and
  selected/considered candidate diagnostics. Valid positive-weight freezer
  mismatches return no-charge `UNCERTAIN` with
  `reason=no_weight_fit_for_vision_candidate_pool` when no candidate-pool
  combination explains the stable removal delta inside freezer tolerance.
- final result and storage result

Trace JSONL/detail files are raw operational evidence. The current wiki policy
is to describe the trace schema and not ingest untracked trace JSON files unless
the user explicitly asks.

The application writes `services/model/logs/frame_split_*.jsonl`; deployed
Jetsons should bound those files with host logrotate or an equivalent OS-level
rotation policy.

## Debugging Heuristic

- If ffprobe reports frames but decoded frames are zero, inspect extractor
  diagnostics and retry branch. After the async failure-propagation fix, this
  should become a video processing error instead of a normal no-detection
  result when frames were expected and all decode retries still produced zero.
- If a trace/session is `status=error` with a `VideoProcessingError`, inspect
  `[VIDEO-ASYNC]` task names such as `top-extractor`, `side-extractor`, and
  `yolo-inference` before treating the event as a product absence.
- If `queue_wait_ms` is high, the serial worker is the bottleneck.
- If `video_ms` is high and `yolo_count` is high, frame stride may help.
- If model result is ready but close is late, inspect Node polling and CLOSE
  cadence separately.
- If a trace has `processing_stage=loadcell_return_only` or
  `weight_diagnostics.trigger_relevance.skip_reason=return_loadcell_only`, the
  service intentionally skipped YOLO for a positive return delta.
- If a trace has `processing_stage=return_waiting_for_stable_loadcell` or
  `waiting_for=stable_loadcell`, the return delta was positive but still came
  from a short/unstable loadcell payload after the stabilization wait. Inspect
  `weight_diagnostics.return_stabilization` for initial/refreshed delta,
  stable-region validity, and whether simple fallback was rejected.
- If `processing_stage=skipped_balanced` or `skip_reason=cancelled_by_return`,
  a later return balanced the queued removal before video started; this is a
  latency optimization, not a no-detection miss.
- If a non-freezer/default negative trigger has `compound_positive_weights_g`
  and `return_segment_targets`, inspect `weight_diagnostics.mixed_return_segments`.
  `accepted=true` means the positive segment was attached as
  `return_weight_hints` and will be deferred to CLOSE reconciliation.
- In freezer mode, mixed-sign internal positive segments should not produce
  `return_weight_hints` or an accepted `mixed_sign_net_masking_guard`. Compare
  `net_delta_weight`, `decision_delta_weight`, `compound_positive_weights_g`,
  and `compound_negative_weights_g`; for freezer they should show the stable
  net decision, such as `+47.6g/-295.3g` becoming about `-247.7g`, while the
  segment weights remain diagnostic evidence for CLOSE aggregate eligibility.
- In freezer mode, inspect
  `final_weight_validation.freezerCloseAggregate` and
  `[OPS][FREEZER-CLOSE-AGGREGATE]` for unstable door sessions.
  `policy=signed_net_delta` and `globalNetDelta`/`global_net_delta` identify
  the signed target. `outputZone` identifies the zone that will receive the
  whole final basket, `role=rerouted` zones should have
  `weightDeltaOverride=0.0`, and the output zone should have an override equal
  to the signed global net. A near-zero global net clears participant products;
  a negative net with no fitting candidate combination records a no-charge
  reason instead of preserving a mismatched provisional basket.
- In freezer mode, a log sequence with `[OPS][FREEZER-CANDIDATE-FILTER] ...
  reason=vision_identity_passthrough`, `[OPS][CANDIDATES] ... count_hint=1`,
  and `[OPS][RESULT] ... product_count=2` is expected when the ordered solver
  accepts a same-product repeat. For the bagel field shape
  `delta=-309.5g`, `BAG_NULLDAM_BAGEL_140G`, `unit_weight=156g`, and
  `confidence=0.528`, inspect
  `weight_diagnostics.freezer_vision_first.selected[*].count=2` and
  `combinationResidual~=2.5g`. If the final result stays empty with
  `reason=no_weight_fit_for_vision_candidate_pool`, ordered candidate-pool
  fitting failed and the trigger was intentionally no-charge.
- In freezer mode, if a lower-rank product has a smaller residual but a higher
  rank product already fits tolerance, the higher-rank product should win. The
  cheese burger field shape is `target=183.7g`, rank-1 cheese burger `176g`,
  and lower-rank dumpling `189g`; `orderedCombinationSearch.attempts[0]`
  should be the cheese burger `x1` attempt and the selected product should be
  cheese burger.
- If `delta_weight=0.0g` skipped inference, inspect `loadcell.payload_state`
  before blaming vision. `empty_payload` means Camera sent no loadcell samples,
  `invalid_only` means filtered values did not parse, `all_zero` means the
  filtered zone channels were literally zero, and `nonzero` means the payload
  had usable values but the computed delta was still within the low-weight
  threshold.
- If CLOSE still prints `zones=none`, inspect optional
  `decisionSummary.diagnosticZoneLines`,
  `decisionSummary.zones[*].noChargeDiagnostics`, and
  `[OPS][CLOSE_DIAGNOSTIC]`. These fields explain no-charge skipped loadcell
  payloads without changing payment totals.
- If raw detections are zero, first check `active_product_diagnostics`. An empty
  allowlist with `empty_allowlist_fail_closed` means the active inventory
  snapshot blocked inference before vision had a chance to detect products.
- If `final_result.status=removal_waiting_for_stable_loadcell`, also inspect
  `active_product_diagnostics` and
  `weight_diagnostics.active_product_failure_reason`. The trigger may be
  waiting for a stable loadcell tail while also lacking Node active-product
  context; the model keeps both causes visible and still excludes the trigger
  from payment.
- If a top-camera class is missing after raw detection, inspect
  `stage_counts_by_class[*].roi_filtered`, `roi_y_avg`, `roi_y_limit`, and
  `roi_direction` before treating it as a model recall miss.
- If a side-camera product appears as `source=roi_rescue`, inspect
  `roi_x_avg`, `roi_x_limit`, and `side_motion_passed`. Current ROI rescue
  rejects static evidence and candidates whose average center is to the right
  of the side ROI limit.
- If a side-camera class is present in raw detections but missing from regular
  candidates, inspect `side_roi_soft_passed`, `soft_margin_filtered`,
  `roi_filtered`, and `motion_filtered`. Pepsi-like boundary evidence around
  `x=402..404` should appear as `side_roi_soft_passed` before motion filtering.
- `vision_config` records `side_roi_soft_margin_px`, the computed
  `side_roi_soft_x_max`, FFmpeg gamma/contrast values, and the async frame
  stride so field traces can separate ROI/candidate issues from decode or YOLO
  latency changes.
- If a product appears as `source=threshold_rescue`, inspect `roi_conflict` and
  `threshold_rescue_rejected_reason`. A weak low-confidence rescue should not be
  accepted when stronger same-class side evidence was filtered outside the ROI.
- A trace with `status=complete` and expected frame counters can still produce
  no products when `active_product_diagnostics.inference_fail_closed_reason` is
  `missing_active_product_snapshot_fail_closed`. Treat that as missing inventory
  context, not as proof that CLOSE interrupted frame processing.
- If `snapshot_source=last_valid` and `used_last_valid_snapshot=true`, the
  trigger recovered from current ActiveProductStore cleanup by using the fresh
  last-valid inventory snapshot. If the same trace still has no raw detections,
  continue debugging video/model recall separately.
- Multi-zone CLOSE pending responses expose `pendingTriggerCount`,
  `pendingChargeableVisionCount`, `pendingTriggerZones`, and
  `pendingTriggerSessionIds`; final CLOSE summaries expose
  `missing_active_products` plus optional no-charge diagnostics so
  Node/operator logs do not misread the result as a vision-only miss.
- If strict mismatch occurs, compare active product snapshot, candidate weights,
  `weight_diagnostics`, and `StrictWeightMatcher.last_diagnostics`.
- If a rank-1 regular candidate and a lower-rank threshold/ROI rescue candidate
  are both inside strict single-item tolerance, the final strict ordering should
  prefer the regular candidate unless another higher-priority rule applies.
  Check `StrictWeightMatcher.last_diagnostics.valid_combinations` for the raw
  residuals and candidate sources when investigating this edge.
- Also inspect `weight_diagnostics.strict_candidate_priority_selection`.
  `matcher_raw_top_combinations` shows the weight-first matcher order, while
  `post_sort_top_combinations` shows the engine's final candidate-priority
  order. `reason=regular_single_candidate_priority` means a regular single
  final candidate such as Hatban was intentionally selected over a lower-rank
  item such as HOT6 despite a 1-2g residual disadvantage.
- Inspect `weight_diagnostics.stage_weight_gate_candidates` when a product has
  many stage votes but misses the regular candidate threshold. Accepted entries
  are promoted as `source=stage_weight_gate` only when the stage count passed
  the weight gate, the active product has valid stock/weight, votes meet the
  detected-single minimum, and confidence is at least `0.08`. If strict
  candidate priority chooses that promoted product, the strict diagnostics
  should show `reason=stage_weight_gate_candidate_priority`.
- If the final branch is `segment_weight_matching`, inspect
  `weight_diagnostics.segment_weight_matching.targets`,
  `segment_options`, and `selections`. This branch means separable negative
  loadcell movements were judged before the aggregate delta, so a split
  repeated-product removal can intentionally beat a simpler one-item aggregate
  match.
- If `loadcell.channel_removal_segment_targets` exists, inspect
  `loadcell.channel_delta_diagnostics` first. `accepted=true` means physical
  loadcell channels split a simultaneous same-zone removal; positive channel
  deltas or only one negative channel should reject the split. In decision
  diagnostics, `target_source=channel_removal_segment_targets` with
  `reason=channel_supported_split_preferred` means the channel-supported basket
  beat an aggregate single rescue candidate.
- Segment option diagnostics include evidence source, trusted/strong flags, and
  motion gate status when class evidence exists. They also include
  `stage_score`, `side_confidence`, `top_confidence`, `side_votes`,
  `top_votes`, and `score_reason` for stage-backed options. If an exact
  active-only residual loses to a slightly looser stage/diagnostic option,
  compare those fields with `evidence_score`, `evidence_source`, and
  `allowed_residual` in `segment_options` and `selections`.
- Segment options also expose `option_kind`, `selection_tier`,
  `selection_reason`, and compound `items`. A merged segment such as
  Fanta + Pepsi should appear as `option_kind=compound` with two `count=1`
  items; small repeated products rejected behind that option should carry
  `rejected_reason=trusted_or_single_item_segment_preferred`.
- Segment diagnostics also record `max_items_per_segment`,
  `segment_grip_limit`, `rejected_option_count`, and `rejected_options`. A
  one-segment small-product repeat above the grip cap should show
  `rejected_reason=count_exceeds_segment_grip_limit`; an aggregate override
  candidate above `len(targets) * max_items_per_segment` should show
  `reason=count_exceeds_segment_grip_limit`.
- For repeated 500ml bottle removals, inspect
  `segment_weight_matching.same_weight_bottle_collision` and
  `segment_weight_matching.repeated_segment_reuse_guard`. An accepted
  same-weight bottle collision means one class had enough repeated stage
  coverage to explain all separated bottle-weight segments. A rejected entry
  with `reason=repeated_segment_evidence_insufficient` means a top-only or weak
  one-off candidate was not allowed to be reused across multiple separated
  segment targets.
- If final candidates are empty but `stage_counts_by_class` saw multiple
  products, do not inspect raw detection count alone. `stage_score` caps raw
  votes with `log1p` and weights side confidence plus ROI/motion evidence above
  low-confidence top-only raw counts; this is the expected recovery path for
  Welchs x2 versus top-raw-heavy Cupban traces.
- In the same diagnostics, inspect `candidate_supported_override` when a final
  vision candidate appears in OPS logs but a different active product has a
  tighter segment residual. An accepted override means the detected product
  explained the aggregate segment total as a repeated count inside
  `same_product_count_tolerance * count + tolerance_grams`, so evidence beat an
  active-only residual-only segment fit.
- Inspect `aggregate_evidence_override` when a large removal is split into many
  short negative segments and the result is a low-confidence mixed basket. An
  accepted override means a strong final/stage/diagnostic product explained the
  aggregate segment total as a repeated count, which is the expected recovery
  path for collision-like loadcell fragmentation such as Trevi x5. This
  override still obeys the segment-derived total grip cap.
- In `segment_weight_matching.segment_options` and `selections`, inspect
  `evidence_supported`. Weak stage traces with low confidence or too few votes
  should show `false`; those traces are ranked like active-only options and
  should not turn small-item repeated counts into `COMPLETE`.
- If `aggregate_evidence_override.accepted=false` and
  `reason=selected_segment_already_confident`, the segment-level all-evidence
  selection was kept. Low-count threshold/ROI rescue noise should appear in the
  candidate list as `insufficient_aggregate_evidence` rather than winning an
  aggregate repeat by residual alone.
- If `aggregate_evidence_override.accepted=false` and
  `reason=clean_supported_segment_match_preferred`, every selected loadcell
  segment had supported evidence and the segment total residual was no worse
  than the aggregate repeated candidate. In this case the final products should
  follow `segment_weight_matching.selections`, such as Binch + Haruyache +
  Letsbe instead of Chapagetti x4.
- For `<=5g` tail triggers with video, `decision_branch` can be
  `low_weight_video_diagnostic`. In that branch `engine_skipped=true`,
  `excluded_from_close_summary=true`, and `candidates`/`stage_counts_by_class`
  are diagnostic evidence only; no product should be added to DoorSession. An
  active global session may still expose `noChargeDiagnostics` at CLOSE so the
  skipped trigger is visible to operators.
- If OPS candidates show a regular rank-1 product but the result would
  otherwise select a same-weight active product or lower-rank rescue candidate,
  inspect `weight_diagnostics.same_weight_candidate_collision`. An accepted
  diagnostic means regular final-candidate identity won even with a slightly
  larger residual inside the scaled allowance.
- If CLOSE shows no products but a nonzero positive `weightDelta`, inspect
  `unmatched_returns` and `weight_diagnostics.effective_count_guard`.
  Unmatched return hints are excluded from basket-facing effective delta; if no
  products remain and the residual is within weight tolerance, CLOSE should log
  `0.0g`.
- If a final candidate appears in OPS logs but the result includes extra
  candidate-outside stage-count products, inspect
  `weight_diagnostics.same_product_count_match`. When
  `stage_count_preempted=true` and `accepted=true`, the detected candidate won
  as a repeated same-product aggregate before stage-count expansion. The
  selected item records `residual` and `allowed_residual`, where the allowance
  is `same_product_count_tolerance * count + tolerance_grams`.
- If a removal trigger happens within three seconds of another same-zone
  trigger, inspect `loadcell.compound_segments` and
  `loadcell.recent_same_zone_events`. A positive segment or recent positive
  event is treated as a returned-weight hint during strict combination ranking.
- If same-zone return matching appears out of order, compare the stored
  `TriggerResult.timestamp` values. Door-session aggregation replays by event
  timestamp so a loadcell-only return that completes first can still apply
  after the earlier video-backed removal.
- If final output uses `weight_diagnostics.forced_final_fallback`, the engine
  intentionally avoided `none` by selecting from the active product snapshot.
  Check `inside_tolerance`, `target_source`, `expected_weight`, and `residual`
  before treating the result as high-confidence. For pair fallbacks, also check
  `mode`, `pair_support_rank`, and `pair_support`: a supported repeated product
  should appear as `detected_same_product_pair`, while unsupported mixed active
  products should be ranked behind it even if their residual is smaller.

## Related Wiki Pages

- [Latency and frame stride](../synthesis/latency-and-frame-stride.md)
- [Loadcell and trigger](loadcell-and-trigger.md)
- [Decision and weight](decision-and-weight.md)
