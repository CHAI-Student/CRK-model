# Source Code Map: Loadcell And Trigger

Source: [core/loadcell_stats.py](../../../services/model/model_service/core/loadcell_stats.py),
[api/routes/trigger.py](../../../services/model/model_service/api/routes/trigger.py),
[service/trigger_service.py](../../../services/model/model_service/service/trigger_service.py)

Status: current trigger path map

## Current Thesis

The trigger path converts Camera loadcell samples and AVI paths into queued
inference work. The worker is intentionally serial to protect the Jetson GPU,
so queue wait is a real latency dimension.

## Loadcell Delta

- `parse_loadcell_value()` handles raw string-like values.
- `avg_loadcell_channels()` is still named as an average helper, but current
  README/tests document that physical zone channels are summed into the zone
  total.
- `filter_peaks()` removes temporary loadcell spikes.
- `analyze_weight_delta()` now makes chargeable deltas from confirmed stable
  plateaus only. The baseline is the first stable plateau and the final value
  is the last stable plateau whose tail reaches the end of the parsed payload
  within `stable_window_size - 1` samples. First/last sample deltas, raw
  extreme deltas, and simple fallbacks are diagnostics, not chargeable
  `decision_delta` sources except for the freezer endpoint fallback below.
- Freezer triggers can enable a conservative endpoint fallback through
  `MODEL__LOADCELL__FREEZER_ENDPOINT_FALLBACK_ENABLED=true`. When stable
  history would otherwise produce no chargeable negative removal, a nonzero
  payload with at least 10 samples over 2 seconds can use the filtered first
  and last zone totals if the first total is near the payload high point and
  the last total is near the payload low point. This handles freezer Camera
  histories such as `+10g -> -60g` as `decision_delta=-70g` instead of `0g`.
- Loadcell history is stable-plateau based. The service records
  `stable_plateaus`, `compound_segments`, paired opposite movements, ignored
  movement diagnostics, and ordered `purchase_delta_candidates`.
- Separable removal movements are now exposed as `removal_segment_targets`.
  These are paired-out-free negative stable segments in time order, so a
  multi-step removal such as `210g + 105g + 103g + 107g` can be judged before
  falling back to the aggregate start/end delta.
- Simultaneous same-zone removals can also expose
  `channel_removal_segment_targets`. These are derived from physical loadcell
  channel deltas across the stable start/end plateaus when two or more channels
  drop at the same time and their summed negative delta matches the aggregate
  removal within tolerance. They are evidence-required targets for cases such
  as `144g + 375g` being merged into one `519g` time segment. Each accepted
  target carries `channel_position` and `channel_side` (`left`, then `right`
  for the first two physical removal channels) so freezer decisions can keep
  one product group per loadcell.
- Stable start/end channel deltas are also exposed as
  `channel_movement_targets`. This broader diagnostic list includes one-sided
  removals and positive return-side movements, with `direction=removal` or
  `direction=return`, channel index/position, and channel side. Freezer
  trigger judgment reads negative movement targets when the stricter
  `channel_removal_segment_targets` list is absent, so a single left/right
  loadcell movement can still be solved as one product group. Freezer
  positive-only return triggers carry positive movement targets into
  DoorSession for location-aware return reconciliation.
- Channel split diagnostics are recorded in `channel_delta_diagnostics`,
  including per-channel start/end/delta values, rejection reasons for positive
  channel offsets or insufficient negative channels, and accepted channel
  targets.
- Paired-out-free positive stable segments are exposed as
  `return_segment_targets` for the default/non-freezer path. When a negative
  chargeable non-freezer trigger also contains such a positive segment, both
  `TriggerService` and the compatibility `/trigger` route convert it into an
  internal `return_weight_hints` entry so DoorSession aggregation can replay
  the return while the decision engine still judges the negative
  `decision_delta`. Freezer mixed-sign negative triggers suppress these segment
  targets and keep the positive movement as diagnostics only.
- Positive-then-negative pairs that look like press/release movement are not
  promoted into normal purchase targets. The negative side is recorded under
  `vision_required_segment_targets`, which the decision engine may use only
  when product evidence exists in vision, stage counts, or diagnostics.
- `decision_delta` is the movement used for removal judgment. It stays with the
  stable start/end net delta when that is valid. The default/non-freezer path
  can still switch to an unpaired negative segment when return/removal history
  is merged in one trigger; freezer mode now opts into
  `stable_net_delta_only`, so mixed-sign internal positive segments do not
  override the stable net delta. A freezer `+70g` then `-150g` payload remains
  `decision_delta=-80g`, and the `+47.6g` then `-295.3g` field shape remains
  about `decision_delta=-247.7g`. The freezer endpoint fallback remains the
  narrow exception when stable plateau evidence is unavailable but endpoint
  checks are accepted.
- Stable-tail diagnostics include `stable_delta_source`,
  `baseline_stable_avg`, `final_stable_avg`,
  `trailing_unstable_sample_count`, `raw_simple_delta`, and
  `raw_extreme_delta`. Freezer mixed-sign internal segments are retained as
  compound segment diagnostics and as `TriggerResult.loadcell_diagnostics` for
  CLOSE aggregate eligibility, but the retired freezer
  `mixed_sign_net_masking_guard` is not accepted in the default path. Endpoint
  fallback diagnostics add
  `decision_delta_reliable`, `endpoint_delta_weight`,
  `endpoint_fallback_applied`, and `endpoint_fallback_reason`.
  `raw_extreme_delta` is useful for identifying transient max/min swings, but
  it is never the chargeable movement.
- Opposite-sign movements within tolerance are paired out. This prevents pure
  remove-return cycles and press-hold-release patterns from becoming purchase
  candidates, while below-threshold micro bumps stay ignored.
- `summarize_loadcell_payload()` records payload-shape evidence separately
  from delta calculation: empty payloads, invalid-only filtered channels,
  all-zero filtered channels, and nonzero payloads are distinguishable in
  traces and OPS logs.
- Cabinet type does not change the effective loadcell payload. Refrigerated
  and freezer modes both use zone-sliced `/trigger.loadcells`; the deprecated
  `/trigger.global_loadcells` field is ignored for decision input. OPS/trace
  metadata still records `cabinet_type`, `loadcell_scope=zone`,
  `requested_zone`, `effective_channel_count`, and any received
  `global_channel_count` for diagnostics.
- `calculate_weight_delta()` is the shared route/service helper.
- `api/routes/trigger.py` and `TriggerService` delegate cabinet-specific
  loadcell options to `core/loadcell_stats.py`
  (`endpoint_fallback_enabled_for_cabinet()`,
  `stable_net_delta_only_for_cabinet()`, and the retired
  `prefer_mixed_sign_removal_delta_for_cabinet()`), so compatibility
  `/trigger` and the queued service path share one source of truth for freezer
  endpoint fallback and mixed-sign stable-net behavior.

## Trigger Inputs And Outputs

Key dataclasses in `TriggerService`:

- `LoadcellReading`
- `TriggerTimingMetadata`
- `TriggerInput`
- `TriggerOutput`
- `LoadcellEvent`
- `QueueItem`

`QueueItem` keeps the session id, idempotency key, delta analysis, active
product snapshot, product weights, allowed class ids, return-weight hints, and
trace context needed by the background worker. It also carries active-product
snapshot metadata so the worker trace preserves whether the snapshot came from
current inventory or the last-valid fallback. It carries a loadcell event id so
queued work can be skipped if a later return balances it before video starts.

## Queue And Dedup

- Dedup key is based on zone and video paths.
- Dedup TTL and max size come from `config.trigger`.
- Queue max size comes from `MODEL__TRIGGER__QUEUE_MAX_SIZE`.
- `enqueue_trigger()` returns quickly with `queued` unless the trigger is
  skipped, duplicate, invalid, return-only, or blocked.
- Positive return deltas use a loadcell-only fast path by default
  (`MODEL__TRIGGER__RETURN_VIDEO_SKIP_ENABLED=true`). The service records the
  raw trigger/trace data and adds a return trigger to DoorSession only when it
  is needed to repair already-processed removals.
- Return fast-path commits now wait for return loadcell stabilization first.
  `MODEL__TRIGGER__RETURN_STABILIZATION_WAIT_SECONDS=1.0` delays the commit
  path, and `MODEL__TRIGGER__RETURN_STABILIZATION_REQUIRE_STABLE_REGIONS=true`
  blocks short first/last fallback samples from being saved as a final return.
  If the payload still lacks a stable tail after the wait, the response is
  `status=waiting`, `waiting_for=stable_loadcell`, and no DoorSession return is
  added; traces include `weight_diagnostics.return_stabilization`.
- Negative removal triggers also require a confirmed stable tail before any
  chargeable work. If the observed trend is negative but the analysis is
  unstable, truncated, or only a simple fallback, `TriggerService` returns
  `status=waiting`, `waiting_for=stable_loadcell`, saves
  `processing_stage=removal_waiting_for_stable_loadcell`, records
  `weight_diagnostics.removal_stabilization`, skips video and
  `ProductDecisionEngine.judge()`, and keeps the trigger out of
  DoorSession/payment aggregation. The freezer endpoint fallback is the narrow
  exception: when it marks `decision_delta_reliable=true`, the removal proceeds
  to normal freezer judgment instead of the stable-loadcell waiting branch.
- When that waiting path also has a missing or empty active-product inference
  snapshot, the trigger records active-product diagnostics before returning.
  `SessionData.failure_reason`, trace `final_result.failure_reason`,
  `weight_diagnostics.active_product_failure_reason`, and the multi-zone
  waiting response expose `missing_active_products` or
  `empty_active_product_allowlist` without making the trigger chargeable.
- `MODEL__TRIGGER__BALANCED_EVENT_CANCEL_ENABLED=true` lets a later return
  cancel still-queued removal triggers when the loadcell weights balance within
  DoorSession tolerance. Cancelled items remain in the queue but the worker
  skips them before video processing.
- `MODEL__TRIGGER__RAPID_SAME_ZONE_WINDOW_SECONDS=3.0` records recent same-zone
  loadcell events in trace loadcell metadata. Negative follow-up triggers can
  therefore see recent positive return weights, which helps avoid charging a
  product that was just put back.
- The trigger service passes `decision_delta` to the decision engine for
  chargeable work, while trace metadata still preserves the raw net delta as
  `net_delta_weight`. Trace metadata also carries segment and channel movement
  targets so the engine can run segment-first or freezer left/right matching
  without changing the public trigger schema.
- The earlier freezer `[OPS][LOADCELL] mixed_sign_net_masking_guard` path is
  retired in the default configuration. Freezer mixed-sign payloads still show
  compound positive/negative segment diagnostics in traces and carry compact
  `loadcell_diagnostics` into DoorSession for signed-net CLOSE aggregate
  eligibility.
- Chargeable negative trigger results are added to DoorSession/payment only
  after the decision engine result explains the full stable removal delta
  inside the existing branch tolerance. Partial sub-segment fallbacks return a
  no-charge `UNCERTAIN` result with `final_weight_mismatch_guard` diagnostics.
- Freezer `freezer_vision_first` valid-weight mismatches are subject to that
  same full-delta rule. The video/OPS candidate list may still show one
  handled product with `count_hint=1`, but the engine can correct the final
  `product_count` to `2+` through repeat fitting or suppress the result as
  no-charge `UNCERTAIN` when the full delta cannot be explained.
- Both trigger entrypoints keep public request/response schemas unchanged while
  adding internal OPS fields: candidate lines include `count_hint` and
  `freezer_exit_votes`, freezer filter lines include selected count, expected
  weight, count residual, `repeatEvidenceMode`, and repeat rejection reason,
  and result lines include final engine `product_count`.
- Mixed return/removal triggers keep the public response shape unchanged. A
  trigger like `+216.7g` then `-16.5g` is judged as the `-16.5g` removal, while
  the hidden return is attached to the internal DoorSession `TriggerResult` as
  `return_weight_hints` for CLOSE deferred reconciliation and delta accounting
  on the default/non-freezer path.
- For freezer sessions, internal mixed-sign positive segments no longer create
  trigger-time `return_weight_hints`. If the door-open session has mixed-sign
  internal segment diagnostics, two or more meaningful freezer triggers, or
  freezer triggers across zones, CLOSE first validates the already-selected
  trigger products against the signed sum of participating `delta_weight`
  values. If that basket fits freezer tolerance, per-zone output is preserved;
  otherwise the freezer aggregate resolver re-solves from raw-confidence-gated
  trigger candidates and assigns fallback output to the latest participating
  freezer trigger zone. This keeps the trigger schema unchanged while avoiding
  shared dual-top camera overlap from locking in a mismatched per-zone basket.
- Freezer CLOSE final-weight validation can now repair a no-product or
  final-weight-mismatch partial result by borrowing a later unused candidate
  snapshot from the same global door session. The repair is single-removal
  only, requires the later candidate's unit weight to match the target removal
  inside close tolerance, excludes candidates already consumed by their own
  matched trigger result, and records `deferredCandidateRepair` diagnostics
  with source zone/session/rank. It does not delay the `/trigger` response and
  does not override already complete weight-matched results.
- Compatibility `/trigger` metadata now stays aligned with `TriggerService`:
  traces include `return_segment_targets`, mixed return diagnostics,
  stable-net freezer mixed-sign metadata, and `effective_count_guard`
  diagnostics when default/non-freezer return hints affect the removal decision
  or reduce a raw repeated-count result.
- Async enqueue registers an in-flight trigger by session id in
  `DoorSessionStore` before returning. Pending entries distinguish chargeable
  vision work from non-chargeable/cancelled diagnostics. Worker start changes the state to
  `processing`, and the worker only clears it after video processing, engine
  judgment, `SessionStore` save, `DoorSession` aggregation, OPS logging, and
  trace finalization are complete. Error paths clear the same session id once
  with status `error`.
- Fatal async video processing errors propagate out of `VideoProcessor` into
  the worker. The worker marks the loadcell event and trigger session as
  `error`, finalizes the trace with `status=error`, notifies
  `DoorSessionStore`, and does not call the decision engine or add products.
- Worker lifecycle is started/stopped by FastAPI lifespan.
- The older inline `TriggerService` loadcell helper implementations were
  removed after being shadowed by the current `core/loadcell_stats.py`
  delegates; this is a dead-code cleanup, not a behavior change.

## Low-Weight Behavior

- If `abs(delta_weight)` is less than or equal to
  `MODEL__TRIGGER__MIN_WEIGHT_CHANGE_GRAMS`, the trigger is non-chargeable.
- Ignored low-weight triggers save a skipped session and trace diagnostics, but
  are excluded from DoorSession/global close aggregation. If a global door
  session is active, the service also records a no-charge diagnostic on the
  global session; this does not add products, counts, prices, or trigger count.
- When `MODEL__TRIGGER__LOW_WEIGHT_VISION_FALLBACK=true` and a video path is
  present, the service and compatibility `/trigger` route run video processing
  for diagnostics only. They record candidates/stage counts, save
  `processing_stage=low_weight_video_diagnostic`, set
  `engine_skipped=true`, and do not call `ProductDecisionEngine.judge()` or add
  a DoorSession charge.
- When the same flag is disabled, low-weight triggers remain hard-skipped
  before video processing.
- Low-weight traces include `low_weight_ignored`, `threshold_grams`, and
  `excluded_from_close_summary` style diagnostics so field logs do not look
  like a vision miss.
- CLOSE responses expose skipped low-weight/loadcell-payload diagnostics
  through optional `decisionSummary.diagnosticZoneLines` and
  `decisionSummary.zones[*].noChargeDiagnostics`. The summary text and payment
  totals remain unchanged, so a zone can show
  `diagnostic=loadcell_payload_all_zero` while `zones=none` and
  `total_price=0` are still correct.
- Low-weight skip behavior remains non-chargeable, but suspect payloads now get
  an explicit `loadcell_payload_diagnostic` branch in weight diagnostics.
  `loadcell_payload_reason` distinguishes empty payloads, invalid-only filtered
  values, all-zero filtered values, and filtered-all-zero/raw-nonzero cases.
- Use `payload_state`, raw/filtered states, channel counts, and first/last
  totals to tell whether `0.0g` came from missing/invalid/all-zero data or a
  nonzero but stable/no-change payload.

## Worker Processing

The worker performs:

1. queue wait measurement
2. active product snapshot handling, including last-valid fallback when current
   Node inventory context is missing or invalid
3. cancellation check for balanced-out queued removals
4. video processing through `VideoProcessor` only for chargeable removal work
5. vote-to-engine candidate conversion
6. `ProductDecisionEngine.judge()`
7. session storage
8. door-session aggregation
9. OPS and trace finalization

## Related Wiki Pages

- [Runtime flow](../synthesis/runtime-flow.md)
- [API routes](api-routes.md)
- [Observability and traces](observability-and-traces.md)
- [Scenario readiness and 0g diagnostics](../synthesis/scenario-readiness-and-0g.md)
