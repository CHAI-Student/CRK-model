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
  `decision_delta` sources.
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
  as `144g + 375g` being merged into one `519g` time segment.
- Channel split diagnostics are recorded in `channel_delta_diagnostics`,
  including per-channel start/end/delta values, rejection reasons for positive
  channel offsets or insufficient negative channels, and accepted channel
  targets.
- Paired-out-free positive stable segments are exposed as
  `return_segment_targets`. When a negative chargeable trigger also contains
  such a positive segment, both `TriggerService` and the compatibility
  `/trigger` route convert it into an internal `return_weight_hints` entry so
  DoorSession aggregation can replay the return while the decision engine still
  judges the negative `decision_delta`.
- Positive-then-negative pairs that look like press/release movement are not
  promoted into normal purchase targets. The negative side is recorded under
  `vision_required_segment_targets`, which the decision engine may use only
  when product evidence exists in vision, stage counts, or diagnostics.
- `decision_delta` is the movement used for removal judgment. It stays with the
  stable start/end net delta when that is valid, but can switch to an unpaired
  negative segment when return/removal history is merged in one trigger.
- Stable-tail diagnostics include `stable_delta_source`,
  `baseline_stable_avg`, `final_stable_avg`,
  `trailing_unstable_sample_count`, `raw_simple_delta`, and
  `raw_extreme_delta`. `raw_extreme_delta` is useful for identifying transient
  max/min swings, but it is never the chargeable movement.
- Opposite-sign movements within tolerance are paired out. This prevents pure
  remove-return cycles and press-hold-release patterns from becoming purchase
  candidates, while below-threshold micro bumps stay ignored.
- `summarize_loadcell_payload()` records payload-shape evidence separately
  from delta calculation: empty payloads, invalid-only filtered channels,
  all-zero filtered channels, and nonzero payloads are distinguishable in
  traces and OPS logs.
- `calculate_weight_delta()` is the shared route/service helper.
- `api/routes/trigger.py` and `TriggerService` keep compatibility wrapper
  helpers, but the implementation delegates to `core/loadcell_stats.py` so
  loadcell math has a single source of truth.

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
  DoorSession/payment aggregation.
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
  `net_delta_weight`. Trace metadata also carries segment targets so the engine
  can run segment-first matching without changing the public trigger schema.
- Chargeable negative trigger results are added to DoorSession/payment only
  after the decision engine result explains the full stable removal delta
  inside the existing branch tolerance. Partial sub-segment fallbacks return a
  no-charge `UNCERTAIN` result with `final_weight_mismatch_guard` diagnostics.
- Mixed return/removal triggers keep the public response shape unchanged. A
  trigger like `+216.7g` then `-16.5g` is judged as the `-16.5g` removal, while
  the hidden return is attached to the internal DoorSession `TriggerResult` as
  `return_weight_hints` for CLOSE deferred reconciliation and delta accounting.
- Compatibility `/trigger` metadata now stays aligned with `TriggerService`:
  traces include `return_segment_targets`, mixed return diagnostics, and
  `effective_count_guard` diagnostics when return hints can reduce a raw
  repeated-count result.
- Async enqueue registers an in-flight trigger by session id in
  `DoorSessionStore` before returning. Pending entries distinguish chargeable
  vision work from non-chargeable/cancelled diagnostics. Worker start changes the state to
  `processing`, and the worker only clears it after video processing, engine
  judgment, `SessionStore` save, `DoorSession` aggregation, OPS logging, and
  trace finalization are complete. Error paths clear the same session id once
  with status `error`.
- Worker lifecycle is started/stopped by FastAPI lifespan.
- The older inline `TriggerService` loadcell helper implementations were
  removed after being shadowed by the current `core/loadcell_stats.py`
  delegates; this is a dead-code cleanup, not a behavior change.

## Low-Weight Behavior

- If `abs(delta_weight)` is less than or equal to
  `MODEL__TRIGGER__MIN_WEIGHT_CHANGE_GRAMS`, the trigger is non-chargeable.
- Ignored low-weight triggers save a skipped session and trace diagnostics, but
  are excluded from DoorSession/global close aggregation.
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
