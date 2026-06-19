# Runtime Flow

## Current Thesis

The core model path is a queued trigger worker followed by video processing,
decision fusion, and door-session aggregation. When debugging behavior, split
model processing latency from Node polling/CLOSE timing.

## End-To-End Flow

```text
Camera POST /trigger
  -> api/routes/trigger.py validates payload and computes delta_weight
  -> TriggerService.enqueue_trigger() records loadcell relevance
  -> return-only or balanced-out triggers skip video
  -> TriggerService worker processes chargeable vision work one trigger at a time
  -> VideoProcessor.process_videos_async() decodes top/side AVI streams
  -> YOLOWrapper detects per processed frame
  -> filters and VotingEnsemble produce vision candidates
  -> ProductDecisionEngine.judge() fuses candidates, delta_weight, active_products
  -> SessionStore stores the trigger result
  -> DoorSessionStore.add_trigger_with_global() updates active door session
  -> ProductAggregator rebuilds removal/return counts
  -> Node POST /api/judge/multi-zone polls or CLOSE finalizes
```

## Decision Inputs

- `delta_weight < 0`: product removal.
- `delta_weight > 0`: product return.
- Loadcell channels are currently documented by README/tests as summed into the
  zone total. Older docs that say "average" are historical and should be
  cross-checked against `core/loadcell_stats.py` and tests before changing
  behavior.
- Refrigerated and freezer modes both consume the zone-sliced
  `/trigger.loadcells` payload. The deprecated `global_loadcells` field is
  accepted for compatibility but is not the effective decision payload.
- `active_products`: live Node product snapshot; required for strict and
  loadcell-only matching.
- `stock_qty = 0`: sold-out signal; strict matching must exclude the product.
- Top/side vision candidates: produced from processed frames after filtering
  and voting.

## Inference Branch Order

`ProductDecisionEngine.judge()` is currently documented as evaluating:

1. `vision_only`
2. freezer `freezer_vision_first` branch when
   `MODEL__MACHINE__CABINET_TYPE=freezer` and the delta is negative
3. stage-count/no-final-candidate recovery
4. legacy `loadcell_only_no_vision` only under explicit `weight_aware`
5. `no_detection_min_weight`
6. `strict_match`
7. Relaxed fallback paths:
   `single_product_match`, `combination_match`, `partial_result`,
   `loadcell_only_no_estimates`

If strict matching fails and `MODEL__WEIGHT__STRICT_MODE_FALLBACK=true`, the
engine continues into relaxed fallback. If fallback is disabled, strict mismatch
can terminate as no detection.

## Return Recovery

Return handling is layered:

- Loadcell-first scheduling runs before the worker. Positive return deltas skip
  YOLO by default, and queued removals that are balanced by a later return are
  marked `skipped_balanced` before video starts.
- Same-zone `ProductAggregator._handle_return()` attempts direct rollback,
  return-combination rollback, then unmatched-return recording.
- `DoorSessionStore._handle_cross_zone_returns()` matches unmatched returns
  across active zones before net-delta validation. It uses bounded combination
  matching, so one trigger can repair same-product multi-count returns,
  multi-product bundles, or bundles removed from multiple zones.
- `DoorSessionStore._validate_net_delta()` then repairs full or partial returns
  using effective zone delta, computed as raw trigger delta minus outgoing
  cross-zone returns plus incoming cross-zone returns.
- CLOSE summaries expose effective basket deltas; raw `+xxg` return triggers
  stay in session/trace diagnostics.
- CLOSE waits only for pending chargeable vision work. Non-chargeable
  return-only or balanced-out diagnostics can remain visible in pending
  snapshots without blocking finalization.

## Active Product Snapshot Recovery

- `ActiveProductStore` keeps current Node product data plus a TTL-bounded
  last-valid stock-positive, positive-weight snapshot.
- Door `CLOSE` cleanup clears only the current snapshot; trigger enqueue and
  fallback `/trigger` use the effective snapshot, falling back to last-valid
  when current data is missing or invalid.
- This fixes traces where `missing_active_product_snapshot_fail_closed` came
  from model-side context loss after repeated judgment/recovery cycles. It
  does not fix cases where raw video detections are genuinely absent.

## Observability Points

- `[TRIGGER-WORKER][LATENCY]`: queue wait, video time, engine time, YOLO count,
  frame stride, original/processed/skipped frames.
- `[VIDEO-ASYNC][LATENCY]`: async video processing totals and stride counts.
- `[CLOSE][LATENCY]`: close wait behavior and pending trigger counts.
- Trace JSON and frame traces: detailed per-session diagnostics.

## Evidence

- [Startup and DI](../source-code/startup-and-di.md)
- [API routes](../source-code/api-routes.md)
- [Loadcell and trigger](../source-code/loadcell-and-trigger.md)
- [Session and persistence](../source-code/session-and-persistence.md)
- [Architecture guide](../source-docs/agent-guides-architecture.md)
- [Trigger inference recovery notes](../source-docs/trigger-inference-recovery-notes-2026-03-31.md)
- [Product detection flow](../source-docs/product-detection-flow.md)
- [Product detection detail](../source-docs/product-detection-detail.md)
- [Node protocol](../source-docs/protocols-node.md)
