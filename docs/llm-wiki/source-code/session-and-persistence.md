# Source Code Map: Session And Persistence

Source: [session/session_store.py](../../../services/model/model_service/session/session_store.py),
[session/door_session.py](../../../services/model/model_service/session/door_session.py),
[session/door_session_store.py](../../../services/model/model_service/session/door_session_store.py),
[session/freezer_close_aggregate.py](../../../services/model/model_service/session/freezer_close_aggregate.py),
[session/global_door_session.py](../../../services/model/model_service/session/global_door_session.py),
[session/product_aggregator.py](../../../services/model/model_service/session/product_aggregator.py),
[session/active_product_store.py](../../../services/model/model_service/session/active_product_store.py),
[session/yaml_persistence.py](../../../services/model/model_service/session/yaml_persistence.py)

Status: current session map

## Current Thesis

There are two session layers: trigger-session storage for individual inference
results, and door/global sessions for basket-level aggregation across zones
while the door is open.

## Trigger Session Layer

- `SessionStore` keeps per-trigger `SessionData` with TTL and max session count.
- `SessionData.failure_reason` records non-chargeable trigger blockers such as
  missing active products even when the session is still `waiting` for a stable
  loadcell retry.
- `ProductResult` is the stored product item shape used by routes and sessions.
- `generate_session_id(zone)` includes microseconds to reduce collisions.
- Worker errors should propagate status rather than leaving sessions stuck in
  `processing`. Fatal async video errors now reach the worker catch block,
  which saves `status=error` / `processing_stage=error` and clears pending
  trigger state.

## Door Session Layer

- `TriggerResult` represents one `/trigger` result inside a door session and
  can carry `failure_reason` such as `missing_active_products`. It can also
  carry internal `return_weight_hints` for default/non-freezer mixed
  return/removal loadcell triggers and compact `loadcell_diagnostics` for
  freezer signed-net CLOSE aggregate eligibility, without changing the HTTP
  response schema.
- `TriggerResult.vision_candidates` stores a compact close-correction snapshot
  of ranked candidates joined with active product weight, price, stock, and
  top/side evidence. Old YAML without this field still restores with an empty
  snapshot.
- `DoorSession` stores triggers, aggregated products, unmatched returns,
  deferred returns, cross-zone return records, timestamps, summary fields, and
  optional `final_weight_validation` diagnostics from CLOSE correction.
- `GlobalDoorSession` groups active zone sessions for the whole door-open
  interval.
- `DoorSessionStore` manages active sessions, global sessions, close debounce,
  finalization, timeout cleanup, YAML recovery, and cross-zone return repair.
- CLOSE finalization is gated by a session-id based in-flight trigger registry,
  not only a count. Any `queued`, `processing`, or `finalizing` trigger keeps
  `handle_close_signal()` at `ready=false reason=pending_trigger`; finalization
  is allowed only after the worker clears the registry entry.

## Product Aggregation

- `ProductAggregator` rebuilds the basket by trigger event timestamp, not by
  completion/append order. This matters because loadcell-only returns can
  finish before an earlier video-backed removal.
- Negative deltas remove/add products to customer basket.
- Positive deltas are returns. During ordinary trigger aggregation, only a
  single same-zone product within flat strict tolerance is deducted
  immediately. Same-product `x2+`, multi-product combinations, cross-zone
  candidates, and internal `return_weight_hints` are stored as
  `deferred_returns` instead of mutating the live basket.
- Negative triggers with `return_weight_hints` preserve those hints by segment
  position for CLOSE reconciliation. They are no longer replayed into the
  intermediate basket, because doing so can hide the actual final delta.
- Same-zone returns, deferred CLOSE reconciliation, cross-zone repair,
  effective net-delta repair, final repeat correction, and freezer CLOSE
  aggregate solving are separate recovery layers. The CLOSE order is local
  aggregation, deferred return reconciliation, net-delta validation, final
  repeat correction, then freezer aggregate solving when the freezer session is
  eligible.
- Cross-zone repair expands positive-count products from other active zones
  into bounded unit candidates, then matches unresolved return deltas against
  deterministic combinations at CLOSE. This covers one item, same-product
  multi-count, multi-product bundles, and bundles originally removed from
  multiple zones without changing the intermediate same-zone basket.
- Cross-zone repair is all-or-nothing for each unmatched return trigger. If no
  complete combination is within tolerance, the return stays unmatched instead
  of partially hiding raw weight.
- Effective zone delta is `raw trigger delta + mixed return hints - outgoing
  cross-zone return weight + incoming cross-zone return weight`; raw trigger
  deltas and return hints remain in the session for diagnostics. Freezer
  signed-net aggregate output can override this basket-facing value at CLOSE
  through `final_weight_validation.freezerCloseAggregate.weightDeltaOverride`.
- Remaining `unmatched_returns` are subtracted from the basket-facing effective
  delta because they did not change the basket. This prevents an unmatched
  positive segment inside a removal trigger from leaving a phantom positive
  CLOSE `weightDelta` after the actual product has been returned.
- When a finalized zone has no active products and the remaining effective
  delta is within the configured weight tolerance, CLOSE output normalizes the
  zone and total `weightDelta` to `0.0g`.
- At global CLOSE only, `DoorSessionStore` compares each zone's final effective
  negative delta against the aggregated basket weight. If the current basket is
  over-fragmented and a repeated ranked candidate explains the final delta
  inside count-scaled tolerance, the basket can be replaced by that same-product
  repeat even when its residual is slightly worse than the current mixed basket.
- CLOSE final-weight correction cannot replace an all-vision-supported current
  basket with a different product identity only because another candidate has a
  tighter loadcell residual. If every current product id is backed by strong
  regular vision evidence, the current identity set is preserved. A same-product
  replacement is still allowed so CLOSE can validate or adjust the count of an
  already vision-supported product.
- CLOSE final-weight correction normally allows a replacement only within one
  base-tolerance residual disadvantage. If the current basket contains
  unsupported small fragments such as stage-only Condition Stick or Binch
  repeats, the residual gap allowance expands narrowly to
  `base + same_product_count_tolerance`; diagnostics expose
  `currentHasUnsupportedFragments`, `residualGap`, and `residualGapAllowed`.
- CLOSE final-weight correction is not blocked solely because return triggers
  or internal return hints were present. It runs after deferred reconciliation;
  unresolved unmatched returns or cross-zone repair records can still block it
  so incomplete return state does not get overwritten by repeat correction.
- CLOSE final repeat correction applies a count cap:
  `min(stock_qty, same_product_max_count, max_count_per_item,
  removal_trigger_count * max_items_per_segment)`. Candidates above this cap
  are rejected with `count_exceeds_close_repeat_cap`, which prevents final
  corrections such as HomeRunBall `64g x33` while preserving supported cases
  such as Sky Barley `x3`.
- Freezer has a signed-net CLOSE aggregate resolver for unstable sessions. It
  runs only when `MODEL__MACHINE__CABINET_TYPE=freezer` and only when there are
  mixed-sign internal segment diagnostics, two or more meaningful freezer
  triggers, or freezer triggers spanning multiple zones. A simple single stable
  negative freezer trigger without mixed-sign diagnostics stays on the existing
  per-zone path.
- `FreezerCloseAggregateResolver` collects participating trigger candidate
  snapshots and trigger products, sums participating signed `delta_weight`
  values as `globalNetDelta`, and solves one vision-supported basket against
  `abs(globalNetDelta)` only when that net is negative. It does not create
  unseen active-product-only identities; products without usable positive
  weight are excluded from aggregate count correction and remain
  diagnostic-only. Low-delta candidate-only freezer triggers at or below
  freezer tolerance do not force aggregate eligibility.
- If `abs(globalNetDelta)` is within freezer tolerance, aggregate CLOSE clears
  all participant products and returns `weightDeltaOverride=0.0` for every
  participant zone. If `globalNetDelta` is positive, the aggregate output is
  also no-charge. If `globalNetDelta` is negative but no handled/final
  candidate combination fits the target, provisional participant products are
  cleared and diagnostics record a no-charge reason instead of preserving a
  mismatched basket.
- Accepted freezer aggregate output is attributed to the latest participating
  trigger zone, with insertion order as the tie-breaker for same-tick
  timestamps. The output zone receives the full selected basket and a
  close-time `weightDeltaOverride` equal to the signed `globalNetDelta`; other
  participating freezer zones are emptied and receive `weightDeltaOverride=0.0`.
  Public response schemas stay unchanged: existing zone `products` and
  `weightDelta` fields are redistributed at CLOSE.
- `final_weight_validation.freezerCloseAggregate` records
  `policy=signed_net_delta`, eligibility reason, participating zones/triggers,
  `globalNetDelta`, final target, output zone, selected products, residual,
  no-charge reason when applicable, role, and weight delta override.

## ActiveProductStore

- Node product snapshots are normalized into model-side product records.
- The store maps current YOLO engine class names to product metadata and
  weights through Edge class-name keys.
- In node-first mode, class-id resolution ignores
  `trainingidx`/`training_idx`/`trainingIdx`/`yolo_class_id` aliases and
  `yolo_class_name`; those fields can drift from the deployed engine. Official
  input is `product_eng_name` and successful matches record
  `class_id_source=product_eng_name_engine`. During Edge migration,
  engine-matching `name` records `name_engine_compat` and legacy
  `product_name` records `product_name_engine_legacy`.
- It is the source of product weights for strict/loadcell-only matching.
- Product weight input accepts `product_weight`, `productWeight`,
  `product_weight_g`, `unit_weight_g`, and `weight` aliases. Alias repairs are
  logged in `repaired_weight_diagnostics` so the trace can show which field
  supplied the usable weight.
- If a mapped stock-positive product arrives with weight `<= 0`, the store
  repairs it from the current snapshot, then the fresh last-valid snapshot for
  the same YOLO class or product index. As a narrow final guard, class `44`
  (`BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML`) repairs to `520.0g`.
- Active snapshots are treated as live inventory candidates. Multi-zone
  `CLOSE` responses and zero-stock/zero-weight product payloads are not allowed
  to replace a valid stock-positive, positive-weight snapshot.
- `set_products(..., preserve_on_invalid_existing=True)` preserves the previous
  valid snapshot when a new mapped payload has no stock-positive,
  positive-weight products.
- The store also keeps a TTL-bounded last-valid snapshot. Door `CLOSE`
  cleanup clears the current snapshot but preserves this fallback so a trigger
  that arrives before the next valid Node product snapshot can still build
  `allowed_class_ids` and product weights.
- Trigger traces expose whether inference used `snapshot_source=current`,
  `last_valid`, or `missing`. On shutdown/container cleanup, current and
  last-valid snapshots are both cleared.

## Persistence

- Door sessions are serialized through `YamlPersistence`.
- YAML directory and retention are configured by `MODEL__DOOR_SESSION__*`.
- Startup can recover active sessions; shutdown can clean old YAML files using
  `MODEL__DOOR_SESSION__YAML_RETENTION_DAYS`.
- Frame trace JSONL retention is not handled by `YamlPersistence`; deployed
  hosts should use OS log rotation for `services/model/logs/frame_split_*.jsonl`.

## Related Wiki Pages

- [Runtime flow](../synthesis/runtime-flow.md)
- [Protocol contracts](../synthesis/protocol-contracts.md)
- [Historical risk and fixes](../synthesis/historical-risk-and-fixes.md)
