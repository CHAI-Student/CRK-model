# Trigger Inference Recovery Notes

## Why This Note Exists

The older product-detection documents still describe the general pipeline, but
they do not capture the latest trigger hardening and return-recovery behavior.
This note records the current operational rules.

## Inference Updates

- The preferred path is `TriggerService`, but the fallback `/trigger` route
  must behave the same way for weight-aware inference.
- Both paths now forward the live `active_products` snapshot into
  `ProductDecisionEngine.judge(...)`.
- `strict_mode` remains enabled by default.
- `strict_mode_fallback` also defaults to enabled, so strict weight misses
  degrade into the relaxed single/combo/partial path instead of immediately
  returning `NO_DETECTION`.
- `stock_qty = 0` remains a hard sold-out filter for strict matching.

## Return Recovery Updates

### Same-Zone Recovery

`ProductAggregator._handle_return()` tries, in order:

1. single-product weight rollback
2. return-combination rollback
3. unmatched-return recording

### Session-Level Repair

`DoorSessionStore._validate_net_delta()` compares the full session net delta
against the aggregated counts and repairs full-return or partial-return cases.

### Cross-Zone Repair

`DoorSessionStore._handle_cross_zone_returns()` tries to resolve unmatched
returns against other active sessions.

## Practical Debugging Checklist

1. Confirm the trigger path logged an `active_products` snapshot size.
2. Confirm the engine log shows one of:
   `strict_match`, `single_product_match`, `combination_match`,
   `partial_result`, or a reason-coded `NO_DETECTION`.
3. For return issues, inspect:
   `ProductAggregator` same-zone handling,
   then net-delta repair,
   then cross-zone repair.
4. If strict matching is too aggressive for a deployment, set
   `MODEL__WEIGHT__STRICT_MODE_FALLBACK=true` explicitly.
