# Source Summary: TRIGGER_INFERENCE_RECOVERY_NOTES_2026-03-31.md

Source: [docs/TRIGGER_INFERENCE_RECOVERY_NOTES_2026-03-31.md](../../TRIGGER_INFERENCE_RECOVERY_NOTES_2026-03-31.md)
Status: current with caveats

## Use This When

Use this for the latest documented trigger hardening and return-recovery rules.

## Key Facts

- `TriggerService` is the preferred path, but fallback `/trigger` should behave
  the same for weight-aware inference.
- Both paths must forward live `active_products` into
  `ProductDecisionEngine.judge(...)`.
- `strict_mode` remains enabled by default.
- `strict_mode_fallback` also defaults to enabled, so strict weight misses can
  degrade into relaxed single/combo/partial paths.
- `stock_qty = 0` remains a sold-out filter for strict matching.
- Same-zone return recovery tries single-product rollback, then return
  combination rollback, then records unmatched returns.
- `DoorSessionStore._validate_net_delta()` repairs full-return or
  partial-return cases at the session level.
- `DoorSessionStore._handle_cross_zone_returns()` can match unmatched returns
  against other active sessions.

## Related Code

- `services/model/model_service/api/routes/trigger.py`
- `services/model/model_service/service/trigger_service.py`
- `services/model/model_service/engine/decision_engine.py`
- `services/model/model_service/session/product_aggregator.py`
- `services/model/model_service/session/door_session_store.py`

## Caveats

- Later detected-single fallback and frame-stride telemetry are not covered in
  this March note. See latency and decision-recovery synthesis for those.

## Related Wiki Pages

- [Runtime flow](../synthesis/runtime-flow.md)
- [Historical risk and fixes](../synthesis/historical-risk-and-fixes.md)
- [Latency and frame stride](../synthesis/latency-and-frame-stride.md)
