# Source Summary: agent-guides/architecture.md

Source: [docs/agent-guides/architecture.md](../../agent-guides/architecture.md)
Status: current

## Use This When

Use this as the first source for the current trigger-inference architecture.

## Key Facts

- The model runtime flow is:
  `Edge_Environment/CRK-CAMERA -> /trigger -> TriggerService -> video ->
  ProductDecisionEngine -> DoorSessionStore -> ProductAggregator`.
- `TriggerService` is the preferred async orchestration path and owns queueing,
  deduplication, and trace logging.
- `ProductDecisionEngine.judge()` combines vision candidates, `delta_weight`,
  and live `active_products`.
- Fatal async video decode/queue/YOLO task failures propagate as
  `VideoProcessingError` or related model-service exceptions; expected-frame
  zero-decode after retry must not become a normal no-detection or loadcell-only
  fallback result.
- `active_products` from Node is the supported source for strict/loadcell-only
  matching.
- `stock_qty = 0` is a hard sold-out filter for strict matching.
- `strict_mode_fallback=true` lets strict misses continue into relaxed
  matching instead of immediately returning `NO_DETECTION`.
- Return repair has three layers: same-zone `ProductAggregator`, session-level
  `DoorSessionStore._validate_net_delta()`, and cross-zone return repair.

## Related Code

- `services/model/model_service/api/routes/trigger.py`
- `services/model/model_service/service/trigger_service.py`
- `services/model/model_service/video/`
- `services/model/model_service/engine/decision_engine.py`
- `services/model/model_service/session/door_session_store.py`
- `services/model/model_service/session/product_aggregator.py`

## Caveats

- This guide is intentionally scoped to the trigger-inference path. Use protocol
  docs for external service contracts and Jetson docs for deployment details.

## Related Wiki Pages

- [Runtime flow](../synthesis/runtime-flow.md)
- [Product detection pipeline](../synthesis/product-detection-pipeline.md)
- [Protocol contracts](../synthesis/protocol-contracts.md)
