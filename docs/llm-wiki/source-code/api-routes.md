# Source Code Map: API Routes

Source: [api/manager.py](../../../services/model/model_service/api/manager.py),
[routes/health.py](../../../services/model/model_service/api/routes/health.py),
[routes/trigger.py](../../../services/model/model_service/api/routes/trigger.py),
[routes/multi_zone.py](../../../services/model/model_service/api/routes/multi_zone.py)

Status: current API map

## Current Thesis

The model API is small but stateful. `/trigger` accepts Camera inference work,
while `/api/judge/multi-zone` is the Node-facing door-session control and
polling surface.

## Endpoints

| Endpoint | Code | Primary caller | Purpose |
| --- | --- | --- | --- |
| `GET /` | `api/manager.py` | humans/health tools | Root metadata. |
| `GET /api/health` | `routes/health.py` | Node health aggregation | Coarse health and YOLO readiness. |
| `GET /api/health/detailed` | `routes/health.py` | operators/tests | Dependency and runtime config snapshot. |
| `POST /trigger` | `routes/trigger.py` | CRK-CAMERA | Submit zone, loadcell samples, top/side AVI paths. |
| `GET /trigger/stats` | `routes/trigger.py` | operators/tests | Trigger worker stats. |
| `POST /api/judge/multi-zone` | `routes/multi_zone.py` | Node / Edge_Environment | OPEN/CLOSE/polling and product snapshot ingestion. |
| `GET /api/judge/session/{session_id}` | `routes/multi_zone.py` | debug | Inspect one trigger session. |
| `GET /api/judge/sessions/stats` | `routes/multi_zone.py` | debug | Inspect `SessionStore` metrics. |
| `GET /api/judge/door-sessions/stats` | `routes/multi_zone.py` | debug | Inspect `DoorSessionStore` metrics. |
| `GET /api/judge/door-session/{zone}` | `routes/multi_zone.py` | debug | Inspect active zone door session. |
| `POST /api/judge/door-session/{zone}/finalize` | `routes/multi_zone.py` | debug/manual ops | Force door-session finalization. |

## `/trigger` Behavior

- Validates video paths and loadcell payload shape.
- Computes/analyzes weight delta through shared loadcell helpers.
- Uses `/trigger.loadcells` as the effective loadcell payload for every
  cabinet type. `global_loadcells` is accepted only as a deprecated
  compatibility field and does not change decision input.
- Creates a `TriggerTraceContext`.
- Uses `TriggerService` when available; fallback route still performs video,
  decision, and session storage so the behavior stays weight-aware.
- Captures the effective active-product snapshot for inference. If current
  Node inventory context was cleared by door close, a fresh last-valid snapshot
  can provide `allowed_class_ids`, product weights, and trace diagnostics.
- Can skip reliable low-weight events, use vision-only fallback for uncertain
  low-weight video, or enqueue normal worker processing.

## Health Diagnostics

- `/api/health` remains the coarse readiness endpoint.
- `/api/health/detailed` reports initialized dependency flags, runtime host,
  port, model path, and best-effort import/runtime diagnostics for NumPy, Torch
  CUDA visibility, and TensorRT.
- Missing Torch/TensorRT on a local development host should appear as
  diagnostic `error` strings rather than failing the health route import.

## `/api/judge/multi-zone` Behavior

- Accepts object-form or array-form products.
- Stores product snapshots in `ActiveProductStore`. Stock-positive payloads
  need an engine class-name key so the store can match it to the loaded YOLO
  engine class names. Official input is `product_eng_name`; during migration,
  engine-matching `name` and legacy `product_name` are accepted after
  `product_eng_name`. `trainingidx`, `yolo_class_id`, and `yolo_class_name`
  are accepted for API compatibility but ignored for active-product class
  identity. Valid stock-positive, positive-weight payloads update both current
  and last-valid snapshots; CLOSE cleanup clears current data while preserving
  the bounded last-valid fallback.
- Interprets `session_id` as door state:
  `OPEN`, `CLOSE`, `null` polling, `zone_...` lookup, or legacy device id.
- Handles close debounce through `DoorSessionStore.handle_close_signal()`.
- Builds final Node-compatible products, totals, status, and decision summary.
- Zone `weightDelta` and `decisionSummary.totalWeightDelta` use effective
  basket delta after cross-zone return repair; raw trigger deltas remain in
  door-session diagnostics.
- Writes request/response logs through a bounded `ThreadPoolExecutor`.

## API Compatibility Rules

- Node treats `success === true` as terminal.
- Empty basket finals still need `success: true`.
- Product response fields include both `productIdx` and `productId`.
- Do not change response shape without checking Node `Payments.js` and
  `PaymentStore.js`.

## Related Wiki Pages

- [Protocol contracts](../synthesis/protocol-contracts.md)
- [Runtime flow](../synthesis/runtime-flow.md)
- [Loadcell and trigger](loadcell-and-trigger.md)
