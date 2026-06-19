# Protocol Contracts

## Current Thesis

The model has only two direct external API relationships: Camera submits
triggers, and Node controls/polls door-session judgment. IO Board and Payment
are indirect dependencies through Camera and Node.

## Camera To Model: `/trigger`

Camera sends:

- `zone`
- `loadcells`
- `videos.top`
- `videos.side`
- optional capture/loadcell timing metadata

Important behavior:

- Video fields are shared filesystem paths.
- Model validates/uses the AVI paths and computes `delta_weight`.
- Refrigerated and freezer cabinets both use the zone-sliced `loadcells`
  payload as decision input. `global_loadcells` may appear for backward
  compatibility, but CRK-model ignores it for effective freezer/refrigerated
  weight calculation.
- In freezer `dual_top_proxy` deployments, the public payload remains
  `videos.top` and `videos.side`; internally those are treated as top-middle
  and top-side streams for lower-half dual-top filtering.
- Typical statuses include `queued`, `complete`, `skipped`, and `duplicate`.
- Camera does not retry failed `/trigger` calls according to the protocol note.

## Node To Model: `/api/judge/multi-zone`

Node sends:

- `session_id="OPEN"` to start or keep a global door session.
- `session_id="CLOSE"` to initiate close/finalization.
- `session_id=null` for polling.
- `session_id="zone_..."` to look up a concrete inference session.
- `products` as the live active product snapshot.

Important behavior:

- Node treats `success === true` as terminal.
- Empty basket finals must still return `success: true`.
- `products[].productId` is used downstream like `product_idx`.
- Model runtime class identity uses Edge `product_eng_name` matched against
  the current YOLO engine class name. During the Edge migration, an
  engine-matching `name` field and legacy `product_name` are also accepted
  after `product_eng_name`. Direct `trainingidx`/`training_idx`/`trainingIdx`/
  `yolo_class_id` and `yolo_class_name` fields may be present for
  compatibility but are ignored for active-product class identity.
  `product_idx` is preserved for Node-facing identity but is not the stable
  model class key.
- Startup logs emitted by `MODEL__VISION__LOG_ENGINE_CLASSES=on` use
  `[OPS][YOLO-ENGINE-CLASS] id=... name=...`; that `name=` token is the log
  label for the engine class string, not a required payload field.
- Node polling cadence can add 5-10 seconds of apparent latency after model
  work has finished.

## External Follow-Up Risks

- Edge/Camera/IO service URLs are still hardcoded in sibling repos in several
  places; changing deployment topology can break integration before model code
  is reached.
- Camera still needs a retry/repost path when model returns
  `waiting_for=stable_loadcell`; CRK-model can preserve the waiting reason, but
  cannot create a new stable loadcell tail by itself.
- IO Board protocol documentation has stale items relative to current runtime
  wiring, so loadcell schema/timing investigations should verify the actual
  Camera and IO Board code before changing model assumptions.

## IO Board Indirect Contract

- Model does not parse raw IO Board APIs.
- Camera subscribes to IO Board SSE and packages loadcell data into `/trigger`.
- If loadcell event schema changes, check Camera `loadcell.py` before changing
  model code.

## Payment Indirect Contract

- Model does not call Payment.
- Node combines `inferenceResult.totalPrice`, `status`, `products[].productId`,
  and `products[].count` into payment/PNT payloads.
- Model response-shape changes can break Node/Payment integration even if the
  model process is healthy.

## Evidence

- [API routes](../source-code/api-routes.md)
- [Session and persistence](../source-code/session-and-persistence.md)
- [API reference](../source-docs/reference.md)
- [Camera protocol](../source-docs/protocols-camera.md)
- [Node protocol](../source-docs/protocols-node.md)
- [IO Board protocol](../source-docs/protocols-io-board.md)
- [Payment protocol](../source-docs/protocols-payment.md)
