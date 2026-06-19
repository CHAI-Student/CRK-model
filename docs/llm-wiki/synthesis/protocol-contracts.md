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
- Node polling cadence can add 5-10 seconds of apparent latency after model
  work has finished.

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
