# Source Summary: protocols/node.md

Source: [docs/protocols/node.md](../../protocols/node.md)
Status: current

## Use This When

Use this for Edge_Environment/Node contracts with the model.

## Key Facts

- Direct Node-to-Model interfaces are `GET /api/health` and
  `POST /api/judge/multi-zone`.
- Node does not currently use `/api/health/detailed`.
- Node health MQTT is a coarse health check and must not be used alone to
  validate `/trigger` or `/api/judge/multi-zone`.
- Multi-zone request accepts object form with `session_id`, `products`, and
  `zone`, or array form with products directly.
- `session_id` semantics:
  `OPEN` starts/keeps a global door session,
  `CLOSE` processes close/finalization,
  `null` polls,
  `zone_...` looks up a concrete inference session,
  other strings can act like legacy device ids.
- Response states include processing, close pending, final success, and final
  empty basket.
- Node treats `success === true` as terminal, so empty final baskets must still
  return `success: true`.
- Node's PNT transfer logic uses `products[].productId` as if it were
  `product_idx`.

## Related Code

- `Edge_Environment/server/routes/Mqtt/HealthMqtt.js`
- `Edge_Environment/server/routes/RestAPI/Payments.js`
- `Edge_Environment/server/routes/RestAPI/PaymentStore.js`
- `services/model/model_service/api/routes/health.py`
- `services/model/model_service/api/routes/multi_zone.py`

## Caveats

- Node polling cadence can add perceived latency after the model result is
  ready. Separate model-side latency from Node polling delay when debugging.

## Related Wiki Pages

- [Protocol contracts](../synthesis/protocol-contracts.md)
- [Latency and frame stride](../synthesis/latency-and-frame-stride.md)
