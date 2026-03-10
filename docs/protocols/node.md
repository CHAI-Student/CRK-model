# Node Protocol

## Scope

- Node repository: [Edge_Environment](../../../../Edge_Environment)
- Model service repository: [current model repo](../..)

This document describes the direct API contracts between Node and the model service.

## Direct Interfaces

| Direction | Endpoint | Purpose |
| --- | --- | --- |
| Node -> Model | `GET /api/health` | Edge health aggregation |
| Node -> Model | `POST /api/judge/multi-zone` | Door OPEN/CLOSE, polling, final basket lookup |

Node does not currently use `/api/health/detailed`.

## Node -> Model Health

- Node source: [HealthMqtt.js](../../../../Edge_Environment/server/routes/Mqtt/HealthMqtt.js)
- Model source: [health.py](../../services/model/model_service/api/routes/health.py)

Request:

```http
GET /api/health
```

Model response:

```json
{
  "model": "HEALTHY",
  "status": "ok",
  "yolo_loaded": true,
  "session_store_ready": true,
  "timestamp": 1773053049.0
}
```

Notes:

- Treat Node health MQTT as a coarse `/api/health` check only.
- In the current Node code, `edgepc_status` is not a precise model readiness signal.
- Do not use health MQTT alone to validate `/trigger` or `/api/judge/multi-zone`.

## Node -> Model Judge Flow

- Node source: [Payments.js](../../../../Edge_Environment/server/routes/RestAPI/Payments.js)
- Model source: [multi_zone.py](../../services/model/model_service/api/routes/multi_zone.py)

The request body accepts two forms.

### Object form

```json
{
  "session_id": "OPEN",
  "products": [
    {
      "product_idx": "IF11-001",
      "product_name": "Sample",
      "sale_price": 3500,
      "product_weight": "365",
      "stock_qty": 10,
      "has_loadcell": "true"
    }
  ],
  "zone": 1
}
```

### Array form

```json
[
  {
    "product_idx": "IF11-001",
    "product_name": "Sample",
    "sale_price": 3500,
    "product_weight": "365",
    "stock_qty": 10,
    "has_loadcell": "true"
  }
]
```

## `session_id` Semantics

- `OPEN`: start or keep a global door session
- `CLOSE`: process close signal, pending or finalize
- `null`: global session polling
- `zone_...`: look up a concrete inference session
- `DE...` or other general strings: treated like `device_id` in legacy fallback

## Model Response States

Processing example:

```json
{
  "success": false,
  "status": "processing",
  "message": "Waiting for YOLO inference"
}
```

Door close pending example:

```json
{
  "success": false,
  "status": "in_progress",
  "pending_close": true,
  "zones": [],
  "products": [],
  "totalInterimPrice": 0
}
```

Final success example:

```json
{
  "success": true,
  "status": "success",
  "has_products": true,
  "products": [
    {
      "productIdx": "IF11-001",
      "productId": 26,
      "name": "Sample product",
      "count": 1,
      "price": 3500,
      "confidence": 0.92
    }
  ],
  "totalPrice": 3500,
  "totalProductCount": 1
}
```

Final empty-basket example:

```json
{
  "success": true,
  "status": "complete_no_products",
  "has_products": false,
  "products": [],
  "totalPrice": 0,
  "totalProductCount": 0
}
```

## Compatibility Notes

- Node treats `success === true` as the terminal state.
- Empty basket final responses must still return `success: true`.
- Node's PNT transfer logic uses `products[].productId` as if it were `product_idx`.
- Node health logging and the real inference path are separate concerns.

## Source Of Truth

- [Edge_Environment/server/routes/Mqtt/HealthMqtt.js](../../../../Edge_Environment/server/routes/Mqtt/HealthMqtt.js)
- [Edge_Environment/server/routes/RestAPI/Payments.js](../../../../Edge_Environment/server/routes/RestAPI/Payments.js)
- [Edge_Environment/server/routes/RestAPI/PaymentStore.js](../../../../Edge_Environment/server/routes/RestAPI/PaymentStore.js)
- [model_service/api/routes/health.py](../../services/model/model_service/api/routes/health.py)
- [model_service/api/routes/multi_zone.py](../../services/model/model_service/api/routes/multi_zone.py)
