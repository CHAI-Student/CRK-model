# Payment Protocol

## Scope

- Payment repository: [CRK-PAYMENT](../../../../CRK-PAYMENT)
- Node repository: [Edge_Environment](../../../../Edge_Environment)
- Model service repository: [current model repo](../..)

This document explains the relation between the Payment service and the model service. In the current architecture, the model does not call the Payment service directly.

## Directness

- Model -> Payment: none
- Payment -> Model: none
- Actual path:
  - Node -> Payment
  - Node -> Model
  - Node combines both results before sending them to PNT

## Payment API Used By Node

Node source: [Payments.js](../../../../Edge_Environment/server/routes/RestAPI/Payments.js)

Representative endpoints:

- `GET /status`
- `POST /payment/token/approve`
- `POST /payment/token/cancel`
- `POST /payment/samsung-pay/approve`
- `POST /payment/samsung-pay/cancel`

Payment source: [manager.py](../../../../CRK-PAYMENT/src/api/manager.py)

Representative health response:

```json
{
  "status": "ok",
  "response_code": 0,
  "message": "SUCCESS"
}
```

Representative approve response:

```json
{
  "status": "Y",
  "authorization_number": "12345678",
  "authorization_date": "260123",
  "card_info": {
    "SERIAL_NUMBER": "1234567890123456",
    "ISSUER_NAME": "Issuer"
  },
  "vankey": "VANKEY1234567890ABCDEFGH",
  "response_code": 0,
  "message": "Approved"
}
```

## Model-Relevant Indirect Contract

Model inference does not go directly to Payment. Node injects the inference result into the PNT IF_08 payload in [PaymentStore.js](../../../../Edge_Environment/server/routes/RestAPI/PaymentStore.js).

Node uses these model-derived fields:

- `approve_price <- inferenceResult.totalPrice`
- `state <- inferenceResult.status === 'success' ? '0' : '1'`
- `product_idx <- inferenceResult.products[].productId`
- `product_count <- inferenceResult.products[].count`

Node also attaches one `.mp4` file from `archival/cam_0` as `paymentFile`.

## Implications For The Model Team

- Payment issues may not reproduce inside the model process itself.
- For payment-side incidents, check Node's `inferenceResult` handling and PNT payload mapping first.
- If the model response shape changes, [PaymentStore.js](../../../../Edge_Environment/server/routes/RestAPI/PaymentStore.js) is likely to break before the Payment service does.

## Source Of Truth

- [CRK-PAYMENT/src/api/manager.py](../../../../CRK-PAYMENT/src/api/manager.py)
- [Edge_Environment/server/routes/RestAPI/Payments.js](../../../../Edge_Environment/server/routes/RestAPI/Payments.js)
- [Edge_Environment/server/routes/RestAPI/PaymentStore.js](../../../../Edge_Environment/server/routes/RestAPI/PaymentStore.js)
