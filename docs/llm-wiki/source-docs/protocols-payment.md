# Source Summary: protocols/payment.md

Source: [docs/protocols/payment.md](../../protocols/payment.md)
Status: current

## Use This When

Use this to understand the indirect relation between model inference and
payment/PNT payloads.

## Key Facts

- The model does not call Payment directly, and Payment does not call the model.
- Actual path is Node to Payment, Node to Model, then Node combines payment and
  inference results before sending PNT payloads.
- Node payment endpoints include status, token approve/cancel, and Samsung Pay
  approve/cancel.
- Node injects inference fields into PNT IF_08:
  `approve_price <- inferenceResult.totalPrice`,
  `state <- inferenceResult.status === 'success' ? '0' : '1'`,
  `product_idx <- inferenceResult.products[].productId`,
  `product_count <- inferenceResult.products[].count`.
- Node attaches one archival `.mp4` from `archival/cam_0` as `paymentFile`.
- Model response shape changes can break Node/Payment integration even when the
  model itself still works.

## Related Code

- `CRK-PAYMENT/src/api/manager.py`
- `Edge_Environment/server/routes/RestAPI/Payments.js`
- `Edge_Environment/server/routes/RestAPI/PaymentStore.js`
- `services/model/model_service/api/routes/multi_zone.py`

## Caveats

- Payment incidents may need Node/PNT investigation first, not model-process
  debugging.

## Related Wiki Pages

- [Protocol contracts](../synthesis/protocol-contracts.md)
- [System map](../synthesis/system-map.md)
