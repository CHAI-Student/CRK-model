# Source Summary: REFERENCE.md

Source: [docs/REFERENCE.md](../../REFERENCE.md)
Status: current with caveats

## Use This When

Use this for the public model API surface and response shapes.

## Key Facts

- Base URL is `http://<host>:8002`.
- Health endpoints:
  `GET /api/health` and `GET /api/health/detailed`.
- Trigger endpoint:
  `POST /trigger` accepts `zone`, `loadcells`, and `videos.top/side`, and can
  return a queued session id and door-session id.
- Trigger stats endpoint:
  `GET /trigger/stats`.
- Multi-zone endpoint:
  `POST /api/judge/multi-zone` supports object payloads and array payloads.
- Session and door-session inspection endpoints exist for debugging and forced
  finalization.
- Common error codes include video path, validation, FFmpeg, GPU, and model-load
  failures.

## Related Code

- `services/model/model_service/api/routes/health.py`
- `services/model/model_service/api/routes/trigger.py`
- `services/model/model_service/api/routes/multi_zone.py`
- `services/model/model_service/session/session_store.py`
- `services/model/model_service/session/door_session_store.py`

## Caveats

- External Node behavior is better captured by
  [protocols-node](protocols-node.md). In particular, Node treats `success:
  true` as terminal and expects empty basket finals to still succeed.

## Related Wiki Pages

- [Protocol contracts](../synthesis/protocol-contracts.md)
- [Runtime flow](../synthesis/runtime-flow.md)
