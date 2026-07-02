# Source Summary: agent-guides/conventions.md

Source: [docs/agent-guides/conventions.md](../../agent-guides/conventions.md)
Status: current

## Use This When

Use this for current runtime defaults, configuration rules, and repo layout.

## Key Facts

- Runtime verification is Jetson-only.
- Preferred commands after venv activation:
  `model-service` and `pytest services/model/tests -q`.
- Important defaults include:
  `MODEL__API__PORT=8002`,
  `MODEL__VISION__YOLO_MODEL_PATH=models/0204_morning.engine`,
  `MODEL__DOOR_SESSION__CLOSE_INITIAL_WAIT_SECONDS=3.0`,
  `MODEL__DOOR_SESSION__CLOSE_SUBSEQUENT_WAIT_SECONDS=1.0`,
  `MODEL__VIDEO__READY_MAX_WAIT_SECONDS=2.0`,
  `MODEL__VIDEO__READY_POLL_INTERVAL_SECONDS=0.2`,
  `MODEL__ASYNC_STREAMING__FRAME_STRIDE=1`,
  `MODEL__WEIGHT__STRICT_MODE=true`, and
  `MODEL__WEIGHT__STRICT_MODE_FALLBACK=true`.
- `Settings` auto-loads `.env`.
- Jetson venvs must use `--system-site-packages`; NumPy should stay on `1.x`.
- Every weight-aware path should receive the Node `active_products` snapshot.

## Related Code

- `services/model/model_service/core/config.py`
- `services/model/model_service/main.py`
- `services/model/model_service/api/`
- `services/model/model_service/service/`
- `services/model/model_service/video/`
- `services/model/model_service/vision/`
- `services/model/tests/`

## Caveats

- Defaults can drift when `.env` overrides are used on Jetson. Treat this as
  the code-default map, then check deployment env for live behavior.

## Related Wiki Pages

- [Jetson and testing](../synthesis/jetson-and-testing.md)
- [Latency and frame stride](../synthesis/latency-and-frame-stride.md)
