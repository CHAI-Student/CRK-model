# Code Conventions and Runtime Defaults

## Runtime Entry Points

Runtime verification is Jetson-only. The production target is the real Jetson
Orin Nano on Ubuntu 22.04, so do not treat local PC service startup, health
checks, AVI decoding, or TensorRT behavior as authoritative.

Preferred after activating `.venv`:

```bash
model-service
pytest services/model/tests -q
```

Fallback with uv:

```bash
uv run --no-sync model-service
uv run --no-sync pytest services/model/tests -q
```

## Key Runtime Defaults

| Variable | Default |
|----------|---------|
| `MODEL__API__HOST` | `0.0.0.0` |
| `MODEL__API__PORT` | `8002` |
| `MODEL__VISION__YOLO_MODEL_PATH` | `models/0204_morning.engine` |
| `MODEL__BUFFER__TTL_SECONDS` | `300` |
| `MODEL__DOOR_SESSION__SESSION_TIMEOUT_SECONDS` | `30.0` |
| `MODEL__DOOR_SESSION__WEIGHT_TOLERANCE_GRAMS` | `5.0` |
| `MODEL__DOOR_SESSION__MAX_DURATION_SECONDS` | `600.0` |
| `MODEL__DOOR_SESSION__CLOSE_INITIAL_WAIT_SECONDS` | `3.0` |
| `MODEL__DOOR_SESSION__CLOSE_SUBSEQUENT_WAIT_SECONDS` | `1.0` |
| `MODEL__VIDEO__READY_MAX_WAIT_SECONDS` | `2.0` |
| `MODEL__VIDEO__READY_POLL_INTERVAL_SECONDS` | `0.2` |
| `MODEL__ASYNC_STREAMING__FRAME_STRIDE` | `2` |
| `MODEL__WEIGHT__STRICT_MODE` | `true` |
| `MODEL__WEIGHT__STRICT_MODE_FALLBACK` | `true` |

## Configuration Rules

- `Settings` auto-loads `.env`.
- Jetson venvs must be created with `--system-site-packages`.
- Keep NumPy on `1.x`.
- Avoid import-time coupling between FastAPI routes and YOLO-heavy modules.
- Forward the Node.js `active_products` snapshot into every weight-aware path.
- Treat `delta_weight < 0` as removal and `delta_weight > 0` as return.
- Keep `stock_qty = 0` as a real sold-out signal for strict matching.
- Default strict misses should degrade to relaxed matching unless ops explicitly disable `MODEL__WEIGHT__STRICT_MODE_FALLBACK`.

## Project Layout

- `services/model/model_service/api/` - FastAPI layer
- `services/model/model_service/service/` - trigger workflow
- `services/model/model_service/session/` - session state
- `services/model/model_service/video/` - frame extraction and voting
- `services/model/model_service/vision/` - TensorRT YOLO wrapper
- `services/model/tests/` - pytest suite
