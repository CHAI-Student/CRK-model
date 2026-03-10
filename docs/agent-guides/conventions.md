# Code Conventions and Runtime Defaults

## Runtime Entry Points

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
| `MODEL__VISION__YOLO_MODEL_PATH` | `models/siyeon_best.engine` |
| `MODEL__BUFFER__TTL_SECONDS` | `300` |
| `MODEL__DOOR_SESSION__SESSION_TIMEOUT_SECONDS` | `30.0` |
| `MODEL__DOOR_SESSION__WEIGHT_TOLERANCE_GRAMS` | `5.0` |
| `MODEL__DOOR_SESSION__MAX_DURATION_SECONDS` | `600.0` |

## Configuration Rules

- `Settings` auto-loads `.env`.
- Jetson venvs must be created with `--system-site-packages`.
- Keep NumPy on `1.x`.
- Avoid import-time coupling between FastAPI routes and YOLO-heavy modules.

## Project Layout

- `services/model/model_service/api/` - FastAPI layer
- `services/model/model_service/service/` - trigger workflow
- `services/model/model_service/session/` - session state
- `services/model/model_service/video/` - frame extraction and voting
- `services/model/model_service/vision/` - TensorRT YOLO wrapper
- `services/model/tests/` - pytest suite
