# Jetson Setup Notes

Use this when the task is about direct Jetson runtime behavior.

## Required Shape

```bash
uv venv --system-site-packages --python python3.10 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Runtime Rule

Prefer:

```bash
model-service
pytest services/model/tests -q
```

Acceptable fallback:

```bash
uv run --no-sync model-service
uv run --no-sync pytest services/model/tests -q
```

Avoid plain `uv run ...` for normal execution after setup, because a sync step can unexpectedly modify `.venv`.

## Settings Rule

- `.env` is auto-loaded by `Settings`.
- Default engine path is `models/siyeon_best.engine`.
- Override with `MODEL__VISION__YOLO_MODEL_PATH` when the engine file name differs.

## Compatibility Rule

- Jetson runtime must stay on NumPy 1.x.
- `--system-site-packages` is required to inherit CUDA, TensorRT, and JetPack-provided PyTorch.
