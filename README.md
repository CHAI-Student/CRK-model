# Edge Environment - Model Service

Last reviewed: 2026-03-09

This repository contains the FastAPI-based model service for the AI smart vending machine stack. The primary runtime target is Jetson Orin Nano 4GB with TensorRT `.engine` inference.

## Current Runtime Status

- FastAPI import/startup paths were decoupled from heavy YOLO and NumPy imports.
- Runtime settings are now read from `app.state.settings`, and `.env` is loaded automatically by `Settings`.
- The default engine path is now `models/siyeon_best.engine`.
- Targeted FastAPI regression tests passed on 2026-03-09:
  - `services/model/tests/test_fastapi_imports.py`
  - `services/model/tests/test_api_routes.py`
  - `services/model/tests/test_deps.py`
  - `services/model/tests/test_trigger_helpers.py`

## Jetson Quick Start

```bash
cd Edge_Environment

chmod +x scripts/setup_jetson.sh
chmod +x scripts/install_jetson_torch.sh
./scripts/setup_jetson.sh

source .venv/bin/activate
model-service
```

Health checks:

```bash
curl http://localhost:8002/api/health
curl http://localhost:8002/api/health/detailed
```

## Manual Setup

```bash
uv venv --system-site-packages --python python3.10 .venv
source .venv/bin/activate
./scripts/install_jetson_torch.sh
uv pip install --no-deps -e .
uv pip install "fastapi>=0.100.0" "uvicorn[standard]>=0.23.0" "pydantic>=2.0.0" "pydantic-settings>=2.0.0" "python-multipart>=0.0.6" "httpx>=0.24.0" "aiohttp>=3.8.0" "numpy>=1.24.0,<2.0.0" "pillow>=10.0.0" "pyyaml>=6.0.0" "requests>=2.23.0" "scipy>=1.4.1" "matplotlib>=3.3.0" "psutil>=5.8.0" "polars>=0.20.0" "ultralytics-thop>=2.0.18"
uv pip install --no-deps "ultralytics>=8.0.0,<9.0.0"
uv pip install "pytest>=7.0.0" "pytest-asyncio>=0.21.0" "pytest-cov>=4.0.0" "ruff>=0.1.0"

cp .env.example .env
```

Set at least this value in `.env`:

```bash
MODEL__VISION__YOLO_MODEL_PATH=models/siyeon_best.engine
```

If your engine file has a different name, point the variable at the actual file.

## Recommended Commands

After activation, prefer the installed entry points:

```bash
model-service
pytest services/model/tests -q
```

If you want to keep using `uv run`, prefer `--no-sync` once the environment is already prepared:

```bash
uv run --no-sync model-service
uv run --no-sync pytest services/model/tests -q
```

## AVI Split Trace

- `/trigger` still receives AVI file paths from the camera service.
- The model does not run inference on the AVI blob directly. It splits the AVI into frame images and runs YOLO per frame.
- Every trigger now writes a persistent trace entry to `services/model/logs/frame_split_YYYYMMDD.jsonl`.
- Each trace entry records `processing_mode="avi_to_frames"`, `inference_unit="image_frame"`, trigger status, video paths, and per-camera frame counts.
- Optional sample frame export is off by default. Enable it with:

```bash
MODEL__TRACE__SAMPLE_EXPORT_ENABLED=true
MODEL__TRACE__SAMPLE_COUNT_PER_CAMERA=3
MODEL__TRACE__SAMPLE_EXPORT_DIR=logs/frame_samples
```

- When enabled, sample JPEGs are written under `services/model/logs/frame_samples/<session_id>/<camera>/`.

## `uv` Notes for Jetson

- Create the venv with `--system-site-packages`. JetPack system packages provide CUDA, TensorRT, PyTorch, and often OpenCV.
- Keep NumPy on `1.x`. The project pins `numpy>=1.24.0,<2.0.0`.
- Do not use `uv pip install -e ".[dev]"` on Jetson. `ultralytics` can pull CPU-only `torch/torchvision` wheels from PyPI and break CUDA.
- `scripts/install_jetson_torch.sh` now uses the Jetson AI Lab JP6/CUDA 12.6 index at `pypi.jetson-ai-lab.io` and falls back to direct wheel URLs.
- If `torch.version.cuda` is `None`, the active torch build is CPU-only. `nvidia-cudss-cu12` does not convert it into a Jetson CUDA build.
- If `import torch` fails with `libcudss.so`, install `nvidia-cudss-cu12` after the Jetson torch wheel is in place.
- Avoid plain `uv run ...` for day-to-day execution if the environment is already installed. A sync step can unexpectedly reinstall or shadow packages in `.venv`.
- Avoid `uv sync` unless you intentionally want uv to reconcile the environment against the lock file.

## Important Files

- `services/model/model_service/main.py` - process entry point
- `services/model/model_service/api/manager.py` - FastAPI app factory and lifespan
- `services/model/model_service/core/config.py` - runtime settings and `.env` loading
- `services/model/model_service/service/trigger_service.py` - trigger workflow
- `services/model/model_service/api/routes/trigger.py` - `/trigger`
- `services/model/model_service/api/routes/multi_zone.py` - `/api/judge/*`
- `services/model/tests/` - pytest suite
- `docs/JETSON_SETUP.md` - full Jetson setup guide
- `docs/REFERENCE.md` - API reference

## Docker

Direct Jetson execution is the primary path. Docker support is available for:

- `model` - long-running FastAPI service
- `convert` - one-shot `.pt` to `.engine` conversion

Quick commands:

```bash
docker compose --profile convert run --rm convert
docker compose up -d model
docker compose logs -f model
```
