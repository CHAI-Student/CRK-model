# Edge Environment - Model Service

Last reviewed: 2026-03-31

This repository contains the legacy/reference FastAPI-based model service for the AI smart vending machine stack. The runtime target for this Python service remains the real Jetson Orin Nano 4GB device on Ubuntu 22.04 with TensorRT `.engine` inference.

For new clone-based operation, use `CRK-model-go` as the primary model service. The Go service standardizes on ONNX Runtime CUDA with `0204_morning.onnx`; if only `0204_morning.pt` is available, the Go repo's Jetson run script can export FP16 ONNX before starting the Go binary. This avoids per-device TensorRT engine conversion during normal deployment.

Operational checks should be run on the Jetson device, not on a developer PC. Local imports or unit tests can catch syntax-level regressions, but service startup, health, AVI decoding, TensorRT loading, and trigger behavior are only authoritative on the Jetson runtime.

## Current Runtime Status

- This Python service is maintained as the TensorRT `.engine` legacy/reference path.
- The primary operational path for fresh installs is `CRK-model-go` with ONNX.
- This repo may still provide the Jetson Python/Ultralytics environment used by `CRK-model-go/scripts/run-jetson-native.sh` to export `models/0204_morning.pt` into FP16 `models/0204_morning.onnx`.
- Camera sends two physical loadcell channels per vending zone; the model service sums those channels into the zone total and does not average them.
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
chmod +x scripts/jetson_env.sh
./scripts/setup_jetson.sh

source .venv/bin/activate
model-service
```

`./scripts/setup_jetson.sh` is a one-time environment preparation step. After it
completes, opening a new terminal only needs:

```bash
cd Edge_Environment
source .venv/bin/activate
model-service
```

The setup script installs a small activation hook into `.venv/bin/activate`.
That hook restores the Jetson CUDA/TensorRT runtime paths automatically for
future shells, and `model-service` itself now performs the same bootstrap
before importing the service stack. In practice, a fresh terminal should no
longer require rerunning `./scripts/setup_jetson.sh`.

Health checks:

```bash
curl http://localhost:8002/api/health
curl http://localhost:8002/api/health/detailed
```

`/api/health` is ready only when `model` is `HEALTHY`, `yolo_loaded` is `true`, and `status` is `ok`.

## Live Engine Preview

To visually verify camera input and TensorRT inference on the Jetson, use the
standalone preview script. It loads a `.engine` file, opens a live camera source,
and draws real-time bbox labels, confidence, and FPS in an OpenCV window.

```bash
cd CRK-model
source .venv/bin/activate
python scripts/live_engine_preview.py --model models/siyeon_best.engine --source 0
```

Useful options:

```bash
python scripts/live_engine_preview.py \
  --model models/siyeon_best.engine \
  --source 0 \
  --width 640 \
  --height 480 \
  --imgsz 480 \
  --conf 0.25
```

The preview is a Jetson-only operational check. Do not use a developer PC run
to judge TensorRT, camera, or CUDA readiness.

## Manual Setup

```bash
uv venv --system-site-packages --python python3.10 .venv
source .venv/bin/activate
source scripts/jetson_env.sh
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

`model-service` now bootstraps the common Jetson CUDA/TensorRT library paths on
startup, so it does not depend on rerunning `setup_jetson.sh` in each shell.

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

## PT to ONNX Export Role

`CRK-model-go` does not load `.pt` directly. When `models/0204_morning.onnx`
is missing but `models/0204_morning.pt` exists, the Go repo's Jetson run script
can call this Python environment to run Ultralytics export with:

```bash
format=onnx
imgsz=480
half=True
```

CUDA is required and FP32 fallback is intentionally disabled. The output remains
owned by `CRK-model-go` as `models/0204_morning.onnx`; this Python repo remains
the legacy TensorRT `.engine` service for direct Python runtime use.

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
