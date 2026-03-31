# Jetson Orin Nano Setup Guide

Last reviewed: 2026-03-31

This guide is for the direct Jetson runtime, not the Windows dev environment.

## Target Environment

| Item | Required |
|------|----------|
| Device | Jetson Orin Nano 4GB |
| OS | JetPack 6.2 / Ubuntu 22.04 |
| Python | 3.10 |
| CUDA | JetPack-provided |
| TensorRT | JetPack-provided |
| Inference artifact | `.engine` file already built on Jetson |

## Before You Start

Verify the platform:

```bash
cat /etc/nv_tegra_release
python3.10 --version
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

Install FFmpeg if needed:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

## Recommended Setup

```bash
cd ~/Edge_Environment

chmod +x scripts/setup_jetson.sh
chmod +x scripts/install_jetson_torch.sh
chmod +x scripts/jetson_env.sh
./scripts/setup_jetson.sh

source .venv/bin/activate
model-service
```

The setup script now does all of the following:

- verifies Jetson, CUDA, PyTorch, and TensorRT visibility
- creates `.venv` with `--system-site-packages`
- installs a Jetson-compatible `torch/torchvision` pair into `.venv` if CUDA is missing
- installs the project in editable mode without letting `uv` replace Jetson torch
- forces NumPy back to `1.x` if needed
- creates `.env` from `.env.example` if missing
- checks that the `model-service` entry point exists
- installs a `.venv/bin/activate` hook that restores Jetson CUDA/TensorRT paths
  in every future shell

## Manual Setup

```bash
cd ~/Edge_Environment

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

Set the engine path in `.env`:

```bash
MODEL__VISION__YOLO_MODEL_PATH=models/siyeon_best.engine
```

If your engine file uses a different name, point the variable at the real file instead.

## Runtime Notes

- `.env` is now loaded automatically by `model_service.core.config.Settings`.
- The default engine path in code is `models/siyeon_best.engine`.
- FastAPI startup now fails fast if the TensorRT engine cannot be loaded.
- Startup errors now include the root cause from the YOLO loader. If CUDA is unavailable, the message says so instead of looking like a pure path problem.
- `model-service` now bootstraps the common Jetson CUDA/TensorRT runtime paths
  before importing the app. After the one-time setup, a fresh terminal should
  only need `source .venv/bin/activate` followed by `model-service`.
- `scripts/jetson_env.sh` remains available as a manual recovery command when
  you want to inspect CUDA visibility inside the shell itself.

## Recommended Runtime Commands

After activating the venv:

```bash
model-service
pytest services/model/tests/test_fastapi_imports.py -q
```

If you want to use `uv run`, avoid automatic re-sync:

```bash
uv run --no-sync model-service
uv run --no-sync pytest services/model/tests -q
```

## Why `--system-site-packages` Matters

JetPack already provides GPU-aware builds of key packages. The Jetson venv should inherit them instead of trying to replace them.

Without `--system-site-packages`, you can easily lose access to:

- CUDA-enabled PyTorch
- TensorRT Python bindings
- system OpenCV builds

## Torch Rule

Do not use `uv pip install -e ".[dev]"` on Jetson. The `ultralytics` dependency chain can install CPU-only `torch/torchvision` wheels from PyPI into `.venv`.

If `torch.version.cuda` prints `None`, the active torch build is CPU-only:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Fix it with the helper:

```bash
./scripts/install_jetson_torch.sh
```

The helper defaults to the Jetson AI Lab JP6 / CUDA 12.6 index and downloads exact `torch` / `torchvision` wheels before installing them into `.venv`.

The helper now targets `https://pypi.jetson-ai-lab.io/jp6/cu126` and falls back to direct wheel URLs if the simple index path fails.

`nvidia-cudss-cu12` is not a substitute for a Jetson CUDA-enabled torch build. If `torch.version.cuda` is `None`, reinstall torch first. Only install `nvidia-cudss-cu12` if `import torch` later fails with `libcudss.so`.

If the helper reports `Name or service not known`, fix Jetson networking before retrying:

```bash
ping -c 1 8.8.8.8
getent hosts pypi.jetson-ai-lab.io
```

## NumPy Rule

Keep NumPy on `1.x`:

```bash
python -c "import numpy; print(numpy.__version__)"
uv pip install "numpy>=1.24.0,<2.0.0" --force-reinstall
```

The service code was updated to avoid import-time crashes from broken NumPy environments, but Jetson runtime should still stay on NumPy 1.x.

## Engine Verification

List engine files:

```bash
find models -maxdepth 1 -name '*.engine' -print
```

Try a direct Ultralytics load:

```bash
python - <<'PY'
from ultralytics import YOLO

model = YOLO("models/siyeon_best.engine")
print(model.names)
PY
```

## Health Checks

```bash
curl http://localhost:8002/api/health
curl http://localhost:8002/api/health/detailed
```

`/api/health/detailed` now reports the runtime host, port, and model path from the app settings, not stale import-time defaults.

## Systemd Example

```ini
[Unit]
Description=Model Service
After=network.target

[Service]
Type=simple
User=jetson
WorkingDirectory=/home/jetson/Edge_Environment
Environment=PYTHONUNBUFFERED=1
ExecStart=/home/jetson/Edge_Environment/.venv/bin/model-service
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

## Troubleshooting

| Problem | Cause | Action |
|---------|-------|--------|
| `Name or service not known` while downloading torch | DNS failure on Jetson | verify network and `getent hosts pypi.jetson-ai-lab.io`, then rerun the helper |
| `torch.cuda.is_available()` is `False` and `torch.version.cuda` is `None` | CPU-only PyPI torch is installed in `.venv` | run `./scripts/install_jetson_torch.sh`; rerun `./scripts/setup_jetson.sh` only if the venv entry points or activation hook also need to be rebuilt |
| `torch.cuda.is_available()` is `False` but `torch.version.cuda` is set | CUDA libraries are not visible to torch | first retry in a fresh shell with `source .venv/bin/activate && model-service`; if shell tools still cannot see CUDA, run `source scripts/jetson_env.sh` and recheck |
| `model-service` works only after manually rerunning `./scripts/setup_jetson.sh` | Jetson runtime paths were not restored in the new shell | rerun `./scripts/setup_jetson.sh` once to reinstall the activation hook, then use only `source .venv/bin/activate` in later shells |
| `import torch` fails with `libcudss.so` | cuDSS runtime missing for the Jetson wheel | run `pip install nvidia-cudss-cu12` inside `.venv`, then retry |
| `No module named tensorrt` | TensorRT Python bindings unavailable | verify JetPack install and Python path |
| `numpy.core.multiarray failed to import` | NumPy 2.x or mixed packages | reinstall NumPy 1.x in `.venv` |
| `YOLO model load failed` | startup wrapper includes the real root cause now | read the `Root cause:` suffix first, then verify the engine path or CUDA stack |
| `uv run` tries to change the venv | sync step triggered | use `model-service` directly or `uv run --no-sync` |

## Direct Test Commands

```bash
pytest services/model/tests/test_fastapi_imports.py -q
pytest services/model/tests/test_api_routes.py -q
pytest services/model/tests/test_deps.py -q
pytest services/model/tests/test_trigger_helpers.py -q
```
