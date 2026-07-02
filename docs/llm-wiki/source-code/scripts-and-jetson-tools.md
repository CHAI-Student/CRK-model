# Source Code Map: Scripts And Jetson Tools

Source: [scripts](../../../scripts), [README.md](../../../README.md)

Status: current script map

## Current Thesis

Scripts are Jetson-oriented helpers. They should not be treated as proof that a
developer PC can validate TensorRT runtime behavior.

## Scripts

- [setup_jetson.sh](../../../scripts/setup_jetson.sh):
  validates Jetson prerequisites, prepares `.venv` with
  `--system-site-packages`, loads Jetson runtime paths, installs project
  packages without replacing Jetson torch, prepares `.env`, validates imports,
  installs the activation hook, and installs the user-level `model-service`
  auto-venv launcher.
- [install_model_service_launcher.sh](../../../scripts/install_model_service_launcher.sh):
  installs `~/.local/bin/model-service`, which enters this repo, sources
  `.venv/bin/activate`, and execs `.venv/bin/model-service` with the original
  arguments. It refuses to overwrite non-owned launchers unless
  `MODEL_SERVICE_LAUNCHER_FORCE=1` is set.
- [install_jetson_torch.sh](../../../scripts/install_jetson_torch.sh):
  installs a Jetson-compatible torch/torchvision pair when the active torch
  build cannot see CUDA.
- [jetson_env.sh](../../../scripts/jetson_env.sh):
  restores CUDA, TensorRT, binary, and library paths for the current shell.
- [live_engine_preview.py](../../../scripts/live_engine_preview.py):
  standalone Jetson camera/model preview that loads an engine, opens a camera
  source, draws boxes/labels/FPS, and displays via OpenCV or ffplay.
- [convert_engine.sh](../../../scripts/convert_engine.sh):
  repo-local TensorRT engine conversion/export helper. It defaults to this
  repo's `models/`, keeps env overrides such as `PT_FILE`, `IMGSZ`,
  `MODELS_DIR`, `PROJECT_ROOT`, and `PYTHON_BIN`, verifies `yolo`, and fails
  when the active Torch build cannot see CUDA.
- [refresh_scenario_fixture.py](../../../scripts/refresh_scenario_fixture.py):
  operator-run helper that reads the source Excel workbooks, regenerates the
  committed scenario JSON fixture, and refreshes the human fixture report.
- [verify_scenario_readiness.py](../../../scripts/verify_scenario_readiness.py):
  local contract verifier for the committed scenario fixture plus optional
  configured-stride trace latency evidence summary.

## Jetson Setup Flow

```bash
chmod +x scripts/setup_jetson.sh
chmod +x scripts/install_jetson_torch.sh
chmod +x scripts/jetson_env.sh
./scripts/setup_jetson.sh
model-service
```

After setup, the user-level launcher lets a fresh shell run `model-service`
without first activating `.venv`, once `~/.local/bin` is loaded in PATH.

## Live Preview Flow

```bash
python scripts/live_engine_preview.py \
  --model models/0204_morning.engine \
  --source 0 \
  --display-backend ffplay
```

Use this only on Jetson with real camera/CUDA/TensorRT availability.

## TensorRT Engine Export Flow

```bash
source .venv/bin/activate
PT_FILE=0204_morning.pt scripts/convert_engine.sh
```

This creates the direct `.engine` artifact for the Python TensorRT service.
Do not use this page to move ONNX ownership into CRK-model; ONNX export remains
part of the `CRK-model-go` deployment flow.

## Related Wiki Pages

- [Repo overview](repo-overview.md)
- [Jetson and testing](../synthesis/jetson-and-testing.md)
- [Configuration](configuration.md)
