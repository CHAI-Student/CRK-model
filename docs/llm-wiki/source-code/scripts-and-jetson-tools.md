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
  and installs the activation hook.
- [install_jetson_torch.sh](../../../scripts/install_jetson_torch.sh):
  installs a Jetson-compatible torch/torchvision pair when the active torch
  build cannot see CUDA.
- [jetson_env.sh](../../../scripts/jetson_env.sh):
  restores CUDA, TensorRT, binary, and library paths for the current shell.
- [live_engine_preview.py](../../../scripts/live_engine_preview.py):
  standalone Jetson camera/model preview that loads an engine, opens a camera
  source, draws boxes/labels/FPS, and displays via OpenCV or ffplay.
- [convert_engine.sh](../../../scripts/convert_engine.sh):
  helper for TensorRT engine conversion/export workflows.
- [refresh_scenario_fixture.py](../../../scripts/refresh_scenario_fixture.py):
  operator-run helper that reads the source Excel workbooks, regenerates the
  committed scenario JSON fixture, and refreshes the human fixture report.
- [verify_scenario_readiness.py](../../../scripts/verify_scenario_readiness.py):
  local contract verifier for the committed scenario fixture plus optional
  stride-2 trace latency evidence summary.

## Jetson Setup Flow

```bash
chmod +x scripts/setup_jetson.sh
chmod +x scripts/install_jetson_torch.sh
chmod +x scripts/jetson_env.sh
./scripts/setup_jetson.sh
source .venv/bin/activate
model-service
```

## Live Preview Flow

```bash
python scripts/live_engine_preview.py \
  --model models/0204_morning.engine \
  --source 0 \
  --display-backend ffplay
```

Use this only on Jetson with real camera/CUDA/TensorRT availability.

## Related Wiki Pages

- [Repo overview](repo-overview.md)
- [Jetson and testing](../synthesis/jetson-and-testing.md)
- [Configuration](configuration.md)
