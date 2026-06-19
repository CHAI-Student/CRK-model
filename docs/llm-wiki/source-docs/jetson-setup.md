# Source Summary: JETSON_SETUP.md

Source: [docs/JETSON_SETUP.md](../../JETSON_SETUP.md)
Status: current with caveats

## Use This When

Use this for full Jetson Orin Nano setup, verification, and troubleshooting.

## Key Facts

- Target runtime is Jetson Orin Nano 4GB on JetPack 6.2 / Ubuntu 22.04,
  Python 3.10, JetPack CUDA/TensorRT, and a prebuilt `.engine` file.
- `scripts/setup_jetson.sh` is the recommended setup path.
- Manual setup must use `uv venv --system-site-packages --python python3.10`.
- Default engine path is `models/0204_morning.engine`.
- `model-service` bootstraps common Jetson CUDA/TensorRT paths before importing
  the app.
- Do not use `uv pip install -e ".[dev]"` on Jetson because it can pull
  CPU-only PyPI torch wheels.
- Keep NumPy on `1.x`.
- Health checks are useful on Jetson after setup, but local PC health checks do
  not prove production readiness.
- Troubleshooting covers DNS failures, CPU-only torch, missing CUDA library
  paths, missing TensorRT, NumPy 2.x, YOLO model load failures, and unwanted uv
  sync.

## Related Code

- `scripts/setup_jetson.sh`
- `scripts/install_jetson_torch.sh`
- `scripts/jetson_env.sh`
- `services/model/model_service/main.py`
- `services/model/model_service/core/config.py`

## Caveats

- This document is deployment-focused. For normal agent work, use the shorter
  [agent guide](agent-guides-jetson-setup.md) plus
  [agent-guides/conventions](agent-guides-conventions.md).

## Related Wiki Pages

- [Jetson and testing](../synthesis/jetson-and-testing.md)
