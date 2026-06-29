# Source Summary: agent-guides/jetson-setup.md

Source: [docs/agent-guides/jetson-setup.md](../../agent-guides/jetson-setup.md)
Status: current

## Use This When

Use this as the concise Jetson setup and recovery guide.

## Key Facts

- One-time setup should run `scripts/setup_jetson.sh` on the Jetson device.
- The helper creates `.venv` with `--system-site-packages`, repairs CPU-only
  torch if needed, installs dependencies, and adds a runtime hook to
  `.venv/bin/activate`.
- Normal runtime after setup should be:
  `source .venv/bin/activate` then `model-service`.
- Preferred day-to-day commands are `model-service` and
  `pytest services/model/tests -q`.
- Plain `uv run ...` can unexpectedly sync and mutate `.venv`; use
  `uv run --no-sync ...` only when uv is needed.
- `scripts/jetson_env.sh` restores CUDA, TensorRT, and shared library paths.
- Recovery order is activation, `source scripts/jetson_env.sh`, CUDA checks,
  then `scripts/install_jetson_torch.sh` if torch is CPU-only.
- `scripts/convert_engine.sh` is the Jetson-only repo-local TensorRT export
  helper. It checks `yolo` plus Torch CUDA visibility and writes under this
  repo's `models/` by default.
- `/api/health/detailed` exposes best-effort NumPy, Torch CUDA, and TensorRT
  diagnostics, but does not replace Jetson trigger verification.
- YAML retention is configured in the app, while `frame_split_*.jsonl` trace
  retention should be bounded with host log rotation.

## Related Code

- `scripts/setup_jetson.sh`
- `scripts/install_jetson_torch.sh`
- `scripts/jetson_env.sh`
- `scripts/convert_engine.sh`
- `services/model/model_service/main.py`

## Caveats

- This is the short operational version. The root [JETSON_SETUP](jetson-setup.md)
  summary covers the longer troubleshooting guide.

## Related Wiki Pages

- [Jetson and testing](../synthesis/jetson-and-testing.md)
