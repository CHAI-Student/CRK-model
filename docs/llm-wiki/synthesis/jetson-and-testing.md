# Jetson And Testing

## Current Thesis

For this service, local tests prove code-level regressions, not production
runtime readiness. Runtime proof requires the Jetson Orin Nano environment with
CUDA, TensorRT, the `.engine` file, and the real service wiring.
README additionally marks this Python service as the legacy/reference TensorRT
path while fresh clone-based operation should prefer `CRK-model-go`.

## Runtime Target

- Jetson Orin Nano 4GB.
- JetPack 6.2 / Ubuntu 22.04.
- Python 3.10.
- JetPack-provided CUDA and TensorRT.
- TensorRT `.engine` file, default `models/0204_morning.engine`.
- NumPy 1.x.
- `/api/health/detailed` reports best-effort NumPy, Torch CUDA, and TensorRT
  diagnostics, but this does not replace a real Jetson trigger test.

## Preferred Commands

After activating `.venv`:

```bash
model-service
pytest services/model/tests -q
```

Fallback when uv is needed:

```bash
uv run --no-sync model-service
uv run --no-sync pytest services/model/tests -q
```

Avoid plain `uv run ...` on Jetson after setup because it can trigger a sync
step and unexpectedly mutate `.venv`.

## Jetson Setup Rules

- Use `scripts/setup_jetson.sh` for one-time setup.
- Create `.venv` with `--system-site-packages`.
- Use `scripts/jetson_env.sh` or the activation hook to restore CUDA/TensorRT
  paths.
- If `torch.version.cuda` is `None`, reinstall the Jetson torch wheel with
  `scripts/install_jetson_torch.sh`.
- Do not use `uv pip install -e ".[dev]"` on Jetson because it can install
  CPU-only PyPI torch wheels.
- Use `scripts/convert_engine.sh` for repo-local TensorRT `.engine` export from
  `.pt` only on Jetson. The script checks `yolo`, Torch CUDA visibility, and
  writes under this repo's `models/` by default. ONNX export stays with
  `CRK-model-go`.
- Use host log rotation for `services/model/logs/frame_split_*.jsonl` in
  long-running deployments; YAML retention is controlled by
  `MODEL__DOOR_SESSION__YAML_RETENTION_DAYS`.

## Local Verification Rules

- Full default local gate: `pytest services/model/tests -q`.
- Trigger inference changes: include decision engine and return recovery tests.
- AVI decode/loadcell delta changes: include
  `test_trigger_pipeline_regressions.py`.
- Runtime startup/health checks on a local PC must not be described as Jetson
  verification.
- Async video failure behavior is a local unit-testable contract:
  extractor/queue/YOLO task failures must propagate as errors, while decoded
  no-product video remains a normal no-detection case.

## Evidence

- [Repo overview](../source-code/repo-overview.md)
- [Scripts and Jetson tools](../source-code/scripts-and-jetson-tools.md)
- [Tests map](../source-code/tests-map.md)
- [Conventions](../source-docs/agent-guides-conventions.md)
- [Build and test](../source-docs/agent-guides-build-test.md)
- [Jetson setup notes](../source-docs/agent-guides-jetson-setup.md)
- [Full Jetson setup guide](../source-docs/jetson-setup.md)
