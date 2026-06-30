# Source Code Map: Repo Overview

Source: [README.md](../../../README.md),
[pyproject.toml](../../../pyproject.toml),
[dataset.yaml](../../../dataset.yaml),
[.env.example](../../../.env.example)

Status: current repo-level map

## Current Thesis

This repository is the Python FastAPI/TensorRT model service for the CRK smart
vending stack. The README now frames it as the legacy/reference TensorRT
`.engine` path, while fresh clone-based operation is expected to prefer
`CRK-model-go` with ONNX Runtime CUDA.

## Runtime Role

- Direct Python service entry point: `model-service`, mapped to
  `model_service.main:run` in `pyproject.toml`.
- Runtime target: Jetson Orin Nano 4GB, JetPack 6.2 / Ubuntu 22.04,
  Python 3.10, TensorRT `.engine` inference.
- Default engine path: `models/0204_morning.engine`.
- Local PC startup, local health checks, and local TensorRT behavior are not
  authoritative runtime verification.
- This Python repo can still provide the Python/Ultralytics environment used by
  `CRK-model-go` when exporting `.pt` to FP16 ONNX.
- Direct TensorRT `.engine` export for this Python service uses
  `scripts/convert_engine.sh`, which defaults to this repo's `models/` and
  fails fast if Torch cannot see CUDA.

## Package And Dependency Shape

- Project name: `ai-smart-vending`.
- Version: `5.4.0`.
- Python range: `>=3.10,<3.12`.
- Build backend: Hatchling.
- Package root: `services/model/model_service`, exposed as `model_service`.
- Core runtime dependencies: FastAPI, Uvicorn, Pydantic v2, pydantic-settings,
  HTTP clients, NumPy `<2.0`, Pillow, Ultralytics, PyYAML.
- Dev/test extras add pytest, pytest-asyncio, pytest-cov, and ruff.

## Dataset And Class Map

- `dataset.yaml` declares `nc: 133`.
- Class `0` is `hand`.
- Product classes run from `1` through `132`.
- Optional static validation can compare engine names, `dataset.yaml`, and
  `config/yolo_product_mapping.json` when explicitly enabled. Runtime
  active-product class identity does not use the static mapping file.

## Environment Defaults

`.env.example` is the sanitized freezer-first field template. It sets freezer
mode, dual-top camera layout, and `models/set7_v8best.engine` while keeping
credentials as placeholders. Important groups:
Important groups:

- API: `MODEL__API__HOST`, `MODEL__API__PORT`, `MODEL__API__LOG_LEVEL`.
- Vision: model path, Top/Side thresholds, crop policies, motion thresholds,
  rescue settings, voting weights.
- Catalog: node-first product snapshots, engine-backed class-name matching
  with official `product_eng_name` plus migration-compatible `name` and legacy
  `product_name`, and optional legacy static validation.
- Async streaming: enabled flag, queue size, `MODEL__ASYNC_STREAMING__FRAME_STRIDE`.
- Weight: strict mode, fallback, tolerances, rescue tolerance, combination
  limits.
- Door session: enabled flag, session timeout, close debounce waits, YAML
  persistence.
- Trace: sample export enablement, sample count, sample export directory.
- Health: `/api/health/detailed` reports runtime dependency diagnostics for
  NumPy, Torch CUDA, and TensorRT without making those imports hard test
  requirements on the development host.

## Important Freshness Notes

- README was reviewed on 2026-06-04 and records the current local gate as
  `uv run --no-sync ruff check services/model scripts` plus
  `uv run --no-sync pytest services/model/tests -q` with `351 passed`.
  This remains static/unit/contract proof only, not Jetson runtime readiness.
- README says Camera sends two physical loadcell channels per vending zone and
  the model sums those channels into zone total. Older docs that mention channel
  averaging should be treated as historical unless code/tests confirm them.
- README says new clone-based operation should use `CRK-model-go`; this wiki is
  still scoped to the Python `CRK-model` repo unless a task explicitly asks for
  sibling repo ingestion.
- README now records host log rotation as the retention layer for
  `services/model/logs/frame_split_*.jsonl`; app YAML retention does not clean
  those JSONL trace files.

## Related Wiki Pages

- [File inventory](file-inventory.md)
- [Configuration](configuration.md)
- [Scripts and Jetson tools](scripts-and-jetson-tools.md)
- [Jetson and testing](../synthesis/jetson-and-testing.md)
