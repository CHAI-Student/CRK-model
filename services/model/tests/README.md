# Model Service Tests

Last reviewed: 2026-06-04

## Test Bootstrap

`conftest.py` now prepends `services/model` to `sys.path` only when `model_service` is not already importable. This keeps editable installs working while still allowing direct repo-local pytest runs.

## Recommended Commands

After activating `.venv`, prefer direct commands:

```bash
pytest services/model/tests -q
pytest services/model/tests/test_fastapi_imports.py -q
pytest services/model/tests/test_multi_zone_summary.py -q
pytest services/model/tests/test_trigger_helpers.py -q
pytest services/model/tests/test_runtime_env.py -q
pytest services/model/tests/test_video_processor_thresholds.py -q
pytest services/model/tests/test_frame_trace.py::test_trigger_service_worker_marks_video_processing_error -q
```

If you want to use `uv run`, prefer `--no-sync` once the environment is already installed:

```bash
uv run --no-sync pytest services/model/tests -q
```

## Coverage Command

```bash
pytest services/model/tests --cov=services/model/model_service --cov-report=term-missing
```

## Important Suites

| File | Purpose |
|------|---------|
| `test_fastapi_imports.py` | import smoke, runtime settings checks, and best-effort health diagnostics |
| `test_multi_zone_summary.py` | multi-zone close summaries, active snapshot guards, and response contracts |
| `test_trigger_helpers.py` | loadcell and vote helper logic |
| `test_video_processor_thresholds.py` | video thresholds, ROI, async frame stride, and async failure propagation |
| `test_frame_trace.py` | trigger traces, worker lifecycle, and propagated video-error session status |
| `test_session_store_lifecycle.py` | session overwrite logging lifecycle |
| `test_product_aggregator.py` | removal, return, and close aggregation behavior |
| `test_cross_zone_return.py` | cross-zone return repair behavior |

## Latest Local Verification

The full model-service suite is the preferred local gate:

```bash
uv run --no-sync pytest services/model/tests -q
```

Current recorded result on 2026-06-04:

```text
351 passed
```

This is local unit/contract coverage only. It does not prove Jetson TensorRT,
camera, AVI decoding, or service startup readiness.
