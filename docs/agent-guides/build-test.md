# Build and Test Notes

## Preferred Commands

After activating `.venv`:

```bash
pytest services/model/tests -q
pytest services/model/tests/test_fastapi_imports.py -q
pytest services/model/tests/test_multi_zone_summary.py -q
pytest services/model/tests/test_trigger_helpers.py -q
pytest services/model/tests/test_decision_engine.py -q
pytest services/model/tests/test_runtime_env.py -q
pytest services/model/tests/test_product_aggregator.py -q
pytest services/model/tests/test_cross_zone_return.py -q
```

Coverage:

```bash
pytest services/model/tests --cov=services/model/model_service --cov-report=term-missing
```

If uv must be used:

```bash
uv run --no-sync pytest services/model/tests -q
```

Cross-repo regressions added for the April 2026 timing/capture fixes:

```bash
cd ../CRK-CAMERA
uv run python -m unittest discover -s tests -p "test_*.py" -v

cd ../Edge_Environment
node --test server/test/payments-order.test.js
```

## Important Notes

- `services/model/tests/conftest.py` conditionally bootstraps `services/model` onto `sys.path`.
- The full local model-service suite recorded `351 passed` on 2026-06-04.
  Re-run the suite instead of relying on the historical count.
- Unit tests should not require YOLO, TensorRT, or GPU startup.
- `test_runtime_env.py` covers the Jetson-specific startup bootstrap and should
  be rerun whenever `main.py`, `runtime_env.py`, or the Jetson startup scripts
  change.
- When touching trigger inference, rerun both `test_decision_engine.py` and the return-recovery suites (`test_product_aggregator.py`, `test_cross_zone_return.py`) because inference fallback and session repair interact.
- When touching AVI decode or loadcell delta logic, rerun
  `test_trigger_pipeline_regressions.py` in addition to the full suite. That
  file covers partial stdout reads, zero-frame retry, and truncated-vs-stable
  loadcell history.
