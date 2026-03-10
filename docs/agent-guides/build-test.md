# Build and Test Notes

## Preferred Commands

After activating `.venv`:

```bash
pytest services/model/tests -q
pytest services/model/tests/test_fastapi_imports.py -q
pytest services/model/tests/test_api_routes.py -q
pytest services/model/tests/test_deps.py -q
pytest services/model/tests/test_trigger_helpers.py -q
```

Coverage:

```bash
pytest services/model/tests --cov=services/model/model_service --cov-report=term-missing
```

If uv must be used:

```bash
uv run --no-sync pytest services/model/tests -q
```

## Important Notes

- `services/model/tests/conftest.py` conditionally bootstraps `services/model` onto `sys.path`.
- The focused FastAPI suite passed on 2026-03-09 with `43 passed`.
- Unit tests should not require YOLO, TensorRT, or GPU startup.
