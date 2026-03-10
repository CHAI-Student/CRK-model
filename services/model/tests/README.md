# Model Service Tests

Last reviewed: 2026-03-09

## Test Bootstrap

`conftest.py` now prepends `services/model` to `sys.path` only when `model_service` is not already importable. This keeps editable installs working while still allowing direct repo-local pytest runs.

## Recommended Commands

After activating `.venv`, prefer direct commands:

```bash
pytest services/model/tests -q
pytest services/model/tests/test_fastapi_imports.py -q
pytest services/model/tests/test_api_routes.py -q
pytest services/model/tests/test_deps.py -q
pytest services/model/tests/test_trigger_helpers.py -q
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
| `test_fastapi_imports.py` | import smoke and runtime settings checks |
| `test_api_routes.py` | FastAPI route responses |
| `test_deps.py` | dependency container behavior |
| `test_trigger_helpers.py` | loadcell and vote helper logic |
| `test_session_store.py` | in-memory session store behavior |
| `test_door_session_store.py` | door-session YAML lifecycle |

## Latest Focused Verification

The focused FastAPI regression suite below passed on 2026-03-09:

```bash
pytest services/model/tests/test_fastapi_imports.py \
  services/model/tests/test_api_routes.py \
  services/model/tests/test_deps.py \
  services/model/tests/test_trigger_helpers.py -q
```

Result:

```text
43 passed in 0.67s
```
