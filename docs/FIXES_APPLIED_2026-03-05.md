# Edge Environment - Fixes Applied (2026-03-05)

> Status note (2026-03-09): FastAPI import stabilization, runtime settings health fixes, shared loadcell helpers, and focused FastAPI regression coverage were added after this document was written. Treat this file as historical context, not the latest runtime status.

Based on: `docs/REQUIRED_FIXES_REVIEW_2026-03-04.md`

## Overview

| # | Severity | Issue | Verdict | Action |
|---|----------|-------|---------|--------|
| 1 | Critical | Worker failure not propagated as `status="error"` | TRUE | **Fixed** |
| 2 | High | Session ID format mismatch (tests/comments vs actual) | TRUE | **Fixed** |
| 3 | High | Phase 0b safety guard incomplete | PARTIAL | **Fixed** |
| 4 | High | Invalid `[tool.uv]` keys in pyproject.toml | TRUE | **Fixed** |
| 5 | Medium | Fallback uses single channel `filtered_value[0]` | TRUE | **Fixed** |
| 6 | Medium | Version metadata inconsistent (5.3 vs 5.4) | TRUE | **Fixed** |
| 7 | Medium | Unused `engine` dependency injection in multi-zone | TRUE | **Fixed** |
| 8 | Medium | Tests blocked by numpy env | FALSE | **Rejected** |
| Opt | — | Return combination DFS ranking | TRUE | **Deferred** |

---

## 1. [Critical] Worker failure → `status="error"` propagation

**Problem:** `SessionStore.update_stage()` only set `processing_stage`, never `status`. Failed sessions stayed as `"processing"` forever, making `multi_zone.py`'s error response branch dead code.

**Changes:**

| File | Change |
|------|--------|
| `services/model/model_service/session/session_store.py` | Added optional `status` parameter to `update_stage()` |
| `services/model/model_service/service/trigger_service.py` | Added `status="error"` to both error paths (queue-full at L406, worker exception at L532) |
| `services/model/tests/test_session_store.py` | Added `TestUpdateStage` class with 3 tests |

---

## 2. [High] Session ID format — comments & tests aligned

**Problem:** Generators include microseconds (`%f`), producing 5-part IDs like `zone_1_260201_143025_123456`. Comments described 4-part format; tests asserted `len(parts) == 4`.

**Decision:** Keep microseconds (collision prevention).

**Changes:**

| File | Change |
|------|--------|
| `services/model/model_service/session/session_store.py` | Comment: `{HHMMSS}` → `{HHMMSS}_{ffffff}` |
| `services/model/model_service/session/door_session.py` | Comment: same |
| `services/model/model_service/session/global_door_session.py` | Comment: same |
| `services/model/tests/test_session_store.py` | `len(parts) == 4` → `5`, added microseconds assertion |
| `services/model/tests/test_pipeline.py` | `len(parts) == 4` → `5` |

---

## 3. [High] Phase 0b safety guard — abnormally large return deduction

**Problem:** Roadmap specifies "if `estimated_count > agg.count`, deduct only 1 (safety guard)". Code used `min(estimated_count, agg.count)` which deducted all remaining stock instead.

**Changes:**

| File | Change |
|------|--------|
| `services/model/model_service/session/product_aggregator.py` | `min(estimated_count, agg.count)` → `if estimated_count > agg.count: estimated_count = 1` |
| `services/model/tests/test_product_aggregator.py` | Added 2 tests in `TestReturnCountEstimation`: safety guard case + normal multi-return |

---

## 4. [High] Invalid `[tool.uv]` keys removed

**Problem:** `python-version` and `system-site-packages` are not valid `[tool.uv]` configuration keys. Silently ignored by uv.

**Changes:**

| File | Change |
|------|--------|
| `pyproject.toml` | Removed both invalid keys; replaced with explanatory comments pointing to correct mechanisms (`requires-python`, `uv venv --system-site-packages`) |

---

## 5. [Medium] Trigger fallback → multi-channel average

**Problem:** `_calculate_weight_delta()` fallback path used `filtered_value[0]` (single channel) instead of the available `_avg_loadcell_channels()` function.

**Changes:**

| File | Change |
|------|--------|
| `services/model/model_service/api/routes/trigger.py` | Replaced `filtered_value[0]` + `_parse_loadcell_value()` with `_avg_loadcell_channels()` in the `is_valid == False` branch |

---

## 6. [Medium] Version strings unified to 5.4.0

**Problem:** `__init__.py` and `pyproject.toml` said 5.4.0; `main.py` and `manager.py` still said 5.3.0/v5.3.

**Changes:**

| File | Change |
|------|--------|
| `services/model/model_service/main.py` | `v5.3` → `v5.4` in docstring, argparse description, log message |
| `services/model/model_service/api/manager.py` | `5.3.0` → `5.4.0` in FastAPI metadata and root endpoint; `v5.3` → `v5.4` in description |

---

## 7. [Medium] Unused `engine` dependency removed from multi-zone route

**Problem:** `engine: ProductDecisionEngine = Depends(get_decision_engine)` injected at L761 but never referenced in `judge_multi_zone()` function body.

**Changes:**

| File | Change |
|------|--------|
| `services/model/model_service/api/routes/multi_zone.py` | Removed `engine` parameter, `ProductDecisionEngine` import, and `get_decision_engine` import |

---

## 8. [Rejected] numpy environment issue

**Review claim:** numpy `ndarray` AttributeError blocks test execution.

**Verdict at the time: FALSE** — `numpy>=1.24.0,<2.0.0` was properly declared in `pyproject.toml`.

**2026-03-09 follow-up:** the broken Windows environment was real, but additional code changes were still applied later to reduce import-time coupling to NumPy-heavy modules. FastAPI imports now stay lighter even when the environment is partially broken.

---

## Optional: Return combination DFS ranking — Deferred

**Review claim:** `_backtrack_return()` returns first DFS hit without ranking alternatives.

**Verdict: TRUE** — Valid improvement, but not a bug. Requires careful design to avoid performance regression. Deferred for future work.

---

## Verification

Tested with (directly-related test files only):
```bash
.venv/Scripts/python -m pytest services/model/tests/test_session_store.py \
  services/model/tests/test_product_aggregator.py \
  services/model/tests/test_door_session_store.py \
  services/model/tests/test_scenarios_phase4.py \
  services/model/tests/test_scenarios_phase5.py -v
```

Result: **51 passed, 0 failed**

Note: Full suite has pre-existing `numpy AttributeError` on Windows dev machine
(tests importing from `model_service.api`, `.video`, or `.vision` fail at collection).
