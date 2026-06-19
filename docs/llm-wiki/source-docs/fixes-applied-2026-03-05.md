# Source Summary: FIXES_APPLIED_2026-03-05.md

Source: [docs/FIXES_APPLIED_2026-03-05.md](../../FIXES_APPLIED_2026-03-05.md)
Status: historical with current-status note

## Use This When

Use this to see which March 2026 risk-review items were fixed, rejected, or
deferred.

## Key Facts

- Worker failure propagation was fixed by allowing `SessionStore.update_stage()`
  to set `status="error"` and by setting error status in trigger failure paths.
- Session ids kept microseconds to prevent collisions; tests/comments were
  updated to match.
- Abnormally large return deduction got a safety guard.
- Invalid `[tool.uv]` keys were removed from `pyproject.toml`.
- Trigger fallback was changed to use multi-channel average instead of only
  `filtered_value[0]`.
- Version strings were unified to 5.4.0 at that time.
- Unused engine dependency injection was removed from multi-zone route.
- A claimed NumPy issue was rejected as a code bug at that time, though later
  FastAPI import-stabilization work reduced import-time coupling.
- Return-combination DFS ranking was deferred as an improvement rather than a
  bug.

## Related Code

- `services/model/model_service/session/session_store.py`
- `services/model/model_service/service/trigger_service.py`
- `services/model/model_service/session/product_aggregator.py`
- `services/model/model_service/api/routes/trigger.py`
- `services/model/model_service/api/routes/multi_zone.py`
- `services/model/tests/test_session_store.py`
- `services/model/tests/test_product_aggregator.py`

## Caveats

- This document is explicitly marked as historical by its own status note.
  Current defaults should come from agent guides and code.
- The "multi-channel average" wording is historical. Current README/tests
  describe summing physical zone loadcell channels into the zone total.

## Related Wiki Pages

- [Historical risk and fixes](../synthesis/historical-risk-and-fixes.md)
