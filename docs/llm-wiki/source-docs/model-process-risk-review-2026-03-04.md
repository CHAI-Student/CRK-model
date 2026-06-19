# Source Summary: MODEL_PROCESS_RISK_REVIEW_2026-03-04.md

Source: [docs/MODEL_PROCESS_RISK_REVIEW_2026-03-04.md](../../MODEL_PROCESS_RISK_REVIEW_2026-03-04.md)
Status: historical with source encoding caveat

## Use This When

Use this to understand risks identified before the March 2026 fixes.

## Key Facts

- The review covered the full model process from `/trigger` through
  `TriggerService`, frame extraction, YOLO, filtering, decision, session store,
  and Node polling.
- High-priority risks included:
  YOLO load failure looking like a healthy service, session id collision,
  failed sessions remaining in `processing`, dedup timing blocking retries,
  unclear `ActiveProductStore` cleanup, async event-loop blocking/fallback
  incompatibility, CLOSE finalize races, single-count return handling, and
  single-channel loadcell usage.
- It also called out documentation-code mismatches around API contract,
  terminal CLOSE states, health fields, version strings, and model path.

## Related Code

- `services/model/model_service/api/`
- `services/model/model_service/service/trigger_service.py`
- `services/model/model_service/video/`
- `services/model/model_service/vision/`
- `services/model/model_service/engine/`
- `services/model/model_service/session/`

## Caveats

- The raw file currently renders with encoding damage in this environment.
  Cross-check with [FIXES_APPLIED](fixes-applied-2026-03-05.md) and current code
  before treating a risk as still open.

## Related Wiki Pages

- [Historical risk and fixes](../synthesis/historical-risk-and-fixes.md)
