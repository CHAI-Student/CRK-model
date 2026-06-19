# Historical Risk And Fixes

## Current Thesis

The March 2026 docs explain why many current guardrails exist. Treat them as
history unless current agent guides or code confirm the behavior.

## Risk Review Themes

The March risk review identified issues around:

- YOLO load failure appearing as a healthy service.
- Session id collision or overwrite risks.
- Failed triggers remaining in `processing`.
- Dedup registration blocking legitimate retries.
- `ActiveProductStore` cleanup ambiguity.
- Async video path blocking or fallback incompatibility.
- CLOSE finalization races.
- Return handling deducting only one item in multi-return scenarios.
- Single-channel loadcell use instead of channel averaging.
- Docs/code mismatches in API states, health fields, versions, and model path.

## Fixes Applied

The follow-up fixes included:

- Error status propagation from worker failures.
- Microsecond session id expectations in comments/tests.
- Return deduction safety guard.
- Removal of invalid uv config keys.
- Multi-channel loadcell averaging in trigger fallback.
- Version string alignment.
- Removal of unused multi-zone engine injection.
- Deferral of return-combination DFS ranking as an improvement.

## Scenario Roadmap History

The roadmap tracks phases for:

- Data quality foundation.
- Simultaneous multi-item returns.
- Camera interference.
- Loadcell interference.
- Complex extraction/return verification.
- Two-handed extraction and multi-zone aggregation.

It marks many phases complete, but the file still contains older implementation
prompts. Use it for historical intent and test coverage ideas, not as a current
todo list.

## Later Recovery Notes

The March 31 recovery note is more current than the February/March pipeline
docs for trigger hardening:

- Preferred path is `TriggerService`.
- Fallback `/trigger` should match weight-aware behavior.
- `active_products` must be forwarded into `ProductDecisionEngine`.
- Strict mode and strict fallback default to enabled.
- Return repair has same-zone, net-delta, and cross-zone layers.

## Current Freshness Caveat

Current README/tests describe physical zone loadcell channels as summed into a
zone total. The March wording about averaging is useful historical context for
"do not use only channel 0", but future edits should verify the current summed
semantics in `core/loadcell_stats.py` and tests before changing behavior.

## Evidence

- [Repo overview](../source-code/repo-overview.md)
- [Loadcell and trigger](../source-code/loadcell-and-trigger.md)
- [Model process risk review](../source-docs/model-process-risk-review-2026-03-04.md)
- [Fixes applied](../source-docs/fixes-applied-2026-03-05.md)
- [Implementation roadmap](../source-docs/implementation-roadmap.md)
- [Trigger inference recovery notes](../source-docs/trigger-inference-recovery-notes-2026-03-31.md)
