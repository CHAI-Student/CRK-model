## Model Service Architecture

### Scope

This guide covers the trigger-inference path that starts in FastAPI and ends in
door-session aggregation.

### Runtime Flow

0. `Edge_Environment` and `CRK-CAMERA`
   The capture directory is created first, then the camera service arms
   recording/loadcell collection, returns `ready=true`, the model polling loop
   is started, and only then is the door unlocked.
1. `api/routes/trigger.py`
   Accepts a completed trigger with videos, loadcell samples, and zone id.
2. `service/trigger_service.py`
   Preferred async orchestration path. Handles queueing, deduplication, and
   trace logging.
3. `video/`
   Extracts frames and produces per-camera votes.
4. `engine/decision_engine.py`
   Fuses vision results, loadcell delta, and the live `active_products`
   snapshot into a `JudgmentResult`.
5. `session/door_session_store.py`
   Stores the trigger inside the active door session and rebuilds aggregated
   product counts.
6. `session/product_aggregator.py`
   Applies removal and return semantics to the session-level product counts.

### Key Data Contracts

- `delta_weight < 0`: removal from the machine.
- `delta_weight > 0`: return back into the machine.
- `active_products`: live product snapshot from Node.js. This is the only
  supported source for strict/loadcell-only matching.
- `stock_qty = 0`: sold out. Strict matching must exclude the product.
- `timing`: optional metadata forwarded from `CRK-CAMERA`. Supported fields:
  `capture_started_at`, `capture_ended_at`, `loadcell_started_at`,
  `loadcell_ended_at`, `trigger_started_at`, `trigger_end_reason`.

### Inference Branches

`ProductDecisionEngine.judge()` evaluates in this order:

1. `vision_only`
2. `loadcell_only_no_vision`
3. `no_detection_min_weight`
4. `strict_match`
5. relaxed fallback:
   `single_product_match`
   `combination_match`
   `partial_result`
   `loadcell_only_no_estimates`

If strict matching is enabled and fails:

- `MODEL__WEIGHT__STRICT_MODE_FALLBACK=true`
  Continue into the relaxed path.
- `MODEL__WEIGHT__STRICT_MODE_FALLBACK=false`
  Return `NO_DETECTION`.

### Video Extraction Recovery

- `video/frame_extractor.py` no longer treats a short stdout read as EOF.
  Frames are assembled only after the full frame payload is received.
- If `ffprobe` reports frames but async decode returns `0`, the extractor logs
  diagnostics and retries through sync/raw decode paths. If retries still
  decode zero expected frames, `process_videos_async()` propagates a
  `VideoProcessingError`; this must not become a normal no-detection or
  loadcell-only fallback result.
- The extractor diagnostics now include:
  `expected_frames`, `decoded_frames`, `bytes_read`, `partial_reads`,
  `stderr_tail`, and `final_branch`.

### Loadcell Timing Diagnostics

- Weight delta analysis is driven by the configurable
  `MODEL__LOADCELL__STABLE_WINDOW_SIZE` and
  `MODEL__LOADCELL__STABILITY_THRESHOLD_GRAMS` settings.
- The trigger path logs the sample count, time span, stable region indices, and
  baseline/final windows so field logs can show whether the history was cut
  before the stable tail arrived.

### Return Recovery Layers

There are three recovery passes for "removed then put back" behavior:

1. Same-zone immediate recovery
   `ProductAggregator._handle_return()` first tries single-product weight
   matching, then multi-product return combinations.
2. Net-delta repair
   `DoorSessionStore._validate_net_delta()` corrects aggregated counts when the
   session-level net delta shows that all or part of the removal was returned.
3. Cross-zone repair
   `DoorSessionStore._handle_cross_zone_returns()` tries to match unmatched
   returns against other active sessions.

### Operational Notes

- Door unlock must never happen before camera/loadcell capture is armed. If the
  camera service does not report `ready=true`, Node.js should abort the flow and
  keep the door locked.
- The fallback `/trigger` route must forward `active_products` into
  `engine.judge(...)` exactly like `TriggerService` does.
- Logs should keep explicit reason codes such as
  `strict_mismatch`, `no_active_products`, `stock_filtered`, and
  `negative_delta_weight(removal)` so field debugging does not require code
  inspection.
