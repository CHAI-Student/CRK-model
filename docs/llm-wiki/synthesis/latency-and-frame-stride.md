# Latency And Frame Stride

## Current Thesis

Recent latency work targeted three separable delays: trigger queue/video
processing, CLOSE wait behavior, and Node polling cadence. The high-impact
model-side bottleneck in the observed logs was not ffprobe; it was a single
worker queue plus YOLO over every decoded top/side AVI frame.

The 2026-05-21 and 2026-05-27 no-detection traces showed a separate recall
blocker: normal YOLO inference can receive an empty or missing allowed-class
list when `ActiveProductStore` loses current inventory context. That failure
can look like a stride recall problem because trace output has zero raw
detections, but it is an active-product snapshot issue first. Current trigger
paths can fall back to a fresh last-valid snapshot and mark
`snapshot_source=last_valid` in diagnostics.

## Recent Latency Commits

- `bf4cddf Reduce CRK model service latency waits`
  - Added configurable CLOSE waits.
  - Added shorter video readiness polling.
  - Added latency telemetry for trigger, video, and close paths.
- `3a5c306 Add detected single-item weight fallback`
  - Added class-evidence recovery for cases where strict matching misses but a
    single detected item fits weight tolerance.
- `6b50bb3 Add async frame stride latency controls`
  - Added configurable async frame stride.
  - Added frame-stride telemetry fields to latency logs and stats.

## Key Runtime Defaults

- `MODEL__DOOR_SESSION__CLOSE_INITIAL_WAIT_SECONDS=3.0`
- `MODEL__DOOR_SESSION__CLOSE_SUBSEQUENT_WAIT_SECONDS=1.0`
- `MODEL__VIDEO__READY_MAX_WAIT_SECONDS=2.0`
- `MODEL__VIDEO__READY_POLL_INTERVAL_SECONDS=0.2`
- `MODEL__ASYNC_STREAMING__FRAME_STRIDE=2`

The scenario readiness fixture also records `frame_stride=2`. As of
2026-06-02 this is the only accepted async streaming stride; settings
validation rejects `1` or `3`.

## Frame Stride Semantics

`MODEL__ASYNC_STREAMING__FRAME_STRIDE=2` means the async video processor still
walks the video stream, but only runs YOLO on frames whose original frame index
is divisible by `2`. The runtime also pins the processor to `2` so accidental
test monkeypatches or stale injected values cannot switch to another stride.

The stride optimization reduces YOLO inference work, not necessarily all decode
work. If decode, IO, or queue wait dominates a trace, stride alone will not
solve the whole latency problem.

## Telemetry To Compare

Use `[TRIGGER-WORKER][LATENCY]` and trace JSON fields:

- `queue_wait_ms`
- `video_ms`
- `engine_ms`
- `yolo_total_ms`
- `yolo_count`
- `frame_stride`
- `original_frames`
- `processed_frames`
- `skipped_frames`

Use `[CLOSE][LATENCY]` for close finalization and pending-trigger waits. If the
model result is ready but `[OPS][CLOSE]` appears much later, inspect Node polling
cadence separately.

## Accuracy Risk

Frame skipping can reduce evidence for:

- very fast hand/product appearances,
- objects that only appear clearly in a few frames,
- low-confidence products rescued by accumulated votes,
- cases where Top and Side cameras provide asymmetric evidence.

Do not tune stride to fix recognition regressions. The fixed value is `2`;
recall regressions should be handled in evidence filtering, candidate ranking,
or weight/session logic. `docs/jetson-stride2.env.txt` remains a historical
field-test template and matches the current fixed default.

Before blaming stride, verify trace `active_product_diagnostics`. If
`allowed_class_ids_count=0` and
`inference_fail_closed_reason=empty_allowlist_fail_closed`, production
candidates were intentionally blocked by the strict active inventory allowlist.
Diagnostic all-class evidence is only for debugging and must not bypass active
inventory safety.

If `inference_fail_closed_reason=missing_active_product_snapshot_fail_closed`,
check whether `snapshot_source=last_valid` recovered the trigger. Without that
fallback, the failure is model-service inventory context loss, not a raw vision
recall miss. With fallback present but raw detections still empty, continue
with normal video/model evidence analysis.

## Evidence

- [Configuration](../source-code/configuration.md)
- [Video and vision](../source-code/video-and-vision.md)
- [Observability and traces](../source-code/observability-and-traces.md)
- [Conventions](../source-docs/agent-guides-conventions.md)
- [Runtime flow](runtime-flow.md)
- [Product detection pipeline](product-detection-pipeline.md)
- [Scenario readiness and 0g diagnostics](scenario-readiness-and-0g.md)
- Code: `services/model/model_service/video/video_processor.py`
- Code: `services/model/model_service/service/trigger_service.py`
- Code: `services/model/model_service/core/config.py`
