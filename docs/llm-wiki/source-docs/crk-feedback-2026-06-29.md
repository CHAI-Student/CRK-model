# Source Summary: CRK Feedback 2026-06-29

Source: user-provided pasted text attachment,
`C:\Users\user\.codex\attachments\51e2a4ac-51b4-4f55-83fc-9a2554e67766\pasted-text.txt`

Status: current external review input

Use this when: checking whether CRK-model addresses the 2026-06-29 CRK
operations and async video failure review.

## Key Facts

- Scope for this repo is `(2) CRK-Model (EdgePC operating model)`. The
  attachment also asks AI Server handoff questions, but those are outside this
  CRK-model code change.
- CRK emphasized long-term embedded operation on Jetson Orin Nano, including
  JetPack/NumPy compatibility, single-GPU resource serialization, disk/log
  retention, and a clearer TensorRT weight deployment runbook.
- The concrete code risk was in `process_videos_async()`: Top extraction, Side
  extraction, and YOLO inference tasks could fail, be logged, and still allow
  the function to continue into an empty `VideoProcessingResult`.
- Required behavior is fail-closed for fatal video processing errors. A fatal
  extractor/YOLO/queue failure should propagate as a `VideoProcessingError` or
  related model-service exception so `TriggerService` marks the session/error
  path instead of treating the event as a normal no-product case.
- Legitimate decoded video with no product detections remains a valid
  no-detection outcome; the issue is silent conversion of processing failure
  into no-detection.

## Related Code

- `services/model/model_service/video/video_processor.py`
- `services/model/model_service/service/trigger_service.py`
- `services/model/model_service/api/routes/health.py`
- `scripts/convert_engine.sh`

## Caveats

- The attachment is an external review note, not a committed raw source file in
  this repo. Preserve its provenance and do not copy the full pasted text into
  the repo by default.
- Jetson runtime readiness still requires on-device validation. Local tests
  only prove code-level regression behavior.

## Related Wiki Pages

- [Video and vision](../source-code/video-and-vision.md)
- [Loadcell and trigger](../source-code/loadcell-and-trigger.md)
- [Jetson and testing](../synthesis/jetson-and-testing.md)
- [Scripts and Jetson tools](../source-code/scripts-and-jetson-tools.md)
