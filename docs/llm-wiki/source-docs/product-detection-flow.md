# Source Summary: PRODUCT_DETECTION_FLOW.md

Source: [docs/PRODUCT_DETECTION_FLOW.md](../../PRODUCT_DETECTION_FLOW.md)
Status: historical with useful pipeline structure

## Use This When

Use this for the original seven-stage product detection overview.

## Key Facts

- The pipeline is described as seven stages:
  trigger receipt, frame extraction, YOLO TensorRT inference, filtering,
  voting ensemble, decision engine, and session integration/response.
- `POST /trigger` accepts AVI paths and loadcell data, computes
  `delta_weight`, deduplicates, skips low-weight events, and queues work.
- Frame extraction uses FFmpeg/NVDEC-style streaming into 480x480 BGR frames.
- YOLO is TensorRT FP16 at 480x480 with low raw confidence threshold before
  later filters.
- Filters include motion, hand path, side ROI, and confidence.
- Voting combines Top and Side camera evidence into `weighted_confidence`.
- Decision combines vision with `StrictWeightMatcher` and `delta_weight`.
- Node polls `POST /api/judge/multi-zone` and sends OPEN/CLOSE signals.

## Related Code

- `services/model/model_service/api/routes/trigger.py`
- `services/model/model_service/service/trigger_service.py`
- `services/model/model_service/video/frame_extractor.py`
- `services/model/model_service/video/video_processor.py`
- `services/model/model_service/video/voting_ensemble.py`
- `services/model/model_service/vision/yolo_wrapper.py`
- `services/model/model_service/engine/decision_engine.py`
- `services/model/model_service/session/door_session_store.py`

## Caveats

- Some details predate later trigger hardening, recovery behavior, and latency
  telemetry. Prefer [agent-guides/architecture](agent-guides-architecture.md)
  for current inference branches.

## Related Wiki Pages

- [Product detection pipeline](../synthesis/product-detection-pipeline.md)
- [Runtime flow](../synthesis/runtime-flow.md)
- [Latency and frame stride](../synthesis/latency-and-frame-stride.md)
