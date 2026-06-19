# Source Summary: PRODUCT_DETECTION_DETAIL.md

Source: [docs/PRODUCT_DETECTION_DETAIL.md](../../PRODUCT_DETECTION_DETAIL.md)
Status: historical with useful internals

## Use This When

Use this for detailed descriptions of stages 3 through 7: YOLO, filters,
voting, decision, and session aggregation.

## Key Facts

- The detailed flow is:
  BGR frame -> `YOLODetection[]` -> filtered detections -> `VoteCount` ->
  `VoteResult[]` -> `EnsembleResult[]` -> `JudgmentResult` -> Node response.
- YOLO uses TensorRT FP16, 480x480 input, low initial confidence threshold, and
  a maximum detection limit per frame.
- Filtering removes static/background detections, products not on a hand path,
  invalid side-camera ROI, and low confidence.
- Voting stores count, max confidence, and sum confidence per camera/class, then
  combines Top/Side evidence.
- Decision logic uses `ProductDecisionEngine.judge()` and
  `StrictWeightMatcher.find_valid_combinations()`.
- `fusion_confidence` blends vision confidence, weight match quality, and
  simplicity.
- Session integration covers `DoorSession`, `ProductAggregator`, unmatched
  returns, cross-zone returns, global session lifecycle, Node polling response,
  and YAML persistence.

## Related Code

- `services/model/model_service/vision/yolo_wrapper.py`
- `services/model/model_service/vision/hand_path_tracker.py`
- `services/model/model_service/video/video_processor.py`
- `services/model/model_service/video/voting_ensemble.py`
- `services/model/model_service/engine/decision_engine.py`
- `services/model/model_service/weight/strict_weight_matcher.py`
- `services/model/model_service/session/`

## Caveats

- Treat this as a detailed conceptual reference, not the single source of truth
  for current defaults. Later recovery and latency work changed parts of the
  operational behavior.

## Related Wiki Pages

- [Product detection pipeline](../synthesis/product-detection-pipeline.md)
- [Historical risk and fixes](../synthesis/historical-risk-and-fixes.md)
