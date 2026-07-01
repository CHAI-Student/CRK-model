# Source Code Map: Video And Vision

Source: [video/frame_extractor.py](../../../services/model/model_service/video/frame_extractor.py),
[video/freezer_candidate_policy.py](../../../services/model/model_service/video/freezer_candidate_policy.py),
[video/video_processor.py](../../../services/model/model_service/video/video_processor.py),
[video/voting_ensemble.py](../../../services/model/model_service/video/voting_ensemble.py),
[vision/yolo_wrapper.py](../../../services/model/model_service/vision/yolo_wrapper.py),
[vision/hand_path_tracker.py](../../../services/model/model_service/vision/hand_path_tracker.py)

Status: current video/vision map

## Current Thesis

The video stack decodes AVI files frame-by-frame, runs one YOLO wrapper over
selected frames, accumulates per-camera evidence, and returns ranked
`VoteResult` candidates for the decision engine.

## Frame Extraction

- `StreamingFrameExtractor` uses FFmpeg and records per-attempt diagnostics.
- The current freezer field profile applies FFmpeg `eq` gamma/contrast `1.0/1.0`
  on Top and `1.0/1.0` on Side. Trigger traces record these values with the
  frame stride so candidate recall can be compared against latency.
- It tracks expected frames, decoded frames, bytes read, partial reads,
  decoder, ffprobe attempts, ffprobe wait time, return code, and stderr tail.
- If async decode returns zero frames while ffprobe reported frames, sync/raw
  retry paths can recover; if retries still decode zero expected frames, the
  async processor raises `VideoProcessingError`.
- `CV2FrameExtractor` is available as a fallback path.

## VideoProcessor

- `VideoProcessingStats` records top/side processed frames, original frames,
  skipped frames, frame stride, raw detections, filtered detections, YOLO count,
  YOLO total/average time, ROI/motion/hand-path filter counts.
- `process_videos_async()` is the main latency-sensitive path.
- Fatal async streaming failures propagate as model-service exceptions instead
  of returning an empty no-detection result. This includes missing async
  extractor support for a provided video path, frame queue timeout before all
  extractors finish, zero decoded frames after retry when frames were expected,
  and YOLO/task exceptions.
- Existing model-service exceptions such as `YOLOGPUError` and
  `VideoProcessingError` are re-raised; unknown task failures are wrapped in
  `VideoProcessingError` with the task name so the trigger worker can mark the
  session as `error`.
- A successfully decoded video with no product detections still returns normal
  empty candidates; the fail-closed rule is only for processing failure.
- The async frame queue type annotation avoids importing or referencing a
  local `np` symbol in `video_processor.py`; frame payload behavior is
  unchanged.
- Trigger code passes `delta_weight` into both sync and async video processing
  so top-camera ROI can distinguish removal from return.
- Fixed `frame_stride=2` runs YOLO on decoded frame indices divisible by `2`;
  other stride values are rejected by settings validation.
- Threshold rescue and ROI rescue can preserve low-confidence evidence for
  weight-gated matching.
- Diagnostic all-class trace can collect limited evidence outside normal
  active-class filtering.
- YOLO inference receives product class ids from the active product snapshot
  plus the configured hand class id `0` when products exist. Final result
  filtering, rescue filtering, and product outputs still use product ids only,
  so the hand class never becomes product identity evidence. If the active
  product allowlist is missing or empty, inference receives `[]` and remains
  fail-closed. Under the default Node-first catalog policy, product ids come
  only from Edge class-name keys matched against the current YOLO engine class
  names. Official input is `product_eng_name` and successful matches are
  tagged as `product_eng_name_engine`; during Edge migration,
  engine-matching `name` is tagged `name_engine_compat` and legacy
  `product_name` is tagged `product_name_engine_legacy`. Direct
  `trainingidx`/`yolo_class_id` fields and stale static mappings are ignored
  for active-product class identity.
- A stock-positive product with missing or `0g` weight still remains in the
  active-class allowlist; weight availability only affects loadcell
  validation/count paths.

## Filters And Voting

- Motion filtering uses bbox center movement with a 10px default minimum floor
  before the bbox-size dynamic threshold rule.
- Motion filtering is fail-closed for regular candidates: if every regular
  product candidate is `motion_passed=false`, those candidates remain filtered
  instead of being reintroduced by a fail-open fallback.
- Hand-path filtering uses `HandPathTracker` and hand trajectory/product bbox
  intersection. In `dual_top_proxy`, only the physical `top_middle` stream
  receives hand class `0` and updates this tracker; physical `top_side` remains
  product-only so weak side hand detections cannot filter candidates.
- Top/Side preprocessing defaults to left 480x480 crop from 640x480 camera
  frames. `letterbox` remains available for explicit experiments, but the
  operating templates now align ROI coordinates to the 480x480 TensorRT input.
- The logical Top/Side processing profiles are now separated from physical
  camera placement. The default `legacy_top_side` layout preserves current
  behavior; `dual_top_proxy` records `videos.top` as physical `top_middle` and
  `videos.side` as `top_side` while applying the Top processing profile to
  both public streams.
- Top ROI filtering keeps the lower region for both removals
  (`delta_weight < 0`) and returns (`delta_weight > 0`): detections pass when
  `center_y >= 240`. Zero or missing delta skips top ROI.
- In freezer mode with `dual_top_proxy`, both public streams are treated as
  top cameras. Freezer candidates must pass the configured vertical ROI.
  The freezer template uses the upper half:
  `MODEL__VISION__FREEZER_ROI_VERTICAL_REGION=upper` and
  `center_y <= MODEL__VISION__FREEZER_ROI_Y_SPLIT` (default `240`).
  `lower` remains a rollback mode, and
  `MODEL__VISION__FREEZER_LOWER_ROI_Y_SPLIT` is only a deprecated split
  fallback. Threshold rescue and ROI rescue are disabled for freezer
  candidates so only strong in-ROI evidence reaches the decision engine.
- Current freezer product vote floors are `0.70` for both Top and Side through
  `MODEL__VISION__TOP_CONFIDENCE_THRESHOLD` and
  `MODEL__VISION__SIDE_CONFIDENCE_THRESHOLD`. Product detections below the
  relevant raw/max camera floor may still appear in traces as rejected
  observations, but they cannot create regular votes, rescue candidates,
  stage-count fallback, diagnostic fallback identity evidence, or close-time
  aggregate fallback candidates in freezer mode. Weighted/combined confidence
  is logged separately and does not satisfy this raw identity floor.
- Hand tracking has a separate floor:
  `MODEL__VISION__HAND_CONFIDENCE_THRESHOLD=0.30`. Hand detections below this
  value are removed before `HandPathTracker` sees them; hand class id remains
  `MODEL__VISION__HAND_CLASS_ID=0`. This class is included in the top-middle
  inference allowlist only.
- `MODEL__MACHINE__CABINET_TYPE=freezer` alone is not enough to enable freezer
  dual-top candidate-pool filtering. Dual-top freezer deployments must also set
  `MODEL__VISION__CAMERA_LAYOUT=dual_top_proxy`; otherwise the freezer handled
  filter records `disabled_camera_layout` and OPS emits a config warning.
- After video processing, freezer dual-top removals keep a vision candidate
  pool. `VideoProcessor.filter_freezer_handled_candidates()` passes through all
  regular `source=vision` candidates inside top-K that already passed the
  raw product threshold, freezer ROI, motion, and valid top-middle hand-path gates.
  It no longer narrows by loadcell residual, multi-kind weight fit, or
  same-product repeat. The filter normalizes dual-top same-class
  `instance_count_hint` to `1`; repeated counts are inferred later by the
  decision engine's ordered weight solver.
- Stage-only, active-only, weight-nearest, threshold-rescue, and ROI-rescue
  evidence remains diagnostic-only for freezer identity. The freezer candidate
  filter records these sources in trace diagnostics when available, but it does
  not return them as handled candidates or create a chargeable product from
  them.
- Freezer ROI stage names are split by meaning: `freezer_roi_passed` is the
  only stage that increments `freezerExitPathVotes`; `freezer_roi_filtered`
  records rejected ROI evidence and `freezerRoiFilteredVotes` only.
- Freezer handled filtering now records and uses interaction evidence:
  `pathDisplacementPx`, `maxDistancePx`, `centerSpanX/Y`,
  `trajectoryExitPathPassed`, `staticShelfLikely`, `handPathValid`,
  `handPathValidUpperRoi`, `handInteractionPassed`, `handNearFrameCount`,
  `handNearVoteRatio`, `minHandDistancePx`, `handPathPassed`, and
  `handPathBlocked`. In freezer dual-top, hand evidence is only tracked from
  physical `top_middle` inside the active upper ROI, so lower-shelf detections
  or weak `top_side` hand detections do not create hand support. Top-only
  static candidates are demoted when they have no trajectory/hand support. A
  valid top-middle hand-path block removes a top-only candidate only when at
  least one hand-near alternative remains, preserving fail-open behavior when
  hand evidence would remove everything.
- Freezer trigger traces now expose `camera_layout`, raw candidate count,
  handled candidate count, freezer filter reason, and the key freezer vote,
  motion, ROI, exit-path, and multi-candidate thresholds. OPS also writes a
  `[FREEZER-CANDIDATE-FILTER]` line. In the current policy, a normal enabled
  freezer removal should show `reason=vision_identity_passthrough` with
  `raw=N` and `handled=N` after visual gates and hand-path rejection. Candidate
  diagnostics expose weighted `confidence`, raw `identity_confidence`, camera
  raw confidences, the active `identity_threshold`, `count_hint`, and
  `freezer_exit_votes`; final result OPS lines expose engine `product_count`,
  so field logs can distinguish pre-engine candidate hints from chargeable
  basket counts.
- Side ROI filtering protects against side-camera noise outside the useful
  left-side region with hard `center_x <= side_roi_x_max`.
- The side ROI default is hard `center_x <= 400` plus a conditional
  `side_roi_soft_margin_px=5` regular-candidate band. Threshold-passed side
  detections in `400 < center_x <= 405` are allowed to reach motion filtering,
  which stabilizes Pepsi detections around `x=402..404`; farther-right
  detections remain ROI filtered.
- ROI rescue still requires `side_motion_passed=true` and the default strict
  boundary `roi_x_avg <= roi_x_limit`, so static or right-side detections do
  not re-enter candidates as `source=roi_rescue`.
- Top ROI-filtered detections do not enter ROI rescue; low-confidence top
  detections outside the active top ROI are also ineligible for threshold
  rescue.
- `VotingEnsemble` combines Top and Side votes using configured weights,
  top-only/side-only weights, common-class bonus, min vote ratio, and min vote
  count. The freezer dual-top template now biases physical `top_middle` over
  `top_side` with `top=0.60`, `side=0.40`, `top_only=0.60`, and
  `side_only=0.40`.
- Trigger conversion preserves frame-level `VoteResult.vote_count` as
  `EnsembleResult.raw_vote_count` for downstream repeat-count gates. The
  converted `EnsembleResult.vote_count` continues to mean Top/Side consensus
  (`1` or `2`), not total frame votes.
- Threshold rescue still allows low-confidence moving evidence to be retained
  for later weight gating, separate from regular motion-filtered candidates.
- Threshold rescue candidates carry `roi_conflict` diagnostics when the same
  class has stronger side-camera evidence filtered outside `side_roi_x_max`.
  Weak conflicted rescues are rejected during weight gating, preventing
  right-side Trevi-style evidence from beating regular Pepsi evidence by weight
  residual alone.
- When weight-gated rescue candidates are merged into the final candidate list,
  regular `source=vision` candidates now rank ahead of `roi_rescue` and
  `threshold_rescue` candidates before `top_k` trimming.

## YOLO Wrapper

- `YOLOWrapper` loads the TensorRT engine and exposes `detect(frame)`.
- Geometry handling defaults to left 480 crop and still supports explicit
  policies including `letterbox`.
- Class `0` is hand; product classes are positive ids.
- Product inference allowlists are expanded to include hand class `0` only
  when at least one active product class exists. Product result allowlists stay
  product-only.
- If `allowed_class_ids` is an empty list, detection is skipped fail-closed.
  That now indicates missing/invalid active class ids, not merely zero product
  weight.
- Engine load failure is surfaced during FastAPI lifespan startup.

## Related Wiki Pages

- [Product detection pipeline](../synthesis/product-detection-pipeline.md)
- [Latency and frame stride](../synthesis/latency-and-frame-stride.md)
- [Observability and traces](observability-and-traces.md)
