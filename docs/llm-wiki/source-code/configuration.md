# Source Code Map: Configuration

Source: [core/config.py](../../../services/model/model_service/core/config.py),
[.env.example](../../../.env.example),
[docs/jetson-stride2.env.txt](../../jetson-stride2.env.txt),
[docs/jetson-stride2.env.ko.txt](../../jetson-stride2.env.ko.txt),
[pyproject.toml](../../../pyproject.toml)

Status: current configuration map

## Current Thesis

Configuration is Pydantic Settings with `MODEL__` prefix, nested `__`
delimiter, `.env` auto-load, and runtime overrides through CLI host/port.

## Settings Groups

| Group | Class | Purpose |
| --- | --- | --- |
| API | `APIModel` | Host, port, log level, graceful shutdown timeout. |
| Vision | `VisionModel` | YOLO engine path, thresholds, crop/ROI policies, vote weights, rescue knobs. |
| Weight | `WeightModel` | Strict/relaxed tolerance, combination limits, detected-single fallback. |
| Trigger | `TriggerModel` | Dedup TTL/size, queue size, min weight, low-weight video fallback, return-only skip, balanced cancellation. |
| Catalog | `CatalogModel` | Node-first active product catalog policy, engine-backed class-name matching, and optional legacy static validation. |
| Loadcell | `LoadcellModel` | Stable window and stability threshold. |
| Video | `VideoModel` | AVI readiness polling limits. |
| Async streaming | `AsyncStreamingModel` | Async enablement, frame queue, frame stride, zero-frame retry. |
| Trace | `TraceModel` | Sample frame export settings. |
| Buffer | `BufferModel` | Trigger session TTL, max sessions, cleanup interval. |
| Door session | `DoorSessionModel` | Door session enablement, timeout, close waits, YAML retention. |

## High-Impact Defaults

- `MODEL__API__PORT=8002`
- Code default `MODEL__VISION__YOLO_MODEL_PATH=models/0204_morning.engine`;
  the copyable freezer `.env.example` sets `models/set9_imbalance_16.engine`.
- `MODEL__VISION__YOLO_INTERNAL_CONF_THRESHOLD=0.01` keeps raw YOLO
  collection broad; service-side regular candidate thresholds do the main
  filtering.
- `MODEL__VISION__HAND_CLASS_ID=0` identifies hand detections, and
  `MODEL__VISION__HAND_CONFIDENCE_THRESHOLD=0.40` is the separate hand-tracking
  confidence floor. This is intentionally lower than the current freezer
  product vote floor because hand recall is weaker.
- `MODEL__VISION__LOG_ENGINE_CLASSES=off` by default. Set it to `on`/`true`/`1`
  to print every loaded YOLO engine class id/name at startup for field
  debugging.
- `MODEL__CATALOG__SOURCE_POLICY=node_first` makes Node-provided product data
  the runtime source of truth. Class identity is resolved from Edge
  `product_eng_name` matched against loaded YOLO engine class names. During
  Edge migration, engine-matching `name` and legacy `product_name` are
  compatibility class keys after `product_eng_name`.
- Direct class-id/name fields such as `trainingidx`, `training_idx`,
  `trainingIdx`, `yolo_class_id`, `yoloClassId`, and `yolo_class_name` are
  accepted for API compatibility but ignored for active-product class
  identity.
- `MODEL__CATALOG__PRODUCT_NAME_FALLBACK_ENABLED` is a legacy env flag kept for
  deployment compatibility; it no longer enables runtime static-name fallback.
  In the official contract `product_name` is the display name; as a temporary
  bridge it can act as a class key only when it matches the current engine
  class name.
- `MODEL__CATALOG__STATIC_VALIDATION_ENABLED=false` skips startup comparison
  against `dataset.yaml` and `services/config/yolo_product_mapping.json` by
  default, preventing stale static files from warning after an engine swap.
  `yolo_product_mapping.json` is not used for runtime active-product
  allowlists.
- `MODEL__WEIGHT__IDENTITY_POLICY=vision_first` is the default product-identity
  policy. Product identity must come from final vision candidates, strong stage
  evidence, or weight-gated rescue evidence; loadcell-only and active-only
  candidates remain validation/count evidence instead of creating a charged
  product. `weight_aware` keeps the older loadcell/active fallback behavior for
  explicit legacy testing or field rollback.
- `MODEL__WEIGHT__FUSION_VISION_WEIGHT=0.65`,
  `MODEL__WEIGHT__FUSION_LOADCELL_WEIGHT=0.25`, and
  `MODEL__WEIGHT__FUSION_COUNT_WEIGHT=0.10` make fused confidence vision-heavy
  under the default policy.
- `MODEL__WEIGHT__STRICT_MODE=true`
- `MODEL__WEIGHT__STRICT_MODE_FALLBACK=true`
- `MODEL__WEIGHT__TOLERANCE_GRAMS=5.0` in code defaults, used by strict
  combination matching and relaxed count validation.
- Code default `MODEL__MACHINE__CABINET_TYPE=refrigerated` preserves the
  refrigerated loadcell/vision/weight policy. The copyable `.env.example` is
  freezer-first and sets `MODEL__MACHINE__CABINET_TYPE=freezer`, keeping
  zone-sliced loadcell input but using freezer-specific upper-half dual-top
  filtering and a vision-candidate-pool decision branch where weight validates
  ordered count/combination choices.
- `MODEL__WEIGHT__FREEZER_CONFIDENCE_TIE_BAND=0.08` controls how close
  freezer single-item fallback candidates must be in confidence before weight
  residual can break the tie.
- `MODEL__WEIGHT__FREEZER_MULTI_MIN_CONFIDENCE=0.45` is the freezer
  multi-kind vision evidence floor.
- `MODEL__WEIGHT__FREEZER_WEIGHT_TOLERANCE_GRAMS=15.0` is the freezer-only
  loadcell tolerance for ordered candidate-pool count and combination
  validation.
- `MODEL__WEIGHT__FREEZER_VISION_MULTI_WITHOUT_WEIGHT_ENABLED` is retained for
  config compatibility, but ordinary nonzero freezer deltas still require
  segment/combined-weight support inside the freezer tolerance before returning
  multiple product identities. `.env.example` sets this rollback flag to
  `false`.
- `MODEL__WEIGHT__FREEZER_DISTINCT_MIXED_PREFERENCE_ENABLED=true` lets an
  all-single mixed freezer basket replace a same-product repeat with the same
  total item count when the mixed residual is no more than
  `MODEL__WEIGHT__FREEZER_DISTINCT_MIXED_MAX_EXTRA_RESIDUAL_GRAMS=5.0` worse.
- `MODEL__WEIGHT__FREEZER_PRIOR_TRIGGER_DEDUPE_ENABLED=true` excludes products
  already selected by earlier freezer removal triggers in the same global door
  session before later freezer trigger solving. The current policy is
  fail-closed: if the remaining candidate pool cannot fit the later target, the
  trigger returns no-charge diagnostics rather than reusing an earlier product
  group.
- `MODEL__WEIGHT__MULTI_KIND_MIN_CONFIDENCE=0.18` is the per-item confidence
  floor for multi-kind combinations.
- `MODEL__WEIGHT__SAME_PRODUCT_COUNT_TOLERANCE_GRAMS=5.0` applies only to the
  repeated same-product count path.
- `MODEL__WEIGHT__SAME_PRODUCT_MAX_COUNT=8` caps repeated same-product removal
  or return matches without widening general multi-kind strict matching.
- `MODEL__WEIGHT__MAX_ITEMS_PER_SEGMENT=3` caps one detected loadcell removal
  segment to at most three product units. When two segment targets exist, the
  segment-derived aggregate cap is six; when no segment targets exist, the
  existing aggregate `SAME_PRODUCT_MAX_COUNT=8` behavior remains available.
- `MODEL__WEIGHT__MAX_COMBINATION_ITEMS=5` bounds strict matching by total
  units.
- `MODEL__WEIGHT__MAX_COMBINATION_KINDS=3` bounds strict matching by distinct
  product kinds.
- `MODEL__WEIGHT__DETECTED_SINGLE_FALLBACK_*` controls the final one-item
  fallback.
- `.env.example` now lists the same repeated-count and strict-search weight
  knobs as the Jetson templates so a copied env preserves these operational
  guardrails.
- `MODEL__TRIGGER__RETURN_VIDEO_SKIP_ENABLED=true`
- `MODEL__TRIGGER__RETURN_STABILIZATION_WAIT_SECONDS=1.0`
- `MODEL__TRIGGER__RETURN_STABILIZATION_REQUIRE_STABLE_REGIONS=true`
- `MODEL__TRIGGER__BALANCED_EVENT_CANCEL_ENABLED=true`
- `MODEL__TRIGGER__COOPERATIVE_CANCEL_ENABLED=true`
- `MODEL__TRIGGER__RAPID_SAME_ZONE_WINDOW_SECONDS=3.0` controls how far back
  trigger traces expose same-zone loadcell events for rapid follow-up
  disambiguation.
- `MODEL__TRACE__SAMPLE_EXPORT_ENABLED=false` in `.env.example` and Jetson
  templates keeps frame image export opt-in. Turning it on writes sampled frame
  files during inference and can add Jetson disk I/O.
- `MODEL__DOOR_SESSION__CLOSE_INITIAL_WAIT_SECONDS=3.0`
- `MODEL__DOOR_SESSION__CLOSE_SUBSEQUENT_WAIT_SECONDS=1.0`
- `MODEL__DOOR_SESSION__YAML_RETENTION_DAYS=7` controls completed
  door-session YAML cleanup on shutdown. Use host log rotation separately for
  frame trace JSONL files.
- `MODEL__VIDEO__READY_MAX_WAIT_SECONDS=2.0`
- `MODEL__VIDEO__READY_POLL_INTERVAL_SECONDS=0.2`
- `MODEL__VISION__TOP_CROP_POLICY=left`,
  `MODEL__VISION__SIDE_CROP_POLICY=left`, and
  `MODEL__VISION__CROP_WIDTH=480` keep inference on the left 480x480 crop from
  640x480 camera frames.
- Code default `MODEL__VISION__CAMERA_LAYOUT=legacy_top_side` preserves the
  current one Top plus per-zone Side camera mapping. The freezer `.env.example`
  sets `dual_top_proxy`, which keeps the `/trigger` `videos.top/side` contract
  but records `videos.top` as physical `top_middle` and `videos.side` as
  `top_side` using the Top processing profile.
- `MODEL__VISION__TOP_ROI_ENABLED=true`
- `MODEL__VISION__MOTION_MIN_DISPLACEMENT_PX=10.0`; tracker dynamic thresholds
  also use this as the minimum floor before applying the bbox-size rule.
- `MODEL__VISION__FREEZER_MOTION_MIN_DISPLACEMENT_PX=12.0` is the freezer
  motion floor used only when `MODEL__MACHINE__CABINET_TYPE=freezer`.
- The freezer field template uses `MODEL__VISION__FREEZER_MIN_VOTE_RATIO=0.08`
  and `MODEL__VISION__FREEZER_MIN_VOTE_COUNT=4` to tighten freezer candidate
  voting.
- `MODEL__VISION__FREEZER_ROI_VERTICAL_REGION=upper` and
  `MODEL__VISION__FREEZER_ROI_Y_SPLIT=240.0` keep freezer dual-top detections
  only when their 480x480 bbox center is in the upper half
  (`center_y <= 240`). `lower` remains a rollback value. The legacy
  `MODEL__VISION__FREEZER_LOWER_ROI_Y_SPLIT` key is a deprecated split fallback
  only and should not be used in new templates.
- `MODEL__VISION__FREEZER_MIN_EXIT_PATH_VOTES=3` remains part of freezer
  interaction diagnostics and legacy exit-path evidence. The current freezer
  handled filter no longer uses it to narrow candidates by loadcell residual
  before engine judgment.
- `MODEL__VISION__TOP_ROI_Y_SPLIT=240.0`; top-camera ROI uses bbox
  `center_y` with image top at `0`, and non-zero removal/return deltas both
  keep the lower region.
- `MODEL__VISION__TOP_CONFIDENCE_THRESHOLD=0.50` and
  `MODEL__VISION__SIDE_CONFIDENCE_THRESHOLD=0.50` are the current freezer field
  product vote floors. In freezer mode, product detections below the relevant
  raw/max camera floor cannot become regular votes, threshold/ROI rescue,
  weight-gated rescue, stage-count fallback, diagnostic fallback identity
  evidence, or close-time aggregate fallback candidates. Weighted/combined
  confidence remains diagnostic and is not the freezer identity floor.
- `MODEL__VISION__TOP_WEIGHT=0.60`,
  `MODEL__VISION__SIDE_WEIGHT=0.40`,
  `MODEL__VISION__TOP_ONLY_WEIGHT=0.60`, and
  `MODEL__VISION__SIDE_ONLY_WEIGHT=0.40` bias freezer dual-top ranking toward
  physical `top_middle` over physical `top_side`.
- `MODEL__VISION__SIDE_ROI_X_MAX=400.0` keeps the hard side ROI boundary in
  the field-tuned left-crop operating region.
- `MODEL__VISION__SIDE_ROI_SOFT_MARGIN_PX=5.0` opens a conditional regular
  candidate band through `center_x <= 405` for threshold-passed side detections
  that still survive motion filtering; low-confidence threshold rescue remains
  hard-ROI gated.
- The current freezer field template sets `MODEL__VISION__FFMPEG_TOP_GAMMA=1.0`,
  `MODEL__VISION__FFMPEG_TOP_CONTRAST=1.0`,
  `MODEL__VISION__FFMPEG_SIDE_GAMMA=1.0`, and
  `MODEL__VISION__FFMPEG_SIDE_CONTRAST=1.0`.
- `MODEL__VISION__ROI_RESCUE_REQUIRE_MOTION=true` and
  `MODEL__VISION__ROI_RESCUE_MAX_OVER_LIMIT_PX=0.0` keep right-side/static
  ROI-filtered detections from re-entering as rescue candidates.
- `MODEL__ASYNC_STREAMING__FRAME_STRIDE=1`; this accuracy-first default runs
  YOLO on every decoded frame. Set `2` only as a latency rollback.
- `docs/jetson-stride2.env.txt` remains a copy-paste rollback template for
  latency-first field operation.

## Validators And Safety

- API port must be `1..65535`.
- API log level is normalized and validated.
- Crop policy must be one of `left`, `center`, `right`, `offset`, `none`, or
  `letterbox`.
- Camera layout must be `legacy_top_side` or `dual_top_proxy`.
- Catalog source policy must be `node_first` or `static_mapping_compat`.
- Frame queue size must be positive. Frame stride must be `1` or `2`.
- Video readiness waits and door-session waits must be non-negative.
- Trace sample count must be non-negative.

## Jetson Environment Notes

- `pyproject.toml` does not encode system-site-packages. Jetson venv creation
  must use `uv venv --system-site-packages`.
- NumPy is pinned below 2.0.
- Do not let uv reinstall CPU-only torch into the Jetson venv.

## Related Wiki Pages

- [Repo overview](repo-overview.md)
- [Latency and frame stride](../synthesis/latency-and-frame-stride.md)
- [Scenario readiness and 0g diagnostics](../synthesis/scenario-readiness-and-0g.md)
- [Jetson and testing](../synthesis/jetson-and-testing.md)
