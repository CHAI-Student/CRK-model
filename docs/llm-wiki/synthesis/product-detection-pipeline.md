# Product Detection Pipeline

## Current Thesis

The model does not rely on a single YOLO frame. It accumulates frame-level
evidence, filters detections, votes across Top/Side cameras, and then uses
loadcell weight and active product data mainly to validate/count the
vision-supported product identity.

## Seven Stages

| Stage | Input | Output | Main code |
| --- | --- | --- | --- |
| 1. Trigger receipt | AVI paths, loadcell samples, zone | queued trigger item | `api/routes/trigger.py`, `service/trigger_service.py` |
| 2. Frame extraction | top/side AVI | BGR frames | `video/frame_extractor.py` |
| 3. YOLO inference | processed frames | `YOLODetection[]` | `vision/yolo_wrapper.py` |
| 4. Filtering | raw detections | filtered detections | `video/video_processor.py`, `vision/hand_path_tracker.py` |
| 5. Voting | filtered top/side detections | `VoteResult[]` / candidates | `video/voting_ensemble.py` |
| 6. Judgment | candidates, `delta_weight`, `active_products` | `JudgmentResult` | `engine/decision_engine.py`, `weight/strict_weight_matcher.py` |
| 7. Session response | trigger judgments | door/global basket | `session/door_session_store.py`, `session/product_aggregator.py` |

## Vision Processing

- YOLO is TensorRT/FP16-oriented on Jetson with 480x480 input.
- Runtime product identity/class mapping is Node-first by default, but the
  model resolves active-product class ids from Edge class-name keys matched
  against loaded YOLO engine class names. Official input is
  `product_eng_name`, tagged as `product_eng_name_engine`. During migration,
  engine-matching `name` is tagged `name_engine_compat` and legacy
  `product_name` is tagged `product_name_engine_legacy`. Direct
  `trainingidx`, `training_idx`, `trainingIdx`, `yolo_class_id`,
  `yoloClassId`, and `yolo_class_name` payload fields are ignored for runtime
  class identity because they can drift from the deployed engine.
- Top/Side frames are preprocessed with a left 480x480 crop from 640x480 camera
  frames by default, so ROI coordinates are interpreted in that 480x480 space.
- Camera layout is configurable without changing the `/trigger` payload shape.
  `legacy_top_side` means physical Top plus per-zone Side; `dual_top_proxy`
  means `videos.top` is top-middle and `videos.side` is top-side. In freezer
  mode both streams use the Top profile for upper-half dual-top filtering by
  default.
- The current Jetson field profile sets Top FFmpeg gamma/contrast to `1.2/1.2`
  and Side to `1.0/1.0`. Trigger traces record the active values and frame
  stride for latency/recall comparison.
- Raw detection threshold is intentionally low; later filters remove noise.
- Filters include confidence checks, top ROI, side ROI, motion, and hand path.
  Top ROI uses `delta_weight` only as an enablement/direction label: removals
  and returns both keep `center_y >= 240`, while zero/unknown delta skips the
  top ROI.
- Freezer filtering applies only when
  `MODEL__MACHINE__CABINET_TYPE=freezer`. With `dual_top_proxy`, both public
  streams must keep bbox centers in the configured freezer ROI. The field
  template uses the upper 240 pixels of the 480x480 crop
  (`FREEZER_ROI_VERTICAL_REGION=upper`, `center_y <= FREEZER_ROI_Y_SPLIT`),
  pass the freezer motion floor, and pass freezer vote thresholds. Threshold
  rescue and ROI rescue are disabled so weak/static evidence does not enter
  freezer candidates.
- Current freezer product confidence thresholds are `0.70` for both Top and
  Side, matching the smaller freezer product-set deployment. Hand tracking uses
  a separate `0.40` floor for class `0`, but freezer `dual_top_proxy` only
  enables hand class `0` and hand-path filtering on physical `top_middle`.
  Physical `top_side` remains product-only and cannot filter candidates through
  hand evidence.
- Motion filtering defaults to a 10px bbox-center movement floor and no longer
  fails open when all regular candidates are static; low-confidence moving
  evidence is still available through threshold rescue.
- Side ROI uses hard `center_x <= 400` by default, plus a conditional `+5px`
  regular-candidate soft band for threshold-passed side detections that still
  pass motion filtering. This restores Pepsi boundary detections near
  `x=402..404` without reopening far-right Trevi evidence. ROI rescue remains
  strict at the hard boundary (`roi_x_avg <= roi_x_limit`) and still requires
  motion. Threshold rescue also records same-class side ROI conflict and rejects
  weak low-confidence rescues when stronger evidence for that class was
  filtered outside the side ROI.
- Voting tracks per-class evidence such as count, max confidence, average
  confidence, top/side evidence, and vote ratio.
- Top-only and side-only cases have configurable voting weights; current
  freezer dual-top tuning biases physical `top_middle` over `top_side`
  (`top=0.60`, `side=0.40`, `top_only=0.60`, `side_only=0.40`,
  common-class bonus `0.20`).
- Final candidate ranking treats regular `source=vision` output as stronger
  than `roi_rescue` or `threshold_rescue` output, so high-quality YOLO/voting
  evidence is not pushed below rescue-only evidence during `top_k` trimming.

## Weight And Product Fusion

- Strict matching uses product weights from live `active_products`, not a stale
  static catalog.
- The active-product allowlist is based on `stock > 0` and a valid product
  class id. Missing or `0g` product weight is recorded as weight unavailable
  for loadcell validation/count, but it does not suppress vision inference.
- Product identity is vision-first by default. `MODEL__WEIGHT__IDENTITY_POLICY`
  defaults to `vision_first`, so final candidates, strong stage evidence, or
  weight-gated rescue evidence must support the product identity before a
  negative delta can create a charged product. `weight_aware` is the explicit
  legacy mode for active/loadcell fallback behavior.
- Loadcell evidence is still used for count and validation. When vision and
  loadcell agree, the result can be `complete`; when strong vision conflicts
  with loadcell, the vision product remains as `partial` with mismatch
  diagnostics; when no vision-derived identity exists, the result is no-charge
  `no_detection` or `uncertain`.
- Freezer mode narrows this further: final freezer candidates are the only
  chargeable identity source. The normal gram tolerance is not used to reject a
  strong freezer candidate. Raw vision top-K is preserved in diagnostics, but
  the handled candidate list sent to OPS, the engine, and DoorSession is
  narrowed for single freezer removal segments. In that case one product is
  selected by freezer exit-path evidence and weight residual, with top
  confidence-band weight residual used only as fallback. Freezer loadcell
  residual is now reliable to about `15g`, so multi-kind freezer output requires
  segment/compound or combined-candidate weight support inside
  `MODEL__WEIGHT__FREEZER_WEIGHT_TOLERANCE_GRAMS`. Strong dual-camera evidence
  without a weight fit records `freezer_multi_kind_weight_mismatch` and falls
  back to one handled freezer product.
- Freezer same-product repeats are allowed as a conservative candidate-filter
  and vision-first engine override. The repeat count is inferred from
  `target_weight / unit_weight`, then checked against confidence,
  stock/count caps, and freezer residual tolerance. Multi-candidate repeats
  still need freezer exit-path votes and vote count, but a single regular
  `vision` identity can use the tighter weight fit directly when `x2+` is
  inside tolerance and closer than `x1`. A top-only repeat can beat a top-only
  single only within the configured repeat residual gap; a dual-camera
  exit-path single remains preferred. Frame-level vote evidence is carried into
  the engine through `raw_vote_count`, while `vote_count` keeps its consensus
  meaning.
- Freezer interaction evidence now sits between raw vision and weight
  selection. The trace records actual path displacement, max movement, center
  span, trajectory support, static-shelf likelihood, upper-ROI hand validity,
  hand proximity counts/ratios, and hand-path pass/block state from physical
  `top_middle` only. Static top-only shelf-like candidates are demoted, while
  valid top-middle hand-path blocks become hard rejects only when at least one
  hand-near alternative candidate remains.
- When strong vision sees a stock-positive product whose Node weight is missing
  or `0g`, the product identity remains as `partial` with
  `vision_identity_preserved_weight_unavailable`; loadcell-derived count
  correction is not attempted without a positive weight.
- Fused confidence is now configurable and vision-heavy by default:
  `MODEL__WEIGHT__FUSION_VISION_WEIGHT=0.65`,
  `MODEL__WEIGHT__FUSION_LOADCELL_WEIGHT=0.25`, and
  `MODEL__WEIGHT__FUSION_COUNT_WEIGHT=0.10`.
- Relaxed fallback can attempt single product, combination, partial result, or
  loadcell-only behavior depending on available evidence.
- Recent decision recovery includes a detected-single-item fallback that can use
  strong class evidence plus product weight tolerance when strict matching is
  too narrow.
- Rejected rescue evidence is still useful after final candidates are empty.
  In detected-single fallback, a strong 450-560g bottle signal with side
  confidence, side votes, and motion evidence can replace a weak weight-only
  Trevi-style single fallback when the residual fits the narrow
  detected-single-plus-one-per-item grace window. This keeps Pepsi evidence from
  disappearing just because it missed the strict candidate weight gate.
- Strict combination matching now chooses compact vision-backed explanations
  first: fewer total units, then lower weight residual, then higher average
  vision confidence. This keeps A x1 + B x1 ahead of C x4 when both fit.
- Strict matching now uses motion/source evidence when a static one-item match
  competes with motion-supported repeated or two-kind candidates, covering
  rapid same-zone cases where the simpler weight explanation is not the handled
  product.
- Negative follow-up triggers can use returned-weight hints from compound
  loadcell segments and recent same-zone returns to avoid selecting a product
  that was just put back.
- When one time segment hides simultaneous same-zone removals, physical
  loadcell channel deltas can become evidence-required segment targets. A
  supported channel split such as Tteokbokki + Welchs is evaluated before a
  single aggregate threshold-rescue product with a tighter residual.
- Segment-first matching ranks weak stage traces as unsupported evidence. This
  prevents low-confidence small-product repeats from beating active large-bottle
  explanations or being reported as `COMPLETE`.
- In freezer mode, product stage/diagnostic/rescue evidence below the `0.70`
  product floor is also unsupported for identity creation. Trace/debug records
  can still show the rejected observation, but it cannot become a final
  product fallback.
- Stage-count evidence that passed the weight gate can be promoted to a
  synthetic `stage_weight_gate` candidate when active stock/weight are valid,
  votes meet the detected-single minimum, and confidence is at least `0.08`.
  This is the recovery path for products such as Pepsi that are visible in raw
  stage counts but sit below the regular candidate threshold. For single-item
  strict matches, candidate rank stays ahead of stage/source priority once all
  options are inside strict tolerance, so a lower-rank Trevi stage promotion
  cannot replace a higher-rank Sky Barley candidate at a tighter residual.
- Segment-first matching can split one merged loadcell segment into a compound
  two- or three-item option when at least one product is a final/trusted vision
  candidate and companions have supported or weak companion evidence. This lets
  a large segment such as Fanta + Pepsi compete directly with small-item repeat
  explanations such as Chapagetti x8.
- Segment targets impose a grip-size cap: one detected loadcell segment can
  explain at most `MODEL__WEIGHT__MAX_ITEMS_PER_SEGMENT` products. The template
  default is three, but setting the env value to `4` allows four total products
  in one segment. Without segment targets, existing aggregate count limits
  remain.
- Same-weight 500ml bottle segment collisions prefer repeated two-camera or
  very strong stage evidence over repeated reuse of one top-only final
  candidate. This prevents Trevi-style top-only evidence from being stretched
  across separated Pepsi-weight segments.
- Small-item repeats (`unit_weight < 200g`, `count >= 3`) are kept as recovery
  only when no valid trusted compound/single or tight large-item single
  alternative exists.
- Clean evidence-supported segment matches are kept ahead of aggregate repeated
  product overrides when their total residual is no worse. This lets separate
  loadcell changes choose Binch + Haruyache + Letsbe instead of collapsing into
  a same-product repeat such as Chapagetti x4.
- Stage/diagnostic-only small fragments are not clean support for aggregate
  repeat collisions. If a high-rank regular candidate such as Sky Barley
  explains the aggregate segment total as `x2` within count-scaled tolerance, it
  can beat a Condition/Hot6/Binch fragment basket that fits weight slightly
  better but depends on small products absent from final candidates.
- Do not lower global candidate thresholds to fix small-product filler
  regressions. Threshold rescue and diagnostics already preserve low-confidence
  evidence for recovery paths; lowering the regular Top/Side thresholds makes
  tiny products more likely to appear as noisy supported ingredients. The
  decision layer instead blocks aggregate strict/relaxed combinations from
  using `unit_weight < 200g`, `count >= 2` fragments unless they are regular
  final candidates or have strong motion-backed stage evidence. Medium drink
  repeats such as LetsBe `228g x2` can therefore beat a lower-residual
  HALUYACHE-plus-Condition fragment basket.
- Merged loadcell history now has its own purchase-delta selection: stable
  remove-return pairs and press-release pairs are ignored, while the latest
  unpaired removal can become the `decision_delta` even when the raw net delta
  is positive or misleading.
- When vision sees nothing or all regular branches miss, the default
  `vision_first` policy does not charge from the nearest active product. The
  older active-product fallback remains available only through
  `MODEL__WEIGHT__IDENTITY_POLICY=weight_aware`.
- Same-weight 500ml bottle regressions are handled with collision-specific
  guards, not by relaxing strict matching globally. A regular rank-1 final
  vision combination can replace a same-weight lower-rank strict winner inside
  count-scaled tolerance, covering Sky Barley + Fanta versus Corn Silk + Fanta.
  A strong regular Pepsi candidate can also win a two-bottle `x2` match over
  Trevi/Corn rescue or active-only collisions when the residual stays within
  the controlled count-scaled allowance. Single 500ml bottles do not receive
  that count-scaled grace: they must stay inside flat strict tolerance, and
  strict single-item options are then ordered by candidate rank before source
  priority or small residual gaps. The only single-bottle exception is the
  fallback-only identity override for empty-final-candidate cases, where
  rejected strong side/motion evidence can beat weak residual-only weight
  fallback without changing candidate output semantics.
- When strong 500ml bottle `x2` vision evidence conflicts with a negative
  loadcell delta that is materially under the expected two-bottle weight, the
  trigger worker does not sleep and then reuse the same stale payload as proof
  of stability. The model service can only judge the loadcell samples already
  POSTed by Camera/Node, so it stores a `waiting` session with
  `waiting_for="stable_loadcell"`, records
  `weight_diagnostics.removal_stabilization`, skips engine judgment, and waits
  for Camera/Node to retry with a newly stabilized loadcell tail.
- Final candidate combinations are authoritative over stage-count recovery:
  if the final candidates can explain the delta as a multi-item strict
  combination, that candidate-only result is selected before any
  `stage_counts_by_class` pair or supplement.
- If final candidates miss a combination, the engine retries strict
  combination matching with final candidates plus `stage_counts_by_class`,
  capped at 10 candidate classes total. This covers cases where the real
  products appeared in earlier filter stages but not in the final candidate
  list.
- Same-product repeated-count matching runs after strict/compact combinations
  miss, with a dedicated `5g` per-item tolerance. The repeated path is capped
  at `x8` and still respects stock and active-product evidence.
- Multi-kind combinations are conservative: each ingredient must meet
  `MODEL__WEIGHT__MULTI_KIND_MIN_CONFIDENCE=0.18`, and three-kind baskets are
  still rejected when any ingredient is below that floor.
- Return handling uses a hybrid policy. A positive delta immediately deducts
  one same-zone product only when a single unit matches inside flat strict
  tolerance. Multi-count returns, multi-product return combinations,
  cross-zone candidates, and mixed `return_weight_hints` are preserved as
  deferred return records and replayed at CLOSE against the final net delta.
- CLOSE finalization performs one basket-level final-weight validation after
  trigger aggregation. Ranked candidate snapshots saved on each removal trigger
  let an over-fragmented mixed basket be corrected to a supported same-product
  repeat such as Sky Barley `x3` when that repeat explains the final total
  delta inside count-scaled tolerance. When the current basket contains
  unsupported small fragments, the replacement may be worse by up to
  `base + same_product_count_tolerance`; clean supported mixed baskets still
  remain. This does not change individual `/trigger` responses. Return-bearing
  sessions are reconciled first through deferred returns; unresolved unmatched
  or cross-zone records can still block repeat correction.
- CLOSE final-weight validation is identity-stable for clean vision-supported
  baskets. If every current product id is backed by strong regular vision
  evidence, a different repeated candidate cannot replace the basket only
  because its loadcell residual is lower. Same-product count correction remains
  allowed for an already vision-supported product.
- CLOSE repeat correction applies the same operational grip cap used by
  trigger segments: `removal_trigger_count * max_items_per_segment`, also
  bounded by stock, same-product max count, and per-item max count. This keeps
  strong repeat evidence such as Sky Barley `x3`, while rejecting impossible
  corrections such as HomeRunBall `x33`.
- Low/zero-weight tail triggers can run video diagnostics when configured, but
  they skip engine judgment and stay excluded from CLOSE/payment. Active global
  sessions expose optional no-charge close diagnostics, such as
  `decisionSummary.diagnosticZoneLines` and zone-level
  `noChargeDiagnostics`, so a `0.0g` loadcell payload is visible without
  creating products or price.
- Active-only forced fallback does not convert tiny no-vision shelf movement
  into the nearest small product. If a `6-10g` negative delta has no candidate
  evidence and sits below the lightest active product minus strict tolerance,
  it remains a loadcell-only miss rather than charging Condition Stick.

## Failure Modes To Keep Separate

- Decode failure: ffprobe sees frames but async decode/retries yield zero
  frames, or async extractor/queue/YOLO tasks fail. This now propagates as an
  error path and must not be merged with a true vision miss.
- Capture timing failure: loadcell stable tail is missing, so delta is wrong.
- Removal stabilization conflict: vision strongly supports a two-bottle
  removal, but the posted negative loadcell delta is still undercounted; return
  `waiting_for=stable_loadcell` instead of charging a fallback product set.
- Vision miss: no usable YOLO/voting candidates after filters.
- Strict mismatch: candidates exist but do not match active product weights.
- Polling delay: Node checks result after model already finished.

## Evidence

- [Video and vision](../source-code/video-and-vision.md)
- [Decision and weight](../source-code/decision-and-weight.md)
- [Loadcell and trigger](../source-code/loadcell-and-trigger.md)
- [Product detection flow](../source-docs/product-detection-flow.md)
- [Product detection detail](../source-docs/product-detection-detail.md)
- [Architecture guide](../source-docs/agent-guides-architecture.md)
- [Trigger capture debugging](../source-docs/trigger-capture-debugging.md)
- [Trigger inference recovery notes](../source-docs/trigger-inference-recovery-notes-2026-03-31.md)
