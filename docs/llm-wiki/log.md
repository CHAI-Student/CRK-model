# LLM Wiki Log

## [2026-07-01] maintenance | freezer vision candidate pool ordered solver

- Retired freezer loadcell-residual candidate narrowing before engine judgment.
  `VideoProcessor.filter_freezer_handled_candidates()` now keeps every regular
  freezer vision candidate that passed the configured product threshold, ROI,
  motion, and valid top-middle hand-path gates. Stage-only, active-only, and
  weight-nearest products remain diagnostics only and cannot create freezer
  identity.
- Added the freezer ordered weight-combination solver in
  `ProductDecisionEngine`. It tests candidate-pool options in deterministic
  vision order: rank-1 `x1`, rank-2 `x1`, then same-product counts by count and
  rank, then mixed combinations by total count and rank. Weight residual is a
  pass/fail check inside `MODEL__WEIGHT__FREEZER_WEIGHT_TOLERANCE_GRAMS`, not
  an identity ranking signal.
- Recorded the cheese burger field failure mode. With target `183.7g`, rank-1
  `BAG_HANMAC_TRIPLE_CHEESE_HAMBURGER_155G` at `176g` is selected before a
  lower-rank `189g` dumpling because rank-1 `x1` already fits freezer
  tolerance.
- Updated the freezer field `.env.example` profile to
  `models/set9_imbalance_16.engine`, product thresholds `0.50/0.50`, hand
  threshold `0.30`, top/side weights `0.60/0.40`, freezer tolerance `15.0`,
  and Top FFmpeg gamma/contrast `1.0/1.0`.

## [2026-07-01] maintenance | freezer signed-net close aggregate simplification

- Superseded the freezer mixed-sign removal-total rule. Freezer trigger
  analysis now keeps the confirmed stable start/end net delta when one payload
  contains both positive and negative internal segments, so `+70g` then
  `-150g` remains `decision_delta=-80g` and `+47.6g` then `-295.3g` remains
  about `decision_delta=-247.7g`.
- Retired freezer mixed-sign `return_weight_hints`. Internal positive freezer
  segments are diagnostics-only; positive-only freezer return triggers still
  use the existing return path.
- Reworked `FreezerCloseAggregateResolver` to use signed global net deltas
  across participating freezer triggers. CLOSE clears participant products when
  the global net is near zero, solves negative net targets from handled/final
  trigger candidates and trigger products only, and attributes accepted baskets
  to the latest participating freezer trigger zone.
- Updated OPS/diagnostics to report `policy=signed_net_delta`,
  `globalNetDelta`, `finalTargetWeight`, selected products, residual, and
  no-charge reasons instead of raw negative totals and matched positive hints.

## [2026-07-01] maintenance | freezer hybrid close aggregate resolver

- Historical note: this hybrid raw-negative/positive-hint policy is superseded
  by the later signed-net close aggregate simplification above.
- Added a freezer-only Hybrid CLOSE aggregate resolver for unstable door-open
  sessions. Mixed-sign `return_weight_hints`, multiple meaningful negative
  freezer triggers, or negative freezer triggers spanning zones now cause CLOSE
  to solve one aggregate basket and attribute it to the latest participating
  freezer trigger zone.
- Kept simple single stable negative freezer triggers on the existing per-zone
  path. Empty no-product negative triggers inside freezer tolerance do not
  force aggregate eligibility, so small diagnostic/noise movements do not
  reroute an otherwise stable basket.
- Changed positive hint handling to Return If Matched. The resolver first
  solves the raw negative removal target, then subtracts a positive hint only
  when selected products match that weight and residual improves. Unmatched
  positive hints are recorded as pressure/artifact diagnostics.
- Added close-time `weightDelta` overrides and
  `final_weight_validation.freezerCloseAggregate` diagnostics, plus
  `[OPS][FREEZER-CLOSE-AGGREGATE]` logging and regressions for mixed-sign
  reroute, unmatched positive hints, multi-zone aggregate solving, simple
  freezer preservation, and refrigerated non-application.

## [2026-07-01] maintenance | freezer full-delta repeat repair

- Closed the freezer `-309.5g` bagel failure mode end-to-end. A single regular
  `BAG_NULLDAM_BAGEL_140G` candidate with `156g` unit weight and confidence
  above `MODEL__WEIGHT__FREEZER_MULTI_MIN_CONFIDENCE=0.45` can now become
  `x2` through `repeatEvidenceMode=single_regular_vision_identity`, yielding
  about `312g` explained weight and a `2.5g` residual.
- Tightened direct `freezer_vision_first` output: when product weights are
  valid and positive, the final basket must explain the full stable negative
  delta inside `MODEL__WEIGHT__FREEZER_WEIGHT_TOLERANCE_GRAMS=15.0`. If repeat
  or multi-kind fitting cannot explain the delta, the engine returns no-charge
  `UNCERTAIN` with `final_weight_mismatch_guard` instead of preserving a
  mismatched chargeable `PARTIAL x1`.
- Extracted shared freezer repeat/count diagnostics into
  `video/freezer_candidate_policy.py` so `VideoProcessor` handled filtering and
  `ProductDecisionEngine` repeat correction report the same count, expected
  weight, residual, rejection reason, and `repeatEvidenceMode`.
- Added trigger/CLOSE regression coverage for the field log shape and updated
  OPS diagnostics to distinguish pre-engine candidate `count_hint` from final
  engine `product_count`.

## [2026-07-01] maintenance | freezer top-middle hand filtering and vote bias

- Limited freezer `dual_top_proxy` hand-path filtering to the physical
  `top_middle` stream. `videos.side` still maps to physical `top_side` and
  uses the Top processing profile for product detections, but its inference
  allowlist is product-only and its hand detections cannot update
  `HandPathTracker` or filter candidates.
- Rebalanced freezer dual-top voting toward `top_middle`:
  `MODEL__VISION__TOP_WEIGHT=0.60`,
  `MODEL__VISION__SIDE_WEIGHT=0.40`,
  `MODEL__VISION__TOP_ONLY_WEIGHT=0.60`, and
  `MODEL__VISION__SIDE_ONLY_WEIGHT=0.40`.
- Added regressions for top-side hand detections failing open without filtering
  candidates, top-middle hand detections still filtering candidates, per-camera
  inference allowlists, and top-biased default ensemble ranking.

## [2026-07-01] maintenance | freezer mixed-sign removal delta guard

- Historical note: this freezer removal-total guard is superseded by the later
  stable-net freezer mixed-sign rule above.
- Fixed freezer mixed return/removal payloads where a positive return segment
  masked a larger removal in the start/end net delta. In freezer mode,
  unpaired negative segment total now feeds product judgment when both
  positive and negative unpaired stable segments exist, so `+70g` then `-150g`
  is judged as a `150g` removal instead of the masked `80g` net.
- Kept the positive segment as `return_weight_hints` for CLOSE same-zone or
  cross-zone reconciliation, preserving the public trigger schema.
- Added `mixed_sign_net_masking_guard` trace diagnostics and an OPS loadcell
  line with net, return total, removal total, and selected decision delta.

## [2026-07-01] maintenance | freezer single-candidate bagel repeat gate

- Fixed the remaining bagel `x2` field miss where `BAG_NULLDAM_BAGEL_140G`
  appeared as the only regular vision candidate at `-309.5g`, but stayed `x1`
  because repeat inference required freezer exit-path and frame-vote gates
  before checking the tight `156g x2 = 312g` fit.
- Added a freezer-only `single_regular_vision_identity` repeat path: when the
  only candidate identity is a regular vision product, confidence is above the
  freezer floor, stock/count caps allow it, and the repeated count is within
  `MODEL__WEIGHT__FREEZER_WEIGHT_TOLERANCE_GRAMS` while closer than `x1`, the
  count can promote to `x2+` even with weak exit-path evidence.
- Extended freezer candidate-filter OPS diagnostics with selected count,
  expected weight, count residual, and first repeat rejection reason.

## [2026-07-01] maintenance | freezer bagel repeat count fix

- Fixed freezer same-product repeat counting for cases such as
  `BAG_NULLDAM_BAGEL_140G` where one visible candidate with
  `instance_count_hint=1` still matches the measured removal as `x2`
  (`156g x2 ~= 313g`).
- Preserved frame-level `VoteResult.vote_count` as `EnsembleResult.raw_vote_count`
  in both trigger conversion paths, while keeping `EnsembleResult.vote_count`
  as the existing Top/Side consensus scale.
- Added regressions for bagel-only freezer `x2`, conversion vote preservation,
  handled-filter repeat diagnostics, and stock/residual/vote-count rejection
  guards.

## [2026-07-01] maintenance | freezer product and hand confidence floors

- Raised current freezer product vote defaults to `0.70` for Top and Side, and
  added the separate `MODEL__VISION__HAND_CONFIDENCE_THRESHOLD=0.40` hand
  tracking floor while keeping hand class id `0`.
- Documented that product inference allowlists expand to include hand class
  `0` only when active product classes exist. Final product filtering, rescue
  filtering, and product outputs remain product-only and the empty allowlist
  path stays fail-closed.
- Recorded freezer-only identity guards: product evidence below the product
  confidence floor cannot create regular votes, rescue candidates,
  stage-count fallback, or diagnostic fallback identity evidence.

## [2026-06-30] maintenance | freezer stage-only rescue guard

- Documented that freezer stage-only rescue is candidate-miss recovery, not an
  override for a supported product already selected by the video handled
  filter. Direct `freezer_vision_first` now rejects stage-only resurrection for
  classes the handled-filter considered but did not select.
- Recorded the ambiguous dual-camera stage-only priority guard: when a
  non-stage candidate has a strict freezer weight-gate fit, stage-only
  ambiguous evidence is demoted into normal single ranking instead of taking
  the special priority tier.
- Added regression coverage for the zone2 Yomamte/Melona collision where the
  video handled-filter selected Yomamte but engine stage-only rescue had
  previously switched the result to Melona.

## [2026-06-30] maintenance | freezer same-tier residual ordering

- Updated freezer handled-candidate and direct `freezer_vision_first`
  documentation so same-tier single candidates sort by weight residual before
  raw freezer exit-path vote volume. Exit-path votes still gate candidates and
  resolve residual ties, but no longer let a worse-residual top-only candidate
  beat a tighter supported single.
- Documented the multi-segment freezer trace behavior change: compound or
  multiple removal segment evidence no longer automatically passes raw
  candidates through. The filter first checks viable multi-kind weight fits,
  strict/near single fits, and same-product repeats, then records
  `multi_item_trace_evidence_passthrough_unresolved` only when it must fail
  open.
- Added regression coverage for the zone4 Melona/Yomamte collision in both the
  video handled-filter path and direct decision engine path, plus unresolved
  multi-segment fail-open coverage.

## [2026-06-30] maintenance | freezer env template and auto launcher

- Documented the user-level `model-service` launcher installed by
  `scripts/setup_jetson.sh`. The launcher lives at `~/.local/bin/model-service`,
  activates this repo's `.venv`, and execs `.venv/bin/model-service` so fresh
  shells can start the service without manual activation once PATH is loaded.
- Recorded that `.env.example` is now a sanitized freezer-first dual-top
  template using `MODEL__MACHINE__CABINET_TYPE=freezer`,
  `MODEL__VISION__CAMERA_LAYOUT=dual_top_proxy`,
  `MODEL__VISION__YOLO_MODEL_PATH=models/set7_v8best.engine`, and placeholder
  secrets.
- Added regression coverage for `.env.example` parsing plus Bash-gated shell
  syntax and launcher dry-run behavior.

## [2026-06-30] maintenance | freezer static and hand interaction evidence

- Added freezer interaction diagnostics for path displacement, max movement,
  center span, trajectory pass, static shelf likelihood, and hand-path
  pass/block state in trace `stage_counts_by_class` and freezer selection
  diagnostics.
- Documented the new generic static-shelf guard: top-only static candidates are
  softly demoted unless trajectory or hand-path support exists. Valid hand-path
  blocks can hard-reject a candidate only when another candidate remains, so
  hand-path all-blocked cases still fail open.
- Recorded regression coverage for static top-only tight singles losing to
  trajectory-supported candidates, hand-path blocked candidates losing to
  alternatives, hand-path fail-open, and direct `freezer_vision_first`
  alignment with the video handled-filter path.

## [2026-06-30] maintenance | freezer count-aware handled candidate filtering

- Documented freezer same-product repeat inference for handled candidates and
  direct `freezer_vision_first` selection. A final vision candidate can become
  `same_product_repeat_weight_gate` from `target_weight / unit_weight` even
  when `instance_count_hint=1`, while still requiring freezer exit-path votes,
  vote count, confidence, stock/count caps, and both freezer/count-scaled
  residual gates.
- Recorded the conservative selection guards: a repeat may beat a top-only
  single only within the configured repeat residual gap, and dual-camera
  exit-path singles stay preferred over weaker top-only repeats.
- Added trace diagnostics for `sameProductRepeatCandidates`,
  `rejectedSameProductRepeatCandidates`, selected `count`, `expectedWeight`,
  and `countWeightResidual`, plus regression coverage for the zone5
  `BAG_NULLDAM_BAGEL_140G x2` versus jajangbab single collision.

## [2026-06-29] maintenance | freezer weight-gated multi-candidate repair

- Recorded that freezer loadcell matching is again reliable enough to gate
  product selection with `MODEL__WEIGHT__FREEZER_WEIGHT_TOLERANCE_GRAMS=15.0`.
- Updated freezer single-removal policy so strong top-1/top-2/top-3 freezer
  candidates are not returned together unless their combined/segment weight
  fits the measured delta. A `178g` removal with several `170g` candidates now
  resolves to one handled product instead of a roughly `500g` basket.
- Replaced the old vision-only freezer multi exception with
  `freezer_multi_kind_weight_mismatch` diagnostics and a single-candidate
  fallback path for ordinary nonzero freezer deltas.

## [2026-06-29] maintenance | CRK async video failure propagation and ops hardening

- Ingested the CRK 2026-06-29 feedback note as an external review source
  summary and scoped implementation to CRK-model only.
- Documented that fatal async video extractor, frame queue, zero-frame retry,
  and YOLO task failures now propagate as model-service exceptions instead of
  becoming empty no-detection results.
- Recorded `/api/health/detailed` runtime diagnostics for NumPy, Torch CUDA,
  and TensorRT, the repo-local TensorRT `.engine` export helper, and host
  logrotate responsibility for `frame_split_*.jsonl` trace files.

## [2026-06-29] maintenance | freezer deferred candidate repair and filter visibility

- Documented that freezer strict handled-candidate filtering requires both
  `MODEL__MACHINE__CABINET_TYPE=freezer` and
  `MODEL__VISION__CAMERA_LAYOUT=dual_top_proxy`; changing only cabinet type
  leaves handled narrowing disabled and now emits OPS/config diagnostics.
- Added freezer candidate-filter visibility to trigger traces and OPS logs:
  camera layout, raw/handled candidate counts, filter reason, and key freezer
  vote/motion/ROI/exit-path thresholds.
- Added CLOSE-time freezer deferred candidate repair for single-removal
  no-product or final-weight-mismatch partial baskets. It can select a later
  unused same-session candidate by weight while rejecting candidates already
  consumed by their own matched trigger result.

## [2026-06-29] maintenance | freezer endpoint loadcell fallback

- Documented the freezer-only endpoint fallback for nonzero Camera loadcell
  histories where stable plateau analysis would otherwise leave
  `decision_delta=0.0`. A continuous payload such as `+10g -> -60g` can now be
  treated as `-70g` when sample count/span and endpoint high/low checks pass.
- Clarified that missing, invalid-only, all-zero, and too-short payloads remain
  no-charge diagnostics; the fallback is separate from the existing 0g
  diagnostic branch.
- Added regression coverage for endpoint fallback acceptance, default
  refrigerated non-application, short-sample rejection, invalid/all-zero
  rejection, and trigger trace metadata.

## [2026-06-29] maintenance | freezer vision-supported multi and no-charge diagnostics

- Kept the `0g`/low-weight trigger contract fail-closed: engine judgment is
  skipped, products/prices stay out of DoorSession/payment, and diagnostic
  video processing remains trace-only when enabled.
- Added close-time no-charge visibility for skipped loadcell payloads.
  `decisionSummary.diagnosticZoneLines` and
  `decisionSummary.zones[*].noChargeDiagnostics` now expose reasons such as
  `loadcell_payload_all_zero` so a zone 1 `0.0g` event is not mistaken for a
  pure vision miss.
- Added the freezer rollback flag
  `MODEL__WEIGHT__FREEZER_VISION_MULTI_WITHOUT_WEIGHT_ENABLED=true`.
  When enabled, strong dual-camera freezer exit-path evidence can preserve
  multiple product identities as a `partial` freezer result even when the
  combined loadcell residual does not fit.
- Updated freezer handled-candidate policy: weak/static freezer noise is still
  narrowed to a single handled candidate, but strong multi-product freezer
  exit-path evidence is passed through to the decision engine.

## [2026-06-19] maintenance | freezer handled-candidate narrowing

- Documented the freezer dual-top handled-candidate split: raw top-K vision
  candidates remain in trace diagnostics, while `trace.candidates`,
  `[OPS][CANDIDATES]`, engine input, and DoorSession close snapshots use only
  handled candidates.
- Recorded that single-removal freezer events default to one handled product.
  `freezerExitPathVotes` plus weight residual can rescue handled products ahead
  of static shelf false positives; top confidence-band weight residual remains
  the fallback. `instance_count_hint` becomes count evidence only when the
  target weight supports the hinted count.
- Tightened freezer multi-kind behavior: multiple visible products no longer
  create a basket by confidence alone. Multi-kind output requires segment or
  compound loadcell evidence, or a combined candidate weight that fits the
  existing count-scaled tolerance.
- Updated freezer CLOSE behavior: unresolved final-weight mismatches preserve
  detected Edge-facing `products`/`totalPrice` and expose
  `finalWeightValidation.outputPolicy=products_as_detected`.
- Updated the Jetson stride-2 env template for freezer field operation:
  `MODEL__MACHINE__CABINET_TYPE=freezer` with
  `MODEL__VISION__CAMERA_LAYOUT=dual_top_proxy`.

## [2026-06-18] maintenance | freezer vision-first zone loadcell policy

- Removed the whole-cabinet loadcell decision contract from the docs: all
  cabinet types now use zone-sliced `/trigger.loadcells`.
  `global_loadcells` remains only a deprecated compatibility field.
- Documented freezer `dual_top_proxy` behavior: public `videos.top/side`
  remain unchanged, but the streams are treated as top-middle/top-side and
  freezer candidates must pass lower-half ROI, stronger motion, and stronger
  vote filters. Threshold/ROI rescue is disabled for freezer candidates.
- Documented the freezer decision branch: final vision candidates are the only
  chargeable identity source, confidence ranks first, and loadcell weight is
  recorded as a tie-break/diagnostic instead of a hard reject gate.
- Updated product class-key notes: `product_eng_name` remains official, while
  engine-matching `name` and legacy `product_name` are temporary compatibility
  keys. `trainingidx`, `yolo_class_id`, `yolo_class_name`, and stale
  `yolo_product_mapping.json` data are still ignored for runtime identity.
- Clarified that YOLO engine startup log output `name=` is an engine class
  label, not an Edge payload field name.

## [2026-06-17] maintenance | historical product_eng_name runtime class key

- Historical note superseded by the 2026-06-18 compatibility bridge:
  `product_eng_name` remains official, but engine-matching `name` and legacy
  `product_name` are accepted during Edge migration.
- At that point, active product class resolution was narrowed to Edge
  `product_eng_name` matched against the loaded YOLO engine class name. The
  later 2026-06-18 bridge adds engine-matching `name` and legacy
  `product_name` compatibility.
- Recorded that `trainingidx`, `yolo_class_id`, `yolo_class_name`, `product_idx`,
  and stale `services/config/yolo_product_mapping.json` ids are ignored for
  active-product runtime class identity.
- Updated diagnostics language around empty allowlists to focus on unmapped or
  missing `product_eng_name` instead of invalid direct class ids.
- Tightened the field-name contract so camelCase and shortened aliases are not
  accepted for runtime class identity.
- Added `MODEL__VISION__LOG_ENGINE_CLASSES=off` as a startup debug toggle; set
  it to `on` to print every loaded YOLO engine class id/name.

## [2026-06-17] maintenance | historical Korean product_name display-only

- Removed the remaining class-id fallback that compared display `product_name`
  against engine/static class names.
- Recorded the official contract that `product_name` is Korean display
  metadata. The later 2026-06-18 bridge accepts legacy `product_name` only when
  it already matches a loaded engine class name.

## [2026-06-17] maintenance | product_eng_name class-key contract

- Recorded that Edge sends `product_name` as the Korean display name and
  `product_eng_name` as the YOLO engine class name.
- Updated the model-side catalog contract so `product_eng_name` is resolved as
  the stable class key, while `product_idx` remains an external product
  identifier.
- Clarified that engine-loaded class names win over stale static
  `yolo_product_mapping.json` rows with the same class name.

## [2026-06-17] maintenance | Node-first product-name fallback bridge

- Historical note superseded by the later class-key changes.
  `MODEL__CATALOG__PRODUCT_NAME_FALLBACK_ENABLED` remains as a legacy
  deployment flag; current runtime compatibility is controlled by engine-name
  matching, not static name fallback.
- Runtime active-product class identity now ignores direct class-id fields and
  `yolo_class_name`; current allowlists come from engine-matching
  `product_eng_name`, `name`, or legacy `product_name`.
- Recorded sibling-repo follow-up risks that remain outside this CRK-model-only
  change: Edge/Camera/IO hardcoded URLs, missing Camera retry for
  `waiting_for=stable_loadcell`, and stale IO Board protocol docs.

## [2026-06-16] maintenance | Waiting loadcell plus active snapshot diagnostics

- Documented that `removal_waiting_for_stable_loadcell` can now preserve the
  active-product fail-closed reason when Node inventory context is missing at
  the same time as the Camera/IO loadcell payload lacks a stable tail.
- Updated loadcell, observability, session, and index pages for the new
  `SessionData.failure_reason`, trace final-result failure reason, and
  multi-zone waiting `failureReasons` behavior.
- Left raw operational trace JSON files and `result.xlsx` untouched.

## [2026-06-15] maintenance | Node-first catalog and zero-weight vision path

- Historical note: `MODEL__CATALOG__SOURCE_POLICY=node_first` remains the
  default runtime catalog mode, but direct `trainingidx`/`yolo_class_id`
  class-id aliases have been superseded by engine-name class-key matching.
- Recorded that static `dataset.yaml` and `yolo_product_mapping.json`
  validation is opt-in through
  `MODEL__CATALOG__STATIC_VALIDATION_ENABLED=true`; those static files do not
  populate runtime active-product allowlists.
- Updated startup, configuration, video/vision, decision/weight, and pipeline
  pages for zero-weight active products: they stay in the vision allowlist but
  are marked unavailable for loadcell count/validation.
- Recorded the vision-first missing-weight behavior:
  `vision_identity_preserved_weight_unavailable` returns the vision product as
  `partial` instead of falling back to active/loadcell identity.

## [2026-06-11] maintenance | CLOSE vision identity stability

- Documented that CLOSE final-weight correction preserves all-regular
  vision-supported current basket identities and blocks lower-residual
  different-product repeat replacements with `identitySwapBlocked=true`.
- Recorded that CLOSE can still adjust counts for the same vision-supported
  product id when final weight validation supports a different count.

## [2026-06-11] maintenance | Vision-first loadcell policy

- Added `MODEL__WEIGHT__IDENTITY_POLICY=vision_first` as the default decision
  policy and documented `weight_aware` as the explicit legacy fallback mode.
- Documented configurable confidence fusion defaults of vision `0.65`,
  loadcell `0.25`, and count `0.10`.
- Recorded that loadcell-only and active-only candidates no longer create
  product identity under the default policy; loadcell validates/counts
  vision-supported identity and produces mismatch diagnostics when it conflicts.

## [2026-06-11] maintenance | Camera layout role trace

- Added `MODEL__VISION__CAMERA_LAYOUT` with `legacy_top_side` and
  `dual_top_proxy` layouts while preserving the public `/trigger`
  `videos.top/side` contract.
- Recorded camera layout and logical-to-physical role mapping in frame trace
  summary/detail JSON so operators can distinguish a physical Side stream from
  the dual-top proxy stream. The current freezer mapping names those physical
  streams `top_middle` and `top_side`.

## [2026-06-04] maintenance | Documentation command sync

- Updated active README, AGENTS, test README, build-test, and Jetson setup
  docs to stop recommending removed startup/API helper test files.
- Recorded the current code-backed default engine path as
  `models/0204_morning.engine` and left the older conflicting engine-path
  claim only in historical source docs.
- Added scenario fixture and verification scripts to the scripts code map and
  recorded the 2026-06-04 local gate: Ruff over `services/model scripts` and
  the full model test suite with `351 passed`.

## [2026-06-02] maintenance | Full-delta matched-only finalization

- Added the matched-only finalization rule for chargeable removals. Engine
  `COMPLETE`/`PARTIAL` results must explain the full stable negative delta
  inside existing branch tolerances, otherwise the result becomes no-charge
  `UNCERTAIN` with `final_weight_mismatch_guard` diagnostics.
- Documented the Haluyache + LetsBe + Jagabee regression. Failed physical
  channel splits now fall back to ordinary time-based removal segments, and
  segment-local stage/rescue evidence can complete the full product set before
  active-only forced fallback.
- Recorded that forced final fallback rejects partial purchase targets such as
  `last_unpaired_negative_segment` unless they equal the full removal delta,
  and that CLOSE clears unresolved mismatched baskets with
  `finalWeightValidation.reason=unresolved_final_weight_mismatch`.

## [2026-06-02] maintenance | Accuracy-first loadcell and candidate priority

- Documented stable-tail-only chargeable loadcell deltas: first/last samples
  and raw max/min extremes are retained as diagnostics, while removals require
  a confirmed stable final plateau before `decision_delta` can charge.
- Recorded the broader removal stabilization policy. Unstable, truncated, or
  simple-fallback negative removals return `waiting_for=stable_loadcell`,
  skip video and `engine.judge()`, and stay out of DoorSession/payment
  aggregation until a stable payload arrives.
- Updated decision ordering so no-final-candidate stage-count/rescue
  combinations run before loadcell-only and active-only forced fallback, with
  trace branch diagnostics for `stage_count_combination_match`.
- Added regression-map coverage for stable-tail diagnostics, unstable removal
  waiting, ranked candidate repeats over unseen active repeats, no-final
  stage-count combination recovery, and normal session lifecycle logging.

## [2026-06-02] maintenance | CLOSE deferred returns and fixed stride 2

- Documented the hybrid return policy: strict same-zone single returns can be
  deducted immediately, while same-product `x2+`, combo, cross-zone, and mixed
  return-hint deltas are stored in `DoorSession.deferred_returns` until CLOSE.
- Recorded CLOSE deferred return reconciliation diagnostics under
  `finalWeightValidation.deferredReturnReconciliation`, followed by net-delta
  validation and final repeat correction.
- Added the CLOSE repeat count cap
  `min(stock_qty, same_product_max_count, max_count_per_item,
  removal_trigger_count * max_items_per_segment)`, with
  `count_exceeds_close_repeat_cap` for rejected repeat corrections such as
  HomeRunBall `x33`.
- Updated async streaming policy: `MODEL__ASYNC_STREAMING__FRAME_STRIDE=2` is
  fixed, settings reject other values, and the processor runtime also pins
  stride to `2`.

## [2026-06-02] maintenance | LetsBe repeat over unsupported small fragments

- Documented the LetsBe/HALUYACHE regression where a small product absent from
  final candidates was used as residual filler in a mixed basket.
- Recorded the new unsupported small repeat fragment rule:
  `unit_weight < 200g` with `count >= 2` is rejected from aggregate
  strict/relaxed combinations unless it is a regular final vision candidate or
  has strong motion-backed stage evidence.
- Clarified that regular candidate thresholds should not be lowered for this
  class of issue; low-confidence evidence remains available through
  rescue/diagnostic paths, while decision-time guards prevent tiny noisy
  fragments from becoming charged products.

## [2026-06-02] maintenance | Pepsi detected-single identity override

- Documented the zone 5 Pepsi/Trevi single-bottle fallback regression: Pepsi had
  strong side/motion stage evidence but missed the final candidate weight gate,
  while weak Trevi won detected-single fallback only by residual.
- Added the fallback-only 500ml identity override rule. Strong rejected bottle
  evidence can replace a weak residual-only single fallback inside
  `detected_single_fallback_tolerance + same_product_count_tolerance`; strict
  single-candidate ordering and OPS final-candidate logs remain unchanged.
- Added diagnostics guidance for
  `weight_diagnostics.detected_single_item_fallback.single_bottle_identity_override`
  so operators can see when rejected Pepsi evidence was used after candidates
  were empty.

## [2026-06-02] maintenance | SKY repeat over fragment baskets

- Documented the Sky Barley x3 regression where a lower-residual
  Condition/Hot6/Binch fragment basket survived until CLOSE even though ranked
  Sky evidence explained the final delta inside count-scaled tolerance.
- Recorded the segment aggregate rule: stage/diagnostic-only small fragments are
  not clean support when a high-rank regular same-product repeat explains the
  aggregate total.
- Updated CLOSE correction guidance so unsupported small-fragment baskets can
  use `base + same_product_count_tolerance` residual-gap allowance while clean
  supported mixed baskets and return/cross-zone sessions remain protected.

## [2026-06-02] maintenance | Low-delta active-only fallback noise guard

- Documented the no-vision `6-10g` shelf-shake regression where loadcell-only
  matching failed strict tolerance and forced final fallback charged Condition
  Stick as the nearest tiny active product.
- Recorded the new active-only low-weight noise guard: without vision/stage or
  purchase-delta evidence, deltas below the lightest active product minus
  strict tolerance stay as loadcell-only misses instead of creating a product.
- Kept real small-item behavior separate: a true Condition Stick-sized delta
  can still match through strict loadcell nearest-single behavior.

## [2026-06-02] maintenance | CLOSE final-weight candidate correction

- Documented the CLOSE-only final basket validation that compares each zone's
  effective negative delta with the final aggregated product weight.
- Recorded the new DoorSession candidate snapshot contract: trigger results
  persist ranked vision candidates joined with active weight, price, stock, and
  top/side evidence for close-time correction.
- Clarified that over-fragmented mixed baskets can be replaced by a supported
  same-product repeat inside count-scaled tolerance, while return/cross-zone
  sessions and clean supported mixed baskets are preserved.

## [2026-06-02] maintenance | Strict single rank priority over stage promotion

- Documented the zone 4 Sky/Trevi regression root cause: Sky Barley was the raw
  strict single winner at `delta=-525g` (`523g`, residual `2g`), but lower-rank
  Trevi was promoted to `stage_weight_gate` and source priority moved it ahead.
- Updated strict single-item policy: once candidates are inside flat strict
  tolerance, candidate rank sorts before source priority, residual, and
  evidence for `vision`, `threshold_rescue`, and `stage_weight_gate`.
- Clarified stage weight-gate as a recovery path, not a reason to override an
  already valid higher-rank strict single candidate.

## [2026-06-02] maintenance | Removal stabilization and single-bottle strict guard

- Documented the Pepsi x2 undercount contract: when strong regular 500ml
  bottle repeat evidence conflicts with a materially short negative loadcell
  delta, the trigger worker saves a `waiting` session, records
  `weight_diagnostics.removal_stabilization`, skips `engine.judge()`, and waits
  for a new stable loadcell payload.
- Updated same-weight identity guidance: single bottles keep flat strict
  tolerance, so a Trevi `530g` rank-1 candidate at `521g` no longer overrides a
  tighter Corn/Pepsi `520g` weight-gated rescue; controlled count-scaled grace
  remains available only for strong 450-560g `x2` repeats and rank-aware
  combinations.
- Recorded stage weight-gate promotion for already-present non-regular
  candidates, plus the multi-zone `status="waiting"` response contract with
  `reason="waiting_for_stable_loadcell"`.

## [2026-06-02] maintenance | Vision identity collision grace

- Documented the Sky/Corn/Fanta and Pepsi/Trevi regression fix: regular
  rank-1 final vision combinations and strong regular 500ml bottle repeats can
  use count-scaled grace only inside same-weight lower-rank/rescue/active-only
  collision guards.
- Recorded forced fallback behavior for Pepsi-only evidence: supported
  same-product repeat pairs can use the same controlled repeat allowance and
  outrank mixed detected-plus-active pairs that inject unsupported Trevi.
- Updated return aggregation notes: single returns keep flat tolerance, while
  multi-product return combinations use count-scaled tolerance and record
  `last_return_diagnostics`.

## [2026-06-01] maintenance | Pepsi side ROI soft-band recovery

- Documented the candidate miss root cause from field traces: Pepsi had strong
  raw/threshold evidence, but side centers around `x=402..404` were clipped by
  hard `side_roi_x_max=400` and fell back to `threshold_rescue`.
- Added the hard-ROI-plus-soft-band policy: keep hard side ROI at `400`, allow
  threshold-passed side detections through `405` only for regular candidate
  formation and motion filtering, and leave threshold/ROI rescue hard-gated.
- Updated configuration, video/vision, observability, pipeline, and tests maps
  for `MODEL__VISION__SIDE_ROI_SOFT_MARGIN_PX=5.0`, Top gamma/contrast
  `1.2/1.2`, trace `vision_config` timing fields, and Pepsi/Corn same-weight
  candidate-priority coverage.

## [2026-06-01] maintenance | Simultaneous channel split recovery

- Documented `channel_removal_segment_targets` and
  `channel_delta_diagnostics` for same-zone simultaneous removals where
  physical loadcell channels split one aggregate time segment.
- Recorded decision behavior: channel targets are evidence-required, single
  product per channel, and a supported split such as Tteokbokki + Welchs beats
  a same-weight aggregate `threshold_rescue`.
- Added test-map coverage for channel target extraction, positive-channel and
  single-channel rejection, supported channel split selection, and active-only
  channel fallback to aggregate strict matching.

## [2026-06-01] maintenance | Pepsi stage weight-gate recovery

- Documented `source=stage_weight_gate` candidate promotion for classes that
  pass the stage weight gate, have valid active stock/weight, meet the
  detected-single vote minimum, and reach confidence `>=0.08`.
- Recorded same-weight 500ml bottle collision handling for separated segment
  targets: repeated two-camera or very strong stage evidence can explain all
  bottle-weight segments, while a single top-only candidate is rejected with
  `repeated_segment_evidence_insufficient`.
- Updated trace/test maps for `stage_weight_gate_candidates`,
  `same_weight_bottle_collision`, and `repeated_segment_reuse_guard`
  diagnostics.

## [2026-06-01] maintenance | Strict single-candidate priority diagnostics

- Documented strict post-sort diagnostics:
  `weight_diagnostics.strict_candidate_priority_selection` now distinguishes
  matcher raw residual order from final engine candidate-priority order.
- Recorded the Hatban/HOT6 regression policy: regular rank/source inside strict
  tolerance beats a lower-rank regular or rescue candidate with only a 1-2g
  residual advantage.
- Added decision-engine coverage for Hatban `365g` at `delta=-368g` selecting
  Hatban over HOT6 `367g`, while preserving the tighter match when rank 1 is
  outside strict tolerance.

## [2026-06-01] maintenance | Segment grip cap

- Documented `MODEL__WEIGHT__MAX_ITEMS_PER_SEGMENT=3`: segment-first matching
  caps one detected loadcell segment to three product units, while preserving
  existing aggregate `SAME_PRODUCT_MAX_COUNT=8` behavior when segment targets
  are absent.
- Recorded segment-derived total caps for aggregate overrides and strict
  fallback paths: two segments allow up to six units, but x8 repeats are
  rejected with `count_exceeds_segment_grip_limit`.
- Added decision-engine coverage for one-segment Pepero x8 rejection, two
  segment Pepero x6 allowance, and aggregate override cap enforcement.

## [2026-06-01] maintenance | Compound segment split and low-weight diagnostics

- Documented segment compound options for merged loadcell movements: one
  segment can split into two or three `count=1` items when at least one item is
  final/trusted and companions have supported or weak companion evidence.
- Recorded explicit segment selection tiers and the small-item repeat guard:
  `unit_weight < 200g` with `count >= 3` is rejected behind valid compound,
  trusted/supported single, or tight large-item alternatives.
- Updated low-weight behavior: `LOW_WEIGHT_VISION_FALLBACK=true` now runs video
  diagnostics only, records `engine_skipped=true`, and stays excluded from
  DoorSession/CLOSE payment.

## [2026-06-01] maintenance | Clean segment match priority

- Documented segment override ordering: when loadcell segments are cleanly
  matched by evidence-supported products, segment selections beat aggregate
  repeated-count overrides with equal or higher residual.
- Added diagnostics for blocked aggregate overrides:
  `reason=clean_supported_segment_match_preferred`,
  `candidate_aggregate_residual`, and `selected_segment_all_supported`.
- Added regression coverage for Binch + Haruyache + Letsbe versus Chapagetti
  x4, while preserving collision-like aggregate repeated-count recovery.

## [2026-06-01] maintenance | Side ROI candidate promotion

- Relaxed regular side ROI from `center_x <= 250` to `center_x <= 400` in the
  left 480 crop so strong side detections for Bibigo and Pepero become normal
  candidates instead of only stage-count fallback evidence.
- Kept regular Top/Side thresholds at `0.25`, top ROI at `center_y >= 240`,
  and ROI rescue strict at the current side ROI boundary.
- Added regression coverage for side ROI boundary behavior, Bibigo/Pepero trace
  shapes, and Letsbe `0.2926` confidence under the preview-aligned threshold.

## [2026-06-01] maintenance | Pepsi/Trevi preview alignment

- Documented the service/preview alignment: Python service defaults now use
  `models/0204_morning.engine` and regular Top/Side thresholds `0.25`.
- Recorded ROI-strict threshold rescue diagnostics: conflicted weak rescues now
  expose `roi_conflict` and `threshold_rescue_rejected_reason`.
- Added regression coverage for weak Trevi threshold rescue rejection and
  regular Pepsi winning over Trevi rescue residual collisions.

## [2026-06-01] maintenance | Dummy artifact and env-interference audit

- Documented the audit result: runtime-interfering dummy artifacts were limited
  to an enabled sample-frame export default, an unreachable legacy block, and
  confusing YOLO warmup dummy naming.
- Recorded that `.env.example` now keeps frame sample export disabled by
  default, matching the Jetson templates and avoiding extra disk I/O when copied
  into a live `.env`.
- Kept test-only mocks/fixtures and the multi-zone dummy/test product-name
  rejection guard; they are not runtime dummy data and prevent invalid products
  from entering inference.

## [2026-06-01] maintenance | Segment weak-evidence repeat guard

- Documented segment-first matching support tiers: weak stage traces are not
  enough to make small-item repeated counts outrank active large-bottle
  explanations.
- Recorded `evidence_supported` in segment diagnostics and regression coverage
  for Pepsi x2 versus weak Pepero repeats and Sky Barley x2 versus weak Binch
  repeats.
- Aligned `.env.example` weight settings with the Jetson templates so replacing
  a live `.env` does not silently omit repeated-count and strict-search limits.

## [2026-06-01] maintenance | Forced fallback pair support ranking

- Documented forced final fallback pair ordering so supported same-product
  repeats beat unsupported mixed active-product pairs even when the mixed pair
  has a smaller residual.
- Recorded new diagnostics: `mode=detected_same_product_pair`,
  `pair_support_rank`, per-product `pair_support`, and `evidence_score`.
- Added decision-engine regression coverage for Pepsi x2 versus weak Trevi
  pair injection, plus supported mixed-pair preservation.

## [2026-06-01] maintenance | 480-crop ROI alignment

- Documented the new operating geometry: Top/Side inference uses the left
  480x480 crop from 640x480 camera frames so ROI coordinates match the
  TensorRT 480 input contract.
- Updated ROI defaults: side keeps `center_x <= 400`, top keeps
  `center_y >= 240` for both removal and return deltas, and side ROI rescue is
  strict at the ROI boundary.
- Updated configuration, video/vision, product-pipeline, observability, and
  test-map wiki pages for the new 480-crop-aligned filters.

## [2026-06-01] maintenance | Camera-aware stage evidence scoring

- Documented stage-count evidence scoring that caps raw vote influence with
  `log1p` and weights side confidence, ROI/threshold, and motion evidence above
  low-confidence top-only raw detections.
- Recorded segment-first selection ordering where evidence tier is preserved
  but residual beats weak evidence-score differences, allowing Welchs x2 to
  beat top-raw-heavy Cupban x2 in split loadcell traces.
- Added trace/test map notes for `stage_score`, camera confidence/vote fields,
  and stage-count expansion ordering before the 10-class limit.

## [2026-06-01] maintenance | Same-weight candidate and close delta guard

- Documented compatibility `/trigger` parity for `return_segment_targets`,
  mixed return hints, and `effective_count_guard` diagnostics.
- Recorded the same-weight candidate identity guard so regular Pepsi-style
  candidates beat same-weight active-only products and lower-rank rescue
  candidates within the scaled allowance.
- Documented basket-facing CLOSE delta behavior: still-unmatched return hints
  are excluded from effective delta, and empty baskets normalize tolerance-sized
  residuals to `0.0g`.

## [2026-06-01] maintenance | Mixed return segment replay

- Documented `return_segment_targets` for paired-out-free positive loadcell
  segments inside compound trigger histories.
- Recorded the internal `return_weight_hints` path: negative chargeable
  triggers keep judging the removal `decision_delta`, while DoorSession replay
  uses the positive hint to cancel an earlier removal in the basket.
- Updated trace and test maps for `weight_diagnostics.mixed_return_segments`,
  mixed return/removal aggregation, and CLOSE effective delta accounting.

## [2026-06-01] maintenance | Trevi return and King Rush segment recovery

- Documented evidence-aware segment selection where stage/diagnostic evidence
  can beat active-only exact residuals for separable removal segments.
- Recorded aggregate override tightening: low-count threshold/ROI rescue flags
  are not trusted aggregate evidence unless backed by final rank, weight gate,
  or strong stage/diagnostic evidence.
- Documented chronological door-session trigger replay and relaxed same-zone
  single return matching so a `524g` return can cancel a `530g` Trevi removal.

## [2026-06-01] maintenance | Candidate-first repeated count before stage expansion

- Documented the strict-miss recovery order where final candidates get a
  same-product repeated-count match before stage-count products can supplement
  the basket.
- Updated the same-product repeated-count tolerance formula to
  `count * same_product_count_tolerance + tolerance_grams`, matching the
  segment aggregate override allowance.
- Recorded diagnostics for `same_product_count_match.stage_count_preempted`
  and test coverage for the Welchs x2 regression plus the outside-tolerance
  stage-count fallback guard.

## [2026-06-01] maintenance | Candidate rank and repeated Pepsi recovery

- Documented strict single-item ordering that prefers regular final-candidate
  source/rank over lower-confidence rescue candidates when both are inside the
  fixed strict tolerance.
- Recorded regression coverage for Sky Barley rank-1 versus Trevi threshold
  rescue, and for stage-supported Pepsi repeated-count recovery over a mixed
  fragmented segment basket.

## [2026-06-01] maintenance | Trevi collision segment recovery

- Documented aggregate evidence override for collision-like loadcell segment
  fragmentation, where a strong repeated-product stage/diagnostic candidate can
  beat a low-confidence mixed segment basket.
- Recorded the new `aggregate_evidence_override` trace diagnostics for segment
  matching, including candidate residuals, evidence strength, and rejection
  reasons.
- Updated the decision test map for the Trevi x5 fragmentation regression and
  low-evidence guard coverage.

## [2026-06-01] maintenance | Hot6 segment candidate priority

- Documented the segment-weight override where trusted final candidate or
  trusted stage evidence can beat evidence-free active-only products when the
  aggregate segment total supports a repeated same-product count.
- Recorded the diagnostic fields `candidate_supported_override` and
  `evidence_priority_selection` for explaining cases where segment residual
  alone would otherwise select the wrong active product.
- Updated the decision test map for the Hot6-style regression in
  `test_decision_engine.py`.

## [2026-06-01] maintenance | Segment-first loadcell matching

- Documented `removal_segment_targets` and
  `vision_required_segment_targets` as trace metadata derived from stable
  loadcell movement history.
- Recorded the segment-first decision branch that matches separated removal
  segments before aggregate strict matching, while keeping forced final
  fallback as the last no-none guard.
- Updated test maps for loadcell segment target extraction and decision-engine
  segment-first regressions.

## [2026-06-01] maintenance | Stable loadcell history and no-none fallback

- Documented stable plateau history as the basis for `decision_delta`, including
  paired remove-return/press-release suppression and ignored micro movements.
- Recorded that trace metadata now preserves both `net_delta_weight` and
  `decision_delta_weight` plus purchase-delta candidates and pairing details.
- Documented the forced final fallback that returns a low-confidence `PARTIAL`
  active-product guess instead of `none` for chargeable negative deltas when
  active products exist.
- Updated test maps for loadcell history regressions and forced fallback
  decision-engine coverage.

## [2026-06-01] maintenance | Rapid same-zone failure remediation

- Recorded that Top/Side FFmpeg gamma and contrast defaults are now all `1.0`
  because field comparison favored the unadjusted image path.
- Documented compound loadcell segment detection for merged removal/return
  payloads and the `3.0s` recent same-zone trigger context exposed in traces.
- Updated decision/weight notes for motion-aware strict combination ranking and
  returned-weight hints that down-rank likely just-returned products during
  negative follow-up triggers.
- Added regression coverage for compound loadcell segments, recent same-zone
  trace metadata, motion-supported ambiguous strict matches, and returned
  weight hint ranking.

## [2026-05-29] maintenance | Candidate-inclusive stage fallback and weight repair

- Changed stage-count strict fallback ordering so any valid combination that
  includes a final candidate outranks all-stage-count combinations; all-stage
  combos remain recovery when candidate-inclusive combos are outside tolerance.
- Applied the same stage-count strict fallback before relaxed combination
  recovery, keeping final candidates as primary evidence and `stage_counts` as
  supplements up to 10 total classes.
- Added product-weight alias parsing and repair diagnostics for active product
  snapshots, including a narrow class `44`
  `BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML` fallback to `520.0g`.

## [2026-05-29] maintenance | Return loadcell stabilization

- Added return-only stabilization before committing positive loadcell deltas:
  the service waits `MODEL__TRIGGER__RETURN_STABILIZATION_WAIT_SECONDS=1.0`
  and requires stable start/end regions by default.
- Short first/last fallback return payloads now produce
  `status=waiting`, `waiting_for=stable_loadcell` instead of silently adding a
  partial return weight to DoorSession.
- Added trace diagnostics under `weight_diagnostics.return_stabilization` and
  regression tests for stable return commit and partial return waiting.

## [2026-05-29] maintenance | Candidate-only combination priority

- Added a candidate-only strict combination pass inside relaxed combination
  matching before any `stage_counts_by_class` expansion.
- Reordered stage-count fallback selections so final-candidate-only
  combinations beat combinations that include `source=stage_counts`.
- Added regressions for the field case where three final candidates match the
  loadcell delta but two non-candidate stage-count products also match.

## [2026-05-29] maintenance | Confidence and side ROI rescue tightening

- Raised regular Top/Side candidate confidence thresholds to `0.30` after gamma
  tuning made high-confidence detections more reliable.
- Increased vision ranking weights to `top=0.60`, `side=0.65`,
  `top_only=0.55`, `side_only=0.60`, and common-class bonus `0.20`.
- Tightened side ROI from `center_x <= 440` to `center_x <= 400`.
- Limited `source=roi_rescue` to moving side detections near the ROI boundary:
  `side_motion_passed=true` and `roi_x_avg <= roi_x_limit + 20px`.
- Added regression coverage for static/far-right ROI-filtered detections so
  they stay out of final candidates and weight-gated rescue votes.

## [2026-05-29] maintenance | Vision-source candidate priority

- Changed final candidate merge ordering so regular `source=vision` candidates
  rank ahead of `roi_rescue` and `threshold_rescue` candidates before `top_k`
  trimming.
- Applied the shared ordering to both `TriggerService` and the direct API
  trigger path through `VideoProcessor.merge_rescue_votes()`.
- Added regression coverage proving a lower-confidence vision candidate remains
  first even when rescue candidates have higher weight-gated confidence.

## [2026-05-29] maintenance | Stage-count combination fallback

- Added a fallback strict-combination pass that merges final candidates with
  `stage_counts_by_class` evidence up to 10 total candidate classes after the
  final-candidate strict pass misses.
- Kept final candidates first, then fills remaining slots from stage-count
  insertion order so lower-ranked stage evidence can still be considered.
- The fallback accepts only combinations with total count >= 2; one-item
  recovery remains handled by detected-single fallback.
- Added regression coverage for a one-final-candidate miss where the real
  A+B products are present only in the second and eighth stage-count entries.
- Verified locally with focused decision/strict/scenario tests (`60 passed`),
  full model tests (`218 passed`), and `uv tool run ruff check services/model
  scripts`.

## [2026-05-29] maintenance | Compact vision-first combination selection

- Changed strict combination ranking to prefer the smallest total item count
  first, then lower weight residual, higher average vision confidence, and
  fewer product kinds. This keeps A x1 + B x1 ahead of C x4 when both are
  plausible.
- Moved same-product repeated-count matching after strict/compact combination
  matching. Relaxed matching now tries count=1 singles, then combinations, then
  repeated single-product estimates.
- Raised general strict weight tolerance to
  `MODEL__WEIGHT__TOLERANCE_GRAMS=5.0` and added
  `MODEL__WEIGHT__MULTI_KIND_MIN_CONFIDENCE=0.18` for multi-kind items.
- Updated the stride-2 env templates and code maps for the 5g tolerance,
  0.18 confidence floor, 10px motion floor, and same-product x8 settings.
- Verified locally with focused decision/strict/scenario tests
  (`59 passed`), full model tests (`217 passed`), `uv tool run ruff check
  services/model scripts`, and scenario readiness
  (`verified=924`, `failures=0`, `elapsed_ms=34.7`, `stride2_traces=4`).

## [2026-05-29] maintenance | Motion fail-closed and same-product x8

- Lowered the model-service motion filter floor from 15px to 10px and aligned
  bbox dynamic threshold floors with the same setting.
- Removed regular candidate motion fail-open behavior so static products that
  all fail motion remain filtered instead of being restored as candidates.
- Kept hand-path and ROI filters unchanged, while retaining low-confidence
  moving threshold rescue for weight-gated recovery.
- Added same-product repeated-count controls:
  `MODEL__WEIGHT__SAME_PRODUCT_COUNT_TOLERANCE_GRAMS=5.0` and
  `MODEL__WEIGHT__SAME_PRODUCT_MAX_COUNT=8`, leaving general strict matching at
  `MODEL__WEIGHT__TOLERANCE_GRAMS=3.0` and five total units.
- Verified locally with `pytest services/model/tests -q` (`212 passed`),
  `uv tool run ruff check services/model scripts`, and scenario readiness
  verification (`924` cases, `0` failures, `60.75 ms` engine time, 4 stride-2
  traces under the 20s video budget).

## [2026-05-29] maintenance | Scenario readiness and explicit 0g branch

- Added an Excel refresh script, committed scenario JSON fixture, and scenario
  readiness report for 924 expanded cases and 104 checklist rows.
- Added a scenario verification script/report showing 924/924 model-contract
  rows passing, 0 failures, 60.75 ms engine decision time, and 4 stride-2 trace
  files under the 20s video budget.
- Documented strict matching coverage for five total units, three distinct
  product kinds, A+B+C, A2+B2, and A5 scenario expectations.
- Recorded that stride-2 scenario tests enforce the 20-second latency evidence
  contract shape, while Jetson TensorRT traces remain required for production
  timing proof.
- Updated loadcell docs for explicit low-weight payload diagnostics, including
  all-zero and filtered-all-zero/raw-nonzero reasons.
- Recorded the code-derived sibling-service diagnosis that the payment path may
  unlock before Camera recording starts; CRK-model now reports missing/zero
  loadcell evidence but does not modify sibling services.

## [2026-05-28] maintenance | Loadcell 0g diagnostics

- Documented diagnostics-only handling for `delta_weight=0.0g` cases without
  changing low-weight skip behavior.
- Recorded new loadcell payload evidence in traces and OPS logs:
  `payload_state`, raw/filtered channel counts, invalid/zero/nonzero counts,
  and first/last raw or filtered zone totals.
- Clarified that CRK-model receives Camera-packaged `/trigger.loadcells`, so
  `empty_payload`, `invalid_only`, or `all_zero` should be traced at the
  Camera/IO Board boundary before treating the event as a vision miss.

## [2026-05-27] maintenance | Loadcell-first trigger cancellation

- Documented the trigger relevance scheduler that skips YOLO for positive
  return deltas and cancels queued removal triggers when a later return
  balances them before video starts.
- Recorded the new CLOSE distinction between all pending triggers and
  `pendingChargeableVisionCount`, so non-chargeable diagnostics do not block
  finalization.
- Added trace interpretation notes for `return_loadcell_only`,
  `balanced_out`, and `cancelled_by_return` paths.
- Added regression coverage for single and combination balanced cancellation,
  return-only DoorSession updates, and non-chargeable pending close behavior.

## [2026-05-27] maintenance | Cross-zone combination repair and active snapshot fallback

- Documented cross-zone repair as local aggregation, global combination repair,
  then effective net-delta validation.
- Recorded support for same-product multi-count, multi-product, and
  multi-source-zone return repair without changing the public response shape.
- Documented the last-valid `ActiveProductStore` fallback for
  `missing_active_product_snapshot_fail_closed` traces after repeated
  judgment/recovery cycles.
- Added regression coverage for combination returns, repeated recovery before
  the next trigger, effective CLOSE summary output, and active snapshot fallback
  TTL behavior.

## [2026-05-27] maintenance | Cross-zone return effective delta

- Documented that cross-zone return repair now runs before effective net-delta
  validation across active door sessions.
- Recorded the effective delta rule used by CLOSE summaries:
  raw trigger delta minus outgoing cross-zone return weight plus incoming
  cross-zone return weight.
- Added regression coverage for returning an item to a different zone after
  removing another item in the source zone, plus CLOSE summary output.

## [2026-05-27] maintenance | Safe release cleanup and static checks

- Recorded the release-cleanup pass that kept public API and runtime behavior
  unchanged while removing shadowed `TriggerService` loadcell helper code.
- Documented that loadcell route/service compatibility helpers now delegate to
  `core/loadcell_stats.py`, preserving existing test imports.
- Updated code maps for the Ruff-clean import/static-string cleanup and the
  async frame queue annotation fix.
- Left raw operational trace JSON files untouched.

## [2026-05-25] maintenance | Top camera direction-aware ROI

- Documented the top-camera ROI rule: removal keeps detections with
  `center_y >= 200`, return keeps detections with `center_y <= 200`, and
  zero/unknown delta skips top ROI.
- Updated video/vision, configuration, product-pipeline, observability, and
  tests maps for the new top ROI config, trace fields, and regression tests.
- Left raw operational trace JSON files untouched.

## [2026-05-21] ingest | Existing docs and recent latency work

- Created the initial repo-local wiki under `docs/llm-wiki/`.
- Ingested all 17 existing files under `docs/` as source-summary pages.
- Added synthesis pages for system boundaries, runtime flow, product detection,
  protocols, Jetson/testing, historical risks/fixes, and latency/frame stride.
- Left raw docs and untracked trace JSON files untouched.

## [2026-05-21] ingest | Codebase-wide module map

- Added `source-code/` as the code/config/test/script layer.
- Added module-level maps for startup/DI, API routes, configuration, loadcell
  and trigger flow, video/vision, decision/weight, sessions, observability,
  tests, scripts, and repo overview.
- Updated navigation and freshness rules to reflect README-level facts:
  this Python repo is the legacy/reference TensorRT path, fresh clone-based
  operation points toward `CRK-model-go`, and current loadcell channel behavior
  is summed zone total.
- Continued to exclude untracked trace JSON files from durable wiki ingestion.

## [2026-05-21] maintenance | Active product allowlist guard

- Documented the no-detection failure mode where a zero-stock/zero-weight
  active-product snapshot creates an empty strict YOLO allowlist.
- Updated latency/stride guidance to check `active_product_diagnostics` before
  treating zero raw detections as a frame-stride recall loss.
- Updated session and observability maps for the multi-zone snapshot guard,
  store preservation behavior, and empty-allowlist fail-closed trace field.

## [2026-05-21] maintenance | Main branch delivery rule

- Clarified the repo operating rule that final delivery should use `main` and,
  when requested, commit and push changes to `origin/main`.

## [2026-05-21] maintenance | Jetson stride-2 env template

- Added `docs/jetson-stride2.env.txt` as a Jetson copy/paste `.env` template
  with `MODEL__ASYNC_STREAMING__FRAME_STRIDE=2`.
- Documented at the time that the stride-2 template was for field testing and
  did not replace the then-current `frame_stride=1` recall-safe default policy.

## [2026-05-21] maintenance | CLOSE waits for trigger finalization

- Documented that CLOSE final success must wait until the trigger worker has
  finished video processing, session storage, door-session aggregation, OPS
  logging, and trace finalization.
- Recorded the field trace conclusion: the referenced `1335xx`/`1340xx` traces
  completed frame processing; `none` came from
  `missing_active_product_snapshot_fail_closed`, not from CLOSE cutting frames
  short.
- Documented the new close diagnostics for pending trigger session ids and the
  `missing_active_products` final summary reason.

## [2026-05-21] maintenance | Low-weight ignore and same-product counts

- Documented that `abs(delta_weight) <= 5g` is hard-ignored for production
  judgment and excluded from DoorSession close aggregation.
- Recorded the field diagnosis that noisy low-confidence multi-kind weight
  combinations can over-explain a delta and should be rejected unless candidate
  quality is strong enough.
- Added the same-product repeated-count rule: active, detected products may use
  `count * MODEL__WEIGHT__TOLERANCE_GRAMS` residual tolerance for `x2` cases
  such as Pepero, while respecting stock and count limits.

## [2026-06-30] freezer | Upper ROI and hand proximity filtering

- Changed freezer `dual_top_proxy` documentation to the upper-half ROI default:
  `MODEL__VISION__FREEZER_ROI_VERTICAL_REGION=upper` and
  `MODEL__VISION__FREEZER_ROI_Y_SPLIT=240.0`.
- Documented the stage semantic split: `freezer_roi_passed` increments
  `freezerExitPathVotes`, while `freezer_roi_filtered` is rejected ROI evidence
  with `freezerRoiFilteredVotes` diagnostics only.
- Recorded the new upper-ROI hand proximity evidence:
  `handPathValidUpperRoi`, `handInteractionPassed`, `handNearFrameCount`,
  `handNearVoteRatio`, and `minHandDistancePx`.
- Updated the freezer handled-filter and direct `freezer_vision_first` notes so
  top-only/static candidates without hand/trajectory support are demoted or
  rejected only when a supported alternative remains.
