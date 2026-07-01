# Source Code Map: Decision And Weight

Source: [engine/decision_engine.py](../../../services/model/model_service/engine/decision_engine.py),
[engine/models.py](../../../services/model/model_service/engine/models.py),
[service/trigger_service.py](../../../services/model/model_service/service/trigger_service.py),
[weight/count_calculator.py](../../../services/model/model_service/weight/count_calculator.py),
[weight/strict_weight_matcher.py](../../../services/model/model_service/weight/strict_weight_matcher.py),
[session/product_aggregator.py](../../../services/model/model_service/session/product_aggregator.py)

Status: current decision/weight map

## Current Thesis

Final judgment is vision-first for product identity. Final vision candidates,
strong stage evidence, or weight-gated rescue evidence can create product
identity; loadcell and `active_products` validate weight/count and price/stock.
`MODEL__WEIGHT__IDENTITY_POLICY=weight_aware` preserves the older
loadcell/active fallback behavior only when explicitly selected.

## Core Models

- `JudgmentStatus`: `complete`, `partial`, `uncertain`, `no_detection`.
- `EnsembleResult`: normalized vision candidate from video/voting.
- `CountEstimate`: weight-based count estimate.
- `ProductJudgment`: product response item.
- `JudgmentResult`: final decision with products, total price, confidence,
  status, delta, explained weight, residual, and Node response conversion.

## ProductDecisionEngine Branches

The engine handles:

- `vision_only`
- no-final-candidate stage-count combination recovery before loadcell-only
  fallback
- loadcell-only fallback when no candidate or stage/rescue combination exists
  and `MODEL__WEIGHT__IDENTITY_POLICY=weight_aware`
- segment-first loadcell matching when trace metadata exposes separable
  removal segments
- min-weight no-detection
- strict matching
- same-product repeated-count matching after final-candidate strict matching
  misses, before stage-count expansion
- stage-count combination fallback after candidate repeated-count matching misses
- relaxed matching:
  single-count product, combination, repeated product, partial result,
  loadcell-only no estimates
- detected-single-item fallback after strict/relaxed misses
- forced final fallback that returns a low-confidence `PARTIAL` active-product
  guess instead of `none` for chargeable negative deltas only under the
  explicit `weight_aware` identity policy

Important inputs:

- `vision_candidates`
- `delta_weight`
- `vision_only`
- `active_products`
- `trace_context`

## Vision-First Identity Policy

- The default `MODEL__WEIGHT__IDENTITY_POLICY=vision_first` blocks product
  identity from loadcell-only or active-only candidates. If no usable vision,
  stage, or weight-gated rescue evidence exists, negative loadcell deltas return
  no-charge `NO_DETECTION` or `UNCERTAIN` with empty products.
- If strong vision identity exists but loadcell validation conflicts, the
  engine preserves the vision product as `PARTIAL` and records
  `weight_diagnostics.vision_first_identity_validation` with target weight,
  expected weight, residual, tolerance, and whether count validation passed.
- If strong vision identity exists but the active product weight is unavailable
  or `0g`, the engine still preserves the vision product as `PARTIAL` with
  reason `vision_identity_preserved_weight_unavailable`. Count correction and
  loadcell validation are skipped for that product because they require a
  positive Node/current/last-valid weight.
- Loadcell-derived counts are accepted only when stock, max count, same-product
  cap, segment grip cap, and tolerance all agree. If count validation fails,
  the result keeps the vision identity at count `1` as `PARTIAL` instead of
  inventing a loadcell-sized repeated count.
- Non-success vision-first results cannot carry product identity. The guard
  records `weight_diagnostics.vision_first_identity_suppressed` and returns an
  empty-product no-charge result when low-confidence relaxed paths would
  otherwise leak an unsupported product.
- Fused confidence is configurable and vision-heavy by default:
  `vision=0.65`, `loadcell=0.25`, `count=0.10`.

## Freezer Vision-First Branch

- `MODEL__MACHINE__CABINET_TYPE=freezer` activates a freezer-specific branch
  for negative deltas after `vision_only` handling and before the normal
  refrigerated strict/relaxed matching path.
- Freezer results can only use products present in final vision candidates and
  in the active-product snapshot. Loadcell-only, active-only, or weight-nearest
  products that are absent from candidates remain no-charge misses.
- The normal `MODEL__WEIGHT__TOLERANCE_GRAMS` value is not a hard product
  reject gate in freezer mode. Single-item freezer selection first assigns a
  weight/exit-path tier, then sorts same-tier options by interaction penalty,
  source support, weight residual, dual-camera exit-path support,
  `freezerExitPathVotes`, confidence, and rank. The confidence-band residual
  fallback is still used when no strict/near weight-gate option exists. A
  single freezer result is weight-reliable when residual is within
  `MODEL__WEIGHT__FREEZER_WEIGHT_TOLERANCE_GRAMS=15.0`.
- Freezer diagnostics record `decision_branch=freezer_vision_first`,
  `weight_used_as=tiebreaker` or `diagnostic`, `weight_reliable`,
  `weight_residual`, `selectionTier`, `freezerExitPathVotes`, and the full
  considered/selected candidate list. `freezerExitPathVotes` comes from
  `freezer_roi_passed` or explicit legacy vote fields, not ROI-rejected
  detections. A large residual becomes `partial`, not an automatic identity
  rejection.
- Single freezer removal segments now use a handled-candidate narrowing step:
  raw top-K vision candidates stay in trace diagnostics, but OPS candidates,
  engine input, and DoorSession snapshots receive the one handled product
  selected by freezer exit-path evidence and weight residual before falling
  back to top confidence-band weight residual. Multi-segment trace evidence is
  diagnostic support, not a sufficient reason to keep raw freezer candidates:
  viable multi-kind weight fits are preserved, otherwise strict/near single or
  same-product repeat narrowing can still run.
- Stage-only freezer rescue is limited to candidate-miss recovery. When the
  video handled-filter has already accepted one supported freezer candidate,
  direct `freezer_vision_first` will not recreate considered-but-unselected
  classes as `freezer_stage_exit_path`. If a non-stage candidate has a strict
  freezer weight-gate fit, ambiguous dual-camera stage-only evidence is
  demoted into normal single ranking instead of taking the special priority
  tier.
- Freezer product identity creation also respects the freezer product
  confidence floor. Stage-count, diagnostic, threshold-rescue, ROI-rescue, and
  weight-gated rescue evidence below the current `0.70` product threshold can
  remain visible in diagnostics, but it cannot create a final product fallback.
- Multi-kind freezer results require segment/compound or combined-candidate
  weight support inside `MODEL__WEIGHT__FREEZER_WEIGHT_TOLERANCE_GRAMS`.
  Strong dual-camera exit-path evidence alone no longer selects multiple
  freezer candidates for ordinary nonzero deltas; rejected sets record
  `freezer_multi_kind_weight_mismatch` and fall back to the single handled
  freezer selection path.
- Same-product freezer repeats can be inferred from weight even when
  `instance_count_hint=1`. The candidate must still be a final vision
  candidate with confidence above the freezer multi floor, positive stock, and
  a count bounded by stock, `max_items_per_segment`,
  `same_product_max_count`, and `max_count_per_item`. Multi-candidate repeat
  competition still requires freezer exit-path votes and enough vote evidence.
  If the only candidate identity is a regular `vision` product, the count may
  instead promote from weight alone when the repeated residual is inside
  `MODEL__WEIGHT__FREEZER_WEIGHT_TOLERANCE_GRAMS` and is closer than `x1`;
  this covers side-only bagel cases such as `156g x2` against a `309.5g`
  removal. Trace diagnostics expose `sameProductRepeatCandidates`,
  `rejectedSameProductRepeatCandidates`, `singleRegularVisionIdentity`,
  `repeatEvidenceMode`, `count`, `expectedWeight`, and
  `countWeightResidual`. The trigger conversion layer preserves frame-level
  votes as `EnsembleResult.raw_vote_count`; `EnsembleResult.vote_count` remains
  the Top/Side consensus scale, so repeat gates must read raw frame evidence
  from `raw_vote_count` whenever they use the multi-candidate evidence path.
- Direct `freezer_vision_first` selection reads the same interaction evidence
  as the video handled-filter path. `staticShelfLikely` top-only candidates are
  softly demoted unless trajectory or hand-path support exists. The hand
  fields include `handPathValidUpperRoi`, `handInteractionPassed`,
  `handNearFrameCount`, `handNearVoteRatio`, `minHandDistancePx`,
  `handPathPassed`, and `handPathBlocked`. A `handPathBlocked` candidate is
  hard-rejected only when another supported option remains. This keeps direct
  engine calls and video-trigger filtering aligned without adding
  product-name-specific rules.

## StrictWeightMatcher

- Converts active products and vision candidates into `CandidateProduct`
  entries.
- Excludes products without weight and sold-out products.
- This exclusion applies to loadcell matching only. Zero-weight stock-positive
  products can still be detected by vision because `ActiveProductStore`
  separates the YOLO allowlist from loadcell-valid weight rows.
- Searches combinations near `abs(delta_weight)` within gram tolerance.
- Uses separate bounds for total units and distinct product kinds. Current
  defaults cover the scenario matrix: up to five units and up to three kinds.
- Sorts valid combinations compact-first by default: lowest total unit count,
  lowest weight residual, strongest evidence, then lowest kind count.
- Candidate evidence now carries motion and source flags from `EnsembleResult`
  into `CandidateProduct`. When a static one-item match competes with a
  motion-supported repeated or multi-kind match at the same weight, the
  motion-supported explanation can outrank the simpler static item.
- Records `last_diagnostics` for trace/debugging.
- Records `last_return_diagnostics` for return-combination matching. Single
  returns keep strict flat tolerance, while multi-product return combinations
  use `MODEL__WEIGHT__TOLERANCE_GRAMS +
  item_count * MODEL__WEIGHT__SAME_PRODUCT_COUNT_TOLERANCE_GRAMS`.
- Allows high-confidence three-kind combinations such as A+B+C, but still
  rejects multi-kind ingredients below
  `MODEL__WEIGHT__MULTI_KIND_MIN_CONFIDENCE=0.18` and combinations above the
  configured kind limit.
- `ProductDecisionEngine` applies an additional candidate-priority ordering on
  matcher results: any valid combination that includes at least one final
  `vision_candidates` item beats all-stage-count combinations. Within that
  bucket it sorts multi-item combinations by lower total count, final-candidate
  source/rank, lower weight residual, fewer stage-only units, higher average
  confidence, then fewer kinds.
- Chargeable negative decisions have a final full-delta guard. A successful
  `COMPLETE` or `PARTIAL` removal can reach the API only when the final product
  weight sum explains `abs(delta_weight)` inside the existing strict,
  detected-single, rescue, or count-scaled tolerance for that branch. If a
  fallback explains only a sub-segment, the engine returns `UNCERTAIN` with no
  products and records `weight_diagnostics.final_weight_mismatch_guard`.
- For single-item strict candidates, rank is the strongest identity signal once
  every option is inside strict tolerance: candidate rank sorts before source,
  then residual and evidence. This applies to `source=vision`,
  `source=threshold_rescue`, and `source=stage_weight_gate`. It keeps a higher
  ranked Sky Barley `523g` candidate at `delta=-525g` from losing to a lower
  ranked Trevi `530g` stage-weight candidate only because stage evidence has a
  higher source priority.
- Strict traces record the matcher raw order and the engine post-sort order in
  `weight_diagnostics.strict_candidate_priority_selection`. When the raw
  matcher top differs from the final selected single candidate, the reason can
  be `regular_single_candidate_priority`,
  `stage_weight_gate_candidate_priority`, or `ranked_single_candidate_priority`
  for ranked rescue/stage-aware single-item ordering.
- Before strict and segment matching, non-freezer/legacy stage-count detections
  can be promoted to synthetic `source=stage_weight_gate` candidates when they
  passed the weight gate, have an in-stock positive-weight active product, meet
  `MODEL__WEIGHT__DETECTED_SINGLE_FALLBACK_MIN_VOTES`, and reach confidence
  `>=0.08`. In current freezer mode, this same recovery path is additionally
  gated by the product confidence floor, so sub-`0.70` stage evidence cannot
  create freezer product identity. If the same class already exists in
  `vision_candidates` as a non-regular rescue/stage candidate, weight-gated
  stage evidence can upgrade that entry to `source=stage_weight_gate`; regular
  `source=vision` final candidates are never overwritten by stage evidence.
  Diagnostics are recorded in `weight_diagnostics.stage_weight_gate_candidates`.
  Stage evidence is a recovery path when final candidates cannot explain the
  strict single match; it does not let a lower-rank stage-weight candidate
  override a higher-rank final/rescue candidate that is already inside strict
  tolerance.
- Before aggregate strict matching, the engine checks
  `loadcell.removal_segment_targets`. When two or more unpaired removal
  segments exist, each segment is matched independently against all in-stock
  loadcell-enabled active products, then the selected segment explanations are
  aggregated into the final product judgment. This prevents a split same-item
  removal such as `210g + 105g + 103g + 107g` from collapsing into a simpler
  aggregate one-item match near `530g`.
- If `loadcell.channel_removal_segment_targets` contains two or more physical
  channel targets, segment matching evaluates them before ordinary time-based
  `removal_segment_targets`. Channel targets are evidence-required and only
  allow one single-product option per channel; all selected channel products
  must be evidence-supported. A supported channel split such as Tteokbokki
  `144g` plus Welchs `371g` can therefore beat a same-weight aggregate
  threshold rescue such as Kwangdong `520g`. If a channel split cannot explain
  every channel target, the engine records
  `channel_segment_weight_matching` diagnostics and retries ordinary
  time-based `removal_segment_targets` before aggregate strict or fallback
  branches.
- Segment-first matching respects active stock across segments, allows repeated
  same-product counts per segment, and records
  `weight_diagnostics.segment_weight_matching` with segment targets, options,
  selections, residuals, and final status. Without product evidence it returns
  low-confidence
  `PARTIAL`; when selected products have vision/stage/diagnostic evidence it
  can return `COMPLETE`.
- `MODEL__WEIGHT__MAX_ITEMS_PER_SEGMENT` is applied only when
  `loadcell.removal_segment_targets` exist. The repository template default is
  `3`; setting the env value to `4` allows up to four total product units in
  one segment, including mixed products. Two segments with a value of `4` can
  therefore produce eight total units. Without segment targets, aggregate
  same-product logic keeps the existing
  `MODEL__WEIGHT__SAME_PRODUCT_MAX_COUNT=8` behavior.
- Same-weight collision guards preserve regular `source=vision` candidates over
  candidate-outside stage/threshold products in the same bottle-weight band.
  For Pepsi/Corn-style `520g` collisions, a regular Pepsi candidate inside the
  guard tolerance is selected before same-weight Corn Silk Tea stage-gate
  evidence that was not promoted through the regular candidate path.
- Segment evidence now distinguishes supported evidence from weak traces.
  Trusted/strong/weight-gated evidence, or stage evidence with at least
  `MODEL__WEIGHT__DETECTED_SINGLE_FALLBACK_MIN_VOTES` and confidence `>=0.18`,
  is supported. Weak low-confidence traces are ranked like active-only products,
  so small products such as Pepero/Binch repeated many times do not beat an
  active large-bottle x2 explanation just because their residual is smaller.
- Single-product segment recovery can also treat weak stage/rescue evidence as
  supported when its confidence is `>=0.18` and the unit weight matches that
  specific loadcell segment inside strict tolerance. This is segment-local only:
  it does not lower global YOLO thresholds or allow repeated small fragments to
  fill aggregate baskets.
- Aggregate combination matching rejects unsupported small repeat fragments:
  any item with `unit_weight < 200g` and `count >= 2` cannot be used as strict
  or relaxed filler unless it is a regular final `source=vision` candidate or
  has strong stage evidence with motion, votes, and confidence. This blocks
  Condition Stick-style fragments from completing baskets such as
  `HALUYACHE x2 + Condition x2` when the small product was absent from final
  candidates. The rejected reason is recorded as
  `unsupported_small_repeat_fragment` in strict matcher diagnostics. Segment
  matching keeps exact loadcell segment recovery available, but unsupported
  small repeats are ranked lower and exposed in segment diagnostics so stronger
  repeated candidates can override them.
- Segment options can now be `single` or `compound`. A compound segment splits
  one merged loadcell movement into two or three `count=1` products when at
  least one item is a final/trusted vision product and companions have
  supported stage evidence or weak companion evidence
  (`unit_weight >= 200g`, votes `>=20`, confidence `>=0.08`). Compound
  allowance is `MODEL__WEIGHT__TOLERANCE_GRAMS +
  item_count * MODEL__WEIGHT__SAME_PRODUCT_COUNT_TOLERANCE_GRAMS`.
  Compound options are not generated above the per-segment grip cap.
- Segment ranking uses explicit tiers: trusted compound split, trusted single,
  supported single/repeat, weight-tight single, small-item repeated count, then
  active-only repeat. `unit_weight < 200g` with `count >= 3` records
  `rejected_reason=trusted_or_single_item_segment_preferred` when a valid
  single or compound alternative exists.
- Same-product segment repeats that exceed
  `MODEL__WEIGHT__MAX_ITEMS_PER_SEGMENT` are rejected with
  `rejected_reason=count_exceeds_segment_grip_limit`. This blocks one-segment
  explanations such as `Pepero x8` or `Chapagetti x8` while preserving
  per-segment `x3` recovery.
- Evidence-backed single-item segment options may use
  `MODEL__WEIGHT__TOLERANCE_GRAMS +
  MODEL__WEIGHT__SAME_PRODUCT_COUNT_TOLERANCE_GRAMS` residual allowance, while
  active-only single items keep strict tolerance. When stage/diagnostic
  evidence appears in segment options, selection minimizes active-only units,
  then stays within evidence tier and compares residual before evidence score.
  This lets a `523g + 371.8g` sequence select Trevi plus King Rush over exact
  active-only Sky Barley/Welchs fits, while a `370.7g + 374.1g` sequence can
  select Welchs x2 over a top-raw-heavy Cupban stage trace.
- Stage-count evidence uses a camera-aware score instead of raw count alone.
  Side confidence, side ROI/threshold/motion evidence, and capped `log1p`
  vote scores are weighted above low-confidence top-only raw counts. Segment
  diagnostics expose `stage_score`, `side_confidence`, `top_confidence`,
  `side_votes`, `top_votes`, and `score_reason` for each evidence-backed
  option.
- Segment-first also has a candidate-supported repeated-count override. If the
  best segment-by-segment explanation uses only evidence-free active products,
  a trusted final candidate or trusted stage-count product can win by explaining
  the aggregate segment total as the same product repeated. Its allowance is
  `count * MODEL__WEIGHT__SAME_PRODUCT_COUNT_TOLERANCE_GRAMS +
  MODEL__WEIGHT__TOLERANCE_GRAMS`, so a detected `367g x3` product can beat an
  unseen `371g x3` active-only fit when the aggregate residual stays inside
  that bound.
- For collision-like over-fragmented segments, the same aggregate override can
  run even when the selected segment mix contains some evidence. It is eligible
  when the segment selection is multi-kind, partial, includes active-only
  products, or is a stage/diagnostic-supported repeated product over three or
  more segments. A clean all-evidence segment selection is not replaced when
  its total segment residual is less than or equal to the aggregate repeated
  candidate residual; diagnostics record
  `reason=clean_supported_segment_match_preferred`. Stage/diagnostic-only small
  fragments (`unit_weight < 200g`) are not treated as clean support when a
  high-rank regular repeated candidate explains the aggregate total inside
  count-scaled tolerance. This lets a rank-1 Sky Barley `x2` candidate beat a
  Condition/Hot6/Binch fragment basket that only wins on residual.
- Segment aggregate overrides, same-product count match, same-weight candidate
  collision guard, and strict matcher calls all observe the segment-derived
  total cap `len(removal_segment_targets) * MAX_ITEMS_PER_SEGMENT`. For example,
  two segments reject an aggregate `x8` repeat, while seven fragmented segments
  can still recover a supported `x5` collision candidate.
- Channel split matching does not run aggregate segment overrides. If every
  channel target is supported, diagnostics use
  `reason=channel_supported_split_preferred` and may record
  `rejected_aggregate_rescue`; if any channel target is active-only or
  unsupported, the branch records the rejection and falls back to aggregate
  strict matching.
- Same-weight 500ml bottle collisions get a segment repeat pass when separated
  segment targets are all in the bottle-weight band. A single product can cover
  every segment when each segment has a valid one-count option and the class has
  two-camera stage evidence, or very strong single-camera evidence. This lets a
  low-confidence-but-repeated Pepsi trace beat one top-only Trevi candidate
  being reused across several segments. The selected repeat is recorded in
  `segment_weight_matching.same_weight_bottle_collision`; rejected top-only or
  weak repeated reuse is recorded in
  `segment_weight_matching.repeated_segment_reuse_guard` with
  `reason=repeated_segment_evidence_insufficient`.
- Stage `threshold_rescue_candidate` and `roi_rescue_candidate` flags are not
  trusted aggregate evidence by themselves. Aggregate override requires final
  rank, weight gate, or strong stage/diagnostic evidence. Strong stage evidence
  currently means count `>=20`, confidence `>=0.30`, and motion not rejected;
  strong diagnostic evidence means votes `>=5` and confidence `>=0.30`.
- Repaired active weights are valid weight operands, but repair by itself is
  not identity evidence. A product whose `product_weight=0` was repaired from a
  snapshot still needs final, stage, or diagnostic evidence before it receives
  an evidence priority bonus over a same-weight detected product.
- `loadcell.vision_required_segment_targets` are only eligible when product
  evidence exists. This keeps press/release-like positive-then-negative loadcell
  pairs out of active-product-only fallback.
- For negative follow-up triggers, `ProductDecisionEngine` also reads returned
  weight hints from trace loadcell metadata: positive `compound_segments` and
  recent same-zone positive events. A combination whose unit weight matches a
  likely just-returned item is down-ranked, not hard-filtered.
- Before aggregate strict matching, a narrow same-weight candidate identity
  guard protects regular final candidates from losing to candidate-outside
  active products with the same unit weight or to lower-rank rescue candidates
  that only have a smaller strict residual. The guard is limited to actual
  same-weight/rescue collisions, so normal multi-kind strict combinations and
  the existing same-product repeated-count branch keep their priority.
- The same-weight identity guard treats single bottles and two-bottle repeats
  differently. For `count == 1`, a regular final vision candidate can win only
  inside the flat strict tolerance (`MODEL__WEIGHT__TOLERANCE_GRAMS`, currently
  `5g`). A single Trevi `530g` candidate at target `521g` therefore cannot
  override a Corn/Pepsi `520g` weight-gated rescue with `1g` residual. For
  `count == 2` 450-560g bottle repeats, the guard can use the controlled
  count-scaled repeat allowance described below.
- Strict combination selection has a second narrow candidate-priority grace:
  when a regular `source=vision` rank-1 final candidate can replace exactly a
  same-weight lower-rank or rescue collision inside count-scaled tolerance, the
  rank-1 combination may beat the strict residual winner. This is for cases
  like Sky Barley + Fanta at `-1151g`, where Corn Silk + Fanta is inside the
  flat `5g` strict window but replaces a same-weight higher-rank Sky candidate
  that is only one per-item tolerance outside strict. Diagnostics are recorded
  in `weight_diagnostics.candidate_priority_combination_grace`.
- For 450-560g two-bottle repeats, the same-weight candidate collision guard
  can add one extra per-item allowance only for strong regular vision
  same-product repeats. This lets regular Pepsi x2 beat Trevi/Corn rescue or
  active-only collisions when the repeat residual is controlled, without
  changing general strict matching.
- The explicit Pepsi/Trevi single-bottle regression now goes the other way:
  regular single-bottle identity does not get count-scaled grace. If the
  regular single candidate is outside flat strict tolerance and a weight-gated
  rescue is inside strict tolerance, the strict/rescue winner remains selected.
- Detected-single fallback has a separate 500ml identity override for the
  no-final-candidate boundary. When a strong side/motion stage or diagnostic
  bottle evidence is rejected only because the residual is slightly outside the
  detected-single `8g` window, it may use one `same_product_count_tolerance`
  grace unit and replace a weak weight-only single fallback such as Trevi. This
  does not change OPS final-candidate semantics and does not apply when the
  current residual winner also has strong bottle identity evidence.
- The guard records `weight_diagnostics.same_weight_candidate_collision` with
  the selected regular candidate, rejected same-weight active products, any
  rejected strict rescue candidate, residuals, and the scaled allowance.
- `TriggerService` has a pre-engine removal stabilization conflict gate. When a
  negative delta looks like a 500ml bottle `x2` undercount, the rank-1 final
  `source=vision` candidate is strong enough, and the measured weight is short
  by more than the ordinary x2 allowance but within 20% of one unit, the service
  stores `SessionData(status="waiting",
  processing_stage="removal_waiting_for_stable_loadcell")`, records
  `weight_diagnostics.removal_stabilization`, skips `engine.judge()`, and does
  not add anything to DoorSession/payment aggregation.
- The same stabilization branch now covers the broader accuracy-first loadcell
  policy. If a removal has only simple first/last fallback data or the final
  stable plateau does not reach the payload tail, the trigger waits for
  `stable_loadcell` before video and before the engine can fall back to
  active-only products.
- If all ordinary branches fail and active products exist, the forced final
  fallback runs last. It first compares every in-stock loadcell-enabled active
  product as `x1`, then tries pair options from detected evidence plus active
  products. Supported same-product repeat pairs use
  `mode=detected_same_product_pair` and, for regular vision-backed 500ml
  bottle repeats, the same count-scaled repeat allowance as the collision
  guard. This prevents an active-only Trevi from being injected into a
  Pepsi-only `-1058g` fallback just because Pepsi + Trevi has a smaller flat
  residual. A mixed `detected_plus_active_pair` can still win when both
  products have support or when the detected repeat is outside tolerance. If
  nothing is inside tolerance, the fallback still returns the nearest evaluated
  option as `PARTIAL` with explicit diagnostics.
- Active-only forced fallback has a low-weight noise guard. When there are no
  vision/stage/purchase-delta candidates and the negative delta is still below
  the lightest active loadcell product minus strict tolerance, capped by the
  small-delta fallback window, the engine records
  `reason=active_only_low_weight_noise` and returns the loadcell-only miss
  instead of charging a nearby tiny product such as Condition Stick for `6-10g`
  shelf shake.
- Trigger-level judgment is not the last identity check. Door-session CLOSE
  finalization now performs a basket-level final-weight validation using the
  persisted ranked candidate snapshots from each trigger. This close-only pass
  is allowed to replace an over-fragmented mixed basket with a same-product
  repeat when that ranked candidate explains the final effective negative delta
  inside `MODEL__WEIGHT__TOLERANCE_GRAMS + count *
  MODEL__WEIGHT__SAME_PRODUCT_COUNT_TOLERANCE_GRAMS`.
- The CLOSE correction is intentionally narrower than decision-engine strict
  matching: it requires repeated or high-rank regular vision evidence, allows a
  small residual disadvantage within one base tolerance for ordinary
  over-fragmented baskets, and within `base + same_product_count_tolerance` only
  when the current basket contains unsupported small fragments. It preserves
  clean evidence-supported mixed baskets and skips sessions with returns or
  cross-zone repair.
- Under the default vision-first policy, CLOSE does not swap an all-regular
  vision-supported current basket to a different product identity simply because
  that replacement has a lower loadcell residual. The close-only repeat
  correction may still adjust the count when the selected replacement is the
  same product id already present in the current basket.

## Stage-Count Combination Fallback

- When strict matching cannot find a combination from final candidates, the
  decision engine builds a second candidate set from final candidates plus
  `trace_context.stage_counts_by_class`.
- When final candidates are empty, the engine still runs this strict
  stage-count combination pass before `judge_by_weight_only()` and forced final
  fallback. This lets ranked stage/rescue evidence form same-product repeats or
  multi-item pairs before any unseen active-only product is injected.
- Before building that expanded set, strict judgment gives regular final
  candidates one same-product repeated-count chance against the aggregate
  delta. This candidate-first repeat path prevents a detected product such as
  `Welchs x2` from being replaced by candidate-outside stage-count filler
  products when the repeated count stays inside the scaled residual allowance.
- Final candidates keep priority, and stage-count entries fill the remaining
  slots up to 10 total candidate classes by camera-aware stage score rather
  than insertion order. This keeps a side/confidence-backed product that missed
  final voting from being dropped behind many top-only low-confidence raw
  detections.
- If final candidates alone can explain the weight as a multi-item strict
  combination, relaxed matching tries that candidate-only strict combination
  before stage-count expansion. If final candidates need one or more
  stage-count supplements, that candidate-inclusive combination still outranks
  an all-stage-count combination while it remains inside strict tolerance.
- The fallback accepts only total-count >= 2 combinations, so existing
  detected-single fallback behavior remains responsible for one-item recovery.
- It uses the same active-product filters as strict matching. `stage_counts`
  are a supplement to the final candidates, not a replacement: all-stage
  combinations are kept as recovery only when no valid candidate-inclusive
  combination exists.
- Successful stage-count combination recovery records
  `weight_diagnostics.decision_branch=stage_count_combination_match`, making it
  easy to distinguish evidence-supported recovery from active-only forced
  fallback in trace review.

## Same Product Count

- After final-candidate strict matching misses, and again after strict/compact
  combination recovery misses, the decision engine checks whether a detected
  active product can explain the loadcell delta as the same product repeated.
  The first check runs before stage-count expansion and records
  `stage_count_preempted=true` when it accepts.
- For repeated same-product candidates, residual tolerance scales by count:
  `allowed_residual = count * MODEL__WEIGHT__SAME_PRODUCT_COUNT_TOLERANCE_GRAMS
  + MODEL__WEIGHT__TOLERANCE_GRAMS`.
- The repeated-count path is evidence gated and bounded by stock plus
  `MODEL__WEIGHT__SAME_PRODUCT_MAX_COUNT`. It covers same-product counts through
  x8 while rejecting high repeated-count matches that would mask another active
  product already near the target weight as a single item.
- Counts above x2 are also rejected when another high-confidence regular vision
  candidate competes with the selected class, so A+B+C scenarios stay on strict
  multi-kind matching instead of being collapsed into A x7.
- The relaxed count-calculator fallback uses the same x8 cap so a repeated item
  rejected by the dedicated path is not restored later as x9 or higher.
- General strict combination matching remains capped by
  `MODEL__WEIGHT__MAX_COMBINATION_ITEMS=5` and
  `MODEL__WEIGHT__TOLERANCE_GRAMS=5.0`.

## Recovery And Rescue

- Threshold/ROI/no-motion rescue candidates can enter strict matching only when
  weight gates are satisfied.
- Detected-single fallback uses class evidence from stage counts and diagnostic
  detections plus nearest active-product weight tolerance.
- Its `single_bottle_identity_override` diagnostic records whether a rejected
  strong 450-560g bottle evidence source, such as Pepsi side evidence, replaced
  a weak residual-only single fallback. Candidate logs remain final-candidate
  logs; rejected rescue evidence is exposed through trace diagnostics instead.
- Loadcell-only nearest-single fallback rejects matches outside tolerance and
  can reject ambiguous nearest candidates.
- The forced final fallback intentionally supersedes that fail-closed behavior
  only at the final engine boundary, so direct helper tests can still exercise
  the stricter loadcell-only primitive. Pair fallback diagnostics include
  `pair_support_rank`, per-product `pair_support`, and `evidence_score` so
  traces can show whether a mixed active product had enough evidence.
- Forced final fallback is now full-delta-only. It may use
  `net_stable_delta`, `unpaired_negative_total`, or the ordinary
  `decision_delta_weight`, but an individual
  `last_unpaired_negative_segment`/`unpaired_negative_segment` target is
  rejected unless it also equals the full stable removal delta. Rejected partial
  targets are recorded in
  `weight_diagnostics.forced_fallback_rejected_partial_target`.

## Return Aggregation

- `ProductAggregator` immediately deducts only one same-zone product when a
  positive return delta matches that unit inside flat strict tolerance. It does
  not estimate multi-count returns or run combination matching during the
  intermediate replay.
- Same-product `x2+`, multi-product return combinations, cross-zone candidates,
  and `return_weight_hints` are stored as `DoorSession.deferred_returns`.
  CLOSE reconciliation then compares the final effective delta with the current
  basket weight and applies a same-zone subset or cross-zone match only when it
  improves the final residual.
- Deferred reconciliation diagnostics are exposed as
  `finalWeightValidation.deferredReturnReconciliation`. Records that still
  cannot be applied after CLOSE are moved to `unmatched_returns` and excluded
  from basket-facing effective delta.
- CLOSE repeat correction runs after deferred return reconciliation and applies
  a final count cap:
  `min(stock_qty, same_product_max_count, max_count_per_item,
  removal_trigger_count * max_items_per_segment)`. Rejected candidates record
  `count_exceeds_close_repeat_cap`, preventing weight-only repeat corrections
  such as HomeRunBall `64g x33`.
- All-vision-supported current baskets are identity-stable at CLOSE. A lower
  residual repeated candidate with a different product id is recorded as
  `clean_supported_basket_preferred` with `identitySwapBlocked=true`; a
  same-product repeat can still update the supported product count.
- After CLOSE correction, DoorSession applies a matched-only guard to the
  effective negative net delta. If the aggregated basket still cannot explain
  the final removal weight inside close tolerance,
  `finalWeightValidation.reason` becomes `unresolved_final_weight_mismatch`.
  Non-freezer cabinets clear `aggregated_products`; freezer cabinets preserve
  the detected `products`/`totalPrice` for Edge output and record
  `finalWeightValidation.outputPolicy=products_as_detected` plus
  `unresolvedProducts` diagnostics.

## Related Wiki Pages

- [Product detection pipeline](../synthesis/product-detection-pipeline.md)
- [Runtime flow](../synthesis/runtime-flow.md)
- [Observability and traces](observability-and-traces.md)
- [Scenario readiness and 0g diagnostics](../synthesis/scenario-readiness-and-0g.md)
