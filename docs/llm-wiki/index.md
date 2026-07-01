# CRK-model LLM Wiki

Last updated: 2026-07-01

This wiki is the LLM-facing map of the CRK model service. The original
documents under `docs/` remain the raw sources. Pages here summarize and
synthesize those sources so another Codex or LLM can recover the current system
shape without rereading every long historical document.

## Start Here

- [Wiki schema](wiki-schema.md): how to add future source summaries, synthesis
  pages, and work-log entries.
- [Repo overview](source-code/repo-overview.md): runtime role, README-level
  status, package shape, dataset, and env groups.
- [File inventory](source-code/file-inventory.md): every runtime Python module,
  test file, script, and root operational source.
- [Runtime flow](synthesis/runtime-flow.md): the main `/trigger` to final
  judgment path.
- [System map](synthesis/system-map.md): how Edge_Environment, CRK-CAMERA,
  CRK-IO-BOARD, CRK-PAYMENT, and CRK-model relate.
- [Jetson and testing](synthesis/jetson-and-testing.md): runtime verification
  rules and safe local checks.
- [Latency and frame stride](synthesis/latency-and-frame-stride.md): current
  latency work, telemetry, and `frame_stride` tradeoffs.
- [Scenario readiness and 0g diagnostics](synthesis/scenario-readiness-and-0g.md):
  Excel scenario fixture coverage, strict combination limits, stride-2 latency
  contract, and payment-path 0g diagnostics.

## Current Operating Rules

- Runtime target is the Jetson Orin Nano service. Local PC startup, health
  checks, or TensorRT behavior are not valid production-runtime proof.
- This Python repo is currently documented by README as the legacy/reference
  TensorRT `.engine` path; fresh clone-based operation should prefer
  `CRK-model-go` unless the task explicitly targets this Python service.
- Preferred local safety gate is `pytest services/model/tests -q`; the
  2026-06-15 local run recorded `366 passed` alongside a passing Ruff check
  with `uv run --no-sync ruff check services/model scripts`.
- Frame trace image export is opt-in. `.env.example` keeps
  `MODEL__TRACE__SAMPLE_EXPORT_ENABLED=false` so replacing a live `.env` does
  not add sample-frame disk writes on Jetson.
- `.env.example` is now a sanitized freezer-first dual-top template. It sets
  `MODEL__MACHINE__CABINET_TYPE=freezer`,
  `MODEL__VISION__CAMERA_LAYOUT=dual_top_proxy`, and
  `MODEL__VISION__YOLO_MODEL_PATH=models/set7_v8best.engine` while leaving
  credentials and secrets as placeholders.
- Fatal async video processing failures are not valid no-detection outcomes.
  `process_videos_async()` must propagate extractor, queue, and YOLO task
  errors as model-service exceptions so `TriggerService` records
  `status=error` instead of silently returning empty candidates.
- `/api/health/detailed` includes best-effort diagnostics for NumPy, Torch CUDA
  visibility, and TensorRT import availability. These are operator clues, not
  local-PC production proof.
- The model service receives Camera-packaged `/trigger` payloads and Node
  `/api/judge/multi-zone` calls. It does not call IO Board or Payment directly.
- `active_products` from Node is the runtime product-catalog source by
  default. Active-product class identity is resolved from Edge
  `product_eng_name` matched against class names loaded from the current YOLO
  engine. During the Edge migration, engine-matching `name` and legacy
  `product_name` are compatibility class keys after `product_eng_name`.
- `trainingidx`/`training_idx`/`trainingIdx`/`yolo_class_id` and
  `yolo_class_name` may still appear in payloads for API compatibility, but
  they are ignored for active-product class identity because Edge ids can drift
  from the deployed engine ids. Korean `product_name` is display metadata in
  the official contract; it is used as a legacy class key only when it already
  matches the loaded engine class name.
- Edge `product_idx` is not a stable model class key. The model preserves it
  for Node-facing product identity, but resolves YOLO classes from the
  engine-matching class-name fields above.
- Static `dataset.yaml` and `services/config/yolo_product_mapping.json`
  validation is advisory and opt-in through
  `MODEL__CATALOG__STATIC_VALIDATION_ENABLED=true`; default startup records
  the loaded engine class summary without warning on stale static mappings.
  The stale mapping file is not used for runtime active-product allowlists.
- `MODEL__VISION__LOG_ENGINE_CLASSES=on` prints the loaded engine classes
  through OPS logs. The log key `name=` is the engine class label, not an Edge
  payload field name.
- Stock-positive active products with valid class ids remain in the YOLO
  allowlist even when Node sends missing or `0g` product weight. Such rows are
  tracked as `weight_unavailable` for loadcell validation/count diagnostics
  rather than blocking vision detection.
- Product identity is vision-first by default. With
  `MODEL__WEIGHT__IDENTITY_POLICY=vision_first`, final vision candidates,
  strong stage evidence, or weight-gated rescue evidence must support the
  product identity. Loadcell is count/validation evidence, and no-vision
  loadcell-only deltas return no-charge `no_detection` or `uncertain`.
- `MODEL__MACHINE__CABINET_TYPE=refrigerated` keeps the existing
  loadcell/vision/weight path. `freezer` uses the same zone-sliced
  `/trigger.loadcells` payload but switches identity to a freezer
  vision-first policy where loadcell weight is a tie-break/diagnostic signal,
  not a hard product reject gate.
- Strong vision identity with unavailable product weight is preserved as
  `partial` with `vision_identity_preserved_weight_unavailable`; loadcell count
  correction only runs when the active product has a valid positive weight.
- Confidence fusion is vision-heavy by default:
  `MODEL__WEIGHT__FUSION_VISION_WEIGHT=0.65`,
  `MODEL__WEIGHT__FUSION_LOADCELL_WEIGHT=0.25`, and
  `MODEL__WEIGHT__FUSION_COUNT_WEIGHT=0.10`.
- `delta_weight < 0` means removal. `delta_weight > 0` means return.
- Top/Side YOLO preprocessing uses the left 480x480 crop from 640x480 camera
  frames by default, matching the TensorRT 480 input contract.
- `MODEL__VISION__CAMERA_LAYOUT` defaults to `legacy_top_side`. In
  `dual_top_proxy`, Camera still sends `videos.top/side`, but `videos.top` is
  recorded as physical `top_middle` and `videos.side` is recorded as
  `top_side` using the logical Top processing profile.
- Python service code defaults still point at `models/0204_morning.engine`, but
  the current copyable `.env.example` is freezer-field oriented and overrides
  the engine path to `models/set7_v8best.engine`. Current freezer product vote
  floors are `MODEL__VISION__TOP_CONFIDENCE_THRESHOLD=0.70` and
  `MODEL__VISION__SIDE_CONFIDENCE_THRESHOLD=0.70`; hand tracking uses the
  separate `MODEL__VISION__HAND_CONFIDENCE_THRESHOLD=0.40` with hand class id
  `0`.
- Top-camera ROI now uses the lower region for both removals and returns:
  non-zero deltas keep `center_y >= 240`, while zero or missing delta skips
  the top ROI.
- Freezer `dual_top_proxy` treats both public streams as top cameras and keeps
  only detections inside the configured freezer vertical ROI. The field
  template uses the upper half
  (`MODEL__VISION__FREEZER_ROI_VERTICAL_REGION=upper`,
  `center_y <= MODEL__VISION__FREEZER_ROI_Y_SPLIT`, default `240`). It applies
  stronger motion/vote floors and a `0.70` product confidence floor, while
  hand detections are filtered independently at `0.40`. Threshold/ROI/stage/
  diagnostic fallback evidence below the product floor cannot create freezer
  product identity. `freezer_roi_passed` increments exit-path votes; rejected
  `freezer_roi_filtered` evidence remains diagnostic only.
- Freezer handled candidates are narrower than raw vision top-K. Raw top-K
  candidates stay in trace diagnostics, while OPS candidates, engine input, and
  DoorSession close snapshots use handled candidates. For a single freezer
  removal segment, the handled list defaults to one product; weight residual
  breaks ties inside the top confidence band. Same-product `x2+` candidates can
  now be inferred from target weight even when `instance_count_hint=1`, but only
  for final vision candidates with freezer exit-path votes, enough vote
  evidence, confidence above the freezer multi floor, stock/count caps, and a
  residual inside the freezer/count-scaled tolerance.
- Freezer handled selection also uses interaction evidence, not only exit-path
  counts. Trace diagnostics record path displacement, max movement, center
  span, trajectory pass, static shelf likelihood, upper-ROI hand validity,
  hand proximity counts/ratios, and hand-path pass/block state. Top-only
  static shelf candidates are softly demoted, and valid hand-path blocks can
  hard-reject a candidate only when at least one hand-near alternative remains;
  no-near/all-blocked hand-path cases still fail open.
- Freezer same-tier single selection now puts weight residual ahead of raw
  exit-path vote volume. Exit-path votes still gate candidates and break
  residual ties, but a high-vote top-only candidate no longer beats a
  better-weighted supported single. Compound/multi-segment loadcell traces no
  longer automatically pass raw candidates through; they first try viable
  multi-fit, single, and same-product repeat explanations, then fail open only
  when unresolved.
- Freezer stage-only rescue is candidate-miss recovery, not an override for
  already handled candidates. If the handled-filter accepted one supported
  product, direct `freezer_vision_first` rejects stage-only resurrection of
  considered-but-unselected classes; non-stage strict weight-gate candidates
  also keep ambiguous dual-camera stage-only evidence out of the special
  priority tier.
- Freezer loadcell is now treated as reliable to about `15g` by
  `MODEL__WEIGHT__FREEZER_WEIGHT_TOLERANCE_GRAMS=15.0`. Strong dual-camera
  freezer evidence can keep multiple handled identities only when their
  combined/segment weight fits that freezer tolerance; ordinary nonzero freezer
  deltas no longer return top-1/top-2/top-3 candidates together when their sum
  greatly exceeds the measured removal.
- Side-camera ROI keeps hard `center_x <= 400` in the left 480 crop and adds a
  conditional `+5px` regular-candidate soft band. This catches Pepsi detections
  around `x=402..404` while leaving farther-right Trevi-style detections out.
- Current README/tests describe physical zone loadcell channels as summed into
  the zone total. Older docs that mention averaging are historical context.
- The current Jetson template uses Top FFmpeg gamma/contrast `1.2/1.2` and
  Side `1.0/1.0`; traces record these values so field runs can be compared
  against candidate recall and YOLO latency.
- Rapid same-zone judgment now keeps two extra pieces of model-side context:
  compound loadcell segments inside one trigger payload, and same-zone
  loadcell events from the last `3.0s`.
- Simultaneous same-zone removals can use physical loadcell channel deltas as
  evidence-required segment targets. A channel-supported split such as
  Tteokbokki + Welchs is judged before a same-weight aggregate rescue product.
- Chargeable negative deltas use matched-only finalization. After regular
  matching fails, forced fallback can still return a low-confidence `PARTIAL`
  from active product weights, but only when the product weight sum explains
  the full stable removal delta inside the existing branch tolerance. A
  sub-segment-only fallback returns no-charge `UNCERTAIN` with
  `final_weight_mismatch_guard` diagnostics.
- Collision-specific identity guards are preferred over broad tolerance
  changes. Regular rank-1 final vision combinations and strong regular 500ml
  two-bottle repeats can use count-scaled grace only when replacing
  same-weight lower-rank/rescue/active-only collisions. Single 500ml bottles
  stay on flat strict tolerance before strict identity can override tighter
  weight evidence.
- Detected-single fallback has one narrow single-bottle exception: when final
  candidates are empty and a rejected 450-560g bottle has strong side/motion
  stage or diagnostic evidence, it can replace a weak residual-only single
  fallback inside `detected_single_fallback_tolerance +
  same_product_count_tolerance`. Keep this as diagnostics-driven fallback
  identity evidence, not as OPS candidate filtering.
- Strong 500ml bottle `x2` removal evidence with a materially undercounted
  negative loadcell delta returns `waiting_for=stable_loadcell` and is excluded
  from DoorSession/payment until Camera/Node posts a new stable loadcell tail.
- Chargeable loadcell deltas now require stable plateau evidence. The service
  uses the first confirmed stable plateau as baseline and the last confirmed
  stable tail plateau as the final value; first/last samples and raw max/min
  swings are diagnostics only. Negative removals with no confirmed stable tail
  return `waiting_for=stable_loadcell` before video, engine, DoorSession, or
  payment aggregation.
- Negative removals that wait for a stable loadcell tail now keep any active
  product fail-closed reason in the trigger session, trace final result, and
  multi-zone waiting response. A combined `removal_waiting_for_stable_loadcell`
  plus `missing_active_products` result means both Node inventory context and
  Camera/IO loadcell timing need inspection.
- Latest sibling-repo risk notes are documentation-only in this Python repo:
  Edge/Camera/IO URL hardcoding, Camera missing retry for
  `waiting_for=stable_loadcell`, and stale IO protocol docs remain external
  follow-ups rather than CRK-model code changes.
- Return handling is hybrid: a positive delta deducts one same-zone product
  immediately only inside flat strict tolerance. Multi-count/combo/cross-zone
  returns and mixed `return_weight_hints` are deferred until CLOSE and then
  replayed against the final net delta.
- Strict single-item matching records both matcher raw order and engine
  post-sort order. Once candidates are inside flat strict tolerance, single
  products sort by candidate rank before source priority, residual, or evidence;
  this includes `threshold_rescue` and `stage_weight_gate` candidates.
- Stage-count evidence that passed the weight gate can be promoted to
  `source=stage_weight_gate` when votes/confidence are sufficient. It recovers
  products missing from final candidates, but it does not let a lower-rank
  stage candidate override a higher-rank strict single candidate.
- When final candidates are empty, stage-count/rescue evidence can still build
  strict multi-item combinations before loadcell-only or active-only forced
  fallback. Candidate/stage-supported repeats and pairs therefore outrank
  unseen active-only pair injection when they explain the stable target inside
  existing tolerances.
- Segment matching retries ordinary time-based `removal_segment_targets` when
  evidence-required physical channel targets cannot explain every channel. This
  lets sequential removals such as Haluyache + LetsBe + Jagabee recover before
  aggregate strict or active-only forced fallback.
- Segment-first matching treats low-confidence stage traces as unsupported
  evidence, so weak Pepero/Binch-style small-item repeats do not outrank
  active large-bottle explanations or become `COMPLETE`.
- Segment-first matching can split a merged loadcell segment into a trusted
  compound option such as Fanta + Pepsi. Small-item repeats remain recovery,
  but are rejected behind valid compound/single large-item alternatives.
- Segment targets enforce the configured grip-size cap
  `MODEL__WEIGHT__MAX_ITEMS_PER_SEGMENT`. The template default is three
  products per detected loadcell segment; setting the env value to `4` allows
  four total products in that segment, including mixed products.
- Clean evidence-supported loadcell segment matches are kept ahead of aggregate
  repeated-count overrides when their total residual is no worse, so separate
  product removals do not collapse into a same-product repeat.
- Stage/diagnostic-only small fragments are not clean support when a high-rank
  regular candidate explains the aggregate segment total as a same-product
  repeat inside count-scaled tolerance.
- Do not lower regular candidate thresholds to solve small-product filler
  mistakes. Keep low-confidence evidence in rescue/diagnostic paths, and block
  unsupported `unit_weight < 200g`, `count >= 2` fragments at decision time
  unless they are regular final candidates or strong motion-backed stage
  evidence.
- Same-weight 500ml bottle segment collisions prefer repeated two-camera or
  very strong stage evidence over reusing a single top-only candidate across
  several separated segment targets.
- CLOSE finalization performs a final basket-weight validation. Over-fragmented
  mixed baskets can be corrected to a repeated ranked candidate when the final
  effective delta supports that repeat inside count-scaled tolerance, while
  unresolved return/cross-zone sessions and clean supported mixed baskets are preserved.
  Unsupported small-fragment baskets get a narrow wider residual-gap allowance
  so strong Sky Barley `x3` evidence can replace a lower-residual fragment mix.
- CLOSE finalization does not swap an all-regular vision-supported current
  basket to a different product identity only because the alternative has a
  lower loadcell residual. It can still correct the count of the same
  vision-supported product.
- CLOSE repeat correction has a final count cap based on stock, same-product
  max count, per-item max count, and
  `removal_trigger_count * MODEL__WEIGHT__MAX_ITEMS_PER_SEGMENT`; cap failures
  record `count_exceeds_close_repeat_cap`.
- Low/zero-weight tail triggers can run diagnostic video processing, but the
  trace records `engine_skipped=true` and the trigger stays excluded from
  CLOSE/payment. Active global sessions now retain no-charge diagnostics for
  these skipped events, surfaced at close through
  `decisionSummary.diagnosticZoneLines` and
  `decisionSummary.zones[*].noChargeDiagnostics`.
- Completed door-session YAML retention is controlled by
  `MODEL__DOOR_SESSION__YAML_RETENTION_DAYS`; deployed Jetsons should also use
  host log rotation for `services/model/logs/frame_split_*.jsonl`.
- `scripts/convert_engine.sh` is the repo-local TensorRT `.engine` export
  helper. It defaults to this repo's `models/`, checks for `yolo`, and fails if
  the active Torch build cannot see CUDA. ONNX export ownership remains in
  `CRK-model-go`.
- No-vision active-only forced fallback is not part of the default
  `vision_first` identity policy. The older low-weight noise guard still
  applies when `MODEL__WEIGHT__IDENTITY_POLICY=weight_aware` is explicitly
  selected for legacy fallback behavior.
- Future delivery for this repo should happen on `main` unless the user
  explicitly asks for a different branch strategy. When the user asks to
  finalize changes, commit and push them to `origin/main`.

## Synthesis Pages

| Page | Purpose |
| --- | --- |
| [system-map](synthesis/system-map.md) | Service boundaries and cross-repo dependencies. |
| [runtime-flow](synthesis/runtime-flow.md) | End-to-end trigger, queue, video, decision, session, and close flow. |
| [product-detection-pipeline](synthesis/product-detection-pipeline.md) | Seven-stage detection pipeline and judgment logic. |
| [protocol-contracts](synthesis/protocol-contracts.md) | Camera, Node, IO Board, and Payment contracts from the model perspective. |
| [jetson-and-testing](synthesis/jetson-and-testing.md) | Jetson setup, verification boundaries, and test commands. |
| [historical-risk-and-fixes](synthesis/historical-risk-and-fixes.md) | Historical risk review, fixes, roadmap, and recovery notes. |
| [latency-and-frame-stride](synthesis/latency-and-frame-stride.md) | Latency bottlenecks, telemetry, recent commits, and stride policy. |
| [scenario-readiness-and-0g](synthesis/scenario-readiness-and-0g.md) | Scenario matrix fixture, model contract coverage, and 0g diagnostic branch. |

## Code Maps

- [repo-overview](source-code/repo-overview.md): README, package, dataset, and
  `.env.example` operational summary.
- [file-inventory](source-code/file-inventory.md): complete source/test/script
  inventory.
- [startup-and-di](source-code/startup-and-di.md): `main.py`, FastAPI lifespan,
  and `ServiceContainer` graph.
- [api-routes](source-code/api-routes.md): HTTP endpoints and compatibility
  rules.
- [configuration](source-code/configuration.md): Pydantic settings and env var
  groups.
- [loadcell-and-trigger](source-code/loadcell-and-trigger.md): loadcell delta,
  low-weight handling, dedup, and trigger worker.
- [video-and-vision](source-code/video-and-vision.md): frame extraction, YOLO,
  filters, voting, and rescue candidates.
- [decision-and-weight](source-code/decision-and-weight.md): decision engine,
  strict matcher, relaxed matching, and fallbacks.
- [session-and-persistence](source-code/session-and-persistence.md): trigger
  sessions, door/global sessions, active products, YAML persistence.
- [observability-and-traces](source-code/observability-and-traces.md): OPS
  logs, latency logs, trace schema, and debug heuristics.
- [tests-map](source-code/tests-map.md): test files by subsystem and when to
  run them.
- [scripts-and-jetson-tools](source-code/scripts-and-jetson-tools.md): Jetson
  setup, runtime env, torch install, live preview, and engine conversion.

## Source Document Summaries

### Agent Guides

- [architecture](source-docs/agent-guides-architecture.md): current model
  service architecture and inference branches.
- [build-test](source-docs/agent-guides-build-test.md): preferred tests and
  focused regression suites.
- [conventions](source-docs/agent-guides-conventions.md): runtime defaults,
  config rules, and project layout.
- [jetson-setup notes](source-docs/agent-guides-jetson-setup.md): concise
  Jetson setup and recovery notes.

### Core Docs

- [API reference](source-docs/reference.md): model HTTP endpoints and response
  shapes.
- [Jetson setup](source-docs/jetson-setup.md): full Jetson Orin Nano setup and
  troubleshooting guide.
- [Product detection flow](source-docs/product-detection-flow.md): seven-stage
  pipeline overview.
- [Product detection detail](source-docs/product-detection-detail.md): YOLO,
  filters, voting, decision, and session internals.
- [Trigger capture debugging](source-docs/trigger-capture-debugging.md): field
  timing contract across Edge, Camera, and Model.
- [Trigger inference recovery notes](source-docs/trigger-inference-recovery-notes-2026-03-31.md):
  current trigger hardening and return recovery behavior.

### External Review Inputs

- [CRK feedback 2026-06-29](source-docs/crk-feedback-2026-06-29.md):
  external review note covering CRK-model operational hardening and async video
  failure propagation.
- [Freezer weight feedback 2026-06-29](source-docs/freezer-weight-feedback-2026-06-29.md):
  operator feedback that freezer loadcell error is usually about `5g` and up to
  `10g-15g`, with a `178g` single-removal top-three candidate over-selection
  failure.

### Historical Planning And Risk Docs

- [Implementation roadmap](source-docs/implementation-roadmap.md): scenario
  phases and completion history.
- [Model process risk review](source-docs/model-process-risk-review-2026-03-04.md):
  historical risk review, some text encoding issues in the source file.
- [Fixes applied](source-docs/fixes-applied-2026-03-05.md): fixes following
  the March 2026 risk review.

### Protocol Docs

- [Camera protocol](source-docs/protocols-camera.md): Node-to-Camera and
  Camera-to-Model contracts.
- [IO Board protocol](source-docs/protocols-io-board.md): loadcell and door
  data path as seen by the model.
- [Node protocol](source-docs/protocols-node.md): health and multi-zone judge
  contracts.
- [Payment protocol](source-docs/protocols-payment.md): indirect payment
  relation through Node.

## Maintenance Log

See [log](log.md) for append-only wiki maintenance history.
