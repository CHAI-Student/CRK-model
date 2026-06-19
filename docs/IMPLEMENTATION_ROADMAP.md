# Scenario Implementation Roadmap

> **Created**: 2026-03-04 | **Version**: 1.0
> **Reference Document**: `docs/MODEL_PROCESS_RISK_REVIEW_2026-03-04.md`
> **2026-03-09 note**: This roadmap is historical. For current Jetson run/test commands, prefer `model-service`, `pytest`, or `uv run --no-sync ...`.

A step-by-step roadmap for implementing the full extraction/return scenarios of the AI smart vending machine.
Each Phase can be implemented independently. Copy the **Implementation Prompt** at the bottom of each section into a new session to use.

---

## Full Scenario Coverage

### Single Hand - Single Product

| Scenario | Phase | Current Status |
|---------|-------|----------|
| Basic extraction (left/right hand) | Phase 4 | Working (verification needed) |
| Multi-item extraction (sequential) | Phase 4 | Working (verification needed) |
| Same-zone return | Phase 4 | Working (verification needed) |
| Cross-zone return | Phase 4 | Working (verification needed) |
| A extraction → return → B extraction | Phase 4 | Partially working |
| Fast extraction (caught on camera) | Phase 2 | Working (threshold check needed) |
| Ultra-fast extraction (camera miss) | Phase 2 | Loadcell fallback partially working |

### Single Hand - Multi Product

| Scenario | Phase | Current Status |
|---------|-------|----------|
| Multiple same products simultaneous extraction | Phase 0b | ✅ Implemented |
| Multiple different products simultaneous extraction | Phase 0b | ✅ Implemented |
| Multi-item extraction then **full simultaneous return** | **Phase 1** | ✅ **Implemented** |

### Load Cell Interference

| Scenario | Phase | Current Status |
|---------|-------|----------|
| Pushing product sideways then extracting | Phase 3 | ✅ Peak filtering implemented |
| Temporarily placing on another product | Phase 3 | ✅ Peak filtering implemented |

### Vision Interference

| Scenario | Phase | Current Status |
|---------|-------|----------|
| Extraction visible only from top | Phase 2 | ✅ top_only_weight=0.6 applied |
| Extraction visible only from side | Phase 2 | ✅ side_only_weight=0.5 applied |

### Two-Handed Extraction

| Scenario | Phase | Current Status |
|---------|-------|----------|
| 1 product, 1 zone, two hands | Phase 5 | Unverified |
| 2 products, 1 zone, two hands | Phase 5 | Unverified |
| 2 products, 2 zones, two hands | Phase 5 | Basic support (verification needed) |

---

## Implementation Dependencies

```
Phase 0 (Data Quality Foundation)
    └─> Phase 1 (Simultaneous Return Core)
            └─> Phase 2 (Camera Interference)  ─┐
                Phase 3 (Loadcell Interference) ─┼─> Phase 4 (Integration Verification)
                                                 │       └─> Phase 5 (Two-Handed)
                                                 ┘
```

---

## Phase 0: Data Quality Foundation (Medium Priority Fixes)

**Purpose**: Fix two Medium-priority items that form the accuracy foundation for all subsequent Phases

**Estimated Time**: 1-2 hours

### Current Problems

- **Loadcell multi-channel**: Using only `filtered_value[0]` (single channel) → vulnerable to drift/deviation
- **Single-item return deduction**: Always `count -= 1` regardless of weight → errors when returning multiple items sequentially

### Target Files

| File | Changes |
|------|----------|
| `services/model/model_service/service/trigger_service.py` | `_calculate_weight_delta()` multi-channel average |
| `services/model/model_service/api/routes/trigger.py` | `_detect_stable_segment()` multi-channel average |
| `services/model/model_service/session/product_aggregator.py` | `_handle_return()` weight-based count estimation |

### Completion Criteria

- [x] Calculate delta using the average of all channels in `filtered_value` array
- [x] Estimate return count via `delta_weight / product_weight` (rounded)
- [x] All existing tests pass: `uv run pytest services/model/tests -v`

---

### Phase 0 Implementation Prompt

```
Implement 2 Medium-priority items in the Edge_Environment Model service.

**Working Directory**: Edge_Environment/
**Current Version**: 5.4.0

## 0a. Loadcell Multi-Channel Average (trigger_service.py, trigger.py)

Find all locations in the current code that use only `filtered_value[0]`
and replace with the average of all valid channels.

- trigger_service.py `_calculate_weight_delta()` function (around line 1073-1102)
- trigger.py `_detect_stable_segment()` function (around line 158-209)

Valid channel definition: parseable numeric values in the `filtered_value` array (whether to exclude 0 follows existing logic).
If parsing fails or the array is empty, maintain fallback to existing [0] usage.

Helper function example:
def _avg_loadcell_channels(values: list[str]) -> float:
    parsed = [float(v.strip().lstrip('+')) for v in values if v.strip()]
    return sum(parsed) / len(parsed) if parsed else 0.0

## 0b. Multi-Item Return Count Deduction (product_aggregator.py)

Modify the `_handle_return()` function (around line 183-228):

Changes:
1. Use `find_product_by_weight()` to identify the product (keep existing behavior)
2. Calculate `estimated_count = round(delta_weight / agg.weight)` using the identified product's weight
3. Deduct with `agg.count = max(0, agg.count - estimated_count)`
4. If estimated_count is <= 0 or abnormally large (> agg.count), deduct only 1 (safety guard)

After completion, run tests: uv run pytest services/model/tests -v
```

---

## Phase 1: Simultaneous Multi-Item Return Processing [Core Unimplemented]

**Purpose**: Handle combination matching when multiple products are returned in a single trigger

**Estimated Time**: 3-4 hours

### Current Problem

```
Scenario: Simultaneous return of Chicken Mayo (365g) x 2 + Kimbap (250g) x 1
delta_weight = +980g

Current behavior:
  find_product_by_weight(980g) → match failure
  → UnmatchedReturn recorded (return not processed)

Desired behavior:
  Single match fails → attempt combination matching
  combination_search([365g, 250g], 980g)
  → 365x2 + 250x1 = 980 ✓
  → Deduct Chicken Mayo 2, Kimbap 1
```

### Target Files

| File | Changes |
|------|----------|
| `services/model/model_service/weight/strict_weight_matcher.py` | Add return-specific combination search method |
| `services/model/model_service/session/product_aggregator.py` | `_handle_return()` combination matching fallback |

### Completion Criteria

- [x] Single product return: existing behavior preserved
- [x] Multiple same products simultaneous return: count determined by `round(delta / weight)`
- [x] Different products combination simultaneous return: processed via combination search
- [x] Match failure records `UnmatchedReturn` (existing behavior preserved)
- [x] Search depth limit (max 4 product combinations, for performance)
- [x] New tests pass: `test_product_aggregator.py` (TestBatchReturn) + `test_strict_weight_matcher.py` (TestFindReturnCombination)

---

### Phase 1 Implementation Prompt

```
Implement simultaneous multi-item return processing in the Edge_Environment Model service.

**Working Directory**: Edge_Environment/
**Current Version**: 5.4.0

## Background

Currently `product_aggregator.py`'s `_handle_return()` only matches single products.
`StrictWeightMatcher` is only applied for extraction (delta < 0),
and combination matching for returns (delta > 0) is not implemented.

## Implementation

### 1. Add return-specific combination search method to strict_weight_matcher.py

Reference existing extraction combination search logic to add a return-direction method.

```python
def find_return_combination(
    self,
    available_products: dict,  # {product_id: AggregatedProduct}
    delta_weight: float,       # total returned weight (positive)
    tolerance: float = 3.0,    # tolerance (g)
    max_depth: int = 4,        # max combination depth
) -> Optional[dict]:           # {product_id: count} or None
```

Search algorithm:
- Use only products with count > 0 as candidates
- Find closest combination to delta_weight (backtracking)
- Prefer combinations with fewest items within tolerance

### 2. Modify `_handle_return()` in product_aggregator.py

Steps:
1. Attempt existing single product match (keep)
2. On failure, call StrictWeightMatcher.find_return_combination()
3. If combination found, deduct count for each product
4. If combination also fails, record existing UnmatchedReturn

### 3. Write tests/test_batch_return.py

Test cases:
- 2 same products simultaneous return
- 2 different product types simultaneous return
- 3 item combination return
- Match failure case (UnmatchedReturn verification)

After completion: uv run pytest services/model/tests/test_batch_return.py -v
```

---

## Phase 2: Camera Interference Handling

**Purpose**: Improve accuracy for cases where camera detection is poor and tune thresholds

**Estimated Time**: 2-3 hours

### Current Behavior

| Case | Current Confidence | Decision Result |
|--------|-----------|---------|
| Top + Side both detected | Max 1.0 | Normal |
| Top only detected | top_conf x 0.5 (max 0.475) | Normal (exceeds threshold 0.3) |
| Side only detected | side_conf x 0.5 (max 0.44) | Normal (exceeds threshold 0.3) |
| Camera miss + loadcell | Max 0.5 (weight-only) | PARTIAL |
| Camera miss + no active_products | - | NO_DETECTION |

### Target Files

| File | Changes |
|------|----------|
| `services/model/model_service/core/config.py` | Add top/side only weight env vars |
| `services/model/model_service/vision/voting_ensemble.py` | Make single-direction weights configurable |
| `services/model/model_service/engine/product_decision_engine.py` | Strengthen weight-only fallback |

### Completion Criteria

- [x] `MODEL__VISION__TOP_ONLY_WEIGHT` env var for adjusting single-direction weight
- [x] `MODEL__VISION__SIDE_ONLY_WEIGHT` env var added
- [x] `MODEL__WEIGHT__MIN_WEIGHT_CHANGE` — already exists as `MODEL__TRIGGER__MIN_WEIGHT_CHANGE_GRAMS` (TriggerModel)
- [x] Ultra-fast extraction scenario test cases written (`test_camera_interference.py`)

---

### Phase 2 Implementation Prompt

```
Improve camera interference handling in the Edge_Environment Model service.

**Working Directory**: Edge_Environment/
**Current Version**: 5.4.0

## Background

Currently, single-direction camera detection reduces confidence by 50%.
top_only_weight and side_only_weight are hardcoded (0.5) in the code,
making field tuning difficult.

Additionally, when ultra-fast extraction results in camera miss,
the loadcell-only fallback returns NO_DETECTION if there are no active_products.

## Implementation

### 1. Add env vars to config.py

Add the following settings to VisionSettings:
- `top_only_weight: float = 0.6` (MODEL__VISION__TOP_ONLY_WEIGHT)
- `side_only_weight: float = 0.5` (MODEL__VISION__SIDE_ONLY_WEIGHT)
- `min_weight_change_grams: float = 5.0` (MODEL__WEIGHT__MIN_WEIGHT_CHANGE)

### 2. Use config values in voting_ensemble.py

Replace hardcoded 0.5 with config's top_only_weight / side_only_weight.
Inject settings at voting_ensemble initialization or read directly from config.

### 3. Review weight-only fallback in product_decision_engine.py

Read the `judge_by_weight_only()` function:
- Check the branch that returns NO_DETECTION when active_products is empty
- Enhance logging for easier field debugging
- When delta_weight > 0 and no active_products:
  → Investigate whether fallback using the session's last known product info is feasible

### 4. Write tests

tests/test_camera_interference.py:
- Top-only scenario (weighted_conf verification)
- Side-only scenario (weighted_conf verification)
- Camera miss + loadcell scenario

After completion: uv run pytest services/model/tests/test_camera_interference.py -v
```

---

## Phase 3: Loadcell Interference Handling

**Purpose**: Reduce false positives from unintentional weight change patterns

**Estimated Time**: 2-3 hours

### Current Problems

```
Scenario 1: Pushing product sideways then extracting
  Loadcell: [1000, 1005, 1200, 1050, 1010, 635, 640, 638]
  Current: Stable segment detection may include the 1200g peak
  Problem: Delta calculation error possible

Scenario 2: Temporarily placing on another product
  Loadcell: [1000, 1002, 1000, 1365, 1363, 1000, 1002, 1001, 635, 638]
  Current: Mid-peak (1365g segment) may be incorrectly detected as end stable segment
  Problem: End weight judged as 1365g → delta error
```

### Target Files

| File | Changes |
|------|----------|
| `services/model/model_service/api/routes/trigger.py` | Improve stable segment detection algorithm |

### Completion Criteria

- [x] Exclude temporary peaks (appear and revert) from stable segments
- [x] Peak detection: median-based `_filter_peaks()` algorithm (threshold_factor=1.5, min_diff=50g)
- [x] Fallback to full data if insufficient stable segments after peak exclusion
- [x] Tests: `test_loadcell_interference.py` (8 TCs pass)

---

### Phase 3 Implementation Prompt

```
Improve loadcell interference scenario handling in the Edge_Environment Model service.

**Working Directory**: Edge_Environment/
**Current Version**: 5.4.0

## Background

The `_detect_stable_segment()` function in `trigger.py` determines the first
segment where sliding window STD < 15g as the stable segment.

Two interference scenarios cause false positives:
1. Pushing product sideways then extracting: weight peak before extraction
2. Temporarily placing on another product: weight peak before end

## Implementation

### 1. Improve stable segment algorithm (trigger.py)

Modify `_detect_stable_segment()`:

Before improvement:
- First window with STD < 15g → stable segment

After improvement:
- Step 1: Detect "local peaks" in the full data
  - Definition: segments that differ by 2x or more from the preceding 5-point average
  - Mask peak segments (exclude those points)
- Step 2: Run existing STD algorithm on peak-excluded data
- Step 3: If valid points < window_size (5), fallback to original data including peaks

Parameters:
- peak_threshold_multiplier: 2.0 (recommend making configurable)
- peak_context_window: 5 (window before/after peak for determination)

### 2. Write tests (tests/test_loadcell_interference.py)

Test cases:
- Normal loadcell data → existing behavior preserved
- Start segment peak data → correct start weight after peak exclusion
- End segment peak data → correct end weight after peak exclusion
- All peaks (fallback case)

After completion: uv run pytest services/model/tests/test_loadcell_interference.py -v
```

---

## Phase 4: Complex Scenario Verification and Fixes

**Purpose**: Verify that Phase 0-3 implementations work correctly together across all scenarios

**Estimated Time**: 3-4 hours

### Scenarios to Verify

1. Basic extraction (left/right hand agnostic, vision-based)
2. Sequential multi-extraction (multiple triggers within same Door Session)
3. Same-zone return
4. Cross-zone return (zone 1 extraction → zone 2 return)
5. A extraction → return → B extraction (product switch)

### Target Files

| File | Changes |
|------|----------|
| `services/model/tests/test_scenarios_phase4.py` | New integration tests for each scenario |
| `services/model/model_service/session/product_aggregator.py` | Bug fixes (if found during verification) |
| `services/model/model_service/session/door_session_store.py` | Bug fixes (if found during verification) |

### Completion Criteria

- [x] All 5 scenarios pass pytest
- [x] Cross-zone return GlobalSession state accuracy confirmed
- [x] Product switch count accuracy confirmed (no negatives)

---

### Phase 4 Implementation Prompt

```
Write complex scenario integration tests and fix bugs in the Edge_Environment Model service.

**Working Directory**: Edge_Environment/
**Current Version**: 5.4.0

## Background

Phase 0-3 implementations are complete. Now verify that all extraction/return
scenarios work correctly with integration tests, and fix any discovered bugs.

## Implementation

### 1. Write tests/test_scenarios_phase4.py

Write tests that reproduce each scenario using mock trigger data.
Reference existing test patterns in `tests/test_door_session_store.py`.

Test cases:

**TC1: Basic Extraction**
- Trigger: delta=-365g, vision=[Chicken Mayo (conf=0.85)]
- Expected: Chicken Mayo count +1

**TC2: Sequential Multi-Extraction**
- Trigger 1: delta=-365g, vision=[Chicken Mayo]
- Trigger 2: delta=-365g, vision=[Chicken Mayo]
- Expected: Chicken Mayo count +2, accumulated in same DoorSession

**TC3: Same-Zone Return**
- Trigger 1: delta=-365g, vision=[Chicken Mayo]  (extraction)
- Trigger 2: delta=+365g, vision=[]               (return)
- Expected: Chicken Mayo final count 0

**TC4: Cross-Zone Return**
- Zone 1 trigger: delta=-365g, vision=[Chicken Mayo]
- Zone 2 trigger: delta=+365g, vision=[]
- Expected: GlobalSession Chicken Mayo final count 0

**TC5: A → Return → B Extraction**
- Trigger 1: delta=-365g, vision=[Chicken Mayo]
- Trigger 2: delta=+365g (Chicken Mayo return)
- Trigger 3: delta=-250g, vision=[Kimbap]
- Expected: Final Chicken Mayo=0, Kimbap=1

### 2. Bug Fixes

If any TC fails, check relevant files and fix.
Fix scope:
- product_aggregator.py count calculation logic
- door_session_store.py GlobalSession aggregation logic

After completion: uv run pytest services/model/tests/test_scenarios_phase4.py -v
```

---

## Phase 5: Two-Handed Multi-Extraction

**Purpose**: Handle two-handed scenarios, especially verify simultaneous extraction behavior

**Estimated Time**: 2-3 hours

### Current Architecture Understanding

```
Camera Driver sends separate POST /trigger per zone
  → Zone 1 trigger (registered in async queue)
  → Zone 2 trigger (registered in async queue)
  → Worker processes sequentially (asyncio.Queue)
  → Both results accumulate in same GlobalDoorSession
```

Two-handed scenarios are fundamentally supported by this architecture, but concurrency and VotingEnsemble behavior need verification.

### Target Files

| File | Changes |
|------|----------|
| `services/model/tests/test_scenarios_phase5.py` | Two-handed scenario tests |
| `services/model/model_service/session/global_door_session.py` | Concurrent extraction metadata (optional) |

### Completion Criteria

- [x] Two-handed single zone: two extractions accurately accumulate in same DoorSession
- [x] Two-handed multi zone: GlobalSession accurately aggregates both zone results
- [x] Concurrent extraction timestamp detection within N seconds (optional)

---

### Phase 5 Implementation Prompt

```
Implement and verify two-handed multi-extraction scenarios in the Edge_Environment Model service.

**Working Directory**: Edge_Environment/
**Current Version**: 5.4.0

## Background

Phase 0-4 implementations are complete. Now verify two-handed extraction scenarios.
Camera Driver sends separate /trigger calls per zone, so 2-zone extraction is
processed as 2 triggers. Verify that GlobalDoorSession accumulates correctly.

## Implementation

### 1. Write tests/test_scenarios_phase5.py

**TC1: Two-Handed - 1 Product, 1 Zone**
- Zone 1 trigger: delta=-365g, vision=[Chicken Mayo]
- Zone 1 trigger: delta=-365g, vision=[Chicken Mayo]  (nearly simultaneous)
- Expected: Chicken Mayo count = 2, accumulated in zone 1 DoorSession

**TC2: Two-Handed - 2 Products, 1 Zone (Different Items)**
- Zone 1 trigger: delta=-365g, vision=[Chicken Mayo]
- Zone 1 trigger: delta=-250g, vision=[Kimbap]
- Expected: Chicken Mayo=1, Kimbap=1

**TC3: Two-Handed - 2 Products, 2 Zones**
- Zone 1 trigger: delta=-365g, vision=[Chicken Mayo]
- Zone 2 trigger: delta=-250g, vision=[Kimbap]
- Expected: GlobalSession Chicken Mayo=1 (zone1), Kimbap=1 (zone2)

### 2. Concurrent Extraction Detection (Optional)

If two triggers' `trigger_time` difference is within 2 seconds, add
"concurrent_trigger=True" metadata to TriggerResult. (For future dashboard/alert use)

Related file: `service/trigger_service.py`'s `_worker_loop()`

### 3. Bug Fixes

Fix DoorSessionStore or GlobalDoorSession logic if any TC fails.

After completion: uv run pytest services/model/tests/test_scenarios_phase5.py -v
Full suite: uv run pytest services/model/tests -v
```

---

## Full Completion Checklist

### Phase 0
- [x] Loadcell multi-channel average (`trigger_service.py`, `trigger.py`)
- [x] Multi-item return count deduction (`product_aggregator.py`)
- [x] All existing tests pass

### Phase 1
- [x] `strict_weight_matcher.py` return combination search
- [x] `product_aggregator.py` combination matching fallback
- [x] Batch return tests pass (`test_product_aggregator.py`, `test_strict_weight_matcher.py`)

### Phase 2
- [x] config.py env vars added (top/side only weight)
- [x] `voting_ensemble.py` made configurable
- [x] `tests/test_camera_interference.py` new tests pass

### Phase 3
- [x] `trigger.py` + `trigger_service.py` peak filtering algorithm
- [x] `tests/test_loadcell_interference.py` new tests pass

### Phase 4
- [x] `tests/test_scenarios_phase4.py` 5 TCs pass
- [x] Cross-zone return bug fixes complete

### Phase 5
- [x] `tests/test_scenarios_phase5.py` 3 TCs pass
- [x] Full test suite passes: `uv run pytest services/model/tests -v`

---

> **Next Step**: Execute each Phase's implementation prompt sequentially in a new session.
> It is recommended to proceed to Phase 5 only after completing full integration verification in Phase 4.
