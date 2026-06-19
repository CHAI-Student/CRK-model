# 상품 판단 파이프라인 상세 분석 (3~7단계)

> **버전**: v5.4 | **최종 업데이트**: 2026-02-20

YOLO TensorRT 추론부터 반품/교차존 처리까지 **5개 단계 심층 분석**.

---

## 요약

### 5개 단계 핵심 요약

| 단계 | 핵심 | 입력 | 출력 |
|------|------|------|------|
| 3. YOLO 추론 | TensorRT FP16 480x480 | BGR 프레임 | YOLODetection[] |
| 4. 필터링 | Motion + Hand + ROI + Conf | YOLODetection[] | 필터링된 Detection[] |
| 5. Voting | Top/Side 가중 투표 결합 | 필터링된 Detection[] | VoteResult[] |
| 6. 판단 | StrictWeightMatcher + Vision | VoteResult[] + delta_weight | JudgmentResult |
| 7. 세션 | 집계 + 반품 + 교차존 | JudgmentResult | Node.js 응답 JSON |

### 데이터 변환 파이프라인

```
BGR 프레임 (480x480x3)
    │
    │  YOLO TensorRT FP16
    ▼
YOLODetection[]              ← xyxy, conf, cls, name, center, is_hand
    │
    │  Motion + Hand Path + Side ROI + Conf 필터
    ▼
Filtered Detection[]         ← 배경 노이즈, 미이동 객체 제거됨
    │
    │  VotingEnsemble.add_vote() × N 프레임
    ▼
VoteCount (class_id별)       ← count, max_conf, sum_conf (Top/Side 각각)
    │
    │  VotingEnsemble.combine()
    ▼
VoteResult[]                 ← weighted_confidence, vote_ratio
    │
    │  → EnsembleResult 변환
    ▼
EnsembleResult[]             ← class_id, combined_confidence
    │
    │  ProductDecisionEngine.judge()
    ▼
JudgmentResult               ← status, products, confidence
    │
    │  ProductAggregator → DoorSession → GlobalSession
    ▼
Node.js 응답 JSON            ← zones, totalPrice, productCount
```

---

## 3장. YOLO TensorRT 추론 상세

**파일**: `vision/yolo_wrapper.py` (557 lines)

### YOLOWrapper 클래스 구조

```python
class YOLOWrapper:
    INPUT_SIZE = 480           # 입력 해상도
    CROP_WIDTH = 480           # 640→480 우측 크롭
    MAX_DETECTIONS = 20        # 프레임당 최대 탐지
    CACHE_CLEANUP_INTERVAL = 100  # GPU 캐시 청소 주기

    def load(model_path: str)      # TensorRT 엔진 로드 (.engine 전용)
    def detect(frame: ndarray)     # 단일 프레임 추론
    def parse_results(results)     # YOLO 결과 → YOLODetection 변환
```

### TensorRT FP16 최적화

| 항목 | 설정 | 이유 |
|------|------|------|
| 입력 크기 | 480x480 | 4GB 메모리 제약 (640 대비 44% 감소) |
| 정밀도 | FP16 (half=True) | 메모리 50% 절약, 속도 2x 향상 |
| conf | 0.01 | 매우 낮은 임계값 → 4단계 필터링에서 정밀 제거 |
| max_det | 20 | 후처리 부하 제한 |
| 크롭 | 640→480 우측 크롭 | 자판기 유효 영역만 사용 |

### YOLODetection 데이터 구조

```python
@dataclass
class YOLODetection:
    xyxy: tuple[float, float, float, float]  # (x1, y1, x2, y2) 바운딩 박스
    conf: float                               # 신뢰도 (0.0~1.0)
    cls: int                                  # 클래스 ID
    name: str                                 # 클래스명 ("cola", "water", ...)
    center: tuple[float, float]               # bbox 중심 (cx, cy)
    is_hand: bool                             # cls == 0 → 손 감지
```

### GPU 캐시 관리

```
추론 카운터 (inference_count)
    │
    ├─ +1 매 추론마다
    │
    └─ inference_count % 100 == 0 ?
        └─ YES → torch.cuda.empty_cache()
            └─ Jetson 4GB 메모리 안정화
```

- 100회 추론마다 `torch.cuda.empty_cache()` 호출
- Jetson Orin Nano 4GB 환경에서 메모리 단편화 방지

### 프레임당 추론 흐름

```
BGR 프레임 (480, 480, 3)
    │
    ├─ allowed_class_ids 설정 확인
    │
    ├─ YOLO model.predict()
    │   ├─ conf=0.01
    │   ├─ half=True (FP16)
    │   ├─ max_det=20
    │   └─ verbose=False
    │
    ├─ 결과 파싱
    │   └─ boxes → YOLODetection 리스트
    │       ├─ is_hand (cls == 0) 분류
    │       └─ center 좌표 계산
    │
    └─ GPU 캐시 체크 (100회 주기)

출력: List[YOLODetection] (상품 + 손)
```

---

## 4장. 필터링 파이프라인 상세

**파일**: `video/video_processor.py`, `vision/hand_path_tracker.py`

### 필터링 개요

4가지 필터가 순차적으로 적용되어 노이즈를 제거한다:

```
Raw YOLODetection[] (프레임당 최대 20개)
    │
    ├─ 1) Motion Filter ──── 이동하지 않은 객체 제거
    ├─ 2) Hand Path Filter ─ 손 궤적과 무관한 객체 제거
    ├─ 3) Side ROI Filter ── 측면 카메라 우측 영역 제거
    └─ 4) Confidence Filter ─ conf < 0.4 제거
    │
    ▼
Filtered Detection[] (실제 상호작용한 상품만 남음)
```

### 4.1 Motion Filter: BboxTracker

**목적**: 프레임 간 이동하지 않은 객체(배경 오검출) 제거

```python
class BboxTracker:
    first_center: tuple    # 최초 감지 위치
    last_center: tuple     # 최종 감지 위치
    max_distance: float    # 최대 이동 거리
    detection_count: int   # 감지 횟수

    @property
    def dynamic_threshold(self) -> float:
        return max(15.0, bbox_size * 0.10)

    def has_motion(self) -> bool:
        total_displacement = distance(first_center, last_center)
        return (total_displacement >= threshold
                or max_distance >= threshold)
```

**동작 원리:**
1. 클래스별로 BboxTracker 생성
2. 프레임마다 bbox 중심 좌표 업데이트
3. 전체 프레임 처리 후 `has_motion()` 확인
4. **동적 임계값**: `max(15px, bbox_size * 0.10)`
   - 큰 객체는 더 많이 움직여야 "이동"으로 판정
   - 최소 15px 이동 필요

```
프레임 1          프레임 5          프레임 10
┌────┐            ┌────┐            ┌────┐
│ A  │            │ A  │────15px───►│ A  │  → has_motion? YES
└────┘            └────┘            └────┘

┌──┐              ┌──┐              ┌──┐
│B │              │B │              │B │     → has_motion? NO (제거)
└──┘              └──┘              └──┘
```

### 4.2 Hand Path Filter: HandPathTracker

**목적**: 손이 실제로 접근/접촉한 상품만 남기기

**파일**: `vision/hand_path_tracker.py` (352 lines)

```python
class HandTrajectory:
    centers: List[tuple]        # 손 중심 좌표 리스트 (궤적)
    avg_bbox_size: float        # 평균 손 bbox 크기

    def intersects_bbox(product_center, product_size) -> bool:
        tolerance = hand_radius + product_radius * 0.5
        # 궤적의 어느 지점이든 tolerance 내에 있으면 True
```

```python
class ProductBboxHistory:
    avg_center: tuple           # 상품 평균 위치
    avg_bbox_size: float        # 상품 평균 크기
```

**교차 검증 알고리즘:**

```
손 궤적 (HandTrajectory)
    ●──●──●──●──●──●──●
    │                    ╲
    │   상품 A            ╲  tolerance 원
    │  ┌────────┐     ●───●──●
    │  │        │    /
    │  └────────┘   /
    │              /
    ●──●──●──●──●

tolerance = hand_radius + product_radius * 0.5

궤적의 어느 점이라도 상품 중심에서 tolerance 이내
→ intersects = True → 상품 A 유지
```

**필터 조건:**
- `min_hand_detections = 3`: 최소 3프레임 이상 손 감지
- `min_path_length = 30.0px`: 손 궤적 최소 길이
- 유효한 손 궤적이 없으면 → **필터링 자체를 스킵** (전체 통과)

### 4.3 Side ROI Filter

**목적**: 측면 카메라에서 유효 영역만 사용

```
Side 카메라 영상 (480x480)
┌───────────┬───────────┐
│           │           │
│  유효 영역 │  제외 영역 │
│ (x < 240) │ (x ≥ 240) │
│           │           │
│  상품 선반  │  노이즈   │
│           │           │
└───────────┴───────────┘
     사용          미사용
```

- **기준**: `center_x < 240px` (좌측 절반만 사용)
- Side 카메라 전용 필터 (Top 카메라에는 미적용)

### 4.4 Confidence Threshold

최종 필터로 낮은 신뢰도 감지 제거:

```
YOLO conf=0.01 (매우 낮은 임계값으로 추론)
    │
    ├─ 0.01 ~ 0.39: 제거 (노이즈)
    └─ 0.40 ~ 1.00: 유지 (유효 감지)
```

- YOLO 추론 시 conf=0.01로 광범위하게 감지
- 필터링 단계에서 conf ≥ 0.4만 최종 유지
- 이 전략으로 Motion/Hand Path 필터가 중간 신뢰도(0.1~0.4) 감지도 추적 가능

---

## 5장. Voting Ensemble 상세

**파일**: `video/voting_ensemble.py` (385 lines)

### VoteCount 구조

```python
@dataclass
class VoteCount:
    count: int = 0              # 감지된 프레임 수
    max_confidence: float = 0.0 # 최대 신뢰도
    sum_confidence: float = 0.0 # 신뢰도 합계
```

### 단일 카메라 투표 축적

```python
def add_vote(class_id: int, confidence: float):
    vote = votes[class_id]     # VoteCount 조회 또는 생성
    vote.count += 1
    vote.max_confidence = max(vote.max_confidence, confidence)
    vote.sum_confidence += confidence
```

```
Top 카메라: 30 프레임 처리
────────────────────────────────────────
프레임  1: cola(0.8), water(0.3)
프레임  2: cola(0.9)
프레임  3: cola(0.7), water(0.5)
...
프레임 30: cola(0.8)

결과:
  cola:  VoteCount(count=28, max_conf=0.95, sum_conf=22.4)
  water: VoteCount(count=5,  max_conf=0.5,  sum_conf=2.1)
```

### Top + Side combine 알고리즘

```python
def combine(
    top_votes: Dict[int, VoteCount],
    side_votes: Dict[int, VoteCount],
    total_top_frames: int,
    total_side_frames: int,
    top_weight: float = 0.5,      # Top 가중치
    side_weight: float = 0.5,     # Side 가중치
    common_class_bonus: float = 0.2,  # 양쪽 감지 보너스
) -> List[VoteResult]:
```

**결합 공식:**

```
CASE 1: 양쪽 카메라 감지
  weighted_confidence = top_conf * 0.5
                      + side_conf * 0.5
                      + min(top_conf, side_conf) * 0.2
  (클램핑: min(result, 1.0))

CASE 2: Top 카메라만 감지
  weighted_confidence = top_conf * 0.5

CASE 3: Side 카메라만 감지
  weighted_confidence = side_conf * 0.5
```

**예시 계산:**

```
cola: top_conf=0.95, side_conf=0.80
  → 0.95 * 0.5 + 0.80 * 0.5 + min(0.95, 0.80) * 0.2
  → 0.475 + 0.400 + 0.160
  → 1.035 → 클램핑 → 1.0

water: top_conf=0.50, side_conf=없음
  → 0.50 * 0.5
  → 0.25
```

### vote_ratio 필터링

```python
# 조건 (OR):
vote_ratio = vote_count / total_frames >= 0.05   # 5% 이상
OR
vote_count >= 3                                    # 3프레임 이상 (절대 기준)
```

- **상대 기준**: 전체 프레임의 5% 이상 감지
- **절대 기준**: 최소 3프레임 이상 감지 (짧은 영상 대비)
- 두 조건 중 하나만 충족하면 통과

### VoteResult 전체 필드

```python
@dataclass
class VoteResult:
    class_id: int                    # YOLO 클래스 ID
    class_name: str                  # 클래스명
    vote_count: int                  # 총 투표 수 (Top + Side)
    vote_ratio: float                # 투표 비율
    max_confidence: float            # 전체 최대 신뢰도
    avg_confidence: float            # 전체 평균 신뢰도
    weighted_confidence: float       # 가중 결합 신뢰도 (최종 사용)

    # Top 카메라 상세
    top_vote_count: int
    top_max_confidence: float
    top_avg_confidence: float

    # Side 카메라 상세
    side_vote_count: int
    side_max_confidence: float
    side_avg_confidence: float
```

---

## 6장. 판단 엔진 상세

**파일**: `engine/decision_engine.py` (825 lines), `weight/strict_weight_matcher.py` (461 lines)

### ProductDecisionEngine.judge() 의사결정 트리

```
judge(candidates, delta_weight, active_products)
    │
    ├─ [1] vision_only 모드?
    │   └─ YES → _judge_vision_only()
    │       ├─ 최고 신뢰도 후보 1개 선택
    │       ├─ confidence = combined_confidence * 0.7
    │       ├─ count = 1
    │       └─ status = COMPLETE
    │
    ├─ [2] 후보(candidates) 없음?
    │   └─ YES → judge_by_weight_only()
    │       ├─ active_products에서 무게로 매칭
    │       ├─ ±tolerance 이내 상품 검색
    │       └─ 매칭 없으면 → NO_DETECTION
    │
    ├─ [3] |delta_weight| < min_weight_change (5g)?
    │   └─ YES → NO_DETECTION (무게 변화 미미)
    │
    ├─ [4] strict_mode? (기본값: true)
    │   └─ YES → _judge_strict()
    │       ├─ StrictWeightMatcher.find_valid_combinations()
    │       ├─ 유효 조합 있으면 → 최상위 선택 → COMPLETE
    │       └─ 유효 조합 없으면 → [5]로 fallback
    │
    └─ [5] 일반 매칭 (fallback)
        ├─ single_product_match: 단일 상품 무게 매칭
        ├─ combination_match: 복수 상품 조합 매칭
        └─ partial_result: 부분 매칭 (PARTIAL)
```

### StrictWeightMatcher 백트래킹 알고리즘

**입력:**
- `candidates`: YOLO 후보 리스트 (EnsembleResult[])
- `delta_weight`: 무게 변화량 (g)
- `active_products`: 자판기 등록 상품 정보

**처리 단계:**

```
Step 1: 후보 추출
    ├─ YOLO 후보 중 active_products에 존재하는 것만
    ├─ 무게 정보 있음 (weight > 0)
    └─ 재고 있음 (stock > 0)
    → CandidateProduct[] 생성

Step 2: Backtracking 부분집합 합 탐색
    ├─ target_weight = |delta_weight|
    ├─ 모든 조합 탐색 (상품 × 개수)
    ├─ |조합_무게 - target| ≤ tolerance(3g) → 유효 조합
    ├─ 조합_무게 > target + tolerance → 가지치기(pruning)
    └─ 제약:
        ├─ max_items = 5 (조합당 최대 상품 종류)
        ├─ max_count_per_item = 10 (상품당 최대 개수)
        └─ max_combinations = 100 (최대 조합 수)

Step 3: match_score로 정렬
    └─ 내림차순 → 최상위 = 최적 조합
```

**Backtracking 탐색 예시:**

```
target_weight = 206g, tolerance = 3g
후보: cola(200g), water(180g), juice(100g)

탐색 트리:
  cola×1 (200g)
    ├─ |200 - 206| = 6g > 3g → 미달, 계속 탐색
    ├─ + water×1 (380g) > 209g → 가지치기
    └─ + juice×1 (300g) > 209g → 가지치기

  cola×1 + 추가 불가 → 다음

  water×1 (180g)
    └─ |180 - 206| = 26g > 3g → 미달

  juice×1 (100g)
    └─ + juice×1 (200g) → |200 - 206| = 6g > 3g
    └─ + juice×2 (300g) > 209g → 가지치기

→ 유효 조합 없음 (이 경우 fallback으로)
```

**match_score 계산:**

```python
# 1. 무게 점수 (오차 적을수록 높음)
weight_score = max(0.0, 1.0 - (weight_error / tolerance))
# tolerance=3g, error=1g → 1.0 - 0.33 = 0.67
# tolerance=3g, error=0g → 1.0 - 0.00 = 1.00

# 2. Vision 점수 (개수 가중 평균 신뢰도)
vision_score = Σ(conf_i × count_i) / Σ(count_i)
# cola(conf=0.9, count=2) + water(conf=0.7, count=1)
# → (0.9×2 + 0.7×1) / 3 = 0.833

# 3. 단순성 점수 (상품 종류 적을수록 높음)
simplicity_score = max(0.0, 1.0 - (len(items) - 1) * 0.2)
# 1종류 → 1.0
# 2종류 → 0.8
# 3종류 → 0.6
# 4종류 → 0.4
# 5종류 → 0.2

# 최종: 가중 평균
match_score = weight_score * 0.6
            + vision_score * 0.3
            + simplicity_score * 0.1
```

**ValidCombination 구조:**

```python
@dataclass
class ValidCombination:
    items: List[CombinationItem]     # 조합 구성 상품들
    total_weight: float              # 조합 총 무게
    target_weight: float             # 목표 무게 (|delta_weight|)
    weight_error: float              # 무게 오차 (절대값)
    avg_vision_confidence: float     # 평균 Vision 신뢰도
    match_score: float               # 매칭 점수 (정렬용)

    @property
    def total_price(self) -> int     # 총 가격
    @property
    def total_count(self) -> int     # 총 개수

@dataclass
class CombinationItem:
    candidate: CandidateProduct      # 후보 상품 정보
    count: int                       # 개수

@dataclass
class CandidateProduct:
    class_id: int                    # YOLO 클래스 ID
    name: str                        # 상품명
    weight: float                    # 단위 무게 (g)
    stock: int                       # 재고 수량
    vision_confidence: float         # Vision 신뢰도
    unit_price: int                  # 단가 (원)
```

### 최종 신뢰도 계산: fusion_confidence

```
fusion_confidence = vision_conf * 0.4
                  + weight_conf * 0.5
                  + count_score * 0.1
```

| 요소 | 가중치 | 설명 |
|------|--------|------|
| vision_conf | 0.4 | YOLO + Voting Ensemble 기반 비전 신뢰도 |
| weight_conf | 0.5 | 무게 매칭 정확도 (무게가 가장 중요) |
| count_score | 0.1 | 개수 합리성 점수 |

**count_score 계산:**

```
count ≤ 3  → 1.0 (합리적)
count = 4  → 0.9
count = 5  → 0.8
count = 6  → 0.7
...
count ≥ 13 → 0.0
```

### JudgmentResult 상태 분류

| 상태 | 의미 | 조건 |
|------|------|------|
| COMPLETE | 확신 있는 판단 | 무게+비전 모두 매칭 성공 |
| PARTIAL | 부분 매칭 | 일부 무게만 설명 가능 |
| UNCERTAIN | 불확실 | 비전만 감지, 무게 미매칭 |
| NO_DETECTION | 감지 없음 | 후보 없음 또는 무게 변화 미미 |

---

## 7장. 세션 통합 & 반품/교차존 처리 상세

**파일**: `session/door_session.py`, `session/door_session_store.py`, `session/product_aggregator.py`, `session/global_door_session.py`, `api/routes/multi_zone.py`

### DoorSession 구조

```python
@dataclass
class DoorSession:
    session_id: str                              # UUID
    zone: int                                    # Zone 번호 (1-5)
    triggers: List[TriggerResult]                # 축적된 트리거 결과들
    aggregated_products: Dict[str, AggregatedProduct]  # 집계된 상품
    unmatched_returns: List[UnmatchedReturn]     # 미매칭 반품
    cross_zone_returns: List[CrossZoneReturn]    # 교차존 반품
    created_at: datetime
    updated_at: datetime
```

**관련 데이터 모델:**

```python
@dataclass
class TriggerResult:
    trigger_id: str          # 트리거 UUID
    session_id: str          # 소속 세션
    products: List[dict]     # 감지된 상품 리스트
    delta_weight: float      # 무게 변화량
    is_return: bool          # 반품 여부 (delta > 0)

@dataclass
class AggregatedProduct:
    product_id: str          # 상품 고유 ID
    product_idx: int         # 상품 인덱스
    name: str                # 상품명
    count: int               # 수량 (양수: 제거, 음수: 반품)
    unit_price: int          # 단가
    weight: float            # 단위 무게

@dataclass
class UnmatchedReturn:
    trigger_id: str          # 트리거 ID
    delta_weight: float      # 반품 무게
    tolerance_used: float    # 사용된 허용 오차

@dataclass
class CrossZoneReturn:
    source_zone: int         # 반품 발생 zone
    target_zone: int         # 매칭 대상 zone
    product_id: str          # 매칭된 상품 ID
    matched_weight: float    # 매칭된 무게
```

### ProductAggregator 집계 알고리즘

**파일**: `session/product_aggregator.py` (352 lines)

```
ProductAggregator.aggregate(triggers)
    │
    ├─ Trigger별 순차 처리
    │
    ├─ delta_weight < 0 (상품 제거)
    │   └─ _handle_removal()
    │       ├─ YOLO 판단 결과의 상품들 추가
    │       ├─ AggregatedProduct.count 증가
    │       └─ (고객이 상품을 꺼냈음)
    │
    ├─ delta_weight > 0 (반품/되돌려놓음)
    │   └─ _handle_return()
    │       ├─ find_product_by_weight()
    │       │   ├─ 기존 집계 상품 중 무게 매칭
    │       │   └─ tolerance 이내 상품 검색
    │       ├─ 매칭 성공 → count 감소
    │       └─ 매칭 실패 → UnmatchedReturn 기록
    │
    └─ 결과: AggregationResult
        ├─ products: Dict[str, AggregatedProduct]
        └─ unmatched_returns: List[UnmatchedReturn]
```

**상품 제거 예시:**

```
Trigger #1: delta = -200g, YOLO = [cola(200g)]
  → aggregated_products: {cola: count=1}

Trigger #2: delta = -380g, YOLO = [cola(200g), water(180g)]
  → aggregated_products: {cola: count=2, water: count=1}
```

**반품 처리 예시:**

```
Trigger #3: delta = +200g (되돌려놓음)
  → find_product_by_weight(200g)
  → cola(200g) 매칭 성공
  → aggregated_products: {cola: count=1, water: count=1}

Trigger #4: delta = +150g (되돌려놓음)
  → find_product_by_weight(150g)
  → 매칭 실패 (cola=200g, water=180g 모두 tolerance 초과)
  → unmatched_returns: [{trigger_id, delta=150g}]
```

### 교차존 반품 처리 (Cross-Zone Return)

**시나리오**: Zone 1에서 꺼낸 상품을 Zone 2에 되돌려놓는 경우

```
Zone 1 DoorSession           Zone 2 DoorSession
  cola: count=2                (반품 발생)
  water: count=1               delta = +200g
                               │
                               ├─ 자기 zone에서 매칭 시도 → 실패
                               │  (Zone 2에 cola 없음)
                               │
                               └─ UnmatchedReturn 기록
                                    │
    ┌────────────────────────────────┘
    │  교차존 탐색
    │
    ├─ Zone 1의 active session에서 무게 매칭
    │  → cola(200g) 매칭 성공!
    │
    ├─ Zone 1: cola count 2→1
    ├─ Zone 2: CrossZoneReturn 기록
    │   {source=2, target=1, product=cola, weight=200g}
    │
    └─ modified_zones 추적 → YAML 저장
```

**처리 흐름:**

```python
def _handle_cross_zone_returns(unmatched_returns, source_zone):
    for unmatched in unmatched_returns:
        for other_zone, other_session in active_sessions.items():
            if other_zone == source_zone:
                continue
            # 다른 zone의 집계 상품에서 무게 매칭
            matched = find_product_by_weight(
                other_session.aggregated_products,
                unmatched.delta_weight,
                tolerance
            )
            if matched:
                matched.count -= 1  # 대상 zone의 수량 감소
                cross_zone_return = CrossZoneReturn(
                    source_zone=source_zone,
                    target_zone=other_zone,
                    product_id=matched.product_id,
                    matched_weight=unmatched.delta_weight,
                )
                break  # 첫 매칭에서 중단
```

### GlobalDoorSession 생명주기

**파일**: `session/global_door_session.py` (193 lines)

```
                    Node.js OPEN 신호
                         │
                         ▼
              ┌──── GlobalDoorSession 생성 ────┐
              │   zone_sessions: {}             │
              │   status: "active"              │
              └────────────────────────────────┘
                         │
                    Trigger × N
                    (각 zone별 DoorSession 축적)
                         │
                    Node.js CLOSE 신호
                         │
                         ▼
              ┌──── pending_close 전환 ────────┐
              │   first_close_at = now()        │
              │   pending_close = True          │
              └────────────────────────────────┘
                         │
              ┌──── 대기 로직 ────────────────┐
              │                                │
              │  초기 대기: 20초               │
              │  (마지막 trigger 이후)         │
              │                                │
              │  이후 폴링마다:                │
              │  - 큐에 대기 trigger 있으면    │
              │    → 추가 5초 대기             │
              │  - 없으면 → 최종화             │
              └────────────────────────────────┘
                         │
                         ▼
              ┌──── 최종화 (finalize) ────────┐
              │   교차존 반품 처리             │
              │   전체 재집계                  │
              │   net_delta 검증              │
              │   status: "complete"           │
              └────────────────────────────────┘
                         │
                         ▼
                  Node.js에 최종 결과 응답
```

**pending_close 타이밍 상세:**

```python
def is_ready_to_finalize_after_close(self) -> bool:
    elapsed = now() - self.first_close_at

    # 초기 대기: 20초 (마지막 trigger 처리 완료 대기)
    if elapsed < initial_wait(20s):
        return False

    # 큐 대기 trigger 확인 (v4.10)
    if pending_trigger_count > 0:
        return False  # 아직 처리 중인 trigger 있음

    # 이후 대기: 5초 간격 체크
    if elapsed < subsequent_wait(5s) since last check:
        return False

    return True  # 최종화 가능
```

### Node.js 폴링 응답 구조

**엔드포인트**: `POST /api/judge/multi-zone`

**요청:**

```json
{
  "zone": 1,
  "session_id": "global-session-uuid",
  "door_state": "OPEN" | "CLOSE" | null
}
```

**응답 (OPEN 시 interim):**

```json
{
  "status": "processing",
  "globalSessionInfo": {
    "session_id": "uuid",
    "status": "active",
    "zone_count": 2,
    "trigger_count": 3
  },
  "zones": [
    {
      "zone": 1,
      "products": [
        {
          "product_id": "P001",
          "name": "콜라",
          "count": 2,
          "unit_price": 1500,
          "total_price": 3000,
          "weight": 200.0
        }
      ],
      "totalPrice": 3000,
      "productCount": 2
    }
  ],
  "totalPrice": 3000,
  "productCount": 2
}
```

**응답 (CLOSE 후 완료 시):**

```json
{
  "status": "complete",
  "globalSessionInfo": {
    "session_id": "uuid",
    "status": "complete",
    "zone_count": 2,
    "trigger_count": 5,
    "cross_zone_returns": [
      {
        "source_zone": 2,
        "target_zone": 1,
        "product_id": "P001",
        "matched_weight": 200.0
      }
    ]
  },
  "zones": [
    {
      "zone": 1,
      "products": [...],
      "totalPrice": 4500,
      "productCount": 3
    },
    {
      "zone": 2,
      "products": [],
      "totalPrice": 0,
      "productCount": 0
    }
  ],
  "totalPrice": 4500,
  "productCount": 3
}
```

### YAML 영속화

```python
# Background ThreadPoolExecutor로 비동기 저장
executor = ThreadPoolExecutor(max_workers=1)
executor.submit(_save_session_yaml, session, file_path)
```

```
data/sessions/
├── door_session_zone1_2026-02-20T10-30-00.yaml
├── door_session_zone2_2026-02-20T10-30-00.yaml
└── global_session_2026-02-20T10-30-00.yaml
```

- 매 trigger 처리 후 백그라운드 저장
- 서비스 재시작 시 복구용
- ThreadPoolExecutor (max_workers=1)로 I/O 블로킹 방지

---

## 참조 파일 전체 목록

| 파일 | 경로 (model_service/) | 라인 수 | 역할 |
|------|----------------------|---------|------|
| YOLO Wrapper | `vision/yolo_wrapper.py` | 557 | TensorRT 추론 |
| Hand Path Tracker | `vision/hand_path_tracker.py` | 352 | 손 경로 필터링 |
| Video Processor | `video/video_processor.py` | 905 | 영상 처리 + 필터링 |
| Voting Ensemble | `video/voting_ensemble.py` | 385 | 투표 앙상블 |
| Decision Engine | `engine/decision_engine.py` | 825 | 판단 엔진 |
| Weight Matcher | `weight/strict_weight_matcher.py` | 461 | 무게 매칭 |
| Door Session | `session/door_session.py` | 442 | DoorSession 모델 |
| Door Session Store | `session/door_session_store.py` | 1210 | Door Session 관리 |
| Product Aggregator | `session/product_aggregator.py` | 352 | 상품 집계 |
| Global Session | `session/global_door_session.py` | 193 | GlobalSession 모델 |
| Multi-Zone API | `api/routes/multi_zone.py` | 1202 | Node.js 폴링 API |

---

> 전체 흐름 개요: [PRODUCT_DETECTION_FLOW.md](./PRODUCT_DETECTION_FLOW.md) (7단계 파이프라인 개요)
