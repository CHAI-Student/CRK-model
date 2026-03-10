# 상품 판단 파이프라인 전체 흐름

> **버전**: v5.4 | **최종 업데이트**: 2026-02-20

카메라 트리거 수신부터 Node.js 결과 반환까지 **7단계 파이프라인** 개요.

---

## 한눈에 보는 전체 흐름

```
Camera Driver                    Model Service (8002)                        Node.js (8888)
     │                                  │                                        │
     │  POST /trigger                   │                                        │
     │  (AVI + loadcells)               │                                        │
     ├─────────────────────────────────►│                                        │
     │                              1. Trigger 수신                              │
     │                              (큐 등록, 중복제거,                          │
     │                               무게 변화량 계산)                           │
     │                                  │                                        │
     │                              2. 프레임 추출                               │
     │                              (AVI → FFmpeg NVDEC                          │
     │                               → BGR 480x480)                             │
     │                                  │                                        │
     │                              3. YOLO TensorRT 추론                        │
     │                              (FP16, conf=0.01,                            │
     │                               max_det=20)                                │
     │                                  │                                        │
     │                              4. 필터링                                    │
     │                              (Motion, Hand Path,                          │
     │                               Side ROI, Conf)                            │
     │                                  │                                        │
     │                              5. Voting Ensemble                           │
     │                              (Top + Side 카메라                            │
     │                               투표 결합)                                  │
     │                                  │                                        │
     │                              6. 판단 엔진                                 │
     │                              (StrictWeightMatcher                         │
     │                               + Vision)                                  │
     │                                  │                                        │
     │                              7. 세션 통합 & 응답                          │
     │                              (DoorSession →                               │
     │                               GlobalSession)                             │
     │                                  │   POST /api/judge/multi-zone           │
     │                                  │◄───────────────────────────────────────┤
     │                                  │          10초 간격 폴링                │
     │                                  ├───────────────────────────────────────►│
     │                                  │          판단 결과 응답                │
```

## 7단계 요약

| 단계 | 이름 | 입력 | 출력 | 핵심 파일 |
|------|------|------|------|-----------|
| 1 | Trigger 수신 | AVI 경로, 로드셀 데이터 | TriggerItem (큐 등록) | `api/routes/trigger.py`, `service/trigger_service.py` |
| 2 | 프레임 추출 | AVI 파일 | BGR 프레임 스트림 (480x480) | `video/frame_extractor.py` |
| 3 | YOLO 추론 | BGR 프레임 | YOLODetection 리스트 | `vision/yolo_wrapper.py` |
| 4 | 필터링 | YOLODetection 리스트 | 필터링된 Detection 리스트 | `video/video_processor.py`, `vision/hand_path_tracker.py` |
| 5 | Voting Ensemble | Top/Side Detection 결과 | VoteResult 리스트 | `video/voting_ensemble.py` |
| 6 | 판단 엔진 | VoteResult + delta_weight | JudgmentResult | `engine/decision_engine.py`, `weight/strict_weight_matcher.py` |
| 7 | 세션 통합 | JudgmentResult | Node.js 응답 JSON | `session/door_session_store.py`, `session/product_aggregator.py` |

---

## 1단계: Trigger 수신

**파일**: `api/routes/trigger.py`, `service/trigger_service.py`

### 입력

Camera Driver가 AVI 녹화 완료 후 `POST /trigger`로 호출:

```json
{
  "zone": 1,
  "videos": {
    "top": "/path/to/top_camera.avi",
    "side": "/path/to/side_camera.avi"
  },
  "loadcells": [
    {"timestamp": 1700000000.0, "values": [100.5, 200.3, 150.2, 180.1]},
    ...
  ]
}
```

### 처리 흐름

```
POST /trigger
    │
    ├─ TriggerRequest 파싱 (zone, videos, loadcells)
    │
    ├─ 로드셀 → 무게 변화량(delta_weight) 계산
    │   ├─ _detect_stable_regions(): 슬라이딩 윈도우 std < 15.0
    │   ├─ start_avg (초반 안정 구간 평균)
    │   └─ end_avg (후반 안정 구간 평균) → delta = end - start
    │
    └─ TriggerService.enqueue_trigger()
        ├─ 멱등성 키(idempotency key) 중복 체크
        │   └─ MD5(zone + video paths), TTL 5초
        ├─ 비디오 파일 존재 확인
        ├─ 저무게 변화 스킵 (|delta| < 5g → YOLO 생략)
        └─ asyncio.Queue에 등록 (순차 처리 보장)
```

### 출력

- 즉시 응답: `{"status": "queued", "trigger_id": "uuid"}`
- 백그라운드: 큐 워커가 순차 처리 시작

### 핵심 로직

- **순차 처리 큐 (v4.10)**: TensorRT 동시 추론 충돌 방지를 위해 asyncio.Queue 사용
- **멱등성 키 (v4.5)**: 동일 trigger 5초 이내 재전송 시 중복 제거
- **저무게 스킵**: `|delta_weight| < MIN_WEIGHT_CHANGE_GRAMS(5g)` → YOLO 생략, 무게 전용 판단

---

## 2단계: 프레임 추출

**파일**: `video/frame_extractor.py`

### 입력

- AVI 비디오 파일 경로 (Top/Side 각 1개)

### 처리 흐름

```
AVI 파일
    │
    ├─ ffprobe 메타데이터 조회 (3회 재시도, 10초 간격)
    │   └─ 아직 녹화 중인 파일 대비
    │
    ├─ StreamingFrameExtractor (FFmpeg 서브프로세스)
    │   ├─ -hwaccel cuda (NVDEC 하드웨어 디코딩)
    │   ├─ -c:v mjpeg (MJPEG 코덱)
    │   ├─ gamma/contrast 보정 (카메라 타입별)
    │   │   ├─ Top: eq=gamma=1.3:contrast=1.2
    │   │   └─ Side: eq=gamma=1.2:contrast=1.1
    │   └─ rawvideo 출력 (BGR24, 480x480)
    │
    └─ 프레임 스트리밍 (__iter__ / __aiter__)
        ├─ 동기: yield BGR 프레임 (480, 480, 3)
        └─ 비동기: v5.3 Async Streaming 지원
```

### 출력

- BGR numpy 배열 스트림: shape `(480, 480, 3)`, dtype `uint8`
- 프레임별 순차 yield (메모리 효율적)

### v5.3 Async Streaming

```
기존 (v5.2 이전):          Top 추출 → Top 추론 → Side 추출 → Side 추론
                           ──────────────────────────────────────────►

Async (v5.3):              Top 추출 ─┐    ┌─ Top 추론
                                      ├────┤
                           Side 추출 ─┘    └─ Side 추론
                           ──────────────────────►
                           처리 시간 20-30% 감소
```

- Feature Flag: `MODEL__ASYNC_STREAMING__ENABLED` (기본 true)
- asyncio.TaskGroup으로 Top/Side I/O 병렬화

---

## 3단계: YOLO TensorRT 추론

**파일**: `vision/yolo_wrapper.py`

### 입력

- BGR 프레임 (480x480)

### 처리 흐름

```
BGR 프레임 (480, 480, 3)
    │
    ├─ YOLO TensorRT 추론
    │   ├─ 입력 크기: 480x480
    │   ├─ FP16 (half=True)
    │   ├─ conf=0.01 (매우 낮은 임계값 → 필터링에서 걸러냄)
    │   ├─ max_det=20
    │   └─ allowed_class_ids 필터링 (선택적)
    │
    ├─ 결과 파싱 → YOLODetection 리스트
    │   ├─ xyxy: 바운딩 박스 좌표
    │   ├─ conf: 신뢰도
    │   ├─ cls: 클래스 ID
    │   ├─ name: 클래스명
    │   ├─ center: 중심 좌표
    │   └─ is_hand: cls == 0 (손 감지)
    │
    └─ GPU 캐시 관리
        └─ 100회 추론마다 torch.cuda.empty_cache()
```

### 출력

- `List[YOLODetection]`: 프레임당 감지된 객체 리스트 (최대 20개)

---

## 4단계: 필터링

**파일**: `video/video_processor.py`, `vision/hand_path_tracker.py`

### 입력

- 전체 프레임의 YOLODetection 리스트

### 4가지 필터

```
전체 YOLODetection
    │
    ├─ 1) Motion Filter (BboxTracker)
    │   ├─ bbox 중심 이동 거리 추적
    │   ├─ 동적 임계값: max(15px, bbox_size * 0.10)
    │   └─ 이동 없는 객체 제거 (배경 오검출)
    │
    ├─ 2) Hand Path Filter (HandPathTracker)
    │   ├─ 손 궤적(trajectory) 기록
    │   ├─ 상품 bbox와 궤적 교차 검증
    │   ├─ tolerance = hand_radius + product_radius * 0.5
    │   ├─ min_hand_detections = 3, min_path_length = 30px
    │   └─ 손이 지나가지 않은 상품 제거
    │
    ├─ 3) Side ROI Filter (Side 카메라 전용)
    │   ├─ 좌측 절반만 사용: center_x < 240px
    │   └─ 우측 영역은 노이즈가 많아 제외
    │
    └─ 4) Confidence Threshold
        └─ conf < 0.4 미만 최종 제거
```

### 출력

- 필터링된 YOLODetection 리스트 (Top/Side 각각)

---

## 5단계: Voting Ensemble

**파일**: `video/voting_ensemble.py`

### 입력

- Top 카메라 필터링 결과
- Side 카메라 필터링 결과

### 처리 흐름

```
Top 카메라 결과                Side 카메라 결과
    │                              │
    ├─ 투표 축적 (VoteCount)       ├─ 투표 축적 (VoteCount)
    │  ├─ count (프레임 수)        │  ├─ count
    │  ├─ max_confidence           │  ├─ max_confidence
    │  └─ sum_confidence           │  └─ sum_confidence
    │                              │
    └──────────┬───────────────────┘
               │
          combine() 결합
               │
    ├─ weighted_confidence 계산:
    │   top_conf * 0.5 + side_conf * 0.5 + min(top, side) * 0.2
    │   (1.0 클램핑)
    │
    ├─ vote_ratio 필터링:
    │   vote_count / total_frames >= 5%  OR  vote_count >= 3
    │
    └─ VoteResult 리스트 (class_id별)
```

### 출력

- `List[VoteResult]`: 클래스별 투표 결과
  - `class_id`, `class_name`
  - `vote_count`, `vote_ratio`
  - `weighted_confidence`
  - Top/Side 카메라별 상세 투표 정보

### 결합 공식

```
weighted_confidence = top_conf * 0.5
                    + side_conf * 0.5
                    + min(top_conf, side_conf) * 0.2

(양쪽 카메라에서 감지될수록 보너스, 최대 1.0)
```

---

## 6단계: 판단 엔진

**파일**: `engine/decision_engine.py`, `weight/strict_weight_matcher.py`

### 입력

- `List[EnsembleResult]`: Voting Ensemble 결과 (VoteResult → EnsembleResult 변환)
- `delta_weight`: 무게 변화량 (g)
- `active_products`: 현재 자판기에 등록된 상품 정보

### 의사결정 트리

```
ProductDecisionEngine.judge()
    │
    ├─ vision_only 모드?
    │   └─ YES → _judge_vision_only() (confidence * 0.7, count=1)
    │
    ├─ 후보 없음?
    │   └─ YES → judge_by_weight_only() (무게만으로 판단)
    │
    ├─ |delta_weight| < min_weight_change?
    │   └─ YES → NO_DETECTION (무게 변화 미미)
    │
    ├─ strict_mode?  (기본 true)
    │   └─ YES → _judge_strict()
    │       └─ StrictWeightMatcher 사용
    │           ├─ Backtracking 부분집합 합 탐색
    │           ├─ target_weight ± 3g 이내 조합 검색
    │           └─ match_score 최상위 조합 선택
    │
    └─ (fallback) 일반 매칭
        ├─ single_product_match
        ├─ combination_match
        └─ partial_result
```

### 출력

- `JudgmentResult`:
  - `status`: COMPLETE / PARTIAL / UNCERTAIN / NO_DETECTION
  - `products`: 감지된 상품 리스트 (class_id, name, count, confidence)
  - `confidence`: 종합 신뢰도

---

## 7단계: 세션 통합 & 응답

**파일**: `session/door_session_store.py`, `session/product_aggregator.py`, `session/global_door_session.py`, `api/routes/multi_zone.py`

### 입력

- `JudgmentResult`: 6단계 판단 결과
- 현재 DoorSession 상태

### 처리 흐름

```
JudgmentResult
    │
    ├─ DoorSession에 TriggerResult 추가
    │   └─ triggers 리스트에 축적
    │
    ├─ ProductAggregator 집계
    │   ├─ 상품 제거 (delta < 0): YOLO 결과 추가, count 증가
    │   ├─ 반품 (delta > 0): 무게로 기존 상품 찾아 count 감소
    │   └─ 미매칭 반품: UnmatchedReturn으로 기록
    │
    ├─ 교차존 반품 처리 (Cross-Zone Return)
    │   ├─ 미매칭 반품 → 다른 zone의 active session에서 무게 매칭
    │   └─ 매칭 성공 시: 대상 zone의 count 감소 + 기록
    │
    ├─ YAML 영속화 (background ThreadPoolExecutor)
    │
    └─ Node.js 폴링 응답
        │
        ├─ POST /api/judge/multi-zone (10초 간격)
        │
        ├─ OPEN 신호 → GlobalDoorSession 시작
        │   └─ zones 1-5 interim 응답
        │
        ├─ CLOSE 신호 → pending_close 전환
        │   ├─ 초기 대기: 20초 (마지막 trigger 이후)
        │   └─ 이후 대기: 5초 (큐 대기 trigger 확인)
        │
        └─ 완료 시 최종 응답
            ├─ zones: [{zone, products, totalPrice}]
            ├─ totalPrice, productCount
            └─ globalSessionInfo
```

### Node.js 응답 구조

```json
{
  "zones": [
    {
      "zone": 1,
      "products": [
        {
          "product_id": "P001",
          "name": "콜라",
          "count": 2,
          "unit_price": 1500,
          "total_price": 3000
        }
      ],
      "totalPrice": 3000
    }
  ],
  "totalPrice": 3000,
  "productCount": 2,
  "globalSessionInfo": {
    "session_id": "uuid",
    "status": "complete"
  }
}
```

---

## 핵심 설정값

| 설정 | 값 | 설명 | 환경변수 |
|------|----|------|----------|
| 입력 크기 | 480x480 | YOLO 입력 해상도 | - |
| FP16 | true | 반정밀도 추론 | - |
| conf threshold | 0.01 | YOLO 최소 신뢰도 (필터링 전) | - |
| max_det | 20 | 프레임당 최대 탐지 수 | - |
| 무게 허용오차 | ±3.0g | StrictWeightMatcher 기본값 | `MODEL__WEIGHT__TOLERANCE_GRAMS` |
| 저무게 스킵 | 5g | delta_weight 미만 시 YOLO 생략 | - |
| 캐시 청소 | 100회 | GPU empty_cache 주기 | - |
| 세션 TTL | 300초 | DoorSession 만료 시간 | `MODEL__BUFFER__TTL_SECONDS` |
| 폴링 주기 | 10초 | Node.js → Model 폴링 간격 | - |
| pending_close 초기 | 20초 | CLOSE 후 최종화 대기 | - |
| pending_close 이후 | 5초 | 추가 trigger 대기 | - |

## 신뢰도(Confidence) 계산 공식 3가지

### 1. Voting Ensemble: weighted_confidence

Top/Side 카메라 투표 결합 시 사용:

```
weighted_confidence = top_conf * 0.5
                    + side_conf * 0.5
                    + min(top_conf, side_conf) * 0.2

top_conf  = 해당 클래스의 Top 카메라 max_confidence
side_conf = 해당 클래스의 Side 카메라 max_confidence
클램핑: min(result, 1.0)
```

### 2. StrictWeightMatcher: match_score

무게 기반 조합 평가 시 사용:

```
match_score = weight_score * 0.6
            + vision_score * 0.3
            + simplicity_score * 0.1

weight_score     = max(0, 1 - weight_error / tolerance)
vision_score     = avg_vision_confidence (개수 가중 평균)
simplicity_score = max(0, 1 - (상품종류수 - 1) * 0.2)
```

### 3. Decision Engine: fusion_confidence

최종 판단 신뢰도:

```
fusion_confidence = vision_conf * 0.4
                  + weight_conf * 0.5
                  + count_score * 0.1

vision_conf = 비전 기반 신뢰도
weight_conf = 무게 매칭 신뢰도
count_score = 개수 기반 점수 (≤3: 1.0, 그 이상: 1.0 - (count-3)*0.1)
```

---

## 전체 시퀀스 다이어그램

```
Camera        Model Service                Node.js
  │              │                            │
  │   OPEN 신호  │  POST /multi-zone (OPEN)   │
  │              │◄───────────────────────────│
  │              │  GlobalSession 시작         │
  │              │───────────────────────────►│ interim 응답
  │              │                            │
  │  Trigger #1  │                            │
  ├─────────────►│                            │
  │              ├─ 큐 등록                    │
  │  202 queued  │                            │
  │◄─────────────┤                            │
  │              ├─ 프레임 추출               │
  │              ├─ YOLO 추론                 │
  │              ├─ 필터링                    │
  │              ├─ Voting                    │
  │              ├─ 판단 엔진                 │
  │              ├─ DoorSession 저장          │
  │              │                            │
  │  Trigger #2  │                            │
  ├─────────────►│                            │
  │              ├─ (같은 파이프라인)          │
  │              ├─ DoorSession 통합          │
  │              │                            │
  │              │  POST /multi-zone (폴링)   │
  │              │◄───────────────────────────│
  │              │  "아직 처리중" 응답         │
  │              │───────────────────────────►│
  │              │                            │
  │  Trigger #N  │                            │
  ├─────────────►│                            │
  │              ├─ (같은 파이프라인)          │
  │              │                            │
  │              │  POST /multi-zone (CLOSE)  │
  │              │◄───────────────────────────│
  │              ├─ pending_close 전환         │
  │              │  (20초 대기)               │
  │              │                            │
  │              │  POST /multi-zone (폴링)   │
  │              │◄───────────────────────────│
  │              ├─ 최종화 확인               │
  │              ├─ 교차존 반품 처리           │
  │              ├─ 최종 결과 응답             │
  │              │───────────────────────────►│ 결제 진행
  │              │                            │
```

---

## 참조 파일 목록

| 파일 | 경로 (model_service/) | 역할 |
|------|----------------------|------|
| Trigger API | `api/routes/trigger.py` | Trigger 엔드포인트, 무게 계산 |
| Trigger Service | `service/trigger_service.py` | 큐 처리, 워커 루프 |
| Frame Extractor | `video/frame_extractor.py` | FFmpeg 프레임 추출 |
| Video Processor | `video/video_processor.py` | 영상 처리 오케스트레이션 |
| YOLO Wrapper | `vision/yolo_wrapper.py` | YOLO TensorRT 추론 |
| Hand Path Tracker | `vision/hand_path_tracker.py` | 손 경로 필터링 |
| Voting Ensemble | `video/voting_ensemble.py` | 투표 앙상블 |
| Decision Engine | `engine/decision_engine.py` | 판단 엔진 |
| Weight Matcher | `weight/strict_weight_matcher.py` | 무게 매칭 |
| Door Session | `session/door_session.py` | DoorSession 모델 |
| Door Session Store | `session/door_session_store.py` | Door Session 관리 |
| Product Aggregator | `session/product_aggregator.py` | 상품 집계 |
| Global Session | `session/global_door_session.py` | GlobalSession 모델 |
| Multi-Zone API | `api/routes/multi_zone.py` | Node.js 폴링 API |

---

> 상세 분석 문서: [PRODUCT_DETECTION_DETAIL.md](./PRODUCT_DETECTION_DETAIL.md) (3~7단계 심층 분석)
