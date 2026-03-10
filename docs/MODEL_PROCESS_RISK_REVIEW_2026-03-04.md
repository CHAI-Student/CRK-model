# Edge Environment Model Process 리스크 점검 (2026-03-04)

## 검토 범위
- 문서: `CLAUDE.md`, `docs/PRODUCT_DETECTION_FLOW.md`, `docs/PRODUCT_DETECTION_DETAIL.md`, `docs/REFERENCE.md`
- 코드: `services/model/model_service` 전체 흐름
  - API: `api/routes/*.py`, `api/manager.py`
  - 서비스: `service/trigger_service.py`
  - 비전/비디오: `vision/*.py`, `video/*.py`
  - 판정/무게: `engine/*.py`, `weight/*.py`
  - 세션: `session/*.py`

## 전체 흐름(코드 기준)
1. `POST /trigger` 진입 → `TriggerService.enqueue_trigger()`에서 중복 체크/큐 등록/저무게 스킵 처리
2. 워커(`_worker_loop`)가 순차 처리 → `VideoProcessor.process_videos_async()` 또는 `process_videos()`
3. 프레임 추출(`StreamingFrameExtractor`) + YOLO 추론(`YOLOWrapper.detect`)
4. Motion/HandPath/ROI/Confidence 필터 → `VotingEnsemble.combine()`
5. `ProductDecisionEngine.judge()` + `StrictWeightMatcher`로 최종 판단
6. `SessionStore` 저장 + `DoorSessionStore.add_trigger_with_global()` 누적
7. Node.js가 `POST /api/judge/multi-zone`로 OPEN/CLOSE/폴링하며 최종 결과 수신

## 주요 리스크 (우선순위 순)

### 1. [Critical] YOLO 로드 실패 시 서비스가 정상 기동되어 오판/무검출로 흘러갈 수 있음
- 근거
  - `services/model/model_service/api/manager.py:146` `yolo_loaded = yolo.load()` 결과를 검증 없이 무시
  - `services/model/model_service/vision/yolo_wrapper.py:386-397` 로드 실패/모델 없음 시 `[]` 반환
  - `services/model/model_service/api/routes/health.py:65,73` 헬스의 `model`은 파일 존재만 보고 `"HEALTHY"` 판정
- 영향
  - 실제 추론 불능 상태에서도 서비스/헬스가 살아있는 것처럼 보임
  - 트리거가 명시적 에러 대신 빈 후보로 진행될 가능성
- 권장
  - startup에서 `yolo.load()==False`면 기동 실패 처리(즉시 fail-fast)
  - health의 healthy 기준을 `yolo_loaded` 중심으로 변경

### 2. [Critical] 세션 ID 충돌 가능성(초 단위 생성)
- 근거
  - `services/model/model_service/session/session_store.py:413`
  - `services/model/model_service/session/door_session.py:441`
  - `services/model/model_service/session/global_door_session.py:192`
  - 모두 `strftime('%y%m%d_%H%M%S')` 기반
- 영향
  - 같은 초에 동일 zone 트리거가 들어오면 세션 덮어쓰기/혼선 가능
  - YAML 파일명 충돌로 세션 이력 유실 가능
- 권장
  - 밀리초/UUID suffix 추가(예: `..._%f` + short uuid)

### 3. [High] 처리 실패가 `processing` 상태로 남아 Node.js 폴링이 장시간 지속될 수 있음
- 근거
  - 워커 에러 시 `update_stage(..., "error")`만 수행: `services/model/model_service/service/trigger_service.py:521-524`
  - `SessionData.status`는 사실상 `processing|complete`: `services/model/model_service/session/session_store.py:54,73`
  - 폴링 API는 `status=="processing"`이면 계속 처리중 응답: `services/model/model_service/api/routes/multi_zone.py:958-973`
- 영향
  - 실에러가 명시적으로 종료되지 않고 TTL까지 대기 상태 지속 가능
- 권장
  - `status="error"` 상태 추가 및 `multi-zone`에서 즉시 실패 응답 분기

### 4. [High] Dedup 키/등록 타이밍 때문에 실패 후 정상 재시도까지 중복으로 막힐 수 있음
- 근거
  - 키 구성: zone + video path만 사용 `services/model/model_service/service/trigger_service.py:160-165`
  - 큐 등록 직후 dedup 등록 `services/model/model_service/service/trigger_service.py:409`
  - 실패 시에도 같은 session으로 중복 판정될 여지
- 영향
  - 같은 파일 경로 재사용 환경에서 정상 재처리 지연/누락 가능
- 권장
  - 키에 파일 mtime/size/hash 일부 포함
  - dedup 등록 시점을 처리 성공 후로 조정하거나 실패 상태 별도 관리

### 5. [High] ActiveProductStore 정리 시점이 불명확(세션 간 상품 정보 잔존 위험)
- 근거
  - CLOSE 핸들러 인자만 있고 실제 미사용: `services/model/model_service/api/routes/multi_zone.py:537,539,872`
  - DoorSession finalize callback API는 존재하나 wiring 없음: `services/model/model_service/session/door_session_store.py:278` / `services/model/model_service/api/manager.py:176-182`
  - 실제 clear는 shutdown 시점: `services/model/model_service/api/manager.py:239`
- 영향
  - 다음 세션 초기에 이전 상품 필터가 남아 잘못된 class filtering 가능
- 권장
  - CLOSE finalize 성공 시 `active_product_store.clear()` 또는 finalize callback 연결

### 6. [High] Async 경로에서 이벤트 루프 블로킹 및 fallback 비호환 가능성
- 근거
  - `_probe_video`가 블로킹 `time.sleep` 사용: `services/model/model_service/video/frame_extractor.py:86,129`
  - 비동기 iterator 진입 시 `_probe_video` 호출: `services/model/model_service/video/frame_extractor.py:371`
  - async 처리 루프는 `async for frame in extractor`: `services/model/model_service/video/video_processor.py:436`
  - ffmpeg 미존재 시 `CV2FrameExtractor` 반환 가능: `services/model/model_service/video/frame_extractor.py:639` (비동기 iterator 미구현)
- 영향
  - async 모드에서도 루프 블로킹 발생 가능
  - ffmpeg 장애 시 async 경로에서 런타임 에러 가능
- 권장
  - probe를 비동기화(`asyncio.sleep`, `create_subprocess_exec`)
  - async 모드 진입 전 ffmpeg 필수 검증 또는 async fallback 구현

### 7. [Medium] CLOSE finalize 경합 시 `None` 처리 누락 가능
- 근거
  - 호출부는 반환값 null-check 없이 사용: `services/model/model_service/api/routes/multi_zone.py:641-650`
  - 피호출부는 `None` 반환 가능: `services/model/model_service/session/door_session_store.py:560-562`
- 영향
  - 동시 CLOSE/cleanup 상황에서 간헐적 500 가능
- 권장
  - `finalize_global_session()` 결과 null-check 추가

### 8. [Medium] 반품 처리에서 +delta가 커도 기본적으로 수량 1개만 차감
- 근거
  - `services/model/model_service/session/product_aggregator.py:191-193,216`
- 영향
  - 한 트리거에서 2개 이상 동시 반품 시 즉시 반영이 부정확할 수 있음
  - 이후 보정 로직 의존도가 커짐
- 권장
  - 반환 무게 기반 다중 개수 차감 로직 추가

### 9. [Medium] 로드셀 배열에서 첫 채널(`filtered_value[0]`)만 사용
- 근거
  - `services/model/model_service/service/trigger_service.py:1101-1102,1073-1074`
  - `services/model/model_service/api/routes/trigger.py:158-159,208-209`
- 영향
  - 다채널 loadcell에서 특정 채널 편차/드리프트에 취약
- 권장
  - 채널 평균/중앙값/유효채널 가중치 기반으로 delta 계산

## 문서-코드 불일치 (운영 리스크)

### A. API 계약 불일치
- 문서는 `GET /api/products`, `POST /api/products/sync` 제공으로 명시:
  - `docs/REFERENCE.md:311,332`
- 코드에서는 products router 제거:
  - `services/model/model_service/api/routes/__init__.py:5`
  - `services/model/model_service/api/__init__.py:11`

### B. CLOSE 최종 상태값 불일치
- 문서: 완료 상태 `"complete"` 예시
  - `docs/REFERENCE.md:173,177`
- 코드: `"success"` 또는 `"complete_no_products"`
  - `services/model/model_service/api/routes/multi_zone.py:665`

### C. Health 응답 필드 불일치
- 문서: `door_session_store_ready` 포함
  - `docs/REFERENCE.md:22,38`
- 코드 `HealthResponse`에는 해당 필드 없음
  - `services/model/model_service/api/routes/health.py:20-27`

### D. 버전/기본 모델 경로 불일치
- 문서 버전: 5.4.0
  - `CLAUDE.md:6,16`
- 코드/패키지 버전: 5.3.0
  - `services/model/model_service/__init__.py:15`
  - `pyproject.toml:7`
- 기본 모델 경로
  - 문서: `models/siyeon_best.engine`
  - 코드: `models/siyeon_best.engine` (resolved on 2026-03-09)

## 결론
- 현재 파이프라인은 구조적으로 잘 분리되어 있으나, 운영 관점에서 가장 위험한 지점은 `초기화/실패 상태 전파`와 `세션 ID/상태 관리`다.
- 특히 `YOLO 로드 실패 시 정상 서비스처럼 보이는 문제`와 `에러가 processing으로 남는 문제`는 현장 장애를 늦게 발견하게 만드는 핵심 리스크다.
