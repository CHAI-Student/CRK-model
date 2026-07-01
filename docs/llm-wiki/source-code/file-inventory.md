# Source Code Map: File Inventory

Source: `services/model/model_service/**/*.py`, `services/model/tests`,
`scripts`, and repo root operational files.

Status: current complete file map

## Runtime Package Files

### Package Root

- [services/model/model_service/__init__.py](../../../services/model/model_service/__init__.py):
  package metadata/export surface.
- [services/model/model_service/__main__.py](../../../services/model/model_service/__main__.py):
  module execution shim.
- [services/model/model_service/main.py](../../../services/model/model_service/main.py):
  console entry, runtime bootstrap, settings load, Uvicorn start.

### API

- [api/__init__.py](../../../services/model/model_service/api/__init__.py):
  lazy API exports.
- [api/deps.py](../../../services/model/model_service/api/deps.py):
  FastAPI dependency accessors over the global `ServiceContainer`.
- [api/manager.py](../../../services/model/model_service/api/manager.py):
  app factory, lifespan, dependency initialization, shutdown.
- [api/routes/__init__.py](../../../services/model/model_service/api/routes/__init__.py):
  lazy router exports.
- [api/routes/health.py](../../../services/model/model_service/api/routes/health.py):
  `/api/health` and `/api/health/detailed`.
- [api/routes/trigger.py](../../../services/model/model_service/api/routes/trigger.py):
  `/trigger`, trigger fallback path, trace creation, route-level helper logic.
- [api/routes/multi_zone.py](../../../services/model/model_service/api/routes/multi_zone.py):
  `/api/judge/*`, Node OPEN/CLOSE/polling, product snapshot intake.

### Container And Core

- [container/__init__.py](../../../services/model/model_service/container/__init__.py):
  container exports.
- [container/service_container.py](../../../services/model/model_service/container/service_container.py):
  explicit DI container and lazy `VideoProcessor`/`TriggerService` creation.
- [core/__init__.py](../../../services/model/model_service/core/__init__.py):
  core package marker.
- [core/config.py](../../../services/model/model_service/core/config.py):
  `Settings` and all Pydantic env groups.
- [core/exceptions.py](../../../services/model/model_service/core/exceptions.py):
  typed service/video/YOLO/session/loadcell exceptions.
- [core/loadcell_stats.py](../../../services/model/model_service/core/loadcell_stats.py):
  parsing, channel math, stable-window analysis, peak filtering.
- [core/logging_config.py](../../../services/model/model_service/core/logging_config.py):
  structured logging, ops logger, correlation ids, performance logger.
- [core/model_validation.py](../../../services/model/model_service/core/model_validation.py):
  engine/dataset/mapping class validation.
- [core/runtime_env.py](../../../services/model/model_service/core/runtime_env.py):
  Jetson runtime environment detection and re-exec bootstrap.
- [core/terminal.py](../../../services/model/model_service/core/terminal.py):
  terminal state capture/restore around entrypoint shutdown.

### Database And Engine

- [database/__init__.py](../../../services/model/model_service/database/__init__.py):
  reserved package marker.
- [engine/__init__.py](../../../services/model/model_service/engine/__init__.py):
  engine exports.
- [engine/models.py](../../../services/model/model_service/engine/models.py):
  `JudgmentStatus`, `EnsembleResult`, `CountEstimate`,
  `ProductJudgment`, `JudgmentResult`, engine `ProductInfo`.
- [engine/decision_engine.py](../../../services/model/model_service/engine/decision_engine.py):
  final product decision flow, strict/relaxed matching, fallback logic.

### Service

- [service/__init__.py](../../../services/model/model_service/service/__init__.py):
  service package marker.
- [service/trigger_service.py](../../../services/model/model_service/service/trigger_service.py):
  queued trigger worker, dedup, video processing orchestration, storage.
- [service/judgment_service.py](../../../services/model/model_service/service/judgment_service.py):
  service-level judgment DTOs and wrapper service.
- [service/door_session_service.py](../../../services/model/model_service/service/door_session_service.py):
  service-level door-session DTOs and wrapper service.

### Session

- [session/__init__.py](../../../services/model/model_service/session/__init__.py):
  session exports.
- [session/active_product_store.py](../../../services/model/model_service/session/active_product_store.py):
  Node product snapshot normalization, YOLO mapping, weight lookup.
- [session/session_store.py](../../../services/model/model_service/session/session_store.py):
  in-memory trigger session store and session id generation.
- [session/door_session.py](../../../services/model/model_service/session/door_session.py):
  `TriggerResult`, `DoorSession`, aggregated products, returns models.
- [session/global_door_session.py](../../../services/model/model_service/session/global_door_session.py):
  multi-zone global door-session container.
- [session/door_session_store.py](../../../services/model/model_service/session/door_session_store.py):
  active door/global session lifecycle, close handling, recovery.
- [session/product_aggregator.py](../../../services/model/model_service/session/product_aggregator.py):
  removal/return aggregation and net-delta repair support.
- [session/yaml_persistence.py](../../../services/model/model_service/session/yaml_persistence.py):
  YAML persistence for door sessions.

### Video, Vision, And Weight

- [video/__init__.py](../../../services/model/model_service/video/__init__.py):
  lazy video exports.
- [video/frame_extractor.py](../../../services/model/model_service/video/frame_extractor.py):
  FFmpeg/CV2 frame extraction, diagnostics, retry paths.
- [video/frame_trace.py](../../../services/model/model_service/video/frame_trace.py):
  per-trigger JSONL/detail trace and optional sample export.
- [video/freezer_candidate_policy.py](../../../services/model/model_service/video/freezer_candidate_policy.py):
  shared freezer repeat/count fit helpers used by video handled filtering and
  decision-engine repeat correction.
- [video/video_processor.py](../../../services/model/model_service/video/video_processor.py):
  sync/async video processing, filters, rescue candidates, frame stride.
- [video/voting_ensemble.py](../../../services/model/model_service/video/voting_ensemble.py):
  Top/Side vote accumulation and candidate ranking.
- [vision/__init__.py](../../../services/model/model_service/vision/__init__.py):
  vision exports.
- [vision/yolo_wrapper.py](../../../services/model/model_service/vision/yolo_wrapper.py):
  TensorRT YOLO wrapper and geometry handling.
- [vision/hand_path_tracker.py](../../../services/model/model_service/vision/hand_path_tracker.py):
  hand trajectory and product-path interaction logic.
- [vision/hand_filter.py](../../../services/model/model_service/vision/hand_filter.py):
  hand proximity filtering helper.
- [vision/top5_extractor.py](../../../services/model/model_service/vision/top5_extractor.py):
  top-k extraction utility.
- [weight/__init__.py](../../../services/model/model_service/weight/__init__.py):
  weight exports.
- [weight/count_calculator.py](../../../services/model/model_service/weight/count_calculator.py):
  weight-based count estimation for relaxed matching.
- [weight/strict_weight_matcher.py](../../../services/model/model_service/weight/strict_weight_matcher.py):
  strict subset-sum style product/weight matcher.

## Tests

See [Tests map](tests-map.md) for behavioral grouping. The test folder also
contains [conftest.py](../../../services/model/tests/conftest.py) and
[README.md](../../../services/model/tests/README.md). Scenario coverage uses
[test_scenario_matrix_contract.py](../../../services/model/tests/test_scenario_matrix_contract.py)
and the generated
[scenario_matrix.json](../../../services/model/tests/fixtures/scenario_matrix.json)
fixture. Compound same-zone loadcell coverage lives in
[test_loadcell_compound_segments.py](../../../services/model/tests/test_loadcell_compound_segments.py).

## Scripts And Root Files

- [README.md](../../../README.md)
- [pyproject.toml](../../../pyproject.toml)
- [.env.example](../../../.env.example)
- [dataset.yaml](../../../dataset.yaml)
- [scripts/setup_jetson.sh](../../../scripts/setup_jetson.sh)
- [scripts/install_model_service_launcher.sh](../../../scripts/install_model_service_launcher.sh)
- [scripts/install_jetson_torch.sh](../../../scripts/install_jetson_torch.sh)
- [scripts/jetson_env.sh](../../../scripts/jetson_env.sh)
- [scripts/live_engine_preview.py](../../../scripts/live_engine_preview.py)
- [scripts/convert_engine.sh](../../../scripts/convert_engine.sh)
- [scripts/refresh_scenario_fixture.py](../../../scripts/refresh_scenario_fixture.py)
- [scripts/verify_scenario_readiness.py](../../../scripts/verify_scenario_readiness.py)

## Related Wiki Pages

- [Startup and DI](startup-and-di.md)
- [API routes](api-routes.md)
- [Configuration](configuration.md)
- [Tests map](tests-map.md)
- [Scenario readiness and 0g diagnostics](../synthesis/scenario-readiness-and-0g.md)
