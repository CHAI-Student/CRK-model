# Source Code Map: Startup And DI

Source: [main.py](../../../services/model/model_service/main.py),
[api/manager.py](../../../services/model/model_service/api/manager.py),
[container/service_container.py](../../../services/model/model_service/container/service_container.py),
[api/deps.py](../../../services/model/model_service/api/deps.py)

Status: current startup map

## Current Thesis

Startup is intentionally split into an early Jetson runtime bootstrap, late
FastAPI imports, explicit dependency construction in lifespan, and global
container accessors for routes/tests.

## Entry Flow

```text
model-service / python -m model_service
  -> model_service.main.run()
  -> install_terminal_restore()
  -> main()
  -> bootstrap_runtime_environment()
  -> import Settings, logging, serve_api
  -> Settings() loads .env
  -> serve_api(settings)
  -> create_app(settings)
  -> FastAPI lifespan initializes runtime dependencies
```

## Lifespan Initialization

`create_lifespan(settings)` builds the runtime graph:

- `SessionStore`
- `YOLOWrapper`, with fail-fast `yolo.load()` check
- `ActiveProductStore`, seeded with loaded engine name/id mapping and
  `settings.catalog.source_policy`
- Node-first catalog startup logging that records source policy, static
  validation enablement, and loaded engine class count
- optional legacy static class validation against engine names, `dataset.yaml`,
  and `config/yolo_product_mapping.json` only when
  `MODEL__CATALOG__STATIC_VALIDATION_ENABLED=true`
- `ProductDecisionEngine(product_db=active_product_store)`
- optional `DoorSessionStore`
- global dependencies through `init_dependencies(...)`
- lazy `TriggerService` from `ServiceContainer`, then `start_worker()`
- background `session_cleanup_task`

Shutdown stops the trigger worker, cancels cleanup, clears active products,
cleans old YAML sessions, shuts down the multi-zone log executor, and resets
dependencies.

## ServiceContainer Rules

- `ServiceContainer` is explicit and test-replaceable.
- `VideoProcessor` is lazily created from `YOLOWrapper`.
- `TriggerService` is lazily created from `VideoProcessor`,
  `ProductDecisionEngine`, `SessionStore`, optional `DoorSessionStore`, and
  optional `ActiveProductStore`.
- Required getters raise if the dependency is missing; optional getters return
  `None`.
- `get_status()` is used by health routes to report initialization and component
  readiness.

## Why This Matters

- Heavy YOLO/TensorRT imports are delayed until after Jetson runtime bootstrap.
- Startup fails if the TensorRT engine cannot be loaded.
- New TensorRT engines no longer emit stale `missing_in_engine` warnings from
  old static mapping files unless static catalog validation is explicitly
  enabled for compatibility audits.
- Tests can replace the global container without starting the service.
- The app stores runtime `Settings` on `app.state.settings`, so health details
  reflect actual runtime settings rather than stale import-time defaults.

## Related Wiki Pages

- [API routes](api-routes.md)
- [Configuration](configuration.md)
- [Jetson and testing](../synthesis/jetson-and-testing.md)
