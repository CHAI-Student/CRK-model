"""FastAPI application manager for the model service."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from fastapi import FastAPI
from model_service.core.config import Settings
from model_service.core.logging_config import get_logger
from uvicorn import Config as UvicornConfig
from uvicorn import Server

if TYPE_CHECKING:
    from model_service.session import DoorSessionStore, SessionStore

logger = get_logger(__name__)


def maybe_validate_static_catalog(
    *,
    settings: Settings,
    engine_class_names: dict[Any, Any],
    base_dir: Path,
    validate_fn: Callable[..., Any] | None = None,
) -> Any | None:
    """Run legacy static dataset/mapping validation only when explicitly enabled."""
    logger.info(
        "YOLO catalog startup: "
        f"source_policy={settings.catalog.source_policy}, "
        f"static_validation_enabled={settings.catalog.static_validation_enabled}, "
        f"engine_classes={len(engine_class_names or {})}"
    )

    if not settings.catalog.static_validation_enabled:
        logger.info(
            "YOLO static catalog validation skipped: "
            "MODEL__CATALOG__STATIC_VALIDATION_ENABLED=false"
        )
        return None

    if validate_fn is None:
        from model_service.core.model_validation import validate_model_class_mapping

        validate_fn = validate_model_class_mapping

    mapping_path = base_dir / "config" / "yolo_product_mapping.json"
    dataset_path = base_dir.parent / "dataset.yaml"
    return validate_fn(
        engine_class_names=engine_class_names,
        dataset_path=dataset_path,
        mapping_path=mapping_path,
    )


async def session_cleanup_task(
    session_store: SessionStore,
    door_session_store: Optional[DoorSessionStore] = None,
    interval_seconds: float = 60.0,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Periodically clean expired in-memory and door sessions."""
    logger.info(f"Session cleanup task started (interval={interval_seconds}s)")

    sleep_chunk = 5.0
    elapsed_since_cleanup = 0.0

    while True:
        try:
            if stop_event is not None and stop_event.is_set():
                logger.info("Session cleanup task stopping (stop event set)")
                break

            await asyncio.sleep(sleep_chunk)
            elapsed_since_cleanup += sleep_chunk

            if stop_event is not None and stop_event.is_set():
                logger.info("Session cleanup task stopping (stop event set after sleep)")
                break

            if elapsed_since_cleanup >= interval_seconds:
                elapsed_since_cleanup = 0.0

                cleaned = session_store.cleanup_expired()
                if cleaned > 0:
                    logger.info(f"Session cleanup: removed {cleaned} expired sessions")

                if door_session_store is not None:
                    door_cleaned = door_session_store.cleanup_timed_out_sessions()
                    if door_cleaned > 0:
                        logger.info(f"Door session cleanup: removed {door_cleaned} timed out sessions")

        except asyncio.CancelledError:
            logger.info("Session cleanup task cancelled")
            break
        except Exception as exc:
            logger.error(f"Session cleanup task error: {exc}", exc_info=True)
            elapsed_since_cleanup = 0.0

    logger.info("Session cleanup task stopped")


def create_lifespan(settings: Settings):
    """Create the FastAPI lifespan handler bound to runtime settings."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from model_service.api.deps import cleanup_dependencies, init_dependencies
        from model_service.api.routes.multi_zone import shutdown_log_executor
        from model_service.container.service_container import get_global_container
        from model_service.engine import ProductDecisionEngine
        from model_service.session import DoorSessionStore, SessionStore
        from model_service.session.active_product_store import ActiveProductStore
        from model_service.vision import YOLOWrapper

        logger.info("Model service starting...")

        session_store = SessionStore(
            ttl_seconds=settings.buffer.ttl_seconds,
            max_sessions=settings.buffer.max_sessions,
        )

        yolo = YOLOWrapper(model_path=settings.yolo_model_path)
        if not yolo.load():
            details = getattr(yolo, "last_error", None)
            detail_suffix = f" Root cause: {details}" if details else ""
            raise RuntimeError(
                f"YOLO model load failed: {settings.yolo_model_path}. "
                f"Check that the TensorRT engine exists and is readable.{detail_suffix}"
            )

        yolo_name_to_id = {}
        for class_id, class_name in (yolo.class_names or {}).items():
            if class_id > 0:
                yolo_name_to_id[class_name] = class_id

        active_product_store = ActiveProductStore(
            yolo_name_to_id=yolo_name_to_id,
            source_policy=settings.catalog.source_policy,
        )

        base_dir = Path(__file__).parent.parent.parent.parent
        mapping_path = base_dir / "config" / "yolo_product_mapping.json"
        if mapping_path.exists():
            import json

            try:
                with open(mapping_path, "r", encoding="utf-8") as handle:
                    mapping_data = json.load(handle)
                loaded = active_product_store.load_yolo_mapping(mapping_data)
                logger.info(f"ActiveProductStore: loaded {loaded} YOLO mappings")
            except Exception as exc:
                logger.error(f"Failed to load YOLO mapping for ActiveProductStore: {exc}")

        validation = maybe_validate_static_catalog(
            settings=settings,
            engine_class_names=yolo.class_names or {},
            base_dir=base_dir,
        )
        if validation is not None and not validation.ok:
            logger.warning(
                "YOLO class mapping validation found issues: "
                f"mismatches={len(validation.mismatches)}, "
                f"missing_in_engine={validation.missing_in_engine[:10]}, "
                f"missing_in_dataset={validation.missing_in_dataset[:10]}, "
                f"missing_in_mapping={validation.missing_in_mapping[:10]}"
            )

        decision_engine = ProductDecisionEngine(product_db=active_product_store)

        door_session_store = None
        if settings.door_session.enabled:
            door_session_store = DoorSessionStore(
                yaml_dir=settings.door_session.yaml_dir,
                session_timeout=settings.door_session.session_timeout_seconds,
                weight_tolerance=settings.door_session.weight_tolerance_grams,
                max_duration=settings.door_session.max_duration_seconds,
                get_product_weight=lambda pid: active_product_store.get_product_weight(pid),
            )
            recovered = door_session_store.recover_active_sessions()
            logger.info(f"DoorSessionStore: recovered {recovered} active sessions")

        init_dependencies(
            session_store=session_store,
            yolo=yolo,
            engine=decision_engine,
            door_session_store=door_session_store,
            active_product_store=active_product_store,
        )

        trigger_service = get_global_container().get_trigger_service()
        await trigger_service.start_worker()

        cleanup_stop_event = asyncio.Event()
        cleanup_interval = settings.buffer.cleanup_interval_seconds
        cleanup_task = asyncio.create_task(
            session_cleanup_task(
                session_store=session_store,
                door_session_store=door_session_store,
                interval_seconds=cleanup_interval,
                stop_event=cleanup_stop_event,
            )
        )

        logger.info(f"Model service ready on port {settings.port}")
        logger.info(
            f"SessionStore: ttl={settings.buffer.ttl_seconds}s, "
            f"max_sessions={settings.buffer.max_sessions}, "
            f"cleanup_interval={cleanup_interval}s"
        )
        if settings.door_session.enabled:
            logger.info(
                f"DoorSessionStore: timeout={settings.door_session.session_timeout_seconds}s, "
                f"tolerance={settings.door_session.weight_tolerance_grams}g, "
                f"max_duration={settings.door_session.max_duration_seconds}s"
            )

        yield

        logger.info("Model service shutting down...")

        trigger_service = get_global_container().get_trigger_service_optional()
        if trigger_service is not None:
            await trigger_service.stop_worker()

        cleanup_stop_event.set()
        cleanup_task.cancel()
        active_product_store.clear_all()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

        if door_session_store is not None:
            try:
                days_to_keep = settings.door_session.yaml_retention_days
                deleted = door_session_store._persistence.cleanup_old_sessions(
                    days_to_keep=days_to_keep
                )
                if deleted > 0:
                    logger.info(
                        f"YAML cleanup: removed {deleted} old session files "
                        f"(older than {days_to_keep} days)"
                    )
            except Exception as exc:
                logger.error(f"YAML cleanup failed: {exc}")

            door_session_store.shutdown()

        shutdown_log_executor()
        cleanup_dependencies()
        logger.info("Model service stopped")

    return lifespan


def create_app(settings: Settings) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Model Service",
        description="AI product judgment service - AVI trigger + vision + weight fusion",
        version="5.4.0",
        lifespan=create_lifespan(settings),
    )
    app.state.settings = settings

    from model_service.api.routes import health_router, multi_zone_router, trigger_router

    app.include_router(health_router)
    app.include_router(trigger_router)
    app.include_router(multi_zone_router)

    @app.get("/")
    async def root():
        return {
            "service": "model",
            "version": "5.4.0",
            "description": "AI product judgment service (AVI trigger API)",
        }

    return app


class GracefulShutdownServer(Server):
    """Uvicorn server subclass that exposes a stop event."""

    def __init__(self, config: UvicornConfig, stop_event: asyncio.Event | None = None):
        super().__init__(config)
        self.stop_event = stop_event or asyncio.Event()

    async def shutdown(self, *args, **kwargs) -> None:
        self.stop_event.set()
        await super().shutdown(*args, **kwargs)


async def serve_api(settings: Settings) -> None:
    """Start the FastAPI server with graceful shutdown support."""
    app = create_app(settings)

    uvicorn_config = UvicornConfig(
        app=app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        timeout_graceful_shutdown=settings.api.timeout_graceful_shutdown,
    )
    server = GracefulShutdownServer(uvicorn_config)

    logger.info(f"Starting API server on {settings.host}:{settings.port}")
    await server.serve()
