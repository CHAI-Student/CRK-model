"""Service container for dependency injection."""

import logging
import threading
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from model_service.engine import ProductDecisionEngine
    from model_service.service.trigger_service import TriggerService
    from model_service.session import DoorSessionStore, SessionStore
    from model_service.session.active_product_store import ActiveProductStore
    from model_service.video import VideoProcessor
    from model_service.vision import YOLOWrapper

logger = logging.getLogger(__name__)


class ServiceContainer:
    """Explicit dependency container used by FastAPI and tests."""

    def __init__(self):
        self._session_store: Optional["SessionStore"] = None
        self._door_session_store: Optional["DoorSessionStore"] = None
        self._active_product_store: Optional["ActiveProductStore"] = None
        self._yolo: Optional["YOLOWrapper"] = None
        self._engine: Optional["ProductDecisionEngine"] = None
        self._video_processor: Optional["VideoProcessor"] = None
        self._trigger_service: Optional["TriggerService"] = None
        self._initialized: bool = False

    def init(
        self,
        session_store: "SessionStore",
        yolo: "YOLOWrapper",
        engine: "ProductDecisionEngine",
        video_processor: Optional["VideoProcessor"] = None,
        door_session_store: Optional["DoorSessionStore"] = None,
        active_product_store: Optional["ActiveProductStore"] = None,
    ) -> None:
        """Initialize the container with core runtime dependencies."""
        self._session_store = session_store
        self._yolo = yolo
        self._engine = engine
        self._door_session_store = door_session_store
        self._active_product_store = active_product_store
        self._video_processor = video_processor
        self._trigger_service = None
        self._initialized = True

        logger.info(
            "ServiceContainer initialized: "
            f"session_store={session_store is not None}, "
            f"yolo={yolo is not None}, "
            f"engine={engine is not None}, "
            f"video_processor={self._video_processor is not None}, "
            f"door_session_store={door_session_store is not None}, "
            f"active_product_store={active_product_store is not None}"
        )

    def cleanup(self) -> None:
        """Reset all managed instances."""
        if self._session_store is not None:
            self._session_store.clear_all()
        if self._door_session_store is not None:
            self._door_session_store.clear_all()
        if self._active_product_store is not None:
            self._active_product_store.clear()

        self._session_store = None
        self._door_session_store = None
        self._active_product_store = None
        self._yolo = None
        self._engine = None
        self._video_processor = None
        self._trigger_service = None
        self._initialized = False

        logger.info("ServiceContainer cleaned up")

    def _ensure_video_processor(self) -> Optional["VideoProcessor"]:
        if self._video_processor is None and self._yolo is not None:
            from model_service.video import VideoProcessor

            self._video_processor = VideoProcessor(yolo=self._yolo)
        return self._video_processor

    def _ensure_trigger_service(self) -> Optional["TriggerService"]:
        if self._trigger_service is None:
            video_processor = self._ensure_video_processor()
            if (
                video_processor is None
                or self._engine is None
                or self._session_store is None
            ):
                return None

            from model_service.service.trigger_service import TriggerService

            self._trigger_service = TriggerService(
                video_processor=video_processor,
                engine=self._engine,
                session_store=self._session_store,
                door_session_store=self._door_session_store,
                active_product_store=self._active_product_store,
            )
        return self._trigger_service

    def get_session_store(self) -> "SessionStore":
        if self._session_store is None:
            raise RuntimeError("SessionStore not initialized. Call init() first.")
        return self._session_store

    def get_yolo(self) -> "YOLOWrapper":
        if self._yolo is None:
            raise RuntimeError("YOLOWrapper not initialized. Call init() first.")
        return self._yolo

    def get_decision_engine(self) -> "ProductDecisionEngine":
        if self._engine is None:
            raise RuntimeError("ProductDecisionEngine not initialized. Call init() first.")
        return self._engine

    def get_video_processor(self) -> "VideoProcessor":
        video_processor = self._ensure_video_processor()
        if video_processor is None:
            raise RuntimeError("VideoProcessor not initialized. Call init() first.")
        return video_processor

    def get_door_session_store(self) -> "DoorSessionStore":
        if self._door_session_store is None:
            raise RuntimeError("DoorSessionStore not initialized. Call init() first.")
        return self._door_session_store

    def get_active_product_store(self) -> "ActiveProductStore":
        if self._active_product_store is None:
            raise RuntimeError("ActiveProductStore not initialized. Call init() first.")
        return self._active_product_store

    def get_trigger_service(self) -> "TriggerService":
        trigger_service = self._ensure_trigger_service()
        if trigger_service is None:
            raise RuntimeError("TriggerService not initialized. Call init() first.")
        return trigger_service

    def get_session_store_optional(self) -> Optional["SessionStore"]:
        return self._session_store

    def get_yolo_optional(self) -> Optional["YOLOWrapper"]:
        return self._yolo

    def get_decision_engine_optional(self) -> Optional["ProductDecisionEngine"]:
        return self._engine

    def get_video_processor_optional(self) -> Optional["VideoProcessor"]:
        return self._ensure_video_processor()

    def get_door_session_store_optional(self) -> Optional["DoorSessionStore"]:
        return self._door_session_store

    def get_active_product_store_optional(self) -> Optional["ActiveProductStore"]:
        return self._active_product_store

    def get_trigger_service_optional(self) -> Optional["TriggerService"]:
        return self._ensure_trigger_service()

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def get_status(self) -> dict:
        yolo = self._yolo
        door_store = self._door_session_store
        return {
            "initialized": self._initialized,
            "session_store": self._session_store is not None,
            "yolo": yolo is not None,
            "yolo_instance": yolo,
            "yolo_loaded": yolo.is_loaded if yolo else False,
            "engine": self._engine is not None,
            "video_processor": self._video_processor is not None,
            "door_session_store": door_store is not None,
            "door_session_store_instance": door_store,
            "active_product_store": self._active_product_store is not None,
            "trigger_service": self._trigger_service is not None,
        }


_global_container: Optional[ServiceContainer] = None
_global_container_lock = threading.Lock()


def get_global_container() -> ServiceContainer:
    """Return the global container, creating it on first access."""
    global _global_container
    if _global_container is not None:
        return _global_container

    with _global_container_lock:
        if _global_container is None:
            _global_container = ServiceContainer()
            logger.debug("Global container created (thread-safe)")
        return _global_container


def set_global_container(container: ServiceContainer) -> None:
    """Replace the global container, mainly for tests."""
    global _global_container
    with _global_container_lock:
        _global_container = container


def reset_global_container() -> None:
    """Reset and clean up the global container."""
    global _global_container
    with _global_container_lock:
        if _global_container is not None:
            _global_container.cleanup()
        _global_container = None
