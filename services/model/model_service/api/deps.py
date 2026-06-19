"""API dependencies for dependency injection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from model_service.container import ServiceContainer
from model_service.container.service_container import (
    get_global_container,
    reset_global_container,
    set_global_container,
)

if TYPE_CHECKING:
    from model_service.engine import ProductDecisionEngine
    from model_service.service.trigger_service import TriggerService
    from model_service.session import DoorSessionStore, SessionStore
    from model_service.session.active_product_store import ActiveProductStore
    from model_service.video import VideoProcessor
    from model_service.vision import YOLOWrapper

logger = logging.getLogger(__name__)


def init_dependencies(
    session_store: SessionStore,
    yolo: YOLOWrapper,
    engine: ProductDecisionEngine,
    video_processor: Optional[VideoProcessor] = None,
    door_session_store: Optional[DoorSessionStore] = None,
    active_product_store: Optional[ActiveProductStore] = None,
) -> ServiceContainer:
    """Initialize the global dependency container."""
    container = get_global_container()
    container.init(
        session_store=session_store,
        yolo=yolo,
        engine=engine,
        video_processor=video_processor,
        door_session_store=door_session_store,
        active_product_store=active_product_store,
    )
    return container


def cleanup_dependencies() -> None:
    """Reset the global dependency container."""
    reset_global_container()


def get_session_store() -> SessionStore:
    return get_global_container().get_session_store()


def get_yolo() -> YOLOWrapper:
    return get_global_container().get_yolo()


def get_decision_engine() -> ProductDecisionEngine:
    return get_global_container().get_decision_engine()


def get_video_processor() -> VideoProcessor:
    return get_global_container().get_video_processor()


def get_door_session_store() -> DoorSessionStore:
    return get_global_container().get_door_session_store()


def get_active_product_store() -> ActiveProductStore:
    return get_global_container().get_active_product_store()


def get_trigger_service() -> TriggerService:
    return get_global_container().get_trigger_service()


def get_session_store_optional() -> Optional[SessionStore]:
    return get_global_container().get_session_store_optional()


def get_yolo_optional() -> Optional[YOLOWrapper]:
    return get_global_container().get_yolo_optional()


def get_decision_engine_optional() -> Optional[ProductDecisionEngine]:
    return get_global_container().get_decision_engine_optional()


def get_video_processor_optional() -> Optional[VideoProcessor]:
    return get_global_container().get_video_processor_optional()


def get_door_session_store_optional() -> Optional[DoorSessionStore]:
    return get_global_container().get_door_session_store_optional()


def get_active_product_store_optional() -> Optional[ActiveProductStore]:
    return get_global_container().get_active_product_store_optional()


def get_trigger_service_optional() -> Optional[TriggerService]:
    return get_global_container().get_trigger_service_optional()


def is_initialized() -> bool:
    return get_global_container().is_initialized


def get_status() -> dict:
    return get_global_container().get_status()


def create_test_container() -> ServiceContainer:
    return ServiceContainer()


def use_test_container(container: ServiceContainer) -> None:
    set_global_container(container)
