"""Health check routes."""

import time

from fastapi import APIRouter, Request
from model_service.api.deps import get_status, is_initialized
from model_service.core.config import Settings, config
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api", tags=["health"])


class HealthResponse(BaseModel):
    model: str
    status: str = "ok"
    yolo_loaded: bool = False
    session_store_ready: bool = False
    timestamp: float = 0.0


class DetailedHealthResponse(BaseModel):
    service: str = "model"
    version: str = "5.4.0"
    status: str = "ok"
    dependencies: dict = Field(default_factory=dict)
    config: dict = Field(default_factory=dict)
    timestamp: float = 0.0


def _get_runtime_settings(request: Request) -> Settings:
    runtime_settings = getattr(request.app.state, "settings", None)
    if isinstance(runtime_settings, Settings):
        return runtime_settings
    return config


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    _ = _get_runtime_settings(request)
    deps_status = get_status()
    yolo_instance = deps_status.get("yolo_instance")
    yolo_loaded = yolo_instance.is_loaded if yolo_instance else False

    return HealthResponse(
        model="HEALTHY" if yolo_loaded else "UNHEALTHY",
        status="ok" if is_initialized() else "initializing",
        yolo_loaded=yolo_loaded,
        session_store_ready=deps_status.get("session_store", False),
        timestamp=time.time(),
    )


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check(request: Request):
    runtime_settings = _get_runtime_settings(request)
    deps_status = get_status()
    serializable_deps = {
        "initialized": deps_status.get("initialized", False),
        "session_store": deps_status.get("session_store", False),
        "yolo": deps_status.get("yolo", False),
        "yolo_loaded": deps_status.get("yolo_loaded", False),
        "engine": deps_status.get("engine", False),
        "video_processor": deps_status.get("video_processor", False),
        "door_session_store": deps_status.get("door_session_store", False),
        "active_product_store": deps_status.get("active_product_store", False),
        "trigger_service": deps_status.get("trigger_service", False),
    }

    return DetailedHealthResponse(
        service="model",
        version="5.4.0",
        status="ok" if is_initialized() else "initializing",
        dependencies=serializable_deps,
        config={
            "host": runtime_settings.host,
            "port": runtime_settings.port,
            "yolo_model_path": runtime_settings.yolo_model_path,
        },
        timestamp=time.time(),
    )
