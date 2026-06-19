"""API module for Model service."""

from importlib import import_module

_EXPORTS = {
    "health_router": ("model_service.api.routes", "health_router"),
    "trigger_router": ("model_service.api.routes", "trigger_router"),
    "multi_zone_router": ("model_service.api.routes", "multi_zone_router"),
    "init_dependencies": ("model_service.api.deps", "init_dependencies"),
    "cleanup_dependencies": ("model_service.api.deps", "cleanup_dependencies"),
    "get_session_store": ("model_service.api.deps", "get_session_store"),
    "get_yolo": ("model_service.api.deps", "get_yolo"),
    "get_decision_engine": ("model_service.api.deps", "get_decision_engine"),
    "get_video_processor": ("model_service.api.deps", "get_video_processor"),
    "create_app": ("model_service.api.manager", "create_app"),
    "serve_api": ("model_service.api.manager", "serve_api"),
}

__all__ = [
    "health_router",
    "trigger_router",
    "multi_zone_router",
    "init_dependencies",
    "cleanup_dependencies",
    "get_session_store",
    "get_yolo",
    "get_decision_engine",
    "get_video_processor",
    "create_app",
    "serve_api",
]


def __getattr__(name: str):
    """Resolve public API exports lazily to avoid heavy import chains."""
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
