"""API route exports with lazy loading."""

from importlib import import_module

_EXPORTS = {
    "health_router": ("model_service.api.routes.health", "router"),
    "trigger_router": ("model_service.api.routes.trigger", "router"),
    "multi_zone_router": ("model_service.api.routes.multi_zone", "router"),
}

__all__ = [
    "health_router",
    "trigger_router",
    "multi_zone_router",
]


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
