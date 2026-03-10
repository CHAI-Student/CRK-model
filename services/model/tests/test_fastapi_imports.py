"""FastAPI import and runtime settings smoke tests."""

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_manager_import_smoke():
    from model_service.api.manager import create_app

    assert callable(create_app)


def test_route_import_smoke():
    from model_service.api.routes.multi_zone import router as multi_zone_router
    from model_service.api.routes.trigger import router as trigger_router

    assert multi_zone_router is not None
    assert trigger_router is not None


def test_video_processor_import_smoke():
    from model_service.video import VideoProcessor

    assert VideoProcessor is not None


def test_lazy_api_exports_smoke():
    from model_service.api import create_app
    from model_service.api.routes import multi_zone_router

    assert callable(create_app)
    assert multi_zone_router is not None


def test_create_app_stores_runtime_settings():
    from model_service.api.manager import create_app
    from model_service.core.config import Settings

    settings = Settings()
    settings.api.host = "127.0.0.1"
    settings.api.port = 9123
    settings.vision.yolo_model_path = "runtime-model.engine"

    app = create_app(settings)

    assert app.state.settings is settings
    assert app.state.settings.host == "127.0.0.1"
    assert app.state.settings.port == 9123
    assert app.state.settings.yolo_model_path == "runtime-model.engine"


def test_health_route_uses_runtime_settings():
    from model_service.api.routes.health import router
    from model_service.core.config import Settings

    settings = Settings()
    settings.api.host = "127.0.0.1"
    settings.api.port = 9123
    settings.vision.yolo_model_path = "runtime-model.engine"

    app = FastAPI()
    app.state.settings = settings
    app.include_router(router)

    client = TestClient(app)
    response = client.get("/api/health/detailed")

    assert response.status_code == 200
    payload = response.json()
    assert payload["config"]["host"] == "127.0.0.1"
    assert payload["config"]["port"] == 9123
    assert payload["config"]["yolo_model_path"] == "runtime-model.engine"


def test_detailed_health_serializes_initialized_dependencies(tmp_path):
    from model_service.api import deps
    from model_service.api.routes.health import router
    from model_service.container import ServiceContainer
    from model_service.core.config import Settings
    from model_service.session import DoorSessionStore, SessionStore

    container = ServiceContainer()
    mock_yolo = MagicMock()
    mock_yolo.is_loaded = True
    container.init(
        session_store=SessionStore(ttl_seconds=60, max_sessions=10),
        yolo=mock_yolo,
        engine=MagicMock(),
        door_session_store=DoorSessionStore(
            yaml_dir=str(tmp_path),
            session_timeout=5.0,
            weight_tolerance=3.0,
            max_duration=60.0,
        ),
    )
    deps.use_test_container(container)

    app = FastAPI()
    app.state.settings = Settings()
    app.include_router(router)

    client = TestClient(app)
    response = client.get("/api/health/detailed")

    assert response.status_code == 200
    payload = response.json()
    assert payload["dependencies"]["door_session_store"] is True
    assert payload["dependencies"]["yolo_loaded"] is True
    assert "door_session_store_instance" not in payload["dependencies"]


def test_video_processor_source_is_python310_compatible():
    project_root = Path(__file__).resolve().parents[3]
    video_processor_path = project_root / "services" / "model" / "model_service" / "video" / "video_processor.py"
    source = video_processor_path.read_text(encoding="utf-8")

    ast.parse(source, filename=str(video_processor_path), feature_version=(3, 10))


@pytest.mark.asyncio
async def test_lifespan_surfaces_yolo_root_cause(monkeypatch):
    from model_service.api.manager import create_lifespan
    from model_service.core.config import Settings
    import model_service.vision as vision

    settings = Settings()
    settings.vision.yolo_model_path = "models/0204_morning.engine"

    class FakeYOLO:
        def __init__(self, model_path):
            self.model_path = model_path
            self.last_error = (
                "PyTorch was installed without CUDA support. "
                "The default PyPI CPU wheel is active instead of a Jetson-compatible GPU build."
            )

        def load(self):
            return False

    monkeypatch.setattr(vision, "YOLOWrapper", FakeYOLO)

    app = FastAPI(lifespan=create_lifespan(settings))

    with pytest.raises(RuntimeError, match="PyTorch was installed without CUDA support"):
        async with app.router.lifespan_context(app):
            pass
