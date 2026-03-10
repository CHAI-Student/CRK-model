"""
Pytest configuration and fixtures (v5.3).

v5.3 변경사항:
- sys.path 조작 제거: model_service 패키지로 정상 import
- 절대 경로 import 사용 (model_service.*)

v4.2 변경사항:
- ServiceContainer fixture 추가
- 테스트 격리를 위한 컨테이너 리셋
"""

import sys
import tempfile
import time
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock

import pytest


try:
    import model_service  # noqa: F401
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[3]
    model_path = project_root / "services" / "model"
    model_path_str = str(model_path)
    if model_path_str not in sys.path:
        sys.path.insert(0, model_path_str)


@pytest.fixture(autouse=True)
def reset_global_container():
    """각 테스트 전후로 전역 컨테이너를 리셋 (v4.2)."""
    from model_service.container.service_container import reset_global_container
    reset_global_container()
    yield
    reset_global_container()


@pytest.fixture
def session_store():
    """Create a fresh SessionStore for testing."""
    from model_service.session import SessionStore
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    yield store
    store.clear_all()


@pytest.fixture
def sample_session_data():
    """Create sample SessionData for testing."""
    from model_service.session import SessionData, ProductResult

    products = [
        ProductResult(
            product_id=26,
            name="치킨마요",
            count=2,
            price=3500,
            confidence=0.85,
        ),
        ProductResult(
            product_id=15,
            name="참치마요",
            count=1,
            price=3000,
            confidence=0.78,
        ),
    ]

    return SessionData(
        session_id="zone_1_260201_143025",
        zone=1,
        products=products,
        total_price=10000,
        delta_weight=-730.5,
        status="complete",
        confidence=0.82,
        top_frames=150,
        side_frames=150,
        processing_time_ms=2500.3,
    )


@pytest.fixture
def sample_loadcells():
    """Create sample loadcell data for testing."""
    # Simulate loadcell readings over time
    # Start: stable around 5000g, End: stable around 4500g (removed ~500g)
    data = []

    # Start stable region (5000g)
    for i in range(10):
        data.append({
            "timestamp": f"2026-02-01T14:30:{i:02d}.000Z",
            "raw_value": [f"+{5000 + i}", f"+{5000 - i}"],
            "filtered_value": [f"+{5000 + i//2}", f"+{5000 - i//2}"],
            "filter_method": "none",
        })

    # Transition (weight changing)
    for i in range(5):
        val = 5000 - (i * 100)
        data.append({
            "timestamp": f"2026-02-01T14:30:{10+i:02d}.000Z",
            "raw_value": [f"+{val}", f"+{val}"],
            "filtered_value": [f"+{val}", f"+{val}"],
            "filter_method": "none",
        })

    # End stable region (4500g)
    for i in range(10):
        data.append({
            "timestamp": f"2026-02-01T14:30:{15+i:02d}.000Z",
            "raw_value": [f"+{4500 + i}", f"+{4500 - i}"],
            "filtered_value": [f"+{4500 + i//2}", f"+{4500 - i//2}"],
            "filter_method": "none",
        })

    return data


# ============================================================================
# Door Session Fixtures (v4.1)
# ============================================================================


@pytest.fixture
def door_session_store():
    """Create a DoorSessionStore for testing."""
    from model_service.session import DoorSessionStore

    # Use temp directory for YAML persistence
    temp_dir = tempfile.mkdtemp()
    store = DoorSessionStore(
        yaml_dir=temp_dir,
        session_timeout=5.0,  # Short timeout for testing
        weight_tolerance=3.0,
        max_duration=60.0,
    )
    yield store
    store.clear_all()


@pytest.fixture
def sample_trigger_result():
    """Create sample TriggerResult for testing."""
    from model_service.session import ProductResult
    from model_service.session.door_session import TriggerResult

    products = [
        ProductResult(
            product_id=26,
            product_idx="26",
            name="치킨마요",
            count=1,
            price=3500,
            confidence=0.92,
        )
    ]

    return TriggerResult(
        trigger_id="trigger_001",
        session_id="zone_1_260201_143025",
        timestamp=time.time(),
        products=products,
        delta_weight=-365.0,
        confidence=0.92,
        video_paths={"top": "/path/top.avi", "side": "/path/side.avi"},
        is_return=False,
        processing_time_ms=150.5,
    )


@pytest.fixture
def sample_return_trigger_result():
    """Create sample return TriggerResult for testing."""
    from model_service.session.door_session import TriggerResult

    return TriggerResult(
        trigger_id="trigger_002",
        session_id="zone_1_260201_143030",
        timestamp=time.time(),
        products=[],
        delta_weight=363.0,  # Close to 365g (within tolerance)
        confidence=0.0,
        video_paths={"top": "/path/top2.avi"},
        is_return=True,
        processing_time_ms=50.0,
    )


@pytest.fixture
def product_aggregator():
    """Create ProductAggregator for testing."""
    from model_service.session.product_aggregator import ProductAggregator

    def mock_get_weight(product_id: int) -> float:
        """Mock weight getter."""
        weights = {
            26: 365.0,   # 치킨마요
            15: 250.0,   # 참치마요
            10: 320.0,   # 김밥
        }
        return weights.get(product_id, 0.0)

    return ProductAggregator(
        weight_tolerance=3.0,
        get_product_weight=mock_get_weight,
    )


@pytest.fixture
def voting_ensemble():
    """Create VotingEnsemble for testing."""
    from model_service.video import VotingEnsemble

    return VotingEnsemble(min_vote_ratio=0.05)


@pytest.fixture
def sample_aggregated_products():
    """Create sample aggregated products for testing."""
    from model_service.session.door_session import AggregatedProduct

    return {
        26: AggregatedProduct(
            product_id=26,
            product_idx="26",
            name="치킨마요",
            count=2,
            unit_price=3500,
            weight=365.0,
            total_confidence=1.8,
            detection_count=2,
        ),
        15: AggregatedProduct(
            product_id=15,
            product_idx="15",
            name="참치마요",
            count=1,
            unit_price=3000,
            weight=250.0,
            total_confidence=0.85,
            detection_count=1,
        ),
    }
