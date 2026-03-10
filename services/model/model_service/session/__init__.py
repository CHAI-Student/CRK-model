"""
Session module for storing YOLO inference results.

세션별 YOLO 추론 결과 저장 모듈.

v4.5 변경:
- PendingTriggerStore 제거 (전역 상품 관리로 불필요해짐)

v4.4 추가:
- ActiveProductStore: Node.js에서 받은 상품 정보 관리 (YOLO 사전 필터링)

v4.3 추가:
- GlobalDoorSession: session_id="OPEN"/"CLOSE" 기반 전체 zone 통합 관리

v4.2 추가:
- UnmatchedReturn: 무게 매칭 실패 추적

v4.1 추가:
- Door Session: 문 열림~닫힘 동안의 모든 trigger 통합 관리
- ProductAggregator: 상품 합산/반환 처리
- YamlPersistence: YAML 파일 저장/복구
"""

from .session_store import SessionStore, SessionData, ProductResult
from .door_session import (
    DoorSession,
    TriggerResult,
    AggregatedProduct,
    UnmatchedReturn,
    generate_door_session_id,
)
from .global_door_session import GlobalDoorSession, generate_global_session_id
from .door_session_store import DoorSessionStore
from .product_aggregator import ProductAggregator
from .yaml_persistence import YamlPersistence
from .active_product_store import ActiveProductStore

__all__ = [
    # Session Store (v4.0)
    "SessionStore",
    "SessionData",
    "ProductResult",
    # Door Session (v4.1)
    "DoorSession",
    "TriggerResult",
    "AggregatedProduct",
    "UnmatchedReturn",
    "generate_door_session_id",
    "DoorSessionStore",
    "ProductAggregator",
    "YamlPersistence",
    # Global Door Session (v4.3)
    "GlobalDoorSession",
    "generate_global_session_id",
    # Product Pre-filtering (v4.4/v4.5)
    "ActiveProductStore",
]
