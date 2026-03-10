"""
Door Session Service (v4.3).

Door Session 비즈니스 로직 - 세션 관리, 통계 조회, 강제 종료.
라우터에서 분리된 핵심 비즈니스 로직.

v4.3 변경사항:
- add_trigger → add_trigger_with_global 변경 (GlobalSession 연동 보장)
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from model_service.session import DoorSessionStore
from model_service.session.door_session import DoorSession

logger = logging.getLogger(__name__)


@dataclass
class DoorSessionStats:
    """Door Session 통계."""
    enabled: bool
    active_sessions: int = 0
    active_zones: list = None
    session_timeout: float = 0.0
    weight_tolerance: float = 0.0
    max_duration: float = 0.0
    timestamp: float = 0.0
    message: Optional[str] = None

    def __post_init__(self):
        if self.active_zones is None:
            self.active_zones = []


@dataclass
class DoorSessionStatus:
    """Door Session 상태."""
    found: bool
    zone: int
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


@dataclass
class FinalizeResult:
    """Door Session 강제 종료 결과."""
    success: bool
    zone: int
    door_session_id: Optional[str] = None
    trigger_count: int = 0
    product_count: int = 0
    total_price: int = 0
    message: str = ""


class DoorSessionService:
    """
    Door Session 비즈니스 로직 서비스.

    Door Session 관리, 통계 조회, 강제 종료를 담당.
    """

    def __init__(self, door_session_store: Optional[DoorSessionStore] = None):
        """
        Initialize door session service.

        Args:
            door_session_store: DoorSessionStore 인스턴스 (선택)
        """
        self._door_session_store = door_session_store

    @property
    def is_enabled(self) -> bool:
        """DoorSessionStore 활성화 여부."""
        return self._door_session_store is not None

    def get_stats(self) -> DoorSessionStats:
        """
        Door Session 통계 조회.

        Returns:
            DoorSessionStats: 통계 정보
        """
        if not self.is_enabled:
            return DoorSessionStats(
                enabled=False,
                message="DoorSessionStore is not enabled",
            )

        stats = self._door_session_store.get_stats()

        return DoorSessionStats(
            enabled=True,
            active_sessions=stats.get("active_sessions", 0),
            active_zones=stats.get("active_zones", []),
            session_timeout=stats.get("session_timeout", 0.0),
            weight_tolerance=stats.get("weight_tolerance", 0.0),
            max_duration=stats.get("max_duration", 0.0),
            timestamp=time.time(),
        )

    def get_session_status(self, zone: int) -> DoorSessionStatus:
        """
        특정 Zone의 Door Session 상태 조회.

        Args:
            zone: Zone 번호

        Returns:
            DoorSessionStatus: 세션 상태
        """
        if not self.is_enabled:
            return DoorSessionStatus(
                found=False,
                zone=zone,
                message="DoorSessionStore is not enabled",
            )

        door_session = self._door_session_store.get_session(zone)

        if door_session is None:
            return DoorSessionStatus(
                found=False,
                zone=zone,
                message="No active door session for this zone",
            )

        return DoorSessionStatus(
            found=True,
            zone=zone,
            data=door_session.to_dict(),
        )

    def finalize_session(self, zone: int) -> FinalizeResult:
        """
        Door Session 강제 종료.

        Args:
            zone: Zone 번호

        Returns:
            FinalizeResult: 종료 결과
        """
        if not self.is_enabled:
            return FinalizeResult(
                success=False,
                zone=zone,
                message="DoorSessionStore is not enabled",
            )

        door_session = self._door_session_store.finalize_session(zone)

        if door_session is None:
            return FinalizeResult(
                success=False,
                zone=zone,
                message="No active door session to finalize",
            )

        logger.info(
            f"[DOOR SESSION SERVICE] Finalized: zone={zone}, "
            f"door_session_id={door_session.door_session_id}, "
            f"triggers={door_session.trigger_count}, "
            f"products={door_session.product_count}, "
            f"total_price={door_session.total_price}"
        )

        return FinalizeResult(
            success=True,
            zone=zone,
            door_session_id=door_session.door_session_id,
            trigger_count=door_session.trigger_count,
            product_count=door_session.product_count,
            total_price=door_session.total_price,
            message="Door session finalized successfully",
        )

    def get_or_finalize(self, zone: int) -> tuple[Optional[DoorSession], bool]:
        """
        Door Session 조회 또는 타임아웃 시 종료.

        Args:
            zone: Zone 번호

        Returns:
            (DoorSession, is_finalized) 튜플
        """
        if not self.is_enabled:
            return None, False

        return self._door_session_store.get_or_finalize(zone)

    def add_trigger_with_global(self, zone: int, trigger_result) -> Optional[DoorSession]:
        """
        Door Session에 트리거 결과 추가 (GlobalSession 연동).

        v4.3: add_trigger → add_trigger_with_global 변경.
        GlobalSession이 활성화된 경우 자동 연동됩니다.

        Args:
            zone: Zone 번호
            trigger_result: TriggerResult

        Returns:
            DoorSession: 업데이트된 세션 (없으면 None)
        """
        if not self.is_enabled:
            return None

        return self._door_session_store.add_trigger_with_global(zone, trigger_result)
