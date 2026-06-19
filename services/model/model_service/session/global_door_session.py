"""
Global Door Session Data Model (v4.7).

전체 zone(1~5)을 통합 관리하는 GlobalDoorSession.
Node.js가 session_id="OPEN"/"CLOSE" 신호를 보내면,
문 열림~닫힘 동안의 모든 zone trigger를 하나의 세션으로 통합합니다.

v4.7 변경사항:
- pending_close/first_close_at 필드 추가 (빠른 문 열고 닫기 대응)
- is_ready_to_finalize_after_close() 메서드 추가

사용법:
    global_session = GlobalDoorSession(
        global_session_id=generate_global_session_id(),
    )

    # zone별 DoorSession 추가
    global_session.zone_sessions[1] = door_session_zone_1

    # 전체 통계 조회
    total = global_session.total_price          # 총 가격
    count = global_session.total_product_count  # 총 상품 수
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

from .door_session import DoorSession


@dataclass
class GlobalDoorSession:
    """
    전체 zone 통합 관리 세션 (v4.7).

    문 열림(OPEN) ~ 문 닫힘(CLOSE) 동안의 모든 zone trigger를 통합 관리.
    Node.js가 session_id="OPEN"을 처음 보내면 생성,
    session_id="CLOSE"를 보내면 종료.

    v4.7: pending_close 상태 추가 (빠른 문 열고 닫기 대응)
    - 첫 CLOSE 시 pending_close=True 설정
    - 다음 CLOSE 시 first_close_at 이후 trigger가 없으면 finalize

    Attributes:
        global_session_id: GlobalSession ID (예: "global_260203_143000")
        status: 세션 상태 ("active" | "complete")
        zone_sessions: zone별 DoorSession (zone -> DoorSession)
        created_at: 세션 생성 시각 (epoch)
        finalized_at: 세션 종료 시각 (complete일 때만)
        last_trigger_at: 마지막 trigger 시각 (v4.6)
        pending_close: 첫 CLOSE 수신 여부 (v4.7)
        first_close_at: 첫 CLOSE 시각 (v4.7)
    """

    global_session_id: str
    status: str = "active"  # "active" | "complete"
    zone_sessions: Dict[int, DoorSession] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    finalized_at: Optional[float] = None
    last_trigger_at: Optional[float] = None  # 마지막 trigger 시각 (v4.6)
    pending_close: bool = False  # 첫 CLOSE 수신 여부 (v4.7)
    first_close_at: Optional[float] = None  # 첫 CLOSE 시각 (v4.7)

    @property
    def total_price(self) -> int:
        """모든 zone의 총 금액 (count > 0인 상품만)."""
        return sum(s.total_price for s in self.zone_sessions.values())

    @property
    def total_product_count(self) -> int:
        """모든 zone의 총 상품 수 (count > 0인 상품만)."""
        return sum(s.product_count for s in self.zone_sessions.values())

    @property
    def is_active(self) -> bool:
        """활성 상태 여부."""
        return self.status == "active"

    @property
    def total_trigger_count(self) -> int:
        """모든 zone의 총 trigger 수."""
        return sum(s.trigger_count for s in self.zone_sessions.values())

    @property
    def duration_seconds(self) -> float:
        """세션 지속 시간 (초)."""
        end_time = self.finalized_at or time.time()
        return end_time - self.created_at

    @property
    def active_zones(self) -> list:
        """활성 zone 목록 (trigger가 있는 zone)."""
        return sorted(self.zone_sessions.keys())

    def is_ready_to_finalize(self, wait_seconds: float = 30.0) -> bool:
        """
        Finalize 준비 완료 여부 (v4.6, 하위 호환).

        마지막 trigger 후 wait_seconds 이상 지났으면 True.
        trigger가 없으면 True (빈 세션).

        참고: v4.7에서는 is_ready_to_finalize_after_close() 사용 권장

        Args:
            wait_seconds: 대기 시간 (초)

        Returns:
            True if ready to finalize
        """
        if self.last_trigger_at is None:
            return True  # trigger 없음 → 바로 종료 가능
        return time.time() - self.last_trigger_at >= wait_seconds

    def is_ready_to_finalize_after_close(self, wait_seconds: float = 20.0) -> bool:
        """
        CLOSE 이후 finalize 준비 완료 여부 (v4.7).

        pending_close 상태에서:
        1. first_close_at 이후 wait_seconds(기본 20초)가 지났는지 확인
        2. first_close_at 이후 trigger가 없는지 확인

        조건:
        - pending_close=False: 아직 첫 CLOSE 안 옴 → False
        - first_close_at 이후 wait_seconds 안 지남 → False (더 대기 필요)
        - first_close_at 이후 trigger 있음 → False (더 대기 필요)
        - first_close_at 이후 wait_seconds 지났고 trigger 없음 → True

        Args:
            wait_seconds: 첫 CLOSE 이후 대기 시간 (기본 20초)

        Returns:
            True if ready to finalize after CLOSE signal
        """
        if not self.pending_close:
            return False  # 아직 첫 CLOSE 안 옴

        if self.first_close_at is None:
            return False  # 비정상 상태

        # first_close_at 이후 wait_seconds가 지났는지 확인
        elapsed = time.time() - self.first_close_at
        if elapsed < wait_seconds:
            return False  # 아직 대기 시간 안 지남

        # first_close_at 이후에 trigger가 있었는지 확인
        if self.last_trigger_at is not None and self.last_trigger_at > self.first_close_at:
            return False  # first_close 이후 trigger 있음 → 더 대기

        return True  # 대기 시간 지났고 trigger 없음 → finalize 가능

    def get_zone_session(self, zone: int) -> Optional[DoorSession]:
        """특정 zone의 DoorSession 반환."""
        return self.zone_sessions.get(zone)

    def to_dict(self) -> dict:
        """딕셔너리 변환."""
        return {
            "global_session_id": self.global_session_id,
            "status": self.status,
            "zone_sessions": {
                zone: session.to_dict()
                for zone, session in self.zone_sessions.items()
            },
            "created_at": self.created_at,
            "finalized_at": self.finalized_at,
            "last_trigger_at": self.last_trigger_at,
            "pending_close": self.pending_close,  # v4.7
            "first_close_at": self.first_close_at,  # v4.7
            "summary": {
                "total_price": self.total_price,
                "total_product_count": self.total_product_count,
                "total_trigger_count": self.total_trigger_count,
                "zone_count": len(self.zone_sessions),
                "active_zones": self.active_zones,
                "duration_seconds": round(self.duration_seconds, 1),
            },
        }


def generate_global_session_id() -> str:
    """
    Global Session ID 생성.

    Format: global_{YYMMDD}_{HHMMSS}_{ffffff}

    Returns:
        Global Session ID (예: global_260203_143000_123456)
    """
    now = datetime.now()
    return f"global_{now.strftime('%y%m%d_%H%M%S_%f')}"
