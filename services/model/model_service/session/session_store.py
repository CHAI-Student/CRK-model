"""
Session Store for YOLO Inference Results (v4.5).

YOLO 추론 결과를 세션별로 저장하고 관리.
TTL 기반 자동 정리 기능 포함.

v4.5 변경사항:
- _cleanup_to_capacity() 메서드 추가 (max_sessions의 80%까지 정리)
- 동시 요청 시 용량 초과 방지

사용법:
    store = SessionStore(ttl_seconds=300)

    # 결과 저장
    store.save("zone_1_260201_143025", session_data)

    # 결과 조회
    data = store.get("zone_1_260201_143025")
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ProductResult:
    """단일 상품 판단 결과."""
    product_id: int           # YOLO class_id (내부용)
    name: str
    count: int
    price: int
    confidence: float
    product_idx: Optional[str] = None  # IF11 product_idx (Node.js 응답용)


@dataclass
class SessionData:
    """
    세션별 YOLO 추론 결과.

    Attributes:
        session_id: 세션 ID (예: zone_1_260201_143025)
        zone: Zone 번호
        products: 판단된 상품 목록
        total_price: 총 금액
        delta_weight: 무게 변화량 (g)
        created_at: 생성 시간 (timestamp)
        status: 처리 상태 ("processing" | "complete" | "error")
        processing_stage: 현재 처리 단계
            - "extracting_frames": 프레임 추출 중
            - "detecting_products": 상품 후보 도출 중
            - "calculating_count": 개수 판단 중
            - "complete": 완료
        processing_stage_detail: 처리 단계 상세 정보
        confidence: 전체 신뢰도
        top_frames: Top 카메라 프레임 수
        side_frames: Side 카메라 프레임 수
        processing_time_ms: 처리 시간 (ms)
        vision_candidates: YOLO 비전 후보군 (재계산용, EnsembleResult의 dict 목록)
    """
    session_id: str
    zone: int
    products: List[ProductResult] = field(default_factory=list)
    total_price: int = 0
    delta_weight: float = 0.0
    created_at: float = field(default_factory=time.time)
    status: str = "processing"
    processing_stage: str = "extracting_frames"
    processing_stage_detail: str = ""
    confidence: float = 0.0
    top_frames: int = 0
    side_frames: int = 0
    processing_time_ms: float = 0.0
    vision_candidates: Optional[List[dict]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        result = {
            "session_id": self.session_id,
            "zone": self.zone,
            "products": [
                {
                    "productIdx": p.product_idx if p.product_idx else str(p.product_id),
                    "productId": p.product_id,
                    "name": p.name,
                    "count": p.count,
                    "price": p.price,
                    "confidence": round(p.confidence, 4),
                }
                for p in self.products
            ],
            "totalPrice": self.total_price,
            "delta_weight": round(self.delta_weight, 1),
            "created_at": self.created_at,
            "status": self.status,
            "processing_stage": self.processing_stage,
            "processing_stage_detail": self.processing_stage_detail,
            "confidence": round(self.confidence, 4),
            "stats": {
                "top_frames": self.top_frames,
                "side_frames": self.side_frames,
                "processing_time_ms": round(self.processing_time_ms, 1),
            },
        }
        # vision_candidates는 디버깅용으로만 포함 (있을 경우)
        if self.vision_candidates:
            result["vision_candidates_count"] = len(self.vision_candidates)
        return result


class SessionStore:
    """
    세션별 YOLO 추론 결과 저장소.

    Thread-safe한 dict 기반 구현.
    TTL 기반 자동 정리 기능 포함.
    """

    def __init__(self, ttl_seconds: float = 300.0, max_sessions: int = 100):
        """
        Initialize session store.

        Args:
            ttl_seconds: 세션 TTL (기본: 300초 = 5분)
            max_sessions: 최대 세션 수 (기본: 100)
        """
        self._store: Dict[str, SessionData] = {}
        self._lock = threading.Lock()
        self._ttl_seconds = ttl_seconds
        self._max_sessions = max_sessions

        logger.info(
            f"SessionStore initialized: ttl={ttl_seconds}s, max={max_sessions}"
        )

    def save(self, session_id: str, data: SessionData) -> None:
        """
        세션 데이터 저장.

        v4.5: 용량 초과 시 80%까지 정리 (동시 요청 대응)

        Args:
            session_id: 세션 ID
            data: 세션 데이터
        """
        with self._lock:
            # 중복 세션 경고
            if session_id in self._store:
                logger.warning(f"Session overwritten: {session_id}")

            # v4.5: 용량 초과 시 80%까지 정리 (1개만 제거하면 동시 요청 시 한계 초과)
            if len(self._store) >= self._max_sessions:
                self._cleanup_to_capacity()

            self._store[session_id] = data
            logger.debug(f"Session saved: {session_id}")

    def get(self, session_id: str) -> Optional[SessionData]:
        """
        세션 데이터 조회.

        Args:
            session_id: 세션 ID

        Returns:
            SessionData or None if not found or expired
        """
        data, _ = self.get_with_status(session_id)
        return data

    def get_with_status(self, session_id: str) -> Tuple[Optional[SessionData], str]:
        """
        세션 데이터 조회 (상태 포함).

        Args:
            session_id: 세션 ID

        Returns:
            (SessionData, status) 튜플
            - status: "found" | "not_found" | "expired"
        """
        with self._lock:
            data = self._store.get(session_id)

            if data is None:
                return None, "not_found"

            # TTL 체크
            if time.time() - data.created_at > self._ttl_seconds:
                del self._store[session_id]
                logger.debug(f"Session expired and removed: {session_id}")
                return None, "expired"

            return data, "found"

    def update_stage(
        self,
        session_id: str,
        processing_stage: str,
        processing_stage_detail: str = "",
        status: Optional[str] = None,
    ) -> bool:
        """
        세션 처리 단계 업데이트.

        Args:
            session_id: 세션 ID
            processing_stage: 처리 단계
                - "extracting_frames": 프레임 추출 중
                - "detecting_products": 상품 후보 도출 중
                - "calculating_count": 개수 판단 중
                - "complete": 완료
            processing_stage_detail: 상세 정보 (예: "Top 카메라 50/120 프레임")
            status: 세션 상태 (예: "error") - None이면 변경하지 않음

        Returns:
            업데이트 성공 여부
        """
        with self._lock:
            if session_id not in self._store:
                return False

            self._store[session_id].processing_stage = processing_stage
            self._store[session_id].processing_stage_detail = processing_stage_detail
            if status is not None:
                self._store[session_id].status = status
            logger.debug(
                f"Session stage updated: {session_id} -> {processing_stage} ({processing_stage_detail})"
            )
            return True

    def delete(self, session_id: str) -> bool:
        """
        세션 삭제.

        Args:
            session_id: 세션 ID

        Returns:
            삭제 성공 여부
        """
        with self._lock:
            if session_id in self._store:
                del self._store[session_id]
                logger.debug(f"Session deleted: {session_id}")
                return True
            return False

    def exists(self, session_id: str) -> bool:
        """
        세션 존재 여부 확인.

        Args:
            session_id: 세션 ID

        Returns:
            존재 여부 (TTL 만료된 경우 False)
        """
        return self.get(session_id) is not None

    def cleanup_expired(self) -> int:
        """
        만료된 세션 정리.

        Returns:
            정리된 세션 수
        """
        with self._lock:
            now = time.time()
            expired_keys = [
                key for key, data in self._store.items()
                if now - data.created_at > self._ttl_seconds
            ]

            for key in expired_keys:
                del self._store[key]

            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired sessions")

            return len(expired_keys)

    def _cleanup_oldest(self) -> None:
        """가장 오래된 세션 1개 정리 (lock 내부에서 호출)."""
        if not self._store:
            return

        # 가장 오래된 세션 찾기
        oldest_key = min(
            self._store.keys(),
            key=lambda k: self._store[k].created_at
        )
        del self._store[oldest_key]
        logger.debug(f"Removed oldest session: {oldest_key}")

    def _cleanup_to_capacity(self) -> int:
        """
        max_sessions의 80%까지 정리 (v4.5).

        동시 요청 시 한계 초과를 방지하기 위해 여유 공간 확보.
        lock 내부에서 호출됩니다.

        Returns:
            정리된 세션 수
        """
        if not self._store:
            return 0

        # 목표: max_sessions의 80%
        target_count = int(self._max_sessions * 0.8)
        current_count = len(self._store)

        if current_count <= target_count:
            return 0

        # 정리해야 할 개수
        to_remove = current_count - target_count

        # 생성 시간 기준으로 정렬하여 가장 오래된 것부터 제거
        sorted_keys = sorted(
            self._store.keys(),
            key=lambda k: self._store[k].created_at
        )

        removed_count = 0
        for key in sorted_keys[:to_remove]:
            del self._store[key]
            removed_count += 1
            logger.debug(f"Removed session (capacity): {key}")

        if removed_count > 0:
            logger.info(
                f"Session store capacity cleanup: removed {removed_count} sessions "
                f"({current_count} -> {len(self._store)}, target={target_count})"
            )

        return removed_count

    def clear_all(self) -> None:
        """모든 세션 삭제."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            logger.info(f"Cleared all {count} sessions")

    def get_stats(self) -> dict:
        """
        저장소 통계 반환.

        Returns:
            통계 정보 dict
        """
        with self._lock:
            now = time.time()
            active_count = sum(
                1 for data in self._store.values()
                if now - data.created_at <= self._ttl_seconds
            )

            return {
                "total_sessions": len(self._store),
                "active_sessions": active_count,
                "ttl_seconds": self._ttl_seconds,
                "max_sessions": self._max_sessions,
            }

    def get_latest_active(self, zone: Optional[int] = None) -> Optional[SessionData]:
        """
        최근 활성 세션 조회.

        Args:
            zone: Zone 필터 (선택)

        Returns:
            가장 최근 SessionData or None
        """
        with self._lock:
            now = time.time()
            active = [
                data for data in self._store.values()
                if now - data.created_at <= self._ttl_seconds
                and (zone is None or data.zone == zone)
            ]

            if not active:
                return None

            # 가장 최근 세션 반환
            return max(active, key=lambda d: d.created_at)

    @property
    def session_count(self) -> int:
        """현재 저장된 세션 수."""
        with self._lock:
            return len(self._store)


def generate_session_id(zone: int) -> str:
    """
    세션 ID 생성.

    Format: zone_{zone}_{YYMMDD}_{HHMMSS}_{ffffff}

    Args:
        zone: Zone 번호

    Returns:
        세션 ID (예: zone_1_260201_143025_123456)
    """
    now = datetime.now()
    return f"zone_{zone}_{now.strftime('%y%m%d_%H%M%S_%f')}"
