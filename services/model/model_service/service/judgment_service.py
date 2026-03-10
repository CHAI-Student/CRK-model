"""
Judgment Service (v4.2).

판단 비즈니스 로직 - 세션 조회 및 상품 판단 결과 반환.
라우터에서 분리된 핵심 비즈니스 로직.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from model_service.session import SessionStore, SessionData, ProductResult, DoorSessionStore
from model_service.session.door_session import DoorSession, AggregatedProduct
from model_service.session.active_product_store import ActiveProductStore
from model_service.engine import ProductDecisionEngine, EnsembleResult

logger = logging.getLogger(__name__)


@dataclass
class ProductInfo:
    """상품 정보 (Node.js 요청)."""
    product_idx: str
    product_name: str
    sale_price: int
    product_weight: str
    stock_qty: int = 0
    has_loadcell: str = "true"  # v4.8: 추가


@dataclass
class JudgmentInput:
    """판단 요청 입력."""
    session_id: Optional[str]
    zone: Optional[int]
    products: List[ProductInfo] = field(default_factory=list)


@dataclass
class ProductResponse:
    """상품 판단 결과."""
    product_idx: str
    product_id: int
    name: str
    count: int
    price: int
    confidence: float = 0.0


@dataclass
class WeightInfo:
    """무게 정보."""
    delta: float
    is_removal: bool


@dataclass
class ProcessingStats:
    """처리 통계."""
    top_frames: int
    side_frames: int
    processing_time_ms: float


@dataclass
class DoorSessionInfo:
    """Door Session 정보."""
    trigger_count: int
    duration_seconds: float
    created_at: float
    last_trigger_at: Optional[float] = None
    finalized_at: Optional[float] = None
    unmatched_returns: Optional[Dict[str, Any]] = None


@dataclass
class JudgmentResult:
    """판단 결과."""
    success: bool
    status: str  # "processing" | "in_progress" | "complete" | "error"
    zone: Optional[int] = None
    session_id: Optional[str] = None
    door_session_id: Optional[str] = None
    device_id: Optional[str] = None
    message: Optional[str] = None
    reason: Optional[str] = None
    processing_stage: Optional[str] = None
    processing_stage_detail: Optional[str] = None
    products: List[ProductResponse] = field(default_factory=list)
    product_count: int = 0
    total_price: int = 0
    confidence: float = 0.0
    weight_info: Optional[WeightInfo] = None
    stats: Optional[ProcessingStats] = None
    door_session_info: Optional[DoorSessionInfo] = None
    interim_products: Optional[List[ProductResponse]] = None
    interim_product_count: Optional[int] = None
    interim_total_price: Optional[int] = None


class JudgmentService:
    """
    판단 비즈니스 로직 서비스.

    세션 조회 및 상품 판단 결과 반환을 담당.
    """

    def __init__(
        self,
        session_store: SessionStore,
        engine: ProductDecisionEngine,
        door_session_store: Optional[DoorSessionStore] = None,
        active_product_store: Optional[ActiveProductStore] = None,
    ):
        """
        Initialize judgment service.

        Args:
            session_store: SessionStore 인스턴스
            engine: ProductDecisionEngine 인스턴스
            door_session_store: DoorSessionStore 인스턴스 (선택)
            active_product_store: ActiveProductStore 인스턴스 (선택, v5.0)
        """
        self._session_store = session_store
        self._engine = engine
        self._door_session_store = door_session_store
        self._active_product_store = active_product_store

    def judge(self, input_data: JudgmentInput) -> JudgmentResult:
        """
        상품 판단 수행.
        """
        logger.info(
            f"[JUDGMENT] session_id={input_data.session_id}, "
            f"zone={input_data.zone}, products={len(input_data.products)}"
        )

        # session_id 유효성 검사
        is_valid_session_id = (
            input_data.session_id is not None
            and input_data.session_id.startswith("zone_")
        )
        device_id = None if is_valid_session_id else input_data.session_id

        # Door Session 기반 응답 (v4.1)
        if self._door_session_store is not None and input_data.zone is not None:
            door_result = self._process_door_session(
                input_data.zone,
                device_id,
                input_data.session_id,
            )
            if door_result is not None:
                return door_result

        # 기존 SessionStore 기반 응답
        return self._process_session_store(
            input_data,
            is_valid_session_id,
            device_id,
        )

    def _process_door_session(
        self,
        zone: int,
        device_id: Optional[str],
        session_id: Optional[str],
    ) -> Optional[JudgmentResult]:
        """Door Session 기반 응답 처리."""
        door_session, is_finalized = self._door_session_store.get_or_finalize(zone)

        if door_session is None:
            return None

        active_products = door_session.get_active_products()
        products_response = self._convert_aggregated_products(active_products)
        product_count = sum(p.count for p in products_response)
        total_price = door_session.total_price

        if is_finalized:
            return self._create_door_session_complete_response(
                door_session,
                products_response,
                product_count,
                total_price,
                device_id,
            )
        else:
            return self._create_door_session_progress_response(
                door_session,
                products_response,
                product_count,
                total_price,
                device_id,
            )

    def _create_door_session_complete_response(
        self,
        door_session: DoorSession,
        products: List[ProductResponse],
        product_count: int,
        total_price: int,
        device_id: Optional[str],
    ) -> JudgmentResult:
        """Door Session 완료 응답 생성."""
        logger.info(
            f"[JUDGMENT DOOR SESSION COMPLETE] "
            f"door_session_id={door_session.door_session_id}, "
            f"triggers={door_session.trigger_count}, "
            f"products={product_count}, total_price={total_price}"
        )

        unmatched_info = None
        if door_session.has_unmatched_returns:
            unmatched_info = {
                "count": len(door_session.unmatched_returns),
                "totalWeight": round(door_session.unmatched_returns_weight, 1),
                "details": [
                    {
                        "triggerId": r.trigger_id,
                        "deltaWeight": round(r.delta_weight, 1),
                        "timestamp": r.timestamp,
                    }
                    for r in door_session.unmatched_returns
                ],
            }

        total_delta = sum(t.delta_weight for t in door_session.triggers)
        avg_confidence = (
            sum(p.confidence for p in products) / len(products)
            if products else 0.0
        )

        return JudgmentResult(
            success=product_count > 0,
            status="complete",
            zone=door_session.zone,
            session_id=door_session.triggers[-1].session_id if door_session.triggers else None,
            door_session_id=door_session.door_session_id,
            device_id=device_id,
            processing_stage="complete",
            processing_stage_detail=f"Door session 완료: {door_session.trigger_count}개 trigger 통합",
            products=products,
            product_count=product_count,
            total_price=total_price,
            confidence=round(avg_confidence, 4),
            weight_info=WeightInfo(
                delta=round(total_delta, 1),
                is_removal=total_delta < 0,
            ),
            stats=ProcessingStats(
                top_frames=0,
                side_frames=0,
                processing_time_ms=round(
                    sum(t.processing_time_ms for t in door_session.triggers),
                    1,
                ),
            ),
            door_session_info=DoorSessionInfo(
                trigger_count=door_session.trigger_count,
                duration_seconds=round(door_session.duration_seconds, 1),
                created_at=door_session.created_at,
                finalized_at=door_session.finalized_at,
                unmatched_returns=unmatched_info,
            ),
        )

    def _create_door_session_progress_response(
        self,
        door_session: DoorSession,
        products: List[ProductResponse],
        product_count: int,
        total_price: int,
        device_id: Optional[str],
    ) -> JudgmentResult:
        """Door Session 진행 중 응답 생성."""
        logger.info(
            f"[JUDGMENT DOOR SESSION IN_PROGRESS] "
            f"door_session_id={door_session.door_session_id}, "
            f"triggers={door_session.trigger_count}, "
            f"interim_products={product_count}"
        )

        return JudgmentResult(
            success=False,
            status="in_progress",
            zone=door_session.zone,
            session_id=door_session.triggers[-1].session_id if door_session.triggers else None,
            door_session_id=door_session.door_session_id,
            device_id=device_id,
            processing_stage="door_session_active",
            processing_stage_detail=f"Door session 활성: {door_session.trigger_count}개 trigger 수신",
            interim_products=products,
            interim_product_count=product_count,
            interim_total_price=total_price,
            stats=ProcessingStats(
                top_frames=0,
                side_frames=0,
                processing_time_ms=round(
                    sum(t.processing_time_ms for t in door_session.triggers),
                    1,
                ),
            ),
            door_session_info=DoorSessionInfo(
                trigger_count=door_session.trigger_count,
                duration_seconds=round(door_session.duration_seconds, 1),
                created_at=door_session.created_at,
                last_trigger_at=door_session.last_trigger_at,
            ),
        )

    def _process_session_store(
        self,
        input_data: JudgmentInput,
        is_valid_session_id: bool,
        device_id: Optional[str],
    ) -> JudgmentResult:
        """기존 SessionStore 기반 응답 처리."""
        if is_valid_session_id:
            session_data, session_status = self._session_store.get_with_status(
                input_data.session_id
            )
        else:
            if device_id:
                logger.info(
                    f"[JUDGMENT] device_id={device_id}, "
                    f"looking up latest active session for zone={input_data.zone}"
                )
            session_data = self._session_store.get_latest_active(input_data.zone)
            session_status = "found" if session_data else "no_active_session"

        if session_data is None:
            return self._create_processing_response(
                session_status, device_id, input_data.session_id, input_data.zone
            )

        if session_data.status == "processing":
            return self._create_session_processing_response(session_data, device_id)

        return self._create_complete_response(session_data, device_id)

    def _create_processing_response(
        self,
        session_status: str,
        device_id: Optional[str],
        session_id: Optional[str],
        zone: Optional[int],
    ) -> JudgmentResult:
        """처리 중 응답 생성."""
        if session_status == "expired":
            return JudgmentResult(
                success=False,
                status="processing",
                device_id=device_id,
                message="세션이 만료되었습니다. 다시 시도해주세요.",
                reason="expired",
                processing_stage="expired",
                processing_stage_detail="세션 TTL 만료",
            )
        elif session_status == "no_active_session":
            return JudgmentResult(
                success=False,
                status="processing",
                device_id=device_id,
                message="활성 세션이 없습니다",
                reason="no_active_session",
                processing_stage="waiting",
                processing_stage_detail="카메라 트리거 대기 중",
            )
        else:
            return JudgmentResult(
                success=False,
                status="processing",
                device_id=device_id,
                message="YOLO 추론 대기 중",
                reason="not_found",
                processing_stage="waiting",
                processing_stage_detail="세션 생성 대기 중",
            )

    def _create_session_processing_response(
        self,
        session_data: SessionData,
        device_id: Optional[str],
    ) -> JudgmentResult:
        """세션 처리 중 응답 생성."""
        logger.info(
            f"[JUDGMENT RESPONSE] device_id={device_id}, "
            f"session_id={session_data.session_id}, "
            f"status=processing, stage={session_data.processing_stage}"
        )
        return JudgmentResult(
            success=False,
            status="processing",
            device_id=device_id,
            session_id=session_data.session_id,
            zone=session_data.zone,
            message=session_data.processing_stage_detail or "처리 중",
            reason="in_progress",
            processing_stage=session_data.processing_stage,
            processing_stage_detail=session_data.processing_stage_detail,
        )

    def _create_complete_response(
        self,
        session_data: SessionData,
        device_id: Optional[str],
    ) -> JudgmentResult:
        """완료 응답 생성."""
        products = [
            ProductResponse(
                product_idx=p.product_idx if p.product_idx else str(p.product_id),
                product_id=p.product_id,
                name=p.name,
                count=p.count,
                price=p.price,
                confidence=round(p.confidence, 4),
            )
            for p in session_data.products
            if p.count > 0
        ]

        product_count = sum(p.count for p in products)
        is_success = session_data.status == "complete" and product_count > 0

        logger.info(
            f"[JUDGMENT RESPONSE] device_id={device_id}, "
            f"session_id={session_data.session_id}, status=complete, "
            f"zone={session_data.zone}, products={len(products)}, "
            f"count={product_count}, total={session_data.total_price}, "
            f"delta={session_data.delta_weight:.1f}g, success={is_success}"
        )

        return JudgmentResult(
            success=is_success,
            status="complete",
            device_id=device_id,
            zone=session_data.zone,
            session_id=session_data.session_id,
            processing_stage=session_data.processing_stage,
            processing_stage_detail=session_data.processing_stage_detail,
            products=products,
            product_count=product_count,
            total_price=session_data.total_price,
            confidence=round(session_data.confidence, 4),
            weight_info=WeightInfo(
                delta=round(session_data.delta_weight, 1),
                is_removal=session_data.delta_weight < 0,
            ),
            stats=ProcessingStats(
                top_frames=session_data.top_frames,
                side_frames=session_data.side_frames,
                processing_time_ms=round(session_data.processing_time_ms, 1),
            ),
        )

    def _get_product_idx(self, product_id: int) -> Optional[str]:
        """YOLO class_id로 IF11 product_idx 조회."""
        if self._active_product_store is not None:
            product_info = self._active_product_store.get_by_yolo_class_id(product_id)
            if product_info and product_info.product_idx:
                return product_info.product_idx
        return None

    def _convert_aggregated_products(
        self,
        aggregated_products: List[AggregatedProduct],
    ) -> List[ProductResponse]:
        """AggregatedProduct를 ProductResponse로 변환."""
        return [
            ProductResponse(
                product_idx=p.product_idx if p.product_idx else str(p.product_id),
                product_id=p.product_id,
                name=p.name,
                count=p.count,
                price=p.unit_price,
                confidence=round(p.average_confidence, 4),
            )
            for p in aggregated_products
        ]
