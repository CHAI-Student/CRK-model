"""
Multi-Zone Judge API Routes (v5.2).

POST /api/judge/multi-zone - Node.js에서 10초 간격으로 폴링
SessionStore에서 결과를 조회하여 반환.

v5.2 변경사항:
- daemon thread → ThreadPoolExecutor로 변경 (파일 핸들 누수 방지)
- shutdown_log_executor() 함수 추가 (lifespan에서 호출)

v4.8 변경사항:
- Zone 기반 DoorSession 코드 제거 (dead code 정리)
- GlobalDoorSession으로 완전 통합
- Node.js는 항상 OPEN/CLOSE 신호를 보내므로 Zone 타임아웃 불필요

v4.7 변경사항:
- CLOSE 타이밍 문제 해결 (빠른 문 열고 닫기 대응)
- handle_close_signal() 사용: 첫 CLOSE → pending, 두 번째 CLOSE → finalize 여부 결정
- pending_close 상태 응답 추가

v4.5 변경사항:
- 상품 리스트 전역(global) 저장 (zone별 관리 제거)
- ProductInfo.product_weight를 Optional로 변경
- PendingTriggerStore 관련 로직 제거

v4.3 변경사항:
- session_id 필드를 문 상태 신호로 재활용
  - session_id="OPEN" → GlobalSession 시작/유지
  - session_id="CLOSE" → GlobalSession 종료 + 최종 결과
  - session_id=null → 폴링 (현재 상태 조회)
- zones 배열 응답: zone 1~5 결과 분리 반환
- GlobalDoorSession으로 전체 zone 통합 관리

사용 흐름:
1. Node.js가 데드볼트 문 열림 감지 → session_id="OPEN" 전송
2. 문 열린 동안 session_id="OPEN" 반복 전송 (기존 세션 유지)
3. 카메라 trigger 발생 시 각 zone에 상품 누적
4. 문 닫힘 → session_id="CLOSE" 전송 (첫 CLOSE: pending 상태)
5. 두 번째 CLOSE → trigger 없으면 finalize, 있으면 다시 대기
6. Model이 zones 배열로 zone 1~5 최종 결과 반환
"""

import concurrent.futures
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException
from model_service.api.deps import (
    get_active_product_store_optional,
    get_door_session_store_optional,
    get_session_store,
)
from model_service.core.config import config
from model_service.core.logging_config import get_ops_logger
from model_service.session import (
    DoorSession,
    DoorSessionStore,
    GlobalDoorSession,
    SessionStore,
)
from model_service.session.active_product_store import ActiveProductStore
from model_service.session.door_session import (
    trigger_effective_delta_weight,
    unmatched_return_delta_weight,
)
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)
ops_logger = get_ops_logger()

router = APIRouter(prefix="/api/judge", tags=["judge"])

# ============================================================================
# v5.2: ThreadPoolExecutor for Request Logging (replaces daemon threads)
# ============================================================================

# 모듈 레벨 ThreadPoolExecutor (서비스 전체에서 공유)
_log_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
_log_executor_shutdown = False


def _get_log_executor() -> concurrent.futures.ThreadPoolExecutor:
    """로그 기록용 ThreadPoolExecutor 가져오기 (lazy init)."""
    global _log_executor
    if _log_executor is None:
        _log_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="multi_zone_log",
        )
    return _log_executor


def shutdown_log_executor() -> None:
    """
    로그 기록용 ThreadPoolExecutor 종료 (v5.2).

    FastAPI lifespan shutdown에서 호출해야 합니다.
    서비스 종료 시 파일 I/O가 완전히 완료되도록 보장합니다.
    """
    global _log_executor, _log_executor_shutdown
    _log_executor_shutdown = True
    if _log_executor is not None:
        logger.info("Shutting down multi_zone log executor...")
        _log_executor.shutdown(wait=True, cancel_futures=False)
        _log_executor = None
        logger.info("multi_zone log executor shutdown complete")


# ============================================================================
# Request/Response Models
# ============================================================================


class ProductInfo(BaseModel):
    """
    Node.js에서 전달하는 상품 정보.

    Note: product_weight는 Node.js에서 string으로 전달됩니다.
          ActiveProductStore.set_products()에서 float로 변환 처리됩니다.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    product_idx: str = Field(..., description="상품 ID (IF11)")
    product_name: str = Field(..., description="상품명")
    sale_price: int = Field(..., description="판매가격")
    product_weight: Optional[str | float | int] = Field(
        default="0",
        validation_alias=AliasChoices(
            "product_weight",
            "productWeight",
            "product_weight_g",
            "productWeightG",
            "unit_weight_g",
            "unitWeightG",
            "weight",
        ),
        description="상품 무게 (g), string→float 변환됨",
    )
    stock_qty: Optional[int] = Field(default=None, description="재고 수량 (v4.6: None이면 무제한)")
    has_loadcell: str = Field(default="true", description="로드셀 사용 여부 (v4.8)")
    yolo_class_id: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices(
            "yolo_class_id",
            "yoloClassId",
            "trainingidx",
            "training_idx",
            "trainingIdx",
        ),
        description="YOLO class id override from Node/model training index",
    )
    yolo_class_name: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("yolo_class_name", "yoloClassName"),
        description="YOLO class name override",
    )


class MultiZoneRequest(BaseModel):
    """Multi-Zone 판단 요청."""

    session_id: Optional[str] = Field(default=None, description="세션 ID (선택)")
    products: List[ProductInfo] = Field(
        default_factory=list,
        description="상품 목록 (선택, 무게 검증용)",
    )
    zone: Optional[int] = Field(default=None, description="Zone 번호 (세션 필터용)")


class ProductResponse(BaseModel):
    """상품 판단 결과."""

    productIdx: str  # IF11 product_idx (문자열)
    productId: int   # YOLO class_id (하위 호환)
    name: str
    count: int
    price: int
    confidence: float = 0.0


class ProcessingResponse(BaseModel):
    """처리 중 응답 (HTTP 200)."""

    status: str = "processing"
    message: str = "YOLO 추론 대기 중"


class WeightInfo(BaseModel):
    """무게 정보."""

    delta: float  # 무게 변화량 (g), 음수 = 제거
    isRemoval: bool  # 상품 제거 여부


class ProcessingStats(BaseModel):
    """처리 통계."""

    topFrames: int
    sideFrames: int
    processingTimeMs: float


class CompleteResponse(BaseModel):
    """완료 응답 (HTTP 200)."""

    status: str = "complete"
    zone: int
    products: List[ProductResponse]
    productCount: int  # 총 상품 개수
    totalPrice: int
    confidence: float
    weightInfo: WeightInfo
    stats: ProcessingStats


# ============================================================================
# Request Parsing Helper
# ============================================================================


def _parse_request(body: Any) -> MultiZoneRequest:
    """
    유연한 요청 파싱.

    Node.js가 보내는 배열 형식과 기존 객체 형식 모두 처리:
    - 배열: Node.js가 productData 직접 전송 (session_id 없음)
    - 객체: 기존 형식 {session_id?, products, zone?}

    Args:
        body: 요청 본문 (list 또는 dict)

    Returns:
        MultiZoneRequest
    """
    if isinstance(body, list):
        # Node.js 배열 형식: productData 직접 전송
        products = []
        for p in body:
            try:
                products.append(ProductInfo(**p))
            except Exception as e:
                logger.warning(f"[PARSE] Invalid product item: {p}, error: {e}")
        return MultiZoneRequest(products=products)

    elif isinstance(body, dict):
        # 기존 객체 형식
        products = []
        for p in body.get("products", []):
            try:
                products.append(ProductInfo(**p))
            except Exception as e:
                logger.warning(f"[PARSE] Invalid product item: {p}, error: {e}")
        return MultiZoneRequest(
            session_id=body.get("session_id"),
            products=products,
            zone=body.get("zone"),
        )

    raise ValueError(f"Invalid request format: expected list or dict, got {type(body)}")


# ============================================================================
# Product Validation Helper (v4.4 - 간소화)
# ============================================================================


# 비정상 상품명 패턴 (대소문자 무시)
INVALID_PRODUCT_NAME_PATTERNS = ["test", "테스트", "sample", "샘플", "dummy", "더미"]


def _check_invalid_product_names(products: List[ProductInfo]) -> List[str]:
    """
    비정상 상품명 감지.

    "TEST", "테스트", "SAMPLE" 등 더미 상품명을 감지합니다.
    이런 상품명은 YOLO 매핑이 불가능하므로 HTTP 400을 반환해야 합니다.

    Args:
        products: 상품 목록

    Returns:
        비정상 상품명 리스트 (비어있으면 정상)
    """
    invalid_names = []
    for p in products:
        if not p.product_name:
            continue
        name_lower = p.product_name.strip().lower()
        for pattern in INVALID_PRODUCT_NAME_PATTERNS:
            if pattern in name_lower:
                invalid_names.append(f"{p.product_name} (product_idx={p.product_idx})")
                break
    return invalid_names


def _log_product_warnings(products: List[ProductInfo]) -> None:
    """
    기타 경고사항 로깅 (HTTP 400 아님, 로그만).

    음수 가격, 음수 재고 등은 경고만 출력하고 계속 진행합니다.
    """
    warnings = []
    for p in products:
        if p.sale_price < 0:
            warnings.append(f"음수 가격: {p.product_idx}, price={p.sale_price}")
        if p.stock_qty is not None and p.stock_qty < 0:
            warnings.append(f"음수 재고: {p.product_idx}, stock={p.stock_qty}")

    if warnings:
        # 경고를 1회 로깅 (개별 반복 없음)
        logger.warning(f"[PRODUCT WARNINGS] {len(warnings)}개: {warnings[:5]}")


# ============================================================================
# Request Logging Helper (디버깅용, 비동기 - v4.4)
# ============================================================================


def _coerce_product_weight(value: object) -> float:
    try:
        return float(value) if value else 0.0
    except (TypeError, ValueError):
        return 0.0


def _has_stock_positive_weight_request_products(products: List[ProductInfo]) -> bool:
    for product in products:
        stock = product.stock_qty if product.stock_qty is not None else 999
        if stock > 0 and _coerce_product_weight(product.product_weight) > 0:
            return True
    return False


def _has_stock_positive_class_request_products(products: List[ProductInfo]) -> bool:
    for product in products:
        stock = product.stock_qty if product.stock_qty is not None else 999
        if stock > 0 and product.yolo_class_id is not None:
            return True
    return False


def _products_to_store_payload(products: List[ProductInfo]) -> List[dict]:
    payload = []
    for product in products:
        stock = product.stock_qty if product.stock_qty is not None else 999
        payload.append({
            "product_idx": product.product_idx,
            "product_name": product.product_name,
            "sale_price": product.sale_price,
            "product_weight": product.product_weight or "0",
            "stock_qty": stock,
            "has_loadcell": product.has_loadcell,
            "yolo_class_id": product.yolo_class_id,
            "yolo_class_name": product.yolo_class_name,
        })
        logger.debug(
            f"[MULTI-ZONE] Product: {product.product_name}, "
            f"original_stock={product.stock_qty}, used_stock={stock}"
        )
    return payload


def _summarize_product_payload(products: List[ProductInfo], limit: int = 5) -> List[dict]:
    return [
        {
            "product_idx": product.product_idx,
            "name": product.product_name,
            "stock": product.stock_qty,
            "weight": product.product_weight,
            "yolo_class_id": product.yolo_class_id,
        }
        for product in products[:limit]
    ]


def _maybe_update_active_product_snapshot(
    *,
    request: MultiZoneRequest,
    door_state: Optional[str],
    active_product_store: Optional[ActiveProductStore],
) -> None:
    if active_product_store is None or not request.products:
        return

    if door_state == "CLOSE":
        logger.info(
            "[MULTI-ZONE] ignored products payload for active snapshot: "
            "reason=close_request"
        )
        return

    has_candidate_inventory = _has_stock_positive_class_request_products(
        request.products
    )
    has_existing_inventory = bool(
        getattr(
            active_product_store,
            "has_stock_positive_class_products",
            lambda: False,
        )()
    )

    if not has_candidate_inventory:
        reason = "zero_stock_positive_class_products"
        logger.warning(
            "[MULTI-ZONE] ignored products payload for active snapshot: "
            f"reason={reason}, products={len(request.products)}, "
            f"preserved_existing={str(has_existing_inventory).lower()}, "
            f"sample={_summarize_product_payload(request.products)}"
        )
        return

    set_result = active_product_store.set_products(
        products=_products_to_store_payload(request.products),
        preserve_on_invalid_existing=True,
    )

    logger.info(
        f"[MULTI-ZONE] set_products: "
        f"{set_result.mapped_products}/{set_result.total_products} mapped, "
        f"{set_result.allowed_products} allowed (stock > 0), "
        f"{set_result.stock_positive_class_products} stock+class, "
        f"{set_result.stock_positive_weight_products} stock+weight"
    )

    if set_result.preserved_existing:
        logger.warning(
            "[MULTI-ZONE] ignored products payload for active snapshot: "
            "reason=mapped_zero_stock_positive_class_products, "
            "preserved_existing=true, "
            f"sample={_summarize_product_payload(request.products)}"
        )

    if set_result.unmapped_names:
        logger.warning(
            f"[MULTI-ZONE] unmapped products: {set_result.unmapped_names}"
        )


def _log_request_to_file_sync(request: MultiZoneRequest, response: dict) -> None:
    """
    Node.js 요청/응답을 로그 파일에 저장 (동기 버전, 스레드에서 실행).

    로그 파일: services/model/logs/multi_zone_YYYYMMDD.jsonl
    """
    try:
        log_dir = Path(__file__).parent.parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / f"multi_zone_{datetime.now().strftime('%Y%m%d')}.jsonl"

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request": {
                "session_id": request.session_id,
                "zone": request.zone,
                "products_count": len(request.products),
                "products": [
                    {
                        "product_idx": p.product_idx,
                        "product_name": p.product_name,
                        "sale_price": p.sale_price,
                        "product_weight": p.product_weight,
                        "stock_qty": p.stock_qty,
                        "yolo_class_id": p.yolo_class_id,
                        "yolo_class_name": p.yolo_class_name,
                    }
                    for p in request.products[:10]  # 최대 10개만 로깅
                ],
            },
            "response_summary": {
                "status": response.get("status"),
                "success": response.get("success"),
                "zones_count": len(response.get("zones", [])),
                "total_price": response.get("totalPrice") or response.get("totalInterimPrice") or response.get("interimTotalPrice"),
                "product_count": response.get("productCount") or response.get("totalProductCount") or response.get("interimProductCount"),
            },
        }

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    except Exception as e:
        logger.warning(f"[LOG] Failed to write request log: {e}")


def _log_request_to_file(request: MultiZoneRequest, response: dict) -> None:
    """
    Node.js 요청/응답을 로그 파일에 저장 (비동기 - ThreadPoolExecutor).

    v5.2: daemon thread → ThreadPoolExecutor로 변경 (파일 핸들 누수 방지)
    v4.4: 동기 파일 I/O가 Uvicorn 블로킹을 유발하지 않도록
    백그라운드에서 비동기적으로 실행합니다.
    """
    global _log_executor_shutdown

    # v5.2: executor가 이미 shutdown된 경우 (서비스 종료 중)
    if _log_executor_shutdown:
        # 동기로 즉시 저장 시도 (graceful shutdown 보장)
        try:
            _log_request_to_file_sync(request, response)
        except Exception as e:
            logger.warning(f"[LOG] Sync log write failed during shutdown: {e}")
        return

    try:
        executor = _get_log_executor()
        executor.submit(_log_request_to_file_sync, request, response)
    except RuntimeError:
        # ThreadPoolExecutor가 이미 shutdown된 경우
        logger.warning("[LOG] Log executor shutdown, skipping log write")


# ============================================================================
# Door State Helper Functions (v4.3)
# ============================================================================


def _parse_door_state(session_id: Optional[str]) -> Optional[str]:
    """
    session_id에서 문 상태(door_state) 추출.

    Args:
        session_id: 요청의 session_id 필드

    Returns:
        "OPEN", "CLOSE", 또는 None (일반 세션 ID인 경우)
    """
    if session_id is None:
        return None
    upper = session_id.upper()
    if upper in ("OPEN", "CLOSE"):
        return upper
    return None


def _effective_zone_weight_delta(
    global_session: GlobalDoorSession,
    door_session: DoorSession,
) -> float:
    """Return basket-level delta after cross-zone return reconciliation."""
    raw_delta = sum(trigger_effective_delta_weight(t) for t in door_session.triggers)
    unmatched_return_weight = unmatched_return_delta_weight(door_session)
    outgoing_cross_zone_weight = sum(
        record.delta_weight for record in door_session.cross_zone_returns
    )
    incoming_cross_zone_weight = sum(
        record.delta_weight
        for other_session in global_session.zone_sessions.values()
        for record in other_session.cross_zone_returns
        if record.target_zone == door_session.zone
    )
    return (
        raw_delta
        - unmatched_return_weight
        - outgoing_cross_zone_weight
        + incoming_cross_zone_weight
    )


def _build_zone_result(
    global_session: GlobalDoorSession,
    zone: int,
    is_complete: bool,
) -> dict:
    """
    Zone별 결과 구성.

    Args:
        global_session: GlobalDoorSession
        zone: Zone 번호 (1~5)
        is_complete: 완료 상태 여부

    Returns:
        Zone 결과 딕셔너리
    """
    door_session = global_session.zone_sessions.get(zone)

    if door_session is None:
        # 해당 zone에 trigger가 없는 경우
        if is_complete:
            return {
                "zone": zone,
                "products": [],
                "productNames": [],
                "productCounts": [],
                "totalPrice": 0,
                "productCount": 0,
                "weightDelta": 0.0,
            }
        else:
            return {
                "zone": zone,
                "interim_products": [],
                "interimProductNames": [],
                "interimProductCounts": [],
                "interimTotalPrice": 0,
                "interimProductCount": 0,
                "weightDelta": 0.0,
            }

    # 활성 상품 조회 (count > 0)
    active_products = door_session.get_active_products()
    products_list = [
        {
            "productIdx": p.product_idx or str(p.product_id),
            "productId": p.product_id,
            "name": p.name,
            "count": p.count,
            "price": p.unit_price,
            "confidence": round(p.average_confidence, 4),
        }
        for p in active_products
    ]

    # 새로 추가할 배열들
    product_names = [p.name for p in active_products]
    product_counts = [p.count for p in active_products]

    # Raw trigger deltas stay on the session/trace. Node-facing zone output
    # uses the basket-level effective delta after cross-zone returns.
    weight_delta = _effective_zone_weight_delta(global_session, door_session)
    if not active_products and abs(weight_delta) <= float(config.weight.tolerance_grams):
        weight_delta = 0.0
    failure_reasons = sorted(
        {
            reason
            for reason in (
                getattr(trigger, "failure_reason", None)
                for trigger in door_session.triggers
            )
            if reason
        }
    )
    final_weight_validation = dict(
        getattr(door_session, "final_weight_validation", {}) or {}
    )

    if is_complete:
        result = {
            "zone": zone,
            "products": products_list,
            "productNames": product_names,
            "productCounts": product_counts,
            "totalPrice": door_session.total_price,
            "productCount": door_session.product_count,
            "weightDelta": round(weight_delta, 1),
            "door_session_id": door_session.door_session_id,
            "triggerCount": door_session.trigger_count,
            "failureReasons": failure_reasons,
        }
        if final_weight_validation:
            result["finalWeightValidation"] = final_weight_validation
        return result
    else:
        result = {
            "zone": zone,
            "interim_products": products_list,
            "interimProductNames": product_names,
            "interimProductCounts": product_counts,
            "interimTotalPrice": door_session.total_price,
            "interimProductCount": door_session.product_count,
            "weightDelta": round(weight_delta, 1),
            "triggerCount": door_session.trigger_count,
            "failureReasons": failure_reasons,
        }
        if final_weight_validation:
            result["finalWeightValidation"] = final_weight_validation
        return result


def _build_decision_summary(global_session: GlobalDoorSession, zones: List[dict]) -> dict:
    """Build and log the final per-zone decision summary for CLOSE responses."""
    summary_zones = []
    zone_lines = []
    active_zone_lines = []
    for zone in zones:
        products = zone.get("products", [])
        product_text = ", ".join(
            f"{product.get('name')}x{product.get('count')}"
            for product in products
        ) or "none"
        failure_reasons = zone.get("failureReasons", [])
        failure_text = (
            f" reason={','.join(failure_reasons)}"
            if not products and failure_reasons
            else ""
        )
        zone_line = (
            f"zone={zone.get('zone')} "
            f"weight_delta={float(zone.get('weightDelta', 0.0)):.1f}g "
            f"products={product_text}{failure_text}"
        )
        zone_lines.append(zone_line)
        if (
            products
            or float(zone.get("weightDelta", 0.0)) != 0.0
            or int(zone.get("triggerCount", 0)) > 0
        ):
            active_zone_lines.append(zone_line)
        summary_zone = {
            "zone": zone.get("zone"),
            "weightDelta": zone.get("weightDelta", 0.0),
            "triggerCount": zone.get("triggerCount", 0),
            "productCount": zone.get("productCount", 0),
            "totalPrice": zone.get("totalPrice", 0),
            "products": products,
            "failureReasons": failure_reasons,
        }
        if zone.get("finalWeightValidation"):
            summary_zone["finalWeightValidation"] = zone["finalWeightValidation"]
        summary_zones.append(summary_zone)

    total_weight_delta = round(
        sum(float(zone.get("weightDelta", 0.0)) for zone in summary_zones),
        1,
    )
    summary = {
        "globalSessionId": global_session.global_session_id,
        "totalWeightDelta": total_weight_delta,
        "totalProductCount": global_session.total_product_count,
        "totalPrice": global_session.total_price,
        "zones": summary_zones,
        "zoneLines": zone_lines,
    }
    summary_text = (
        "; ".join(active_zone_lines or ["zones=none"])
        + f"; total_weight_delta={total_weight_delta:.1f}g "
        + f"total_price={global_session.total_price}"
    )
    summary["summaryText"] = summary_text

    logger.info(
        "[MULTI-ZONE CLOSE][SUMMARY] "
        f"global_session_id={global_session.global_session_id} "
        f"total_weight_delta={total_weight_delta:.1f}g "
        f"total_products={global_session.total_product_count} "
        f"total_price={global_session.total_price}"
    )
    for zone in summary_zones:
        product_text = ", ".join(
            f"{product.get('name')}x{product.get('count')}"
            for product in zone["products"]
        ) or "none"
        failure_text = (
            f" reason={','.join(zone.get('failureReasons', []))}"
            if not zone["products"] and zone.get("failureReasons")
            else ""
        )
        logger.info(
            "[MULTI-ZONE CLOSE][SUMMARY] "
            f"zone={zone['zone']} "
            f"weight_delta={float(zone['weightDelta']):.1f}g "
            f"triggers={zone['triggerCount']} "
            f"products={product_text}{failure_text}"
        )
    ops_logger.info(f"[OPS][CLOSE] {summary_text}")

    return summary


def _handle_door_open(store: DoorSessionStore) -> dict:
    """
    OPEN 처리 (v4.3).

    session_id="OPEN" 시 호출됩니다.
    이미 GlobalSession이 있으면 기존 것 사용 (초기화 안함)

    Args:
        store: DoorSessionStore

    Returns:
        API 응답 딕셔너리
    """
    if store is None:
        return {
            "success": False,
            "status": "error",
            "message": "DoorSessionStore not enabled",
        }

    # get_or_start: 있으면 기존 것, 없으면 새로 생성
    global_session = store.get_or_start_global_session()

    # Zone 1~5 현재 상태 반환
    zones = []
    for zone_num in range(1, 6):
        zone_result = _build_zone_result(global_session, zone_num, is_complete=False)
        zones.append(zone_result)

    total_interim_price = sum(z.get("interimTotalPrice", 0) for z in zones)
    total_interim_count = sum(z.get("interimProductCount", 0) for z in zones)

    # Node.js 하위 호환: 모든 zone의 interim_products를 합친 products 배열
    all_products = []
    for z in zones:
        all_products.extend(z.get("interim_products", []))

    logger.info(
        f"[MULTI-ZONE OPEN] global_session_id={global_session.global_session_id}, "
        f"total_triggers={global_session.total_trigger_count}, "
        f"interim_price={total_interim_price}"
    )

    return {
        "success": False,
        "status": "in_progress",
        "global_session_id": global_session.global_session_id,
        "zones": zones,
        "products": all_products,  # Node.js 하위 호환
        "totalInterimPrice": total_interim_price,
        "totalInterimProductCount": total_interim_count,
        "interimProductCount": total_interim_count,  # Node.js 하위 호환
        "globalSessionInfo": {
            "totalTriggerCount": global_session.total_trigger_count,
            "durationSeconds": round(global_session.duration_seconds, 1),
            "createdAt": global_session.created_at,
            "activeZones": global_session.active_zones,
        },
    }


def _handle_door_close(
    store: DoorSessionStore,
    active_product_store: Optional[ActiveProductStore] = None,
) -> dict:
    """
    CLOSE 처리 (v4.7 수정).

    session_id="CLOSE" 시 호출됩니다.
    빠른 문 열고 닫기 상황 대응을 위해 handle_close_signal() 사용:
    1. 첫 CLOSE → pending_close 상태, in_progress 반환
    2. 두 번째+ CLOSE → first_close_at 이후 trigger 유무에 따라 결정
       - trigger 없음 → finalize 진행
       - trigger 있음 → first_close_at 갱신, 다시 대기

    v4.7: handle_close_signal() 기반 로직으로 변경
    v4.4: ActiveProductStore도 정리.

    Args:
        store: DoorSessionStore
        active_product_store: ActiveProductStore (v4.4, 선택)

    Returns:
        API 응답 딕셔너리
    """
    if store is None:
        return {
            "success": True,
            "status": "success",
            "message": "DoorSessionStore not enabled",
            "products": [],
            "totalPrice": 0,
            "productCount": 0,
        }

    # v4.7: handle_close_signal() 사용
    is_ready, global_session = store.handle_close_signal()
    pending_snapshot = (
        store.get_pending_trigger_snapshot()
        if hasattr(store, "get_pending_trigger_snapshot")
        else {
            "pendingTriggerCount": 0,
            "pendingTriggerZones": [],
            "pendingTriggerSessionIds": [],
            "pendingTriggerStatuses": {},
        }
    )

    # GlobalSession이 없으면 바로 종료
    if global_session is None:
        logger.info("[MULTI-ZONE CLOSE] No active global session to close")
        return {
            "success": True,
            "status": "success",
            "message": "No active door session to close",
            "zones": [],
            "products": [],
            "totalPrice": 0,
            "totalProductCount": 0,
            "productCount": 0,
            "globalSessionInfo": None,
        }

    # 아직 finalize 준비 안됨 (pending_close 상태)
    if not is_ready:
        elapsed = 0.0
        if global_session.first_close_at is not None:
            elapsed = time.time() - global_session.first_close_at
        pending_trigger_count = pending_snapshot.get("pendingTriggerCount", 0)
        pending_chargeable_vision_count = pending_snapshot.get(
            "pendingChargeableVisionCount",
            pending_trigger_count,
        )
        pending_reason = (
            "pending_trigger" if pending_chargeable_vision_count > 0 else "close_wait"
        )

        # Zone 1~5 현재 interim 상태 반환 (v4.7)
        zones = []
        for zone_num in range(1, 6):
            zone_result = _build_zone_result(global_session, zone_num, is_complete=False)
            zones.append(zone_result)

        total_interim_price = sum(z.get("interimTotalPrice", 0) for z in zones)
        total_interim_count = sum(z.get("interimProductCount", 0) for z in zones)

        # Node.js 하위 호환
        all_products = []
        for z in zones:
            all_products.extend(z.get("interim_products", []))

        logger.info(
            f"[MULTI-ZONE CLOSE] Pending close: "
            f"global_session_id={global_session.global_session_id}, "
            f"elapsed={elapsed:.1f}s, pending_close={global_session.pending_close}, "
            f"last_trigger_at={global_session.last_trigger_at}"
        )

        return {
            "success": False,
            "status": "in_progress",
            "reason": pending_reason,
            "message": f"CLOSE 대기 중 (trigger 도착 대기, {elapsed:.1f}초)",
            "pending_close": True,  # v4.7: pending 상태 표시
            "pendingTriggerCount": pending_trigger_count,
            "pendingTriggerZones": pending_snapshot.get("pendingTriggerZones", []),
            "pendingTriggerSessionIds": pending_snapshot.get("pendingTriggerSessionIds", []),
            "pendingTriggerStatuses": pending_snapshot.get("pendingTriggerStatuses", {}),
            "pendingChargeableVisionCount": pending_chargeable_vision_count,
            "pendingChargeableVisionSessionIds": pending_snapshot.get(
                "pendingChargeableVisionSessionIds",
                [],
            ),
            "skippedBalancedTriggerCount": pending_snapshot.get(
                "skippedBalancedTriggerCount",
                0,
            ),
            "global_session_id": global_session.global_session_id,
            "zones": zones,
            "products": all_products,
            "totalInterimPrice": total_interim_price,
            "totalInterimProductCount": total_interim_count,
            "interimProductCount": total_interim_count,
            "totalPrice": total_interim_price,
            "productCount": total_interim_count,
            "globalSessionInfo": {
                "totalTriggerCount": global_session.total_trigger_count,
                "durationSeconds": round(global_session.duration_seconds, 1),
                "createdAt": global_session.created_at,
                "activeZones": global_session.active_zones,
                "pendingClose": global_session.pending_close,
                "firstCloseAt": global_session.first_close_at,
                "lastTriggerAt": global_session.last_trigger_at,
            },
        }

    # finalize 준비 완료 → finalize 진행
    global_session = store.finalize_global_session()

    # B7: 동시 CLOSE/cleanup 경합 시 None 반환 대응
    if global_session is None:
        logger.warning("[MULTI-ZONE CLOSE] finalize_global_session returned None - concurrent close detected")
        return {
            "success": False,
            "status": "error",
            "message": "세션이 이미 종료되었습니다.",
            "zones": [],
            "products": [],
            "totalPrice": 0,
            "totalProductCount": 0,
            "productCount": 0,
        }

    # B5: 세션 종료 후 ActiveProductStore 정리 (다음 세션 class filtering 오염 방지)
    if active_product_store is not None:
        active_product_store.clear()
        logger.info("[MULTI-ZONE CLOSE] ActiveProductStore cleared after finalize")

    # Zone 1~5 최종 결과 구성
    zones = []
    for zone_num in range(1, 6):
        zone_result = _build_zone_result(global_session, zone_num, is_complete=True)
        zones.append(zone_result)

    total_price = global_session.total_price
    total_product_count = global_session.total_product_count

    # Node.js 하위 호환: 모든 zone의 products를 합친 배열
    all_products = []
    for z in zones:
        all_products.extend(z.get("products", []))

    decision_summary = _build_decision_summary(global_session, zones)
    failure_reasons = sorted(
        {
            reason
            for zone in zones
            for reason in zone.get("failureReasons", [])
        }
    )
    has_missing_active_products = "missing_active_products" in failure_reasons
    has_empty_allowlist = "empty_active_product_allowlist" in failure_reasons
    final_success = not (has_missing_active_products or has_empty_allowlist)
    final_status = (
        "missing_active_products"
        if has_missing_active_products
        else "empty_active_product_allowlist"
        if has_empty_allowlist
        else "success"
        if total_product_count > 0
        else "complete_no_products"
    )

    logger.info(
        f"[MULTI-ZONE CLOSE] Finalized: global_session_id={global_session.global_session_id}, "
        f"zones={len(global_session.zone_sessions)}, "
        f"total_price={total_price}, total_products={total_product_count}"
    )

    return {
        "success": final_success,
        "status": final_status,
        "has_products": total_product_count > 0,
        "failureReasons": failure_reasons,
        "global_session_id": global_session.global_session_id,
        "zones": zones,
        "products": all_products,
        "totalPrice": total_price,
        "totalProductCount": total_product_count,
        "productCount": total_product_count,
        "decisionSummary": decision_summary,
        "decisionSummaryText": decision_summary["summaryText"],
        "globalSessionInfo": {
            "totalTriggerCount": global_session.total_trigger_count,
            "durationSeconds": round(global_session.duration_seconds, 1),
            "createdAt": global_session.created_at,
            "finalizedAt": global_session.finalized_at,
            "activeZones": global_session.active_zones,
        },
    }


def _handle_global_polling(store: DoorSessionStore) -> dict:
    """
    GlobalSession 활성 시 폴링 응답 (v4.3).

    session_id=null 이고 GlobalSession이 활성인 경우.

    Args:
        store: DoorSessionStore

    Returns:
        API 응답 딕셔너리
    """
    global_session = store.get_global_session()

    if global_session is None:
        # 이 경우는 발생하지 않아야 함 (is_global_session_active 체크 후 호출)
        return {
            "success": False,
            "status": "processing",
            "message": "No active global session",
            "products": [],  # Node.js 하위 호환
        }

    # Zone 1~5 현재 상태 반환
    zones = []
    for zone_num in range(1, 6):
        zone_result = _build_zone_result(global_session, zone_num, is_complete=False)
        zones.append(zone_result)

    total_interim_price = sum(z.get("interimTotalPrice", 0) for z in zones)
    total_interim_count = sum(z.get("interimProductCount", 0) for z in zones)

    # Node.js 하위 호환: 모든 zone의 interim_products를 합친 products 배열
    all_products = []
    for z in zones:
        all_products.extend(z.get("interim_products", []))

    return {
        "success": False,
        "status": "in_progress",
        "global_session_id": global_session.global_session_id,
        "zones": zones,
        "products": all_products,  # Node.js 하위 호환
        "totalInterimPrice": total_interim_price,
        "totalInterimProductCount": total_interim_count,
        "interimProductCount": total_interim_count,  # Node.js 하위 호환
        "globalSessionInfo": {
            "totalTriggerCount": global_session.total_trigger_count,
            "durationSeconds": round(global_session.duration_seconds, 1),
            "createdAt": global_session.created_at,
            "activeZones": global_session.active_zones,
        },
    }


# ============================================================================
# Multi-Zone Judge Endpoint
# ============================================================================


@router.post("/multi-zone")
async def judge_multi_zone(
    body: Any = Body(...),
    session_store: SessionStore = Depends(get_session_store),
    door_session_store: DoorSessionStore | None = Depends(get_door_session_store_optional),
    active_product_store: ActiveProductStore | None = Depends(get_active_product_store_optional),
):
    """
    Node.js 폴링용 상품 판단 API (v4.3).

    v4.3 신규:
    - session_id="OPEN" → GlobalSession 시작/유지
    - session_id="CLOSE" → GlobalSession 종료 + zones 배열 반환
    - session_id=null → 폴링 (GlobalSession 있으면 zones 반환)

    데드볼트 문 열리면 10초 간격으로 호출됩니다.
    SessionStore에서 결과를 조회하여:
    - 세션 없음 → "processing" 응답 (success=false)
    - 세션 있음 → "complete" + 상품 정보 응답 (success=true)

    지원하는 요청 형식:
    1. 배열 직접 전송: [{"product_idx": "26", ...}, ...]
    2. 객체 형식: {"session_id": "OPEN|CLOSE|null", "products": [...], "zone": 1}

    Args:
        body: 요청 본문 (list 또는 dict)
        session_store: SessionStore 의존성

    Returns:
        ProcessingResponse or CompleteResponse (모두 success 필드 포함)
        v4.3: zones 배열 포함 응답
    """
    # 요청 파싱
    logger.info(f"[MULTI-ZONE REQUEST] body_type={type(body).__name__}")

    try:
        request = _parse_request(body)
    except ValueError as e:
        logger.error(f"[MULTI-ZONE ERROR] Failed to parse request: {e}")
        return {
            "success": False,
            "status": "error",
            "message": str(e),
            "reason": "parse_error",
        }

    logger.info(
        f"[MULTI-ZONE PARSED] session_id={request.session_id}, "
        f"products={len(request.products)}, zone={request.zone}"
    )

    # ========================================================================
    # 상품 정보 유효성 검사 (v4.4 간소화)
    # - 비정상 상품명(TEST 등)만 HTTP 400 반환
    # - 기타 경고는 로그만 출력 (서비스 블로킹 방지)
    # ========================================================================
    if request.products:
        # 비정상 상품명만 검사 (HTTP 400)
        invalid_names = _check_invalid_product_names(request.products)
        if invalid_names:
            # 로깅 최소화 (1회만)
            logger.warning(f"[INVALID PRODUCTS] {len(invalid_names)}개 감지: {invalid_names[:3]}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "INVALID_PRODUCT_NAME",
                    "message": f"비정상 상품명 감지: {len(invalid_names)}개",
                    "invalid_products": invalid_names,
                },
            )

        # 기타 경고는 로그만 (HTTP 400 아님)
        _log_product_warnings(request.products)

    # ========================================================================
    # v4.3: Door State (OPEN/CLOSE) 처리
    # ========================================================================
    door_state = _parse_door_state(request.session_id)
    logger.info(
        f"[MULTI-ZONE] door_state parsed: "
        f"session_id='{request.session_id}' -> door_state='{door_state}'"
    )

    # ActiveProductStore update must run after door_state parsing so CLOSE/result
    # payloads cannot replace live inventory candidates.
    _maybe_update_active_product_snapshot(
        request=request,
        door_state=door_state,
        active_product_store=active_product_store,
    )

    # OPEN 처리 (반복 호출됨 - 기존 세션 유지)
    if door_state == "OPEN":
        response = _handle_door_open(door_session_store)
        _log_request_to_file(request, response)
        return response

    # CLOSE 처리 (최종 결과 반환)
    if door_state == "CLOSE":
        response = _handle_door_close(door_session_store, active_product_store)
        _log_request_to_file(request, response)
        return response

    # 폴링 (session_id가 null이고 GlobalSession이 활성인 경우)
    if door_session_store is not None and door_session_store.is_global_session_active():
        response = _handle_global_polling(door_session_store)
        _log_request_to_file(request, response)
        return response

    # ========================================================================
    # v4.8: Zone 기반 DoorSession 제거됨
    # - Node.js는 항상 OPEN/CLOSE 신호를 보내므로 GlobalDoorSession만 사용
    # - Zone 기반 타임아웃 로직 삭제 (dead code 정리)
    # ========================================================================

    # 기존 SessionStore 기반 응답 (GlobalSession이 없는 경우 - 레거시 fallback)
    # session_id 유효성 검사: zone_으로 시작하면 실제 세션 ID, 아니면 device_id
    # Node.js가 deviceIdx(예: "DE17683631997086480")를 session_id로 보내는 경우
    is_valid_session_id = (
        request.session_id is not None
        and request.session_id.startswith("zone_")
    )

    # device_id 추출: zone_으로 시작 안하면 device_id로 간주
    device_id = None if is_valid_session_id else request.session_id

    if is_valid_session_id:
        # 유효한 session_id로 조회
        session_data, session_status = session_store.get_with_status(request.session_id)
    else:
        # device_id이거나 없으면 최근 활성 세션 자동 선택
        if device_id:
            logger.info(
                f"[MULTI-ZONE] device_id={device_id}, "
                f"looking up latest active session for zone={request.zone}"
            )
        session_data = session_store.get_latest_active(request.zone)
        session_status = "found" if session_data else "no_active_session"

    if session_data is None:
        # 세션이 없거나 만료된 경우
        if session_status == "expired":
            logger.info(
                f"[MULTI-ZONE RESPONSE] device_id={device_id}, session_id={request.session_id}, "
                f"status=processing, reason=expired"
            )
            return {
                "success": False,
                "status": "processing",
                "message": "세션이 만료되었습니다. 다시 시도해주세요.",
                "reason": "expired",
                "device_id": device_id,
                "processing_stage": "expired",
                "processing_stage_detail": "세션 TTL 만료",
            }
        elif session_status == "no_active_session":
            logger.info(
                f"[MULTI-ZONE RESPONSE] device_id={device_id}, zone={request.zone}, "
                f"status=processing, reason=no_active_session"
            )
            return {
                "success": False,
                "status": "processing",
                "message": "활성 세션이 없습니다",
                "reason": "no_active_session",
                "device_id": device_id,
                "processing_stage": "waiting",
                "processing_stage_detail": "카메라 트리거 대기 중",
            }
        else:
            logger.info(
                f"[MULTI-ZONE RESPONSE] device_id={device_id}, session_id={request.session_id}, "
                f"status=processing, reason=not_found"
            )
            return {
                "success": False,
                "status": "processing",
                "message": "YOLO 추론 대기 중",
                "reason": "not_found",
                "device_id": device_id,
                "processing_stage": "waiting",
                "processing_stage_detail": "세션 생성 대기 중",
            }

    # 처리 에러 상태 (status="error") - 즉시 실패 응답
    if session_data.status == "error":
        logger.warning(
            f"[MULTI-ZONE RESPONSE] device_id={device_id}, session_id={session_data.session_id}, "
            f"status=error, detail={session_data.processing_stage_detail}"
        )
        return {
            "success": False,
            "status": "error",
            "message": session_data.processing_stage_detail or "처리 오류",
            "reason": "processing_failed",
            "device_id": device_id,
            "session_id": session_data.session_id,
            "zone": session_data.zone,
            "processing_stage": session_data.processing_stage,
        }

    # 세션이 아직 처리 중인 경우 (status="processing")
    if session_data.status == "waiting":
        logger.info(
            f"[MULTI-ZONE RESPONSE] device_id={device_id}, session_id={session_data.session_id}, "
            f"status=waiting, stage={session_data.processing_stage}"
        )
        failure_reasons = (
            [session_data.failure_reason]
            if getattr(session_data, "failure_reason", None)
            else []
        )
        return {
            "success": False,
            "status": "waiting",
            "message": (
                session_data.processing_stage_detail
                or "Waiting for stable loadcell retry"
            ),
            "reason": "waiting_for_stable_loadcell",
            "waiting_for": "stable_loadcell",
            "device_id": device_id,
            "session_id": session_data.session_id,
            "zone": session_data.zone,
            "processing_stage": session_data.processing_stage,
            "processing_stage_detail": session_data.processing_stage_detail,
            "failureReasons": failure_reasons,
        }

    if session_data.status == "processing":
        logger.info(
            f"[MULTI-ZONE RESPONSE] device_id={device_id}, session_id={session_data.session_id}, "
            f"status=processing, stage={session_data.processing_stage}"
        )
        return {
            "success": False,
            "status": "processing",
            "message": session_data.processing_stage_detail or "처리 중",
            "reason": "in_progress",
            "device_id": device_id,
            "session_id": session_data.session_id,
            "zone": session_data.zone,
            "processing_stage": session_data.processing_stage,
            "processing_stage_detail": session_data.processing_stage_detail,
        }

    # 결과가 있으면 complete 응답
    # 개수가 0인 상품은 제외 (Node.js에서 요청한 상품 목록에 포함되지 않음)
    products = [
        {
            "productIdx": p.product_idx if p.product_idx else str(p.product_id),
            "productId": p.product_id,
            "name": p.name,
            "count": p.count,
            "price": p.price,
            "confidence": round(p.confidence, 4),
        }
        for p in session_data.products
        if p.count > 0  # 개수 0인 상품 필터링
    ]

    # 새 필드: 상품명/개수 배열
    product_names = [p["name"] for p in products]
    product_counts_list = [p["count"] for p in products]

    # 총 상품 개수 계산 (count > 0인 상품만)
    product_count = sum(p.count for p in session_data.products if p.count > 0)

    is_success = session_data.status == "complete"
    has_products = product_count > 0

    logger.info(
        f"[MULTI-ZONE RESPONSE] device_id={device_id}, session_id={session_data.session_id}, "
        f"status=complete, zone={session_data.zone}, products={len(products)}, "
        f"count={product_count}, total={session_data.total_price}, delta={session_data.delta_weight:.1f}g, "
        f"success={is_success}, has_products={has_products}"
    )

    response = {
        "success": is_success,
        "status": "complete",
        "has_products": has_products,
        "device_id": device_id,
        "zone": session_data.zone,
        "session_id": session_data.session_id,
        "processing_stage": session_data.processing_stage,
        "processing_stage_detail": session_data.processing_stage_detail,
        "products": products,
        "productNames": product_names,
        "productCounts": product_counts_list,
        "productCount": product_count,
        "totalPrice": session_data.total_price,
        "weightDelta": round(session_data.delta_weight, 1),
        "confidence": round(session_data.confidence, 4),
        "weightInfo": {
            "delta": round(session_data.delta_weight, 1),
            "isRemoval": session_data.delta_weight < 0,
        },
        "stats": {
            "topFrames": session_data.top_frames,
            "sideFrames": session_data.side_frames,
            "processingTimeMs": round(session_data.processing_time_ms, 1),
        },
    }
    _log_request_to_file(request, response)
    return response


# ============================================================================
# Session Status Endpoint
# ============================================================================


@router.get("/session/{session_id}")
async def get_session_status(
    session_id: str,
    session_store: SessionStore = Depends(get_session_store),
):
    """
    세션 상태 조회 (디버깅용).

    Args:
        session_id: 세션 ID

    Returns:
        세션 상세 정보 or 404
    """
    session_data = session_store.get(session_id)

    if session_data is None:
        return {
            "found": False,
            "session_id": session_id,
            "message": "Session not found or expired",
        }

    return {
        "found": True,
        "session_id": session_id,
        "data": session_data.to_dict(),
    }


# ============================================================================
# Session Store Stats Endpoint
# ============================================================================


@router.get("/sessions/stats")
async def get_sessions_stats(
    session_store: SessionStore = Depends(get_session_store),
    door_session_store: DoorSessionStore | None = Depends(get_door_session_store_optional),
):
    """
    세션 저장소 통계 조회.

    Returns:
        총 세션 수, 활성 세션 수 등
    """
    result = {
        **session_store.get_stats(),
        "timestamp": time.time(),
    }

    if door_session_store is not None:
        result["door_session_store"] = door_session_store.get_stats()

    return result


# ============================================================================
# Door Session Endpoints (v4.1)
# ============================================================================


@router.get("/door-sessions/stats")
async def get_door_sessions_stats(
    door_session_store: DoorSessionStore | None = Depends(get_door_session_store_optional),
):
    """
    Door Session 저장소 통계 조회.

    Returns:
        활성 Door Session 수, YAML 저장 현황 등
    """
    if door_session_store is None:
        return {
            "enabled": False,
            "message": "DoorSessionStore is not enabled",
        }

    return {
        "enabled": True,
        **door_session_store.get_stats(),
        "timestamp": time.time(),
    }


@router.get("/door-session/{zone}")
async def get_door_session_status(
    zone: int,
    door_session_store: DoorSessionStore | None = Depends(get_door_session_store_optional),
):
    """
    특정 Zone의 Door Session 상태 조회.

    Args:
        zone: Zone 번호

    Returns:
        Door Session 상세 정보 or 404
    """
    if door_session_store is None:
        return {
            "found": False,
            "zone": zone,
            "message": "DoorSessionStore is not enabled",
        }

    door_session = door_session_store.get_session(zone)

    if door_session is None:
        return {
            "found": False,
            "zone": zone,
            "message": "No active door session for this zone",
        }

    return {
        "found": True,
        "zone": zone,
        "data": door_session.to_dict(),
    }


@router.post("/door-session/{zone}/finalize")
async def finalize_door_session(
    zone: int,
    door_session_store: DoorSessionStore | None = Depends(get_door_session_store_optional),
):
    """
    Door Session 강제 종료.

    Args:
        zone: Zone 번호

    Returns:
        종료된 Door Session 정보
    """
    if door_session_store is None:
        return {
            "success": False,
            "zone": zone,
            "message": "DoorSessionStore is not enabled",
        }

    door_session = door_session_store.finalize_session(zone)

    if door_session is None:
        return {
            "success": False,
            "zone": zone,
            "message": "No active door session to finalize",
        }

    return {
        "success": True,
        "zone": zone,
        "door_session_id": door_session.door_session_id,
        "trigger_count": door_session.trigger_count,
        "product_count": door_session.product_count,
        "total_price": door_session.total_price,
        "message": "Door session finalized successfully",
    }
