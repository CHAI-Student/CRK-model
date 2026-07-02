"""
Door Session Data Models (v4.9).

Door Session은 문이 열리고 닫힐 때까지의 모든 trigger를 통합 관리합니다.
여러 번의 /trigger 호출이 하나의 DoorSession으로 묶이며,
상품 제거/반환이 누적되어 최종 결과가 계산됩니다.

v4.9 변경사항:
- CrossZoneReturn 추가: 크로스 존 반환 매칭 기록
- DoorSession에 cross_zone_returns 필드 추가
- Zone A에서 꺼낸 상품을 Zone B에 넣으면 Zone A에서 차감

v4.2 변경사항:
- UnmatchedReturn 추가: 무게 매칭 실패 시 추적
- DoorSession에 unmatched_returns 필드 추가

사용법:
    trigger_result = TriggerResult(
        trigger_id="trigger_001",
        session_id="zone_1_260201_143025",
        timestamp=time.time(),
        products=[...],
        delta_weight=-365.0,
        confidence=0.95,
        video_paths={"top": "/path/top.avi"},
        is_return=False,
    )

    door_session = DoorSession(
        door_session_id="door_zone_1_260201_143000",
        zone=1,
    )
    door_session.add_trigger(trigger_result)
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .session_store import ProductResult


@dataclass
class TriggerResult:
    """
    단일 /trigger 호출 결과.

    Camera에서 녹화 완료 후 호출된 /trigger의 YOLO 추론 결과.

    Attributes:
        trigger_id: 트리거 ID (예: "trigger_001")
        session_id: 기존 SessionStore의 session_id (예: "zone_1_260201_143025")
        timestamp: 트리거 발생 시각 (epoch)
        products: YOLO 추론 결과 상품 목록
        delta_weight: 무게 변화량 (g) - 음수=제거, 양수=반환
        confidence: 전체 신뢰도
        video_paths: 비디오 파일 경로 {"top": path, "side": path}
        is_return: 반환 여부 (delta > 0이면 True)
        processing_time_ms: 처리 시간 (ms)
    """
    trigger_id: str
    session_id: str
    timestamp: float
    products: List[ProductResult]
    delta_weight: float
    confidence: float
    video_paths: Dict[str, str]
    is_return: bool = False
    processing_time_ms: float = 0.0
    timing_metadata: Optional[Dict[str, str]] = None
    failure_reason: Optional[str] = None
    return_weight_hints: List[Dict[str, object]] = field(default_factory=list)
    vision_candidates: List[Dict[str, object]] = field(default_factory=list)
    loadcell_diagnostics: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """딕셔너리 변환."""
        return {
            "trigger_id": self.trigger_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "products": [
                {
                    "product_id": p.product_id,
                    "product_idx": p.product_idx,
                    "name": p.name,
                    "count": p.count,
                    "price": p.price,
                    "confidence": p.confidence,
                    "placement_units": [dict(unit) for unit in p.placement_units],
                }
                for p in self.products
            ],
            "delta_weight": self.delta_weight,
            "confidence": self.confidence,
            "video_paths": self.video_paths,
            "is_return": self.is_return,
            "processing_time_ms": self.processing_time_ms,
            "timing_metadata": self.timing_metadata or {},
            "failure_reason": self.failure_reason,
            "return_weight_hints": [dict(hint) for hint in self.return_weight_hints],
            "vision_candidates": [
                dict(candidate) for candidate in self.vision_candidates
            ],
            "loadcell_diagnostics": dict(self.loadcell_diagnostics),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TriggerResult":
        """딕셔너리에서 복원."""
        products = [
            ProductResult(
                product_id=p["product_id"],
                product_idx=p.get("product_idx"),
                name=p["name"],
                count=p["count"],
                price=p["price"],
                confidence=p.get("confidence", 0.0),
                placement_units=[
                    dict(unit)
                    for unit in p.get("placement_units", [])
                    if isinstance(unit, dict)
                ],
            )
            for p in data.get("products", [])
        ]
        return cls(
            trigger_id=data["trigger_id"],
            session_id=data["session_id"],
            timestamp=data["timestamp"],
            products=products,
            delta_weight=data["delta_weight"],
            confidence=data.get("confidence", 0.0),
            video_paths=data.get("video_paths", {}),
            is_return=data.get("is_return", False),
            processing_time_ms=data.get("processing_time_ms", 0.0),
            timing_metadata=data.get("timing_metadata") or None,
            failure_reason=data.get("failure_reason"),
            return_weight_hints=[
                dict(hint)
                for hint in data.get("return_weight_hints", [])
                if isinstance(hint, dict)
            ],
            vision_candidates=[
                dict(candidate)
                for candidate in data.get("vision_candidates", [])
                if isinstance(candidate, dict)
            ],
            loadcell_diagnostics=(
                dict(data.get("loadcell_diagnostics", {}))
                if isinstance(data.get("loadcell_diagnostics", {}), dict)
                else {}
            ),
        )


def _candidate_value(candidate: object, key: str, default: Any = None) -> Any:
    if isinstance(candidate, dict):
        return candidate.get(key, default)
    return getattr(candidate, key, default)


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _active_product_class_id(product: object) -> Optional[int]:
    class_id = getattr(product, "yolo_class_id", None)
    if class_id is None:
        class_id = getattr(product, "product_id", None)
    if class_id is None:
        return None
    return _coerce_int(class_id)


def _active_product_name(product: object, fallback: str) -> str:
    return str(
        getattr(
            product,
            "product_name",
            getattr(product, "name", fallback),
        )
        or fallback
    )


def build_trigger_candidate_snapshot(
    vision_candidates: Sequence[object],
    active_products: Optional[Sequence[object]],
) -> List[Dict[str, object]]:
    """Return compact candidate metadata for close-time basket validation."""
    active_by_class: Dict[int, object] = {}
    for product in active_products or []:
        class_id = _active_product_class_id(product)
        if class_id is not None:
            active_by_class[class_id] = product

    snapshots: List[Dict[str, object]] = []
    for rank, candidate in enumerate(vision_candidates or [], start=1):
        class_id = _coerce_int(_candidate_value(candidate, "class_id", None), -1)
        if class_id < 0:
            continue

        class_name = str(_candidate_value(candidate, "class_name", "") or "")
        product = active_by_class.get(class_id)
        if product is None:
            continue

        unit_weight = _coerce_float(
            getattr(product, "product_weight", getattr(product, "weight", 0.0)),
        )
        stock_qty = _coerce_int(getattr(product, "stock_qty", getattr(product, "stock", 0)))
        if unit_weight <= 0 or stock_qty <= 0:
            continue

        unit_price = _coerce_int(
            getattr(product, "sale_price", getattr(product, "price", 0)),
        )
        confidence = _coerce_float(
            _candidate_value(
                candidate,
                "combined_confidence",
                _candidate_value(candidate, "weighted_confidence", 0.0),
            )
        )
        top_confidence = _coerce_float(_candidate_value(candidate, "top_confidence", 0.0))
        side_confidence = _coerce_float(_candidate_value(candidate, "side_confidence", 0.0))
        identity_confidence = max(top_confidence, side_confidence)
        if identity_confidence <= 0.0:
            identity_confidence = _coerce_float(
                _candidate_value(candidate, "identity_confidence", confidence),
            )

        snapshots.append(
            {
                "rank": rank,
                "product_id": class_id,
                "product_idx": getattr(product, "product_idx", None),
                "name": _active_product_name(product, class_name),
                "unit_weight": round(unit_weight, 3),
                "unit_price": unit_price,
                "stock_qty": stock_qty,
                "confidence": round(confidence, 4),
                "identity_confidence": round(identity_confidence, 4),
                "source": str(_candidate_value(candidate, "source", "vision") or "vision"),
                "top": bool(top_confidence > 0),
                "side": bool(side_confidence > 0),
                "top_confidence": round(top_confidence, 4),
                "side_confidence": round(side_confidence, 4),
                "vote_count": _coerce_int(_candidate_value(candidate, "vote_count", 0)),
                "raw_vote_count": _coerce_int(
                    _candidate_value(candidate, "raw_vote_count", 0)
                ),
                "freezer_exit_path_votes": _coerce_int(
                    _candidate_value(
                        candidate,
                        "freezer_exit_path_votes",
                        _candidate_value(candidate, "freezerExitPathVotes", 0),
                    )
                ),
                "weight_gate_passed": _candidate_value(
                    candidate,
                    "weight_gate_passed",
                    None,
                ),
            }
        )
    return snapshots


def return_hint_delta_weight(hint: object) -> float:
    """Return a positive delta contribution for an internal return hint."""
    try:
        if isinstance(hint, dict):
            raw_value = hint.get("delta", hint.get("weight", 0.0))
        else:
            raw_value = hint
        return abs(float(raw_value))
    except (TypeError, ValueError):
        return 0.0


def trigger_effective_delta_weight(trigger: TriggerResult) -> float:
    """Return trigger delta including mixed return hints."""
    return float(trigger.delta_weight) + sum(
        return_hint_delta_weight(hint)
        for hint in getattr(trigger, "return_weight_hints", []) or []
    )


def unmatched_return_delta_weight(session: object) -> float:
    """Return still-unmatched return weight that should not affect basket delta."""
    total = 0.0
    for record in getattr(session, "unmatched_returns", []) or []:
        try:
            total += abs(float(getattr(record, "delta_weight", 0.0)))
        except (TypeError, ValueError):
            continue
    return total


def close_weight_delta_override(session: object) -> Optional[float]:
    """Return a close-time response weight override when one was recorded."""
    validation = getattr(session, "final_weight_validation", {}) or {}
    if not isinstance(validation, dict):
        return None
    aggregate = validation.get("freezerCloseAggregate")
    if not isinstance(aggregate, dict) or "weightDeltaOverride" not in aggregate:
        return None
    try:
        return float(aggregate["weightDeltaOverride"])
    except (TypeError, ValueError):
        return None


@dataclass
class UnmatchedReturn:
    """
    무게 매칭 실패한 반환 기록 (v4.2).

    무게 증가(반환)가 감지되었으나 매칭되는 상품을 찾지 못한 경우.

    Attributes:
        trigger_id: 해당 trigger ID
        delta_weight: 반환 무게 (g)
        timestamp: 반환 시각 (epoch)
        tolerance_used: 매칭 시 사용한 허용 오차 (g)
    """
    trigger_id: str
    delta_weight: float
    timestamp: float
    tolerance_used: float = 3.0
    channel_side: Optional[str] = None
    channel_index: Optional[int] = None
    channel_position: Optional[int] = None
    source_zone: Optional[int] = None
    source: str = "positive_return"

    def to_dict(self) -> dict:
        """딕셔너리 변환."""
        return {
            "trigger_id": self.trigger_id,
            "delta_weight": self.delta_weight,
            "timestamp": self.timestamp,
            "tolerance_used": self.tolerance_used,
            "channel_side": self.channel_side,
            "channel_index": self.channel_index,
            "channel_position": self.channel_position,
            "source_zone": self.source_zone,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UnmatchedReturn":
        """딕셔너리에서 복원."""
        return cls(
            trigger_id=data["trigger_id"],
            delta_weight=data["delta_weight"],
            timestamp=data["timestamp"],
            tolerance_used=data.get("tolerance_used", 3.0),
            channel_side=data.get("channel_side"),
            channel_index=data.get("channel_index"),
            channel_position=data.get("channel_position"),
            source_zone=data.get("source_zone"),
            source=data.get("source", "positive_return"),
        )


@dataclass
class DeferredReturn:
    """Return delta kept for close-time basket reconciliation."""

    trigger_id: str
    delta_weight: float
    timestamp: float
    source: str = "positive_return"
    replay_position: str = "return"
    tolerance_used: float = 5.0
    channel_side: Optional[str] = None
    channel_index: Optional[int] = None
    channel_position: Optional[int] = None
    source_zone: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "trigger_id": self.trigger_id,
            "delta_weight": self.delta_weight,
            "timestamp": self.timestamp,
            "source": self.source,
            "replay_position": self.replay_position,
            "tolerance_used": self.tolerance_used,
            "channel_side": self.channel_side,
            "channel_index": self.channel_index,
            "channel_position": self.channel_position,
            "source_zone": self.source_zone,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DeferredReturn":
        return cls(
            trigger_id=data["trigger_id"],
            delta_weight=data["delta_weight"],
            timestamp=data["timestamp"],
            source=data.get("source", "positive_return"),
            replay_position=data.get("replay_position", "return"),
            tolerance_used=data.get("tolerance_used", 5.0),
            channel_side=data.get("channel_side"),
            channel_index=data.get("channel_index"),
            channel_position=data.get("channel_position"),
            source_zone=data.get("source_zone"),
        )


@dataclass
class CrossZoneReturn:
    """
    크로스 존 반환 매칭 기록 (v4.9).

    무게 증가(반환)가 다른 zone의 상품과 매칭된 경우.

    Attributes:
        trigger_id: 반환 trigger ID
        source_zone: 반환 발생 zone (무게 증가)
        target_zone: 매칭된 zone (상품 차감)
        product_id: 매칭된 상품 ID (YOLO class_id)
        product_name: 상품명
        matched_weight: 매칭된 상품 무게 (g)
        delta_weight: 실제 반환 무게 (g)
        timestamp: 매칭 시각 (epoch)
    """
    trigger_id: str
    source_zone: int
    target_zone: int
    product_id: int
    product_name: str
    matched_weight: float
    delta_weight: float
    timestamp: float

    def to_dict(self) -> dict:
        """딕셔너리 변환."""
        return {
            "trigger_id": self.trigger_id,
            "source_zone": self.source_zone,
            "target_zone": self.target_zone,
            "product_id": self.product_id,
            "product_name": self.product_name,
            "matched_weight": self.matched_weight,
            "delta_weight": self.delta_weight,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CrossZoneReturn":
        """딕셔너리에서 복원."""
        return cls(
            trigger_id=data["trigger_id"],
            source_zone=data["source_zone"],
            target_zone=data["target_zone"],
            product_id=data["product_id"],
            product_name=data["product_name"],
            matched_weight=data["matched_weight"],
            delta_weight=data["delta_weight"],
            timestamp=data["timestamp"],
        )


@dataclass
class AggregatedProduct:
    """
    통합 상품 결과.

    여러 trigger에서 감지된 동일 상품을 합산한 결과.

    Attributes:
        product_id: YOLO class_id (내부용)
        product_idx: IF11 product_idx (Node.js 응답용)
        name: 상품명
        count: 수량 (합산 결과, 반환 시 차감)
        unit_price: 단가 (원)
        weight: 단위 무게 (g)
        total_confidence: 누적 신뢰도
        detection_count: 감지 횟수
    """
    product_id: int
    product_idx: Optional[str]
    name: str
    count: int
    unit_price: int
    weight: float
    total_confidence: float = 0.0
    detection_count: int = 0
    placement_units: List[Dict[str, object]] = field(default_factory=list)

    @property
    def total_price(self) -> int:
        """총 가격 계산."""
        return self.unit_price * self.count

    @property
    def average_confidence(self) -> float:
        """평균 신뢰도."""
        if self.detection_count == 0:
            return 0.0
        return self.total_confidence / self.detection_count

    def to_dict(self) -> dict:
        """딕셔너리 변환."""
        return {
            "product_id": self.product_id,
            "product_idx": self.product_idx,
            "name": self.name,
            "count": self.count,
            "unit_price": self.unit_price,
            "weight": self.weight,
            "total_price": self.total_price,
            "average_confidence": round(self.average_confidence, 4),
            "detection_count": self.detection_count,
            "placement_units": [dict(unit) for unit in self.placement_units],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AggregatedProduct":
        """딕셔너리에서 복원."""
        return cls(
            product_id=data["product_id"],
            product_idx=data.get("product_idx"),
            name=data["name"],
            count=data["count"],
            unit_price=data["unit_price"],
            weight=data["weight"],
            total_confidence=data.get("total_confidence", 0.0),
            detection_count=data.get("detection_count", 0),
            placement_units=[
                dict(unit)
                for unit in data.get("placement_units", [])
                if isinstance(unit, dict)
            ],
        )


@dataclass
class DoorSession:
    """
    Door Session - 문 열림~닫힘 동안의 모든 trigger 통합 관리.

    Attributes:
        door_session_id: Door Session ID (예: "door_zone_1_260201_143000")
        zone: Zone 번호
        status: 세션 상태 ("active" | "complete")
        triggers: TriggerResult 목록 (시간순)
        aggregated_products: 통합 상품 결과 (product_id -> AggregatedProduct)
        unmatched_returns: 무게 매칭 실패한 반환 목록 (v4.2)
        cross_zone_returns: 크로스 존 반환 매칭 기록 (v4.9)
        created_at: 세션 생성 시각 (epoch)
        last_trigger_at: 마지막 trigger 시각 (타임아웃 계산용)
        finalized_at: 세션 종료 시각 (complete일 때만)
    """
    door_session_id: str
    zone: int
    status: str = "active"
    triggers: List[TriggerResult] = field(default_factory=list)
    aggregated_products: Dict[int, AggregatedProduct] = field(default_factory=dict)
    unmatched_returns: List[UnmatchedReturn] = field(default_factory=list)
    deferred_returns: List[DeferredReturn] = field(default_factory=list)
    cross_zone_returns: List[CrossZoneReturn] = field(default_factory=list)
    final_weight_validation: Dict[str, object] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_trigger_at: float = field(default_factory=time.time)
    finalized_at: Optional[float] = None

    @property
    def trigger_count(self) -> int:
        """Trigger 수."""
        return len(self.triggers)

    @property
    def total_price(self) -> int:
        """총 금액 (count > 0인 상품만)."""
        return sum(
            p.total_price
            for p in self.aggregated_products.values()
            if p.count > 0
        )

    @property
    def product_count(self) -> int:
        """총 상품 수 (count > 0인 상품만)."""
        return sum(
            p.count
            for p in self.aggregated_products.values()
            if p.count > 0
        )

    @property
    def duration_seconds(self) -> float:
        """세션 지속 시간 (초)."""
        end_time = self.finalized_at or time.time()
        return end_time - self.created_at

    @property
    def is_active(self) -> bool:
        """활성 상태 여부."""
        return self.status == "active"

    def get_active_products(self) -> List[AggregatedProduct]:
        """count > 0인 상품 목록 반환."""
        return [p for p in self.aggregated_products.values() if p.count > 0]

    @property
    def has_unmatched_returns(self) -> bool:
        """매칭 실패한 반환이 있는지 여부 (v4.2)."""
        return len(self.unmatched_returns) > 0

    @property
    def unmatched_returns_weight(self) -> float:
        """매칭 실패한 반환 총 무게 (g) (v4.2)."""
        return sum(r.delta_weight for r in self.unmatched_returns)

    @property
    def deferred_returns_weight(self) -> float:
        """Return weight reserved for close-time reconciliation."""
        return sum(abs(r.delta_weight) for r in self.deferred_returns)

    @property
    def has_cross_zone_returns(self) -> bool:
        """크로스 존 반환이 있는지 여부 (v4.9)."""
        return len(self.cross_zone_returns) > 0

    @property
    def cross_zone_returns_count(self) -> int:
        """크로스 존 반환 횟수 (v4.9)."""
        return len(self.cross_zone_returns)

    def to_dict(self) -> dict:
        """딕셔너리 변환 (YAML 저장용)."""
        return {
            "door_session_id": self.door_session_id,
            "zone": self.zone,
            "status": self.status,
            "triggers": [t.to_dict() for t in self.triggers],
            "aggregated_products": {
                str(pid): p.to_dict()
                for pid, p in self.aggregated_products.items()
            },
            "unmatched_returns": [r.to_dict() for r in self.unmatched_returns],
            "deferred_returns": [r.to_dict() for r in self.deferred_returns],
            "cross_zone_returns": [r.to_dict() for r in self.cross_zone_returns],
            "final_weight_validation": dict(self.final_weight_validation),
            "created_at": self.created_at,
            "last_trigger_at": self.last_trigger_at,
            "finalized_at": self.finalized_at,
            "summary": {
                "trigger_count": self.trigger_count,
                "total_price": self.total_price,
                "product_count": self.product_count,
                "duration_seconds": round(self.duration_seconds, 1),
                "unmatched_returns_count": len(self.unmatched_returns),
                "unmatched_returns_weight": round(self.unmatched_returns_weight, 1),
                "deferred_returns_count": len(self.deferred_returns),
                "deferred_returns_weight": round(self.deferred_returns_weight, 1),
                "cross_zone_returns_count": self.cross_zone_returns_count,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DoorSession":
        """딕셔너리에서 복원."""
        triggers = [
            TriggerResult.from_dict(t)
            for t in data.get("triggers", [])
        ]
        aggregated_products = {
            int(pid): AggregatedProduct.from_dict(p)
            for pid, p in data.get("aggregated_products", {}).items()
        }
        unmatched_returns = [
            UnmatchedReturn.from_dict(r)
            for r in data.get("unmatched_returns", [])
        ]
        deferred_returns = [
            DeferredReturn.from_dict(r)
            for r in data.get("deferred_returns", [])
        ]
        cross_zone_returns = [
            CrossZoneReturn.from_dict(r)
            for r in data.get("cross_zone_returns", [])
        ]
        return cls(
            door_session_id=data["door_session_id"],
            zone=data["zone"],
            status=data.get("status", "active"),
            triggers=triggers,
            aggregated_products=aggregated_products,
            unmatched_returns=unmatched_returns,
            deferred_returns=deferred_returns,
            cross_zone_returns=cross_zone_returns,
            final_weight_validation=dict(data.get("final_weight_validation", {})),
            created_at=data.get("created_at", time.time()),
            last_trigger_at=data.get("last_trigger_at", time.time()),
            finalized_at=data.get("finalized_at"),
        )


def generate_door_session_id(zone: int) -> str:
    """
    Door Session ID 생성.

    Format: door_zone_{zone}_{YYMMDD}_{HHMMSS}_{ffffff}

    Args:
        zone: Zone 번호

    Returns:
        Door Session ID (예: door_zone_1_260201_143000_123456)
    """
    from datetime import datetime
    now = datetime.now()
    return f"door_zone_{zone}_{now.strftime('%y%m%d_%H%M%S_%f')}"
