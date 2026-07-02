"""
Product Aggregator for Door Session (v4.2).

여러 TriggerResult의 상품을 통합하고, 반환 처리를 수행합니다.

합산 로직:
1. 제거(delta<0): 상품 count 증가
2. 반환(delta>0): 무게 매칭하여 count 감소

v4.2 변경사항:
- 무게 매칭 실패 시 UnmatchedReturn으로 추적
- AggregationResult 반환으로 unmatched_returns 포함

사용법:
    aggregator = ProductAggregator(weight_tolerance=3.0)
    result = aggregator.aggregate(triggers)

    # 결과 접근
    products = result.products
    unmatched = result.unmatched_returns
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from model_service.core.config import config
from model_service.weight.strict_weight_matcher import StrictWeightMatcher

from .door_session import (
    AggregatedProduct,
    DeferredReturn,
    ReturnedPositionHint,
    TriggerResult,
    UnmatchedReturn,
    return_hint_delta_weight,
)
from .session_store import ProductResult

logger = logging.getLogger(__name__)


@dataclass
class AggregationResult:
    """
    상품 집계 결과 (v4.2).

    Attributes:
        products: 통합된 상품 (product_id -> AggregatedProduct)
        unmatched_returns: 매칭 실패한 반환 목록
    """
    products: Dict[int, AggregatedProduct] = field(default_factory=dict)
    unmatched_returns: List[UnmatchedReturn] = field(default_factory=list)
    deferred_returns: List[DeferredReturn] = field(default_factory=list)
    location_return_diagnostics: List[dict[str, object]] = field(default_factory=list)
    returned_position_hints: List[ReturnedPositionHint] = field(default_factory=list)


class ProductAggregator:
    """
    상품 통합 및 반환 처리.

    여러 trigger에서 감지된 상품을 합산하고,
    무게 증가(반환) 시 해당 무게의 상품을 차감합니다.
    """

    def __init__(
        self,
        weight_tolerance: float = 3.0,
        get_product_weight: Optional[Callable[[int], float]] = None,
    ):
        """
        Initialize ProductAggregator.

        Args:
            weight_tolerance: 무게 매칭 허용 오차 (g)
            get_product_weight: product_id -> weight 조회 함수 (없으면 aggregated에서 조회)
        """
        self._weight_tolerance = weight_tolerance
        self._get_product_weight = get_product_weight
        self._weight_matcher = StrictWeightMatcher(tolerance=weight_tolerance)

    def aggregate(
        self,
        triggers: List[TriggerResult],
        *,
        zone: Optional[int] = None,
    ) -> Dict[int, AggregatedProduct]:
        """
        여러 TriggerResult의 상품을 통합.

        합산 로직:
        1. 제거(delta<0): 상품 count 증가 (YOLO 결과 사용)
        2. 반환(delta>0): 무게 매칭하여 count 감소

        Args:
            triggers: TriggerResult 목록 (시간순)

        Returns:
            통합된 상품 결과 (product_id -> AggregatedProduct)

        Note:
            하위 호환성을 위해 Dict만 반환합니다.
            unmatched_returns가 필요하면 aggregate_with_unmatched() 사용.
        """
        result = self.aggregate_with_unmatched(triggers, zone=zone)
        return result.products

    def aggregate_with_unmatched(
        self,
        triggers: List[TriggerResult],
        *,
        zone: Optional[int] = None,
    ) -> AggregationResult:
        """
        여러 TriggerResult의 상품을 통합 (unmatched_returns 포함, v4.2).

        합산 로직:
        1. 제거(delta<0): 상품 count 증가 (YOLO 결과 사용)
        2. 반환(delta>0): 무게 매칭하여 count 감소
        3. 매칭 실패 시 unmatched_returns에 기록

        Args:
            triggers: TriggerResult 목록 (시간순)

        Returns:
            AggregationResult (products + unmatched_returns)
        """
        aggregated: Dict[int, AggregatedProduct] = {}
        unmatched_returns: List[UnmatchedReturn] = []
        deferred_returns: List[DeferredReturn] = []
        location_return_diagnostics: List[dict[str, object]] = []
        returned_position_hints: List[ReturnedPositionHint] = []

        # Replay by event timestamp rather than completion order. Loadcell-only
        # returns can finish before video-backed removals that happened earlier.
        ordered_triggers = [
            trigger
            for _, trigger in sorted(
                enumerate(triggers),
                key=lambda item: (float(item[1].timestamp), item[0]),
            )
        ]
        for trigger in ordered_triggers:
            if trigger.is_return:
                # 반환 처리: 무게 매칭하여 차감
                matched_id = self._handle_freezer_location_return(
                    aggregated,
                    unmatched_returns,
                    trigger,
                    source_zone=zone,
                    diagnostics=location_return_diagnostics,
                    returned_position_hints=returned_position_hints,
                )
                if matched_id is None:
                    matched_id = self._handle_return(
                        aggregated,
                        trigger.delta_weight,
                        trigger=trigger,
                        source_zone=zone,
                        returned_position_hints=returned_position_hints,
                    )
                if matched_id is None:
                    # 매칭 실패 → 기록
                    self._record_deferred_return(
                        deferred_returns,
                        trigger,
                        source="positive_return",
                        replay_position="return",
                    )
            else:
                # 제거 처리: YOLO 결과 합산
                self._handle_return_hints(
                    deferred_returns,
                    trigger,
                    replay_position="before_removal",
                )
                self._handle_removal(aggregated, trigger)
                self._handle_return_hints(
                    deferred_returns,
                    trigger,
                    replay_position="after_removal",
                )

        logger.debug(
            f"Aggregated {len(triggers)} triggers -> {len(aggregated)} products, "
            f"total count={sum(p.count for p in aggregated.values())}, "
            f"unmatched_returns={len(unmatched_returns)}"
        )

        return AggregationResult(
            products=aggregated,
            unmatched_returns=unmatched_returns,
            deferred_returns=deferred_returns,
            location_return_diagnostics=location_return_diagnostics,
            returned_position_hints=returned_position_hints,
        )

    def _record_unmatched_return(
        self,
        unmatched_returns: List[UnmatchedReturn],
        trigger: TriggerResult,
    ) -> None:
        """Persist an unmatched return so later recovery passes can inspect it."""
        self._record_unmatched_return_delta(
            unmatched_returns,
            trigger,
            trigger.delta_weight,
        )

    def _record_unmatched_return_delta(
        self,
        unmatched_returns: List[UnmatchedReturn],
        trigger: TriggerResult,
        delta_weight: float,
        *,
        target: Optional[dict[str, object]] = None,
        source_zone: Optional[int] = None,
        source: str = "positive_return",
    ) -> None:
        """Persist an unmatched return delta tied to the source trigger."""
        unmatched_returns.append(
            UnmatchedReturn(
                trigger_id=trigger.trigger_id,
                delta_weight=delta_weight,
                timestamp=trigger.timestamp,
                tolerance_used=self._weight_tolerance,
                channel_side=(
                    str(target.get("channel_side"))
                    if isinstance(target, dict) and target.get("channel_side") is not None
                    else None
                ),
                channel_index=(
                    int(target["channel_index"])
                    if isinstance(target, dict) and target.get("channel_index") is not None
                    else None
                ),
                channel_position=(
                    int(target["channel_position"])
                    if isinstance(target, dict) and target.get("channel_position") is not None
                    else None
                ),
                source_zone=source_zone,
                source=source,
            )
        )

    def _record_deferred_return(
        self,
        deferred_returns: List[DeferredReturn],
        trigger: TriggerResult,
        *,
        source: str,
        replay_position: str,
        delta_weight: Optional[float] = None,
        target: Optional[dict[str, object]] = None,
        source_zone: Optional[int] = None,
    ) -> None:
        """Persist a return delta for CLOSE-only reconciliation."""
        deferred_returns.append(
            DeferredReturn(
                trigger_id=trigger.trigger_id,
                delta_weight=(
                    trigger.delta_weight if delta_weight is None else delta_weight
                ),
                timestamp=trigger.timestamp,
                source=source,
                replay_position=replay_position,
                tolerance_used=self._weight_tolerance,
                channel_side=(
                    str(target.get("channel_side"))
                    if isinstance(target, dict) and target.get("channel_side") is not None
                    else None
                ),
                channel_index=(
                    int(target["channel_index"])
                    if isinstance(target, dict) and target.get("channel_index") is not None
                    else None
                ),
                channel_position=(
                    int(target["channel_position"])
                    if isinstance(target, dict) and target.get("channel_position") is not None
                    else None
                ),
                source_zone=source_zone,
            )
        )

    def _append_returned_position_hint(
        self,
        returned_position_hints: Optional[List[ReturnedPositionHint]],
        *,
        product: AggregatedProduct,
        trigger: TriggerResult,
        count: int,
        source_zone: Optional[int],
        source: str,
        reason: str,
        target: Optional[dict[str, object]] = None,
        selected_units: Optional[List[dict[str, object]]] = None,
    ) -> None:
        if returned_position_hints is None or count <= 0:
            return
        unit = selected_units[0] if selected_units else None
        channel_side = None
        channel_index = None
        channel_position = None
        if isinstance(target, dict):
            channel_side = target.get("channel_side") or target.get("channelSide")
            channel_index = target.get("channel_index") or target.get("channelIndex")
            channel_position = (
                target.get("channel_position")
                if target.get("channel_position") is not None
                else target.get("channelPosition")
            )
        if isinstance(unit, dict):
            channel_side = (
                channel_side
                or unit.get("channelSide")
                or unit.get("channel_side")
            )
            channel_index = (
                channel_index
                if channel_index is not None
                else unit.get("channelIndex", unit.get("channel_index"))
            )
            channel_position = (
                channel_position
                if channel_position is not None
                else unit.get("channelPosition", unit.get("channel_position"))
            )
        try:
            parsed_channel_index = (
                int(channel_index) if channel_index is not None else None
            )
        except (TypeError, ValueError):
            parsed_channel_index = None
        try:
            parsed_channel_position = (
                int(channel_position) if channel_position is not None else None
            )
        except (TypeError, ValueError):
            parsed_channel_position = None
        returned_position_hints.append(
            ReturnedPositionHint(
                product_id=int(product.product_id),
                product_idx=product.product_idx,
                name=product.name,
                unit_weight=round(float(product.weight), 1),
                count=int(count),
                zone=source_zone,
                trigger_id=trigger.trigger_id,
                session_id=trigger.session_id,
                timestamp=float(trigger.timestamp),
                source=source,
                confidence=float(product.average_confidence),
                reason=reason,
                channel_side=str(channel_side) if channel_side is not None else None,
                channel_index=parsed_channel_index,
                channel_position=parsed_channel_position,
            )
        )

    def _handle_return_hints(
        self,
        deferred_returns: List[DeferredReturn],
        trigger: TriggerResult,
        *,
        replay_position: str,
    ) -> None:
        for hint in getattr(trigger, "return_weight_hints", []) or []:
            if not isinstance(hint, dict):
                continue
            hint_position = str(hint.get("replay_position", "before_removal"))
            if hint_position != replay_position:
                continue
            delta_weight = return_hint_delta_weight(hint)
            if delta_weight <= 0:
                continue
            source = str(hint.get("source", "mixed_return_hint"))
            self._record_deferred_return(
                deferred_returns,
                trigger,
                source=source or "mixed_return_hint",
                replay_position=hint_position,
                delta_weight=delta_weight,
            )

    @staticmethod
    def _estimate_return_count(unit_weight: float, delta_weight: float, count: int) -> int:
        """Estimate how many units were returned while avoiding over-deduction.

        When the inferred count is larger than the currently aggregated count,
        the delta is treated as noisy and only a single unit is rolled back.
        """
        if unit_weight <= 0:
            return 1

        estimated_count = max(1, round(delta_weight / unit_weight))
        if estimated_count > count:
            return 1
        return estimated_count

    def _placement_units_for_product(
        self,
        product: ProductResult,
        trigger: TriggerResult,
    ) -> List[dict[str, object]]:
        count = max(0, int(product.count))
        units = [
            dict(unit)
            for unit in getattr(product, "placement_units", []) or []
            if isinstance(unit, dict)
        ][:count]
        unit_weight = self._get_weight_for_product(product)
        while len(units) < count:
            units.append(
                {
                    "product_id": int(product.product_id),
                    "product_idx": product.product_idx,
                    "name": product.name,
                    "unitWeight": round(float(unit_weight), 1),
                    "zone": None,
                    "sourceTriggerId": trigger.trigger_id,
                    "sourceSessionId": trigger.session_id,
                    "sourceTimestamp": trigger.timestamp,
                    "channelSide": "unknown",
                    "source": "aggregator_synthesized",
                }
            )
        for unit in units:
            unit.setdefault("product_id", int(product.product_id))
            unit.setdefault("product_idx", product.product_idx)
            unit.setdefault("name", product.name)
            unit.setdefault("unitWeight", round(float(unit_weight), 1))
            unit.setdefault("sourceTriggerId", trigger.trigger_id)
            unit.setdefault("sourceSessionId", trigger.session_id)
            unit.setdefault("sourceTimestamp", trigger.timestamp)
            unit.setdefault("channelSide", "unknown")
        return units

    @staticmethod
    def _remove_product_units(
        product: AggregatedProduct,
        count: int,
        *,
        selected_units: Optional[List[dict[str, object]]] = None,
    ) -> None:
        if count <= 0:
            return
        if selected_units:
            remaining = list(product.placement_units)
            for selected in selected_units[:count]:
                remove_index: Optional[int] = None
                for index, unit in enumerate(remaining):
                    if unit is selected or unit == selected:
                        remove_index = index
                        break
                if remove_index is not None:
                    remaining.pop(remove_index)
                elif remaining:
                    remaining.pop()
            product.placement_units = remaining
            return
        product.placement_units = list(product.placement_units)[: max(0, product.count)]

    def _handle_removal(
        self,
        aggregated: Dict[int, AggregatedProduct],
        trigger: TriggerResult,
    ) -> None:
        """
        제거 처리: YOLO 결과 합산.

        Args:
            aggregated: 통합 상품 딕셔너리 (in-place 수정)
            trigger: 제거 trigger 결과
        """
        for product in trigger.products:
            if product.count <= 0:
                continue

            product_id = product.product_id
            placement_units = self._placement_units_for_product(product, trigger)

            if product_id in aggregated:
                # 기존 상품에 합산
                agg = aggregated[product_id]
                agg.count += product.count
                agg.total_confidence += product.confidence * product.count
                agg.detection_count += product.count
                agg.placement_units.extend(placement_units)
            else:
                # 새 상품 추가
                aggregated[product_id] = AggregatedProduct(
                    product_id=product_id,
                    product_idx=product.product_idx,
                    name=product.name,
                    count=product.count,
                    unit_price=product.price,
                    weight=self._get_weight_for_product(product),
                    total_confidence=product.confidence * product.count,
                    detection_count=product.count,
                    placement_units=placement_units,
                )

            logger.debug(
                f"Added product: {product.name} x{product.count} "
                f"(id={product_id}, total={aggregated[product_id].count})"
            )

    @staticmethod
    def _unit_channel_side(unit: dict[str, object]) -> str:
        return str(unit.get("channelSide") or unit.get("channel_side") or "unknown")

    @staticmethod
    def _unit_zone(unit: dict[str, object]) -> Optional[int]:
        try:
            raw_zone = unit.get("zone")
            return int(raw_zone) if raw_zone is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _positive_channel_return_targets(
        trigger: TriggerResult,
    ) -> List[dict[str, object]]:
        diagnostics = getattr(trigger, "loadcell_diagnostics", {}) or {}
        if not isinstance(diagnostics, dict):
            return []
        targets: List[dict[str, object]] = []
        for entry in diagnostics.get("channel_movement_targets") or []:
            if not isinstance(entry, dict):
                continue
            direction = str(entry.get("direction", "")).lower()
            try:
                delta = float(entry.get("delta", 0.0) or 0.0)
                weight = abs(float(entry.get("weight", delta) or delta))
            except (TypeError, ValueError):
                continue
            if weight <= 0:
                continue
            if direction == "return" or delta > 0:
                target = dict(entry)
                target["weight"] = round(weight, 1)
                targets.append(target)
        return targets

    def _freezer_return_tolerance(self) -> float:
        return max(
            float(self._weight_tolerance),
            float(config.weight.freezer_weight_tolerance_grams),
        )

    def _find_same_zone_location_return_match(
        self,
        aggregated: Dict[int, AggregatedProduct],
        *,
        target: dict[str, object],
        source_zone: Optional[int],
    ) -> Optional[dict[str, object]]:
        target_weight = abs(float(target.get("weight", 0.0) or 0.0))
        if target_weight <= 0:
            return None
        target_side = str(target.get("channel_side") or "unknown")
        tolerance = self._freezer_return_tolerance()
        tiers = [
            ("same_zone_same_side", True),
            ("same_zone_other_side", False),
        ]
        for tier_name, same_side in tiers:
            matches: List[dict[str, object]] = []
            for product_id, product in aggregated.items():
                if product.count <= 0 or product.weight <= 0:
                    continue
                product_units = [
                    unit
                    for unit in product.placement_units
                    if isinstance(unit, dict)
                    and (
                        source_zone is None
                        or self._unit_zone(unit) in {None, int(source_zone)}
                    )
                    and (
                        self._unit_channel_side(unit) == target_side
                        if same_side
                        else self._unit_channel_side(unit) != target_side
                    )
                ]
                if not product_units:
                    continue
                product_units = sorted(
                    product_units,
                    key=lambda unit: float(unit.get("sourceTimestamp", 0.0) or 0.0),
                    reverse=True,
                )
                max_count = min(product.count, len(product_units))
                for count in range(1, max_count + 1):
                    expected = float(product.weight) * count
                    residual = abs(expected - target_weight)
                    if residual > tolerance:
                        continue
                    latest_timestamp = max(
                        float(unit.get("sourceTimestamp", 0.0) or 0.0)
                        for unit in product_units[:count]
                    )
                    matches.append(
                        {
                            "product_id": int(product_id),
                            "product": product,
                            "count": int(count),
                            "expected": expected,
                            "residual": residual,
                            "tier": tier_name,
                            "units": product_units[:count],
                            "latestTimestamp": latest_timestamp,
                        }
                    )
            if matches:
                return min(
                    matches,
                    key=lambda item: (
                        int(item["count"]),
                        float(item["residual"]),
                        -float(item["latestTimestamp"]),
                    ),
                )
        return None

    def _handle_freezer_location_return(
        self,
        aggregated: Dict[int, AggregatedProduct],
        unmatched_returns: List[UnmatchedReturn],
        trigger: TriggerResult,
        *,
        source_zone: Optional[int],
        diagnostics: List[dict[str, object]],
        returned_position_hints: Optional[List[ReturnedPositionHint]] = None,
    ) -> Optional[int]:
        targets = self._positive_channel_return_targets(trigger)
        if not targets:
            return None

        reconciliation = {
            "triggerId": trigger.trigger_id,
            "sourceZone": source_zone,
            "policy": "freezer_location_return_same_zone_first",
            "targets": [],
        }
        matched_product_ids: List[int] = []
        for target in targets:
            target_diag = {
                "channelSide": target.get("channel_side"),
                "channelIndex": target.get("channel_index"),
                "channelPosition": target.get("channel_position"),
                "targetWeight": round(float(target.get("weight", 0.0) or 0.0), 1),
                "accepted": False,
            }
            match = self._find_same_zone_location_return_match(
                aggregated,
                target=target,
                source_zone=source_zone,
            )
            if match is None:
                self._record_unmatched_return_delta(
                    unmatched_returns,
                    trigger,
                    float(target.get("weight", trigger.delta_weight) or trigger.delta_weight),
                    target=target,
                    source_zone=source_zone,
                    source="freezer_location_return",
                )
                target_diag["reason"] = "no_same_zone_location_match"
                reconciliation["targets"].append(target_diag)
                continue

            product = match["product"]
            count = int(match["count"])
            product.count = max(0, int(product.count) - count)
            self._append_returned_position_hint(
                returned_position_hints,
                product=product,
                trigger=trigger,
                count=count,
                source_zone=source_zone,
                source="freezer_location_return",
                reason="same_zone_location_return_reconciled",
                target=target,
                selected_units=match["units"],
            )
            self._remove_product_units(
                product,
                count,
                selected_units=match["units"],
            )
            matched_product_ids.append(int(match["product_id"]))
            target_diag.update(
                {
                    "accepted": True,
                    "matchTier": match["tier"],
                    "matchedProductId": int(match["product_id"]),
                    "matchedProductName": product.name,
                    "matchedCount": count,
                    "matchedWeight": round(float(match["expected"]), 1),
                    "residual": round(float(match["residual"]), 1),
                }
            )
            reconciliation["targets"].append(target_diag)

        reconciliation["accepted"] = bool(matched_product_ids)
        diagnostics.append(reconciliation)
        trigger.loadcell_diagnostics.setdefault(
            "freezerLocationReturnReconciliation",
            [],
        ).append(reconciliation)
        return matched_product_ids[0] if matched_product_ids else -1

    def _handle_return(
        self,
        aggregated: Dict[int, AggregatedProduct],
        delta_weight: float,
        *,
        trigger: Optional[TriggerResult] = None,
        source_zone: Optional[int] = None,
        returned_position_hints: Optional[List[ReturnedPositionHint]] = None,
    ) -> Optional[int]:
        """
        반환 처리: 무게 매칭하여 차감.

        무게 증가(delta > 0) 시 반환 처리.
        매칭되는 상품의 count를 1 감소시킵니다.

        예시:
        - aggregated에 치킨마요(365g) x2, 김밥(250g) x1 있음
        - delta_weight = +363g 들어옴
        - 365g ± 3g = 362~368g 범위에 363g 포함
        - 치킨마요 count -= 1 → 치킨마요 x1 남음

        Args:
            aggregated: 통합 상품 딕셔너리 (in-place 수정)
            delta_weight: 무게 증가량 (양수)

        Returns:
            차감된 product_id 또는 None (매칭 실패)
        """
        # Only positive deltas represent "put back" operations. Negative deltas
        # are handled upstream as removals.
        if delta_weight <= 0:
            return None

        # 무게로 상품 찾기
        matched_product_id = self.find_product_by_weight(
            aggregated,
            delta_weight,
            tolerance=self._weight_tolerance,
        )

        if matched_product_id is not None:
            agg = aggregated[matched_product_id]
            if agg.count > 0:
                # 무게 기반 개수 추정: delta / unit_weight (최소 1)
                # Phase 0b 안전장치: 비정상적으로 큰 추정치(> 재고)면 1개만 차감
                # Estimate the returned unit count from the load delta, but
                # stay conservative when the rounded count would erase more
                # inventory than is currently aggregated.
                estimated_count = 1
                selected_units = list(agg.placement_units)[-estimated_count:]
                agg.count = max(0, agg.count - 1)
                if trigger is not None:
                    self._append_returned_position_hint(
                        returned_position_hints,
                        product=agg,
                        trigger=trigger,
                        count=estimated_count,
                        source_zone=source_zone,
                        source="positive_return",
                        reason="weight_return_reconciled",
                        selected_units=selected_units,
                    )
                self._remove_product_units(agg, estimated_count)
                logger.info(
                    f"Return processed: {agg.name} x{estimated_count} "
                    f"(delta={delta_weight:.1f}g, unit={agg.weight:.1f}g, "
                    f"remaining={agg.count})"
                )
                return matched_product_id

        # Multi-count and combination returns are deferred to CLOSE.
        logger.warning(
            f"Return weight matching failed: {delta_weight:.1f}g "
            f"(tolerance={self._weight_tolerance}g)"
        )
        return None

    def find_product_by_weight(
        self,
        aggregated: Dict[int, AggregatedProduct],
        weight: float,
        tolerance: Optional[float] = None,
    ) -> Optional[int]:
        """
        무게로 상품 찾기.

        허용 오차 내에서 가장 가까운 무게의 상품을 찾습니다.
        count > 0인 상품만 대상으로 합니다.

        Args:
            aggregated: 통합 상품 딕셔너리
            weight: 찾을 무게 (g)

        Returns:
            매칭된 product_id 또는 None
        """
        best_match: Optional[int] = None
        best_diff = float("inf")
        tolerance_to_use = self._weight_tolerance if tolerance is None else tolerance

        for product_id, agg in aggregated.items():
            if agg.count <= 0:
                continue
            if agg.weight <= 0:
                continue

            diff = abs(agg.weight - weight)
            if diff <= tolerance_to_use and diff < best_diff:
                best_diff = diff
                best_match = product_id

        if best_match is not None:
            logger.debug(
                f"Weight match found: {weight:.1f}g -> "
                f"product_id={best_match} (diff={best_diff:.1f}g)"
            )

        return best_match

    def _return_single_match_tolerance(self) -> float:
        """Allow narrow loadcell noise for single returned items only."""
        return max(self._weight_tolerance, min(self._weight_tolerance + 3.0, 10.0))

    def _get_weight_for_product(self, product: ProductResult) -> float:
        """
        상품의 무게 조회.

        Args:
            product: 상품 결과

        Returns:
            무게 (g)
        """
        if self._get_product_weight is not None:
            weight = self._get_product_weight(product.product_id)
            if weight > 0:
                return weight

        # Fallback: ProductResult에는 무게 정보가 없으므로 0 반환
        # DoorSessionStore에서 product_db를 통해 설정해야 함
        return 0.0

    def update_weights_from_db(
        self,
        aggregated: Dict[int, AggregatedProduct],
        get_weight: Callable[[int], float],
    ) -> int:
        """
        ProductDatabase에서 무게 정보 업데이트.

        Args:
            aggregated: 통합 상품 딕셔너리 (in-place 수정)
            get_weight: product_id -> weight 조회 함수

        Returns:
            업데이트된 상품 수
        """
        updated = 0
        for product_id, agg in aggregated.items():
            if agg.weight <= 0:
                weight = get_weight(product_id)
                if weight > 0:
                    agg.weight = weight
                    updated += 1
        return updated

    # ========================================================================
    # Incremental Update Methods (v4.2) - O(N²) → O(N) 최적화
    # ========================================================================

    def add_trigger_incremental(
        self,
        aggregated: Dict[int, AggregatedProduct],
        unmatched_returns: List[UnmatchedReturn],
        trigger: TriggerResult,
        deferred_returns: Optional[List[DeferredReturn]] = None,
        *,
        zone: Optional[int] = None,
    ) -> Tuple[Dict[int, AggregatedProduct], List[UnmatchedReturn]]:
        """
        단일 trigger를 증분 추가 (v4.2).

        전체 재집계 대신 새 trigger만 처리하여 O(N²) → O(N) 최적화.

        Args:
            aggregated: 기존 통합 상품 딕셔너리 (in-place 수정)
            unmatched_returns: 기존 매칭 실패 목록 (in-place 수정)
            trigger: 추가할 TriggerResult

        Returns:
            (업데이트된 aggregated, 업데이트된 unmatched_returns)
        """
        if trigger.is_return:
            # 반환 처리: 무게 매칭하여 차감
            matched_id = self._handle_freezer_location_return(
                aggregated,
                unmatched_returns,
                trigger,
                source_zone=zone,
                diagnostics=[],
            )
            if matched_id is None:
                matched_id = self._handle_return(aggregated, trigger.delta_weight)
            if matched_id is None:
                # 매칭 실패 → 기록
                if deferred_returns is None:
                    self._record_unmatched_return(unmatched_returns, trigger)
                else:
                    self._record_deferred_return(
                        deferred_returns,
                        trigger,
                        source="positive_return",
                        replay_position="return",
                    )
        else:
            target_deferred_returns = (
                deferred_returns if deferred_returns is not None else []
            )
            # 제거 처리: YOLO 결과 합산
            self._handle_return_hints(
                target_deferred_returns,
                trigger,
                replay_position="before_removal",
            )
            self._handle_removal(aggregated, trigger)
            self._handle_return_hints(
                target_deferred_returns,
                trigger,
                replay_position="after_removal",
            )

        return aggregated, unmatched_returns
