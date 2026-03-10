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

from .door_session import TriggerResult, AggregatedProduct, UnmatchedReturn
from .session_store import ProductResult
from model_service.weight.strict_weight_matcher import StrictWeightMatcher

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
        result = self.aggregate_with_unmatched(triggers)
        return result.products

    def aggregate_with_unmatched(
        self,
        triggers: List[TriggerResult],
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

        for trigger in triggers:
            if trigger.is_return:
                # 반환 처리: 무게 매칭하여 차감
                matched_id = self._handle_return(aggregated, trigger.delta_weight)
                if matched_id is None:
                    # 매칭 실패 → 기록
                    unmatched_returns.append(UnmatchedReturn(
                        trigger_id=trigger.trigger_id,
                        delta_weight=trigger.delta_weight,
                        timestamp=trigger.timestamp,
                        tolerance_used=self._weight_tolerance,
                    ))
            else:
                # 제거 처리: YOLO 결과 합산
                self._handle_removal(aggregated, trigger)

        logger.debug(
            f"Aggregated {len(triggers)} triggers -> {len(aggregated)} products, "
            f"total count={sum(p.count for p in aggregated.values())}, "
            f"unmatched_returns={len(unmatched_returns)}"
        )

        return AggregationResult(
            products=aggregated,
            unmatched_returns=unmatched_returns,
        )

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

            if product_id in aggregated:
                # 기존 상품에 합산
                agg = aggregated[product_id]
                agg.count += product.count
                agg.total_confidence += product.confidence * product.count
                agg.detection_count += product.count
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
                )

            logger.debug(
                f"Added product: {product.name} x{product.count} "
                f"(id={product_id}, total={aggregated[product_id].count})"
            )

    def _handle_return(
        self,
        aggregated: Dict[int, AggregatedProduct],
        delta_weight: float,
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
        if delta_weight <= 0:
            return None

        # 무게로 상품 찾기
        matched_product_id = self.find_product_by_weight(aggregated, delta_weight)

        if matched_product_id is not None:
            agg = aggregated[matched_product_id]
            if agg.count > 0:
                # 무게 기반 개수 추정: delta / unit_weight (최소 1)
                # Phase 0b 안전장치: 비정상적으로 큰 추정치(> 재고)면 1개만 차감
                if agg.weight > 0:
                    estimated_count = max(1, round(delta_weight / agg.weight))
                    if estimated_count > agg.count:
                        estimated_count = 1
                else:
                    estimated_count = 1
                agg.count = max(0, agg.count - estimated_count)
                logger.info(
                    f"Return processed: {agg.name} x{estimated_count} "
                    f"(delta={delta_weight:.1f}g, unit={agg.weight:.1f}g, "
                    f"remaining={agg.count})"
                )
                return matched_product_id

        # Step 2: 조합 매칭 폴백 (Phase 1) - 다중 상품 동시 반납
        combo = self._weight_matcher.find_return_combination(aggregated, delta_weight)
        if combo is not None:
            names = []
            first_id = None
            for product_id, count in combo.items():
                if product_id in aggregated:
                    agg = aggregated[product_id]
                    agg.count = max(0, agg.count - count)
                    names.append(f"{agg.name}x{count}")
                    if first_id is None:
                        first_id = product_id
            logger.info(
                f"Return combo processed: {', '.join(names)} "
                f"(delta={delta_weight:.1f}g)"
            )
            return first_id

        logger.warning(
            f"Return weight matching failed: {delta_weight:.1f}g "
            f"(tolerance={self._weight_tolerance}g)"
        )
        return None

    def find_product_by_weight(
        self,
        aggregated: Dict[int, AggregatedProduct],
        weight: float,
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

        for product_id, agg in aggregated.items():
            if agg.count <= 0:
                continue
            if agg.weight <= 0:
                continue

            diff = abs(agg.weight - weight)
            if diff <= self._weight_tolerance and diff < best_diff:
                best_diff = diff
                best_match = product_id

        if best_match is not None:
            logger.debug(
                f"Weight match found: {weight:.1f}g -> "
                f"product_id={best_match} (diff={best_diff:.1f}g)"
            )

        return best_match

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
            matched_id = self._handle_return(aggregated, trigger.delta_weight)
            if matched_id is None:
                # 매칭 실패 → 기록
                unmatched_returns.append(UnmatchedReturn(
                    trigger_id=trigger.trigger_id,
                    delta_weight=trigger.delta_weight,
                    timestamp=trigger.timestamp,
                    tolerance_used=self._weight_tolerance,
                ))
        else:
            # 제거 처리: YOLO 결과 합산
            self._handle_removal(aggregated, trigger)

        return aggregated, unmatched_returns
