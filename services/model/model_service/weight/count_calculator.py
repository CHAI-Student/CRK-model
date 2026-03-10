"""
Weight-Based Count Calculator.

무게 변화량 기반 상품 개수 계산기.

핵심 알고리즘:
1. Vision 후보군의 각 상품에 대해 개수 추정
2. count = round(abs(delta_weight) / unit_weight)
3. 허용 오차 내 검증 (tolerance_percent)
4. match_score 계산으로 최적 후보 선별

v4.10 변경사항:
- ProductDatabase 완전 제거 - Node.js active_products만 사용

v4.7 변경사항:
- active_products 파라미터 추가: ActiveProductStore의 상품 정보 우선 사용
- stock 필터링 비활성화 가능: use_stock_limit=False로 stock=0 필터링 스킵

사용 예시:
    calculator = WeightBasedCountCalculator()
    estimates = calculator.calculate(candidates, delta_weight=-365.0, active_products=products)
    best = estimates[0]  # 가장 높은 match_score
"""

from typing import Any, Dict, List, Optional
import logging

from model_service.engine.models import EnsembleResult, CountEstimate
from model_service.session.active_product_store import ActiveProductStore
logger = logging.getLogger(__name__)


class WeightBasedCountCalculator:
    """
    무게 기반 개수 계산기.

    v4.10: ProductDatabase 제거 - Node.js active_products만 사용

    Vision 후보군의 각 상품에 대해 무게 변화량으로 개수를 추정하고,
    허용 오차 내에서 검증합니다.

    Attributes:
        tolerance_percent: 기본 허용 오차 비율 (8%)
        max_count: 최대 개수 제한
        min_weight_change: 최소 무게 변화량 (g)
    """

    def __init__(
        self,
        tolerance_percent: float = 0.08,
        tolerance_grams: float = 3.0,
        max_count: int = 10,
        min_weight_change: float = 5.0,
        use_stock_limit: bool = True,
    ):
        """
        개수 계산기 초기화.

        v5.1: tolerance_grams 기본값 5g → 3g 변경 (로드셀 정확도 반영)
        v4.10: ProductDatabase 제거 - Node.js active_products만 사용

        Args:
            tolerance_percent: 기본 허용 오차 비율 (기본값 8%)
            tolerance_grams: 고정 허용 오차 (기본값 3g, v5.1)
            max_count: 최대 개수 제한 (기본값 10)
            min_weight_change: 최소 무게 변화량 (기본값 5g)
            use_stock_limit: 재고 상한 사용 여부 (기본값 True, v4.3)
        """
        self.tolerance_percent = tolerance_percent
        self.tolerance_grams = tolerance_grams
        self.max_count = max_count
        self.min_weight_change = min_weight_change
        self.use_stock_limit = use_stock_limit

    def calculate(
        self,
        candidates: List[EnsembleResult],
        delta_weight: float,
        use_category_tolerance: bool = True,
        active_products: Optional[ActiveProductStore] = None,
    ) -> List[CountEstimate]:
        """
        각 후보에 대한 개수 추정.

        Vision 후보군의 각 상품에 대해 무게 기반 개수를 계산하고,
        match_score 기준으로 정렬하여 반환합니다.

        v4.7: active_products가 있으면 해당 정보를 우선 사용.
        ActiveProductStore의 상품 정보로 YOLO class_id → 상품 매핑.

        Args:
            candidates: Multi-View Ensemble 결과 (Top-5)
            delta_weight: 무게 변화량 (음수 = 제거)
            use_category_tolerance: 카테고리별 허용 오차 사용 여부
            active_products: ActiveProductStore에서 가져온 상품 정보 (v4.7)

        Returns:
            CountEstimate 리스트 (match_score 내림차순 정렬)
        """
        abs_weight = abs(delta_weight)

        logger.info(f"[COUNT] ========== 개수 추정 ==========")
        logger.info(f"[COUNT] 후보: {len(candidates)}개, delta_weight={abs_weight:.1f}g")

        # v4.7: active_products 빠른 조회용 맵 생성
        active_product_map: Dict[int, Any] = {}
        if active_products:
            for ap in active_products:
                if ap.yolo_class_id is not None:
                    active_product_map[ap.yolo_class_id] = ap
            logger.info(f"[COUNT] v4.7: active_products {len(active_product_map)}개 로드")

        # 최소 무게 변화량 체크
        if abs_weight < self.min_weight_change:
            logger.info(f"[COUNT] 무게 변화 너무 작음: {abs_weight}g < {self.min_weight_change}g")
            return []

        estimates = []

        for candidate in candidates:
            # v4.7: active_products에서 먼저 상품 정보 조회
            active_product = active_product_map.get(candidate.class_id)

            if active_product is not None:
                # ActiveProductStore에서 상품 정보 사용 (v4.7)
                product_name = active_product.product_name
                product_weight = active_product.product_weight
                stock = active_product.stock_qty

                logger.debug(
                    f"[COUNT] v4.7: Using active_product: {product_name} "
                    f"(class_id={candidate.class_id}, weight={product_weight}g, stock={stock})"
                )

                if product_weight <= 0:
                    logger.debug(f"[COUNT] Skipping product with zero weight: {product_name}")
                    continue

                # v4.8: active_products의 실제 stock_qty를 사용하여 상한 적용
                # stock=0이면 재고 소진 상태이므로 count=0 처리
                count = self._estimate_count(abs_weight, product_weight, stock=stock)
                if count <= 0:
                    logger.info(
                        f"[COUNT] v4.8: Stock depleted: {product_name} "
                        f"(stock={stock}, vision detected but filtered)"
                    )
                    continue

                # 예상 무게 계산
                expected_weight = product_weight * count
                weight_error = abs(abs_weight - expected_weight)

                # 허용 오차: 고정 5g 사용
                tolerance_amount = self.tolerance_grams

                # 검증
                validated = weight_error <= tolerance_amount

                # 매칭 점수 계산
                match_score = self._calculate_match_score(
                    weight_error=weight_error,
                    expected_weight=expected_weight,
                    vision_confidence=candidate.combined_confidence,
                )

                # v4.7: active_product에서 가격 정보 가져오기
                unit_price = active_product.sale_price

                estimate = CountEstimate(
                    product_id=candidate.class_id,
                    product_name=product_name,
                    count=count,
                    unit_weight=product_weight,
                    expected_weight=expected_weight,
                    actual_weight=abs_weight,
                    match_score=match_score,
                    vision_confidence=candidate.combined_confidence,
                    validated=validated,
                    unit_price=unit_price,  # v4.7: 가격 정보 추가
                )

                estimates.append(estimate)
                logger.debug(
                    f"[COUNT] v4.7 Estimate: {product_name} x{count}, "
                    f"expected={expected_weight:.1f}g, actual={abs_weight:.1f}g, "
                    f"error={weight_error:.1f}g, validated={validated}, score={match_score:.3f}, "
                    f"unit_price={unit_price}원"
                )
            else:
                # v4.9: ProductDatabase fallback 완전 비활성화
                # Node.js에서 받은 active_products에 없는 상품은 스킵
                # (재고가 없거나 등록되지 않은 상품)
                logger.debug(
                    f"[COUNT] v4.9: Skipping class_id={candidate.class_id} "
                    f"(not in active_products)"
                )

        # match_score 기준 정렬
        estimates.sort(key=lambda e: e.match_score, reverse=True)

        # 결과 로깅
        for est in estimates[:5]:
            logger.info(
                f"[COUNT] {est.product_name}: count={est.count}, "
                f"expected={est.expected_weight:.1f}g, "
                f"match_score={est.match_score:.3f}, validated={est.validated}"
            )

        return estimates

    def _estimate_count(
        self,
        abs_weight: float,
        unit_weight: float,
        stock: int = 0,
    ) -> int:
        """
        수량 추정 (반올림), 재고 상한 적용 (v4.3).

        Args:
            abs_weight: 절대 무게 변화량 (g)
            unit_weight: 단위 무게 (g)
            stock: 재고 수량 (v4.3)

        Returns:
            추정 개수 (1 ~ min(max_count, stock))
        """
        if unit_weight <= 0:
            return 0

        count = round(abs_weight / unit_weight)

        # 범위 제한
        if count < 1:
            return 0
        if count > self.max_count:
            count = self.max_count

        # v4.3: 재고 상한 적용
        if self.use_stock_limit and stock > 0 and count > stock:
            logger.info(f"[COUNT] 재고 상한 적용: estimated={count} -> stock={stock}")
            count = stock

        return count

    def _calculate_match_score(
        self,
        weight_error: float,
        expected_weight: float,
        vision_confidence: float,
    ) -> float:
        """
        매칭 점수 계산.

        점수 구성:
        - 무게 매칭 점수 (50%): 오차가 적을수록 높음
        - Vision 신뢰도 (40%): 앙상블 결과 신뢰도
        - 개수 합리성 (10%): 개수가 적을수록 높음 (단순성 선호)

        Args:
            weight_error: 무게 오차 (g)
            expected_weight: 예상 무게 (g)
            vision_confidence: Vision 신뢰도

        Returns:
            매칭 점수 (0.0 ~ 1.0)
        """
        # 1. 무게 매칭 점수 (0.0 ~ 1.0)
        # tolerance_grams 기준으로 계산 (고정 5g)
        if expected_weight <= 0:
            weight_score = 0.0
        else:
            # 허용 오차(5g)의 2배(10g)까지 선형 감소
            weight_score = max(0.0, 1.0 - (weight_error / (2 * self.tolerance_grams)))

        # 2. Vision 신뢰도 (0.0 ~ 1.0)
        vision_score = min(max(vision_confidence, 0.0), 1.0)

        # 3. 가중 평균
        # 무게 매칭 50%, Vision 40%, 기타 10%
        match_score = (
            weight_score * 0.5 +
            vision_score * 0.4 +
            0.1  # 기본 점수
        )

        return min(match_score, 1.0)

    def calculate_combination(
        self,
        candidates: List[EnsembleResult],
        delta_weight: float,
        max_combination_size: int = 2,
        active_products: Optional[List[Any]] = None,
    ) -> Optional[List[CountEstimate]]:
        """
        다중 상품 조합 계산.

        단일 상품으로 무게를 설명할 수 없을 때,
        2개 이상의 상품 조합으로 시도합니다.

        전략:
        1. 동일 상품 다중 개수 (A x N)
        2. 서로 다른 상품 조합 (A x 1 + B x 1)
        3. 서로 다른 상품 다중 개수 (A x N + B x M)

        v4.7: active_products가 있으면 해당 정보를 우선 사용.

        Args:
            candidates: Multi-View Ensemble 결과
            delta_weight: 무게 변화량
            max_combination_size: 최대 조합 크기 (기본값 2)
            active_products: ActiveProductStore에서 가져온 상품 정보 (v4.7)

        Returns:
            매칭되는 CountEstimate 리스트 또는 None
        """
        from itertools import combinations, product as iterproduct

        abs_weight = abs(delta_weight)

        if abs_weight < self.min_weight_change:
            return None

        # v4.7: active_products 빠른 조회용 맵 생성
        active_product_map: Dict[int, Any] = {}
        if active_products:
            for ap in active_products:
                if ap.yolo_class_id is not None:
                    active_product_map[ap.yolo_class_id] = ap

        # 후보군에서 상품 정보 추출 (v4.7: active_products 우선)
        product_candidates = []
        for candidate in candidates[:5]:  # 상위 5개만 고려
            active_product = active_product_map.get(candidate.class_id)

            if active_product is not None:
                # v4.7: ActiveProductStore에서 상품 정보 사용
                if active_product.product_weight > 0:
                    # 가상의 ProductInfo-like 객체 생성
                    pseudo_prod = type('PseudoProduct', (), {
                        'name': active_product.product_name,
                        'weight': active_product.product_weight,
                        'stock': active_product.stock_qty,
                    })()
                    product_candidates.append((candidate, pseudo_prod))
            else:
                # v4.9: ProductDatabase fallback 완전 비활성화
                # Node.js에서 받은 active_products에 없는 상품은 스킵
                logger.debug(
                    f"[COMBINATION] v4.9: Skipping class_id={candidate.class_id} "
                    f"(not in active_products)"
                )

        if not product_candidates:
            return None

        best_combination = None
        best_error = float("inf")
        best_score = 0.0

        # 전략 2 & 3: 서로 다른 상품 조합 (다양한 개수)
        if len(product_candidates) >= 2:
            for (cand1, prod1), (cand2, prod2) in combinations(product_candidates, 2):
                # v4.3: 재고 상한 적용
                max_count1 = min(3, prod1.stock) if self.use_stock_limit and prod1.stock > 0 else 3
                max_count2 = min(3, prod2.stock) if self.use_stock_limit and prod2.stock > 0 else 3
                # 각 상품 1~재고수량(최대3)개씩 조합 시도
                for count1, count2 in iterproduct(range(1, max_count1 + 1), range(1, max_count2 + 1)):
                    combined_weight = prod1.weight * count1 + prod2.weight * count2
                    error = abs(abs_weight - combined_weight)
                    tolerance = self.tolerance_grams  # 고정 5g

                    if error <= tolerance:
                        # 매칭 점수 계산 (오차 적을수록 + 개수 적을수록 높음)
                        error_score = 1.0 - (error / combined_weight) if combined_weight > 0 else 0
                        count_penalty = 1.0 - ((count1 + count2 - 2) * 0.1)  # 2개일 때 최고
                        avg_confidence = (cand1.combined_confidence + cand2.combined_confidence) / 2
                        score = error_score * 0.5 + avg_confidence * 0.4 + count_penalty * 0.1

                        if error < best_error or (error == best_error and score > best_score):
                            best_error = error
                            best_score = score

                            # 각 상품의 기여 무게 비율로 actual_weight 분배
                            weight1 = prod1.weight * count1
                            weight2 = prod2.weight * count2
                            total_expected = weight1 + weight2

                            best_combination = [
                                CountEstimate(
                                    product_id=cand1.class_id,
                                    product_name=prod1.name,
                                    count=count1,
                                    unit_weight=prod1.weight,
                                    expected_weight=weight1,
                                    actual_weight=abs_weight * (weight1 / total_expected),
                                    match_score=self._calculate_match_score(
                                        weight_error=error * (weight1 / total_expected),
                                        expected_weight=weight1,
                                        vision_confidence=cand1.combined_confidence,
                                    ),
                                    vision_confidence=cand1.combined_confidence,
                                    validated=True,
                                ),
                                CountEstimate(
                                    product_id=cand2.class_id,
                                    product_name=prod2.name,
                                    count=count2,
                                    unit_weight=prod2.weight,
                                    expected_weight=weight2,
                                    actual_weight=abs_weight * (weight2 / total_expected),
                                    match_score=self._calculate_match_score(
                                        weight_error=error * (weight2 / total_expected),
                                        expected_weight=weight2,
                                        vision_confidence=cand2.combined_confidence,
                                    ),
                                    vision_confidence=cand2.combined_confidence,
                                    validated=True,
                                ),
                            ]

        if best_combination:
            products_str = " + ".join(
                f"{e.product_name}x{e.count}" for e in best_combination
            )
            logger.info(
                f"Found combination match: {products_str}, "
                f"error={best_error:.1f}g, score={best_score:.3f}"
            )

        return best_combination

    def validate_estimate(self, estimate: CountEstimate) -> bool:
        """
        개수 추정 결과 검증.

        Args:
            estimate: CountEstimate 인스턴스

        Returns:
            검증 통과 여부
        """
        # 기본 검증
        if estimate.count <= 0:
            return False

        if estimate.count > self.max_count:
            return False

        # 오차 검증 (고정 5g)
        return estimate.weight_error <= self.tolerance_grams
