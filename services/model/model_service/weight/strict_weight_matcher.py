"""
Strict Weight Matcher.

무게 우선 엄격 매칭기 (v5.1).

로드셀이 매우 정확(±3g)하므로:
1. 무게로 가능한 모든 상품 조합을 찾음
2. 그 중 YOLO가 감지한 것만 필터링
3. Vision 신뢰도로 최종 선택

알고리즘: Weight-First Subset Sum (Backtracking)

사용 예시:
    matcher = StrictWeightMatcher(tolerance=3.0)
    valid_combos = matcher.find_valid_combinations(
        candidates=vision_candidates,
        delta_weight=100.0,
        active_products=products,
    )
    if valid_combos:
        best = valid_combos[0]  # 이미 신뢰도 순 정렬됨
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

from model_service.engine.models import EnsembleResult
from model_service.core.config import config as app_config

logger = logging.getLogger(__name__)


@dataclass
class CandidateProduct:
    """
    조합 검색용 후보 상품.

    Attributes:
        class_id: YOLO 클래스 ID
        name: 상품명
        weight: 단위 무게 (g)
        stock: 재고 수량
        vision_confidence: Vision 신뢰도
        unit_price: 단가 (원)
    """
    class_id: int
    name: str
    weight: float
    stock: int
    vision_confidence: float
    unit_price: int = 0


@dataclass
class CombinationItem:
    """
    조합 내 개별 상품.

    Attributes:
        candidate: 후보 상품 정보
        count: 개수
    """
    candidate: CandidateProduct
    count: int

    @property
    def total_weight(self) -> float:
        """총 무게."""
        return self.candidate.weight * self.count

    @property
    def total_price(self) -> int:
        """총 가격."""
        return self.candidate.unit_price * self.count


@dataclass
class ValidCombination:
    """
    유효한 상품 조합 결과.

    Attributes:
        items: 조합을 구성하는 상품들
        total_weight: 조합 총 무게
        target_weight: 목표 무게 (delta_weight 절대값)
        weight_error: 무게 오차 (절대값)
        avg_vision_confidence: 평균 Vision 신뢰도
        match_score: 매칭 점수 (정렬용)
    """
    items: List[CombinationItem] = field(default_factory=list)
    total_weight: float = 0.0
    target_weight: float = 0.0
    weight_error: float = 0.0
    avg_vision_confidence: float = 0.0
    match_score: float = 0.0

    @property
    def total_price(self) -> int:
        """총 가격."""
        return sum(item.total_price for item in self.items)

    @property
    def total_count(self) -> int:
        """총 개수."""
        return sum(item.count for item in self.items)

    def to_dict(self) -> dict:
        """딕셔너리 변환."""
        return {
            "items": [
                {
                    "class_id": item.candidate.class_id,
                    "name": item.candidate.name,
                    "count": item.count,
                    "unit_weight": item.candidate.weight,
                    "total_weight": item.total_weight,
                    "unit_price": item.candidate.unit_price,
                    "total_price": item.total_price,
                    "vision_confidence": item.candidate.vision_confidence,
                }
                for item in self.items
            ],
            "total_weight": round(self.total_weight, 1),
            "target_weight": round(self.target_weight, 1),
            "weight_error": round(self.weight_error, 1),
            "total_count": self.total_count,
            "total_price": self.total_price,
            "avg_vision_confidence": round(self.avg_vision_confidence, 4),
            "match_score": round(self.match_score, 4),
        }


class StrictWeightMatcher:
    """
    무게 우선 엄격 매칭기.

    로드셀이 매우 정확(±3g)하므로:
    1. 무게로 가능한 모든 상품 조합을 찾음
    2. 그 중 YOLO가 감지한 것만 필터링
    3. Vision 신뢰도로 최종 선택

    Attributes:
        tolerance: 무게 허용 오차 (g, 기본값 3.0)
        max_items: 조합당 최대 상품 수 (기본값 5)
        max_count_per_item: 상품당 최대 개수 (기본값 10)
        max_combinations: 최대 조합 수 (성능 제한, 기본값 100)
    """

    def __init__(
        self,
        tolerance: Optional[float] = None,
        max_items: Optional[int] = None,
        max_count_per_item: Optional[int] = None,
        max_combinations: Optional[int] = None,
    ):
        """
        매칭기 초기화.

        v5.2: config에서 기본값 가져옴

        Args:
            tolerance: 무게 허용 오차 (g), None이면 config 사용
            max_items: 조합당 최대 상품 수, None이면 config 사용
            max_count_per_item: 상품당 최대 개수, None이면 config 사용
            max_combinations: 최대 조합 수 (성능 제한), None이면 config 사용
        """
        self.tolerance = tolerance if tolerance is not None else app_config.weight.tolerance_grams
        self.max_items = max_items if max_items is not None else app_config.weight.max_combination_items
        self.max_count_per_item = max_count_per_item if max_count_per_item is not None else app_config.weight.max_count_per_item
        self.max_combinations = max_combinations if max_combinations is not None else app_config.weight.max_combinations

    def find_valid_combinations(
        self,
        candidates: List[EnsembleResult],
        delta_weight: float,
        active_products: Optional[List[Any]] = None,
    ) -> List[ValidCombination]:
        """
        무게를 정확히 설명하는 조합만 반환.

        알고리즘:
        1. YOLO 후보 중 무게 정보가 있는 것만 추출
        2. Backtracking으로 target_weight ± tolerance 되는 모든 조합 찾기
        3. 조합별로 총 Vision 신뢰도 계산
        4. match_score 순으로 정렬하여 반환

        시간복잡도: O(n * max_count * max_items) ≈ O(5 * 10 * 5) = O(250)

        Args:
            candidates: YOLO 후보 (EnsembleResult 리스트)
            delta_weight: 무게 변화량 (음수 = 제거)
            active_products: ActiveProductStore의 상품 정보

        Returns:
            유효한 조합 리스트 (match_score 내림차순 정렬)
        """
        target_weight = abs(delta_weight)

        logger.info(f"[STRICT] ========== 엄격 무게 매칭 ==========")
        logger.info(
            f"[STRICT] target_weight={target_weight:.1f}g, "
            f"tolerance=±{self.tolerance}g, candidates={len(candidates)}개"
        )

        # 최소 무게 체크
        if target_weight < self.tolerance:
            logger.info(f"[STRICT] 무게 변화 너무 작음: {target_weight:.1f}g < {self.tolerance}g")
            return []

        # active_products 빠른 조회용 맵 생성
        active_product_map: Dict[int, Any] = {}
        if active_products:
            for ap in active_products:
                if ap.yolo_class_id is not None:
                    active_product_map[ap.yolo_class_id] = ap

        # 후보 상품 추출 (YOLO 감지 + 무게 정보 있음)
        candidate_products = self._extract_candidate_products(
            candidates, active_product_map
        )

        if not candidate_products:
            logger.warning("[STRICT] 유효한 후보 상품 없음 (무게 정보 없거나 재고 없음)")
            return []

        logger.info(f"[STRICT] 유효 후보: {len(candidate_products)}개")
        for cp in candidate_products:
            logger.debug(
                f"[STRICT]   - {cp.name}: weight={cp.weight}g, "
                f"stock={cp.stock}, vision={cp.vision_confidence:.3f}"
            )

        # Backtracking으로 모든 유효 조합 찾기
        valid_combinations: List[ValidCombination] = []
        self._backtrack(
            candidates=candidate_products,
            target=target_weight,
            current_combo=[],
            current_weight=0.0,
            start_idx=0,
            results=valid_combinations,
        )

        if not valid_combinations:
            logger.info(
                f"[STRICT] 유효 조합 없음 (target={target_weight:.1f}g ± {self.tolerance}g)"
            )
            return []

        # match_score 기준 정렬 (내림차순)
        valid_combinations.sort(key=lambda c: c.match_score, reverse=True)

        # 결과 로깅
        logger.info(f"[STRICT] 유효 조합 {len(valid_combinations)}개 발견")
        for i, combo in enumerate(valid_combinations[:3]):
            items_str = " + ".join(
                f"{item.candidate.name}x{item.count}" for item in combo.items
            )
            logger.info(
                f"[STRICT]   #{i+1}: {items_str}, "
                f"weight={combo.total_weight:.1f}g (err={combo.weight_error:.1f}g), "
                f"score={combo.match_score:.3f}"
            )

        return valid_combinations

    def _extract_candidate_products(
        self,
        candidates: List[EnsembleResult],
        active_product_map: Dict[int, Any],
    ) -> List[CandidateProduct]:
        """
        YOLO 후보에서 유효한 CandidateProduct 추출.

        조건:
        - active_products에 존재
        - 무게 정보 있음 (weight > 0)
        - 재고 있음 (stock > 0)

        Args:
            candidates: YOLO 후보
            active_product_map: class_id → ActiveProduct 맵

        Returns:
            CandidateProduct 리스트
        """
        result = []

        for candidate in candidates:
            active_product = active_product_map.get(candidate.class_id)

            if active_product is None:
                logger.debug(
                    f"[STRICT] Skip class_id={candidate.class_id}: "
                    "not in active_products"
                )
                continue

            weight = getattr(active_product, 'product_weight', 0)
            stock = getattr(active_product, 'stock_qty', 0)

            if weight <= 0:
                logger.debug(
                    f"[STRICT] Skip {active_product.product_name}: "
                    f"weight={weight}g (invalid)"
                )
                continue

            if stock <= 0:
                logger.debug(
                    f"[STRICT] Skip {active_product.product_name}: "
                    f"stock={stock} (out of stock)"
                )
                continue

            cp = CandidateProduct(
                class_id=candidate.class_id,
                name=active_product.product_name,
                weight=weight,
                stock=stock,
                vision_confidence=candidate.combined_confidence,
                unit_price=getattr(active_product, 'sale_price', 0),
            )
            result.append(cp)

        return result

    def _backtrack(
        self,
        candidates: List[CandidateProduct],
        target: float,
        current_combo: List[CombinationItem],
        current_weight: float,
        start_idx: int,
        results: List[ValidCombination],
    ) -> None:
        """
        Backtracking으로 유효한 조합 탐색.

        Args:
            candidates: 후보 상품 리스트
            target: 목표 무게
            current_combo: 현재 조합
            current_weight: 현재 조합의 총 무게
            start_idx: 탐색 시작 인덱스 (중복 방지)
            results: 결과 저장 리스트
        """
        # 성능 제한: 최대 조합 수 초과 시 중단
        if len(results) >= self.max_combinations:
            return

        # 유효한 조합 발견 (target ± tolerance)
        if current_combo and abs(current_weight - target) <= self.tolerance:
            combo = self._make_valid_combination(
                current_combo.copy(), current_weight, target
            )
            results.append(combo)
            # 계속 탐색 (더 나은 조합 있을 수 있음)

        # Pruning: 초과하면 중단
        if current_weight > target + self.tolerance:
            return

        # 조합 크기 제한
        if len(current_combo) >= self.max_items:
            return

        # 다음 상품들 시도
        for i in range(start_idx, len(candidates)):
            cand = candidates[i]

            # 재고 상한 적용
            max_count = min(cand.stock, self.max_count_per_item)

            for count in range(1, max_count + 1):
                new_weight = current_weight + cand.weight * count

                # Early exit: 이미 초과
                if new_weight > target + self.tolerance:
                    break

                # 조합에 추가
                item = CombinationItem(candidate=cand, count=count)
                current_combo.append(item)

                # 재귀 탐색 (다음 상품부터 - 중복 방지)
                self._backtrack(
                    candidates=candidates,
                    target=target,
                    current_combo=current_combo,
                    current_weight=new_weight,
                    start_idx=i + 1,  # 다음 상품부터 (같은 상품 중복 방지)
                    results=results,
                )

                # 백트래킹
                current_combo.pop()

    def _make_valid_combination(
        self,
        items: List[CombinationItem],
        total_weight: float,
        target_weight: float,
    ) -> ValidCombination:
        """
        ValidCombination 객체 생성.

        Args:
            items: 조합 아이템 리스트
            total_weight: 조합 총 무게
            target_weight: 목표 무게

        Returns:
            ValidCombination 객체 (match_score 포함)
        """
        weight_error = abs(total_weight - target_weight)

        # 평균 Vision 신뢰도 (개수 가중 평균)
        total_count = sum(item.count for item in items)
        if total_count > 0:
            weighted_conf = sum(
                item.candidate.vision_confidence * item.count
                for item in items
            )
            avg_vision_confidence = weighted_conf / total_count
        else:
            avg_vision_confidence = 0.0

        # match_score 계산
        # 1. 무게 점수 (오차 적을수록 높음, tolerance 범위 내 1.0)
        if self.tolerance > 0:
            weight_score = max(0.0, 1.0 - (weight_error / self.tolerance))
        else:
            weight_score = 1.0 if weight_error == 0 else 0.0

        # 2. Vision 점수
        vision_score = min(max(avg_vision_confidence, 0.0), 1.0)

        # 3. 단순성 점수 (상품 종류 적을수록 높음)
        # 1종류: 1.0, 2종류: 0.8, 3종류: 0.6, ...
        simplicity_score = max(0.0, 1.0 - (len(items) - 1) * 0.2)

        # 가중 평균: 무게 60%, Vision 30%, 단순성 10%
        match_score = (
            weight_score * 0.6 +
            vision_score * 0.3 +
            simplicity_score * 0.1
        )

        return ValidCombination(
            items=items,
            total_weight=total_weight,
            target_weight=target_weight,
            weight_error=weight_error,
            avg_vision_confidence=avg_vision_confidence,
            match_score=match_score,
        )

    def find_return_combination(
        self,
        aggregated: dict,
        delta_weight: float,
    ) -> Optional[Dict[int, int]]:
        """
        반품 조합 탐색 (vision 없이 무게만으로).

        단일 상품 매칭 실패 후 폴백으로 사용.
        aggregated에서 count>0, weight>0인 상품들로 delta_weight를
        설명하는 조합을 backtracking으로 탐색합니다.

        Args:
            aggregated: {product_id: AggregatedProduct}
            delta_weight: 반납 무게 (양수)

        Returns:
            {product_id: count} 또는 None (조합 없음)
        """
        if delta_weight <= self.tolerance:
            return None

        # 후보 추출: count > 0, weight > 0 (무거운 것부터 - pruning 효율화)
        candidates = sorted(
            [
                (pid, agg.weight, agg.count)
                for pid, agg in aggregated.items()
                if agg.count > 0 and agg.weight > 0
            ],
            key=lambda x: -x[1],
        )

        if not candidates:
            return None

        result: Dict[int, int] = {}
        found = self._backtrack_return(candidates, delta_weight, 0, result)
        return result if found else None

    def _backtrack_return(
        self,
        candidates: list,
        remaining: float,
        start_idx: int,
        result: Dict[int, int],
    ) -> bool:
        """
        반품 조합 백트래킹.

        Args:
            candidates: [(product_id, weight, max_count), ...]
            remaining: 아직 설명해야 할 무게 (g)
            start_idx: 탐색 시작 인덱스
            result: 현재까지의 조합 (in-place 수정)

        Returns:
            True if valid combination found
        """
        # 기저 조건: 허용 오차 내 → 성공
        if abs(remaining) <= self.tolerance:
            return True

        # Pruning: remaining이 음수 → 초과
        if remaining < -self.tolerance:
            return False

        # 모든 후보 소진
        if start_idx >= len(candidates):
            return False

        # 깊이 제한
        if sum(result.values()) >= self.max_items:
            return False

        product_id, weight, max_count = candidates[start_idx]
        capped = min(max_count, self.max_count_per_item)

        # 이 상품을 1~capped개 사용
        for count in range(1, capped + 1):
            used_weight = weight * count
            if used_weight > remaining + self.tolerance:
                break  # 이미 초과, 더 늘려도 의미없음
            result[product_id] = count
            if self._backtrack_return(candidates, remaining - used_weight, start_idx + 1, result):
                return True
            del result[product_id]

        # 이 상품 스킵
        return self._backtrack_return(candidates, remaining, start_idx + 1, result)
