"""
Hand Proximity Filter.

손과 가까운 상품만 필터링.

알고리즘:
1. 손(cls=0)과 상품(cls>0) 분리
2. 각 손에 대해 가장 가까운 상품 선택
3. 거리 제한 내의 상품만 포함
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging

from .yolo_wrapper import YOLODetection
from model_service.core.config import config

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """필터링 결과."""
    hands: List[YOLODetection]              # 감지된 손들
    filtered_products: List[YOLODetection]  # 손 근접 필터링된 상품들
    all_products: List[YOLODetection]       # 모든 상품 (필터 전)
    hand_product_pairs: List[Tuple[YOLODetection, YOLODetection]]  # (손, 상품) 쌍


class HandProximityFilter:
    """
    손 근접 상품 필터.

    각 손에서 가장 가까운 상품만 선택하여 노이즈 제거.

    Attributes:
        max_distance_px: 최대 거리 제한 (픽셀)
        hand_class_id: 손 클래스 ID (기본값 0)
    """

    def __init__(
        self,
        max_distance_px: Optional[float] = None,
        hand_class_id: Optional[int] = None,
    ):
        """
        필터 초기화.

        Args:
            max_distance_px: 최대 거리 제한 (픽셀)
            hand_class_id: 손 클래스 ID
        """
        self.max_distance_px = max_distance_px or config.vision.max_distance_px
        self.hand_class_id = hand_class_id if hand_class_id is not None else config.vision.hand_class_id

    def filter(
        self,
        detections: List[YOLODetection],
    ) -> FilterResult:
        """
        손 근접 필터링 수행.

        Args:
            detections: 전체 YOLO 감지 결과

        Returns:
            FilterResult
        """
        # 1. 손과 상품 분리
        hands = [d for d in detections if d.cls == self.hand_class_id]
        products = [d for d in detections if d.cls != self.hand_class_id]

        logger.debug(f"Separated: {len(hands)} hands, {len(products)} products")

        # 손이 없으면 모든 상품 반환
        if not hands:
            logger.info("No hands detected, returning all products")
            return FilterResult(
                hands=[],
                filtered_products=products,
                all_products=products,
                hand_product_pairs=[],
            )

        # 상품이 없으면 빈 결과
        if not products:
            logger.info("No products detected")
            return FilterResult(
                hands=hands,
                filtered_products=[],
                all_products=[],
                hand_product_pairs=[],
            )

        # 2. 각 손에 대해 가장 가까운 상품 선택
        filtered = []
        pairs = []

        for hand in hands:
            nearest = self._find_nearest_product(hand, products)
            if nearest:
                if nearest not in filtered:
                    filtered.append(nearest)
                pairs.append((hand, nearest))
                logger.debug(
                    f"Hand at ({hand.center_x:.1f}, {hand.center_y:.1f}) -> "
                    f"{nearest.name} (dist={hand.distance_to(nearest):.1f}px)"
                )

        logger.info(f"Filtered: {len(filtered)} products from {len(products)} total")

        return FilterResult(
            hands=hands,
            filtered_products=filtered,
            all_products=products,
            hand_product_pairs=pairs,
        )

    def _find_nearest_product(
        self,
        hand: YOLODetection,
        products: List[YOLODetection],
    ) -> Optional[YOLODetection]:
        """
        손에서 가장 가까운 상품 찾기.

        Args:
            hand: 손 Detection
            products: 상품 Detection 리스트

        Returns:
            가장 가까운 상품 또는 None (거리 초과 시)
        """
        nearest = None
        min_distance = float('inf')

        for product in products:
            distance = hand.distance_to(product)

            if distance < min_distance and distance <= self.max_distance_px:
                min_distance = distance
                nearest = product

        return nearest

    def filter_and_sort(
        self,
        detections: List[YOLODetection],
        top_k: int = 5,
    ) -> List[YOLODetection]:
        """
        필터링 후 confidence 기준 상위 K개 반환.

        Args:
            detections: 전체 YOLO 감지 결과
            top_k: 상위 K개

        Returns:
            필터링 + 정렬된 상품 리스트
        """
        result = self.filter(detections)

        # confidence 기준 정렬
        sorted_products = sorted(
            result.filtered_products,
            key=lambda d: d.conf,
            reverse=True,
        )

        return sorted_products[:top_k]
