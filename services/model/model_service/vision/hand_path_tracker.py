"""
Hand Path Tracker (v4.6).

손 이동 경로 추적 및 상품 bbox 교차 검증.

핵심 알고리즘:
1. 프레임별 손 중심점 추적 -> 손 이동 경로(trajectory) 구축
2. 상품 bbox가 손 경로와 교차하는지 검증
3. 손 경로와 일치하는 상품만 후보로 유지

사용법:
    tracker = HandPathTracker()

    # 각 프레임에서 손/상품 탐지 업데이트
    for frame_idx, detections in enumerate(all_detections):
        tracker.update_frame(detections, frame_idx)

    # 손 경로와 교차하는 상품 클래스 ID 필터링
    valid_class_ids = tracker.filter_products_by_path(candidate_class_ids)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class HandTrajectory:
    """
    손 이동 경로.

    Attributes:
        centers: 손 중심점 이동 경로 [(x, y), ...]
        frame_indices: 각 중심점이 기록된 프레임 인덱스
        bbox_history: bbox 크기 히스토리 [(width, height), ...]
    """
    centers: List[Tuple[float, float]] = field(default_factory=list)
    frame_indices: List[int] = field(default_factory=list)
    bbox_history: List[Tuple[float, float]] = field(default_factory=list)

    def add_point(
        self,
        center: Tuple[float, float],
        frame_idx: int,
        bbox_size: Optional[Tuple[float, float]] = None,
    ) -> None:
        """손 위치 추가."""
        self.centers.append(center)
        self.frame_indices.append(frame_idx)
        if bbox_size:
            self.bbox_history.append(bbox_size)

    @property
    def avg_bbox_size(self) -> float:
        """평균 bbox 크기 (대각선 길이)."""
        if not self.bbox_history:
            return 100.0  # 기본값
        total = sum(
            math.sqrt(w**2 + h**2) for w, h in self.bbox_history
        )
        return total / len(self.bbox_history)

    def intersects_bbox(
        self,
        bbox_center: Tuple[float, float],
        bbox_size: Tuple[float, float],
        tolerance: Optional[float] = None,
    ) -> bool:
        """
        손 경로가 bbox 근처를 지나갔는지 확인.

        Args:
            bbox_center: 상품 bbox 중심점 (x, y)
            bbox_size: 상품 bbox 크기 (width, height)
            tolerance: 허용 거리 (None이면 손 크기 + bbox 크기의 50% 사용)

        Returns:
            손 경로가 bbox 근처를 지나갔으면 True
        """
        if not self.centers:
            return False

        # 동적 허용 거리: 손 크기와 상품 크기 기반
        if tolerance is None:
            hand_radius = self.avg_bbox_size / 2
            product_radius = math.sqrt(bbox_size[0]**2 + bbox_size[1]**2) / 2
            tolerance = hand_radius + product_radius * 0.5

        px, py = bbox_center

        # 손 경로의 각 점과 상품 중심점 거리 체크
        for hx, hy in self.centers:
            distance = math.sqrt((px - hx)**2 + (py - hy)**2)
            if distance <= tolerance:
                return True

        return False

    def get_path_length(self) -> float:
        """경로 총 길이."""
        if len(self.centers) < 2:
            return 0.0

        total = 0.0
        for i in range(1, len(self.centers)):
            x1, y1 = self.centers[i - 1]
            x2, y2 = self.centers[i]
            total += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return total


@dataclass
class ProductBboxHistory:
    """
    상품 bbox 히스토리.

    Attributes:
        class_id: YOLO 클래스 ID
        class_name: YOLO 클래스명
        centers: 감지된 중심점들
        frame_indices: 감지된 프레임 인덱스들
        bbox_sizes: bbox 크기들 [(width, height), ...]
    """
    class_id: int
    class_name: str = ""
    centers: List[Tuple[float, float]] = field(default_factory=list)
    frame_indices: List[int] = field(default_factory=list)
    bbox_sizes: List[Tuple[float, float]] = field(default_factory=list)

    def add_detection(
        self,
        center: Tuple[float, float],
        frame_idx: int,
        bbox_size: Tuple[float, float],
    ) -> None:
        """탐지 추가."""
        self.centers.append(center)
        self.frame_indices.append(frame_idx)
        self.bbox_sizes.append(bbox_size)

    @property
    def avg_center(self) -> Optional[Tuple[float, float]]:
        """평균 중심점."""
        if not self.centers:
            return None
        avg_x = sum(c[0] for c in self.centers) / len(self.centers)
        avg_y = sum(c[1] for c in self.centers) / len(self.centers)
        return (avg_x, avg_y)

    @property
    def avg_bbox_size(self) -> Tuple[float, float]:
        """평균 bbox 크기."""
        if not self.bbox_sizes:
            return (50.0, 50.0)
        avg_w = sum(s[0] for s in self.bbox_sizes) / len(self.bbox_sizes)
        avg_h = sum(s[1] for s in self.bbox_sizes) / len(self.bbox_sizes)
        return (avg_w, avg_h)


class HandPathTracker:
    """
    손 경로 추적 및 상품 필터링.

    YOLO 탐지 결과에서 손(hand) 클래스를 추적하고,
    손 이동 경로와 교차하는 상품만 유효한 후보로 판단.
    """

    # 손 클래스 ID (YOLO 모델에서 hand로 라벨링된 클래스)
    HAND_CLASS_IDS: Set[int] = {0}  # 기본값, 모델에 따라 조정 필요

    def __init__(
        self,
        hand_class_ids: Optional[Set[int]] = None,
        min_hand_detections: int = 3,
        min_path_length: float = 30.0,
    ):
        """
        Initialize HandPathTracker.

        Args:
            hand_class_ids: 손으로 인식할 YOLO 클래스 ID 집합
            min_hand_detections: 최소 손 감지 횟수 (경로 유효성)
            min_path_length: 최소 손 이동 거리 (픽셀, 경로 유효성)
        """
        if hand_class_ids is not None:
            self.HAND_CLASS_IDS = hand_class_ids
        self.min_hand_detections = min_hand_detections
        self.min_path_length = min_path_length

        # 손 경로들 (여러 손이 있을 수 있음)
        self._hand_trajectories: List[HandTrajectory] = []
        # 현재 프레임의 손 경로 (단일 손으로 단순화)
        self._current_trajectory = HandTrajectory()

        # 상품 bbox 히스토리: class_id -> ProductBboxHistory
        self._product_histories: Dict[int, ProductBboxHistory] = {}

        # 프레임 카운트
        self._frame_count = 0

    def update_frame(
        self,
        detections: List,  # List[YOLODetection]
        frame_idx: int,
    ) -> None:
        """
        프레임 탐지 결과로 상태 업데이트.

        Args:
            detections: YOLO 탐지 결과 리스트
            frame_idx: 현재 프레임 인덱스
        """
        self._frame_count = max(self._frame_count, frame_idx + 1)

        for det in detections:
            # 손 탐지 처리
            if det.is_hand or det.cls in self.HAND_CLASS_IDS:
                center = det.center
                bbox_size = (det.x2 - det.x1, det.y2 - det.y1)
                self._current_trajectory.add_point(center, frame_idx, bbox_size)
            else:
                # 상품 탐지 처리
                class_id = det.cls
                if class_id not in self._product_histories:
                    self._product_histories[class_id] = ProductBboxHistory(
                        class_id=class_id,
                        class_name=det.name,
                    )

                center = det.center
                bbox_size = (det.x2 - det.x1, det.y2 - det.y1)
                self._product_histories[class_id].add_detection(
                    center, frame_idx, bbox_size
                )

    def has_valid_hand_path(self) -> bool:
        """
        유효한 손 경로가 있는지 확인.

        Returns:
            손 경로가 유효하면 True
        """
        trajectory = self._current_trajectory

        # 최소 감지 횟수 체크
        if len(trajectory.centers) < self.min_hand_detections:
            return False

        # 최소 이동 거리 체크
        if trajectory.get_path_length() < self.min_path_length:
            return False

        return True

    def filter_products_by_path(
        self,
        candidate_class_ids: Optional[List[int]] = None,
    ) -> List[int]:
        """
        손 경로와 교차하는 상품만 필터링.

        Args:
            candidate_class_ids: 후보 클래스 ID 리스트 (None이면 모든 감지된 상품)

        Returns:
            손 경로와 교차하는 유효한 클래스 ID 리스트
        """
        # 유효한 손 경로가 없으면 모든 후보 반환 (필터링 안 함)
        if not self.has_valid_hand_path():
            logger.info(
                f"[HAND_PATH] 유효한 손 경로 없음 "
                f"(detections={len(self._current_trajectory.centers)}, "
                f"path_length={self._current_trajectory.get_path_length():.1f}px), "
                f"필터링 스킵"
            )
            if candidate_class_ids is not None:
                return candidate_class_ids
            return list(self._product_histories.keys())

        trajectory = self._current_trajectory
        logger.info(
            f"[HAND_PATH] 손 경로 유효: "
            f"detections={len(trajectory.centers)}, "
            f"path_length={trajectory.get_path_length():.1f}px"
        )

        # 후보 결정
        if candidate_class_ids is not None:
            candidates = candidate_class_ids
        else:
            candidates = list(self._product_histories.keys())

        # 손 경로와 교차하는 상품 필터링
        valid_ids: List[int] = []
        filtered_ids: List[int] = []

        for class_id in candidates:
            history = self._product_histories.get(class_id)

            if history is None:
                # 히스토리가 없으면 필터링
                filtered_ids.append(class_id)
                continue

            avg_center = history.avg_center
            if avg_center is None:
                filtered_ids.append(class_id)
                continue

            avg_size = history.avg_bbox_size

            # 손 경로와 교차 체크
            if trajectory.intersects_bbox(avg_center, avg_size):
                valid_ids.append(class_id)
                logger.debug(
                    f"[HAND_PATH] class {class_id} ({history.class_name}): "
                    f"PASSED (center={avg_center}, size={avg_size})"
                )
            else:
                filtered_ids.append(class_id)
                logger.info(
                    f"[HAND_PATH] class {class_id} ({history.class_name}): "
                    f"FILTERED (손 경로와 미교차, center={avg_center})"
                )

        logger.info(
            f"[HAND_PATH] 필터링 결과: 통과={len(valid_ids)}개, 제외={len(filtered_ids)}개"
        )

        return valid_ids

    def get_stats(self) -> dict:
        """통계 정보 반환."""
        trajectory = self._current_trajectory
        return {
            "frame_count": self._frame_count,
            "hand_detections": len(trajectory.centers),
            "hand_path_length": trajectory.get_path_length(),
            "hand_path_valid": self.has_valid_hand_path(),
            "product_classes": len(self._product_histories),
            "product_class_ids": list(self._product_histories.keys()),
        }

    def clear(self) -> None:
        """상태 초기화."""
        self._current_trajectory = HandTrajectory()
        self._product_histories.clear()
        self._frame_count = 0
