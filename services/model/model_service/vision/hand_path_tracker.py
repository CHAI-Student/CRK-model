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
from typing import Any, Dict, List, Optional, Set, Tuple

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
class HandTrack:
    track_id: int
    trajectory: HandTrajectory = field(default_factory=HandTrajectory)
    last_center: Optional[Tuple[float, float]] = None
    last_frame_idx: int = -1

    def add_point(
        self,
        center: Tuple[float, float],
        frame_idx: int,
        bbox_size: Tuple[float, float],
    ) -> None:
        self.trajectory.add_point(center, frame_idx, bbox_size)
        self.last_center = center
        self.last_frame_idx = int(frame_idx)


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

    @property
    def detection_count(self) -> int:
        return len(self.centers)


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
        roi_y_split: Optional[float] = None,
        roi_vertical_region: Optional[str] = None,
        max_distance_px: float = 150.0,
        frame_window: int = 2,
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
        self.roi_y_split = roi_y_split
        self.roi_vertical_region = (
            roi_vertical_region.strip().lower()
            if isinstance(roi_vertical_region, str)
            else None
        )
        if self.roi_vertical_region not in {None, "upper", "lower"}:
            self.roi_vertical_region = None
        self.max_distance_px = max(0.0, float(max_distance_px))
        self.frame_window = max(0, int(frame_window))

        # 손 경로들 (여러 손이 있을 수 있음)
        self._hand_trajectories: List[HandTrajectory] = []
        # 현재 프레임의 손 경로 (단일 손으로 단순화)
        self._current_trajectory = HandTrajectory()
        self._hand_tracks: List[HandTrack] = []
        self._next_track_id = 1

        # 상품 bbox 히스토리: class_id -> ProductBboxHistory
        self._product_histories: Dict[int, ProductBboxHistory] = {}

        # 프레임 카운트
        self._frame_count = 0

    def _center_in_roi(self, center: Tuple[float, float]) -> bool:
        if self.roi_y_split is None or self.roi_vertical_region is None:
            return True
        center_y = float(center[1])
        split = float(self.roi_y_split)
        if self.roi_vertical_region == "lower":
            return center_y >= split
        return center_y <= split

    def _track_is_valid(self, track: HandTrack) -> bool:
        trajectory = track.trajectory
        if len(trajectory.centers) < self.min_hand_detections:
            return False
        return trajectory.get_path_length() >= self.min_path_length

    def _valid_hand_tracks(self) -> List[HandTrack]:
        return [track for track in self._hand_tracks if self._track_is_valid(track)]

    def _max_track_path_length(self) -> float:
        if not self._hand_tracks:
            return 0.0
        return max(track.trajectory.get_path_length() for track in self._hand_tracks)

    def _assign_hand_detections(
        self,
        hands: List[tuple[Tuple[float, float], Tuple[float, float]]],
        frame_idx: int,
    ) -> None:
        if not hands:
            return

        candidate_pairs: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self._hand_tracks):
            if track.last_center is None:
                continue
            if int(frame_idx) - int(track.last_frame_idx) > self.frame_window:
                continue
            for hand_index, (center, _bbox_size) in enumerate(hands):
                distance = self._center_distance(track.last_center, center)
                if distance <= self.max_distance_px:
                    candidate_pairs.append((distance, track_index, hand_index))

        assigned_tracks: Set[int] = set()
        assigned_hands: Set[int] = set()
        for _distance, track_index, hand_index in sorted(candidate_pairs):
            if track_index in assigned_tracks or hand_index in assigned_hands:
                continue
            center, bbox_size = hands[hand_index]
            self._hand_tracks[track_index].add_point(center, frame_idx, bbox_size)
            assigned_tracks.add(track_index)
            assigned_hands.add(hand_index)

        for hand_index, (center, bbox_size) in enumerate(hands):
            if hand_index in assigned_hands:
                continue
            track = HandTrack(track_id=self._next_track_id)
            self._next_track_id += 1
            track.add_point(center, frame_idx, bbox_size)
            self._hand_tracks.append(track)

    @property
    def hand_path_valid_upper_roi(self) -> bool:
        return self.has_valid_hand_path()

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

        hands: List[tuple[Tuple[float, float], Tuple[float, float]]] = []
        for det in detections:
            # 손 탐지 처리
            if det.is_hand or det.cls in self.HAND_CLASS_IDS:
                center = det.center
                if not self._center_in_roi(center):
                    continue
                bbox_size = (det.x2 - det.x1, det.y2 - det.y1)
                self._current_trajectory.add_point(center, frame_idx, bbox_size)
                hands.append((center, bbox_size))
            else:
                # 상품 탐지 처리
                class_id = det.cls
                center = det.center
                if not self._center_in_roi(center):
                    continue
                if class_id not in self._product_histories:
                    self._product_histories[class_id] = ProductBboxHistory(
                        class_id=class_id,
                        class_name=det.name,
                    )

                bbox_size = (det.x2 - det.x1, det.y2 - det.y1)
                self._product_histories[class_id].add_detection(
                    center, frame_idx, bbox_size
                )
        self._assign_hand_detections(hands, frame_idx)

    @staticmethod
    def _center_distance(
        first: Tuple[float, float],
        second: Tuple[float, float],
    ) -> float:
        return math.sqrt((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2)

    @staticmethod
    def _bbox_diagonal(size: Tuple[float, float]) -> float:
        return math.sqrt(float(size[0]) ** 2 + float(size[1]) ** 2)

    def _hand_points_near_frame(
        self,
        track: HandTrack,
        frame_idx: int,
    ) -> list[tuple[Tuple[float, float], Tuple[float, float]]]:
        trajectory = track.trajectory
        paired = list(
            zip(
                trajectory.centers,
                trajectory.frame_indices,
                trajectory.bbox_history,
            )
        )
        near = [
            (center, bbox_size)
            for center, hand_frame_idx, bbox_size in paired
            if abs(int(hand_frame_idx) - int(frame_idx)) <= self.frame_window
        ]
        return near

    def _product_interaction_metrics(
        self,
        history: Optional[ProductBboxHistory],
    ) -> dict[str, Any]:
        valid_tracks = self._valid_hand_tracks()
        base_payload = {
            "handInteractionPassed": False,
            "handNearFrameCount": 0,
            "handNearVoteRatio": 0.0,
            "minHandDistancePx": None,
            "handTrackId": None,
            "handTrackCount": len(self._hand_tracks),
            "validHandTrackCount": len(valid_tracks),
            "handTrackNearFrameCount": 0,
        }
        if history is None or not history.centers:
            return base_payload
        if not valid_tracks:
            return base_payload

        best_metrics: Optional[dict[str, Any]] = None
        for track in valid_tracks:
            metrics = self._product_interaction_metrics_for_track(track, history)
            if best_metrics is None or self._is_better_track_metric(
                metrics,
                best_metrics,
            ):
                best_metrics = metrics

        if best_metrics is None:
            return base_payload
        return {
            **base_payload,
            **best_metrics,
            "handInteractionPassed": int(best_metrics["handNearFrameCount"]) > 0,
        }

    @staticmethod
    def _metric_distance(metric: dict[str, Any]) -> float:
        value = metric.get("minHandDistancePx")
        if value is None:
            return float("inf")
        return float(value)

    def _is_better_track_metric(
        self,
        candidate: dict[str, Any],
        current: dict[str, Any],
    ) -> bool:
        candidate_near = int(candidate.get("handNearFrameCount", 0) or 0)
        current_near = int(current.get("handNearFrameCount", 0) or 0)
        if candidate_near != current_near:
            return candidate_near > current_near

        candidate_ratio = float(candidate.get("handNearVoteRatio", 0.0) or 0.0)
        current_ratio = float(current.get("handNearVoteRatio", 0.0) or 0.0)
        if candidate_ratio != current_ratio:
            return candidate_ratio > current_ratio

        return self._metric_distance(candidate) < self._metric_distance(current)

    def _product_interaction_metrics_for_track(
        self,
        track: HandTrack,
        history: ProductBboxHistory,
    ) -> dict[str, Any]:
        near_count = 0
        min_distance: Optional[float] = None
        for center, frame_idx, product_size in zip(
            history.centers,
            history.frame_indices,
            history.bbox_sizes,
        ):
            product_near = False
            for hand_center, hand_size in self._hand_points_near_frame(
                track,
                frame_idx,
            ):
                distance = self._center_distance(center, hand_center)
                min_distance = (
                    distance
                    if min_distance is None
                    else min(float(min_distance), distance)
                )
                hand_radius = self._bbox_diagonal(hand_size) / 2.0
                product_radius = self._bbox_diagonal(product_size) / 2.0
                distance_tolerance = min(
                    self.max_distance_px,
                    hand_radius + product_radius * 0.5,
                )
                expanded_bbox_intersects = (
                    abs(hand_center[0] - center[0])
                    <= (float(product_size[0]) / 2.0 + hand_radius)
                    and abs(hand_center[1] - center[1])
                    <= (float(product_size[1]) / 2.0 + hand_radius)
                )
                if distance <= distance_tolerance or expanded_bbox_intersects:
                    product_near = True
                    break
            if product_near:
                near_count += 1

        detection_count = max(1, history.detection_count)
        return {
            "handInteractionPassed": near_count > 0,
            "handNearFrameCount": int(near_count),
            "handNearVoteRatio": round(float(near_count) / float(detection_count), 4),
            "minHandDistancePx": (
                round(float(min_distance), 1) if min_distance is not None else None
            ),
            "handTrackId": int(track.track_id),
            "handTrackCount": len(self._hand_tracks),
            "validHandTrackCount": len(self._valid_hand_tracks()),
            "handTrackNearFrameCount": int(near_count),
        }

    def hand_interaction_metrics(
        self,
        candidate_class_ids: Optional[List[int]] = None,
    ) -> Dict[int, dict[str, Any]]:
        if candidate_class_ids is not None:
            candidates = [int(class_id) for class_id in candidate_class_ids]
        else:
            candidates = list(self._product_histories.keys())

        path_valid = self.has_valid_hand_path()
        hand_track_count = len(self._hand_tracks)
        valid_hand_track_count = len(self._valid_hand_tracks())
        metrics: Dict[int, dict[str, Any]] = {}
        for class_id in candidates:
            payload = {
                "handPathValid": bool(path_valid),
                "handPathValidUpperRoi": bool(path_valid),
                "handInteractionPassed": False,
                "handNearFrameCount": 0,
                "handNearVoteRatio": 0.0,
                "minHandDistancePx": None,
                "handTrackId": None,
                "handTrackCount": hand_track_count,
                "validHandTrackCount": valid_hand_track_count,
                "handTrackNearFrameCount": 0,
            }
            if path_valid:
                payload.update(
                    self._product_interaction_metrics(
                        self._product_histories.get(class_id)
                    )
                )
            metrics[class_id] = payload
        return metrics

    def has_valid_hand_path(self) -> bool:
        """
        유효한 손 경로가 있는지 확인.

        Returns:
            손 경로가 유효하면 True
        """
        if self._hand_tracks:
            return any(self._track_is_valid(track) for track in self._hand_tracks)

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
                f"tracks={len(self._hand_tracks)}, "
                f"max_path_length={self._max_track_path_length():.1f}px), "
                f"필터링 스킵"
            )
            if candidate_class_ids is not None:
                return candidate_class_ids
            return list(self._product_histories.keys())

        logger.info(
            f"[HAND_PATH] 손 경로 유효: "
            f"tracks={len(self._hand_tracks)}, "
            f"valid_tracks={len(self._valid_hand_tracks())}, "
            f"max_path_length={self._max_track_path_length():.1f}px"
        )

        # 후보 결정
        if candidate_class_ids is not None:
            candidates = candidate_class_ids
        else:
            candidates = list(self._product_histories.keys())

        # 손 경로와 교차하는 상품 필터링
        valid_ids: List[int] = []
        filtered_ids: List[int] = []

        metrics_by_class = self.hand_interaction_metrics(candidates)
        for class_id in candidates:
            metrics = metrics_by_class.get(class_id, {})
            if bool(metrics.get("handInteractionPassed")):
                valid_ids.append(class_id)
                logger.debug(
                    f"[HAND_PATH] class {class_id}: PASSED "
                    f"(near_frames={metrics.get('handNearFrameCount')}, "
                    f"min_distance={metrics.get('minHandDistancePx')})"
                )
            else:
                filtered_ids.append(class_id)
                logger.info(
                    f"[HAND_PATH] class {class_id}: FILTERED "
                    f"(near_frames={metrics.get('handNearFrameCount')}, "
                    f"min_distance={metrics.get('minHandDistancePx')})"
                )

        logger.info(
            f"[HAND_PATH] filter result: passed={len(valid_ids)} "
            f"filtered={len(filtered_ids)}"
        )
        return valid_ids

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
        valid_tracks = self._valid_hand_tracks()
        return {
            "frame_count": self._frame_count,
            "hand_detections": len(trajectory.centers),
            "hand_path_length": self._max_track_path_length(),
            "hand_path_valid": self.has_valid_hand_path(),
            "hand_path_valid_upper_roi": self.hand_path_valid_upper_roi,
            "hand_track_count": len(self._hand_tracks),
            "valid_hand_track_count": len(valid_tracks),
            "roi_y_split": self.roi_y_split,
            "roi_vertical_region": self.roi_vertical_region,
            "product_classes": len(self._product_histories),
            "product_class_ids": list(self._product_histories.keys()),
        }

    def clear(self) -> None:
        """상태 초기화."""
        self._current_trajectory = HandTrajectory()
        self._hand_tracks.clear()
        self._next_track_id = 1
        self._product_histories.clear()
        self._frame_count = 0
