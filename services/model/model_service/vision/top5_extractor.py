"""
Top-5 Candidate Extractor.

YOLO 감지 결과에서 Top-5 후보군 추출.

파이프라인:
1. YOLO 감지 (conf=0.01, 매우 낮은 threshold)
2. 손/상품 분리
3. 손 근접 필터링
4. Top-5 confidence 추출
5. (옵션) Multi-View Ensemble
"""

from dataclasses import dataclass
from typing import List, Optional
import logging

from .yolo_wrapper import YOLODetection
from .hand_filter import HandProximityFilter
from model_service.engine.models import EnsembleResult
from model_service.core.config import config

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """추출 결과."""
    candidates: List[YOLODetection]  # Top-K 후보군
    hands: List[YOLODetection]       # 감지된 손
    total_detected: int              # 총 감지 수
    filtered_count: int              # 필터 후 수


class Top5Extractor:
    """
    Top-5 후보군 추출기.

    손 근접 필터링 후 confidence 상위 5개 추출.

    Attributes:
        hand_filter: 손 근접 필터
        top_k: 추출할 후보 수 (기본값 5)
    """

    def __init__(
        self,
        max_distance_px: Optional[float] = None,
        top_k: Optional[int] = None,
        hand_class_id: Optional[int] = None,
    ):
        """
        추출기 초기화.

        Args:
            max_distance_px: 손-상품 최대 거리
            top_k: 추출할 후보 수
            hand_class_id: 손 클래스 ID
        """
        self.hand_filter = HandProximityFilter(
            max_distance_px=max_distance_px,
            hand_class_id=hand_class_id,
        )
        self.top_k = top_k or config.vision.top_k

    def extract(
        self,
        detections: List[YOLODetection],
    ) -> ExtractionResult:
        """
        Top-K 후보군 추출.

        Args:
            detections: YOLO 감지 결과

        Returns:
            ExtractionResult
        """
        # 손 근접 필터링
        filter_result = self.hand_filter.filter(detections)

        # confidence 정렬 후 Top-K 추출
        sorted_products = sorted(
            filter_result.filtered_products,
            key=lambda d: d.conf,
            reverse=True,
        )
        candidates = sorted_products[:self.top_k]

        logger.info(
            f"Extracted Top-{len(candidates)} from "
            f"{len(filter_result.filtered_products)} filtered "
            f"({len(filter_result.all_products)} total products)"
        )

        return ExtractionResult(
            candidates=candidates,
            hands=filter_result.hands,
            total_detected=len(filter_result.all_products),
            filtered_count=len(filter_result.filtered_products),
        )

    def extract_from_raw(
        self,
        detection_data: List[dict],
    ) -> ExtractionResult:
        """
        딕셔너리 데이터에서 추출 (테스트/API용).

        Args:
            detection_data: [{"xyxy": [...], "conf": ..., "cls": ..., "name": ...}, ...]

        Returns:
            ExtractionResult
        """
        from .yolo_wrapper import YOLOWrapper
        detections = YOLOWrapper.parse_detection_list(detection_data)
        return self.extract(detections)

    def process_single_camera(
        self,
        detections: List[YOLODetection],
    ) -> List[EnsembleResult]:
        """
        단일 카메라 처리 (앙상블 없음).

        Top-K 추출 후 EnsembleResult 형식으로 변환.

        Args:
            detections: YOLO 감지 결과

        Returns:
            EnsembleResult 리스트
        """
        result = self.extract(detections)

        ensemble_results = []
        for det in result.candidates:
            if det.is_hand:
                continue

            er = EnsembleResult(
                class_id=det.cls,
                class_name=det.name,
                top_confidence=det.conf,
                side_confidence=0.0,
                combined_confidence=det.conf,
                vote_count=1,
            )
            ensemble_results.append(er)

        return ensemble_results
