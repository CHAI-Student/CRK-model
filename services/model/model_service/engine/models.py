"""
Data Models for Model Service.

상품 판단 서비스의 핵심 데이터 모델 정의.

핵심 플로우:
1. EnsembleResult: Top+Side 카메라 앙상블 결과
2. CountEstimate: 무게 기반 개수 추정
3. ProductJudgment: 개별 상품 판단 결과
4. JudgmentResult: 최종 상품 판단 결과 (Node.js 전달용)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import time


class JudgmentStatus(Enum):
    """상품 판단 상태."""

    COMPLETE = "complete"         # 무게가 완전히 설명됨
    PARTIAL = "partial"           # 일부만 설명됨 (잔여 무게 있음)
    UNCERTAIN = "uncertain"       # 확신할 수 없음 (신뢰도 낮음)
    NO_DETECTION = "no_detection" # 감지된 상품 없음


@dataclass
class EnsembleResult:
    """
    Multi-View Ensemble 결과.

    Top + Side 카메라에서 앙상블된 상품 후보.

    Attributes:
        class_id: 클래스 ID
        class_name: 클래스 이름
        top_confidence: Top 카메라 신뢰도
        side_confidence: Side 카메라 신뢰도
        combined_confidence: 앙상블 결합 신뢰도
        vote_count: 양쪽 합의 (2=양쪽, 1=한쪽)
    """
    class_id: int
    class_name: str
    top_confidence: float
    side_confidence: float
    combined_confidence: float
    vote_count: int = 1  # 1 or 2 (양쪽 합의)

    @property
    def is_consensus(self) -> bool:
        """양쪽 카메라에서 합의되었는지."""
        return self.vote_count == 2

    def to_dict(self) -> dict:
        """딕셔너리 변환."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "top_confidence": round(self.top_confidence, 4),
            "side_confidence": round(self.side_confidence, 4),
            "combined_confidence": round(self.combined_confidence, 4),
            "vote_count": self.vote_count,
            "is_consensus": self.is_consensus,
        }


@dataclass
class CountEstimate:
    """
    무게 기반 개수 추정 결과.

    Attributes:
        product_id: 상품 ID
        product_name: 상품 이름
        count: 추정 개수
        unit_weight: 단위 무게 (g)
        expected_weight: 예상 무게 (unit_weight * count)
        actual_weight: 실제 무게 변화량 (절대값)
        match_score: 매칭 점수 (0.0 ~ 1.0)
        vision_confidence: Vision 신뢰도
        validated: 허용 오차 내 검증 여부
        unit_price: 단가 (원) - v4.7 추가
    """
    product_id: int
    product_name: str
    count: int
    unit_weight: float
    expected_weight: float
    actual_weight: float
    match_score: float
    vision_confidence: float
    validated: bool
    unit_price: int = 0  # v4.7: active_products에서 가격 정보

    @property
    def weight_error(self) -> float:
        """무게 오차 (절대값)."""
        return abs(self.actual_weight - self.expected_weight)

    @property
    def error_rate(self) -> float:
        """오차율 (0.0 ~ 1.0)."""
        if self.expected_weight == 0:
            return 1.0
        return self.weight_error / self.expected_weight

    def to_dict(self) -> dict:
        """딕셔너리 변환."""
        return {
            "product_id": self.product_id,
            "product_name": self.product_name,
            "count": self.count,
            "unit_weight": round(self.unit_weight, 1),
            "expected_weight": round(self.expected_weight, 1),
            "actual_weight": round(self.actual_weight, 1),
            "weight_error": round(self.weight_error, 1),
            "error_rate": round(self.error_rate, 4),
            "match_score": round(self.match_score, 4),
            "vision_confidence": round(self.vision_confidence, 4),
            "validated": self.validated,
            "unit_price": self.unit_price,  # v4.7
        }


@dataclass
class ProductJudgment:
    """
    개별 상품 판단 결과.

    Node.js로 전달되는 개별 상품 정보.

    Attributes:
        product_id: 상품 ID
        name: 상품 이름
        count: 개수
        unit_price: 단가 (원)
        total_price: 총 가격 (원)
        confidence: 신뢰도 (0.0 ~ 1.0)
        unit_weight: 단위 무게 (g)
    """
    product_id: int
    name: str
    count: int
    unit_price: int
    total_price: int
    confidence: float
    unit_weight: float = 0.0

    def to_dict(self) -> dict:
        """딕셔너리 변환 (Node.js 형식)."""
        return {
            "productId": self.product_id,
            "name": self.name,
            "count": self.count,
            "unitPrice": self.unit_price,
            "totalPrice": self.total_price,
            "confidence": round(self.confidence, 2),
        }


@dataclass
class JudgmentResult:
    """
    최종 상품 판단 결과.

    Node.js Orchestrator로 전달되는 전체 결과.

    Attributes:
        products: 판단된 상품 리스트
        total_price: 총 가격 (원)
        confidence: 전체 신뢰도 (0.0 ~ 1.0)
        status: 판단 상태 (complete/partial/uncertain/no_detection)
        weight_delta: 무게 변화량 (음수 = 제거)
        weight_explained: 설명된 무게 (양수)
        weight_residual: 잔여 무게 (설명 안 됨)
        timestamp: 판단 시각 (Unix timestamp)
    """
    products: List[ProductJudgment] = field(default_factory=list)
    total_price: int = 0
    confidence: float = 0.0
    status: JudgmentStatus = JudgmentStatus.NO_DETECTION
    weight_delta: float = 0.0
    weight_explained: float = 0.0
    weight_residual: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def is_removal(self) -> bool:
        """상품 제거 여부 (무게 감소)."""
        return self.weight_delta < 0

    @property
    def is_success(self) -> bool:
        """성공적인 판단 여부."""
        return self.status in [JudgmentStatus.COMPLETE, JudgmentStatus.PARTIAL]

    @property
    def product_count(self) -> int:
        """총 상품 개수."""
        return sum(p.count for p in self.products)

    def to_node_response(self) -> dict:
        """
        Node.js 서버 응답 형식으로 변환.

        Returns:
            {
                "success": bool,
                "products": [...],
                "totalPrice": int,
                "status": str,
                "confidence": float,
                "weightInfo": {...},
                "timestamp": float,
            }
        """
        return {
            "success": self.is_success,
            "products": [p.to_dict() for p in self.products],
            "totalPrice": self.total_price,
            "status": self.status.value,
            "confidence": round(self.confidence, 2),
            "weightInfo": {
                "delta": round(self.weight_delta, 1),
                "explained": round(self.weight_explained, 1),
                "residual": round(self.weight_residual, 1),
            },
            "productCount": self.product_count,
            "isRemoval": self.is_removal,
            "timestamp": self.timestamp,
        }


@dataclass
class ProductInfo:
    """
    상품 정보 (데이터베이스용).

    Attributes:
        product_id: 상품 ID (내부 ID)
        name: 상품 이름 (한글)
        category: 카테고리
        weight: 단위 무게 (g)
        price: 가격 (원)
        barcode: 바코드 (선택)
        stock: 재고 수량 (선택)
        image_count: 등록된 이미지 수
        yolo_class_id: YOLO 모델의 클래스 ID (선택)
        yolo_class_name: YOLO 모델의 클래스 이름 (선택)
        product_idx: IF11 상품 ID (선택, 예: "P17689508539755305")
        has_loadcell: 로드셀 사용 여부 (v4.8, "true"/"false"/"null")
    """
    product_id: int
    name: str
    category: str
    weight: float  # grams
    price: int     # won
    barcode: Optional[str] = None
    stock: int = 0
    image_count: int = 0
    # YOLO-IF11 매핑 필드
    yolo_class_id: Optional[int] = None
    yolo_class_name: Optional[str] = None
    product_idx: Optional[str] = None
    has_loadcell: str = "true"  # v4.8: 추가 (기본값: true)

    def to_dict(self) -> dict:
        """딕셔너리 변환."""
        return {
            "product_id": self.product_id,
            "name": self.name,
            "category": self.category,
            "weight": self.weight,
            "price": self.price,
            "barcode": self.barcode,
            "stock": self.stock,
            "image_count": self.image_count,
            "yolo_class_id": self.yolo_class_id,
            "yolo_class_name": self.yolo_class_name,
            "product_idx": self.product_idx,
            "has_loadcell": self.has_loadcell,  # v4.8
        }
