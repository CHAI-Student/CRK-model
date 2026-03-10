"""
Voting-based Ensemble for Multi-Frame YOLO Results.

Aggregates YOLO detections across multiple frames using voting.
Each class is counted by appearance frequency and confidence scores.

Usage:
    ensemble = VotingEnsemble()

    # Add votes from each frame
    for detection in detections:
        ensemble.add_vote(detection.cls, detection.conf)

    # Get final results
    results = ensemble.get_results(total_frames=100)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class VoteCount:
    """
    Vote count for a single class.

    Attributes:
        count: Number of frames where this class appeared
        max_confidence: Maximum confidence score
        sum_confidence: Sum of all confidence scores (for averaging)
        class_name: Class name from YOLO
    """
    count: int = 0
    max_confidence: float = 0.0
    sum_confidence: float = 0.0
    class_name: str = ""

    def add(self, confidence: float, class_name: str = "") -> None:
        """Add a vote with confidence score."""
        self.count += 1
        self.max_confidence = max(self.max_confidence, confidence)
        self.sum_confidence += confidence
        if class_name and not self.class_name:
            self.class_name = class_name

    @property
    def avg_confidence(self) -> float:
        """Average confidence score."""
        if self.count == 0:
            return 0.0
        return self.sum_confidence / self.count


@dataclass
class VoteResult:
    """
    Final vote result for a class.

    Attributes:
        class_id: YOLO class ID
        class_name: YOLO class name
        vote_count: Number of frames where this class appeared
        max_confidence: Maximum confidence score across all frames
        avg_confidence: Average confidence score
        vote_ratio: Ratio of frames where this class appeared (0.0 ~ 1.0)
        top_detected: Whether detected in top camera
        side_detected: Whether detected in side camera
        top_vote_count: Vote count from top camera
        side_vote_count: Vote count from side camera
        top_max_confidence: Max confidence from top camera
        side_max_confidence: Max confidence from side camera
        weighted_confidence: Weighted combined confidence (with bonus applied)
    """
    class_id: int
    class_name: str
    vote_count: int
    max_confidence: float
    avg_confidence: float
    vote_ratio: float = 0.0
    # Camera-specific fields
    top_detected: bool = False
    side_detected: bool = False
    top_vote_count: int = 0
    side_vote_count: int = 0
    top_max_confidence: float = 0.0
    side_max_confidence: float = 0.0
    weighted_confidence: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "vote_count": self.vote_count,
            "max_confidence": round(self.max_confidence, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "vote_ratio": round(self.vote_ratio, 4),
            "top_detected": self.top_detected,
            "side_detected": self.side_detected,
            "top_vote_count": self.top_vote_count,
            "side_vote_count": self.side_vote_count,
            "top_max_confidence": round(self.top_max_confidence, 4),
            "side_max_confidence": round(self.side_max_confidence, 4),
            "weighted_confidence": round(self.weighted_confidence, 4),
        }


class VotingEnsemble:
    """
    Voting-based ensemble for multi-frame YOLO results.

    Aggregates detections across frames and produces ranked results
    based on vote count and confidence.
    """

    def __init__(self, min_vote_ratio: float = 0.05):
        """
        Initialize voting ensemble.

        Args:
            min_vote_ratio: Minimum vote ratio to include in results (default: 5%)
        """
        self._votes: Dict[int, VoteCount] = {}
        self._total_frames = 0
        self._min_vote_ratio = min_vote_ratio

    def add_vote(
        self,
        class_id: int,
        confidence: float,
        class_name: str = "",
    ) -> None:
        """
        Add a vote for a class.

        Args:
            class_id: YOLO class ID
            confidence: Detection confidence (0.0 ~ 1.0)
            class_name: YOLO class name
        """
        if class_id not in self._votes:
            self._votes[class_id] = VoteCount()
        self._votes[class_id].add(confidence, class_name)

    def increment_frame_count(self) -> None:
        """Increment total frame count."""
        self._total_frames += 1

    def set_frame_count(self, count: int) -> None:
        """Set total frame count."""
        self._total_frames = count

    @property
    def total_frames(self) -> int:
        """Total number of frames processed."""
        return self._total_frames

    @property
    def votes(self) -> Dict[int, VoteCount]:
        """Raw vote counts by class ID."""
        return self._votes

    def get_results(
        self,
        total_frames: Optional[int] = None,
        top_k: int = 10,
        min_vote_ratio: Optional[float] = None,
    ) -> List[VoteResult]:
        """
        Get voting results sorted by vote count.

        Args:
            total_frames: Override total frame count for ratio calculation
            top_k: Maximum number of results to return
            min_vote_ratio: Override minimum vote ratio filter

        Returns:
            List of VoteResult sorted by vote_count descending
        """
        frames = total_frames if total_frames is not None else self._total_frames
        min_ratio = min_vote_ratio if min_vote_ratio is not None else self._min_vote_ratio

        results = []
        for class_id, vote in self._votes.items():
            vote_ratio = vote.count / frames if frames > 0 else 0.0

            # Filter by minimum vote ratio
            if vote_ratio < min_ratio:
                continue

            result = VoteResult(
                class_id=class_id,
                class_name=vote.class_name,
                vote_count=vote.count,
                max_confidence=vote.max_confidence,
                avg_confidence=vote.avg_confidence,
                vote_ratio=vote_ratio,
            )
            results.append(result)

        # Sort by vote count (descending), then by max confidence
        results.sort(key=lambda x: (x.vote_count, x.max_confidence), reverse=True)

        return results[:top_k]

    def clear(self) -> None:
        """Clear all votes."""
        self._votes.clear()
        self._total_frames = 0

    def merge(self, other: "VotingEnsemble") -> None:
        """
        Merge another ensemble's votes into this one.

        Args:
            other: Another VotingEnsemble to merge
        """
        for class_id, other_vote in other.votes.items():
            if class_id not in self._votes:
                self._votes[class_id] = VoteCount()

            vote = self._votes[class_id]
            vote.count += other_vote.count
            vote.max_confidence = max(vote.max_confidence, other_vote.max_confidence)
            vote.sum_confidence += other_vote.sum_confidence
            if other_vote.class_name and not vote.class_name:
                vote.class_name = other_vote.class_name

        self._total_frames += other.total_frames

    @staticmethod
    def combine(
        top_ensemble: "VotingEnsemble",
        side_ensemble: "VotingEnsemble",
        top_weight: float = 0.5,
        side_weight: float = 0.5,
        common_class_bonus: float = 0.2,
        product_weights: Optional[Dict[int, float]] = None,
        top_only_weight: Optional[float] = None,
        side_only_weight: Optional[float] = None,
    ) -> List[VoteResult]:
        """
        Combine Top and Side camera ensembles with weighted confidence.

        Weight calculation:
        - Both cameras detected: top_conf * top_weight + side_conf * side_weight + bonus
        - Top only: top_conf * top_weight
        - Side only: side_conf * side_weight

        Args:
            top_ensemble: Top camera voting results
            side_ensemble: Side camera voting results
            top_weight: Weight for top camera votes (default: 0.5)
            side_weight: Weight for side camera votes (default: 0.5)
            common_class_bonus: Bonus for classes detected in both cameras (default: 0.2)
            product_weights: {class_id: weight_in_grams} for logging (v4.6)
            top_only_weight: Weight for top-only detection (default: None → uses top_weight)
            side_only_weight: Weight for side-only detection (default: None → uses side_weight)

        Returns:
            Combined and sorted VoteResult list (by weighted_confidence descending)
        """
        if product_weights is None:
            product_weights = {}
        logger.info(f"[ENSEMBLE] ========== Top/Side 결합 ==========")
        logger.info(
            f"[ENSEMBLE] Top: {top_ensemble.total_frames}프레임, "
            f"{len(top_ensemble.votes)}개 클래스"
        )
        logger.info(
            f"[ENSEMBLE] Side: {side_ensemble.total_frames}프레임, "
            f"{len(side_ensemble.votes)}개 클래스"
        )
        logger.info(
            f"[ENSEMBLE] 가중치: top={top_weight}, side={side_weight}, "
            f"bonus={common_class_bonus}"
        )

        combined: Dict[int, VoteResult] = {}
        total_frames = top_ensemble.total_frames + side_ensemble.total_frames

        # Add top camera votes
        for class_id, vote in top_ensemble.votes.items():
            combined[class_id] = VoteResult(
                class_id=class_id,
                class_name=vote.class_name,
                vote_count=vote.count,
                max_confidence=vote.max_confidence,
                avg_confidence=vote.avg_confidence,
                vote_ratio=0.0,
                # Camera-specific fields
                top_detected=True,
                side_detected=False,
                top_vote_count=vote.count,
                side_vote_count=0,
                top_max_confidence=vote.max_confidence,
                side_max_confidence=0.0,
            )

        # Merge side camera votes
        for class_id, vote in side_ensemble.votes.items():
            if class_id in combined:
                # Both cameras detected this class
                existing = combined[class_id]
                existing.side_detected = True
                existing.side_vote_count = vote.count
                existing.side_max_confidence = vote.max_confidence
                existing.vote_count = existing.top_vote_count + vote.count
                existing.max_confidence = max(existing.max_confidence, vote.max_confidence)
                existing.avg_confidence = (
                    (existing.avg_confidence * existing.top_vote_count +
                     vote.avg_confidence * vote.count) /
                    (existing.top_vote_count + vote.count)
                )
            else:
                # Side only
                combined[class_id] = VoteResult(
                    class_id=class_id,
                    class_name=vote.class_name,
                    vote_count=vote.count,
                    max_confidence=vote.max_confidence,
                    avg_confidence=vote.avg_confidence,
                    vote_ratio=0.0,
                    # Camera-specific fields
                    top_detected=False,
                    side_detected=True,
                    top_vote_count=0,
                    side_vote_count=vote.count,
                    top_max_confidence=0.0,
                    side_max_confidence=vote.max_confidence,
                )

        # Calculate vote ratios and weighted confidence
        for result in combined.values():
            result.vote_ratio = result.vote_count / total_frames if total_frames > 0 else 0.0

            # Calculate weighted confidence with bonus
            top_conf = result.top_max_confidence
            side_conf = result.side_max_confidence

            if result.top_detected and result.side_detected:
                # Both cameras: weighted average + 동적 보너스
                # 동적 보너스: 두 신뢰도의 최소값 기반 (0 ~ common_class_bonus)
                # 두 카메라 모두 높은 신뢰도일 때만 최대 보너스 적용
                dynamic_bonus = min(top_conf, side_conf) * common_class_bonus
                weighted_conf = (
                    top_conf * top_weight +
                    side_conf * side_weight +
                    dynamic_bonus
                )
                logger.debug(
                    f"[ENSEMBLE] class {result.class_id}: 동적 보너스={dynamic_bonus:.3f} "
                    f"(top={top_conf:.2f}, side={side_conf:.2f})"
                )
            elif result.top_detected:
                # Top only
                effective_weight = top_only_weight if top_only_weight is not None else top_weight
                weighted_conf = top_conf * effective_weight
            else:
                # Side only
                effective_weight = side_only_weight if side_only_weight is not None else side_weight
                weighted_conf = side_conf * effective_weight

            result.weighted_confidence = min(weighted_conf, 1.0)

        # Sort by weighted_confidence descending
        results = sorted(
            combined.values(),
            key=lambda x: x.weighted_confidence,
            reverse=True,
        )

        # Log consensus count
        consensus_count = sum(1 for r in results if r.top_detected and r.side_detected)
        logger.info(f"[ENSEMBLE] 결합 결과: {len(results)}개 후보")
        logger.info(f"[ENSEMBLE] 양쪽 감지: {consensus_count}개")
        for i, r in enumerate(results[:5]):
            # v4.6: combined_conf와 product_weight 출력
            weight_str = ""
            if r.class_id in product_weights:
                weight_str = f", product_weight={product_weights[r.class_id]:.0f}g"
            logger.info(
                f"  [{i+1}] {r.class_name}: combined_conf={r.weighted_confidence:.3f}, "
                f"top={r.top_detected}, side={r.side_detected}{weight_str}"
            )

        return results
