"""
Product Decision Engine.

최종 상품 결정 엔진.

Vision 후보군 + 무게 검증을 결합하여 최종 상품과 개수를 결정합니다.

전략:
1. 단일 상품 매칭 우선 시도
2. 실패 시 다중 상품 조합 시도
3. 완전/불완전 상태 판별
4. Vision 실패 시 Loadcell-only 폴백 (무게로 최근접 상품 추정)
5. Node.js 응답 형식으로 결과 반환
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any, List, Optional

from model_service.core.config import config
from model_service.session.active_product_store import ActiveProductStore

from .models import (
    CountEstimate,
    EnsembleResult,
    JudgmentResult,
    JudgmentStatus,
    ProductJudgment,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _WeightOnlyCandidate:
    """Minimal product view used by the loadcell-only fallback path."""

    product_id: int
    product_idx: str
    name: str
    weight: float
    price: int
    stock: int


@dataclass(frozen=True)
class _DetectedSingleEvidence:
    """Detected class evidence used by the final fallback paths."""

    class_id: int
    name: str
    votes: int
    confidence: float
    source: str
    trusted: bool = False
    motion_gate_passed: bool = True
    weight_gate_passed: Optional[bool] = None
    strong: bool = False
    rank: Optional[int] = None
    stage_score: float = 0.0
    top_confidence: float = 0.0
    side_confidence: float = 0.0
    top_votes: int = 0
    side_votes: int = 0
    score_reason: str = ""


@dataclass(frozen=True)
class _StageEvidenceSummary:
    """Camera-aware stage-count signal used for candidate-outside evidence."""

    votes: int
    confidence: float
    stage_score: float
    top_confidence: float
    side_confidence: float
    top_votes: int
    side_votes: int
    top_passed: int
    side_passed: int
    motion_gate_passed: bool
    top_motion_passed: bool
    side_motion_passed: bool
    score_reason: str


@dataclass(frozen=True)
class _SegmentWeightTarget:
    """One loadcell movement target used by segment-first matching."""

    weight: float
    source: str
    segment_index: int
    evidence_required: bool = False


@dataclass(frozen=True)
class _SegmentMatchItem:
    """One product inside a segment match option."""

    product: _WeightOnlyCandidate
    count: int
    evidence_score: float = 0.0
    evidence_source: Optional[str] = None
    evidence_trusted: bool = False
    evidence_strong: bool = False
    evidence_motion_gate_passed: bool = True
    stage_score: float = 0.0
    evidence_confidence: float = 0.0
    evidence_votes: int = 0
    top_confidence: float = 0.0
    side_confidence: float = 0.0
    top_votes: int = 0
    side_votes: int = 0
    score_reason: str = ""
    weak_companion: bool = False
    weight_tight: bool = False


@dataclass(frozen=True)
class _SegmentMatchOption:
    """One product/count explanation for a loadcell segment."""

    target: _SegmentWeightTarget
    product: _WeightOnlyCandidate
    count: int
    expected_weight: float
    residual: float
    allowed_residual: float
    weight_score: float
    evidence_score: float
    evidence_source: Optional[str] = None
    evidence_trusted: bool = False
    evidence_strong: bool = False
    evidence_motion_gate_passed: bool = True
    stage_score: float = 0.0
    evidence_confidence: float = 0.0
    evidence_votes: int = 0
    top_confidence: float = 0.0
    side_confidence: float = 0.0
    top_votes: int = 0
    side_votes: int = 0
    score_reason: str = ""
    items: tuple[_SegmentMatchItem, ...] = ()
    option_kind: str = "single"
    selection_tier: int = 50
    selection_rank: int = 9999
    selection_reason: str = "single_segment_match"
    rejected_reason: Optional[str] = None


def _get_count_calculator():
    """Lazy import the count calculator to avoid circular imports."""
    from model_service.weight.count_calculator import WeightBasedCountCalculator
    return WeightBasedCountCalculator


class ProductDecisionEngine:
    """
    최종 상품 결정 엔진.

    Vision 후보군과 무게 검증을 결합하여 최종 상품을 결정합니다.
    """

    # 신뢰도 가중치
    VISION_WEIGHT = 0.4      # Vision 신뢰도 가중치
    WEIGHT_MATCH_WEIGHT = 0.5  # 무게 매칭 가중치
    COUNT_WEIGHT = 0.1       # 개수 합리성 가중치
    STAGE_COUNT_COMBINATION_LIMIT = 10
    _FREEZER_AMBIGUOUS_PRODUCT_CLASSES = frozenset({30, 42, 44})

    def __init__(
        self,
        product_db: Optional[ActiveProductStore] = None,
        tolerance_percent: Optional[float] = None,
        confidence_threshold: float = 0.3,
        max_combination_size: Optional[int] = None,
        min_weight_change: Optional[float] = None,
        partial_threshold: float = 0.7,
        strict_mode: Optional[bool] = None,
    ):
        """
        판단 엔진 초기화.

        Args:
            product_db: 상품 데이터베이스
            tolerance_percent: 허용 오차 비율 (기본값 10%)
            confidence_threshold: 최소 신뢰도 임계값 (기본값 0.3)
            max_combination_size: 최대 조합 크기 (기본값 2)
            min_weight_change: 최소 무게 변화량 (기본값 5g)
            partial_threshold: PARTIAL/UNCERTAIN 구분 임계값 (기본값 0.7)
            strict_mode: 엄격 무게 검증 모드 (v5.1, 기본값 True)
        """
        self.product_db = product_db
        self.tolerance_percent = tolerance_percent or config.weight.tolerance_percent
        self.confidence_threshold = confidence_threshold
        self.max_combination_size = max_combination_size or config.weight.max_combination_size
        self.min_weight_change = min_weight_change or config.weight.min_weight_change
        self.partial_threshold = partial_threshold
        self.strict_mode = strict_mode if strict_mode is not None else config.weight.strict_mode

        WeightBasedCountCalculator = _get_count_calculator()
        self.count_calculator = WeightBasedCountCalculator(
            tolerance_percent=self.tolerance_percent,
            tolerance_grams=config.weight.tolerance_grams,
            max_count=config.weight.same_product_max_count,
        )

    def judge(
        self,
        vision_candidates: List[EnsembleResult],
        delta_weight: float,
        vision_only: bool = False,
        active_products: Optional[List] = None,
        trace_context: Optional[object] = None,
    ) -> JudgmentResult:
        """
        상품 판단 수행.

        v5.1: strict_mode 추가 - 무게로 설명 불가 시 NO_DETECTION 반환
        v4.7: active_products 파라미터 추가.
        ActiveProductStore의 상품 정보를 count_calculator에 전달하여
        stock 필터링 및 정적 상품 fallback 문제 해결.

        Args:
            vision_candidates: Multi-View Ensemble 결과 (Top-K)
            delta_weight: 무게 변화량 (음수 = 제거)
            vision_only: Vision 전용 모드 (로드셀 없이 카메라만 사용)
            active_products: ActiveProductStore의 상품 정보 (v4.7)

        Returns:
            JudgmentResult (Node.js 전달용)
        """
        # The active-product snapshot is the authoritative inventory context for
        # strict and loadcell-only inference. Logging it here makes it obvious
        # when the trigger path forgot to forward that snapshot.
        timestamp = time.time()
        abs_weight = abs(delta_weight)
        active_product_count, zero_stock_filtered = self._summarize_active_products(
            active_products
        )
        delta_reason = self._get_delta_reason(delta_weight)

        logger.info("[ENGINE] ========== 판단 엔진 ==========")
        logger.info(
            f"[ENGINE] 후보: {len(vision_candidates)}개, "
            f"delta_weight={delta_weight:.1f}g, vision_only={vision_only}, "
            f"strict_mode={self.strict_mode}"
        )
        logger.info(
            f"[ENGINE][reason={delta_reason}] active_products={active_product_count}, "
            f"zero_stock_filtered={zero_stock_filtered}"
        )
        if active_product_count == 0:
            logger.warning(
                "[ENGINE][reason=no_active_products] active_products snapshot is empty"
            )
        if active_products:
            logger.info(f"[ENGINE] v4.7: active_products {len(active_products)}개 수신")

        # Vision 전용 모드: 카메라만으로 판단
        if vision_only:
            result = self._judge_vision_only(vision_candidates, timestamp)
            self._log_final_branch("vision_only", result)
            return result

        freezer_result = self._try_freezer_vision_first(
            vision_candidates=vision_candidates,
            delta_weight=delta_weight,
            timestamp=timestamp,
            active_products=active_products,
            trace_context=trace_context,
        )
        if freezer_result is not None:
            self._log_final_branch("freezer_vision_first", freezer_result)
            return freezer_result

        vision_candidates = self._augment_stage_weight_gate_candidates(
            vision_candidates=vision_candidates,
            delta_weight=delta_weight,
            active_products=active_products,
            trace_context=trace_context,
        )

        segment_result = self._try_segment_weight_matching(
            vision_candidates=vision_candidates,
            delta_weight=delta_weight,
            timestamp=timestamp,
            active_products=active_products,
            trace_context=trace_context,
        )
        if segment_result is not None:
            segment_result = self._enforce_full_delta_match(
                segment_result,
                trace_context=trace_context,
                branch="segment_weight_matching",
            )
            self._log_final_branch("segment_weight_matching", segment_result)
            return segment_result

        # 1. 후보군이 없는 경우 → Loadcell-only 폴백
        if not vision_candidates:
            logger.warning(
                f"[ENGINE][reason=no_vision_candidates] "
                f"delta_weight={delta_weight:.1f}g, trying loadcell-only fallback"
            )
            stage_combo = self._try_stage_count_combination_match(
                vision_candidates=vision_candidates,
                delta_weight=delta_weight,
                timestamp=timestamp,
                active_products=active_products,
                trace_context=trace_context,
            )
            if stage_combo is not None:
                stage_combo = self._enforce_full_delta_match(
                    stage_combo,
                    trace_context=trace_context,
                    branch="stage_count_combination_match",
                )
                self._log_final_branch("stage_count_combination_match", stage_combo)
                return stage_combo
            detected_single = self._try_detected_single_item_fallback(
                vision_candidates=vision_candidates,
                delta_weight=delta_weight,
                timestamp=timestamp,
                active_products=active_products,
                trace_context=trace_context,
            )
            if detected_single is not None:
                detected_single = self._enforce_full_delta_match(
                    detected_single,
                    trace_context=trace_context,
                    branch="detected_single_item_fallback",
                )
                self._log_final_branch(
                    "detected_single_item_fallback",
                    detected_single,
                )
                return detected_single
            if self._is_vision_first_identity_policy():
                result = self._create_loadcell_identity_suppressed_result(
                    delta_weight=delta_weight,
                    timestamp=timestamp,
                    trace_context=trace_context,
                    branch="loadcell_only_no_vision",
                )
                self._log_final_branch("loadcell_identity_suppressed", result)
                return result
            result = self.judge_by_weight_only(
                delta_weight, timestamp, active_products=active_products
            )
            if not result.is_success:
                forced_final = self._try_forced_final_fallback(
                    vision_candidates=vision_candidates,
                    delta_weight=delta_weight,
                    timestamp=timestamp,
                    active_products=active_products,
                    trace_context=trace_context,
                    previous_status=result.status.value,
                )
                if forced_final is not None:
                    forced_final = self._enforce_full_delta_match(
                        forced_final,
                        trace_context=trace_context,
                        branch="forced_final_fallback",
                    )
                    self._log_final_branch("forced_final_fallback", forced_final)
                    return forced_final
            result = self._enforce_full_delta_match(
                result,
                trace_context=trace_context,
                branch="loadcell_only_no_vision",
            )
            self._log_final_branch("loadcell_only_no_vision", result)
            return result

        # 2. 무게 변화가 없는 경우
        if abs_weight < self.min_weight_change:
            logger.info(
                f"[ENGINE][reason=min_weight_change] "
                f"{abs_weight:.1f}g < {self.min_weight_change}g"
            )
            result = self._create_no_detection_result(delta_weight, timestamp)
            self._log_final_branch("no_detection_min_weight", result)
            return result

        same_weight_candidate = self._try_same_weight_candidate_collision_guard(
            vision_candidates=vision_candidates,
            delta_weight=delta_weight,
            timestamp=timestamp,
            active_products=active_products,
            trace_context=trace_context,
        )
        if same_weight_candidate is not None:
            same_weight_candidate = self._enforce_full_delta_match(
                same_weight_candidate,
                trace_context=trace_context,
                branch="same_weight_candidate_collision",
            )
            self._log_final_branch(
                "same_weight_candidate_collision",
                same_weight_candidate,
            )
            return same_weight_candidate

        if self.strict_mode:
            logger.info("[ENGINE][reason=strict_mode] v5.1 strict matching enabled")
            strict_result = self._judge_strict(
                vision_candidates,
                delta_weight,
                timestamp,
                active_products,
                trace_context=trace_context,
            )
            if strict_result is not None:
                branch = (
                    "strict_match"
                    if strict_result.status == JudgmentStatus.COMPLETE
                    else "strict_no_detection"
                )
                strict_result = self._enforce_full_delta_match(
                    strict_result,
                    trace_context=trace_context,
                    branch=branch,
                )
                self._log_final_branch(branch, strict_result)
                return strict_result

        same_product_count_result = self._try_same_product_count_match(
            vision_candidates=vision_candidates,
            delta_weight=delta_weight,
            timestamp=timestamp,
            active_products=active_products,
            trace_context=trace_context,
        )
        if same_product_count_result is not None:
            same_product_count_result = self._enforce_full_delta_match(
                same_product_count_result,
                trace_context=trace_context,
                branch="same_product_count_match",
            )
            self._log_final_branch(
                "same_product_count_match",
                same_product_count_result,
            )
            return same_product_count_result

        result, branch = self._judge_relaxed(
            vision_candidates=vision_candidates,
            delta_weight=delta_weight,
            timestamp=timestamp,
            active_products=active_products,
            trace_context=trace_context,
        )
        if self._is_vision_first_identity_policy() and result.status != JudgmentStatus.COMPLETE:
            vision_first_result = self._try_vision_first_identity_partial(
                vision_candidates=vision_candidates,
                delta_weight=delta_weight,
                timestamp=timestamp,
                active_products=active_products,
                trace_context=trace_context,
            )
            if vision_first_result is not None:
                self._log_final_branch(
                    "vision_first_identity_partial",
                    vision_first_result,
                )
                return vision_first_result
        if not result.is_success:
            detected_single = self._try_detected_single_item_fallback(
                vision_candidates=vision_candidates,
                delta_weight=delta_weight,
                timestamp=timestamp,
                active_products=active_products,
                trace_context=trace_context,
            )
            if detected_single is not None:
                detected_single = self._enforce_full_delta_match(
                    detected_single,
                    trace_context=trace_context,
                    branch="detected_single_item_fallback",
                )
                self._log_final_branch(
                    "detected_single_item_fallback",
                    detected_single,
                )
                return detected_single
            if self._is_vision_first_identity_policy():
                result = self._suppress_non_success_identity_result(
                    result,
                    trace_context=trace_context,
                    branch=branch,
                )
                result = self._enforce_full_delta_match(
                    result,
                    trace_context=trace_context,
                    branch=branch,
                )
                self._log_final_branch(branch, result)
                return result
            forced_final = self._try_forced_final_fallback(
                vision_candidates=vision_candidates,
                delta_weight=delta_weight,
                timestamp=timestamp,
                active_products=active_products,
                trace_context=trace_context,
                previous_status=result.status.value,
            )
            if forced_final is not None:
                forced_final = self._enforce_full_delta_match(
                    forced_final,
                    trace_context=trace_context,
                    branch="forced_final_fallback",
                )
                self._log_final_branch("forced_final_fallback", forced_final)
                return forced_final
        result = self._enforce_full_delta_match(
            result,
            trace_context=trace_context,
            branch=branch,
        )
        self._log_final_branch(branch, result)
        return result

    def _try_freezer_vision_first(
        self,
        *,
        vision_candidates: List[EnsembleResult],
        delta_weight: float,
        timestamp: float,
        active_products: Optional[List],
        trace_context: Optional[object],
    ) -> Optional[JudgmentResult]:
        """Use freezer-filtered vision candidates as identity and weight as a tie-breaker."""
        if not self._is_freezer_mode() or delta_weight >= 0:
            return None

        target_weight = abs(float(delta_weight))
        min_exit_path_votes = max(
            0,
            int(getattr(config.vision, "freezer_min_exit_path_votes", 3)),
        )
        strict_residual_limit = max(
            0.0,
            float(config.weight.detected_single_fallback_tolerance_grams),
        )
        near_residual_limit = strict_residual_limit + max(
            0.0,
            float(config.weight.same_product_count_tolerance_grams),
        )
        vision_candidates = self._augment_freezer_stage_exit_path_candidates(
            vision_candidates=vision_candidates,
            active_products=active_products,
            trace_context=trace_context,
            target_weight=target_weight,
            min_exit_path_votes=min_exit_path_votes,
            near_residual_limit=near_residual_limit,
        )
        if not vision_candidates:
            self._record_weight_diagnostics(
                trace_context,
                {
                    "decision_branch": "freezer_vision_first",
                    "freezer_vision_first": {
                        "accepted": False,
                        "reason": "no_vision_candidates",
                        "target_weight": round(target_weight, 1),
                    },
                },
            )
            return self._create_no_detection_result(delta_weight, timestamp)

        active_map = self._active_products_by_class(active_products)
        considered: list[dict[str, Any]] = []
        options: list[dict[str, Any]] = []
        top_k = max(1, int(config.vision.top_k))
        candidate_pool = [
            *vision_candidates[:top_k],
            *[
                candidate
                for candidate in vision_candidates[top_k:]
                if str(getattr(candidate, "source", "") or "")
                == "freezer_stage_exit_path"
            ],
        ]

        for rank, candidate in enumerate(candidate_pool, start=1):
            class_id = int(candidate.class_id)
            confidence = float(candidate.combined_confidence)
            stage_entry = self._freezer_stage_entry(trace_context, class_id)
            camera_exit_counts = self._freezer_stage_camera_exit_counts(stage_entry)
            roi_x_avg, roi_y_avg = self._freezer_stage_center(stage_entry)
            source = str(getattr(candidate, "source", "vision") or "vision")
            product = active_map.get(class_id)
            diag: dict[str, Any] = {
                "rank": rank,
                "class_id": class_id,
                "name": candidate.class_name,
                "source": source,
                "confidence": round(confidence, 4),
                "raw_vote_count": int(getattr(candidate, "raw_vote_count", 0) or 0),
                "freezerExitPathVotes": self._freezer_exit_path_votes(
                    candidate,
                    trace_context,
                ),
                "cameraExitCounts": dict(camera_exit_counts),
                "dualCameraExitPath": len(camera_exit_counts) >= 2,
                "stageOnly": source == "freezer_stage_exit_path",
                "roiXAvg": round(float(roi_x_avg), 1) if roi_x_avg is not None else None,
                "roiYAvg": round(float(roi_y_avg), 1) if roi_y_avg is not None else None,
                "instance_count_hint": max(
                    1,
                    int(getattr(candidate, "instance_count_hint", 1) or 1),
                ),
                "motion_gate_passed": bool(
                    getattr(candidate, "motion_gate_passed", True)
                ),
            }
            if product is None:
                diag["reason"] = "not_in_active_products"
                considered.append(diag)
                continue
            if not self._candidate_has_vision_identity_evidence(candidate):
                unit_weight = self._coerce_float(
                    getattr(product, "product_weight", 0.0)
                )
                residual = (
                    abs(target_weight - unit_weight)
                    if unit_weight > 0
                    else target_weight
                )
                diag.update(
                    {
                        "product_name": getattr(
                            product,
                            "product_name",
                            candidate.class_name,
                        ),
                        "unit_weight": round(unit_weight, 1),
                        "count": 1,
                        "expected_weight": round(max(0.0, unit_weight), 1),
                        "weight_residual": round(residual, 1),
                        "weight_used_as": "diagnostic",
                    }
                )
                diag["reason"] = "insufficient_vision_identity_evidence"
                considered.append(diag)
                continue
            stock = self._coerce_int(getattr(product, "stock_qty", 0))
            if stock <= 0:
                diag["reason"] = "invalid_stock"
                considered.append(diag)
                continue

            unit_weight = self._coerce_float(getattr(product, "product_weight", 0.0))
            instance_count_hint = max(1, int(diag["instance_count_hint"]))
            count = self._freezer_supported_count(
                unit_weight=unit_weight,
                target_weight=target_weight,
                instance_count_hint=instance_count_hint,
                stock=stock,
            )
            expected_weight = max(0.0, unit_weight) * count
            residual = (
                abs(target_weight - expected_weight)
                if expected_weight > 0
                else target_weight
            )
            diag.update(
                {
                    "product_name": getattr(product, "product_name", candidate.class_name),
                    "product_eng_name": getattr(product, "product_eng_name", ""),
                    "stock": stock,
                    "unit_weight": round(unit_weight, 1),
                    "raw_instance_count_hint": instance_count_hint,
                    "instance_count_hint": count,
                    "count": count,
                    "expected_weight": round(expected_weight, 1),
                    "weight_residual": round(residual, 1),
                    "weight_used_as": "tiebreaker",
                    "weight_reliable": residual <= float(config.weight.tolerance_grams),
                    "reason": "candidate",
                }
            )
            option = {
                "rank": rank,
                "candidate": candidate,
                "product": product,
                "confidence": confidence,
                "count": count,
                "unit_weight": unit_weight,
                "expected_weight": expected_weight,
                "residual": residual,
                "freezer_exit_path_votes": int(diag["freezerExitPathVotes"]),
                "camera_exit_counts": camera_exit_counts,
                "dual_camera_exit_path": bool(diag["dualCameraExitPath"]),
                "stage_only": bool(diag["stageOnly"]),
                "source": source,
                "source_priority": 1 if source == "freezer_stage_exit_path" else 0,
                "diagnostics": diag,
            }
            options.append(option)
            considered.append(diag)

        if not options:
            self._record_weight_diagnostics(
                trace_context,
                {
                    "decision_branch": "freezer_vision_first",
                    "freezer_vision_first": {
                        "accepted": False,
                        "reason": "no_supported_vision_candidates",
                        "target_weight": round(target_weight, 1),
                        "considered": considered,
                    },
                },
            )
            return self._create_no_detection_result(delta_weight, timestamp)

        multi_kind_selection = self._select_freezer_multi_kind_options(
            options,
            target_weight=target_weight,
            trace_context=trace_context,
        )
        if multi_kind_selection is not None:
            selected_options, reason = multi_kind_selection
            return self._create_freezer_vision_result(
                selected_options=selected_options,
                delta_weight=delta_weight,
                target_weight=target_weight,
                timestamp=timestamp,
                trace_context=trace_context,
                considered=considered,
                reason=reason,
            )

        selected, reason = self._select_freezer_single_handled_option(options)
        return self._create_freezer_vision_result(
            selected_options=[selected],
            delta_weight=delta_weight,
            target_weight=target_weight,
            timestamp=timestamp,
            trace_context=trace_context,
            considered=considered,
            reason=reason,
        )

    @staticmethod
    def _freezer_stage_entry(
        trace_context: Optional[object],
        class_id: int,
    ) -> dict[str, Any]:
        stage_counts = getattr(trace_context, "stage_counts_by_class", {}) or {}
        if not isinstance(stage_counts, dict):
            return {}
        entry = stage_counts.get(str(class_id)) or stage_counts.get(class_id)
        return entry if isinstance(entry, dict) else {}

    @staticmethod
    def _freezer_stage_int(entry: dict[str, Any], *keys: str) -> int:
        values: list[int] = []
        for key in keys:
            try:
                values.append(int(entry.get(key, 0) or 0))
            except (TypeError, ValueError):
                pass
        return max(values or [0])

    @staticmethod
    def _freezer_stage_float(entry: dict[str, Any], *keys: str) -> float:
        values: list[float] = []
        for key in keys:
            try:
                values.append(float(entry.get(key, 0.0) or 0.0))
            except (TypeError, ValueError):
                pass
        return max(values or [0.0])

    @classmethod
    def _freezer_stage_camera_exit_counts(
        cls,
        entry: dict[str, Any],
    ) -> dict[str, int]:
        cameras = entry.get("cameras") or {}
        if not isinstance(cameras, dict):
            return {}
        counts: dict[str, int] = {}
        for camera, camera_entry in cameras.items():
            if not isinstance(camera_entry, dict):
                continue
            count = cls._freezer_stage_int(
                camera_entry,
                "freezerExitPathVotes",
                "freezer_exit_path_votes",
                "freezer_roi_filtered",
            )
            if count > 0:
                counts[str(camera)] = count
        return counts

    @staticmethod
    def _freezer_stage_center(
        entry: dict[str, Any],
    ) -> tuple[Optional[float], Optional[float]]:
        try:
            x_avg = (
                float(entry["roi_x_avg"])
                if entry.get("roi_x_avg") is not None
                else None
            )
        except (TypeError, ValueError):
            x_avg = None
        try:
            y_avg = (
                float(entry["roi_y_avg"])
                if entry.get("roi_y_avg") is not None
                else None
            )
        except (TypeError, ValueError):
            y_avg = None
        return x_avg, y_avg

    def _augment_freezer_stage_exit_path_candidates(
        self,
        *,
        vision_candidates: List[EnsembleResult],
        active_products: Optional[List],
        trace_context: Optional[object],
        target_weight: float,
        min_exit_path_votes: int,
        near_residual_limit: float,
    ) -> List[EnsembleResult]:
        stage_counts = getattr(trace_context, "stage_counts_by_class", {}) or {}
        if not isinstance(stage_counts, dict) or not active_products:
            return vision_candidates

        active_map = self._active_products_by_class(active_products)
        if not active_map:
            return vision_candidates

        existing_class_ids = {
            int(candidate.class_id)
            for candidate in vision_candidates
            if getattr(candidate, "class_id", None) is not None
        }
        augmented = list(vision_candidates)
        min_stage_votes = max(1, int(config.weight.detected_single_fallback_min_votes))
        min_confidence = float(config.weight.multi_kind_min_confidence)

        for raw_class_id, entry in stage_counts.items():
            if not isinstance(entry, dict):
                continue
            try:
                class_id = int(entry.get("class_id", raw_class_id))
            except (TypeError, ValueError):
                continue
            if class_id in existing_class_ids:
                continue
            product = active_map.get(class_id)
            if product is None or not self._active_product_has_loadcell(product):
                continue
            stock = self._coerce_int(getattr(product, "stock_qty", 0))
            unit_weight = self._coerce_float(getattr(product, "product_weight", 0.0))
            if stock <= 0 or unit_weight <= 0.0:
                continue
            residual = abs(target_weight - unit_weight)
            if residual > near_residual_limit:
                continue
            exit_path_votes = self._freezer_stage_int(
                entry,
                "freezerExitPathVotes",
                "freezer_exit_path_votes",
                "freezer_roi_filtered",
            )
            threshold_votes = self._freezer_stage_int(
                entry,
                "threshold_passed",
                "motion_passed",
                "raw",
            )
            confidence = self._freezer_stage_float(
                entry,
                "freezer_roi_filtered_max_confidence",
                "threshold_passed_max_confidence",
                "raw_max_confidence",
            )
            if (
                exit_path_votes < min_exit_path_votes
                or threshold_votes < min_stage_votes
                or confidence < min_confidence
            ):
                continue

            cameras = entry.get("cameras") if isinstance(entry.get("cameras"), dict) else {}
            top_entry = cameras.get("top", {}) if isinstance(cameras, dict) else {}
            side_entry = cameras.get("side", {}) if isinstance(cameras, dict) else {}
            if not isinstance(top_entry, dict):
                top_entry = {}
            if not isinstance(side_entry, dict):
                side_entry = {}
            top_votes = self._freezer_stage_int(
                top_entry,
                "freezerExitPathVotes",
                "freezer_roi_filtered",
                "threshold_passed",
                "raw",
            )
            side_votes = self._freezer_stage_int(
                side_entry,
                "freezerExitPathVotes",
                "freezer_roi_filtered",
                "threshold_passed",
                "raw",
            )
            augmented.append(
                EnsembleResult(
                    class_id=class_id,
                    class_name=str(
                        entry.get("name") or getattr(product, "product_name", "")
                    ),
                    top_confidence=self._freezer_stage_float(
                        top_entry,
                        "freezer_roi_filtered_max_confidence",
                        "threshold_passed_max_confidence",
                        "raw_max_confidence",
                    ),
                    side_confidence=self._freezer_stage_float(
                        side_entry,
                        "freezer_roi_filtered_max_confidence",
                        "threshold_passed_max_confidence",
                        "raw_max_confidence",
                    ),
                    combined_confidence=confidence,
                    vote_count=2 if top_votes > 0 and side_votes > 0 else 1,
                    source="freezer_stage_exit_path",
                    raw_vote_count=self._freezer_stage_int(
                        entry,
                        "raw",
                        "threshold_passed",
                    ),
                    top_motion_passed=bool(
                        top_entry.get("motion_passed")
                        or top_entry.get("motion_filtered")
                    ),
                    side_motion_passed=bool(
                        side_entry.get("motion_passed")
                        or side_entry.get("motion_filtered")
                    ),
                    motion_gate_passed=True,
                    weight_gate_passed=residual
                    <= float(config.weight.detected_single_fallback_tolerance_grams),
                    rescue_tolerance_g=near_residual_limit,
                    rescue_weight_residual_g=residual,
                    freezer_exit_path_votes=exit_path_votes,
                )
            )
        return augmented

    def _freezer_exit_path_votes(
        self,
        candidate: EnsembleResult,
        trace_context: Optional[object],
    ) -> int:
        values: list[int] = []
        for value in (
            getattr(candidate, "freezer_exit_path_votes", 0),
            getattr(candidate, "freezerExitPathVotes", 0),
        ):
            try:
                values.append(int(value or 0))
            except (TypeError, ValueError):
                pass

        entry = self._freezer_stage_entry(trace_context, int(candidate.class_id))
        for key in ("freezerExitPathVotes", "freezer_exit_path_votes", "freezer_roi_filtered"):
            try:
                values.append(int(entry.get(key, 0) or 0))
            except (TypeError, ValueError):
                pass
        return max(values or [0])

    @classmethod
    def _freezer_handled_tier(
        cls,
        option: dict[str, Any],
        *,
        min_exit_path_votes: int,
        strict_residual_limit: float,
        near_residual_limit: float,
    ) -> tuple[int, str]:
        residual = float(option["residual"])
        exit_path_votes = int(option.get("freezer_exit_path_votes", 0) or 0)
        if (
            bool(option.get("stage_only"))
            and int(getattr(option.get("candidate"), "class_id", -1))
            in cls._FREEZER_AMBIGUOUS_PRODUCT_CLASSES
            and bool(option.get("dual_camera_exit_path"))
            and residual <= near_residual_limit
            and exit_path_votes >= min_exit_path_votes
        ):
            return -1, "freezer_single_ambiguous_dual_camera_stage_exit_path"
        if residual <= strict_residual_limit and exit_path_votes >= min_exit_path_votes:
            return 0, "freezer_single_weight_gate_exit_path"
        if residual <= near_residual_limit and exit_path_votes >= min_exit_path_votes:
            return 1, "freezer_single_near_weight_exit_path"
        return 2, "freezer_single_confidence_weight_tiebreak"

    def _select_freezer_single_handled_option(
        self,
        options: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], str]:
        min_exit_path_votes = max(
            0,
            int(getattr(config.vision, "freezer_min_exit_path_votes", 3)),
        )
        strict_residual_limit = max(
            0.0,
            float(config.weight.detected_single_fallback_tolerance_grams),
        )
        near_residual_limit = strict_residual_limit + max(
            0.0,
            float(config.weight.same_product_count_tolerance_grams),
        )
        scored: list[tuple[int, str, dict[str, Any]]] = []
        for option in options:
            tier, reason = self._freezer_handled_tier(
                option,
                min_exit_path_votes=min_exit_path_votes,
                strict_residual_limit=strict_residual_limit,
                near_residual_limit=near_residual_limit,
            )
            option["selection_tier"] = reason
            option["freezer_min_exit_path_votes"] = min_exit_path_votes
            option["freezer_strict_residual_limit"] = strict_residual_limit
            option["freezer_near_residual_limit"] = near_residual_limit
            option["diagnostics"]["selectionTier"] = reason
            option["diagnostics"]["freezerMinExitPathVotes"] = min_exit_path_votes
            option["diagnostics"]["strictResidualLimit"] = round(strict_residual_limit, 1)
            option["diagnostics"]["nearResidualLimit"] = round(near_residual_limit, 1)
            scored.append((tier, reason, option))

        handled = [item for item in scored if item[0] < 2]
        if handled:
            tier, reason, selected = min(
                handled,
                key=lambda item: (
                    item[0],
                    int(item[2].get("source_priority", 0) or 0),
                    -int(item[2].get("freezer_exit_path_votes", 0) or 0),
                    float(item[2]["residual"]),
                    -float(item[2]["confidence"]),
                    int(item[2]["rank"]),
                ),
            )
            return selected, reason

        best_confidence = max(option["confidence"] for option in options)
        confidence_band = max(0.0, float(config.weight.freezer_confidence_tie_band))
        tie_pool = [
            option
            for option in options
            if option["confidence"] >= best_confidence - confidence_band
        ]
        selected = min(
            tie_pool,
            key=lambda option: (
                float(option["residual"]),
                int(option["rank"]),
                -float(option["confidence"]),
            ),
        )
        return selected, "freezer_single_confidence_weight_tiebreak"

    def _freezer_supported_count(
        self,
        *,
        unit_weight: float,
        target_weight: float,
        instance_count_hint: int,
        stock: int,
    ) -> int:
        if unit_weight <= 0.0:
            return 1
        max_hint = max(
            1,
            min(
                int(instance_count_hint),
                int(stock),
                int(config.weight.max_count_per_item),
            ),
        )
        if max_hint <= 1:
            return 1

        best_count = 1
        best_residual = abs(target_weight - unit_weight)
        for count in range(2, max_hint + 1):
            residual = abs(target_weight - unit_weight * count)
            if residual <= self._count_scaled_weight_tolerance(count) and residual < best_residual:
                best_count = count
                best_residual = residual
        return best_count

    def _select_freezer_multi_kind_options(
        self,
        options: list[dict[str, Any]],
        *,
        target_weight: float,
        trace_context: Optional[object],
    ) -> Optional[tuple[list[dict[str, Any]], str]]:
        multi_min_confidence = float(config.weight.freezer_multi_min_confidence)
        high_confidence = [
            option for option in options if option["confidence"] >= multi_min_confidence
        ]
        high_confidence.sort(key=lambda item: (item["rank"], -item["confidence"]))
        if len(high_confidence) < 2:
            return None

        max_kinds = max(2, int(config.weight.max_combination_kinds))
        if self._freezer_trace_has_multi_item_evidence(trace_context):
            return (
                high_confidence[: min(max_kinds, len(high_confidence))],
                "freezer_multi_kind_segment_evidence",
            )

        viable: list[tuple[int, float, float, int, list[dict[str, Any]]]] = []
        for size in range(2, min(max_kinds, len(high_confidence)) + 1):
            for combo in combinations(high_confidence, size):
                selected = list(combo)
                total_count = sum(int(option["count"]) for option in selected)
                expected_weight = sum(float(option["expected_weight"]) for option in selected)
                residual = abs(target_weight - expected_weight)
                allowed = self._count_scaled_weight_tolerance(total_count)
                if residual > allowed:
                    continue
                avg_confidence = sum(
                    float(option["confidence"]) for option in selected
                ) / len(selected)
                rank_sum = sum(int(option["rank"]) for option in selected)
                viable.append(
                    (
                        total_count,
                        residual,
                        -avg_confidence,
                        rank_sum,
                        selected,
                    )
                )
        if not viable:
            vision_supported = self._select_freezer_vision_supported_multi_options(
                high_confidence,
                max_kinds=max_kinds,
            )
            if vision_supported is not None:
                return vision_supported, "freezer_multi_kind_vision_supported"
            return None
        viable.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        return viable[0][4], "freezer_multi_kind_weight_supported"

    @staticmethod
    def _select_freezer_vision_supported_multi_options(
        options: list[dict[str, Any]],
        *,
        max_kinds: int,
    ) -> Optional[list[dict[str, Any]]]:
        if not bool(config.weight.freezer_vision_multi_without_weight_enabled):
            return None

        min_exit_path_votes = max(
            0,
            int(getattr(config.vision, "freezer_min_exit_path_votes", 3)),
        )
        strong_options = [
            option
            for option in options
            if int(option.get("freezer_exit_path_votes", 0) or 0)
            >= min_exit_path_votes
            and bool(option.get("dual_camera_exit_path"))
            and bool(option.get("diagnostics", {}).get("motion_gate_passed", True))
        ]
        if len(strong_options) < 2:
            return None

        strong_options.sort(
            key=lambda item: (int(item["rank"]), -float(item["confidence"]))
        )
        return strong_options[: min(max_kinds, len(strong_options))]

    @staticmethod
    def _freezer_trace_has_multi_item_evidence(trace_context: Optional[object]) -> bool:
        loadcell = getattr(trace_context, "loadcell", None)
        if not isinstance(loadcell, dict):
            return False

        removal_targets = loadcell.get("removal_segment_targets") or []
        if isinstance(removal_targets, list) and len(removal_targets) >= 2:
            return True
        if bool(loadcell.get("compound_event")):
            return True
        try:
            return int(loadcell.get("compound_negative_segment_count", 0) or 0) >= 2
        except (TypeError, ValueError):
            return False

    def _create_freezer_vision_result(
        self,
        *,
        selected_options: list[dict[str, Any]],
        delta_weight: float,
        target_weight: float,
        timestamp: float,
        trace_context: Optional[object],
        considered: list[dict[str, Any]],
        reason: str,
    ) -> JudgmentResult:
        products: list[ProductJudgment] = []
        total_price = 0
        explained_weight = 0.0
        confidences: list[float] = []
        for option in selected_options:
            product = option["product"]
            candidate = option["candidate"]
            count = max(1, int(option["count"]))
            unit_price = self._coerce_int(getattr(product, "sale_price", 0))
            unit_weight = self._coerce_float(getattr(product, "product_weight", 0.0))
            confidence = float(option["confidence"])
            product_judgment = ProductJudgment(
                product_id=int(getattr(product, "yolo_class_id", candidate.class_id)),
                name=getattr(product, "product_name", candidate.class_name),
                count=count,
                unit_price=unit_price,
                total_price=unit_price * count,
                confidence=confidence,
                unit_weight=unit_weight,
            )
            products.append(product_judgment)
            total_price += product_judgment.total_price
            explained_weight += max(0.0, unit_weight) * count
            confidences.append(confidence)

        residual = abs(target_weight - explained_weight)
        tolerance = float(config.weight.tolerance_grams)
        weight_reliable = residual <= tolerance
        status = JudgmentStatus.COMPLETE if weight_reliable else JudgmentStatus.PARTIAL
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        diagnostics = {
            "accepted": True,
            "reason": reason,
            "target_weight": round(target_weight, 1),
            "explained_weight": round(explained_weight, 1),
            "weight_residual": round(residual, 1),
            "tolerance": round(tolerance, 1),
            "weight_used_as": (
                "diagnostic"
                if reason == "freezer_multi_kind_vision_supported"
                else "tiebreaker"
            ),
            "weight_reliable": weight_reliable,
            "selected": [option["diagnostics"] for option in selected_options],
            "considered": considered,
        }
        ambiguous_candidates = [
            item
            for item in considered
            if int(item.get("class_id", -1)) in self._FREEZER_AMBIGUOUS_PRODUCT_CLASSES
            and float(item.get("weight_residual", target_weight) or target_weight)
            <= max(
                0.0,
                float(config.weight.detected_single_fallback_tolerance_grams),
            )
            + max(0.0, float(config.weight.same_product_count_tolerance_grams))
            and int(item.get("freezerExitPathVotes", 0) or 0)
            >= max(0, int(getattr(config.vision, "freezer_min_exit_path_votes", 3)))
        ]
        if ambiguous_candidates:
            diagnostics["ambiguousCandidates"] = ambiguous_candidates
            diagnostics["hardNegativeCandidates"] = ambiguous_candidates
        self._record_weight_diagnostics(
            trace_context,
            {
                "decision_branch": "freezer_vision_first",
                "freezer_vision_first": diagnostics,
            },
        )
        logger.info(
            "[ENGINE][reason=freezer_vision_first] "
            f"items={len(products)}, target={target_weight:.1f}g, "
            f"explained={explained_weight:.1f}g, residual={residual:.1f}g, "
            f"status={status.value}"
        )
        return JudgmentResult(
            products=products,
            total_price=total_price,
            confidence=confidence,
            status=status,
            weight_delta=delta_weight,
            weight_explained=explained_weight,
            weight_residual=round(residual, 1),
            timestamp=timestamp,
        )

    def _judge_relaxed(
        self,
        vision_candidates: List[EnsembleResult],
        delta_weight: float,
        timestamp: float,
        active_products: Optional[List] = None,
        trace_context: Optional[object] = None,
    ) -> tuple[JudgmentResult, str]:
        """Run the legacy non-strict flow and report which branch was used."""
        # Prefer fully validated matches first, then progressively relax the
        # matching requirements if the stricter branches fail.
        scoped_active_products = self._scope_active_products_to_vision_candidates(
            active_products,
            vision_candidates,
        )
        estimates = self.count_calculator.calculate(
            vision_candidates, delta_weight, active_products=scoped_active_products
        )
        segment_grip_limit = self._segment_grip_limit_from_trace(trace_context)
        if segment_grip_limit is not None:
            rejected_estimates = [
                estimate
                for estimate in estimates
                if int(estimate.count) > segment_grip_limit
            ]
            if rejected_estimates:
                self._record_weight_diagnostics(
                    trace_context,
                    {
                        "relaxed_count_segment_grip_filter": {
                            "accepted": False,
                            "segment_grip_limit": segment_grip_limit,
                            "rejected": [
                                {
                                    "class_id": int(estimate.product_id),
                                    "name": estimate.product_name,
                                    "count": int(estimate.count),
                                    "reason": "count_exceeds_segment_grip_limit",
                                }
                                for estimate in rejected_estimates
                            ],
                        }
                    },
                )
            estimates = [
                estimate
                for estimate in estimates
                if int(estimate.count) <= segment_grip_limit
            ]

        if not estimates:
            logger.warning(
                "[ENGINE][reason=no_count_estimates] trying nearest single fallback"
            )
            nearest_single = self._try_loadcell_nearest_single(
                delta_weight,
                timestamp,
                scoped_active_products,
                require_unique_margin=False,
            )
            if nearest_single is not None:
                return nearest_single, "loadcell_nearest_single_no_estimates"

            result = self.judge_by_weight_only(
                delta_weight, timestamp, active_products=scoped_active_products
            )
            return result, "vision_scoped_loadcell_only_no_estimates"

        single_count_estimates = [
            estimate for estimate in estimates if int(estimate.count) == 1
        ]
        repeated_count_estimates = [
            estimate for estimate in estimates if int(estimate.count) > 1
        ]

        logger.info("[ENGINE] 전략: single_product_match(count=1) 시도...")
        single_result = self._try_single_product_match(
            single_count_estimates, delta_weight, timestamp
        )
        if single_result and single_result.status == JudgmentStatus.COMPLETE:
            logger.info(f"[ENGINE] 단일 상품 매칭 성공: {single_result.products[0].name}")
            return single_result, "single_product_match"

        logger.info("[ENGINE] 전략: combination_match 시도...")
        combo_result = self._try_combination_match(
            vision_candidates,
            delta_weight,
            timestamp,
            active_products=active_products,
            trace_context=trace_context,
        )
        if combo_result and combo_result.status == JudgmentStatus.COMPLETE:
            names = [p.name for p in combo_result.products]
            logger.info(f"[ENGINE] 조합 매칭 성공: {names}")
            return combo_result, "combination_match"

        logger.info("[ENGINE] 전략: repeated_product_match 시도...")
        repeated_result = self._try_single_product_match(
            repeated_count_estimates, delta_weight, timestamp
        )
        if repeated_result and repeated_result.status == JudgmentStatus.COMPLETE:
            logger.info(
                f"[ENGINE] 반복 상품 매칭 성공: {repeated_result.products[0].name}"
            )
            return repeated_result, "repeated_product_match"

        nearest_single = self._try_loadcell_nearest_single(
            delta_weight,
            timestamp,
            scoped_active_products,
            require_unique_margin=False,
        )
        if nearest_single is not None:
            return nearest_single, "loadcell_nearest_single"

        # 6. 불완전 결과 반환 (최선의 추정)
        logger.info("[ENGINE] 전략: partial_result 반환 (매칭 실패)")
        return self._create_partial_result(estimates, delta_weight, timestamp), "partial_result"

    def _try_same_weight_candidate_collision_guard(
        self,
        *,
        vision_candidates: List[EnsembleResult],
        delta_weight: float,
        timestamp: float,
        active_products: Optional[List] = None,
        trace_context: Optional[object] = None,
    ) -> Optional[JudgmentResult]:
        """Prefer regular final candidates over same-weight active/rescue collisions."""
        if delta_weight >= 0 or not vision_candidates or not active_products:
            return None

        active_map = {
            int(product.yolo_class_id): product
            for product in active_products
            if getattr(product, "yolo_class_id", None) is not None
        }
        if not active_map:
            return None

        target_weight = abs(float(delta_weight))
        tolerance = float(config.weight.tolerance_grams)
        tolerance_per_item = float(config.weight.same_product_count_tolerance_grams)
        max_count_per_item = int(config.weight.max_count_per_item)
        max_same_product_count = max(2, int(config.weight.same_product_max_count))
        segment_grip_limit = self._segment_grip_limit_from_trace(trace_context)
        top_k = max(1, int(config.vision.top_k))
        candidate_class_ids = {
            int(candidate.class_id)
            for candidate in vision_candidates[:top_k]
            if getattr(candidate, "class_id", None) is not None
        }

        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for rank, candidate in enumerate(vision_candidates[:top_k], start=1):
            source = str(getattr(candidate, "source", "vision") or "vision")
            product = active_map.get(int(candidate.class_id))
            candidate_diag = {
                "rank": rank,
                "class_id": int(candidate.class_id),
                "name": str(candidate.class_name),
                "source": source,
                "confidence": round(float(candidate.combined_confidence), 4),
            }
            if product is None:
                candidate_diag["reason"] = "not_in_active_products"
                rejected.append(candidate_diag)
                continue
            if source != "vision":
                candidate_diag["reason"] = "non_regular_candidate"
                rejected.append(candidate_diag)
                continue
            if not self._active_product_has_loadcell(product):
                candidate_diag["reason"] = "no_loadcell_product"
                rejected.append(candidate_diag)
                continue

            unit_weight = self._coerce_float(getattr(product, "product_weight", 0.0))
            stock = self._coerce_int(getattr(product, "stock_qty", 0))
            if unit_weight <= 0 or stock <= 0:
                candidate_diag["reason"] = "invalid_weight_or_stock"
                rejected.append(candidate_diag)
                continue

            count = max(1, int(round(target_weight / unit_weight)))
            allowed_count = min(stock, max_count_per_item, max_same_product_count)
            if segment_grip_limit is not None:
                allowed_count = min(allowed_count, segment_grip_limit)
            expected_weight = unit_weight * count
            residual = abs(target_weight - expected_weight)
            base_allowed_residual = (
                tolerance if count == 1 else tolerance + tolerance_per_item * count
            )
            allowed_residual = self._regular_bottle_repeat_allowed_residual(
                candidate=candidate,
                rank=rank,
                count=count,
                unit_weight=unit_weight,
                base_allowed_residual=base_allowed_residual,
            )
            candidate_diag.update(
                {
                    "unit_weight": round(unit_weight, 1),
                    "nearest_count": count,
                    "expected_weight": round(expected_weight, 1),
                    "residual": round(residual, 1),
                    "allowed_residual": round(allowed_residual, 1),
                    "base_allowed_residual": round(base_allowed_residual, 1),
                    "bottle_repeat_grace": allowed_residual > base_allowed_residual,
                    "segment_grip_limit": segment_grip_limit,
                }
            )
            if segment_grip_limit is not None and count > segment_grip_limit:
                candidate_diag["reason"] = "count_exceeds_segment_grip_limit"
                rejected.append(candidate_diag)
                continue
            if count > allowed_count:
                candidate_diag["reason"] = "count_exceeds_stock_or_limit"
                rejected.append(candidate_diag)
                continue
            if residual > allowed_residual:
                candidate_diag["reason"] = "residual_exceeds_candidate_guard_tolerance"
                rejected.append(candidate_diag)
                continue

            same_weight_active = self._same_weight_active_collisions(
                selected_product=product,
                active_products=active_products,
                candidate_class_ids=candidate_class_ids,
                target_weight=target_weight,
                selected_unit_weight=unit_weight,
                count=count,
                allowed_residual=allowed_residual,
                unit_tolerance=tolerance,
            )
            competing_strict = self._best_strict_weight_competing_candidate(
                selected_class_id=int(candidate.class_id),
                vision_candidates=vision_candidates,
                active_map=active_map,
                target_weight=target_weight,
                tolerance=tolerance,
            )
            has_non_regular_strict_competition = (
                competing_strict is not None
                and str(competing_strict.get("source")) != "vision"
            )
            if not same_weight_active and not has_non_regular_strict_competition:
                candidate_diag["reason"] = "no_same_weight_or_rescue_collision"
                rejected.append(candidate_diag)
                continue

            weight_score = max(
                0.5,
                1.0 - (residual / max(allowed_residual, 0.001)) * 0.5,
            )
            confidence = self._calculate_fusion_confidence(
                vision_score=float(candidate.combined_confidence),
                weight_score=weight_score,
                count=count,
            )
            accepted.append(
                {
                    "rank": rank,
                    "product": product,
                    "count": count,
                    "unit_weight": unit_weight,
                    "expected_weight": expected_weight,
                    "residual": residual,
                    "allowed_residual": allowed_residual,
                    "confidence": confidence,
                    "same_weight_active": same_weight_active,
                    "competing_strict": competing_strict,
                }
            )

        if not accepted:
            if rejected:
                self._record_weight_diagnostics(
                    trace_context,
                    {
                        "same_weight_candidate_collision": {
                            "accepted": False,
                            "target_weight": round(target_weight, 1),
                            "candidates": rejected,
                        }
                    },
                )
            return None

        best = sorted(
            accepted,
            key=lambda item: (
                int(item["rank"]),
                float(item["residual"]) / max(float(item["allowed_residual"]), 0.001),
                -float(item["confidence"]),
            ),
        )[0]
        product = best["product"]
        count = int(best["count"])
        unit_price = int(getattr(product, "sale_price", 0) or 0)
        product_judgment = ProductJudgment(
            product_id=int(product.yolo_class_id),
            name=str(product.product_name),
            count=count,
            unit_price=unit_price,
            total_price=unit_price * count,
            confidence=float(best["confidence"]),
            unit_weight=float(best["unit_weight"]),
        )
        diagnostics = {
            "accepted": True,
            "target_weight": round(target_weight, 1),
            "selected": {
                "class_id": product_judgment.product_id,
                "name": product_judgment.name,
                "count": product_judgment.count,
                "unit_weight": round(float(best["unit_weight"]), 1),
                "expected_weight": round(float(best["expected_weight"]), 1),
                "residual": round(float(best["residual"]), 1),
                "allowed_residual": round(float(best["allowed_residual"]), 1),
                "confidence": round(float(best["confidence"]), 4),
                "rank": int(best["rank"]),
                "source": "vision",
            },
            "same_weight_active_collisions": best["same_weight_active"],
            "rejected_best_strict": best["competing_strict"],
            "rejected_candidates": rejected[:10],
            "reason": "regular_candidate_identity_over_weight_collision",
        }
        self._record_weight_diagnostics(
            trace_context,
            {
                "decision_branch": "same_weight_candidate_collision",
                "same_weight_candidate_collision": diagnostics,
            },
        )
        logger.info(
            "[ENGINE][reason=same_weight_candidate_collision] "
            f"selected={product_judgment.name}x{product_judgment.count}, "
            f"expected={best['expected_weight']:.1f}g, actual={target_weight:.1f}g, "
            f"residual={best['residual']:.1f}g, "
            f"allowed={best['allowed_residual']:.1f}g"
        )
        return JudgmentResult(
            products=[product_judgment],
            total_price=product_judgment.total_price,
            confidence=product_judgment.confidence,
            status=JudgmentStatus.COMPLETE,
            weight_delta=delta_weight,
            weight_explained=float(best["expected_weight"]),
            weight_residual=round(float(best["residual"]), 1),
            timestamp=timestamp,
        )

    def _augment_stage_weight_gate_candidates(
        self,
        *,
        vision_candidates: List[EnsembleResult],
        delta_weight: float,
        active_products: Optional[List] = None,
        trace_context: Optional[object] = None,
    ) -> List[EnsembleResult]:
        """Promote tight weight-gated stage evidence into strict matching."""
        if trace_context is None or not active_products or delta_weight >= 0:
            return vision_candidates

        stage_counts = getattr(trace_context, "stage_counts_by_class", {}) or {}
        if not isinstance(stage_counts, dict) or not stage_counts:
            return vision_candidates

        active_map = {
            int(product.yolo_class_id): product
            for product in active_products
            if getattr(product, "yolo_class_id", None) is not None
        }
        if not active_map:
            return vision_candidates

        min_votes = max(0, int(config.weight.detected_single_fallback_min_votes))
        min_confidence = 0.08
        augmented = list(vision_candidates)
        existing_index_by_class: dict[int, int] = {
            int(candidate.class_id): index
            for index, candidate in enumerate(augmented)
            if getattr(candidate, "class_id", None) is not None
        }
        diagnostics: dict[str, Any] = {
            "accepted": False,
            "target_weight": round(abs(float(delta_weight)), 1),
            "min_votes": min_votes,
            "min_confidence": min_confidence,
            "candidates": [],
        }

        for entry in stage_counts.values():
            if not isinstance(entry, dict):
                continue
            try:
                class_id = int(entry.get("class_id"))
            except (TypeError, ValueError):
                continue
            summary = self._stage_evidence_summary(entry)
            product = active_map.get(class_id)
            product_weight = (
                self._coerce_float(getattr(product, "product_weight", 0.0))
                if product is not None
                else 0.0
            )
            stock = (
                self._coerce_int(getattr(product, "stock_qty", 0))
                if product is not None
                else 0
            )
            candidate_diag = {
                "class_id": class_id,
                "name": str(entry.get("name") or ""),
                "votes": summary.votes,
                "confidence": round(float(summary.confidence), 4),
                "top_votes": summary.top_votes,
                "side_votes": summary.side_votes,
                "top_confidence": round(float(summary.top_confidence), 4),
                "side_confidence": round(float(summary.side_confidence), 4),
                "weight_gate_passed": bool(entry.get("weight_gate_passed")),
            }

            existing_candidate = (
                augmented[existing_index_by_class[class_id]]
                if class_id in existing_index_by_class
                else None
            )
            if (
                existing_candidate is not None
                and self._source_includes_regular_vision(
                    getattr(existing_candidate, "source", "vision")
                )
            ):
                candidate_diag["reason"] = "regular_candidate_already_present"
            elif product is None:
                candidate_diag["reason"] = "not_active"
            elif not self._active_product_has_loadcell(product):
                candidate_diag["reason"] = "no_loadcell_product"
            elif product_weight <= 0 or stock <= 0:
                candidate_diag["reason"] = "invalid_weight_or_stock"
            elif not bool(entry.get("weight_gate_passed")):
                candidate_diag["reason"] = "weight_gate_not_passed"
            elif summary.votes < min_votes:
                candidate_diag["reason"] = "insufficient_votes"
            elif summary.confidence < min_confidence:
                candidate_diag["reason"] = "insufficient_confidence"
            else:
                candidate = EnsembleResult(
                    class_id=class_id,
                    class_name=str(entry.get("name") or getattr(product, "product_name", "")),
                    top_confidence=summary.top_confidence,
                    side_confidence=summary.side_confidence,
                    combined_confidence=summary.confidence,
                    vote_count=max(1, summary.votes),
                    source="stage_weight_gate",
                    raw_vote_count=summary.votes,
                    top_motion_passed=summary.top_motion_passed,
                    side_motion_passed=summary.side_motion_passed,
                    motion_gate_passed=summary.motion_gate_passed,
                    weight_gate_passed=True,
                )
                if existing_candidate is not None:
                    augmented[existing_index_by_class[class_id]] = candidate
                    candidate_diag["reason"] = "upgraded_existing_candidate"
                else:
                    augmented.append(candidate)
                    existing_index_by_class[class_id] = len(augmented) - 1
                    candidate_diag["reason"] = "accepted"
                diagnostics["accepted"] = True

            diagnostics["candidates"].append(candidate_diag)

        if diagnostics["candidates"]:
            self._record_weight_diagnostics(
                trace_context,
                {"stage_weight_gate_candidates": diagnostics},
            )
        return augmented

    def _same_weight_active_collisions(
        self,
        *,
        selected_product: object,
        active_products: List,
        candidate_class_ids: set[int],
        target_weight: float,
        selected_unit_weight: float,
        count: int,
        allowed_residual: float,
        unit_tolerance: float,
    ) -> list[dict[str, Any]]:
        selected_class_id = getattr(selected_product, "yolo_class_id", None)
        collisions: list[dict[str, Any]] = []
        for product in active_products:
            class_id = getattr(product, "yolo_class_id", None)
            if class_id is None or class_id == selected_class_id:
                continue
            if int(class_id) in candidate_class_ids:
                continue
            if not self._active_product_has_loadcell(product):
                continue
            unit_weight = self._coerce_float(getattr(product, "product_weight", 0.0))
            stock = self._coerce_int(getattr(product, "stock_qty", 0))
            if unit_weight <= 0 or stock < count:
                continue
            if abs(unit_weight - selected_unit_weight) > unit_tolerance:
                continue
            expected_weight = unit_weight * count
            residual = abs(target_weight - expected_weight)
            if residual <= allowed_residual:
                collisions.append(
                    {
                        "class_id": int(class_id),
                        "name": str(getattr(product, "product_name", "")),
                        "unit_weight": round(unit_weight, 1),
                        "count": int(count),
                        "expected_weight": round(expected_weight, 1),
                        "residual": round(residual, 1),
                    }
                )
        collisions.sort(key=lambda item: (float(item["residual"]), int(item["class_id"])))
        return collisions

    def _best_strict_weight_competing_candidate(
        self,
        *,
        selected_class_id: int,
        vision_candidates: List[EnsembleResult],
        active_map: dict[int, object],
        target_weight: float,
        tolerance: float,
    ) -> Optional[dict[str, Any]]:
        options: list[dict[str, Any]] = []
        for rank, candidate in enumerate(vision_candidates, start=1):
            class_id = int(candidate.class_id)
            if class_id == selected_class_id:
                continue
            product = active_map.get(class_id)
            if product is None or not self._active_product_has_loadcell(product):
                continue
            unit_weight = self._coerce_float(getattr(product, "product_weight", 0.0))
            stock = self._coerce_int(getattr(product, "stock_qty", 0))
            if unit_weight <= 0 or stock <= 0:
                continue
            count = max(1, int(round(target_weight / unit_weight)))
            if count > stock:
                continue
            expected_weight = unit_weight * count
            residual = abs(target_weight - expected_weight)
            if residual <= tolerance:
                options.append(
                    {
                        "rank": rank,
                        "class_id": class_id,
                        "name": str(getattr(product, "product_name", candidate.class_name)),
                        "source": str(getattr(candidate, "source", "vision") or "vision"),
                        "confidence": round(float(candidate.combined_confidence), 4),
                        "count": int(count),
                        "unit_weight": round(unit_weight, 1),
                        "expected_weight": round(expected_weight, 1),
                        "residual": round(residual, 1),
                    }
                )
        if not options:
            return None
        return sorted(
            options,
            key=lambda item: (
                0 if item["source"] == "vision" else 1,
                float(item["residual"]),
                int(item["rank"]),
            ),
        )[0]

    @staticmethod
    def _is_500ml_bottle_weight(unit_weight: float) -> bool:
        return 450.0 <= float(unit_weight) <= 560.0

    @staticmethod
    def _count_scaled_weight_tolerance(count: int, *, extra_units: int = 0) -> float:
        return float(config.weight.tolerance_grams) + (
            float(config.weight.same_product_count_tolerance_grams)
            * max(0, int(count) + int(extra_units))
        )

    @staticmethod
    def _source_includes_regular_vision(source: object) -> bool:
        return "vision" in {
            part.strip()
            for part in str(source or "").split("+")
            if part.strip()
        }

    def _regular_bottle_repeat_allowed_residual(
        self,
        *,
        candidate: EnsembleResult,
        rank: int,
        count: int,
        unit_weight: float,
        base_allowed_residual: float,
    ) -> float:
        if count != 2:
            return float(base_allowed_residual)
        if not self._is_500ml_bottle_weight(unit_weight):
            return float(base_allowed_residual)
        if not self._source_includes_regular_vision(getattr(candidate, "source", "vision")):
            return float(base_allowed_residual)
        if not self._has_same_product_count_evidence(candidate, rank):
            return float(base_allowed_residual)
        return max(
            float(base_allowed_residual),
            self._count_scaled_weight_tolerance(count, extra_units=1),
        )

    def _try_same_product_count_match(
        self,
        vision_candidates: List[EnsembleResult],
        delta_weight: float,
        timestamp: float,
        active_products: Optional[List] = None,
        trace_context: Optional[object] = None,
        stage_count_preempted: bool = False,
    ) -> Optional[JudgmentResult]:
        target_weight = abs(float(delta_weight))
        diagnostics: dict = {
            "decision_branch": "same_product_count_match",
            "accepted": False,
            "target_weight": round(target_weight, 1),
            "stage_count_preempted": bool(stage_count_preempted),
            "candidates": [],
        }
        if not vision_candidates or not active_products:
            self._record_weight_diagnostics(
                trace_context,
                {"same_product_count_match": diagnostics},
            )
            return None

        active_map = {
            int(product.yolo_class_id): product
            for product in active_products
            if getattr(product, "yolo_class_id", None) is not None
        }
        tolerance_per_item = float(config.weight.same_product_count_tolerance_grams)
        max_count_per_item = int(config.weight.max_count_per_item)
        max_same_product_count = max(2, int(config.weight.same_product_max_count))
        segment_grip_limit = self._segment_grip_limit_from_trace(trace_context)
        if segment_grip_limit is not None:
            diagnostics["segment_grip_limit"] = segment_grip_limit
        accepted: list[dict] = []

        for rank, candidate in enumerate(
            vision_candidates[: max(1, int(config.vision.top_k))],
            start=1,
        ):
            product = active_map.get(candidate.class_id)
            candidate_diag = {
                "rank": rank,
                "class_id": candidate.class_id,
                "name": candidate.class_name,
                "confidence": round(float(candidate.combined_confidence), 4),
            }
            if product is None:
                candidate_diag["reason"] = "not_in_active_products"
                diagnostics["candidates"].append(candidate_diag)
                continue
            if not self._active_product_has_loadcell(product):
                candidate_diag["reason"] = "no_loadcell_product"
                diagnostics["candidates"].append(candidate_diag)
                continue

            unit_weight = self._coerce_float(getattr(product, "product_weight", 0.0))
            stock = self._coerce_int(getattr(product, "stock_qty", 0))
            if unit_weight <= 0 or stock <= 0:
                candidate_diag["reason"] = "invalid_weight_or_stock"
                diagnostics["candidates"].append(candidate_diag)
                continue

            count = int(round(target_weight / unit_weight)) if unit_weight else 0
            candidate_diag["unit_weight"] = round(unit_weight, 1)
            candidate_diag["nearest_count"] = count
            if segment_grip_limit is not None:
                candidate_diag["segment_grip_limit"] = segment_grip_limit
            if count < 2:
                candidate_diag["reason"] = "single_count"
                diagnostics["candidates"].append(candidate_diag)
                continue
            if segment_grip_limit is not None and count > segment_grip_limit:
                candidate_diag["reason"] = "count_exceeds_segment_grip_limit"
                diagnostics["candidates"].append(candidate_diag)
                continue
            if count > min(stock, max_count_per_item, max_same_product_count):
                candidate_diag["reason"] = "count_exceeds_stock_or_limit"
                diagnostics["candidates"].append(candidate_diag)
                continue

            expected_weight = unit_weight * count
            residual = abs(target_weight - expected_weight)
            allowed_residual = tolerance_per_item * count + float(
                config.weight.tolerance_grams
            )
            candidate_diag["expected_weight"] = round(expected_weight, 1)
            candidate_diag["residual"] = round(residual, 1)
            candidate_diag["allowed_residual"] = round(allowed_residual, 1)
            if residual > allowed_residual:
                candidate_diag["reason"] = "residual_exceeds_count_tolerance"
                diagnostics["candidates"].append(candidate_diag)
                continue
            competing_candidate = self._same_product_competing_candidate(
                vision_candidates=vision_candidates,
                selected_candidate=candidate,
            )
            if count > 2 and competing_candidate is not None:
                candidate_diag["reason"] = "competing_multi_candidate_evidence"
                candidate_diag["competing_candidate"] = competing_candidate
                diagnostics["candidates"].append(candidate_diag)
                continue
            near_single = self._near_single_active_product(
                target_weight=target_weight,
                selected_product=product,
                active_products=active_products,
            )
            if count > 2 and near_single is not None:
                candidate_diag["reason"] = "near_single_active_product"
                candidate_diag["near_single"] = near_single
                diagnostics["candidates"].append(candidate_diag)
                continue
            if not self._has_same_product_count_evidence(candidate, rank):
                candidate_diag["reason"] = "insufficient_same_product_evidence"
                diagnostics["candidates"].append(candidate_diag)
                continue

            weight_score = max(
                0.5,
                1.0 - (residual / max(allowed_residual, 0.001)) * 0.5,
            )
            confidence = self._calculate_fusion_confidence(
                vision_score=candidate.combined_confidence,
                weight_score=weight_score,
                count=count,
            )
            accepted.append(
                {
                    "rank": rank,
                    "product": product,
                    "count": count,
                    "unit_weight": unit_weight,
                    "expected_weight": expected_weight,
                    "residual": residual,
                    "allowed_residual": allowed_residual,
                    "confidence": confidence,
                }
            )
            candidate_diag["reason"] = "accepted_candidate"
            diagnostics["candidates"].append(candidate_diag)

        if not accepted:
            diagnostics["reason"] = "no_same_product_count_match"
            self._record_weight_diagnostics(
                trace_context,
                {"same_product_count_match": diagnostics},
            )
            return None

        best = sorted(
            accepted,
            key=lambda item: (
                item["residual"] / max(item["allowed_residual"], 0.001),
                item["rank"],
                -item["confidence"],
            ),
        )[0]
        product = best["product"]
        unit_price = int(getattr(product, "sale_price", 0) or 0)
        product_judgment = ProductJudgment(
            product_id=int(product.yolo_class_id),
            name=str(product.product_name),
            count=int(best["count"]),
            unit_price=unit_price,
            total_price=unit_price * int(best["count"]),
            confidence=float(best["confidence"]),
            unit_weight=float(best["unit_weight"]),
        )
        diagnostics.update(
            {
                "accepted": True,
                "selected": {
                    "class_id": product_judgment.product_id,
                    "name": product_judgment.name,
                    "count": product_judgment.count,
                    "unit_weight": round(float(best["unit_weight"]), 1),
                    "expected_weight": round(float(best["expected_weight"]), 1),
                    "residual": round(float(best["residual"]), 1),
                    "allowed_residual": round(float(best["allowed_residual"]), 1),
                    "confidence": round(float(best["confidence"]), 4),
                    "rank": int(best["rank"]),
                },
            }
        )
        self._record_weight_diagnostics(
            trace_context,
            {
                "decision_branch": "same_product_count_match",
                "same_product_count_match": diagnostics,
            },
        )
        logger.info(
            "[ENGINE][reason=same_product_count_match] "
            f"selected={product_judgment.name}x{product_judgment.count}, "
            f"expected={best['expected_weight']:.1f}g, actual={target_weight:.1f}g, "
            f"residual={best['residual']:.1f}g, "
            f"allowed={best['allowed_residual']:.1f}g"
        )
        return JudgmentResult(
            products=[product_judgment],
            total_price=product_judgment.total_price,
            confidence=product_judgment.confidence,
            status=JudgmentStatus.COMPLETE,
            weight_delta=delta_weight,
            weight_explained=float(best["expected_weight"]),
            weight_residual=float(best["residual"]),
            timestamp=timestamp,
        )

    def _same_product_competing_candidate(
        self,
        *,
        vision_candidates: List[EnsembleResult],
        selected_candidate: EnsembleResult,
    ) -> Optional[dict[str, Any]]:
        selected_confidence = float(selected_candidate.combined_confidence)
        competing_threshold = max(
            float(config.weight.multi_kind_min_confidence),
            selected_confidence * 0.75,
        )
        for candidate in vision_candidates[: max(1, int(config.vision.top_k))]:
            if candidate.class_id == selected_candidate.class_id:
                continue
            confidence = float(candidate.combined_confidence)
            if confidence < competing_threshold:
                continue
            return {
                "class_id": int(candidate.class_id),
                "name": str(candidate.class_name),
                "confidence": round(confidence, 4),
                "threshold": round(competing_threshold, 4),
            }
        return None

    def _near_single_active_product(
        self,
        *,
        target_weight: float,
        selected_product: object,
        active_products: Optional[List],
    ) -> Optional[dict[str, Any]]:
        if not active_products:
            return None

        selected_class_id = getattr(selected_product, "yolo_class_id", None)
        tolerance = float(config.weight.detected_single_fallback_tolerance_grams)
        best: Optional[dict[str, Any]] = None
        for product in active_products:
            class_id = getattr(product, "yolo_class_id", None)
            if class_id is None or class_id == selected_class_id:
                continue
            if not self._active_product_has_loadcell(product):
                continue
            unit_weight = self._coerce_float(getattr(product, "product_weight", 0.0))
            stock = self._coerce_int(getattr(product, "stock_qty", 0))
            if unit_weight <= 0 or stock <= 0:
                continue
            residual = abs(target_weight - unit_weight)
            if residual > tolerance:
                continue
            candidate = {
                "class_id": int(class_id),
                "name": str(getattr(product, "product_name", "")),
                "unit_weight": round(unit_weight, 1),
                "residual": round(residual, 1),
            }
            if best is None or residual < float(best["residual"]):
                best = candidate
        return best

    def _has_same_product_count_evidence(
        self,
        candidate: EnsembleResult,
        rank: int,
    ) -> bool:
        confidence = float(candidate.combined_confidence)
        raw_votes = int(candidate.raw_vote_count or 0)
        votes = max(raw_votes, int(candidate.vote_count or 0))
        if candidate.source in {"threshold_rescue", "roi_rescue"}:
            return bool(candidate.weight_gate_passed) and confidence >= 0.18
        if confidence >= self.confidence_threshold:
            return True
        relaxed_rank1_threshold = max(0.18, self.confidence_threshold * 0.6)
        return (
            rank == 1
            and confidence >= relaxed_rank1_threshold
            and votes >= int(config.weight.detected_single_fallback_min_votes)
        )

    def _judge_strict(
        self,
        vision_candidates: List[EnsembleResult],
        delta_weight: float,
        timestamp: float,
        active_products: Optional[List] = None,
        trace_context: Optional[object] = None,
    ) -> Optional[JudgmentResult]:
        """
        무게 우선 엄격 판단 (v5.1).

        로드셀이 매우 정확(±3g)하므로:
        1. 무게로 가능한 모든 상품 조합을 찾음
        2. 그 중 YOLO가 감지한 것만 필터링
        3. Vision 신뢰도로 최종 선택
        4. 무게로 설명 불가 시 NO_DETECTION 반환

        Args:
            vision_candidates: YOLO 후보
            delta_weight: 무게 변화량
            timestamp: 판단 시각
            active_products: ActiveProductStore의 상품 정보

        Returns:
            JudgmentResult on strict success, None when relaxed fallback should continue.
        """
        # The strict path is weight-first. If it cannot fully explain the load
        # delta, the caller decides whether to degrade to the relaxed path.
        from model_service.weight.strict_weight_matcher import StrictWeightMatcher

        # StrictWeightMatcher 생성
        matcher = StrictWeightMatcher(
            tolerance=config.weight.tolerance_grams,
            max_items=self._strict_max_items_for_trace(trace_context),
            max_kinds=config.weight.max_combination_kinds,
        )

        # 유효한 조합 찾기
        valid_combos = matcher.find_valid_combinations(
            candidates=vision_candidates,
            delta_weight=delta_weight,
            active_products=active_products,
        )
        if trace_context is not None and hasattr(trace_context, "record_weight_diagnostics"):
            trace_context.record_weight_diagnostics(matcher.last_diagnostics)

        if not valid_combos:
            same_product_count_result = self._try_same_product_count_match(
                vision_candidates=vision_candidates,
                delta_weight=delta_weight,
                timestamp=timestamp,
                active_products=active_products,
                trace_context=trace_context,
                stage_count_preempted=True,
            )
            if same_product_count_result is not None:
                return same_product_count_result

            stage_combo_result = self._try_stage_count_combination_match(
                vision_candidates=vision_candidates,
                delta_weight=delta_weight,
                timestamp=timestamp,
                active_products=active_products,
                trace_context=trace_context,
            )
            if stage_combo_result is not None:
                return stage_combo_result

            rescue_result = self._try_rescue_single_match(
                vision_candidates,
                delta_weight,
                timestamp,
                active_products,
            )
            if rescue_result is not None:
                return rescue_result

            if config.weight.strict_mode_fallback:
                logger.warning(
                    f"[ENGINE][reason=strict_mismatch] "
                    f"delta={delta_weight:.1f}g, tolerance=±{config.weight.tolerance_grams}g, "
                    "fallback=enabled"
                )
                return None

            logger.warning(
                f"[ENGINE][reason=strict_mismatch] "
                f"delta={delta_weight:.1f}g, tolerance=±{config.weight.tolerance_grams}g, "
                "fallback=disabled -> NO_DETECTION"
            )
            return self._create_no_detection_result(delta_weight, timestamp)

        # 가장 신뢰도 높은 조합 선택 (이미 match_score 순 정렬됨)
        matcher_raw_top_combinations = list(
            matcher.last_diagnostics.get("valid_combinations", [])[:5]
        )

        self._sort_candidate_priority_combinations(
            valid_combos,
            vision_candidates,
            delta_weight=delta_weight,
            trace_context=trace_context,
        )
        best = valid_combos[0]
        self._record_strict_candidate_priority_selection(
            trace_context=trace_context,
            selected=best,
            post_sort_combinations=valid_combos,
            matcher_raw_top_combinations=matcher_raw_top_combinations,
        )
        priority_grace_result = self._try_candidate_priority_combination_grace(
            vision_candidates=vision_candidates,
            delta_weight=delta_weight,
            timestamp=timestamp,
            active_products=active_products,
            trace_context=trace_context,
            strict_best=best,
        )
        if priority_grace_result is not None:
            return priority_grace_result

        logger.info(
            f"[ENGINE][reason=strict_match] 최적 조합 선택 - "
            f"weight={best.total_weight:.1f}g (err={best.weight_error:.1f}g), "
            f"score={best.match_score:.3f}"
        )

        # 조합에서 JudgmentResult 생성
        return self._create_result_from_strict_combo(best, delta_weight, timestamp)

    def _try_candidate_priority_combination_grace(
        self,
        *,
        vision_candidates: List[EnsembleResult],
        delta_weight: float,
        timestamp: float,
        active_products: Optional[List] = None,
        trace_context: Optional[object] = None,
        strict_best=None,
    ) -> Optional[JudgmentResult]:
        """Let a rank-1 regular vision combo replace a same-weight lower-rank hit."""
        if delta_weight >= 0 or strict_best is None:
            return None
        if not vision_candidates or not active_products or strict_best.total_count < 2:
            return None

        from model_service.weight.strict_weight_matcher import (
            CandidateProduct,
            CombinationItem,
            ValidCombination,
        )

        target_weight = abs(float(delta_weight))
        strict_tolerance = float(config.weight.tolerance_grams)
        max_kinds = max(2, int(config.weight.max_combination_kinds))
        top_k = max(1, int(config.vision.top_k))
        active_map = {
            int(product.yolo_class_id): product
            for product in active_products
            if getattr(product, "yolo_class_id", None) is not None
        }
        if not active_map:
            return None

        candidate_products: list[tuple[int, CandidateProduct]] = []
        rejected: list[dict[str, Any]] = []
        for rank, candidate in enumerate(vision_candidates[:top_k], start=1):
            source = getattr(candidate, "source", "vision")
            diag = {
                "rank": rank,
                "class_id": int(candidate.class_id),
                "name": str(candidate.class_name),
                "source": str(source or "vision"),
                "confidence": round(float(candidate.combined_confidence), 4),
            }
            if not self._source_includes_regular_vision(source):
                diag["reason"] = "non_regular_candidate"
                rejected.append(diag)
                continue
            if float(candidate.combined_confidence) < float(
                config.weight.multi_kind_min_confidence
            ):
                diag["reason"] = "below_multi_kind_confidence"
                rejected.append(diag)
                continue
            product = active_map.get(int(candidate.class_id))
            if product is None:
                diag["reason"] = "not_in_active_products"
                rejected.append(diag)
                continue
            if not self._active_product_has_loadcell(product):
                diag["reason"] = "no_loadcell_product"
                rejected.append(diag)
                continue
            unit_weight = self._coerce_float(getattr(product, "product_weight", 0.0))
            stock = self._coerce_int(getattr(product, "stock_qty", 0))
            if unit_weight <= 0 or stock <= 0:
                diag["reason"] = "invalid_weight_or_stock"
                rejected.append(diag)
                continue

            candidate_products.append(
                (
                    rank,
                    CandidateProduct(
                        class_id=int(candidate.class_id),
                        name=str(getattr(product, "product_name", candidate.class_name)),
                        weight=unit_weight,
                        stock=stock,
                        vision_confidence=float(candidate.combined_confidence),
                        unit_price=int(getattr(product, "sale_price", 0) or 0),
                        vote_count=int(getattr(candidate, "vote_count", 1) or 1),
                        raw_vote_count=int(
                            getattr(candidate, "raw_vote_count", 0) or 0
                        ),
                        source="vision",
                        top_motion_passed=bool(
                            getattr(candidate, "top_motion_passed", False)
                        ),
                        side_motion_passed=bool(
                            getattr(candidate, "side_motion_passed", False)
                        ),
                        motion_gate_passed=bool(
                            getattr(candidate, "motion_gate_passed", True)
                        ),
                        weight_gate_passed=getattr(
                            candidate, "weight_gate_passed", None
                        ),
                    ),
                )
            )

        rank_by_class = {
            product.class_id: rank
            for rank, product in candidate_products
        }
        options: list[dict[str, Any]] = []
        considered: list[dict[str, Any]] = []
        for kind_count in range(2, min(max_kinds, len(candidate_products)) + 1):
            for combo_entries in combinations(candidate_products, kind_count):
                if not any(rank == 1 for rank, _ in combo_entries):
                    continue
                items = [
                    CombinationItem(candidate=product, count=1)
                    for _, product in combo_entries
                ]
                total_weight = sum(item.total_weight for item in items)
                residual = abs(target_weight - total_weight)
                allowed_residual = self._count_scaled_weight_tolerance(kind_count)
                combo_diag = {
                    "items": [
                        {
                            "class_id": item.candidate.class_id,
                            "name": item.candidate.name,
                            "rank": rank_by_class.get(item.candidate.class_id),
                            "unit_weight": round(float(item.candidate.weight), 1),
                        }
                        for item in items
                    ],
                    "total_weight": round(float(total_weight), 1),
                    "residual": round(float(residual), 1),
                    "allowed_residual": round(float(allowed_residual), 1),
                }
                if residual <= strict_tolerance:
                    combo_diag["reason"] = "already_strict_match"
                    considered.append(combo_diag)
                    continue
                if residual > allowed_residual:
                    combo_diag["reason"] = "residual_exceeds_grace_tolerance"
                    considered.append(combo_diag)
                    continue
                replacement = self._candidate_priority_grace_replacement(
                    items=items,
                    strict_best=strict_best,
                    rank_by_class=rank_by_class,
                    strict_tolerance=strict_tolerance,
                )
                if replacement is None:
                    combo_diag["reason"] = "not_same_weight_priority_replacement"
                    considered.append(combo_diag)
                    continue

                avg_confidence = sum(
                    item.candidate.vision_confidence for item in items
                ) / len(items)
                weight_score = max(
                    0.0,
                    1.0 - residual / max(allowed_residual, 0.001),
                )
                simplicity_score = max(0.0, 1.0 - (len(items) - 1) * 0.2)
                match_score = (
                    weight_score * 0.6
                    + min(max(avg_confidence, 0.0), 1.0) * 0.3
                    + simplicity_score * 0.1
                )
                valid_combo = ValidCombination(
                    items=items,
                    total_weight=total_weight,
                    target_weight=target_weight,
                    weight_error=residual,
                    avg_vision_confidence=avg_confidence,
                    match_score=match_score,
                )
                options.append(
                    {
                        "combo": valid_combo,
                        "allowed_residual": allowed_residual,
                        "replacement": replacement,
                        "rank_tuple": tuple(rank for rank, _ in combo_entries),
                    }
                )

        if not options:
            if considered:
                self._record_weight_diagnostics(
                    trace_context,
                    {
                        "candidate_priority_combination_grace": {
                            "accepted": False,
                            "target_weight": round(target_weight, 1),
                            "strict_best": strict_best.to_dict(),
                            "considered": considered[:10],
                            "rejected_candidates": rejected[:10],
                            "reason": "no_grace_candidate",
                        }
                    },
                )
            return None

        selected = sorted(
            options,
            key=lambda option: (
                option["rank_tuple"],
                float(option["combo"].weight_error)
                / max(float(option["allowed_residual"]), 0.001),
                -float(option["combo"].avg_vision_confidence),
            ),
        )[0]
        combo = selected["combo"]
        selected_dict = combo.to_dict()
        selected_dict["allowed_residual"] = round(
            float(selected["allowed_residual"]), 1
        )
        diagnostics = {
            "accepted": True,
            "target_weight": round(target_weight, 1),
            "strict_tolerance": round(strict_tolerance, 1),
            "selected": selected_dict,
            "rejected_strict_best": strict_best.to_dict(),
            "replacement": selected["replacement"],
            "rejected_candidates": rejected[:10],
            "reason": "rank1_regular_vision_over_same_weight_lower_rank_collision",
        }
        self._record_weight_diagnostics(
            trace_context,
            {
                "decision_branch": "candidate_priority_combination_grace",
                "candidate_priority_combination_grace": diagnostics,
            },
        )
        logger.info(
            "[ENGINE][reason=candidate_priority_combination_grace] "
            f"target={target_weight:.1f}g, expected={combo.total_weight:.1f}g, "
            f"residual={combo.weight_error:.1f}g"
        )
        return self._create_result_from_strict_combo(combo, delta_weight, timestamp)

    def _candidate_priority_grace_replacement(
        self,
        *,
        items: list,
        strict_best,
        rank_by_class: dict[int, int],
        strict_tolerance: float,
    ) -> Optional[dict[str, Any]]:
        grace_ids = [int(item.candidate.class_id) for item in items]
        strict_ids = [int(item.candidate.class_id) for item in strict_best.items]
        if len(grace_ids) != len(strict_ids):
            return None
        grace_set = set(grace_ids)
        strict_set = set(strict_ids)
        if grace_set == strict_set:
            return None
        overlap = grace_set & strict_set
        if len(overlap) < max(1, min(len(grace_set), len(strict_set)) - 1):
            return None

        grace_only = [
            item
            for item in items
            if int(item.candidate.class_id) not in strict_set
        ]
        strict_only = [
            item
            for item in strict_best.items
            if int(item.candidate.class_id) not in grace_set
        ]
        for grace_item in grace_only:
            grace_rank = rank_by_class.get(int(grace_item.candidate.class_id), 9999)
            if grace_rank != 1:
                continue
            for strict_item in strict_only:
                strict_class_id = int(strict_item.candidate.class_id)
                strict_rank = rank_by_class.get(strict_class_id, 9999)
                unit_delta = abs(
                    float(grace_item.candidate.weight)
                    - float(strict_item.candidate.weight)
                )
                same_weight = unit_delta <= strict_tolerance
                lower_priority = (
                    grace_rank < strict_rank
                    or str(strict_item.candidate.source) != "vision"
                )
                if same_weight and lower_priority:
                    return {
                        "rank1_class_id": int(grace_item.candidate.class_id),
                        "rank1_name": grace_item.candidate.name,
                        "replaced_class_id": strict_class_id,
                        "replaced_name": strict_item.candidate.name,
                        "rank1_rank": int(grace_rank),
                        "replaced_rank": int(strict_rank),
                        "unit_weight_delta": round(unit_delta, 1),
                    }
        return None

    def _try_rescue_single_match(
        self,
        vision_candidates: List[EnsembleResult],
        delta_weight: float,
        timestamp: float,
        active_products: Optional[List] = None,
    ) -> Optional[JudgmentResult]:
        """Recover only weight-gated vision rescue candidates after strict miss."""
        rescue_candidates = [
            candidate
            for candidate in vision_candidates
            if candidate.source in {"threshold_rescue", "roi_rescue"}
        ]
        if not rescue_candidates or not active_products:
            return None

        active_map = {
            int(product.yolo_class_id): product
            for product in active_products
            if getattr(product, "yolo_class_id", None) is not None
        }
        target_weight = abs(float(delta_weight))
        rescue_tolerance = float(config.weight.rescue_tolerance_grams)
        no_motion_tolerance = float(
            config.vision.weight_rescue_no_motion_max_residual_grams
        )
        viable: list[tuple[float, float, int, EnsembleResult, object]] = []

        for candidate in rescue_candidates:
            product = active_map.get(candidate.class_id)
            if product is None or not self._active_product_has_loadcell(product):
                continue
            product_weight = float(getattr(product, "product_weight", 0.0) or 0.0)
            stock_qty = int(getattr(product, "stock_qty", 0) or 0)
            if product_weight <= 0 or stock_qty <= 0:
                continue

            residual = abs(target_weight - product_weight)
            if residual > rescue_tolerance:
                continue

            motion_gate_passed = bool(
                candidate.top_motion_passed
                or candidate.side_motion_passed
                or (
                    candidate.source != "threshold_rescue"
                    and candidate.motion_gate_passed
                )
            )
            if candidate.source == "threshold_rescue" and not motion_gate_passed:
                if not config.vision.weight_rescue_no_motion_enabled:
                    continue
                if candidate.raw_vote_count < config.vision.weight_rescue_no_motion_min_raw_votes:
                    continue
                if residual > no_motion_tolerance:
                    continue

            viable.append(
                (
                    residual,
                    -float(candidate.combined_confidence),
                    -int(candidate.raw_vote_count or candidate.vote_count),
                    candidate,
                    product,
                )
            )

        if not viable:
            return None

        viable.sort(key=lambda item: (item[0], item[1], item[2]))
        residual, _, _, candidate, product = viable[0]
        product_weight = float(getattr(product, "product_weight", 0.0) or 0.0)
        price = int(getattr(product, "sale_price", 0) or 0)
        confidence = min(max(float(candidate.combined_confidence), 0.01), 1.0)
        product_judgment = ProductJudgment(
            product_id=int(getattr(product, "yolo_class_id", candidate.class_id)),
            name=getattr(product, "product_name", candidate.class_name),
            count=1,
            unit_price=price,
            total_price=price,
            confidence=confidence,
            unit_weight=product_weight,
        )

        logger.info(
            "[ENGINE][reason=rescue_weight_match] "
            f"source={candidate.source}, selected={product_judgment.name}, "
            f"expected={product_weight:.1f}g, actual={target_weight:.1f}g, "
            f"residual={residual:.1f}g, tolerance={rescue_tolerance:.1f}g"
        )

        return JudgmentResult(
            products=[product_judgment],
            total_price=product_judgment.total_price,
            confidence=confidence,
            status=JudgmentStatus.COMPLETE,
            weight_delta=delta_weight,
            weight_explained=product_weight,
            weight_residual=round(residual, 1),
            timestamp=timestamp,
        )

    def _try_stage_count_combination_match(
        self,
        vision_candidates: List[EnsembleResult],
        delta_weight: float,
        timestamp: float,
        active_products: Optional[List] = None,
        trace_context: Optional[object] = None,
    ) -> Optional[JudgmentResult]:
        """Try compact strict matching with candidates plus stage-count evidence."""
        if trace_context is None or not active_products:
            return None

        merged_candidates, diagnostics = self._build_stage_count_combination_candidates(
            vision_candidates=vision_candidates,
            trace_context=trace_context,
        )
        if diagnostics["evidence_candidates_added"] <= 0:
            return None

        from model_service.weight.strict_weight_matcher import StrictWeightMatcher

        matcher = StrictWeightMatcher(
            tolerance=config.weight.tolerance_grams,
            max_items=self._strict_max_items_for_trace(trace_context),
            max_kinds=config.weight.max_combination_kinds,
        )
        valid_combos = matcher.find_valid_combinations(
            candidates=merged_candidates,
            delta_weight=delta_weight,
            active_products=active_products,
        )
        single_match_count = sum(1 for combo in valid_combos if combo.total_count < 2)
        valid_combos = [combo for combo in valid_combos if combo.total_count >= 2]
        diagnostics["accepted"] = bool(valid_combos)
        diagnostics["single_matches_ignored"] = single_match_count
        diagnostics["strict_diagnostics"] = matcher.last_diagnostics

        if not valid_combos:
            self._record_weight_diagnostics(
                trace_context,
                {"stage_count_combination_match": diagnostics},
            )
            return None

        self._sort_candidate_priority_combinations(
            valid_combos,
            vision_candidates,
            delta_weight=delta_weight,
            trace_context=trace_context,
        )
        best = valid_combos[0]
        diagnostics["selected"] = best.to_dict()
        self._record_weight_diagnostics(
            trace_context,
            {
                "decision_branch": "stage_count_combination_match",
                "stage_count_combination_match": diagnostics,
            },
        )
        logger.info(
            "[ENGINE][reason=stage_count_combination_match] "
            f"merged_candidates={diagnostics['merged_candidate_count']}, "
            f"stage_added={diagnostics['stage_candidates_added']}, "
            f"weight={best.total_weight:.1f}g, residual={best.weight_error:.1f}g"
        )
        return self._create_result_from_strict_combo(best, delta_weight, timestamp)

    def _try_candidate_only_strict_combination_match(
        self,
        vision_candidates: List[EnsembleResult],
        delta_weight: float,
        timestamp: float,
        active_products: Optional[List] = None,
        trace_context: Optional[object] = None,
    ) -> Optional[JudgmentResult]:
        """Try final-candidate-only strict combinations before stage-count expansion."""
        if not vision_candidates or not active_products:
            return None

        from model_service.weight.strict_weight_matcher import StrictWeightMatcher

        matcher = StrictWeightMatcher(
            tolerance=config.weight.tolerance_grams,
            max_items=self._strict_max_items_for_trace(trace_context),
            max_kinds=config.weight.max_combination_kinds,
        )
        valid_combos = matcher.find_valid_combinations(
            candidates=vision_candidates,
            delta_weight=delta_weight,
            active_products=active_products,
        )
        valid_combos = [combo for combo in valid_combos if combo.total_count >= 2]
        if not valid_combos:
            return None

        self._sort_candidate_priority_combinations(
            valid_combos,
            vision_candidates,
            delta_weight=delta_weight,
            trace_context=trace_context,
        )
        best = valid_combos[0]
        self._record_weight_diagnostics(
            trace_context,
            {
                "relaxed_candidate_only_strict_combination_match": {
                    "reason": "candidate_only_strict_combination_match",
                    "input_candidate_count": len(vision_candidates),
                    "selected": best.to_dict(),
                    "strict_diagnostics": matcher.last_diagnostics,
                }
            },
        )
        logger.info(
            "[ENGINE][reason=candidate_only_strict_combination_match] "
            f"candidates={len(vision_candidates)}, "
            f"weight={best.total_weight:.1f}g, residual={best.weight_error:.1f}g"
        )
        return self._create_result_from_strict_combo(best, delta_weight, timestamp)

    @classmethod
    def _sort_candidate_priority_combinations(
        cls,
        valid_combos: list,
        vision_candidates: List[EnsembleResult],
        *,
        delta_weight: float = 0.0,
        trace_context: Optional[object] = None,
    ) -> None:
        vision_class_ids = {
            int(candidate.class_id)
            for candidate in vision_candidates
            if getattr(candidate, "class_id", None) is not None
        }
        candidate_priority_by_class: dict[int, tuple[int, int]] = {}
        for rank, candidate in enumerate(vision_candidates, start=1):
            try:
                class_id = int(candidate.class_id)
            except (TypeError, ValueError):
                continue
            source = getattr(candidate, "source", "vision")
            source_priority = cls._candidate_source_priority(source)
            existing = candidate_priority_by_class.get(class_id)
            priority = (rank, source_priority)
            if existing is None or priority < existing:
                candidate_priority_by_class[class_id] = priority

        def combination_source_counts(combo) -> tuple[int, int]:
            candidate_units = 0
            stage_units = 0
            for item in combo.items:
                try:
                    class_id = int(item.candidate.class_id)
                except (TypeError, ValueError):
                    stage_units += int(item.count)
                    continue
                if class_id in vision_class_ids and item.candidate.source != "stage_counts":
                    candidate_units += int(item.count)
                else:
                    stage_units += int(item.count)
            return candidate_units, stage_units

        def combination_candidate_priority(combo) -> tuple[int, int]:
            priorities: list[tuple[int, int]] = []
            for item in combo.items:
                try:
                    class_id = int(item.candidate.class_id)
                except (TypeError, ValueError):
                    continue
                priority = candidate_priority_by_class.get(class_id)
                if priority is not None:
                    priorities.append(priority)
            return min(priorities) if priorities else (2, 9999)

        returned_weight_hints = cls._returned_weight_hints(trace_context)
        has_static_single_match = any(
            combo.total_count == 1
            and getattr(combo, "motion_evidence_ratio", 0.0) <= 0.0
            for combo in valid_combos
        )

        def sort_key(
            combo,
        ) -> tuple[int, int, int, int, int, int, float, int, float, float, int, float]:
            candidate_units, stage_units = combination_source_counts(combo)
            returned_overlap = (
                cls._returned_weight_overlap(combo, returned_weight_hints)
                if delta_weight < 0 and returned_weight_hints
                else 0
            )
            motion_supported_multi = (
                has_static_single_match
                and combo.total_count > 1
                and getattr(combo, "motion_evidence_ratio", 0.0) > 0.0
            )
            evidence_score = float(getattr(combo, "evidence_score", 0.0))
            candidate_rank, candidate_source_priority = combination_candidate_priority(
                combo
            )
            if combo.total_count != 1:
                candidate_rank = 0
                candidate_source_priority = 0
            return (
                0 if motion_supported_multi else 1,
                int(candidate_units <= 0),
                returned_overlap,
                combo.total_count,
                candidate_rank,
                candidate_source_priority,
                combo.weight_error,
                stage_units,
                -evidence_score,
                -combo.avg_vision_confidence,
                combo.kind_count,
                -combo.match_score,
            )

        valid_combos.sort(key=sort_key)

    @staticmethod
    def _candidate_source_priority(source: object) -> int:
        source_name = str(source or "vision")
        if source_name == "vision":
            return 0
        if source_name == "stage_weight_gate":
            return 1
        if source_name == "stage_counts":
            return 2
        if source_name in {"threshold_rescue", "roi_rescue"}:
            return 3
        if source_name == "diagnostic":
            return 4
        return 5

    @classmethod
    def _record_strict_candidate_priority_selection(
        cls,
        *,
        trace_context: Optional[object],
        selected,
        post_sort_combinations: list,
        matcher_raw_top_combinations: list[dict[str, Any]],
    ) -> None:
        """Record engine post-sort strict selection separately from matcher order."""
        if trace_context is None or not hasattr(trace_context, "record_weight_diagnostics"):
            return

        selected_dict = selected.to_dict()
        post_sort_top = [
            combo.to_dict()
            for combo in post_sort_combinations[:5]
        ]
        reason = "strict_match"
        raw_top = matcher_raw_top_combinations[0] if matcher_raw_top_combinations else None
        selected_source = cls._strict_combo_first_source(selected_dict)
        if (
            raw_top is not None
            and cls._strict_combo_first_class_id(raw_top)
            != cls._strict_combo_first_class_id(selected_dict)
            and cls._strict_combo_is_candidate_priority_single(selected_dict)
        ):
            reason = (
                "stage_weight_gate_candidate_priority"
                if selected_source == "stage_weight_gate"
                else (
                    "regular_single_candidate_priority"
                    if selected_source == "vision"
                    else "ranked_single_candidate_priority"
                )
            )

        cls._record_weight_diagnostics(
            trace_context,
            {
                "strict_candidate_priority_selection": {
                    "reason": reason,
                    "selected": selected_dict,
                    "matcher_raw_top_combinations": matcher_raw_top_combinations,
                    "post_sort_top_combinations": post_sort_top,
                }
            },
        )

    @staticmethod
    def _strict_combo_first_class_id(combo: dict[str, Any]) -> int | None:
        items = combo.get("items") or []
        if not items:
            return None
        try:
            return int(items[0].get("class_id"))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _strict_combo_first_source(combo: dict[str, Any]) -> str:
        items = combo.get("items") or []
        if not items:
            return ""
        return str(items[0].get("source") or "vision")

    @staticmethod
    def _strict_combo_is_candidate_priority_single(combo: dict[str, Any]) -> bool:
        if int(combo.get("total_count") or 0) != 1:
            return False
        items = combo.get("items") or []
        if len(items) != 1:
            return False
        return str(items[0].get("source") or "vision") in {
            "vision",
            "stage_weight_gate",
            "threshold_rescue",
            "roi_rescue",
        }

    @staticmethod
    def _returned_weight_hints(trace_context: Optional[object]) -> list[float]:
        if trace_context is None:
            return []
        loadcell = getattr(trace_context, "loadcell", {}) or {}
        if not isinstance(loadcell, dict):
            return []

        hints: list[float] = []
        for key in ("compound_positive_weights_g", "recent_return_weights_g"):
            values = loadcell.get(key) or []
            if isinstance(values, (int, float)):
                values = [values]
            for value in values:
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    continue
                if parsed > 0:
                    hints.append(parsed)

        for event in loadcell.get("recent_same_zone_events") or []:
            if not isinstance(event, dict) or int(event.get("sign", 0) or 0) <= 0:
                continue
            try:
                parsed = float(event.get("abs_weight", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                hints.append(parsed)

        return hints

    @staticmethod
    def _returned_weight_overlap(combo, returned_weight_hints: list[float]) -> int:
        if not returned_weight_hints:
            return 0
        tolerance = max(0.0, float(config.weight.tolerance_grams))
        overlap = 0
        for item in combo.items:
            unit_weight = float(getattr(item.candidate, "weight", 0.0) or 0.0)
            if any(
                abs(unit_weight - hint) <= tolerance
                for hint in returned_weight_hints
            ):
                overlap += int(item.count)
        return overlap

    def _build_stage_count_combination_candidates(
        self,
        *,
        vision_candidates: List[EnsembleResult],
        trace_context: object,
    ) -> tuple[list[EnsembleResult], dict[str, Any]]:
        limit = self.STAGE_COUNT_COMBINATION_LIMIT
        merged: list[EnsembleResult] = []
        seen_class_ids: set[int] = set()

        for candidate in vision_candidates:
            if len(merged) >= limit:
                break
            try:
                class_id = int(candidate.class_id)
            except (TypeError, ValueError):
                continue
            if class_id in seen_class_ids:
                continue
            merged.append(candidate)
            seen_class_ids.add(class_id)

        stage_counts = getattr(trace_context, "stage_counts_by_class", {}) or {}
        stage_entries: list[tuple[_StageEvidenceSummary, dict[str, Any]]] = []
        for entry in stage_counts.values():
            if not isinstance(entry, dict):
                continue
            try:
                class_id = int(entry.get("class_id"))
            except (TypeError, ValueError):
                continue
            if class_id in seen_class_ids:
                continue
            summary = self._stage_evidence_summary(entry)
            if summary.votes <= 0 and summary.confidence <= 0:
                continue
            stage_entries.append((summary, entry))

        stage_entries.sort(
            key=lambda item: (
                -round(float(item[0].stage_score), 4),
                -round(float(item[0].side_confidence), 4),
                -round(float(item[0].confidence), 4),
                -int(item[0].side_votes),
                -int(item[0].votes),
            )
        )

        stage_added: list[dict[str, Any]] = []
        for summary, entry in stage_entries:
            if len(merged) >= limit:
                break
            class_id = int(entry.get("class_id"))
            candidate = EnsembleResult(
                class_id=class_id,
                class_name=str(entry.get("name") or ""),
                top_confidence=summary.top_confidence,
                side_confidence=summary.side_confidence,
                combined_confidence=summary.confidence,
                vote_count=max(1, summary.votes),
                source="stage_counts",
                raw_vote_count=summary.votes,
                top_motion_passed=summary.top_motion_passed,
                side_motion_passed=summary.side_motion_passed,
                motion_gate_passed=bool(
                    summary.motion_gate_passed
                    or entry.get("final_rank") is not None
                    or entry.get("weight_gate_passed")
                ),
                weight_gate_passed=entry.get("weight_gate_passed"),
            )
            merged.append(candidate)
            seen_class_ids.add(class_id)
            stage_added.append(
                {
                    "class_id": class_id,
                    "name": candidate.class_name,
                    "votes": summary.votes,
                    "confidence": round(summary.confidence, 4),
                    "stage_score": round(summary.stage_score, 4),
                    "side_confidence": round(summary.side_confidence, 4),
                    "top_confidence": round(summary.top_confidence, 4),
                    "side_votes": summary.side_votes,
                    "top_votes": summary.top_votes,
                    "score_reason": summary.score_reason,
                }
            )

        rescue_added: list[dict[str, Any]] = []
        for source_name, attr_name in (
            ("threshold_rescue", "threshold_rescue_candidates"),
            ("roi_rescue", "roi_rescue_candidates"),
        ):
            rescue_candidates = getattr(trace_context, attr_name, []) or []
            for raw_candidate in rescue_candidates:
                if len(merged) >= limit:
                    break
                if not isinstance(raw_candidate, dict):
                    continue
                try:
                    class_id = int(raw_candidate.get("class_id"))
                except (TypeError, ValueError):
                    continue
                if class_id in seen_class_ids:
                    continue
                confidence = self._coerce_float(raw_candidate.get("confidence", 0.0))
                votes = max(
                    1 if confidence > 0 else 0,
                    self._coerce_int(raw_candidate.get("votes", 0)),
                    self._coerce_int(raw_candidate.get("raw_vote_count", 0)),
                )
                if votes <= 0 and confidence <= 0:
                    continue
                candidate = EnsembleResult(
                    class_id=class_id,
                    class_name=str(raw_candidate.get("name") or ""),
                    top_confidence=confidence if raw_candidate.get("top") else 0.0,
                    side_confidence=confidence if raw_candidate.get("side") else confidence,
                    combined_confidence=confidence,
                    vote_count=max(1, votes),
                    source=source_name,
                    raw_vote_count=votes,
                    top_motion_passed=bool(raw_candidate.get("top_motion_passed")),
                    side_motion_passed=bool(raw_candidate.get("side_motion_passed")),
                    motion_gate_passed=bool(raw_candidate.get("motion_gate_passed")),
                    weight_gate_passed=raw_candidate.get("weight_gate_passed"),
                )
                merged.append(candidate)
                seen_class_ids.add(class_id)
                rescue_added.append(
                    {
                        "class_id": class_id,
                        "name": candidate.class_name,
                        "source": source_name,
                        "votes": votes,
                        "confidence": round(confidence, 4),
                    }
                )

        diagnostic_added: list[dict[str, Any]] = []
        evidence_by_class = self._collect_detected_single_evidence([], trace_context)
        diagnostic_evidence = [
            evidence
            for evidence in evidence_by_class.values()
            if "diagnostic" in set(str(evidence.source).split("+"))
        ]
        diagnostic_evidence.sort(
            key=lambda evidence: (
                -int(evidence.strong),
                -int(evidence.votes),
                -round(float(evidence.confidence), 4),
            )
        )
        min_votes = max(1, int(config.weight.detected_single_fallback_min_votes))
        for evidence in diagnostic_evidence:
            if len(merged) >= limit:
                break
            if evidence.class_id in seen_class_ids:
                continue
            if not evidence.strong and evidence.votes < min_votes:
                continue
            candidate = EnsembleResult(
                class_id=evidence.class_id,
                class_name=evidence.name,
                top_confidence=evidence.top_confidence,
                side_confidence=evidence.side_confidence,
                combined_confidence=evidence.confidence,
                vote_count=max(1, evidence.votes),
                source="diagnostic",
                raw_vote_count=evidence.votes,
                top_motion_passed=evidence.top_votes > 0,
                side_motion_passed=evidence.side_votes > 0,
                motion_gate_passed=evidence.motion_gate_passed,
            )
            merged.append(candidate)
            seen_class_ids.add(evidence.class_id)
            diagnostic_added.append(
                {
                    "class_id": evidence.class_id,
                    "name": candidate.class_name,
                    "votes": evidence.votes,
                    "confidence": round(float(evidence.confidence), 4),
                }
            )

        diagnostics = {
            "reason": "stage_count_combination_match",
            "candidate_limit": limit,
            "input_candidate_count": len(vision_candidates),
            "merged_candidate_count": len(merged),
            "stage_candidates_added": len(stage_added),
            "rescue_candidates_added": len(rescue_added),
            "diagnostic_candidates_added": len(diagnostic_added),
            "evidence_candidates_added": len(stage_added)
            + len(rescue_added)
            + len(diagnostic_added),
            "stage_candidates": stage_added,
            "rescue_candidates": rescue_added,
            "diagnostic_candidates": diagnostic_added,
            "accepted": False,
        }
        return merged, diagnostics

    def _try_detected_single_item_fallback(
        self,
        vision_candidates: List[EnsembleResult],
        delta_weight: float,
        timestamp: float,
        active_products: Optional[List] = None,
        trace_context: Optional[object] = None,
    ) -> Optional[JudgmentResult]:
        """Recover a one-item removal from detected evidence after all matches miss."""
        if not config.weight.detected_single_fallback_enabled:
            return None
        if delta_weight >= 0:
            return None
        if not active_products:
            return None

        target_weight = abs(float(delta_weight))
        if target_weight < self.min_weight_change:
            return None

        active_map: dict[int, object] = {}
        for product in active_products:
            class_id = getattr(product, "yolo_class_id", None)
            if class_id is None:
                continue
            try:
                active_map[int(class_id)] = product
            except (TypeError, ValueError):
                continue

        evidence_by_class = self._collect_detected_single_evidence(
            vision_candidates,
            trace_context,
        )
        if not evidence_by_class:
            return None

        tolerance = float(config.weight.detected_single_fallback_tolerance_grams)
        min_votes = max(0, int(config.weight.detected_single_fallback_min_votes))
        identity_allowed_residual = tolerance + float(
            config.weight.same_product_count_tolerance_grams
        )
        diagnostics: dict[str, Any] = {
            "reason": "detected_single_item_fallback",
            "target_weight": round(target_weight, 1),
            "tolerance": round(tolerance, 1),
            "min_votes": min_votes,
            "considered": len(evidence_by_class),
            "accepted": None,
            "candidates": [],
        }
        viable: list[tuple[float, int, float, _DetectedSingleEvidence, object, float, int]] = []
        identity_viable: list[
            tuple[float, int, float, _DetectedSingleEvidence, object, float, int, float]
        ] = []

        for evidence in evidence_by_class.values():
            product = active_map.get(evidence.class_id)
            product_weight = 0.0
            stock_qty = 0
            price = 0
            residual: float | None = None
            reason = "accepted"

            if product is None:
                reason = "not_active"
            elif not self._active_product_has_loadcell(product):
                reason = "no_loadcell"
            else:
                product_weight = self._coerce_float(
                    getattr(product, "product_weight", 0.0)
                )
                stock_qty = self._coerce_int(getattr(product, "stock_qty", 0))
                price = self._coerce_int(getattr(product, "sale_price", 0))
                residual = abs(target_weight - product_weight) if product_weight > 0 else None
                if product_weight <= 0:
                    reason = "invalid_weight"
                elif stock_qty <= 0:
                    reason = "zero_stock"
                elif (
                    evidence.source == "threshold_rescue"
                    and not evidence.motion_gate_passed
                    and evidence.weight_gate_passed is not True
                ):
                    reason = "motion_rejected_threshold_rescue"
                elif not evidence.trusted and evidence.votes < min_votes:
                    reason = "insufficient_votes"
                elif residual is None or residual > tolerance:
                    reason = "weight_mismatch"

            identity_rescue_eligible = self._has_strong_single_bottle_identity_evidence(
                evidence=evidence,
                unit_weight=product_weight,
                residual=residual,
                allowed_residual=identity_allowed_residual,
            )
            diagnostics["candidates"].append(
                {
                    "class_id": evidence.class_id,
                    "name": evidence.name,
                    "source": evidence.source,
                    "votes": evidence.votes,
                    "confidence": round(evidence.confidence, 4),
                    "top_confidence": round(evidence.top_confidence, 4),
                    "side_confidence": round(evidence.side_confidence, 4),
                    "top_votes": evidence.top_votes,
                    "side_votes": evidence.side_votes,
                    "motion_gate_passed": evidence.motion_gate_passed,
                    "unit_weight_g": round(product_weight, 1) if product_weight > 0 else None,
                    "stock_qty": stock_qty,
                    "weight_residual_g": (
                        round(float(residual), 1) if residual is not None else None
                    ),
                    "single_bottle_identity_override_eligible": identity_rescue_eligible,
                    "single_bottle_identity_allowed_residual_g": (
                        round(identity_allowed_residual, 1)
                        if product_weight > 0
                        else None
                    ),
                    "reason": reason,
                }
            )

            if identity_rescue_eligible and residual is not None:
                identity_viable.append(
                    (
                        residual,
                        -int(evidence.side_votes),
                        -float(evidence.side_confidence),
                        evidence,
                        product,
                        product_weight,
                        price,
                        identity_allowed_residual,
                    )
                )

            if reason != "accepted" or residual is None:
                continue

            viable.append(
                (
                    residual,
                    -int(evidence.votes),
                    -float(evidence.confidence),
                    evidence,
                    product,
                    product_weight,
                    price,
                )
            )

        normal_selection = None
        if viable:
            viable.sort(key=lambda item: (item[0], item[1], item[2]))
            normal_selection = viable[0]

        identity_selection, identity_diagnostics = (
            self._select_single_bottle_identity_override(
                normal_selection=normal_selection,
                identity_candidates=identity_viable,
                base_tolerance=tolerance,
            )
        )
        diagnostics["single_bottle_identity_override"] = identity_diagnostics

        if identity_selection is not None:
            (
                residual,
                _,
                _,
                evidence,
                product,
                product_weight,
                price,
                selection_tolerance,
            ) = identity_selection
            selected_by_identity_override = True
        elif normal_selection is not None:
            (
                residual,
                _,
                _,
                evidence,
                product,
                product_weight,
                price,
            ) = normal_selection
            selection_tolerance = tolerance
            selected_by_identity_override = False
        else:
            self._record_detected_single_fallback(trace_context, diagnostics)
            return None

        product_name = getattr(product, "product_name", evidence.name)
        weight_score = (
            max(0.0, 1.0 - (residual / selection_tolerance))
            if selection_tolerance > 0
            else 0.0
        )
        confidence = min(
            0.65,
            max(0.05, 0.45 * weight_score + 0.25 * evidence.confidence + 0.10),
        )
        product_judgment = ProductJudgment(
            product_id=int(getattr(product, "yolo_class_id", evidence.class_id)),
            name=product_name,
            count=1,
            unit_price=price,
            total_price=price,
            confidence=confidence,
            unit_weight=product_weight,
        )

        accepted = {
            "class_id": product_judgment.product_id,
            "name": product_judgment.name,
            "source": evidence.source,
            "votes": evidence.votes,
            "confidence": round(confidence, 4),
            "unit_weight_g": round(product_weight, 1),
            "weight_residual_g": round(float(residual), 1),
            "single_bottle_identity_override": selected_by_identity_override,
        }
        diagnostics["accepted"] = accepted
        diagnostics["fallback_reason"] = "detected_single_item_fallback"
        self._record_detected_single_fallback(trace_context, diagnostics)

        logger.info(
            "[ENGINE][reason=detected_single_item_fallback] "
            f"selected={product_judgment.name}, expected={product_weight:.1f}g, "
            f"actual={target_weight:.1f}g, residual={residual:.1f}g, "
            f"votes={evidence.votes}, source={evidence.source}"
        )

        return JudgmentResult(
            products=[product_judgment],
            total_price=product_judgment.total_price,
            confidence=confidence,
            status=JudgmentStatus.COMPLETE,
            weight_delta=delta_weight,
            weight_explained=product_weight,
            weight_residual=round(float(residual), 1),
            timestamp=timestamp,
        )

    def _has_strong_single_bottle_identity_evidence(
        self,
        *,
        evidence: _DetectedSingleEvidence,
        unit_weight: float,
        residual: Optional[float] = None,
        allowed_residual: Optional[float] = None,
    ) -> bool:
        if unit_weight <= 0 or not self._is_500ml_bottle_weight(unit_weight):
            return False
        if residual is not None and allowed_residual is not None:
            if residual > allowed_residual:
                return False
        source_parts = {
            part.strip()
            for part in str(evidence.source or "").split("+")
            if part.strip()
        }
        if not ({"stage_counts", "diagnostic"} & source_parts):
            return False
        if not evidence.motion_gate_passed:
            return False
        if not evidence.strong:
            return False

        min_side_votes = max(
            int(config.weight.detected_single_fallback_min_votes),
            4,
        )
        side_threshold = max(0.45, float(self.confidence_threshold))
        return (
            int(evidence.side_votes) >= min_side_votes
            and float(evidence.side_confidence) >= side_threshold
            and float(evidence.confidence) >= side_threshold
        )

    def _select_single_bottle_identity_override(
        self,
        *,
        normal_selection: Optional[
            tuple[float, int, float, _DetectedSingleEvidence, object, float, int]
        ],
        identity_candidates: list[
            tuple[float, int, float, _DetectedSingleEvidence, object, float, int, float]
        ],
        base_tolerance: float,
    ) -> tuple[
        Optional[
            tuple[float, int, float, _DetectedSingleEvidence, object, float, int, float]
        ],
        dict[str, Any],
    ]:
        residual_gap_allowed = float(base_tolerance) + float(
            config.weight.same_product_count_tolerance_grams
        )
        diagnostics: dict[str, Any] = {
            "accepted": False,
            "reason": "no_strong_identity_candidate",
            "allowed_residual_g": round(
                float(base_tolerance)
                + float(config.weight.same_product_count_tolerance_grams),
                1,
            ),
            "residual_gap_allowed_g": round(residual_gap_allowed, 1),
            "candidate": None,
            "replaced": None,
        }
        if not identity_candidates:
            return None, diagnostics

        identity_candidates.sort(
            key=lambda item: (
                -float(item[3].side_confidence),
                -int(item[3].side_votes),
                -float(item[3].stage_score),
                float(item[0]),
            )
        )
        identity_selection = identity_candidates[0]
        (
            identity_residual,
            _,
            _,
            identity_evidence,
            _identity_product,
            identity_weight,
            _identity_price,
            _identity_allowed,
        ) = identity_selection
        diagnostics["candidate"] = {
            "class_id": identity_evidence.class_id,
            "name": identity_evidence.name,
            "source": identity_evidence.source,
            "votes": identity_evidence.votes,
            "confidence": round(identity_evidence.confidence, 4),
            "side_votes": identity_evidence.side_votes,
            "side_confidence": round(identity_evidence.side_confidence, 4),
            "unit_weight_g": round(float(identity_weight), 1),
            "weight_residual_g": round(float(identity_residual), 1),
        }

        if normal_selection is None:
            diagnostics["accepted"] = True
            diagnostics["reason"] = "strong_identity_without_base_tolerance_match"
            return identity_selection, diagnostics

        (
            normal_residual,
            _,
            _,
            normal_evidence,
            _normal_product,
            normal_weight,
            _normal_price,
        ) = normal_selection
        diagnostics["replaced"] = {
            "class_id": normal_evidence.class_id,
            "name": normal_evidence.name,
            "source": normal_evidence.source,
            "votes": normal_evidence.votes,
            "confidence": round(normal_evidence.confidence, 4),
            "side_votes": normal_evidence.side_votes,
            "side_confidence": round(normal_evidence.side_confidence, 4),
            "unit_weight_g": round(float(normal_weight), 1),
            "weight_residual_g": round(float(normal_residual), 1),
        }

        if self._has_strong_single_bottle_identity_evidence(
            evidence=normal_evidence,
            unit_weight=normal_weight,
        ):
            diagnostics["reason"] = "current_single_identity_strong"
            return None, diagnostics

        residual_gap = max(0.0, float(identity_residual) - float(normal_residual))
        diagnostics["residual_gap_g"] = round(residual_gap, 1)
        if residual_gap > residual_gap_allowed:
            diagnostics["reason"] = "residual_gap_exceeds_allowance"
            return None, diagnostics

        diagnostics["accepted"] = True
        diagnostics["reason"] = "strong_identity_over_weak_weight_single"
        return identity_selection, diagnostics

    def _try_segment_weight_matching(
        self,
        *,
        vision_candidates: List[EnsembleResult],
        delta_weight: float,
        timestamp: float,
        active_products: Optional[List],
        trace_context: Optional[object],
        skip_channel_targets: bool = False,
    ) -> Optional[JudgmentResult]:
        """Match separable loadcell removal segments before aggregate weight."""
        channel_targets, regular_targets, vision_required_targets = (
            self._segment_weight_targets_from_trace(trace_context)
        )
        if not channel_targets and not regular_targets and not vision_required_targets:
            return None

        evidence_by_class = self._collect_detected_single_evidence(
            vision_candidates,
            trace_context,
        )
        if not skip_channel_targets and len(channel_targets) >= 2 and evidence_by_class:
            targets = channel_targets
            target_source = "channel_removal_segment_targets"
        elif len(regular_targets) >= 2:
            targets = regular_targets
            target_source = "removal_segment_targets"
        elif vision_required_targets and evidence_by_class:
            targets = vision_required_targets
            target_source = "vision_required_segment_targets"
        else:
            self._record_weight_diagnostics(
                trace_context,
                {
                    "segment_weight_matching": {
                        "accepted": False,
                        "reason": "insufficient_segment_targets",
                        "channel_target_count": len(channel_targets),
                        "regular_target_count": len(regular_targets),
                        "vision_required_target_count": len(vision_required_targets),
                        "has_evidence": bool(evidence_by_class),
                    }
                },
            )
            return None

        active_candidates = self._build_weight_only_candidates(active_products)
        diagnostics: dict[str, Any] = {
            "accepted": False,
            "target_source": target_source,
            "aggregate_delta_weight": round(float(delta_weight), 1),
            "max_items_per_segment": self._max_items_per_segment(),
            "segment_grip_limit": len(targets) * self._max_items_per_segment(),
            "targets": [
                {
                    "segment_index": target.segment_index,
                    "weight": round(float(target.weight), 1),
                    "source": target.source,
                    "evidence_required": target.evidence_required,
                }
                for target in targets
            ],
            "segment_options": [],
        }
        if not active_candidates:
            diagnostics["reason"] = "no_valid_active_products"
            self._record_weight_diagnostics(
                trace_context,
                {"segment_weight_matching": diagnostics},
            )
            return None

        options_by_segment: list[list[_SegmentMatchOption]] = []
        failed_segment_index: int | None = None
        for target in targets:
            options = self._segment_match_options_for_target(
                target=target,
                active_candidates=active_candidates,
                evidence_by_class=evidence_by_class,
            )
            diagnostics["segment_options"].append(
                {
                    "segment_index": target.segment_index,
                    "target_weight": round(float(target.weight), 1),
                    "option_count": len(
                        [
                            option
                            for option in options
                            if option.rejected_reason
                            != "count_exceeds_segment_grip_limit"
                        ]
                    ),
                    "rejected_option_count": len(
                        [option for option in options if option.rejected_reason]
                    ),
                    "top_options": [
                        self._segment_option_diagnostics(option)
                        for option in options[:5]
                    ],
                    "rejected_options": [
                        self._segment_option_diagnostics(option)
                        for option in options
                        if option.rejected_reason
                    ][:5],
                }
            )
            selectable_options = [
                option
                for option in options
                if option.rejected_reason != "count_exceeds_segment_grip_limit"
            ]
            if target_source == "channel_removal_segment_targets":
                selectable_options = [
                    option
                    for option in selectable_options
                    if option.option_kind == "single" and option.count == 1
                ]
            if not selectable_options:
                if failed_segment_index is None:
                    failed_segment_index = target.segment_index
                continue
            options_by_segment.append(selectable_options[:10])

        if failed_segment_index is not None:
            if len(targets) >= 3:
                candidate_override = self._try_candidate_supported_segment_override(
                    selections=[],
                    targets=targets,
                    active_candidates=active_candidates,
                    evidence_by_class=evidence_by_class,
                    delta_weight=delta_weight,
                    timestamp=timestamp,
                    target_source=target_source,
                    diagnostics=diagnostics,
                    trace_context=trace_context,
                    force_evaluate_aggregate=True,
                )
                if candidate_override is not None:
                    return candidate_override
            diagnostics["reason"] = "segment_without_valid_option"
            diagnostics["failed_segment_index"] = failed_segment_index
            if target_source == "channel_removal_segment_targets" and len(regular_targets) >= 2:
                self._record_weight_diagnostics(
                    trace_context,
                    {"channel_segment_weight_matching": dict(diagnostics)},
                )
                regular_result = self._try_segment_weight_matching(
                    vision_candidates=vision_candidates,
                    delta_weight=delta_weight,
                    timestamp=timestamp,
                    active_products=active_products,
                    trace_context=trace_context,
                    skip_channel_targets=True,
                )
                if regular_result is not None:
                    return regular_result
            self._record_weight_diagnostics(
                trace_context,
                {"segment_weight_matching": diagnostics},
            )
            return None

        best_selection = self._select_segment_match_options(options_by_segment)
        if not best_selection:
            diagnostics["reason"] = "stock_limit_prevents_segment_match"
            self._record_weight_diagnostics(
                trace_context,
                {"segment_weight_matching": diagnostics},
            )
            return None

        best_selection = self._prefer_same_weight_bottle_segment_repeat(
            options_by_segment=options_by_segment,
            targets=targets,
            current_selection=best_selection,
            diagnostics=diagnostics,
        )

        if target_source == "channel_removal_segment_targets":
            if not all(
                self._segment_option_supported(option)
                for option in best_selection
            ):
                diagnostics["reason"] = "channel_split_without_supported_evidence"
                diagnostics["rejected_selection"] = [
                    self._segment_option_diagnostics(option)
                    for option in best_selection
                ]
                if len(regular_targets) >= 2:
                    self._record_weight_diagnostics(
                        trace_context,
                        {"channel_segment_weight_matching": dict(diagnostics)},
                    )
                    regular_result = self._try_segment_weight_matching(
                        vision_candidates=vision_candidates,
                        delta_weight=delta_weight,
                        timestamp=timestamp,
                        active_products=active_products,
                        trace_context=trace_context,
                        skip_channel_targets=True,
                    )
                    if regular_result is not None:
                        return regular_result
                self._record_weight_diagnostics(
                    trace_context,
                    {"segment_weight_matching": diagnostics},
                )
                return None
            diagnostics["reason"] = "channel_supported_split_preferred"
            rejected_rescue = self._aggregate_rescue_rejected_by_channel_split(
                vision_candidates=vision_candidates,
                active_candidates=active_candidates,
                delta_weight=delta_weight,
            )
            if rejected_rescue is not None:
                diagnostics["rejected_aggregate_rescue"] = rejected_rescue
        else:
            candidate_override = self._try_candidate_supported_segment_override(
                selections=best_selection,
                targets=targets,
                active_candidates=active_candidates,
                evidence_by_class=evidence_by_class,
                delta_weight=delta_weight,
                timestamp=timestamp,
                target_source=target_source,
                diagnostics=diagnostics,
                trace_context=trace_context,
            )
            if candidate_override is not None:
                return candidate_override

        return self._create_segment_weight_result(
            selections=best_selection,
            delta_weight=delta_weight,
            timestamp=timestamp,
            evidence_by_class=evidence_by_class,
            target_source=target_source,
            diagnostics=diagnostics,
            trace_context=trace_context,
        )

    def _aggregate_rescue_rejected_by_channel_split(
        self,
        *,
        vision_candidates: List[EnsembleResult],
        active_candidates: list[_WeightOnlyCandidate],
        delta_weight: float,
    ) -> Optional[dict[str, Any]]:
        active_by_class = {
            int(candidate.product_id): candidate
            for candidate in active_candidates
        }
        target_weight = abs(float(delta_weight))
        tolerance = max(
            float(config.weight.tolerance_grams),
            float(config.weight.rescue_tolerance_grams),
        )
        rejected: list[dict[str, Any]] = []
        for candidate in vision_candidates:
            source = str(getattr(candidate, "source", "vision") or "vision")
            if source not in {"threshold_rescue", "roi_rescue"}:
                continue
            product = active_by_class.get(int(candidate.class_id))
            if product is None or product.weight <= 0:
                continue
            residual = abs(target_weight - float(product.weight))
            if residual > tolerance:
                continue
            rejected.append(
                {
                    "class_id": int(candidate.class_id),
                    "name": str(candidate.class_name),
                    "source": source,
                    "unit_weight": round(float(product.weight), 1),
                    "residual": round(float(residual), 1),
                    "confidence": round(float(candidate.combined_confidence), 4),
                    "reason": "channel_supported_split_preferred",
                }
            )
        if not rejected:
            return None
        return sorted(
            rejected,
            key=lambda entry: (
                float(entry["residual"]),
                -float(entry["confidence"]),
                int(entry["class_id"]),
            ),
        )[0]

    def _segment_weight_targets_from_trace(
        self,
        trace_context: Optional[object],
    ) -> tuple[
        list[_SegmentWeightTarget],
        list[_SegmentWeightTarget],
        list[_SegmentWeightTarget],
    ]:
        loadcell = getattr(trace_context, "loadcell", {}) if trace_context else {}
        if not isinstance(loadcell, dict):
            return [], [], []

        def parse_targets(
            key: str,
            *,
            default_evidence_required: bool,
        ) -> list[_SegmentWeightTarget]:
            parsed: list[_SegmentWeightTarget] = []
            for fallback_index, entry in enumerate(loadcell.get(key) or []):
                if not isinstance(entry, dict):
                    continue
                weight = abs(self._coerce_float(entry.get("weight", 0.0)))
                if weight < self.min_weight_change:
                    continue
                segment_index = self._coerce_int(entry.get("segment_index"))
                if "segment_index" not in entry:
                    indices = entry.get("segment_indices") or []
                    if indices:
                        segment_index = self._coerce_int(indices[0])
                    else:
                        segment_index = fallback_index
                parsed.append(
                    _SegmentWeightTarget(
                        weight=weight,
                        source=str(entry.get("source") or key),
                        segment_index=segment_index,
                        evidence_required=bool(
                            entry.get(
                                "evidence_required",
                                default_evidence_required,
                            )
                        ),
                    )
                )
            return parsed

        return (
            parse_targets(
                "channel_removal_segment_targets",
                default_evidence_required=True,
            ),
            parse_targets("removal_segment_targets", default_evidence_required=False),
            parse_targets(
                "vision_required_segment_targets",
                default_evidence_required=True,
            ),
        )

    @staticmethod
    def _max_items_per_segment() -> int:
        return max(1, int(config.weight.max_items_per_segment))

    def _segment_grip_limit_from_trace(
        self,
        trace_context: Optional[object],
    ) -> Optional[int]:
        channel_targets, regular_targets, _ = self._segment_weight_targets_from_trace(
            trace_context
        )
        if len(channel_targets) >= 2:
            return len(channel_targets) * self._max_items_per_segment()
        if not regular_targets:
            return None
        return len(regular_targets) * self._max_items_per_segment()

    def _strict_max_items_for_trace(
        self,
        trace_context: Optional[object],
    ) -> int:
        segment_grip_limit = self._segment_grip_limit_from_trace(trace_context)
        if segment_grip_limit is not None:
            return segment_grip_limit
        return int(config.weight.max_combination_items)

    def _segment_match_options_for_target(
        self,
        *,
        target: _SegmentWeightTarget,
        active_candidates: list[_WeightOnlyCandidate],
        evidence_by_class: dict[int, _DetectedSingleEvidence],
    ) -> list[_SegmentMatchOption]:
        options: list[_SegmentMatchOption] = []
        strict_tolerance = float(config.weight.tolerance_grams)
        per_item_tolerance = float(config.weight.same_product_count_tolerance_grams)
        segment_count_cap = self._max_items_per_segment()
        max_count_cap = max(
            1,
            min(
                int(config.weight.max_count_per_item),
                int(config.weight.same_product_max_count),
            ),
        )

        for product in active_candidates:
            if product.weight <= 0 or product.stock <= 0:
                continue
            evidence = evidence_by_class.get(int(product.product_id))
            evidence_score = self._forced_evidence_score(
                product.product_id,
                evidence_by_class,
            )
            if target.evidence_required and evidence_score <= 0.0:
                continue
            max_count = min(product.stock, max_count_cap)
            for count in range(1, max_count + 1):
                expected_weight = product.weight * count
                allowed_residual = max(strict_tolerance, per_item_tolerance * count)
                if count == 1 and evidence_score > 0.0:
                    allowed_residual = max(
                        allowed_residual,
                        strict_tolerance + per_item_tolerance,
                    )
                residual = abs(target.weight - expected_weight)
                if residual > allowed_residual:
                    continue
                rejected_reason = (
                    "count_exceeds_segment_grip_limit"
                    if count > segment_count_cap
                    else None
                )
                weight_score = max(
                    0.0,
                    1.0 - residual / max(allowed_residual, 0.001),
                )
                evidence_confidence = (
                    max(
                        float(evidence.confidence),
                        float(evidence.top_confidence),
                        float(evidence.side_confidence),
                    )
                    if evidence is not None
                    else 0.0
                )
                weight_tight = (
                    count == 1
                    and evidence_score > 0.0
                    and evidence_confidence >= 0.18
                    and residual <= strict_tolerance
                )
                item = self._segment_item_for_product(
                    product,
                    evidence,
                    count=count,
                    weak_companion=False,
                    weight_tight=weight_tight,
                )
                selection_tier, selection_reason = self._single_segment_selection(
                    item,
                    residual=residual,
                    allowed_residual=allowed_residual,
                )
                if rejected_reason:
                    selection_tier = 99
                    selection_reason = rejected_reason
                options.append(
                    _SegmentMatchOption(
                        target=target,
                        product=product,
                        count=count,
                        expected_weight=expected_weight,
                        residual=residual,
                        allowed_residual=allowed_residual,
                        weight_score=weight_score,
                        evidence_score=evidence_score,
                        evidence_source=evidence.source if evidence else None,
                        evidence_trusted=bool(evidence.trusted) if evidence else False,
                        evidence_strong=bool(evidence.strong) if evidence else False,
                        evidence_motion_gate_passed=(
                            bool(evidence.motion_gate_passed) if evidence else True
                        ),
                        stage_score=(
                            float(evidence.stage_score) if evidence else 0.0
                        ),
                        evidence_confidence=(
                            float(evidence.confidence) if evidence else 0.0
                        ),
                        evidence_votes=int(evidence.votes) if evidence else 0,
                        top_confidence=(
                            float(evidence.top_confidence) if evidence else 0.0
                        ),
                        side_confidence=(
                            float(evidence.side_confidence) if evidence else 0.0
                        ),
                        top_votes=int(evidence.top_votes) if evidence else 0,
                        side_votes=int(evidence.side_votes) if evidence else 0,
                        score_reason=evidence.score_reason if evidence else "",
                        items=(item,),
                        option_kind="single",
                        selection_tier=selection_tier,
                        selection_rank=self._segment_evidence_selection_rank(
                            evidence
                        ),
                        selection_reason=selection_reason,
                        rejected_reason=rejected_reason,
                    )
                )

        options.extend(
            self._compound_segment_match_options_for_target(
                target=target,
                active_candidates=active_candidates,
                evidence_by_class=evidence_by_class,
                strict_tolerance=strict_tolerance,
                per_item_tolerance=per_item_tolerance,
            )
        )

        has_preferred_alternative = any(
            option.selection_tier <= 3
            and not self._segment_option_is_small_repeat(option)
            for option in options
        )
        if has_preferred_alternative:
            options = [
                replace(
                    option,
                    rejected_reason="trusted_or_single_item_segment_preferred",
                )
                if (
                    option.rejected_reason is None
                    and self._segment_option_is_small_repeat(option)
                )
                else option
                for option in options
            ]

        return sorted(
            options,
            key=lambda option: (
                option.selection_tier,
                option.selection_rank if option.option_kind == "compound" else 9999,
                option.residual / max(option.allowed_residual, 0.001),
                option.selection_rank,
                option.count,
                -option.evidence_score,
            ),
        )

    def _compound_segment_match_options_for_target(
        self,
        *,
        target: _SegmentWeightTarget,
        active_candidates: list[_WeightOnlyCandidate],
        evidence_by_class: dict[int, _DetectedSingleEvidence],
        strict_tolerance: float,
        per_item_tolerance: float,
    ) -> list[_SegmentMatchOption]:
        options: list[_SegmentMatchOption] = []
        pool: list[_SegmentMatchItem] = []
        if self._max_items_per_segment() < 2:
            return options
        for product in active_candidates:
            if product.weight <= 0 or product.stock <= 0:
                continue
            evidence = evidence_by_class.get(int(product.product_id))
            if evidence is None:
                continue
            weak_companion = self._segment_product_has_weak_companion_evidence(
                product,
                evidence,
            )
            item = self._segment_item_for_product(
                product,
                evidence,
                count=1,
                weak_companion=weak_companion,
            )
            if not self._segment_item_supported(item):
                continue
            pool.append(item)

        for item_count in range(2, min(3, self._max_items_per_segment()) + 1):
            for combo in combinations(pool, item_count):
                trusted_items = [
                    item
                    for item in combo
                    if item.evidence_trusted
                    or self._segment_evidence_selection_rank(
                        evidence_by_class.get(int(item.product.product_id))
                    )
                    < 9999
                ]
                if not trusted_items:
                    continue
                if sum(1 for item in combo if item.weak_companion) > 1:
                    continue

                expected_weight = sum(item.product.weight for item in combo)
                allowed_residual = strict_tolerance + per_item_tolerance * item_count
                residual = abs(target.weight - expected_weight)
                if residual > allowed_residual:
                    continue
                ordered_items = tuple(
                    sorted(
                        combo,
                        key=lambda item: (
                            0
                            if item.evidence_trusted
                            or self._segment_evidence_selection_rank(
                                evidence_by_class.get(int(item.product.product_id))
                            )
                            < 9999
                            else 1,
                            self._segment_evidence_selection_rank(
                                evidence_by_class.get(int(item.product.product_id))
                            ),
                            item.product.product_id,
                        ),
                    )
                )
                weight_score = max(
                    0.0,
                    1.0 - residual / max(allowed_residual, 0.001),
                )
                evidence_score = sum(item.evidence_score for item in ordered_items)
                primary = ordered_items[0].product
                options.append(
                    _SegmentMatchOption(
                        target=target,
                        product=primary,
                        count=1,
                        expected_weight=expected_weight,
                        residual=residual,
                        allowed_residual=allowed_residual,
                        weight_score=weight_score,
                        evidence_score=evidence_score,
                        evidence_source="+".join(
                            sorted(
                                {
                                    str(item.evidence_source)
                                    for item in ordered_items
                                    if item.evidence_source
                                }
                            )
                        )
                        or None,
                        evidence_trusted=any(
                            item.evidence_trusted for item in ordered_items
                        ),
                        evidence_strong=any(
                            item.evidence_strong for item in ordered_items
                        ),
                        evidence_motion_gate_passed=all(
                            item.evidence_motion_gate_passed
                            for item in ordered_items
                        ),
                        stage_score=sum(item.stage_score for item in ordered_items),
                        evidence_confidence=max(
                            item.evidence_confidence for item in ordered_items
                        ),
                        evidence_votes=sum(item.evidence_votes for item in ordered_items),
                        top_confidence=max(item.top_confidence for item in ordered_items),
                        side_confidence=max(item.side_confidence for item in ordered_items),
                        top_votes=sum(item.top_votes for item in ordered_items),
                        side_votes=sum(item.side_votes for item in ordered_items),
                        score_reason="+".join(
                            item.score_reason
                            for item in ordered_items
                            if item.score_reason
                        ),
                        items=ordered_items,
                        option_kind="compound",
                        selection_tier=0,
                        selection_rank=min(
                            self._segment_evidence_selection_rank(
                                evidence_by_class.get(int(item.product.product_id))
                            )
                            for item in trusted_items
                        ),
                        selection_reason="trusted_compound_segment_split",
                    )
                )
        return options

    def _segment_item_for_product(
        self,
        product: _WeightOnlyCandidate,
        evidence: Optional[_DetectedSingleEvidence],
        *,
        count: int,
        weak_companion: bool = False,
        weight_tight: bool = False,
    ) -> _SegmentMatchItem:
        return _SegmentMatchItem(
            product=product,
            count=count,
            evidence_score=self._forced_evidence_score(
                product.product_id,
                {int(product.product_id): evidence} if evidence is not None else {},
            ),
            evidence_source=evidence.source if evidence else None,
            evidence_trusted=bool(evidence.trusted) if evidence else False,
            evidence_strong=bool(evidence.strong) if evidence else False,
            evidence_motion_gate_passed=(
                bool(evidence.motion_gate_passed) if evidence else True
            ),
            stage_score=float(evidence.stage_score) if evidence else 0.0,
            evidence_confidence=float(evidence.confidence) if evidence else 0.0,
            evidence_votes=int(evidence.votes) if evidence else 0,
            top_confidence=float(evidence.top_confidence) if evidence else 0.0,
            side_confidence=float(evidence.side_confidence) if evidence else 0.0,
            top_votes=int(evidence.top_votes) if evidence else 0,
            side_votes=int(evidence.side_votes) if evidence else 0,
            score_reason=evidence.score_reason if evidence else "",
            weak_companion=weak_companion,
            weight_tight=weight_tight,
        )

    @staticmethod
    def _segment_evidence_selection_rank(
        evidence: Optional[_DetectedSingleEvidence],
    ) -> int:
        if evidence is None or evidence.rank is None:
            return 9999
        rank = ProductDecisionEngine._coerce_int(evidence.rank)
        return rank if rank > 0 else 9999

    @staticmethod
    def _segment_product_has_weak_companion_evidence(
        product: _WeightOnlyCandidate,
        evidence: Optional[_DetectedSingleEvidence],
    ) -> bool:
        if evidence is None or product.stock <= 0 or product.weight < 200.0:
            return False
        votes = max(
            int(evidence.votes),
            int(evidence.top_votes),
            int(evidence.side_votes),
        )
        confidence = max(
            float(evidence.confidence),
            float(evidence.top_confidence),
            float(evidence.side_confidence),
        )
        return votes >= 20 and confidence >= 0.08

    def _single_segment_selection(
        self,
        item: _SegmentMatchItem,
        *,
        residual: float,
        allowed_residual: float,
    ) -> tuple[int, str]:
        if self._segment_item_is_unsupported_small_repeat_fragment(item):
            return 99, "unsupported_small_repeat_fragment"
        if self._segment_option_is_small_repeat_item(item):
            return 4, "small_item_repeated_count_segment_match"
        strict_tolerance = float(config.weight.tolerance_grams)
        if item.weight_tight:
            return 1, "weight_tight_single_segment_match"
        if item.evidence_trusted or item.evidence_strong:
            if residual > strict_tolerance:
                return 3, "trusted_loose_single_segment_match"
            return 1, "trusted_single_segment_match"
        if self._segment_item_supported(item):
            if residual > strict_tolerance:
                return 4, "supported_loose_single_segment_match"
            return 2, "supported_single_segment_match"
        if (
            item.count == 1
            and item.product.weight >= 200.0
            and residual <= strict_tolerance
        ):
            return 3, "weight_tight_weak_single_segment_match"
        if item.count > 1:
            return 5, "active_only_repeated_count_segment_match"
        if residual <= strict_tolerance:
            return 5, "active_only_single_segment_match"
        return 5, "active_only_segment_match"

    @staticmethod
    def _segment_option_diagnostics(
        option: _SegmentMatchOption,
    ) -> dict[str, Any]:
        diagnostics = {
            "class_id": option.product.product_id,
            "name": option.product.name,
            "count": option.count,
            "unit_weight": round(float(option.product.weight), 1),
            "expected_weight": round(float(option.expected_weight), 1),
            "residual": round(float(option.residual), 1),
            "allowed_residual": round(float(option.allowed_residual), 1),
            "evidence_score": round(float(option.evidence_score), 4),
            "evidence_supported": ProductDecisionEngine._segment_option_supported(
                option
            ),
            "option_kind": option.option_kind,
            "selection_tier": int(option.selection_tier),
            "selection_reason": option.selection_reason,
        }
        if option.rejected_reason:
            diagnostics["rejected_reason"] = option.rejected_reason
        items = ProductDecisionEngine._segment_option_items(option)
        if len(items) > 1:
            diagnostics["items"] = [
                ProductDecisionEngine._segment_item_diagnostics(item)
                for item in items
            ]
        if option.evidence_score > 0.0:
            diagnostics.update(
                {
                    "evidence_source": option.evidence_source,
                    "evidence_trusted": bool(option.evidence_trusted),
                    "evidence_strong": bool(option.evidence_strong),
                    "evidence_motion_gate_passed": bool(
                        option.evidence_motion_gate_passed
                    ),
                    "stage_score": round(float(option.stage_score), 4),
                    "evidence_confidence": round(float(option.evidence_confidence), 4),
                    "evidence_votes": int(option.evidence_votes),
                    "side_confidence": round(float(option.side_confidence), 4),
                    "top_confidence": round(float(option.top_confidence), 4),
                    "side_votes": int(option.side_votes),
                    "top_votes": int(option.top_votes),
                    "score_reason": option.score_reason,
                }
            )
        return diagnostics

    @staticmethod
    def _segment_item_diagnostics(item: _SegmentMatchItem) -> dict[str, Any]:
        return {
            "class_id": item.product.product_id,
            "name": item.product.name,
            "count": item.count,
            "unit_weight": round(float(item.product.weight), 1),
            "evidence_score": round(float(item.evidence_score), 4),
            "evidence_source": item.evidence_source,
            "evidence_trusted": bool(item.evidence_trusted),
            "evidence_strong": bool(item.evidence_strong),
            "evidence_confidence": round(float(item.evidence_confidence), 4),
            "evidence_votes": int(item.evidence_votes),
            "side_confidence": round(float(item.side_confidence), 4),
            "top_confidence": round(float(item.top_confidence), 4),
            "side_votes": int(item.side_votes),
            "top_votes": int(item.top_votes),
            "weak_companion": bool(item.weak_companion),
            "weight_tight": bool(item.weight_tight),
            "evidence_supported": ProductDecisionEngine._segment_item_supported(
                item
            ),
        }

    @staticmethod
    def _segment_option_items(
        option: _SegmentMatchOption,
    ) -> tuple[_SegmentMatchItem, ...]:
        if option.items:
            return option.items
        return (
            _SegmentMatchItem(
                product=option.product,
                count=option.count,
                evidence_score=option.evidence_score,
                evidence_source=option.evidence_source,
                evidence_trusted=option.evidence_trusted,
                evidence_strong=option.evidence_strong,
                evidence_motion_gate_passed=option.evidence_motion_gate_passed,
                stage_score=option.stage_score,
                evidence_confidence=option.evidence_confidence,
                evidence_votes=option.evidence_votes,
                top_confidence=option.top_confidence,
                side_confidence=option.side_confidence,
                top_votes=option.top_votes,
                side_votes=option.side_votes,
                score_reason=option.score_reason,
            ),
        )

    @staticmethod
    def _segment_item_supported(item: _SegmentMatchItem) -> bool:
        if item.evidence_trusted or item.evidence_strong:
            return True
        if item.weak_companion or item.weight_tight:
            return True
        if item.evidence_score <= 0.0:
            return False
        votes = max(
            int(item.evidence_votes),
            int(item.top_votes),
            int(item.side_votes),
        )
        confidence = max(
            float(item.evidence_confidence),
            float(item.top_confidence),
            float(item.side_confidence),
        )
        min_votes = max(0, int(config.weight.detected_single_fallback_min_votes))
        return votes >= min_votes and confidence >= 0.18

    @staticmethod
    def _segment_option_supported(option: _SegmentMatchOption) -> bool:
        return all(
            ProductDecisionEngine._segment_item_supported(item)
            for item in ProductDecisionEngine._segment_option_items(option)
        )

    @staticmethod
    def _segment_option_is_small_repeat_item(item: _SegmentMatchItem) -> bool:
        return item.product.weight < 200.0 and item.count >= 3

    @staticmethod
    def _segment_item_is_unsupported_small_repeat_fragment(
        item: _SegmentMatchItem,
    ) -> bool:
        if item.product.weight >= 200.0 or item.count < 2:
            return False
        if ProductDecisionEngine._source_includes_regular_vision(item.evidence_source):
            return False
        if not item.evidence_motion_gate_passed:
            return True
        if item.evidence_strong:
            return False

        votes = max(
            int(item.evidence_votes),
            int(item.top_votes),
            int(item.side_votes),
        )
        confidence = max(
            float(item.evidence_confidence),
            float(item.top_confidence),
            float(item.side_confidence),
        )
        return not (votes >= 20 and confidence >= 0.30)

    @staticmethod
    def _segment_option_is_small_repeat(option: _SegmentMatchOption) -> bool:
        return any(
            ProductDecisionEngine._segment_option_is_small_repeat_item(item)
            for item in ProductDecisionEngine._segment_option_items(option)
        )

    def _select_segment_match_options(
        self,
        options_by_segment: list[list[_SegmentMatchOption]],
    ) -> list[_SegmentMatchOption]:
        best_selection: list[_SegmentMatchOption] = []
        best_rank: tuple | None = None
        has_stage_or_diagnostic_evidence = any(
            item.evidence_score > 0.0
            and bool(
                set(str(item.evidence_source or "").split("+"))
                & {"stage_counts", "diagnostic"}
            )
            for options in options_by_segment
            for option in options
            for item in self._segment_option_items(option)
        )

        def search(
            index: int,
            selected: list[_SegmentMatchOption],
            used_stock: dict[int, int],
            residual_sum: float,
            score_sum: float,
        ) -> None:
            nonlocal best_selection, best_rank
            if index >= len(options_by_segment):
                selected_items = [
                    item
                    for option in selected
                    for item in self._segment_option_items(option)
                ]
                kind_count = len({item.product.product_id for item in selected_items})
                total_units = sum(item.count for item in selected_items)
                selection_tier_sum = sum(option.selection_tier for option in selected)
                compound_rank_sum = sum(
                    option.selection_rank
                    for option in selected
                    if option.option_kind == "compound"
                )
                if has_stage_or_diagnostic_evidence:
                    active_only_units = sum(
                        item.count
                        for item in selected_items
                        if not self._segment_item_supported(item)
                    )
                    evidence_sum = sum(item.evidence_score for item in selected_items)
                    strong_or_trusted_units = sum(
                        item.count
                        for item in selected_items
                        if item.evidence_strong or item.evidence_trusted
                    )
                    weak_evidence_units = total_units - strong_or_trusted_units
                    rank = (
                        selection_tier_sum,
                        active_only_units,
                        weak_evidence_units,
                        compound_rank_sum,
                        round(float(residual_sum), 4),
                        kind_count,
                        total_units,
                        -round(float(evidence_sum), 4),
                        -round(float(score_sum), 4),
                    )
                else:
                    rank = (
                        selection_tier_sum,
                        compound_rank_sum,
                        round(float(residual_sum), 4),
                        kind_count,
                        total_units,
                        -round(float(score_sum), 4),
                    )
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_selection = list(selected)
                return

            previous_product_id = (
                self._segment_option_items(selected[-1])[-1].product.product_id
                if selected
                else None
            )
            for option in options_by_segment[index]:
                increments: dict[int, int] = {}
                stock_by_product: dict[int, int] = {}
                for item in self._segment_option_items(option):
                    product_id = item.product.product_id
                    increments[product_id] = increments.get(product_id, 0) + item.count
                    stock_by_product[product_id] = item.product.stock
                if any(
                    used_stock.get(product_id, 0) + count
                    > stock_by_product[product_id]
                    for product_id, count in increments.items()
                ):
                    continue
                continuity_bonus = (
                    0.15
                    if previous_product_id
                    in {item.product.product_id for item in self._segment_option_items(option)}
                    else 0.0
                )
                option_score = (
                    option.weight_score * 2.0
                    + min(option.evidence_score, 1.0) * 0.5
                    + continuity_bonus
                    - sum(item.count for item in self._segment_option_items(option)) * 0.02
                )
                for product_id, count in increments.items():
                    used_stock[product_id] = used_stock.get(product_id, 0) + count
                selected.append(option)
                search(
                    index + 1,
                    selected,
                    used_stock,
                    residual_sum + option.residual,
                    score_sum + option_score,
                )
                selected.pop()
                for product_id, count in increments.items():
                    previous_used = used_stock.get(product_id, 0)
                    if previous_used <= count:
                        used_stock.pop(product_id, None)
                    else:
                        used_stock[product_id] = previous_used - count

        search(0, [], {}, 0.0, 0.0)
        return best_selection

    def _prefer_same_weight_bottle_segment_repeat(
        self,
        *,
        options_by_segment: list[list[_SegmentMatchOption]],
        targets: list[_SegmentWeightTarget],
        current_selection: list[_SegmentMatchOption],
        diagnostics: dict[str, Any],
    ) -> list[_SegmentMatchOption]:
        if len(targets) < 2 or len(options_by_segment) != len(targets):
            return current_selection

        current_residual = sum(option.residual for option in current_selection)
        reuse_rejections = self._repeated_segment_reuse_rejections(current_selection)
        repeat_candidates: list[dict[str, Any]] = []
        product_ids = {
            int(option.product.product_id)
            for options in options_by_segment
            for option in options
            if self._same_weight_bottle_segment_option(option)
        }

        for product_id in sorted(product_ids):
            per_segment_options: list[_SegmentMatchOption] = []
            for options in options_by_segment:
                matches = [
                    option
                    for option in options
                    if (
                        int(option.product.product_id) == product_id
                        and self._same_weight_bottle_segment_option(option)
                    )
                ]
                if not matches:
                    per_segment_options = []
                    break
                per_segment_options.append(
                    sorted(
                        matches,
                        key=lambda option: (
                            option.residual,
                            option.selection_tier,
                            -option.evidence_score,
                        ),
                    )[0]
                )
            if len(per_segment_options) != len(options_by_segment):
                continue

            support = self._same_weight_bottle_repeat_support(per_segment_options)
            total_residual = sum(option.residual for option in per_segment_options)
            repeat_candidates.append(
                {
                    "class_id": product_id,
                    "name": per_segment_options[0].product.name,
                    "count": len(per_segment_options),
                    "unit_weight": round(float(per_segment_options[0].product.weight), 1),
                    "total_residual": round(float(total_residual), 1),
                    "coverage_rank": support["coverage_rank"],
                    "supported": support["supported"],
                    "top_votes": support["top_votes"],
                    "side_votes": support["side_votes"],
                    "votes": support["votes"],
                    "confidence": round(float(support["confidence"]), 4),
                    "reason": (
                        "accepted_candidate"
                        if support["supported"]
                        else "insufficient_repeated_segment_evidence"
                    ),
                    "options": [
                        {
                            "segment_index": option.target.segment_index,
                            "target_weight": round(float(option.target.weight), 1),
                            "residual": round(float(option.residual), 1),
                            "selection_tier": int(option.selection_tier),
                            "selection_reason": option.selection_reason,
                        }
                        for option in per_segment_options
                    ],
                    "_options": per_segment_options,
                    "_total_residual": total_residual,
                }
            )

        accepted = [
            candidate for candidate in repeat_candidates if bool(candidate["supported"])
        ]
        selected_candidate: dict[str, Any] | None = None
        if accepted:
            selected_candidate = sorted(
                accepted,
                key=lambda candidate: (
                    int(candidate["coverage_rank"]),
                    round(float(candidate["_total_residual"]), 4),
                    -int(candidate["votes"]),
                    -float(candidate["confidence"]),
                ),
            )[0]

        current_product_ids = {
            item.product.product_id
            for option in current_selection
            for item in self._segment_option_items(option)
        }
        should_replace = (
            selected_candidate is not None
            and current_product_ids != {int(selected_candidate["class_id"])}
            and float(selected_candidate["_total_residual"])
            <= current_residual + float(config.weight.same_product_count_tolerance_grams)
        )

        if reuse_rejections:
            diagnostics["repeated_segment_reuse_guard"] = {
                "accepted": False,
                "rejected": reuse_rejections,
            }

        diagnostics["same_weight_bottle_collision"] = {
            "accepted": bool(should_replace),
            "target_count": len(targets),
            "current_total_residual": round(float(current_residual), 1),
            "candidates": [
                {
                    key: value
                    for key, value in candidate.items()
                    if not key.startswith("_")
                }
                for candidate in repeat_candidates
            ],
        }

        if not should_replace or selected_candidate is None:
            return current_selection

        selected_options = [
            replace(
                option,
                selection_tier=min(option.selection_tier, 1),
                selection_reason="same_weight_bottle_repeat_preferred",
                rejected_reason=None,
            )
            for option in selected_candidate["_options"]
        ]
        diagnostics["same_weight_bottle_collision"].update(
            {
                "accepted": True,
                "reason": "same_product_bottle_repeat_preferred",
                "selected": {
                    key: value
                    for key, value in selected_candidate.items()
                    if not key.startswith("_")
                },
                "replaced_products": [
                    {
                        "class_id": item.product.product_id,
                        "name": item.product.name,
                        "count": item.count,
                    }
                    for option in current_selection
                    for item in self._segment_option_items(option)
                ],
            }
        )
        if reuse_rejections:
            diagnostics["repeated_segment_reuse_guard"]["accepted"] = True
        return selected_options

    @staticmethod
    def _same_weight_bottle_segment_option(option: _SegmentMatchOption) -> bool:
        return (
            option.option_kind == "single"
            and option.count == 1
            and option.rejected_reason is None
            and 450.0 <= float(option.product.weight) <= 560.0
        )

    def _same_weight_bottle_repeat_support(
        self,
        options: list[_SegmentMatchOption],
    ) -> dict[str, Any]:
        min_votes = max(0, int(config.weight.detected_single_fallback_min_votes))
        top_votes = max((int(option.top_votes) for option in options), default=0)
        side_votes = max((int(option.side_votes) for option in options), default=0)
        votes = max((int(option.evidence_votes) for option in options), default=0)
        confidence = max(
            (float(option.evidence_confidence) for option in options),
            default=0.0,
        )
        two_camera = (
            top_votes >= min_votes
            and side_votes >= min_votes
            and confidence >= 0.08
        )
        very_strong_single_camera = (
            votes >= max(1, min_votes) * max(1, len(options)) * 20
            and confidence >= 0.30
        )
        coverage_rank = 0 if two_camera else 1 if side_votes >= min_votes else 2
        return {
            "supported": two_camera or very_strong_single_camera,
            "coverage_rank": coverage_rank,
            "top_votes": top_votes,
            "side_votes": side_votes,
            "votes": votes,
            "confidence": confidence,
        }

    def _repeated_segment_reuse_rejections(
        self,
        selection: list[_SegmentMatchOption],
    ) -> list[dict[str, Any]]:
        by_product: dict[int, list[_SegmentMatchOption]] = {}
        for option in selection:
            for item in self._segment_option_items(option):
                if item.count != 1:
                    continue
                by_product.setdefault(item.product.product_id, []).append(option)

        rejected: list[dict[str, Any]] = []
        for product_id, options in by_product.items():
            if len(options) < 2:
                continue
            support = self._same_weight_bottle_repeat_support(options)
            if bool(support["supported"]):
                continue
            rejected.append(
                {
                    "class_id": product_id,
                    "name": options[0].product.name,
                    "count": len(options),
                    "reason": "repeated_segment_evidence_insufficient",
                    "top_votes": support["top_votes"],
                    "side_votes": support["side_votes"],
                    "votes": support["votes"],
                    "confidence": round(float(support["confidence"]), 4),
                }
            )
        return rejected

    def _try_candidate_supported_segment_override(
        self,
        *,
        selections: list[_SegmentMatchOption],
        targets: list[_SegmentWeightTarget],
        active_candidates: list[_WeightOnlyCandidate],
        evidence_by_class: dict[int, _DetectedSingleEvidence],
        delta_weight: float,
        timestamp: float,
        target_source: str,
        diagnostics: dict[str, Any],
        trace_context: Optional[object],
        force_evaluate_aggregate: bool = False,
    ) -> Optional[JudgmentResult]:
        """Prefer a trusted detected repeated product over active-only segments."""
        selection_summary = self._segment_selection_summary(selections)
        selected_has_evidence = any(
            item.evidence_score > 0.0
            for option in selections
            for item in self._segment_option_items(option)
        )
        over_fragmented_segments = len(targets) >= 3
        selected_has_stage_or_diagnostic = any(
            item.evidence_score > 0.0
            and bool(
                set(str(item.evidence_source or "").split("+"))
                & {"stage_counts", "diagnostic"}
            )
            for option in selections
            for item in self._segment_option_items(option)
        )
        should_evaluate_aggregate = (
            force_evaluate_aggregate
            or bool(selection_summary["has_active_only_product"])
            or not bool(selection_summary["all_products_have_evidence"])
            or (
                over_fragmented_segments
                and (
                    int(selection_summary["kind_count"]) > 1
                    or selected_has_stage_or_diagnostic
                )
            )
        )
        target_total = sum(target.weight for target in targets)
        strict_tolerance = float(config.weight.tolerance_grams)
        per_item_tolerance = float(config.weight.same_product_count_tolerance_grams)
        segment_grip_limit = len(targets) * self._max_items_per_segment()
        max_count_cap = max(
            1,
            min(
                int(config.weight.max_count_per_item),
                int(config.weight.same_product_max_count),
                segment_grip_limit,
            ),
        )
        override_diagnostics: dict[str, Any] = {
            "accepted": False,
            "target_weight": round(float(target_total), 1),
            "selected_segment_has_evidence": selected_has_evidence,
            "selected_segment_summary": selection_summary,
            "segment_grip_limit": segment_grip_limit,
            "candidates": [],
        }
        selected_has_unsupported_small_fragments = (
            self._segment_selection_has_unsupported_small_fragments(selections)
        )
        override_diagnostics["selected_has_unsupported_small_fragments"] = (
            selected_has_unsupported_small_fragments
        )

        accepted: list[dict[str, Any]] = []
        for product in active_candidates:
            evidence = evidence_by_class.get(int(product.product_id))
            candidate_diag: dict[str, Any] = {
                "class_id": product.product_id,
                "name": product.name,
                "unit_weight": round(float(product.weight), 1),
            }
            if evidence is None:
                candidate_diag["evidence_score"] = 0.0
                candidate_diag["reason"] = "no_candidate_stage_or_diagnostic_evidence"
                override_diagnostics["candidates"].append(candidate_diag)
                continue
            if not self._has_strong_aggregate_evidence(evidence):
                candidate_diag["evidence_score"] = round(
                    float(
                        self._forced_evidence_score(
                            product.product_id,
                            evidence_by_class,
                        )
                    ),
                    4,
                )
                candidate_diag["evidence_source"] = evidence.source
                candidate_diag["strong_evidence"] = bool(evidence.strong)
                candidate_diag["trusted_evidence"] = bool(evidence.trusted)
                candidate_diag["motion_gate_passed"] = bool(evidence.motion_gate_passed)
                candidate_diag["rank"] = evidence.rank
                candidate_diag["reason"] = "insufficient_aggregate_evidence"
                override_diagnostics["candidates"].append(candidate_diag)
                continue
            evidence_score = self._forced_evidence_score(
                product.product_id,
                evidence_by_class,
            )
            candidate_diag["evidence_score"] = round(float(evidence_score), 4)
            candidate_diag["evidence_source"] = evidence.source
            candidate_diag["strong_evidence"] = bool(evidence.strong)
            candidate_diag["trusted_evidence"] = bool(evidence.trusted)
            candidate_diag["motion_gate_passed"] = bool(evidence.motion_gate_passed)
            candidate_diag["rank"] = evidence.rank
            if product.weight <= 0 or product.stock <= 0:
                candidate_diag["reason"] = "invalid_weight_or_stock"
                override_diagnostics["candidates"].append(candidate_diag)
                continue

            count = int(round(target_total / product.weight))
            candidate_diag["nearest_count"] = count
            candidate_diag["segment_grip_limit"] = segment_grip_limit
            if count < 2:
                candidate_diag["reason"] = "single_count"
                override_diagnostics["candidates"].append(candidate_diag)
                continue
            if count > segment_grip_limit:
                candidate_diag["reason"] = "count_exceeds_segment_grip_limit"
                override_diagnostics["candidates"].append(candidate_diag)
                continue
            if count > min(product.stock, max_count_cap):
                candidate_diag["reason"] = "count_exceeds_stock_or_limit"
                override_diagnostics["candidates"].append(candidate_diag)
                continue

            expected_weight = product.weight * count
            residual = abs(target_total - expected_weight)
            allowed_residual = per_item_tolerance * count + strict_tolerance
            candidate_diag.update(
                {
                    "expected_weight": round(float(expected_weight), 1),
                    "aggregate_residual": round(float(residual), 1),
                    "allowed_residual": round(float(allowed_residual), 1),
                }
            )
            if residual > allowed_residual:
                candidate_diag["reason"] = "aggregate_residual_exceeds_tolerance"
                override_diagnostics["candidates"].append(candidate_diag)
                continue

            weight_score = max(
                0.0,
                1.0 - residual / max(allowed_residual, 0.001),
            )
            residual_ratio = residual / max(allowed_residual, 0.001)
            accepted.append(
                {
                    "product": product,
                    "count": count,
                    "expected_weight": expected_weight,
                    "residual": residual,
                    "allowed_residual": allowed_residual,
                    "residual_ratio": residual_ratio,
                    "weight_score": weight_score,
                    "evidence_score": evidence_score,
                    "evidence_source": evidence.source,
                    "evidence_rank": evidence.rank,
                    "status": self._status_for_aggregate_evidence(evidence),
                }
            )
            candidate_diag["reason"] = "accepted_candidate_supported_repeat"
            override_diagnostics["candidates"].append(candidate_diag)

        if not accepted:
            override_diagnostics["reason"] = "no_candidate_supported_repeat_match"
            diagnostics["candidate_supported_override"] = override_diagnostics
            diagnostics["aggregate_evidence_override"] = dict(override_diagnostics)
            return None

        best = sorted(
            accepted,
            key=lambda item: (
                item["evidence_rank"] if item["evidence_rank"] is not None else 9999,
                int(float(item["residual_ratio"]) * 10),
                -item["evidence_score"],
                item["residual_ratio"],
                item["count"],
            ),
        )[0]
        selected_products = selection_summary.get("products", [])
        same_product_repeat_already_selected = (
            len(selected_products) == 1
            and int(selected_products[0]["class_id"]) == int(best["product"].product_id)
            and int(selected_products[0]["count"]) == int(best["count"])
            and not over_fragmented_segments
        )
        if not should_evaluate_aggregate:
            if not same_product_repeat_already_selected:
                override_diagnostics["reason"] = "selected_segment_already_confident"
                override_diagnostics["accepted_candidate_count"] = len(accepted)
                diagnostics["candidate_supported_override"] = override_diagnostics
                diagnostics["aggregate_evidence_override"] = dict(override_diagnostics)
                return None
        selected_segment_all_supported = bool(
            selection_summary["all_products_have_evidence"]
        ) and not bool(selection_summary["has_active_only_product"])
        selected_segment_total_residual = float(
            sum(option.residual for option in selections)
        )
        selected_segments_within_tolerance = all(
            option.residual <= option.allowed_residual for option in selections
        )
        clean_supported_segment_match = (
            len(selections) >= 2
            and selected_segment_all_supported
            and selected_segments_within_tolerance
            and selected_segment_total_residual <= float(best["residual"])
            and not same_product_repeat_already_selected
            and not selected_has_unsupported_small_fragments
        )
        if clean_supported_segment_match:
            override_diagnostics.update(
                {
                    "accepted": False,
                    "reason": "clean_supported_segment_match_preferred",
                    "selected_segment_all_supported": True,
                    "candidate_aggregate_residual": round(
                        float(best["residual"]),
                        1,
                    ),
                }
            )
            diagnostics.update(
                {
                    "candidate_supported_override": override_diagnostics,
                    "aggregate_evidence_override": dict(override_diagnostics),
                    "selected_segment_all_supported": True,
                }
            )
            return None

        product = best["product"]
        count = int(best["count"])
        weight_score = float(best["weight_score"])
        evidence_score = float(best["evidence_score"])
        status = best["status"]
        confidence = min(0.95, 0.55 * weight_score + 0.40 * min(evidence_score, 1.0) + 0.05)
        if status == JudgmentStatus.PARTIAL:
            confidence = min(0.65, confidence)
        product_judgment = ProductJudgment(
            product_id=product.product_id,
            name=product.name,
            count=count,
            unit_price=product.price,
            total_price=product.price * count,
            confidence=confidence,
            unit_weight=product.weight,
        )
        selected_diag = {
            "class_id": product.product_id,
            "name": product.name,
            "count": count,
            "unit_weight": round(float(product.weight), 1),
            "expected_weight": round(float(best["expected_weight"]), 1),
            "aggregate_residual": round(float(best["residual"]), 1),
            "allowed_residual": round(float(best["allowed_residual"]), 1),
            "residual_ratio": round(float(best["residual_ratio"]), 4),
            "evidence_score": round(evidence_score, 4),
            "evidence_source": best["evidence_source"],
            "evidence_rank": best["evidence_rank"],
            "status": status.value,
            "reason": "aggregate_evidence_repeated_count",
        }
        override_diagnostics.update(
            {
                "accepted": True,
                **selected_diag,
                "selected": selected_diag,
            }
        )
        diagnostics.update(
            {
                "accepted": True,
                "target_source": target_source,
                "status": status.value,
                "target_weight": round(float(target_total), 1),
                "expected_weight": round(float(best["expected_weight"]), 1),
                "total_residual": round(float(best["residual"]), 1),
                "confidence": round(float(confidence), 4),
                "candidate_supported_override": override_diagnostics,
                "aggregate_evidence_override": dict(override_diagnostics),
                "evidence_priority_selection": {
                    "accepted": True,
                    "reason": "aggregate_evidence_repeat_over_fragmented_segments",
                    "active_only_segment_residual": round(
                        float(sum(option.residual for option in selections)),
                        1,
                    ),
                    "candidate_aggregate_residual": round(float(best["residual"]), 1),
                    "selected_has_unsupported_small_fragments": (
                        selected_has_unsupported_small_fragments
                    ),
                },
                "selections": [
                    {
                        "segment_index": option.target.segment_index,
                        "target_weight": round(float(option.target.weight), 1),
                        **self._segment_option_diagnostics(option),
                    }
                    for option in selections
                ],
                "products": [
                    {
                        "class_id": product_judgment.product_id,
                        "name": product_judgment.name,
                        "count": product_judgment.count,
                        "unit_weight": round(float(product_judgment.unit_weight), 1),
                    }
                ],
            }
        )
        self._record_weight_diagnostics(
            trace_context,
            {
                "decision_branch": "segment_weight_matching",
                "segment_weight_matching": diagnostics,
            },
        )
        logger.info(
            "[ENGINE][reason=segment_weight_matching_candidate_override] "
            f"selected={product_judgment.name}x{count}, target={target_total:.1f}g, "
            f"expected={float(best['expected_weight']):.1f}g, "
            f"residual={float(best['residual']):.1f}g, "
            f"allowed={float(best['allowed_residual']):.1f}g"
        )
        return JudgmentResult(
            products=[product_judgment],
            total_price=product_judgment.total_price,
            confidence=confidence,
            status=status,
            weight_delta=delta_weight,
            weight_explained=float(best["expected_weight"]),
            weight_residual=round(float(best["residual"]), 1),
            timestamp=timestamp,
        )

    def _segment_selection_summary(
        self,
        selections: list[_SegmentMatchOption],
    ) -> dict[str, Any]:
        product_counts: dict[int, int] = {}
        product_names: dict[int, str] = {}
        product_evidence: dict[int, float] = {}
        product_supported: dict[int, bool] = {}
        for option in selections:
            for item in self._segment_option_items(option):
                product_id = item.product.product_id
                product_counts[product_id] = (
                    product_counts.get(product_id, 0) + item.count
                )
                product_names[product_id] = item.product.name
                product_evidence[product_id] = max(
                    product_evidence.get(product_id, 0.0),
                    item.evidence_score,
                )
                product_supported[product_id] = (
                    product_supported.get(product_id, False)
                    or self._segment_item_supported(item)
                )

        return {
            "kind_count": len(product_counts),
            "total_units": sum(product_counts.values()),
            "total_residual": round(float(sum(option.residual for option in selections)), 1),
            "all_products_have_evidence": all(
                product_supported.get(product_id, False)
                for product_id in product_counts
            ),
            "has_active_only_product": any(
                not product_supported.get(product_id, False)
                for product_id in product_counts
            ),
            "max_residual": round(
                float(max((option.residual for option in selections), default=0.0)),
                1,
            ),
            "all_segment_residuals_within_tolerance": all(
                option.residual <= option.allowed_residual for option in selections
            ),
            "products": [
                {
                    "class_id": product_id,
                    "name": product_names[product_id],
                    "count": product_counts[product_id],
                    "evidence_score": round(float(product_evidence[product_id]), 4),
                    "evidence_supported": product_supported.get(product_id, False),
                }
                for product_id in product_counts
            ],
        }

    @staticmethod
    def _segment_selection_has_unsupported_small_fragments(
        selections: list[_SegmentMatchOption],
    ) -> bool:
        for option in selections:
            for item in ProductDecisionEngine._segment_option_items(option):
                if (
                    ProductDecisionEngine._segment_item_is_unsupported_small_repeat_fragment(
                        item
                    )
                ):
                    return True
        return False

    @staticmethod
    def _has_strong_aggregate_evidence(evidence: _DetectedSingleEvidence) -> bool:
        if not evidence.motion_gate_passed and not evidence.weight_gate_passed:
            return False
        return bool(evidence.trusted or evidence.strong)

    @staticmethod
    def _status_for_aggregate_evidence(
        evidence: _DetectedSingleEvidence,
    ) -> JudgmentStatus:
        if evidence.trusted or (
            evidence.strong and "stage_counts" in evidence.source.split("+")
        ):
            return JudgmentStatus.COMPLETE
        return JudgmentStatus.PARTIAL

    def _create_segment_weight_result(
        self,
        *,
        selections: list[_SegmentMatchOption],
        delta_weight: float,
        timestamp: float,
        evidence_by_class: dict[int, _DetectedSingleEvidence],
        target_source: str,
        diagnostics: dict[str, Any],
        trace_context: Optional[object],
    ) -> JudgmentResult:
        product_order: list[int] = []
        product_counts: dict[int, int] = {}
        product_by_id: dict[int, _WeightOnlyCandidate] = {}
        for option in selections:
            for item in self._segment_option_items(option):
                product_id = item.product.product_id
                if product_id not in product_counts:
                    product_order.append(product_id)
                    product_counts[product_id] = 0
                    product_by_id[product_id] = item.product
                product_counts[product_id] += item.count

        all_products_have_evidence = all(
            self._segment_item_supported(item)
            for option in selections
            for item in self._segment_option_items(option)
        )
        status = (
            JudgmentStatus.COMPLETE
            if all_products_have_evidence
            else JudgmentStatus.PARTIAL
        )
        avg_weight_score = sum(option.weight_score for option in selections) / max(
            len(selections),
            1,
        )
        avg_evidence_score = sum(
            min(option.evidence_score, 1.0) for option in selections
        ) / max(len(selections), 1)
        confidence = 0.55 * avg_weight_score + 0.40 * avg_evidence_score
        if status == JudgmentStatus.COMPLETE:
            confidence = min(0.95, confidence + 0.05)
        else:
            confidence = min(0.65, max(0.25, confidence))

        products: list[ProductJudgment] = []
        total_price = 0
        for product_id in product_order:
            product = product_by_id[product_id]
            count = product_counts[product_id]
            total_price += product.price * count
            products.append(
                ProductJudgment(
                    product_id=product.product_id,
                    name=product.name,
                    count=count,
                    unit_price=product.price,
                    total_price=product.price * count,
                    confidence=confidence,
                    unit_weight=product.weight,
                )
            )

        target_total = sum(option.target.weight for option in selections)
        expected_total = sum(option.expected_weight for option in selections)
        total_residual = sum(option.residual for option in selections)
        diagnostics.update(
            {
                "accepted": True,
                "target_source": target_source,
                "status": status.value,
                "target_weight": round(float(target_total), 1),
                "expected_weight": round(float(expected_total), 1),
                "total_residual": round(float(total_residual), 1),
                "confidence": round(float(confidence), 4),
                "selections": [
                    {
                        "segment_index": option.target.segment_index,
                        "target_weight": round(float(option.target.weight), 1),
                        **self._segment_option_diagnostics(option),
                    }
                    for option in selections
                ],
                "products": [
                    {
                        "class_id": product.product_id,
                        "name": product.name,
                        "count": product.count,
                        "unit_weight": round(float(product.unit_weight), 1),
                    }
                    for product in products
                ],
            }
        )
        self._record_weight_diagnostics(
            trace_context,
            {
                "decision_branch": "segment_weight_matching",
                "segment_weight_matching": diagnostics,
            },
        )
        logger.info(
            "[ENGINE][reason=segment_weight_matching] "
            f"target={target_total:.1f}g, expected={expected_total:.1f}g, "
            f"residual={total_residual:.1f}g, status={status.value}"
        )
        return JudgmentResult(
            products=products,
            total_price=total_price,
            confidence=confidence,
            status=status,
            weight_delta=delta_weight,
            weight_explained=expected_total,
            weight_residual=round(float(total_residual), 1),
            timestamp=timestamp,
        )

    def _try_forced_final_fallback(
        self,
        *,
        vision_candidates: List[EnsembleResult],
        delta_weight: float,
        timestamp: float,
        active_products: Optional[List],
        trace_context: Optional[object],
        previous_status: str,
    ) -> Optional[JudgmentResult]:
        """Return a best-effort active-product guess so chargeable removals are not none."""
        if delta_weight >= 0 or abs(delta_weight) < self.min_weight_change:
            return None

        active_candidates = self._build_weight_only_candidates(active_products)
        if not active_candidates:
            self._record_weight_diagnostics(
                trace_context,
                {
                    "forced_final_fallback": {
                        "accepted": False,
                        "reason": "no_valid_active_products",
                        "previous_status": previous_status,
                    }
                },
            )
            return None

        evidence_by_class = self._collect_detected_single_evidence(
            vision_candidates,
            trace_context,
        )
        target_weight = abs(float(delta_weight))
        low_weight_noise_diagnostics = (
            self._active_only_low_weight_noise_diagnostics(
                target_weight=target_weight,
                active_candidates=active_candidates,
                evidence_by_class=evidence_by_class,
                trace_context=trace_context,
                previous_status=previous_status,
            )
        )
        if low_weight_noise_diagnostics is not None:
            self._record_weight_diagnostics(
                trace_context,
                {"forced_final_fallback": low_weight_noise_diagnostics},
            )
            logger.warning(
                "[ENGINE][reason=active_only_low_weight_noise] "
                f"target={target_weight:.1f}g, "
                f"min_active={low_weight_noise_diagnostics['min_active_weight']:.1f}g, "
                f"ceiling={low_weight_noise_diagnostics['noise_ceiling']:.1f}g"
            )
            return None

        target_weights, rejected_partial_targets = self._forced_fallback_target_weights(
            delta_weight,
            trace_context,
        )
        if rejected_partial_targets:
            self._record_weight_diagnostics(
                trace_context,
                {
                    "forced_fallback_rejected_partial_target": {
                        "accepted": False,
                        "reason": "partial_target_does_not_explain_full_delta",
                        "target_weight": round(abs(float(delta_weight)), 1),
                        "rejected": rejected_partial_targets,
                        "previous_status": previous_status,
                    }
                },
            )
        tolerance = max(
            float(config.weight.tolerance_grams),
            float(config.weight.detected_single_fallback_tolerance_grams),
        )
        single_options = self._forced_single_options(
            active_candidates=active_candidates,
            evidence_by_class=evidence_by_class,
            target_weights=target_weights,
            tolerance=tolerance,
        )
        inside_single_options = [
            option for option in single_options if bool(option["inside_tolerance"])
        ]
        if inside_single_options:
            selected = sorted(
                inside_single_options,
                key=lambda option: (
                    int(option["target_rank"]),
                    -float(option["evidence_score"]),
                    float(option["residual"]),
                ),
            )[0]
            return self._create_forced_fallback_result(
                option=selected,
                delta_weight=delta_weight,
                timestamp=timestamp,
                tolerance=tolerance,
                trace_context=trace_context,
                previous_status=previous_status,
            )

        pair_options = self._forced_pair_options(
            active_candidates=active_candidates,
            evidence_by_class=evidence_by_class,
            target_weights=target_weights,
            tolerance=tolerance,
        )
        inside_pair_options = [
            option for option in pair_options if bool(option["inside_tolerance"])
        ]
        if inside_pair_options:
            selected = sorted(
                inside_pair_options,
                key=lambda option: (
                    int(option["target_rank"]),
                    int(option.get("pair_support_rank", 1)),
                    -float(option["evidence_score"]),
                    float(option["residual"]),
                ),
            )[0]
            return self._create_forced_fallback_result(
                option=selected,
                delta_weight=delta_weight,
                timestamp=timestamp,
                tolerance=tolerance,
                trace_context=trace_context,
                previous_status=previous_status,
            )

        fallback_options = single_options + pair_options
        if not fallback_options:
            return None
        selected = sorted(
            fallback_options,
            key=lambda option: (
                float(option["residual"]),
                int(option["unit_count"]),
                -float(option["evidence_score"]),
            ),
        )[0]
        return self._create_forced_fallback_result(
            option=selected,
            delta_weight=delta_weight,
            timestamp=timestamp,
            tolerance=tolerance,
            trace_context=trace_context,
            previous_status=previous_status,
        )

    def _active_only_low_weight_noise_diagnostics(
        self,
        *,
        target_weight: float,
        active_candidates: list[_WeightOnlyCandidate],
        evidence_by_class: dict[int, _DetectedSingleEvidence],
        trace_context: Optional[object],
        previous_status: str,
    ) -> Optional[dict[str, object]]:
        """Reject tiny active-only fallback guesses below the lightest product."""
        if evidence_by_class:
            return None
        if self._trace_has_purchase_delta_candidates(trace_context):
            return None

        valid_weights = [
            float(candidate.weight)
            for candidate in active_candidates
            if float(candidate.weight) > 0
        ]
        if not valid_weights:
            return None

        strict_tolerance = max(0.0, float(config.weight.tolerance_grams))
        fallback_tolerance = max(
            0.0,
            float(config.weight.detected_single_fallback_tolerance_grams),
        )
        min_active_weight = min(valid_weights)
        under_min_ceiling = min_active_weight - strict_tolerance
        if under_min_ceiling <= self.min_weight_change:
            return None

        # Keep this as a noise guard, not a broad fallback disable. Large
        # active-only misses still use the existing forced fallback policy.
        noise_ceiling = min(
            under_min_ceiling,
            strict_tolerance + fallback_tolerance,
        )
        if target_weight > noise_ceiling:
            return None

        return {
            "accepted": False,
            "reason": "active_only_low_weight_noise",
            "previous_status": previous_status,
            "target_weight": round(float(target_weight), 1),
            "min_active_weight": round(float(min_active_weight), 1),
            "strict_tolerance": round(float(strict_tolerance), 1),
            "noise_ceiling": round(float(noise_ceiling), 1),
        }

    @staticmethod
    def _trace_has_purchase_delta_candidates(trace_context: Optional[object]) -> bool:
        loadcell = getattr(trace_context, "loadcell", {}) if trace_context else {}
        return (
            isinstance(loadcell, dict)
            and bool(loadcell.get("purchase_delta_candidates"))
        )

    @staticmethod
    def _forced_fallback_target_weights(
        delta_weight: float,
        trace_context: Optional[object],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        targets: list[dict[str, object]] = []
        rejected_partial_targets: list[dict[str, object]] = []
        loadcell = getattr(trace_context, "loadcell", {}) if trace_context else {}
        fallback_weight = abs(float(delta_weight))
        strict_tolerance = max(0.1, float(config.weight.tolerance_grams))
        if isinstance(loadcell, dict):
            for entry in loadcell.get("purchase_delta_candidates") or []:
                if not isinstance(entry, dict):
                    continue
                try:
                    weight = abs(float(entry.get("weight", 0.0)))
                except (TypeError, ValueError):
                    continue
                if weight <= 0:
                    continue
                source = str(entry.get("source") or "loadcell_history")
                if abs(weight - fallback_weight) > strict_tolerance:
                    rejected_partial_targets.append(
                        {
                            "weight": round(float(weight), 1),
                            "source": source,
                            "residual_to_full_delta": round(
                                abs(float(fallback_weight) - weight),
                                1,
                            ),
                        }
                    )
                    continue
                targets.append(
                    {
                        "weight": weight,
                        "source": source,
                        "rank": len(targets),
                    }
                )
        if not any(abs(float(target["weight"]) - fallback_weight) < 0.1 for target in targets):
            targets.append(
                {
                    "weight": fallback_weight,
                    "source": "decision_delta_weight",
                    "rank": len(targets),
                }
            )
        return targets, rejected_partial_targets

    @staticmethod
    def _forced_evidence_score(
        product_id: int,
        evidence_by_class: dict[int, _DetectedSingleEvidence],
    ) -> float:
        evidence = evidence_by_class.get(int(product_id))
        if evidence is None:
            return 0.0
        if evidence.stage_score > 0.0 and "stage_counts" in evidence.source:
            score = float(evidence.stage_score)
        else:
            score = float(evidence.confidence)
            score += max(0, int(evidence.votes)) * 0.05
            if evidence.motion_gate_passed:
                score += 0.2
        if evidence.trusted:
            score += 1.0
        return score

    @staticmethod
    def _forced_pair_product_supported(
        product_id: int,
        evidence_by_class: dict[int, _DetectedSingleEvidence],
    ) -> bool:
        evidence = evidence_by_class.get(int(product_id))
        return ProductDecisionEngine._detected_evidence_supported(evidence)

    @staticmethod
    def _detected_evidence_supported(evidence: Optional[_DetectedSingleEvidence]) -> bool:
        if evidence is None:
            return False
        if evidence.trusted or evidence.strong or bool(evidence.weight_gate_passed):
            return True
        min_votes = max(0, int(config.weight.detected_single_fallback_min_votes))
        return evidence.votes >= min_votes and evidence.confidence >= 0.18

    def _forced_pair_support(
        self,
        products: list[tuple[_WeightOnlyCandidate, int]],
        evidence_by_class: dict[int, _DetectedSingleEvidence],
    ) -> tuple[int, float, list[dict[str, object]]]:
        details: list[dict[str, object]] = []
        all_supported = True
        evidence_score = 0.0
        for product, count in products:
            product_score = self._forced_evidence_score(
                product.product_id,
                evidence_by_class,
            )
            evidence_score += product_score
            evidence = evidence_by_class.get(product.product_id)
            supported = self._forced_pair_product_supported(
                product.product_id,
                evidence_by_class,
            )
            all_supported = all_supported and supported
            details.append(
                {
                    "class_id": product.product_id,
                    "name": product.name,
                    "count": int(count),
                    "supported": supported,
                    "evidence_score": round(product_score, 4),
                    "votes": int(evidence.votes) if evidence else 0,
                    "confidence": (
                        round(float(evidence.confidence), 4) if evidence else 0.0
                    ),
                    "source": evidence.source if evidence else None,
                }
            )
        return (0 if all_supported else 1), evidence_score, details

    def _forced_single_options(
        self,
        *,
        active_candidates: list[_WeightOnlyCandidate],
        evidence_by_class: dict[int, _DetectedSingleEvidence],
        target_weights: list[dict[str, object]],
        tolerance: float,
    ) -> list[dict[str, object]]:
        options: list[dict[str, object]] = []
        for target in target_weights:
            target_weight = float(target["weight"])
            for product in active_candidates:
                residual = abs(target_weight - product.weight)
                options.append(
                    {
                        "mode": "single_active_product",
                        "products": [(product, 1)],
                        "unit_count": 1,
                        "expected_weight": product.weight,
                        "target_weight": target_weight,
                        "target_source": target["source"],
                        "target_rank": int(target["rank"]),
                        "residual": residual,
                        "inside_tolerance": residual <= tolerance,
                        "evidence_score": self._forced_evidence_score(
                            product.product_id,
                            evidence_by_class,
                        ),
                    }
                )
        return options

    def _forced_pair_options(
        self,
        *,
        active_candidates: list[_WeightOnlyCandidate],
        evidence_by_class: dict[int, _DetectedSingleEvidence],
        target_weights: list[dict[str, object]],
        tolerance: float,
    ) -> list[dict[str, object]]:
        if not evidence_by_class:
            return []
        product_by_id = {product.product_id: product for product in active_candidates}
        detected_products = [
            product_by_id[class_id]
            for class_id in evidence_by_class
            if class_id in product_by_id
        ]
        detected_products.sort(
            key=lambda product: -self._forced_evidence_score(
                product.product_id,
                evidence_by_class,
            )
        )
        options: list[dict[str, object]] = []
        for target in target_weights:
            target_weight = float(target["weight"])
            for detected_product in detected_products:
                for other_product in active_candidates:
                    same_product_pair = (
                        detected_product.product_id == other_product.product_id
                    )
                    if same_product_pair and detected_product.stock < 2:
                        continue
                    expected_weight = detected_product.weight + other_product.weight
                    residual = abs(target_weight - expected_weight)
                    products = (
                        [(detected_product, 2)]
                        if same_product_pair
                        else [(detected_product, 1), (other_product, 1)]
                    )
                    option_tolerance = float(tolerance)
                    repeat_tolerance_grace = False
                    evidence = evidence_by_class.get(detected_product.product_id)
                    if (
                        same_product_pair
                        and evidence is not None
                        and self._source_includes_regular_vision(evidence.source)
                        and self._detected_evidence_supported(evidence)
                        and self._is_500ml_bottle_weight(detected_product.weight)
                    ):
                        option_tolerance = max(
                            option_tolerance,
                            self._count_scaled_weight_tolerance(2, extra_units=1),
                        )
                        repeat_tolerance_grace = option_tolerance > float(tolerance)
                    pair_support_rank, evidence_score, pair_support = (
                        self._forced_pair_support(products, evidence_by_class)
                    )
                    options.append(
                        {
                            "mode": (
                                "detected_same_product_pair"
                                if same_product_pair
                                else "detected_plus_active_pair"
                            ),
                            "products": products,
                            "unit_count": 2,
                            "expected_weight": expected_weight,
                            "target_weight": target_weight,
                            "target_source": target["source"],
                            "target_rank": int(target["rank"]),
                            "residual": residual,
                            "tolerance": option_tolerance,
                            "base_tolerance": float(tolerance),
                            "repeat_tolerance_grace": repeat_tolerance_grace,
                            "inside_tolerance": residual <= option_tolerance,
                            "evidence_score": evidence_score,
                            "pair_support_rank": pair_support_rank,
                            "pair_support": pair_support,
                        }
                    )
        return options

    def _create_forced_fallback_result(
        self,
        *,
        option: dict[str, object],
        delta_weight: float,
        timestamp: float,
        tolerance: float,
        trace_context: Optional[object],
        previous_status: str,
    ) -> JudgmentResult:
        products: list[ProductJudgment] = []
        total_price = 0
        for product, count in option["products"]:
            product_count = int(count)
            total_price += product.price * product_count
            products.append(
                ProductJudgment(
                    product_id=product.product_id,
                    name=product.name,
                    count=product_count,
                    unit_price=product.price,
                    total_price=product.price * product_count,
                    confidence=0.0,
                    unit_weight=product.weight,
                )
            )

        residual = float(option["residual"])
        inside_tolerance = bool(option["inside_tolerance"])
        target_weight = float(option["target_weight"])
        option_tolerance = float(option.get("tolerance", tolerance))
        weight_score = max(
            0.0,
            1.0 - residual / max(target_weight, option_tolerance, 1.0),
        )
        confidence = (
            min(0.55, 0.25 + 0.25 * weight_score)
            if inside_tolerance
            else min(0.25, 0.05 + 0.20 * weight_score)
        )
        for product in products:
            product.confidence = confidence

        diagnostics = {
            "accepted": True,
            "previous_status": previous_status,
            "mode": option["mode"],
            "target_weight": round(target_weight, 1),
            "target_source": option["target_source"],
            "expected_weight": round(float(option["expected_weight"]), 1),
            "residual": round(residual, 1),
            "tolerance": round(float(option_tolerance), 1),
            "inside_tolerance": inside_tolerance,
            "confidence": round(confidence, 4),
            "products": [
                {
                    "class_id": product.product_id,
                    "name": product.name,
                    "count": product.count,
                    "unit_weight": round(float(product.unit_weight), 1),
                }
                for product in products
            ],
        }
        if "evidence_score" in option:
            diagnostics["evidence_score"] = round(float(option["evidence_score"]), 4)
        if "pair_support_rank" in option:
            diagnostics["pair_support_rank"] = int(option["pair_support_rank"])
        if "pair_support" in option:
            diagnostics["pair_support"] = option["pair_support"]
        if "base_tolerance" in option:
            diagnostics["base_tolerance"] = round(float(option["base_tolerance"]), 1)
        if "repeat_tolerance_grace" in option:
            diagnostics["repeat_tolerance_grace"] = bool(
                option["repeat_tolerance_grace"]
            )
        self._record_weight_diagnostics(
            trace_context,
            {
                "decision_branch": "forced_final_fallback",
                "forced_final_fallback": diagnostics,
            },
        )
        logger.warning(
            "[ENGINE][reason=forced_final_fallback] "
            f"mode={option['mode']}, target={target_weight:.1f}g, "
            f"expected={float(option['expected_weight']):.1f}g, "
            f"residual={residual:.1f}g, inside_tolerance={inside_tolerance}"
        )
        return JudgmentResult(
            products=products,
            total_price=total_price,
            confidence=confidence,
            status=JudgmentStatus.PARTIAL,
            weight_delta=delta_weight,
            weight_explained=float(option["expected_weight"]),
            weight_residual=round(residual, 1),
            timestamp=timestamp,
        )

    def _collect_detected_single_evidence(
        self,
        vision_candidates: List[EnsembleResult],
        trace_context: Optional[object],
    ) -> dict[int, _DetectedSingleEvidence]:
        evidence_by_class: dict[int, _DetectedSingleEvidence] = {}

        def add_evidence(evidence: _DetectedSingleEvidence) -> None:
            existing = evidence_by_class.get(evidence.class_id)
            if existing is None:
                evidence_by_class[evidence.class_id] = evidence
                return
            source_parts = sorted(
                set(existing.source.split("+")) | set(evidence.source.split("+"))
            )
            preferred = existing
            if (
                int(evidence.trusted) > int(existing.trusted)
                or evidence.votes > existing.votes
                or (
                    evidence.votes == existing.votes
                    and evidence.confidence > existing.confidence
                )
            ):
                preferred = evidence
            rank_values = [
                rank
                for rank in (existing.rank, evidence.rank)
                if rank is not None
            ]
            weight_gate_passed = (
                bool(existing.weight_gate_passed) or bool(evidence.weight_gate_passed)
            )
            evidence_by_class[evidence.class_id] = _DetectedSingleEvidence(
                class_id=preferred.class_id,
                name=preferred.name or existing.name or evidence.name,
                votes=max(existing.votes, evidence.votes),
                confidence=max(existing.confidence, evidence.confidence),
                source="+".join(source_parts),
                trusted=existing.trusted or evidence.trusted,
                motion_gate_passed=(
                    existing.motion_gate_passed or evidence.motion_gate_passed
                ),
                weight_gate_passed=weight_gate_passed or None,
                strong=existing.strong or evidence.strong,
                rank=min(rank_values) if rank_values else None,
                stage_score=max(existing.stage_score, evidence.stage_score),
                top_confidence=max(
                    existing.top_confidence,
                    evidence.top_confidence,
                ),
                side_confidence=max(
                    existing.side_confidence,
                    evidence.side_confidence,
                ),
                top_votes=max(existing.top_votes, evidence.top_votes),
                side_votes=max(existing.side_votes, evidence.side_votes),
                score_reason=(
                    evidence.score_reason
                    if evidence.stage_score >= existing.stage_score
                    else existing.score_reason
                ),
            )

        for rank, candidate in enumerate(vision_candidates, start=1):
            raw_votes = int(candidate.raw_vote_count or 0)
            votes = max(raw_votes, int(candidate.vote_count or 0))
            motion_gate_passed = bool(
                candidate.motion_gate_passed
                or candidate.top_motion_passed
                or candidate.side_motion_passed
            )
            add_evidence(
                _DetectedSingleEvidence(
                    class_id=int(candidate.class_id),
                    name=candidate.class_name,
                    votes=votes,
                    confidence=float(candidate.combined_confidence),
                    source=getattr(candidate, "source", "vision"),
                    trusted=True,
                    motion_gate_passed=motion_gate_passed,
                    weight_gate_passed=candidate.weight_gate_passed,
                    strong=True,
                    rank=rank,
                )
            )

        if trace_context is None:
            return evidence_by_class

        stage_counts = getattr(trace_context, "stage_counts_by_class", {}) or {}
        for entry in stage_counts.values():
            if not isinstance(entry, dict):
                continue
            class_id = entry.get("class_id")
            try:
                class_id_int = int(class_id)
            except (TypeError, ValueError):
                continue
            summary = self._stage_evidence_summary(entry)
            votes = summary.votes
            confidence = summary.confidence
            motion_gate_passed = bool(
                summary.motion_gate_passed
                or entry.get("motion_gate_passed", True)
            )
            trusted = bool(
                entry.get("final_rank") is not None
                or entry.get("weight_gate_passed")
            )
            strong = (
                votes >= 20
                and confidence >= 0.30
                and motion_gate_passed
            )
            rank = self._coerce_int(entry.get("final_rank"))
            if votes <= 0 and not trusted:
                continue
            add_evidence(
                _DetectedSingleEvidence(
                    class_id=class_id_int,
                    name=str(entry.get("name") or ""),
                    votes=votes,
                    confidence=confidence,
                    source="stage_counts",
                    trusted=trusted,
                    motion_gate_passed=motion_gate_passed,
                    weight_gate_passed=entry.get("weight_gate_passed"),
                    strong=strong,
                    rank=rank if rank > 0 else None,
                    stage_score=summary.stage_score,
                    top_confidence=summary.top_confidence,
                    side_confidence=summary.side_confidence,
                    top_votes=summary.top_votes,
                    side_votes=summary.side_votes,
                    score_reason=summary.score_reason,
                )
            )

        diagnostic_detections = getattr(trace_context, "diagnostic_detections", []) or []
        diagnostic_by_class: dict[int, dict[str, Any]] = {}
        for detection in diagnostic_detections:
            if not isinstance(detection, dict):
                continue
            try:
                class_id_int = int(detection.get("class_id"))
            except (TypeError, ValueError):
                continue
            entry = diagnostic_by_class.setdefault(
                class_id_int,
                {
                    "name": detection.get("name") or "",
                    "votes": 0,
                    "confidence": 0.0,
                },
            )
            entry["votes"] += 1
            entry["confidence"] = max(
                float(entry["confidence"]),
                self._coerce_float(detection.get("confidence", 0.0)),
            )

        diagnostic_entries = sorted(
            diagnostic_by_class.items(),
            key=lambda item: (int(item[1]["votes"]), float(item[1]["confidence"])),
            reverse=True,
        )
        for rank, (class_id_int, entry) in enumerate(diagnostic_entries, start=1):
            votes = int(entry["votes"])
            confidence = float(entry["confidence"])
            add_evidence(
                _DetectedSingleEvidence(
                    class_id=class_id_int,
                    name=str(entry["name"]),
                    votes=votes,
                    confidence=confidence,
                    source="diagnostic",
                    trusted=False,
                    strong=votes >= 5 and confidence >= 0.30,
                    rank=rank,
                )
            )

        return evidence_by_class

    @staticmethod
    def _max_stage_count(entry: dict[str, Any]) -> int:
        count_keys = (
            "raw",
            "threshold_passed",
            "threshold_filtered",
            "roi_passed",
            "roi_filtered",
            "motion_passed",
            "hand_path_passed",
        )
        return max(
            [0]
            + [
                ProductDecisionEngine._coerce_int(entry.get(key, 0))
                for key in count_keys
            ]
        )

    @staticmethod
    def _max_stage_confidence(entry: dict[str, Any]) -> float:
        confidence_values = [
            ProductDecisionEngine._coerce_float(value)
            for key, value in entry.items()
            if key.endswith("_max_confidence")
        ]
        if not confidence_values:
            return 0.0
        return max(confidence_values)

    @staticmethod
    def _max_stage_passed_count(entry: dict[str, Any]) -> int:
        passed_keys = (
            "threshold_passed",
            "roi_passed",
            "roi_filtered",
            "motion_filtered",
            "motion_passed",
            "hand_path_passed",
        )
        return max(
            [0]
            + [
                ProductDecisionEngine._coerce_int(entry.get(key, 0))
                for key in passed_keys
            ]
        )

    @staticmethod
    def _bounded_log_count_score(count: int, cap: int) -> float:
        if count <= 0 or cap <= 0:
            return 0.0
        return min(1.0, math.log1p(float(count)) / math.log1p(float(cap)))

    @classmethod
    def _stage_evidence_summary(
        cls,
        entry: dict[str, Any],
    ) -> _StageEvidenceSummary:
        cameras = entry.get("cameras", {})
        top_entry = cameras.get("top", {}) if isinstance(cameras, dict) else {}
        side_entry = cameras.get("side", {}) if isinstance(cameras, dict) else {}
        if not isinstance(top_entry, dict):
            top_entry = {}
        if not isinstance(side_entry, dict):
            side_entry = {}

        aggregate_votes = cls._max_stage_count(entry)
        aggregate_confidence = cls._max_stage_confidence(entry)
        aggregate_passed = cls._max_stage_passed_count(entry)

        top_votes = cls._max_stage_count(top_entry)
        side_votes = cls._max_stage_count(side_entry)
        top_confidence = cls._max_stage_confidence(top_entry)
        side_confidence = cls._max_stage_confidence(side_entry)
        top_passed = cls._max_stage_passed_count(top_entry)
        side_passed = cls._max_stage_passed_count(side_entry)

        has_camera_breakdown = bool(top_entry or side_entry)
        generic_votes = 0 if has_camera_breakdown else aggregate_votes
        generic_confidence = 0.0 if has_camera_breakdown else aggregate_confidence
        generic_passed = 0 if has_camera_breakdown else aggregate_passed

        top_motion_passed = bool(
            top_entry.get("motion_passed")
            or top_entry.get("motion_filtered")
            or top_entry.get("hand_path_passed")
        )
        side_motion_passed = bool(
            side_entry.get("motion_passed")
            or side_entry.get("motion_filtered")
            or side_entry.get("hand_path_passed")
        )
        motion_gate_passed = bool(
            entry.get("motion_gate_passed")
            or entry.get("motion_passed")
            or entry.get("motion_filtered")
            or entry.get("hand_path_passed")
            or top_motion_passed
            or side_motion_passed
        )

        score = 0.0
        components: list[str] = []

        side_vote_score = 0.35 * cls._bounded_log_count_score(side_votes, 20)
        if side_vote_score:
            score += side_vote_score
            components.append("side_votes")

        top_vote_score = 0.10 * cls._bounded_log_count_score(top_votes, 50)
        if top_vote_score:
            score += top_vote_score
            components.append("top_votes")

        generic_vote_score = 0.20 * cls._bounded_log_count_score(generic_votes, 50)
        if generic_vote_score:
            score += generic_vote_score
            components.append("aggregate_votes")

        side_confidence_score = 1.50 * max(0.0, min(1.0, side_confidence))
        if side_confidence_score:
            score += side_confidence_score
            components.append("side_confidence")

        top_confidence_score = 0.35 * max(0.0, min(1.0, top_confidence))
        if top_confidence_score:
            score += top_confidence_score
            components.append("top_confidence")

        generic_confidence_score = 0.80 * max(0.0, min(1.0, generic_confidence))
        if generic_confidence_score:
            score += generic_confidence_score
            components.append("aggregate_confidence")

        side_pass_score = 0.25 * cls._bounded_log_count_score(side_passed, 5)
        if side_pass_score:
            score += side_pass_score
            components.append("side_passed")

        top_pass_score = 0.08 * cls._bounded_log_count_score(top_passed, 20)
        if top_pass_score:
            score += top_pass_score
            components.append("top_passed")

        generic_pass_score = 0.15 * cls._bounded_log_count_score(generic_passed, 20)
        if generic_pass_score:
            score += generic_pass_score
            components.append("aggregate_passed")

        if side_passed > 0:
            score += 0.15
            components.append("side_roi_or_threshold")
        elif top_passed > 0:
            score += 0.04
            components.append("top_roi_or_threshold")

        if motion_gate_passed:
            score += 0.20
            components.append("motion")

        confidence = max(aggregate_confidence, top_confidence, side_confidence)
        votes = max(aggregate_votes, top_votes, side_votes)
        return _StageEvidenceSummary(
            votes=votes,
            confidence=confidence,
            stage_score=max(0.0, score),
            top_confidence=top_confidence,
            side_confidence=side_confidence,
            top_votes=top_votes,
            side_votes=side_votes,
            top_passed=top_passed,
            side_passed=side_passed,
            motion_gate_passed=motion_gate_passed,
            top_motion_passed=top_motion_passed,
            side_motion_passed=side_motion_passed,
            score_reason="+".join(components) if components else "no_stage_signal",
        )

    @staticmethod
    def _record_detected_single_fallback(
        trace_context: Optional[object],
        diagnostics: dict[str, Any],
    ) -> None:
        if trace_context is None:
            return
        if hasattr(trace_context, "record_detected_single_fallback"):
            trace_context.record_detected_single_fallback(diagnostics)
            return
        weight_diagnostics = getattr(trace_context, "weight_diagnostics", None)
        if isinstance(weight_diagnostics, dict):
            weight_diagnostics["detected_single_item_fallback"] = diagnostics
            if diagnostics.get("accepted"):
                weight_diagnostics["fallback_reason"] = "detected_single_item_fallback"

    @staticmethod
    def _coerce_float(value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _coerce_int(value: object) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _create_result_from_strict_combo(
        self,
        combo,  # ValidCombination
        delta_weight: float,
        timestamp: float,
    ) -> JudgmentResult:
        """
        StrictWeightMatcher 조합에서 JudgmentResult 생성.

        Args:
            combo: ValidCombination 객체
            delta_weight: 무게 변화량
            timestamp: 판단 시각

        Returns:
            JudgmentResult (status: COMPLETE)
        """
        # Strict combinations already fully explain the load delta, so each
        # item can safely inherit the combination-level confidence.
        products = []
        total_price = 0

        for item in combo.items:
            # 무게 기반이므로 신뢰도를 조합 match_score로 설정
            confidence = combo.match_score

            product = ProductJudgment(
                product_id=item.candidate.class_id,
                name=item.candidate.name,
                count=item.count,
                unit_price=item.candidate.unit_price,
                total_price=item.candidate.unit_price * item.count,
                confidence=confidence,
                unit_weight=item.candidate.weight,
            )
            products.append(product)
            total_price += product.total_price

        # 평균 신뢰도
        avg_confidence = combo.match_score

        items_str = " + ".join(
            f"{p.name}x{p.count}" for p in products
        )
        logger.info(
            f"[ENGINE] v5.1 strict 결과: COMPLETE, {items_str}, "
            f"total_price={total_price}원, confidence={avg_confidence:.3f}"
        )

        return JudgmentResult(
            products=products,
            total_price=total_price,
            confidence=avg_confidence,
            status=JudgmentStatus.COMPLETE,
            weight_delta=delta_weight,
            weight_explained=combo.total_weight,
            weight_residual=combo.weight_error,
            timestamp=timestamp,
        )

    def _judge_vision_only(
        self,
        vision_candidates: List[EnsembleResult],
        timestamp: float,
    ) -> JudgmentResult:
        """
        Vision 전용 판단 (로드셀 없이 카메라만 사용).

        무게 검증 없이 Vision 신뢰도만으로 상품을 판단합니다.
        개수는 1로 고정되며, 신뢰도는 Vision 신뢰도의 70%로 감소합니다.

        YOLO class_id 기반 조회:
        1. get_by_yolo_class_id로 매핑된 상품 조회
        2. 실패 시 get_by_yolo_class_name으로 폴백
        3. 최종 실패 시 Vision 결과에서 직접 생성

        Args:
            vision_candidates: Vision 후보군
            timestamp: 판단 시각

        Returns:
            JudgmentResult (status: PARTIAL, 무게 검증 없음)
        """
        logger.info(f"Vision-only judgment: {len(vision_candidates)} candidates")

        if not vision_candidates:
            logger.warning("No vision candidates for vision-only judgment")
            return self._create_no_detection_result(0.0, timestamp)

        best_candidate = max(vision_candidates, key=lambda c: c.combined_confidence)

        # Vision-only mode should stay available even when the product DB has
        # not been injected yet. In that case we still emit a partial answer
        # using the raw class metadata from the detector.
        if self.product_db is None:
            confidence = best_candidate.combined_confidence * 0.7
            logger.warning(
                f"Vision-only fallback without product DB: "
                f"class_id={best_candidate.class_id}, name={best_candidate.class_name}"
            )
            return JudgmentResult(
                products=[
                    ProductJudgment(
                        product_id=best_candidate.class_id,
                        name=best_candidate.class_name,
                        count=1,
                        unit_price=0,
                        total_price=0,
                        confidence=confidence,
                        unit_weight=0.0,
                    )
                ],
                total_price=0,
                confidence=confidence,
                status=JudgmentStatus.PARTIAL,
                weight_delta=0.0,
                weight_explained=0.0,
                weight_residual=0.0,
                timestamp=timestamp,
            )

        # 후보군 로그 출력
        candidate_limit = max(1, int(config.vision.top_k))
        for i, c in enumerate(vision_candidates[:candidate_limit]):
            logger.debug(
                f"  Candidate {i+1}: class_id={c.class_id}, name={c.class_name}, "
                f"conf={c.combined_confidence:.3f}"
            )

        # 가장 높은 신뢰도의 후보 선택
        best_candidate = max(vision_candidates, key=lambda c: c.combined_confidence)

        if best_candidate.combined_confidence < self.confidence_threshold:
            logger.info(
                f"Vision confidence too low: {best_candidate.combined_confidence:.3f} "
                f"< {self.confidence_threshold}"
            )
            return self._create_no_detection_result(0.0, timestamp)

        # 상품 정보 조회 (YOLO 매핑 우선)
        # 1. YOLO class_id로 매핑된 상품 조회
        product = self._lookup_product_for_candidate(best_candidate)

        # 2. 폴백: YOLO 클래스명으로 조회
        if product is None:
            product = self.product_db.get_by_yolo_class_name(best_candidate.class_name)
            if product:
                logger.debug(
                    f"Found product by yolo_class_name: {best_candidate.class_name} -> "
                    f"product_id={product.product_id}, name={product.name}"
                )

        # 3. 폴백: 기존 방식 (product_id = class_id)
        if product is None:
            product = self.product_db.get_product(best_candidate.class_id)
            if product:
                logger.debug(
                    f"Found product by class_id as product_id: {best_candidate.class_id} -> "
                    f"name={product.name}"
                )

        # ProductDatabase에 없으면 Vision 결과에서 직접 생성 (fallback)
        if product is None:
            logger.warning(
                f"Product not found in DB for class_id={best_candidate.class_id}, "
                f"using Vision class_name as fallback: {best_candidate.class_name}"
            )
            # Vision 전용: 신뢰도 70%로 감소, 개수는 1로 고정
            confidence = best_candidate.combined_confidence * 0.7
            count = 1

            # Vision 결과에서 상품 정보 생성 (가격은 0원, 무게는 0g)
            product_judgment = ProductJudgment(
                product_id=best_candidate.class_id,
                name=best_candidate.class_name,
                count=count,
                unit_price=0,  # 가격 미정
                total_price=0,
                confidence=confidence,
                unit_weight=0.0,  # 무게 미정
            )

            logger.info(
                f"Vision-only result (fallback): {best_candidate.class_name}, "
                f"class_id={best_candidate.class_id}, "
                f"vision_conf={best_candidate.combined_confidence:.3f}, "
                f"final_conf={confidence:.3f}, count={count}"
            )

            return JudgmentResult(
                products=[product_judgment],
                total_price=0,
                confidence=confidence,
                status=JudgmentStatus.PARTIAL,  # 무게 미검증
                weight_delta=0.0,
                weight_explained=0.0,
                weight_residual=0.0,
                timestamp=timestamp,
            )

        # Vision 전용: 신뢰도 70%로 감소, 개수는 1로 고정
        confidence = best_candidate.combined_confidence * 0.7
        count = 1  # 무게 없이 개수 추정 불가

        # IF11 매핑된 상품 정보 사용 (가격, 무게)
        logger.info(
            f"Vision-only result: {product.name} (product_id={product.product_id}), "
            f"price={product.price}원, weight={product.weight}g, "
            f"vision_conf={best_candidate.combined_confidence:.3f}, "
            f"final_conf={confidence:.3f}, count={count}, "
            f"yolo_class_id={best_candidate.class_id}, product_idx={product.product_idx}"
        )

        product_judgment = ProductJudgment(
            product_id=product.product_id,
            name=product.name,
            count=count,
            unit_price=product.price,
            total_price=product.price * count,
            confidence=confidence,
            unit_weight=product.weight,
        )

        return JudgmentResult(
            products=[product_judgment],
            total_price=product_judgment.total_price,
            confidence=confidence,
            status=JudgmentStatus.PARTIAL,  # 무게 미검증
            weight_delta=0.0,  # 무게 데이터 없음
            weight_explained=0.0,
            weight_residual=0.0,
            timestamp=timestamp,
        )

    def judge_by_weight_only(
        self,
        delta_weight: float,
        timestamp: Optional[float] = None,
        active_products: Optional[List] = None,  # v4.8: 추가
    ) -> JudgmentResult:
        """
        무게만으로 가장 가까운 상품 추정 (Vision 실패 시 폴백).

        v4.8: active_products가 있으면 Node.js에서 보낸 최신 무게 정보 우선 사용.
        has_loadcell 필드도 확인하여 로드셀 없는 상품은 제외.

        Args:
            delta_weight: 무게 변화량 (음수 = 제거)
            timestamp: 판단 시각 (기본값: 현재 시각)
            active_products: ActiveProductStore에서 가져온 상품 정보 (v4.8)

        Returns:
            JudgmentResult (status: PARTIAL 또는 UNCERTAIN)
        """
        if timestamp is None:
            timestamp = time.time()

        abs_weight = abs(delta_weight)

        logger.info(
            f"[ENGINE][reason=loadcell_only] delta_weight={delta_weight:.1f}g, "
            f"active_products={len(active_products) if active_products else 0}"
        )

        # 무게 변화가 너무 작은 경우
        if abs_weight < self.min_weight_change:
            logger.info(f"Weight change too small for fallback: {abs_weight:.1f}g")
            return self._create_no_detection_result(delta_weight, timestamp)

        # v4.8: active_products 우선 사용 (Node.js 최신 무게)
        # Normalize the live snapshot into a minimal candidate list before
        # doing any weight math so type quirks stay isolated here.
        candidate_products = self._build_weight_only_candidates(active_products)
        # v4.9: active_products가 없으면 no_detection 반환
        if candidate_products:
            logger.info(
                f"[LOADCELL-ONLY] using {len(candidate_products)} products "
                "from normalized active_products snapshot"
            )

        if not candidate_products:
            logger.warning(
                f"[ENGINE][reason=no_active_products] [LOADCELL-ONLY] v4.9: No active_products available "
                f"(delta_weight={delta_weight:.1f}g, input_count={len(active_products) if active_products else 0}), "
                f"ProductDB fallback disabled → no_detection"
            )
            return self._create_no_detection_result(delta_weight, timestamp)

        nearest_single = self._try_loadcell_nearest_single(
            delta_weight,
            timestamp,
            active_products,
        )
        if nearest_single is not None:
            return nearest_single

        logger.warning(
            "[ENGINE][reason=loadcell_only_no_single_within_5g] "
            f"delta_weight={delta_weight:.1f}g, "
            f"candidates={len(candidate_products)}, fail_closed=true"
        )
        return JudgmentResult(
            products=[],
            total_price=0,
            confidence=0.0,
            status=JudgmentStatus.UNCERTAIN,
            weight_delta=delta_weight,
            weight_explained=0.0,
            weight_residual=abs_weight,
            timestamp=timestamp,
        )

    def _try_loadcell_nearest_single(
        self,
        delta_weight: float,
        timestamp: float,
        active_products: Optional[List],
        require_unique_margin: bool = True,
    ) -> Optional[JudgmentResult]:
        """Return the nearest single active product when loadcell error is < 5g."""
        abs_weight = abs(delta_weight)
        candidates = self._build_weight_only_candidates(active_products)
        if not candidates:
            return None

        ranked = sorted(
            (
                (abs(abs_weight - product.weight), product)
                for product in candidates
            ),
            key=lambda item: item[0],
        )
        best_error, best_match = ranked[0]

        if best_match is None:
            return None

        if best_error >= 5.0:
            logger.info(
                "[ENGINE][reason=loadcell_nearest_single_rejected] "
                f"delta_weight={delta_weight:.1f}g, nearest={best_match.name}, "
                f"unit_weight={best_match.weight:.1f}g, residual={best_error:.1f}g"
            )
            return None

        if require_unique_margin and len(ranked) > 1:
            second_error = ranked[1][0]
            margin = second_error - best_error
            if margin < config.weight.nearest_single_margin_grams:
                logger.info(
                    "[ENGINE][reason=loadcell_nearest_single_ambiguous] "
                    f"delta_weight={delta_weight:.1f}g, nearest={best_match.name}, "
                    f"best_residual={best_error:.1f}g, second_residual={second_error:.1f}g, "
                    f"margin={margin:.1f}g"
                )
                return None

        match_score = max(0.0, 1.0 - (best_error / 5.0))
        confidence = match_score * 0.7
        product_judgment = ProductJudgment(
            product_id=best_match.product_id,
            name=best_match.name,
            count=1,
            unit_price=best_match.price,
            total_price=best_match.price,
            confidence=confidence,
            unit_weight=best_match.weight,
        )

        logger.info(
            "[ENGINE][reason=loadcell_nearest_single] "
            f"selected={best_match.name}, expected={best_match.weight:.1f}g, "
            f"actual={abs_weight:.1f}g, residual={best_error:.1f}g, "
            f"confidence={confidence:.3f}"
        )

        return JudgmentResult(
            products=[product_judgment],
            total_price=product_judgment.total_price,
            confidence=confidence,
            status=JudgmentStatus.PARTIAL,
            weight_delta=delta_weight,
            weight_explained=best_match.weight,
            weight_residual=best_error,
            timestamp=timestamp,
        )

    def _try_single_product_match(
        self,
        estimates: List[CountEstimate],
        delta_weight: float,
        timestamp: float,
    ) -> Optional[JudgmentResult]:
        """
        단일 상품 매칭 시도.

        검증된(validated=True) 추정 중 가장 높은 match_score를 선택.
        """
        validated_estimates = [e for e in estimates if e.validated]

        if not validated_estimates:
            return None

        best = validated_estimates[0]

        confidence = self._calculate_fusion_confidence(
            vision_score=best.vision_confidence,
            weight_score=best.match_score,
            count=best.count,
        )

        if confidence < self.confidence_threshold:
            logger.info(f"[ENGINE] 신뢰도 부족: {confidence:.3f} < {self.confidence_threshold}")
            return None

        product = self._create_product_judgment(best, confidence)

        logger.info(
            f"[ENGINE] 최종: status=COMPLETE, products=1, confidence={confidence:.3f}"
        )

        return JudgmentResult(
            products=[product],
            total_price=product.total_price,
            confidence=confidence,
            status=JudgmentStatus.COMPLETE,
            weight_delta=delta_weight,
            weight_explained=best.expected_weight,
            weight_residual=abs(abs(delta_weight) - best.expected_weight),
            timestamp=timestamp,
        )

    def _try_combination_match(
        self,
        candidates: List[EnsembleResult],
        delta_weight: float,
        timestamp: float,
        active_products: Optional[List] = None,
        trace_context: Optional[object] = None,
    ) -> Optional[JudgmentResult]:
        """다중 상품 조합 매칭 시도. v4.7: active_products 전달."""
        min_multi_kind_confidence = float(config.weight.multi_kind_min_confidence)
        weak_candidates = [
            {
                "class_id": candidate.class_id,
                "name": candidate.class_name,
                "confidence": round(float(candidate.combined_confidence), 4),
            }
            for candidate in candidates
            if candidate.combined_confidence < min_multi_kind_confidence
        ]
        combination = self.count_calculator.calculate_combination(
            candidates=candidates,
            delta_weight=delta_weight,
            max_combination_size=self.max_combination_size,
            active_products=active_products,
        )
        if not combination:
            candidate_strict_result = self._try_candidate_only_strict_combination_match(
                vision_candidates=candidates,
                delta_weight=delta_weight,
                timestamp=timestamp,
                active_products=active_products,
                trace_context=trace_context,
            )
            if candidate_strict_result is not None:
                return candidate_strict_result

        if not combination and trace_context is not None:
            stage_strict_result = self._try_stage_count_combination_match(
                vision_candidates=candidates,
                delta_weight=delta_weight,
                timestamp=timestamp,
                active_products=active_products,
                trace_context=trace_context,
            )
            if stage_strict_result is not None:
                return stage_strict_result

            merged_candidates, stage_diagnostics = (
                self._build_stage_count_combination_candidates(
                    vision_candidates=candidates,
                    trace_context=trace_context,
                )
            )
            if stage_diagnostics["stage_candidates_added"] > 0:
                combination = self.count_calculator.calculate_combination(
                    candidates=merged_candidates,
                    delta_weight=delta_weight,
                    max_combination_size=self.max_combination_size,
                    active_products=active_products,
                )
                stage_diagnostics["accepted"] = bool(combination)
                self._record_weight_diagnostics(
                    trace_context,
                    {"relaxed_stage_count_combination_match": stage_diagnostics},
                )

        if not combination:
            if weak_candidates:
                self._record_weight_diagnostics(
                    trace_context,
                    {
                        "relaxed_rejected": {
                            "rejected_combo_reason": "low_confidence_combo",
                            "weak_candidates": weak_candidates[:10],
                        }
                    },
                )
            return None

        products = []
        total_price = 0
        total_explained = 0.0

        for estimate in combination:
            confidence = self._calculate_fusion_confidence(
                vision_score=estimate.vision_confidence,
                weight_score=estimate.match_score,
                count=estimate.count,
            )
            product = self._create_product_judgment(estimate, confidence)
            products.append(product)
            total_price += product.total_price
            total_explained += estimate.expected_weight

        avg_confidence = sum(p.confidence for p in products) / len(products)

        logger.info(
            f"[ENGINE] 최종: status=COMPLETE, products={len(products)}, "
            f"confidence={avg_confidence:.3f}"
        )

        return JudgmentResult(
            products=products,
            total_price=total_price,
            confidence=avg_confidence,
            status=JudgmentStatus.COMPLETE,
            weight_delta=delta_weight,
            weight_explained=total_explained,
            weight_residual=abs(abs(delta_weight) - total_explained),
            timestamp=timestamp,
        )

    def _create_partial_result(
        self,
        estimates: List[CountEstimate],
        delta_weight: float,
        timestamp: float,
    ) -> JudgmentResult:
        """불완전 결과 생성."""
        # Keep the best available estimate so downstream aggregation can still
        # reason about likely products even when strict validation fails.
        if not estimates:
            return self._create_no_detection_result(delta_weight, timestamp)

        best = estimates[0]

        confidence = self._calculate_fusion_confidence(
            vision_score=best.vision_confidence,
            weight_score=best.match_score,
            count=best.count,
        )

        if best.match_score > self.partial_threshold:
            status = JudgmentStatus.PARTIAL
        else:
            status = JudgmentStatus.UNCERTAIN

        product = self._create_product_judgment(best, confidence)

        return JudgmentResult(
            products=[product],
            total_price=product.total_price,
            confidence=confidence,
            status=status,
            weight_delta=delta_weight,
            weight_explained=best.expected_weight,
            weight_residual=abs(abs(delta_weight) - best.expected_weight),
            timestamp=timestamp,
        )

    def _create_no_detection_result(
        self,
        delta_weight: float,
        timestamp: float,
    ) -> JudgmentResult:
        """감지된 상품 없음 결과 생성."""
        # Preserve the raw delta even for NO_DETECTION so operators can tell
        # whether inference failed after a meaningful weight change.
        return JudgmentResult(
            products=[],
            total_price=0,
            confidence=0.0,
            status=JudgmentStatus.NO_DETECTION,
            weight_delta=delta_weight,
            weight_explained=0.0,
            weight_residual=abs(delta_weight),
            timestamp=timestamp,
        )

    def _create_loadcell_identity_suppressed_result(
        self,
        *,
        delta_weight: float,
        timestamp: float,
        trace_context: Optional[object],
        branch: str,
    ) -> JudgmentResult:
        """Return no-charge result when vision-first policy blocks loadcell identity."""
        target_weight = abs(float(delta_weight))
        diagnostics = {
            "accepted": False,
            "policy": config.weight.identity_policy,
            "branch": branch,
            "target_weight": round(target_weight, 1),
            "reason": "vision_first_requires_product_evidence",
        }
        self._record_weight_diagnostics(
            trace_context,
            {
                "decision_branch": "loadcell_identity_suppressed",
                "loadcell_identity_suppressed": diagnostics,
            },
        )
        logger.info(
            "[ENGINE][reason=loadcell_identity_suppressed] "
            f"policy={config.weight.identity_policy}, target={target_weight:.1f}g"
        )
        return self._create_no_detection_result(delta_weight, timestamp)

    def _suppress_non_success_identity_result(
        self,
        result: JudgmentResult,
        *,
        trace_context: Optional[object],
        branch: str,
    ) -> JudgmentResult:
        """Clear product identity from non-chargeable vision-first results."""
        if result.is_success or not result.products:
            return result

        diagnostics = {
            "accepted": False,
            "policy": config.weight.identity_policy,
            "branch": branch,
            "status": result.status.value,
            "suppressed_products": [
                {
                    "product_id": int(product.product_id),
                    "name": product.name,
                    "count": int(product.count),
                }
                for product in result.products
            ],
            "reason": "non_success_result_cannot_carry_identity",
        }
        self._record_weight_diagnostics(
            trace_context,
            {"vision_first_identity_suppressed": diagnostics},
        )
        return JudgmentResult(
            products=[],
            total_price=0,
            confidence=0.0,
            status=result.status,
            weight_delta=result.weight_delta,
            weight_explained=0.0,
            weight_residual=abs(float(result.weight_delta)),
            timestamp=result.timestamp,
        )

    def _try_vision_first_identity_partial(
        self,
        *,
        vision_candidates: List[EnsembleResult],
        delta_weight: float,
        timestamp: float,
        active_products: Optional[List],
        trace_context: Optional[object],
    ) -> Optional[JudgmentResult]:
        """Preserve strong vision identity when loadcell validation conflicts."""
        if not self._is_vision_first_identity_policy():
            return None
        if delta_weight >= 0 or not vision_candidates or not active_products:
            return None

        active_map = self._active_products_by_class(active_products)
        target_weight = abs(float(delta_weight))
        if target_weight < self.min_weight_change:
            return None

        considered: list[dict[str, Any]] = []
        selected: dict[str, Any] | None = None
        for rank, candidate in enumerate(vision_candidates, start=1):
            candidate_diag: dict[str, Any] = {
                "rank": rank,
                "class_id": int(candidate.class_id),
                "name": candidate.class_name,
                "source": getattr(candidate, "source", "vision"),
                "confidence": round(float(candidate.combined_confidence), 4),
            }
            product = active_map.get(int(candidate.class_id))
            if product is None:
                candidate_diag["reason"] = "not_in_active_products"
                considered.append(candidate_diag)
                continue
            if not self._candidate_has_vision_identity_evidence(candidate):
                candidate_diag["reason"] = "insufficient_vision_identity_evidence"
                considered.append(candidate_diag)
                continue
            if not self._active_product_has_loadcell(product):
                candidate_diag["reason"] = "no_loadcell_product"
                considered.append(candidate_diag)
                continue

            unit_weight = self._coerce_float(getattr(product, "product_weight", 0.0))
            stock = self._coerce_int(getattr(product, "stock_qty", 0))
            if stock <= 0:
                candidate_diag["reason"] = "invalid_stock"
                considered.append(candidate_diag)
                continue
            if unit_weight <= 0:
                candidate_diag.update(
                    {
                        "unit_weight": round(unit_weight, 1),
                        "weight_status": "unavailable",
                        "count": 1,
                        "expected_weight": 0.0,
                        "residual": round(target_weight, 1),
                        "tolerance": 0.0,
                        "weight_validation_passed": False,
                        "reason": "selected_weight_unavailable",
                    }
                )
                selected = {
                    "candidate": candidate,
                    "product": product,
                    "count": 1,
                    "expected_weight": 0.0,
                    "residual": target_weight,
                    "tolerance": 0.0,
                    "weight_validation_passed": False,
                    "diagnostics": candidate_diag,
                }
                considered.append(candidate_diag)
                break

            nearest_count = max(1, int(round(target_weight / unit_weight)))
            count_cap = max(
                1,
                min(
                    stock,
                    int(config.weight.max_count_per_item),
                    max(2, int(config.weight.same_product_max_count)),
                ),
            )
            segment_grip_limit = self._segment_grip_limit_from_trace(trace_context)
            if segment_grip_limit is not None:
                count_cap = min(count_cap, segment_grip_limit)
            validation_count = min(nearest_count, count_cap)
            validation_expected_weight = unit_weight * validation_count
            validation_residual = abs(target_weight - validation_expected_weight)
            tolerance = self._full_delta_match_tolerance(
                [
                    ProductJudgment(
                        product_id=int(candidate.class_id),
                        name=candidate.class_name,
                        count=validation_count,
                        unit_price=self._coerce_int(getattr(product, "sale_price", 0)),
                        total_price=0,
                        unit_weight=unit_weight,
                        confidence=float(candidate.combined_confidence),
                    )
                ],
                branch="vision_first_identity_partial",
            )
            count_validation_reason = "validated"
            if nearest_count > count_cap:
                count_validation_reason = "count_exceeds_validation_limit"
            weight_validation_passed = (
                count_validation_reason == "validated"
                and validation_residual <= tolerance
            )
            count = validation_count if weight_validation_passed else 1
            expected_weight = unit_weight * count
            residual = abs(target_weight - expected_weight)
            candidate_diag.update(
                {
                    "unit_weight": round(unit_weight, 1),
                    "nearest_count": nearest_count,
                    "count_cap": count_cap,
                    "validation_count": validation_count,
                    "count_validation_reason": count_validation_reason,
                    "count": count,
                    "validation_expected_weight": round(validation_expected_weight, 1),
                    "validation_residual": round(validation_residual, 1),
                    "expected_weight": round(expected_weight, 1),
                    "residual": round(residual, 1),
                    "tolerance": round(tolerance, 1),
                    "weight_validation_passed": weight_validation_passed,
                    "reason": "selected",
                }
            )
            selected = {
                "candidate": candidate,
                "product": product,
                "count": count,
                "expected_weight": expected_weight,
                "residual": residual,
                "tolerance": tolerance,
                "weight_validation_passed": weight_validation_passed,
                "diagnostics": candidate_diag,
            }
            considered.append(candidate_diag)
            break

        if selected is None:
            self._record_weight_diagnostics(
                trace_context,
                {
                    "vision_first_identity_validation": {
                        "accepted": False,
                        "policy": config.weight.identity_policy,
                        "target_weight": round(target_weight, 1),
                        "reason": "no_supported_vision_identity",
                        "considered": considered,
                    }
                },
            )
            return None

        candidate = selected["candidate"]
        product = selected["product"]
        count = int(selected["count"])
        unit_price = self._coerce_int(getattr(product, "sale_price", 0))
        total_price = unit_price * count
        residual = float(selected["residual"])
        tolerance = float(selected["tolerance"])
        weight_validation_passed = bool(selected["weight_validation_passed"])
        weight_score = (
            max(0.0, 1.0 - residual / max(tolerance, 0.001))
            if weight_validation_passed
            else 0.0
        )
        confidence = self._calculate_fusion_confidence(
            vision_score=float(candidate.combined_confidence),
            weight_score=weight_score,
            count=count,
        )
        product_judgment = ProductJudgment(
            product_id=int(getattr(product, "yolo_class_id", candidate.class_id)),
            name=getattr(product, "product_name", candidate.class_name),
            count=count,
            unit_price=unit_price,
            total_price=total_price,
            confidence=confidence,
            unit_weight=self._coerce_float(getattr(product, "product_weight", 0.0)),
        )
        status = (
            JudgmentStatus.COMPLETE
            if weight_validation_passed
            else JudgmentStatus.PARTIAL
        )
        if weight_validation_passed:
            reason = "vision_identity_weight_validated"
        elif selected["diagnostics"].get("weight_status") == "unavailable":
            reason = "vision_identity_preserved_weight_unavailable"
        else:
            reason = "vision_identity_preserved_weight_mismatch"
        diagnostics = {
            "accepted": True,
            "policy": config.weight.identity_policy,
            "target_weight": round(target_weight, 1),
            "reason": reason,
            "weight_validation_passed": weight_validation_passed,
            "selected": selected["diagnostics"],
            "considered": considered,
        }
        self._record_weight_diagnostics(
            trace_context,
            {
                "decision_branch": "vision_first_identity_partial",
                "vision_first_identity_validation": diagnostics,
            },
        )
        logger.info(
            "[ENGINE][reason=vision_first_identity] "
            f"selected={product_judgment.name}, count={count}, "
            f"target={target_weight:.1f}g, expected={selected['expected_weight']:.1f}g, "
            f"residual={residual:.1f}g, status={status.value}"
        )
        return JudgmentResult(
            products=[product_judgment],
            total_price=total_price,
            confidence=confidence,
            status=status,
            weight_delta=delta_weight,
            weight_explained=float(selected["expected_weight"]),
            weight_residual=round(residual, 1),
            timestamp=timestamp,
        )

    def _enforce_full_delta_match(
        self,
        result: JudgmentResult,
        *,
        trace_context: Optional[object],
        branch: str,
    ) -> JudgmentResult:
        """Reject chargeable removal results that only explain a partial target."""
        if result.weight_delta >= 0 or not result.is_success:
            return result

        target_weight = abs(float(result.weight_delta))
        if target_weight < self.min_weight_change:
            return result

        explained_weight = self._result_explained_weight(result)
        residual = abs(target_weight - explained_weight)
        tolerance = self._full_delta_match_tolerance(
            result.products,
            branch=branch,
        )
        unit_count = max(0, sum(int(product.count) for product in result.products))
        accepted = residual <= tolerance

        diagnostics = {
            "accepted": accepted,
            "branch": branch,
            "target_weight": round(float(target_weight), 1),
            "explained_weight": round(float(explained_weight), 1),
            "residual": round(float(residual), 1),
            "tolerance": round(float(tolerance), 1),
            "unit_count": unit_count,
            "status_before_guard": result.status.value,
        }
        if accepted:
            diagnostics["reason"] = "full_delta_matched"
            self._record_weight_diagnostics(
                trace_context,
                {"final_weight_mismatch_guard": diagnostics},
            )
            return result

        diagnostics["reason"] = "result_does_not_explain_full_delta"
        self._record_weight_diagnostics(
            trace_context,
            {"final_weight_mismatch_guard": diagnostics},
        )
        logger.warning(
            "[ENGINE][reason=final_weight_mismatch_guard] "
            "branch=%s target=%.1fg explained=%.1fg residual=%.1fg "
            "tolerance=%.1fg",
            branch,
            target_weight,
            explained_weight,
            residual,
            tolerance,
        )
        return JudgmentResult(
            products=[],
            total_price=0,
            confidence=0.0,
            status=JudgmentStatus.UNCERTAIN,
            weight_delta=result.weight_delta,
            weight_explained=explained_weight,
            weight_residual=round(float(residual), 1),
            timestamp=result.timestamp,
        )

    @staticmethod
    def _result_explained_weight(result: JudgmentResult) -> float:
        product_weight = sum(
            float(product.unit_weight) * int(product.count)
            for product in result.products
            if float(product.unit_weight) > 0 and int(product.count) > 0
        )
        if product_weight > 0:
            return product_weight
        return max(0.0, float(result.weight_explained))

    @classmethod
    def _full_delta_match_tolerance(
        cls,
        products: list[ProductJudgment],
        *,
        branch: str,
    ) -> float:
        base_tolerance = max(0.0, float(config.weight.tolerance_grams))
        unit_count = max(0, sum(int(product.count) for product in products))
        if unit_count <= 1:
            if branch in {
                "detected_single_item_fallback",
                "forced_final_fallback",
                "strict_match",
            }:
                return max(
                    base_tolerance,
                    float(config.weight.detected_single_fallback_tolerance_grams),
                    float(config.weight.rescue_tolerance_grams),
                )
            return base_tolerance

        same_product_ids = {
            int(product.product_id)
            for product in products
            if int(product.count) > 0
        }
        bottle_repeat_extra = (
            1
            if (
                len(same_product_ids) == 1
                and any(
                    cls._is_500ml_bottle_weight(float(product.unit_weight))
                    for product in products
                )
            )
            else 0
        )
        return cls._count_scaled_weight_tolerance(
            unit_count,
            extra_units=bottle_repeat_extra,
        )

    @staticmethod
    def _get_delta_reason(delta_weight: float) -> str:
        if delta_weight < 0:
            return "negative_delta_weight(removal)"
        if delta_weight > 0:
            return "positive_delta_weight(return)"
        return "zero_delta_weight"

    @staticmethod
    def _is_vision_first_identity_policy() -> bool:
        return str(config.weight.identity_policy).lower() == "vision_first"

    @staticmethod
    def _is_freezer_mode() -> bool:
        return str(config.machine.cabinet_type).lower() == "freezer"

    @staticmethod
    def _active_products_by_class(active_products: Optional[List]) -> dict[int, object]:
        active_map: dict[int, object] = {}
        for product in active_products or []:
            class_id = getattr(product, "yolo_class_id", None)
            if class_id is None:
                continue
            try:
                active_map[int(class_id)] = product
            except (TypeError, ValueError):
                continue
        return active_map

    def _candidate_has_vision_identity_evidence(
        self,
        candidate: EnsembleResult,
    ) -> bool:
        source = str(getattr(candidate, "source", "vision") or "vision")
        confidence = float(getattr(candidate, "combined_confidence", 0.0) or 0.0)
        if source == "vision":
            return confidence >= max(0.0, float(self.confidence_threshold))
        if source in {"threshold_rescue", "roi_rescue"}:
            return bool(getattr(candidate, "weight_gate_passed", False))
        if source == "freezer_stage_exit_path":
            return (
                self._is_freezer_mode()
                and int(getattr(candidate, "freezer_exit_path_votes", 0) or 0)
                >= max(0, int(getattr(config.vision, "freezer_min_exit_path_votes", 3)))
                and confidence >= float(config.weight.multi_kind_min_confidence)
            )
        if source == "stage_weight_gate":
            return confidence >= float(config.weight.multi_kind_min_confidence)
        return False

    @staticmethod
    def _summarize_active_products(active_products: Optional[List]) -> tuple[int, int]:
        if not active_products:
            return 0, 0

        zero_stock_filtered = sum(
            1 for product in active_products
            if getattr(product, "stock_qty", 0) <= 0
        )
        return len(active_products), zero_stock_filtered

    @staticmethod
    def _scope_active_products_to_vision_candidates(
        active_products: Optional[List],
        vision_candidates: List[EnsembleResult],
    ) -> Optional[List]:
        """Limit relaxed fallback inventory to products seen by vision."""
        if not active_products or not vision_candidates:
            return active_products

        candidate_ids = {candidate.class_id for candidate in vision_candidates}
        scoped = [
            product
            for product in active_products
            if getattr(product, "yolo_class_id", None) in candidate_ids
        ]

        if len(scoped) != len(active_products):
            logger.info(
                "[ENGINE][reason=vision_scoped_fallback] "
                f"active_products={len(active_products)} -> scoped={len(scoped)}"
            )

        return scoped

    @staticmethod
    def _active_product_has_loadcell(product: object) -> bool:
        """Normalize `has_loadcell` values from Node.js into a bool."""
        raw_value = getattr(product, "has_loadcell", "true")
        if isinstance(raw_value, bool):
            return raw_value
        if raw_value is None:
            return False
        if isinstance(raw_value, str):
            return raw_value.strip().lower() not in {"", "0", "false", "null", "none"}
        return bool(raw_value)

    def _lookup_product_for_candidate(self, candidate: EnsembleResult):
        """Resolve a vision candidate against the configured product DB."""
        if self.product_db is None:
            return None

        product = self.product_db.get_by_yolo_class_id(candidate.class_id)
        if product is not None:
            return product

        product = self.product_db.get_by_yolo_class_name(candidate.class_name)
        if product is not None:
            logger.debug(
                f"Resolved by yolo_class_name: {candidate.class_name} -> "
                f"product_id={product.product_id}, name={product.name}"
            )
            return product

        product = self.product_db.get_product(candidate.class_id)
        if product is not None:
            logger.debug(
                f"Resolved by legacy product_id lookup: class_id={candidate.class_id} -> "
                f"name={product.name}"
            )
        return product

    def _build_weight_only_candidates(
        self,
        active_products: Optional[List],
    ) -> List[_WeightOnlyCandidate]:
        """Convert active-product snapshots into weight-only candidates."""
        if not active_products:
            return []

        candidates: List[_WeightOnlyCandidate] = []
        for active_product in active_products:
            if not self._active_product_has_loadcell(active_product):
                logger.debug(
                    f"[LOADCELL-ONLY] Skip no-loadcell product: "
                    f"{active_product.product_name}"
                )
                continue

            if (
                active_product.yolo_class_id is None
                or active_product.product_weight <= 0
                or active_product.stock_qty <= 0
            ):
                continue

            candidates.append(
                _WeightOnlyCandidate(
                    product_id=active_product.yolo_class_id,
                    product_idx=active_product.product_idx,
                    name=active_product.product_name,
                    weight=active_product.product_weight,
                    price=active_product.sale_price,
                    stock=active_product.stock_qty,
                )
            )

        return candidates

    def _log_final_branch(self, branch: str, result: JudgmentResult) -> None:
        logger.info(
            f"[ENGINE][final_branch={branch}] "
            f"status={result.status.value}, products={len(result.products)}, "
            f"confidence={result.confidence:.3f}"
        )

    @staticmethod
    def _record_weight_diagnostics(
        trace_context: Optional[object],
        diagnostics: dict,
    ) -> None:
        if trace_context is None or not hasattr(trace_context, "record_weight_diagnostics"):
            return
        existing = dict(getattr(trace_context, "weight_diagnostics", {}) or {})
        existing.update(diagnostics)
        trace_context.record_weight_diagnostics(existing)

    def _create_product_judgment(
        self,
        estimate: CountEstimate,
        confidence: float,
    ) -> ProductJudgment:
        """CountEstimate에서 ProductJudgment 생성. v4.7: estimate.unit_price 우선 사용."""
        # v4.7: active_products에서 가격이 있으면 우선 사용
        # Prefer live prices from the active-product snapshot over the static
        # catalog because stock and pricing can drift during operations.
        if estimate.unit_price > 0:
            price = estimate.unit_price
        elif self.product_db is not None:
            price = self.product_db.get_price(estimate.product_id)
        else:
            price = 0
        total_price = price * estimate.count

        return ProductJudgment(
            product_id=estimate.product_id,
            name=estimate.product_name,
            count=estimate.count,
            unit_price=price,
            total_price=total_price,
            confidence=confidence,
            unit_weight=estimate.unit_weight,
        )

    def _calculate_fusion_confidence(
        self,
        vision_score: float,
        weight_score: float,
        count: int,
    ) -> float:
        """
        퓨전 신뢰도 계산.

        가중 평균:
        - Vision 신뢰도: 40%
        - 무게 매칭 점수: 50%
        - 개수 합리성: 10%
        """
        # Clamp upstream values into the expected confidence domain first so
        # malformed inputs cannot skew the fused score.
        vision_normalized = min(max(vision_score, 0.0), 1.0)
        weight_normalized = min(max(weight_score, 0.0), 1.0)

        if count <= 3:
            count_score = 1.0
        else:
            count_score = max(0.0, 1.0 - (count - 3) * 0.1)

        vision_weight = float(config.weight.fusion_vision_weight)
        loadcell_weight = float(config.weight.fusion_loadcell_weight)
        count_weight = float(config.weight.fusion_count_weight)

        confidence = (
            vision_weight * vision_normalized +
            loadcell_weight * weight_normalized +
            count_weight * count_score
        )

        logger.debug("[ENGINE] 신뢰도 계산:")
        logger.debug(
            f"  vision={vision_normalized:.3f} × {vision_weight} = "
            f"{vision_normalized * vision_weight:.3f}"
        )
        logger.debug(
            f"  weight={weight_normalized:.3f} × {loadcell_weight} = "
            f"{weight_normalized * loadcell_weight:.3f}"
        )
        logger.debug(
            f"  count={count_score:.3f} × {count_weight} = "
            f"{count_score * count_weight:.3f}"
        )
        logger.debug(f"  total = {min(confidence, 1.0):.3f}")

        return min(confidence, 1.0)
