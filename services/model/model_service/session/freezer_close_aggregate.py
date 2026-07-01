"""Freezer close-time aggregate basket resolver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

from model_service.core.config import config

from .door_session import AggregatedProduct, DoorSession, TriggerResult


@dataclass(frozen=True)
class _Participant:
    zone: int
    session: DoorSession
    trigger: TriggerResult


@dataclass
class _CandidateGroup:
    product_id: int
    product_idx: Optional[str]
    name: str
    unit_weight: float
    unit_price: int
    stock_qty: int
    best_rank: int = 999
    best_confidence: float = 0.0
    evidence_count: int = 0
    zones: set[int] = field(default_factory=set)
    trigger_ids: set[str] = field(default_factory=set)

    @property
    def count_cap(self) -> int:
        caps = [
            max(1, int(config.weight.max_count_per_item)),
            max(1, int(config.weight.same_product_max_count)),
        ]
        if self.stock_qty > 0:
            caps.append(int(self.stock_qty))
        return max(1, min(caps))


@dataclass(frozen=True)
class _Combination:
    counts: Dict[int, int]
    total_weight: float
    residual: float
    total_count: int
    kind_count: int
    score: float
    rank_sum: int


class FreezerCloseAggregateResolver:
    """Re-solve unstable freezer close baskets and attribute them to last zone."""

    def __init__(
        self,
        *,
        get_product_weight: Optional[Callable[[int], float]] = None,
    ) -> None:
        self._get_product_weight = get_product_weight
        self._tolerance = max(
            0.0,
            float(config.weight.freezer_weight_tolerance_grams),
        )

    def apply(self, sessions: Dict[int, DoorSession]) -> Optional[dict[str, object]]:
        if str(config.machine.cabinet_type).strip().lower() != "freezer":
            return None

        participants = self._negative_participants(sessions.values())
        if not self._is_eligible(participants):
            return None

        raw_negative_total = sum(abs(float(item.trigger.delta_weight)) for item in participants)
        positive_hints = self._positive_hints(participants)
        output_zone = max(
            enumerate(participants),
            key=lambda indexed: (
                float(indexed[1].trigger.timestamp),
                float(indexed[1].session.last_trigger_at),
                int(indexed[0]),
            ),
        )[1].zone
        diagnostics = self._base_diagnostics(
            participants=participants,
            raw_negative_total=raw_negative_total,
            positive_hints=positive_hints,
            output_zone=output_zone,
        )

        candidate_groups = self._candidate_groups(participants)
        diagnostics["candidateCount"] = len(candidate_groups)
        combination = self._find_best_combination(
            candidate_groups,
            raw_negative_total,
            max_total_count=self._max_total_count(participants),
        )
        if combination is None:
            diagnostics.update(
                {
                    "accepted": False,
                    "reason": "no_weight_fit_for_freezer_close_aggregate",
                    "selectedProducts": [],
                }
            )
            self._record_diagnostics(participants, diagnostics)
            return diagnostics

        selected_counts = dict(combination.counts)
        selected_weight = float(combination.total_weight)
        matched_hints, unmatched_hints, matched_return_total = self._apply_return_hints(
            selected_counts,
            candidate_groups,
            raw_negative_total,
            selected_weight,
            positive_hints,
        )
        final_target = max(0.0, raw_negative_total - matched_return_total)
        final_weight = self._counts_weight(selected_counts, candidate_groups)
        final_residual = abs(final_target - final_weight)

        diagnostics.update(
            {
                "accepted": final_residual <= self._tolerance,
                "reason": (
                    "freezer_close_aggregate_applied"
                    if final_residual <= self._tolerance
                    else "freezer_close_aggregate_residual_exceeds_tolerance"
                ),
                "matchedReturnTotal": round(float(matched_return_total), 1),
                "finalTargetWeight": round(float(final_target), 1),
                "selectedWeight": round(float(final_weight), 1),
                "residual": round(float(final_residual), 1),
                "allowedResidual": round(float(self._tolerance), 1),
                "selectedProducts": self._selected_product_diagnostics(
                    selected_counts,
                    candidate_groups,
                ),
                "matchedPositiveHints": matched_hints,
                "unmatchedPositiveHints": unmatched_hints,
            }
        )
        if not diagnostics["accepted"]:
            self._record_diagnostics(participants, diagnostics)
            return diagnostics

        self._apply_output(
            participants,
            output_zone=output_zone,
            selected_counts=selected_counts,
            candidate_groups=candidate_groups,
            final_target=final_target,
            diagnostics=diagnostics,
        )
        return diagnostics

    @staticmethod
    def _negative_participants(
        sessions: Iterable[DoorSession],
    ) -> List[_Participant]:
        participants: List[_Participant] = []
        noise_floor = max(
            0.0,
            float(config.weight.freezer_weight_tolerance_grams),
        )
        for session in sessions:
            for trigger in session.triggers:
                if trigger.is_return or float(trigger.delta_weight) >= 0:
                    continue
                has_products = any(
                    int(getattr(product, "count", 0) or 0) > 0
                    for product in trigger.products
                )
                has_return_hint = bool(
                    getattr(trigger, "return_weight_hints", None)
                )
                if (
                    abs(float(trigger.delta_weight)) <= noise_floor
                    and not has_products
                    and not has_return_hint
                ):
                    continue
                participants.append(
                    _Participant(
                        zone=int(session.zone),
                        session=session,
                        trigger=trigger,
                    )
                )
        return participants

    @staticmethod
    def _is_eligible(participants: List[_Participant]) -> bool:
        if not participants:
            return False
        has_return_hint = any(
            bool(getattr(item.trigger, "return_weight_hints", None))
            for item in participants
        )
        zones = {item.zone for item in participants}
        return has_return_hint or len(participants) >= 2 or len(zones) >= 2

    @staticmethod
    def _positive_hints(participants: List[_Participant]) -> List[dict[str, object]]:
        hints: List[dict[str, object]] = []
        for item in participants:
            for hint in getattr(item.trigger, "return_weight_hints", []) or []:
                if not isinstance(hint, dict):
                    continue
                weight = _positive_hint_weight(hint)
                if weight <= 0:
                    continue
                hints.append(
                    {
                        "zone": item.zone,
                        "triggerId": item.trigger.trigger_id,
                        "weight": round(float(weight), 1),
                        "source": str(hint.get("source") or hint.get("reason") or "return_hint"),
                    }
                )
        return hints

    def _candidate_groups(
        self,
        participants: List[_Participant],
    ) -> Dict[int, _CandidateGroup]:
        groups: Dict[int, _CandidateGroup] = {}
        for item in participants:
            for candidate in getattr(item.trigger, "vision_candidates", []) or []:
                if not isinstance(candidate, dict):
                    continue
                parsed = self._candidate_from_snapshot(candidate)
                if parsed is None:
                    continue
                self._merge_candidate(groups, parsed, item, source_rank=parsed.best_rank)
            for product in item.trigger.products:
                if int(product.count) <= 0:
                    continue
                parsed = self._candidate_from_trigger_product(product)
                if parsed is None:
                    continue
                self._merge_candidate(groups, parsed, item, source_rank=999)
        return groups

    @staticmethod
    def _candidate_from_snapshot(
        candidate: dict[str, object],
    ) -> Optional[_CandidateGroup]:
        try:
            product_id = int(candidate.get("product_id"))
            unit_weight = float(candidate.get("unit_weight", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        if product_id < 0 or unit_weight <= 0:
            return None
        return _CandidateGroup(
            product_id=product_id,
            product_idx=(
                str(candidate.get("product_idx"))
                if candidate.get("product_idx") is not None
                else None
            ),
            name=str(candidate.get("name") or product_id),
            unit_weight=unit_weight,
            unit_price=int(candidate.get("unit_price", 0) or 0),
            stock_qty=int(candidate.get("stock_qty", 0) or 0),
            best_rank=int(candidate.get("rank", 999) or 999),
            best_confidence=float(candidate.get("confidence", 0.0) or 0.0),
        )

    def _candidate_from_trigger_product(
        self,
        product: object,
    ) -> Optional[_CandidateGroup]:
        product_id = int(getattr(product, "product_id", -1))
        if product_id < 0:
            return None
        unit_weight = 0.0
        if self._get_product_weight is not None:
            try:
                unit_weight = float(self._get_product_weight(product_id))
            except (TypeError, ValueError):
                unit_weight = 0.0
        if unit_weight <= 0:
            return None
        return _CandidateGroup(
            product_id=product_id,
            product_idx=getattr(product, "product_idx", None),
            name=str(getattr(product, "name", product_id)),
            unit_weight=unit_weight,
            unit_price=int(getattr(product, "price", 0) or 0),
            stock_qty=max(int(getattr(product, "count", 1) or 1), 1),
            best_confidence=float(getattr(product, "confidence", 0.0) or 0.0),
        )

    @staticmethod
    def _merge_candidate(
        groups: Dict[int, _CandidateGroup],
        candidate: _CandidateGroup,
        participant: _Participant,
        *,
        source_rank: int,
    ) -> None:
        group = groups.get(candidate.product_id)
        if group is None:
            group = candidate
            groups[candidate.product_id] = group
        else:
            group.best_rank = min(group.best_rank, source_rank)
            group.best_confidence = max(group.best_confidence, candidate.best_confidence)
            group.stock_qty = max(group.stock_qty, candidate.stock_qty)
            if group.unit_price <= 0 and candidate.unit_price > 0:
                group.unit_price = candidate.unit_price
            if group.product_idx is None and candidate.product_idx is not None:
                group.product_idx = candidate.product_idx
        group.evidence_count += 1
        group.zones.add(participant.zone)
        group.trigger_ids.add(participant.trigger.trigger_id)

    @staticmethod
    def _max_total_count(participants: List[_Participant]) -> int:
        return max(
            1,
            len(participants) * max(1, int(config.weight.max_items_per_segment)),
        )

    def _find_best_combination(
        self,
        candidate_groups: Dict[int, _CandidateGroup],
        target_weight: float,
        *,
        max_total_count: int,
    ) -> Optional[_Combination]:
        groups = sorted(
            candidate_groups.values(),
            key=lambda group: (group.best_rank, -group.best_confidence, group.product_id),
        )
        max_kinds = max(1, int(config.weight.max_combination_kinds))
        matches: List[_Combination] = []

        def search(
            index: int,
            counts: Dict[int, int],
            total_weight: float,
            total_count: int,
            kind_count: int,
        ) -> None:
            if total_weight > target_weight + self._tolerance:
                return
            if total_count > max_total_count or kind_count > max_kinds:
                return
            if index >= len(groups):
                if not counts:
                    return
                residual = abs(target_weight - total_weight)
                if residual > self._tolerance:
                    return
                matches.append(
                    _Combination(
                        counts=dict(counts),
                        total_weight=total_weight,
                        residual=residual,
                        total_count=total_count,
                        kind_count=kind_count,
                        score=sum(
                            candidate_groups[product_id].best_confidence * count
                            for product_id, count in counts.items()
                        ),
                        rank_sum=sum(
                            candidate_groups[product_id].best_rank * count
                            for product_id, count in counts.items()
                        ),
                    )
                )
                return

            group = groups[index]
            search(index + 1, counts, total_weight, total_count, kind_count)
            cap = min(group.count_cap, max_total_count - total_count)
            for count in range(1, cap + 1):
                counts[group.product_id] = count
                search(
                    index + 1,
                    counts,
                    total_weight + group.unit_weight * count,
                    total_count + count,
                    kind_count + 1,
                )
            counts.pop(group.product_id, None)

        search(0, {}, 0.0, 0, 0)
        if not matches:
            return None
        return sorted(
            matches,
            key=lambda item: (
                item.residual,
                item.total_count,
                item.kind_count,
                -item.score,
                item.rank_sum,
            ),
        )[0]

    def _apply_return_hints(
        self,
        selected_counts: Dict[int, int],
        candidate_groups: Dict[int, _CandidateGroup],
        raw_negative_total: float,
        selected_weight: float,
        positive_hints: List[dict[str, object]],
    ) -> tuple[List[dict[str, object]], List[dict[str, object]], float]:
        matched: List[dict[str, object]] = []
        unmatched: List[dict[str, object]] = []
        matched_return_total = 0.0
        current_weight = selected_weight
        for hint in positive_hints:
            hint_weight = float(hint["weight"])
            match = self._find_return_match(
                selected_counts,
                candidate_groups,
                hint_weight,
            )
            if match is None:
                unmatched.append({**hint, "reason": "no_selected_product_weight_match"})
                continue
            adjusted_target = max(0.0, raw_negative_total - matched_return_total - hint_weight)
            keep_residual = abs(adjusted_target - current_weight)
            new_weight = current_weight - match.total_weight
            new_residual = abs(adjusted_target - new_weight)
            if new_residual >= keep_residual or new_residual > self._tolerance:
                unmatched.append({**hint, "reason": "no_residual_improvement"})
                continue
            for product_id, count in match.counts.items():
                selected_counts[product_id] = selected_counts.get(product_id, 0) - count
                if selected_counts[product_id] <= 0:
                    selected_counts.pop(product_id, None)
            current_weight = new_weight
            matched_return_total += hint_weight
            matched.append(
                {
                    **hint,
                    "matchedWeight": round(float(match.total_weight), 1),
                    "products": self._selected_product_diagnostics(
                        match.counts,
                        candidate_groups,
                    ),
                    "residualAfter": round(float(new_residual), 1),
                }
            )
        return matched, unmatched, matched_return_total

    def _find_return_match(
        self,
        selected_counts: Dict[int, int],
        candidate_groups: Dict[int, _CandidateGroup],
        target_weight: float,
    ) -> Optional[_Combination]:
        subset = {
            product_id: candidate_groups[product_id]
            for product_id, count in selected_counts.items()
            if count > 0 and product_id in candidate_groups
        }
        if not subset:
            return None
        capped = {
            product_id: _copy_group_with_stock(group, selected_counts[product_id])
            for product_id, group in subset.items()
        }
        return self._find_best_combination(
            capped,
            target_weight,
            max_total_count=sum(selected_counts.values()),
        )

    @staticmethod
    def _counts_weight(
        counts: Dict[int, int],
        candidate_groups: Dict[int, _CandidateGroup],
    ) -> float:
        return sum(
            candidate_groups[product_id].unit_weight * count
            for product_id, count in counts.items()
            if product_id in candidate_groups
        )

    @staticmethod
    def _selected_product_diagnostics(
        counts: Dict[int, int],
        candidate_groups: Dict[int, _CandidateGroup],
    ) -> List[dict[str, object]]:
        products: List[dict[str, object]] = []
        for product_id in sorted(
            counts,
            key=lambda item: (
                candidate_groups[item].best_rank,
                candidate_groups[item].name,
            ),
        ):
            count = int(counts[product_id])
            if count <= 0:
                continue
            group = candidate_groups[product_id]
            products.append(
                {
                    "productId": int(product_id),
                    "name": group.name,
                    "count": count,
                }
            )
        return products

    def _apply_output(
        self,
        participants: List[_Participant],
        *,
        output_zone: int,
        selected_counts: Dict[int, int],
        candidate_groups: Dict[int, _CandidateGroup],
        final_target: float,
        diagnostics: dict[str, object],
    ) -> None:
        participant_sessions = {item.zone: item.session for item in participants}
        output_products = self._aggregated_products(selected_counts, candidate_groups)
        for zone, session in participant_sessions.items():
            zone_diagnostics = dict(diagnostics)
            zone_diagnostics["role"] = "output" if zone == output_zone else "rerouted"
            zone_diagnostics["weightDeltaOverride"] = (
                round(-float(final_target), 1) if zone == output_zone else 0.0
            )
            if zone != output_zone:
                zone_diagnostics["reroutedToZone"] = output_zone
            session.final_weight_validation = dict(session.final_weight_validation or {})
            session.final_weight_validation["freezerCloseAggregate"] = zone_diagnostics
            session.aggregated_products = (
                dict(output_products) if zone == output_zone else {}
            )

    @staticmethod
    def _aggregated_products(
        counts: Dict[int, int],
        candidate_groups: Dict[int, _CandidateGroup],
    ) -> Dict[int, AggregatedProduct]:
        products: Dict[int, AggregatedProduct] = {}
        for product_id, count in counts.items():
            if count <= 0 or product_id not in candidate_groups:
                continue
            group = candidate_groups[product_id]
            products[product_id] = AggregatedProduct(
                product_id=product_id,
                product_idx=group.product_idx,
                name=group.name,
                count=int(count),
                unit_price=int(group.unit_price),
                weight=float(group.unit_weight),
                total_confidence=float(group.best_confidence) * int(count),
                detection_count=int(count),
            )
        return products

    @staticmethod
    def _base_diagnostics(
        *,
        participants: List[_Participant],
        raw_negative_total: float,
        positive_hints: List[dict[str, object]],
        output_zone: int,
    ) -> dict[str, object]:
        zones = sorted({item.zone for item in participants})
        return {
            "accepted": False,
            "reason": "not_evaluated",
            "eligibilityReason": _eligibility_reason(participants),
            "participatingZones": zones,
            "participatingTriggers": [
                {
                    "zone": item.zone,
                    "triggerId": item.trigger.trigger_id,
                    "sessionId": item.trigger.session_id,
                    "deltaWeight": round(float(item.trigger.delta_weight), 1),
                    "returnHintWeight": round(
                        sum(
                            _positive_hint_weight(hint)
                            for hint in getattr(item.trigger, "return_weight_hints", []) or []
                            if isinstance(hint, dict)
                        ),
                        1,
                    ),
                }
                for item in participants
            ],
            "rawNegativeTotal": round(float(raw_negative_total), 1),
            "positiveHintTotal": round(
                sum(float(hint["weight"]) for hint in positive_hints),
                1,
            ),
            "outputZone": int(output_zone),
        }

    @staticmethod
    def _record_diagnostics(
        participants: List[_Participant],
        diagnostics: dict[str, object],
    ) -> None:
        for item in participants:
            item.session.final_weight_validation = dict(
                item.session.final_weight_validation or {}
            )
            item.session.final_weight_validation["freezerCloseAggregate"] = dict(
                diagnostics
            )


def _positive_hint_weight(hint: dict[str, object]) -> float:
    try:
        return abs(float(hint.get("delta", hint.get("weight", 0.0)) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _eligibility_reason(participants: List[_Participant]) -> str:
    if any(getattr(item.trigger, "return_weight_hints", None) for item in participants):
        return "return_weight_hints_present"
    zones = {item.zone for item in participants}
    if len(zones) >= 2:
        return "multiple_freezer_zones"
    if len(participants) >= 2:
        return "multiple_freezer_negative_triggers"
    return "not_eligible"


def _copy_group_with_stock(
    group: _CandidateGroup,
    stock_qty: int,
) -> _CandidateGroup:
    return _CandidateGroup(
        product_id=group.product_id,
        product_idx=group.product_idx,
        name=group.name,
        unit_weight=group.unit_weight,
        unit_price=group.unit_price,
        stock_qty=stock_qty,
        best_rank=group.best_rank,
        best_confidence=group.best_confidence,
        evidence_count=group.evidence_count,
        zones=set(group.zones),
        trigger_ids=set(group.trigger_ids),
    )
