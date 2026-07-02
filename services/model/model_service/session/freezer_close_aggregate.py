"""Freezer close-time signed net basket resolver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

from model_service.core.config import config

from .door_session import (
    AggregatedProduct,
    DoorSession,
    ReturnedPositionHint,
    TriggerResult,
)


@dataclass(frozen=True)
class _Participant:
    zone: int
    session: DoorSession
    trigger: TriggerResult

    @property
    def delta_weight(self) -> float:
        return float(self.trigger.delta_weight)

    @property
    def has_mixed_sign_diagnostics(self) -> bool:
        return _trigger_has_mixed_sign_diagnostics(self.trigger)


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
    """Re-solve unstable freezer close baskets from signed trigger net delta."""

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

        participants = self._participants(sessions.values())
        if not self._is_eligible(participants):
            return None

        global_net_delta = sum(item.delta_weight for item in participants)
        output_zone = self._output_zone(participants)
        diagnostics = self._base_diagnostics(
            participants=participants,
            global_net_delta=global_net_delta,
            output_zone=output_zone,
        )
        returned_hints = self._returned_position_hints(participants)
        if returned_hints:
            diagnostics["returnedPositionHints"] = [
                self._returned_hint_diagnostic(hint) for hint in returned_hints
            ]

        if abs(global_net_delta) <= self._tolerance:
            diagnostics.update(
                {
                    "accepted": True,
                    "reason": "freezer_close_aggregate_net_zero",
                    "noChargeReason": "signed_net_delta_within_tolerance",
                    "finalTargetWeight": 0.0,
                    "selectedWeight": 0.0,
                    "residual": 0.0,
                    "allowedResidual": round(float(self._tolerance), 1),
                    "selectedProducts": [],
                }
            )
            self._apply_no_charge_output(
                participants,
                output_zone=output_zone,
                weight_delta_override=0.0,
                diagnostics=diagnostics,
            )
            return diagnostics

        if global_net_delta > self._tolerance:
            diagnostics.update(
                {
                    "accepted": True,
                    "reason": "freezer_close_aggregate_net_return",
                    "noChargeReason": "signed_net_delta_is_positive",
                    "finalTargetWeight": 0.0,
                    "selectedWeight": 0.0,
                    "residual": 0.0,
                    "allowedResidual": round(float(self._tolerance), 1),
                    "selectedProducts": [],
                }
            )
            self._apply_no_charge_output(
                participants,
                output_zone=output_zone,
                weight_delta_override=0.0,
                diagnostics=diagnostics,
            )
            return diagnostics

        target_weight = abs(float(global_net_delta))
        (
            current_counts,
            current_groups,
            current_weight,
        ) = self._current_trigger_product_selection(participants)
        current_residual = abs(target_weight - current_weight)
        diagnostics.update(
            {
                "finalTargetWeight": round(float(target_weight), 1),
                "triggerSelectedWeight": round(float(current_weight), 1),
                "triggerSelectedResidual": round(float(current_residual), 1),
                "triggerSelectedProducts": self._selected_product_diagnostics(
                    current_counts,
                    current_groups,
                ),
            }
        )
        current_suppressed, current_allowed = self._selection_returned_position_context(
            current_counts,
            current_groups,
            participants,
            returned_hints,
        )
        if current_suppressed:
            diagnostics["returnedPositionSuppressedCandidates"] = current_suppressed
            diagnostics["triggerProductsPreserveBlockedByReturnedPosition"] = True
        if current_allowed:
            diagnostics["samePositionReturnedProductAllowed"] = current_allowed
        if current_counts and current_residual <= self._tolerance and not current_suppressed:
            diagnostics.update(
                {
                    "accepted": True,
                    "reason": "freezer_close_aggregate_trigger_products_preserved",
                    "selectedWeight": round(float(current_weight), 1),
                    "residual": round(float(current_residual), 1),
                    "allowedResidual": round(float(self._tolerance), 1),
                    "selectedProducts": self._selected_product_diagnostics(
                        current_counts,
                        current_groups,
                    ),
                }
            )
            self._apply_preserve_output(
                participants,
                output_zone=output_zone,
                diagnostics=diagnostics,
            )
            return diagnostics

        raw_candidate_groups = self._candidate_groups(participants)
        (
            candidate_groups,
            suppressed_candidates,
            allowed_candidates,
        ) = self._filter_returned_position_candidates(
            raw_candidate_groups,
            participants,
            returned_hints,
        )
        if suppressed_candidates:
            existing = list(diagnostics.get("returnedPositionSuppressedCandidates", []))
            diagnostics["returnedPositionSuppressedCandidates"] = [
                *existing,
                *suppressed_candidates,
            ]
        if allowed_candidates:
            existing_allowed = list(
                diagnostics.get("samePositionReturnedProductAllowed", [])
            )
            diagnostics["samePositionReturnedProductAllowed"] = [
                *existing_allowed,
                *allowed_candidates,
            ]
        diagnostics["rawCandidateCount"] = len(raw_candidate_groups)
        diagnostics["candidateCount"] = len(candidate_groups)
        combination = self._find_best_combination(
            candidate_groups,
            target_weight,
            max_total_count=self._max_total_count(participants),
        )
        if combination is None:
            diagnostics.update(
                {
                    "accepted": False,
                    "reason": "no_weight_fit_for_freezer_close_aggregate",
                    "noChargeReason": "no_candidate_combination_for_signed_net_delta",
                    "selectedWeight": 0.0,
                    "residual": round(float(target_weight), 1),
                    "allowedResidual": round(float(self._tolerance), 1),
                    "selectedProducts": [],
                }
            )
            self._apply_no_charge_output(
                participants,
                output_zone=output_zone,
                weight_delta_override=round(float(global_net_delta), 1),
                diagnostics=diagnostics,
            )
            return diagnostics

        selected_counts = dict(combination.counts)
        selected_weight = float(combination.total_weight)
        residual = abs(target_weight - selected_weight)
        diagnostics.update(
            {
                "accepted": residual <= self._tolerance,
                "reason": (
                    "freezer_close_aggregate_applied"
                    if residual <= self._tolerance
                    else "freezer_close_aggregate_residual_exceeds_tolerance"
                ),
                "selectedWeight": round(float(selected_weight), 1),
                "residual": round(float(residual), 1),
                "allowedResidual": round(float(self._tolerance), 1),
                "selectedProducts": self._selected_product_diagnostics(
                    selected_counts,
                    candidate_groups,
                ),
            }
        )
        if residual > self._tolerance:
            diagnostics["noChargeReason"] = "candidate_combination_residual_exceeds_tolerance"
            self._apply_no_charge_output(
                participants,
                output_zone=output_zone,
                weight_delta_override=round(float(global_net_delta), 1),
                diagnostics=diagnostics,
            )
            return diagnostics

        self._apply_output(
            participants,
            output_zone=output_zone,
            selected_counts=selected_counts,
            candidate_groups=candidate_groups,
            output_delta=global_net_delta,
            diagnostics=diagnostics,
        )
        return diagnostics

    def _participants(
        self,
        sessions: Iterable[DoorSession],
    ) -> List[_Participant]:
        participants: List[_Participant] = []
        noise_floor = self._tolerance
        for session in sessions:
            for trigger in session.triggers:
                delta = float(trigger.delta_weight)
                has_products = any(
                    int(getattr(product, "count", 0) or 0) > 0
                    for product in trigger.products
                )
                has_mixed_sign = _trigger_has_mixed_sign_diagnostics(trigger)
                if (
                    abs(delta) <= noise_floor
                    and not has_products
                    and not has_mixed_sign
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

    def _is_eligible(self, participants: List[_Participant]) -> bool:
        if not participants:
            return False
        if any(item.has_mixed_sign_diagnostics for item in participants):
            return True
        zones = {item.zone for item in participants}
        if len(zones) >= 2:
            return True
        meaningful_count = sum(
            1
            for item in participants
            if abs(item.delta_weight) > self._tolerance
            or _trigger_has_products_or_candidates(item.trigger)
        )
        return meaningful_count >= 2

    @staticmethod
    def _output_zone(participants: List[_Participant]) -> int:
        return max(
            enumerate(participants),
            key=lambda indexed: (
                float(indexed[1].trigger.timestamp),
                float(indexed[1].session.last_trigger_at),
                int(indexed[0]),
            ),
        )[1].zone

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
    def _returned_position_hints(
        participants: List[_Participant],
    ) -> List[ReturnedPositionHint]:
        hints: List[ReturnedPositionHint] = []
        seen: set[tuple] = set()
        for item in participants:
            for hint in getattr(item.session, "returned_position_hints", []) or []:
                if not isinstance(hint, ReturnedPositionHint):
                    continue
                key = (
                    int(hint.product_id),
                    hint.product_idx,
                    hint.zone,
                    hint.channel_side,
                    hint.channel_index,
                    hint.channel_position,
                    hint.trigger_id,
                    hint.source,
                    hint.reason,
                )
                if key in seen:
                    continue
                seen.add(key)
                hints.append(hint)
        return hints

    @staticmethod
    def _returned_hint_diagnostic(
        hint: ReturnedPositionHint,
    ) -> dict[str, object]:
        return {
            "productId": int(hint.product_id),
            "productIdx": hint.product_idx,
            "name": hint.name,
            "count": int(hint.count),
            "unitWeight": round(float(hint.unit_weight), 1),
            "zone": hint.zone,
            "channelSide": hint.channel_side,
            "channelIndex": hint.channel_index,
            "channelPosition": hint.channel_position,
            "triggerId": hint.trigger_id,
            "sessionId": hint.session_id,
            "source": hint.source,
            "reason": hint.reason,
            "timestamp": round(float(hint.timestamp), 3),
        }

    def _filter_returned_position_candidates(
        self,
        candidate_groups: Dict[int, _CandidateGroup],
        participants: List[_Participant],
        returned_hints: List[ReturnedPositionHint],
    ) -> tuple[
        Dict[int, _CandidateGroup],
        List[dict[str, object]],
        List[dict[str, object]],
    ]:
        if not returned_hints:
            return dict(candidate_groups), [], []

        filtered: Dict[int, _CandidateGroup] = {}
        suppressed: List[dict[str, object]] = []
        allowed: List[dict[str, object]] = []
        for product_id, group in candidate_groups.items():
            context = self._returned_position_context_for_group(
                group,
                participants,
                returned_hints,
            )
            if context.get("suppressed"):
                suppressed.append(context["suppressed"])
                continue
            if context.get("allowed"):
                allowed.append(context["allowed"])
            filtered[product_id] = group
        return filtered, suppressed, allowed

    def _selection_returned_position_context(
        self,
        counts: Dict[int, int],
        candidate_groups: Dict[int, _CandidateGroup],
        participants: List[_Participant],
        returned_hints: List[ReturnedPositionHint],
    ) -> tuple[List[dict[str, object]], List[dict[str, object]]]:
        if not returned_hints:
            return [], []

        suppressed: List[dict[str, object]] = []
        allowed: List[dict[str, object]] = []
        for product_id, count in counts.items():
            if int(count) <= 0 or product_id not in candidate_groups:
                continue
            context = self._returned_position_context_for_group(
                candidate_groups[product_id],
                participants,
                returned_hints,
            )
            if context.get("suppressed"):
                suppressed.append(context["suppressed"])
            elif context.get("allowed"):
                allowed.append(context["allowed"])
        return suppressed, allowed

    def _returned_position_context_for_group(
        self,
        group: _CandidateGroup,
        participants: List[_Participant],
        returned_hints: List[ReturnedPositionHint],
    ) -> dict[str, dict[str, object]]:
        matching_hints = [
            hint
            for hint in returned_hints
            if int(hint.product_id) == int(group.product_id)
        ]
        if not matching_hints:
            return {}

        evidence_participants = [
            item
            for item in participants
            if item.trigger.trigger_id in group.trigger_ids
        ]
        for hint in matching_hints:
            hint_key = self._hint_position_key(hint)
            if hint_key is None:
                continue
            for item in evidence_participants:
                if hint_key in self._removal_position_keys(item.trigger, item.zone):
                    return {
                        "allowed": {
                            "productId": int(group.product_id),
                            "name": group.name,
                            "reason": "same_position_returned_product_allowed",
                            "hint": self._returned_hint_diagnostic(hint),
                            "triggerIds": sorted(group.trigger_ids),
                        }
                    }

        first_hint = matching_hints[0]
        return {
            "suppressed": {
                "productId": int(group.product_id),
                "name": group.name,
                "reason": "returned_original_position_different_or_unknown_target",
                "hint": self._returned_hint_diagnostic(first_hint),
                "triggerIds": sorted(group.trigger_ids),
            }
        }

    @classmethod
    def _hint_position_key(
        cls,
        hint: ReturnedPositionHint,
    ) -> Optional[tuple]:
        return cls._position_key(
            zone=hint.zone,
            channel_side=hint.channel_side,
            channel_index=hint.channel_index,
            channel_position=hint.channel_position,
        )

    @classmethod
    def _removal_position_keys(
        cls,
        trigger: TriggerResult,
        zone: int,
    ) -> set[tuple]:
        diagnostics = getattr(trigger, "loadcell_diagnostics", {}) or {}
        if not isinstance(diagnostics, dict):
            return set()
        raw_targets = list(diagnostics.get("channel_removal_segment_targets") or [])
        if not raw_targets:
            raw_targets = list(diagnostics.get("channel_movement_targets") or [])

        keys: set[tuple] = set()
        for entry in raw_targets:
            if not isinstance(entry, dict):
                continue
            try:
                delta = float(entry.get("delta", 0.0) or 0.0)
            except (TypeError, ValueError):
                delta = 0.0
            direction = str(entry.get("direction", "")).lower()
            if delta >= 0 and direction != "removal":
                continue
            key = cls._position_key(
                zone=zone,
                channel_side=entry.get("channel_side") or entry.get("channelSide"),
                channel_index=entry.get("channel_index", entry.get("channelIndex")),
                channel_position=entry.get(
                    "channel_position",
                    entry.get("channelPosition"),
                ),
            )
            if key is not None:
                keys.add(key)
        return keys

    @staticmethod
    def _position_key(
        *,
        zone: Optional[int],
        channel_side: object,
        channel_index: object,
        channel_position: object,
    ) -> Optional[tuple]:
        if zone is None:
            return None
        side = str(channel_side).strip().lower() if channel_side is not None else None
        if side in {"", "unknown", "none", "null"}:
            side = None
        try:
            index = int(channel_index) if channel_index is not None else None
        except (TypeError, ValueError):
            index = None
        try:
            position = (
                int(channel_position)
                if channel_position is not None
                else None
            )
        except (TypeError, ValueError):
            position = None
        if side is None and index is None and position is None:
            return None
        return (int(zone), side, index, position)

    @classmethod
    def _candidate_from_snapshot(
        cls,
        candidate: dict[str, object],
    ) -> Optional[_CandidateGroup]:
        if str(candidate.get("source", "vision") or "vision") != "vision":
            return None
        if not cls._snapshot_identity_confidence_passed(candidate):
            return None
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
            best_confidence=float(
                candidate.get(
                    "identity_confidence",
                    candidate.get("confidence", 0.0),
                )
                or 0.0
            ),
        )

    @staticmethod
    def _snapshot_identity_confidence_passed(candidate: dict[str, object]) -> bool:
        try:
            top_confidence = float(candidate.get("top_confidence", 0.0) or 0.0)
            side_confidence = float(candidate.get("side_confidence", 0.0) or 0.0)
            identity_confidence = float(
                candidate.get(
                    "identity_confidence",
                    candidate.get("confidence", 0.0),
                )
                or 0.0
            )
        except (TypeError, ValueError):
            return False

        top_detected = bool(candidate.get("top")) or top_confidence > 0.0
        side_detected = bool(candidate.get("side")) or side_confidence > 0.0
        top_threshold = _snapshot_threshold_for_camera("top")
        side_threshold = _snapshot_threshold_for_camera("side")
        if top_detected and top_confidence >= top_threshold:
            return True
        if side_detected and side_confidence >= side_threshold:
            return True
        if not top_detected and not side_detected:
            return identity_confidence >= min(top_threshold, side_threshold)
        return False

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
            stock_qty=0,
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
        ordered_matches = sorted(
            matches,
            key=lambda item: (
                item.total_count,
                item.kind_count,
                item.rank_sum,
                item.residual,
                -item.score,
            ),
        )
        best = ordered_matches[0]
        if (
            bool(config.weight.freezer_distinct_mixed_preference_enabled)
            and best.kind_count == 1
            and best.total_count > 1
        ):
            max_extra_residual = max(
                0.0,
                float(config.weight.freezer_distinct_mixed_max_extra_residual_grams),
            )
            distinct_mixed = [
                item
                for item in matches
                if item.total_count == best.total_count
                and item.kind_count == item.total_count
                and all(int(count) == 1 for count in item.counts.values())
                and float(item.residual) <= float(best.residual) + max_extra_residual
            ]
            if distinct_mixed:
                return sorted(
                    distinct_mixed,
                    key=lambda item: (
                        item.residual,
                        item.rank_sum,
                        -item.score,
                    ),
                )[0]
        return best

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
        output_delta: float,
        diagnostics: dict[str, object],
    ) -> None:
        participant_sessions = {item.zone: item.session for item in participants}
        output_products = self._aggregated_products(selected_counts, candidate_groups)
        for zone, session in participant_sessions.items():
            zone_diagnostics = dict(diagnostics)
            zone_diagnostics["role"] = "output" if zone == output_zone else "rerouted"
            zone_diagnostics["weightDeltaOverride"] = (
                round(float(output_delta), 1) if zone == output_zone else 0.0
            )
            if zone != output_zone:
                zone_diagnostics["reroutedToZone"] = output_zone
            session.final_weight_validation = dict(session.final_weight_validation or {})
            session.final_weight_validation["freezerCloseAggregate"] = zone_diagnostics
            session.aggregated_products = (
                dict(output_products) if zone == output_zone else {}
            )

    def _apply_preserve_output(
        self,
        participants: List[_Participant],
        *,
        output_zone: int,
        diagnostics: dict[str, object],
    ) -> None:
        preserved_products_by_zone = self._preserved_trigger_products_by_zone(
            participants
        )
        selected_products_by_zone = [
            {
                "zone": int(zone),
                "products": self._aggregated_product_diagnostics(products),
            }
            for zone, products in sorted(preserved_products_by_zone.items())
        ]
        for item in participants:
            zone_diagnostics = dict(diagnostics)
            zone_diagnostics["role"] = "preserved"
            zone_diagnostics["selectedProductsByZone"] = selected_products_by_zone
            if item.zone == output_zone:
                zone_diagnostics["outputZoneRole"] = "latest_trigger_zone"
            item.session.final_weight_validation = dict(
                item.session.final_weight_validation or {}
            )
            item.session.final_weight_validation[
                "freezerCloseAggregate"
            ] = zone_diagnostics
            item.session.aggregated_products = dict(
                preserved_products_by_zone.get(item.zone, {})
            )

    def _apply_no_charge_output(
        self,
        participants: List[_Participant],
        *,
        output_zone: int,
        weight_delta_override: float,
        diagnostics: dict[str, object],
    ) -> None:
        for item in participants:
            zone_diagnostics = dict(diagnostics)
            zone_diagnostics["role"] = (
                "output" if item.zone == output_zone else "rerouted"
            )
            zone_diagnostics["weightDeltaOverride"] = (
                round(float(weight_delta_override), 1)
                if item.zone == output_zone
                else 0.0
            )
            if item.zone != output_zone:
                zone_diagnostics["reroutedToZone"] = output_zone
            item.session.final_weight_validation = dict(
                item.session.final_weight_validation or {}
            )
            item.session.final_weight_validation[
                "freezerCloseAggregate"
            ] = zone_diagnostics
            item.session.aggregated_products = {}

    def _current_trigger_product_selection(
        self,
        participants: List[_Participant],
    ) -> tuple[Dict[int, int], Dict[int, _CandidateGroup], float]:
        counts: Dict[int, int] = {}
        groups: Dict[int, _CandidateGroup] = {}
        for item in participants:
            for product in item.trigger.products:
                count = int(getattr(product, "count", 0) or 0)
                if count <= 0:
                    continue
                parsed = self._candidate_from_trigger_product(product)
                if parsed is None:
                    continue
                product_id = int(parsed.product_id)
                counts[product_id] = counts.get(product_id, 0) + count
                self._merge_candidate(groups, parsed, item, source_rank=999)
        selected_weight = sum(
            groups[product_id].unit_weight * count
            for product_id, count in counts.items()
            if product_id in groups
        )
        return counts, groups, float(selected_weight)

    def _preserved_trigger_products_by_zone(
        self,
        participants: List[_Participant],
    ) -> Dict[int, Dict[int, AggregatedProduct]]:
        products_by_zone: Dict[int, Dict[int, AggregatedProduct]] = {}
        for item in participants:
            zone_products = products_by_zone.setdefault(int(item.zone), {})
            for product in item.trigger.products:
                count = int(getattr(product, "count", 0) or 0)
                if count <= 0:
                    continue
                parsed = self._candidate_from_trigger_product(product)
                if parsed is None:
                    continue
                product_id = int(parsed.product_id)
                existing = zone_products.get(product_id)
                product_confidence = float(getattr(product, "confidence", 0.0) or 0.0)
                placement_units = [
                    dict(unit)
                    for unit in getattr(product, "placement_units", []) or []
                    if isinstance(unit, dict)
                ][:count]
                if existing is None:
                    zone_products[product_id] = AggregatedProduct(
                        product_id=product_id,
                        product_idx=parsed.product_idx,
                        name=parsed.name,
                        count=count,
                        unit_price=int(parsed.unit_price),
                        weight=float(parsed.unit_weight),
                        total_confidence=product_confidence * count,
                        detection_count=count,
                        placement_units=placement_units,
                    )
                    continue

                existing.count += count
                existing.total_confidence += product_confidence * count
                existing.detection_count += count
                existing.placement_units.extend(placement_units)
                if existing.unit_price <= 0 and parsed.unit_price > 0:
                    existing.unit_price = int(parsed.unit_price)
                if existing.product_idx is None and parsed.product_idx is not None:
                    existing.product_idx = parsed.product_idx
        return products_by_zone

    @staticmethod
    def _aggregated_product_diagnostics(
        products: Dict[int, AggregatedProduct],
    ) -> List[dict[str, object]]:
        diagnostics: List[dict[str, object]] = []
        for product in sorted(
            products.values(),
            key=lambda item: (item.name, int(item.product_id)),
        ):
            if int(product.count) <= 0:
                continue
            diagnostics.append(
                {
                    "productId": int(product.product_id),
                    "name": product.name,
                    "count": int(product.count),
                }
            )
        return diagnostics

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
        global_net_delta: float,
        output_zone: int,
    ) -> dict[str, object]:
        zones = sorted({item.zone for item in participants})
        return {
            "policy": "signed_net_delta",
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
                    "isReturn": bool(item.trigger.is_return),
                    "productCount": sum(
                        int(getattr(product, "count", 0) or 0)
                        for product in item.trigger.products
                    ),
                    "candidateCount": len(
                        getattr(item.trigger, "vision_candidates", []) or []
                    ),
                    "mixedSignInternalSegments": item.has_mixed_sign_diagnostics,
                    "netDeltaWeight": _diagnostic_float(
                        item.trigger,
                        "net_delta_weight",
                    ),
                    "decisionDeltaWeight": _diagnostic_float(
                        item.trigger,
                        "decision_delta_weight",
                    ),
                }
                for item in participants
            ],
            "globalNetDelta": round(float(global_net_delta), 1),
            "outputZone": int(output_zone),
        }


def _trigger_has_products_or_candidates(trigger: TriggerResult) -> bool:
    if any(int(getattr(product, "count", 0) or 0) > 0 for product in trigger.products):
        return True
    return any(
        isinstance(candidate, dict)
        for candidate in getattr(trigger, "vision_candidates", []) or []
    )


def _trigger_has_mixed_sign_diagnostics(trigger: TriggerResult) -> bool:
    diagnostics = getattr(trigger, "loadcell_diagnostics", {}) or {}
    if not isinstance(diagnostics, dict):
        return False
    if bool(diagnostics.get("mixed_sign_internal_segments")):
        return True
    try:
        positive_count = int(diagnostics.get("compound_positive_segment_count", 0) or 0)
        negative_count = int(diagnostics.get("compound_negative_segment_count", 0) or 0)
    except (TypeError, ValueError):
        return False
    return positive_count > 0 and negative_count > 0


def _diagnostic_float(
    trigger: TriggerResult,
    key: str,
) -> Optional[float]:
    diagnostics = getattr(trigger, "loadcell_diagnostics", {}) or {}
    if not isinstance(diagnostics, dict) or key not in diagnostics:
        return None
    try:
        return round(float(diagnostics[key]), 1)
    except (TypeError, ValueError):
        return None


def _snapshot_threshold_for_camera(camera: str) -> float:
    normalized = (camera or "top").strip().lower()
    if (
        str(config.machine.cabinet_type).strip().lower() == "freezer"
        and str(config.vision.camera_layout).strip().lower() == "dual_top_proxy"
        and normalized in {"top", "side"}
    ):
        return float(config.vision.top_confidence_threshold)
    if normalized == "side":
        return float(config.vision.side_confidence_threshold)
    return float(config.vision.top_confidence_threshold)


def _eligibility_reason(participants: List[_Participant]) -> str:
    if any(item.has_mixed_sign_diagnostics for item in participants):
        return "mixed_sign_internal_segments"
    zones = {item.zone for item in participants}
    if len(zones) >= 2:
        return "multiple_freezer_zones"
    if len(participants) >= 2:
        return "multiple_freezer_triggers"
    return "not_eligible"
