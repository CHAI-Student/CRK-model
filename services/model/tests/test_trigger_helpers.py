"""
Trigger API helper functions tests.

Tests:
- 로드셀 값 파싱
- 안정 구간 감지
- 무게 변화량 계산
"""

import pytest
from pydantic import BaseModel
from typing import List


# Mock LoadcellData for testing (same structure as in trigger.py)
class LoadcellData(BaseModel):
    timestamp: str
    raw_value: List[str]
    filtered_value: List[str]
    filter_method: str = "none"


class TestParseLoadcellValue:
    """로드셀 값 파싱 테스트."""

    def test_parse_positive_value(self):
        """양수 값 파싱 테스트."""
        from model_service.api.routes.trigger import _parse_loadcell_value

        assert _parse_loadcell_value("+12345") == 12345.0
        assert _parse_loadcell_value("+500") == 500.0
        assert _parse_loadcell_value("+0") == 0.0

    def test_parse_negative_value(self):
        """음수 값 파싱 테스트."""
        from model_service.api.routes.trigger import _parse_loadcell_value

        assert _parse_loadcell_value("-12345") == -12345.0
        assert _parse_loadcell_value("-500") == -500.0

    def test_parse_plain_number(self):
        """부호 없는 숫자 파싱 테스트."""
        from model_service.api.routes.trigger import _parse_loadcell_value

        assert _parse_loadcell_value("12345") == 12345.0
        assert _parse_loadcell_value("0") == 0.0

    def test_parse_with_whitespace(self):
        """공백 포함 값 파싱 테스트."""
        from model_service.api.routes.trigger import _parse_loadcell_value

        assert _parse_loadcell_value("  +12345  ") == 12345.0
        assert _parse_loadcell_value(" -500 ") == -500.0

    def test_parse_invalid_value(self):
        """잘못된 값 파싱 테스트."""
        from model_service.api.routes.trigger import _parse_loadcell_value

        assert _parse_loadcell_value("abc") == 0.0
        assert _parse_loadcell_value("") == 0.0
        assert _parse_loadcell_value(None) == 0.0


class TestCalculateWeightDelta:
    """무게 변화량 계산 테스트."""

    def test_calculate_weight_delta_removal(self, sample_loadcells):
        """상품 제거 시 무게 변화량 테스트 (음수)."""
        from model_service.api.routes.trigger import _calculate_weight_delta

        loadcells = [LoadcellData(**lc) for lc in sample_loadcells]
        delta = _calculate_weight_delta(loadcells)

        # 5000g -> 4500g = -500g
        assert delta < 0
        assert -600 < delta < -400  # 약 -500g 근처

    def test_calculate_weight_delta_empty(self):
        """빈 로드셀 데이터 테스트."""
        from model_service.api.routes.trigger import _calculate_weight_delta

        delta = _calculate_weight_delta([])
        assert delta == 0.0

    def test_calculate_weight_delta_single(self):
        """단일 데이터 테스트."""
        from model_service.api.routes.trigger import _calculate_weight_delta

        loadcells = [
            LoadcellData(
                timestamp="2026-02-01T14:30:00.000Z",
                raw_value=["+5000"],
                filtered_value=["+5000"],
                filter_method="none",
            )
        ]
        delta = _calculate_weight_delta(loadcells)
        assert delta == 0.0

    def test_calculate_weight_delta_no_change(self):
        """무게 변화 없음 테스트."""
        from model_service.api.routes.trigger import _calculate_weight_delta

        loadcells = []
        for i in range(20):
            loadcells.append(
                LoadcellData(
                    timestamp=f"2026-02-01T14:30:{i:02d}.000Z",
                    raw_value=[f"+{5000 + (i % 5)}"],
                    filtered_value=[f"+{5000 + (i % 3)}"],
                    filter_method="none",
                )
            )

        delta = _calculate_weight_delta(loadcells)
        assert abs(delta) < 50  # 거의 변화 없음

    def test_calculate_weight_delta_addition(self):
        """상품 추가 시 무게 변화량 테스트 (양수)."""
        from model_service.api.routes.trigger import _calculate_weight_delta

        loadcells = []

        # Start: 3000g
        for i in range(10):
            loadcells.append(
                LoadcellData(
                    timestamp=f"2026-02-01T14:30:{i:02d}.000Z",
                    raw_value=["+3000", "+3000"],
                    filtered_value=["+3000", "+3000"],
                    filter_method="none",
                )
            )

        # End: 3500g (500g 추가)
        for i in range(10):
            loadcells.append(
                LoadcellData(
                    timestamp=f"2026-02-01T14:30:{10+i:02d}.000Z",
                    raw_value=["+3500", "+3500"],
                    filtered_value=["+3500", "+3500"],
                    filter_method="none",
                )
            )

        delta = _calculate_weight_delta(loadcells)
        assert delta > 0
        assert 400 < delta < 600  # 약 +500g 근처


class TestDetectStableRegions:
    """안정 구간 감지 테스트."""

    def test_detect_stable_regions_valid(self, sample_loadcells):
        """유효한 안정 구간 감지 테스트."""
        from model_service.api.routes.trigger import _detect_stable_regions

        loadcells = [LoadcellData(**lc) for lc in sample_loadcells]
        start_avg, end_avg, is_valid = _detect_stable_regions(loadcells)

        assert is_valid is True
        assert 4900 < start_avg < 5100  # 시작: ~5000g
        assert 4400 < end_avg < 4600    # 종료: ~4500g

    def test_detect_stable_regions_short_data(self):
        """짧은 데이터에서 안정 구간 감지 테스트."""
        from model_service.api.routes.trigger import _detect_stable_regions

        loadcells = [
            LoadcellData(
                timestamp="2026-02-01T14:30:00.000Z",
                raw_value=["+5000"],
                filtered_value=["+5000"],
                filter_method="none",
            ),
            LoadcellData(
                timestamp="2026-02-01T14:30:01.000Z",
                raw_value=["+4500"],
                filtered_value=["+4500"],
                filter_method="none",
            ),
        ]

        start_avg, end_avg, is_valid = _detect_stable_regions(loadcells)

        # 짧은 데이터는 simple_delta_values 폴백 사용
        assert start_avg == 5000.0
        assert end_avg == 4500.0


class TestAvgLoadcellChannels:
    """로드셀 다채널 평균 계산 테스트 (Phase 0a)."""

    def test_avg_loadcell_channels_multi(self):
        """다중 채널 평균 계산 검증."""
        from model_service.api.routes.trigger import _avg_loadcell_channels
        assert _avg_loadcell_channels(["+1000", "+2000"]) == 1500.0

    def test_avg_loadcell_channels_single(self):
        from model_service.api.routes.trigger import _avg_loadcell_channels
        assert _avg_loadcell_channels(["+1000"]) == 1000.0

    def test_avg_loadcell_channels_empty(self):
        from model_service.api.routes.trigger import _avg_loadcell_channels
        assert _avg_loadcell_channels([]) == 0.0

    def test_avg_loadcell_channels_with_plus_sign(self):
        from model_service.api.routes.trigger import _avg_loadcell_channels
        assert _avg_loadcell_channels(["+3000", "+3000", "+3000"]) == 3000.0

    def test_avg_loadcell_channels_invalid_skipped(self):
        from model_service.api.routes.trigger import _avg_loadcell_channels
        # 유효하지 않은 값은 무시
        assert _avg_loadcell_channels(["+1000", "abc", "+3000"]) == 2000.0


class TestVoteResultsToEnsemble:
    """VoteResult -> EnsembleResult 변환 테스트."""

    def test_vote_results_to_ensemble_both_detected(self):
        """Top/Side 둘 다 감지된 경우 변환 테스트."""
        from model_service.api.routes.trigger import _vote_results_to_ensemble
        from model_service.video.voting_ensemble import VoteResult

        vote_results = [
            VoteResult(
                class_id=26,
                class_name="치킨마요",
                vote_count=100,
                max_confidence=0.95,
                avg_confidence=0.85,
                vote_ratio=0.5,
                top_detected=True,
                side_detected=True,
                top_max_confidence=0.9,
                side_max_confidence=0.85,
                weighted_confidence=0.875,
            )
        ]

        ensemble_results = _vote_results_to_ensemble(vote_results)

        assert len(ensemble_results) == 1
        result = ensemble_results[0]
        assert result.class_id == 26
        assert result.class_name == "치킨마요"
        assert result.top_confidence == 0.9
        assert result.side_confidence == 0.85
        assert result.combined_confidence == 0.875
        assert result.vote_count == 2  # 양쪽 감지

    def test_vote_results_to_ensemble_top_only(self):
        """Top만 감지된 경우 변환 테스트."""
        from model_service.api.routes.trigger import _vote_results_to_ensemble
        from model_service.video.voting_ensemble import VoteResult

        vote_results = [
            VoteResult(
                class_id=15,
                class_name="참치마요",
                vote_count=50,
                max_confidence=0.8,
                avg_confidence=0.7,
                vote_ratio=0.3,
                top_detected=True,
                side_detected=False,
                top_max_confidence=0.8,
                side_max_confidence=0.0,
                weighted_confidence=0.4,
            )
        ]

        ensemble_results = _vote_results_to_ensemble(vote_results)

        assert len(ensemble_results) == 1
        result = ensemble_results[0]
        assert result.vote_count == 1  # 한쪽만 감지

    def test_vote_results_to_ensemble_empty(self):
        """빈 결과 변환 테스트."""
        from model_service.api.routes.trigger import _vote_results_to_ensemble

        ensemble_results = _vote_results_to_ensemble([])
        assert ensemble_results == []
