from model_service.core import loadcell_stats
from model_service.service.trigger_service import (
    LoadcellReading,
    TriggerInput,
    TriggerService,
)
from model_service.session import SessionStore


def _reading(second: int, total_weight: float) -> LoadcellReading:
    half = total_weight / 2
    value = f"{half:+06.1f}"
    return LoadcellReading(
        timestamp=f"2026-06-01T00:00:{second:02d}+00:00",
        raw_value=[value, value],
        filtered_value=[value, value],
    )


def _plateau(start_second: int, total_weight: float) -> list[LoadcellReading]:
    return [_reading(start_second + offset, total_weight) for offset in range(5)]


def _multi_channel_reading(second: int, values: list[float]) -> LoadcellReading:
    encoded = [f"{value:+06.1f}" for value in values]
    return LoadcellReading(
        timestamp=f"2026-06-01T00:00:{second:02d}+00:00",
        raw_value=encoded,
        filtered_value=encoded,
    )


def _multi_channel_plateau(
    start_second: int,
    values: list[float],
) -> list[LoadcellReading]:
    return [
        _multi_channel_reading(start_second + offset, values)
        for offset in range(5)
    ]


def _single_channel_plateau(
    start_second: int,
    total_weight: float,
) -> list[LoadcellReading]:
    value = f"{total_weight:+06.1f}"
    return [
        LoadcellReading(
            timestamp=f"2026-06-01T00:00:{start_second + offset:02d}+00:00",
            raw_value=[value],
            filtered_value=[value],
        )
        for offset in range(5)
    ]


def _spanned_two_channel_series(
    values: list[float],
    *,
    span_seconds: float = 3.0,
) -> list[LoadcellReading]:
    step = span_seconds / max(len(values) - 1, 1)
    loadcells: list[LoadcellReading] = []
    for index, total_weight in enumerate(values):
        half = total_weight / 2
        value = f"{half:+06.1f}"
        loadcells.append(
            LoadcellReading(
                timestamp=f"2026-06-01T00:00:{index * step:06.3f}+00:00",
                raw_value=[value, value],
                filtered_value=[value, value],
            )
        )
    return loadcells


def test_compound_segments_preserve_net_delta_but_expose_internal_movements():
    loadcells = (
        _plateau(0, 1000.0)
        + _plateau(5, 900.0)
        + _plateau(10, 960.0)
        + _plateau(15, 850.0)
    )

    analysis = loadcell_stats.analyze_weight_delta(loadcells, window_size=2)

    assert analysis.delta == -150.0
    assert analysis.reason == "stable_regions"
    assert [segment.sign for segment in analysis.segments] == [-1, 1, -1]
    assert [round(segment.delta, 1) for segment in analysis.segments] == [
        -100.0,
        60.0,
        -110.0,
    ]


def test_history_pairs_remove_return_and_uses_remaining_removal_as_decision_delta():
    loadcells = (
        _plateau(0, 1000.0)
        + _plateau(5, 700.0)
        + _plateau(10, 1000.0)
        + _plateau(15, 800.0)
    )

    analysis = loadcell_stats.analyze_weight_delta(loadcells, window_size=2)

    assert analysis.delta == -200.0
    assert analysis.decision_delta == -200.0
    assert analysis.paired_loadcell_movements[0]["reason"] == "removal_return_pair"
    assert analysis.purchase_delta_candidates[0]["weight"] == 200.0
    assert analysis.purchase_delta_candidates[0]["source"] == "net_stable_delta"
    assert [target["weight"] for target in analysis.removal_segment_targets] == [
        200.0
    ]


def test_history_can_find_removal_even_when_net_delta_is_positive():
    loadcells = (
        _plateau(0, 1000.0)
        + _plateau(5, 1300.0)
        + _plateau(10, 1100.0)
    )

    analysis = loadcell_stats.analyze_weight_delta(loadcells, window_size=2)

    assert analysis.delta == 100.0
    assert analysis.decision_delta == -200.0
    assert analysis.purchase_delta_candidates[0]["weight"] == 200.0
    assert analysis.purchase_delta_candidates[0]["source"] == (
        "unpaired_negative_total"
    )
    assert [target["weight"] for target in analysis.removal_segment_targets] == [
        200.0
    ]


def test_history_exposes_return_segment_before_followup_removal():
    loadcells = (
        _single_channel_plateau(0, 3093.9)
        + _single_channel_plateau(5, 3310.6)
        + _single_channel_plateau(10, 3294.1)
    )

    analysis = loadcell_stats.analyze_weight_delta(
        loadcells,
        stability_threshold=2.0,
    )

    assert round(analysis.delta, 1) == 200.2
    assert round(analysis.decision_delta, 1) == -16.5
    assert [target["weight"] for target in analysis.return_segment_targets] == [
        216.7
    ]
    assert [target["weight"] for target in analysis.removal_segment_targets] == [
        16.5
    ]


def test_trigger_service_builds_mixed_return_weight_hint():
    service = TriggerService(
        video_processor=None,
        engine=None,
        session_store=SessionStore(),
    )
    loadcells = (
        _single_channel_plateau(0, 3093.9)
        + _single_channel_plateau(5, 3310.6)
        + _single_channel_plateau(10, 3294.1)
    )
    analysis = loadcell_stats.analyze_weight_delta(
        loadcells,
        stability_threshold=2.0,
    )

    hints = service._mixed_return_hints_from_analysis(
        analysis,
        decision_delta=analysis.decision_delta,
    )

    assert hints == [
        {
            "source": "unpaired_positive_segment",
            "weight": 216.7,
            "delta": 216.7,
            "segment_index": 0,
            "segment_indices": [0],
            "start_timestamp": "2026-06-01T00:00:04+00:00",
            "end_timestamp": "2026-06-01T00:00:05+00:00",
            "duration_seconds": 1.0,
            "reason": "unpaired_return_segment",
            "replay_position": "before_removal",
        }
    ]


def test_trigger_route_exposes_return_segments_and_builds_mixed_return_weight_hint():
    from model_service.api.routes.trigger import (
        _loadcell_trace_metadata,
        _mixed_return_hints_from_analysis,
    )

    loadcells = (
        _single_channel_plateau(0, 3093.9)
        + _single_channel_plateau(5, 3310.6)
        + _single_channel_plateau(10, 3294.1)
    )
    analysis = loadcell_stats.analyze_weight_delta(
        loadcells,
        stability_threshold=2.0,
    )

    metadata = _loadcell_trace_metadata(loadcells, analysis)
    hints = _mixed_return_hints_from_analysis(
        analysis,
        decision_delta=analysis.decision_delta,
    )

    assert [target["weight"] for target in metadata["return_segment_targets"]] == [
        216.7
    ]
    assert hints == [
        {
            "source": "unpaired_positive_segment",
            "weight": 216.7,
            "delta": 216.7,
            "segment_index": 0,
            "segment_indices": [0],
            "start_timestamp": "2026-06-01T00:00:04+00:00",
            "end_timestamp": "2026-06-01T00:00:05+00:00",
            "duration_seconds": 1.0,
            "reason": "unpaired_return_segment",
            "replay_position": "before_removal",
        }
    ]


def test_history_exposes_separate_removal_segment_targets_in_time_order():
    loadcells = (
        _plateau(0, 1000.0)
        + _plateau(5, 790.0)
        + _plateau(10, 685.0)
        + _plateau(15, 582.0)
        + _plateau(20, 475.0)
    )

    analysis = loadcell_stats.analyze_weight_delta(loadcells, window_size=2)

    assert analysis.delta == -525.0
    assert analysis.decision_delta == -525.0
    assert [target["weight"] for target in analysis.removal_segment_targets] == [
        210.0,
        105.0,
        103.0,
        107.0,
    ]
    assert [
        target["segment_index"] for target in analysis.removal_segment_targets
    ] == [0, 1, 2, 3]


def test_history_exposes_simultaneous_channel_removal_targets():
    loadcells = (
        _multi_channel_plateau(0, [647.0, 2608.0])
        + _multi_channel_plateau(5, [503.0, 2233.0])
    )

    analysis = loadcell_stats.analyze_weight_delta(loadcells, window_size=2)

    assert analysis.delta == -519.0
    assert analysis.decision_delta == -519.0
    assert [target["weight"] for target in analysis.channel_removal_segment_targets] == [
        144.0,
        375.0,
    ]
    assert all(
        target["source"] == "simultaneous_channel_delta"
        for target in analysis.channel_removal_segment_targets
    )
    assert analysis.channel_delta_diagnostics["accepted"] is True
    assert analysis.channel_delta_diagnostics["negative_channel_count"] == 2


def test_history_rejects_channel_targets_when_positive_channel_offsets_total():
    loadcells = (
        _multi_channel_plateau(0, [647.0, 2608.0, 1000.0])
        + _multi_channel_plateau(5, [503.0, 2233.0, 1100.0])
    )

    analysis = loadcell_stats.analyze_weight_delta(loadcells, window_size=2)

    assert analysis.delta == -419.0
    assert analysis.channel_removal_segment_targets == []
    assert analysis.channel_delta_diagnostics["accepted"] is False
    assert analysis.channel_delta_diagnostics["reason"] == "positive_channel_delta_present"


def test_history_rejects_channel_targets_when_only_one_channel_changes():
    loadcells = (
        _multi_channel_plateau(0, [1000.0, 2000.0])
        + _multi_channel_plateau(5, [500.0, 2000.0])
    )

    analysis = loadcell_stats.analyze_weight_delta(loadcells, window_size=2)

    assert analysis.delta == -500.0
    assert analysis.channel_removal_segment_targets == []
    assert analysis.channel_delta_diagnostics["accepted"] is False
    assert analysis.channel_delta_diagnostics["reason"] == "insufficient_negative_channels"


def test_history_ignores_balanced_return_and_pressure_like_release():
    remove_return = (
        _plateau(0, 1000.0)
        + _plateau(5, 700.0)
        + _plateau(10, 1000.0)
    )
    press_release = (
        _plateau(0, 1000.0)
        + _plateau(5, 1300.0)
        + _plateau(10, 1000.0)
    )

    remove_return_analysis = loadcell_stats.analyze_weight_delta(remove_return)
    press_release_analysis = loadcell_stats.analyze_weight_delta(press_release)

    assert remove_return_analysis.decision_delta == 0.0
    assert remove_return_analysis.purchase_delta_candidates == []
    assert remove_return_analysis.return_segment_targets == []
    assert remove_return_analysis.pressure_like_event is True
    assert press_release_analysis.decision_delta == 0.0
    assert press_release_analysis.purchase_delta_candidates == []
    assert press_release_analysis.return_segment_targets == []
    assert press_release_analysis.removal_segment_targets == []
    assert [
        target["weight"]
        for target in press_release_analysis.vision_required_segment_targets
    ] == [300.0]
    assert press_release_analysis.pressure_like_event is True


def test_history_does_not_promote_one_gram_middle_bump_to_purchase():
    loadcells = (
        _plateau(0, 1000.0)
        + _plateau(5, 1001.0)
        + _plateau(10, 1000.0)
    )

    analysis = loadcell_stats.analyze_weight_delta(loadcells)

    assert analysis.decision_delta == 0.0
    assert analysis.purchase_delta_candidates == []
    assert analysis.return_segment_targets == []
    assert analysis.removal_segment_targets == []
    assert analysis.vision_required_segment_targets == []
    assert analysis.segments == []


def test_trigger_metadata_records_compound_segments_and_recent_same_zone_returns():
    service = TriggerService(
        video_processor=None,
        engine=None,
        session_store=SessionStore(),
    )
    service._register_loadcell_event(
        session_id="previous-return",
        zone=3,
        delta_weight=60.0,
        state="return_only",
    )
    loadcells = (
        _plateau(0, 1000.0)
        + _plateau(5, 900.0)
        + _plateau(10, 960.0)
        + _plateau(15, 850.0)
    )
    analysis = service._analyze_weight_delta(loadcells)

    metadata = service._loadcell_trace_metadata(loadcells, analysis, zone=3)

    assert metadata["compound_event"] is True
    assert metadata["compound_segment_count"] == 3
    assert metadata["compound_positive_weights_g"] == [60.0]
    assert metadata["compound_negative_weights_g"] == [100.0, 110.0]
    assert metadata["decision_delta_weight"] == -150.0
    assert [target["weight"] for target in metadata["return_segment_targets"]] == [
        60.0
    ]
    assert metadata["stable_plateaus"]
    assert metadata["purchase_delta_candidates"]
    assert [target["weight"] for target in metadata["removal_segment_targets"]] == [
        100.0,
        110.0,
    ]
    assert metadata["recent_return_weights_g"] == [60.0]
    assert metadata["recent_same_zone_events"][0]["session_id"] == "previous-return"


def test_freezer_endpoint_fallback_trace_metadata_avoids_low_weight_skip():
    service = TriggerService(
        video_processor=None,
        engine=None,
        session_store=SessionStore(),
    )
    loadcells = _spanned_two_channel_series(
        [10.0, -40.0, 5.0, -45.0, 0.0, -50.0, -5.0, -55.0, -10.0, -60.0]
    )
    input_data = TriggerInput(
        zone=2,
        loadcells=loadcells,
        top_video_path=None,
        side_video_path=None,
        cabinet_type="freezer",
    )

    analysis = service._analyze_weight_delta(loadcells, cabinet_type="freezer")
    metadata = service._loadcell_trace_metadata(
        loadcells,
        analysis,
        zone=2,
        input_data=input_data,
    )

    assert analysis.decision_delta == -70.0
    assert service._should_skip_low_weight(input_data, analysis, analysis.decision_delta) is False
    assert service._removal_stabilization_from_analysis(analysis) is None
    assert metadata["cabinet_type"] == "freezer"
    assert metadata["decision_delta_weight"] == -70.0
    assert metadata["decision_delta_reliable"] is True
    assert metadata["endpoint_delta_weight"] == -70.0
    assert metadata["endpoint_fallback_applied"] is True
    assert metadata["endpoint_fallback_reason"] == "freezer_endpoint_delta"
    assert metadata["stable_delta_source"] == "freezer_endpoint_fallback"
