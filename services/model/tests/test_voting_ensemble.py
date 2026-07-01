import pytest


def test_default_ensemble_weights_bias_top_middle_for_consensus():
    from model_service.video.voting_ensemble import VotingEnsemble

    top = VotingEnsemble()
    side = VotingEnsemble()
    top.set_frame_count(1)
    side.set_frame_count(1)
    top.add_vote(1, 0.8, "top-and-side")
    side.add_vote(1, 0.5, "top-and-side")

    results = VotingEnsemble.combine(top, side)

    expected = (0.8 * 0.60) + (0.5 * 0.40) + (min(0.8, 0.5) * 0.2)
    assert len(results) == 1
    assert results[0].weighted_confidence == pytest.approx(expected)


def test_default_top_only_weight_is_higher_than_side_only_weight():
    from model_service.video.voting_ensemble import VotingEnsemble

    top = VotingEnsemble()
    side = VotingEnsemble()
    top.set_frame_count(1)
    side.set_frame_count(1)
    top.add_vote(1, 0.8, "top-only")
    side.add_vote(2, 0.8, "side-only")

    results = VotingEnsemble.combine(top, side)
    by_name = {result.class_name: result for result in results}

    assert by_name["top-only"].weighted_confidence == pytest.approx(0.8 * 0.60)
    assert by_name["side-only"].weighted_confidence == pytest.approx(0.8 * 0.40)
    assert by_name["top-only"].weighted_confidence > by_name["side-only"].weighted_confidence


def test_top_only_candidate_sorts_above_same_confidence_side_only_candidate():
    from model_service.video.voting_ensemble import VotingEnsemble

    top = VotingEnsemble()
    side = VotingEnsemble()
    top.set_frame_count(1)
    side.set_frame_count(1)
    top.add_vote(1, 0.9, "top-only")
    side.add_vote(2, 0.9, "side-only")

    results = VotingEnsemble.combine(top, side)

    assert [result.class_name for result in results] == ["top-only", "side-only"]
