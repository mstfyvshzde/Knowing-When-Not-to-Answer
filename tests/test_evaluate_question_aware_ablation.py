import pytest

from src.evaluation.evaluate_question_aware_ablation import (
    create_ranked_indices,
    extract_self_verification_score,
    geometric_mean_score,
)


def test_self_verification_score_is_normalized_to_zero_one():
    assert extract_self_verification_score(
        {"self_verification_score": -1.0},
        "self_verification_score",
    ) == pytest.approx(0.0)

    assert extract_self_verification_score(
        {"self_verification_score": 0.0},
        "self_verification_score",
    ) == pytest.approx(0.5)

    assert extract_self_verification_score(
        {"self_verification_score": 1.0},
        "self_verification_score",
    ) == pytest.approx(1.0)


def test_geometric_mean_score_combines_signals_correctly():
    assert geometric_mean_score(0.81, 0.25) == pytest.approx(0.45)


def test_ranked_indices_use_original_index_as_tie_breaker():
    scores = [0.7, 0.9, 0.9, 0.4]

    assert create_ranked_indices(scores) == [1, 2, 0, 3]
