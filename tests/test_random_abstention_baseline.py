import pytest

from src.baselines.random_abstention_baseline import (
    apply_random_abstention,
    summarize_decisions,
)


def build_predictions(count: int = 10) -> list[dict]:
    return [
        {
            "id": str(index),
            "prediction_text": f"answer-{index}",
            "decision": "ANSWER",
        }
        for index in range(count)
    ]


def test_zero_coverage_abstains_on_every_prediction() -> None:
    predictions = build_predictions()

    results = apply_random_abstention(
        predictions=predictions,
        coverage=0.0,
        seed=42,
    )

    assert all(result["decision"] == "ABSTAIN" for result in results)
    assert all(result["final_answer"] == "I do not know" for result in results)


def test_full_coverage_answers_every_prediction() -> None:
    predictions = build_predictions()

    results = apply_random_abstention(
        predictions=predictions,
        coverage=1.0,
        seed=42,
    )

    assert all(result["decision"] == "ANSWER" for result in results)

    assert [result["final_answer"] for result in results] == [
        prediction["prediction_text"] for prediction in predictions
    ]


def test_same_seed_produces_same_decisions() -> None:
    predictions = build_predictions(count=100)

    first_results = apply_random_abstention(
        predictions=predictions,
        coverage=0.5,
        seed=42,
    )

    second_results = apply_random_abstention(
        predictions=predictions,
        coverage=0.5,
        seed=42,
    )

    assert [result["decision"] for result in first_results] == [
        result["decision"] for result in second_results
    ]


@pytest.mark.parametrize(
    "coverage",
    [-0.01, 1.01],
)
def test_invalid_coverage_raises_value_error(
    coverage: float,
) -> None:
    predictions = build_predictions()

    with pytest.raises(
        ValueError,
        match="Coverage must be between 0 and 1",
    ):
        apply_random_abstention(
            predictions=predictions,
            coverage=coverage,
            seed=42,
        )


def test_summarize_decisions_calculates_coverage() -> None:
    predictions = [
        {"decision": "ANSWER"},
        {"decision": "ANSWER"},
        {"decision": "ABSTAIN"},
        {"decision": "ABSTAIN"},
    ]

    summary = summarize_decisions(predictions)

    assert summary == {
        "total_examples": 4,
        "answered_examples": 2,
        "abstained_examples": 2,
        "coverage": 0.5,
        "abstention_rate": 0.5,
    }


def test_summarize_decisions_rejects_empty_predictions() -> None:
    with pytest.raises(
        ValueError,
        match="Prediction list cannot be empty",
    ):
        summarize_decisions([])


def test_exact_target_coverage_is_respected() -> None:
    predictions = build_predictions(count=200)

    results = apply_random_abstention(
        predictions=predictions,
        coverage=0.5,
        seed=17,
    )

    answered = sum(result["decision"] == "ANSWER" for result in results)

    assert answered == 100
