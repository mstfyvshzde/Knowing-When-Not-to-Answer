import json

import pytest

from src.evaluation.evaluate_random_abstention import (
    evaluate_one_seed,
    run_multi_seed_evaluation,
    summarize_runs,
    validate_coverages,
    validate_seeds,
)


def build_predictions() -> list[dict]:
    return [
        {
            "id": "1",
            "prediction_text": "alpha",
            "reference_answers": ["alpha"],
            "is_answerable": True,
        },
        {
            "id": "2",
            "prediction_text": "wrong",
            "reference_answers": ["beta"],
            "is_answerable": True,
        },
        {
            "id": "3",
            "prediction_text": "gamma",
            "reference_answers": ["gamma"],
            "is_answerable": True,
        },
        {
            "id": "4",
            "prediction_text": "wrong",
            "reference_answers": ["delta"],
            "is_answerable": True,
        },
    ]


def test_validate_coverages_accepts_valid_values() -> None:
    validate_coverages([0.1, 0.5, 1.0])


@pytest.mark.parametrize(
    "coverages",
    [
        [],
        [0.0],
        [-0.1],
        [1.1],
    ],
)
def test_validate_coverages_rejects_invalid_values(
    coverages: list[float],
) -> None:
    with pytest.raises(ValueError):
        validate_coverages(coverages)


def test_validate_seeds_accepts_non_negative_values() -> None:
    validate_seeds([0, 1, 17])


@pytest.mark.parametrize(
    "seeds",
    [
        [],
        [-1],
        [0, -5],
    ],
)
def test_validate_seeds_rejects_invalid_values(
    seeds: list[int],
) -> None:
    with pytest.raises(ValueError):
        validate_seeds(seeds)


def test_evaluate_one_seed_returns_expected_structure() -> None:
    result = evaluate_one_seed(
        predictions=build_predictions(),
        coverage=0.5,
        seed=17,
    )

    assert result["seed"] == 17
    assert result["target_coverage"] == 0.5
    assert result["actual_coverage"] == 0.5
    assert result["answered"] == 2
    assert result["abstain_rate"] == 0.5
    assert 0.0 <= result["answer_accuracy"] <= 1.0
    assert 0.0 <= result["selective_risk"] <= 1.0


def test_summarize_runs_calculates_population_statistics() -> None:
    runs = [
        {
            "seed": 0,
            "target_coverage": 0.5,
            "actual_coverage": 0.5,
            "answered": 2,
            "answer_accuracy": 0.5,
            "selective_risk": 0.5,
            "abstain_rate": 0.5,
        },
        {
            "seed": 1,
            "target_coverage": 0.5,
            "actual_coverage": 0.5,
            "answered": 2,
            "answer_accuracy": 1.0,
            "selective_risk": 0.0,
            "abstain_rate": 0.5,
        },
    ]

    summary = summarize_runs(runs)

    assert summary["number_of_seeds"] == 2
    assert summary["mean_answer_accuracy"] == pytest.approx(0.75)
    assert summary["mean_selective_risk"] == pytest.approx(0.25)
    assert summary["answer_accuracy_std"] == pytest.approx(0.25)
    assert summary["selective_risk_std"] == pytest.approx(0.25)


def test_full_coverage_has_zero_variance_across_seeds(
    tmp_path,
) -> None:
    input_path = tmp_path / "predictions.jsonl"

    lines = [
        json.dumps(
            {
                "id": "1",
                "prediction_text": "alpha",
                "reference_answers": ["alpha"],
                "is_answerable": True,
            }
        ),
        json.dumps(
            {
                "id": "2",
                "prediction_text": "wrong",
                "reference_answers": ["beta"],
                "is_answerable": True,
            }
        ),
    ]

    input_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    results = run_multi_seed_evaluation(
        input_path=input_path,
        coverages=[1.0],
        seeds=[0, 1, 2],
    )

    summary = results["results"][0]["summary"]

    assert summary["actual_coverage"] == 1.0
    assert summary["answered_per_seed"] == 2
    assert summary["mean_selective_risk"] == pytest.approx(0.5)
    assert summary["selective_risk_std"] == pytest.approx(0.0)
