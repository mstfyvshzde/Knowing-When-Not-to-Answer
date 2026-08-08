"""Tests the data preparation functions to make sure answerability labels, dataset splitting, reproducibility, validation, and dataset statistics work correctly."""

import pytest
from datasets import Dataset, DatasetDict

from src.data.prepare_data import (
    add_answerability_column,
    build_statistics,
    is_answerable,
    stratified_split_indices,
)


# Checks that an example with at least one answer is recognized as answerable.
def test_is_answerable_returns_true_when_answer_exists() -> None:
    example = {
        "answers": {
            "text": ["James Watt"],
            "answer_start": [10],
        }
    }

    assert is_answerable(example) is True


# Checks that an example with no answer text is recognized as unanswerable.
def test_is_answerable_returns_false_when_answer_is_empty() -> None:
    example = {
        "answers": {
            "text": [],
            "answer_start": [],
        }
    }

    assert is_answerable(example) is False


# Checks that a missing answers field is treated as unanswerable.
def test_is_answerable_returns_false_when_answers_missing() -> None:
    assert is_answerable({}) is False


# Checks that add_answerability_column adds 1 for answerable
# examples and 0 for unanswerable examples.
def test_add_answerability_column_adds_correct_labels() -> None:
    dataset = Dataset.from_dict(
        {
            "question": [
                "Who patented the steam engine?",
                "What is the missing answer?",
            ],
            "answers": [
                {
                    "text": ["James Watt"],
                    "answer_start": [0],
                },
                {
                    "text": [],
                    "answer_start": [],
                },
            ],
        }
    )

    result = add_answerability_column(dataset)

    assert result["is_answerable"] == [1, 0]


# Checks that the stratified split keeps all examples
# without losing or duplicating indices.
def test_stratified_split_indices_preserves_all_examples() -> None:
    labels = [0, 0, 0, 0, 1, 1, 1, 1]

    calibration_indices, test_indices = stratified_split_indices(
        labels=labels,
        calibration_fraction=0.50,
        seed=17,
    )

    combined_indices = calibration_indices + test_indices

    assert len(combined_indices) == len(labels)
    assert len(set(combined_indices)) == len(labels)
    assert set(combined_indices) == set(range(len(labels)))


# Checks that a 50% stratified split keeps both classes
# balanced between calibration and test.
def test_stratified_split_indices_keeps_class_balance() -> None:
    labels = [0, 0, 0, 0, 1, 1, 1, 1]

    calibration_indices, test_indices = stratified_split_indices(
        labels=labels,
        calibration_fraction=0.50,
        seed=17,
    )

    calibration_labels = [labels[index] for index in calibration_indices]
    test_labels = [labels[index] for index in test_indices]

    assert calibration_labels.count(0) == 2
    assert calibration_labels.count(1) == 2

    assert test_labels.count(0) == 2
    assert test_labels.count(1) == 2


# Checks that the same seed produces the same split.
def test_stratified_split_indices_is_reproducible() -> None:
    labels = [0, 0, 0, 0, 1, 1, 1, 1]

    first_split = stratified_split_indices(
        labels=labels,
        calibration_fraction=0.50,
        seed=17,
    )

    second_split = stratified_split_indices(
        labels=labels,
        calibration_fraction=0.50,
        seed=17,
    )

    assert first_split == second_split


# Checks that invalid calibration fractions are rejected.
@pytest.mark.parametrize(
    "calibration_fraction",
    [
        0.0,
        1.0,
        -0.10,
        1.10,
    ],
)
def test_stratified_split_indices_rejects_invalid_fraction(
    calibration_fraction: float,
) -> None:
    with pytest.raises(ValueError):
        stratified_split_indices(
            labels=[0, 0, 1, 1],
            calibration_fraction=calibration_fraction,
            seed=17,
        )


# Checks that dataset statistics are calculated correctly.
def test_build_statistics_calculates_correct_counts() -> None:
    train = Dataset.from_dict(
        {
            "is_answerable": [1, 1, 0, 1],
        }
    )

    calibration = Dataset.from_dict(
        {
            "is_answerable": [1, 0],
        }
    )

    test = Dataset.from_dict(
        {
            "is_answerable": [0, 0, 1, 1],
        }
    )

    dataset = DatasetDict(
        {
            "train": train,
            "calibration": calibration,
            "test": test,
        }
    )

    statistics = build_statistics(dataset)

    assert statistics["train"]["total_examples"] == 4
    assert statistics["train"]["answerable_examples"] == 3
    assert statistics["train"]["unanswerable_examples"] == 1
    assert statistics["train"]["answerable_fraction"] == pytest.approx(0.75)

    assert statistics["calibration"]["total_examples"] == 2
    assert statistics["calibration"]["answerable_examples"] == 1
    assert statistics["calibration"]["unanswerable_examples"] == 1
    assert statistics["calibration"]["answerable_fraction"] == pytest.approx(0.50)

    assert statistics["test"]["total_examples"] == 4
    assert statistics["test"]["answerable_examples"] == 2
    assert statistics["test"]["unanswerable_examples"] == 2
    assert statistics["test"]["answerable_fraction"] == pytest.approx(0.50)
