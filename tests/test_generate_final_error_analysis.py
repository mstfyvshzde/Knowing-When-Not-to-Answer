"""Tests for canonical final verifier error analysis."""

from pathlib import Path

import pytest

from src.analysis.generate_final_error_analysis import generate_summary


def record(**overrides: object) -> dict[str, object]:
    """Build the minimal schema required by the summary generator."""

    row: dict[str, object] = {
        "is_answerable": True,
        "prediction_text": "Paris",
        "reference_answers": ["Paris"],
        "qa_claim_valid": True,
        "qa_nli_label": "ENTAILMENT",
        "self_verification_label": "SUPPORTED",
        "qa_entailment_probability": 0.9,
        "qa_claim_validation_reasons": [],
    }
    row.update(overrides)
    return row


def test_summary_keeps_punctuation_only_unanswerable_candidate_incorrect() -> None:
    """Forced-answer correctness must not turn '.' into a correct abstention."""

    rows = [
        record(),
        record(
            is_answerable=False,
            prediction_text=".",
            reference_answers=[],
            qa_nli_label="CONTRADICTION",
            self_verification_label="REJECTED",
            qa_entailment_probability=0.1,
        ),
        record(
            prediction_text="London",
            reference_answers=["Paris"],
            qa_entailment_probability=0.8,
            self_verification_label="UNCERTAIN",
        ),
        record(
            prediction_text="The Nile",
            reference_answers=["Nile"],
            qa_nli_label="NEUTRAL",
            qa_entailment_probability=0.2,
            self_verification_label="UNCERTAIN",
        ),
    ]

    summary = generate_summary(rows, Path("predictions.jsonl"))

    assert summary["correct_predictions"] == 2
    assert summary["incorrect_predictions"] == 2
    assert summary["full_accuracy"] == pytest.approx(0.5)
    assert summary["qa_nli_label"]["CONTRADICTION"]["correct"] == 0
    assert summary["diagnostic_counts"] == {
        "high_entailment_incorrect_ge_0.8": 1,
        "low_entailment_correct_le_0.2": 1,
    }


def test_summary_rejects_non_boolean_claim_validity() -> None:
    """Persisted claim validity must remain a real boolean."""

    row = record(qa_claim_valid="true")

    with pytest.raises(TypeError, match="qa_claim_valid"):
        generate_summary([row], Path("predictions.jsonl"))
