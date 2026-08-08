"""Tests the decision engine to make sure it correctly reads confidence and evidence information, combines them into ANSWER, VERIFY, or ABSTAIN decisions, handles invalid inputs, and calculates final decision metrics correctly."""
import pytest


from src.decision.decision_engine import (
    calculate_decision_metrics,
    combine_decisions,
    get_confidence,
    get_evidence_support,
    get_threshold_decision,
    process_prediction,
    validate_predictions,
)


# Checks that a normal threshold decision is returned correctly.
def test_get_threshold_decision_returns_valid_decision() -> None:
    prediction = {
        "threshold_decision": "ANSWER",
    }

    assert get_threshold_decision(prediction) == "ANSWER"


# Checks that lowercase decision values are standardized to uppercase.
def test_get_threshold_decision_normalizes_case() -> None:
    prediction = {
        "threshold_decision": "verify",
    }

    assert get_threshold_decision(prediction) == "VERIFY"


# Checks that the function can use another supported decision field.
def test_get_threshold_decision_uses_fallback_field() -> None:
    prediction = {
        "confidence_decision": "ABSTAIN",
    }

    assert get_threshold_decision(prediction) == "ABSTAIN"


# Checks that an error is raised when no valid threshold decision exists.
def test_get_threshold_decision_rejects_invalid_input() -> None:
    prediction = {
        "threshold_decision": "UNKNOWN",
    }

    with pytest.raises(ValueError):
        get_threshold_decision(prediction)


# Checks that a valid evidence label is returned correctly.
def test_get_evidence_support_returns_valid_label() -> None:
    prediction = {
        "evidence_support": "SUPPORTED",
    }

    assert get_evidence_support(prediction) == "SUPPORTED"


# Checks that lowercase evidence labels are standardized to uppercase.
def test_get_evidence_support_normalizes_case() -> None:
    prediction = {
        "evidence_support": "weak",
    }

    assert get_evidence_support(prediction) == "WEAK"


# Checks that invalid evidence labels are rejected.
def test_get_evidence_support_rejects_invalid_label() -> None:
    prediction = {
        "evidence_support": "MAYBE",
    }

    with pytest.raises(ValueError):
        get_evidence_support(prediction)


# Checks that calibrated confidence is preferred when available.
def test_get_confidence_uses_calibrated_confidence_first() -> None:
    prediction = {
        "calibrated_confidence": 0.75,
        "confidence": 0.40,
    }

    assert get_confidence(prediction) == pytest.approx(0.75)


# Checks that normal confidence is used when calibrated confidence is missing.
def test_get_confidence_uses_confidence_fallback() -> None:
    prediction = {
        "confidence": 0.60,
    }

    assert get_confidence(prediction) == pytest.approx(0.60)


# Checks that confidence values outside 0–1 are rejected.
def test_get_confidence_rejects_out_of_range_value() -> None:
    prediction = {
        "confidence": 1.20,
    }

    with pytest.raises(ValueError):
        get_confidence(prediction)


# Checks that missing confidence information is rejected.
def test_get_confidence_rejects_missing_confidence() -> None:
    with pytest.raises(ValueError):
        get_confidence({})


# High-confidence ANSWER decisions should remain ANSWER.
def test_combine_decisions_preserves_answer() -> None:
    final_decision, reason = combine_decisions(
        threshold_decision="ANSWER",
        evidence_support="UNSUPPORTED",
    )

    assert final_decision == "ANSWER"
    assert reason == "high_confidence_answer_preserved"


# Low-confidence ABSTAIN decisions should remain ABSTAIN.
def test_combine_decisions_preserves_abstain() -> None:
    final_decision, reason = combine_decisions(
        threshold_decision="ABSTAIN",
        evidence_support="SUPPORTED",
    )

    assert final_decision == "ABSTAIN"
    assert reason == "confidence_below_abstain_threshold"


# VERIFY + SUPPORTED evidence should become ANSWER.
def test_combine_decisions_verify_supported_becomes_answer() -> None:
    final_decision, reason = combine_decisions(
        threshold_decision="VERIFY",
        evidence_support="SUPPORTED",
    )

    assert final_decision == "ANSWER"
    assert reason == "medium_confidence_with_supported_evidence"


# VERIFY + WEAK evidence should remain VERIFY.
def test_combine_decisions_verify_weak_remains_verify() -> None:
    final_decision, reason = combine_decisions(
        threshold_decision="VERIFY",
        evidence_support="WEAK",
    )

    assert final_decision == "VERIFY"
    assert reason == "medium_confidence_and_weak_evidence"


# VERIFY + UNSUPPORTED evidence should become ABSTAIN.
def test_combine_decisions_verify_unsupported_becomes_abstain() -> None:
    final_decision, reason = combine_decisions(
        threshold_decision="VERIFY",
        evidence_support="UNSUPPORTED",
    )

    assert final_decision == "ABSTAIN"
    assert reason == "medim_confidence_and_unsupported_evidence"


# Checks that one prediction is processed and enriched correctly.
def test_process_prediction_adds_final_decision_fields() -> None:
    prediction = {
        "threshold_decision": "VERIFY",
        "evidence_support": "SUPPORTED",
        "confidence": 0.65,
        "prediction_text": "James Watt",
    }

    result = process_prediction(prediction)

    assert result["threshold_decision"] == "VERIFY"
    assert result["final_decision"] == "ANSWER"
    assert result["decision_reason"] == "medium_confidence_with_supported_evidence"
    assert result["decision_engine"] == "confidence_evidence_rule_based"
    assert result["decision_confidence"] == pytest.approx(0.65)


# Checks that processing does not modify the original prediction dictionary.
def test_process_prediction_does_not_modify_original() -> None:
    prediction = {
        "threshold_decision": "VERIFY",
        "evidence_support": "WEAK",
        "confidence": 0.50,
    }

    process_prediction(prediction)

    assert "final_decision" not in prediction
    assert "decision_reason" not in prediction


# Checks that an empty prediction list is rejected.
def test_validate_predictions_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        validate_predictions([])


# Checks that predictions without evidence support are rejected.
def test_validate_predictions_requires_evidence_support() -> None:
    predictions = [
        {
            "threshold_decision": "ANSWER",
            "confidence": 0.90,
        }
    ]

    with pytest.raises(ValueError):
        validate_predictions(predictions)


# Checks the final ANSWER / VERIFY / ABSTAIN counts and rates.
def test_calculate_decision_metrics_counts_decisions() -> None:
    predictions = [
        {"final_decision": "ANSWER"},
        {"final_decision": "ANSWER"},
        {"final_decision": "VERIFY"},
        {"final_decision": "ABSTAIN"},
    ]

    metrics = calculate_decision_metrics(predictions)

    assert metrics["total"] == 4
    assert metrics["answer_count"] == 2
    assert metrics["verify_count"] == 1
    assert metrics["abstain_count"] == 1

    assert metrics["answer_rate"] == pytest.approx(0.50)
    assert metrics["verify_rate"] == pytest.approx(0.25)
    assert metrics["abstain_rate"] == pytest.approx(0.25)


# Checks that accuracy is calculated separately for each final decision group.
def test_calculate_decision_metrics_calculates_group_accuracy() -> None:
    predictions = [
        {
            "final_decision": "ANSWER",
            "is_correct": True,
        },
        {
            "final_decision": "ANSWER",
            "is_correct": False,
        },
        {
            "final_decision": "VERIFY",
            "is_correct": True,
        },
        {
            "final_decision": "ABSTAIN",
            "is_correct": False,
        },
    ]

    metrics = calculate_decision_metrics(predictions)

    assert metrics["answer_accuracy"] == pytest.approx(0.50)
    assert metrics["verify_accuracy"] == pytest.approx(1.00)
    assert metrics["abstain_accuracy"] == pytest.approx(0.00)
    assert metrics["answer_risk"] == pytest.approx(0.50)