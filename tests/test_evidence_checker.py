"""
This file checks whether evidence_checker.py works correctly by testing answer normalization, answer-overlap F1 scoring, evidence classification, and evidence-summary calculations.
"""

# assert means: “Check that this condition is true. If it is false, fail the test.”
import pytest

from src.verification.evidence_checker import (
    answer_overlap_f1,
    classify_evidence,
    normalize_answer,
    summarize_evidence,
)


# checks that normalize_answer() correctly converts uppercase letters to lowercase.
def test_normalize_answerr_lowercases_text() -> None:
    assert normalize_answer("James Watt") == "james watt"


# Checks that normalize_answer() removes punctuation like !.
def test_normalize__answer_removes_punctation() -> None:
    assert normalize_answer("James Watt!") == "james watt"


# Checks that normalize_answer() removes articles like a, an, and the.
def test_normalize_answerr_removes_articles() -> None:
    assert normalize_answer("The James Watt") == "james watt"


# Checks that two identical answers get an F1 similarity score of about 1.0.
def test_answer_overlap_f1_perfect_match() -> None:
    score = answer_overlap_f1("James Watt", "James Watt")

    assert score == pytest.approx(1.0)


# Checks that partially matching answers get a score between 0 and 1, not a perfect match and not zero.
def test_answer_overlap_f1_partial_match() -> None:
    score = answer_overlap_f1("James Watt", "Watt")

    assert 0.0 < score < 1.0


# Checks that two completely different answers get an F1 score of 0.0.
def test_answer_overlap_f1_returns_zero_for_no_overlap() -> None:
    score = answer_overlap_f1("James Watt", "Albert Einstein")

    assert score == pytest.approx(0.0)


# Checks that if one answer is empty, the F1 score is 0.0.
def test_answer_overlap_f1_returns_zero_for_empty_overlap() -> None:
    score = answer_overlap_f1("", "James Watt")

    assert score == pytest.approx(0.0)


# Checks that when the generated answer and verifier answer match strongly, and the verifier confidence is high enough, the function returns SUPPORTED.
def test_classify_evidence_supported() -> None:
    correct_label, reason, match = classify_evidence(
        generated_answer="James Watt",
        verifier_answer="James Watt",
        verifier_score=0.95,
        support_threshold=0.30,
        match_threshold=0.80,
        rejection_threshold=0.50,
    )

    assert correct_label == "SUPPORTED"
    assert match >= 0.80
    assert "matching answer" in reason.lower()


# Checks that when the generated answer and verifier answer are completely different, and the verifier is highly confident, the function returns UNSUPPORTED.
def test_classify_evidence_unsupported_for_different_answer() -> None:
    correct_label, reason, match = classify_evidence(
        generated_answer="James Watt",
        verifier_answer="Albert Einstein",
        verifier_score=0.95,
        support_threshold=0.30,
        match_threshold=0.80,
        rejection_threshold=0.50,
    )

    assert correct_label == "UNSUPPORTED"
    assert match < 0.20
    assert "different answer" in reason.lower()


# Checks that when the verifier returns no answer and is highly confident about it, the generated answer is classified as UNSUPPORTED.
def test_classify_evidence_unsupported_for_confident_no_answer() -> None:
    correct_label, reason, match = classify_evidence(
        generated_answer="James Watt",
        verifier_answer="",
        verifier_score=0.90,
        support_threshold=0.30,
        match_threshold=0.80,
        rejection_threshold=0.50,
    )

    assert correct_label == "UNSUPPORTED"
    assert match == 0.0
    assert "unanswerable" in reason.lower()


# Checks that when the verifier gives no answer but has low confidence, the result is UNCERTAIN instead of UNSUPPORTED.
def test_classify_evidence_uncertain_for_low_confidence_no_answer() -> None:
    correct_label, reason, match = classify_evidence(
        generated_answer="James Watt",
        verifier_answer="",
        verifier_score=0.20,
        support_threshold=0.30,
        match_threshold=0.80,
        rejection_threshold=0.50,
    )

    assert correct_label == "UNCERTAIN"
    assert match == 0.0
    assert "not high enough" in reason.lower()


# Checks that summarize_evidence() correctly counts each evidence label and calculates their rates.
def test_summarize_evidence_counts_labels() -> None:
    predictions = [
        {"evidence_label": "SUPPORTED"},
        {"evidence_label": "SUPPORTED"},
        {"evidence_label": "UNSUPPORTED"},
        {"evidence_label": "UNCERTAIN"},
    ]

    summary = summarize_evidence(predictions)

    assert summary["total"] == 4
    assert summary["supported"] == 2
    assert summary["unsupported"] == 1
    assert summary["uncertain"] == 1
    assert summary["supported_rate"] == pytest.approx(0.50)
    assert summary["unsupported_rate"] == pytest.approx(0.25)
    assert summary["uncertain_rate"] == pytest.approx(0.25)


# Checks that summarize_evidence() rejects an empty prediction list by raising a ValueError.
def test_summarize_evidence_rejects_empty_input() -> None:
    # Means: “I expect the code below to produce a ValueError.”
    with pytest.raises(ValueError):
        # Gives the function an empty list.
        summarize_evidence([])
