"""Tests for the independent self-verification module."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.verification.self_verifier import (
    SelfVerifier,
    batched_indices,
    build_self_verification_claim,
    clean_text,
    get_context,
    get_predicted_answer,
    get_question,
    run_self_verification,
    validate_predictions,
)


def test_clean_text_normalizes_whitespace() -> None:
    assert clean_text("  James   Watt\n invented  ") == "James Watt invented"


def test_clean_text_handles_none() -> None:
    assert clean_text(None) == ""


def test_get_question_supports_aliases() -> None:
    assert get_question({"question": "Who invented it?"}) == "Who invented it?"
    assert get_question({"question_text": "Where?"}) == "Where?"
    assert get_question({"query": "When?"}) == "When?"


def test_get_predicted_answer_supports_aliases() -> None:
    assert get_predicted_answer({"prediction_text": "James Watt"}) == "James Watt"
    assert get_predicted_answer({"predicted_answer": "Paris"}) == "Paris"
    assert get_predicted_answer({"answer": "42"}) == "42"


def test_get_context_supports_aliases() -> None:
    assert get_context({"context": "Main context"}) == "Main context"
    assert get_context({"passage": "Passage text"}) == "Passage text"
    assert get_context({"evidence_context": "Evidence"}) == "Evidence"


def test_build_self_verification_claim() -> None:
    claim = build_self_verification_claim(
        question="Who developed the theory of relativity?",
        answer="Albert Einstein",
    )

    assert claim == (
        'The answer to the question '
        '"Who developed the theory of relativity?" '
        'is "Albert Einstein".'
    )


def test_batched_indices() -> None:
    assert list(
        batched_indices(
            total_size=10,
            batch_size=4,
        )
    ) == [
        (0, 4),
        (4, 8),
        (8, 10),
    ]


def test_batched_indices_rejects_invalid_batch_size() -> None:
    with pytest.raises(
        ValueError,
        match="Batch size must be positive",
    ):
        list(
            batched_indices(
                total_size=10,
                batch_size=0,
            )
        )


def test_validate_predictions_rejects_empty_input() -> None:
    with pytest.raises(
        ValueError,
        match="Prediction list cannot be empty",
    ):
        validate_predictions([])


def test_validate_predictions_rejects_missing_question() -> None:
    predictions = [
        {
            "context": "Some evidence.",
            "predicted_answer": "Answer",
        }
    ]

    with pytest.raises(
        ValueError,
        match="has no question",
    ):
        validate_predictions(predictions)


def test_validate_predictions_rejects_missing_context() -> None:
    predictions = [
        {
            "question": "Question?",
            "predicted_answer": "Answer",
        }
    ]

    with pytest.raises(
        ValueError,
        match="has no context",
    ):
        validate_predictions(predictions)


def make_verifier(
    supported_threshold: float = 0.70,
    reject_threshold: float = 0.70,
) -> SelfVerifier:
    verifier = object.__new__(SelfVerifier)

    verifier.supported_threshold = supported_threshold
    verifier.reject_threshold = reject_threshold

    return verifier


def test_assign_label_supported() -> None:
    verifier = make_verifier()

    label = verifier._assign_label(
        entailment_probability=0.90,
        neutral_probability=0.05,
        contradiction_probability=0.05,
    )

    assert label == "SUPPORTED"


def test_assign_label_rejected() -> None:
    verifier = make_verifier()

    label = verifier._assign_label(
        entailment_probability=0.05,
        neutral_probability=0.05,
        contradiction_probability=0.90,
    )

    assert label == "REJECTED"


def test_assign_label_uncertain() -> None:
    verifier = make_verifier()

    label = verifier._assign_label(
        entailment_probability=0.50,
        neutral_probability=0.40,
        contradiction_probability=0.10,
    )

    assert label == "UNCERTAIN"


def test_contradiction_has_priority_over_entailment() -> None:
    verifier = make_verifier()

    label = verifier._assign_label(
        entailment_probability=0.80,
        neutral_probability=0.00,
        contradiction_probability=0.80,
    )

    assert label == "REJECTED"


def test_resolve_label_ids() -> None:
    verifier = object.__new__(SelfVerifier)

    verifier.model = SimpleNamespace(
        config=SimpleNamespace(
            id2label={
                0: "CONTRADICTION",
                1: "NEUTRAL",
                2: "ENTAILMENT",
            }
        )
    )

    assert verifier._resolve_label_ids() == {
        "contradiction": 0,
        "neutral": 1,
        "entailment": 2,
    }


def test_resolve_label_ids_rejects_invalid_model_labels() -> None:
    verifier = object.__new__(SelfVerifier)

    verifier.model = SimpleNamespace(
        config=SimpleNamespace(
            id2label={
                0: "LABEL_0",
                1: "LABEL_1",
                2: "LABEL_2",
            }
        )
    )

    with pytest.raises(
        ValueError,
        match="Could not resolve NLI labels",
    ):
        verifier._resolve_label_ids()


def test_run_self_verification_end_to_end_without_real_model(
    tmp_path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    records = [
        {
            "id": "example-1",
            "question": "Who developed relativity?",
            "context": "Albert Einstein developed the theory of relativity.",
            "predicted_answer": "Albert Einstein",
        },
        {
            "id": "example-2",
            "question": "Who developed relativity?",
            "context": "Albert Einstein developed the theory of relativity.",
            "predicted_answer": "",
        },
    ]

    with input_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(record) + "\n"
            )

    class FakeSelfVerifier:
        def __init__(
            self,
            **kwargs,
        ) -> None:
            self.kwargs = kwargs

        def verify_batch(
            self,
            contexts,
            claims,
        ):
            assert contexts
            assert claims
            assert len(contexts) == len(claims)

            return [
                {
                    "self_verification_label": "SUPPORTED",
                    "self_verification_score": 0.90,
                    "self_entailment_probability": 0.95,
                    "self_neutral_probability": 0.03,
                    "self_contradiction_probability": 0.02,
                }
                for _ in claims
            ]

    monkeypatch.setattr(
        "src.verification.self_verifier.SelfVerifier",
        FakeSelfVerifier,
    )

    results = run_self_verification(
        input_path=input_path,
        output_path=output_path,
        model_name="fake-model",
        batch_size=2,
        max_context_tokens=100,
        max_claim_tokens=50,
        supported_threshold=0.70,
        reject_threshold=0.70,
    )

    assert len(results) == 2
    assert output_path.exists()

    supported = results[0]

    assert supported["id"] == "example-1"
    assert supported["self_verification_label"] == "SUPPORTED"
    assert supported["self_verification_score"] == pytest.approx(0.90)
    assert supported["self_verification_model"] == "fake-model"

    assert (
        supported["self_verification_claim"]
        == 'The answer to the question '
        '"Who developed relativity?" '
        'is "Albert Einstein".'
    )

    rejected = results[1]

    assert rejected["id"] == "example-2"
    assert rejected["self_verification_label"] == "REJECTED"
    assert rejected["self_verification_score"] == pytest.approx(-1.0)
    assert rejected["self_contradiction_probability"] == pytest.approx(1.0)
    assert rejected["self_verification_claim"] is None


@pytest.mark.parametrize(
    ("supported_threshold", "reject_threshold"),
    [
        (-0.01, 0.70),
        (1.01, 0.70),
        (0.70, -0.01),
        (0.70, 1.01),
    ],
)
def test_run_self_verification_rejects_invalid_thresholds(
    tmp_path,
    supported_threshold,
    reject_threshold,
) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    input_path.write_text(
        json.dumps(
            {
                "question": "Question?",
                "context": "Context.",
                "predicted_answer": "Answer",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        run_self_verification(
            input_path=input_path,
            output_path=output_path,
            model_name="unused-model",
            batch_size=1,
            max_context_tokens=100,
            max_claim_tokens=50,
            supported_threshold=supported_threshold,
            reject_threshold=reject_threshold,
        )