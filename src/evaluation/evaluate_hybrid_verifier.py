"""
Compare confidence-only, lexical-verifier, and hybrid-verifier prototype policies.

This module evaluates three earlier selective-QA operating points:

1. Confidence-only:
   answer when calibrated confidence exceeds a fixed threshold.

2. Lexical verifier:
   answer only when lexical evidence is labeled SUPPORTED.

3. Hybrid verifier:
   answer only when the combined confidence/lexical/NLI verifier is labeled
   SUPPORTED.

Underlying QA candidate correctness is independent of the later ANSWER/REJECT
policy.

For answerable examples, this diagnostic prototype uses relaxed correctness:

    Exact Match == 1
        OR
    token F1 >= relaxed_f1_threshold

For unanswerable examples, the forced-answer QA candidate is always incorrect.
A later verifier may correctly reject that candidate, but rejection must not
retroactively make the underlying candidate correct.

`overall_relaxed_accuracy` is retained for historical output compatibility.
Despite its name, it equals `routing_action_accuracy`:

    (correct_answered + correct_rejected) / total

This file belongs to the earlier fixed-policy prototype analysis and should not
be mixed with the project's final Exact-Match-based score-ranking/AURC results.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.evaluation.metrics import normalize_answer, parse_answerability
from src.utils.io import load_jsonl

DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/calibration_with_hybrid_evidence.jsonl"
)
DEFAULT_CONFIDENCE_THRESHOLD = 0.50
DEFAULT_LEXICAL_SUPPORTED_LABEL = "SUPPORTED"
DEFAULT_HYBRID_SUPPORTED_LABEL = "SUPPORTED"
DEFAULT_RELAXED_F1_THRESHOLD = 0.80
DEFAULT_MAX_EXAMPLES = 10

VALID_SUPPORT_LABELS = {"SUPPORTED", "WEAK", "UNSUPPORTED"}

CALIBRATED_CONFIDENCE_FIELDS = (
    "calibrated_confidence",
    "confidence_calibrated",
    "calibrated_probability",
    "temperature_scaled_confidence",
    "qa_calibrated_confidence",
)
LEXICAL_LABEL_FIELDS = (
    "evidence_support",
    "evidence_label",
    "lexical_evidence_support",
    "lexical_label",
)
HYBRID_LABEL_FIELDS = (
    "hybrid_evidence_support",
    "hybrid_support",
    "hybrid_label",
)
HYBRID_SCORE_FIELDS = ("hybrid_evidence_score", "hybrid_score")
PREDICTION_FIELDS = (
    "predicted_answer",
    "prediction_text",
    "prediction_answer",
    "answer",
)
REFERENCE_FIELDS = (
    "reference_answers",
    "gold_answers",
    "answers",
    "reference_answer",
    "gold_answer",
)
ANSWERABILITY_FIELDS = (
    "is_answerable",
    "answerable",
    "gold_is_answerable",
)


def get_first_value(
    prediction: dict[str, Any],
    field_names: Sequence[str],
    default: Any = None,
) -> Any:
    """Return the first available non-None value from ordered aliases."""
    for field_name in field_names:
        if field_name in prediction and prediction[field_name] is not None:
            return prediction[field_name]
    return default


def get_first_numeric_value(
    prediction: dict[str, Any],
    field_names: Sequence[str],
    default: float | None = None,
) -> float:
    """Return the first finite numeric value; malformed present values fail."""
    for field_name in field_names:
        if field_name not in prediction or prediction[field_name] is None:
            continue

        value = prediction[field_name]

        if isinstance(value, bool):
            raise ValueError(  # noqa: TRY004
                f"Boolean value is invalid for numeric field "
                f"{field_name!r}: {value!r}."
            )

        try:
            numeric_value = float(value)

        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid numeric value in {field_name!r}: {value!r}."
            ) from error

        if not math.isfinite(numeric_value):
            raise ValueError(
                f"Non-finite numeric value in "
                f"{field_name!r}: {numeric_value}."
            )

        return numeric_value

    if default is not None:
        return float(default)

    raise ValueError(
        "None of the expected numeric fields were found: "
        f"{list(field_names)}"
    )


def get_optional_numeric_value(
    prediction: dict[str, Any],
    field_names: Sequence[str],
) -> float | None:
    """Return an optional finite numeric value without inventing a default."""
    if not any(
        field_name in prediction
        and prediction[field_name] is not None
        for field_name in field_names
    ):
        return None

    return get_first_numeric_value(
        prediction,
        field_names,
    )


def validate_probability(
    value: float,
    signal_name: str,
) -> float:
    """Require a finite probability-like value in [0, 1]."""
    if (
        not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(
            f"{signal_name} must be a finite value "
            f"in [0, 1], received {value}."
        )

    return value


def get_calibrated_confidence(
    prediction: dict[str, Any],
) -> float:
    """Retrieve calibrated confidence without falling back to raw confidence."""
    explicit_value = get_optional_numeric_value(
        prediction,
        CALIBRATED_CONFIDENCE_FIELDS,
    )

    if explicit_value is not None:
        return validate_probability(
            explicit_value,
            "Calibrated confidence",
        )

    if (
        prediction.get("confidence_is_calibrated") is True
        and prediction.get("confidence") is not None
    ):
        confidence = get_first_numeric_value(
            prediction,
            ("confidence",),
        )

        return validate_probability(
            confidence,
            "Calibrated confidence",
        )

    raise ValueError(
        "Prediction does not contain usable "
        "calibrated confidence."
    )


def get_predicted_answer(
    prediction: dict[str, Any],
) -> str:
    """Extract the predicted answer from supported field aliases."""
    value = get_first_value(
        prediction,
        PREDICTION_FIELDS,
        default="",
    )

    return (
        ""
        if value is None
        else str(value).strip()
    )


def extract_reference_value(
    value: Any,
) -> list[str]:
    """Recursively extract reference strings from common answer structures."""
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, dict):
        for field_name in (
            "text",
            "answer",
            "answers",
            "answer_text",
        ):
            if field_name in value:
                return extract_reference_value(
                    value[field_name]
                )

        return []

    if isinstance(
        value,
        (list, tuple),
    ):
        references: list[str] = []

        for item in value:
            references.extend(
                extract_reference_value(item)
            )

        return references

    return [str(value)]


def get_reference_answers(
    prediction: dict[str, Any],
) -> list[str]:
    """Extract references from the first supported reference field."""
    value = get_first_value(
        prediction,
        REFERENCE_FIELDS,
        default=None,
    )

    if value is None:
        return []

    return [
        str(reference).strip()
        for reference in extract_reference_value(value)
    ]


def parse_answerability_value(
    value: Any,
) -> bool:
    """Compatibility wrapper around the shared strict answerability parser."""
    return parse_answerability(value)


def get_is_answerable(
    prediction: dict[str, Any],
) -> bool:
    """Retrieve required explicit answerability metadata."""
    for field_name in ANSWERABILITY_FIELDS:
        if (
            field_name not in prediction
            or prediction[field_name] is None
        ):
            continue

        try:
            return parse_answerability_value(
                prediction[field_name]
            )

        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid answerability value in "
                f"{field_name!r}: "
                f"{prediction[field_name]!r}."
            ) from error

    raise ValueError(
        "Prediction does not contain explicit "
        "answerability metadata."
    )


def exact_match_score(
    prediction: str,
    reference: str,
) -> float:
    """Return normalized Exact Match for one prediction/reference pair."""
    return float(
        normalize_answer(prediction)
        == normalize_answer(reference)
    )


def token_f1_score(
    prediction: str,
    reference: str,
) -> float:
    """Calculate token-level F1 for one prediction/reference pair."""
    prediction_tokens = (
        normalize_answer(prediction).split()
    )

    reference_tokens = (
        normalize_answer(reference).split()
    )

    if (
        not prediction_tokens
        and not reference_tokens
    ):
        return 1.0

    if (
        not prediction_tokens
        or not reference_tokens
    ):
        return 0.0

    common_tokens = (
        Counter(prediction_tokens)
        & Counter(reference_tokens)
    )

    overlap_count = sum(
        common_tokens.values()
    )

    if overlap_count == 0:
        return 0.0

    precision = (
        overlap_count
        / len(prediction_tokens)
    )

    recall = (
        overlap_count
        / len(reference_tokens)
    )

    return (
        2.0
        * precision
        * recall
        / (precision + recall)
    )


def calculate_answer_scores(
    predicted_answer: str,
    reference_answers: list[str],
) -> tuple[float, float]:
    """
    Calculate best EM/F1 over usable references.

    No references means an unanswerable forced-answer candidate, which receives
    zero answer-text credit. Rejection is evaluated separately.
    """
    usable_references = [
        reference
        for reference in reference_answers
        if normalize_answer(reference)
    ]

    if not usable_references:
        return (
            0.0,
            0.0,
        )

    exact_match = max(
        exact_match_score(
            predicted_answer,
            reference,
        )
        for reference in usable_references
    )

    token_f1 = max(
        token_f1_score(
            predicted_answer,
            reference,
        )
        for reference in usable_references
    )

    return (
        exact_match,
        token_f1,
    )


def is_prediction_correct(
    prediction: dict[str, Any],
    relaxed_f1_threshold: float,
) -> tuple[bool, float, float]:
    """Determine underlying forced-answer QA candidate correctness."""
    if (
        not math.isfinite(relaxed_f1_threshold)
        or not 0.0
        <= relaxed_f1_threshold
        <= 1.0
    ):
        raise ValueError(
            "relaxed_f1_threshold must be finite "
            "and lie in [0, 1]."
        )

    predicted_answer = get_predicted_answer(
        prediction
    )

    reference_answers = get_reference_answers(
        prediction
    )

    is_answerable = get_is_answerable(
        prediction
    )

    usable_references = [
        reference
        for reference in reference_answers
        if normalize_answer(reference)
    ]

    if is_answerable:
        if not usable_references:
            raise ValueError(
                "Answerable prediction does not contain "
                "a usable reference answer. "
                f"Prediction id: "
                f"{prediction.get('id')!r}"
            )

        (
            exact_match,
            token_f1,
        ) = calculate_answer_scores(
            predicted_answer,
            usable_references,
        )

        correct = (
            exact_match == 1.0
            or token_f1
            >= relaxed_f1_threshold
        )

        return (
            correct,
            exact_match,
            token_f1,
        )

    if usable_references:
        raise ValueError(
            "Unanswerable prediction contains "
            "non-empty reference answers. "
            f"Prediction id: "
            f"{prediction.get('id')!r}"
        )

    return (
        False,
        0.0,
        0.0,
    )


def get_support_label(
    prediction: dict[str, Any],
    field_names: Sequence[str],
    signal_name: str,
) -> str:
    """Retrieve and validate a lexical or hybrid support label."""
    value = get_first_value(
        prediction,
        field_names,
        default=None,
    )

    if value is None:
        raise ValueError(
            f"Prediction does not contain "
            f"{signal_name}."
        )

    label = (
        str(value)
        .strip()
        .upper()
    )

    if label not in VALID_SUPPORT_LABELS:
        raise ValueError(
            f"Invalid {signal_name}: "
            f"{label!r}."
        )

    return label


def get_lexical_label(
    prediction: dict[str, Any],
) -> str:
    """Retrieve the lexical evidence-support label."""
    return get_support_label(
        prediction,
        LEXICAL_LABEL_FIELDS,
        "lexical support label",
    )


def get_hybrid_label(
    prediction: dict[str, Any],
) -> str:
    """Retrieve the hybrid evidence-support label."""
    return get_support_label(
        prediction,
        HYBRID_LABEL_FIELDS,
        "hybrid support label",
    )


def get_hybrid_score(
    prediction: dict[str, Any],
) -> float | None:
    """Retrieve the optional hybrid score for diagnostics."""
    score = get_optional_numeric_value(
        prediction,
        HYBRID_SCORE_FIELDS,
    )

    if score is None:
        return None

    return validate_probability(
        score,
        "Hybrid evidence score",
    )


def confidence_only_answers(
    prediction: dict[str, Any],
    confidence_threshold: float,
) -> bool:
    """Return whether the fixed confidence-only policy chooses ANSWER."""
    return (
        get_calibrated_confidence(prediction)
        >= confidence_threshold
    )


def lexical_verifier_answers(
    prediction: dict[str, Any],
) -> bool:
    """Return whether the lexical-verifier policy chooses ANSWER."""
    return (
        get_lexical_label(prediction)
        == DEFAULT_LEXICAL_SUPPORTED_LABEL
    )


def hybrid_verifier_answers(
    prediction: dict[str, Any],
) -> bool:
    """Return whether the hybrid-verifier policy chooses ANSWER."""
    return (
        get_hybrid_label(prediction)
        == DEFAULT_HYBRID_SUPPORTED_LABEL
    )


def build_example_diagnostic(
    index: int,
    prediction: dict[str, Any],
    exact_match: float,
    token_f1: float,
) -> dict[str, Any]:
    """Build a compact record for qualitative inspection."""
    return {
        "index": index,
        "id": prediction.get("id"),
        "question": prediction.get(
            "question",
            "",
        ),
        "is_answerable": get_is_answerable(
            prediction
        ),
        "predicted_answer": get_predicted_answer(
            prediction
        ),
        "reference_answers": get_reference_answers(
            prediction
        ),
        "exact_match": exact_match,
        "token_f1": token_f1,
        "confidence": get_calibrated_confidence(
            prediction
        ),
        "lexical_support": get_lexical_label(
            prediction
        ),
        "hybrid_support": get_hybrid_label(
            prediction
        ),
        "hybrid_score": get_hybrid_score(
            prediction
        ),
    }


def evaluate_system(
    predictions: list[dict[str, Any]],
    answer_decisions: list[bool],
    relaxed_f1_threshold: float,
) -> dict[str, Any]:
    """Evaluate one prototype ANSWER/REJECT policy."""
    if not predictions:
        raise ValueError(
            "Prediction list cannot be empty."
        )

    if (
        len(predictions)
        != len(answer_decisions)
    ):
        raise ValueError(
            "Prediction and decision counts must match."
        )

    if any(
        not isinstance(
            decision,
            bool,
        )
        for decision in answer_decisions
    ):
        raise ValueError(
            "answer_decisions must contain "
            "only Boolean values."
        )

    if (
        not math.isfinite(
            relaxed_f1_threshold
        )
        or not 0.0
        <= relaxed_f1_threshold
        <= 1.0
    ):
        raise ValueError(
            "relaxed_f1_threshold must be finite "
            "and lie in [0, 1]."
        )

    total_count = len(
        predictions
    )

    answered_count = 0
    abstained_count = 0

    correct_answered_count = 0
    wrong_answered_count = 0

    correct_rejected_count = 0
    wrong_rejected_count = 0

    total_exact_match = 0.0
    total_token_f1 = 0.0

    answered_exact_match = 0.0
    answered_token_f1 = 0.0

    wrong_answered_examples: list[
        dict[str, Any]
    ] = []

    correct_rejected_examples: list[
        dict[str, Any]
    ] = []

    for index, (
        prediction,
        should_answer,
    ) in enumerate(
        zip(
            predictions,
            answer_decisions,
        ),
        start=1,
    ):
        (
            is_correct,
            exact_match,
            token_f1,
        ) = is_prediction_correct(
            prediction,
            relaxed_f1_threshold,
        )

        total_exact_match += (
            exact_match
        )

        total_token_f1 += (
            token_f1
        )

        if should_answer:
            answered_count += 1

            answered_exact_match += (
                exact_match
            )

            answered_token_f1 += (
                token_f1
            )

            if is_correct:
                correct_answered_count += 1

            else:
                wrong_answered_count += 1

                wrong_answered_examples.append(
                    build_example_diagnostic(
                        index,
                        prediction,
                        exact_match,
                        token_f1,
                    )
                )

        else:
            abstained_count += 1

            if is_correct:
                wrong_rejected_count += 1

            else:
                correct_rejected_count += 1

                correct_rejected_examples.append(
                    build_example_diagnostic(
                        index,
                        prediction,
                        exact_match,
                        token_f1,
                    )
                )

    coverage = (
        answered_count
        / total_count
    )

    abstention_rate = (
        abstained_count
        / total_count
    )

    if answered_count > 0:
        selective_accuracy = (
            correct_answered_count
            / answered_count
        )

        selective_risk = (
            wrong_answered_count
            / answered_count
        )

        answered_average_exact_match = (
            answered_exact_match
            / answered_count
        )

        answered_average_token_f1 = (
            answered_token_f1
            / answered_count
        )

    else:
        selective_accuracy = 0.0
        selective_risk = 0.0
        answered_average_exact_match = 0.0
        answered_average_token_f1 = 0.0

    routing_action_accuracy = (
        correct_answered_count
        + correct_rejected_count
    ) / total_count

    return {
        "total": total_count,
        "answered": answered_count,
        "abstained": abstained_count,
        "coverage": coverage,
        "abstention_rate": (
            abstention_rate
        ),
        "selective_accuracy": (
            selective_accuracy
        ),
        "selective_risk": (
            selective_risk
        ),
        "correct_answered": (
            correct_answered_count
        ),
        "wrong_answered": (
            wrong_answered_count
        ),
        "correct_rejected": (
            correct_rejected_count
        ),
        "wrong_rejected": (
            wrong_rejected_count
        ),
        "routing_action_accuracy": (
            routing_action_accuracy
        ),
        "overall_relaxed_accuracy": (
            routing_action_accuracy
        ),
        "average_exact_match": (
            total_exact_match
            / total_count
        ),
        "average_token_f1": (
            total_token_f1
            / total_count
        ),
        "answered_average_exact_match": (
            answered_average_exact_match
        ),
        "answered_average_token_f1": (
            answered_average_token_f1
        ),
        "wrong_answered_examples": (
            wrong_answered_examples
        ),
        "correct_rejected_examples": (
            correct_rejected_examples
        ),
    }


def validate_system_metrics(
    metrics: dict[str, Any],
) -> None:
    """Check structural and mathematical invariants for one result."""
    total = int(
        metrics[
            "total"
        ]
    )

    answered = int(
        metrics[
            "answered"
        ]
    )

    abstained = int(
        metrics[
            "abstained"
        ]
    )

    if (
        answered
        + abstained
        != total
    ):
        raise RuntimeError(
            "Answered and rejected counts "
            "do not sum to total."
        )

    if (
        int(
            metrics[
                "correct_answered"
            ]
        )
        + int(
            metrics[
                "wrong_answered"
            ]
        )
        != answered
    ):
        raise RuntimeError(
            "Answered correctness counts "
            "are inconsistent."
        )

    if (
        int(
            metrics[
                "correct_rejected"
            ]
        )
        + int(
            metrics[
                "wrong_rejected"
            ]
        )
        != abstained
    ):
        raise RuntimeError(
            "Rejected correctness counts "
            "are inconsistent."
        )

    if not math.isclose(
        float(
            metrics[
                "coverage"
            ]
        ),
        answered
        / total,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Coverage is inconsistent "
            "with answered count."
        )

    if not math.isclose(
        float(
            metrics[
                "abstention_rate"
            ]
        ),
        abstained
        / total,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Rejection rate is inconsistent "
            "with rejected count."
        )

    if (
        answered > 0
        and not math.isclose(
            float(
                metrics[
                    "selective_risk"
                ]
            ),
            1.0
            - float(
                metrics[
                    "selective_accuracy"
                ]
            ),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError(
            "Selective risk is inconsistent "
            "with selective accuracy."
        )

    expected_routing_accuracy = (
        int(
            metrics[
                "correct_answered"
            ]
        )
        + int(
            metrics[
                "correct_rejected"
            ]
        )
    ) / total

    if not math.isclose(
        float(
            metrics[
                "routing_action_accuracy"
            ]
        ),
        expected_routing_accuracy,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Routing-action accuracy is "
            "inconsistent with action counts."
        )

    if not math.isclose(
        float(
            metrics[
                "overall_relaxed_accuracy"
            ]
        ),
        expected_routing_accuracy,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Historical overall_relaxed_accuracy "
            "is inconsistent with "
            "routing-action accuracy."
        )


def format_optional_metric(
    value: float | None,
) -> str:
    """Format an optional floating-point diagnostic."""
    return (
        "N/A"
        if value is None
        else f"{value:.4f}"
    )


def print_system_metrics(
    system_name: str,
    metrics: dict[str, Any],
) -> None:
    """Print main operating-point metrics for one prototype policy."""
    print(
        "\n"
        + "=" * 60
    )

    print(
        system_name
    )

    print(
        "=" * 60
    )

    print(
        f"Total:                 "
        f"{metrics['total']}"
    )

    print(
        f"Answered:              "
        f"{metrics['answered']}"
    )

    print(
        f"Rejected:              "
        f"{metrics['abstained']}"
    )

    print(
        f"Coverage:              "
        f"{metrics['coverage']:.4f}"
    )

    print(
        f"Rejection rate:        "
        f"{metrics['abstention_rate']:.4f}"
    )

    print(
        f"Selective accuracy:    "
        f"{metrics['selective_accuracy']:.4f}"
    )

    print(
        f"Selective risk:        "
        f"{metrics['selective_risk']:.4f}"
    )

    print(
        f"Correct answered:      "
        f"{metrics['correct_answered']}"
    )

    print(
        f"Wrong answered:        "
        f"{metrics['wrong_answered']}"
    )

    print(
        f"Correct rejected:      "
        f"{metrics['correct_rejected']}"
    )

    print(
        f"Wrong rejected:        "
        f"{metrics['wrong_rejected']}"
    )

    print(
        f"Routing action acc.:   "
        f"{metrics['routing_action_accuracy']:.4f}"
    )

    print(
        "Answered average EM:   "
        f"{metrics['answered_average_exact_match']:.4f}"
    )

    print(
        "Answered average F1:   "
        f"{metrics['answered_average_token_f1']:.4f}"
    )


def print_example(
    example: dict[str, Any],
) -> None:
    """Print one qualitative diagnostic example."""
    print(
        f"\nIndex: "
        f"{example['index']}"
    )

    print(
        f"ID: "
        f"{example['id']}"
    )

    print(
        f"Answerable: "
        f"{example['is_answerable']}"
    )

    print(
        f"Question: "
        f"{example['question']}"
    )

    print(
        f"Prediction: "
        f"{example['predicted_answer']}"
    )

    print(
        f"Reference: "
        f"{example['reference_answers']}"
    )

    print(
        f"Token F1: "
        f"{example['token_f1']:.4f}"
    )

    print(
        f"Confidence: "
        f"{example['confidence']:.4f}"
    )

    print(
        f"Lexical: "
        f"{example['lexical_support']}"
    )

    print(
        f"Hybrid: "
        f"{example['hybrid_support']}"
    )

    print(
        "Hybrid score: "
        f"{format_optional_metric(
            example['hybrid_score']
        )}"
    )


def print_critical_examples(
    system_name: str,
    metrics: dict[str, Any],
    max_examples: int,
) -> None:
    """Print harmful answers and beneficial rejections."""
    wrong_examples = (
        metrics[
            "wrong_answered_examples"
        ]
    )

    print(
        "\n"
        + "-" * 60
    )

    print(
        f"{system_name} — "
        "WRONG ANSWERED examples"
    )

    print(
        "-" * 60
    )

    if not wrong_examples:
        print(
            "None."
        )

    for example in wrong_examples[
        :max_examples
    ]:
        print_example(
            example
        )

    correct_rejected_examples = (
        metrics[
            "correct_rejected_examples"
        ]
    )

    print(
        "\n"
        + "-" * 60
    )

    print(
        f"{system_name} — "
        "CORRECTLY REJECTED "
        "INCORRECT CANDIDATES"
    )

    print(
        "-" * 60
    )

    if not correct_rejected_examples:
        print(
            "None."
        )

    for example in correct_rejected_examples[
        :max_examples
    ]:
        print_example(
            example
        )


def validate_runtime_settings(
    confidence_threshold: float,
    relaxed_f1_threshold: float,
    max_examples: int,
) -> None:
    """Validate thresholds and display settings."""
    if (
        not math.isfinite(
            confidence_threshold
        )
        or not 0.0
        <= confidence_threshold
        <= 1.0
    ):
        raise ValueError(
            "confidence_threshold must be finite "
            "and lie in [0, 1]."
        )

    if (
        not math.isfinite(
            relaxed_f1_threshold
        )
        or not 0.0
        <= relaxed_f1_threshold
        <= 1.0
    ):
        raise ValueError(
            "relaxed_f1_threshold must be finite "
            "and lie in [0, 1]."
        )

    if max_examples < 0:
        raise ValueError(
            "max_examples cannot be negative."
        )


def validate_predictions(
    predictions: list[dict[str, Any]],
) -> None:
    """Validate all inputs required by the three prototype policies."""
    if not predictions:
        raise ValueError(
            "Prediction list cannot be empty."
        )

    for index, prediction in enumerate(
        predictions,
        start=1,
    ):
        try:
            if not any(
                field_name in prediction
                for field_name
                in PREDICTION_FIELDS
            ):
                raise ValueError(
                    "No supported prediction-answer "
                    "field was found."
                )

            is_answerable = (
                get_is_answerable(
                    prediction
                )
            )

            references = (
                get_reference_answers(
                    prediction
                )
            )

            usable_references = [
                reference
                for reference in references
                if normalize_answer(
                    reference
                )
            ]

            if (
                is_answerable
                and not usable_references
            ):
                raise ValueError(
                    "Answerable record has no "
                    "usable reference answer."
                )

            if (
                not is_answerable
                and usable_references
            ):
                raise ValueError(
                    "Unanswerable record contains "
                    "non-empty references."
                )

            get_calibrated_confidence(
                prediction
            )

            get_lexical_label(
                prediction
            )

            get_hybrid_label(
                prediction
            )

            get_hybrid_score(
                prediction
            )

        except (
            ValueError,
            TypeError,
            KeyError,
        ) as error:
            raise ValueError(
                f"Prediction {index} "
                f"failed validation: {error}"
            ) from error


def run_evaluation(
    input_path: str | Path,
    confidence_threshold: float,
    relaxed_f1_threshold: float,
    max_examples: int,
) -> dict[str, dict[str, Any]]:
    """Compare confidence-only, lexical, and hybrid prototype policies."""
    validate_runtime_settings(
        confidence_threshold,
        relaxed_f1_threshold,
        max_examples,
    )

    predictions = (
        load_jsonl(
            input_path
        )
    )

    validate_predictions(
        predictions
    )

    confidence_decisions = [
        confidence_only_answers(
            prediction,
            confidence_threshold,
        )
        for prediction in predictions
    ]

    lexical_decisions = [
        lexical_verifier_answers(
            prediction
        )
        for prediction in predictions
    ]

    hybrid_decisions = [
        hybrid_verifier_answers(
            prediction
        )
        for prediction in predictions
    ]

    confidence_metrics = (
        evaluate_system(
            predictions,
            confidence_decisions,
            relaxed_f1_threshold,
        )
    )

    lexical_metrics = (
        evaluate_system(
            predictions,
            lexical_decisions,
            relaxed_f1_threshold,
        )
    )

    hybrid_metrics = (
        evaluate_system(
            predictions,
            hybrid_decisions,
            relaxed_f1_threshold,
        )
    )

    for metrics in (
        confidence_metrics,
        lexical_metrics,
        hybrid_metrics,
    ):
        validate_system_metrics(
            metrics
        )

    results = {
        "confidence_only": (
            confidence_metrics
        ),
        "lexical_verifier": (
            lexical_metrics
        ),
        "hybrid_verifier": (
            hybrid_metrics
        ),
    }

    print(
        "\nComparative prototype "
        "selective-QA evaluation"
    )

    print(
        f"Input: "
        f"{input_path}"
    )

    print(
        "Correctness: forced-answer candidates; "
        "unanswerable candidates are incorrect"
    )

    print(
        "Answerable relaxed F1 threshold: "
        f"{relaxed_f1_threshold:.4f}"
    )

    print(
        f"Confidence threshold: "
        f"{confidence_threshold:.4f}"
    )

    print_system_metrics(
        "CONFIDENCE-ONLY BASELINE",
        confidence_metrics,
    )

    print_system_metrics(
        "LEXICAL VERIFIER",
        lexical_metrics,
    )

    print_system_metrics(
        "HYBRID VERIFIER",
        hybrid_metrics,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        f"{'System':<24}"
        f"{'Coverage':>10}"
        f"{'Accuracy':>12}"
        f"{'Risk':>10}"
        f"{'Wrong':>8}"
        f"{'Correct Reject':>16}"
    )

    print(
        "-" * 80
    )

    for system_name, metrics in (
        (
            "Confidence-only",
            confidence_metrics,
        ),
        (
            "Lexical verifier",
            lexical_metrics,
        ),
        (
            "Hybrid verifier",
            hybrid_metrics,
        ),
    ):
        print(
            f"{system_name:<24}"
            f"{metrics['coverage']:>10.4f}"
            f"{metrics['selective_accuracy']:>12.4f}"
            f"{metrics['selective_risk']:>10.4f}"
            f"{metrics['wrong_answered']:>8}"
            f"{metrics['correct_rejected']:>16}"
        )

    print_critical_examples(
        "HYBRID VERIFIER",
        hybrid_metrics,
        max_examples,
    )

    return results


def parse_arguments() -> argparse.Namespace:
    """Parse prototype hybrid-verifier evaluation settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare confidence-only, lexical, "
            "and hybrid selective-QA prototype policies."
        )
    )

    parser.add_argument(
        "--input",
        default=str(
            DEFAULT_INPUT_PATH
        ),
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=(
            DEFAULT_CONFIDENCE_THRESHOLD
        ),
    )

    parser.add_argument(
        "--relaxed-f1-threshold",
        type=float,
        default=(
            DEFAULT_RELAXED_F1_THRESHOLD
        ),
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=(
            DEFAULT_MAX_EXAMPLES
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_evaluation(
        input_path=args.input,
        confidence_threshold=(
            args.confidence_threshold
        ),
        relaxed_f1_threshold=(
            args.relaxed_f1_threshold
        ),
        max_examples=(
            args.max_examples
        ),
    )