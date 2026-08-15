"""
Analyze errors made by the prototype lexical evidence verifier.

The analysis compares lexical verifier labels (SUPPORTED, WEAK, UNSUPPORTED)
with the correctness of the underlying forced-answer QA candidate.

Two correctness views are reported for answerable examples:

- strict correctness: normalized Exact Match
- relaxed correctness: Exact Match or token F1 >= 0.80

For unanswerable examples, the underlying forced-answer QA candidate is always
incorrect. A later ANSWER / VERIFY / ABSTAIN routing decision is diagnostic
metadata only and must not change candidate correctness.

This file analyzes the earlier lexical-verifier prototype. It is not the final
score-ranking/AURC evaluator.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.calibration.calibration_metrics import (
    is_prediction_correct as calibration_is_prediction_correct,
)
from src.calibration.calibration_metrics import (
    normalize_answer as calibration_normalize_answer,
)
from src.evaluation.metrics import parse_answerability
from src.utils.io import load_jsonl, save_jsonl

DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/calibration_with_evidence.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "outputs/analysis/evidence_error_cases.jsonl"
)

DEFAULT_RELAXED_F1_THRESHOLD = 0.80


VALID_EVIDENCE_LABELS = {
    "SUPPORTED",
    "WEAK",
    "UNSUPPORTED",
}

VALID_ROUTING_DECISIONS = {
    "ANSWER",
    "VERIFY",
    "ABSTAIN",
}


PREDICTION_FIELDS = (
    "prediction_text",
    "predicted_answer",
    "prediction_answer",
    "answer",
)

REFERENCE_FIELDS = (
    "reference_answers",
    "gold_answers",
    "gold_answer",
    "reference_answer",
    "answers",
)

EVIDENCE_LABEL_FIELDS = (
    "evidence_support",
    "evidence_label",
    "lexical_evidence_support",
    "lexical_label",
)

EVIDENCE_SCORE_FIELDS = (
    "combined_evidence_score",
    "evidence_score",
    "lexical_evidence_score",
)

CALIBRATED_CONFIDENCE_FIELDS = (
    "calibrated_confidence",
    "temperature_scaled_confidence",
    "qa_calibrated_confidence",
    "confidence_calibrated",
    "calibrated_probability",
)


def get_first_value(
    prediction: dict[str, Any],
    field_names: Sequence[str],
    default: Any = None,
) -> Any:
    """Return the first available non-None value from ordered aliases."""

    for field_name in field_names:
        if (
            field_name in prediction
            and prediction[field_name] is not None
        ):
            return prediction[field_name]

    return default


def get_predicted_answer(
    prediction: dict[str, Any],
) -> str:
    """Return the first available predicted-answer field as text."""

    value = get_first_value(
        prediction,
        PREDICTION_FIELDS,
        default="",
    )

    return (
        ""
        if value is None
        else str(value)
    )


def has_prediction_field(
    prediction: dict[str, Any],
) -> bool:
    """Return whether the record contains a supported prediction field."""

    return any(
        field_name in prediction
        for field_name in PREDICTION_FIELDS
    )


def extract_answer_texts(
    value: Any,
) -> list[str]:
    """
    Extract answer strings from common nested reference structures.

    Unknown dictionaries are not traversed blindly because metadata such as
    character offsets must not become accidental answer text.
    """

    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, dict):
        for key in (
            "text",
            "answer",
            "answers",
            "answer_text",
            "reference_answer",
            "reference_answers",
        ):
            if key in value:
                return extract_answer_texts(
                    value[key]
                )

        return []

    if isinstance(
        value,
        (list, tuple),
    ):
        answers: list[str] = []

        for item in value:
            answers.extend(
                extract_answer_texts(
                    item
                )
            )

        return answers

    return [
        str(value)
    ]


def get_gold_answers(
    prediction: dict[str, Any],
) -> list[str]:
    """
    Return the first non-empty supported reference collection.

    If reference fields are present but all are empty, return an empty list.
    """

    for field_name in REFERENCE_FIELDS:
        if field_name not in prediction:
            continue

        answers = [
            str(answer).strip()
            for answer in extract_answer_texts(
                prediction[field_name]
            )
            if str(answer).strip()
        ]

        if answers:
            return list(
                dict.fromkeys(
                    answers
                )
            )

    return []


def normalize_answer(
    text: Any,
) -> str:
    """Normalize text using the same rule as calibration correctness."""

    return calibration_normalize_answer(
        str(text)
    )


def parse_boolean(
    value: Any,
    default: bool = True,
) -> bool:
    """
    Compatibility wrapper for historical callers.

    None uses the supplied default; malformed non-None values raise an error.
    """

    if value is None:
        return default

    try:
        return parse_answerability(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            "Could not interpret Boolean value: "
            f"{value!r}."
        ) from error


def get_is_answerable(
    prediction: dict[str, Any],
) -> bool:
    """Retrieve required answerability metadata using strict parsing."""

    if (
        "is_answerable"
        not in prediction
    ):
        raise ValueError(
            "Prediction does not contain "
            "'is_answerable'."
        )

    try:
        return parse_answerability(
            prediction[
                "is_answerable"
            ]
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            "Invalid is_answerable value: "
            f"{prediction.get('is_answerable')!r}."
        ) from error


def calculate_exact_match(
    predicted_answer: str,
    gold_answer: str,
) -> float:
    """Return normalized Exact Match for one prediction/reference pair."""

    return float(
        normalize_answer(
            predicted_answer
        )
        == normalize_answer(
            gold_answer
        )
    )


def calculate_token_f1(
    predicted_answer: str,
    gold_answer: str,
) -> float:
    """Calculate token-level F1 for one prediction/reference pair."""

    predicted_tokens = (
        normalize_answer(
            predicted_answer
        ).split()
    )

    gold_tokens = (
        normalize_answer(
            gold_answer
        ).split()
    )

    if (
        not predicted_tokens
        and not gold_tokens
    ):
        return 1.0

    if (
        not predicted_tokens
        or not gold_tokens
    ):
        return 0.0

    common_tokens = (
        Counter(
            predicted_tokens
        )
        & Counter(
            gold_tokens
        )
    )

    shared_count = sum(
        common_tokens.values()
    )

    if shared_count == 0:
        return 0.0

    precision = (
        shared_count
        / len(
            predicted_tokens
        )
    )

    recall = (
        shared_count
        / len(
            gold_tokens
        )
    )

    return (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
        )
    )


def build_calibration_correctness_record(
    prediction: dict[str, Any],
    predicted_answer: str,
    gold_answers: list[str],
    is_answerable: bool,
) -> dict[str, Any]:
    """
    Build canonical fields expected by calibration correctness.

    Converting answerability to a real bool prevents truthiness errors in
    historical implementations of the calibration helper.
    """

    canonical_prediction = dict(
        prediction
    )

    canonical_prediction[
        "prediction_text"
    ] = predicted_answer

    canonical_prediction[
        "reference_answers"
    ] = gold_answers

    canonical_prediction[
        "is_answerable"
    ] = is_answerable

    return canonical_prediction


def calculate_answer_metrics(
    prediction: dict[str, Any],
) -> dict[str, Any]:
    """
    Calculate forced-answer candidate quality for one record.

    Strict correctness is aligned with calibration threshold selection.

    Relaxed correctness adds token-F1 credit only for answerable examples.

    Unanswerable forced-answer candidates are always incorrect, regardless of
    prediction text or later routing decisions.
    """

    predicted_answer = (
        get_predicted_answer(
            prediction
        )
    )

    gold_answers = (
        get_gold_answers(
            prediction
        )
    )

    is_answerable = (
        get_is_answerable(
            prediction
        )
    )

    usable_gold_answers = [
        answer
        for answer in gold_answers
        if normalize_answer(
            answer
        )
    ]

    if is_answerable:
        if not usable_gold_answers:
            raise ValueError(
                "Answerable prediction does not "
                "contain a usable reference. "
                f"Prediction id: "
                f"{prediction.get('id')!r}"
            )

        exact_match = max(
            calculate_exact_match(
                predicted_answer,
                gold_answer,
            )
            for gold_answer
            in usable_gold_answers
        )

        token_f1 = max(
            calculate_token_f1(
                predicted_answer,
                gold_answer,
            )
            for gold_answer
            in usable_gold_answers
        )

        canonical_prediction = (
            build_calibration_correctness_record(
                prediction=(
                    prediction
                ),
                predicted_answer=(
                    predicted_answer
                ),
                gold_answers=(
                    usable_gold_answers
                ),
                is_answerable=True,
            )
        )

        strict_correct = bool(
            calibration_is_prediction_correct(
                canonical_prediction
            )
        )

        if (
            strict_correct
            != (
                exact_match
                == 1.0
            )
        ):
            raise RuntimeError(
                "Strict correctness disagrees "
                "with Exact Match."
            )

        relaxed_correct = (
            strict_correct
            or token_f1
            >= DEFAULT_RELAXED_F1_THRESHOLD
        )

        if strict_correct:
            error_type = (
                "CORRECT_EXACT"
            )

        elif (
            token_f1
            >= DEFAULT_RELAXED_F1_THRESHOLD
        ):
            error_type = (
                "MINOR_SPAN_MISMATCH"
            )

        elif (
            token_f1
            >= 0.50
        ):
            error_type = (
                "PARTIAL_ANSWER"
            )

        else:
            error_type = (
                "WRONG_ANSWER"
            )

        return {
            "exact_match": (
                exact_match
            ),
            "token_f1": (
                token_f1
            ),
            "strict_correct": (
                strict_correct
            ),
            "relaxed_correct": (
                relaxed_correct
            ),
            "error_type": (
                error_type
            ),
        }

    if usable_gold_answers:
        raise ValueError(
            "Unanswerable prediction contains "
            "non-empty references. "
            f"Prediction id: "
            f"{prediction.get('id')!r}"
        )

    canonical_prediction = (
        build_calibration_correctness_record(
            prediction=(
                prediction
            ),
            predicted_answer=(
                predicted_answer
            ),
            gold_answers=[],
            is_answerable=False,
        )
    )

    strict_correct = bool(
        calibration_is_prediction_correct(
            canonical_prediction
        )
    )

    if strict_correct:
        raise RuntimeError(
            "Calibration correctness marked an "
            "unanswerable forced-answer candidate "
            "as correct."
        )

    return {
        "exact_match": 0.0,
        "token_f1": 0.0,
        "strict_correct": False,
        "relaxed_correct": False,
        "error_type": (
            "UNANSWERABLE_FORCED_ANSWER_CANDIDATE"
        ),
    }


def get_evidence_support(
    prediction: dict[str, Any],
) -> str:
    """Retrieve and validate the lexical evidence-support label."""

    value = get_first_value(
        prediction,
        EVIDENCE_LABEL_FIELDS,
        default=None,
    )

    if value is None:
        raise ValueError(
            "Prediction does not contain a "
            "lexical evidence-support label."
        )

    support = (
        str(value)
        .strip()
        .upper()
    )

    if (
        support
        not in VALID_EVIDENCE_LABELS
    ):
        raise ValueError(
            "Invalid evidence-support value: "
            f"{support!r}."
        )

    return support


def classify_evidence_case(
    relaxed_correct: bool,
    evidence_support: str,
) -> str:
    """
    Combine candidate correctness with lexical evidence support.

    The historical parameter name is preserved although strict correctness can
    also be supplied when constructing `strict_category`.
    """

    support = (
        str(
            evidence_support
        )
        .strip()
        .upper()
    )

    if (
        support
        not in VALID_EVIDENCE_LABELS
    ):
        raise ValueError(
            "Invalid evidence-support value: "
            f"{support!r}."
        )

    prefix = (
        "CORRECT"
        if relaxed_correct
        else "WRONG"
    )

    return (
        f"{prefix}_{support}"
    )


def get_consistent_decision(
    prediction: dict[str, Any],
    field_names: Sequence[str],
    decision_name: str,
) -> str | None:
    """
    Return one consistent routing decision from historical aliases.

    Routing decisions are diagnostics only; they never determine candidate
    correctness.
    """

    observed: dict[
        str,
        str,
    ] = {}

    for field_name in field_names:
        if (
            field_name
            not in prediction
            or prediction[
                field_name
            ]
            is None
        ):
            continue

        decision = (
            str(
                prediction[
                    field_name
                ]
            )
            .strip()
            .upper()
        )

        if (
            decision
            not in VALID_ROUTING_DECISIONS
        ):
            raise ValueError(
                f"Invalid {field_name} routing "
                f"decision: {decision!r}."
            )

        observed[
            field_name
        ] = decision

    if not observed:
        return None

    unique_values = set(
        observed.values()
    )

    if (
        len(
            unique_values
        )
        > 1
    ):
        raise ValueError(
            f"Conflicting {decision_name} "
            f"fields: {observed}."
        )

    return next(
        iter(
            unique_values
        )
    )


def get_normalized_decision(
    prediction: dict[str, Any],
    field_names: tuple[str, ...],
) -> str | None:
    """Compatibility wrapper for historical callers."""

    return get_consistent_decision(
        prediction=(
            prediction
        ),
        field_names=(
            field_names
        ),
        decision_name=(
            "routing decision"
        ),
    )


def get_optional_probability(
    prediction: dict[str, Any],
    field_names: Sequence[str],
    signal_name: str,
) -> float | None:
    """
    Return the first optional probability-like diagnostic in [0, 1].

    Present malformed values raise instead of silently disappearing.
    """

    for field_name in field_names:
        if (
            field_name
            not in prediction
            or prediction[
                field_name
            ]
            is None
        ):
            continue

        value = prediction[
            field_name
        ]

        if isinstance(
            value,
            bool,
        ):
            raise ValueError(  # noqa: TRY004
                f"{signal_name} "
                "cannot be Boolean."
            )

        try:
            numeric_value = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                f"Invalid {signal_name} "
                f"in {field_name!r}: "
                f"{value!r}."
            ) from error

        if (
            not math.isfinite(
                numeric_value
            )
            or not (
                0.0
                <= numeric_value
                <= 1.0
            )
        ):
            raise ValueError(
                f"{signal_name} must be "
                "a finite value in [0, 1], "
                f"received {numeric_value}."
            )

        return numeric_value

    return None


def get_optional_finite_numeric(
    prediction: dict[str, Any],
    field_name: str,
) -> float | None:
    """Return one optional finite numeric diagnostic."""

    if (
        field_name
        not in prediction
        or prediction[
            field_name
        ]
        is None
    ):
        return None

    value = prediction[
        field_name
    ]

    if isinstance(
        value,
        bool,
    ):
        raise ValueError(  # noqa: TRY004
            f"{field_name} "
            "cannot be Boolean."
        )

    try:
        numeric_value = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            f"Invalid {field_name}: "
            f"{value!r}."
        ) from error

    if not math.isfinite(
        numeric_value
    ):
        raise ValueError(
            f"{field_name} "
            "must be finite."
        )

    return numeric_value


def get_evidence_score(
    prediction: dict[str, Any],
) -> float | None:
    """Retrieve the lexical evidence score across historical aliases."""

    return get_optional_probability(
        prediction=(
            prediction
        ),
        field_names=(
            EVIDENCE_SCORE_FIELDS
        ),
        signal_name=(
            "evidence score"
        ),
    )


def get_display_confidence(
    prediction: dict[str, Any],
) -> float | None:
    """
    Retrieve calibrated confidence for diagnostics when explicitly available.

    Generic `confidence` is accepted only when explicitly marked calibrated.
    """

    value = (
        get_optional_probability(
            prediction=(
                prediction
            ),
            field_names=(
                CALIBRATED_CONFIDENCE_FIELDS
            ),
            signal_name=(
                "calibrated confidence"
            ),
        )
    )

    if value is not None:
        return value

    if (
        prediction.get(
            "confidence_is_calibrated"
        )
        is True
    ):
        return get_optional_probability(
            prediction=(
                prediction
            ),
            field_names=(
                "confidence",
            ),
            signal_name=(
                "calibrated confidence"
            ),
        )

    return None


def build_analysis_case(
    prediction: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    """
    Build one lexical-verifier diagnostic record.

    `category` preserves the relaxed-correctness view.
    `strict_category` uses forced-answer Exact Match correctness.
    """

    metrics = (
        calculate_answer_metrics(
            prediction
        )
    )

    evidence_support = (
        get_evidence_support(
            prediction
        )
    )

    relaxed_category = (
        classify_evidence_case(
            metrics[
                "relaxed_correct"
            ],
            evidence_support,
        )
    )

    strict_category = (
        classify_evidence_case(
            metrics[
                "strict_correct"
            ],
            evidence_support,
        )
    )

    threshold_decision = (
        get_consistent_decision(
            prediction=(
                prediction
            ),
            field_names=(
                "threshold_decision",
                "confidence_decision",
                "selective_decision",
                "decision",
            ),
            decision_name=(
                "threshold decision"
            ),
        )
    )

    final_decision = (
        get_consistent_decision(
            prediction=(
                prediction
            ),
            field_names=(
                "final_decision",
            ),
            decision_name=(
                "final decision"
            ),
        )
    )

    return {
        "index": (
            index
        ),
        "id": prediction.get(
            "id"
        ),
        "category": (
            relaxed_category
        ),
        "strict_category": (
            strict_category
        ),
        "error_type": (
            metrics[
                "error_type"
            ]
        ),
        "is_answerable": (
            get_is_answerable(
                prediction
            )
        ),
        "strict_correct": (
            metrics[
                "strict_correct"
            ]
        ),
        "relaxed_correct": (
            metrics[
                "relaxed_correct"
            ]
        ),
        "exact_match": (
            metrics[
                "exact_match"
            ]
        ),
        "token_f1": (
            metrics[
                "token_f1"
            ]
        ),
        "question": prediction.get(
            "question",
            "",
        ),
        "predicted_answer": (
            get_predicted_answer(
                prediction
            )
        ),
        "gold_answers": (
            get_gold_answers(
                prediction
            )
        ),
        "context": prediction.get(
            "context",
            "",
        ),
        "evidence_text": prediction.get(
            "evidence_text",
            "",
        ),
        "answer_context_score": (
            get_optional_finite_numeric(
                prediction,
                "answer_context_score",
            )
        ),
        "question_evidence_overlap": (
            get_optional_finite_numeric(
                prediction,
                "question_evidence_overlap",
            )
        ),
        "evidence_score": (
            get_evidence_score(
                prediction
            )
        ),
        "evidence_support": (
            evidence_support
        ),
        "confidence": (
            get_display_confidence(
                prediction
            )
        ),
        "threshold_decision": (
            threshold_decision
        ),
        "final_decision": (
            final_decision
        ),
    }


def validate_predictions(
    predictions: list[dict[str, Any]],
) -> None:
    """Validate records before lexical-verifier error analysis."""

    if not predictions:
        raise ValueError(
            "Prediction list cannot be empty."
        )

    for index, prediction in enumerate(
        predictions,
        start=1,
    ):
        try:
            if not has_prediction_field(
                prediction
            ):
                raise ValueError(
                    "No supported prediction "
                    "field was found."
                )

            if (
                "question"
                not in prediction
            ):
                raise ValueError(
                    "Missing question field."
                )

            if (
                "context"
                not in prediction
            ):
                raise ValueError(
                    "Missing context field."
                )

            is_answerable = (
                get_is_answerable(
                    prediction
                )
            )

            gold_answers = (
                get_gold_answers(
                    prediction
                )
            )

            usable_gold_answers = [
                answer
                for answer
                in gold_answers
                if normalize_answer(
                    answer
                )
            ]

            if (
                is_answerable
                and not usable_gold_answers
            ):
                raise ValueError(
                    "Answerable record has no "
                    "usable reference answer."
                )

            if (
                not is_answerable
                and usable_gold_answers
            ):
                raise ValueError(
                    "Unanswerable record contains "
                    "non-empty references."
                )

            get_evidence_support(
                prediction
            )

            get_evidence_score(
                prediction
            )

            get_display_confidence(
                prediction
            )

            get_optional_finite_numeric(
                prediction,
                "answer_context_score",
            )

            get_optional_finite_numeric(
                prediction,
                "question_evidence_overlap",
            )

            calculate_answer_metrics(
                prediction
            )

            get_consistent_decision(
                prediction=(
                    prediction
                ),
                field_names=(
                    "threshold_decision",
                    "confidence_decision",
                    "selective_decision",
                    "decision",
                ),
                decision_name=(
                    "threshold decision"
                ),
            )

            get_consistent_decision(
                prediction=(
                    prediction
                ),
                field_names=(
                    "final_decision",
                ),
                decision_name=(
                    "final decision"
                ),
            )

        except (
            ValueError,
            TypeError,
            KeyError,
            RuntimeError,
        ) as error:
            raise ValueError(
                f"Prediction {index} "
                f"failed validation: "
                f"{error}"
            ) from error


def print_case(
    case: dict[str, Any],
) -> None:
    """Print one diagnostic case for manual inspection."""

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"Index: "
        f"{case['index']}"
    )

    print(
        f"ID: "
        f"{case['id']}"
    )

    print(
        "Relaxed category: "
        f"{case['category']}"
    )

    print(
        "Strict category: "
        f"{case['strict_category']}"
    )

    print(
        f"Error type: "
        f"{case['error_type']}"
    )

    print(
        f"Answerable: "
        f"{case['is_answerable']}"
    )

    print(
        "Exact Match: "
        f"{case['exact_match']:.4f}"
    )

    print(
        "Token F1: "
        f"{case['token_f1']:.4f}"
    )

    print(
        f"Strict correct: "
        f"{case['strict_correct']}"
    )

    print(
        f"Relaxed correct: "
        f"{case['relaxed_correct']}"
    )

    print(
        f"\nQuestion:\n"
        f"{case['question']}"
    )

    print(
        f"\nPrediction:\n"
        f"{case['predicted_answer']}"
    )

    print(
        f"\nGold answers:\n"
        f"{case['gold_answers']}"
    )

    print(
        f"\nEvidence text:\n"
        f"{case['evidence_text']}"
    )

    print(
        "\nVerifier:"
    )

    print(
        "  answer_context_score: "
        f"{case['answer_context_score']}"
    )

    print(
        "  question_evidence_overlap: "
        f"{case['question_evidence_overlap']}"
    )

    print(
        "  evidence_score: "
        f"{case['evidence_score']}"
    )

    print(
        "  evidence_support: "
        f"{case['evidence_support']}"
    )

    print(
        f"  confidence: "
        f"{case['confidence']}"
    )

    print(
        "  threshold_decision: "
        f"{case['threshold_decision']}"
    )

    print(
        f"  final_decision: "
        f"{case['final_decision']}"
    )


def analyze_evidence_errors(
    input_path: str | Path,
    output_path: str | Path,
    max_examples: int,
) -> list[dict[str, Any]]:
    """Run lexical evidence-verifier error analysis and save per-case output."""

    if (
        not isinstance(
            max_examples,
            int,
        )
        or isinstance(
            max_examples,
            bool,
        )
        or max_examples < 0
    ):
        raise ValueError(
            "max_examples must be a "
            "non-negative integer."
        )

    predictions = (
        load_jsonl(
            input_path
        )
    )

    validate_predictions(
        predictions
    )

    analysis_cases = [
        build_analysis_case(
            prediction,
            index,
        )
        for index, prediction
        in enumerate(
            predictions,
            start=1,
        )
    ]

    save_jsonl(
        analysis_cases,
        output_path,
    )

    category_counts = Counter(
        case[
            "category"
        ]
        for case
        in analysis_cases
    )

    strict_category_counts = Counter(
        case[
            "strict_category"
        ]
        for case
        in analysis_cases
    )

    error_type_counts = Counter(
        case[
            "error_type"
        ]
        for case
        in analysis_cases
    )

    strict_correct_count = sum(
        int(
            case[
                "strict_correct"
            ]
        )
        for case
        in analysis_cases
    )

    relaxed_correct_count = sum(
        int(
            case[
                "relaxed_correct"
            ]
        )
        for case
        in analysis_cases
    )

    average_f1 = (
        sum(
            float(
                case[
                    "token_f1"
                ]
            )
            for case
            in analysis_cases
        )
        / len(
            analysis_cases
        )
    )

    print(
        "\nEvidence error analysis completed."
    )

    print(
        f"Total predictions: "
        f"{len(analysis_cases)}"
    )

    print(
        "\nForced-answer QA diagnostics:"
    )

    print(
        "Strict Exact Match accuracy: "
        f"{strict_correct_count / len(analysis_cases):.4f}"
    )

    print(
        "Relaxed answerable accuracy "
        f"(EM or F1 >= "
        f"{DEFAULT_RELAXED_F1_THRESHOLD:.2f}; "
        "unanswerable candidates remain incorrect): "
        f"{relaxed_correct_count / len(analysis_cases):.4f}"
    )

    print(
        f"Average Token F1: "
        f"{average_f1:.4f}"
    )

    category_order = (
        "WRONG_SUPPORTED",
        "WRONG_WEAK",
        "WRONG_UNSUPPORTED",
        "CORRECT_SUPPORTED",
        "CORRECT_WEAK",
        "CORRECT_UNSUPPORTED",
    )

    print(
        "\nRelaxed evidence category summary:"
    )

    for category in category_order:
        print(
            f"{category}: "
            f"{category_counts.get(category, 0)}"
        )

    print(
        "\nStrict evidence category summary:"
    )

    for category in category_order:
        print(
            f"{category}: "
            f"{strict_category_counts.get(category, 0)}"
        )

    print(
        "\nAnswer error types:"
    )

    for (
        error_type,
        count,
    ) in error_type_counts.most_common():
        print(
            f"{error_type}: "
            f"{count}"
        )

    critical_cases = [
        case
        for case
        in analysis_cases
        if (
            case[
                "category"
            ]
            == "WRONG_SUPPORTED"
            and case[
                "threshold_decision"
            ]
            == "ANSWER"
        )
    ]

    print(
        "\nCritical cases: "
        "relaxed WRONG_SUPPORTED "
        "+ threshold ANSWER"
    )

    print(
        f"Count: "
        f"{len(critical_cases)}"
    )

    for case in critical_cases[
        :max_examples
    ]:
        print_case(
            case
        )

    print(
        f"\nResults saved to: "
        f"{output_path}"
    )

    return analysis_cases


def parse_arguments() -> argparse.Namespace:
    """Parse evidence-error analysis settings."""

    parser = argparse.ArgumentParser(
        description=(
            "Analyze forced-answer QA candidate "
            "correctness against lexical "
            "evidence-support labels."
        )
    )

    parser.add_argument(
        "--input",
        default=str(
            DEFAULT_INPUT_PATH
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT_PATH
        ),
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    analyze_evidence_errors(
        input_path=args.input,
        output_path=args.output,
        max_examples=args.max_examples,
    )