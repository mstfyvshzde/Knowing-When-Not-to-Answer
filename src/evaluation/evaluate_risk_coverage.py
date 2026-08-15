"""
Evaluate confidence-only and hybrid prototype ranking for selective QA.

This module compares two earlier project signals:

1. calibrated confidence
2. hybrid evidence score

It provides two complementary analyses:

- threshold sweeps, which show the operational behavior of fixed score cutoffs;
- exact score ranking, which produces a full risk-coverage curve and AURC.

The underlying QA correctness definition used by this prototype is relaxed for
answerable questions:

    Exact Match == 1
        OR
    token F1 >= relaxed_f1_threshold

The default relaxed F1 threshold is 0.80.

Important
---------
This file belongs to the earlier confidence-vs-hybrid prototype analysis. Its
relaxed correctness criterion differs from the Exact-Match correctness used by
the project's final question-aware AURC experiments.

For unanswerable examples, the underlying forced-answer QA candidate is always
treated as incorrect. Later ANSWER/ABSTAIN routing decisions do not change
candidate correctness.

AURC uses the same discrete definition as the final ranking evaluator:

    arithmetic mean of risk over every non-empty ranked prefix

Lower AURC is better.

Threshold-grid coverage comparisons are approximate because two score
distributions can reach slightly different actual coverages at their nearest
thresholds. The output records the resulting coverage gap explicitly.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.evaluation.metrics import (
    normalize_answer as shared_normalize_answer,
)
from src.evaluation.metrics import (
    parse_answerability,
)
from src.utils.io import (
    load_jsonl,
)
from src.utils.io import (
    save_json as shared_save_json,
)

DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/calibration_with_hybrid_evidence.jsonl"
)

DEFAULT_OUTPUT_DIR = Path(
    "outputs/evaluation/risk_coverage"
)

DEFAULT_RELAXED_F1_THRESHOLD = 0.80

DEFAULT_THRESHOLD_START = 0.00
DEFAULT_THRESHOLD_END = 1.00
DEFAULT_THRESHOLD_STEP = 0.01

DEFAULT_TARGET_COVERAGES = (
    0.10,
    0.20,
    0.30,
    0.32,
    0.40,
    0.50,
    0.54,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
)

DEFAULT_DISPLAY_THRESHOLDS = (
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
)


CALIBRATED_CONFIDENCE_FIELDS = (
    "calibrated_confidence",
    "confidence_calibrated",
    "calibrated_probability",
)

GENERIC_CONFIDENCE_FIELD = "confidence"
CONFIDENCE_CALIBRATED_FLAG = "confidence_is_calibrated"

HYBRID_SCORE_FIELDS = (
    "hybrid_evidence_score",
    "hybrid_score",
)

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
    field_names: tuple[str, ...],
    default: Any = None,
) -> Any:
    """Return the first available non-None value from ordered field aliases."""

    for field_name in field_names:
        value = prediction.get(field_name)

        if value is not None:
            return value

    return default


def get_first_numeric_value(
    prediction: dict[str, Any],
    field_names: tuple[str, ...],
    default: float | None = None,
) -> float:
    """
    Return the first usable finite numeric value from ordered aliases.

    Present-but-malformed values raise an error instead of silently becoming
    zero, because silent substitution could alter ranking or threshold results.
    """

    for field_name in field_names:
        if field_name not in prediction:
            continue

        value = prediction[field_name]

        if value is None:
            continue

        if isinstance(value, bool):
            raise ValueError(  # noqa: TRY004
                f"Boolean value is not a valid numeric score in "
                f"{field_name!r}: {value!r}."
            )

        try:
            numeric_value = float(value)

        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid numeric value in "
                f"{field_name!r}: {value!r}."
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


def clamp_probability(
    value: float,
) -> float:
    """
    Clamp a derived value to [0, 1].

    This helper is retained for compatibility. Experimental input scores are
    validated rather than silently clamped.
    """

    return max(
        0.0,
        min(
            1.0,
            float(value),
        ),
    )


def get_predicted_answer(
    prediction: dict[str, Any],
) -> str:
    """Extract the predicted answer from supported historical field aliases."""

    value = get_first_value(
        prediction=prediction,
        field_names=PREDICTION_FIELDS,
        default="",
    )

    return str(value).strip()


def extract_reference_value(
    value: Any,
) -> list[str]:
    """Recursively extract reference-answer strings from common structures."""

    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, dict):
        for field_name in (
            "text",
            "answer",
            "answers",
        ):
            if field_name in value:
                return extract_reference_value(
                    value[field_name]
                )

        return []

    if isinstance(
        value,
        (list, tuple, set),
    ):
        references: list[str] = []

        for item in value:
            references.extend(
                extract_reference_value(
                    item
                )
            )

        return references

    return [
        str(value)
    ]


def get_reference_answers(
    prediction: dict[str, Any],
) -> list[str]:
    """Extract all reference answers from the first supported reference field."""

    value = get_first_value(
        prediction=prediction,
        field_names=REFERENCE_FIELDS,
        default=None,
    )

    if value is None:
        return []

    return [
        str(reference).strip()
        for reference
        in extract_reference_value(
            value
        )
    ]


def get_is_answerable(
    prediction: dict[str, Any],
) -> bool:
    """
    Retrieve answerability from explicit metadata when available.

    If explicit answerability is absent, infer it from whether at least one
    non-empty normalized reference answer exists.
    """

    for field_name in ANSWERABILITY_FIELDS:
        if field_name not in prediction:
            continue

        value = prediction[field_name]

        if value is None:
            continue

        try:
            return parse_answerability(
                value
            )

        except (
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                f"Invalid answerability value in "
                f"{field_name!r}: {value!r}."
            ) from error

    reference_answers = (
        get_reference_answers(
            prediction
        )
    )

    return any(
        normalize_answer(
            answer
        )
        for answer
        in reference_answers
    )


def normalize_answer(
    text: str,
) -> str:
    """Normalize answer text using the shared project comparison rule."""

    return shared_normalize_answer(
        str(text)
    )


def exact_match_score(
    predicted_answer: str,
    reference_answer: str,
) -> float:
    """Return normalized Exact Match for one prediction/reference pair."""

    return float(
        normalize_answer(
            predicted_answer
        )
        == normalize_answer(
            reference_answer
        )
    )


def token_f1_score(
    predicted_answer: str,
    reference_answer: str,
) -> float:
    """Calculate token-level F1 for one prediction/reference pair."""

    predicted_tokens = (
        normalize_answer(
            predicted_answer
        ).split()
    )

    reference_tokens = (
        normalize_answer(
            reference_answer
        ).split()
    )

    if (
        not predicted_tokens
        and not reference_tokens
    ):
        return 1.0

    if (
        not predicted_tokens
        or not reference_tokens
    ):
        return 0.0

    common_tokens = (
        Counter(predicted_tokens)
        & Counter(reference_tokens)
    )

    overlap_count = sum(
        common_tokens.values()
    )

    if overlap_count == 0:
        return 0.0

    precision = (
        overlap_count
        / len(predicted_tokens)
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


def get_clean_raw_answer(
    value: Any,
) -> str:
    """
    Normalize whitespace only, without removing punctuation.

    This is used when deciding whether a raw unanswerable prediction is
    genuinely empty.
    """

    if value is None:
        return ""

    return " ".join(
        str(value).split()
    )


def calculate_answer_scores(
    predicted_answer: str,
    reference_answers: list[str],
) -> tuple[float, float]:
    """
    Calculate best Exact Match and token F1 across available references.

    For records without references, the historical prototype diagnostic assigns
    1.0 to an actually empty prediction and 0.0 otherwise.
    """

    if not reference_answers:
        return (
            0.0,
            0.0,
        )

    exact_match = max(
        exact_match_score(
            predicted_answer,
            reference_answer,
        )
        for reference_answer
        in reference_answers
    )

    token_f1 = max(
        token_f1_score(
            predicted_answer,
            reference_answer,
        )
        for reference_answer
        in reference_answers
    )

    return (
        exact_match,
        token_f1,
    )


def is_prediction_correct(
    prediction: dict[str, Any],
    relaxed_f1_threshold: float,
) -> tuple[
    bool,
    float,
    float,
]:
    """
    Determine underlying QA correctness for the prototype analysis.

    Answerable questions are correct when Exact Match is 1.0 or token F1 meets
    the relaxed threshold.

    For unanswerable examples, the underlying forced-answer QA candidate is always
    incorrect, regardless of later routing decisions.
    """

    if not (
        0.0
        <= relaxed_f1_threshold
        <= 1.0
    ):
        raise ValueError(
            "relaxed_f1_threshold must be "
            "between 0 and 1."
        )

    predicted_answer = (
        get_predicted_answer(
            prediction
        )
    )

    reference_answers = (
        get_reference_answers(
            prediction
        )
    )

    is_answerable = (
        get_is_answerable(
            prediction
        )
    )

    usable_references = [
        reference
        for reference
        in reference_answers
        if normalize_answer(
            reference
        )
    ]

    if (
        is_answerable
        and not usable_references
    ):
        raise ValueError(
            "Answerable prediction does not contain "
            "a usable reference answer. "
            f"Prediction id: "
            f"{prediction.get('id')!r}"
        )

    if (
        not is_answerable
        and usable_references
    ):
        raise ValueError(
            "Unanswerable prediction contains "
            "non-empty reference answers. "
            f"Prediction id: "
            f"{prediction.get('id')!r}"
        )

    (
        exact_match,
        token_f1,
    ) = calculate_answer_scores(
        predicted_answer=(
            predicted_answer
        ),
        reference_answers=(
            reference_answers
        ),
    )

    if not is_answerable:
        return (
            False,
            0.0,
            0.0,
        )

    is_correct = (
        exact_match == 1.0
        or token_f1
        >= relaxed_f1_threshold
    )

    return (
        is_correct,
        exact_match,
        token_f1,
    )


def validate_probability_score(
    score: float,
    score_name: str,
) -> float:
    """Require a finite probability-like score in [0, 1]."""

    if not math.isfinite(
        score
    ):
        raise ValueError(
            f"{score_name} must be finite."
        )

    if not (
        0.0
        <= score
        <= 1.0
    ):
        raise ValueError(
            f"{score_name} must be "
            "between 0 and 1, "
            f"received {score}."
        )

    return score


def get_calibrated_confidence(
    prediction: dict[str, Any],
) -> float:
    """
    Retrieve the calibrated confidence used by the prototype comparison.

    Explicit calibrated-confidence fields are preferred. The generic
    `confidence` field is accepted only when the record explicitly marks it as
    calibrated. Raw confidence is intentionally not used as a fallback.
    """

    for field_name in (
        CALIBRATED_CONFIDENCE_FIELDS
    ):
        if (
            field_name
            in prediction
            and prediction[
                field_name
            ]
            is not None
        ):
            score = (
                get_first_numeric_value(
                    prediction,
                    (field_name,),
                )
            )

            return (
                validate_probability_score(
                    score,
                    "Calibrated confidence",
                )
            )

    if (
        prediction.get(
            CONFIDENCE_CALIBRATED_FLAG
        )
        is True
        and prediction.get(
            GENERIC_CONFIDENCE_FIELD
        )
        is not None
    ):
        score = (
            get_first_numeric_value(
                prediction,
                (
                    GENERIC_CONFIDENCE_FIELD,
                ),
            )
        )

        return (
            validate_probability_score(
                score,
                "Calibrated confidence",
            )
        )

    raise ValueError(
        "Prediction does not contain usable "
        "calibrated confidence."
    )


def get_hybrid_score(
    prediction: dict[str, Any],
) -> float:
    """Retrieve the hybrid evidence score as a validated [0, 1] value."""

    score = (
        get_first_numeric_value(
            prediction=prediction,
            field_names=(
                HYBRID_SCORE_FIELDS
            ),
        )
    )

    return (
        validate_probability_score(
            score,
            "Hybrid evidence score",
        )
    )


def validate_predictions(
    predictions: list[
        dict[
            str,
            Any,
        ]
    ],
) -> None:
    """
    Validate score, answerability, and reference information for every record.

    Missing or malformed scores fail early rather than silently becoming zero.
    """

    if not predictions:
        raise ValueError(
            "Prediction list cannot be empty."
        )

    for index, prediction in enumerate(
        predictions,
        start=1,
    ):
        try:
            get_calibrated_confidence(
                prediction
            )

            get_hybrid_score(
                prediction
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
                for reference
                in references
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
                    "non-empty reference answers."
                )

        except (
            ValueError,
            TypeError,
        ) as error:
            raise ValueError(
                f"Prediction {index} "
                f"failed validation: {error}"
            ) from error


def generate_thresholds(
    start: float,
    end: float,
    step: float,
) -> list[float]:
    """Generate an inclusive deterministic threshold grid."""

    for name, value in (
        (
            "threshold start",
            start,
        ),
        (
            "threshold end",
            end,
        ),
        (
            "threshold step",
            step,
        ),
    ):
        if not math.isfinite(
            float(value)
        ):
            raise ValueError(
                f"{name} must be finite."
            )

    if step <= 0.0:
        raise ValueError(
            "Threshold step must be positive."
        )

    if start > end:
        raise ValueError(
            "Threshold start must not exceed end."
        )

    if (
        start < 0.0
        or end > 1.0
    ):
        raise ValueError(
            "Threshold range must be "
            "inside [0, 1]."
        )

    thresholds: list[
        float
    ] = []

    current_value = start

    while (
        current_value
        <= end + 1e-12
    ):
        thresholds.append(
            round(
                current_value,
                10,
            )
        )

        current_value += step

    return thresholds


def evaluate_at_threshold(
    predictions: list[
        dict[
            str,
            Any,
        ]
    ],
    correctness: list[bool],
    score_function: Callable[
        [dict[str, Any]],
        float,
    ],
    threshold: float,
) -> dict[str, Any]:
    """
    Evaluate a score-based ANSWER/ABSTAIN policy at one threshold.

    When zero examples are answered, selective accuracy/risk use the historical
    plotting convention 1.0/0.0. Such zero-coverage points are not used in AURC.
    """

    if not predictions:
        raise ValueError(
            "Prediction list cannot be empty."
        )

    if (
        len(predictions)
        != len(correctness)
    ):
        raise ValueError(
            "Prediction and correctness "
            "counts must match."
        )

    if (
        not math.isfinite(
            threshold
        )
        or not (
            0.0
            <= threshold
            <= 1.0
        )
    ):
        raise ValueError(
            "Threshold must be finite "
            "and between 0 and 1."
        )

    answered_indices = [
        index
        for index, prediction
        in enumerate(
            predictions
        )
        if score_function(
            prediction
        )
        >= threshold
    ]

    answered_count = len(
        answered_indices
    )

    total_count = len(
        predictions
    )

    abstained_count = (
        total_count
        - answered_count
    )

    correct_answered = sum(
        1
        for index
        in answered_indices
        if correctness[
            index
        ]
    )

    wrong_answered = (
        answered_count
        - correct_answered
    )

    coverage = (
        answered_count
        / total_count
    )

    if answered_count:
        selective_accuracy = (
            correct_answered
            / answered_count
        )

        selective_risk = (
            wrong_answered
            / answered_count
        )

    else:
        selective_accuracy = 1.0
        selective_risk = 0.0

    return {
        "threshold": threshold,
        "total": total_count,
        "answered": (
            answered_count
        ),
        "abstained": (
            abstained_count
        ),
        "coverage": coverage,
        "selective_accuracy": (
            selective_accuracy
        ),
        "selective_risk": (
            selective_risk
        ),
        "correct_answered": (
            correct_answered
        ),
        "wrong_answered": (
            wrong_answered
        ),
    }


def sweep_thresholds(
    predictions: list[
        dict[
            str,
            Any,
        ]
    ],
    correctness: list[bool],
    score_function: Callable[
        [dict[str, Any]],
        float,
    ],
    thresholds: list[float],
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """Evaluate one score function across a deterministic threshold grid."""

    if not thresholds:
        raise ValueError(
            "Threshold list cannot be empty."
        )

    return [
        evaluate_at_threshold(
            predictions=predictions,
            correctness=correctness,
            score_function=(
                score_function
            ),
            threshold=threshold,
        )
        for threshold
        in thresholds
    ]


def build_exact_risk_coverage_curve(
    predictions: list[
        dict[
            str,
            Any,
        ]
    ],
    correctness: list[bool],
    score_function: Callable[
        [dict[str, Any]],
        float,
    ],
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """
    Build the exact prefix-based risk-coverage curve.

    Records are ranked by score descending. Score ties are broken by original
    input index, matching the deterministic tie rule used by the final ablation
    evaluator.

    A zero-coverage origin is included for plotting, but excluded from AURC.
    """

    if not predictions:
        raise ValueError(
            "Prediction list cannot be empty."
        )

    if (
        len(predictions)
        != len(correctness)
    ):
        raise ValueError(
            "Prediction and correctness "
            "counts must match."
        )

    scored_examples: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for index, (
        prediction,
        is_correct,
    ) in enumerate(
        zip(
            predictions,
            correctness,
        )
    ):
        score = (
            score_function(
                prediction
            )
        )

        validate_probability_score(
            score,
            "Ranking score",
        )

        scored_examples.append(
            {
                "index": index,
                "score": score,
                "correct": bool(
                    is_correct
                ),
            }
        )

    scored_examples.sort(
        key=lambda item: (
            -float(
                item[
                    "score"
                ]
            ),
            int(
                item[
                    "index"
                ]
            ),
        )
    )

    total_count = len(
        scored_examples
    )

    curve: list[
        dict[
            str,
            Any,
        ]
    ] = [
        {
            "answered": 0,
            "coverage": 0.0,
            "selective_accuracy": 1.0,
            "selective_risk": 0.0,
            "minimum_score": None,
        }
    ]

    correct_answered = 0

    for rank, example in enumerate(
        scored_examples,
        start=1,
    ):
        correct_answered += int(
            example[
                "correct"
            ]
        )

        wrong_answered = (
            rank
            - correct_answered
        )

        curve.append(
            {
                "answered": rank,
                "coverage": (
                    rank
                    / total_count
                ),
                "selective_accuracy": (
                    correct_answered
                    / rank
                ),
                "selective_risk": (
                    wrong_answered
                    / rank
                ),
                "minimum_score": (
                    example[
                        "score"
                    ]
                ),
            }
        )

    return curve


def calculate_aurc(
    curve: list[
        dict[
            str,
            Any,
        ]
    ],
) -> float:
    """
    Calculate discrete AURC as mean risk over all non-empty ranked prefixes.

    This matches the canonical AURC definition used by
    `evaluate_question_aware_ablation.py`.
    """

    non_empty_points = [
        point
        for point
        in curve
        if int(
            point.get(
                "answered",
                0,
            )
        )
        > 0
    ]

    if not non_empty_points:
        raise ValueError(
            "Cannot calculate AURC without "
            "non-empty ranked prefixes."
        )

    return (
        sum(
            float(
                point[
                    "selective_risk"
                ]
            )
            for point
            in non_empty_points
        )
        / len(
            non_empty_points
        )
    )


def find_closest_coverage_result(
    results: list[
        dict[
            str,
            Any,
        ]
    ],
    target_coverage: float,
) -> dict[str, Any]:
    """
    Select the threshold result closest to a requested positive coverage.

    Zero-answer rows are ignored when a positive-coverage result exists.
    Equal-distance ties prefer higher actual coverage before lower risk.
    """

    if not results:
        raise ValueError(
            "Results cannot be empty."
        )

    if (
        not math.isfinite(
            target_coverage
        )
        or not (
            0.0
            < target_coverage
            <= 1.0
        )
    ):
        raise ValueError(
            "target_coverage must be finite "
            "and lie in (0, 1]."
        )

    positive_results = [
        result
        for result
        in results
        if float(
            result[
                "coverage"
            ]
        )
        > 0.0
    ]

    candidates = (
        positive_results
        if positive_results
        else results
    )

    return min(
        candidates,
        key=lambda result: (
            abs(
                float(
                    result[
                        "coverage"
                    ]
                )
                - target_coverage
            ),
            -float(
                result[
                    "coverage"
                ]
            ),
            float(
                result[
                    "selective_risk"
                ]
            ),
        ),
    )


def compare_at_target_coverages(
    confidence_results: list[
        dict[
            str,
            Any,
        ]
    ],
    hybrid_results: list[
        dict[
            str,
            Any,
        ]
    ],
    target_coverages: list[float],
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """
    Compare threshold sweeps near requested coverage levels.

    This is an approximate threshold-grid comparison, not the exact prefix-based
    matched-coverage analysis used by the final ablation evaluator.

    `coverage_gap` exposes the actual coverage mismatch between the two selected
    threshold rows.
    """

    comparisons: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for target_coverage in (
        target_coverages
    ):
        confidence_result = (
            find_closest_coverage_result(
                results=(
                    confidence_results
                ),
                target_coverage=(
                    target_coverage
                ),
            )
        )

        hybrid_result = (
            find_closest_coverage_result(
                results=(
                    hybrid_results
                ),
                target_coverage=(
                    target_coverage
                ),
            )
        )

        confidence_coverage = float(
            confidence_result[
                "coverage"
            ]
        )

        hybrid_coverage = float(
            hybrid_result[
                "coverage"
            ]
        )

        comparisons.append(
            {
                "target_coverage": (
                    target_coverage
                ),
                "confidence_threshold": (
                    confidence_result[
                        "threshold"
                    ]
                ),
                "confidence_coverage": (
                    confidence_coverage
                ),
                "confidence_accuracy": (
                    confidence_result[
                        "selective_accuracy"
                    ]
                ),
                "confidence_risk": (
                    confidence_result[
                        "selective_risk"
                    ]
                ),
                "confidence_answered": (
                    confidence_result[
                        "answered"
                    ]
                ),
                "hybrid_threshold": (
                    hybrid_result[
                        "threshold"
                    ]
                ),
                "hybrid_coverage": (
                    hybrid_coverage
                ),
                "hybrid_accuracy": (
                    hybrid_result[
                        "selective_accuracy"
                    ]
                ),
                "hybrid_risk": (
                    hybrid_result[
                        "selective_risk"
                    ]
                ),
                "hybrid_answered": (
                    hybrid_result[
                        "answered"
                    ]
                ),
                "coverage_gap": (
                    hybrid_coverage
                    - confidence_coverage
                ),
                "risk_difference": (
                    float(
                        hybrid_result[
                            "selective_risk"
                        ]
                    )
                    - float(
                        confidence_result[
                            "selective_risk"
                        ]
                    )
                ),
            }
        )

    return comparisons


def save_csv(
    rows: list[
        dict[
            str,
            Any,
        ]
    ],
    output_path: str | Path,
) -> None:
    """Save result rows as UTF-8 CSV with deterministic Unix line endings."""

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        output_path.write_text(
            "",
            encoding="utf-8",
        )

        return

    field_names = list(
        rows[
            0
        ].keys()
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=(
                field_names
            ),
            lineterminator="\n",
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def save_json(
    data: Any,
    output_path: str | Path,
) -> None:
    """Save formatted JSON through the repository's shared I/O helper."""

    shared_save_json(
        data,
        output_path,
    )


def print_threshold_summary(
    system_name: str,
    results: list[
        dict[
            str,
            Any,
        ]
    ],
    selected_thresholds: list[float],
) -> None:
    """Print representative rows from a threshold sweep."""

    print(
        "\n"
        + "=" * 78
    )

    print(
        system_name
    )

    print(
        "=" * 78
    )

    print(
        f"{'Threshold':>10}"
        f"{'Coverage':>12}"
        f"{'Accuracy':>12}"
        f"{'Risk':>12}"
        f"{'Answered':>12}"
        f"{'Wrong':>10}"
    )

    print(
        "-" * 78
    )

    for selected_threshold in (
        selected_thresholds
    ):
        closest_result = min(
            results,
            key=lambda result: abs(
                float(
                    result[
                        "threshold"
                    ]
                )
                - selected_threshold
            ),
        )

        print(
            f"{closest_result['threshold']:>10.2f}"
            f"{closest_result['coverage']:>12.4f}"
            f"{closest_result['selective_accuracy']:>12.4f}"
            f"{closest_result['selective_risk']:>12.4f}"
            f"{closest_result['answered']:>12}"
            f"{closest_result['wrong_answered']:>10}"
        )


def print_matched_coverage_table(
    comparisons: list[
        dict[
            str,
            Any,
        ]
    ],
) -> None:
    """
    Print approximate threshold-grid coverage comparisons.

    The coverage-gap column makes mismatched actual coverage explicit.
    """

    print(
        "\n"
        + "=" * 124
    )

    print(
        "APPROXIMATE THRESHOLD-GRID "
        "COVERAGE COMPARISON"
    )

    print(
        "=" * 124
    )

    print(
        f"{'Target':>8}"
        f"{'Conf cov':>10}"
        f"{'Conf risk':>11}"
        f"{'Conf thr':>10}"
        f"{'Hybrid cov':>12}"
        f"{'Hybrid risk':>13}"
        f"{'Hybrid thr':>12}"
        f"{'Cov gap':>10}"
        f"{'Risk Δ':>10}"
    )

    print(
        "-" * 124
    )

    for comparison in comparisons:
        print(
            f"{comparison['target_coverage']:>8.2f}"
            f"{comparison['confidence_coverage']:>10.4f}"
            f"{comparison['confidence_risk']:>11.4f}"
            f"{comparison['confidence_threshold']:>10.2f}"
            f"{comparison['hybrid_coverage']:>12.4f}"
            f"{comparison['hybrid_risk']:>13.4f}"
            f"{comparison['hybrid_threshold']:>12.2f}"
            f"{comparison['coverage_gap']:>10.4f}"
            f"{comparison['risk_difference']:>10.4f}"
        )


def validate_runtime_settings(
    relaxed_f1_threshold: float,
    threshold_start: float,
    threshold_end: float,
    threshold_step: float,
) -> None:
    """Validate prototype evaluation thresholds before processing data."""

    if (
        not math.isfinite(
            relaxed_f1_threshold
        )
        or not (
            0.0
            <= relaxed_f1_threshold
            <= 1.0
        )
    ):
        raise ValueError(
            "relaxed_f1_threshold must be "
            "finite and between 0 and 1."
        )

    generate_thresholds(
        start=(
            threshold_start
        ),
        end=(
            threshold_end
        ),
        step=(
            threshold_step
        ),
    )


def run_risk_coverage_evaluation(
    input_path: str | Path,
    output_dir: str | Path,
    relaxed_f1_threshold: float,
    threshold_start: float,
    threshold_end: float,
    threshold_step: float,
) -> dict[str, Any]:
    """
    Run the complete confidence-vs-hybrid prototype evaluation.

    Threshold sweeps are saved for operational diagnostics. AURC is calculated
    from exact score-ranked prefixes rather than from the coarse threshold grid.
    """

    validate_runtime_settings(
        relaxed_f1_threshold=(
            relaxed_f1_threshold
        ),
        threshold_start=(
            threshold_start
        ),
        threshold_end=(
            threshold_end
        ),
        threshold_step=(
            threshold_step
        ),
    )

    predictions = (
        load_jsonl(
            input_path
        )
    )

    validate_predictions(
        predictions
    )

    correctness: list[
        bool
    ] = []

    exact_match_scores: list[
        float
    ] = []

    token_f1_scores: list[
        float
    ] = []

    for prediction in predictions:
        (
            is_correct,
            exact_match,
            token_f1,
        ) = is_prediction_correct(
            prediction=(
                prediction
            ),
            relaxed_f1_threshold=(
                relaxed_f1_threshold
            ),
        )

        correctness.append(
            is_correct
        )

        exact_match_scores.append(
            exact_match
        )

        token_f1_scores.append(
            token_f1
        )

    thresholds = (
        generate_thresholds(
            start=(
                threshold_start
            ),
            end=(
                threshold_end
            ),
            step=(
                threshold_step
            ),
        )
    )

    confidence_results = (
        sweep_thresholds(
            predictions=(
                predictions
            ),
            correctness=(
                correctness
            ),
            score_function=(
                get_calibrated_confidence
            ),
            thresholds=(
                thresholds
            ),
        )
    )

    hybrid_results = (
        sweep_thresholds(
            predictions=(
                predictions
            ),
            correctness=(
                correctness
            ),
            score_function=(
                get_hybrid_score
            ),
            thresholds=(
                thresholds
            ),
        )
    )

    confidence_curve = (
        build_exact_risk_coverage_curve(
            predictions=(
                predictions
            ),
            correctness=(
                correctness
            ),
            score_function=(
                get_calibrated_confidence
            ),
        )
    )

    hybrid_curve = (
        build_exact_risk_coverage_curve(
            predictions=(
                predictions
            ),
            correctness=(
                correctness
            ),
            score_function=(
                get_hybrid_score
            ),
        )
    )

    confidence_aurc = (
        calculate_aurc(
            confidence_curve
        )
    )

    hybrid_aurc = (
        calculate_aurc(
            hybrid_curve
        )
    )

    comparisons = (
        compare_at_target_coverages(
            confidence_results=(
                confidence_results
            ),
            hybrid_results=(
                hybrid_results
            ),
            target_coverages=list(
                DEFAULT_TARGET_COVERAGES
            ),
        )
    )

    total_count = len(
        predictions
    )

    correct_count = sum(
        int(value)
        for value
        in correctness
    )

    incorrect_count = (
        total_count
        - correct_count
    )

    average_exact_match = (
        sum(
            exact_match_scores
        )
        / total_count
    )

    average_token_f1 = (
        sum(
            token_f1_scores
        )
        / total_count
    )

    aurc_difference = (
        hybrid_aurc
        - confidence_aurc
    )

    aurc_relative_change = (
        None
        if math.isclose(
            confidence_aurc,
            0.0,
            abs_tol=1e-15,
        )
        else (
            aurc_difference
            / confidence_aurc
        )
    )

    summary = {
        "input_path": (
            str(input_path)
        ),
        "total_predictions": (
            total_count
        ),
        "correct_predictions": (
            correct_count
        ),
        "incorrect_predictions": (
            incorrect_count
        ),
        "relaxed_f1_threshold": (
            relaxed_f1_threshold
        ),
        "correctness_definition": (
            "answerable: exact match OR token F1 "
            ">= relaxed threshold; unanswerable "
            "forced-answer candidates are incorrect"
        ),
        "average_exact_match": (
            average_exact_match
        ),
        "average_token_f1": (
            average_token_f1
        ),
        "aurc_definition": (
            "mean selective risk over every "
            "non-empty score-ranked prefix"
        ),
        "tie_break_rule": (
            "score descending, "
            "original index ascending"
        ),
        "confidence_aurc": (
            confidence_aurc
        ),
        "hybrid_aurc": (
            hybrid_aurc
        ),
        "aurc_difference": (
            aurc_difference
        ),
        "aurc_relative_change": (
            aurc_relative_change
        ),
        "threshold_grid": {
            "start": (
                threshold_start
            ),
            "end": (
                threshold_end
            ),
            "step": (
                threshold_step
            ),
        },
        "matched_coverage_note": (
            "Threshold-grid comparisons are "
            "approximate; coverage_gap reports "
            "hybrid coverage minus confidence coverage."
        ),
    }

    output_directory = Path(
        output_dir
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_csv(
        rows=(
            confidence_results
        ),
        output_path=(
            output_directory
            / "confidence_threshold_sweep.csv"
        ),
    )

    save_csv(
        rows=(
            hybrid_results
        ),
        output_path=(
            output_directory
            / "hybrid_threshold_sweep.csv"
        ),
    )

    save_csv(
        rows=(
            confidence_curve
        ),
        output_path=(
            output_directory
            / "confidence_risk_coverage_curve.csv"
        ),
    )

    save_csv(
        rows=(
            hybrid_curve
        ),
        output_path=(
            output_directory
            / "hybrid_risk_coverage_curve.csv"
        ),
    )

    save_csv(
        rows=(
            comparisons
        ),
        output_path=(
            output_directory
            / "matched_coverage_comparison.csv"
        ),
    )

    save_json(
        data=(
            summary
        ),
        output_path=(
            output_directory
            / "risk_coverage_summary.json"
        ),
    )

    print(
        "\nRisk-coverage evaluation completed."
    )

    print(
        f"Input: "
        f"{input_path}"
    )

    print(
        f"Total predictions: "
        f"{total_count}"
    )

    print(
        f"Correct predictions: "
        f"{correct_count}"
    )

    print(
        f"Incorrect predictions: "
        f"{incorrect_count}"
    )

    print(
        f"Average Exact Match: "
        f"{average_exact_match:.4f}"
    )

    print(
        f"Average Token F1: "
        f"{average_token_f1:.4f}"
    )

    print_threshold_summary(
        system_name=(
            "CONFIDENCE THRESHOLD SWEEP"
        ),
        results=(
            confidence_results
        ),
        selected_thresholds=list(
            DEFAULT_DISPLAY_THRESHOLDS
        ),
    )

    print_threshold_summary(
        system_name=(
            "HYBRID THRESHOLD SWEEP"
        ),
        results=(
            hybrid_results
        ),
        selected_thresholds=list(
            DEFAULT_DISPLAY_THRESHOLDS
        ),
    )

    print_matched_coverage_table(
        comparisons
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "AURC SUMMARY"
    )

    print(
        "=" * 60
    )

    print(
        f"Confidence AURC: "
        f"{confidence_aurc:.6f}"
    )

    print(
        f"Hybrid AURC:     "
        f"{hybrid_aurc:.6f}"
    )

    print(
        "Hybrid - Confidence: "
        f"{aurc_difference:.6f}"
    )

    if (
        hybrid_aurc
        < confidence_aurc
    ):
        print(
            "Result: Hybrid ranking is better "
            "(lower AURC)."
        )

    elif (
        hybrid_aurc
        > confidence_aurc
    ):
        print(
            "Result: Confidence ranking is better "
            "(lower AURC)."
        )

    else:
        print(
            "Result: Both systems have equal AURC."
        )

    print(
        f"\nResults saved to: "
        f"{output_directory}"
    )

    return {
        "summary": (
            summary
        ),
        "confidence_threshold_sweep": (
            confidence_results
        ),
        "hybrid_threshold_sweep": (
            hybrid_results
        ),
        "confidence_curve": (
            confidence_curve
        ),
        "hybrid_curve": (
            hybrid_curve
        ),
        "matched_coverage": (
            comparisons
        ),
    }


def parse_arguments() -> argparse.Namespace:
    """Parse confidence-vs-hybrid risk-coverage evaluation settings."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare calibrated confidence and "
            "hybrid verification using threshold "
            "sweeps and exact risk-coverage ranking."
        )
    )

    parser.add_argument(
        "--input",
        default=str(
            DEFAULT_INPUT_PATH
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIR
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
        "--threshold-start",
        type=float,
        default=(
            DEFAULT_THRESHOLD_START
        ),
    )

    parser.add_argument(
        "--threshold-end",
        type=float,
        default=(
            DEFAULT_THRESHOLD_END
        ),
    )

    parser.add_argument(
        "--threshold-step",
        type=float,
        default=(
            DEFAULT_THRESHOLD_STEP
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = (
        parse_arguments()
    )

    run_risk_coverage_evaluation(
        input_path=(
            arguments.input
        ),
        output_dir=(
            arguments.output_dir
        ),
        relaxed_f1_threshold=(
            arguments.relaxed_f1_threshold
        ),
        threshold_start=(
            arguments.threshold_start
        ),
        threshold_end=(
            arguments.threshold_end
        ),
        threshold_step=(
            arguments.threshold_step
        ),
    )
