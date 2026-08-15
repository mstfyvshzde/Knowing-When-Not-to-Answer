"""
Evaluate score-ranking ablations for selective question answering.

This module compares confidence and verifier-derived ranking scores by asking:

    If predictions are sorted from most to least trustworthy, how much risk
    remains as coverage increases?

The canonical methods evaluated here are:

1. Confidence only
2. Question-aware semantic V2
3. Confidence + question-aware semantic V2
4. Self-verifier only
5. Confidence + self-verifier

Optional lexical and older semantic scores are included only when complete
compatible fields are present.

For every method, the evaluator:

- reconstructs underlying QA correctness,
- ranks records by score in descending order,
- breaks score ties by original record index,
- evaluates matched coverage levels,
- builds the full discrete risk-coverage curve,
- calculates AURC,
- calculates normalized AURC relative to oracle and random ranking,
- saves JSON/CSV summaries and a risk-coverage plot.

Important
---------
Correctness is reconstructed from the prediction, answerability/reference
information, and explicit ANSWER/ABSTAIN decision when necessary. Stored
correctness fields are used only as a compatibility fallback.

For unanswerable examples, an explicit decision takes precedence. This prevents
a punctuation-only forced answer such as "." from becoming correct merely
because answer normalization removes punctuation.

Question-aware V2 records with invalid generated claims receive semantic score
0.0 by design.

`self_verification_score` is defined on [-1, 1] and is mapped linearly to [0, 1]:

    normalized_self_score = (score + 1) / 2

The two fixed fusion methods use equal-weight geometric means:

    sqrt(confidence * verifier_score)

AURC in this repository is the arithmetic mean of risk at every non-empty
prefix of the ranked list. Lower AURC is better.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.evaluation.metrics import (
    normalize_answer as shared_normalize_answer,
)
from src.evaluation.metrics import (
    parse_answerability,
)
from src.utils.io import (
    load_jsonl as shared_load_jsonl,
)
from src.utils.io import (
    save_json as shared_save_json,
)

DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/"
    "calibration_with_question_aware_semantic_evidence_v2.jsonl"
)

DEFAULT_OUTPUT_DIRECTORY = Path(
    "outputs/evaluation/question_aware_ablation"
)

DEFAULT_COVERAGE_LEVELS = (
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
)


CALIBRATED_CONFIDENCE_FIELDS = (
    "calibrated_confidence",
    "temperature_scaled_confidence",
    "qa_calibrated_confidence",
)

GENERIC_CONFIDENCE_FIELD = "confidence"
CONFIDENCE_CALIBRATED_FLAG = "confidence_is_calibrated"

# Historical public constant retained for compatibility.
CONFIDENCE_FIELDS = (
    *CALIBRATED_CONFIDENCE_FIELDS,
    GENERIC_CONFIDENCE_FIELD,
)

LEXICAL_SCORE_FIELDS = (
    "combined_evidence_score",
    "evidence_score",
    "lexical_verification_score",
    "lexical_score",
    "lexical_evidence_score",
    "lexical_support_score",
    "lexical_overlap_score",
)

OLD_SEMANTIC_SCORE_FIELDS = (
    "semantic_entailment_probability",
    "semantic_entailment_score",
    "semantic_verification_score",
    "semantic_score",
    "entailment_probability",
)

QUESTION_AWARE_SCORE_FIELDS = (
    "qa_entailment_probability",
)

SELF_VERIFICATION_SCORE_FIELDS = (
    "self_verification_score",
)

QUESTION_AWARE_VALIDITY_FIELDS = (
    "qa_claim_valid",
)

PREDICTION_FIELDS = (
    "predicted_answer",
    "prediction_text",
    "prediction_answer",
    "prediction",
    "model_answer",
    "generated_answer",
    "answer",
)

REFERENCE_FIELDS = (
    "reference_answers",
    "gold_answers",
    "accepted_answers",
    "references",
    "reference_answer",
    "gold_answer",
    "target_answer",
    "ground_truth",
    "answers",
    "answer_texts",
    "gold",
)

ANSWERABILITY_FIELDS = (
    "is_answerable",
    "answerable",
    "gold_is_answerable",
)

CORRECTNESS_FIELDS = (
    "exact_match",
    "em",
    "is_exact_match",
    "exact_match_score",
    "em_score",
    "prediction_correct",
    "answer_correct",
    "prediction_is_correct",
    "qa_is_correct",
    "correct",
    "is_correct",
)


@dataclass(frozen=True)
class SelectiveMetrics:
    """
    Metrics at one requested coverage level.

    `correct_abstained` and `incorrect_abstained` preserve historical output
    names. They refer to the correctness of the underlying QA predictions that
    were abstained on, not to whether abstention itself was a correct action.
    """

    requested_coverage: float
    answered: int
    total: int
    actual_coverage: float
    selective_accuracy: float
    risk: float
    wrong_answered: int
    correct_answered: int
    abstained: int
    correct_abstained: int
    incorrect_abstained: int


@dataclass(frozen=True)
class MethodResult:
    """
    Complete risk-coverage evaluation for one ranking method.

    `full_accuracy` is underlying QA prediction accuracy before selective
    ranking. It is therefore identical across methods evaluated on the same
    records.
    """

    method: str
    score_field: str
    available_records: int
    total_records: int
    full_accuracy: float
    aurc: float
    normalized_aurc: float | None
    matched_coverage: list[SelectiveMetrics]


def load_jsonl(
    path: str | Path,
) -> list[dict[str, Any]]:
    """
    Load JSONL records through the repository's shared I/O implementation.

    The historical evaluator required at least one record, so that behavior is
    preserved here.
    """

    records = shared_load_jsonl(path)

    if not records:
        raise ValueError(
            f"No records were found in {Path(path)}"
        )

    return records


def save_json(
    data: Any,
    path: str | Path,
) -> None:
    """Save JSON through the repository's shared UTF-8 JSON writer."""

    shared_save_json(
        data,
        path,
    )


def normalize_answer(
    answer: Any,
) -> str:
    """
    Normalize answer text using the shared project normalization rule.

    The local wrapper preserves this module's historical public function.
    """

    return shared_normalize_answer(
        str(answer)
    )


def get_first_value(
    record: dict[str, Any],
    field_names: Sequence[str],
    default: Any = None,
) -> Any:
    """Return the first available non-None value from ordered field aliases."""

    for field_name in field_names:
        if (
            field_name in record
            and record[field_name] is not None
        ):
            return record[field_name]

    return default


def clean_text(text: Any) -> str:
    """Collapse repeated whitespace and remove surrounding whitespace."""

    if text is None:
        return ""

    return " ".join(
        str(text).split()
    )


def coerce_boolean(
    value: Any,
) -> bool | None:
    """
    Convert common stored Boolean/correctness representations.

    None is returned when the value cannot be interpreted unambiguously.
    """

    if isinstance(value, bool):
        return value

    if (
        isinstance(value, int)
        and not isinstance(value, bool)
    ):
        if value == 1:
            return True

        if value == 0:
            return False

    if isinstance(value, float) and math.isfinite(value):
        if math.isclose(value, 1.0):
            return True

        if math.isclose(value, 0.0):
            return False

    if isinstance(value, str):
        normalized_value = (
            value.strip().lower()
        )

        if normalized_value in {
            "true",
            "yes",
            "correct",
            "1",
            "1.0",
        }:
            return True

        if normalized_value in {
            "false",
            "no",
            "incorrect",
            "0",
            "0.0",
        }:
            return False

    return None


def coerce_float(
    value: Any,
) -> float | None:
    """
    Convert a stored numeric value to a finite float.

    Boolean values are rejected because True/False should not silently become
    ranking scores 1.0/0.0.
    """

    if (
        value is None
        or isinstance(value, bool)
    ):
        return None

    try:
        number = float(value)

    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def clamp_probability(
    value: float,
) -> float:
    """
    Clamp a derived value to [0, 1].

    Retained for API compatibility. Canonical experimental input scores are
    validated instead of silently clamped.
    """

    return max(
        0.0,
        min(
            1.0,
            float(value),
        ),
    )


def extract_reference_value(
    value: Any,
) -> list[str]:
    """
    Recursively extract answer strings from supported reference structures.
    """

    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, dict):
        for possible_field in (
            "text",
            "answer",
            "answers",
        ):
            if possible_field in value:
                return extract_reference_value(
                    value[possible_field]
                )

        return []

    if isinstance(value, Iterable):
        references: list[str] = []

        for item in value:
            references.extend(
                extract_reference_value(item)
            )

        return references

    return [str(value)]


def extract_references(
    record: dict[str, Any],
) -> list[str]:
    """Extract reference answers from the first supported reference field."""

    raw_value = get_first_value(
        record,
        REFERENCE_FIELDS,
        default=None,
    )

    if raw_value is None:
        return []

    return extract_reference_value(
        raw_value
    )


def get_reference_field(
    record: dict[str, Any],
) -> tuple[bool, Any]:
    """
    Return whether a reference field exists and its raw value.

    Presence is tracked separately from extracted content so an explicitly empty
    reference list can represent an unanswerable example.
    """

    for field_name in REFERENCE_FIELDS:
        if field_name in record:
            return (
                True,
                record[field_name],
            )

    return (
        False,
        None,
    )


def get_explicit_answerability(
    record: dict[str, Any],
) -> bool | None:
    """
    Retrieve explicit answerability metadata when available.

    Malformed explicit values are rejected instead of silently inferred from
    references.
    """

    for field_name in ANSWERABILITY_FIELDS:
        if field_name not in record:
            continue

        value = record[field_name]

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

    return None


def get_stored_correctness(
    record: dict[str, Any],
) -> bool | None:
    """
    Retrieve a stored correctness value only as a compatibility fallback.

    Raw prediction/reference reconstruction is preferred because generic fields
    such as `is_correct` can have different meanings in different pipeline
    stages.
    """

    for field_name in CORRECTNESS_FIELDS:
        if field_name not in record:
            continue

        correctness = coerce_boolean(
            record[field_name]
        )

        if correctness is not None:
            return correctness

    return None


def infer_correctness(
    record: dict[str, Any],
) -> bool:
    """
    Reconstruct underlying QA prediction correctness.

    Preferred reconstruction uses:

    - predicted answer text,
    - reference answers,
    - optional explicit answerability metadata,
    - explicit ANSWER/ABSTAIN decision for unanswerable examples.

    Answerable examples are correct only when normalized prediction text exactly
    matches a normalized non-empty reference.

    For unanswerable examples:
    - if an explicit decision exists, only ABSTAIN is correct;
    - otherwise, only a genuinely empty raw prediction is treated as correct.

    The raw-text fallback intentionally does not use answer normalization, so a
    punctuation-only forced answer such as "." is not mistaken for abstention.
    """

    prediction_value = get_first_value(
        record,
        PREDICTION_FIELDS,
        default=None,
    )

    (
        reference_field_found,
        raw_reference_value,
    ) = get_reference_field(
        record
    )

    if (
        prediction_value is not None
        and reference_field_found
    ):
        normalized_prediction = (
            normalize_answer(
                prediction_value
            )
        )

        references = (
            extract_reference_value(
                raw_reference_value
            )
        )

        normalized_references = [
            normalize_answer(
                reference
            )
            for reference
            in references
        ]

        normalized_references = [
            reference
            for reference
            in normalized_references
            if reference
        ]

        explicit_answerability = (
            get_explicit_answerability(
                record
            )
        )

        inferred_answerability = bool(
            normalized_references
        )

        if explicit_answerability is not None:
            if (
                explicit_answerability
                and not normalized_references
            ):
                raise ValueError(
                    "Record is explicitly answerable "
                    "but has no usable reference answer."
                )

            if (
                not explicit_answerability
                and normalized_references
            ):
                raise ValueError(
                    "Record is explicitly unanswerable "
                    "but contains non-empty reference answers."
                )

            is_answerable = (
                explicit_answerability
            )

        else:
            is_answerable = (
                inferred_answerability
            )

        if is_answerable:
            return any(
                normalized_prediction
                == reference
                for reference
                in normalized_references
            )

        decision = clean_text(
            record.get(
                "decision",
                "",
            )
        ).upper()

        if decision:
            if decision not in {
                "ANSWER",
                "ABSTAIN",
            }:
                raise ValueError(
                    "Unanswerable-record decision must "
                    f"be ANSWER or ABSTAIN, received "
                    f"{decision!r}."
                )

            return decision == "ABSTAIN"

        return (
            clean_text(
                prediction_value
            )
            == ""
        )

    stored_correctness = (
        get_stored_correctness(
            record
        )
    )

    if stored_correctness is not None:
        return stored_correctness

    if prediction_value is None:
        raise ValueError(
            "Could not infer correctness because "
            "no prediction field was found. "
            f"Available keys: "
            f"{sorted(record.keys())}"
        )

    if not reference_field_found:
        raise ValueError(
            "Could not infer correctness because "
            "no reference-answer field was found. "
            f"Available keys: "
            f"{sorted(record.keys())}"
        )

    raise ValueError(
        "Could not infer correctness from "
        "the available record."
    )


def find_available_field(
    records: Sequence[dict[str, Any]],
    candidates: Sequence[str],
) -> str | None:
    """
    Return the candidate numeric field with the greatest usable coverage.

    Candidate order breaks ties, keeping field selection deterministic.
    """

    best_field: str | None = None
    best_count = 0

    for field_name in candidates:
        numeric_count = sum(
            coerce_float(
                record.get(
                    field_name
                )
            )
            is not None
            for record in records
        )

        if numeric_count > best_count:
            best_field = field_name
            best_count = numeric_count

    return best_field


def find_calibrated_confidence_field(
    records: Sequence[dict[str, Any]],
) -> str:
    """
    Resolve the confidence field used by canonical ranking/fusion.

    Explicit calibrated fields are preferred. The generic `confidence` field is
    allowed only when every record explicitly marks it as calibrated.
    """

    field_name = find_available_field(
        records,
        CALIBRATED_CONFIDENCE_FIELDS,
    )

    if field_name is not None:
        return field_name

    has_generic_confidence = all(
        coerce_float(
            record.get(
                GENERIC_CONFIDENCE_FIELD
            )
        )
        is not None
        for record in records
    )

    generic_is_calibrated = all(
        coerce_boolean(
            record.get(
                CONFIDENCE_CALIBRATED_FLAG
            )
        )
        is True
        for record in records
    )

    if (
        has_generic_confidence
        and generic_is_calibrated
    ):
        return (
            GENERIC_CONFIDENCE_FIELD
        )

    raise ValueError(
        "No complete calibrated confidence "
        "field was found. "
        "Checked explicit fields: "
        f"{CALIBRATED_CONFIDENCE_FIELDS}. "
        "The generic 'confidence' field is "
        "accepted only when "
        "'confidence_is_calibrated' is true "
        "for every record."
    )


def extract_probability_score(
    record: dict[str, Any],
    field_name: str,
) -> float | None:
    """
    Extract a finite probability-like score in [0, 1].

    Out-of-range values are rejected rather than silently clamped.
    """

    value = coerce_float(
        record.get(
            field_name
        )
    )

    if value is None:
        return None

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"Score field {field_name!r} "
            "must lie in [0, 1], "
            f"received {value}."
        )

    return value


def extract_numeric_score(
    record: dict[str, Any],
    field_name: str,
) -> float | None:
    """
    Compatibility wrapper for probability-like experimental scores.

    Unlike the historical implementation, out-of-range values raise an error
    instead of being silently clamped.
    """

    return extract_probability_score(
        record,
        field_name,
    )


def extract_complete_probability_scores(
    records: Sequence[dict[str, Any]],
    field_name: str,
    signal_name: str,
) -> list[float]:
    """
    Extract one required probability score from every record.
    """

    scores: list[float] = []

    for index, record in enumerate(
        records,
        start=1,
    ):
        score = (
            extract_probability_score(
                record,
                field_name,
            )
        )

        if score is None:
            raise ValueError(
                f"{signal_name} field "
                f"{field_name!r} is missing "
                "or non-numeric in record "
                f"{index}."
            )

        scores.append(
            score
        )

    return scores


def extract_question_aware_score(
    record: dict[str, Any],
    score_field: str,
) -> float:
    """
    Extract the question-aware V2 semantic score.

    Invalid generated claims receive score 0.0 by the predeclared V2 policy.

    A valid claim must contain a usable entailment probability; missing scores
    are treated as malformed experiment data rather than silently becoming zero.
    """

    claim_valid_value = (
        get_first_value(
            record,
            QUESTION_AWARE_VALIDITY_FIELDS,
            default=None,
        )
    )

    if claim_valid_value is None:
        raise ValueError(
            "Question-aware record does not "
            "contain qa_claim_valid."
        )

    claim_valid = coerce_boolean(
        claim_valid_value
    )

    if claim_valid is None:
        raise ValueError(
            "qa_claim_valid could not be "
            "interpreted as Boolean: "
            f"{claim_valid_value!r}."
        )

    if not claim_valid:
        return 0.0

    score = extract_probability_score(
        record,
        score_field,
    )

    if score is None:
        raise ValueError(
            "Valid question-aware claim is "
            f"missing {score_field!r}."
        )

    return score


def extract_self_verification_score(
    record: dict[str, Any],
    score_field: str,
) -> float | None:
    """
    Normalize self-verification score from [-1, 1] to [0, 1].
    """

    score = coerce_float(
        record.get(
            score_field
        )
    )

    if score is None:
        return None

    if not -1.0 <= score <= 1.0:
        raise ValueError(
            "Self-verification score must "
            "lie in [-1, 1], "
            f"received {score}."
        )

    return (
        score + 1.0
    ) / 2.0


def geometric_mean_score(
    first_score: float,
    second_score: float,
) -> float:
    """
    Calculate the equal-weight geometric mean of two [0, 1] scores.
    """

    for score_name, score in (
        (
            "first_score",
            first_score,
        ),
        (
            "second_score",
            second_score,
        ),
    ):
        if not math.isfinite(
            score
        ):
            raise ValueError(
                f"{score_name} must be finite."
            )

        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"{score_name} must lie "
                "in [0, 1], "
                f"received {score}."
            )

    return math.sqrt(
        first_score
        * second_score
    )


def create_ranked_indices(
    scores: Sequence[float],
) -> list[int]:
    """
    Rank records by score descending with deterministic original-index ties.

    Tie rule:

        higher score first
        then smaller original index first
    """

    if not scores:
        raise ValueError(
            "Cannot rank an empty score sequence."
        )

    if any(
        not math.isfinite(score)
        for score in scores
    ):
        raise ValueError(
            "Ranking scores must all be finite."
        )

    return sorted(
        range(
            len(scores)
        ),
        key=lambda index: (
            -scores[index],
            index,
        ),
    )


def metrics_at_answer_count(
    correctness: Sequence[bool],
    ranked_indices: Sequence[int],
    answer_count: int,
    requested_coverage: float,
) -> SelectiveMetrics:
    """
    Evaluate selective risk when answering the top-ranked `answer_count` items.
    """

    total = len(
        correctness
    )

    if total == 0:
        raise ValueError(
            "Cannot evaluate an empty "
            "correctness sequence."
        )

    if (
        len(ranked_indices)
        != total
    ):
        raise ValueError(
            "Ranked-index count must match "
            "correctness count."
        )

    answer_count = max(
        1,
        min(
            answer_count,
            total,
        ),
    )

    answered_indices = set(
        ranked_indices[
            :answer_count
        ]
    )

    correct_answered = sum(
        int(
            correctness[index]
        )
        for index
        in answered_indices
    )

    wrong_answered = (
        answer_count
        - correct_answered
    )

    total_correct = sum(
        int(value)
        for value
        in correctness
    )

    total_incorrect = (
        total
        - total_correct
    )

    correct_abstained = (
        total_correct
        - correct_answered
    )

    incorrect_abstained = (
        total_incorrect
        - wrong_answered
    )

    selective_accuracy = (
        correct_answered
        / answer_count
    )

    risk = (
        wrong_answered
        / answer_count
    )

    return SelectiveMetrics(
        requested_coverage=(
            requested_coverage
        ),
        answered=(
            answer_count
        ),
        total=total,
        actual_coverage=(
            answer_count
            / total
        ),
        selective_accuracy=(
            selective_accuracy
        ),
        risk=risk,
        wrong_answered=(
            wrong_answered
        ),
        correct_answered=(
            correct_answered
        ),
        abstained=(
            total
            - answer_count
        ),
        correct_abstained=(
            correct_abstained
        ),
        incorrect_abstained=(
            incorrect_abstained
        ),
    )


def build_risk_coverage_curve(
    correctness: Sequence[bool],
    scores: Sequence[float],
) -> list[
    dict[
        str,
        float | int,
    ]
]:
    """
    Build the full discrete risk-coverage curve.

    One point is emitted for every non-empty ranked prefix from 1/N through
    full coverage.
    """

    if (
        len(correctness)
        != len(scores)
    ):
        raise ValueError(
            "Correctness and score lengths "
            "must match."
        )

    if not correctness:
        raise ValueError(
            "Cannot build a curve from "
            "empty inputs."
        )

    ranked_indices = (
        create_ranked_indices(
            scores
        )
    )

    curve: list[
        dict[
            str,
            float | int,
        ]
    ] = []

    correct_answered = 0

    for (
        answer_count,
        record_index,
    ) in enumerate(
        ranked_indices,
        start=1,
    ):
        correct_answered += int(
            correctness[
                record_index
            ]
        )

        wrong_answered = (
            answer_count
            - correct_answered
        )

        curve.append(
            {
                "answered": (
                    answer_count
                ),
                "coverage": (
                    answer_count
                    / len(correctness)
                ),
                "selective_accuracy": (
                    correct_answered
                    / answer_count
                ),
                "risk": (
                    wrong_answered
                    / answer_count
                ),
                "wrong_answered": (
                    wrong_answered
                ),
            }
        )

    return curve


def calculate_aurc(
    curve: Sequence[
        dict[
            str,
            float | int,
        ]
    ],
) -> float:
    """
    Calculate the repository's discrete AURC.

    AURC is the arithmetic mean of risk across all non-empty ranked prefixes.
    This definition is intentionally preserved for reproducibility.
    """

    if not curve:
        raise ValueError(
            "Cannot calculate AURC from "
            "an empty curve."
        )

    return (
        sum(
            float(
                point[
                    "risk"
                ]
            )
            for point
            in curve
        )
        / len(curve)
    )


def calculate_optimal_aurc(
    correctness: Sequence[bool],
) -> float:
    """
    Calculate oracle AURC by ranking all correct predictions before errors.
    """

    if not correctness:
        raise ValueError(
            "Cannot calculate optimal AURC "
            "from empty correctness."
        )

    oracle_scores = [
        (
            1.0
            if is_correct
            else 0.0
        )
        for is_correct
        in correctness
    ]

    oracle_curve = (
        build_risk_coverage_curve(
            correctness,
            oracle_scores,
        )
    )

    return calculate_aurc(
        oracle_curve
    )


def calculate_random_aurc(
    correctness: Sequence[bool],
) -> float:
    """
    Calculate expected AURC under a uniformly random ranking.

    At every non-empty prefix, expected risk equals the dataset error rate.

    Therefore:

        random AURC = 1 - full accuracy
    """

    if not correctness:
        raise ValueError(
            "Cannot calculate random AURC "
            "from empty correctness."
        )

    return (
        1.0
        - sum(
            int(value)
            for value
            in correctness
        )
        / len(correctness)
    )


def calculate_normalized_aurc(
    aurc: float,
    optimal_aurc: float,
    random_aurc: float,
) -> float | None:
    """
    Normalize AURC between oracle and expected-random reference points.

    0 corresponds to oracle ranking.

    1 corresponds to expected random ranking.

    Values above 1 are possible when a ranking performs worse than random.
    """

    denominator = (
        random_aurc
        - optimal_aurc
    )

    if math.isclose(
        denominator,
        0.0,
        abs_tol=1e-12,
    ):
        return None

    return (
        aurc
        - optimal_aurc
    ) / denominator


def validate_coverage_levels(
    coverage_levels: Sequence[float],
) -> tuple[float, ...]:
    """
    Validate coverage levels and return a deterministic sorted tuple.
    """

    if not coverage_levels:
        raise ValueError(
            "At least one coverage level "
            "is required."
        )

    normalized_levels: list[
        float
    ] = []

    for value in coverage_levels:
        numeric_value = float(
            value
        )

        if (
            not math.isfinite(
                numeric_value
            )
            or not (
                0.0
                < numeric_value
                <= 1.0
            )
        ):
            raise ValueError(
                "Coverage levels must be "
                "finite and lie in (0, 1]."
            )

        normalized_levels.append(
            numeric_value
        )

    return tuple(
        sorted(
            set(
                normalized_levels
            )
        )
    )


def evaluate_method(
    method_name: str,
    score_field_description: str,
    correctness: Sequence[bool],
    scores: Sequence[float],
    coverage_levels: Sequence[float],
) -> tuple[
    MethodResult,
    list[
        dict[
            str,
            float | int,
        ]
    ],
]:
    """
    Evaluate one ranking method at matched coverage and across the full curve.
    """

    if (
        len(correctness)
        != len(scores)
    ):
        raise ValueError(
            f"Length mismatch for "
            f"method {method_name}."
        )

    if not correctness:
        raise ValueError(
            f"Method {method_name} "
            "has no records."
        )

    validated_coverage_levels = (
        validate_coverage_levels(
            coverage_levels
        )
    )

    total = len(
        correctness
    )

    ranked_indices = (
        create_ranked_indices(
            scores
        )
    )

    matched_metrics: list[
        SelectiveMetrics
    ] = []

    for requested_coverage in (
        validated_coverage_levels
    ):
        answer_count = max(
            1,
            math.ceil(
                requested_coverage
                * total
            ),
        )

        matched_metrics.append(
            metrics_at_answer_count(
                correctness=correctness,
                ranked_indices=(
                    ranked_indices
                ),
                answer_count=(
                    answer_count
                ),
                requested_coverage=(
                    requested_coverage
                ),
            )
        )

    curve = (
        build_risk_coverage_curve(
            correctness,
            scores,
        )
    )

    aurc = calculate_aurc(
        curve
    )

    optimal_aurc = (
        calculate_optimal_aurc(
            correctness
        )
    )

    random_aurc = (
        calculate_random_aurc(
            correctness
        )
    )

    normalized_aurc = (
        calculate_normalized_aurc(
            aurc=aurc,
            optimal_aurc=(
                optimal_aurc
            ),
            random_aurc=(
                random_aurc
            ),
        )
    )

    result = MethodResult(
        method=(
            method_name
        ),
        score_field=(
            score_field_description
        ),
        available_records=sum(
            math.isfinite(
                score
            )
            for score
            in scores
        ),
        total_records=(
            total
        ),
        full_accuracy=(
            sum(
                int(value)
                for value
                in correctness
            )
            / total
        ),
        aurc=aurc,
        normalized_aurc=(
            normalized_aurc
        ),
        matched_coverage=(
            matched_metrics
        ),
    )

    return (
        result,
        curve,
    )


def save_matched_coverage_csv(
    results: Sequence[
        MethodResult
    ],
    path: str | Path,
) -> None:
    """
    Save matched-coverage metrics for every evaluated method.
    """

    output_path = Path(
        path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    field_names = [
        "method",
        "requested_coverage",
        "actual_coverage",
        "answered",
        "total",
        "selective_accuracy",
        "risk",
        "wrong_answered",
        "correct_answered",
        "abstained",
        "correct_abstained",
        "incorrect_abstained",
    ]

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

        for result in results:
            for metric in (
                result.matched_coverage
            ):
                writer.writerow(
                    {
                        "method": (
                            result.method
                        ),
                        **asdict(
                            metric
                        ),
                    }
                )


def save_summary_csv(
    results: Sequence[
        MethodResult
    ],
    path: str | Path,
) -> None:
    """
    Save one aggregate summary row per ranking method.
    """

    output_path = Path(
        path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "method",
                "score_field",
                "total_records",
                "full_accuracy",
                "aurc",
                "normalized_aurc",
            ],
            lineterminator="\n",
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "method": (
                        result.method
                    ),
                    "score_field": (
                        result.score_field
                    ),
                    "total_records": (
                        result.total_records
                    ),
                    "full_accuracy": (
                        result.full_accuracy
                    ),
                    "aurc": (
                        result.aurc
                    ),
                    "normalized_aurc": (
                        result.normalized_aurc
                    ),
                }
            )


def save_curve_csv(
    curves: dict[
        str,
        list[
            dict[
                str,
                float | int,
            ]
        ],
    ],
    path: str | Path,
) -> None:
    """
    Save every point from every risk-coverage curve.
    """

    output_path = Path(
        path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "method",
                "answered",
                "coverage",
                "selective_accuracy",
                "risk",
                "wrong_answered",
            ],
            lineterminator="\n",
        )

        writer.writeheader()

        for (
            method_name,
            curve,
        ) in curves.items():
            for point in curve:
                writer.writerow(
                    {
                        "method": (
                            method_name
                        ),
                        **point,
                    }
                )


def plot_risk_coverage_curves(
    curves: dict[
        str,
        list[
            dict[
                str,
                float | int,
            ]
        ],
    ],
    path: str | Path,
) -> None:
    """
    Plot and save risk against coverage for all evaluated methods.
    """

    figure = plt.figure(
        figsize=(9, 6)
    )

    axis = (
        figure.add_subplot(
            111
        )
    )

    for (
        method_name,
        curve,
    ) in curves.items():
        axis.plot(
            [
                float(
                    point[
                        "coverage"
                    ]
                )
                for point
                in curve
            ],
            [
                float(
                    point[
                        "risk"
                    ]
                )
                for point
                in curve
            ],
            label=(
                method_name
            ),
        )

    axis.set_xlabel(
        "Coverage"
    )

    axis.set_ylabel(
        "Selective risk"
    )

    axis.set_title(
        "Question-Aware Verification Ablation"
    )

    axis.set_xlim(
        0.0,
        1.0,
    )

    axis.set_ylim(
        bottom=0.0
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    output_path = Path(
        path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def print_summary(
    results: Sequence[
        MethodResult
    ],
) -> None:
    """
    Print methods ordered from lowest to highest AURC.
    """

    print(
        "\n"
        + "=" * 88
    )

    print(
        "QUESTION-AWARE ABLATION SUMMARY"
    )

    print(
        "=" * 88
    )

    print(
        f"{'Method':<42}"
        f"{'AURC':>12}"
        f"{'Norm. AURC':>16}"
        f"{'Full Acc.':>14}"
    )

    print(
        "-" * 88
    )

    for result in sorted(
        results,
        key=lambda item: (
            item.aurc
        ),
    ):
        normalized_text = (
            f"{result.normalized_aurc:.6f}"
            if (
                result.normalized_aurc
                is not None
            )
            else "N/A"
        )

        print(
            f"{result.method:<42}"
            f"{result.aurc:>12.6f}"
            f"{normalized_text:>16}"
            f"{result.full_accuracy:>14.4f}"
        )

    print(
        "\nLower AURC is better."
    )


def print_matched_coverage(
    results: Sequence[
        MethodResult
    ],
) -> None:
    """
    Print method risk comparisons at identical requested coverage levels.
    """

    if not results:
        raise ValueError(
            "No method results are available."
        )

    coverage_levels = [
        metric.requested_coverage
        for metric
        in results[
            0
        ].matched_coverage
    ]

    print(
        "\n"
        + "=" * 88
    )

    print(
        "MATCHED-COVERAGE RISK"
    )

    print(
        "=" * 88
    )

    for coverage in (
        coverage_levels
    ):
        print(
            "\nCoverage target: "
            f"{coverage:.0%}"
        )

        print(
            f"{'Method':<42}"
            f"{'Risk':>10}"
            f"{'Accuracy':>12}"
            f"{'Wrong':>10}"
            f"{'Answered':>12}"
        )

        print(
            "-" * 88
        )

        rows: list[
            tuple[
                str,
                SelectiveMetrics,
            ]
        ] = []

        for result in results:
            metric = next(
                item
                for item
                in result.matched_coverage
                if math.isclose(
                    item.requested_coverage,
                    coverage,
                )
            )

            rows.append(
                (
                    result.method,
                    metric,
                )
            )

        for (
            method_name,
            metric,
        ) in sorted(
            rows,
            key=lambda item: (
                item[1].risk
            ),
        ):
            print(
                f"{method_name:<42}"
                f"{metric.risk:>10.4f}"
                f"{metric.selective_accuracy:>12.4f}"
                f"{metric.wrong_answered:>10}"
                f"{metric.answered:>12}"
            )


def parse_coverage_levels(
    raw_value: str,
) -> tuple[
    float,
    ...,
]:
    """
    Parse comma-separated coverage levels from the command line.
    """

    values: list[
        float
    ] = []

    for item in raw_value.split(
        ","
    ):
        stripped_item = (
            item.strip()
        )

        if not stripped_item:
            continue

        try:
            value = float(
                stripped_item
            )

        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "Invalid coverage value: "
                f"{stripped_item!r}."
            ) from error

        if (
            not math.isfinite(
                value
            )
            or not (
                0.0
                < value
                <= 1.0
            )
        ):
            raise argparse.ArgumentTypeError(
                "Coverage values must be "
                "finite and lie in (0, 1]."
            )

        values.append(
            value
        )

    if not values:
        raise argparse.ArgumentTypeError(
            "At least one coverage value "
            "is required."
        )

    return tuple(
        sorted(
            set(
                values
            )
        )
    )


def validate_required_question_aware_fields(
    records: Sequence[
        dict[
            str,
            Any,
        ]
    ],
) -> None:
    """
    Require a usable claim-validity value for every V2 record.
    """

    for index, record in enumerate(
        records,
        start=1,
    ):
        value = get_first_value(
            record,
            QUESTION_AWARE_VALIDITY_FIELDS,
            default=None,
        )

        if value is None:
            raise ValueError(
                f"Record {index} does not "
                "contain qa_claim_valid."
            )

        if (
            coerce_boolean(
                value
            )
            is None
        ):
            raise ValueError(
                f"Record {index} has invalid "
                "qa_claim_valid value: "
                f"{value!r}."
            )


def evaluate_ablation(
    input_path: str | Path,
    output_directory: str | Path,
    coverage_levels: Sequence[float],
) -> list[
    MethodResult
]:
    """
    Run the complete score-ranking ablation evaluation.

    Required canonical signals:

    - calibrated confidence
    - question-aware V2 entailment
    - question-aware V2 claim validity
    - self-verification score

    Optional lexical and earlier semantic methods are added only when complete
    compatible score fields exist.
    """

    records = load_jsonl(
        input_path
    )

    validated_coverage_levels = (
        validate_coverage_levels(
            coverage_levels
        )
    )

    correctness = [
        infer_correctness(
            record
        )
        for record
        in records
    ]

    confidence_field = (
        find_calibrated_confidence_field(
            records
        )
    )

    lexical_field = (
        find_available_field(
            records,
            LEXICAL_SCORE_FIELDS,
        )
    )

    old_semantic_field = (
        find_available_field(
            records,
            OLD_SEMANTIC_SCORE_FIELDS,
        )
    )

    question_aware_field = (
        find_available_field(
            records,
            QUESTION_AWARE_SCORE_FIELDS,
        )
    )

    self_verification_field = (
        find_available_field(
            records,
            SELF_VERIFICATION_SCORE_FIELDS,
        )
    )

    if question_aware_field is None:
        raise ValueError(
            "No question-aware entailment "
            "field was found. "
            f"Checked: "
            f"{QUESTION_AWARE_SCORE_FIELDS}"
        )

    if self_verification_field is None:
        raise ValueError(
            "No self-verification score "
            "field was found. "
            f"Checked: "
            f"{SELF_VERIFICATION_SCORE_FIELDS}"
        )

    validate_required_question_aware_fields(
        records
    )

    confidence_scores = (
        extract_complete_probability_scores(
            records=records,
            field_name=(
                confidence_field
            ),
            signal_name=(
                "Calibrated confidence"
            ),
        )
    )

    question_aware_scores = [
        extract_question_aware_score(
            record,
            question_aware_field,
        )
        for record
        in records
    ]

    self_verification_scores_optional = [
        extract_self_verification_score(
            record,
            self_verification_field,
        )
        for record
        in records
    ]

    if any(
        score is None
        for score
        in self_verification_scores_optional
    ):
        missing_count = sum(
            score is None
            for score
            in self_verification_scores_optional
        )

        raise ValueError(
            "Self-verification field "
            f"{self_verification_field!r} "
            "is missing or non-numeric in "
            f"{missing_count} records."
        )

    self_verification_scores = [
        float(
            score
        )
        for score
        in self_verification_scores_optional
        if score is not None
    ]

    confidence_question_aware_scores = [
        geometric_mean_score(
            confidence_score,
            semantic_score,
        )
        for (
            confidence_score,
            semantic_score,
        ) in zip(
            confidence_scores,
            question_aware_scores,
        )
    ]

    confidence_self_scores = [
        geometric_mean_score(
            confidence_score,
            self_score,
        )
        for (
            confidence_score,
            self_score,
        ) in zip(
            confidence_scores,
            self_verification_scores,
        )
    ]

    methods: list[
        tuple[
            str,
            str,
            list[
                float
            ],
        ]
    ] = [
        (
            "Confidence only",
            confidence_field,
            confidence_scores,
        ),
        (
            "Question-aware semantic V2",
            (
                f"{question_aware_field}; "
                "invalid claims=0"
            ),
            question_aware_scores,
        ),
        (
            "Confidence + question-aware semantic V2",
            (
                f"sqrt({confidence_field} * "
                f"{question_aware_field}); "
                "invalid claims=0"
            ),
            confidence_question_aware_scores,
        ),
        (
            "Self-verifier only",
            (
                "normalized("
                f"{self_verification_field})"
            ),
            self_verification_scores,
        ),
        (
            "Confidence + self-verifier",
            (
                f"sqrt({confidence_field} * "
                "normalized("
                f"{self_verification_field}))"
            ),
            confidence_self_scores,
        ),
    ]

    if lexical_field is not None:
        try:
            lexical_scores = (
                extract_complete_probability_scores(
                    records=records,
                    field_name=(
                        lexical_field
                    ),
                    signal_name=(
                        "Lexical verifier"
                    ),
                )
            )

        except ValueError:
            print(
                "Skipping lexical verifier: "
                f"field {lexical_field!r} "
                "is incomplete or invalid."
            )

        else:
            methods.insert(
                1,
                (
                    "Lexical verifier only",
                    lexical_field,
                    lexical_scores,
                ),
            )

    if old_semantic_field is not None:
        if (
            old_semantic_field
            == question_aware_field
        ):
            print(
                "Skipping old semantic verifier "
                "because its detected field is "
                "identical to the question-aware "
                "field."
            )

        else:
            try:
                old_semantic_scores = (
                    extract_complete_probability_scores(
                        records=records,
                        field_name=(
                            old_semantic_field
                        ),
                        signal_name=(
                            "Old semantic verifier"
                        ),
                    )
                )

            except ValueError:
                print(
                    "Skipping old semantic verifier: "
                    f"field {old_semantic_field!r} "
                    "is incomplete or invalid."
                )

            else:
                lexical_present = any(
                    method_name
                    == "Lexical verifier only"
                    for (
                        method_name,
                        _,
                        _,
                    ) in methods
                )

                insertion_index = (
                    2
                    if lexical_present
                    else 1
                )

                methods.insert(
                    insertion_index,
                    (
                        "Old semantic verifier only",
                        old_semantic_field,
                        old_semantic_scores,
                    ),
                )

    results: list[
        MethodResult
    ] = []

    curves: dict[
        str,
        list[
            dict[
                str,
                float | int,
            ]
        ],
    ] = {}

    for (
        method_name,
        field_description,
        scores,
    ) in methods:
        (
            result,
            curve,
        ) = evaluate_method(
            method_name=(
                method_name
            ),
            score_field_description=(
                field_description
            ),
            correctness=(
                correctness
            ),
            scores=scores,
            coverage_levels=(
                validated_coverage_levels
            ),
        )

        results.append(
            result
        )

        curves[
            method_name
        ] = curve

    output_directory_path = Path(
        output_directory
    )

    output_directory_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    correct_predictions = sum(
        int(value)
        for value
        in correctness
    )

    summary_payload = {
        "input_path": str(
            input_path
        ),
        "total_records": (
            len(records)
        ),
        "correct_predictions": (
            correct_predictions
        ),
        "incorrect_predictions": (
            len(correctness)
            - correct_predictions
        ),
        "full_accuracy": (
            correct_predictions
            / len(correctness)
        ),
        "detected_fields": {
            "confidence": (
                confidence_field
            ),
            "lexical": (
                lexical_field
            ),
            "old_semantic": (
                old_semantic_field
            ),
            "question_aware_semantic": (
                question_aware_field
            ),
            "self_verification": (
                self_verification_field
            ),
        },
        "combination_rules": {
            "confidence_question_aware": (
                "equal-weight geometric mean: "
                "sqrt(confidence * "
                "qa_entailment)"
            ),
            "confidence_self_verification": (
                "equal-weight geometric mean: "
                "sqrt(confidence * "
                "normalized_self_verification)"
            ),
        },
        "self_verification_normalization": (
            "self_verification_score "
            "[-1, 1] mapped linearly "
            "to [0, 1]"
        ),
        "invalid_claim_policy": (
            "qa semantic score = 0.0"
        ),
        "methods": [
            {
                **asdict(
                    result
                ),
                "matched_coverage": [
                    asdict(
                        metric
                    )
                    for metric
                    in result.matched_coverage
                ],
            }
            for result
            in results
        ],
    }

    save_json(
        summary_payload,
        output_directory_path
        / "ablation_summary.json",
    )

    save_summary_csv(
        results,
        output_directory_path
        / "ablation_summary.csv",
    )

    save_matched_coverage_csv(
        results,
        output_directory_path
        / "matched_coverage.csv",
    )

    save_curve_csv(
        curves,
        output_directory_path
        / "risk_coverage_curves.csv",
    )

    plot_risk_coverage_curves(
        curves,
        output_directory_path
        / "risk_coverage_curves.png",
    )

    print_summary(
        results
    )

    print_matched_coverage(
        results
    )

    print(
        "\n"
        + "=" * 88
    )

    print(
        "OUTPUT FILES"
    )

    print(
        "=" * 88
    )

    for output_name in (
        "ablation_summary.json",
        "ablation_summary.csv",
        "matched_coverage.csv",
        "risk_coverage_curves.csv",
        "risk_coverage_curves.png",
    ):
        print(
            output_directory_path
            / output_name
        )

    return results


def parse_arguments() -> argparse.Namespace:
    """
    Parse score-ranking ablation evaluation settings.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate confidence and verifier "
            "ranking signals using risk-coverage, "
            "AURC, and matched-coverage analysis."
        )
    )

    parser.add_argument(
        "--input",
        default=str(
            DEFAULT_INPUT_PATH
        ),
        help=(
            "Input JSONL predictions."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIRECTORY
        ),
        help=(
            "Directory for evaluation outputs."
        ),
    )

    parser.add_argument(
        "--coverage-levels",
        type=parse_coverage_levels,
        default=(
            DEFAULT_COVERAGE_LEVELS
        ),
        help=(
            "Comma-separated matched coverage "
            "levels, e.g. 0.1,0.2,0.5,1.0."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = (
        parse_arguments()
    )

    evaluate_ablation(
        input_path=(
            arguments.input
        ),
        output_directory=(
            arguments.output_dir
        ),
        coverage_levels=(
            arguments.coverage_levels
        ),
    )