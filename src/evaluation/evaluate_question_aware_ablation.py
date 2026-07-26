"""
Question-aware verifier ablation evaluation.

This script compares selective QA ranking methods without tuning
thresholds or learning combination weights.

Compared methods
----------------
1. Confidence only
2. Lexical verifier only, if available
3. Old semantic verifier only, if available
4. Question-aware semantic verifier V2
5. Confidence + question-aware semantic V2

Evaluation
----------
- Risk-coverage curve
- Empirical AURC
- Matched-coverage risk
- Selective accuracy
- Wrong answered
- Correct abstained
- Incorrect abstained

Important
---------
The combined score uses a pre-declared equal-weight geometric mean:

    sqrt(confidence * qa_entailment)

No weights or thresholds are optimized on the evaluation data.

Invalid question-aware claims receive semantic score 0.0. This ensures
they are ranked below valid evidence without treating them as factual
contradictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import string
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/calibration_with_question_aware_semantic_evidence_v2.jsonl"
)

DEFAULT_OUTPUT_DIRECTORY = Path("outputs/evaluation/question_aware_ablation")

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


CONFIDENCE_FIELDS = (
    "calibrated_confidence",
    "temperature_scaled_confidence",
    "qa_calibrated_confidence",
    "confidence",
    "qa_confidence",
    "prediction_confidence",
    "max_probability",
    "probability",
)

LEXICAL_SCORE_FIELDS = (
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

QUESTION_AWARE_SCORE_FIELDS = ("qa_entailment_probability",)

QUESTION_AWARE_VALIDITY_FIELDS = ("qa_claim_valid",)

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

CORRECTNESS_FIELDS = (
    "is_correct",
    "correct",
    "exact_match",
    "em",
    "prediction_correct",
)


@dataclass(frozen=True)
class SelectiveMetrics:
    """Metrics at one coverage level."""

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
    """Complete evaluation result for one ranking method."""

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
    """Load JSON objects from a JSONL file."""

    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    records: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            stripped_line = line.strip()

            if not stripped_line:
                continue

            try:
                record = json.loads(stripped_line)

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise TypeError(f"Line {line_number} must contain a JSON object.")

            records.append(record)

    if not records:
        raise ValueError(f"No records were found in {input_path}")

    return records


def save_json(
    data: Any,
    path: str | Path,
) -> None:
    """Save a JSON-compatible object."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            data,
            output_file,
            indent=2,
            ensure_ascii=False,
        )


def get_first_value(
    record: dict[str, Any],
    field_names: Sequence[str],
    default: Any = None,
) -> Any:
    """Return the first present, non-None field value."""

    for field_name in field_names:
        if field_name in record and record[field_name] is not None:
            return record[field_name]

    return default


def clean_text(
    text: Any,
) -> str:
    """Normalize whitespace."""

    return re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip()


def normalize_answer(
    answer: Any,
) -> str:
    """
    Apply a SQuAD-style answer normalization.

    This fallback is used only when no explicit correctness field exists.
    """

    text = clean_text(answer).lower()

    text = "".join(
        character for character in text if character not in string.punctuation
    )

    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text,
    )

    return " ".join(text.split())


def coerce_boolean(
    value: Any,
) -> bool | None:
    """Convert common correctness representations into a boolean."""

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        if value == 1:
            return True

        if value == 0:
            return False

    if isinstance(value, float):
        if math.isclose(value, 1.0):
            return True

        if math.isclose(value, 0.0):
            return False

    if isinstance(value, str):
        normalized_value = value.strip().lower()

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
    """Safely convert a value to a finite float."""

    if value is None:
        return None

    if isinstance(value, bool):
        return float(value)

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
    """Clamp a score to the probability interval."""

    return max(
        0.0,
        min(1.0, value),
    )


def extract_references(
    record: dict[str, Any],
) -> list[str]:
    """Extract one or more reference answers."""

    raw_value = get_first_value(
        record,
        REFERENCE_FIELDS,
        default=None,
    )

    if raw_value is None:
        return []

    if isinstance(raw_value, str):
        return [raw_value]

    if isinstance(raw_value, dict):
        for possible_field in (
            "text",
            "answers",
            "answer",
        ):
            if possible_field in raw_value:
                return extract_reference_value(raw_value[possible_field])

        return []

    return extract_reference_value(raw_value)


def extract_reference_value(
    value: Any,
) -> list[str]:
    """Normalize a nested reference-answer value."""

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
                return extract_reference_value(value[possible_field])

        return []

    if isinstance(value, Iterable):
        references: list[str] = []

        for item in value:
            references.extend(extract_reference_value(item))

        return references

    return [str(value)]


def infer_correctness(
    record: dict[str, Any],
) -> bool:
    """
    Infer whether the prediction is correct.

    Priority
    --------
    1. Explicit correctness field
    2. Exact match against reference answers
    3. Empty-reference handling for unanswerable questions
    """

    expanded_correctness_fields = (
        *CORRECTNESS_FIELDS,
        "is_exact_match",
        "exact_match_score",
        "em_score",
        "answer_correct",
        "prediction_is_correct",
        "qa_is_correct",
    )

    # 1. Prefer an explicit correctness field.
    for field_name in expanded_correctness_fields:
        if field_name not in record:
            continue

        correctness = coerce_boolean(record[field_name])

        if correctness is not None:
            return correctness

    prediction_value = get_first_value(
        record,
        PREDICTION_FIELDS,
        default=None,
    )

    if prediction_value is None:
        raise ValueError(
            "Could not infer correctness because no prediction "
            f"field was found. Available keys: {sorted(record.keys())}"
        )

    normalized_prediction = normalize_answer(prediction_value)

    # Determine whether a reference field genuinely exists.
    reference_field_found = False
    raw_reference_value: Any = None

    for field_name in REFERENCE_FIELDS:
        if field_name in record:
            reference_field_found = True
            raw_reference_value = record[field_name]
            break

    if not reference_field_found:
        raise ValueError(
            "Could not infer correctness because no reference-answer "
            f"field was found. Available keys: {sorted(record.keys())}"
        )

    references = extract_reference_value(raw_reference_value)

    normalized_references = [normalize_answer(reference) for reference in references]

    # Empty list represents an unanswerable example.
    if not normalized_references:
        return normalized_prediction == ""

    return any(
        normalized_prediction == reference for reference in normalized_references
    )


def find_available_field(
    records: Sequence[dict[str, Any]],
    candidates: Sequence[str],
) -> str | None:
    """
    Find the candidate field available in the greatest number of records.

    A field must have at least one numeric value to be selected.
    """

    best_field: str | None = None
    best_count = 0

    for field_name in candidates:
        numeric_count = sum(
            coerce_float(record.get(field_name)) is not None for record in records
        )

        if numeric_count > best_count:
            best_field = field_name
            best_count = numeric_count

    return best_field


def extract_numeric_score(
    record: dict[str, Any],
    field_name: str,
) -> float | None:
    """Extract and clamp one numeric method score."""

    value = coerce_float(record.get(field_name))

    if value is None:
        return None

    return clamp_probability(value)


def extract_question_aware_score(
    record: dict[str, Any],
    score_field: str,
) -> float:
    """
    Extract the question-aware semantic score.

    Invalid claims receive 0.0 rather than being removed. This preserves
    the complete evaluation population and ranks unreliable verifier
    outputs at the bottom.
    """

    claim_valid = get_first_value(
        record,
        QUESTION_AWARE_VALIDITY_FIELDS,
        default=False,
    )

    claim_valid_boolean = coerce_boolean(claim_valid)

    if claim_valid_boolean is not True:
        return 0.0

    score = extract_numeric_score(
        record,
        score_field,
    )

    if score is None:
        return 0.0

    return score


def geometric_mean_score(
    first_score: float,
    second_score: float,
) -> float:
    """
    Equal-weight geometric mean.

    This combination has no fitted weights:

        sqrt(first_score * second_score)
    """

    return math.sqrt(max(0.0, first_score) * max(0.0, second_score))


def create_ranked_indices(
    scores: Sequence[float],
) -> list[int]:
    """
    Rank records by descending score.

    Original record index is used as a deterministic tie-breaker.
    """

    return sorted(
        range(len(scores)),
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
    """Calculate selective metrics for the top-k ranked records."""

    total = len(correctness)

    if total == 0:
        raise ValueError("Cannot evaluate an empty correctness sequence.")

    answer_count = max(
        1,
        min(answer_count, total),
    )

    answered_indices = set(ranked_indices[:answer_count])

    correct_answered = sum(correctness[index] for index in answered_indices)

    wrong_answered = answer_count - correct_answered

    total_correct = sum(correctness)
    total_incorrect = total - total_correct

    correct_abstained = total_correct - correct_answered

    incorrect_abstained = total_incorrect - wrong_answered

    selective_accuracy = correct_answered / answer_count

    risk = wrong_answered / answer_count

    return SelectiveMetrics(
        requested_coverage=requested_coverage,
        answered=answer_count,
        total=total,
        actual_coverage=answer_count / total,
        selective_accuracy=selective_accuracy,
        risk=risk,
        wrong_answered=wrong_answered,
        correct_answered=correct_answered,
        abstained=total - answer_count,
        correct_abstained=correct_abstained,
        incorrect_abstained=incorrect_abstained,
    )


def build_risk_coverage_curve(
    correctness: Sequence[bool],
    scores: Sequence[float],
) -> list[dict[str, float | int]]:
    """
    Construct the empirical risk-coverage curve.

    One point is produced for every answer count from 1 to N.
    """

    if len(correctness) != len(scores):
        raise ValueError("Correctness and score lengths must match.")

    ranked_indices = create_ranked_indices(scores)

    curve: list[dict[str, float | int]] = []

    correct_answered = 0

    for answer_count, record_index in enumerate(
        ranked_indices,
        start=1,
    ):
        correct_answered += int(correctness[record_index])

        wrong_answered = answer_count - correct_answered

        curve.append(
            {
                "answered": answer_count,
                "coverage": (answer_count / len(correctness)),
                "selective_accuracy": (correct_answered / answer_count),
                "risk": (wrong_answered / answer_count),
                "wrong_answered": wrong_answered,
            }
        )

    return curve


def calculate_aurc(
    curve: Sequence[dict[str, float | int]],
) -> float:
    """
    Calculate empirical AURC.

    For equally spaced coverage points k/N, empirical AURC is the
    arithmetic mean of selective risk across k = 1 ... N.
    """

    if not curve:
        raise ValueError("Cannot calculate AURC from an empty curve.")

    return sum(float(point["risk"]) for point in curve) / len(curve)


def calculate_optimal_aurc(
    correctness: Sequence[bool],
) -> float:
    """
    Calculate the oracle AURC for the same correctness distribution.

    The oracle ranks all correct predictions before all incorrect ones.
    """

    oracle_scores = [1.0 if is_correct else 0.0 for is_correct in correctness]

    oracle_curve = build_risk_coverage_curve(
        correctness,
        oracle_scores,
    )

    return calculate_aurc(oracle_curve)


def calculate_random_aurc(
    correctness: Sequence[bool],
) -> float:
    """
    Expected random-ranking AURC.

    Under random ranking, expected selective risk equals full error rate.
    """

    return 1.0 - sum(correctness) / len(correctness)


def calculate_normalized_aurc(
    aurc: float,
    optimal_aurc: float,
    random_aurc: float,
) -> float | None:
    """
    Normalize AURC between oracle and expected random ranking.

    0.0 = oracle ranking
    1.0 = expected random ranking
    Values above 1.0 are worse than random.
    """

    denominator = random_aurc - optimal_aurc

    if math.isclose(
        denominator,
        0.0,
        abs_tol=1e-12,
    ):
        return None

    return (aurc - optimal_aurc) / denominator


def evaluate_method(
    method_name: str,
    score_field_description: str,
    correctness: Sequence[bool],
    scores: Sequence[float],
    coverage_levels: Sequence[float],
) -> tuple[
    MethodResult,
    list[dict[str, float | int]],
]:
    """Evaluate one selective-ranking method."""

    if len(correctness) != len(scores):
        raise ValueError(f"Length mismatch for method {method_name}.")

    total = len(correctness)
    ranked_indices = create_ranked_indices(scores)

    matched_metrics: list[SelectiveMetrics] = []

    for requested_coverage in coverage_levels:
        answer_count = max(
            1,
            math.ceil(requested_coverage * total),
        )

        matched_metrics.append(
            metrics_at_answer_count(
                correctness=correctness,
                ranked_indices=ranked_indices,
                answer_count=answer_count,
                requested_coverage=requested_coverage,
            )
        )

    curve = build_risk_coverage_curve(
        correctness,
        scores,
    )

    aurc = calculate_aurc(curve)

    optimal_aurc = calculate_optimal_aurc(correctness)

    random_aurc = calculate_random_aurc(correctness)

    normalized_aurc = calculate_normalized_aurc(
        aurc=aurc,
        optimal_aurc=optimal_aurc,
        random_aurc=random_aurc,
    )

    result = MethodResult(
        method=method_name,
        score_field=score_field_description,
        available_records=sum(math.isfinite(score) for score in scores),
        total_records=total,
        full_accuracy=sum(correctness) / total,
        aurc=aurc,
        normalized_aurc=normalized_aurc,
        matched_coverage=matched_metrics,
    )

    return result, curve


def save_matched_coverage_csv(
    results: Sequence[MethodResult],
    path: str | Path,
) -> None:
    """Save matched-coverage results."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
            fieldnames=field_names,
        )

        writer.writeheader()

        for result in results:
            for metric in result.matched_coverage:
                writer.writerow(
                    {
                        "method": result.method,
                        **asdict(metric),
                    }
                )


def save_summary_csv(
    results: Sequence[MethodResult],
    path: str | Path,
) -> None:
    """Save one summary row per method."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "method": result.method,
                    "score_field": result.score_field,
                    "total_records": result.total_records,
                    "full_accuracy": result.full_accuracy,
                    "aurc": result.aurc,
                    "normalized_aurc": result.normalized_aurc,
                }
            )


def save_curve_csv(
    curves: dict[
        str,
        list[dict[str, float | int]],
    ],
    path: str | Path,
) -> None:
    """Save all risk-coverage curves in long CSV format."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        )

        writer.writeheader()

        for method_name, curve in curves.items():
            for point in curve:
                writer.writerow(
                    {
                        "method": method_name,
                        **point,
                    }
                )


def plot_risk_coverage_curves(
    curves: dict[
        str,
        list[dict[str, float | int]],
    ],
    path: str | Path,
) -> None:
    """Plot risk against coverage."""

    figure = plt.figure(figsize=(9, 6))

    axis = figure.add_subplot(111)

    for method_name, curve in curves.items():
        axis.plot(
            [float(point["coverage"]) for point in curve],
            [float(point["risk"]) for point in curve],
            label=method_name,
        )

    axis.set_xlabel("Coverage")
    axis.set_ylabel("Selective risk")
    axis.set_title("Question-Aware Verification Ablation")

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(bottom=0.0)
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def print_summary(
    results: Sequence[MethodResult],
) -> None:
    """Print the main ablation summary."""

    print("\n" + "=" * 88)

    print("QUESTION-AWARE ABLATION SUMMARY")

    print("=" * 88)

    print(f"{'Method':<42}{'AURC':>12}{'Norm. AURC':>16}{'Full Acc.':>14}")

    print("-" * 88)

    for result in sorted(
        results,
        key=lambda item: item.aurc,
    ):
        normalized_text = (
            f"{result.normalized_aurc:.6f}"
            if result.normalized_aurc is not None
            else "N/A"
        )

        print(
            f"{result.method:<42}"
            f"{result.aurc:>12.6f}"
            f"{normalized_text:>16}"
            f"{result.full_accuracy:>14.4f}"
        )

    print("\nLower AURC is better.")


def print_matched_coverage(
    results: Sequence[MethodResult],
) -> None:
    """Print matched-coverage comparisons."""

    coverage_levels = [
        metric.requested_coverage for metric in results[0].matched_coverage
    ]

    print("\n" + "=" * 88)

    print("MATCHED-COVERAGE RISK")

    print("=" * 88)

    for coverage in coverage_levels:
        print(f"\nCoverage target: {coverage:.0%}")

        print(
            f"{'Method':<42}{'Risk':>10}{'Accuracy':>12}{'Wrong':>10}{'Answered':>12}"
        )

        print("-" * 88)

        rows: list[tuple[str, SelectiveMetrics]] = []

        for result in results:
            metric = next(
                item
                for item in result.matched_coverage
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

        for method_name, metric in sorted(
            rows,
            key=lambda item: item[1].risk,
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
) -> tuple[float, ...]:
    """Parse comma-separated coverage values."""

    values: list[float] = []

    for item in raw_value.split(","):
        stripped_item = item.strip()

        if not stripped_item:
            continue

        value = float(stripped_item)

        if not 0.0 < value <= 1.0:
            raise argparse.ArgumentTypeError("Coverage values must be in (0, 1].")

        values.append(value)

    if not values:
        raise argparse.ArgumentTypeError("At least one coverage value is required.")

    return tuple(sorted(set(values)))


def evaluate_ablation(
    input_path: str | Path,
    output_directory: str | Path,
    coverage_levels: Sequence[float],
) -> list[MethodResult]:
    """Run all available ablation methods."""

    records = load_jsonl(input_path)

    correctness = [infer_correctness(record) for record in records]

    confidence_field = find_available_field(
        records,
        CONFIDENCE_FIELDS,
    )

    lexical_field = find_available_field(
        records,
        LEXICAL_SCORE_FIELDS,
    )

    old_semantic_field = find_available_field(
        records,
        OLD_SEMANTIC_SCORE_FIELDS,
    )

    question_aware_field = find_available_field(
        records,
        QUESTION_AWARE_SCORE_FIELDS,
    )

    if confidence_field is None:
        raise ValueError(
            f"No confidence score field was found. Checked: {CONFIDENCE_FIELDS}"
        )

    if question_aware_field is None:
        raise ValueError(
            "No question-aware entailment field was found. "
            f"Checked: {QUESTION_AWARE_SCORE_FIELDS}"
        )

    confidence_scores = [
        extract_numeric_score(
            record,
            confidence_field,
        )
        for record in records
    ]

    if any(score is None for score in confidence_scores):
        missing_count = sum(score is None for score in confidence_scores)

        raise ValueError(
            f"Confidence field '{confidence_field}' is missing "
            f"or non-numeric in {missing_count} records."
        )

    confidence_scores_clean = [
        float(score) for score in confidence_scores if score is not None
    ]

    question_aware_scores = [
        extract_question_aware_score(
            record,
            question_aware_field,
        )
        for record in records
    ]

    combined_scores = [
        geometric_mean_score(
            confidence_score,
            semantic_score,
        )
        for confidence_score, semantic_score in zip(
            confidence_scores_clean,
            question_aware_scores,
        )
    ]

    methods: list[
        tuple[
            str,
            str,
            list[float],
        ]
    ] = [
        (
            "Confidence only",
            confidence_field,
            confidence_scores_clean,
        ),
        (
            "Question-aware semantic V2",
            (f"{question_aware_field}; invalid claims=0"),
            question_aware_scores,
        ),
        (
            "Confidence + question-aware semantic V2",
            (f"sqrt({confidence_field} * {question_aware_field}); invalid claims=0"),
            combined_scores,
        ),
    ]

    if lexical_field is not None:
        lexical_scores_optional = [
            extract_numeric_score(
                record,
                lexical_field,
            )
            for record in records
        ]

        if all(score is not None for score in lexical_scores_optional):
            methods.insert(
                1,
                (
                    "Lexical verifier only",
                    lexical_field,
                    [
                        float(score)
                        for score in lexical_scores_optional
                        if score is not None
                    ],
                ),
            )

        else:
            print(f"Skipping lexical verifier: field '{lexical_field}' is incomplete.")

    if old_semantic_field is not None:
        if old_semantic_field == question_aware_field:
            print(
                "Skipping old semantic verifier because its detected "
                "field is identical to the question-aware field."
            )

        else:
            old_semantic_scores_optional = [
                extract_numeric_score(
                    record,
                    old_semantic_field,
                )
                for record in records
            ]

            if all(score is not None for score in old_semantic_scores_optional):
                insertion_index = 2 if lexical_field is not None else 1

                methods.insert(
                    insertion_index,
                    (
                        "Old semantic verifier only",
                        old_semantic_field,
                        [
                            float(score)
                            for score in old_semantic_scores_optional
                            if score is not None
                        ],
                    ),
                )

            else:
                print(
                    f"Skipping old semantic verifier: field "
                    f"'{old_semantic_field}' is incomplete."
                )

    results: list[MethodResult] = []
    curves: dict[
        str,
        list[dict[str, float | int]],
    ] = {}

    for (
        method_name,
        field_description,
        scores,
    ) in methods:
        result, curve = evaluate_method(
            method_name=method_name,
            score_field_description=field_description,
            correctness=correctness,
            scores=scores,
            coverage_levels=coverage_levels,
        )

        results.append(result)
        curves[method_name] = curve

    output_directory_path = Path(output_directory)

    output_directory_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_payload = {
        "input_path": str(input_path),
        "total_records": len(records),
        "correct_predictions": sum(correctness),
        "incorrect_predictions": (len(correctness) - sum(correctness)),
        "full_accuracy": (sum(correctness) / len(correctness)),
        "detected_fields": {
            "confidence": confidence_field,
            "lexical": lexical_field,
            "old_semantic": old_semantic_field,
            "question_aware_semantic": (question_aware_field),
        },
        "combination_rule": (
            "equal-weight geometric mean: sqrt(confidence * qa_entailment)"
        ),
        "invalid_claim_policy": ("qa semantic score = 0.0"),
        "methods": [
            {
                **asdict(result),
                "matched_coverage": [
                    asdict(metric) for metric in result.matched_coverage
                ],
            }
            for result in results
        ],
    }

    save_json(
        summary_payload,
        output_directory_path / "ablation_summary.json",
    )

    save_summary_csv(
        results,
        output_directory_path / "ablation_summary.csv",
    )

    save_matched_coverage_csv(
        results,
        output_directory_path / "matched_coverage.csv",
    )

    save_curve_csv(
        curves,
        output_directory_path / "risk_coverage_curves.csv",
    )

    plot_risk_coverage_curves(
        curves,
        output_directory_path / "risk_coverage_curves.png",
    )

    print_summary(results)
    print_matched_coverage(results)

    print("\n" + "=" * 88)

    print("OUTPUT FILES")

    print("=" * 88)

    print(output_directory_path / "ablation_summary.json")

    print(output_directory_path / "ablation_summary.csv")

    print(output_directory_path / "matched_coverage.csv")

    print(output_directory_path / "risk_coverage_curves.csv")

    print(output_directory_path / "risk_coverage_curves.png")

    return results


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate confidence, lexical evidence, old semantic "
            "evidence, and question-aware semantic evidence using "
            "risk-coverage and matched-coverage analysis."
        )
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Input JSONL predictions.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIRECTORY),
        help="Directory for evaluation outputs.",
    )

    parser.add_argument(
        "--coverage-levels",
        type=parse_coverage_levels,
        default=DEFAULT_COVERAGE_LEVELS,
        help=(
            "Comma-separated matched coverage levels. Example: 0.1,0.2,0.3,0.4,0.5,1.0"
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    evaluate_ablation(
        input_path=arguments.input,
        output_directory=arguments.output_dir,
        coverage_levels=arguments.coverage_levels,
    )
