"""Diagnose question-aware semantic verifier V2 errors without tuning."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import string
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

INPUT = Path(
    "outputs/predictions/calibration_with_question_aware_semantic_evidence_v2.jsonl"
)

OUT = Path("outputs/analysis/question_aware_verifier")


CONF = (
    "calibrated_confidence",
    "temperature_scaled_confidence",
    "qa_calibrated_confidence",
    "confidence",
    "qa_confidence",
    "prediction_confidence",
    "max_probability",
    "probability",
)

ENT = (
    "qa_entailment_probability",
    "question_aware_entailment_probability",
    "question_aware_semantic_score",
    "qa_semantic_score",
)

LABEL = (
    "qa_nli_label",
    "question_aware_nli_label",
    "qa_verification_label",
    "semantic_label",
    "nli_label",
)

VALID = (
    "qa_claim_valid",
    "claim_valid",
    "question_aware_claim_valid",
)

REASONS = (
    "qa_invalid_claim_reasons",
    "invalid_claim_reasons",
    "claim_invalid_reasons",
    "qa_claim_invalid_reasons",
)

QUESTION = (
    "question",
    "query",
    "input_question",
)

PRED = (
    "predicted_answer",
    "prediction_text",
    "prediction_answer",
    "prediction",
    "model_answer",
    "generated_answer",
    "answer",
)

REF = (
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

CORRECT = (
    "is_correct",
    "correct",
    "exact_match",
    "em",
    "prediction_correct",
    "is_exact_match",
    "exact_match_score",
    "em_score",
    "answer_correct",
    "prediction_is_correct",
    "qa_is_correct",
)

CLAIM = (
    "qa_claim",
    "generated_claim",
    "claim",
    "question_aware_claim",
)

EVIDENCE = (
    "hybrid_evidence",
    "evidence",
    "retrieved_evidence",
    "evidence_text",
    "context",
    "passage",
    "supporting_evidence",
)

CONTRA = (
    "qa_contradiction_probability",
    "question_aware_contradiction_probability",
)

NEUTRAL = (
    "qa_neutral_probability",
    "question_aware_neutral_probability",
)

BINS = tuple((index / 10, (index + 1) / 10) for index in range(10))


def first(
    record: dict[str, Any],
    names: Sequence[str],
    default: Any = None,
) -> Any:
    """Return the first non-None field value."""

    for name in names:
        if name in record and record[name] is not None:
            return record[name]

    return default


def search_nested_field(
    value: Any,
    candidate_fields: Sequence[str],
) -> Any:
    """
    Recursively find the first matching field.

    This function is schema-flexible and does not rely on a
    particular QA dataset structure.
    """

    if isinstance(value, dict):
        # Prefer fields at the current level.
        for field_name in candidate_fields:
            if field_name in value:
                return value[field_name]

        # Then inspect nested objects.
        for nested_value in value.values():
            result = search_nested_field(
                nested_value,
                candidate_fields,
            )

            if result is not None:
                return result

    elif isinstance(value, (list, tuple)):
        for item in value:
            if not isinstance(item, (dict, list, tuple)):
                continue

            result = search_nested_field(
                item,
                candidate_fields,
            )

            if result is not None:
                return result

    return None


def first_or_nested(
    record: dict[str, Any],
    names: Sequence[str],
    default: Any = None,
) -> Any:
    """Search top-level fields first, then nested structures."""

    direct_value = first(
        record,
        names,
        default=None,
    )

    if direct_value is not None:
        return direct_value

    nested_value = search_nested_field(
        record,
        names,
    )

    if nested_value is not None:
        return nested_value

    return default


def fnum(value: Any) -> float | None:
    """Convert a value to a finite float."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def fbool(value: Any) -> bool | None:
    """Convert common boolean representations."""

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        if value == 1:
            return True

        if value == 0:
            return False

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {
            "true",
            "yes",
            "correct",
            "1",
            "1.0",
            "valid",
        }:
            return True

        if normalized in {
            "false",
            "no",
            "incorrect",
            "0",
            "0.0",
            "invalid",
        }:
            return False

    return None


def clamp(value: float) -> float:
    """Restrict a probability to [0, 1]."""

    return max(
        0.0,
        min(1.0, value),
    )


def clean(value: Any) -> str:
    """Normalize whitespace."""

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def norm(value: Any) -> str:
    """SQuAD-style simplified exact-match normalization."""

    text = clean(value).lower()

    text = "".join(
        character for character in text if character not in string.punctuation
    )

    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text,
    )

    return " ".join(text.split())


def refs(value: Any) -> list[str]:
    """Convert different reference-answer formats to a flat list."""

    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, dict):
        for key in (
            "text",
            "answer",
            "answers",
            "value",
            "label",
        ):
            if key in value:
                return refs(value[key])

        return []

    if isinstance(value, Iterable):
        output: list[str] = []

        for item in value:
            output.extend(refs(item))

        return output

    return [str(value)]


def extract_reference_answers(
    record: dict[str, Any],
) -> list[str]:
    """
    Extract gold answers from top-level or nested structures.

    An explicitly present empty list remains a valid representation
    for an unanswerable question.
    """

    for field_name in REF:
        if field_name in record:
            return refs(record[field_name])

    nested_value = search_nested_field(
        record,
        REF,
    )

    return refs(nested_value)


def infer_correct(
    record: dict[str, Any],
) -> bool:
    """Infer exact-match correctness."""

    for field_name in CORRECT:
        if field_name not in record:
            continue

        boolean_value = fbool(record[field_name])

        if boolean_value is not None:
            return boolean_value

    prediction = first_or_nested(
        record,
        PRED,
        default=None,
    )

    if prediction is None:
        raise ValueError(
            "No prediction or correctness field at "
            f"record {record['_index']}; "
            f"keys={sorted(record)}"
        )

    reference_field_exists = any(field_name in record for field_name in REF)

    if not reference_field_exists:
        nested_reference = search_nested_field(
            record,
            REF,
        )

        reference_field_exists = nested_reference is not None

    if not reference_field_exists:
        raise ValueError(
            "No reference or correctness field at "
            f"record {record['_index']}; "
            f"keys={sorted(record)}"
        )

    normalized_prediction = norm(prediction)

    normalized_references = [
        norm(reference) for reference in extract_reference_answers(record)
    ]

    # Empty references represent an unanswerable example.
    if not normalized_references:
        return normalized_prediction == ""

    return any(
        normalized_prediction == reference for reference in normalized_references
    )


def field(
    records: Sequence[dict[str, Any]],
    names: Sequence[str],
    numeric: bool = False,
) -> str | None:
    """Find the candidate field with the highest coverage."""

    best_field: str | None = None
    best_count = 0

    for name in names:
        if numeric:
            count = sum(fnum(record.get(name)) is not None for record in records)
        else:
            count = sum(
                name in record and record[name] is not None for record in records
            )

        if count > best_count:
            best_field = name
            best_count = count

    return best_field


def reason_list(value: Any) -> list[str]:
    """Normalize invalid-claim reasons."""

    if value is None:
        return []

    if isinstance(value, str):
        stripped = value.strip()

        if not stripped:
            return []

        try:
            decoded = json.loads(stripped)

            if decoded != value:
                return reason_list(decoded)
        except json.JSONDecodeError:
            pass

        return [
            clean(item).upper().replace(" ", "_")
            for item in re.split(r"[,;|]", stripped)
            if clean(item)
        ]

    if isinstance(value, dict):
        return [
            clean(key).upper().replace(" ", "_")
            for key, active in value.items()
            if fbool(active) is True
        ]

    if isinstance(value, Iterable):
        output: list[str] = []

        for item in value:
            output.extend(reason_list(item))

        return output

    return [clean(value).upper().replace(" ", "_")]


def infer_label(
    record: dict[str, Any],
    label_field: str | None,
    valid: bool,
    entailment: float | None,
) -> str:
    """Infer the canonical NLI label."""

    if not valid:
        return "INVALID_CLAIM"

    raw_label = ""

    if label_field is not None:
        raw_label = clean(record.get(label_field, "")).upper()

    for label in (
        "ENTAILMENT",
        "CONTRADICTION",
        "NEUTRAL",
        "EMPTY_ANSWER",
    ):
        if label in raw_label:
            return label

    candidates: list[tuple[str, float]] = []

    if entailment is not None:
        candidates.append(("ENTAILMENT", entailment))

    contradiction = fnum(
        first_or_nested(
            record,
            CONTRA,
            default=None,
        )
    )

    neutral = fnum(
        first_or_nested(
            record,
            NEUTRAL,
            default=None,
        )
    )

    if contradiction is not None:
        candidates.append(("CONTRADICTION", contradiction))

    if neutral is not None:
        candidates.append(("NEUTRAL", neutral))

    if not candidates:
        return "UNKNOWN"

    return max(
        candidates,
        key=lambda item: item[1],
    )[0]


def mean(
    values: Sequence[float],
) -> float | None:
    if not values:
        return None

    return sum(values) / len(values)


def pearson(
    first_values: Sequence[float],
    second_values: Sequence[float],
) -> float | None:
    if len(first_values) < 2:
        return None

    first_mean = mean(first_values)
    second_mean = mean(second_values)

    if first_mean is None or second_mean is None:
        return None

    numerator = sum(
        (first - first_mean) * (second - second_mean)
        for first, second in zip(
            first_values,
            second_values,
        )
    )

    denominator = math.sqrt(
        sum((value - first_mean) ** 2 for value in first_values)
        * sum((value - second_mean) ** 2 for value in second_values)
    )

    if math.isclose(
        denominator,
        0.0,
        abs_tol=1e-15,
    ):
        return None

    return numerator / denominator


def ranks(
    values: Sequence[float],
) -> list[float]:
    """Calculate average ranks, including tied values."""

    ordered = sorted(
        enumerate(values),
        key=lambda item: item[1],
    )

    output = [0.0] * len(values)
    position = 0

    while position < len(ordered):
        end = position + 1

        while end < len(ordered) and math.isclose(
            ordered[end][1],
            ordered[position][1],
            abs_tol=1e-15,
        ):
            end += 1

        average_rank = (position + 1 + end) / 2

        for index in range(
            position,
            end,
        ):
            original_index = ordered[index][0]
            output[original_index] = average_rank

        position = end

    return output


def write_csv(
    rows: Sequence[dict[str, Any]],
    path: Path,
    fields: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if fields is None:
        fields = list(rows[0]) if rows else []

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(fields),
        )

        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(
    rows: Sequence[dict[str, Any]],
    path: Path,
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for row in rows:
            output_file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def compact(
    record: dict[str, Any],
    correct: bool,
    confidence: float | None,
    entailment: float | None,
    valid: bool,
    label: str,
    reasons: Sequence[str],
) -> dict[str, Any]:
    """
    Preserve fields required for qualitative error analysis.

    source_record retains the complete original input record so that
    no dataset or pipeline fields are lost.
    """

    return {
        "index": record["_index"],
        "correct": correct,
        "confidence": confidence,
        "entailment_probability": entailment,
        "claim_valid": valid,
        "nli_label": label,
        "invalid_claim_reasons": list(reasons),
        "question": first_or_nested(
            record,
            QUESTION,
            default=None,
        ),
        "predicted_answer": first_or_nested(
            record,
            PRED,
            default=None,
        ),
        "reference_answers": (extract_reference_answers(record)),
        "claim": first_or_nested(
            record,
            CLAIM,
            default=None,
        ),
        "evidence": first_or_nested(
            record,
            EVIDENCE,
            default=None,
        ),
        "contradiction_probability": fnum(
            first_or_nested(
                record,
                CONTRA,
                default=None,
            )
        ),
        "neutral_probability": fnum(
            first_or_nested(
                record,
                NEUTRAL,
                default=None,
            )
        ),
        "source_record": {
            key: value for key, value in record.items() if key != "_index"
        },
    }


def run(
    input_path: str | Path,
    output_dir: str | Path,
    high_ent: float,
    low_ent: float,
    high_conf: float,
    low_conf: float,
    max_examples: int,
) -> None:
    records: list[dict[str, Any]] = []

    with Path(input_path).open(
        encoding="utf-8",
    ) as input_file:
        for line_number, line in enumerate(
            input_file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise TypeError(f"Line {line_number} must contain a JSON object.")

            record["_index"] = len(records)
            records.append(record)

    if not records:
        raise ValueError("Input JSONL is empty.")

    confidence_field = field(
        records,
        CONF,
        numeric=True,
    )

    entailment_field = field(
        records,
        ENT,
        numeric=True,
    )

    label_field = field(
        records,
        LABEL,
    )

    validity_field = field(
        records,
        VALID,
    )

    if not confidence_field:
        raise ValueError(f"No confidence field found. Checked: {CONF}")

    if not entailment_field:
        raise ValueError(f"No entailment field found. Checked: {ENT}")

    data: list[dict[str, Any]] = []

    for record in records:
        correct = infer_correct(record)

        confidence_value = fnum(record.get(confidence_field))

        entailment_value = fnum(record.get(entailment_field))

        confidence = clamp(confidence_value) if confidence_value is not None else None

        entailment = clamp(entailment_value) if entailment_value is not None else None

        if validity_field:
            valid = fbool(record.get(validity_field)) is True
        else:
            raw_label = clean(
                first_or_nested(
                    record,
                    LABEL,
                    default="",
                )
            ).upper()

            valid = "INVALID_CLAIM" not in raw_label

        nli_label = infer_label(
            record,
            label_field,
            valid,
            entailment,
        )

        invalid_reasons = reason_list(
            first_or_nested(
                record,
                REASONS,
                default=None,
            )
        )

        data.append(
            compact(
                record,
                correct,
                confidence,
                entailment,
                valid,
                nli_label,
                invalid_reasons,
            )
        )

    output_path = Path(output_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = len(data)

    correct_count = sum(item["correct"] for item in data)

    by_label: dict[
        str,
        Counter[str],
    ] = defaultdict(Counter)

    for item in data:
        correctness_key = "correct" if item["correct"] else "incorrect"

        by_label[item["nli_label"]][correctness_key] += 1

    label_rows: list[dict[str, Any]] = []

    for label in sorted(by_label):
        correct = by_label[label]["correct"]
        incorrect = by_label[label]["incorrect"]
        count = correct + incorrect

        label_rows.append(
            {
                "nli_label": label,
                "count": count,
                "correct": correct,
                "incorrect": incorrect,
                "accuracy": correct / count,
                "error_rate": incorrect / count,
                "share_of_all_records": count / total,
            }
        )

    by_validity: dict[
        str,
        Counter[str],
    ] = defaultdict(Counter)

    for item in data:
        validity_key = "VALID" if item["claim_valid"] else "INVALID"

        correctness_key = "correct" if item["correct"] else "incorrect"

        by_validity[validity_key][correctness_key] += 1

    validity_rows: list[dict[str, Any]] = []

    for key in (
        "VALID",
        "INVALID",
    ):
        correct = by_validity[key]["correct"]
        incorrect = by_validity[key]["incorrect"]
        count = correct + incorrect

        validity_rows.append(
            {
                "claim_validity": key,
                "count": count,
                "correct": correct,
                "incorrect": incorrect,
                "accuracy": (correct / count if count else None),
                "error_rate": (incorrect / count if count else None),
                "share_of_all_records": (count / total),
            }
        )

    bin_counts: dict[
        tuple[float, float],
        Counter[str],
    ] = defaultdict(Counter)

    bin_scores: dict[
        tuple[float, float],
        list[float],
    ] = defaultdict(list)

    for item in data:
        score = item["entailment_probability"]

        if score is None:
            continue

        bin_index = min(
            int(score * 10),
            9,
        )

        bin_key = BINS[bin_index]

        correctness_key = "correct" if item["correct"] else "incorrect"

        bin_counts[bin_key][correctness_key] += 1

        bin_scores[bin_key].append(score)

    bin_rows: list[dict[str, Any]] = []

    for lower, upper in BINS:
        correct = bin_counts[(lower, upper)]["correct"]

        incorrect = bin_counts[(lower, upper)]["incorrect"]

        count = correct + incorrect

        closing_bracket = "]" if upper == 1 else ")"

        bin_rows.append(
            {
                "score_bin": (f"[{lower:.1f}, {upper:.1f}{closing_bracket}"),
                "lower_bound": lower,
                "upper_bound": upper,
                "count": count,
                "correct": correct,
                "incorrect": incorrect,
                "accuracy": (correct / count if count else None),
                "error_rate": (incorrect / count if count else None),
                "mean_entailment_score": mean(bin_scores[(lower, upper)]),
            }
        )

    reason_counts: Counter[str] = Counter()
    reason_correct: Counter[str] = Counter()
    reason_incorrect: Counter[str] = Counter()

    for item in data:
        if item["claim_valid"]:
            continue

        reasons = item["invalid_claim_reasons"] or ["UNSPECIFIED"]

        for reason in reasons:
            reason_counts[reason] += 1

            if item["correct"]:
                reason_correct[reason] += 1
            else:
                reason_incorrect[reason] += 1

    reason_rows = [
        {
            "reason": reason,
            "count": count,
            "correct": reason_correct[reason],
            "incorrect": reason_incorrect[reason],
            "accuracy": (reason_correct[reason] / count),
        }
        for reason, count in reason_counts.most_common()
    ]

    paired = [
        (
            item["confidence"],
            item["entailment_probability"],
        )
        for item in data
        if item["confidence"] is not None and item["entailment_probability"] is not None
    ]

    first_scores = [first_score for first_score, _ in paired]

    second_scores = [second_score for _, second_score in paired]

    high_entailment_incorrect = sorted(
        [
            item
            for item in data
            if not item["correct"]
            and item["entailment_probability"] is not None
            and item["entailment_probability"] >= high_ent
        ],
        key=lambda item: (
            -item["entailment_probability"],
            -(item["confidence"] or 0),
            item["index"],
        ),
    )[:max_examples]

    low_entailment_correct = sorted(
        [
            item
            for item in data
            if item["correct"]
            and item["entailment_probability"] is not None
            and item["entailment_probability"] <= low_ent
        ],
        key=lambda item: (
            item["entailment_probability"],
            -(item["confidence"] or 0),
            item["index"],
        ),
    )[:max_examples]

    confidence_high_semantic_low = sorted(
        [
            item
            for item in data
            if item["confidence"] is not None
            and item["entailment_probability"] is not None
            and item["confidence"] >= high_conf
            and item["entailment_probability"] <= low_ent
        ],
        key=lambda item: -(item["confidence"] - item["entailment_probability"]),
    )[:max_examples]

    confidence_low_semantic_high = sorted(
        [
            item
            for item in data
            if item["confidence"] is not None
            and item["entailment_probability"] is not None
            and item["confidence"] <= low_conf
            and item["entailment_probability"] >= high_ent
        ],
        key=lambda item: -(item["entailment_probability"] - item["confidence"]),
    )[:max_examples]

    correct_entailment = [
        item["entailment_probability"]
        for item in data
        if item["correct"] and item["entailment_probability"] is not None
    ]

    incorrect_entailment = [
        item["entailment_probability"]
        for item in data
        if not item["correct"] and item["entailment_probability"] is not None
    ]

    summary = {
        "input_path": str(input_path),
        "total_records": total,
        "correct_predictions": correct_count,
        "incorrect_predictions": (total - correct_count),
        "full_accuracy": (correct_count / total),
        "detected_fields": {
            "confidence": confidence_field,
            "entailment": entailment_field,
            "nli_label": label_field,
            "claim_validity": validity_field,
        },
        "thresholds_used_for_example_extraction_only": {
            "high_entailment": high_ent,
            "low_entailment": low_ent,
            "high_confidence": high_conf,
            "low_confidence": low_conf,
        },
        "claim_validity": {
            "valid_claims": sum(item["claim_valid"] for item in data),
            "invalid_claims": sum(not item["claim_valid"] for item in data),
        },
        "mean_entailment": {
            "correct_predictions": mean(correct_entailment),
            "incorrect_predictions": mean(incorrect_entailment),
        },
        "confidence_entailment_relationship": {
            "paired_records": len(paired),
            "pearson_correlation": pearson(
                first_scores,
                second_scores,
            ),
            "spearman_correlation": (
                pearson(
                    ranks(first_scores),
                    ranks(second_scores),
                )
                if paired
                else None
            ),
        },
        "diagnostic_group_counts": {
            "high_entailment_incorrect": len(high_entailment_incorrect),
            "low_entailment_correct": len(low_entailment_correct),
            "confidence_high_semantic_low": len(confidence_high_semantic_low),
            "confidence_low_semantic_high": len(confidence_low_semantic_high),
        },
        "correctness_by_nli_label": label_rows,
        "correctness_by_claim_validity": validity_rows,
        "entailment_score_bins": bin_rows,
        "invalid_claim_reasons": reason_rows,
    }

    (output_path / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_csv(
        label_rows,
        output_path / "correctness_by_nli_label.csv",
    )

    write_csv(
        validity_rows,
        output_path / "correctness_by_claim_validity.csv",
    )

    write_csv(
        bin_rows,
        output_path / "entailment_score_bins.csv",
    )

    write_csv(
        reason_rows,
        output_path / "invalid_claim_reasons.csv",
        (
            "reason",
            "count",
            "correct",
            "incorrect",
            "accuracy",
        ),
    )

    write_jsonl(
        high_entailment_incorrect,
        output_path / "high_entailment_incorrect.jsonl",
    )

    write_jsonl(
        low_entailment_correct,
        output_path / "low_entailment_correct.jsonl",
    )

    write_jsonl(
        confidence_high_semantic_low,
        output_path / "confidence_high_semantic_low.jsonl",
    )

    write_jsonl(
        confidence_low_semantic_high,
        output_path / "confidence_low_semantic_high.jsonl",
    )

    figure = plt.figure(figsize=(8, 6))

    axis = figure.add_subplot(111)

    for correctness, label in (
        (True, "Correct"),
        (False, "Incorrect"),
    ):
        points = [
            item
            for item in data
            if item["correct"] == correctness
            and item["confidence"] is not None
            and item["entailment_probability"] is not None
        ]

        if points:
            axis.scatter(
                [item["confidence"] for item in points],
                [item["entailment_probability"] for item in points],
                alpha=0.65,
                label=label,
            )

    axis.set(
        xlabel="Calibrated confidence",
        ylabel=("Question-aware entailment probability"),
        title=("Confidence vs Question-Aware Entailment"),
        xlim=(0, 1),
        ylim=(0, 1),
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    axis.legend()
    figure.tight_layout()

    figure.savefig(
        output_path / "confidence_entailment_scatter.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    correct_mean = mean(correct_entailment)

    incorrect_mean = mean(incorrect_entailment)

    pearson_value = summary["confidence_entailment_relationship"]["pearson_correlation"]

    spearman_value = summary["confidence_entailment_relationship"][
        "spearman_correlation"
    ]

    print("\n" + "=" * 92 + "\nQUESTION-AWARE VERIFIER ERROR ANALYSIS\n" + "=" * 92)

    print(
        f"Total: {total} | "
        f"Correct: {correct_count} | "
        f"Incorrect: {total - correct_count} | "
        f"Accuracy: {correct_count / total:.4f}"
    )

    print(
        "Valid claims: "
        f"{summary['claim_validity']['valid_claims']} | "
        "Invalid claims: "
        f"{summary['claim_validity']['invalid_claims']}"
    )

    print(
        "Mean entailment — correct: "
        f"{correct_mean:.4f} | "
        "incorrect: "
        f"{incorrect_mean:.4f}"
    )

    print(f"Pearson: {pearson_value:.4f} | Spearman: {spearman_value:.4f}")

    print("\nCORRECTNESS BY NLI LABEL")

    for row in label_rows:
        print(
            f"{row['nli_label']:<20} "
            f"count={row['count']:<4} "
            f"accuracy={row['accuracy']:.4f}"
        )

    print("\nDIAGNOSTIC GROUPS")

    for key, value in summary["diagnostic_group_counts"].items():
        print(f"{key:<35} {value}")

    print(f"\nOutputs: {output_path}")


def probability_argument(
    value: str,
) -> float:
    number = float(value)

    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("Probability must be between 0 and 1.")

    return number


def positive_integer_argument(
    value: str,
) -> int:
    number = int(value)

    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be positive.")

    return number


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=str(INPUT),
    )

    parser.add_argument(
        "--output-dir",
        default=str(OUT),
    )

    parser.add_argument(
        "--high-entailment",
        type=probability_argument,
        default=0.8,
    )

    parser.add_argument(
        "--low-entailment",
        type=probability_argument,
        default=0.2,
    )

    parser.add_argument(
        "--high-confidence",
        type=probability_argument,
        default=0.8,
    )

    parser.add_argument(
        "--low-confidence",
        type=probability_argument,
        default=0.2,
    )

    parser.add_argument(
        "--max-examples",
        type=positive_integer_argument,
        default=50,
    )

    arguments = parser.parse_args()

    run(
        arguments.input,
        arguments.output_dir,
        arguments.high_entailment,
        arguments.low_entailment,
        arguments.high_confidence,
        arguments.low_confidence,
        arguments.max_examples,
    )


if __name__ == "__main__":
    main()
