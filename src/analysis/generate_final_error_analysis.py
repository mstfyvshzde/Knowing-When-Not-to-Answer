"""
Generate the canonical final verifier error-analysis summary.

This analysis uses the same forced-answer correctness definition as the final
selective-QA evaluation:

- answerable examples are correct when the normalized prediction exactly
  matches at least one normalized reference answer;
- unanswerable forced-answer candidates are incorrect.

Correctness is imported from
`src.calibration.calibration_metrics.is_prediction_correct` so this diagnostic
cannot silently drift away from the project's canonical evaluation semantics.

The analysis is descriptive only. It does not tune thresholds, ranking rules,
fusion weights, prompts, or any other experimental parameter.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.calibration.calibration_metrics import is_prediction_correct
from src.utils.io import load_jsonl, save_json

DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/"
    "test_with_question_aware_v2_and_self_verification.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "outputs/analysis/"
    "final_error_analysis/"
    "summary.json"
)

HIGH_ENTAILMENT_THRESHOLD = 0.8
LOW_ENTAILMENT_THRESHOLD = 0.2


def require_field(
    record: dict[str, Any],
    field_name: str,
    record_index: int,
) -> Any:
    """Return a required field or fail with a precise record-level error."""

    if field_name not in record:
        raise ValueError(
            f"Missing required field {field_name!r} "
            f"at record {record_index}."
        )

    return record[field_name]


def require_probability(
    record: dict[str, Any],
    field_name: str,
    record_index: int,
) -> float:
    """Return a finite probability in the closed interval [0, 1]."""

    value = require_field(
        record,
        field_name,
        record_index,
    )

    try:
        probability = float(value)

    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field_name!r} must be numeric at record "
            f"{record_index}; received {value!r}."
        ) from error

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"{field_name!r} must lie in [0, 1] at record "
            f"{record_index}; received {probability!r}."
        )

    return probability


def normalize_label(
    value: Any,
    field_name: str,
    record_index: int,
) -> str:
    """Normalize one required categorical label."""

    label = str(value).strip().upper()

    if not label:
        raise ValueError(
            f"{field_name!r} is empty at record {record_index}."
        )

    return label


def normalize_validation_reasons(
    value: Any,
    record_index: int,
) -> list[str]:
    """
    Normalize claim-validation reasons.

    Validation reasons are not mutually exclusive, so every listed reason is
    counted independently.
    """

    if value is None:
        return []

    if isinstance(value, str):
        reason = value.strip().upper()

        return [reason] if reason else []

    if isinstance(
        value,
        (list, tuple, set),
    ):
        reasons: list[str] = []

        for item in value:
            reason = str(item).strip().upper()

            if reason:
                reasons.append(reason)

        return reasons

    raise ValueError(
        "qa_claim_validation_reasons must be a string, sequence, "
        f"or null at record {record_index}; received {type(value).__name__}."
    )


def update_group(
    groups: dict[str, Counter[str]],
    group_name: str,
    correct: bool,
) -> None:
    """Update count/correct/incorrect totals for one diagnostic group."""

    groups[group_name]["count"] += 1

    if correct:
        groups[group_name]["correct"] += 1

    else:
        groups[group_name]["incorrect"] += 1


def finalize_groups(
    groups: dict[str, Counter[str]],
) -> dict[str, dict[str, int | float]]:
    """Convert grouped counters into JSON-serializable metric dictionaries."""

    summary: dict[
        str,
        dict[str, int | float],
    ] = {}

    for group_name in sorted(groups):
        counts = groups[group_name]

        count = int(
            counts["count"]
        )

        correct = int(
            counts["correct"]
        )

        incorrect = int(
            counts["incorrect"]
        )

        summary[group_name] = {
            "count": count,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": (
                correct / count
                if count
                else 0.0
            ),
        }

    return summary


def generate_summary(
    records: list[dict[str, Any]],
    input_path: Path,
) -> dict[str, Any]:
    """Generate the complete canonical final error-analysis summary."""

    if not records:
        raise ValueError(
            "Final prediction file is empty."
        )

    qa_claim_validity: dict[
        str,
        Counter[str],
    ] = defaultdict(Counter)

    qa_nli_labels: dict[
        str,
        Counter[str],
    ] = defaultdict(Counter)

    self_verification_labels: dict[
        str,
        Counter[str],
    ] = defaultdict(Counter)

    invalid_claim_reasons: Counter[str] = Counter()

    correct_predictions = 0

    high_entailment_incorrect = 0
    low_entailment_correct = 0

    for index, record in enumerate(records):
        correct = bool(
            is_prediction_correct(
                record
            )
        )

        if correct:
            correct_predictions += 1

        qa_claim_valid_value = require_field(
            record,
            "qa_claim_valid",
            index,
        )

        if not isinstance(
            qa_claim_valid_value,
            bool,
        ):
            raise TypeError(
                "'qa_claim_valid' must be boolean at record "
                f"{index}; received {qa_claim_valid_value!r}."
            )

        qa_claim_valid_key = str(
            qa_claim_valid_value
        )

        update_group(
            qa_claim_validity,
            qa_claim_valid_key,
            correct,
        )

        qa_nli_label = normalize_label(
            require_field(
                record,
                "qa_nli_label",
                index,
            ),
            "qa_nli_label",
            index,
        )

        update_group(
            qa_nli_labels,
            qa_nli_label,
            correct,
        )

        self_verification_label = normalize_label(
            require_field(
                record,
                "self_verification_label",
                index,
            ),
            "self_verification_label",
            index,
        )

        update_group(
            self_verification_labels,
            self_verification_label,
            correct,
        )

        entailment_probability = require_probability(
            record,
            "qa_entailment_probability",
            index,
        )

        if (
            not correct
            and entailment_probability
            >= HIGH_ENTAILMENT_THRESHOLD
        ):
            high_entailment_incorrect += 1

        if (
            correct
            and entailment_probability
            <= LOW_ENTAILMENT_THRESHOLD
        ):
            low_entailment_correct += 1

        validation_reasons = (
            normalize_validation_reasons(
                record.get(
                    "qa_claim_validation_reasons"
                ),
                index,
            )
        )

        if not qa_claim_valid_value:
            invalid_claim_reasons.update(
                validation_reasons
            )

    total_records = len(records)

    incorrect_predictions = (
        total_records
        - correct_predictions
    )

    return {
        "input_path": str(input_path),
        "analysis_type": (
            "canonical_final_verifier_error_analysis"
        ),
        "correctness_definition": (
            "src.calibration.calibration_metrics."
            "is_prediction_correct"
        ),
        "total_records": total_records,
        "correct_predictions": (
            correct_predictions
        ),
        "incorrect_predictions": (
            incorrect_predictions
        ),
        "full_accuracy": (
            correct_predictions
            / total_records
        ),
        "qa_claim_validity": (
            finalize_groups(
                qa_claim_validity
            )
        ),
        "qa_nli_label": (
            finalize_groups(
                qa_nli_labels
            )
        ),
        "self_verification_label": (
            finalize_groups(
                self_verification_labels
            )
        ),
        "invalid_claim_reasons": dict(
            sorted(
                invalid_claim_reasons.items()
            )
        ),
        "diagnostic_counts": {
            "high_entailment_incorrect_ge_0.8": (
                high_entailment_incorrect
            ),
            "low_entailment_correct_le_0.2": (
                low_entailment_correct
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate the canonical final verifier "
            "error-analysis summary."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=(
            "Final held-out prediction JSONL."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Destination summary JSON."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run canonical final error analysis."""

    args = parse_args()

    records = load_jsonl(
        args.input
    )

    summary = generate_summary(
        records=records,
        input_path=args.input,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        summary,
        args.output,
    )

    print(
        f"Records: "
        f"{summary['total_records']}"
    )

    print(
        f"Correct: "
        f"{summary['correct_predictions']}"
    )

    print(
        f"Incorrect: "
        f"{summary['incorrect_predictions']}"
    )

    print(
        f"Accuracy: "
        f"{summary['full_accuracy']:.6f}"
    )

    print(
        f"Saved: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()