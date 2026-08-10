"""
Checks whether different verification signals agree with each other and labels each prediction as CONSISTENT, MIXED, or CONFLICTING.
"""

import argparse 
from pathlib import Path
from typing import Any

from src.utils.io import load_jsonl, save_jsonl


DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/calibration_with_hybrid_evidence.jsonl"
)


DEFAULT_OUTPUT_PATH = Path(
    "outputs/predictions/calibration_with_consistency.jsonl"
)

# SUPPORTED -> The evidence/context supports the predicted answer.
# Entailment -> The context supports the statement.
POSITIVE_LABELS = {
    "SUPPORTED",
    "ENTAILMENT",
    "ANSWER",
}


# UNSUPPORTED -> The evidence/context does not provide enough support for the predicted answer.
# Contradiction -> The context conflicts with the statement.
NEGATIVE_LABELS = {
    "UNSUPPORTED",
    "CONTRADICTION",
    "ABSTAIN",
}


UNCERTAIN_LABELS = {
    "WEAK",
    "NEUTRAL",
    "VERIFY",
    "UNCERTAIN",
}


# standart any label into a clean uppercase string.
def normalize_label(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip().upper()



# converts different labels into a simple direction score.
def label_direction(label: str) -> int:
    label = normalize_label(label)

    if label in POSITIVE_LABELS:
        return 1

    if label in NEGATIVE_LABELS:
        return -1

    if label in UNCERTAIN_LABELS:
        return 0

    return 0



# collects the available verification results from one prediction and organize them into one consistent dictionary
def collect_verification_signals(
    prediction: dict[str, Any]
) -> dict[str, str]:
    possible_fields = {
        "lexical": (
            "evidence_support",
            "evidence_label"
        ),
        "semantic": (
            "semantic_label"
        ),
        "hybrid": (
            "hybrid_evidence_support"
        ),
        "decision": (
            "final_decision",
            "threshold_decision",
            "decision"
        )
    }


    # Example:
    # signals = {
    #     "lexical": "SUPPORTED",
    #     "semantic": "ENTAILMENT",
    # }
    signals: dict[str, str] = {}


    # signal_name = the category name, like "lexical", "semantic", "hybrid", "decision".
    # field_names = the possible fields for that category, like ("evidence_support", "evidence_label").
    for signal_name, field_names in possible_fields.items():
        for field_name in field_names:
            value = prediction.get(field_name)

            if value is None:
                continue

            normalized = normalize_label(value)

            if normalized:
                signals[signal_name] = normalized
                break

    return signals


# measures how much the different verification signals agree with each other.
def calculate_conssistencey_score(
    signals: dict[str, str]
) -> float:
    # Example:
    # signals = {
    # "lexical": "SUPPORTED",
    # "semantic": "ENTAILMENT",
    # "decision": "ANSWER",
    # }

    # Directions: [1, 1, 1]
    # So: dominant_count = 3, total = 3
    # score = 3 / 3 = 1.0
    # Perfect agreement
    if not signals:
        return 0.0

    directions = [
        label_direction(label)
        for label in signals.values()
    ]

    if len(directions) == 1:
        return 1.0

    positive = sum(direction == 1 for direction in directions)
    negative = sum(direction == -1 for direction in directions)
    neutral = sum(direction == 0 for direction in directions)

    total = len(directions)

    dominant_count = max(
        positive,
        negative,
        neutral
    )


    return float(dominant_count / total)


# classify the overall agreement between the signals as CONSISTENT, MIXED, or CONFLICTING
def classify_consistency(
    signals: dict[str, str],
    consistent_threshold: float = 0.75
) -> str:
    # consistent_threshold is the minimum consistency score required to call the signals CONSISTENT.
    # CONSISTENT means the signals mostly agree with each other.
    if not 0.0 <= consistent_threshold <= 1.0:
        raise ValueError(
            "consistent_threshold must be between 0 and 1."
        )

    # the signals are not clearly conflicting, but they also do not agree strongly enough.
    if not signals:
        return 'MIXED'

    directions = {
        label_direction(label)
        for label in signals.values()
    }

    # CONFLICTING -> the signals directly disagree with each other.
    if 1 in directions and -1 in directions:
        return 'CONFLICTING'

    score = calculate_conssistencey_score(signals)

    if score >= consistent_threshold:
        return 'CONSISTENT'

    return 'MIXED'



# To take one prediction, check how consistent all its verification signals are, and attach the consistency results back to that prediction.
def checck_prediction_consistencey(
    prediction: dict[str, Any],
    consistent_threshold: float
) -> dict[str, Any]:
    signals = collect_verification_signals(prediction)

    consistency_score = calculate_conssistencey_score(
        signals
    )

    consistency_label = classify_consistency(
        signals=signals,
        consistent_threshold=consistent_threshold
    )

    updated_prediction = prediction.copy()
    
    updated_prediction.update(
        {
            "consistency_signals": signals,
            "consistency_score": consistency_score,
            "consistency_label": consistency_label,
            "consistency_checker": (
                "verification_signal_agreement_v1"
            )
        }
    )

    return updated_prediction


# To make sure the prediction list is valid before running the consistency checker.
def validate_predictions(
    predictions: list[dict[str, Any]]
) -> None:
    if not predictions:
        raise ValueError(
            "Prediction list cannot be empty."
        )

    for index, prediction in enumerate(predictions, start=1):
        signals = collect_verification_signals(prediction)

        if not signals:
            raise ValueError(
                f"Prediction {index} contains no "
                "verification signals. "
                f"Available keys: "
                f"{list(prediction.keys())}"
            )


# To run the consistency checker on the whole prediction file, save the enriched predictions, and report how many are CONSISTENT, MIXED, or CONFLICTING.
def run_consistencey_check(
    input_path: str | Path,
    output_path: str | Path,
    consistent_threshold: float
) -> list[dict[str, Any]]:
    predictions = load_jsonl(input_path)
    validate_predictions(predictions)

    checked_predictions: list[dict[str, Any]] = []

    counts = {
        "CONSISTENT": 0,
        "MIXED": 0,
        "CONFLICTING": 0
    }

    for index, prediction in enumerate(predictions, start=1):
        checked_prediction = (
            checck_prediction_consistencey(prediction=prediction, consistent_threshold=consistent_threshold)
        )

        checked_predictions.append(checked_prediction)

        label = checked_prediction[
            'consistency_label'
        ]

        counts[label] += 1

        print(
            f"{index}/{len(predictions)} | "
            f"consistency="
            f"{checked_prediction['consistency_score']:.4f} | "
            f"label={label}"
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    save_jsonl(
        checked_prediction,
        output_path
    )

    print(
        "\nConsistency checking completed."
    )

    print(
        f"CONSISTENT: "
        f"{counts['CONSISTENT']}"
    )

    print(
        f"MIXED: "
        f"{counts['MIXED']}"
    )

    print(
        f"CONFLICTING: "
        f"{counts['CONFLICTING']}"
    )

    print(
        f"Results saved to: {output_path}"
    )

    return checked_predictions


# To define and read command-line arguments for the consistency checker.
def parse_arguments() -> argparse.Namespace:
    parser = argparse.Namespace(
        description=(
            "Measure agreement among "
            "verification signals."
        )
    )

    parser.add_argument(
        '--input',
        default=str(DEFAULT_INPUT_PATH)
    )

    parser.add_argument(
        '--output',
        default=str(DEFAULT_OUTPUT_PATH)
    )

    parser.add_argument(
        "--consistent-threshold",
        type=float,
        default=0.75,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_consistencey_check(
        input_path=args.input,
        output_path=args.output,
        consistent_threshold=(
            args.consistent_threshold
        ),
    )