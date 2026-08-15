"""
Measure agreement among the project's prototype verification signals.

Different components may express support using different labels, such as:

- SUPPORTED / UNSUPPORTED
- ENTAILMENT / CONTRADICTION / NEUTRAL
- ANSWER / VERIFY / ABSTAIN

This module maps those heterogeneous labels into a common direction:

    +1 -> supports answering
     0 -> uncertain or intermediate
    -1 -> supports not answering

It then summarizes whether the available signals are:

- CONSISTENT: mostly point in the same direction
- MIXED: agreement is incomplete or dominated by uncertainty
- CONFLICTING: at least one positive and one negative signal directly disagree

This checker belongs to the earlier prototype verification pipeline and should
be interpreted as a descriptive agreement diagnostic, not as an independent
verification model or a final selective-QA ranking method.
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

# Labels interpreted as evidence in favor of answering.
POSITIVE_LABELS = {
    "SUPPORTED",
    "ENTAILMENT",
    "ANSWER",
}


# Labels interpreted as evidence against answering.
NEGATIVE_LABELS = {
    "UNSUPPORTED",
    "CONTRADICTION",
    "ABSTAIN",
}


# Labels representing uncertainty or an intermediate decision.
UNCERTAIN_LABELS = {
    "WEAK",
    "NEUTRAL",
    "VERIFY",
    "UNCERTAIN",
}


def normalize_label(value: Any) -> str:
    """
    Normalize heterogeneous labels into a comparable uppercase representation.

    Missing values become an empty string so they can be ignored safely when
    collecting available verification signals.
    """

    if value is None:
        return ""

    return str(value).strip().upper()



def label_direction(label: str) -> int:
    """
    Map a verification label to a shared directional representation.

    +1 means evidence favors answering.
    -1 means evidence favors abstaining.
     0 means uncertain, intermediate, or unrecognized.
    """

    label = normalize_label(label)

    if label in POSITIVE_LABELS:
        return 1

    if label in NEGATIVE_LABELS:
        return -1

    if label in UNCERTAIN_LABELS:
        return 0

    return 0



def collect_verification_signals(
    prediction: dict[str, Any],
) -> dict[str, str]:
    """
    Collect the first available label from each verification-signal family.

    Several historical pipeline stages use different field names for similar
    concepts. Each signal family therefore lists its possible field names in
    priority order.

    At most one label is retained for each family so duplicate aliases do not
    count as separate pieces of evidence.
    """

    # Field aliases are searched from left to right. Once one usable field is
    # found for a signal family, later aliases from that family are ignored.
    possible_fields = {
        "lexical": (
            "evidence_support",
            "evidence_label",
        ),
        "semantic": (
            "semantic_label",
        ),
        "hybrid": (
            "hybrid_evidence_support",
        ),
        "decision": (
            "final_decision",
            "threshold_decision",
            "decision",
        ),
    }


    signals: dict[str, str] = {}


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


def calculate_consistency_score(signals: dict[str, str]) -> float:
    """
    Measure how strongly the available signals agree on one direction.

    Every label is mapped to +1, 0, or -1. The score is then the fraction of
    signals belonging to the most common direction.

    Example:

        directions = [+1, +1, -1]

        dominant direction count = 2
        total signals = 3
        consistency score = 2 / 3

    A score of 1.0 means all available signals share the same direction.
    Lower values indicate greater disagreement.
    """

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



def classify_consistency(
    signals: dict[str, str],
    consistent_threshold: float = 0.75,
) -> str:
    """
    Convert signal agreement into CONSISTENT, MIXED, or CONFLICTING.

    CONFLICTING has priority whenever both a positive (+1) and negative (-1)
    direction are present, because the signals explicitly oppose each other.

    Otherwise, CONSISTENT is assigned when the dominant-direction agreement
    reaches consistent_threshold.

    All remaining cases are labeled MIXED.
    """

    # consistent_threshold is the minimum consistency score required to call the signals CONSISTENT.
    # CONSISTENT means the signals mostly agree with each other.
    if not 0.0 <= consistent_threshold <= 1.0:
        raise ValueError(
            "consistent_threshold must be between 0 and 1."
        )

    # the signals are not clearly conflicting, but they also do not agree strongly enough.
    if not signals:
        return "MIXED"

    directions = {
        label_direction(label)
        for label in signals.values()
    }

    # Explicit positive-vs-negative disagreement takes priority over the numerical
    # agreement score. For example, ANSWER and ABSTAIN together are conflicting
    # even if additional neutral signals are present.
    if 1 in directions and -1 in directions:
        return "CONFLICTING"

    score = calculate_consistency_score(signals)

    if score >= consistent_threshold:
        return "CONSISTENT"

    return "MIXED"



def check_prediction_consistency(
    prediction: dict[str, Any],
    consistent_threshold: float,
) -> dict[str, Any]:
    """
    Measure agreement for one prediction and attach the diagnostic results.

    The original prediction is preserved and enriched with the collected
    signals, numerical agreement score, categorical consistency label, and
    checker version.
    """

    signals = collect_verification_signals(prediction)

    consistency_score = calculate_consistency_score(
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
            "consistency_checker": "verification_signal_agreement_v1"
        }
    )

    return updated_prediction


def validate_predictions(
    predictions: list[dict[str, Any]],
) -> None:
    """
    Ensure every prediction contains at least one usable verification signal.

    Running an agreement checker without any available signal would produce a
    meaningless consistency result, so such records are rejected early.
    """

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


def run_consistency_check(
    input_path: str | Path,
    output_path: str | Path,
    consistent_threshold: float,
) -> list[dict[str, Any]]:
    """
    Run the agreement diagnostic over a prediction file.

    Each prediction is enriched with its consistency score and label, then the
    complete annotated collection is saved for later prototype analysis.
    """

    predictions = load_jsonl(input_path)
    validate_predictions(predictions)

    checked_predictions: list[dict[str, Any]] = []

    counts = {
        "CONSISTENT": 0,
        "MIXED": 0,
        "CONFLICTING": 0
    }

    for index, prediction in enumerate(predictions, start=1):
        checked_prediction = check_prediction_consistency(
            prediction=prediction,
            consistent_threshold=consistent_threshold,
        )

        checked_predictions.append(checked_prediction)

        label = checked_prediction["consistency_label"]

        counts[label] += 1

        print(
            f"{index}/{len(predictions)} | "
            f"consistency="
            f"{checked_prediction['consistency_score']:.4f} | "
            f"label={label}"
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_jsonl(checked_predictions, output_path)

    print("\nConsistency checking completed.")
    print(f"CONSISTENT: {counts['CONSISTENT']}")
    print(f"MIXED: {counts['MIXED']}")
    print(f"CONFLICTING: {counts['CONFLICTING']}")
    print(f"Results saved to: {output_path}")

    return checked_predictions


def parse_arguments() -> argparse.Namespace:
    """Parse consistency-checker paths and agreement threshold."""

    parser = argparse.ArgumentParser(
        description="Measure agreement among prototype verification signals."
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
    )

    parser.add_argument(
        "--consistent-threshold",
        type=float,
        default=0.75,
        help="Minimum dominant-direction agreement required for CONSISTENT.",
    )

    return parser.parse_args()



if __name__ == "__main__":
    args = parse_arguments()

    run_consistency_check(
        input_path=args.input,
        output_path=args.output,
        consistent_threshold=args.consistent_threshold,
    )