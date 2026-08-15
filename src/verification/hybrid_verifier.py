"""
Combine calibrated confidence, lexical evidence, and semantic NLI signals.

This prototype hybrid verifier combines three signals for each QA prediction:

1. calibrated QA confidence,
2. lexical context-evidence score,
3. semantic NLI entailment probability.

The default weighted score is:

    hybrid_score =
        0.25 * calibrated_confidence
        + 0.25 * lexical_score
        + 0.50 * entailment_probability

The weights are normalized before use, so custom non-negative weights do not
need to sum to exactly 1.0.

A sufficiently high NLI contradiction probability overrides the weighted score
and labels the prediction UNSUPPORTED.

Otherwise, the hybrid score is mapped to:

- SUPPORTED
- WEAK
- UNSUPPORTED

This module belongs to the earlier prototype verification pipeline. Its inputs
are different diagnostic signals, but they should not automatically be treated
as statistically independent sources of evidence.
"""

import argparse
import math
from pathlib import Path
from typing import Any

from src.utils.io import load_jsonl, save_jsonl

DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/calibration_with_semantic_evidence.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "outputs/predictions/calibration_with_hybrid_evidence.jsonl"
)

DEFAULT_CONFIDENCE_WEIGHT = 0.25
DEFAULT_LEXICAL_WEIGHT = 0.25
DEFAULT_SEMANTIC_WEIGHT = 0.50

DEFAULT_SUPPORTED_THRESHOLD = 0.70
DEFAULT_WEAK_THRESHOLD = 0.40
DEFAULT_CONTRADICTION_THRESHOLD = 0.70


CALIBRATED_CONFIDENCE_FIELDS = (
    "calibrated_confidence",
    "confidence_calibrated",
    "calibrated_probability",
)

LEXICAL_SCORE_FIELDS = (
    "combined_evidence_score",
    "evidence_score",
    "lexical_evidence_score",
)

ENTAILMENT_FIELDS = (
    "entailment_probability",
    "semantic_entailment_probability",
)

CONTRADICTION_FIELDS = (
    "contradiction_probability",
    "semantic_contradiction_probability",
)


def clamp_probability(value: float) -> float:
    """
    Clamp a calculated probability-like value to the interval [0, 1].

    This is used for derived weighted scores where tiny floating-point
    deviations could otherwise produce values slightly outside the interval.
    """

    return max(0.0, min(1.0, float(value)))


def validate_probability(
    value: float,
    signal_name: str,
) -> float:
    """
    Validate that a stored signal can be interpreted as a probability.

    Input signals must be finite and remain inside [0, 1]. Invalid experiment
    data is rejected rather than silently modified.
    """

    numeric_value = float(value)

    if not math.isfinite(numeric_value):
        raise ValueError(
            f"{signal_name} must be finite. Received: {numeric_value}"
        )

    if not 0.0 <= numeric_value <= 1.0:
        raise ValueError(
            f"{signal_name} must be between 0 and 1. "
            f"Received: {numeric_value}"
        )

    return numeric_value


def get_first_numeric_value(
    prediction: dict[str, Any],
    field_names: tuple[str, ...],
) -> float:
    """
    Return the first usable numeric value from a group of field aliases.

    Historical pipeline stages may store the same signal under slightly
    different field names. Aliases are therefore checked from left to right.

    Missing fields are skipped, but malformed numeric values are not silently
    replaced with zero.
    """

    observed_fields: list[str] = []
    invalid_fields: list[str] = []

    for field_name in field_names:
        if field_name not in prediction:
            continue

        value = prediction[field_name]

        if value is None:
            continue

        observed_fields.append(field_name)

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            invalid_fields.append(field_name)
            continue

        if not math.isfinite(numeric_value):
            invalid_fields.append(field_name)
            continue

        return numeric_value

    if observed_fields:
        raise ValueError(
            "No usable numeric value found in fields "
            f"{observed_fields}. Invalid fields: {invalid_fields}"
        )

    raise ValueError(
        "None of the expected fields were found: "
        f"{list(field_names)}"
    )


def get_calibrated_confidence(
    prediction: dict[str, Any],
) -> float:
    """
    Retrieve a temperature-calibrated QA confidence value.

    Explicit calibrated-confidence fields are preferred.

    The generic `confidence` field is accepted only when the prediction
    explicitly records `confidence_is_calibrated=True`. This prevents the
    hybrid verifier from accidentally using the earlier uncalibrated sigmoid
    confidence.
    """

    has_explicit_calibrated_field = any(
        prediction.get(field_name) is not None
        for field_name in CALIBRATED_CONFIDENCE_FIELDS
    )

    if has_explicit_calibrated_field:
        confidence = get_first_numeric_value(
            prediction=prediction,
            field_names=CALIBRATED_CONFIDENCE_FIELDS,
        )

        return validate_probability(
            confidence,
            "calibrated confidence",
        )

    if (
        prediction.get("confidence_is_calibrated", False)
        and prediction.get("confidence") is not None
    ):
        confidence = get_first_numeric_value(
            prediction=prediction,
            field_names=("confidence",),
        )

        return validate_probability(
            confidence,
            "calibrated confidence",
        )

    raise ValueError(
        "No calibrated confidence was found. "
        "Run temperature scaling before hybrid verification."
    )


def get_lexical_score(
    prediction: dict[str, Any],
) -> float:
    """Return the available lexical evidence score."""

    lexical_score = get_first_numeric_value(
        prediction=prediction,
        field_names=LEXICAL_SCORE_FIELDS,
    )

    return validate_probability(
        lexical_score,
        "lexical evidence score",
    )


def get_entailment_probability(
    prediction: dict[str, Any],
) -> float:
    """Return the semantic NLI entailment probability."""

    entailment_probability = get_first_numeric_value(
        prediction=prediction,
        field_names=ENTAILMENT_FIELDS,
    )

    return validate_probability(
        entailment_probability,
        "entailment probability",
    )


def get_contradiction_probability(
    prediction: dict[str, Any],
) -> float:
    """Return the semantic NLI contradiction probability."""

    contradiction_probability = get_first_numeric_value(
        prediction=prediction,
        field_names=CONTRADICTION_FIELDS,
    )

    return validate_probability(
        contradiction_probability,
        "contradiction probability",
    )


def validate_weights(
    confidence_weight: float,
    lexical_weight: float,
    semantic_weight: float,
) -> None:
    """
    Validate the weights used by the hybrid score.

    Weights must be finite and non-negative, and at least one signal must have
    positive weight.
    """

    weights = (
        confidence_weight,
        lexical_weight,
        semantic_weight,
    )

    if any(not math.isfinite(weight) for weight in weights):
        raise ValueError("Hybrid weights must be finite.")

    if any(weight < 0.0 for weight in weights):
        raise ValueError("Hybrid weights cannot be negative.")

    if sum(weights) <= 0.0:
        raise ValueError(
            "At least one hybrid weight must be greater than zero."
        )


def normalize_weights(
    confidence_weight: float,
    lexical_weight: float,
    semantic_weight: float,
) -> tuple[float, float, float]:
    """
    Normalize hybrid weights so their total contribution equals 1.0.

    For example, weights (1, 1, 2) become (0.25, 0.25, 0.50) while preserving
    their relative importance.
    """

    validate_weights(
        confidence_weight=confidence_weight,
        lexical_weight=lexical_weight,
        semantic_weight=semantic_weight,
    )

    total_weight = (
        confidence_weight
        + lexical_weight
        + semantic_weight
    )

    return (
        confidence_weight / total_weight,
        lexical_weight / total_weight,
        semantic_weight / total_weight,
    )


def calculate_hybrid_score(
    calibrated_confidence: float,
    lexical_score: float,
    entailment_probability: float,
    confidence_weight: float,
    lexical_weight: float,
    semantic_weight: float,
) -> float:
    """
    Combine confidence, lexical evidence, and entailment into one score.

    Each input is interpreted on the [0, 1] scale. Weights are normalized before
    multiplication so only their relative sizes affect the final score.
    """

    calibrated_confidence = validate_probability(
        calibrated_confidence,
        "calibrated confidence",
    )

    lexical_score = validate_probability(
        lexical_score,
        "lexical evidence score",
    )

    entailment_probability = validate_probability(
        entailment_probability,
        "entailment probability",
    )

    (
        normalized_confidence_weight,
        normalized_lexical_weight,
        normalized_semantic_weight,
    ) = normalize_weights(
        confidence_weight=confidence_weight,
        lexical_weight=lexical_weight,
        semantic_weight=semantic_weight,
    )

    hybrid_score = (
        normalized_confidence_weight * calibrated_confidence
        + normalized_lexical_weight * lexical_score
        + normalized_semantic_weight * entailment_probability
    )

    return clamp_probability(hybrid_score)


def validate_thresholds(
    supported_threshold: float,
    weak_threshold: float,
    contradiction_threshold: float,
) -> None:
    """Validate support and contradiction thresholds."""

    thresholds = (
        supported_threshold,
        weak_threshold,
        contradiction_threshold,
    )

    if any(not math.isfinite(value) for value in thresholds):
        raise ValueError("Hybrid thresholds must be finite.")

    if not (
        0.0
        <= weak_threshold
        <= supported_threshold
        <= 1.0
    ):
        raise ValueError(
            "Thresholds must satisfy: "
            "0 <= weak_threshold <= supported_threshold <= 1."
        )

    if not 0.0 <= contradiction_threshold <= 1.0:
        raise ValueError(
            "contradiction_threshold must be between 0 and 1."
        )


def classify_hybrid_support(
    hybrid_score: float,
    contradiction_probability: float,
    supported_threshold: float,
    weak_threshold: float,
    contradiction_threshold: float,
) -> str:
    """
    Convert hybrid evidence into SUPPORTED, WEAK, or UNSUPPORTED.

    Strong contradiction has priority over the weighted support score.

    If contradiction does not cross its threshold, the final label is determined
    by the hybrid-score thresholds.
    """

    validate_thresholds(
        supported_threshold=supported_threshold,
        weak_threshold=weak_threshold,
        contradiction_threshold=contradiction_threshold,
    )

    hybrid_score = validate_probability(
        hybrid_score,
        "hybrid evidence score",
    )

    contradiction_probability = validate_probability(
        contradiction_probability,
        "contradiction probability",
    )

    # Strong NLI contradiction overrides positive support from other signals.
    if contradiction_probability >= contradiction_threshold:
        return "UNSUPPORTED"

    if hybrid_score >= supported_threshold:
        return "SUPPORTED"

    if hybrid_score >= weak_threshold:
        return "WEAK"

    return "UNSUPPORTED"


def validate_prediction(
    prediction: dict[str, Any],
    index: int,
) -> None:
    """
    Ensure one prediction contains usable inputs for hybrid verification.

    Validation checks the values themselves rather than only checking whether
    expected field names exist.
    """

    validators = (
        ("calibrated confidence", get_calibrated_confidence),
        ("lexical evidence", get_lexical_score),
        ("entailment probability", get_entailment_probability),
        ("contradiction probability", get_contradiction_probability),
    )

    for signal_name, getter in validators:
        try:
            getter(prediction)
        except ValueError as error:
            raise ValueError(
                f"Prediction {index} has invalid {signal_name}: {error}"
            ) from error


def verify_prediction(
    prediction: dict[str, Any],
    confidence_weight: float,
    lexical_weight: float,
    semantic_weight: float,
    supported_threshold: float,
    weak_threshold: float,
    contradiction_threshold: float,
) -> dict[str, Any]:
    """
    Apply the complete hybrid verification rule to one prediction.

    The original record is preserved and enriched with the three input signals,
    combined hybrid score, support label, configured weights, thresholds, and
    verifier identifier.
    """

    calibrated_confidence = get_calibrated_confidence(prediction)
    lexical_score = get_lexical_score(prediction)
    entailment_probability = get_entailment_probability(prediction)
    contradiction_probability = get_contradiction_probability(prediction)

    hybrid_score = calculate_hybrid_score(
        calibrated_confidence=calibrated_confidence,
        lexical_score=lexical_score,
        entailment_probability=entailment_probability,
        confidence_weight=confidence_weight,
        lexical_weight=lexical_weight,
        semantic_weight=semantic_weight,
    )

    hybrid_support = classify_hybrid_support(
        hybrid_score=hybrid_score,
        contradiction_probability=contradiction_probability,
        supported_threshold=supported_threshold,
        weak_threshold=weak_threshold,
        contradiction_threshold=contradiction_threshold,
    )

    updated_prediction = prediction.copy()

    updated_prediction.update(
        {
            "hybrid_confidence_input": calibrated_confidence,
            "hybrid_lexical_input": lexical_score,
            "hybrid_entailment_input": entailment_probability,
            "hybrid_contradiction_input": contradiction_probability,
            "hybrid_evidence_score": hybrid_score,
            "hybrid_evidence_support": hybrid_support,
            "hybrid_confidence_weight": confidence_weight,
            "hybrid_lexical_weight": lexical_weight,
            "hybrid_semantic_weight": semantic_weight,
            "hybrid_supported_threshold": supported_threshold,
            "hybrid_weak_threshold": weak_threshold,
            "hybrid_contradiction_threshold": contradiction_threshold,
            "hybrid_verifier": "confidence_lexical_nli_v1",
        }
    )

    return updated_prediction


def run_hybrid_verification(
    input_path: str | Path,
    output_path: str | Path,
    confidence_weight: float,
    lexical_weight: float,
    semantic_weight: float,
    supported_threshold: float,
    weak_threshold: float,
    contradiction_threshold: float,
) -> list[dict[str, Any]]:
    """
    Run the prototype hybrid verifier over a complete prediction file.

    Every prediction is validated, enriched with hybrid evidence, saved to disk,
    and counted by its resulting support label.
    """

    predictions = load_jsonl(input_path)

    if not predictions:
        raise ValueError("Prediction list cannot be empty.")

    validate_weights(
        confidence_weight=confidence_weight,
        lexical_weight=lexical_weight,
        semantic_weight=semantic_weight,
    )

    validate_thresholds(
        supported_threshold=supported_threshold,
        weak_threshold=weak_threshold,
        contradiction_threshold=contradiction_threshold,
    )

    verified_predictions: list[dict[str, Any]] = []

    support_counts = {
        "SUPPORTED": 0,
        "WEAK": 0,
        "UNSUPPORTED": 0,
    }

    for index, prediction in enumerate(
        predictions,
        start=1,
    ):
        validate_prediction(
            prediction=prediction,
            index=index,
        )

        verified_prediction = verify_prediction(
            prediction=prediction,
            confidence_weight=confidence_weight,
            lexical_weight=lexical_weight,
            semantic_weight=semantic_weight,
            supported_threshold=supported_threshold,
            weak_threshold=weak_threshold,
            contradiction_threshold=contradiction_threshold,
        )

        verified_predictions.append(
            verified_prediction
        )

        support_label = verified_prediction[
            "hybrid_evidence_support"
        ]

        support_counts[support_label] += 1

        print(
            f"{index}/{len(predictions)} | "
            f"confidence="
            f"{verified_prediction['hybrid_confidence_input']:.4f} | "
            f"lexical="
            f"{verified_prediction['hybrid_lexical_input']:.4f} | "
            f"entailment="
            f"{verified_prediction['hybrid_entailment_input']:.4f} | "
            f"contradiction="
            f"{verified_prediction['hybrid_contradiction_input']:.4f} | "
            f"hybrid="
            f"{verified_prediction['hybrid_evidence_score']:.4f} | "
            f"support={support_label}"
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_jsonl(
        verified_predictions,
        output_path,
    )

    print("\nHybrid verification completed.")
    print(f"SUPPORTED: {support_counts['SUPPORTED']}")
    print(f"WEAK: {support_counts['WEAK']}")
    print(f"UNSUPPORTED: {support_counts['UNSUPPORTED']}")
    print(f"Results saved to: {output_path}")

    return verified_predictions


def parse_arguments() -> argparse.Namespace:
    """Parse hybrid-verification paths, weights, and thresholds."""

    parser = argparse.ArgumentParser(
        description=(
            "Combine calibrated confidence, lexical evidence, "
            "and semantic NLI signals."
        )
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
        "--confidence-weight",
        type=float,
        default=DEFAULT_CONFIDENCE_WEIGHT,
    )

    parser.add_argument(
        "--lexical-weight",
        type=float,
        default=DEFAULT_LEXICAL_WEIGHT,
    )

    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=DEFAULT_SEMANTIC_WEIGHT,
    )

    parser.add_argument(
        "--supported-threshold",
        type=float,
        default=DEFAULT_SUPPORTED_THRESHOLD,
    )

    parser.add_argument(
        "--weak-threshold",
        type=float,
        default=DEFAULT_WEAK_THRESHOLD,
    )

    parser.add_argument(
        "--contradiction-threshold",
        type=float,
        default=DEFAULT_CONTRADICTION_THRESHOLD,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_hybrid_verification(
        input_path=args.input,
        output_path=args.output,
        confidence_weight=args.confidence_weight,
        lexical_weight=args.lexical_weight,
        semantic_weight=args.semantic_weight,
        supported_threshold=args.supported_threshold,
        weak_threshold=args.weak_threshold,
        contradiction_threshold=args.contradiction_threshold,
    )