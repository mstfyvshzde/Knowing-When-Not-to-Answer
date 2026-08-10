"""
Combines confidence, lexical evidence, and semantic NLI signals into one final support decision.
"""


import argparse
from pathlib import Path
from typing import Any

from src.utils.io import (
    load_jsonl,
    save_jsonl
)

DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/calibration_with_semantic_evidence.jsonl"
)

DEFAULT_OUTPUT_PATH = Path("outputs/predictions/calibration_with_hybrid_evidence.jsonl")


# Question: “What is the capital of France?”
# Context: “Paris is the capital and largest city of France.”
# Predicted answer: Paris
# QA confidence = 0.92 -> the answering model is very sure about Paris.
# Lexical evidence = 1.0 -> the word Paris appears directly in the context.
# Semantic/NLI = 0.98 entailment -> the meaning of the context clearly supports “The answer is Paris.”
DEFAULT_CONFIDENCE_WEIGHT = 0.25
DEFAULT_LEXICAL_WEIGHT = 0.25
DEFAULT_SEMANTIC_WEIGHT = 0.50


# if the hybrid evidence score is 0.70 or higher, the answer is labeled SUPPORTED.
DEFAULT_SUPPORTED_THRESHOLD = 0.70
# if the score is between 0.40 and 0.70, it is labeled WEAK.
# If it is below 0.40, it would usually be treated as UNSUPPORTED.
DEFAULT_WEAK_THRESHOLD = 0.40

# If the NLI model gives contradiction probability ≥ 0.70, the system treats the answer as strongly contradicted by the evidence.
# Contradiction means the evidence says the opposite of the predicted answer.
# Example:
# Question: “Is the Earth flat?”
# Predicted answer: “Yes”
# Evidence: “The Earth is approximately spherical.”
# The evidence conflicts with the answer, so NLI would likely classify it as CONTRADICTION.
DEFAULT_CONTRADICTION_THRESHOLD = 0.70


# force any probability value to stay between 0.0 and 1.0.
def clamp_probability(value: float) -> float:
    return max(
        0.0,
        min(
            1.0,
            float(value)
        )
    )


# To search several possible fields in a prediction and return the first value that can be converted into a number.
# Example:
# field_names = ("calibrated_confidence", "confidence", "score")
# If:
# prediction = {
#     "calibrated_confidence": None,
#     "confidence": "0.82"
# }
# it returns: 0.82
def get_first_numeric_value(
    prediction: dict[str, Any],
    field_names: tuple[str, ...],
    default: float = 0.0
) -> float:
    for field_name in field_names:
        value = prediction.get(field_name)

        if value is None:
            continue

        try:
            return float(value)

        except (
            TypeError,
            ValueError
        ):
            continue

    return float(default)


# To find the best available confidence value from several possible field names, then make sure it stays between 0 and 1
def get_calibrated_confidence(prediction: dict[str, Any]) -> float:
    confidence = get_first_numeric_value(
        prediction=prediction,
        field_names=(
            "calibrated_confidence",
            "confidence_calibrated",
            "calibrated_probability",
            "confidence",
            "raw_confidence"
        ),
        default=0.0
    )

    return clamp_probability(confidence)


# To find the available lexical evidence score and make sure it is a valid probability between 0 and 1.
def get_lexical_score(prediction: dict[str, Any]) -> float:
    lexical_score = get_first_numeric_value(
        prediction=prediction,
        field_names=(
            "combined_evidence_score",
            "evidence_score",
            "lexical_evidence_score",
        ),
        default=0.0
    )

    return clamp_probability(lexical_score)


# To get the semantic/NLI ENTAILMENT probability from the prediction and ensure it stays between 0 and 1.
def get_entailment_probability(prediction: dict[str, Any]) -> float:
    entailment_probability = get_first_numeric_value(
        prediction=prediction,
        field_names=(
            "entailment_probability",
            "semantic_entailment_probability",
        ),
        default=0.0
    )

    return clamp_probability(entailment_probability)


# To get the NLI model’s CONTRADICTION probability and make sure it is between 0 and 1.
def get_contradiction_probability(prediction: dict[str, Any]) -> float:
    contradiction_probability = get_first_numeric_value(
        prediction=prediction,
        field_names=(
            "contradiction_probability",
            "semantic_contradiction_probability",
        ),
        default=0.0
    )

    return clamp_probability(contradiction_probability)


# To make sure the three hybrid weights are valid before using them.
def validate_weights(
    confidence_weight: float,
    lexical_weight: float,
    semantic_weight: float
) -> None:
    weights = (
        confidence_weight,
        lexical_weight,
        semantic_weight
    )

    if any(weight < 0.0 for weight in weights): 
        raise ValueError("Hybrid weights cannot be negative.")

    total_weight = sum(weights)

    if total_weight <= 0.0:
        raise ValueError("At least one hybrid weight must be greater than zero.")


# To convert the three hybrid weights so that together they sum to exactly 1.0.
def normalize_weights(
    confidence_weight: float,
    lexical_weight: float,
    semantic_weight: float
) -> tuple[float, float, float]:
    validate_weights(
        confidence_weight=confidence_weight,
        lexical_weight=lexical_weight,
        semantic_weight=semantic_weight
    )

    total_weight = confidence_weight + lexical_weight + semantic_weight

    return (
        confidence_weight / total_weight,
        lexical_weight / total_weight,
        semantic_weight / total_weight
    )


# To combine the three signals into one final hybrid verification score.
def calculate_hybrid_score(
    calibrated_confidence: float,
    lexical_score: float,
    entailment_probability: float,
    confidence_weight: float,
    lexical_weight: float,
    semantic_weight: float
) -> float:
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


# To convert the final hybrid score into one of three labels: SUPPORTED, WEAK, or UNSUPPORTED.
def classify_hybrid_support(
    hybrid_score: float,
    contradiction_probability: float,
    supported_threshold: float,
    weak_threshold: float,
    contradiction_threshold: float
) -> str:
    if not (0.0 <= weak_threshold <= supported_threshold <= 1.0):
        raise ValueError(
            "Thresholds must satisfy: 0 <= weak_threshold <= supported_threshold <= 1."
        )

    if not (0.0 <= contradiction_threshold <= 1.0):
        raise ValueError("contradiction_threshold must be between 0 and 1.")

    if contradiction_probability >= contradiction_threshold:
        return 'UNSUPPORTED'

    if hybrid_score >= supported_threshold:
        return 'SUPPORTED'

    if hybrid_score >= weak_threshold:
        return 'WEAK'

    return 'UNSUPPORTED'


# To check that one prediction contains all the signal groups required by the hybrid verifier before calculation starts
def validate_prediction(
    prediction: dict[str, Any],
    index: int
) -> None:
    confidence_fields = (
        "calibrated_confidence",
        "confidence_calibrated",
        "calibrated_probability",
        "confidence",
        "raw_confidence",
    )

    lexical_fields = (
        "combined_evidence_score",
        "evidence_score",
        "lexical_evidence_score",
    )

    semantic_fields = (
        "entailment_probability",
        "semantic_entailment_probability",
    )

    contradiction_fields = (
        "contradiction_probability",
        "semantic_contradiction_probability",
    )

    missing_groups: list[str] = []

    if not any(field in prediction for field in confidence_fields):
        missing_groups.append("calibrated confidence")

    if not any(field in prediction for field in lexical_fields):
        missing_groups.append("lexical evidence score")

    if not any(field in prediction for field in semantic_fields):
        missing_groups.append("entailment probability")

    if not any(field in prediction for field in contradiction_fields):
        missing_groups.append("contradiction probability")

    if missing_groups:
        raise ValueError(
            f"Prediction {index} is missing: "
            f"{missing_groups}. "
            f"Available keys: "
            f"{list(prediction.keys())}"
        )


# to process one prediction through the complete hybrid verification logic and return an updated prediction with the hybrid score, support label, inputs, weights, and thresholds.
def verify_prediction(
    prediction: dict[str, Any],
    confidence_weight: float,
    lexical_weight: float,
    semantic_weight: float,
    supported_threshold: float,
    weak_threshold: float,
    contradiction_threshold: float
) -> dict[str, Any]:
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
        contradiction_threshold=contradiction_threshold
    )

    updated_prediction = prediction.copy()

    updated_prediction.update(
        {
            "hybrid_confidence_input": calibrated_confidence,
            "hybrid_lexical_input": (lexical_score),
            "hybrid_entailment_input": (entailment_probability),
            "hybrid_contradiction_input": (contradiction_probability),
            "hybrid_evidence_score": (hybrid_score),
            "hybrid_evidence_support": (hybrid_support),
            "hybrid_confidence_weight": (confidence_weight),
            "hybrid_lexical_weight": (lexical_weight),
            "hybrid_semantic_weight": (semantic_weight),
            "hybrid_supported_threshold": (supported_threshold),
            "hybrid_weak_threshold": (weak_threshold),
            "hybrid_contradiction_threshold": (contradiction_threshold),
            "hybrid_verifier": ("confidence_lexical_nli_v1")
        }
    )

    return updated_prediction



# runs the complete hybrid verifier on all predictions, classifies each as SUPPORTED, WEAK, or UNSUPPORTED, saves the enriched results, and prints summary counts.
def run_hybrid_verification(
    input_path: str | Path,
    output_path: str | Path,
    confidence_weight: float,
    lexical_weight: float,
    semantic_weight: float,
    supported_threshold: float,
    weak_threshold: float,
    contradiction_threshold: float
) -> list[dict[str, Any]]:
    predictions = load_jsonl(input_path)

    if not predictions:
        raise ValueError("Prediction list cannot be empty.")

    normalize_weights(
        confidence_weight=confidence_weight,
        lexical_weight=lexical_weight,
        semantic_weight=semantic_weight
    )

    verified_predictions: list[dict[str, Any]] = []

    support_counts = {
        "SUPPORTED": 0,
        "WEAK": 0,
        "UNSUPPORTED": 0,
    }

    for index, prediction in enumerate(
        predictions, start=1
    ):
        validate_prediction(
            prediction=prediction,
            index=index
        )

        verified_prediction = verify_prediction(
            prediction=prediction,
            confidence_weight=confidence_weight,
            lexical_weight=lexical_weight,
            semantic_weight=semantic_weight,
            supported_threshold=supported_threshold,
            weak_threshold=weak_threshold,
            contradiction_threshold=contradiction_threshold
        )

        verified_predictions.append(verified_prediction)

        support_label = verified_prediction['hybrid_evidence_support']
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
        exist_ok=True
    )

    save_jsonl(
        verified_predictions,
        output_path
    )

    print("\nHybrid verification completed.")

    print(f"SUPPORTED: {support_counts['SUPPORTED']}")

    print(f"WEAK: {support_counts['WEAK']}")

    print(f"UNSUPPORTED: {support_counts['UNSUPPORTED']}")

    print(f"Results saved to: {output_path}")

    return verified_predictions



def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine calibrated confidence, lexical evidence and semantic NLI signals."
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
        '--confidence-weight',
        type=float,
        default=DEFAULT_CONFIDENCE_WEIGHT
    )

    parser.add_argument(
        "--lexical-weight",
        type=float,
        default=DEFAULT_LEXICAL_WEIGHT,
    )

    parser.add_argument(
        '--semantic-weight',
        type=float,
        default=DEFAULT_SEMANTIC_WEIGHT
    )

    parser.add_argument(
        '--supported-threshold',
        type=float,
        default=DEFAULT_SUPPORTED_THRESHOLD
    )


    parser.add_argument(
        "--weak-threshold",
        type=float,
        default=(DEFAULT_WEAK_THRESHOLD)
    )

    parser.add_argument(
        "--contradiction-threshold",
        type=float,
        default=(DEFAULT_CONTRADICTION_THRESHOLD)
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_hybrid_verification(
        input_path=args.input,
        output_path=args.output,
        confidence_weight=(args.confidence_weight),
        lexical_weight=(args.lexical_weight),
        semantic_weight=(args.semantic_weight),
        supported_threshold=(args.supported_threshold),
        weak_threshold=(args.weak_threshold),
        contradiction_threshold=(args.contradiction_threshold)
    )
