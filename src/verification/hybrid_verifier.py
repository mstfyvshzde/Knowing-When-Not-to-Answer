"""
Hybrid evidence verifier.

Combines three general signals:

1. Calibrated QA confidence
2. Lexical evidence verification
3. Semantic NLI verification

This module does not use gold labels such as:
    - is_answerable
    - is_correct
    - reference answers

Those fields may only be used later during evaluation.
"""

import argparse
from pathlib import Path
from typing import Any

from src.utils.io import (
    load_jsonl,
    save_jsonl,
)

DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/calibration_with_semantic_evidence.jsonl"
)

DEFAULT_OUTPUT_PATH = Path("outputs/predictions/calibration_with_hybrid_evidence.jsonl")


DEFAULT_CONFIDENCE_WEIGHT = 0.25
DEFAULT_LEXICAL_WEIGHT = 0.25
DEFAULT_SEMANTIC_WEIGHT = 0.50

DEFAULT_SUPPORTED_THRESHOLD = 0.70
DEFAULT_WEAK_THRESHOLD = 0.40

DEFAULT_CONTRADICTION_THRESHOLD = 0.70


def clamp_probability(value: float) -> float:
    """
    Bir değeri 0 ile 1 arasında sınırlar.
    """

    return max(
        0.0,
        min(
            1.0,
            float(value),
        ),
    )


def get_first_numeric_value(
    prediction: dict[str, Any],
    field_names: tuple[str, ...],
    default: float = 0.0,
) -> float:
    """
    Verilen alan isimlerinden bulunan ilk sayısal değeri döndürür.
    """

    for field_name in field_names:
        value = prediction.get(field_name)

        if value is None:
            continue

        try:
            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            continue

    return float(default)


def get_calibrated_confidence(prediction: dict[str, Any]) -> float:
    """
    Prediction kaydındaki calibrated confidence değerini çıkarır.

    Alan isimleri farklı pipeline sürümlerine karşı
    toleranslı olacak şekilde kontrol edilir.
    """

    confidence = get_first_numeric_value(
        prediction=prediction,
        field_names=(
            "calibrated_confidence",
            "confidence_calibrated",
            "calibrated_probability",
            "confidence",
            "raw_confidence",
        ),
        default=0.0,
    )

    return clamp_probability(confidence)


def get_lexical_score(prediction: dict[str, Any]) -> float:
    """
    Lexical evidence verifier tarafından üretilen skoru çıkarır.
    """

    lexical_score = get_first_numeric_value(
        prediction=prediction,
        field_names=(
            "combined_evidence_score",
            "evidence_score",
            "lexical_evidence_score",
        ),
        default=0.0,
    )

    return clamp_probability(lexical_score)


def get_entailment_probability(prediction: dict[str, Any]) -> float:
    """
    Semantic verifier entailment olasılığını çıkarır.
    """

    entailment_probability = get_first_numeric_value(
        prediction=prediction,
        field_names=(
            "entailment_probability",
            "semantic_entailment_probability",
        ),
        default=0.0,
    )

    return clamp_probability(entailment_probability)


def get_contradiction_probability(prediction: dict[str, Any]) -> float:
    """
    Semantic verifier contradiction olasılığını çıkarır.
    """

    contradiction_probability = get_first_numeric_value(
        prediction=prediction,
        field_names=(
            "contradiction_probability",
            "semantic_contradiction_probability",
        ),
        default=0.0,
    )

    return clamp_probability(contradiction_probability)


def validate_weights(
    confidence_weight: float,
    lexical_weight: float,
    semantic_weight: float,
) -> None:
    """
    Hybrid score ağırlıklarını doğrular.
    """

    weights = (
        confidence_weight,
        lexical_weight,
        semantic_weight,
    )

    if any(weight < 0.0 for weight in weights):
        raise ValueError("Hybrid weights cannot be negative.")

    total_weight = sum(weights)

    if total_weight <= 0.0:
        raise ValueError("At least one hybrid weight must be greater than zero.")


def normalize_weights(
    confidence_weight: float,
    lexical_weight: float,
    semantic_weight: float,
) -> tuple[
    float,
    float,
    float,
]:
    """
    Ağırlıkları toplamları 1 olacak şekilde normalize eder.
    """

    validate_weights(
        confidence_weight=confidence_weight,
        lexical_weight=lexical_weight,
        semantic_weight=semantic_weight,
    )

    total_weight = confidence_weight + lexical_weight + semantic_weight

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
    Confidence, lexical ve semantic sinyalleri birleştirir.
    """

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


def classify_hybrid_support(
    hybrid_score: float,
    contradiction_probability: float,
    supported_threshold: float,
    weak_threshold: float,
    contradiction_threshold: float,
) -> str:
    """
    Hybrid score ve contradiction olasılığına göre
    evidence support etiketi üretir.

    Güçlü contradiction, yüksek lexical veya confidence
    skorundan bağımsız olarak desteği düşürür.
    """

    if not (0.0 <= weak_threshold <= supported_threshold <= 1.0):
        raise ValueError(
            "Thresholds must satisfy: 0 <= weak_threshold <= supported_threshold <= 1."
        )

    if not (0.0 <= contradiction_threshold <= 1.0):
        raise ValueError("contradiction_threshold must be between 0 and 1.")

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
    Hybrid verifier için gerekli alanların bulunduğunu kontrol eder.
    """

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
    Tek bir prediction kaydı için hybrid verification yapar.
    """

    calibrated_confidence = get_calibrated_confidence(prediction)

    lexical_score = get_lexical_score(prediction)

    entailment_probability = get_entailment_probability(prediction)

    contradiction_probability = get_contradiction_probability(prediction)

    hybrid_score = calculate_hybrid_score(
        calibrated_confidence=(calibrated_confidence),
        lexical_score=(lexical_score),
        entailment_probability=(entailment_probability),
        confidence_weight=(confidence_weight),
        lexical_weight=(lexical_weight),
        semantic_weight=(semantic_weight),
    )

    hybrid_support = classify_hybrid_support(
        hybrid_score=hybrid_score,
        contradiction_probability=(contradiction_probability),
        supported_threshold=(supported_threshold),
        weak_threshold=(weak_threshold),
        contradiction_threshold=(contradiction_threshold),
    )

    updated_prediction = prediction.copy()

    updated_prediction.update(
        {
            "hybrid_confidence_input": (calibrated_confidence),
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
            "hybrid_verifier": ("confidence_lexical_nli_v1"),
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
    Tüm prediction kayıtlarında hybrid verification çalıştırır.
    """

    predictions = load_jsonl(input_path)

    if not predictions:
        raise ValueError("Prediction list cannot be empty.")

    normalize_weights(
        confidence_weight=confidence_weight,
        lexical_weight=lexical_weight,
        semantic_weight=semantic_weight,
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
            confidence_weight=(confidence_weight),
            lexical_weight=(lexical_weight),
            semantic_weight=(semantic_weight),
            supported_threshold=(supported_threshold),
            weak_threshold=(weak_threshold),
            contradiction_threshold=(contradiction_threshold),
        )

        verified_predictions.append(verified_prediction)

        support_label = verified_prediction["hybrid_evidence_support"]

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
    """
    Command-line argümanlarını okur.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Combine calibrated confidence, lexical evidence and semantic NLI signals."
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
        default=(DEFAULT_CONFIDENCE_WEIGHT),
    )

    parser.add_argument(
        "--lexical-weight",
        type=float,
        default=(DEFAULT_LEXICAL_WEIGHT),
    )

    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=(DEFAULT_SEMANTIC_WEIGHT),
    )

    parser.add_argument(
        "--supported-threshold",
        type=float,
        default=(DEFAULT_SUPPORTED_THRESHOLD),
    )

    parser.add_argument(
        "--weak-threshold",
        type=float,
        default=(DEFAULT_WEAK_THRESHOLD),
    )

    parser.add_argument(
        "--contradiction-threshold",
        type=float,
        default=(DEFAULT_CONTRADICTION_THRESHOLD),
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
        contradiction_threshold=(args.contradiction_threshold),
    )
