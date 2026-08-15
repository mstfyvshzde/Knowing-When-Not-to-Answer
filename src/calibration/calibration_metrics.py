"""
Evaluate how well QA confidence scores reflect actual prediction correctness.

Calibration (kalibrasyon) asks a simple question:
when the system reports a confidence such as 0.80, are predictions with similar
confidence actually correct about 80% of the time?

This module compares confidence values with forced-answer correctness and
reports several calibration metrics:

- ECE: average confidence-vs-accuracy mismatch across confidence bins
- MCE: largest mismatch observed in any confidence bin
- Brier score: squared error between confidence and binary correctness
- NLL: probabilistic penalty for confident wrong predictions
- confidence-accuracy gap: difference between overall mean confidence and accuracy

In the forced-answer setup, unanswerable examples are treated as incorrect
because the model is required to return an answer and cannot make a correct
abstention.
"""

import argparse
import re
import string
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.io import load_jsonl, save_json


def normalize_answer(text: str) -> str:
    """
    Normalize answer text before exact-match comparison.

    Normalization converts text to lowercase, removes punctuation and English
    articles (a, an, the), and collapses extra whitespace.

    This prevents superficial formatting differences from being counted as
    different answers.
    """

    text = text.lower()

    text = "".join(
        character for character in text if character not in string.punctuation
    )

    text = re.sub(r"\b(a|an|the)\b", " ", text)

    return " ".join(text.split())


def is_prediction_correct(prediction: dict[str, Any]) -> int:
    """
    Convert one forced-answer QA prediction into a binary correctness label.

    1 -> the normalized predicted answer exactly matches at least one gold answer
    0 -> the prediction is incorrect

    For unanswerable examples, correctness is always 0 in this forced-answer
    evaluation because the system was not allowed to abstain.
    """

    is_answerable = bool(prediction["is_answerable"])

    if not is_answerable:
        return 0

    predicted_answer = normalize_answer(prediction.get("prediction_text", ""))

    reference_answers = prediction.get("reference_answers", [])

    normalized_references = [
        normalize_answer(reference) for reference in reference_answers
    ]

    return int(predicted_answer in normalized_references)



def validate_confidences(confidences: np.ndarray) -> None:
    """
    Validate that confidence values can be interpreted as probabilities.

    Every confidence must be finite, non-empty, and remain inside the probability
    range [0, 1].
    """

    if np.any(~np.isfinite(confidences)):
        raise ValueError("Confidence values must be finite.")

    if confidences.size == 0:
        raise ValueError("Confidence array cannot be empty.")

    if np.any(confidences < 0.0):
        raise ValueError("Confidence values cannot be below zero.")

    if np.any(confidences > 1.0):
        raise ValueError("Confidence values cannot exceed one.")


def calculate_brier_score(
    confidences: np.ndarray, correct_labels: np.ndarray
) -> float:
    """
    Calculate the Brier score (olasılık tahmininin karesel hatası).

    For each example, the confidence probability is compared with the true binary
    outcome:

    1 -> prediction was correct
    0 -> prediction was incorrect

    Lower Brier score means better calibrated probabilities.
    """

    return float(np.mean((confidences - correct_labels) ** 2))


def calculate_negative_log_likelihood(
    confidences: np.ndarray,
    correct_labels: np.ndarray,
    epsilon: float = 1e-12,
) -> float:
    """
    Calculate binary Negative Log-Likelihood (NLL).

    NLL strongly penalizes confident mistakes. For example, assigning very high
    confidence to an incorrect prediction produces a much larger penalty than
    assigning moderate confidence to the same mistake.

    Lower NLL indicates better probabilistic calibration.
    """

    # Clip (sınırla) extreme probabilities slightly away from exactly 0 and 1.
    # This prevents log(0), which is mathematically undefined.
    clipped_confidences = np.clip(confidences, epsilon, 1.0 - epsilon)

    losses = -(
        correct_labels * np.log(clipped_confidences)
        + (1 - correct_labels) * np.log(1.0 - clipped_confidences)
    )

    return float(np.mean(losses))



def calculate_calibration_bins(
    confidences: np.ndarray,
    correct_labels: np.ndarray,
    number_of_bins: int = 10,
) -> list[dict[str, Any]]:
    """
    Group predictions into equal-width confidence bins.

    A confidence bin (güven aralığı) collects predictions with similar confidence
    values. For each bin, we compare:

    - mean_confidence: modelin ortalama güveni
    - accuracy: gerçek doğruluk oranı
    - calibration_gap: bu ikisi arasındaki mutlak fark

    Example:
    If a bin has mean confidence 0.90 but accuracy 0.70, its calibration gap is 0.20.

    Smaller gaps indicate better calibration.
    """

    if number_of_bins <= 0:
        raise ValueError("number_of_bins must be greater than zero.")

    validate_confidences(confidences)

    # Divide the full probability range [0, 1] into equal-width intervals.
    # With 10 bins, for example, the ranges are approximately:
    # [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0].
    bin_edges = np.linspace(0.0, 1.0, number_of_bins + 1)

    calibration_bins: list[dict[str, Any]] = []

    for bin_index in range(number_of_bins):
        lower_bound = float(bin_edges[bin_index])
        upper_bound = float(bin_edges[bin_index + 1])

        # Include confidence=1.0 in the final bin.
        if bin_index == number_of_bins - 1:
            in_bin = (confidences >= lower_bound) & (confidences <= upper_bound)

        else:
            in_bin = (confidences >= lower_bound) & (confidences < upper_bound)

        count = int(np.sum(in_bin))

        if count == 0:
            calibration_bins.append(
                {
                    "bin_index": bin_index,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "count": 0,
                    "mean_confidence": None,
                    "accuracy": None,
                    "calibration_gap": None,
                }
            )

            continue

        mean_confidence = float(np.mean(confidences[in_bin]))
        accuracy = float(np.mean(correct_labels[in_bin]))

        # Calibration gap (kalibrasyon farkı) measures how far the model's average
        # confidence in this bin is from the actual fraction of correct predictions.
        calibration_gap = abs(mean_confidence - accuracy)

        calibration_bins.append(
            {
                "bin_index": bin_index,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "calibration_gap": calibration_gap,
            }
        )

    return calibration_bins


def calculate_ece(
    calibration_bins: list[dict[str, Any]], total_examples: int
) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    ECE combines calibration gaps from all bins into one weighted average.
    Bins containing more examples contribute more strongly to the final score.

    Lower ECE means confidence and empirical accuracy are better aligned overall.
    """

    if total_examples <= 0:
        raise ValueError("total_examples must be greater than zero.")

    ece = 0.0

    for calibration_bin in calibration_bins:
        count = calibration_bin["count"]
        gap = calibration_bin["calibration_gap"]
        if count == 0 or gap is None:
            continue

        bin_weight = count / total_examples

        ece += bin_weight * gap

    return float(ece)



def calculate_mce(calibration_bins: list[dict[str, Any]]) -> float:
    """
    Calculate Maximum Calibration Error (MCE).

    MCE reports the single largest calibration gap among all non-empty bins.
    Unlike ECE, which summarizes average calibration quality, MCE highlights the
    worst-calibrated confidence region.
    """

    gaps = [
        calibration_bin["calibration_gap"]
        for calibration_bin in calibration_bins
        if calibration_bin["calibration_gap"] is not None
    ]

    if not gaps:
        return 0.0

    return float(max(gaps))


def calculate_calibration_metrics(
    predictions: list[dict[str, Any]], number_of_bins: int = 10
) -> dict[str, Any]:
    """
    Compute the complete calibration report for stored QA predictions.

    The `confidence` field is interpreted as the estimated probability that the
    forced-answer prediction is correct.

    Correctness is converted to binary labels and then compared with confidence
    using bin-based metrics (ECE and MCE) and probability-based metrics
    (Brier score and NLL).
    """

    if not predictions:
        raise ValueError("Prediction list cannot be empty.")

    if "confidence" not in predictions[0]:
        raise ValueError(
            "Confidence scores are missing. Run confidence_estimator.py first."
        )

    # Collect the confidence probabilities that will be evaluated for calibration.
    confidences = np.asarray(
        [float(prediction["confidence"]) for prediction in predictions],
        dtype=np.float64,
    )

    # Convert every prediction into a binary correctness target:
    # 1 = correct, 0 = incorrect.
    correct_labels = np.asarray(
        [is_prediction_correct(prediction) for prediction in predictions],
        dtype=np.float64,
    )

    validate_confidences(confidences)

    calibration_bins = calculate_calibration_bins(
        confidences=confidences,
        correct_labels=correct_labels,
        number_of_bins=number_of_bins,
    )

    ece = calculate_ece(
        calibration_bins=calibration_bins, total_examples=len(predictions)
    )

    mce = calculate_mce(calibration_bins)

    brier_score = calculate_brier_score(
        confidences=confidences, correct_labels=correct_labels
    )

    negative_log_likelihood = calculate_negative_log_likelihood(
        confidences=confidences, correct_labels=correct_labels
    )

    accuracy = float(np.mean(correct_labels))

    mean_confidence = float(np.mean(confidences))

    # Overall confidence-accuracy gap compares mean reported confidence with
    # overall empirical accuracy. It is useful as a simple summary, but unlike ECE
    # it does not show calibration differences across confidence regions.
    return {
        "system": predictions[0].get("system", "unknown"),
        "total_examples": len(predictions),
        "correct_examples": int(np.sum(correct_labels)),
        "accuracy": accuracy,
        "mean_confidence": mean_confidence,
        "confidence_accuracy_gap": abs(mean_confidence - accuracy),
        "expected_calibration_error": ece,
        "maximum_calibration_error": mce,
        "brier_score": brier_score,
        "negative_log_likelihood": negative_log_likelihood,
        "number_of_bins": number_of_bins,
        "calibration_bins": calibration_bins,
    }


def run_calibration_analysis(
    input_path: str | Path,
    output_path: str | Path,
    number_of_bins: int,
) -> dict[str, Any]:
    """
    Load prediction records, compute calibration metrics, and save the report.

    The output JSON contains both overall metrics and detailed confidence-bin
    statistics for later analysis and visualization.
    """

    predictions = load_jsonl(input_path)

    metrics = calculate_calibration_metrics(
        predictions=predictions, number_of_bins=number_of_bins
    )

    save_json(metrics, output_path)

    print("\nCalibration analysis completed.")

    print(f"Examples: {metrics['total_examples']}")

    print(f"Accuracy: {metrics['accuracy']:.4f}")

    print(f"Mean confidence: {metrics['mean_confidence']:.4f}")

    print(f"ECE: {metrics['expected_calibration_error']:.4f}")

    print(f"MCE: {metrics['maximum_calibration_error']:.4f}")

    print(f"Brier score: {metrics['brier_score']:.4f}")

    print(f"NLL: {metrics['negative_log_likelihood']:.4f}")

    print(f"Saved to: {output_path}")

    return metrics


def parse_arguments() -> argparse.Namespace:
    """Parse calibration-analysis paths and bin settings."""

    parser = argparse.ArgumentParser(
        description="Measure calibration quality of QA confidence scores."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Prediction JSONL file containing confidence scores.",
    )

    parser.add_argument(
        "--output", default="outputs/tables/raw_confidence_calibration.json"
    )


    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of equal-width confidence bins.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_calibration_analysis(
        input_path=args.input,
        output_path=args.output,
        number_of_bins=args.bins,
    )
