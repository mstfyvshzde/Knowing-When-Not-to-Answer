""" 
The main purpose of this file is to evaluate the calibration quality of a question-answering model's confidence scores. It loads model predictions, determines which predictions are correct, computes calibration metrics (such as ECE, MCE, Brier Score, Negative Log-Likelihood, Accuracy, and Confidence Gap), groups predictions into confidence bins, and saves a complete calibration report for analysis.
"""

import argparse
import numpy as np
import re
import string
from typing import Any
from pathlib import Path

from src.utils.io import load_jsonl, save_json


# cleans an answer so fair comparison becomes easier.
# It converts text to lowercase, removes punctuation and articles like a/an/the, removes extra spaces, and returns the normalized answer.
def normalize_answer(text: str) -> str:
    text = text.lower()

    text = ''.join(
        character for character in text if character not in string.punctuation
    )

    text = re.sub(r'\b(a | an| the |)\b', ' ', text)

    return ' '.join(text.split())


# It determines whether the model's predicted answer is correct (1) or incorrect (0) by comparing the normalized prediction with the normalized reference answers.
def is_prediction_correct(prediction: dict[str, Any]) -> int:
    is_answerable = bool(prediction['is_answerable'])

    if not is_answerable:
        return 0.0

    # normalizes the model's predicted answer.
    predicted_answer = normalize_answer(prediction.get('prediction_text', ''))

    # gets all correct reference answers.
    reference_answer = prediction.get('reference_answers', [])

    # normalizes every reference answer.
    normalized_references = [
        normalize_answer(reference) for reference in reference_answer
    ]

    # returns 1 if the prediction matches any reference answer; otherwise returns 0.
    return int(predicted_answer in normalized_references)



# checks whether the confidence scores are valid before they are used.
def validate_confidences(confidences: np.ndarray) -> None:
    if confidences.size == 0:
        raise ValueError("Confidence array cannot be empty.")

    if np.any(confidences < 0.0):
        raise ValueError("Confidence values cannot be below zero.")

    if np.any(confidences > 1.0):
        raise ValueError("Confidence values cannot exceed one.")


# Brier score measures how close the model’s confidence scores are to the real outcomes
def calculate_brier_score(confidences: np.ndarray, correct_labels: np.ndarray) -> float:

    return float(np.mean((confidences - correct_labels) ** 2))


# Negative Log-Likelihood measures whether the model’s confidence scores are appropriate for the real outcomes.
# For each prediction:
# If the prediction is correct (label = 1), NLL expects a confidence close to 1.
# If the prediction is incorrect (label = 0), NLL expects a confidence close to 0.
def calculate_negative_log_likelihood(
    confidences: np.ndarray, correct_labels: np.ndarray, epsilon: float =1e-12
) -> float:

    clipped_confidences = np.clip(confidences, epsilon, 1.0 - epsilon)

    # The minus sign is needed because log(x) is negative when 0 < x < 1.
    # It converts the negative log value into a positive loss value.
    losses = -(
        correct_labels * np.log(clipped_confidences)
        + (1 - correct_labels) * np.log(1.0 - clipped_confidences)
    )

    return float(np.mean(losses))



# groups predictions into confidence intervals (bins) and measures how well the model's confidence matches its actual accuracy in each interval.
# Confidence bins group predictions by similar confidence scores.
# Instead of checking every prediction separately, we divide the confidence range from 0.0 to 1.0 into equal intervals.
# We need bins because individual confidence scores are too scattered to judge calibration clearly.
# By grouping similar confidence values, we can compare average confidence vs actual accuracy in each range and see where the model is overconfident or underconfident.
# Example:
# number_of_bins = 10

# Bin 0 -> 0.0 to 0.1
# Bin 9 -> 0.9 to 1.0

# Each bin stores predictions whose confidence falls inside that range.
# Example predictions in the 0.9 to 1.0 bin:

# Confidence   Correct
# 0.92         True
# 0.95         False
# 0.98         True
# 0.94         True

# Mean confidence:
# (0.92 + 0.95 + 0.98 + 0.94) / 4 = 0.9475

# Accuracy:
# 3 correct predictions / 4 predictions = 0.75

# Calibration gap:
# abs(0.9475 - 0.75) = 0.1975
#
# This means the model is about 95% confident, but it is actually correct only 75% of the time.
def calculate_calibration_bins(
    confidences: np.ndarray, correct_labels: np.ndarray, number_of_bins: int = 10
) -> list[dict[str, Any]]:
    if number_of_bins <= 0:
        raise ValueError("number_of_bins must be greater than zero.")

    validate_confidences(confidences)

    # creates equally spaced boundary values between 0 and 1.
    # bin_edges = [0.0, 0.1, 0.2, ..., 0.9, 1.0]
    bin_edges = np.linspace(0.0, 1.0, number_of_bins + 1)

    # creates an empty list that will store the summary of each confidence bin.
    # calibration_bins = [
#     {
#         "bin_index": 0,
#         "lower_bound": 0.0,
#         "upper_bound": 0.1,
#     },
#     {
#         "bin_index": 9,
#         "lower_bound": 0.9,
#         "upper_bound": 1.0,
#     }
#   ]
    calibration_bins: list[dict[str, Any]] = []

    for bin_index in range(number_of_bins):
        lower_bound = float(bin_edges[bin_index])
        upper_bound = float(bin_edges[bin_index + 1])

        # For normal bins, the lower edge is included and the upper edge is excluded:
        # 0.2 <= confidence < 0.3
        if bin_index == number_of_bins - 1:
            in_bin = (confidences >= lower_bound) & (confidences <= upper_bound)

        # But the final bin includes both edges:
        # 0.9 <= confidence <= 1.0
        # This is necessary so a confidence of exactly 1.0 is not left outside all bins.
        else: 
            # in_bin is a Boolean array that shows which confidence scores belong to the current bin.
            # Example:
            # confidences = [0.15, 0.25, 0.28, 0.92]
            # For the 0.2–0.3 bin:
            # in_bin = [False, True, True, False]
            # So it acts like a mask: only the True positions are selected.
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
        # Example:
        # confidences[in_bin] = [0.92, 0.95, 0.98, 0.94]
        # correct_labels[in_bin] = [1, 0, 1, 1]

        # measures the difference between the bin’s average confidence and its real accuracy.
        # A smaller gap means better calibration; 0 means confidence exactly matches accuracy.
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



# Expected Calibration Error (ECE) measures the model's overall calibration error across all confidence bins.
# Instead of looking at one bin, it combines the calibration gaps from all bins into a single score.
def calculate_ece(calibration_bins: list[dict[str, Any]], total_examples: int) -> float:
    if total_examples <= 0:
        raise ValueError("total_examples must be greater than zero.")

    ece = 0.0

    for calibration_bin in calibration_bins:
        count = calibration_bin['count']
        gap = calibration_bin["calibration_gap"]
        if count == 0 or gap is None:
            continue

        bin_weight = count / total_examples

        ece += bin_weight * gap

    return float(ece)


# Maximum Calibration Error (MCE) measures the largest calibration gap among all confidence bins.
# It tells you the worst-calibrated confidence bin.
def calculate_mce(calibration_bins: list[dict[str,Any]]) -> float:
    gaps = [
        calibration_bin['calibration_gap']
        for calibration_bin in calibration_bins
        if calibration_bin['calibration_gap'] is not None
    ]

    if not gaps:
        return 0.0

    return(float(max(gaps)))


# calculates all important calibration metrics for a set of predictions and returns them together in one dictionary.
def calculate_calibration_metrics(
    predictions: list[dict[str, Any]], number_of_bins: int = 10
) -> dict[str, Any]:
    if not predictions:
        raise ValueError("Prediction list cannot be empty.")

    if 'confidence' not in predictions[0]:
        raise ValueError(
            "True confidence scores are not available yet. "
            "Run confidence_estimator.py first."
        )

    # It returns a NumPy array containing all confidence scores as float64 value
    confidences = np.asarray(
        [float(prediction['confidence']) for prediction in predictions],
        dtype=np.float64
    )

    # It returns a NumPy array containing 1s and 0s
    correct_labels = np.asarray(
        [is_prediction_correct(prediction) for prediction in predictions],
        dtype=np.float64
    )

    validate_confidences(confidences)

    # This function returns a list of dictionaries, where each dictionary summarizes one confidence bin.
    calibration_bins = calculate_calibration_bins(
        confidences=confidences, correct_labels=correct_labels, number_of_bins=number_of_bins
    )

    ece = calculate_ece(
        calibration_bins=calibration_bins, total_examples=len(predictions)
    )

    mce = calculate_mce(calibration_bins)

    brier_score = calculate_brier_score(confidences=confidences, correct_labels=correct_labels)

    negative_log_likelihood = calculate_negative_log_likelihood(
        confidences=confidences, correct_labels=correct_labels
    )

    accuracy = float(np.mean(correct_labels))

    mean_confidence = float(np.mean(confidences))

    return {
        # predictions[0].get("system", "unknown") gets the "system" value from the first prediction.
        # Example:
        # {"system": "confidence_baseline"}
        "system": predictions[0].get("system", "unknown"),
        "total_examples": len(predictions),
        "correct_examples": int(np.sum(correct_labels)),
        "accuracy": accuracy,
        "mean_confidence": mean_confidence,
        "confidence_accuracy_gap": abs(mean_confidence - accuracy),
        "expected_calibration_error": ece,
        "maximum_calibration_error": mce,
        "brier_score": brier_score,
        "negative_log_likelihood": (negative_log_likelihood),
        "number_of_bins": number_of_bins,
        "calibration_bins": calibration_bins,
    }


# runs the complete calibration evaluation pipeline.
# It loads the prediction file, calculates all calibration metrics (such as Accuracy, ECE, MCE, Brier Score, and NLL), saves the results to a JSON file, prints a summary, and returns the calculated metrics.
def run_calibration_analysis(
    input_path: str | Path,
    output_path: str | Path,
    number_of_bins: int
) -> dict[str, Any]:
    predictions = load_jsonl(input_path)

    metrics = calculate_calibration_metrics(
        predictions=predictions,
        number_of_bins=number_of_bins
    )

    save_json(
        metrics,
        output_path
    )

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
    parser = argparse.ArgumentParser(
        description=("Measure calibration of raw QA confidence scores.")
    )

    parser.add_argument(
        '--input',
        required=True,
        help='Raw prediction JSONL file'
    )

    parser.add_argument(
        '--output',
        default="outputs/tables/raw_confidence_calibration.json"
    )

    parser.add_argument(
        '--bins',
        type=int,
        default=10
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_calibration_analysis(
        input_path=args.input,
        output_path=args.output,
        number_of_bins=args.bins,
    )
