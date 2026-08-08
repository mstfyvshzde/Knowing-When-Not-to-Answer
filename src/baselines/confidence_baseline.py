"""
This file applies a confidence-based abstention strategy to raw question-answering predictions.
It uses a confidence threshold to decide whether to return the model's answer or abstain, then saves the updated predictions and reports summary statistics.
"""

# argparse is used to read command-line arguments when you run the script from the terminal.
import argparse

# is used to create and manage file paths safely.
from pathlib import Path

# Any is a flexible type hint that allows a variable or dictionary value to contain any Python data type.
from typing import Any

from src.utils.io import load_jsonl, save_jsonl

DEFAULT_INPUT_PATH = Path("outputs/predictions/raw_baseline_calibration.jsonl")

OUTPUT_DIR = Path("outputs/predictions")


# decide whether to answer or abstain based on the model's confidence score.
# If the confidence is above the threshold, it keeps the predicted answer; otherwise, it replaces it with "I do not know" and marks the decision as ABSTAIN.
def apply_confidence_threshold(
    predictions: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:

    # Threshold is the minimum confidence score the model must have to return an answer instead of abstaining.
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    updated_predictions: list[dict[str, Any]] = []

    for prediction in predictions:
        confidence = float(prediction["pipeline_score"])

        if confidence >= threshold:
            decision = "ANSWER"
            final_answer = prediction["prediction_text"]

        else:
            decision = "ABSTAIN"
            final_answer = "I do not know"

        updated_prediction = prediction.copy()

        updated_prediction.update(
            {
                "final_answer": final_answer,
                "decision": decision,
                "threshold": threshold,
                "system": "confidence_baseline",
            }
        )

        updated_predictions.append(updated_prediction)

    return updated_predictions


# calculates overall statistics about the model's decisions, such as how many questions were answered, abstained, and the corresponding rates.
def summarize_decisions(predictions: list[dict[str, Any]]) -> dict[str, float | int]:
    total = len(predictions)

    if total == 0:
        raise ValueError("Prediction list cannot be empty.")

    answered = sum(prediction["decision"] == "ANSWER" for prediction in predictions)

    abstained = total - answered

    return {
        "total_examples": total,
        "answered_examples": answered,
        "abstained-examples": abstained,
        "coverage": answered / total,
        "abstention_rate": abstained / total,
    }


# applies a confidence threshold to raw predictions, saves the updated predictions, prints a summary, and returns the final results.
def run_confidence_baseline(
    input_path: str | Path, threshold: float
) -> list[dict[str, Any]]:
    input_path = Path(input_path)

    raw_predictions = load_jsonl(input_path)

    predictions = apply_confidence_threshold(
        predictions=raw_predictions, threshold=threshold
    )

    summary = summarize_decisions(predictions)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    threshold_name = str(threshold).replace(".", "-")

    output_path = OUTPUT_DIR / f"confidence_baseline_{threshold_name}.jsonl"

    save_jsonl(predictions, output_path)

    print("\nConfidence baseline completed.")
    print(f"Threshold: {threshold:.2f}")
    print(f"Answered: {summary['answered_examples']}/{summary['total_examples']}")
    print(f"Coverage: {summary['coverage']:.4f}")
    print(f"Abstention rate: {summary['abstention_rate']:.4f}")
    print(f"Saved to: {output_path}")

    return predictions


# eads terminal options for the input predictions file and the confidence threshold.
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply confidence-based abstention to raw QA predictions."
    )

    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_PATH))

    parser.add_argument("--threshold", type=float, default=0.50)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_confidence_baseline(
        input_path=args.input,
        threshold=args.threshold,
    )
