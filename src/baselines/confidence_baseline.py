"""
Apply a fixed threshold to raw forced-answer QA pipeline scores.

For each prediction, the raw QA pipeline score is compared with a fixed
threshold (eşik değer):

- score >= threshold -> ANSWER
- score < threshold  -> ABSTAIN

Abstention (cevap vermekten kaçınma) means that the system deliberately
refuses to return the predicted answer when its score is too low.

This module is an earlier threshold-based baseline used for comparison.
The final selective-QA experiments rank examples by their scoring signals and
evaluate risk across different coverage levels rather than relying on one
fixed threshold.
"""


import argparse
from pathlib import Path
from typing import Any

from src.utils.io import load_jsonl, save_jsonl

# Forced-answer QA predictions used by this baseline.
# Their pipeline_score values are raw model scores, not temperature-calibrated
# confidence probabilities used later in the project.
DEFAULT_INPUT_PATH = Path("outputs/predictions/raw_baseline_calibration.jsonl")

OUTPUT_DIR = Path("outputs/predictions")



def apply_confidence_threshold(
    predictions: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    """
    Convert raw QA scores into ANSWER or ABSTAIN decisions.

    The threshold (eşik değer) is the minimum raw pipeline score required for the
    system to keep the model's predicted answer.

    This is a simple baseline: it uses only the QA score and does not use
    calibration, semantic verification, or any other evidence signal.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    updated_predictions: list[dict[str, Any]] = []

    for prediction in predictions:
        # Use the raw Hugging Face QA pipeline score produced by the forced-answer
        # baseline. Calling this pipeline_score avoids confusing it with the calibrated
        # confidence calculated later in the project.
        pipeline_score = float(prediction["pipeline_score"])

        if pipeline_score >= threshold:
            decision = "ANSWER"
            final_answer = prediction["prediction_text"]

        else:
        # ABSTAIN (cevap vermekten kaçınma) replaces the model's proposed answer
        # with an explicit refusal because its score is below the threshold.
            decision = "ABSTAIN"
            final_answer = "I do not know"

        # Keep the original QA prediction unchanged and add the threshold-based
        # decision as extra experiment metadata.
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



def summarize_decisions(predictions: list[dict[str, Any]]) -> dict[str, float | int]:
    """
    Summarize ANSWER and ABSTAIN decisions.

    Coverage (kapsama oranı) is the fraction of all examples for which the system
    chooses to answer.

    Abstention rate (kaçınma oranı) is the fraction for which it refuses to answer.
    """

    total = len(predictions)

    if total == 0:
        raise ValueError("Prediction list cannot be empty.")

    answered = sum(prediction["decision"] == "ANSWER" for prediction in predictions)

    abstained = total - answered

    return {
        "total_examples": total,
        "answered_examples": answered,
        "abstained_examples": abstained,
        "coverage": answered / total,
        "abstention_rate": abstained / total,
    }



def run_confidence_baseline(
    input_path: str | Path, threshold: float
) -> list[dict[str, Any]]:
    """
    Run the fixed-threshold baseline on stored forced-answer predictions.

    The function loads existing QA predictions, converts their raw pipeline scores
    into ANSWER/ABSTAIN decisions, reports coverage statistics, and saves the
    annotated predictions for later evaluation.
    """

    input_path = Path(input_path)

    # Load previously generated forced-answer QA predictions.
    raw_predictions = load_jsonl(input_path)

    predictions = apply_confidence_threshold(
        predictions=raw_predictions, threshold=threshold
    )

    summary = summarize_decisions(predictions)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Encode the threshold in the output filename for experiment traceability.
    threshold_name = str(threshold).replace(".", "-")

    output_path = OUTPUT_DIR / f"confidence_baseline_{threshold_name}.jsonl"

    save_jsonl(predictions, output_path)

    print("\nConfidence baseline completed.")
    print(f"Raw pipeline-score threshold: {threshold:.2f}")
    print(f"Answered: {summary['answered_examples']}/{summary['total_examples']}")
    print(f"Coverage: {summary['coverage']:.4f}")
    print(f"Abstention rate: {summary['abstention_rate']:.4f}")
    print(f"Saved to: {output_path}")

    return predictions


def parse_arguments() -> argparse.Namespace:
    """Parse the input prediction path and confidence threshold."""

    parser = argparse.ArgumentParser(
        description="Run the fixed-threshold confidence abstention baseline."
    )


    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="Path to the raw forced-answer prediction JSONL file.",
    )


    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Minimum raw QA pipeline score required for an ANSWER decision.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_confidence_baseline(
        input_path=args.input,
        threshold=args.threshold,
    )
