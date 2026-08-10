"""
Evaluates the raw QA baseline by measuring its prediction quality with metrics such as accuracy, Exact Match, F1, coverage, abstention rate, and selective risk, then saves the results.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from src.evaluation.metrics import (
    calculate_metrics,
    evaluate_single_prediction,
)
from src.utils.io import (
    load_jsonl,
    save_jsonl,
)


DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/raw_baseline_calibration.jsonl"
)

DEFAULT_EVALUATED_OUTPUT_PATH = Path(
    "outputs/predictions/raw_baseline_evaluated_calibration.jsonl"
)

DEFAULT_METRICS_OUTPUT_PATH = Path(
    "outputs/tables/raw_baseline_metrics.json"
)


# Checks that the prediction list is not empty and that every prediction contains the required fields: decision and is_answerable.
def validate_predictions(
    predictions: list[dict[str, Any]],
) -> None:
    if not predictions:
        raise ValueError(
            "Prediction list cannot be empty."
        )

    required_fields = {
        "decision",
        "is_answerable"
    }

    for index, prediction in enumerate(
        predictions,
        start=1
    ):
        missing_fields = (
            required_fields - prediction.keys()
        )

        if missing_fields:
            missing_text = ", ".join(
                sorted(missing_fields)
            )

            raise ValueError(
                f"Prediction {index} is missing "
                f"required fields: {missing_text}. "
                f"Available keys: "
                f"{list(prediction.keys())}"
            )


# Validates the predictions and evaluates each one individually, returning the evaluated results as a list.
def evaluate_predictions(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    validate_predictions(predictions)

    return [
        evaluate_single_prediction(prediction)
        for prediction in predictions
    ]


# Creates a baseline evaluation summary containing the system/model information, dataset split, number of predictions, and calculated performance metrics.
def build_baseline_summary(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    validate_predictions(predictions)

    metrics = calculate_metrics(predictions)

    first_prediction = predictions[0]

    return {
        "evaluation_type": "raw_qa_baseline",
        "system": first_prediction.get(
            "system",
            "unknown"
        ),
        "model": first_prediction.get(
            "model",
            "unknown"
        ),
        "split": first_prediction.get(
            "split",
            "unknown"
        ),
        "total_predictions": len(predictions),
        "metrics": metrics
    }


# Saves the evaluation summary as a readable JSON file and creates the output folder if it does not already exist.
def save_metrics(
    summary: dict[str, Any],
    output_path: str | Path
) -> None:
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as output_file:
        json.dump(
            summary,
            output_file,
            indent=2,
            ensure_ascii=False
        )


# Runs the complete baseline evaluation: loads predictions, evaluates them, calculates metrics, saves both detailed results and the summary, and prints the main performance scores.
def run_baseline_evaluation(
    input_path: str | Path,
    evaluated_output_path: str | Path,
    metrics_output_path: str | Path
) -> dict[str, Any]:
    predictions = load_jsonl(input_path)

    evaluated_predictions = (
        evaluate_predictions(predictions)
    )

    summary = build_baseline_summary(
        predictions
    )

    evaluated_output_path = Path(
        evaluated_output_path
    )

    evaluated_output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    save_jsonl(
        evaluated_predictions,
        evaluated_output_path
    )

    save_metrics(
        summary=summary,
        output_path=metrics_output_path
    )

    metrics = summary["metrics"]

    print(
        "\nBaseline evaluation completed."
    )

    print(
        f"Total predictions: "
        f"{summary['total_predictions']}"
    )

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Exact Match: "
        f"{metrics['exact_match']:.4f}"
    )

    print(
        f"Token F1: "
        f"{metrics['token_f1']:.4f}"
    )

    print(
        f"Coverage: "
        f"{metrics['coverage']:.4f}"
    )

    print(
        f"Abstention rate: "
        f"{metrics['abstention_rate']:.4f}"
    )

    print(
        f"Answered accuracy: "
        f"{metrics['answered_accuracy']:.4f}"
    )

    print(
        f"Selective risk: "
        f"{metrics['selective_risk']:.4f}"
    )

    print(
        f"Evaluated predictions saved to: "
        f"{evaluated_output_path}"
    )

    print(
        f"Metrics saved to: "
        f"{metrics_output_path}"
    )

    return summary


# Defines and reads command-line arguments for the baseline evaluation, including input predictions and output paths for evaluated results and metrics.
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the raw QA baseline "
            "using standard project metrics."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH
    )

    parser.add_argument(
        "--evaluated-output",
        type=Path,
        default=DEFAULT_EVALUATED_OUTPUT_PATH
    )

    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=DEFAULT_METRICS_OUTPUT_PATH
    )

    return parser.parse_args()


# Starts the baseline evaluation by reading command-line arguments and passing them to run_baseline_evaluation().
def main() -> None:
    args = parse_arguments()

    run_baseline_evaluation(
        input_path=args.input,
        evaluated_output_path=(
            args.evaluated_output
        ),
        metrics_output_path=(
            args.metrics_output
        )
    )


if __name__ == "__main__":
    main()