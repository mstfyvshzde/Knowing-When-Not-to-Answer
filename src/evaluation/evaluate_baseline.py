"""
Evaluate the raw forced-answer QA baseline.

This module applies the project's shared selective-QA evaluation metrics to
raw baseline predictions and saves:

1. per-example evaluated predictions, and
2. an aggregate JSON metrics summary.

The raw baseline is evaluated using the same metric implementation used by
other project components so that correctness definitions remain consistent.

Important
---------
`accuracy` represents selective-QA task correctness.

`exact_match` and `token_f1` follow the project's answer-text evaluation
convention and should not automatically be presented as official SQuAD v2
metrics.
"""

import argparse
from pathlib import Path
from typing import Any

from src.evaluation.metrics import (
    calculate_metrics_from_evaluated,
)
from src.evaluation.metrics import (
    evaluate_predictions as evaluate_metric_predictions,
)
from src.utils.io import (
    load_jsonl,
    save_json,
    save_jsonl,
)

DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/raw_baseline_calibration.jsonl"
)

DEFAULT_EVALUATED_OUTPUT_PATH = Path(
    "outputs/predictions/"
    "raw_baseline_evaluated_calibration.jsonl"
)

DEFAULT_METRICS_OUTPUT_PATH = Path(
    "outputs/tables/raw_baseline_metrics.json"
)


def validate_predictions(
    predictions: list[dict[str, Any]],
) -> None:
    """
    Validate fields required by the shared selective-QA evaluator.

    Every raw baseline record must contain:

    - `decision`
    - `is_answerable`

    More detailed validation of decision values, answerability values, and
    reference-answer structure is performed by the shared metrics module.
    """

    if not predictions:
        raise ValueError(
            "Prediction list cannot be empty."
        )

    required_fields = {
        "decision",
        "is_answerable",
    }

    for index, prediction in enumerate(
        predictions,
        start=1,
    ):
        missing_fields = (
            required_fields
            - prediction.keys()
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


def evaluate_predictions(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Evaluate all raw baseline predictions using the shared metric logic.

    This wrapper preserves the baseline evaluator's public interface while
    delegating actual correctness calculations to `src.evaluation.metrics`.
    """

    validate_predictions(
        predictions
    )

    return evaluate_metric_predictions(
        predictions
    )


def build_baseline_summary(
    predictions: list[dict[str, Any]],
    evaluated_predictions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Build the aggregate raw-baseline evaluation summary.

    When already evaluated records are supplied, metrics are aggregated directly
    from those records rather than evaluating the same predictions a second
    time.
    """

    validate_predictions(
        predictions
    )

    if evaluated_predictions is None:
        evaluated_predictions = (
            evaluate_predictions(
                predictions
            )
        )

    if (
        len(evaluated_predictions)
        != len(predictions)
    ):
        raise ValueError(
            "Evaluated prediction count does not "
            "match raw prediction count."
        )

    metrics = (
        calculate_metrics_from_evaluated(
            evaluated_predictions
        )
    )

    first_prediction = predictions[0]

    return {
        "evaluation_type": "raw_qa_baseline",
        "system": first_prediction.get(
            "system",
            "unknown",
        ),
        "model": first_prediction.get(
            "model",
            "unknown",
        ),
        "split": first_prediction.get(
            "split",
            "unknown",
        ),
        "total_predictions": len(
            predictions
        ),
        "metrics": metrics,
    }


def save_metrics(
    summary: dict[str, Any],
    output_path: str | Path,
) -> None:
    """
    Save the aggregate baseline summary as formatted JSON.

    Shared repository I/O is used so JSON serialization behavior remains
    consistent across experiment scripts.
    """

    save_json(
        summary,
        output_path,
    )


def run_baseline_evaluation(
    input_path: str | Path,
    evaluated_output_path: str | Path,
    metrics_output_path: str | Path,
) -> dict[str, Any]:
    """
    Run the complete raw-baseline evaluation workflow.

    The function:

    1. loads raw predictions,
    2. validates and evaluates every prediction,
    3. aggregates project metrics,
    4. saves per-example evaluated records,
    5. saves the aggregate metrics summary,
    6. prints the main results.
    """

    predictions = load_jsonl(
        input_path
    )

    evaluated_predictions = (
        evaluate_predictions(
            predictions
        )
    )

    summary = (
        build_baseline_summary(
            predictions=predictions,
            evaluated_predictions=(
                evaluated_predictions
            ),
        )
    )

    save_jsonl(
        evaluated_predictions,
        evaluated_output_path,
    )

    save_metrics(
        summary=summary,
        output_path=metrics_output_path,
    )

    metrics = summary[
        "metrics"
    ]

    print(
        "\nBaseline evaluation completed."
    )

    print(
        f"Total predictions: "
        f"{summary['total_predictions']}"
    )

    print(
        f"Task accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Project Exact Match: "
        f"{metrics['exact_match']:.4f}"
    )

    print(
        f"Project Token F1: "
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
        "Evaluated predictions saved to: "
        f"{evaluated_output_path}"
    )

    print(
        f"Metrics saved to: "
        f"{metrics_output_path}"
    )

    return summary


def parse_arguments() -> argparse.Namespace:
    """Parse raw-baseline evaluation paths."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the raw QA baseline "
            "using shared project metrics."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
    )

    parser.add_argument(
        "--evaluated-output",
        type=Path,
        default=(
            DEFAULT_EVALUATED_OUTPUT_PATH
        ),
    )

    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=(
            DEFAULT_METRICS_OUTPUT_PATH
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run raw-baseline evaluation from command-line arguments."""

    args = parse_arguments()

    run_baseline_evaluation(
        input_path=args.input,
        evaluated_output_path=(
            args.evaluated_output
        ),
        metrics_output_path=(
            args.metrics_output
        ),
    )


if __name__ == "__main__":
    main()