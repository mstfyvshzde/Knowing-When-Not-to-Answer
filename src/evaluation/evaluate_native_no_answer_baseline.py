"""
Evaluate the native SQuAD 2.0 no-answer baseline.

This module evaluates predictions produced by the QA model's native
no-answer behavior and reports metrics for:

1. the complete evaluation set,
2. answerable examples only,
3. unanswerable examples only.

Every prediction is evaluated once using the shared selective-QA metric
implementation before subgroup metrics are calculated.

The native baseline can make two decisions:

- ANSWER
- ABSTAIN

Important
---------
`accuracy` is selective-QA task accuracy and therefore rewards both:

- correct answers to answerable questions, and
- correct abstentions on unanswerable questions.

`exact_match` and `token_f1` follow this repository's answer-text convention.
Correct abstentions receive zero lexical EM/F1 credit, so these values should
not be presented as official SQuAD v2 Exact Match/F1 scores.

The answerable-only and unanswerable-only summaries are useful for separating
two distinct behaviors:

- unnecessary abstention on answerable questions,
- incorrect answering on unanswerable questions.
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
    "outputs/predictions/native_no_answer_baseline_test.jsonl"
)

DEFAULT_EVALUATED_OUTPUT_PATH = Path(
    "outputs/predictions/"
    "native_no_answer_baseline_evaluated_test.jsonl"
)

DEFAULT_METRICS_OUTPUT_PATH = Path(
    "outputs/tables/native_no_answer_baseline_metrics.json"
)


def validate_predictions(
    predictions: list[dict[str, Any]],
) -> None:
    """
    Validate fields required by the shared selective-QA evaluator.

    Every native-baseline record must contain:

    - `decision`
    - `is_answerable`

    Detailed validation of decision values, Boolean answerability values,
    references, and prediction text is handled by the shared metrics module.
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


def validate_metadata_consistency(
    predictions: list[dict[str, Any]],
) -> None:
    """
    Check that system, model, and split metadata are internally consistent.

    The summary records one value for each metadata field. Mixing predictions
    from different systems, models, or splits in one evaluation file would
    therefore make the aggregate summary misleading.
    """

    metadata_fields = (
        "system",
        "model",
        "split",
    )

    for field in metadata_fields:
        observed_values = {
            str(prediction[field])
            for prediction in predictions
            if (
                field in prediction
                and prediction[field] is not None
            )
        }

        if len(observed_values) > 1:
            raise ValueError(
                f"Inconsistent {field!r} metadata "
                f"across predictions: "
                f"{sorted(observed_values)}"
            )


def evaluate_predictions(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Evaluate native-baseline predictions exactly once.

    The shared evaluator normalizes decisions and answerability values and
    attaches per-example correctness, Exact Match, and token F1 fields.
    """

    validate_predictions(
        predictions
    )

    validate_metadata_consistency(
        predictions
    )

    return evaluate_metric_predictions(
        predictions
    )


def split_by_answerability(
    evaluated_predictions: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    Split evaluated predictions into answerable and unanswerable subsets.

    The shared evaluator has already converted `is_answerable` into a strict
    Boolean, avoiding Python truthiness errors such as treating the string
    "False" as True.
    """

    answerable_predictions = [
        prediction
        for prediction in evaluated_predictions
        if prediction["is_answerable"] is True
    ]

    unanswerable_predictions = [
        prediction
        for prediction in evaluated_predictions
        if prediction["is_answerable"] is False
    ]

    if (
        len(answerable_predictions)
        + len(unanswerable_predictions)
        != len(evaluated_predictions)
    ):
        raise RuntimeError(
            "Answerability split does not account "
            "for every evaluated prediction."
        )

    if not answerable_predictions:
        raise ValueError(
            "Native no-answer evaluation requires "
            "at least one answerable prediction."
        )

    if not unanswerable_predictions:
        raise ValueError(
            "Native no-answer evaluation requires "
            "at least one unanswerable prediction."
        )

    return (
        answerable_predictions,
        unanswerable_predictions,
    )


def build_baseline_summary(
    predictions: list[dict[str, Any]],
    evaluated_predictions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Build overall and answerability-specific native-baseline metrics.

    If evaluated records are supplied, they are reused directly so predictions
    are not evaluated a second time.
    """

    validate_predictions(
        predictions
    )

    validate_metadata_consistency(
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

    (
        answerable_predictions,
        unanswerable_predictions,
    ) = split_by_answerability(
        evaluated_predictions
    )

    overall_metrics = (
        calculate_metrics_from_evaluated(
            evaluated_predictions
        )
    )

    answerable_metrics = (
        calculate_metrics_from_evaluated(
            answerable_predictions
        )
    )

    unanswerable_metrics = (
        calculate_metrics_from_evaluated(
            unanswerable_predictions
        )
    )

    first_prediction = predictions[0]

    return {
        "evaluation_type": (
            "native_no_answer_baseline"
        ),
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
        "metrics": overall_metrics,
        "answerable_only": (
            answerable_metrics
        ),
        "unanswerable_only": (
            unanswerable_metrics
        ),
    }


def save_metrics(
    summary: dict[str, Any],
    output_path: str | Path,
) -> None:
    """
    Save the aggregate native-baseline summary as formatted JSON.
    """

    save_json(
        summary,
        output_path,
    )


def print_subset_summary(
    subset_name: str,
    metrics: dict[str, Any],
) -> None:
    """
    Print the most informative metrics for one answerability subset.
    """

    print(
        f"\n{subset_name}:"
    )

    print(
        f"  Examples: "
        f"{metrics['total']}"
    )

    print(
        f"  Task accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"  Coverage: "
        f"{metrics['coverage']:.4f}"
    )

    print(
        f"  Abstention rate: "
        f"{metrics['abstention_rate']:.4f}"
    )

    print(
        f"  Answered accuracy: "
        f"{metrics['answered_accuracy']:.4f}"
    )

    print(
        f"  Selective risk: "
        f"{metrics['selective_risk']:.4f}"
    )


def run_baseline_evaluation(
    input_path: str | Path,
    evaluated_output_path: str | Path,
    metrics_output_path: str | Path,
) -> dict[str, Any]:
    """
    Run the complete native no-answer baseline evaluation.

    Workflow:

    1. load predictions,
    2. validate records and metadata,
    3. evaluate every prediction once,
    4. calculate overall metrics,
    5. calculate answerable-only metrics,
    6. calculate unanswerable-only metrics,
    7. save evaluated predictions,
    8. save the aggregate JSON summary.
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
        "\nNative no-answer baseline "
        "evaluation completed."
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

    print_subset_summary(
        subset_name="Answerable-only",
        metrics=summary[
            "answerable_only"
        ],
    )

    print_subset_summary(
        subset_name="Unanswerable-only",
        metrics=summary[
            "unanswerable_only"
        ],
    )

    print(
        "\nEvaluated predictions saved to: "
        f"{evaluated_output_path}"
    )

    print(
        f"Metrics saved to: "
        f"{metrics_output_path}"
    )

    return summary


def parse_arguments() -> argparse.Namespace:
    """Parse native no-answer baseline evaluation paths."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the native SQuAD 2.0 "
            "no-answer baseline using shared "
            "project metrics."
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
    """Run native no-answer baseline evaluation from the command line."""

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