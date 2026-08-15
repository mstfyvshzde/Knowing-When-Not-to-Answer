"""
Evaluate random abstention across multiple deterministic seeds.

Random abstention is a control baseline for selective question answering.

The baseline does not use confidence, semantic verification, lexical evidence,
or any other prediction-quality signal. For a requested target coverage, it
randomly selects a fixed number of QA predictions to answer and abstains on
the rest.

Evaluation proceeds in two conceptually separate stages:

1. Determine the underlying correctness of every raw QA answer candidate.
2. Randomly choose which candidates are answered for each seed and measure
   selective accuracy/risk among those answered candidates.

This separation is important because underlying answer correctness must remain
fixed across random-abstention policies.

For each target coverage, the evaluator reports the distribution across seeds
of:

- answer accuracy,
- selective risk,
- actual coverage,
- number of answered examples.

The reported standard deviations use population standard deviation
(`statistics.pstdev`) across the explicitly evaluated seed set.

Important
---------
The random baseline uses the implementation in
`src.baselines.random_abstention_baseline`, which selects:

    round(number_of_predictions * target_coverage)

examples for ANSWER using a local `random.Random(seed)` instance.

Therefore actual coverage can differ slightly from requested coverage because
the number of answered examples must be an integer.

Selective risk is defined only when at least one example is answered:

    selective_risk = 1 - answer_accuracy
"""

import argparse
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.baselines.random_abstention_baseline import (
    apply_random_abstention,
)
from src.evaluation.metrics import (
    evaluate_predictions as evaluate_metric_predictions,
)
from src.utils.io import (
    load_jsonl,
    save_json,
)

DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/raw_baseline_calibration.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "outputs/tables/"
    "random_abstention_multi_seed_metrics.json"
)

DEFAULT_COVERAGES = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
]

DEFAULT_SEEDS = list(
    range(20)
)


def validate_coverages(
    coverages: list[float],
) -> None:
    """
    Validate requested random-abstention coverage levels.

    Coverage zero is intentionally excluded because selective accuracy and risk
    are undefined when no examples are answered.
    """

    if not coverages:
        raise ValueError(
            "At least one coverage value is required."
        )

    for coverage in coverages:
        if not math.isfinite(
            float(coverage)
        ):
            raise ValueError(
                "Coverage values must be finite, "
                f"received {coverage}."
            )

        if not (
            0.0
            < coverage
            <= 1.0
        ):
            raise ValueError(
                "Coverage must be greater than "
                f"0 and at most 1: {coverage}"
            )


def validate_seeds(
    seeds: list[int],
) -> None:
    """
    Validate the deterministic random seeds used by the control baseline.
    """

    if not seeds:
        raise ValueError(
            "At least one seed is required."
        )

    for seed in seeds:
        if (
            not isinstance(seed, int)
            or isinstance(seed, bool)
        ):
            raise ValueError(  # noqa: TRY004
                "Seeds must be integers, "
                f"received {seed!r}."
            )

        if seed < 0:
            raise ValueError(
                "Seeds must be non-negative integers."
            )


def infer_underlying_correctness(
    predictions: list[dict[str, Any]],
) -> list[bool]:
    """
    Determine raw QA candidate correctness before random abstention.

    Every raw candidate is temporarily evaluated as ANSWER, regardless of what
    later random-abstention policies will do.

    This creates one fixed correctness label per QA candidate:

    - correct extractive answer -> True
    - incorrect extractive answer -> False
    - answering an unanswerable question -> False

    Correctness is therefore independent of the random ANSWER/ABSTAIN decision.
    """

    if not predictions:
        raise ValueError(
            "Prediction list cannot be empty."
        )

    forced_answer_predictions: list[
        dict[str, Any]
    ] = []

    for prediction in predictions:
        forced_answer_prediction = dict(
            prediction
        )

        forced_answer_prediction[
            "decision"
        ] = "ANSWER"

        forced_answer_predictions.append(
            forced_answer_prediction
        )

    evaluated_predictions = (
        evaluate_metric_predictions(
            forced_answer_predictions
        )
    )

    return [
        bool(
            prediction[
                "is_correct"
            ]
        )
        for prediction
        in evaluated_predictions
    ]


def get_random_answer_indices(
    predictions: Sequence[
        dict[str, Any]
    ],
) -> list[int]:
    """
    Extract the indices selected for ANSWER by random abstention.

    Only ANSWER and ABSTAIN are accepted so malformed random-policy output
    cannot silently enter the evaluation.
    """

    answer_indices: list[int] = []

    for index, prediction in enumerate(
        predictions
    ):
        decision = (
            str(
                prediction.get(
                    "decision",
                    "",
                )
            )
            .strip()
            .upper()
        )

        if decision == "ANSWER":
            answer_indices.append(
                index
            )

        elif decision == "ABSTAIN":
            continue

        else:
            raise ValueError(
                "Random-abstention prediction "
                "contains an invalid decision: "
                f"{decision!r} at index {index}."
            )

    return answer_indices


def evaluate_random_selection(
    random_predictions: list[
        dict[str, Any]
    ],
    underlying_correctness: Sequence[bool],
    coverage: float,
    seed: int,
) -> dict[str, Any]:
    """
    Evaluate one already-generated random ANSWER/ABSTAIN selection.

    Underlying QA correctness is supplied separately and remains fixed across
    seeds.
    """

    if (
        len(random_predictions)
        != len(underlying_correctness)
    ):
        raise ValueError(
            "Prediction and correctness counts "
            "must match."
        )

    total = len(
        random_predictions
    )

    if total == 0:
        raise ValueError(
            "Prediction list cannot be empty."
        )

    answer_indices = (
        get_random_answer_indices(
            random_predictions
        )
    )

    answered = len(
        answer_indices
    )

    if answered == 0:
        raise ValueError(
            "Target coverage produced zero "
            "answered examples after integer "
            "rounding. Selective accuracy and "
            "risk are undefined. "
            f"total={total}, "
            f"target_coverage={coverage}"
        )

    correct_answered = sum(
        int(
            underlying_correctness[
                index
            ]
        )
        for index
        in answer_indices
    )

    answer_accuracy = (
        correct_answered
        / answered
    )

    selective_risk = (
        1.0
        - answer_accuracy
    )

    actual_coverage = (
        answered
        / total
    )

    abstain_rate = (
        1.0
        - actual_coverage
    )

    return {
        "seed": seed,
        "target_coverage": coverage,
        "actual_coverage": (
            actual_coverage
        ),
        "answered": answered,
        "answer_accuracy": (
            answer_accuracy
        ),
        "selective_risk": (
            selective_risk
        ),
        "abstain_rate": (
            abstain_rate
        ),
    }


def evaluate_one_seed(
    predictions: list[dict[str, Any]],
    coverage: float,
    seed: int,
) -> dict[str, Any]:
    """
    Evaluate one random-abstention run for one coverage and seed.

    This public function retains the historical API used by the test suite.

    For multi-seed evaluation, underlying correctness is calculated once by
    `run_multi_seed_evaluation` rather than once per seed.
    """

    validate_coverages(
        [coverage]
    )

    validate_seeds(
        [seed]
    )

    underlying_correctness = (
        infer_underlying_correctness(
            predictions
        )
    )

    random_predictions = (
        apply_random_abstention(
            predictions=predictions,
            coverage=coverage,
            seed=seed,
        )
    )

    return evaluate_random_selection(
        random_predictions=(
            random_predictions
        ),
        underlying_correctness=(
            underlying_correctness
        ),
        coverage=coverage,
        seed=seed,
    )


def validate_run_consistency(
    runs: list[dict[str, Any]],
) -> None:
    """
    Ensure all runs being summarized belong to the same coverage setting.

    Because random abstention uses a fixed integer answer count for a given
    dataset size and target coverage, answered count and actual coverage must
    also be identical across seeds.
    """

    if not runs:
        raise ValueError(
            "Run list cannot be empty."
        )

    expected_target_coverage = float(
        runs[0][
            "target_coverage"
        ]
    )

    expected_actual_coverage = float(
        runs[0][
            "actual_coverage"
        ]
    )

    expected_answered = int(
        runs[0][
            "answered"
        ]
    )

    for run in runs[1:]:
        if not math.isclose(
            float(
                run[
                    "target_coverage"
                ]
            ),
            expected_target_coverage,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Cannot summarize runs from "
                "different target coverages."
            )

        if not math.isclose(
            float(
                run[
                    "actual_coverage"
                ]
            ),
            expected_actual_coverage,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Actual coverage unexpectedly "
                "differs across seeds."
            )

        if (
            int(
                run[
                    "answered"
                ]
            )
            != expected_answered
        ):
            raise ValueError(
                "Answered count unexpectedly "
                "differs across seeds."
            )


def summarize_runs(
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Summarize repeated random-abstention runs at one target coverage.

    Mean, minimum, maximum, and population standard deviation are reported for
    answer accuracy and selective risk across the explicitly supplied seed set.

    The standard deviation is descriptive across seeds; it is not a confidence
    interval or standard error.
    """

    validate_run_consistency(
        runs
    )

    accuracies = [
        float(
            run[
                "answer_accuracy"
            ]
        )
        for run in runs
        if (
            run[
                "answer_accuracy"
            ]
            is not None
        )
    ]

    risks = [
        float(
            run[
                "selective_risk"
            ]
        )
        for run in runs
        if (
            run[
                "selective_risk"
            ]
            is not None
        )
    ]

    if (
        len(accuracies)
        != len(runs)
        or len(risks)
        != len(runs)
    ):
        raise ValueError(
            "Every random-abstention run must "
            "contain defined answer accuracy "
            "and selective risk."
        )

    for metric_name, values in (
        (
            "answer accuracy",
            accuracies,
        ),
        (
            "selective risk",
            risks,
        ),
    ):
        if any(
            (
                not math.isfinite(
                    value
                )
                or not (
                    0.0
                    <= value
                    <= 1.0
                )
            )
            for value in values
        ):
            raise ValueError(
                f"Invalid {metric_name} value "
                "found while summarizing runs."
            )

    accuracy_std = (
        statistics.pstdev(
            accuracies
        )
    )

    risk_std = (
        statistics.pstdev(
            risks
        )
    )

    return {
        "number_of_seeds": len(
            runs
        ),
        "target_coverage": (
            runs[0][
                "target_coverage"
            ]
        ),
        "actual_coverage": (
            runs[0][
                "actual_coverage"
            ]
        ),
        "answered_per_seed": (
            runs[0][
                "answered"
            ]
        ),
        "mean_answer_accuracy": (
            statistics.fmean(
                accuracies
            )
        ),
        "answer_accuracy_std": (
            accuracy_std
        ),
        "min_answer_accuracy": min(
            accuracies
        ),
        "max_answer_accuracy": max(
            accuracies
        ),
        "mean_selective_risk": (
            statistics.fmean(
                risks
            )
        ),
        "selective_risk_std": (
            risk_std
        ),
        "min_selective_risk": min(
            risks
        ),
        "max_selective_risk": max(
            risks
        ),
    }


def run_multi_seed_evaluation(
    input_path: str | Path,
    coverages: list[float],
    seeds: list[int],
) -> dict[str, Any]:
    """
    Evaluate random abstention across all requested coverages and seeds.

    Underlying QA correctness is computed exactly once before random selection.
    Every seed then changes only which candidates are answered.
    """

    validate_coverages(
        coverages
    )

    validate_seeds(
        seeds
    )

    predictions = load_jsonl(
        input_path
    )

    if not predictions:
        raise ValueError(
            "Prediction file cannot be empty."
        )

    underlying_correctness = (
        infer_underlying_correctness(
            predictions
        )
    )

    coverage_results: list[
        dict[str, Any]
    ] = []

    for coverage in coverages:
        expected_answer_count = round(
            len(predictions)
            * coverage
        )

        if expected_answer_count == 0:
            raise ValueError(
                "Coverage produces zero answered "
                "examples after random baseline "
                "rounding: "
                f"coverage={coverage}, "
                f"total={len(predictions)}."
            )

        runs: list[
            dict[str, Any]
        ] = []

        for seed in seeds:
            random_predictions = (
                apply_random_abstention(
                    predictions=predictions,
                    coverage=coverage,
                    seed=seed,
                )
            )

            run = (
                evaluate_random_selection(
                    random_predictions=(
                        random_predictions
                    ),
                    underlying_correctness=(
                        underlying_correctness
                    ),
                    coverage=coverage,
                    seed=seed,
                )
            )

            runs.append(
                run
            )

        coverage_results.append(
            {
                "coverage": coverage,
                "summary": (
                    summarize_runs(
                        runs
                    )
                ),
                "runs": runs,
            }
        )

    return {
        "input_path": str(
            input_path
        ),
        "total_predictions": len(
            predictions
        ),
        "seeds": seeds,
        "results": (
            coverage_results
        ),
    }


def save_results(
    results: dict[str, Any],
    output_path: str | Path,
) -> None:
    """
    Save multi-seed random-abstention results as formatted JSON.
    """

    save_json(
        results,
        output_path,
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parse random-abstention evaluation settings.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate random abstention across "
            "multiple deterministic seeds."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    parser.add_argument(
        "--coverages",
        type=float,
        nargs="+",
        default=DEFAULT_COVERAGES,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
    )

    return parser.parse_args()


def main() -> None:
    """
    Run multi-seed random-abstention evaluation from the command line.
    """

    args = parse_arguments()

    results = (
        run_multi_seed_evaluation(
            input_path=args.input,
            coverages=args.coverages,
            seeds=args.seeds,
        )
    )

    save_results(
        results=results,
        output_path=args.output,
    )

    print(
        "Random abstention multi-seed "
        "evaluation completed."
    )

    print(
        f"Total predictions: "
        f"{results['total_predictions']}"
    )

    print(
        f"Seeds: "
        f"{len(results['seeds'])}"
    )

    for result in results[
        "results"
    ]:
        summary = result[
            "summary"
        ]

        print(
            "\n"
            "Target coverage: "
            f"{summary['target_coverage']:.4f}\n"
            "Actual coverage: "
            f"{summary['actual_coverage']:.4f}\n"
            "Mean answer accuracy: "
            f"{summary['mean_answer_accuracy']:.4f}\n"
            "Mean selective risk: "
            f"{summary['mean_selective_risk']:.4f}\n"
            "Selective-risk population std: "
            f"{summary['selective_risk_std']:.4f}"
        )

    print(
        f"\nSaved to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()