"""
Evaluate random abstention across multiple seeds.

The evaluator runs the deterministic random-abstention policy repeatedly
and reports the distribution of selective accuracy and selective risk.
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from src.baselines.random_abstention_baseline import apply_random_abstention
from src.evaluation.evaluate_decisions import calculate_final_metrics
from src.utils.io import load_jsonl

DEFAULT_INPUT_PATH = Path("outputs/predictions/raw_baseline_calibration.jsonl")

DEFAULT_OUTPUT_PATH = Path("outputs/tables/random_abstention_multi_seed_metrics.json")


# Checks that coverage values exist and that every coverage is within the valid range: greater than 0 and at most 1.
def validate_coverages(
    coverages: list[float]
) -> None:
    if not coverages:
        raise ValueError("At least one coverage value is required.")

    for coverage in coverages:
        if not 0.0 < coverage <= 1.0:
            raise ValueError(
                f"Coverage must be greater than 0 and at most 1: {coverage}"
            )


# Checks that at least one seed is provided and that every seed is a non-negative integer.
def validate_seeds(
    seeds: list[int]
) -> None:
    if not seeds:
        raise ValueError("At least one seed is required.")

    if any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be non-negative integers.")


# Evaluates one random-abstention experiment for a specific coverage and seed, then returns its coverage, accuracy, risk, and abstention metrics.
def evaluate_one_seed(
    predictions: list[dict[str, Any]],
    coverage: float,
    seed: int
) -> dict[str, Any]:
    random_predictions = apply_random_abstention(
        predictions=predictions,
        coverage=coverage,
        seed=seed
    )

    metrics = calculate_final_metrics(random_predictions)

    final_policy = metrics["final_policy"]
    answer_metrics = final_policy["answer"]

    return {
        "seed": seed,
        "target_coverage": coverage,
        "actual_coverage": final_policy["answer_coverage"],
        "answered": answer_metrics["count"],
        "answer_accuracy": answer_metrics["accuracy"],
        "selective_risk": final_policy["selective_risk"],
        "abstain_rate": final_policy["abstain_rate"]
    }


# Summarizes multiple random-abstention runs by calculating the mean, standard deviation, minimum, and maximum accuracy and risk across seeds.
def summarize_runs(
    runs: list[dict[str, Any]]
) -> dict[str, Any]:
    if not runs:
        raise ValueError("Run list cannot be empty.")

    accuracies = [
        float(run["answer_accuracy"])
        for run in runs
        if run["answer_accuracy"] is not None
    ]

    risks = [
        float(run["selective_risk"])
        for run in runs
        if run["selective_risk"] is not None
    ]

    if not accuracies or not risks:
        raise ValueError(
            "Accuracy and risk cannot be summarized when no examples are answered."
        )

    risk_std = statistics.pstdev(risks)
    accuracy_std = statistics.pstdev(accuracies)

    return {
        "number_of_seeds": len(runs),
        "target_coverage": runs[0]["target_coverage"],
        "actual_coverage": runs[0]["actual_coverage"],
        "answered_per_seed": runs[0]["answered"],
        "mean_answer_accuracy": statistics.fmean(accuracies),
        "answer_accuracy_std": accuracy_std,
        "min_answer_accuracy": min(accuracies),
        "max_answer_accuracy": max(accuracies),
        "mean_selective_risk": statistics.fmean(risks),
        "selective_risk_std": risk_std,
        "min_selective_risk": min(risks),
        "max_selective_risk": max(risks)
    }


# Runs random-abstention evaluation across multiple coverage levels and random seeds, then summarizes and returns all experiment results.
def run_multi_seed_evaluation(
    input_path: str | Path,
    coverages: list[float],
    seeds: list[int]
) -> dict[str, Any]:
    validate_coverages(coverages)
    validate_seeds(seeds)

    predictions = load_jsonl(input_path)

    if not predictions:
        raise ValueError("Prediction file cannot be empty.")

    coverage_results: list[dict[str, Any]] = []

    for coverage in coverages:
        runs = [
            evaluate_one_seed(
                predictions=predictions,
                coverage=coverage,
                seed=seed
            )
            for seed in seeds
        ]

        coverage_results.append(
            {
                "coverage": coverage,
                "summary": summarize_runs(runs),
                "runs": runs
            }
        )

    return {
        "input_path": str(input_path),
        "total_predictions": len(predictions),
        "seeds": seeds,
        "results": coverage_results
    }


# Saves the evaluation results as a formatted JSON file, creating the output directory first if necessary.
def save_results(
    results: dict[str, Any],
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
            results,
            output_file,
            indent=2,
            ensure_ascii=False
        )


# Defines command-line arguments for the input file, output file, coverage levels, and random seeds used in the multi-seed evaluation.
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate random abstention across multiple seeds."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH
    )

    parser.add_argument(
        "--coverages",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(20))
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    results = run_multi_seed_evaluation(
        input_path=args.input,
        coverages=args.coverages,
        seeds=args.seeds
    )

    save_results(
        results=results,
        output_path=args.output
    )

    print("Random abstention multi-seed evaluation completed.")
    print(f"Total predictions: {results['total_predictions']}")
    print(f"Seeds: {len(results['seeds'])}")

    for result in results["results"]:
        summary = result["summary"]

        print(
            "\n"
            f"Coverage: {summary['actual_coverage']:.4f}\n"
            f"Mean accuracy: {summary['mean_answer_accuracy']:.4f}\n"
            f"Mean risk: {summary['mean_selective_risk']:.4f}\n"
            f"Risk std: {summary['selective_risk_std']:.4f}"
        )

    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
