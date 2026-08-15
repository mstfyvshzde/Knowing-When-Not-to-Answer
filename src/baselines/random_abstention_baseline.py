"""
Run a reproducible random-abstention control baseline.

This baseline ignores all model-quality signals such as confidence or
verification scores. Instead, it randomly chooses which examples to answer
until the requested target coverage is reached approximately.

Coverage (kapsama oranı) is the fraction of examples for which the system
chooses to answer. The remaining examples are assigned ABSTAIN.

Because the selection is random, this baseline acts as a control condition:
it shows whether confidence-based ranking performs better than simply choosing
the same number of examples at random.

A fixed seed makes the random selection reproducible across runs.
"""


import argparse
import random
from pathlib import Path
from typing import Any

from src.utils.io import load_jsonl, save_jsonl

# Default inputs and reproducibility settings for the random control baseline.
DEFAULT_INPUT_PATH = Path("outputs/predictions/raw_baseline_calibration.jsonl")
DEFAULT_OUTPUT_DIR = Path("outputs/predictions")
DEFAULT_SEED = 17


def apply_random_abstention(
    predictions: list[dict[str, Any]], coverage: float, seed: int = DEFAULT_SEED
) -> list[dict[str, Any]]:
    """
    Assign ANSWER or ABSTAIN decisions randomly at a target coverage.

    Coverage (kapsama oranı) determines what fraction of all predictions should
    receive an ANSWER decision.

    The number of answered examples is:

        round(number_of_predictions * coverage)

    The fixed random seed makes the selected example indices reproducible.
    Model confidence and verifier scores are deliberately ignored.
    """

    if not 0.0 <= coverage <= 1.0:
        raise ValueError("Coverage must be between 0 and 1.")

    total_predictions = len(predictions)
    answer_count = round(total_predictions * coverage)

    # Use a local random generator with a fixed seed.
    # Local generator (yerel random üreteci) means this baseline gets reproducible
    # random choices without changing the global random state used elsewhere in
    # the project.
    rng = random.Random(seed)


    # Randomly choose exactly answer_count example indices.
    # Every selected index becomes ANSWER; every remaining index becomes ABSTAIN.
    answer_indices = set(rng.sample(range(total_predictions), k=answer_count))

    updated_predictions: list[dict[str, Any]] = []

    for index, prediction in enumerate(predictions):
        # The decision depends only on random selection. Model scores are intentionally
        # ignored so this remains a true random control baseline.
        should_answer = index in answer_indices

        # Preserve the original QA prediction and append the random baseline decision.
        updated_prediction = prediction.copy()
        updated_prediction.update(
            {
                "decision": "ANSWER" if should_answer else "ABSTAIN",
                "final_decision": "ANSWER" if should_answer else "ABSTAIN",
                "final_answer": (
                    prediction.get("prediction_text", "")
                    if should_answer
                    else "I do not know"
                ),
                "target_coverage": coverage,
                "seed": seed,
                "system": "random_abstention_baseline",
            }
        )

        updated_predictions.append(updated_prediction)

    return updated_predictions


def summarize_decisions(predictions: list[dict[str, Any]]) -> dict[str, float | int]:
    """
    Summarize the ANSWER/ABSTAIN distribution produced by the baseline.

    Actual coverage is calculated as answered_examples / total_examples.
    Because the requested number of answers is rounded to an integer, actual
    coverage may differ very slightly from the requested target coverage.
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


def run_random_abstention_baseline(
    input_path: str | Path,
    coverage: float,
    seed: int = DEFAULT_SEED,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> list[dict[str, Any]]:
    """
    Apply random abstention to stored QA predictions and save the decisions.
    """

    input_path = Path(input_path)
    output_dir = Path(output_dir)

    # Start from the same forced-answer QA predictions used by confidence-based
    # methods so the comparison changes only the selection strategy.
    raw_predictions = load_jsonl(input_path)

    predictions = apply_random_abstention(
        predictions=raw_predictions, coverage=coverage, seed=seed
    )

    summary = summarize_decisions(predictions)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Encode the requested coverage and seed in the filename for traceability.
    coverage_name = str(coverage).replace(".", "-")

    output_path = output_dir / f"random_abstention_{coverage_name}_seed_{seed}.jsonl"

    save_jsonl(predictions, output_path)

    print("\nRandom abstention baseline completed.")
    print(f"Target coverage: {coverage:.2f}")
    print(f"Actual coverage: {summary['coverage']:.4f}")
    print(f"Answered: " f"{summary['answered_examples']}/{summary['total_examples']}")
    print(f"Abstention rate: {summary['abstention_rate']:.4f}")
    print(f"Seed: {seed}")
    print(f"Saved to: {output_path}")

    return predictions


def parse_arguments() -> argparse.Namespace:
    """Parse random-abstention baseline settings from the command line."""

    parser = argparse.ArgumentParser(
        description="Run the reproducible random-abstention control baseline."
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to the forced-answer prediction JSONL file.",
    )

    parser.add_argument(
        "--coverage",
        type=float,
        default=0.50,
        help="Target fraction of examples that should receive an ANSWER decision.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used to select answered examples.",
    )

    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where baseline predictions will be saved.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    run_random_abstention_baseline(
        input_path=arguments.input,
        coverage=arguments.coverage,
        seed=arguments.seed,
        output_dir=arguments.output_dir,
    )
