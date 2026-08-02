"""
The main purpose of this file is to create a random abstention baseline for comparison.
It randomly selects which predictions will be answered according to a target coverage, marks the rest as ABSTAIN, calculates decision statistics, and saves the results. Unlike the confidence baseline, it does not use model confidence—its decisions are random but reproducible with a fixed seed.
"""

# argparse is used to read command-line arguments when you run the script from the terminal.
import argparse
import random

# is used to create and manage file paths safely.
from pathlib import Path

# Any is a flexible type hint that allows a variable or dictionary value to contain any Python data type.
from typing import Any

from src.utils.io import load_jsonl, save_jsonl

DEFAULT_INPUT_PATH = Path("outputs/predictions/raw_baseline_calibration.jsonl")
DEFAULT_OUTPUT_DIR = Path("outputs/predictions")
DEFAULT_SEED = 17


# randomly decides whether to ANSWER or ABSTAIN for each prediction to achieve a target coverage.
# It updates the predictions with these random decisions and returns the modified list.
def apply_random_abstention(
    predictions: list[dict[str, Any]], coverage: float, seed: int = DEFAULT_SEED
) -> list[dict[str, Any]]:
    if not 0.0 <= coverage <= 1.0:
        raise ValueError("Coverage must be between 0 and 1.")

    total_predictions = len(predictions)
    answer_count = round(total_predictions * coverage)

    # Creates a separate random number generator with a fixed seed.
    # The same seed produces the same random choices every time, which makes the experiment reproducible without changing Python's global random state.

    # Example:
    # rng = random.Random(17)
    # rng.sample(range(10), k=3)  -> always selects the same 3 indices
    rng = random.Random(seed)

    # stores the indices of predictions that the system will answer.
    # Example: if it becomes {1, 4, 7}, only predictions at indices 1, 4, and 7 will be marked ANSWER; the others will be ABSTAIN.
    answer_indices = set(rng.sample(range(total_predictions), k=answer_count))

    updated_predictions: list[dict[str, Any]] = []

    for index, prediction in enumerate(predictions):
        should_answer = index in answer_indices

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


# calculates overall results from the prediction list.
def summarize_decisions(predictions: list[dict[str, Any]]) -> dict[str, float | int]:
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


# loads raw predictions, randomly chooses which ones to answer according to the target coverage, calculates summary statistics, and saves the updated predictions as a JSONL file.
def run_random_abstention_baseline(
    input_path: str | Path,
    coverage: float,
    seed: int = DEFAULT_SEED,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> list[dict[str, Any]]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    raw_predictions = load_jsonl(input_path)

    predictions = apply_random_abstention(
        predictions=raw_predictions, coverage=coverage, seed=seed
    )

    summary = summarize_decisions(predictions)

    output_dir.mkdir(parents=True, exist_ok=True)

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


# reads the random-abstention settings from the terminal.
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply random abstention to raw QA predictions."
    )

    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))

    parser.add_argument("--coverage", type=float, default=0.50)

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    run_random_abstention_baseline(
        input_path=arguments.input,
        coverage=arguments.coverage,
        seed=arguments.seed,
        output_dir=arguments.output_dir,
    )
