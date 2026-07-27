"""
Random abstention baseline.

This baseline randomly selects an exact number of predictions according to the
requested coverage and abstains otherwise.

It is intended as a control baseline for selective question answering.
"""

import argparse
import random
from pathlib import Path
from typing import Any

from src.utils.io import load_jsonl, save_jsonl

DEFAULT_INPUT_PATH = Path("outputs/predictions/raw_baseline_calibration.jsonl")
DEFAULT_OUTPUT_DIR = Path("outputs/predictions")
DEFAULT_SEED = 42


def apply_random_abstention(
    predictions: list[dict[str, Any]],
    coverage: float,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """
    Randomly select an exact number of predictions to answer.
    """

    if not 0.0 <= coverage <= 1.0:
        raise ValueError("Coverage must be between 0 and 1.")

    total_predictions = len(predictions)
    answer_count = round(total_predictions * coverage)

    rng = random.Random(seed)
    answer_indices = set(
        rng.sample(
            range(total_predictions),
            k=answer_count,
        )
    )

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


def summarize_decisions(
    predictions: list[dict[str, Any]],
) -> dict[str, float | int]:
    """
    Summarize answer and abstention counts.
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
    Load predictions, apply random abstention, and save the result.
    """

    input_path = Path(input_path)
    output_dir = Path(output_dir)

    raw_predictions = load_jsonl(input_path)

    predictions = apply_random_abstention(
        predictions=raw_predictions,
        coverage=coverage,
        seed=seed,
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


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Apply random abstention to raw QA predictions."
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
    )

    parser.add_argument(
        "--coverage",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
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
