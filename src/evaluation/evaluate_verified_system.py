"""
Evaluates the final verified selective-QA system after the decision engine and summarizes its ANSWER, VERIFY, ABSTAIN, coverage, accuracy, and selective-risk behavior.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from src.evaluation.evaluate_decisions import (
    calculate_final_metrics,
    validate_predictions,
)
from src.utils.io import load_jsonl


DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/calibration_final_decisions.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "outputs/tables/verified_system_metrics.json"
)


# Builds a summary of the verified selective-QA system by reporting ANSWER, VERIFY, and ABSTAIN performance and comparing it with the threshold-only policy.
def build_verified_summary(
    predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    validate_predictions(predictions)

    metrics = calculate_final_metrics(predictions)

    final_policy = metrics["final_policy"]

    answer_metrics = final_policy["answer"]
    verify_metrics = final_policy["verify"]
    abstain_metrics = final_policy["abstain"]

    return {
        "evaluation_type": "verified_selective_qa_system",
        "total_predictions": metrics["total_predictions"],
        "full_coverage_baseline_accuracy": (
            metrics["baseline_full_coverage_accuracy"]
        ),
        "verified_policy": {
            "answer_count": answer_metrics["count"],
            "answer_coverage": final_policy["answer_coverage"],
            "answer_accuracy": answer_metrics["accuracy"],
            "selective_risk": final_policy["selective_risk"],
            "verify_count": verify_metrics["count"],
            "verify_rate": final_policy["verify_rate"],
            "verify_accuracy": verify_metrics["accuracy"],
            "abstain_count": abstain_metrics["count"],
            "abstain_rate": final_policy["abstain_rate"],
            "abstain_accuracy": abstain_metrics["accuracy"],
        },
        "threshold_only_policy": metrics.get(
            "threshold_only_policy"
        ),
        "policy_comparison": metrics.get(
            "policy_comparison"
        ),
        "evidence_distribution": metrics.get(
            "evidence_distribution",
            {},
        ),
        "decision_reason_distribution": metrics.get(
            "decision_reason_distribution",
            {}
        )
    }



# Saves the verified evaluation summary as a formatted JSON file, creating the output directory first if necessary.
def save_summary(
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



# Converts an optional numeric metric into a readable string, returning "N/A" for missing values and formatting numbers to four decimal places.
def format_optional_metric(
    value: float | None
) -> str:
    if value is None:
        return "N/A"

    return f"{value:.4f}"



# Prints a readable summary of the verified selective-QA evaluation, including baseline accuracy, ANSWER/VERIFY/ABSTAIN statistics, selective risk, and changes from the threshold-only policy.
def print_verified_summary(
    summary: dict[str, Any]
) -> None:
    verified_policy = summary["verified_policy"]

    print("\nVerified system evaluation completed.")

    print(
        f"Total predictions: "
        f"{summary['total_predictions']}"
    )

    print(
        "Full-coverage baseline accuracy: "
        f"{format_optional_metric(
            summary['full_coverage_baseline_accuracy']
        )}"
    )

    print("\nVerified selective policy:")

    print(
        f"ANSWER count: "
        f"{verified_policy['answer_count']}"
    )

    print(
        "Answer coverage: "
        f"{format_optional_metric(
            verified_policy['answer_coverage']
        )}"
    )

    print(
        "Answer accuracy: "
        f"{format_optional_metric(
            verified_policy['answer_accuracy']
        )}"
    )

    print(
        "Selective risk: "
        f"{format_optional_metric(
            verified_policy['selective_risk']
        )}"
    )

    print(
        f"VERIFY count: "
        f"{verified_policy['verify_count']}"
    )

    print(
        "Verify rate: "
        f"{format_optional_metric(
            verified_policy['verify_rate']
        )}"
    )

    print(
        f"ABSTAIN count: "
        f"{verified_policy['abstain_count']}"
    )

    print(
        "Abstain rate: "
        f"{format_optional_metric(
            verified_policy['abstain_rate']
        )}"
    )

    policy_comparison = summary.get(
        "policy_comparison"
    )

    if policy_comparison is not None:
        print("\nChange from threshold-only policy:")

        print(
            "Risk change: "
            f"{format_optional_metric(
                policy_comparison['risk_change']
            )}"
        )

        print(
            "Coverage change: "
            f"{format_optional_metric(
                policy_comparison['coverage_change']
            )}"
        )


# Runs the complete verified-system evaluation by loading predictions, validating them, building the summary, saving the results, printing key metrics, and returning the summary.
def run_verified_system_evaluation(
    input_path: str | Path,
    output_path: str | Path
) -> dict[str, Any]:
    predictions = load_jsonl(input_path)

    if not predictions:
        raise ValueError(
            "Prediction file cannot be empty."
        )

    summary = build_verified_summary(
        predictions
    )

    save_summary(
        summary=summary,
        output_path=output_path,
    )

    print_verified_summary(summary)

    print(
        f"\nMetrics saved to: "
        f"{output_path}"
    )

    return summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the final verified "
            "selective-QA system."
        )
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

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    run_verified_system_evaluation(
        input_path=args.input,
        output_path=args.output
    )


if __name__ == "__main__":
    main()