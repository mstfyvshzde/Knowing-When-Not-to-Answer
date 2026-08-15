"""
Summarize the rule-based verified selective-QA prototype.

This module is a reporting and consistency-checking layer over
`evaluate_decisions.py`.

It does not implement a second correctness definition or selective-risk
formula. All numerical evaluation is delegated to `calculate_final_metrics`.

The prototype contains three final routing states:

- ANSWER
- VERIFY
- ABSTAIN

Underlying QA correctness
-------------------------
The shared decision evaluator uses the same forced-answer correctness definition
as confidence-threshold selection.

Under that definition:

- an answerable QA candidate is correct only when its normalized predicted span
  exactly matches a reference answer;
- an unanswerable forced-answer candidate is incorrect.

Later routing decisions do not change that underlying correctness label.

For example, an incorrect unanswerable candidate that is later routed to
ABSTAIN remains an incorrect underlying QA candidate. The abstention can be a
desirable routing action without making the original candidate correct.

Metric interpretation
---------------------
`full_coverage_baseline_accuracy` is retained for output compatibility. It
means full-coverage forced-answer QA candidate accuracy before selective
routing.

Within `verified_policy`:

- `answer_accuracy` is underlying QA candidate accuracy among final ANSWER
  records;
- `verify_accuracy` is underlying QA candidate accuracy among final VERIFY
  records;
- `abstain_accuracy` is underlying QA candidate accuracy among final ABSTAIN
  records.

Therefore VERIFY/ABSTAIN accuracy does not measure whether the routing action
itself was correct.

Selective risk is defined only for final ANSWER records:

    selective_risk = 1 - answer_accuracy

The comparison with the original confidence-threshold routing is an
operating-point comparison. Final ANSWER coverage can change when VERIFY
examples are promoted or blocked, so the reported risk change is not a
matched-coverage effect.

This file evaluates the earlier rule-based prototype and should not be confused
with the project's final score-ranking/AURC experiments.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from src.evaluation.evaluate_decisions import (
    calculate_final_metrics,
    validate_predictions,
)
from src.utils.io import (
    load_jsonl,
    save_json,
)

DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/calibration_final_decisions.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "outputs/tables/verified_system_metrics.json"
)


def build_verified_summary(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build a compact summary of the rule-based verified prototype.

    All correctness, coverage, and risk calculations are delegated to
    `calculate_final_metrics` so this reporting layer cannot drift into a
    competing metric implementation.
    """

    validate_predictions(
        predictions
    )

    metrics = calculate_final_metrics(
        predictions
    )

    final_policy = metrics[
        "final_policy"
    ]

    answer_metrics = final_policy[
        "answer"
    ]

    verify_metrics = final_policy[
        "verify"
    ]

    abstain_metrics = final_policy[
        "abstain"
    ]

    summary = {
        "evaluation_type": (
            "verified_selective_qa_system"
        ),
        "analysis_scope": (
            "prototype fixed-threshold "
            "rule-based routing evaluation"
        ),
        "correctness_definition": (
            metrics.get(
                "correctness_definition",
                (
                    "forced-answer Exact-Match "
                    "correctness shared with "
                    "calibration threshold selection"
                ),
            )
        ),
        "total_predictions": (
            metrics[
                "total_predictions"
            ]
        ),
        "full_coverage_baseline_accuracy": (
            metrics[
                "baseline_full_coverage_accuracy"
            ]
        ),
        "verified_policy": {
            "answer_count": (
                answer_metrics[
                    "count"
                ]
            ),
            "answer_coverage": (
                final_policy[
                    "answer_coverage"
                ]
            ),
            "answer_accuracy": (
                answer_metrics[
                    "accuracy"
                ]
            ),
            "selective_risk": (
                final_policy[
                    "selective_risk"
                ]
            ),
            "verify_count": (
                verify_metrics[
                    "count"
                ]
            ),
            "verify_rate": (
                final_policy[
                    "verify_rate"
                ]
            ),
            "verify_accuracy": (
                verify_metrics[
                    "accuracy"
                ]
            ),
            "abstain_count": (
                abstain_metrics[
                    "count"
                ]
            ),
            "abstain_rate": (
                final_policy[
                    "abstain_rate"
                ]
            ),
            "abstain_accuracy": (
                abstain_metrics[
                    "accuracy"
                ]
            ),
        },
        "threshold_only_policy": (
            metrics.get(
                "threshold_only_policy"
            )
        ),
        "policy_comparison": (
            metrics.get(
                "policy_comparison"
            )
        ),
        "evidence_distribution": (
            metrics.get(
                "evidence_distribution",
                {},
            )
        ),
        "decision_reason_distribution": (
            metrics.get(
                "decision_reason_distribution",
                {},
            )
        ),
    }

    validate_summary_consistency(
        summary
    )

    return summary


def validate_probability_metric(
    value: Any,
    metric_name: str,
    allow_none: bool = False,
) -> None:
    """
    Validate an optional or required probability/rate metric in [0, 1].
    """

    if value is None:
        if allow_none:
            return

        raise RuntimeError(
            f"{metric_name} must be defined."
        )

    try:
        numeric_value = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise RuntimeError(
            f"{metric_name} must be numeric."
        ) from error

    if (
        not math.isfinite(
            numeric_value
        )
        or not (
            0.0
            <= numeric_value
            <= 1.0
        )
    ):
        raise RuntimeError(
            f"{metric_name} must be a finite "
            "value in [0, 1]."
        )


def validate_distribution_total(
    distribution: dict[str, Any],
    expected_total: int,
    distribution_name: str,
) -> None:
    """
    Require a count distribution to account for every prediction.
    """

    total = 0

    for key, value in (
        distribution.items()
    ):
        if (
            not isinstance(
                value,
                int,
            )
            or isinstance(
                value,
                bool,
            )
            or value < 0
        ):
            raise RuntimeError(
                f"{distribution_name} contains "
                f"invalid count for {key!r}: "
                f"{value!r}."
            )

        total += value

    if total != expected_total:
        raise RuntimeError(
            f"{distribution_name} counts do not "
            "sum to total predictions: "
            f"{total} != {expected_total}."
        )


def validate_verified_policy(
    verified_policy: dict[str, Any],
    total: int,
) -> None:
    """
    Validate counts, rates, and ANSWER selective-risk consistency.
    """

    answer_count = int(
        verified_policy[
            "answer_count"
        ]
    )

    verify_count = int(
        verified_policy[
            "verify_count"
        ]
    )

    abstain_count = int(
        verified_policy[
            "abstain_count"
        ]
    )

    for name, count in (
        (
            "answer_count",
            answer_count,
        ),
        (
            "verify_count",
            verify_count,
        ),
        (
            "abstain_count",
            abstain_count,
        ),
    ):
        if count < 0:
            raise RuntimeError(
                f"{name} cannot be negative."
            )

    routed_total = (
        answer_count
        + verify_count
        + abstain_count
    )

    if routed_total != total:
        raise RuntimeError(
            "Final decision counts do not sum "
            "to total predictions: "
            f"ANSWER={answer_count}, "
            f"VERIFY={verify_count}, "
            f"ABSTAIN={abstain_count}, "
            f"total={total}."
        )

    answer_coverage = (
        verified_policy[
            "answer_coverage"
        ]
    )

    verify_rate = (
        verified_policy[
            "verify_rate"
        ]
    )

    abstain_rate = (
        verified_policy[
            "abstain_rate"
        ]
    )

    for name, value in (
        (
            "answer_coverage",
            answer_coverage,
        ),
        (
            "verify_rate",
            verify_rate,
        ),
        (
            "abstain_rate",
            abstain_rate,
        ),
    ):
        validate_probability_metric(
            value,
            name,
        )

    if not math.isclose(
        (
            float(
                answer_coverage
            )
            + float(
                verify_rate
            )
            + float(
                abstain_rate
            )
        ),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "ANSWER, VERIFY, and ABSTAIN rates "
            "do not sum to 1.0."
        )

    expected_answer_coverage = (
        answer_count
        / total
    )

    expected_verify_rate = (
        verify_count
        / total
    )

    expected_abstain_rate = (
        abstain_count
        / total
    )

    for name, observed, expected in (
        (
            "answer_coverage",
            float(
                answer_coverage
            ),
            expected_answer_coverage,
        ),
        (
            "verify_rate",
            float(
                verify_rate
            ),
            expected_verify_rate,
        ),
        (
            "abstain_rate",
            float(
                abstain_rate
            ),
            expected_abstain_rate,
        ),
    ):
        if not math.isclose(
            observed,
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"{name} is inconsistent "
                "with its decision count."
            )

    answer_accuracy = (
        verified_policy[
            "answer_accuracy"
        ]
    )

    verify_accuracy = (
        verified_policy[
            "verify_accuracy"
        ]
    )

    abstain_accuracy = (
        verified_policy[
            "abstain_accuracy"
        ]
    )

    validate_probability_metric(
        answer_accuracy,
        "answer_accuracy",
        allow_none=(
            answer_count == 0
        ),
    )

    validate_probability_metric(
        verify_accuracy,
        "verify_accuracy",
        allow_none=(
            verify_count == 0
        ),
    )

    validate_probability_metric(
        abstain_accuracy,
        "abstain_accuracy",
        allow_none=(
            abstain_count == 0
        ),
    )

    selective_risk = (
        verified_policy[
            "selective_risk"
        ]
    )

    if answer_count == 0:
        if (
            answer_accuracy is not None
            or selective_risk is not None
        ):
            raise RuntimeError(
                "ANSWER accuracy and selective "
                "risk must be undefined when "
                "no examples are answered."
            )

        return

    if (
        answer_accuracy is None
        or selective_risk is None
    ):
        raise RuntimeError(
            "ANSWER accuracy and selective "
            "risk must be defined when ANSWER "
            "count is positive."
        )

    validate_probability_metric(
        selective_risk,
        "selective_risk",
    )

    if not math.isclose(
        float(
            selective_risk
        ),
        1.0
        - float(
            answer_accuracy
        ),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Selective risk is inconsistent "
            "with ANSWER accuracy."
        )


def validate_threshold_policy(
    threshold_policy: dict[str, Any],
    total: int,
) -> None:
    """
    Validate the reconstructed original confidence-threshold routing policy.
    """

    if int(
        threshold_policy[
            "total"
        ]
    ) != total:
        raise RuntimeError(
            "Threshold-policy total does not "
            "match summary total."
        )

    answer_count = int(
        threshold_policy[
            "answer_count"
        ]
    )

    verify_count = int(
        threshold_policy[
            "verify_count"
        ]
    )

    abstain_count = int(
        threshold_policy[
            "abstain_count"
        ]
    )

    if (
        answer_count
        + verify_count
        + abstain_count
        != total
    ):
        raise RuntimeError(
            "Threshold ANSWER, VERIFY, and "
            "ABSTAIN counts do not sum to total."
        )

    answer_coverage = (
        threshold_policy[
            "answer_coverage"
        ]
    )

    verify_rate = (
        threshold_policy[
            "verify_rate"
        ]
    )

    abstain_rate = (
        threshold_policy[
            "abstain_rate"
        ]
    )

    for name, value in (
        (
            "threshold answer coverage",
            answer_coverage,
        ),
        (
            "threshold verify rate",
            verify_rate,
        ),
        (
            "threshold abstain rate",
            abstain_rate,
        ),
    ):
        validate_probability_metric(
            value,
            name,
        )

    if not math.isclose(
        (
            float(
                answer_coverage
            )
            + float(
                verify_rate
            )
            + float(
                abstain_rate
            )
        ),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Threshold-routing rates do not "
            "sum to 1.0."
        )

    expected_rates = {
        "answer": (
            answer_count
            / total
        ),
        "verify": (
            verify_count
            / total
        ),
        "abstain": (
            abstain_count
            / total
        ),
    }

    for name, observed, expected in (
        (
            "threshold answer coverage",
            float(
                answer_coverage
            ),
            expected_rates[
                "answer"
            ],
        ),
        (
            "threshold verify rate",
            float(
                verify_rate
            ),
            expected_rates[
                "verify"
            ],
        ),
        (
            "threshold abstain rate",
            float(
                abstain_rate
            ),
            expected_rates[
                "abstain"
            ],
        ),
    ):
        if not math.isclose(
            observed,
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"{name} is inconsistent "
                "with its count."
            )

    answer_accuracy = (
        threshold_policy[
            "answer_accuracy"
        ]
    )

    selective_risk = (
        threshold_policy[
            "selective_risk"
        ]
    )

    if answer_count == 0:
        if (
            answer_accuracy is not None
            or selective_risk is not None
        ):
            raise RuntimeError(
                "Threshold ANSWER accuracy and "
                "risk must be undefined when "
                "ANSWER count is zero."
            )

        return

    validate_probability_metric(
        answer_accuracy,
        "threshold answer accuracy",
    )

    validate_probability_metric(
        selective_risk,
        "threshold selective risk",
    )

    if not math.isclose(
        float(
            selective_risk
        ),
        1.0
        - float(
            answer_accuracy
        ),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Threshold selective risk is "
            "inconsistent with threshold "
            "ANSWER accuracy."
        )


def validate_policy_comparison(
    summary: dict[str, Any],
) -> None:
    """
    Check that reported changes equal final minus threshold operating point.
    """

    comparison = summary.get(
        "policy_comparison"
    )

    threshold_policy = summary.get(
        "threshold_only_policy"
    )

    if comparison is None:
        if (
            threshold_policy
            is not None
        ):
            raise RuntimeError(
                "Threshold policy is available "
                "but policy_comparison is missing."
            )

        return

    if threshold_policy is None:
        raise RuntimeError(
            "policy_comparison exists without "
            "threshold_only_policy."
        )

    verified_policy = summary[
        "verified_policy"
    ]

    final_risk = (
        verified_policy[
            "selective_risk"
        ]
    )

    threshold_risk = (
        threshold_policy[
            "selective_risk"
        ]
    )

    final_coverage = (
        verified_policy[
            "answer_coverage"
        ]
    )

    threshold_coverage = (
        threshold_policy[
            "answer_coverage"
        ]
    )

    observed_risk_change = (
        comparison.get(
            "risk_change"
        )
    )

    if (
        final_risk is None
        or threshold_risk is None
    ):
        if (
            observed_risk_change
            is not None
        ):
            raise RuntimeError(
                "Risk change must be undefined "
                "when either operating-point "
                "risk is undefined."
            )

    else:
        expected_risk_change = (
            float(
                final_risk
            )
            - float(
                threshold_risk
            )
        )

        if (
            observed_risk_change
            is None
            or not math.isclose(
                float(
                    observed_risk_change
                ),
                expected_risk_change,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError(
                "policy_comparison risk_change "
                "is inconsistent with final and "
                "threshold risks."
            )

    observed_coverage_change = (
        comparison.get(
            "coverage_change"
        )
    )

    expected_coverage_change = (
        float(
            final_coverage
        )
        - float(
            threshold_coverage
        )
    )

    if (
        observed_coverage_change
        is None
        or not math.isclose(
            float(
                observed_coverage_change
            ),
            expected_coverage_change,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError(
            "policy_comparison coverage_change "
            "is inconsistent with final and "
            "threshold coverage."
        )


def validate_summary_consistency(
    summary: dict[str, Any],
) -> None:
    """
    Check structural and mathematical invariants in the summary.

    These checks do not alter metrics. They make malformed or internally
    inconsistent upstream results fail before being written as research output.
    """

    total = int(
        summary[
            "total_predictions"
        ]
    )

    if total <= 0:
        raise ValueError(
            "Verified-system summary must "
            "contain at least one prediction."
        )

    validate_probability_metric(
        summary[
            "full_coverage_baseline_accuracy"
        ],
        "full_coverage_baseline_accuracy",
    )

    validate_verified_policy(
        verified_policy=(
            summary[
                "verified_policy"
            ]
        ),
        total=total,
    )

    threshold_policy = (
        summary.get(
            "threshold_only_policy"
        )
    )

    if (
        threshold_policy
        is not None
    ):
        validate_threshold_policy(
            threshold_policy=(
                threshold_policy
            ),
            total=total,
        )

    validate_policy_comparison(
        summary
    )

    validate_distribution_total(
        distribution=(
            summary[
                "evidence_distribution"
            ]
        ),
        expected_total=total,
        distribution_name=(
            "evidence_distribution"
        ),
    )

    validate_distribution_total(
        distribution=(
            summary[
                "decision_reason_distribution"
            ]
        ),
        expected_total=total,
        distribution_name=(
            "decision_reason_distribution"
        ),
    )


def save_summary(
    summary: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save the verified-system summary with shared repository JSON I/O."""

    save_json(
        summary,
        output_path,
    )


def format_optional_metric(
    value: float | None,
) -> str:
    """Format an optional metric to four decimal places."""

    if value is None:
        return "N/A"

    return f"{value:.4f}"


def print_verified_summary(
    summary: dict[str, Any],
) -> None:
    """Print the prototype summary using precise scientific labels."""

    verified_policy = (
        summary[
            "verified_policy"
        ]
    )

    print(
        "\nVerified prototype "
        "evaluation completed."
    )

    print(
        f"Total predictions: "
        f"{summary['total_predictions']}"
    )

    print(
        "Full-coverage forced-answer "
        "QA accuracy: "
        f"{format_optional_metric(
            summary[
                'full_coverage_baseline_accuracy'
            ]
        )}"
    )

    print(
        "\nFinal rule-based routing:"
    )

    print(
        f"ANSWER count: "
        f"{verified_policy['answer_count']}"
    )

    print(
        "Answer coverage: "
        f"{format_optional_metric(
            verified_policy[
                'answer_coverage'
            ]
        )}"
    )

    print(
        "ANSWER-group forced-answer "
        "QA accuracy: "
        f"{format_optional_metric(
            verified_policy[
                'answer_accuracy'
            ]
        )}"
    )

    print(
        "Selective risk: "
        f"{format_optional_metric(
            verified_policy[
                'selective_risk'
            ]
        )}"
    )

    print(
        f"VERIFY count: "
        f"{verified_policy['verify_count']}"
    )

    print(
        "Verify rate: "
        f"{format_optional_metric(
            verified_policy[
                'verify_rate'
            ]
        )}"
    )

    print(
        "VERIFY-group forced-answer "
        "QA accuracy: "
        f"{format_optional_metric(
            verified_policy[
                'verify_accuracy'
            ]
        )}"
    )

    print(
        f"ABSTAIN count: "
        f"{verified_policy['abstain_count']}"
    )

    print(
        "Abstain rate: "
        f"{format_optional_metric(
            verified_policy[
                'abstain_rate'
            ]
        )}"
    )

    print(
        "ABSTAIN-group forced-answer "
        "QA accuracy: "
        f"{format_optional_metric(
            verified_policy[
                'abstain_accuracy'
            ]
        )}"
    )

    threshold_policy = (
        summary.get(
            "threshold_only_policy"
        )
    )

    if (
        threshold_policy
        is not None
    ):
        print(
            "\nOriginal confidence-threshold "
            "routing:"
        )

        print(
            "Direct-ANSWER coverage: "
            f"{format_optional_metric(
                threshold_policy[
                    'answer_coverage'
                ]
            )}"
        )

        print(
            "Direct-ANSWER forced-answer "
            "QA accuracy: "
            f"{format_optional_metric(
                threshold_policy[
                    'answer_accuracy'
                ]
            )}"
        )

        print(
            "Direct-ANSWER selective risk: "
            f"{format_optional_metric(
                threshold_policy[
                    'selective_risk'
                ]
            )}"
        )

        print(
            "VERIFY rate: "
            f"{format_optional_metric(
                threshold_policy[
                    'verify_rate'
                ]
            )}"
        )

        print(
            "ABSTAIN rate: "
            f"{format_optional_metric(
                threshold_policy[
                    'abstain_rate'
                ]
            )}"
        )

    policy_comparison = (
        summary.get(
            "policy_comparison"
        )
    )

    if (
        policy_comparison
        is not None
    ):
        print(
            "\nEvidence-aware operating-point "
            "change (not matched coverage):"
        )

        print(
            "Risk change: "
            f"{format_optional_metric(
                policy_comparison[
                    'risk_change'
                ]
            )}"
        )

        print(
            "Coverage change: "
            f"{format_optional_metric(
                policy_comparison[
                    'coverage_change'
                ]
            )}"
        )


def run_verified_system_evaluation(
    input_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """
    Run the verified-prototype reporting and consistency-check workflow.
    """

    predictions = load_jsonl(
        input_path
    )

    summary = (
        build_verified_summary(
            predictions
        )
    )

    save_summary(
        summary=summary,
        output_path=output_path,
    )

    print_verified_summary(
        summary
    )

    print(
        f"\nMetrics saved to: "
        f"{output_path}"
    )

    return summary


def parse_arguments() -> argparse.Namespace:
    """Parse verified-prototype evaluation paths."""

    parser = argparse.ArgumentParser(
        description=(
            "Summarize the rule-based verified "
            "selective-QA prototype."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=(
            DEFAULT_INPUT_PATH
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_OUTPUT_PATH
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run verified-prototype reporting from command-line arguments."""

    args = parse_arguments()

    run_verified_system_evaluation(
        input_path=args.input,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()