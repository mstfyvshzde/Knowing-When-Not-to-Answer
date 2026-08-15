"""
Estimate uncertainty around the final held-out AURC results with paired bootstrap.

This experiment quantifies how much the reported AURC values and their
differences from confidence-only ranking vary under resampling of the held-out
examples.

The bootstrap is paired across methods:

- one bootstrap sample of example indices is drawn;
- the same sampled examples are used for every ranking method;
- AURC is recomputed for every method;
- alternative-minus-confidence AURC is calculated within the same replicate.

This pairing is important because all methods are evaluated on the same QA
examples. It isolates uncertainty in their difference more directly than
bootstrapping each method independently.

For each method, the script reports:

- full-sample AURC;
- 95% percentile bootstrap interval for AURC;
- AURC difference relative to confidence-only;
- 95% percentile interval for that paired difference.

Because lower AURC is better:

    delta AURC > 0  -> alternative is worse than confidence
    delta AURC < 0  -> alternative is better than confidence

`bootstrap_fraction_better_than_confidence` is a descriptive bootstrap
frequency, not a formal p-value.

No model parameter, threshold, or fusion weight is selected in this script.
The held-out labels are used only to estimate uncertainty after the evaluation
procedure has already been frozen.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.compare_question_aware_ablation_sample_sizes import (
    deterministic_nested_order,
)
from src.evaluation.evaluate_question_aware_ablation import (
    CONFIDENCE_FIELDS,
    QUESTION_AWARE_SCORE_FIELDS,
    SELF_VERIFICATION_SCORE_FIELDS,
    build_risk_coverage_curve,
    calculate_aurc,
    extract_numeric_score,
    extract_question_aware_score,
    extract_self_verification_score,
    find_available_field,
    geometric_mean_score,
    infer_correctness,
)
from src.utils.io import (
    load_jsonl,
    save_json,
)

DEFAULT_INPUT = Path(
    "outputs/predictions/"
    "test_with_question_aware_v2_and_self_verification.jsonl"
)

DEFAULT_OUTPUT_DIR = Path(
    "outputs/evaluation/"
    "final_sample_size_comparison/bootstrap"
)

DEFAULT_BOOTSTRAP_SAMPLES = 5000
DEFAULT_BOOTSTRAP_SEED = 17
DEFAULT_ORDER_SEED = 17


def compute_aurc(
    correctness: list[bool],
    scores: list[float],
) -> float:
    """
    Compute the project's canonical discrete AURC for one ranked score vector.

    The shared evaluator first ranks examples by score and then averages
    selective risk over every non-empty ranked prefix.
    """

    if (
        not correctness
        or len(correctness)
        != len(scores)
    ):
        raise ValueError(
            "Correctness and score vectors must "
            "be non-empty and have equal length."
        )

    curve = build_risk_coverage_curve(
        correctness,
        scores,
    )

    return calculate_aurc(
        curve
    )


def percentile_ci(
    values: list[float],
) -> tuple[float, float]:
    """
    Return the central 95% percentile bootstrap interval.

    The interval uses the 2.5th and 97.5th percentiles of the empirical
    bootstrap distribution.
    """

    if not values:
        raise ValueError(
            "Cannot calculate a percentile "
            "interval from an empty sequence."
        )

    array = np.asarray(
        values,
        dtype=float,
    )

    if not np.all(
        np.isfinite(
            array
        )
    ):
        raise ValueError(
            "Bootstrap distribution contains "
            "non-finite values."
        )

    lower, upper = np.percentile(
        array,
        [
            2.5,
            97.5,
        ],
    )

    return (
        float(
            lower
        ),
        float(
            upper
        ),
    )


def validate_scores(
    scores: list[float],
    name: str,
    expected_length: int,
) -> None:
    """Validate one probability-like ranking score vector."""

    if len(
        scores
    ) != expected_length:
        raise ValueError(
            f"{name} contains {len(scores)} scores "
            f"but expected {expected_length}."
        )

    for index, score in enumerate(
        scores
    ):
        if (
            not math.isfinite(
                score
            )
            or not (
                0.0
                <= score
                <= 1.0
            )
        ):
            raise ValueError(
                f"{name} contains invalid score "
                f"at index {index}: {score!r}."
            )


def build_method_scores(
    records: list[dict[str, Any]],
) -> tuple[
    list[bool],
    dict[
        str,
        list[float],
    ],
]:
    """
    Reconstruct the five canonical ranking methods from final prediction records.

    The same shared score extractors used by the main held-out evaluator are
    reused here so the bootstrap analyzes exactly the same ranking definitions.

    Question-aware invalid claims receive score zero through the shared
    extractor. Self-verification scores are normalized from [-1, 1] to [0, 1]
    by the shared evaluator.
    """

    if not records:
        raise ValueError(
            "Prediction records cannot be empty."
        )

    confidence_field = (
        find_available_field(
            records,
            CONFIDENCE_FIELDS,
        )
    )

    question_aware_field = (
        find_available_field(
            records,
            QUESTION_AWARE_SCORE_FIELDS,
        )
    )

    self_verification_field = (
        find_available_field(
            records,
            SELF_VERIFICATION_SCORE_FIELDS,
        )
    )

    if confidence_field is None:
        raise ValueError(
            "Confidence score field "
            "was not found."
        )

    if question_aware_field is None:
        raise ValueError(
            "Question-aware semantic score "
            "field was not found."
        )

    if self_verification_field is None:
        raise ValueError(
            "Self-verification score field "
            "was not found."
        )

    correctness = [
        infer_correctness(
            record
        )
        for record in records
    ]

    confidence_scores_optional = [
        extract_numeric_score(
            record,
            confidence_field,
        )
        for record in records
    ]

    if any(
        score is None
        for score
        in confidence_scores_optional
    ):
        missing_count = sum(
            score is None
            for score
            in confidence_scores_optional
        )

        raise ValueError(
            "Confidence score is missing "
            f"in {missing_count} records."
        )

    confidence_scores = [
        float(
            score
        )
        for score
        in confidence_scores_optional
        if score is not None
    ]

    question_aware_scores = [
        extract_question_aware_score(
            record,
            question_aware_field,
        )
        for record in records
    ]

    self_scores_optional = [
        extract_self_verification_score(
            record,
            self_verification_field,
        )
        for record in records
    ]

    if any(
        score is None
        for score
        in self_scores_optional
    ):
        missing_count = sum(
            score is None
            for score
            in self_scores_optional
        )

        raise ValueError(
            "Self-verification score is missing "
            f"in {missing_count} records."
        )

    self_scores = [
        float(
            score
        )
        for score
        in self_scores_optional
        if score is not None
    ]

    total = len(
        records
    )

    validate_scores(
        confidence_scores,
        "confidence scores",
        total,
    )

    validate_scores(
        question_aware_scores,
        "question-aware scores",
        total,
    )

    validate_scores(
        self_scores,
        "self-verification scores",
        total,
    )

    # These are the fixed equal-weight geometric-mean fusions evaluated in the
    # final five-method comparison. Calibration-only tuning is a separate
    # experiment and does not alter these historical fixed-fusion definitions.
    confidence_question_scores = [
        geometric_mean_score(
            confidence,
            semantic,
        )
        for confidence, semantic
        in zip(
            confidence_scores,
            question_aware_scores,
        )
    ]

    confidence_self_scores = [
        geometric_mean_score(
            confidence,
            self_score,
        )
        for confidence, self_score
        in zip(
            confidence_scores,
            self_scores,
        )
    ]

    methods = {
        "Confidence only": (
            confidence_scores
        ),
        "Question-aware semantic V2": (
            question_aware_scores
        ),
        "Confidence + question-aware semantic V2": (
            confidence_question_scores
        ),
        "Self-verifier only": (
            self_scores
        ),
        "Confidence + self-verifier": (
            confidence_self_scores
        ),
    }

    return (
        correctness,
        methods,
    )


def validate_bootstrap_settings(
    bootstrap_samples: int,
    seed: int,
    order_seed: int,
) -> None:
    """Validate reproducibility settings before any resampling begins."""

    if (
        not isinstance(
            bootstrap_samples,
            int,
        )
        or isinstance(
            bootstrap_samples,
            bool,
        )
        or bootstrap_samples <= 0
    ):
        raise ValueError(
            "bootstrap_samples must be "
            "a positive integer."
        )

    for name, value in (
        (
            "seed",
            seed,
        ),
        (
            "order_seed",
            order_seed,
        ),
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
            raise ValueError(
                f"{name} must be a "
                "non-negative integer."
            )


def run_bootstrap(
    input_path: Path,
    output_dir: Path,
    bootstrap_samples: int,
    seed: int,
    order_seed: int,
) -> list[dict[str, Any]]:
    """
    Run paired non-parametric bootstrap uncertainty analysis.

    Each bootstrap replicate samples N examples with replacement from the N
    held-out examples.

    The sampled indices are shared across all five methods, making differences
    relative to confidence-only paired within each replicate.

    Sampled indices are sorted after resampling. Sorting does not change the
    bootstrap multiset; it restores the deterministic base-record order inside
    that multiset. This matters because the ranking evaluator resolves exact
    score ties using record order.
    """

    validate_bootstrap_settings(
        bootstrap_samples,
        seed,
        order_seed,
    )

    records = load_jsonl(
        input_path
    )

    if not records:
        raise ValueError(
            "Input prediction file is empty."
        )

    # Match the deterministic seed-based ordering used by the final nested
    # sample-size evaluation before any score-based ranking takes place.
    records = (
        deterministic_nested_order(
            records,
            seed=(
                order_seed
            ),
        )
    )

    correctness, methods = (
        build_method_scores(
            records
        )
    )

    total = len(
        records
    )

    rng = np.random.default_rng(
        seed
    )

    # These are the reported AURCs on the complete held-out dataset. Bootstrap
    # replicates quantify uncertainty around these fixed point estimates.
    point_aurcs = {
        method: compute_aurc(
            correctness,
            scores,
        )
        for method, scores
        in methods.items()
    }

    bootstrap_aurcs: dict[
        str,
        list[float],
    ] = {
        method: []
        for method in methods
    }

    method_arrays = {
        method: np.asarray(
            scores,
            dtype=float,
        )
        for method, scores
        in methods.items()
    }

    correctness_array = (
        np.asarray(
            correctness,
            dtype=bool,
        )
    )

    for iteration in range(
        bootstrap_samples
    ):
        # Resample examples, not scores independently. The same example indices
        # are then reused for every method, which creates the paired bootstrap.
        sampled_indices = np.sort(
            rng.integers(
                0,
                total,
                size=total,
            )
        )

        sampled_correctness = (
            correctness_array[
                sampled_indices
            ].tolist()
        )

        for (
            method,
            score_array,
        ) in method_arrays.items():
            sampled_scores = (
                score_array[
                    sampled_indices
                ].tolist()
            )

            bootstrap_aurcs[
                method
            ].append(
                compute_aurc(
                    sampled_correctness,
                    sampled_scores,
                )
            )

        if (
            iteration
            + 1
        ) % 500 == 0:
            print(
                "Bootstrap: "
                f"{iteration + 1}/"
                f"{bootstrap_samples}"
            )

    baseline_name = (
        "Confidence only"
    )

    baseline_bootstrap = (
        np.asarray(
            bootstrap_aurcs[
                baseline_name
            ],
            dtype=float,
        )
    )

    rows: list[
        dict[str, Any]
    ] = []

    for method in methods:
        method_values = (
            bootstrap_aurcs[
                method
            ]
        )

        (
            ci_low,
            ci_high,
        ) = percentile_ci(
            method_values
        )

        if (
            method
            == baseline_name
        ):
            delta_point = 0.0
            delta_low = 0.0
            delta_high = 0.0
            fraction_better = None

        else:
            # Because the same bootstrap replicate is used for both methods,
            # this is a paired distribution of AURC differences.
            delta_values = (
                np.asarray(
                    method_values,
                    dtype=float,
                )
                - baseline_bootstrap
            )

            (
                delta_low,
                delta_high,
            ) = percentile_ci(
                delta_values.tolist()
            )

            delta_point = (
                point_aurcs[
                    method
                ]
                - point_aurcs[
                    baseline_name
                ]
            )

            # Lower AURC is better, so delta < 0 means the alternative beats
            # confidence in that bootstrap replicate. This proportion is
            # descriptive and should not be reported as a formal p-value.
            fraction_better = float(
                np.mean(
                    delta_values
                    < 0.0
                )
            )

        rows.append(
            {
                "method": (
                    method
                ),
                "n": (
                    total
                ),
                "aurc": (
                    point_aurcs[
                        method
                    ]
                ),
                "aurc_ci95_low": (
                    ci_low
                ),
                "aurc_ci95_high": (
                    ci_high
                ),
                "delta_aurc_vs_confidence": (
                    delta_point
                ),
                "delta_ci95_low": (
                    delta_low
                ),
                "delta_ci95_high": (
                    delta_high
                ),
                "bootstrap_fraction_better_than_confidence": (
                    fraction_better
                ),
                "bootstrap_samples": (
                    bootstrap_samples
                ),
                "bootstrap_seed": (
                    seed
                ),
                "order_seed": (
                    order_seed
                ),
            }
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "bootstrap_aurc_summary.csv"
    )

    json_path = (
        output_dir
        / "bootstrap_aurc_summary.json"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[
                    0
                ].keys()
            ),
            lineterminator="\n",
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    save_json(
        rows,
        json_path,
    )

    print()

    print(
        "Bootstrap uncertainty "
        "analysis complete."
    )

    print(
        f"Examples: "
        f"{total}"
    )

    print(
        f"Bootstrap samples: "
        f"{bootstrap_samples}"
    )

    print(
        f"Bootstrap seed: "
        f"{seed}"
    )

    print(
        f"Order seed: "
        f"{order_seed}"
    )

    print()

    for row in rows:
        print(
            f"{row['method']}: "
            f"AURC="
            f"{row['aurc']:.6f} "
            f"[{row['aurc_ci95_low']:.6f}, "
            f"{row['aurc_ci95_high']:.6f}]"
        )

        if (
            row[
                "method"
            ]
            != baseline_name
        ):
            print(
                "  Δ vs confidence="
                f"{row[
                    'delta_aurc_vs_confidence'
                ]:+.6f} "
                f"[{row['delta_ci95_low']:.6f}, "
                f"{row['delta_ci95_high']:.6f}]"
            )

    print()

    print(
        f"Saved: "
        f"{csv_path}"
    )

    print(
        f"Saved: "
        f"{json_path}"
    )

    return rows


def parse_arguments() -> argparse.Namespace:
    """Parse held-out bootstrap uncertainty settings."""

    parser = argparse.ArgumentParser(
        description=(
            "Run paired bootstrap uncertainty "
            "analysis for the final held-out "
            "AURC comparison."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=(
            DEFAULT_INPUT
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            DEFAULT_OUTPUT_DIR
        ),
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=(
            DEFAULT_BOOTSTRAP_SAMPLES
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=(
            DEFAULT_BOOTSTRAP_SEED
        ),
        help=(
            "Seed controlling bootstrap "
            "resampling."
        ),
    )

    parser.add_argument(
        "--order-seed",
        type=int,
        default=(
            DEFAULT_ORDER_SEED
        ),
        help=(
            "Seed controlling deterministic "
            "held-out record order and AURC "
            "tie resolution."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run bootstrap uncertainty analysis from command-line arguments."""

    args = parse_arguments()

    run_bootstrap(
        input_path=(
            args.input
        ),
        output_dir=(
            args.output_dir
        ),
        bootstrap_samples=(
            args.bootstrap_samples
        ),
        seed=(
            args.seed
        ),
        order_seed=(
            args.order_seed
        ),
    )


if __name__ == "__main__":
    main()