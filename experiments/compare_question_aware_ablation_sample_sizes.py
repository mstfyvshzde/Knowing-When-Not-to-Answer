"""
Compare selective-QA ranking methods across nested held-out sample sizes.

This experiment studies whether the relative ranking performance of the five
main selective-QA methods is stable as the number of evaluated examples grows.

A single deterministic shuffle is created with seed 17. Every evaluated subset
is then taken from the beginning of that same order:

    200 ⊂ 500 ⊂ 1000 ⊂ 2000 ⊂ 3000

Using nested subsets is important because differences across sample sizes then
come from adding more examples rather than evaluating unrelated random samples.

For every subset, the shared final ablation evaluator computes:

- AURC;
- normalized AURC;
- matched-coverage selective risk;
- full-coverage QA accuracy.

The script also records descriptive question-aware verifier diagnostics such as
the number of high-entailment incorrect candidates.

The high-entailment threshold is used only for diagnostic counting. It is not a
decision threshold, is not tuned here, and does not affect the ranking scores or
AURC calculation.

The held-out test labels are used only for evaluation. No model parameter,
fusion weight, or threshold is selected from these sample-size results.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.evaluation.evaluate_question_aware_ablation import (
    DEFAULT_COVERAGE_LEVELS,
    DEFAULT_INPUT_PATH,
    QUESTION_AWARE_SCORE_FIELDS,
    coerce_boolean,
    evaluate_ablation,
    extract_question_aware_score,
    find_available_field,
    infer_correctness,
    load_jsonl,
    parse_coverage_levels,
)
from src.utils.io import (
    save_json,
    save_jsonl,
)

DEFAULT_SAMPLE_SIZES = (
    200,
    500,
    1000,
    2000,
    3000,
)

DEFAULT_OUTPUT_DIRECTORY = Path(
    "outputs/evaluation/question_aware_ablation_sample_sizes"
)

DEFAULT_SEED = 17

DEFAULT_HIGH_ENTAILMENT_THRESHOLD = 0.80


def parse_sample_sizes(
    raw_value: str,
) -> tuple[int, ...]:
    """
    Parse positive sample sizes from a comma-separated command-line value.

    Duplicate values are removed and the remaining sizes are sorted so nested
    evaluation always proceeds from the smallest subset to the largest.
    """

    sample_sizes: list[int] = []

    for item in raw_value.split(","):
        stripped_item = item.strip()

        if not stripped_item:
            continue

        try:
            sample_size = int(
                stripped_item
            )

        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"Invalid sample size: "
                f"{stripped_item!r}"
            ) from error

        if sample_size <= 0:
            raise argparse.ArgumentTypeError(
                "Sample sizes must be "
                "positive integers."
            )

        sample_sizes.append(
            sample_size
        )

    if not sample_sizes:
        raise argparse.ArgumentTypeError(
            "At least one sample size "
            "is required."
        )

    return tuple(
        sorted(
            set(
                sample_sizes
            )
        )
    )


def deterministic_nested_order(
    records: Sequence[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    """
    Create the one record order shared by every nested subset.

    The shuffle is performed once. Subsets are then created by taking the first
    N records from this same order, which guarantees that smaller subsets are
    contained in larger ones.

    This ordering also provides deterministic original-index tie resolution
    when the downstream evaluator encounters equal ranking scores.
    """

    if seed < 0:
        raise ValueError(
            "seed must be non-negative."
        )

    ordered_records = list(
        records
    )

    random.Random(
        seed
    ).shuffle(
        ordered_records
    )

    return ordered_records


def calculate_high_entailment_error_stats(
    records: Sequence[dict[str, Any]],
    threshold: float,
) -> dict[str, int | float | str | None]:
    """
    Describe high question-aware entailment among QA errors.

    These statistics help diagnose a specific verifier failure mode: incorrect
    QA candidates that nevertheless receive strong semantic-support scores.

    They are descriptive only and do not participate in model selection or
    ranking evaluation.
    """

    if not records:
        raise ValueError(
            "Cannot analyze an empty subset."
        )

    if not (
        0.0
        <= threshold
        <= 1.0
    ):
        raise ValueError(
            "High-entailment threshold "
            "must lie in [0, 1]."
        )

    question_aware_field = (
        find_available_field(
            records,
            QUESTION_AWARE_SCORE_FIELDS,
        )
    )

    if question_aware_field is None:
        return {
            "question_aware_score_field": None,
            "incorrect_predictions": sum(
                not infer_correctness(
                    record
                )
                for record in records
            ),
            "valid_claims": 0,
            "invalid_claims": len(
                records
            ),
            "invalid_claim_rate": 1.0,
            "high_entailment_predictions": 0,
            "high_entailment_incorrect": 0,
            "high_entailment_incorrect_rate_among_incorrect": None,
            "error_rate_within_high_entailment": None,
        }

    correctness = [
        infer_correctness(
            record
        )
        for record in records
    ]

    # Invalid claims are assigned question-aware score zero by the shared score
    # extractor, matching the final V2 verifier ranking definition.
    valid_claims = sum(
        coerce_boolean(
            record.get(
                "qa_claim_valid"
            )
        )
        is True
        for record in records
    )

    invalid_claims = (
        len(
            records
        )
        - valid_claims
    )

    semantic_scores = [
        extract_question_aware_score(
            record,
            question_aware_field,
        )
        for record in records
    ]

    high_entailment_indices = [
        index
        for index, score
        in enumerate(
            semantic_scores
        )
        if score
        >= threshold
    ]

    high_entailment_incorrect = sum(
        not correctness[
            index
        ]
        for index
        in high_entailment_indices
    )

    incorrect_predictions = sum(
        not value
        for value
        in correctness
    )

    return {
        "question_aware_score_field": (
            question_aware_field
        ),
        "incorrect_predictions": (
            incorrect_predictions
        ),
        "valid_claims": (
            valid_claims
        ),
        "invalid_claims": (
            invalid_claims
        ),
        "invalid_claim_rate": (
            invalid_claims
            / len(
                records
            )
        ),
        "high_entailment_predictions": (
            len(
                high_entailment_indices
            )
        ),
        "high_entailment_incorrect": (
            high_entailment_incorrect
        ),
        "high_entailment_incorrect_rate_among_incorrect": (
            high_entailment_incorrect
            / incorrect_predictions
            if incorrect_predictions > 0
            else None
        ),
        "error_rate_within_high_entailment": (
            high_entailment_incorrect
            / len(
                high_entailment_indices
            )
            if high_entailment_indices
            else None
        ),
    }


def save_rows_csv(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
) -> None:
    """
    Save aggregated experiment rows with a stable union of encountered fields.

    Different result types can contain optional diagnostics, so column names are
    collected in first-seen order instead of assuming one hard-coded schema.
    """

    if not rows:
        return

    output_path = Path(
        path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(
                    key
                )

                seen.add(
                    key
                )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=(
                fieldnames
            ),
            extrasaction="ignore",
            lineterminator="\n",
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def plot_metric_by_sample_size(
    comparison_rows: Sequence[dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    path: str | Path,
) -> None:
    """
    Plot one ranking metric as increasingly large nested subsets are evaluated.

    These plots visualize stability with sample size. They should not be read as
    independent repeated experiments because every smaller subset is contained
    in the larger subsets.
    """

    methods = sorted(
        {
            str(
                row[
                    "method"
                ]
            )
            for row
            in comparison_rows
        }
    )

    figure = plt.figure(
        figsize=(
            9,
            6,
        )
    )

    axis = figure.add_subplot(
        111
    )

    plotted_any = False

    for method in methods:
        method_rows = sorted(
            (
                row
                for row
                in comparison_rows
                if (
                    row[
                        "method"
                    ]
                    == method
                    and row.get(
                        metric
                    )
                    is not None
                    and math.isfinite(
                        float(
                            row[
                                metric
                            ]
                        )
                    )
                )
            ),
            key=lambda row: (
                int(
                    row[
                        "sample_size"
                    ]
                )
            ),
        )

        if not method_rows:
            continue

        axis.plot(
            [
                int(
                    row[
                        "sample_size"
                    ]
                )
                for row
                in method_rows
            ],
            [
                float(
                    row[
                        metric
                    ]
                )
                for row
                in method_rows
            ],
            marker="o",
            label=(
                method
            ),
        )

        plotted_any = True

    axis.set_xlabel(
        "Sample size"
    )

    axis.set_ylabel(
        ylabel
    )

    axis.set_title(
        title
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    if plotted_any:
        axis.legend()

    figure.tight_layout()

    output_path = Path(
        path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def validate_sample_size_settings(
    sample_sizes: Sequence[int],
    coverage_levels: Sequence[float],
    seed: int,
    high_entailment_threshold: float,
) -> None:
    """Validate experiment settings before constructing any held-out subsets."""

    if not sample_sizes:
        raise ValueError(
            "At least one sample size "
            "is required."
        )

    if any(
        sample_size <= 0
        for sample_size
        in sample_sizes
    ):
        raise ValueError(
            "Sample sizes must be positive."
        )

    if not coverage_levels:
        raise ValueError(
            "At least one coverage level "
            "is required."
        )

    if any(
        not (
            0.0
            < coverage
            <= 1.0
        )
        for coverage
        in coverage_levels
    ):
        raise ValueError(
            "Coverage levels must lie "
            "in (0, 1]."
        )

    if seed < 0:
        raise ValueError(
            "seed must be non-negative."
        )

    if not (
        0.0
        <= high_entailment_threshold
        <= 1.0
    ):
        raise ValueError(
            "high_entailment_threshold "
            "must lie in [0, 1]."
        )


def run_sample_size_comparison(
    input_path: str | Path,
    output_directory: str | Path,
    sample_sizes: Sequence[int],
    coverage_levels: Sequence[float],
    seed: int,
    high_entailment_threshold: float,
    strict_max_size: bool,
    save_subsets: bool,
) -> list[dict[str, Any]]:
    """
    Evaluate the five ranking methods on deterministic nested held-out subsets.

    One shuffled order is generated and reused throughout the experiment.
    Therefore:

        subset(N1) ⊂ subset(N2)

    whenever N1 < N2.

    Each subset is passed unchanged to the shared ablation evaluator, ensuring
    that AURC, normalized AURC, score definitions, and matched-coverage metrics
    remain identical to the main final evaluation.

    Temporary subset JSONL files exist only because the shared evaluator accepts
    a file path. They can be removed after evaluation unless `save_subsets` is
    requested.
    """

    validate_sample_size_settings(
        sample_sizes=(
            sample_sizes
        ),
        coverage_levels=(
            coverage_levels
        ),
        seed=(
            seed
        ),
        high_entailment_threshold=(
            high_entailment_threshold
        ),
    )

    records = load_jsonl(
        input_path
    )

    if not records:
        raise ValueError(
            "Input prediction file "
            "cannot be empty."
        )

    ordered_records = (
        deterministic_nested_order(
            records,
            seed,
        )
    )

    requested_sizes = tuple(
        sorted(
            set(
                sample_sizes
            )
        )
    )

    largest_requested = max(
        requested_sizes
    )

    if (
        largest_requested
        > len(
            ordered_records
        )
        and strict_max_size
    ):
        raise ValueError(
            "The input file does not contain "
            "enough records for the requested "
            "comparison. "
            f"Requested maximum: "
            f"{largest_requested}; "
            f"available: "
            f"{len(ordered_records)}."
        )

    usable_sizes = tuple(
        sample_size
        for sample_size
        in requested_sizes
        if sample_size
        <= len(
            ordered_records
        )
    )

    if not usable_sizes:
        raise ValueError(
            "None of the requested sample sizes "
            "can be evaluated. "
            f"Available records: "
            f"{len(ordered_records)}."
        )

    skipped_sizes = [
        sample_size
        for sample_size
        in requested_sizes
        if sample_size
        > len(
            ordered_records
        )
    ]

    output_root = Path(
        output_directory
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_rows: list[
        dict[str, Any]
    ] = []

    matched_coverage_rows: list[
        dict[str, Any]
    ] = []

    error_rows: list[
        dict[str, Any]
    ] = []

    for sample_size in usable_sizes:
        # Because every subset comes from the same ordered list, adding a larger
        # N adds examples without changing membership of the smaller subset.
        subset = (
            ordered_records[
                :sample_size
            ]
        )

        size_directory = (
            output_root
            / f"n_{sample_size}"
        )

        subset_path = (
            size_directory
            / "subset.jsonl"
        )

        # `evaluate_ablation` consumes a JSONL path, so the deterministic subset
        # is materialized temporarily. This file is not a new data split.
        save_jsonl(
            subset,
            subset_path,
        )

        print(
            "\n"
            + "#" * 88
        )

        print(
            f"RUNNING SAMPLE SIZE: "
            f"{sample_size}"
        )

        print(
            "#" * 88
        )

        results = evaluate_ablation(
            input_path=(
                subset_path
            ),
            output_directory=(
                size_directory
            ),
            coverage_levels=(
                coverage_levels
            ),
        )

        # This diagnostic asks whether incorrect candidates can still receive
        # high question-aware semantic support as sample size increases.
        error_stats = (
            calculate_high_entailment_error_stats(
                subset,
                threshold=(
                    high_entailment_threshold
                ),
            )
        )

        error_row = {
            "sample_size": (
                sample_size
            ),
            "seed": (
                seed
            ),
            "high_entailment_threshold": (
                high_entailment_threshold
            ),
            **error_stats,
        }

        error_rows.append(
            error_row
        )

        for result in results:
            comparison_rows.append(
                {
                    "sample_size": (
                        sample_size
                    ),
                    "seed": (
                        seed
                    ),
                    "method": (
                        result.method
                    ),
                    "score_field": (
                        result.score_field
                    ),
                    "total_records": (
                        result.total_records
                    ),
                    "full_accuracy": (
                        result.full_accuracy
                    ),
                    "aurc": (
                        result.aurc
                    ),
                    "normalized_aurc": (
                        result.normalized_aurc
                    ),
                    "incorrect_predictions": (
                        error_stats[
                            "incorrect_predictions"
                        ]
                    ),
                    "invalid_claim_rate": (
                        error_stats[
                            "invalid_claim_rate"
                        ]
                    ),
                    "high_entailment_incorrect": (
                        error_stats[
                            "high_entailment_incorrect"
                        ]
                    ),
                    "high_entailment_incorrect_rate_among_incorrect": (
                        error_stats[
                            "high_entailment_incorrect_rate_among_incorrect"
                        ]
                    ),
                    "error_rate_within_high_entailment": (
                        error_stats[
                            "error_rate_within_high_entailment"
                        ]
                    ),
                }
            )

            # Matched coverage compares methods after answering the same number
            # of top-ranked examples, avoiding threshold-specific coverage
            # differences between score distributions.
            for metric in result.matched_coverage:
                matched_coverage_rows.append(
                    {
                        "sample_size": (
                            sample_size
                        ),
                        "seed": (
                            seed
                        ),
                        "method": (
                            result.method
                        ),
                        **asdict(
                            metric
                        ),
                    }
                )

        if not save_subsets:
            subset_path.unlink(
                missing_ok=True
            )

    save_rows_csv(
        comparison_rows,
        output_root
        / "sample_size_comparison.csv",
    )

    save_rows_csv(
        matched_coverage_rows,
        output_root
        / "matched_coverage_by_sample_size.csv",
    )

    save_rows_csv(
        error_rows,
        output_root
        / "high_entailment_error_summary.csv",
    )

    payload = {
        "analysis_type": (
            "nested_heldout_sample_size_stability"
        ),
        "input_path": (
            str(
                input_path
            )
        ),
        "available_records": (
            len(
                ordered_records
            )
        ),
        "seed": (
            seed
        ),
        "requested_sample_sizes": list(
            requested_sizes
        ),
        "evaluated_sample_sizes": list(
            usable_sizes
        ),
        "skipped_sample_sizes": (
            skipped_sizes
        ),
        "nested_subset_policy": (
            "One deterministic shuffled order "
            "is created with the supplied seed. "
            "Each subset contains the first N "
            "records, so every smaller subset "
            "is contained in every larger subset."
        ),
        "interpretation_note": (
            "Nested subset results measure "
            "stability as more held-out examples "
            "are added; they are not independent "
            "replicate experiments."
        ),
        "coverage_levels": list(
            coverage_levels
        ),
        "high_entailment_threshold": (
            high_entailment_threshold
        ),
        "high_entailment_threshold_role": (
            "diagnostic counting only; "
            "does not affect ranking or AURC"
        ),
        "methods": (
            comparison_rows
        ),
        "high_entailment_error_summary": (
            error_rows
        ),
    }

    save_json(
        payload,
        output_root
        / "sample_size_comparison.json",
    )

    plot_metric_by_sample_size(
        comparison_rows,
        metric="aurc",
        ylabel=(
            "AURC (lower is better)"
        ),
        title=(
            "AURC Stability Across "
            "Nested Sample Sizes"
        ),
        path=(
            output_root
            / "aurc_by_sample_size.png"
        ),
    )

    plot_metric_by_sample_size(
        comparison_rows,
        metric="normalized_aurc",
        ylabel=(
            "Normalized AURC "
            "(lower is better)"
        ),
        title=(
            "Normalized AURC Across "
            "Nested Sample Sizes"
        ),
        path=(
            output_root
            / "normalized_aurc_by_sample_size.png"
        ),
    )

    print(
        "\n"
        + "=" * 88
    )

    print(
        "SAMPLE-SIZE COMPARISON COMPLETE"
    )

    print(
        "=" * 88
    )

    print(
        f"Available input records: "
        f"{len(ordered_records)}"
    )

    print(
        f"Evaluated sizes: "
        f"{list(usable_sizes)}"
    )

    if skipped_sizes:
        print(
            f"Skipped sizes: "
            f"{skipped_sizes}"
        )

    print(
        output_root
        / "sample_size_comparison.csv"
    )

    print(
        output_root
        / "sample_size_comparison.json"
    )

    print(
        output_root
        / "matched_coverage_by_sample_size.csv"
    )

    print(
        output_root
        / "high_entailment_error_summary.csv"
    )

    print(
        output_root
        / "aurc_by_sample_size.png"
    )

    print(
        output_root
        / "normalized_aurc_by_sample_size.png"
    )

    return comparison_rows


def parse_arguments() -> argparse.Namespace:
    """Parse nested held-out sample-size experiment settings."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the five selective-QA "
            "ranking methods across deterministic "
            "nested held-out sample sizes."
        )
    )

    parser.add_argument(
        "--input",
        default=str(
            DEFAULT_INPUT_PATH
        ),
        help=(
            "Fully processed held-out prediction "
            "JSONL containing the verifier and "
            "confidence scores."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIRECTORY
        ),
        help=(
            "Root directory for sample-size "
            "comparison artifacts."
        ),
    )

    parser.add_argument(
        "--sample-sizes",
        type=(
            parse_sample_sizes
        ),
        default=(
            DEFAULT_SAMPLE_SIZES
        ),
        help=(
            "Comma-separated nested sizes. "
            "Default: 200,500,1000,2000,3000"
        ),
    )

    parser.add_argument(
        "--coverage-levels",
        type=(
            parse_coverage_levels
        ),
        default=(
            DEFAULT_COVERAGE_LEVELS
        ),
        help=(
            "Comma-separated matched "
            "coverage levels."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=(
            DEFAULT_SEED
        ),
        help=(
            "Seed used once to create the "
            "deterministic nested record order."
        ),
    )

    parser.add_argument(
        "--high-entailment-threshold",
        type=float,
        default=(
            DEFAULT_HIGH_ENTAILMENT_THRESHOLD
        ),
        help=(
            "Diagnostic threshold used only "
            "to count high-entailment "
            "incorrect candidates."
        ),
    )

    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Evaluate only sample sizes that "
            "fit the available input instead "
            "of failing when the requested "
            "maximum is unavailable."
        ),
    )

    parser.add_argument(
        "--save-subsets",
        action="store_true",
        help=(
            "Keep deterministic intermediate "
            "subset JSONL files after evaluation."
        ),
    )

    arguments = (
        parser.parse_args()
    )

    if arguments.seed < 0:
        parser.error(
            "--seed must be non-negative."
        )

    if not (
        0.0
        <= arguments.high_entailment_threshold
        <= 1.0
    ):
        parser.error(
            "--high-entailment-threshold "
            "must be in [0, 1]."
        )

    return arguments


def main() -> None:
    """Run nested sample-size stability analysis."""

    args = (
        parse_arguments()
    )

    run_sample_size_comparison(
        input_path=(
            args.input
        ),
        output_directory=(
            args.output_dir
        ),
        sample_sizes=(
            args.sample_sizes
        ),
        coverage_levels=(
            args.coverage_levels
        ),
        seed=(
            args.seed
        ),
        high_entailment_threshold=(
            args.high_entailment_threshold
        ),
        strict_max_size=(
            not args.allow_partial
        ),
        save_subsets=(
            args.save_subsets
        ),
    )


if __name__ == "__main__":
    main()