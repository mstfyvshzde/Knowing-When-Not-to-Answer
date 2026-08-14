"""
Compare question-aware selective-QA ablations across nested sample sizes.

Expected project layout
-----------------------
This file should be placed at the project root and imports:

    src.evaluation.evaluate_question_aware_ablation

The input JSONL must already contain at least the largest requested number of
fully processed predictions (QA confidence, evidence/verifier outputs, QA2D,
and question-aware NLI fields).

Default comparison
------------------
    200, 500, 1000, 2000, 5000

The same deterministically shuffled record order is reused for every size, so:

    200 ⊂ 500 ⊂ 1000 ⊂ 2000 ⊂ 5000

Outputs
-------
- One complete ablation directory per sample size
- sample_size_comparison.csv
- sample_size_comparison.json
- matched_coverage_by_sample_size.csv
- high_entailment_error_summary.csv
- aurc_by_sample_size.png
- normalized_aurc_by_sample_size.png
"""

from __future__ import annotations

import argparse
import csv
import json
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

DEFAULT_SAMPLE_SIZES = (200, 500, 1000, 2000, 5000)
DEFAULT_OUTPUT_DIRECTORY = Path(
    "outputs/evaluation/question_aware_ablation_sample_sizes"
)
DEFAULT_SEED = 17
DEFAULT_HIGH_ENTAILMENT_THRESHOLD = 0.80


def parse_sample_sizes(raw_value: str) -> tuple[int, ...]:
    """Parse a comma-separated list of positive sample sizes."""

    sample_sizes: list[int] = []

    for item in raw_value.split(","):
        stripped_item = item.strip()
        if not stripped_item:
            continue

        try:
            sample_size = int(stripped_item)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"Invalid sample size: {stripped_item!r}"
            ) from error

        if sample_size <= 0:
            raise argparse.ArgumentTypeError("Sample sizes must be positive integers.")

        sample_sizes.append(sample_size)

    if not sample_sizes:
        raise argparse.ArgumentTypeError("At least one sample size is required.")

    return tuple(sorted(set(sample_sizes)))


def save_json(data: Any, path: str | Path) -> None:
    """Write a JSON-compatible object."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False)


def save_jsonl(records: Sequence[dict[str, Any]], path: str | Path) -> None:
    """Write records to JSONL."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def deterministic_nested_order(
    records: Sequence[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    """Return one deterministic shuffled order used by every subset."""

    ordered_records = list(records)
    random.Random(seed).shuffle(ordered_records)
    return ordered_records


def calculate_high_entailment_error_stats(
    records: Sequence[dict[str, Any]],
    threshold: float,
) -> dict[str, int | float | None]:
    """Summarise high-entailment errors for one subset."""

    question_aware_field = find_available_field(
        records,
        QUESTION_AWARE_SCORE_FIELDS,
    )

    if question_aware_field is None:
        return {
            "question_aware_score_field": None,
            "incorrect_predictions": sum(
                not infer_correctness(record) for record in records
            ),
            "valid_claims": 0,
            "invalid_claims": len(records),
            "invalid_claim_rate": 1.0,
            "high_entailment_predictions": 0,
            "high_entailment_incorrect": 0,
            "high_entailment_incorrect_rate_among_incorrect": None,
            "error_rate_within_high_entailment": None,
        }

    correctness = [infer_correctness(record) for record in records]

    valid_claims = sum(
        coerce_boolean(record.get("qa_claim_valid")) is True for record in records
    )
    invalid_claims = len(records) - valid_claims

    semantic_scores = [
        extract_question_aware_score(record, question_aware_field) for record in records
    ]

    high_entailment_indices = [
        index for index, score in enumerate(semantic_scores) if score >= threshold
    ]

    high_entailment_incorrect = sum(
        not correctness[index] for index in high_entailment_indices
    )
    incorrect_predictions = sum(not value for value in correctness)

    return {
        "question_aware_score_field": question_aware_field,
        "incorrect_predictions": incorrect_predictions,
        "valid_claims": valid_claims,
        "invalid_claims": invalid_claims,
        "invalid_claim_rate": invalid_claims / len(records),
        "high_entailment_predictions": len(high_entailment_indices),
        "high_entailment_incorrect": high_entailment_incorrect,
        "high_entailment_incorrect_rate_among_incorrect": (
            high_entailment_incorrect / incorrect_predictions
            if incorrect_predictions > 0
            else None
        ),
        "error_rate_within_high_entailment": (
            high_entailment_incorrect / len(high_entailment_indices)
            if high_entailment_indices
            else None
        ),
    }


def save_rows_csv(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
) -> None:
    """Save homogeneous dictionaries to CSV."""

    if not rows:
        return

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_by_sample_size(
    comparison_rows: Sequence[dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    path: str | Path,
) -> None:
    """Plot one metric against sample size for every method."""

    methods = sorted({str(row["method"]) for row in comparison_rows})

    figure = plt.figure(figsize=(9, 6))
    axis = figure.add_subplot(111)

    plotted_any = False

    for method in methods:
        method_rows = sorted(
            (
                row
                for row in comparison_rows
                if row["method"] == method
                and row.get(metric) is not None
                and math.isfinite(float(row[metric]))
            ),
            key=lambda row: int(row["sample_size"]),
        )

        if not method_rows:
            continue

        axis.plot(
            [int(row["sample_size"]) for row in method_rows],
            [float(row[metric]) for row in method_rows],
            marker="o",
            label=method,
        )
        plotted_any = True

    axis.set_xlabel("Sample size")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)

    if plotted_any:
        axis.legend()

    figure.tight_layout()

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


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
    """Run nested sample-size experiments and aggregate their results."""

    records = load_jsonl(input_path)
    ordered_records = deterministic_nested_order(records, seed)

    requested_sizes = tuple(sorted(set(sample_sizes)))
    largest_requested = max(requested_sizes)

    if largest_requested > len(ordered_records) and strict_max_size:
        raise ValueError(
            "The input file does not contain enough records for the requested "
            f"comparison. Requested maximum: {largest_requested}; "
            f"available: {len(ordered_records)}. Generate a fully processed "
            "prediction JSONL with at least the requested maximum first."
        )

    usable_sizes = tuple(
        sample_size
        for sample_size in requested_sizes
        if sample_size <= len(ordered_records)
    )

    if not usable_sizes:
        raise ValueError(
            "None of the requested sample sizes can be evaluated. "
            f"Available records: {len(ordered_records)}."
        )

    skipped_sizes = [
        sample_size
        for sample_size in requested_sizes
        if sample_size > len(ordered_records)
    ]

    output_root = Path(output_directory)
    output_root.mkdir(parents=True, exist_ok=True)

    comparison_rows: list[dict[str, Any]] = []
    matched_coverage_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for sample_size in usable_sizes:
        subset = ordered_records[:sample_size]
        size_directory = output_root / f"n_{sample_size}"
        subset_path = size_directory / "subset.jsonl"

        # evaluate_ablation consumes a path, so a deterministic subset file is
        # written for reproducibility. It can optionally be deleted afterward.
        save_jsonl(subset, subset_path)

        print("\n" + "#" * 88)
        print(f"RUNNING SAMPLE SIZE: {sample_size}")
        print("#" * 88)

        results = evaluate_ablation(
            input_path=subset_path,
            output_directory=size_directory,
            coverage_levels=coverage_levels,
        )

        error_stats = calculate_high_entailment_error_stats(
            subset,
            threshold=high_entailment_threshold,
        )

        error_row = {
            "sample_size": sample_size,
            "seed": seed,
            "high_entailment_threshold": high_entailment_threshold,
            **error_stats,
        }
        error_rows.append(error_row)

        for result in results:
            comparison_rows.append(
                {
                    "sample_size": sample_size,
                    "seed": seed,
                    "method": result.method,
                    "score_field": result.score_field,
                    "total_records": result.total_records,
                    "full_accuracy": result.full_accuracy,
                    "aurc": result.aurc,
                    "normalized_aurc": result.normalized_aurc,
                    "incorrect_predictions": error_stats["incorrect_predictions"],
                    "invalid_claim_rate": error_stats["invalid_claim_rate"],
                    "high_entailment_incorrect": error_stats[
                        "high_entailment_incorrect"
                    ],
                    "high_entailment_incorrect_rate_among_incorrect": (
                        error_stats["high_entailment_incorrect_rate_among_incorrect"]
                    ),
                    "error_rate_within_high_entailment": error_stats[
                        "error_rate_within_high_entailment"
                    ],
                }
            )

            for metric in result.matched_coverage:
                matched_coverage_rows.append(
                    {
                        "sample_size": sample_size,
                        "seed": seed,
                        "method": result.method,
                        **asdict(metric),
                    }
                )

        if not save_subsets:
            subset_path.unlink(missing_ok=True)

    save_rows_csv(
        comparison_rows,
        output_root / "sample_size_comparison.csv",
    )
    save_rows_csv(
        matched_coverage_rows,
        output_root / "matched_coverage_by_sample_size.csv",
    )
    save_rows_csv(
        error_rows,
        output_root / "high_entailment_error_summary.csv",
    )

    payload = {
        "input_path": str(input_path),
        "available_records": len(ordered_records),
        "seed": seed,
        "requested_sample_sizes": list(requested_sizes),
        "evaluated_sample_sizes": list(usable_sizes),
        "skipped_sample_sizes": skipped_sizes,
        "nested_subset_policy": (
            "One deterministic shuffled order; each subset is the first N "
            "records, so smaller subsets are contained in larger subsets."
        ),
        "coverage_levels": list(coverage_levels),
        "high_entailment_threshold": high_entailment_threshold,
        "methods": comparison_rows,
        "high_entailment_error_summary": error_rows,
    }

    save_json(
        payload,
        output_root / "sample_size_comparison.json",
    )

    plot_metric_by_sample_size(
        comparison_rows,
        metric="aurc",
        ylabel="AURC (lower is better)",
        title="AURC Stability Across Sample Sizes",
        path=output_root / "aurc_by_sample_size.png",
    )
    plot_metric_by_sample_size(
        comparison_rows,
        metric="normalized_aurc",
        ylabel="Normalized AURC (lower is better)",
        title="Normalized AURC Across Sample Sizes",
        path=output_root / "normalized_aurc_by_sample_size.png",
    )

    print("\n" + "=" * 88)
    print("SAMPLE-SIZE COMPARISON COMPLETE")
    print("=" * 88)
    print(f"Available input records: {len(ordered_records)}")
    print(f"Evaluated sizes: {list(usable_sizes)}")

    if skipped_sizes:
        print(f"Skipped sizes: {skipped_sizes}")

    print(output_root / "sample_size_comparison.csv")
    print(output_root / "sample_size_comparison.json")
    print(output_root / "matched_coverage_by_sample_size.csv")
    print(output_root / "high_entailment_error_summary.csv")
    print(output_root / "aurc_by_sample_size.png")
    print(output_root / "normalized_aurc_by_sample_size.png")

    return comparison_rows


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate nested prediction subsets and compare selective-QA "
            "methods across sample sizes."
        )
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help=(
            "Fully processed prediction JSONL. It must contain at least the "
            "largest requested sample size."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIRECTORY),
        help="Root directory for all sample-size comparison outputs.",
    )
    parser.add_argument(
        "--sample-sizes",
        type=parse_sample_sizes,
        default=DEFAULT_SAMPLE_SIZES,
        help="Comma-separated sizes. Default: 200,500,1000,2000,5000",
    )
    parser.add_argument(
        "--coverage-levels",
        type=parse_coverage_levels,
        default=DEFAULT_COVERAGE_LEVELS,
        help="Comma-separated matched coverage levels.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed used for the one deterministic record ordering.",
    )
    parser.add_argument(
        "--high-entailment-threshold",
        type=float,
        default=DEFAULT_HIGH_ENTAILMENT_THRESHOLD,
        help="Threshold used to count high-entailment incorrect predictions.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Evaluate only sizes available in the input instead of failing "
            "when the largest requested size is unavailable."
        ),
    )
    parser.add_argument(
        "--save-subsets",
        action="store_true",
        help="Keep the deterministic subset JSONL files after evaluation.",
    )

    arguments = parser.parse_args()

    if not 0.0 <= arguments.high_entailment_threshold <= 1.0:
        parser.error("--high-entailment-threshold must be in [0, 1].")

    return arguments


if __name__ == "__main__":
    args = parse_arguments()

    run_sample_size_comparison(
        input_path=args.input,
        output_directory=args.output_dir,
        sample_sizes=args.sample_sizes,
        coverage_levels=args.coverage_levels,
        seed=args.seed,
        high_entailment_threshold=args.high_entailment_threshold,
        strict_max_size=not args.allow_partial,
        save_subsets=args.save_subsets,
    )
