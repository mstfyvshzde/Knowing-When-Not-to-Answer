from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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
from src.utils.io import load_jsonl

DEFAULT_INPUT = Path(
    "outputs/predictions/"
    "test_with_question_aware_v2_and_self_verification.jsonl"
)

DEFAULT_OUTPUT_DIR = Path(
    "outputs/evaluation/final_sample_size_comparison/bootstrap"
)


def compute_aurc(
    correctness: list[bool],
    scores: list[float],
) -> float:
    curve = build_risk_coverage_curve(correctness, scores)
    return calculate_aurc(curve)


def percentile_ci(
    values: list[float],
) -> tuple[float, float]:
    lower, upper = np.percentile(values, [2.5, 97.5])
    return float(lower), float(upper)


def build_method_scores(
    records: list[dict],
) -> tuple[list[bool], dict[str, list[float]]]:
    confidence_field = find_available_field(records, CONFIDENCE_FIELDS)
    question_aware_field = find_available_field(
        records,
        QUESTION_AWARE_SCORE_FIELDS,
    )
    self_verification_field = find_available_field(
        records,
        SELF_VERIFICATION_SCORE_FIELDS,
    )

    if confidence_field is None:
        raise ValueError("Confidence score field was not found.")

    if question_aware_field is None:
        raise ValueError("Question-aware semantic score field was not found.")

    if self_verification_field is None:
        raise ValueError("Self-verification score field was not found.")

    correctness = [infer_correctness(record) for record in records]

    confidence_scores_optional = [
        extract_numeric_score(record, confidence_field)
        for record in records
    ]

    if any(score is None for score in confidence_scores_optional):
        raise ValueError("Missing confidence score.")

    confidence_scores = [
        float(score)
        for score in confidence_scores_optional
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

    if any(score is None for score in self_scores_optional):
        raise ValueError("Missing self-verification score.")

    self_scores = [
        float(score)
        for score in self_scores_optional
        if score is not None
    ]

    confidence_question_scores = [
        geometric_mean_score(confidence, semantic)
        for confidence, semantic in zip(
            confidence_scores,
            question_aware_scores,
        )
    ]

    confidence_self_scores = [
        geometric_mean_score(confidence, self_score)
        for confidence, self_score in zip(
            confidence_scores,
            self_scores,
        )
    ]

    methods = {
        "Confidence only": confidence_scores,
        "Question-aware semantic V2": question_aware_scores,
        "Confidence + question-aware semantic V2": (
            confidence_question_scores
        ),
        "Self-verifier only": self_scores,
        "Confidence + self-verifier": confidence_self_scores,
    }

    return correctness, methods


def run_bootstrap(
    input_path: Path,
    output_dir: Path,
    bootstrap_samples: int,
    seed: int,
    order_seed: int,
) -> list[dict]:
    records = load_jsonl(input_path)

    if not records:
        raise ValueError("Input prediction file is empty.")

    # Match the exact deterministic ordering used by the final
    # nested sample-size evaluation. This also fixes score tie-breaking.
    records = deterministic_nested_order(records, seed=order_seed)

    correctness, methods = build_method_scores(records)

    total = len(records)
    rng = np.random.default_rng(seed)

    point_aurcs = {
        method: compute_aurc(correctness, scores)
        for method, scores in methods.items()
    }

    bootstrap_aurcs = {
        method: []
        for method in methods
    }

    method_arrays = {
        method: np.asarray(scores, dtype=float)
        for method, scores in methods.items()
    }

    correctness_array = np.asarray(correctness, dtype=bool)

    for iteration in range(bootstrap_samples):
        # Sorting preserves the final deterministic record order
        # inside each bootstrap multiset, preventing random tie-breaking
        # from becoming an extra source of noise.
        sampled_indices = np.sort(
            rng.integers(
                0,
                total,
                size=total,
            )
        )

        sampled_correctness = correctness_array[
            sampled_indices
        ].tolist()

        for method, score_array in method_arrays.items():
            sampled_scores = score_array[sampled_indices].tolist()

            bootstrap_aurcs[method].append(
                compute_aurc(
                    sampled_correctness,
                    sampled_scores,
                )
            )

        if (iteration + 1) % 500 == 0:
            print(
                f"Bootstrap: {iteration + 1}/{bootstrap_samples}"
            )

    baseline_name = "Confidence only"
    baseline_bootstrap = np.asarray(
        bootstrap_aurcs[baseline_name],
        dtype=float,
    )

    rows: list[dict] = []

    for method in methods:
        method_values = bootstrap_aurcs[method]
        ci_low, ci_high = percentile_ci(method_values)

        if method == baseline_name:
            delta_point = 0.0
            delta_low = 0.0
            delta_high = 0.0
            fraction_better = None
        else:
            delta_values = (
                np.asarray(method_values, dtype=float)
                - baseline_bootstrap
            )

            delta_low, delta_high = percentile_ci(
                delta_values.tolist()
            )

            delta_point = (
                point_aurcs[method]
                - point_aurcs[baseline_name]
            )

            fraction_better = float(
                np.mean(delta_values < 0.0)
            )

        rows.append(
            {
                "method": method,
                "n": total,
                "aurc": point_aurcs[method],
                "aurc_ci95_low": ci_low,
                "aurc_ci95_high": ci_high,
                "delta_aurc_vs_confidence": delta_point,
                "delta_ci95_low": delta_low,
                "delta_ci95_high": delta_high,
                "bootstrap_fraction_better_than_confidence": (
                    fraction_better
                ),
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_seed": seed,
                "order_seed": order_seed,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "bootstrap_aurc_summary.csv"
    json_path = output_dir / "bootstrap_aurc_summary.json"

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            rows,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("Bootstrap uncertainty analysis complete.")
    print(f"Examples: {total}")
    print(f"Bootstrap samples: {bootstrap_samples}")
    print(f"Bootstrap seed: {seed}")
    print(f"Order seed: {order_seed}")
    print()

    for row in rows:
        print(
            f"{row['method']}: "
            f"AURC={row['aurc']:.6f} "
            f"[{row['aurc_ci95_low']:.6f}, "
            f"{row['aurc_ci95_high']:.6f}]"
        )

        if row["method"] != baseline_name:
            print(
                "  Δ vs confidence="
                f"{row['delta_aurc_vs_confidence']:.6f} "
                f"[{row['delta_ci95_low']:.6f}, "
                f"{row['delta_ci95_high']:.6f}]"
            )

    print()
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")

    return rows


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired bootstrap uncertainty analysis for final AURC results."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Bootstrap resampling seed.",
    )

    parser.add_argument(
        "--order-seed",
        type=int,
        default=17,
        help="Deterministic final-evaluation ordering seed.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_bootstrap(
        input_path=args.input,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        order_seed=args.order_seed,
    )
