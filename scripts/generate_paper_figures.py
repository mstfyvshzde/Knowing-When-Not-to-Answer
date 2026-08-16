"""Generate publication-facing paper figures from canonical tracked artifacts.

This script does not rerun model inference or recompute experimental results.
It reads the version-controlled final evaluation CSV files and regenerates the
paper figures with publication terminology while preserving the historical
internal method names stored in experiment artifacts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SAMPLE_SIZE_INPUT = Path(
    "outputs/evaluation/final_sample_size_comparison/"
    "sample_size_comparison.csv"
)

RISK_COVERAGE_INPUT = Path(
    "outputs/evaluation/final_sample_size_comparison/"
    "n_3000/risk_coverage_curves.csv"
)

SAMPLE_SIZE_OUTPUT = Path(
    "assets/figures/aurc_by_sample_size.png"
)

RISK_COVERAGE_OUTPUT = Path(
    "assets/figures/risk_coverage_curves.png"
)

# Canonical experiment artifacts retain historical internal method names.
# Publication figures map those names to the terminology used in the paper.
METHOD_ORDER = (
    "Confidence only",
    "Question-aware semantic V2",
    "Confidence + question-aware semantic V2",
    "Self-verifier only",
    "Confidence + self-verifier",
)

PAPER_LABELS = {
    "Confidence only": "Confidence only",
    "Question-aware semantic V2": "Question-aware verification",
    "Confidence + question-aware semantic V2": (
        "Confidence + question-aware"
    ),
    "Self-verifier only": "Answer-support verification",
    "Confidence + self-verifier": "Confidence + answer-support",
}


def require_columns(
    dataframe: pd.DataFrame,
    required: set[str],
    source: Path,
) -> None:
    """Fail loudly when a canonical artifact has an unexpected schema."""

    missing = required - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"{source} is missing required columns: {sorted(missing)}."
        )


def require_primary_methods(
    dataframe: pd.DataFrame,
    source: Path,
) -> None:
    """Ensure every primary ranking method is present before plotting."""

    observed = set(
        dataframe["method"]
        .dropna()
        .astype(str)
    )

    missing = set(METHOD_ORDER) - observed

    if missing:
        raise ValueError(
            f"{source} is missing primary methods: {sorted(missing)}."
        )


def generate_sample_size_figure() -> None:
    """Plot canonical AURC across deterministic nested sample sizes."""

    dataframe = pd.read_csv(SAMPLE_SIZE_INPUT)

    require_columns(
        dataframe,
        {"sample_size", "method", "aurc"},
        SAMPLE_SIZE_INPUT,
    )
    require_primary_methods(
        dataframe,
        SAMPLE_SIZE_INPUT,
    )

    figure = plt.figure(figsize=(9, 6))
    axis = figure.add_subplot(111)

    for method in METHOD_ORDER:
        subset = (
            dataframe[
                dataframe["method"] == method
            ]
            .sort_values("sample_size")
        )

        axis.plot(
            subset["sample_size"],
            subset["aurc"],
            marker="o",
            label=PAPER_LABELS[method],
        )

    axis.set_xlabel("Sample size")
    axis.set_ylabel("AURC (lower is better)")
    axis.set_title("AURC Stability Across Nested Sample Sizes")
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()
    SAMPLE_SIZE_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        SAMPLE_SIZE_OUTPUT,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def generate_risk_coverage_figure() -> None:
    """Plot canonical N=3000 risk--coverage curves with paper terminology."""

    dataframe = pd.read_csv(RISK_COVERAGE_INPUT)

    require_columns(
        dataframe,
        {"method", "coverage", "risk"},
        RISK_COVERAGE_INPUT,
    )
    require_primary_methods(
        dataframe,
        RISK_COVERAGE_INPUT,
    )

    figure = plt.figure(figsize=(10, 6))
    axis = figure.add_subplot(111)

    for method in METHOD_ORDER:
        subset = (
            dataframe[
                dataframe["method"] == method
            ]
            .sort_values("coverage")
        )

        axis.plot(
            subset["coverage"],
            subset["risk"],
            label=PAPER_LABELS[method],
        )

    axis.set_xlabel("Coverage")
    axis.set_ylabel("Selective risk")
    axis.set_title("Risk-Coverage Curves on the Final Held-Out Set")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()
    RISK_COVERAGE_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        RISK_COVERAGE_OUTPUT,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    """Regenerate both publication-facing evaluation figures."""

    generate_sample_size_figure()
    generate_risk_coverage_figure()

    print(f"Saved: {SAMPLE_SIZE_OUTPUT}")
    print(f"Saved: {RISK_COVERAGE_OUTPUT}")


if __name__ == "__main__":
    main()
