"""
Generate selected diagnostic figures from existing experiment artifacts.

This script currently produces two figures:

1. confidence calibration before and after temperature scaling;
2. a bar-chart summary of the question-aware verifier ablation artifact.

The calibration figure must use exactly the same QA-candidate correctness
definition as the calibration experiment itself. Therefore correctness is
imported from `src.calibration.calibration_metrics` rather than reimplemented
locally.

In the calibration experiment:

- answerable candidates are evaluated with normalized Exact Match;
- unanswerable forced-answer candidates are treated as incorrect.

The ablation bar chart is a visualization of an existing evaluation artifact.
It does not calculate new AURC values and should not be treated as an
independent experiment.

Final paper figures should always be checked against the canonical held-out
result tables before publication.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.calibration.calibration_metrics import (
    is_prediction_correct,
)
from src.utils.io import load_jsonl

CALIBRATION_INPUT_PATH = Path(
    "outputs/predictions/"
    "raw_baseline_calibrated_calibration.jsonl"
)

CALIBRATION_FIGURE_PATH = Path(
    "assets/figures/calibration_curve.png"
)

ABLATION_INPUT_PATH = Path(
    "outputs/evaluation/final_sample_size_comparison/"
    "n_3000/ablation_summary.csv"
)

ABLATION_FIGURE_PATH = Path(
    "assets/figures/ablation_results.png"
)


def validate_confidence(
    value: Any,
    field_name: str,
    record_index: int,
) -> float:
    """
    Convert one stored confidence value to a valid probability.

    Figure generation fails on missing or malformed confidence values instead
    of silently dropping records, because doing so could make the plotted
    calibration sample differ from the reported calibration experiment.
    """

    try:
        confidence = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            f"{field_name} is not numeric "
            f"at record {record_index}: "
            f"{value!r}."
        ) from error

    if (
        not np.isfinite(
            confidence
        )
        or not (
            0.0
            <= confidence
            <= 1.0
        )
    ):
        raise ValueError(
            f"{field_name} must be a finite "
            "probability in [0, 1] at "
            f"record {record_index}; "
            f"received {confidence!r}."
        )

    return confidence


def load_calibration_predictions(
    path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Load confidence pairs and the calibration experiment's correctness labels.

    Every record must contain both the confidence before temperature scaling and
    the corresponding calibrated confidence afterward. The same records and
    correctness definition are then used for both plotted curves.
    """

    records = load_jsonl(
        path
    )

    if not records:
        raise ValueError(
            "Calibration prediction file "
            "is empty."
        )

    uncalibrated: list[
        float
    ] = []

    calibrated: list[
        float
    ] = []

    correctness: list[
        int
    ] = []

    for index, row in enumerate(
        records
    ):
        if (
            "uncalibrated_confidence"
            not in row
        ):
            raise ValueError(
                "Missing "
                "uncalibrated_confidence "
                f"at record {index}."
            )

        if (
            "calibrated_confidence"
            not in row
        ):
            raise ValueError(
                "Missing calibrated_confidence "
                f"at record {index}."
            )

        uncalibrated.append(
            validate_confidence(
                row[
                    "uncalibrated_confidence"
                ],
                "uncalibrated_confidence",
                index,
            )
        )

        calibrated.append(
            validate_confidence(
                row[
                    "calibrated_confidence"
                ],
                "calibrated_confidence",
                index,
            )
        )

        # Reuse the exact correctness definition used by temperature-scaling
        # calibration rather than introducing a second metric definition here.
        correctness.append(
            int(
                is_prediction_correct(
                    row
                )
            )
        )

    return (
        np.asarray(
            uncalibrated,
            dtype=float,
        ),
        np.asarray(
            calibrated,
            dtype=float,
        ),
        np.asarray(
            correctness,
            dtype=float,
        ),
    )


def bin_accuracy(
    confidences: np.ndarray,
    correctness: np.ndarray,
    bins: int = 10,
) -> tuple[
    list[float],
    list[float],
]:
    """
    Aggregate confidence and empirical correctness into equal-width bins.

    Each plotted point represents:

        x-axis -> mean confidence inside one bin
        y-axis -> empirical candidate accuracy inside that bin

    Points closer to the diagonal indicate better agreement between confidence
    and observed correctness.

    This visualization complements quantitative calibration metrics such as ECE,
    Brier score, and NLL; it does not replace them.
    """

    if bins <= 0:
        raise ValueError(
            "bins must be positive."
        )

    if (
        confidences.size == 0
        or correctness.size == 0
    ):
        raise ValueError(
            "Calibration arrays cannot "
            "be empty."
        )

    if (
        confidences.shape
        != correctness.shape
    ):
        raise ValueError(
            "Confidence and correctness "
            "arrays must have matching shape."
        )

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    mean_confidence: list[
        float
    ] = []

    mean_accuracy: list[
        float
    ] = []

    for lower, upper in pairwise(
        edges
    ):
        # All normal bins are left-closed and right-open. The final bin includes
        # 1.0 so perfectly confident predictions are not omitted.
        if upper == 1.0:
            mask = (
                (
                    confidences
                    >= lower
                )
                & (
                    confidences
                    <= upper
                )
            )

        else:
            mask = (
                (
                    confidences
                    >= lower
                )
                & (
                    confidences
                    < upper
                )
            )

        if not np.any(
            mask
        ):
            continue

        mean_confidence.append(
            float(
                confidences[
                    mask
                ].mean()
            )
        )

        mean_accuracy.append(
            float(
                correctness[
                    mask
                ].mean()
            )
        )

    return (
        mean_confidence,
        mean_accuracy,
    )


def generate_calibration_figure() -> None:
    """
    Plot confidence reliability before and after temperature scaling.

    Both curves use identical examples and correctness labels. Temperature
    scaling changes probability calibration but, because it is monotonic in the
    confidence margin, this figure should not be interpreted as evidence of a
    changed confidence ranking.
    """

    (
        uncalibrated,
        calibrated,
        correctness,
    ) = load_calibration_predictions(
        CALIBRATION_INPUT_PATH
    )

    (
        raw_confidence,
        raw_accuracy,
    ) = bin_accuracy(
        uncalibrated,
        correctness,
    )

    (
        calibrated_confidence,
        calibrated_accuracy,
    ) = bin_accuracy(
        calibrated,
        correctness,
    )

    figure = plt.figure(
        figsize=(
            8,
            6,
        )
    )

    axis = figure.add_subplot(
        111
    )

    # The diagonal represents ideal calibration: predictions that receive
    # confidence p should be correct approximately p of the time.
    axis.plot(
        [
            0,
            1,
        ],
        [
            0,
            1,
        ],
        linestyle="--",
        label=(
            "Perfect calibration"
        ),
    )

    axis.plot(
        raw_confidence,
        raw_accuracy,
        marker="o",
        label=(
            "Uncalibrated"
        ),
    )

    axis.plot(
        calibrated_confidence,
        calibrated_accuracy,
        marker="o",
        label=(
            "Temperature scaled"
        ),
    )

    axis.set_xlabel(
        "Mean confidence"
    )

    axis.set_ylabel(
        "Empirical candidate accuracy"
    )

    axis.set_title(
        "Confidence Calibration"
    )

    axis.set_xlim(
        0.0,
        1.0,
    )

    axis.set_ylim(
        0.0,
        1.0,
    )

    axis.legend()

    axis.grid(
        alpha=0.3
    )

    figure.tight_layout()

    CALIBRATION_FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        CALIBRATION_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"Calibration samples: "
        f"{len(correctness)}"
    )

    print(
        "Candidate accuracy: "
        f"{correctness.mean():.4f}"
    )

    print(
        "Saved calibration figure to: "
        f"{CALIBRATION_FIGURE_PATH}"
    )


def generate_ablation_figure() -> None:
    """
    Visualize AURC values already produced by the ablation evaluator.

    This function does not recompute correctness, rankings, or AURC. It reads
    the stored experiment table and converts the reported values into a bar
    chart.

    The source artifact must therefore be checked for freshness before this
    figure is used in final documentation or a paper.
    """

    import pandas as pd

    if not ABLATION_INPUT_PATH.exists():
        raise FileNotFoundError(
            "Ablation summary was not found: "
            f"{ABLATION_INPUT_PATH}"
        )

    dataframe = pd.read_csv(
        ABLATION_INPUT_PATH
    )

    required_columns = {
        "method",
        "aurc",
    }

    missing_columns = (
        required_columns
        - set(
            dataframe.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Ablation summary is missing "
            "required columns: "
            f"{sorted(missing_columns)}."
        )

    if dataframe.empty:
        raise ValueError(
            "Ablation summary is empty."
        )

    # Short labels improve figure readability. Unknown/new method names are
    # preserved rather than silently converted to missing labels.
    labels = {
        "Confidence only": (
            "Confidence"
        ),
        "Old semantic verifier only": (
            "Old semantic"
        ),
        "Question-aware semantic V2": (
            "QA semantic V2"
        ),
        "Confidence + question-aware semantic V2": (
            "Confidence + QA"
        ),
        "Self-verifier only": (
            "Answer-support verifier"
        ),
        "Confidence + self-verifier": (
            "Confidence + answer-support"
        ),
    }

    dataframe[
        "label"
    ] = dataframe[
        "method"
    ].map(
        lambda method: labels.get(
            method,
            method,
        )
    )

    dataframe[
        "aurc"
    ] = pd.to_numeric(
        dataframe[
            "aurc"
        ],
        errors="raise",
    )

    if not np.all(
        np.isfinite(
            dataframe[
                "aurc"
            ].to_numpy(
                dtype=float
            )
        )
    ):
        raise ValueError(
            "Ablation AURC column contains "
            "non-finite values."
        )

    # Lower AURC is better, so sorting ascending places the strongest ranking
    # method first in the visual comparison.
    dataframe = (
        dataframe.sort_values(
            "aurc",
            ascending=True,
        )
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

    bars = axis.barh(
        dataframe[
            "label"
        ],
        dataframe[
            "aurc"
        ],
    )

    axis.set_xlabel(
        "AURC — lower is better"
    )

    axis.set_ylabel(
        "Method"
    )

    axis.set_title(
        "Question-Aware Verification Ablation"
    )

    for bar, value in zip(
        bars,
        dataframe[
            "aurc"
        ],
    ):
        axis.text(
            float(
                value
            )
            + 0.004,
            bar.get_y()
            + bar.get_height()
            / 2,
            f"{value:.3f}",
            va="center",
        )

    axis.set_xlim(
        0.0,
        float(
            dataframe[
                "aurc"
            ].max()
        )
        + 0.07,
    )

    axis.grid(
        axis="x",
        alpha=0.3,
    )

    figure.tight_layout()

    ABLATION_FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        ABLATION_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        "Saved ablation figure to: "
        f"{ABLATION_FIGURE_PATH}"
    )


def main() -> None:
    """
    Regenerate project figures from their existing experiment artifacts.
    """

    generate_calibration_figure()

    generate_ablation_figure()


if __name__ == "__main__":
    main()