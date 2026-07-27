import json
import re
import string
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INPUT_PATH = Path(
    "outputs/predictions/calibration_final_decisions.jsonl"
)

OUTPUT_PATH = Path(
    "assets/figures/calibration_curve.png"
)

RELAXED_F1_THRESHOLD = 0.80


def normalize_answer(text: str) -> str:
    text = text.lower()

    text = "".join(
        character
        for character in text
        if character not in string.punctuation
    )

    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())

    return text


def exact_match_score(
    prediction: str,
    references: list[str],
) -> float:
    normalized_prediction = normalize_answer(prediction)

    return max(
        (
            float(
                normalized_prediction
                == normalize_answer(reference)
            )
            for reference in references
        ),
        default=0.0,
    )


def token_f1_score(
    prediction: str,
    references: list[str],
) -> float:
    prediction_tokens = normalize_answer(prediction).split()

    best_score = 0.0

    for reference in references:
        reference_tokens = normalize_answer(reference).split()

        common_tokens = Counter(prediction_tokens) & Counter(reference_tokens)
        overlap = sum(common_tokens.values())

        if not prediction_tokens or not reference_tokens:
            score = float(prediction_tokens == reference_tokens)

        elif overlap == 0:
            score = 0.0

        else:
            precision = overlap / len(prediction_tokens)
            recall = overlap / len(reference_tokens)

            score = (
                2 * precision * recall
                / (precision + recall)
            )

        best_score = max(best_score, score)

    return best_score


def calculate_correctness(row: dict) -> int:
    prediction = row.get("prediction_text", "")
    references = row.get("reference_answers", [])
    is_answerable = row.get("is_answerable", True)

    if not is_answerable:
        return int(normalize_answer(prediction) == "")

    exact_match = exact_match_score(
        prediction,
        references,
    )

    token_f1 = token_f1_score(
        prediction,
        references,
    )

    return int(
        exact_match == 1.0
        or token_f1 >= RELAXED_F1_THRESHOLD
    )


def load_predictions(path: Path):
    uncalibrated = []
    calibrated = []
    correctness = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)

            if (
                "uncalibrated_confidence" not in row
                or "calibrated_confidence" not in row
            ):
                continue

            uncalibrated.append(
                row["uncalibrated_confidence"]
            )

            calibrated.append(
                row["calibrated_confidence"]
            )

            correctness.append(
                calculate_correctness(row)
            )

    return (
        np.array(uncalibrated),
        np.array(calibrated),
        np.array(correctness),
    )


def bin_accuracy(
    confidences,
    correctness,
    bins=10,
):
    edges = np.linspace(0.0, 1.0, bins + 1)

    mean_confidence = []
    mean_accuracy = []

    for lower, upper in zip(
        edges[:-1],
        edges[1:],
    ):
        mask = (
            (confidences >= lower)
            & (confidences < upper)
        )

        if upper == 1.0:
            mask = (
                (confidences >= lower)
                & (confidences <= upper)
            )

        if not np.any(mask):
            continue

        mean_confidence.append(
            confidences[mask].mean()
        )

        mean_accuracy.append(
            correctness[mask].mean()
        )

    return mean_confidence, mean_accuracy

def generate_ablation_figure():
    import pandas as pd

    input_path = Path(
        "outputs/evaluation/question_aware_ablation/ablation_summary.csv"
    )

    output_path = Path(
        "assets/figures/ablation_results.png"
    )

    df = pd.read_csv(input_path)

    labels = {
        "Confidence only": "Confidence",
        "Old semantic verifier only": "Old semantic",
        "Question-aware semantic V2": "QA semantic V2",
        "Confidence + question-aware semantic V2": "Confidence + QA",
    }

    df["label"] = df["method"].map(labels)

    df = df.sort_values(
        "aurc",
        ascending=True,
    )

    plt.figure(figsize=(9, 6))

    bars = plt.barh(
        df["label"],
        df["aurc"],
    )

    plt.xlabel("AURC — lower is better")
    plt.ylabel("Method")
    plt.title("Question-Aware Semantic Verifier Ablation")

    for bar, value in zip(
        bars,
        df["aurc"],
    ):
        plt.text(
            value + 0.004,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
        )

    plt.xlim(
        0,
        df["aurc"].max() + 0.07,
    )

    plt.grid(
        axis="x",
        alpha=0.3,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved figure to: {output_path}")

def main():
    (
        uncalibrated,
        calibrated,
        correctness,
    ) = load_predictions(INPUT_PATH)

    raw_confidence, raw_accuracy = bin_accuracy(
        uncalibrated,
        correctness,
    )

    calibrated_confidence, calibrated_accuracy = bin_accuracy(
        calibrated,
        correctness,
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    plt.plot(
        raw_confidence,
        raw_accuracy,
        marker="o",
        label="Uncalibrated",
    )

    plt.plot(
        calibrated_confidence,
        calibrated_accuracy,
        marker="o",
        label="Temperature scaled",
    )

    plt.xlabel("Mean confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Confidence Calibration")
    plt.legend()
    plt.grid(alpha=0.3)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    print(f"Samples: {len(correctness)}")
    print(f"Accuracy: {correctness.mean():.4f}")
    print(f"Saved figure to: {OUTPUT_PATH}")
    generate_ablation_figure()


if __name__ == "__main__":
    main()