"""
Risk–coverage evaluation for selective question answering.

This script compares:

1. Calibrated confidence
2. Hybrid verification score

It calculates:

- Coverage
- Selective accuracy
- Selective risk
- Risk–coverage curves
- Area Under the Risk–Coverage Curve (AURC)
- Matched-coverage comparisons

Gold labels are used only for evaluation.
"""

import argparse
import csv
import json
import re
import string
from collections import Counter
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Any

from src.utils.io import load_jsonl

DEFAULT_INPUT_PATH = Path("outputs/predictions/calibration_with_hybrid_evidence.jsonl")

DEFAULT_OUTPUT_DIR = Path("outputs/evaluation/risk_coverage")

DEFAULT_RELAXED_F1_THRESHOLD = 0.80

DEFAULT_THRESHOLD_START = 0.00
DEFAULT_THRESHOLD_END = 1.00
DEFAULT_THRESHOLD_STEP = 0.01


def get_first_value(
    prediction: dict[str, Any],
    field_names: tuple[str, ...],
    default: Any = None,
) -> Any:
    """
    Return the first available non-None field value.
    """

    for field_name in field_names:
        value = prediction.get(field_name)

        if value is not None:
            return value

    return default


def get_first_numeric_value(
    prediction: dict[str, Any],
    field_names: tuple[str, ...],
    default: float = 0.0,
) -> float:
    """
    Return the first available numeric field value.
    """

    value = get_first_value(
        prediction=prediction,
        field_names=field_names,
        default=default,
    )

    try:
        return float(value)

    except (TypeError, ValueError):
        return float(default)


def clamp_probability(
    value: float,
) -> float:
    """
    Clamp a numeric value to the [0, 1] interval.
    """

    return max(
        0.0,
        min(
            1.0,
            float(value),
        ),
    )


def get_predicted_answer(
    prediction: dict[str, Any],
) -> str:
    """
    Extract the QA model's predicted answer.
    """

    value = get_first_value(
        prediction=prediction,
        field_names=(
            "predicted_answer",
            "prediction_text",
            "prediction_answer",
            "answer",
        ),
        default="",
    )

    return str(value).strip()


def get_reference_answers(
    prediction: dict[str, Any],
) -> list[str]:
    """
    Extract reference answers from common dataset formats.
    """

    value = get_first_value(
        prediction=prediction,
        field_names=(
            "reference_answers",
            "gold_answers",
            "answers",
            "reference_answer",
            "gold_answer",
        ),
        default=[],
    )

    if isinstance(value, dict):
        text_values = value.get(
            "text",
            [],
        )

        if isinstance(text_values, list):
            return [str(answer).strip() for answer in text_values if answer is not None]

        if text_values is not None:
            return [str(text_values).strip()]

        return []

    if isinstance(value, list):
        answers: list[str] = []

        for item in value:
            if isinstance(item, dict):
                text_value = item.get(
                    "text",
                    "",
                )

                if isinstance(
                    text_value,
                    list,
                ):
                    answers.extend(
                        str(answer).strip()
                        for answer in text_value
                        if answer is not None
                    )

                elif text_value is not None:
                    answers.append(str(text_value).strip())

            elif item is not None:
                answers.append(str(item).strip())

        return answers

    if value is None:
        return []

    return [str(value).strip()]


def get_is_answerable(
    prediction: dict[str, Any],
) -> bool:
    """
    Extract the gold answerability label.

    If no explicit label is present, infer answerability
    from whether reference answers contain non-empty text.
    """

    explicit_value = get_first_value(
        prediction=prediction,
        field_names=(
            "is_answerable",
            "answerable",
            "gold_is_answerable",
        ),
        default=None,
    )

    if explicit_value is not None:
        if isinstance(
            explicit_value,
            bool,
        ):
            return explicit_value

        if isinstance(
            explicit_value,
            (int, float),
        ):
            return bool(explicit_value)

        normalized_value = str(explicit_value).strip().lower()

        if normalized_value in {
            "true",
            "1",
            "yes",
            "answerable",
        }:
            return True

        if normalized_value in {
            "false",
            "0",
            "no",
            "unanswerable",
        }:
            return False

    reference_answers = get_reference_answers(prediction)

    return any(answer.strip() for answer in reference_answers)


def normalize_answer(
    text: str,
) -> str:
    """
    Apply standard SQuAD-style normalization.
    """

    normalized_text = str(text).lower()

    normalized_text = "".join(
        character
        for character in normalized_text
        if character not in string.punctuation
    )

    normalized_text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        normalized_text,
    )

    normalized_text = " ".join(normalized_text.split())

    return normalized_text


def exact_match_score(
    predicted_answer: str,
    reference_answer: str,
) -> float:
    """
    Calculate normalized exact match.
    """

    return float(
        normalize_answer(predicted_answer) == normalize_answer(reference_answer)
    )


def token_f1_score(
    predicted_answer: str,
    reference_answer: str,
) -> float:
    """
    Calculate token-level F1.
    """

    predicted_tokens = normalize_answer(predicted_answer).split()

    reference_tokens = normalize_answer(reference_answer).split()

    if not predicted_tokens and not reference_tokens:
        return 1.0

    if not predicted_tokens or not reference_tokens:
        return 0.0

    common_tokens = Counter(predicted_tokens) & Counter(reference_tokens)

    overlap_count = sum(common_tokens.values())

    if overlap_count == 0:
        return 0.0

    precision = overlap_count / len(predicted_tokens)

    recall = overlap_count / len(reference_tokens)

    return 2.0 * precision * recall / (precision + recall)


def calculate_answer_scores(
    predicted_answer: str,
    reference_answers: list[str],
) -> tuple[float, float]:
    """
    Return maximum EM and F1 across reference answers.
    """

    if not reference_answers:
        is_empty_prediction = float(normalize_answer(predicted_answer) == "")

        return (
            is_empty_prediction,
            is_empty_prediction,
        )

    exact_match = max(
        exact_match_score(
            predicted_answer,
            reference_answer,
        )
        for reference_answer in reference_answers
    )

    token_f1 = max(
        token_f1_score(
            predicted_answer,
            reference_answer,
        )
        for reference_answer in reference_answers
    )

    return (
        exact_match,
        token_f1,
    )


def is_prediction_correct(
    prediction: dict[str, Any],
    relaxed_f1_threshold: float,
) -> tuple[
    bool,
    float,
    float,
]:
    """
    Determine whether a QA prediction is correct.

    Answerable examples are correct when:

    - Exact Match equals 1, or
    - Token F1 reaches the relaxed threshold.

    Unanswerable examples are correct only when the
    predicted answer is empty.
    """

    predicted_answer = get_predicted_answer(prediction)

    reference_answers = get_reference_answers(prediction)

    is_answerable = get_is_answerable(prediction)

    exact_match, token_f1 = calculate_answer_scores(
        predicted_answer=(predicted_answer),
        reference_answers=(reference_answers),
    )

    if not is_answerable:
        is_correct = normalize_answer(predicted_answer) == ""

        return (
            is_correct,
            exact_match,
            token_f1,
        )

    is_correct = exact_match == 1.0 or token_f1 >= relaxed_f1_threshold

    return (
        is_correct,
        exact_match,
        token_f1,
    )


def get_calibrated_confidence(
    prediction: dict[str, Any],
) -> float:
    """
    Extract calibrated QA confidence.
    """

    confidence = get_first_numeric_value(
        prediction=prediction,
        field_names=(
            "calibrated_confidence",
            "confidence_calibrated",
            "calibrated_probability",
            "confidence",
            "raw_confidence",
        ),
        default=0.0,
    )

    return clamp_probability(confidence)


def get_hybrid_score(
    prediction: dict[str, Any],
) -> float:
    """
    Extract the hybrid verification score.
    """

    hybrid_score = get_first_numeric_value(
        prediction=prediction,
        field_names=(
            "hybrid_evidence_score",
            "hybrid_score",
        ),
        default=0.0,
    )

    return clamp_probability(hybrid_score)


def validate_predictions(
    predictions: list[dict[str, Any]],
) -> None:
    """
    Validate fields required for risk–coverage analysis.
    """

    if not predictions:
        raise ValueError("Prediction list cannot be empty.")

    confidence_fields = (
        "calibrated_confidence",
        "confidence_calibrated",
        "calibrated_probability",
        "confidence",
        "raw_confidence",
    )

    hybrid_fields = (
        "hybrid_evidence_score",
        "hybrid_score",
    )

    for index, prediction in enumerate(
        predictions,
        start=1,
    ):
        if not any(field_name in prediction for field_name in confidence_fields):
            raise ValueError(f"Prediction {index} does not contain a confidence field.")

        if not any(field_name in prediction for field_name in hybrid_fields):
            raise ValueError(
                f"Prediction {index} does not contain a hybrid score field."
            )


def generate_thresholds(
    start: float,
    end: float,
    step: float,
) -> list[float]:
    """
    Generate inclusive thresholds using stable rounding.
    """

    if step <= 0.0:
        raise ValueError("Threshold step must be positive.")

    if start > end:
        raise ValueError("Threshold start must not exceed end.")

    if start < 0.0 or end > 1.0:
        raise ValueError("Threshold range must be inside [0, 1].")

    thresholds: list[float] = []

    current_value = start

    while current_value <= (end + 1e-12):
        thresholds.append(
            round(
                current_value,
                10,
            )
        )

        current_value += step

    return thresholds


def evaluate_at_threshold(
    predictions: list[dict[str, Any]],
    correctness: list[bool],
    score_function: Callable[
        [dict[str, Any]],
        float,
    ],
    threshold: float,
) -> dict[str, Any]:
    """
    Evaluate one scoring system at one threshold.
    """

    if len(predictions) != len(correctness):
        raise ValueError("Prediction and correctness counts must match.")

    answered_indices = [
        index
        for index, prediction in enumerate(predictions)
        if score_function(prediction) >= threshold
    ]

    answered_count = len(answered_indices)

    total_count = len(predictions)

    abstained_count = total_count - answered_count

    correct_answered = sum(1 for index in answered_indices if correctness[index])

    wrong_answered = answered_count - correct_answered

    coverage = answered_count / total_count if total_count else 0.0

    selective_accuracy = correct_answered / answered_count if answered_count else 1.0

    selective_risk = wrong_answered / answered_count if answered_count else 0.0

    return {
        "threshold": threshold,
        "total": total_count,
        "answered": answered_count,
        "abstained": abstained_count,
        "coverage": coverage,
        "selective_accuracy": (selective_accuracy),
        "selective_risk": (selective_risk),
        "correct_answered": (correct_answered),
        "wrong_answered": (wrong_answered),
    }


def sweep_thresholds(
    predictions: list[dict[str, Any]],
    correctness: list[bool],
    score_function: Callable[
        [dict[str, Any]],
        float,
    ],
    thresholds: list[float],
) -> list[dict[str, Any]]:
    """
    Evaluate a scoring system across thresholds.
    """

    results: list[dict[str, Any]] = []

    for threshold in thresholds:
        metrics = evaluate_at_threshold(
            predictions=predictions,
            correctness=correctness,
            score_function=(score_function),
            threshold=threshold,
        )

        results.append(metrics)

    return results


def build_exact_risk_coverage_curve(
    predictions: list[dict[str, Any]],
    correctness: list[bool],
    score_function: Callable[
        [dict[str, Any]],
        float,
    ],
) -> list[dict[str, Any]]:
    """
    Build an exact risk–coverage curve by ranking all
    examples from highest score to lowest score.

    This avoids dependence on an arbitrary threshold step.
    """

    scored_examples = [
        {
            "score": score_function(prediction),
            "correct": is_correct,
        }
        for prediction, is_correct in zip(
            predictions,
            correctness,
        )
    ]

    scored_examples.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    total_count = len(scored_examples)

    curve: list[dict[str, Any]] = [
        {
            "answered": 0,
            "coverage": 0.0,
            "selective_accuracy": 1.0,
            "selective_risk": 0.0,
            "minimum_score": 1.0,
        }
    ]

    correct_answered = 0
    wrong_answered = 0

    for index, example in enumerate(
        scored_examples,
        start=1,
    ):
        if example["correct"]:
            correct_answered += 1

        else:
            wrong_answered += 1

        selective_accuracy = correct_answered / index

        selective_risk = wrong_answered / index

        coverage = index / total_count

        curve.append(
            {
                "answered": index,
                "coverage": coverage,
                "selective_accuracy": (selective_accuracy),
                "selective_risk": (selective_risk),
                "minimum_score": (example["score"]),
            }
        )

    return curve


def calculate_aurc(
    curve: list[dict[str, Any]],
) -> float:
    """
    Calculate Area Under the Risk–Coverage Curve
    using the trapezoidal rule.

    Lower AURC is better.
    """

    if len(curve) < 2:
        return 0.0

    sorted_curve = sorted(
        curve,
        key=lambda point: point["coverage"],
    )

    area = 0.0

    for left_point, right_point in pairwise(sorted_curve):
        coverage_difference = right_point["coverage"] - left_point["coverage"]

        average_risk = (
            left_point["selective_risk"] + right_point["selective_risk"]
        ) / 2.0

        area += coverage_difference * average_risk

    return area


def find_closest_coverage_result(
    results: list[dict[str, Any]],
    target_coverage: float,
) -> dict[str, Any]:
    """
    Find the threshold result closest to target coverage.
    """

    if not results:
        raise ValueError("Results cannot be empty.")

    return min(
        results,
        key=lambda result: (
            abs(result["coverage"] - target_coverage),
            result["selective_risk"],
        ),
    )


def compare_at_target_coverages(
    confidence_results: list[dict[str, Any]],
    hybrid_results: list[dict[str, Any]],
    target_coverages: list[float],
) -> list[dict[str, Any]]:
    """
    Compare confidence and hybrid systems near
    identical coverage levels.
    """

    comparisons: list[dict[str, Any]] = []

    for target_coverage in target_coverages:
        confidence_result = find_closest_coverage_result(
            results=confidence_results,
            target_coverage=(target_coverage),
        )

        hybrid_result = find_closest_coverage_result(
            results=hybrid_results,
            target_coverage=(target_coverage),
        )

        comparisons.append(
            {
                "target_coverage": (target_coverage),
                "confidence_threshold": (confidence_result["threshold"]),
                "confidence_coverage": (confidence_result["coverage"]),
                "confidence_accuracy": (confidence_result["selective_accuracy"]),
                "confidence_risk": (confidence_result["selective_risk"]),
                "confidence_answered": (confidence_result["answered"]),
                "hybrid_threshold": (hybrid_result["threshold"]),
                "hybrid_coverage": (hybrid_result["coverage"]),
                "hybrid_accuracy": (hybrid_result["selective_accuracy"]),
                "hybrid_risk": (hybrid_result["selective_risk"]),
                "hybrid_answered": (hybrid_result["answered"]),
                "risk_difference": (
                    hybrid_result["selective_risk"]
                    - confidence_result["selective_risk"]
                ),
            }
        )

    return comparisons


def save_csv(
    rows: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """
    Save dictionaries as a CSV file.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        output_path.write_text(
            "",
            encoding="utf-8",
        )

        return

    field_names = list(rows[0].keys())

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=field_names,
        )

        writer.writeheader()
        writer.writerows(rows)


def save_json(
    data: Any,
    output_path: str | Path,
) -> None:
    """
    Save data as formatted JSON.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            data,
            output_file,
            indent=2,
            ensure_ascii=False,
        )


def print_threshold_summary(
    system_name: str,
    results: list[dict[str, Any]],
    selected_thresholds: list[float],
) -> None:
    """
    Print selected threshold results.
    """

    print("\n" + "=" * 78)

    print(system_name)

    print("=" * 78)

    print(
        f"{'Threshold':>10}"
        f"{'Coverage':>12}"
        f"{'Accuracy':>12}"
        f"{'Risk':>12}"
        f"{'Answered':>12}"
        f"{'Wrong':>10}"
    )

    print("-" * 78)

    for selected_threshold in selected_thresholds:
        closest_result = min(
            results,
            key=lambda result: abs(result["threshold"] - selected_threshold),
        )

        print(
            f"{closest_result['threshold']:>10.2f}"
            f"{closest_result['coverage']:>12.4f}"
            f"{closest_result['selective_accuracy']:>12.4f}"
            f"{closest_result['selective_risk']:>12.4f}"
            f"{closest_result['answered']:>12}"
            f"{closest_result['wrong_answered']:>10}"
        )


def print_matched_coverage_table(
    comparisons: list[dict[str, Any]],
) -> None:
    """
    Print matched-coverage comparisons.
    """

    print("\n" + "=" * 112)

    print("MATCHED-COVERAGE COMPARISON")

    print("=" * 112)

    print(
        f"{'Target':>8}"
        f"{'Conf cov':>10}"
        f"{'Conf risk':>11}"
        f"{'Conf thr':>10}"
        f"{'Hybrid cov':>12}"
        f"{'Hybrid risk':>13}"
        f"{'Hybrid thr':>12}"
        f"{'Risk Δ':>10}"
    )

    print("-" * 112)

    for comparison in comparisons:
        print(
            f"{comparison['target_coverage']:>8.2f}"
            f"{comparison['confidence_coverage']:>10.4f}"
            f"{comparison['confidence_risk']:>11.4f}"
            f"{comparison['confidence_threshold']:>10.2f}"
            f"{comparison['hybrid_coverage']:>12.4f}"
            f"{comparison['hybrid_risk']:>13.4f}"
            f"{comparison['hybrid_threshold']:>12.2f}"
            f"{comparison['risk_difference']:>10.4f}"
        )


def run_risk_coverage_evaluation(
    input_path: str | Path,
    output_dir: str | Path,
    relaxed_f1_threshold: float,
    threshold_start: float,
    threshold_end: float,
    threshold_step: float,
) -> dict[str, Any]:
    """
    Run complete risk–coverage evaluation.
    """

    predictions = load_jsonl(input_path)

    validate_predictions(predictions)

    if not (0.0 <= relaxed_f1_threshold <= 1.0):
        raise ValueError("relaxed_f1_threshold must be between 0 and 1.")

    correctness: list[bool] = []

    exact_match_scores: list[float] = []

    token_f1_scores: list[float] = []

    for prediction in predictions:
        (
            is_correct,
            exact_match,
            token_f1,
        ) = is_prediction_correct(
            prediction=prediction,
            relaxed_f1_threshold=(relaxed_f1_threshold),
        )

        correctness.append(is_correct)

        exact_match_scores.append(exact_match)

        token_f1_scores.append(token_f1)

    thresholds = generate_thresholds(
        start=threshold_start,
        end=threshold_end,
        step=threshold_step,
    )

    confidence_results = sweep_thresholds(
        predictions=predictions,
        correctness=correctness,
        score_function=(get_calibrated_confidence),
        thresholds=thresholds,
    )

    hybrid_results = sweep_thresholds(
        predictions=predictions,
        correctness=correctness,
        score_function=(get_hybrid_score),
        thresholds=thresholds,
    )

    confidence_curve = build_exact_risk_coverage_curve(
        predictions=predictions,
        correctness=correctness,
        score_function=(get_calibrated_confidence),
    )

    hybrid_curve = build_exact_risk_coverage_curve(
        predictions=predictions,
        correctness=correctness,
        score_function=(get_hybrid_score),
    )

    confidence_aurc = calculate_aurc(confidence_curve)

    hybrid_aurc = calculate_aurc(hybrid_curve)

    target_coverages = [
        0.10,
        0.20,
        0.30,
        0.32,
        0.40,
        0.50,
        0.54,
        0.60,
        0.70,
        0.80,
        0.90,
        1.00,
    ]

    comparisons = compare_at_target_coverages(
        confidence_results=(confidence_results),
        hybrid_results=(hybrid_results),
        target_coverages=(target_coverages),
    )

    total_count = len(predictions)

    correct_count = sum(correctness)

    incorrect_count = total_count - correct_count

    average_exact_match = sum(exact_match_scores) / total_count

    average_token_f1 = sum(token_f1_scores) / total_count

    summary = {
        "input_path": str(input_path),
        "total_predictions": (total_count),
        "correct_predictions": (correct_count),
        "incorrect_predictions": (incorrect_count),
        "relaxed_f1_threshold": (relaxed_f1_threshold),
        "average_exact_match": (average_exact_match),
        "average_token_f1": (average_token_f1),
        "confidence_aurc": (confidence_aurc),
        "hybrid_aurc": (hybrid_aurc),
        "aurc_difference": (hybrid_aurc - confidence_aurc),
        "aurc_relative_change": (
            (hybrid_aurc - confidence_aurc) / confidence_aurc
            if confidence_aurc
            else 0.0
        ),
    }

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_csv(
        rows=confidence_results,
        output_path=(output_dir / "confidence_threshold_sweep.csv"),
    )

    save_csv(
        rows=hybrid_results,
        output_path=(output_dir / "hybrid_threshold_sweep.csv"),
    )

    save_csv(
        rows=confidence_curve,
        output_path=(output_dir / "confidence_risk_coverage_curve.csv"),
    )

    save_csv(
        rows=hybrid_curve,
        output_path=(output_dir / "hybrid_risk_coverage_curve.csv"),
    )

    save_csv(
        rows=comparisons,
        output_path=(output_dir / "matched_coverage_comparison.csv"),
    )

    save_json(
        data=summary,
        output_path=(output_dir / "risk_coverage_summary.json"),
    )

    print("\nRisk–coverage evaluation completed.")

    print(f"Input: {input_path}")

    print(f"Total predictions: {total_count}")

    print(f"Correct predictions: {correct_count}")

    print(f"Incorrect predictions: {incorrect_count}")

    print(f"Average Exact Match: {average_exact_match:.4f}")

    print(f"Average Token F1: {average_token_f1:.4f}")

    selected_thresholds = [
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95,
    ]

    print_threshold_summary(
        system_name=("CONFIDENCE THRESHOLD SWEEP"),
        results=confidence_results,
        selected_thresholds=(selected_thresholds),
    )

    print_threshold_summary(
        system_name=("HYBRID THRESHOLD SWEEP"),
        results=hybrid_results,
        selected_thresholds=(selected_thresholds),
    )

    print_matched_coverage_table(comparisons)

    print("\n" + "=" * 60)

    print("AURC SUMMARY")

    print("=" * 60)

    print(f"Confidence AURC: {confidence_aurc:.6f}")

    print(f"Hybrid AURC:     {hybrid_aurc:.6f}")

    print(f"Hybrid - Confidence: {hybrid_aurc - confidence_aurc:.6f}")

    if hybrid_aurc < confidence_aurc:
        print("Result: Hybrid ranking is better (lower AURC).")

    elif hybrid_aurc > confidence_aurc:
        print("Result: Confidence ranking is better (lower AURC).")

    else:
        print("Result: Both systems have equal AURC.")

    print(f"\nResults saved to: {output_dir}")

    return {
        "summary": summary,
        "confidence_threshold_sweep": (confidence_results),
        "hybrid_threshold_sweep": (hybrid_results),
        "confidence_curve": (confidence_curve),
        "hybrid_curve": (hybrid_curve),
        "matched_coverage": (comparisons),
    }


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Compare calibrated confidence and hybrid "
            "verification using risk–coverage curves."
        )
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )

    parser.add_argument(
        "--relaxed-f1-threshold",
        type=float,
        default=(DEFAULT_RELAXED_F1_THRESHOLD),
    )

    parser.add_argument(
        "--threshold-start",
        type=float,
        default=(DEFAULT_THRESHOLD_START),
    )

    parser.add_argument(
        "--threshold-end",
        type=float,
        default=(DEFAULT_THRESHOLD_END),
    )

    parser.add_argument(
        "--threshold-step",
        type=float,
        default=(DEFAULT_THRESHOLD_STEP),
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    run_risk_coverage_evaluation(
        input_path=arguments.input,
        output_dir=arguments.output_dir,
        relaxed_f1_threshold=(arguments.relaxed_f1_threshold),
        threshold_start=(arguments.threshold_start),
        threshold_end=(arguments.threshold_end),
        threshold_step=(arguments.threshold_step),
    )
