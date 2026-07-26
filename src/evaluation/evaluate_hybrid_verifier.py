"""
Evaluate confidence-only, lexical, and hybrid abstention systems.

The evaluator measures:

1. Coverage
2. Selective accuracy
3. Selective risk
4. Wrong answered count
5. Correct rejected count
6. Wrong-supported count

Gold labels are used only during evaluation.
They are never used by the verifier itself.
"""

import argparse
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.io import load_jsonl

DEFAULT_INPUT_PATH = Path("outputs/predictions/calibration_with_hybrid_evidence.jsonl")

DEFAULT_CONFIDENCE_THRESHOLD = 0.50

DEFAULT_LEXICAL_SUPPORTED_LABEL = "SUPPORTED"
DEFAULT_HYBRID_SUPPORTED_LABEL = "SUPPORTED"

DEFAULT_RELAXED_F1_THRESHOLD = 0.80


def get_first_value(
    prediction: dict[str, Any],
    field_names: tuple[str, ...],
    default: Any = None,
) -> Any:
    """
    Returns the first available non-None value.
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
    Returns the first available numeric value.
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


def get_predicted_answer(prediction: dict[str, Any]) -> str:
    """
    Extracts the model's predicted answer.
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


def get_reference_answers(prediction: dict[str, Any]) -> list[str]:
    """
    Extracts all available gold/reference answers.

    Supports:
        - reference_answers
        - gold_answers
        - answers
        - reference_answer
        - gold_answer
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
        text_values = value.get("text", [])

        if isinstance(text_values, list):
            return [str(answer).strip() for answer in text_values]

        if text_values is not None:
            return [str(text_values).strip()]

    if isinstance(value, list):
        answers: list[str] = []

        for item in value:
            if isinstance(item, dict):
                text = item.get("text", "")

                if isinstance(text, list):
                    answers.extend(str(answer).strip() for answer in text)
                elif text is not None:
                    answers.append(str(text).strip())

            elif item is not None:
                answers.append(str(item).strip())

        return answers

    if value is None:
        return []

    return [str(value).strip()]


def get_is_answerable(prediction: dict[str, Any]) -> bool:
    """
    Extracts the gold answerability label.

    Supports explicit fields and falls back to whether
    reference answers contain a non-empty answer.
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
        if isinstance(explicit_value, bool):
            return explicit_value

        if isinstance(explicit_value, (int, float)):
            return bool(explicit_value)

        normalized = str(explicit_value).strip().lower()

        if normalized in {
            "true",
            "1",
            "yes",
            "answerable",
        }:
            return True

        if normalized in {
            "false",
            "0",
            "no",
            "unanswerable",
        }:
            return False

    reference_answers = get_reference_answers(prediction)

    return any(answer.strip() for answer in reference_answers)


def normalize_answer(text: str) -> str:
    """
    Standard SQuAD-style answer normalization.
    """

    text = str(text).lower()

    text = "".join(
        character for character in text if character not in string.punctuation
    )

    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text,
    )

    text = " ".join(text.split())

    return text


def exact_match_score(
    prediction: str,
    reference: str,
) -> float:
    """
    Computes normalized exact match.
    """

    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1_score(
    prediction: str,
    reference: str,
) -> float:
    """
    Computes token-level F1.
    """

    prediction_tokens = normalize_answer(prediction).split()

    reference_tokens = normalize_answer(reference).split()

    if not prediction_tokens and not reference_tokens:
        return 1.0

    if not prediction_tokens or not reference_tokens:
        return 0.0

    common_tokens = Counter(prediction_tokens) & Counter(reference_tokens)

    overlap_count = sum(common_tokens.values())

    if overlap_count == 0:
        return 0.0

    precision = overlap_count / len(prediction_tokens)

    recall = overlap_count / len(reference_tokens)

    return 2.0 * precision * recall / (precision + recall)


def calculate_answer_scores(
    predicted_answer: str,
    reference_answers: list[str],
) -> tuple[float, float]:
    """
    Returns maximum Exact Match and Token F1 scores
    across all reference answers.
    """

    if not reference_answers:
        normalized_prediction = normalize_answer(predicted_answer)

        empty_correct = float(normalized_prediction == "")

        return empty_correct, empty_correct

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

    return exact_match, token_f1


def is_prediction_correct(
    prediction: dict[str, Any],
    relaxed_f1_threshold: float,
) -> tuple[bool, float, float]:
    """
    Determines whether the QA prediction is correct.

    Correct when:
        - answerable and exact match is 1
        - answerable and Token F1 exceeds relaxed threshold
        - unanswerable and model returns an empty answer
    """

    predicted_answer = get_predicted_answer(prediction)

    reference_answers = get_reference_answers(prediction)

    is_answerable = get_is_answerable(prediction)

    exact_match, token_f1 = calculate_answer_scores(
        predicted_answer=predicted_answer,
        reference_answers=reference_answers,
    )

    if not is_answerable:
        correct = normalize_answer(predicted_answer) == ""

        return correct, exact_match, token_f1

    correct = exact_match == 1.0 or token_f1 >= relaxed_f1_threshold

    return correct, exact_match, token_f1


def confidence_only_answers(
    prediction: dict[str, Any],
    confidence_threshold: float,
) -> bool:
    """
    Confidence-only baseline decision.
    """

    calibrated_confidence = get_first_numeric_value(
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

    return calibrated_confidence >= confidence_threshold


def lexical_verifier_answers(
    prediction: dict[str, Any],
) -> bool:
    """
    Lexical verifier decision.

    Only lexical SUPPORTED predictions are answered.
    WEAK and UNSUPPORTED are rejected.
    """

    lexical_label = (
        str(
            get_first_value(
                prediction=prediction,
                field_names=(
                    "evidence_support",
                    "evidence_label",
                    "lexical_evidence_support",
                    "lexical_label",
                ),
                default="UNSUPPORTED",
            )
        )
        .strip()
        .upper()
    )

    return lexical_label == DEFAULT_LEXICAL_SUPPORTED_LABEL


def hybrid_verifier_answers(
    prediction: dict[str, Any],
) -> bool:
    """
    Hybrid verifier decision.

    Only hybrid SUPPORTED predictions are answered.
    WEAK and UNSUPPORTED are rejected.
    """

    hybrid_label = (
        str(
            get_first_value(
                prediction=prediction,
                field_names=(
                    "hybrid_evidence_support",
                    "hybrid_support",
                    "hybrid_label",
                ),
                default="UNSUPPORTED",
            )
        )
        .strip()
        .upper()
    )

    return hybrid_label == DEFAULT_HYBRID_SUPPORTED_LABEL


def evaluate_system(
    predictions: list[dict[str, Any]],
    answer_decisions: list[bool],
    relaxed_f1_threshold: float,
) -> dict[str, Any]:
    """
    Evaluates one selective QA system.
    """

    if len(predictions) != len(answer_decisions):
        raise ValueError("Prediction and decision counts must match.")

    total_count = len(predictions)

    answered_count = 0
    abstained_count = 0

    correct_answered_count = 0
    wrong_answered_count = 0

    correct_rejected_count = 0
    wrong_rejected_count = 0

    total_exact_match = 0.0
    total_token_f1 = 0.0

    answered_exact_match = 0.0
    answered_token_f1 = 0.0

    wrong_answered_examples: list[dict[str, Any]] = []

    correct_rejected_examples: list[dict[str, Any]] = []

    for index, (
        prediction,
        should_answer,
    ) in enumerate(
        zip(
            predictions,
            answer_decisions,
        ),
        start=1,
    ):
        (
            is_correct,
            exact_match,
            token_f1,
        ) = is_prediction_correct(
            prediction=prediction,
            relaxed_f1_threshold=(relaxed_f1_threshold),
        )

        total_exact_match += exact_match
        total_token_f1 += token_f1

        if should_answer:
            answered_count += 1

            answered_exact_match += exact_match

            answered_token_f1 += token_f1

            if is_correct:
                correct_answered_count += 1

            else:
                wrong_answered_count += 1

                wrong_answered_examples.append(
                    {
                        "index": index,
                        "question": prediction.get(
                            "question",
                            "",
                        ),
                        "predicted_answer": (get_predicted_answer(prediction)),
                        "reference_answers": (get_reference_answers(prediction)),
                        "exact_match": (exact_match),
                        "token_f1": token_f1,
                        "confidence": (
                            get_first_numeric_value(
                                prediction,
                                (
                                    "calibrated_confidence",
                                    "confidence",
                                ),
                            )
                        ),
                        "lexical_support": (
                            get_first_value(
                                prediction,
                                (
                                    "evidence_support",
                                    "evidence_label",
                                    "lexical_evidence_support",
                                ),
                                "",
                            )
                        ),
                        "hybrid_support": (
                            prediction.get(
                                "hybrid_evidence_support",
                                "",
                            )
                        ),
                        "hybrid_score": (
                            prediction.get(
                                "hybrid_evidence_score",
                                0.0,
                            )
                        ),
                    }
                )

        else:
            abstained_count += 1

            if is_correct:
                correct_rejected_count += 1

                correct_rejected_examples.append(
                    {
                        "index": index,
                        "question": prediction.get(
                            "question",
                            "",
                        ),
                        "predicted_answer": (get_predicted_answer(prediction)),
                        "reference_answers": (get_reference_answers(prediction)),
                        "exact_match": (exact_match),
                        "token_f1": token_f1,
                        "confidence": (
                            get_first_numeric_value(
                                prediction,
                                (
                                    "calibrated_confidence",
                                    "confidence",
                                ),
                            )
                        ),
                        "lexical_support": (
                            get_first_value(
                                prediction,
                                (
                                    "evidence_support",
                                    "evidence_label",
                                    "lexical_evidence_support",
                                ),
                                "",
                            )
                        ),
                        "hybrid_support": (
                            prediction.get(
                                "hybrid_evidence_support",
                                "",
                            )
                        ),
                        "hybrid_score": (
                            prediction.get(
                                "hybrid_evidence_score",
                                0.0,
                            )
                        ),
                    }
                )

            else:
                wrong_rejected_count += 1

    coverage = answered_count / total_count if total_count else 0.0

    selective_accuracy = (
        correct_answered_count / answered_count if answered_count else 0.0
    )

    selective_risk = wrong_answered_count / answered_count if answered_count else 0.0

    abstention_rate = abstained_count / total_count if total_count else 0.0

    overall_relaxed_accuracy = (
        (correct_answered_count + correct_rejected_count) / total_count
        if total_count
        else 0.0
    )

    average_exact_match = total_exact_match / total_count if total_count else 0.0

    average_token_f1 = total_token_f1 / total_count if total_count else 0.0

    answered_average_exact_match = (
        answered_exact_match / answered_count if answered_count else 0.0
    )

    answered_average_token_f1 = (
        answered_token_f1 / answered_count if answered_count else 0.0
    )

    return {
        "total": total_count,
        "answered": answered_count,
        "abstained": abstained_count,
        "coverage": coverage,
        "abstention_rate": abstention_rate,
        "selective_accuracy": (selective_accuracy),
        "selective_risk": selective_risk,
        "correct_answered": (correct_answered_count),
        "wrong_answered": (wrong_answered_count),
        "correct_rejected": (correct_rejected_count),
        "wrong_rejected": (wrong_rejected_count),
        "overall_relaxed_accuracy": (overall_relaxed_accuracy),
        "average_exact_match": (average_exact_match),
        "average_token_f1": (average_token_f1),
        "answered_average_exact_match": (answered_average_exact_match),
        "answered_average_token_f1": (answered_average_token_f1),
        "wrong_answered_examples": (wrong_answered_examples),
        "correct_rejected_examples": (correct_rejected_examples),
    }


def print_system_metrics(
    system_name: str,
    metrics: dict[str, Any],
) -> None:
    """
    Prints evaluation metrics for one system.
    """

    print("\n" + "=" * 60)

    print(system_name)

    print("=" * 60)

    print(f"Total:                 {metrics['total']}")

    print(f"Answered:              {metrics['answered']}")

    print(f"Abstained:             {metrics['abstained']}")

    print(f"Coverage:              {metrics['coverage']:.4f}")

    print(f"Abstention rate:       {metrics['abstention_rate']:.4f}")

    print(f"Selective accuracy:    {metrics['selective_accuracy']:.4f}")

    print(f"Selective risk:        {metrics['selective_risk']:.4f}")

    print(f"Correct answered:      {metrics['correct_answered']}")

    print(f"Wrong answered:        {metrics['wrong_answered']}")

    print(f"Correct rejected:      {metrics['correct_rejected']}")

    print(f"Wrong rejected:        {metrics['wrong_rejected']}")

    print(f"Answered average EM:   {metrics['answered_average_exact_match']:.4f}")

    print(f"Answered average F1:   {metrics['answered_average_token_f1']:.4f}")


def print_critical_examples(
    system_name: str,
    metrics: dict[str, Any],
    max_examples: int,
) -> None:
    """
    Prints wrong answered and correct rejected examples.
    """

    wrong_examples = metrics["wrong_answered_examples"]

    print("\n" + "-" * 60)

    print(f"{system_name} — WRONG ANSWERED examples")

    print("-" * 60)

    if not wrong_examples:
        print("None.")

    for example in wrong_examples[:max_examples]:
        print(f"\nIndex: {example['index']}")

        print(f"Question: {example['question']}")

        print(f"Prediction: {example['predicted_answer']}")

        print(f"Reference: {example['reference_answers']}")

        print(f"Token F1: {example['token_f1']:.4f}")

        print(f"Confidence: {example['confidence']:.4f}")

        print(f"Lexical: {example['lexical_support']}")

        print(f"Hybrid: {example['hybrid_support']}")

        print(f"Hybrid score: {float(example['hybrid_score']):.4f}")

    correct_rejected_examples = metrics["correct_rejected_examples"]

    print("\n" + "-" * 60)

    print(f"{system_name} — CORRECT REJECTED examples")

    print("-" * 60)

    if not correct_rejected_examples:
        print("None.")

    for example in correct_rejected_examples[:max_examples]:
        print(f"\nIndex: {example['index']}")

        print(f"Question: {example['question']}")

        print(f"Prediction: {example['predicted_answer']}")

        print(f"Reference: {example['reference_answers']}")

        print(f"Token F1: {example['token_f1']:.4f}")

        print(f"Confidence: {example['confidence']:.4f}")

        print(f"Lexical: {example['lexical_support']}")

        print(f"Hybrid: {example['hybrid_support']}")

        print(f"Hybrid score: {float(example['hybrid_score']):.4f}")


def run_evaluation(
    input_path: str | Path,
    confidence_threshold: float,
    relaxed_f1_threshold: float,
    max_examples: int,
) -> dict[str, dict[str, Any]]:
    """
    Runs comparative evaluation.
    """

    predictions = load_jsonl(input_path)

    if not predictions:
        raise ValueError("Prediction list cannot be empty.")

    confidence_decisions = [
        confidence_only_answers(
            prediction=prediction,
            confidence_threshold=(confidence_threshold),
        )
        for prediction in predictions
    ]

    lexical_decisions = [
        lexical_verifier_answers(prediction) for prediction in predictions
    ]

    hybrid_decisions = [
        hybrid_verifier_answers(prediction) for prediction in predictions
    ]

    confidence_metrics = evaluate_system(
        predictions=predictions,
        answer_decisions=(confidence_decisions),
        relaxed_f1_threshold=(relaxed_f1_threshold),
    )

    lexical_metrics = evaluate_system(
        predictions=predictions,
        answer_decisions=(lexical_decisions),
        relaxed_f1_threshold=(relaxed_f1_threshold),
    )

    hybrid_metrics = evaluate_system(
        predictions=predictions,
        answer_decisions=(hybrid_decisions),
        relaxed_f1_threshold=(relaxed_f1_threshold),
    )

    results = {
        "confidence_only": (confidence_metrics),
        "lexical_verifier": (lexical_metrics),
        "hybrid_verifier": (hybrid_metrics),
    }

    print("\nComparative selective QA evaluation")

    print(f"Input: {input_path}")

    print(f"Confidence threshold: {confidence_threshold:.4f}")

    print(f"Relaxed F1 threshold: {relaxed_f1_threshold:.4f}")

    print_system_metrics(
        system_name=("CONFIDENCE-ONLY BASELINE"),
        metrics=confidence_metrics,
    )

    print_system_metrics(
        system_name=("LEXICAL VERIFIER"),
        metrics=lexical_metrics,
    )

    print_system_metrics(
        system_name=("HYBRID VERIFIER"),
        metrics=hybrid_metrics,
    )

    print("\n" + "=" * 60)

    print("SUMMARY")

    print("=" * 60)

    print(
        f"{'System':<24}"
        f"{'Coverage':>10}"
        f"{'Accuracy':>12}"
        f"{'Risk':>10}"
        f"{'Wrong':>8}"
        f"{'Correct Reject':>16}"
    )

    print("-" * 80)

    for system_name, metrics in (
        (
            "Confidence-only",
            confidence_metrics,
        ),
        (
            "Lexical verifier",
            lexical_metrics,
        ),
        (
            "Hybrid verifier",
            hybrid_metrics,
        ),
    ):
        print(
            f"{system_name:<24}"
            f"{metrics['coverage']:>10.4f}"
            f"{metrics['selective_accuracy']:>12.4f}"
            f"{metrics['selective_risk']:>10.4f}"
            f"{metrics['wrong_answered']:>8}"
            f"{metrics['correct_rejected']:>16}"
        )

    print_critical_examples(
        system_name="HYBRID VERIFIER",
        metrics=hybrid_metrics,
        max_examples=max_examples,
    )

    return results


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate confidence-only, lexical, and hybrid selective QA systems."
        )
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=(DEFAULT_CONFIDENCE_THRESHOLD),
    )

    parser.add_argument(
        "--relaxed-f1-threshold",
        type=float,
        default=(DEFAULT_RELAXED_F1_THRESHOLD),
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_evaluation(
        input_path=args.input,
        confidence_threshold=(args.confidence_threshold),
        relaxed_f1_threshold=(args.relaxed_f1_threshold),
        max_examples=args.max_examples,
    )
