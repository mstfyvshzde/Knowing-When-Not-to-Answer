"""
Core evaluation metrics for selective question answering.

This module evaluates systems that make one of two final actions:

- ANSWER
- ABSTAIN

A prediction is considered task-correct when:

- the system ANSWERs an answerable question with an exact-match answer, or
- the system ABSTAINs on an unanswerable question.

The module reports:

- task accuracy
- coverage
- abstention rate
- answered accuracy
- selective risk
- extractive Exact Match
- token-level F1
- unnecessary abstention rate
- unanswerable answer rate
- correct abstention rate

Important
---------
The `accuracy` metric includes correct abstentions as successful selective-QA
decisions.

The `exact_match` and `token_f1` metrics do not award lexical credit for a
correct abstention. They therefore describe answer-text quality under this
project's evaluation convention and should not be presented as official
SQuAD v2 Exact Match/F1 scores.

When the system answers no examples, answered accuracy and selective risk are
reported as 0.0 by repository convention.
"""

import argparse
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.io import load_jsonl, save_jsonl

VALID_DECISIONS = {
    "ANSWER",
    "ABSTAIN",
}


def normalize_answer(text: str) -> str:
    """
    Normalize answer text before lexical comparison.

    The normalization follows the project's SQuAD-style comparison rule:

    1. lowercase text,
    2. remove punctuation,
    3. remove the English articles a/an/the,
    4. collapse repeated whitespace.
    """

    if not isinstance(text, str):
        text = str(text)

    text = text.lower()

    text = "".join(
        character
        for character in text
        if character not in string.punctuation
    )

    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text,
    )

    return " ".join(
        text.split()
    )


def exact_match_score(
    prediction: str,
    references: list[str],
) -> float:
    """
    Return 1.0 when the prediction exactly matches any reference answer.

    Comparison is performed after answer normalization.

    If no reference answer is available, the score is 0.0.
    """

    if not references:
        return 0.0

    normalized_prediction = (
        normalize_answer(
            prediction
        )
    )

    return float(
        any(
            normalized_prediction
            == normalize_answer(reference)
            for reference in references
        )
    )


def token_f1_score(
    prediction: str,
    references: list[str],
) -> float:
    """
    Calculate token-level F1 against the best matching reference answer.

    Token overlap is counted with multiplicity using Counter intersection.
    When multiple reference answers are available, the maximum F1 is returned.
    """

    if not references:
        return 0.0

    prediction_tokens = (
        normalize_answer(
            prediction
        ).split()
    )

    if not prediction_tokens:
        return 0.0

    best_f1 = 0.0

    for reference in references:
        reference_tokens = (
            normalize_answer(
                reference
            ).split()
        )

        if not reference_tokens:
            continue

        common_tokens = (
            Counter(prediction_tokens)
            & Counter(reference_tokens)
        )

        overlap = sum(
            common_tokens.values()
        )

        if overlap == 0:
            continue

        precision = (
            overlap
            / len(prediction_tokens)
        )

        recall = (
            overlap
            / len(reference_tokens)
        )

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

        best_f1 = max(
            best_f1,
            f1,
        )

    return best_f1


def parse_answerability(
    value: Any,
) -> bool:
    """
    Convert an answerability field into a strict Boolean value.

    JSON Boolean values are preferred. Integer 0/1 and common textual Boolean
    representations are accepted for compatibility with historical artifacts.

    Ambiguous values are rejected instead of relying on Python's generic
    truthiness, where for example the string "False" would otherwise be True.
    """

    if isinstance(value, bool):
        return value

    if (
        isinstance(value, int)
        and value in {0, 1}
    ):
        return bool(value)

    if isinstance(value, str):
        normalized = (
            value.strip().lower()
        )

        if normalized in {
            "true",
            "1",
            "yes",
        }:
            return True

        if normalized in {
            "false",
            "0",
            "no",
        }:
            return False

    raise TypeError(
        "is_answerable must be a Boolean "
        f"or an equivalent 0/1 value, received {value!r}."
    )


def get_reference_answers(
    prediction: dict[str, Any],
) -> list[str]:
    """
    Retrieve reference answers in the format required by text metrics.

    Missing or explicitly null reference collections become an empty list.
    """

    references = prediction.get(
        "reference_answers",
        [],
    )

    if references is None:
        references = []

    if not isinstance(
        references,
        list,
    ):
        raise TypeError(
            "'reference_answers' must be a list of strings."
        )

    return [
        str(reference)
        for reference in references
    ]


def evaluate_single_prediction(
    prediction: dict[str, Any],
) -> dict[str, Any]:
    """
    Evaluate one selective-QA prediction.

    ANSWER
    ------
    For an answerable question, correctness requires normalized Exact Match
    against at least one reference answer.

    Answering an unanswerable question is always incorrect.

    ABSTAIN
    -------
    Abstaining is correct only when the question is unanswerable.

    Exact Match and token F1 are set to zero for abstentions because these
    fields measure answer-text quality rather than abstention correctness.
    """

    required_fields = {
        "decision",
        "is_answerable",
    }

    missing_fields = (
        required_fields
        - prediction.keys()
    )

    if missing_fields:
        missing_text = ", ".join(
            sorted(missing_fields)
        )

        raise KeyError(
            "Prediction is missing required "
            f"fields: {missing_text}"
        )

    decision = (
        str(
            prediction["decision"]
        )
        .strip()
        .upper()
    )

    if decision not in VALID_DECISIONS:
        raise ValueError(
            f"Unknown decision: {decision!r}. "
            "Expected 'ANSWER' or 'ABSTAIN'."
        )

    is_answerable = (
        parse_answerability(
            prediction[
                "is_answerable"
            ]
        )
    )

    predicted_answer = str(
        prediction.get(
            "prediction_text",
            "",
        )
    )

    references = (
        get_reference_answers(
            prediction
        )
    )

    if decision == "ANSWER":
        if not is_answerable:
            exact_match = 0.0
            token_f1 = 0.0
            is_correct = False

        else:
            exact_match = (
                exact_match_score(
                    predicted_answer,
                    references,
                )
            )

            token_f1 = (
                token_f1_score(
                    predicted_answer,
                    references,
                )
            )

            # Task correctness uses exact answer correctness rather than
            # partial token overlap.
            is_correct = (
                exact_match == 1.0
            )

    else:
        exact_match = 0.0
        token_f1 = 0.0
        is_correct = (
            not is_answerable
        )

    evaluated = (
        prediction.copy()
    )

    evaluated.update(
        {
            "decision": decision,
            "is_answerable": (
                is_answerable
            ),
            "exact_match": (
                exact_match
            ),
            "token_f1": (
                token_f1
            ),
            "is_correct": (
                is_correct
            ),
        }
    )

    return evaluated


def evaluate_predictions(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Evaluate every prediction once and return enriched records.
    """

    if not predictions:
        raise ValueError(
            "Prediction list cannot be empty."
        )

    return [
        evaluate_single_prediction(
            prediction
        )
        for prediction in predictions
    ]



def calculate_metrics_from_evaluated(
    evaluated_predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate metrics from already evaluated prediction records.

    `accuracy` is selective-QA task accuracy and therefore includes correct
    abstentions.

    `answered_accuracy` measures correctness only among predictions the system
    chose to answer.

    `selective_risk` is:

        1 - answered_accuracy

    when at least one example is answered.
    """

    if not evaluated_predictions:
        raise ValueError(
            "Prediction list cannot be empty."
        )

    total = len(
        evaluated_predictions
    )

    answered_predictions = [
        prediction
        for prediction
        in evaluated_predictions
        if (
            prediction["decision"]
            == "ANSWER"
        )
    ]

    abstained_predictions = [
        prediction
        for prediction
        in evaluated_predictions
        if (
            prediction["decision"]
            == "ABSTAIN"
        )
    ]

    answered = len(
        answered_predictions
    )

    abstained = len(
        abstained_predictions
    )

    total_correct = sum(
        int(
            prediction[
                "is_correct"
            ]
        )
        for prediction
        in evaluated_predictions
    )

    answered_correct = sum(
        int(
            prediction[
                "is_correct"
            ]
        )
        for prediction
        in answered_predictions
    )

    answerable_count = sum(
        int(
            prediction[
                "is_answerable"
            ]
        )
        for prediction
        in evaluated_predictions
    )

    unanswerable_count = (
        total
        - answerable_count
    )

    unnecessary_abstentions = sum(
        int(
            prediction["decision"]
            == "ABSTAIN"
            and prediction[
                "is_answerable"
            ]
        )
        for prediction
        in evaluated_predictions
    )

    answered_unanswerable = sum(
        int(
            prediction["decision"]
            == "ANSWER"
            and not prediction[
                "is_answerable"
            ]
        )
        for prediction
        in evaluated_predictions
    )

    correct_abstentions = sum(
        int(
            prediction["decision"]
            == "ABSTAIN"
            and not prediction[
                "is_answerable"
            ]
        )
        for prediction
        in evaluated_predictions
    )

    exact_match_total = sum(
        float(
            prediction[
                "exact_match"
            ]
        )
        for prediction
        in evaluated_predictions
    )

    token_f1_total = sum(
        float(
            prediction[
                "token_f1"
            ]
        )
        for prediction
        in evaluated_predictions
    )

    answered_exact_match_total = sum(
        float(
            prediction[
                "exact_match"
            ]
        )
        for prediction
        in answered_predictions
    )

    answered_token_f1_total = sum(
        float(
            prediction[
                "token_f1"
            ]
        )
        for prediction
        in answered_predictions
    )

    accuracy = (
        total_correct
        / total
    )

    coverage = (
        answered
        / total
    )

    abstention_rate = (
        abstained
        / total
    )

    if answered > 0:
        answered_accuracy = (
            answered_correct
            / answered
        )

        selective_risk = (
            1.0
            - answered_accuracy
        )

        answered_exact_match = (
            answered_exact_match_total
            / answered
        )

        answered_token_f1 = (
            answered_token_f1_total
            / answered
        )

    else:
        # Repository convention for an empty answered set.
        answered_accuracy = 0.0
        selective_risk = 0.0
        answered_exact_match = 0.0
        answered_token_f1 = 0.0

    # Correct abstentions intentionally receive no lexical EM/F1 credit.
    exact_match = (
        exact_match_total
        / total
    )

    token_f1 = (
        token_f1_total
        / total
    )

    unnecessary_abstention_rate = (
        unnecessary_abstentions
        / answerable_count
        if answerable_count > 0
        else 0.0
    )

    unanswerable_answer_rate = (
        answered_unanswerable
        / unanswerable_count
        if unanswerable_count > 0
        else 0.0
    )

    correct_abstention_rate = (
        correct_abstentions
        / unanswerable_count
        if unanswerable_count > 0
        else 0.0
    )

    return {
        "total": total,
        "answered": answered,
        "abstained": abstained,
        "answerable_count": (
            answerable_count
        ),
        "unanswerable_count": (
            unanswerable_count
        ),
        "total_correct": (
            total_correct
        ),
        "answered_correct": (
            answered_correct
        ),
        "correct_abstentions": (
            correct_abstentions
        ),
        "unnecessary_abstentions": (
            unnecessary_abstentions
        ),
        "answered_unanswerable": (
            answered_unanswerable
        ),
        "accuracy": accuracy,
        "coverage": coverage,
        "abstention_rate": (
            abstention_rate
        ),
        "answered_accuracy": (
            answered_accuracy
        ),
        "selective_risk": (
            selective_risk
        ),
        "exact_match": (
            exact_match
        ),
        "token_f1": (
            token_f1
        ),
        "answered_exact_match": (
            answered_exact_match
        ),
        "answered_token_f1": (
            answered_token_f1
        ),
        "unnecessary_abstention_rate": (
            unnecessary_abstention_rate
        ),
        "unanswerable_answer_rate": (
            unanswerable_answer_rate
        ),
        "correct_abstention_rate": (
            correct_abstention_rate
        ),
    }


def calculate_metrics(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Evaluate predictions and calculate aggregate selective-QA metrics.

    This public helper accepts raw prediction records. Each record is evaluated
    once before metric aggregation.
    """

    evaluated_predictions = (
        evaluate_predictions(
            predictions
        )
    )

    return (
        calculate_metrics_from_evaluated(
            evaluated_predictions
        )
    )


def print_metrics(
    metrics: dict[str, Any],
) -> None:
    """Print evaluation metrics in a human-readable format."""

    print(
        "\nEvaluation Results"
    )

    print(
        "=" * 40
    )

    for name, value in metrics.items():
        readable_name = (
            name
            .replace("_", " ")
            .title()
        )

        if isinstance(
            value,
            float,
        ):
            print(
                f"{readable_name}: "
                f"{value:.4f}"
            )

        else:
            print(
                f"{readable_name}: "
                f"{value}"
            )


def parse_arguments() -> argparse.Namespace:
    """Parse selective-QA evaluation input and optional output paths."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ANSWER/ABSTAIN predictions "
            "for selective question answering."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Input predictions JSONL file."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional JSONL path for "
            "evaluated predictions."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run selective-QA evaluation from the command line."""

    args = parse_arguments()

    predictions = load_jsonl(
        args.input
    )

    evaluated_predictions = (
        evaluate_predictions(
            predictions
        )
    )

    metrics = (
        calculate_metrics_from_evaluated(
            evaluated_predictions
        )
    )

    if args.output is not None:
        save_jsonl(
            evaluated_predictions,
            args.output,
        )

        print(
            "Evaluated predictions saved to: "
            f"{args.output}"
        )

    print_metrics(
        metrics
    )


if __name__ == "__main__":
    main()