"""
Analyze decision-level errors in the rule-based selective-QA prototype.

This module studies the final ANSWER / VERIFY / ABSTAIN routing decisions
produced by the earlier confidence + lexical-evidence decision pipeline.

It combines two kinds of information:

1. underlying forced-answer QA candidate quality;
2. the final routing action taken by the selective system.

For answerable questions, candidate quality uses the prototype's relaxed rule:

    Exact Match == 1
        OR
    token F1 >= 0.80

For unanswerable questions, the underlying forced-answer candidate remains
incorrect. However, an ABSTAIN action on an unanswerable question is still
classified as a correct abstention at the routing level.

The diagnostic category `UNSAFE_WRONG_ANSWER` is a project-specific name for an
incorrect candidate that the system chose to answer. It should not be read as a
general safety certification or broader model-safety claim.

This file belongs to the earlier rule-based prototype analysis and should not
be confused with the project's canonical final score-ranking/AURC evaluation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from src.evaluation.analyze_evidence_errors import (
    calculate_answer_metrics,
    get_gold_answers,
    get_is_answerable,
    get_normalized_decision,
    get_predicted_answer,
)
from src.utils.io import (
    load_jsonl,
    save_json,
    save_jsonl,
)

DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/calibration_final_decisions.jsonl"
)

DEFAULT_SUMMARY_OUTPUT_PATH = Path(
    "outputs/tables/final_error_analysis.json"
)

DEFAULT_CASES_OUTPUT_PATH = Path(
    "outputs/analysis/final_error_cases.jsonl"
)


VALID_DECISIONS = {
    "ANSWER",
    "VERIFY",
    "ABSTAIN",
}


def get_final_decision(
    prediction: dict[str, Any],
) -> str:
    """
    Return the validated final routing decision.

    `final_decision` is the decision-engine output after confidence routing and
    lexical evidence resolution.
    """

    value = prediction.get(
        "final_decision"
    )

    if value is None:
        raise ValueError(
            "Prediction does not contain "
            "final_decision."
        )

    decision = (
        str(value)
        .strip()
        .upper()
    )

    if decision not in VALID_DECISIONS:
        raise ValueError(
            f"Invalid final decision: "
            f"{decision!r}."
        )

    return decision


def classify_error_case(
    final_decision: str,
    is_answerable: bool,
    prediction_correct: bool,
) -> str:
    """
    Classify one final routing outcome.

    ANSWER:
        correct candidate
            -> CORRECT_ANSWER

        incorrect candidate
            -> UNSAFE_WRONG_ANSWER

    ABSTAIN:
        unanswerable question
            -> CORRECT_UNANSWERABLE_ABSTENTION

        answerable + correct candidate
            -> UNNECESSARY_ABSTENTION

        answerable + incorrect candidate
            -> PROTECTIVE_ABSTENTION

    VERIFY:
        correct unresolved candidate
            -> VERIFY_CORRECT_PREDICTION

        incorrect unresolved candidate
            -> VERIFY_INCORRECT_PREDICTION

    Candidate correctness and routing quality remain separate concepts.
    """

    if final_decision not in VALID_DECISIONS:
        raise ValueError(
            f"Unsupported final decision: "
            f"{final_decision!r}."
        )

    if final_decision == "ANSWER":
        if prediction_correct:
            return "CORRECT_ANSWER"

        return "UNSAFE_WRONG_ANSWER"

    if final_decision == "ABSTAIN":
        if not is_answerable:
            return (
                "CORRECT_UNANSWERABLE_ABSTENTION"
            )

        if prediction_correct:
            return "UNNECESSARY_ABSTENTION"

        return "PROTECTIVE_ABSTENTION"

    if prediction_correct:
        return "VERIFY_CORRECT_PREDICTION"

    return "VERIFY_INCORRECT_PREDICTION"


def get_threshold_decision(
    prediction: dict[str, Any],
) -> str | None:
    """
    Retrieve the original confidence-threshold routing decision.

    Current artifacts can store this decision in `decision`, while earlier
    versions used more explicit field names.

    The threshold decision is diagnostic metadata only and does not determine
    underlying QA correctness.
    """

    return get_normalized_decision(
        prediction,
        (
            "threshold_decision",
            "confidence_decision",
            "selective_decision",
            "decision",
        ),
    )


def build_error_case(
    prediction: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    """
    Build one complete decision-level diagnostic record.

    `calculate_answer_metrics` determines the quality of the underlying QA
    candidate.

    `classify_error_case` then examines what the selective system did with that
    candidate: answer it, abstain, or leave it unresolved in VERIFY.
    """

    answer_metrics = (
        calculate_answer_metrics(
            prediction
        )
    )

    final_decision = (
        get_final_decision(
            prediction
        )
    )

    is_answerable = (
        get_is_answerable(
            prediction
        )
    )

    prediction_correct = bool(
        answer_metrics[
            "relaxed_correct"
        ]
    )

    category = (
        classify_error_case(
            final_decision=(
                final_decision
            ),
            is_answerable=(
                is_answerable
            ),
            prediction_correct=(
                prediction_correct
            ),
        )
    )

    return {
        "index": index,
        "id": prediction.get(
            "id"
        ),
        "category": (
            category
        ),
        "final_decision": (
            final_decision
        ),
        "is_answerable": (
            is_answerable
        ),
        "prediction_correct": (
            prediction_correct
        ),
        "strict_correct": (
            answer_metrics[
                "strict_correct"
            ]
        ),
        "exact_match": (
            answer_metrics[
                "exact_match"
            ]
        ),
        "token_f1": (
            answer_metrics[
                "token_f1"
            ]
        ),
        "answer_error_type": (
            answer_metrics[
                "error_type"
            ]
        ),
        "question": prediction.get(
            "question",
            "",
        ),
        "predicted_answer": (
            get_predicted_answer(
                prediction
            )
        ),
        "gold_answers": (
            get_gold_answers(
                prediction
            )
        ),
        "calibrated_confidence": prediction.get(
            "calibrated_confidence"
        ),
        "evidence_support": prediction.get(
            "evidence_support"
        ),
        "hybrid_evidence_support": prediction.get(
            "hybrid_evidence_support"
        ),
        "hybrid_evidence_score": prediction.get(
            "hybrid_evidence_score"
        ),
        "threshold_decision": (
            get_threshold_decision(
                prediction
            )
        ),
        "decision_reason": prediction.get(
            "decision_reason"
        ),
    }


def calculate_error_summary(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate the main decision-error categories.

    `unsafe_answer_rate` measures the fraction of final ANSWER decisions whose
    underlying QA candidate is incorrect.

    `unnecessary_abstention_rate` measures the fraction of final ABSTAIN
    decisions that rejected an answerable candidate that was already correct
    under the prototype's relaxed correctness definition.

    These rates describe the fixed rule-based operating point and are not
    risk-coverage or AURC metrics.
    """

    if not cases:
        raise ValueError(
            "Error case list cannot be empty."
        )

    total = len(
        cases
    )

    category_counts = Counter(
        case[
            "category"
        ]
        for case in cases
    )

    unsafe_wrong_answers = (
        category_counts[
            "UNSAFE_WRONG_ANSWER"
        ]
    )

    unnecessary_abstentions = (
        category_counts[
            "UNNECESSARY_ABSTENTION"
        ]
    )

    protective_abstentions = (
        category_counts[
            "PROTECTIVE_ABSTENTION"
        ]
    )

    correct_unanswerable_abstentions = (
        category_counts[
            "CORRECT_UNANSWERABLE_ABSTENTION"
        ]
    )

    verify_count = (
        category_counts[
            "VERIFY_CORRECT_PREDICTION"
        ]
        + category_counts[
            "VERIFY_INCORRECT_PREDICTION"
        ]
    )

    answered_count = sum(
        case[
            "final_decision"
        ]
        == "ANSWER"
        for case in cases
    )

    abstained_count = sum(
        case[
            "final_decision"
        ]
        == "ABSTAIN"
        for case in cases
    )

    if (
        answered_count
        + verify_count
        + abstained_count
        != total
    ):
        raise RuntimeError(
            "ANSWER, VERIFY, and ABSTAIN "
            "counts do not sum to total."
        )

    unsafe_answer_rate = (
        unsafe_wrong_answers
        / answered_count
        if answered_count
        else 0.0
    )

    unnecessary_abstention_rate = (
        unnecessary_abstentions
        / abstained_count
        if abstained_count
        else 0.0
    )

    return {
        "analysis_type": (
            "prototype_decision_error_analysis"
        ),
        "correctness_definition": (
            "forced-answer candidate quality; "
            "answerable examples use Exact Match "
            "or token F1 >= 0.80, while "
            "unanswerable candidates remain incorrect"
        ),
        "total_predictions": (
            total
        ),
        "answered_count": (
            answered_count
        ),
        "verify_count": (
            verify_count
        ),
        "abstained_count": (
            abstained_count
        ),
        "unsafe_wrong_answers": (
            unsafe_wrong_answers
        ),
        "protective_abstentions": (
            protective_abstentions
        ),
        "correct_unanswerable_abstentions": (
            correct_unanswerable_abstentions
        ),
        "unnecessary_abstentions": (
            unnecessary_abstentions
        ),
        "unsafe_answer_rate": (
            unsafe_answer_rate
        ),
        "unnecessary_abstention_rate": (
            unnecessary_abstention_rate
        ),
        "category_counts": dict(
            category_counts
        ),
    }


def save_summary(
    summary: dict[str, Any],
    output_path: str | Path,
) -> None:
    """
    Save the aggregate diagnostic summary using shared repository JSON I/O.
    """

    save_json(
        summary,
        output_path,
    )


def print_summary(
    summary: dict[str, Any],
) -> None:
    """
    Print the most important decision-level errors and routing outcomes.
    """

    print(
        "\nPrototype decision error "
        "analysis completed."
    )

    print(
        f"Total predictions: "
        f"{summary['total_predictions']}"
    )

    print(
        f"ANSWER: "
        f"{summary['answered_count']}"
    )

    print(
        f"VERIFY: "
        f"{summary['verify_count']}"
    )

    print(
        f"ABSTAIN: "
        f"{summary['abstained_count']}"
    )

    print(
        "\nImportant routing categories:"
    )

    print(
        "Wrong answered candidates: "
        f"{summary['unsafe_wrong_answers']}"
    )

    print(
        "Protective abstentions "
        "(answerable but candidate incorrect): "
        f"{summary['protective_abstentions']}"
    )

    print(
        "Correct unanswerable abstentions: "
        f"{summary[
            'correct_unanswerable_abstentions'
        ]}"
    )

    print(
        "Unnecessary abstentions "
        "(answerable candidate already correct): "
        f"{summary['unnecessary_abstentions']}"
    )

    print(
        "Wrong-answer rate among "
        "ANSWER decisions: "
        f"{summary['unsafe_answer_rate']:.4f}"
    )

    print(
        "Unnecessary-abstention rate "
        "among ABSTAIN decisions: "
        f"{summary[
            'unnecessary_abstention_rate'
        ]:.4f}"
    )

    print(
        "\nCategory distribution:"
    )

    for category, count in sorted(
        summary[
            "category_counts"
        ].items()
    ):
        print(
            f"{category}: "
            f"{count}"
        )


def run_error_analysis(
    input_path: str | Path,
    summary_output_path: str | Path,
    cases_output_path: str | Path,
) -> dict[str, Any]:
    """
    Run the complete decision-error analysis.

    The workflow loads final prototype decisions, classifies every example,
    aggregates the diagnostic categories, and saves both summary and per-case
    outputs for later qualitative inspection.
    """

    predictions = load_jsonl(
        input_path
    )

    if not predictions:
        raise ValueError(
            "Prediction file cannot be empty."
        )

    cases = [
        build_error_case(
            prediction=(
                prediction
            ),
            index=(
                index
            ),
        )
        for index, prediction
        in enumerate(
            predictions,
            start=1,
        )
    ]

    summary = (
        calculate_error_summary(
            cases
        )
    )

    save_summary(
        summary=(
            summary
        ),
        output_path=(
            summary_output_path
        ),
    )

    save_jsonl(
        cases,
        cases_output_path,
    )

    print_summary(
        summary
    )

    print(
        f"\nSummary saved to: "
        f"{summary_output_path}"
    )

    print(
        f"Cases saved to: "
        f"{cases_output_path}"
    )

    return summary


def parse_arguments() -> argparse.Namespace:
    """
    Parse input and output paths for prototype decision-error analysis.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Analyze decision-level errors "
            "in the rule-based selective-QA "
            "prototype."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=(
            DEFAULT_INPUT_PATH
        ),
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=(
            DEFAULT_SUMMARY_OUTPUT_PATH
        ),
    )

    parser.add_argument(
        "--cases-output",
        type=Path,
        default=(
            DEFAULT_CASES_OUTPUT_PATH
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run decision-error analysis from command-line arguments."""

    args = parse_arguments()

    run_error_analysis(
        input_path=(
            args.input
        ),
        summary_output_path=(
            args.summary_output
        ),
        cases_output_path=(
            args.cases_output
        ),
    )


if __name__ == "__main__":
    main()