"""
Analyze the final selective-QA system's decision errors.

This module classifies final ANSWER / VERIFY / ABSTAIN decisions into
interpretable diagnostic categories such as unsafe wrong answers,
protective abstentions, unnecessary abstentions, and unresolved VERIFY cases.
"""


import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.evaluation.analyze_evidence_errors import (
    calculate_answer_metrics,
    get_gold_answers,
    get_predicted_answer,
    parse_boolean,
)
from src.utils.io import load_jsonl, save_jsonl

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


# get final_decision -> clean it -> check ANSWER / VERIFY / ABSTAIN -> return it; otherwise raise an error.
def get_final_decision(
    prediction: dict[str, Any]
) -> str:
    value = prediction.get("final_decision")

    if value is None:
        raise ValueError(
            "Prediction does not contain final_decision."
        )

    decision = str(value).strip().upper()

    if decision not in VALID_DECISIONS:
        raise ValueError(
            f"Invalid final decision: {decision!r}"
        )

    return decision


# Classifies each final ANSWER, ABSTAIN, or VERIFY decision based on whether the question was answerable and whether the prediction was correct.
def classify_error_case(
    final_decision: str,
    is_answerable: bool,
    prediction_correct: bool
) -> str:
    if final_decision == "ANSWER":
        if prediction_correct:
            return "CORRECT_ANSWER"

        return "UNSAFE_WRONG_ANSWER"

    if final_decision == "ABSTAIN":
        if not is_answerable:
            return "CORRECT_UNANSWERABLE_ABSTENTION"

        if prediction_correct:
            return "UNNECESSARY_ABSTENTION"

        return "PROTECTIVE_ABSTENTION"

    if final_decision == "VERIFY":
        if prediction_correct:
            return "VERIFY_CORRECT_PREDICTION"

        return "VERIFY_INCORRECT_PREDICTION"

    raise ValueError(
        f"Unsupported final decision: {final_decision}"
    )


# take one prediction -> calculate whether it was correct -> classify the final decision -> collect all important QA, evidence, confidence, and decision fields into one error-analysis case. ✅
def build_error_case(
    prediction: dict[str, Any],
    index: int
) -> dict[str, Any]:
    answer_metrics = calculate_answer_metrics(
        prediction
    )

    final_decision = get_final_decision(
        prediction
    )

    is_answerable = parse_boolean(
        prediction.get(
            "is_answerable",
            True
        )
    )

    prediction_correct = bool(
        answer_metrics["relaxed_correct"]
    )

    category = classify_error_case(
        final_decision=final_decision,
        is_answerable=is_answerable,
        prediction_correct=prediction_correct
    )

    return {
        "index": index,
        "id": prediction.get("id"),
        "category": category,
        "final_decision": final_decision,
        "is_answerable": is_answerable,
        "prediction_correct": prediction_correct,
        "strict_correct": answer_metrics[
            "strict_correct"
        ],
        "exact_match": answer_metrics[
            "exact_match"
        ],
        "token_f1": answer_metrics[
            "token_f1"
        ],
        "answer_error_type": answer_metrics[
            "error_type"
        ],
        "question": prediction.get(
            "question",
            "",
        ),
        "predicted_answer": (
            get_predicted_answer(prediction)
        ),
        "gold_answers": (
            get_gold_answers(prediction)
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
        "threshold_decision": prediction.get(
            "threshold_decision"
        ),
        "decision_reason": prediction.get(
            "decision_reason"
        )
    }



# take all classified cases -> count ANSWER / VERIFY / ABSTAIN outcomes -> measure dangerous wrong answers, helpful abstentions, unnecessary abstentions, and their rates.
def calculate_error_summary(
    cases: list[dict[str, Any]]
) -> dict[str, Any]:
    if not cases:
        raise ValueError(
            "Error case list cannot be empty."
        )

    total = len(cases)

    category_counts = Counter(
        case["category"]
        for case in cases
    )

    unsafe_wrong_answers = category_counts[
        "UNSAFE_WRONG_ANSWER"
    ]

    unnecessary_abstentions = category_counts[
        "UNNECESSARY_ABSTENTION"
    ]

    protective_abstentions = category_counts[
        "PROTECTIVE_ABSTENTION"
    ]

    verify_count = (
        category_counts[
            "VERIFY_CORRECT_PREDICTION"
        ]
        + category_counts[
            "VERIFY_INCORRECT_PREDICTION"
        ]
    )

    answered_count = sum(
        case["final_decision"] == "ANSWER"
        for case in cases
    )

    abstained_count = sum(
        case["final_decision"] == "ABSTAIN"
        for case in cases
    )

    unsafe_answer_rate = (
        unsafe_wrong_answers / answered_count
        if answered_count
        else 0.0
    )

    unnecessary_abstention_rate = (
        unnecessary_abstentions / abstained_count
        if abstained_count
        else 0.0
    )

    return {
        "total_predictions": total,
        "answered_count": answered_count,
        "verify_count": verify_count,
        "abstained_count": abstained_count,
        "unsafe_wrong_answers": (
            unsafe_wrong_answers
        ),
        "protective_abstentions": (
            protective_abstentions
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
        )
    }


# summary dictionary -> create folder if needed -> save as readable JSON.
def save_summary(
    summary: dict[str, Any],
    output_path: str | Path
) -> None:
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as output_file:
        json.dump(
            summary,
            output_file,
            indent=2,
            ensure_ascii=False
        )



# take the error summary -> display ANSWER / VERIFY / ABSTAIN counts -> show unsafe answers, protective/unnecessary abstentions, their rates, and all category counts.
def print_summary(
    summary: dict[str, Any]
) -> None:
    print("\nFinal system error analysis completed.")

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

    print("\nImportant error categories:")

    print(
        "Unsafe wrong answers: "
        f"{summary['unsafe_wrong_answers']}"
    )

    print(
        "Protective abstentions: "
        f"{summary['protective_abstentions']}"
    )

    print(
        "Unnecessary abstentions: "
        f"{summary['unnecessary_abstentions']}"
    )

    print(
        "Unsafe answer rate: "
        f"{summary['unsafe_answer_rate']:.4f}"
    )

    print(
        "Unnecessary abstention rate: "
        f"{summary[
            'unnecessary_abstention_rate'
        ]:.4f}"
    )

    print("\nCategory distribution:")

    for category, count in sorted(
        summary["category_counts"].items()
    ):
        print(
            f"{category}: {count}"
        )


# load predictions -> classify every case -> calculate error summary -> save summary + detailed cases -> print results -> return summary.
def run_error_analysis(
    input_path: str | Path,
    summary_output_path: str | Path,
    cases_output_path: str | Path
) -> dict[str, Any]:
    predictions = load_jsonl(
        input_path
    )

    if not predictions:
        raise ValueError(
            "Prediction file cannot be empty."
        )

    cases = [
        build_error_case(
            prediction=prediction,
            index=index
        )
        for index, prediction
        in enumerate(
            predictions,
            start=1
        )
    ]

    summary = calculate_error_summary(
        cases
    )

    save_summary(
        summary=summary,
        output_path=summary_output_path
    )

    save_jsonl(
        cases,
        cases_output_path
    )

    print_summary(summary)

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
    parser = argparse.ArgumentParser(
        description=(
            "Analyze errors made by the final "
            "selective-QA decision policy."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT_PATH
    )

    parser.add_argument(
        "--cases-output",
        type=Path,
        default=DEFAULT_CASES_OUTPUT_PATH
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    run_error_analysis(
        input_path=args.input,
        summary_output_path=(
            args.summary_output
        ),
        cases_output_path=(
            args.cases_output
        )
    )


if __name__ == "__main__":
    main()