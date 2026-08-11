"""

"""

import argparse
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.utils.io import load_jsonl, save_jsonl

DEFAULT_INPUT_PATH = Path("outputs/predictions/calibration_final_decisions.jsonl")

DEFAULT_METRICS_OUTPUT_PATH = Path("outputs/tables/evidence_impact_analysis.json")

DEFAULT_CASES_OUTPUT_PATH = Path("outputs/predictions/evidence_impact_cases.jsonl")


VALID_DECISIONS = {
    "ANSWER",
    "VERIFY",
    "ABSTAIN",
}

VALID_EVIDENCE_LABELS = {
    "SUPPORTED",
    "WEAK",
    "UNSUPPORTED",
}

# accept different True/False formats -> normalize them -> return True or False; invalid values raise an error.
def normalize_boolean(
    value: Any,
) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        if value == 1:
            return True

        if value == 0:
            return False

    if isinstance(value, float):
        if value == 1.0:
            return True

        if value == 0.0:
            return False

    if isinstance(value, str):
        normalized_value = value.strip().lower()

        true_values = {
            "true",
            "1",
            "correct",
            "yes"
        }

        false_values = {
            "false",
            "0",
            "incorrect",
            "no"
        }

        if normalized_value in true_values:
            return True

        if normalized_value in false_values:
            return False

    raise ValueError(f"Could not convert value to boolean: {value!r}")



# Normalizes answer text by converting it to lowercase, removing punctuation and articles, and cleaning extra spaces.
def normalize_answer(
    text: Any
) -> str:
    normalized_text = str(text or "").lower()

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


# read reference_answers -> handle string, dict, list, or mixed formats -> extract the actual answer text -> return list[str]
def extract_reference_answers(
    prediction: dict[str, Any],
) -> list[str]:
    reference_answers = prediction.get(
        "reference_answers",
        []
    )

    if reference_answers is None:
        return []

    if isinstance(
        reference_answers,
        str
    ):
        return [reference_answers]

    if isinstance(
        reference_answers,
        dict
    ):
        possible_fields = (
            "text",
            "answers",
            "answer_text"
        )

        for field in possible_fields:
            values = reference_answers.get(field)

            if values is None:
                continue

            if isinstance(
                values,
                str
            ):
                return [values]

            if isinstance(
                values,
                list
            ):
                extracted_answers: list[str] = []

                for value in values:
                    if isinstance(
                        value,
                        dict
                    ):
                        answer_text = (
                            value.get("text")
                            or value.get("answer")
                            or value.get("answer_text")
                        )

                        if answer_text is not None:
                            extracted_answers.append(str(answer_text))

                    else:
                        extracted_answers.append(str(value))

                return extracted_answers

        return []

    if isinstance(
        reference_answers,
        list
    ):
        extracted_answers = []

        for item in reference_answers:
            if isinstance(
                item,
                str
            ):
                extracted_answers.append(item)

            elif isinstance(
                item,
                dict
            ):
                answer_text = (
                    item.get("text") or item.get("answer") or item.get("answer_text")
                )

                if answer_text is not None:
                    extracted_answers.append(str(answer_text))

            else:
                extracted_answers.append(str(item))

        return extracted_answers

    return [str(reference_answers)]



# use is_correct / exact_match if available -> otherwise compare the predicted answer with gold answers -> handle unanswerable questions correctly.
def get_correctness(
    prediction: dict[str, Any]
) -> bool:
    possible_fields = (
        "is_correct",
        "correct",
        "prediction_correct",
        "exact_match",
        "em"
    )

    for field in possible_fields:
        if field not in prediction:
            continue

        value = prediction[field]

        if field in {
            "exact_match",
            "em"
        }:
            try:
                numeric_value = float(value)

            except (
                TypeError,
                ValueError
            ):
                return normalize_boolean(value)

            return numeric_value in {
                1.0,
                100.0
            }

        return normalize_boolean(value)

    prediction_text = normalize_answer(
        prediction.get(
            "prediction_text",
            ""
        )
    )

    reference_answers = extract_reference_answers(prediction)

    normalized_references = [
        normalize_answer(reference_answer) for reference_answer in reference_answers
    ]

    normalized_references = [
        reference_answer
        for reference_answer in normalized_references
        if reference_answer
    ]

    is_answerable_value = prediction.get(
        "is_answerable",
        True
    )

    try:
        is_answerable = normalize_boolean(is_answerable_value)

    except ValueError:
        is_answerable = bool(is_answerable_value)

    if not is_answerable:
        return prediction_text == ""

    if not normalized_references:
        raise ValueError(
            "Answerable prediction does not contain "
            "a usable reference answer. "
            f"Prediction id: {prediction.get('id')!r}"
        )

    return prediction_text in normalized_references


# check possible decision fields -> clean the value -> if it is a valid decision, return it; otherwise raise an error.
def get_threshold_decision(
    prediction: dict[str, Any]
) -> str:
    possible_fields = (
        "threshold_decision",
        "confidence_decision",
        "selective_decision"
    )

    for field in possible_fields:
        value = prediction.get(field)

        if value is None:
            continue

        decision = str(value).strip().upper()

        if decision in VALID_DECISIONS:
            return decision

    raise ValueError("Prediction does not contain a valid threshold decision.")



# get final_decision -> clean it -> check it is valid -> return it
def get_final_decision(
    prediction: dict[str, Any]
) -> str:
    value = prediction.get("final_decision")

    if value is None:
        raise ValueError("Prediction does not contain final_decision.")

    decision = str(value).strip().upper()

    if decision not in VALID_DECISIONS:
        raise ValueError(f"Invalid final decision: {decision!r}")

    return decision



# get evidence_support -> clean it -> check that it is one of the valid evidence labels -> return it.
def get_evidence_support(
    prediction: dict[str, Any]
) -> str:
    value = prediction.get("evidence_support")

    if value is None:
        raise ValueError("Prediction does not contain evidence_support.")

    evidence_support = str(value).strip().upper()

    if evidence_support not in (VALID_EVIDENCE_LABELS):
        raise ValueError(f"Invalid evidence support label: {evidence_support!r}")

    return evidence_support


# read field -> try converting to float -> if missing or invalid, return None
def get_numeric_value(
    prediction: dict[str, Any],
    field: str
) -> float | None:
    value = prediction.get(field)

    if value is None:
        return None

    try:
        return float(value)

    except (
        TypeError,
        ValueError
    ):
        return None



# if denominator = 0 -> return None; otherwise -> return the division result as float.
def safe_divide(
    numerator: float,
    denominator: float
) -> float | None:
    if denominator == 0:
        return None

    return float(numerator / denominator)



# take a group of predictions -> count correct/incorrect cases -> calculate accuracy -> compute average numeric signals like confidence and evidence scores.
def calculate_group_metrics(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    count = len(predictions)
    correct_count = sum(int(get_correctness(prediction)) for prediction in predictions)
    incorrect_count = count - correct_count

    accuracy = safe_divide(
        correct_count,
        count
    )

    numeric_fields = (
        "calibrated_confidence",
        "evidence_score",
        "answer_context_score",
        "question_evidence_overlap"
    )

    averages: dict[str, float | None] = {}

    for field in numeric_fields:
        values = [
            value
            for prediction in predictions
            if (
                value := get_numeric_value(
                    prediction,
                    field
                )
            )
            is not None
        ]

        averages[f"average_{field}"] = None if not values else sum(values) / len(values)

    return {
        "count": count,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "accuracy": accuracy,
        **averages
    }


# combine the original threshold decision and final decision into one transition name.
# Example: "ANSWER" + "ABSTAIN" → "ANSWER_TO_ABSTAIN"
def build_transition_name(
    threshold_decision: str,
    final_decision: str
) -> str:
    return f"{threshold_decision}_TO_{final_decision}"



# compare threshold decision + final decision + correctness => assign a human-readable case label.
def classify_case(
    threshold_decision: str,
    final_decision: str,
    is_correct: bool
) -> str:
    if threshold_decision == "VERIFY" and final_decision == "ANSWER" and is_correct:
        return "verify_correct_promoted_to_answer"

    if threshold_decision == "VERIFY" and final_decision == "ANSWER" and not is_correct:
        return "verify_incorrect_promoted_to_answer"

    if threshold_decision == "VERIFY" and final_decision == "ABSTAIN" and is_correct:
        return "verify_correct_blocked"

    if threshold_decision == "VERIFY" and final_decision == "ABSTAIN" and not is_correct:
        return "verify_incorrect_blocked"

    if threshold_decision == "VERIFY" and final_decision == "VERIFY" and is_correct:
        return "verify_correct_preserved"

    if threshold_decision == "VERIFY" and final_decision == "VERIFY" and not is_correct:
        return "verify_incorrect_preserved"

    return "other_transition"



# compare threshold decision vs final decision -> classify what happened to each prediction -> group similar cases -> calculate metrics → measure whether evidence verification blocks wrong answers without rejecting too many correct ones.
def analyze_predictions(
    predictions: list[dict[str, Any]]
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]]
]:
    if not predictions:
        raise ValueError(
            "Prediction list cannot be empty."
        )

    transition_groups: dict[
        str,
        list[dict[str, Any]]
    ] = defaultdict(list)

    evidence_groups: dict[
        str,
        list[dict[str, Any]]
    ] = defaultdict(list)

    diagnostic_groups: dict[
        str,
        list[dict[str, Any]]
    ] = defaultdict(list)

    analyzed_cases: list[dict[str, Any]] = []

    for index, prediction in enumerate(
        predictions,
        start=1
    ):
        try:
            threshold_decision = (
                get_threshold_decision(prediction)
            )

            final_decision = (
                get_final_decision(prediction)
            )

            evidence_support = (
                get_evidence_support(prediction)
            )

            is_correct = get_correctness(prediction)

        except ValueError as error:
            raise ValueError(
                f"Prediction {index} failed analysis: "
                f"{error}"
            ) from error

        transition = build_transition_name(
            threshold_decision,
            final_decision
        )

        diagnostic_category = classify_case(
            threshold_decision=threshold_decision,
            final_decision=final_decision,
            is_correct=is_correct
        )

        transition_groups[
            transition
        ].append(prediction)

        evidence_groups[
            evidence_support
        ].append(prediction)

        diagnostic_groups[
            diagnostic_category
        ].append(prediction)

        analyzed_case = {
            "id": prediction.get("id"),
            "question": prediction.get("question"),
            "prediction_text": prediction.get(
                "prediction_text"
            ),
            "reference_answers": (
                extract_reference_answers(prediction)
            ),
            "is_answerable": prediction.get(
                "is_answerable"
            ),
            "is_correct": is_correct,
            "calibrated_confidence": (
                get_numeric_value(
                    prediction,
                    "calibrated_confidence"
                )
            ),
            "evidence_score": (
                get_numeric_value(
                    prediction,
                    "evidence_score"
                )
            ),
            "answer_context_score": (
                get_numeric_value(
                    prediction,
                    "answer_context_score"
                )
            ),
            "question_evidence_overlap": (
                get_numeric_value(
                    prediction,
                    "question_evidence_overlap"
                )
            ),
            "evidence_support": evidence_support,
            "threshold_decision": (
                threshold_decision
            ),
            "final_decision": final_decision,
            "decision_reason": prediction.get(
                "decision_reason"
            ),
            "transition": transition,
            "diagnostic_category": (
                diagnostic_category
            ),
            "evidence_text": prediction.get(
                "evidence_text"
            )
        }

        analyzed_cases.append(analyzed_case)

    transition_metrics = {
        transition: calculate_group_metrics(
            group_predictions
        )
        for transition, group_predictions
        in sorted(transition_groups.items())
    }

    evidence_metrics = {
        evidence_label: calculate_group_metrics(
            group_predictions
        )
        for evidence_label, group_predictions
        in sorted(evidence_groups.items())
    }

    diagnostic_metrics = {
        category: calculate_group_metrics(
            group_predictions
        )
        for category, group_predictions
        in sorted(diagnostic_groups.items())
    }

    threshold_answer_predictions = [
        prediction
        for prediction in predictions
        if get_threshold_decision(prediction)
        == "ANSWER"
    ]

    final_answer_predictions = [
        prediction
        for prediction in predictions
        if get_final_decision(prediction)
        == "ANSWER"
    ]

    threshold_verify_predictions = [
        prediction
        for prediction in predictions
        if get_threshold_decision(prediction)
        == "VERIFY"
    ]

    threshold_answer_metrics = (
        calculate_group_metrics(
            threshold_answer_predictions
        )
    )

    final_answer_metrics = (
        calculate_group_metrics(
            final_answer_predictions
        )
    )

    threshold_verify_metrics = (
        calculate_group_metrics(
            threshold_verify_predictions
        )
    )

    verify_correct_promoted_count = sum(
        1
        for case in analyzed_cases
        if case["diagnostic_category"]
        == "verify_correct_promoted_to_answer"
    )

    verify_incorrect_promoted_count = sum(
        1
        for case in analyzed_cases
        if case["diagnostic_category"]
        == "verify_incorrect_promoted_to_answer"
    )

    verify_correct_blocked_count = sum(
        1
        for case in analyzed_cases
        if case["diagnostic_category"]
        == "verify_correct_blocked"
    )

    verify_incorrect_blocked_count = sum(
        1
        for case in analyzed_cases
        if case["diagnostic_category"]
        == "verify_incorrect_blocked"
    )

    verify_correct_preserved_count = sum(
        1
        for case in analyzed_cases
        if case["diagnostic_category"]
        == "verify_correct_preserved"
    )

    verify_incorrect_preserved_count = sum(
        1
        for case in analyzed_cases
        if case["diagnostic_category"]
        == "verify_incorrect_preserved"
    )

    summary = {
        "total_predictions": len(predictions),

        "threshold_answer_policy": (
            threshold_answer_metrics
        ),

        "threshold_verify_policy": (
            threshold_verify_metrics
        ),

        "final_answer_policy": (
            final_answer_metrics
        ),

        "evidence_label_metrics": (
            evidence_metrics
        ),

        "transition_metrics": (
            transition_metrics
        ),

        "diagnostic_metrics": (
            diagnostic_metrics
        ),

        "evidence_impact_summary": {
            "verify_predictions": len(
                threshold_verify_predictions
            ),
            "verify_correct_promoted_to_answer": (
                verify_correct_promoted_count
            ),
            "verify_incorrect_promoted_to_answer": (
                verify_incorrect_promoted_count
            ),
            "verify_correct_blocked": (
                verify_correct_blocked_count
            ),
            "verify_incorrect_blocked": (
                verify_incorrect_blocked_count
            ),
            "verify_correct_preserved": (
                verify_correct_preserved_count
            ),
            "verify_incorrect_preserved": (
                verify_incorrect_preserved_count
            )
        },

        "diagnostic_category_counts": dict(
            Counter(
                case["diagnostic_category"]
                for case in analyzed_cases
            )
        )
    }

    return (
        summary,
        analyzed_cases
    )

# Saves a dictionary as a formatted JSON file, creating the output directory first if necessary.
def save_json(
    data: dict[str, Any],
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
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )



# None -> "N/A"; number -> formatted like 0.8472.
def format_metric(
    value: float | None
) -> str:
    if value is None:
        return "N/A"

    return f"{value:.4f}"



# take the analysis results -> compare confidence-only vs evidence-aware policy -> show how many wrong answers were blocked/verified, how many correct answers were downgraded, and print evidence/transition metrics.
def print_summary(
    analysis: dict[str, Any]
) -> None:
    impact = analysis[
        "evidence_impact_summary"
    ]

    threshold_policy = analysis[
        "threshold_answer_policy"
    ]

    threshold_verify_policy = analysis[
        "threshold_verify_policy"
    ]

    final_policy = analysis[
        "final_answer_policy"
    ]

    print(
        "\nEvidence impact analysis completed."
    )

    print(
        f"Total predictions: "
        f"{analysis['total_predictions']}"
    )

    print(
        "\nConfidence-only ANSWER policy:"
    )

    print(
        f"Answer count: "
        f"{threshold_policy['count']}"
    )

    print(
        "Answer accuracy: "
        f"{format_metric(
            threshold_policy['accuracy']
        )}"
    )

    print(
        "\nThreshold VERIFY region:"
    )

    print(
        f"VERIFY count: "
        f"{threshold_verify_policy['count']}"
    )

    print(
        "VERIFY-region accuracy: "
        f"{format_metric(
            threshold_verify_policy['accuracy']
        )}"
    )

    print(
        "\nEvidence-aware ANSWER policy:"
    )

    print(
        f"Answer count: "
        f"{final_policy['count']}"
    )

    print(
        "Answer accuracy: "
        f"{format_metric(
            final_policy['accuracy']
        )}"
    )

    print(
        "\nEvidence impact on VERIFY predictions:"
    )

    print(
        f"VERIFY predictions: "
        f"{impact['verify_predictions']}"
    )

    print(
        "Correct VERIFY promoted to ANSWER: "
        f"{impact[
            'verify_correct_promoted_to_answer'
        ]}"
    )

    print(
        "Incorrect VERIFY promoted to ANSWER: "
        f"{impact[
            'verify_incorrect_promoted_to_answer'
        ]}"
    )

    print(
        "Correct VERIFY blocked: "
        f"{impact[
            'verify_correct_blocked'
        ]}"
    )

    print(
        "Incorrect VERIFY blocked: "
        f"{impact[
            'verify_incorrect_blocked'
        ]}"
    )

    print(
        "Correct VERIFY preserved: "
        f"{impact[
            'verify_correct_preserved'
        ]}"
    )

    print(
        "Incorrect VERIFY preserved: "
        f"{impact[
            'verify_incorrect_preserved'
        ]}"
    )

    print(
        "\nEvidence label accuracy:"
    )

    evidence_metrics = analysis[
        "evidence_label_metrics"
    ]

    for evidence_label in (
        "SUPPORTED",
        "WEAK",
        "UNSUPPORTED"
    ):
        metrics = evidence_metrics.get(
            evidence_label
        )

        if metrics is None:
            continue

        print(
            f"{evidence_label}: "
            f"count={metrics['count']} | "
            f"accuracy="
            f"{format_metric(
                metrics['accuracy']
            )}"
        )

    print(
        "\nDecision transitions:"
    )

    for transition, metrics in (
        analysis["transition_metrics"].items()
    ):
        print(
            f"{transition}: "
            f"count={metrics['count']} | "
            f"accuracy="
            f"{format_metric(
                metrics['accuracy']
            )}"
        )



# load predictions -> analyze evidence impact -> save summary + case details -> print results -> return analysis.
def run_analysis(
    input_path: str | Path,
    metrics_output_path: str | Path,
    cases_output_path: str | Path
) -> dict[str, Any]:
    predictions = load_jsonl(input_path)

    analysis, analyzed_cases = analyze_predictions(predictions)

    save_json(
        data=analysis,
        output_path=metrics_output_path
    )

    save_jsonl(
        analyzed_cases,
        cases_output_path
    )

    print_summary(analysis)

    print(f"\nMetrics saved to: {metrics_output_path}")

    print(f"Diagnostic cases saved to: {cases_output_path}")

    return analysis


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze how evidence verification changes selective QA decisions."
        )
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help=("JSONL file containing final decision engine predictions.")
    )

    parser.add_argument(
        "--metrics-output",
        default=str(DEFAULT_METRICS_OUTPUT_PATH),
        help=("JSON output path for evidence impact metrics.")
    )

    parser.add_argument(
        "--cases-output",
        default=str(DEFAULT_CASES_OUTPUT_PATH),
        help=("JSONL output path for diagnostic prediction cases.")
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_analysis(
        input_path=args.input,
        metrics_output_path=(args.metrics_output),
        cases_output_path=(args.cases_output),
    )
