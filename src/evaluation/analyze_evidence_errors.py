"""
check whether each QA answer is actually correct -> compare that with the verifier’s SUPPORTED / WEAK / UNSUPPORTED label -> find verifier mistakes and especially dangerous WRONG_SUPPORTED cases.
"""
import argparse
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.io import load_jsonl, save_jsonl

DEFAULT_INPUT_PATH = Path("outputs/predictions/calibration_with_evidence.jsonl")

DEFAULT_OUTPUT_PATH = Path("outputs/analysis/evidence_error_cases.jsonl")

NO_ANSWER_VALUES = {
    "",
    "no answer",
    "unanswerable",
    "none",
    "null",
    "cls",
    "n a",
}


# check predicted_answer -> if missing, try prediction_text -> then prediction_answer -> then answer -> otherwise return an empty string.
def get_predicted_answer(prediction: dict[str, Any]) -> str:
    value = prediction.get(
        "predicted_answer",
        prediction.get(
            "prediction_text",
            prediction.get("prediction_answer", prediction.get("answer", ""))
        )
    )

    if value is None:
        return ""

    return str(value)



# handle None, string, number, list, or nested dictionary -> keep digging until actual answer text is found -> return all extracted answers.
def extract_answer_texts(value: Any) -> list[str]:
    answers: list[str] = []

    if value is None:
        return answers

    if isinstance(value, str):
        cleaned = value.strip()

        if cleaned:
            answers.append(cleaned)

        return answers

    if isinstance(value, (int, float, bool)):
        return [str(value)]

    if isinstance(value, list):
        for item in value:
            answers.extend(extract_answer_texts(item))

        return answers

    if isinstance(value, dict):
        answer_keys = (
            "text",
            "answer",
            "answers",
            "value",
            "values",
            "reference_answer",
            "reference_answers"
        )

        matching_key_found = False

        for key in answer_keys:
            if key not in value:
                continue

            matching_key_found = True

            answers.extend(extract_answer_texts(value[key]))

        if matching_key_found:
            return answers

        for nested_value in value.values():
            answers.extend(extract_answer_texts(nested_value))

        return answers

    return answers



# check possible gold-answer fields -> extract answer texts -> remove empty values -> remove duplicates -> return the final gold-answer list.
def get_gold_answers(prediction: dict[str, Any]) -> list[str]:
    possible_fields = (
        "gold_answers",
        "gold_answer",
        "reference_answers",
        "reference_answer",
        "answers"
    )

    for field in possible_fields:
        if field not in prediction:
            continue

        answers = extract_answer_texts(prediction[field])

        return list(dict.fromkeys(answer for answer in answers if answer.strip()))

    return []



# Normalizes an answer by lowercasing it, removing punctuation and articles, and collapsing extra whitespace.
def normalize_answer(text: str) -> str:
    normalized = str(text).lower()

    # removes punctuation
    normalized = normalized.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    normalized = re.sub(
        r"\b(a|an|the)\b",
        " ",
        normalized
    )

    # removes extra spaces
    normalized = re.sub(
        r"\s+",
        " ",
        normalized
    ).strip()

    return normalized



# True, 1, "yes", "answerable" -> True; False, 0, "no", "unanswerable" -> False; unknown value -> use default.
def parse_boolean(value: Any, default: bool = True) -> bool:
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        normalized = value.lower().strip()

        if normalized in {
            "true",
            "1",
            "yes",
            "answerable"
        }:
            return True

        if normalized in {
            "false",
            "0",
            "no",
            "unanswerable"
        }:
            return False

    return default


# normalize both answers -> compare them -> same = 1.0, different = 0.0.
def calculate_exact_match(predicted_answer: str, gold_answer: str) -> float:
    return float(normalize_answer(predicted_answer) == normalize_answer(gold_answer))



# normalize both answers -> split into tokens -> count shared words -> calculate precision and recall -> combine them into F1
def calculate_token_f1(predicted_answer: str, gold_answer: str) -> float:
    predicted_tokens = normalize_answer(predicted_answer).split()

    gold_tokens = normalize_answer(gold_answer).split()

    if not predicted_tokens and not gold_tokens:
        return 1.0

    if not predicted_tokens or not gold_tokens:
        return 0.0

    common_tokens = Counter(predicted_tokens) & Counter(gold_tokens)

    shared_count = sum(common_tokens.values())

    if shared_count == 0:
        return 0.0

    precision = shared_count / len(predicted_tokens)
    recall = shared_count / len(gold_tokens)

    f1_score = 2 * precision * recall / (precision + recall)

    return f1_score



# evaluate one prediction -> handle answerable vs unanswerable cases -> compare with all gold answers -> calculate EM/F1 -> classify the result as exact, minor mismatch, partial, or wrong.
def calculate_answer_metrics(prediction: dict[str, Any]) -> dict[str, Any]:
    predicted_answer = get_predicted_answer(prediction)
    normalized_prediction = normalize_answer(predicted_answer)
    gold_answers = get_gold_answers(prediction)

    is_answerable = parse_boolean(
        prediction.get(
            "is_answerable",
            True
        )
    )

    if not is_answerable:
        predicted_no_answer = normalized_prediction in NO_ANSWER_VALUES

        return {
            "exact_match": float(predicted_no_answer),
            "token_f1": float(predicted_no_answer),
            "strict_correct": (predicted_no_answer),
            "relaxed_correct": (predicted_no_answer),
            "error_type": (
                "CORRECT_NO_ANSWER"
                if predicted_no_answer
                else "UNANSWERABLE_MODEL_ANSWERED"
            )
        }

    if not gold_answers:
        return {
            "exact_match": 0.0,
            "token_f1": 0.0,
            "strict_correct": False,
            "relaxed_correct": False,
            "error_type": ("ANSWERABLE_WITHOUT_GOLD")
        }

    exact_match = max(
        calculate_exact_match(
            predicted_answer,
            gold_answer
        )
        for gold_answer in gold_answers
    )

    token_f1 = max(
        calculate_token_f1(
            predicted_answer,
            gold_answer
        )
        for gold_answer in gold_answers
    )

    strict_correct = exact_match == 1.0

    # Yüksek token örtüşmesine sahip span farklarını gerçek QA hatasından ayırıyoruz.
    relaxed_correct = strict_correct or token_f1 >= 0.80

    if strict_correct:
        error_type = "CORRECT_EXACT"

    elif token_f1 >= 0.80:
        error_type = "MINOR_SPAN_MISMATCH"

    elif token_f1 >= 0.50:
        error_type = "PARTIAL_ANSWER"

    else:
        error_type = "WRONG_ANSWER"

    return {
        "exact_match": exact_match,
        "token_f1": token_f1,
        "strict_correct": strict_correct,
        "relaxed_correct": relaxed_correct,
        "error_type": error_type
    }


# check whether the answer is correct -> combine it with SUPPORTED / WEAK / UNSUPPORTED -> return one diagnostic label.
def classify_evidence_case(relaxed_correct: bool, evidence_support: str) -> str:
    support = evidence_support.upper().strip()

    correctness_prefix = "CORRECT" if relaxed_correct else "WRONG"

    if support not in {
        "SUPPORTED",
        "WEAK",
        "UNSUPPORTED"
    }:
        return "UNKNOWN"

    return f"{correctness_prefix}_{support}"



# take one prediction -> calculate EM/F1 and correctness -> classify its evidence case -> collect all important fields into one analysis dictionary.
def build_analysis_case(prediction: dict[str, Any], index: int) -> dict[str, Any]:
    metrics = calculate_answer_metrics(prediction)

    evidence_support = str(prediction.get("evidence_support", "UNKNOWN"))

    category = classify_evidence_case(
        relaxed_correct=metrics["relaxed_correct"],
        evidence_support=evidence_support
    )

    return {
        "index": index,
        "id": prediction.get("id"),
        "category": category,
        "error_type": metrics["error_type"],
        "is_answerable": parse_boolean(
            prediction.get(
                "is_answerable",
                True
            )
        ),
        "strict_correct": metrics["strict_correct"],
        "relaxed_correct": metrics["relaxed_correct"],
        "exact_match": metrics["exact_match"],
        "token_f1": metrics["token_f1"],
        "question": prediction.get("question", ""),
        "predicted_answer": (get_predicted_answer(prediction)),
        "gold_answers": get_gold_answers(prediction),
        "context": prediction.get("context", ""),
        "evidence_text": prediction.get("evidence_text", ""),
        "answer_context_score": prediction.get("answer_context_score", 0.0),
        "question_evidence_overlap": prediction.get("question_evidence_overlap", 0.0),
        "evidence_score": prediction.get("evidence_score", 0.0),
        "evidence_support": evidence_support,
        "confidence": prediction.get(
            "calibrated_confidence", prediction.get("confidence", None)
        ),
        "threshold_decision": prediction.get(
            "threshold_decision",
            prediction.get("decision", prediction.get("confidence_decision", None))
        ),
        "final_decision": prediction.get("final_decision", None)
    }


# take one analyzed prediction case -> display correctness, EM/F1, question, prediction, gold answers, evidence scores, confidence, and decisions for manual inspection.
def print_case(case: dict[str, Any]) -> None:
    print("\n" + "=" * 80)

    print(f"Index: {case['index']}")

    print(f"ID: {case['id']}")

    print(f"Category: {case['category']}")

    print(f"Error type: {case['error_type']}")

    print(f"Answerable: {case['is_answerable']}")

    print(f"Exact Match: {case['exact_match']:.4f}")

    print(f"Token F1: {case['token_f1']:.4f}")

    print(f"Strict correct: {case['strict_correct']}")

    print(f"Relaxed correct: {case['relaxed_correct']}")

    print(f"\nQuestion:\n{case['question']}")

    print(f"\nPrediction:\n{case['predicted_answer']}")

    print(f"\nGold answers:\n{case['gold_answers']}")

    print(f"\nEvidence text:\n{case['evidence_text']}")

    print("\nVerifier:")

    print(f"  answer_context_score: {case['answer_context_score']}")

    print(f"  question_evidence_overlap: {case['question_evidence_overlap']}")

    print(f"  evidence_score: {case['evidence_score']}")

    print(f"  evidence_support: {case['evidence_support']}")

    print(f"  confidence: {case['confidence']}")

    print(f"  threshold_decision: {case['threshold_decision']}")

    print(f"  final_decision: {case['final_decision']}")


# load predictions -> build analysis cases -> calculate QA/error statistics -> count evidence categories _> find dangerous WRONG_SUPPORTED + ANSWER cases -> print them -> save results.
def analyze_evidence_errors(
    input_path: str | Path, output_path: str | Path, max_examples: int
) -> list[dict[str, Any]]:
    predictions = load_jsonl(input_path)

    if not predictions:
        raise ValueError("Prediction list cannot be empty.")

    analysis_cases: list[dict[str, Any]] = []

    for index, prediction in enumerate(predictions, start=1):
        analysis_cases.append(
            build_analysis_case(
                prediction=prediction,
                index=index
            )
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    save_jsonl(
        analysis_cases,
        output_path
    )

    category_counts = Counter(case["category"] for case in analysis_cases)
    error_type_counts = Counter(case["error_type"] for case in analysis_cases)
    strict_correct_count = sum(case["strict_correct"] for case in analysis_cases)
    relaxed_correct_count = sum(case["relaxed_correct"] for case in analysis_cases)

    average_f1 = sum(case["token_f1"] for case in analysis_cases) / len(analysis_cases)

    print("\nEvidence error analysis completed.")

    print(f"Total predictions: {len(analysis_cases)}")

    print("\nQA metrics:")

    print(
        f"Strict Exact Match accuracy: {strict_correct_count / len(analysis_cases):.4f}"
    )

    print(f"Relaxed accuracy: {relaxed_correct_count / len(analysis_cases):.4f}")

    print(f"Average Token F1: {average_f1:.4f}")

    print("\nEvidence category summary:")

    for category in (
        "WRONG_SUPPORTED",
        "WRONG_WEAK",
        "WRONG_UNSUPPORTED",
        "CORRECT_SUPPORTED",
        "CORRECT_WEAK",
        "CORRECT_UNSUPPORTED",
        "UNKNOWN"
    ):
        print(f"{category}: {category_counts.get(category, 0)}")

    print("\nAnswer error types:")

    for error_type, count in error_type_counts.most_common():
        print(f"{error_type}: {count}")

    critical_cases = [
        case
        for case in analysis_cases
        if (
            case["category"] == "WRONG_SUPPORTED"
            and case["threshold_decision"] == "ANSWER"
        )
    ]

    print("\nCritical cases: WRONG_SUPPORTED + ANSWER")

    print(f"Count: {len(critical_cases)}")

    for case in critical_cases[:max_examples]:
        print_case(case)

    print(f"\nResults saved to: {output_path}")

    return analysis_cases


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Analyze QA correctness and evidence verifier errors.")
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH)
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH)
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=10
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    analyze_evidence_errors(
        input_path=args.input,
        output_path=args.output,
        max_examples=args.max_examples
    )
