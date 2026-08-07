"""
This file verifies whether each predicted answer is supported by the context by extracting evidence, calculating evidence scores, and classifying the evidence as SUPPORTED, WEAK, or UNSUPPORTED.
"""

import argparse
import re
import string
from pathlib import Path
from typing import Any

from src.utils.io import load_jsonl, save_jsonl

DEFAULT_INPUT_PATH = Path("outputs/predictions/calibration_with_decisions.jsonl")

DEFAULT_OUTPUT_PATH = Path("outputs/predictions/calibration_with_evidence.jsonl")


# Common words removed before comparing text so that only meaningful words remain for evidence matching.
STOP_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "from",
    "with",
    "by",
    "and",
    "or",
    "but",
    "if",
    "because",
    "while",
    "as",
    "than",
    "then",
    "also",
    "into",
    "over",
    "under",
    "between",
    "after",
    "before",
    "during",
    "through",
    "about",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "when",
    "where",
    "why",
    "how",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "they",
    "them",
    "their",
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "we",
    "us",
    "our",
    "ours",
    "you",
    "your",
    "yours",
    "i",
    "me",
    "my",
    "mine",
    "all",
    "any",
    "each",
    "every",
    "other",
    "such",
    "some",
    "many",
    "much",
    "more",
    "most",
    "few",
    "less",
    "very",
    "can",
    "could",
    "should",
    "would",
    "will",
    "may",
    "might",
    "according",
    "name",
}


# converts text into a clean, standardized format so it can be compared consistently.
def normalize_text(text: str) -> str:
    text = text.lower()

    # removes punctuation like . , ? ! : ;
    text = text.translate(str.maketrans("", "", string.punctuation))

    # replaces multiple spaces/newlines/tabs with one space and removes spaces from the beginning/end.
    text = re.sub(r"\s+", " ", text).strip()

    return text


# converts text into a set of meaningful words for content comparison.
def tokenize_content_words(text: str) -> set[str]:
    normalized = normalize_text(text)

    tokens = normalized.split()

    content_words = {
        token for token in tokens if token not in STOP_WORDS and len(token) > 1
    }

    return content_words


# locates the answer in the context and returns its character positions.
def find_answer_position(context: str, answer: str) -> tuple[int, int] | None:
    if not answer.strip():
        return None

    context_lower = context.lower()
    answer_lower = answer.lower().strip()

    start_index = context_lower.find(answer_lower)

    if start_index == -1:
        return None

    end_index = start_index + len(answer_lower)

    return start_index, end_index


# returns the surrounding context of the answer to use as evidence for verification.
# A window is a small portion of the context taken around the answer instead of using the entire passage.
# Example:
# Context: "... Albert Einstein was born in Ulm, Germany, in 1879. He later moved to Switzerland ..."
# Answer: "Ulm"
# With window_size = 20, the extracted window might be: "... was born in Ulm, Germany, in 1879 ..."
def extract_evidence_window(
    context: str, answer_start: int, answer_end: int, window_size: int = 120
) -> str:
    window_start = max(0, answer_start - window_size)
    window_end = min(len(context), answer_end + window_size)

    evidence = context[window_start:window_end].strip()

    return evidence


# measures how much the evidence overlaps with the important words in the question. It simply checks how many important words from the question also appear in the evidence.
def calculate_question_evidence_overlap(question: str, evidence: str) -> float:
    question_words = tokenize_content_words(question)
    evidence_words = tokenize_content_words(evidence)

    if not question_words:
        return 0.0

    shared_words = question_words & evidence_words

    # the part that two things have in common.
    overlap = len(shared_words) / len(question_words)

    return float(overlap)


# measures how well the answer is supported by the context by checking whether the answer (or its important words) appears in the context.
def calculate_answer_context_score(asnwer: str, context: str) -> float:
    normalized_answer = normalize_text(asnwer)
    normalized_context = normalize_text(context)

    if not normalized_answer:
        return 0.0

    if normalized_answer in normalized_context:
        return 1.0

    answer_words = tokenize_content_words(asnwer)
    context_words = tokenize_content_words(context)

    if not answer_words:
        return 0.0

    shared_words = answer_words & context_words

    return float(len(shared_words) / len(answer_words))


# classifies the evidence as SUPPORTED, WEAK, or UNSUPPORTED based on the answer-context score and the question-overlap score.
def classify_evidence_support(
    answer_context_score: float,
    question_overlap: float,
    supported_threshold: float,
    weak_threshold: float,
) -> str:
    # answer_context_score -> How well the answer is supported by the context.
    # question_overlap -> How well the evidence matches the important words in the question.
    combined_score = 0.65 * answer_context_score + 0.35 * question_overlap

    # supported_threshold -> the minimum combined score required to label the evidence as SUPPORTED
    if (
        answer_context_score >= 1.0
        and question_overlap >= 0.45
        and combined_score >= supported_threshold
    ):
        return "SUPPORTED"

    # WEAK_THRESHOLD -> the minimum combined score required to label it as WEAK.
    if answer_context_score >= 0.50 and combined_score >= weak_threshold:
        return "WEAK"

    return "UNSUPPORTED"


# verifies a single prediction by extracting evidence from the context, calculating evidence scores, classifying the evidence as SUPPORTED, WEAK, or UNSUPPORTED, and returning the updated prediction with verification results.
def verify_prediction(
    prediction: dict[str, Any],
    evidence_window_size: int,
    supported_threshold: float,
    weak_threshold: float,
) -> dict[str, Any]:
    question = str(prediction.get("question", ""))
    context = str(prediction.get("context", ""))

    predicted_answer = str(
        prediction.get(
            "predicted_answer",
            prediction.get("prediction_text", prediction.get("answer", "")),
        )
    )

    answer_position = find_answer_position(context=context, answer=predicted_answer)

    if answer_position is None:
        evidence_text = ""

        answer_context_score = calculate_answer_context_score(
            asnwer=predicted_answer, context=context
        )

        question_overlap = 0.0

        combined_evidece_score = 0.65 * answer_context_score + 0.35 * question_overlap

    else:
        answer_start, answer_end = answer_position

        evidence_text = extract_evidence_window(
            context=context,
            answer_start=answer_start,
            answer_end=answer_end,
            window_size=evidence_window_size,
        )

        answer_context_score = calculate_answer_context_score(
            asnwer=question, context=context
        )

        question_overlap = calculate_question_evidence_overlap(
            question=question, evidence=evidence_text
        )

        combined_evidece_score = 0.65 * answer_context_score + 0.35 * question_overlap

    support_label = classify_evidence_support(
        answer_context_score=answer_context_score,
        question_overlap=question_overlap,
        supported_threshold=supported_threshold,
        weak_threshold=weak_threshold,
    )

    updated_prediction = prediction.copy()

    updated_prediction.update(
        {
            "evidence_text": evidence_text,
            "answer_context_score": (answer_context_score),
            "question_evidence_overlap": (question_overlap),
            "evidence_score": (combined_evidece_score),
            "evidence_support": (support_label),
            "evidence_verifier": ("lexical_extractive_baseline"),
        }
    )

    return updated_prediction


# checks that the prediction list is valid and that every prediction contains all the required fields before evidence verification begins.
def validate_predictions(predictions: list[dict[str, Any]]) -> None:
    if not predictions:
        raise ValueError("Prediction list cannot be empty.")

    required_fields = {"question", "context"}

    for index, prediction in enumerate(predictions, start=1):
        missing_fields = [field for field in required_fields if field not in prediction]

        # Check if the prediction has an answer in any supported answer field.
        # is a Python built-in function that returns True if at least one condition is True, otherwise it returns False.
        has_answer_field = any(
            field in prediction
            for field in ("prediction_answer", "prediction_text", "answer")
        )

        if not has_answer_field:
            missing_fields.append("predicted_answer")

            if missing_fields:
                raise ValueError(
                    f"Prediction {index} is missing "
                    f"fields: {missing_fields}. "
                    f"Available keys: "
                    f"{list(prediction.keys())}"
                )


# runs the complete evidence verification pipeline: it loads predictions, validates them, verifies the evidence for each prediction, saves the verified results, prints a summary, and returns the verified predictions.
def run_evidence_verification(
    input_path: str | Path,
    output_path: str | Path,
    evidence_window_size: int,
    supported_threshold: float,
    weak_threshold: float,
) -> list[dict[str, Any]]:
    if not (0.0 <= weak_threshold < supported_threshold <= 1.0):
        raise ValueError("Thresholds must satisfy: 0 <= weak < supported <= 1.")

    predictions = load_jsonl(input_path)
    validate_predictions(predictions)
    verified_predictions: list[dict[str, Any]] = []

    for index, prediction in enumerate(predictions, start=1):
        verified_prediction = verify_prediction(
            prediction=prediction,
            evidence_window_size=evidence_window_size,
            supported_threshold=supported_threshold,
            weak_threshold=weak_threshold,
        )

        verified_predictions.append(verified_prediction)

        print(
            f"{index}/{len(predictions)} | "
            f"evidence="
            f"{verified_prediction['evidence_score']:.4f} | "
            f"support="
            f"{verified_prediction['evidence_support']}"
        )

    save_jsonl(verified_predictions, output_path)

    support_counts = {
        "SUPPORTED": 0,
        "WEAK": 0,
        "UNSUPPORTED": 0,
    }

    for prediction in verified_predictions:
        support_counts[prediction["evidence_support"]] += 1

    print("\nEvidence verification completed.")

    print(f"SUPPORTED: {support_counts['SUPPORTED']}")

    print(f"WEAK: {support_counts['WEAK']}")

    print(f"UNSUPPORTED: {support_counts['UNSUPPORTED']}")

    print(f"Results saved to: {output_path}")

    return verified_predictions


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify whether QA predictions are supported by their context."
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
    )

    parser.add_argument(
        "--evidence-window-size",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--supported-threshold",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--weak-threshold",
        type=float,
        default=0.40,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_evidence_verification(
        input_path=args.input,
        output_path=args.output,
        evidence_window_size=(args.evidence_window_size),
        supported_threshold=(args.supported_threshold),
        weak_threshold=(args.weak_threshold),
    )
