"""
Verify predicted answers with a lexical context-evidence heuristic.

This prototype verifier checks whether an extractive QA answer is present in
the context and whether the surrounding evidence window overlaps with important
content words from the question.

For each prediction, it computes:

- answer_context_score: lexical support for the predicted answer in the context
- question_evidence_overlap: overlap between question content words and the
  local evidence window
- evidence_score: weighted combination of those two signals

The resulting label is SUPPORTED, WEAK, or UNSUPPORTED.

This is a lightweight lexical heuristic, not a semantic entailment model.
A SUPPORTED label therefore indicates strong lexical/contextual agreement,
not proof that the answer is logically or factually correct.
"""

import argparse
import re
import string
from pathlib import Path
from typing import Any

from src.utils.io import load_jsonl, save_jsonl

DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/calibration_with_decisions.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "outputs/predictions/calibration_with_evidence.jsonl"
)

# Remove common function words before lexical overlap calculation so the score
# focuses more strongly on content-bearing words from the question.
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

# Minimum question/evidence overlap required for the strongest support label.
MIN_SUPPORTED_QUESTION_OVERLAP = 0.45


def normalize_text(text: str) -> str:
    """
    Normalize text before lexical comparison.

    Text is lowercased, punctuation is removed, and repeated whitespace is
    collapsed so superficial formatting differences do not affect matching.
    """

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_content_words(text: str) -> set[str]:
    """
    Extract normalized content words used for lexical overlap.

    Stop words (anlam yükü düşük yaygın kelimeler) and one-character tokens
    are removed so overlap focuses on more informative terms.
    """

    normalized = normalize_text(text)

    tokens = normalized.split()

    content_words = {
        token
        for token in tokens
        if token not in STOP_WORDS and len(token) > 1
    }

    return content_words


def find_answer_position(
    context: str,
    answer: str,
) -> tuple[int, int] | None:
    """
    Find the first case-insensitive occurrence of an answer in the context.

    Returns the character start/end positions when found, otherwise None.

    This function acts as a fallback when stored answer-span coordinates are
    unavailable.
    """

    if not answer.strip():
        return None

    context_lower = context.lower()
    answer_lower = answer.lower().strip()

    start_index = context_lower.find(answer_lower)

    if start_index == -1:
        return None

    end_index = start_index + len(answer_lower)

    return start_index, end_index


def extract_evidence_window(
    context: str,
    answer_start: int,
    answer_end: int,
    window_size: int = 120,
) -> str:
    """
    Extract local context surrounding the predicted answer.

    The evidence window (cevabın çevresindeki yerel metin) preserves nearby
    context while avoiding comparison against the entire passage.
    """

    window_start = max(0, answer_start - window_size)

    window_end = min(
        len(context),
        answer_end + window_size,
    )

    return context[window_start:window_end].strip()


def calculate_question_evidence_overlap(
    question: str,
    evidence: str,
) -> float:
    """
    Measure how much of the question's content vocabulary appears in evidence.

    The score is the fraction of question content words that are also present
    in the extracted evidence window.

    A larger value means the local evidence contains more terms related to the
    question.
    """

    question_words = tokenize_content_words(question)
    evidence_words = tokenize_content_words(evidence)

    if not question_words:
        return 0.0

    shared_words = question_words & evidence_words

    overlap = len(shared_words) / len(question_words)

    return float(overlap)


def calculate_answer_context_score(
    answer: str,
    context: str,
) -> float:
    """
    Measure lexical support for the predicted answer in the context.

    Exact normalized containment receives score 1.0.

    If the complete normalized answer is not found, the function falls back to
    the fraction of answer content words that occur somewhere in the context.
    """

    normalized_answer = normalize_text(answer)
    normalized_context = normalize_text(context)

    if not normalized_answer:
        return 0.0

    if normalized_answer in normalized_context:
        return 1.0

    answer_words = tokenize_content_words(answer)
    context_words = tokenize_content_words(context)

    if not answer_words:
        return 0.0

    shared_words = answer_words & context_words

    return float(len(shared_words) / len(answer_words))


def classify_evidence_support(
    answer_context_score: float,
    question_overlap: float,
    supported_threshold: float,
    weak_threshold: float,
) -> str:
    """
    Classify lexical evidence as SUPPORTED, WEAK, or UNSUPPORTED.

    The heuristic combines:

        65% answer-context support
        35% question-evidence overlap

    SUPPORTED requires the answer to be fully present in the context together
    with sufficiently relevant surrounding evidence.

    WEAK represents partial lexical support.

    Otherwise the evidence is labeled UNSUPPORTED.
    """

    # Give direct answer/context agreement more weight than question-word overlap.
    combined_score = (
        0.65 * answer_context_score
        + 0.35 * question_overlap
    )

    if (
        answer_context_score >= 1.0
        and question_overlap >= MIN_SUPPORTED_QUESTION_OVERLAP
        and combined_score >= supported_threshold
    ):
        return "SUPPORTED"

    if (
        answer_context_score >= 0.50
        and combined_score >= weak_threshold
    ):
        return "WEAK"

    return "UNSUPPORTED"


def verify_prediction(
    prediction: dict[str, Any],
    evidence_window_size: int,
    supported_threshold: float,
    weak_threshold: float,
) -> dict[str, Any]:
    """
    Apply the lexical evidence heuristic to one QA prediction.

    The function identifies the predicted answer span, extracts surrounding
    evidence, measures answer/context support and question/evidence overlap,
    combines those signals, and attaches the resulting support label to the
    original prediction.
    """

    question = str(prediction.get("question", ""))
    context = str(prediction.get("context", ""))

    predicted_answer = str(
        prediction.get(
            "predicted_answer",
            prediction.get(
                "prediction_text",
                prediction.get("answer", ""),
            ),
        )
    )

    # Prefer the exact character span selected by the original QA model.
    # Text search is used only when stored coordinates are unavailable.
    if "start" in prediction and "end" in prediction:
        answer_start = int(prediction["start"])
        answer_end = int(prediction["end"])

        if not (0 <= answer_start < answer_end <= len(context)):
            raise ValueError(
                "Invalid stored answer span for prediction "
                f"{prediction.get('id', 'unknown')}."
            )

        answer_position: tuple[int, int] | None = (
            answer_start,
            answer_end,
        )

    else:
        answer_position = find_answer_position(
            context=context,
            answer=predicted_answer,
        )

    if answer_position is None:
        evidence_text = ""

        answer_context_score = calculate_answer_context_score(
            answer=predicted_answer,
            context=context,
        )

        question_overlap = 0.0

    else:
        answer_start, answer_end = answer_position

        evidence_text = extract_evidence_window(
            context=context,
            answer_start=answer_start,
            answer_end=answer_end,
            window_size=evidence_window_size,
        )

        answer_context_score = calculate_answer_context_score(
            answer=predicted_answer,
            context=context,
        )

        question_overlap = calculate_question_evidence_overlap(
            question=question,
            evidence=evidence_text,
        )

    combined_evidence_score = (
        0.65 * answer_context_score
        + 0.35 * question_overlap
    )

    support_label = classify_evidence_support(
        answer_context_score=answer_context_score,
        question_overlap=question_overlap,
        supported_threshold=supported_threshold,
        weak_threshold=weak_threshold,
    )

    # Preserve the original prediction and attach the lexical evidence diagnostics.
    updated_prediction = prediction.copy()

    updated_prediction.update(
        {
            "evidence_text": evidence_text,
            "answer_context_score": answer_context_score,
            "question_evidence_overlap": question_overlap,
            "evidence_score": combined_evidence_score,
            "evidence_support": support_label,
            "evidence_verifier": "lexical_extractive_baseline",
        }
    )

    return updated_prediction


def validate_predictions(
    predictions: list[dict[str, Any]],
) -> None:
    """
    Validate fields required by the lexical evidence verifier.

    Every record must contain a question, context, and at least one supported
    prediction-answer field.
    """

    if not predictions:
        raise ValueError("Prediction list cannot be empty.")

    required_fields = {
        "question",
        "context",
    }

    for index, prediction in enumerate(predictions, start=1):
        missing_fields = [
            field
            for field in required_fields
            if field not in prediction
        ]

        has_answer_field = any(
            field in prediction
            for field in (
                "predicted_answer",
                "prediction_text",
                "answer",
            )
        )

        if not has_answer_field:
            missing_fields.append("predicted_answer")

        if missing_fields:
            raise ValueError(
                f"Prediction {index} is missing fields: {missing_fields}. "
                f"Available keys: {list(prediction.keys())}"
            )


def run_evidence_verification(
    input_path: str | Path,
    output_path: str | Path,
    evidence_window_size: int,
    supported_threshold: float,
    weak_threshold: float,
) -> list[dict[str, Any]]:
    """
    Run the lexical evidence verifier over a prediction file.

    Predictions are validated, enriched with evidence scores and support labels,
    saved to disk, and summarized by support category.
    """

    if evidence_window_size <= 0:
        raise ValueError(
            "evidence_window_size must be greater than zero."
        )

    if not (
        0.0 <= weak_threshold < supported_threshold <= 1.0
    ):
        raise ValueError(
            "Thresholds must satisfy: "
            "0 <= weak < supported <= 1."
        )

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
            f"evidence={verified_prediction['evidence_score']:.4f} | "
            f"support={verified_prediction['evidence_support']}"
        )

    save_jsonl(
        verified_predictions,
        output_path,
    )

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
    """Parse lexical-evidence verification settings."""

    parser = argparse.ArgumentParser(
        description=(
            "Verify QA predictions using a lexical context-evidence heuristic."
        )
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
        help=(
            "Number of context characters kept on each side of the answer."
        ),
    )

    parser.add_argument(
        "--supported-threshold",
        type=float,
        default=0.75,
        help="Minimum combined score required for SUPPORTED.",
    )

    parser.add_argument(
        "--weak-threshold",
        type=float,
        default=0.40,
        help="Minimum combined score required for WEAK.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_evidence_verification(
        input_path=args.input,
        output_path=args.output,
        evidence_window_size=args.evidence_window_size,
        supported_threshold=args.supported_threshold,
        weak_threshold=args.weak_threshold,
    )