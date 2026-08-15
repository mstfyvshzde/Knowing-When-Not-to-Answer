"""
Verify forced-answer QA predictions using a secondary extractive QA model.

The verifier receives the same question and context as the original QA model
and independently produces its own answer or native no-answer prediction.

The original answer and verifier output are compared using token-overlap F1
together with the verifier's pipeline score.

Each prediction is assigned one heuristic evidence label:

- SUPPORTED: verifier gives a sufficiently confident matching answer
- UNSUPPORTED: verifier confidently gives no answer or a strongly different answer
- UNCERTAIN: verifier evidence is not decisive

This is a prototype evidence-checking heuristic. Agreement between two QA
models should not be interpreted as proof that an answer is factually correct,
and disagreement does not necessarily mean logical contradiction.
"""

import argparse
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import pipeline

from src.utils.io import load_jsonl, save_jsonl

# Use a different pretrained extractive QA backbone as a secondary verifier.
# Keeping it separate from the main RoBERTa QA model provides an additional
# model-based signal, although the two systems should not be treated as
# statistically independent evidence sources.
VERIFIER_MODEL_NAME = "deepset/deberta-v3-base-squad2"


DEFAULT_INPUT_PATH = Path("outputs/predictions/raw_baseline_calibration.jsonl")

OUTPUT_PATH = Path("outputs/predictions/evidence_verified_calibration.jsonl")



def normalize_answer(text: str) -> str:
    """
    Normalize answer text before lexical overlap comparison.

    Text is lowercased, punctuation and English articles are removed, and
    repeated whitespace is collapsed.

    This reduces superficial differences such as capitalization or punctuation
    before comparing the two QA answers.
    """

    text = text.lower()

    # This block removes punctuation from the text.
    text = "".join(
        character for character in text if character not in string.punctuation
    )

    # It removes the articles a, an, and the from the text before comparison.
    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text,
    )

    return " ".join(text.split())


def answer_overlap_f1(first_answer: str, second_answer: str) -> float:
    """
    Measure token-level overlap between the original and verifier answers.

    Precision measures how much of the original answer overlaps with the
    verifier answer, while recall measures how much of the verifier answer is
    covered by the original answer.

    The resulting F1 score lies between 0 and 1:

    1.0 -> complete normalized token overlap
    0.0 -> no normalized token overlap
    """

    first_tokens = normalize_answer(first_answer).split()
    second_tokens = normalize_answer(second_answer).split()

    if not first_tokens or not second_tokens:
        return 0.0

    common_tokens = Counter(first_tokens) & Counter(second_tokens)

    overlap = sum(common_tokens.values())

    if overlap == 0:
        return 0.0

    # Precision uses the first answer because the first answer is treated as the prediction.
    precision = overlap / len(first_tokens)

    # Recall uses the second answer because the second answer is treated as the reference/verifier answer.
    recall = overlap / len(second_tokens)

    f1_score = 2 * precision * recall / (precision + recall)

    return f1_score


def select_device(device_name: str) -> torch.device:
    """
    Select the hardware device used for verifier inference.

    CUDA refers to NVIDIA GPU execution, while MPS is Apple's GPU backend.
    If the requested accelerator is unavailable, an error is raised instead of
    silently changing the experiment hardware.
    """

    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")

        return torch.device("mps")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

        return torch.device("cuda")

    return torch.device("cpu")


def classify_evidence(
    generated_answer: str,
    verifier_answer: str,
    verifier_score: float,
    support_threshold: float,
    match_threshold: float,
    rejection_threshold: float,
) -> tuple[str, str, float]:
    """
    Convert verifier output into a heuristic evidence label.

    The decision combines two signals:

    1. lexical answer agreement measured by token-overlap F1
    2. the secondary QA model's own pipeline score

    SUPPORTED requires both strong answer overlap and sufficient verifier score.

    UNSUPPORTED is used when the verifier confidently predicts no answer or
    confidently produces a substantially different answer.

    Otherwise the result is UNCERTAIN.

    These labels describe agreement between QA systems; they are not formal
    logical entailment or contradiction judgments.
    """

    LOW_MATCH_THRESHOLD = 0.20

    answer_match = answer_overlap_f1(generated_answer, verifier_answer)

    # verifier_answer -> The answer given by the verifier model.
    if not verifier_answer.strip():
        # verifier_score -> how confident the verifier model is in its own answer.
        # rejection_threshold -> The minimum verifier confidence needed to strongly reject the generated answer as unsupported.
        if verifier_score >= rejection_threshold:
            return (
                "UNSUPPORTED",
                (
                    "Verifier predicted that the question is unanswerable from the context."
                ),
                answer_match,
            )

        return (
            "UNCERTAIN",
            (
                "Verifier returned no answer but its score "
                "was not high enough for rejection."
            ),
            answer_match,
        )

    # answer_match -> How similar the original answer and verifier answer are.
    # match_threshold -> The minimum similarity needed to consider the two answers a match.
    # verifier_score -> How confident the verifier model is in its own answer.
    # support_threshold -> The minimum verifier confidence needed to label the original answer as SUPPORTED.
    if answer_match >= match_threshold and verifier_score >= support_threshold:
        return (
            "SUPPORTED",
            "Independent verifier produced a matching answer.",
            answer_match,
        )

    if (
        answer_match < LOW_MATCH_THRESHOLD
        and verifier_score >= rejection_threshold
    ):
        return (
            "UNSUPPORTED",
            "Independent verifier confidently produced a different answer.",
            answer_match,
        )

    return (
        "UNCERTAIN",
        "Verifier evidence was not decisive.",
        answer_match,
    )


def verify_predictions(
    predictions: list[dict[str, Any]],
    device_name: str = "cpu",
    support_threshold: float = 0.30,
    match_threshold: float = 0.80,
    rejection_threshold: float = 0.50,
) -> list[dict[str, Any]]:
    """
    Run the secondary QA verifier over stored forced-answer predictions.

    For every example, the verifier receives the original question and context.
    Native no-answer behavior remains enabled so the verifier may either return
    an answer span or indicate that the context does not support an answer.

    Its output is then compared with the original forced answer and attached to
    the prediction as prototype evidence metadata.
    """

    device = select_device(device_name)

    print(f"Loading verifier model: {VERIFIER_MODEL_NAME}")

    print(f"Using device: {device}")

    # It loads the verifier QA model and creates a question-answering pipeline that can answer questions from a given context.
    verifier = pipeline(
        task="question-answering",
        model=VERIFIER_MODEL_NAME,
        tokenizer=VERIFIER_MODEL_NAME,
        device=device,
    )

    verified_predictions: list[dict[str, Any]] = []

    for index, prediction in enumerate(predictions, start=1):
        question = prediction["question"]
        context = prediction["context"]

        generated_answer = prediction["prediction_text"]


        # Keep only the verifier's highest-ranked outcome and allow its native
        # SQuAD v2 no-answer option. Unlike the original forced-answer baseline,
        # this verifier is therefore allowed to return an empty answer.
        verifier_result = verifier(
            question=question,
            context=context,
            top_k=1,
            handle_impossible_answer=True
        )

        verifier_answer = str(verifier_result["answer"]).strip()
        verifier_score = float(verifier_result["score"])

        evidence_label, evidence_reason, answer_match = classify_evidence(
            generated_answer=generated_answer,
            verifier_answer=verifier_answer,
            verifier_score=verifier_score,
            support_threshold=support_threshold,
            match_threshold=match_threshold,
            rejection_threshold=rejection_threshold,
        )

        # Preserve the original prediction and attach the secondary verifier output,
        # lexical agreement score, and resulting heuristic evidence label.
        verified_prediction = prediction.copy()

        verified_prediction.update(
            {
                "verifier_answer": verifier_answer,
                "verifier_score": verifier_score,
                "answer_match_f1": answer_match,
                "evidence_label": evidence_label,
                # short explanation of why the evidence was classified as SUPPORTED, UNSUPPORTED, or UNCERTAIN.
                "evidence_reason": evidence_reason,
                "verifier_model": VERIFIER_MODEL_NAME,
            }
        )

        verified_predictions.append(verified_prediction)

        print(f"\nExample {index}/{len(predictions)}")

        print(f"Question: {question}")

        print(f"Generated answer: {generated_answer}")

        print(f"Verifier answer: {verifier_answer or '[NO ANSWER]'}")

        print(f"Verifier pipeline score: {verifier_score:.4f}")

        print(f"Answer match F1: {answer_match:.4f}")

        print(f"Evidence label: {evidence_label}")

    return verified_predictions


def summarize_evidence(
    predictions: list[dict[str, Any]],
) -> dict[str, int | float]:
    """
    Count SUPPORTED, UNSUPPORTED, and UNCERTAIN verifier outcomes.

    The returned rates describe the distribution of heuristic evidence labels
    and should not be interpreted directly as prediction accuracy.
    """

    total = len(predictions)

    if total == 0:
        raise ValueError("Prediction list cannot be empty.")

    supported = sum(
        prediction["evidence_label"] == "SUPPORTED" for prediction in predictions
    )

    unsupported = sum(
        prediction["evidence_label"] == "UNSUPPORTED" for prediction in predictions
    )

    uncertain = total - (supported + unsupported)

    return {
        "total": total,
        "supported": supported,
        "unsupported": unsupported,
        "uncertain": uncertain,
        "supported_rate": supported / total,
        "unsupported_rate": unsupported / total,
        "uncertain_rate": uncertain / total,
    }


def run_evidence_checker(
    input_path: str | Path,
    output_path: str | Path,
    device_name: str,
) -> list[dict[str, Any]]:
    """
    Run the complete secondary-QA evidence-checking workflow.

    The function loads forced-answer predictions, verifies them with the
    secondary QA model, attaches evidence metadata, saves the enriched records,
    and reports the distribution of evidence labels.
    """

    predictions = load_jsonl(input_path)

    missing_context = any("context" not in prediction for prediction in predictions)

    if missing_context:
        raise ValueError(
            "Some predictions do not contain 'context'. "
            "Run raw_answer_baseline.py again after adding "
            "the context field."
        )

    verified_predictions = verify_predictions(
        predictions=predictions, device_name=device_name
    )

    save_jsonl(verified_predictions, output_path)

    summary = summarize_evidence(verified_predictions)

    print("\nEvidence verification completed.")

    print(f"SUPPORTED: {summary['supported']}")

    print(f"UNSUPPORTED: {summary['unsupported']}")

    print(f"UNCERTAIN: {summary['uncertain']}")

    print(f"Results saved to: {output_path}")

    return verified_predictions


def parse_arguments() -> argparse.Namespace:
    """Parse evidence-verifier paths and inference-device settings."""

    parser = argparse.ArgumentParser(
        description="Verify forced answers using a secondary QA model."
    )

    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))

    parser.add_argument("--output", default=str(OUTPUT_PATH))

    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_evidence_checker(
        input_path=args.input,
        output_path=args.output,
        device_name=args.device,
    )
