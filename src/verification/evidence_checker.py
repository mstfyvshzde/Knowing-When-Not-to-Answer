"""
This file uses an independent QA model to verify the original model's answers, classify them as SUPPORTED, UNSUPPORTED, or UNCERTAIN, and save the verification results for later decision-making.
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

# It names the independent QA model used to check the original model’s answer.
# We need it so the system can compare the original answer with a second model’s answer and judge whether the evidence supports it.
VERIFIER_MODEL_NAME = "deepset/deberta-v3-base-squad2"


DEFAULT_INPUT_PATH = Path("outputs/predictions/raw_baseline_calibration.jsonl")

OUTPUT_PATH = Path("outputs/predictions/evidence_verified_calibration.jsonl")


# converts answers into a clean, standardized format so they can be compared fairly, even if they have different capitalization, punctuation, articles, or spacing._
def normalize_answe(text: str) -> str:
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


# measures how similar two answers are by comparing their words and returns an F1 similarity score between 0 and 1.
def answer_overlap_f1(first_answer: str, second_answer: str) -> float:
    # The model’s answer is the answer produced by your main QA system first
    first_tokens = normalize_answe(first_answer).split()

    # the verifier model’s answer is a second, independent answer generated from the same question and context to check whether it agrees.
    second_tokens = normalize_answe(second_answer).split()

    if not first_answer or not second_answer:
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


# it checks whether the requested device is available and returns the correct torch.device for efficiency
def select_device(device_name: str) -> torch.device:
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")

        return torch.device("mps")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

        return torch.device("cuda")

    return torch.device("cpu")


# decides whether the generated answer is SUPPORTED, UNSUPPORTED, or UNCERTAIN by comparing it with the verifier model’s answer and confidence score.
def classify_evdence(
    generated_answer: str,
    verifier_answer: str,
    verifier_score: float,
    support_threshold: float,
    match_threshold: float,
    contradiction_threshold: float,
) -> tuple[str, str, float]:
    answer_match = answer_overlap_f1(generated_answer, verifier_answer)

    # verifier_answer -> The answer given by the verifier model.
    if not verifier_answer.strip():
        # verifier_score -> how confident the verifier model is in its own answer.
        # contradiction_threshold -> The minimum verifier confidence needed to strongly reject the generated answer as unsupported.
        if verifier_score >= contradiction_threshold:
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

    if answer_match < 0.20 and verifier_score >= contradiction_threshold:
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


# uses an independent verifier model to check every generated answer, classify the evidence, and return the updated predictions with verification results.
def verify_predictions(
    predictions: list[dict[str, Any]],
    device_name: str = "cpu",
    support_threshold: float = 0.30,
    match_threshold: float = 0.80,
    contradiction_threshold: float = 0.50,
) -> list[dict[str, Any]]:
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

        # top_k=1 -> Return only the single best answer.
        # handle_impossible_answer=True -> Allow the model to return no answer if the context does not contain one.
        verifier_result = verifier(
            question=question, context=context, top_k=1, handle_impossible_answer=True
        )

        verifier_answer = str(verifier_result["answer"])
        verifier_score = float(verifier_result["score"])

        evidence_label, evidence_reason, answer_match = classify_evdence(
            generated_answer=generated_answer,
            verifier_answer=verifier_answer,
            verifier_score=verifier_score,
            support_threshold=support_threshold,
            match_threshold=match_threshold,
            contradiction_threshold=contradiction_threshold,
        )

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

        print(f"Verifier score: {verifier_score:.4f}")

        print(f"Answer match F1: {answer_match:.4f}")

        print(f"Evidence label: {evidence_label}")

    return verified_predictions


# counts how many predictions are SUPPORTED, UNSUPPORTED, and UNCERTAIN, then calculates their rates.
def summarize_evidence(
    predictions: list[dict[str, Any]],
) -> dict[str, int | float]:
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


# runs the complete evidence checking pipeline: it loads predictions, verifies them with the verifier model, saves the results, prints a summary, and returns the verified predictions.
def run_evidence_checker(
    input_path: str | Path, output_path: str | Path, device_name: str
) -> list[dict[str, Any]]:
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
    """
    Terminal argümanlarını okur.
    """

    parser = argparse.ArgumentParser(
        description=("Verify generated answers using an independent QA model.")
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
