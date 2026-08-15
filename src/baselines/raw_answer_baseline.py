"""
Run the project's forced-answer extractive QA baseline.

The pretrained RoBERTa SQuAD v2 model is applied to each example and is forced
to return a non-empty answer span by disabling its native no-answer behavior.
This baseline therefore never abstains.

The resulting predictions are saved with their raw pipeline scores, reference
answers, answerability labels, and experiment metadata for later calibration,
verification, and selective-ranking evaluation.
"""

import argparse
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import pipeline

from src.utils.io import save_jsonl

# Pretrained extractive QA backbone (ana soru-cevap modeli) used throughout
# the project. Extractive QA means the model selects an answer span directly
# from the provided context instead of generating a new free-form answer.
MODEL_NAME = "deepset/roberta-base-squad2"

DATASET_PATH = Path("data/processed/squad_v2")

OUTPUT_DIR = Path("outputs/predictions")


def select_device(device_name: str) -> torch.device:
    """
    Select the hardware device used for QA inference.

    CUDA refers to NVIDIA GPU execution, while MPS is Apple's GPU backend.
    If the requested accelerator is unavailable, the function raises an error
    instead of silently running the experiment on different hardware.
    """
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

        return torch.device("cuda")

    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")

        return torch.device("mps")

    return torch.device("cpu")


def load_split(split_name: str) -> Dataset:
    """
    Load one split from the processed SQuAD v2 dataset.

    A split (veri bölümü) is one of train, calibration, or test. Loading the
    requested split explicitly helps keep calibration and held-out evaluation
    data separate during experiments.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at: {DATASET_PATH}\n"
            "Run python -m src.data.prepare_data first."
        )

    dataset = load_from_disk(DATASET_PATH)

    if not isinstance(dataset, DatasetDict):
        raise TypeError("Expected the processed data to be a DatasetDict.")

    if split_name not in dataset:
        raise ValueError(
            f"Unknown split: {split_name}. Available splits: {list(dataset.keys())}"
        )

    return dataset[split_name]



def build_reference_answers(example: dict[str, Any]) -> list[str]:
    """Return all gold answer texts, or an empty list for unanswerable examples."""
    answers = example.get("answers", {})

    return list(answers.get("text", []))



def run_raw_baseline(
    split_name: str = "calibration", limit: int | None = 10, device_name: str = "cpu"
) -> list[dict[str, Any]]:
    """
    Run the forced-answer extractive QA baseline on one dataset split.

    Forced-answer (zorunlu cevap) means the model is not allowed to use its
    native no-answer option. Every example must therefore receive a non-empty
    answer span, including questions that are actually unanswerable.

    These raw predictions become the common starting point for later confidence
    calibration, verification, and selective-ranking experiments.

    The raw Hugging Face pipeline score stored here is not yet the calibrated
    confidence used in the final evaluation.
    """

    # Load only the requested experimental split.
    dataset = load_split(split_name)

    # Optionally restrict the number of examples for a smoke test (hızlı sistem
    # kontrolü) or debugging. limit=None processes the entire selected split.
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        dataset = dataset.select(range(min(limit, len(dataset))))

    device = select_device(device_name)

    print(f"Loading model: {MODEL_NAME}")
    print(f"Using device: {device}")
    print(f"Number of examples: {len(dataset)}")

    # Build the pretrained extractive QA inference pipeline.
    qa_model = pipeline(
        task="question-answering", model=MODEL_NAME, tokenizer=MODEL_NAME, device=device
    )

    # Store one structured prediction record for every processed example.
    predictions: list[dict[str, Any]] = []

    for index, example in enumerate(dataset, start=1):
        # Disable the model's native no-answer mechanism, forcing it to propose answer
        # spans even for examples that may actually be unanswerable.
        # top_k=5 returns several ranked candidate spans. If the first candidate is
        # empty or has an invalid character span, the next valid candidate can be used
        # without changing the forced-answer nature of the baseline.
        candidates = qa_model(
            question=example["question"],
            context=example["context"],
            top_k=5,
            handle_impossible_answer=False,
        )

        # Hugging Face may return either one dictionary or a list depending on the
        # pipeline output. Convert both cases to a list so filtering is consistent.
        if isinstance(candidates, dict):
            candidates = [candidates]

        # Select the highest-ranked valid extractive span.
        # A valid span must contain non-empty text and its end character position must
        # be greater than its start position.
        result = next(
            (
                candidate
                for candidate in candidates
                if str(candidate["answer"]).strip()
                and int(candidate["end"]) > int(candidate["start"])
            ),
            None,
        )

        if result is None:
            raise ValueError(
                f"No valid non-empty answer span found for example: {example['id']}"
            )

        # Store both the QA result and experiment metadata (deneyi tanımlayan ek bilgi)
        # required by later stages such as calibration, verification, and evaluation.
        # pipeline_score is the raw Hugging Face span score. It must not be confused
        # with the temperature-calibrated confidence computed later in the pipeline.
        prediction = {
            "id": example["id"],
            # the question given to the model.
            "question": example["question"],
            # the answer generated by the model.
            "prediction_text": result["answer"],
            # the model’s confidence score for that answer.
            "pipeline_score": float(result["score"]),
            # character position where the predicted answer starts in the context.
            "start": int(result["start"]),
            #  character position where the predicted answer ends.
            "end": int(result["end"]),
            # correct answers from the dataset.
            "reference_answers": build_reference_answers(example),
            # whether the question actually has an answer in the context.
            "is_answerable": bool(example["is_answerable"]),
            # the system’s final action; here it always answers.
            "decision": "ANSWER",
            # name of the method being tested.
            "system": "raw_baseline",
            # Hugging Face model used.
            "model": MODEL_NAME,
            # dataset split, such as calibration or test.
            "split": split_name,
            # passage used to answer the question.
            "context": example["context"],
        }

        predictions.append(prediction)

        print(f"\nExample {index}/{len(dataset)}")
        print(f"Question: {example['question']}")
        print(f"Prediction: {result['answer']}")
        print(f"Pipeline score: {result['score']:.4f}")
        print(f"Answerable: {example['is_answerable']}")
        print("Decision: ANSWER")

    # Predictions are experiment artifacts, so they are stored under outputs/
    # rather than modifying the processed dataset itself.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / f"raw_baseline_{split_name}.jsonl"

    save_jsonl(predictions, output_path)

    print(f"\nPredictions saved to: {output_path}")

    return predictions


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options for split, sample limit, and inference device."""

    parser = argparse.ArgumentParser(
        description="Run the forced-answer extractive QA baseline."
    )

    parser.add_argument(
        "--split", choices=["train", "calibration", "test"], default="calibration"
    )

    parser.add_argument("--limit", type=int, default=10)

    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_raw_baseline(
        split_name=args.split,
        limit=args.limit,
        device_name=args.device,
    )
