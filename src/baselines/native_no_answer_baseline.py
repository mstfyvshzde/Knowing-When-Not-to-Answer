"""
Run the model's native SQuAD v2 no-answer baseline.

Unlike the forced-answer baseline, this baseline keeps the pretrained QA
model's built-in ability to return an empty answer when it predicts that the
context does not support an answer.

The resulting ANSWER or ABSTAIN decisions are saved with prediction scores,
reference answers, answerability labels, and experiment metadata for later
evaluation and comparison.
"""


import argparse
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import pipeline

from src.utils.io import save_jsonl

# Use the same pretrained extractive QA backbone (ana soru-cevap modeli) as
# the forced-answer baseline. Keeping the model fixed makes the comparison
# focus on abstention behavior rather than differences between QA models.
MODEL_NAME = "deepset/roberta-base-squad2"


DATASET_PATH = Path("data/processed/squad_v2")

OUTPUT_DIR = Path("outputs/predictions")


def select_device(device_name: str) -> torch.device:
    """
    Select the hardware device used for QA inference.

    CUDA refers to NVIDIA GPU execution, while MPS is Apple's GPU backend.
    If the requested accelerator is unavailable, an error is raised instead of
    silently changing the experiment hardware.
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

    A split (veri bölümü) is one of train, calibration, or test. Explicit split
    selection helps keep calibration data separate from held-out evaluation data.
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
    """
    Run the QA model with its native no-answer mechanism enabled.

    Native no-answer (modelin kendi cevap vermeme mekanizması) allows the
    SQuAD v2 model to choose between returning an extractive answer span and
    predicting that no valid answer exists in the context.

    A non-empty prediction becomes ANSWER, while an empty prediction becomes
    ABSTAIN.

    Because this baseline uses the same QA backbone as the forced-answer setup,
    it shows how well the model's built-in abstention behavior performs without
    the project's additional selective-ranking methods.
    """

    answers = example.get("answers", {})

    return list(answers.get("text", []))


def run_native_no_answer_baseline(
    split_name: str = "calibration", limit: int | None = 10, device_name: str = "cpu"
) -> list[dict[str, Any]]:
    """
    Run the QA backbone with its native no-answer behavior enabled.

    An empty model prediction is interpreted as ABSTAIN; a non-empty span is
    interpreted as ANSWER. This provides a direct comparison with the
    project's forced-answer selective-ranking setup.
    """

    # Load only the requested experimental split.
    dataset = load_split(split_name)

    # Optionally restrict the number of examples for a smoke test (hızlı sistem
    # kontrolü) or debugging. limit=None processes the complete selected split.
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        dataset = dataset.select(range(min(limit, len(dataset))))

    device = select_device(device_name)

    print(f"Loading model: {MODEL_NAME}")
    print(f"Using device: {device}")
    print(f"Number of examples: {len(dataset)}")

    # Build the same pretrained QA pipeline used by the forced-answer baseline.
    # The important experimental difference appears below, where the model's
    # native impossible-answer option is enabled.
    qa_model = pipeline(
        task="question-answering", model=MODEL_NAME, tokenizer=MODEL_NAME, device=device
    )

    # Store one structured prediction record for every processed example.
    predictions: list[dict[str, Any]] = []

    for index, example in enumerate(dataset, start=1):
        # handle_impossible_answer=True enables the model's native null/no-answer
        # option. Unlike the forced-answer baseline, the model is therefore allowed
        # to decide that the context does not contain a valid answer.
        # top_k=1 keeps only the model's highest-scoring final choice.
        result = qa_model(
            question=example["question"],
            context=example["context"],
            top_k=1,
            handle_impossible_answer=True,
        )

        # Normalize the Hugging Face output to one prediction dictionary.
        if isinstance(result, list):
            result = result[0]

        # In this pipeline, the native null/no-answer decision is represented by an
        # empty answer string. We map that model output to the project's ABSTAIN label.
        prediction_text = str(result["answer"]).strip()
        decision = "ANSWER" if prediction_text else "ABSTAIN"


        # Store the model output together with experiment metadata required for later# evaluation.
        # pipeline_score is the Hugging Face score of the model's selected native
        # outcome. That outcome may represent either an answer span or the null/no-answer
        # choice, so this score should not be confused with the calibrated confidence
        # used by the project's selective-ranking experiments.
        prediction = {
            # unique example ID.
            "id": example["id"],
            # the question given to the model.
            "question": example["question"],
            # the answer generated by the model.
            "prediction_text": prediction_text,
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
            # the system’s final action based on the model native no-answer prediction.
            "decision": decision,
            # name of the method being tested.
            "system": "native_no_answer_baseline",
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
        print(f"Pipeline score: {result['score']:.4f}")
        print(f"Confidence: {result['score']:.4f}")
        print(f"Answerable: {example['is_answerable']}")
        print(f"Decision: {decision}")

    # Keep native no-answer outputs separate from forced-answer predictions so the
    # two experimental baselines can be evaluated independently.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / f"native_no_answer_baseline_{split_name}.jsonl"

    save_jsonl(predictions, output_path)

    print(f"\nPredictions saved to: {output_path}")

    return predictions


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options for split, sample limit, and inference device."""

    parser = argparse.ArgumentParser(
        description="Run the native SQuAD v2 no-answer baseline."
    )

    parser.add_argument(
        "--split", choices=["train", "calibration", "test"], default="calibration"
    )

    parser.add_argument("--limit", type=int, default=10)

    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_native_no_answer_baseline(
        split_name=args.split,
        limit=args.limit,
        device_name=args.device,
    )
