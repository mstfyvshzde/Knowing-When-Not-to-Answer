"""
The main purpose of this file is to run a raw question-answering baseline that always gives an answer and never abstains.
It loads a selected dataset split, runs the RoBERTa SQuAD 2.0 model on each example, stores the predicted answer, confidence, reference answers, and metadata, then saves all predictions as a JSONL file for later evaluation and comparison.
"""

# argparse is used to read command-line arguments when you run the script from the terminal.
import argparse

# is used to create and manage file paths safely.
from pathlib import Path

# Any is a flexible type hint that allows a variable or dictionary value to contain any Python data type.
from typing import Any

# is the main PyTorch library used for tensor operations and running neural-network models.
import torch

# Dataset represents one dataset split, such as only train.
# DatasetDict represents several splits together, such as train, calibration, and test.
# load_from_disk loads a dataset that was previously saved locally.
from datasets import Dataset, DatasetDict, load_from_disk

# creates a ready-to-use Transformers task pipeline, such as question answering, without manually writing the full model and tokenizer inference code.
from transformers import pipeline

from src.utils.io import save_jsonl

# stores the Hugging Face model identifier that the QA pipeline will load.
# Here, it selects a RoBERTa model trained for SQuAD 2.0, so it can answer questions and also handle unanswerable ones.
MODEL_NAME = "deepset/roberta-base-squad2"


DATASET_PATH = Path("data/processed/squad_v2")

OUTPUT_DIR = Path("outputs/predictions")


# chooses which hardware PyTorch should use:
# We need it so the model runs on the best available hardware instead of always using the CPU. A GPU can make inference much faster, while the availability checks prevent the program from requesting hardware that does not exist.
def select_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but isnt avalable yet")

        return torch.device("cuda")

    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available yet")

        return torch.device("mps")

    return torch.device("cpu")


# loads the prepared dataset from disk, checks that it has the expected structure, verifies that the requested split exists, and returns only that split.
def load_split(split_name: str) -> Dataset:
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


# extracts all correct answer texts from one dataset example and returns them as a list of strings.
# For an unanswerable example, it returns an empty list.
def build_reference_answers(example: dict[str, Any]) -> list[str]:
    answers = example.get("answers", {})

    return list(answers.get("text", []))


# runs the basic question-answering model on a selected dataset split and saves all predictions as a JSONL file.
# It loads the split, optionally limits the number of examples, selects CPU/GPU, generates one answer for each question, stores the prediction details, and always uses the decision "ANSWER" without abstaining.
def run_raw_baseline(
    split_name: str = "calibration", limit: int | None = 10, device_name: str = "cpu"
) -> list[dict[str, Any]]:
    # loads only the requested dataset split, such as "calibration" or "test".
    dataset = load_split(split_name)

    # limit means the maximum number of dataset examples the function will process.
    # For example, limit=10 processes only the first 10 examples; limit=None processes the entire selected split.
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        dataset = dataset.select(range(min(limit, len(dataset))))

    device = select_device(device_name)

    print(f"Loading model: {MODEL_NAME}")
    print(f"Using device: {device}")
    print(f"Number of examples: {len(dataset)}")

    # creates a ready-to-use Hugging Face question-answering system.
    qa_model = pipeline(
        task="question-answering", model=MODEL_NAME, tokenizer=MODEL_NAME, device=device
    )

    # creates an empty list that will store one prediction dictionary for each dataset example.
    predictions: list[dict[str, Any]] = []

    # loops through every dataset example and also gives each one a visible number starting from 1.
    for index, example in enumerate(dataset, start=1):
        candidates = qa_model(
            question=example["question"],
            context=example["context"],
            top_k=5,
            handle_impossible_answer=False,
        )

        if isinstance(candidates, dict):
            candidates = [candidates]

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

        # This is one prediction record stored as a Python dictionary. It collects everything about one question, the model’s answer, and the experiment setup.
        prediction = {
            # unique example ID.
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
        print(f"Confidence: {result['score']:.4f}")
        print(f"Answerable: {example['is_answerable']}")
        print("Decision: ANSWER")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / f"raw_baseline_{split_name}.jsonl"

    save_jsonl(predictions, output_path)

    print(f"\nPredictions saved to: {output_path}")

    return predictions


# reads options given from the terminal and converts them into a Python object.
# It lets the user choose the dataset split, number of examples, and device without changing the code.
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run the raw question-answering baseline.")
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
