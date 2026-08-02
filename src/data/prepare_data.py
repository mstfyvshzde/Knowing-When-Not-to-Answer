"""
The main purpose of this file is to transform the raw SQuAD v2 dataset into an experiment-ready dataset.
It adds answerability labels, splits the validation data into balanced calibration and test sets, calculates statistics, and saves the final train, calibration, and test data with metadata.
"""

# shutil is used for high-level file and folder operations, such as copying, moving, or deleting them.
import shutil

# is used to create and manage file paths safely.
from pathlib import Path

# Any is a flexible type hint that allows a variable or dictionary value to contain any Python data type.
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk

from src.utils.io import save_json

RAW_DATA_DIR = Path("data/raw/squad_v2")

OUTPUT_DIR = Path("data/processed/squad_v2")

SEED = 17

# calibration means adjusting a model’s confidence so that the scores better match reality.
# Calibration makes confidence scores more realistic. For example, among predictions with 0.8 confidence, about 80% should actually be correct.
# We do it using a separate calibration set and methods such as temperature scaling, which adjusts confidence values without changing the model’s predicted answers.
# CALIBRATION_FRACTION is the portion of the dataset reserved for this calibration step. For example, 0.2 means 20% of the data is used for calibration, while the rest is used for other purposes such as testing or evaluation.
CALIBRATION_FRACTION = 0.50


# checks whether the example contains at least one answer text.
# It returns True when answers["text"] is not empty, otherwise False.
def is_answerable(example: dict[str, Any]) -> bool:
    answers = example.get("answers", {})

    answer_texts = answers.get("text", [])

    return len(answer_texts) > 0


# checks every example and adds a new is_answerable column. 1 means the example has at least one answer; 0 means it has no answer.
def add_answerability_column(dataset: Dataset) -> Dataset:
    # applies the given function to every example in the dataset.
    return dataset.map(
        lambda example: {"is_answerable": int(is_answerable(example))},
        desc="adding answerablity lables",
    )


# splits the dataset indices into calibration and test sets while keeping the same class distribution (stratified split).
# For example, if 70% of the data is answerable and 30% is unanswerable, both the calibration and test sets will keep approximately the same 70/30 ratio.
def stratified_split_indices(
    labels: list[int], calibration_fraction: float, seed: int
) -> tuple[list[int], list[int]]:

    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1.")

    # creates a NumPy random number generator with a fixed seed.
    # We need it to shuffle the indices randomly, while the fixed seed makes that random order repeatable each time.
    rng = np.random.default_rng(seed)

    calibration_indices: list[int] = []
    test_indices: list[int] = []

    # converts the Python list of labels into a NumPy array.
    # We need it because NumPy operations such as np.unique() and np.where() work more easily and efficiently with arrays.
    labels_array = np.asarray(labels)

    # finds each different class label only once, such as 0 and 1.
    # The loop then processes the indices of each class separately so the calibration and test sets keep a similar class balance.
    for label in np.unique(labels_array):

        # Finds the indices of all examples whose label equals the current label.
        # Example:
        # labels_array = [0, 1, 0, 1]
        # label = 0

        # Result:
        # class_indices = [0, 2]
        class_indices = np.where(labels_array == label)[0]

        # randomly changes the order of the indices inside class_indices.
        # Because the data may be processed in order, seeing only one class first can make evaluation or batches unbalanced. Shuffling mixes the classes and creates a fairer, more random order.
        rng.shuffle(class_indices)

        split_point = int(len(class_indices) * calibration_fraction)

        calibration_indices.extend(class_indices[:split_point].tolist())

        test_indices.extend(class_indices[split_point:].tolist())

    # randomly changes the order of calibration-set indices.
    rng.shuffle(calibration_indices)

    # randomly changes the order of test-set indices.
    rng.shuffle(test_indices)

    return calibration_indices, test_indices


# calculates summary statistics for each dataset split, such as total, answerable, unanswerable, and the answerable ratio.
# It returns all these results in one dictionary.
def build_statistics(dataset: DatasetDict) -> dict[str, Any]:
    statistics: dict[str, Any] = {}

    # split_name is the name of the dataset section, such as "train", "calibration", or "test".
    # split_data is the actual data stored inside that section.
    for split_name, split_data in dataset.items():
        # The result is usually a list of 1s and 0s, where 1 means answerable and 0 means unanswerable.
        labels = split_data["is_answerable"]

        answerable_count = int(sum(labels))

        total_count = len(split_data)

        unanswerable_count = total_count - answerable_count

        statistics[split_name] = {
            "total_examples": total_count,
            "answerable_examples": answerable_count,
            "unanswerable_examples": unanswerable_count,
            "answerable_fraction": (
                answerable_count / total_count if total_count > 0 else 0.0
            ),
        }

    return statistics


# prepares the raw SQuAD v2 data for experiments.
# -loads the raw dataset
# -checks that train and validation splits exist
# -adds the is_answerable label
# -splits validation data into balanced calibration and test sets
# -creates the final train, calibration, and test dataset
# -saves the processed dataset, statistics, and split information to disk
def prepare_dataset(overwrite: bool = False) -> DatasetDict:
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at: {RAW_DATA_DIR}\n"
            "Run python -m src.data.download_data first."
        )

    if OUTPUT_DIR.exists():
        # checks whether permission to replace the existing processed dataset was not given.
        # When overwrite=False, it raises an error to protect the existing files; when overwrite=True, the old folder can be deleted and recreated.
        if not overwrite:
            raise FileExistsError(f"Processed dataset already exists at: {OUTPUT_DIR}")

        # deletes the entire processed dataset folder and everything inside it.
        shutil.rmtree(OUTPUT_DIR)

    print(f"Loading raw dataset from: {RAW_DATA_DIR}")

    raw_dataset = load_from_disk(str(RAW_DATA_DIR))

    if not isinstance(raw_dataset, DatasetDict):
        raise TypeError("Expected the raw dataset to be a DatasetDict.")

    required_splits = {"train", "validation"}

    # finds which required split names are missing from the dataset.
    missing_splits = required_splits.difference(raw_dataset.keys())

    if missing_splits:
        raise ValueError(f"Missing dataset splits: {missing_splits}")

    # adds an is_answerable label to every training example.
    # train is the part used to teach the model patterns from examples.
    train_dataset = add_answerability_column(raw_dataset["train"])

    # adds an is_answerable label to every validation example.
    # validation is a separate part used to tune settings and check performance during development without touching the final test set.
    validation_dataset = add_answerability_column(raw_dataset["validation"])

    calibration_indices, test_indices = stratified_split_indices(
        labels=validation_dataset["is_answerable"],
        calibration_franction=CALIBRATION_FRACTION,
        seed=SEED,
    )

    # takes only the rows whose indices belong to the calibration set.
    # calibration is used to tune confidence scores or decision thresholds without changing the model itself.
    calibration_dataset = validation_dataset.select(calibration_indices)

    # takes the remaining selected rows for the test set.
    # test is kept separate and used only for the final unbiased evaluation of the system.
    test_dataset = validation_dataset.select(test_indices)

    prepared_dataset = DatasetDict(
        {
            "train": train_dataset,
            "calibration": calibration_dataset,
            "test": test_dataset,
        }
    )

    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

    prepared_dataset.save_to_disk(str(OUTPUT_DIR))
    statistics = build_statistics(prepared_dataset)

    save_json(statistics, OUTPUT_DIR / "statistics.json")

    save_json(
        {
            "source_dataset": "rajpurkar/squad_v2",
            "seed": SEED,
            "calibration_fraction": CALIBRATION_FRACTION,
            "splits": {
                split_name: len(split_data)
                for split_name, split_data in prepared_dataset.items()
            },
        },
        OUTPUT_DIR / "split_manifest.json",
    )

    print(f"\nPrepared dataset saved to: {OUTPUT_DIR}")

    for split_name, split_stats in statistics.items():
        print(
            f"{split_name}: "
            f"{split_stats['total_examples']:,} total | "
            f"{split_stats['answerable_examples']:,} answerable | "
            f"{split_stats['unanswerable_examples']:,} unanswerable"
        )

    return prepared_dataset


if __name__ == "__main__":
    prepare_dataset()
