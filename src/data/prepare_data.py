"""
Prepare the SQuAD v2 dataset for the project's experimental protocol.

The original validation split is deterministically divided into two disjoint
sets: a calibration split and a held-out test split. The split is stratified
by answerability so both subsets preserve a similar ratio of answerable and
unanswerable examples.

The calibration split may be used for confidence calibration and parameter
selection. The held-out test split is reserved for final evaluation and is
not used for tuning.
"""

import shutil
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk

from src.utils.io import save_json

RAW_DATA_DIR = Path("data/raw/squad_v2")

OUTPUT_DIR = Path("data/processed/squad_v2")

SEED = 17


# Half of the original SQuAD v2 validation split is reserved for calibration.
# Calibration (kalibrasyon) means adjusting confidence scores so they better
# reflect how often predictions are actually correct. This split may also be
# used to select experiment parameters.
# The held-out test split is kept separate and is not used for tuning.
CALIBRATION_FRACTION = 0.50



def is_answerable(example: dict[str, Any]) -> bool:
    """
    Return True if a SQuAD v2 example contains at least one gold/reference answer.

    A gold answer (referans cevap) is a correct answer provided by the dataset.
    Unanswerable SQuAD v2 examples have an empty answers["text"] list.
    """
    answers = example.get("answers", {})

    answer_texts = answers.get("text", [])

    return len(answer_texts) > 0



def add_answerability_column(dataset: Dataset) -> Dataset:
    """
    Add a binary answerability label to every example.

    is_answerable = 1 means the context contains a gold answer.
    is_answerable = 0 means the question is unanswerable from the given context.

    This label is later used for stratified splitting and evaluation.
    """

    # applies the given function to every example in the dataset.
    return dataset.map(
        lambda example: {"is_answerable": int(is_answerable(example))},
        desc="adding answerability labels",
    )



def stratified_split_indices(
    labels: list[int], calibration_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """
    Split example indices into deterministic calibration and test subsets.

    Stratified splitting (katmanlı bölme) processes answerable and unanswerable
    examples separately. This keeps approximately the same answerability
    distribution in both subsets.

    The fixed seed makes the resulting split reproducible across runs.
    """

    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1.")

    # A fixed random seed makes the split exactly reproducible across runs.
    rng = np.random.default_rng(seed)

    calibration_indices: list[int] = []
    test_indices: list[int] = []

    # Convert labels to an array for class-wise index selection.
    labels_array = np.asarray(labels)

    # Process each answerability class separately so both final subsets keep
    # approximately the same answerable/unanswerable distribution.
    for label in np.unique(labels_array):

        # Collect the positions of all examples belonging to the current class
        # (normally 0 = unanswerable and 1 = answerable).
        class_indices = np.where(labels_array == label)[0]

        # Shuffle within the class before assigning examples to each subset.
        rng.shuffle(class_indices)

        split_point = int(len(class_indices) * calibration_fraction)

        calibration_indices.extend(class_indices[:split_point].tolist())

        test_indices.extend(class_indices[split_point:].tolist())

    # The classes were split separately, so shuffle the final subsets again to
    # mix answerable and unanswerable examples. The seeded generator keeps this
    # final ordering reproducible.
    rng.shuffle(calibration_indices)
    rng.shuffle(test_indices)

    return calibration_indices, test_indices



def build_statistics(dataset: DatasetDict) -> dict[str, Any]:
    """
    Compute answerability statistics for every prepared dataset split.

    These counts provide a simple integrity check (veri bütünlüğü kontrolü) and
    make it easy to verify that calibration and test kept similar class ratios.
    """
    statistics: dict[str, Any] = {}


    for split_name, split_data in dataset.items():
        # Because is_answerable is stored as 1 or 0, summing the labels directly
        # gives the number of answerable examples.
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



def prepare_dataset(overwrite: bool = False) -> DatasetDict:
    """
    Build and save the experiment-ready SQuAD v2 dataset.

    The original training split is preserved. The original validation split is
    deterministically divided into stratified calibration and held-out test sets.

    Calibration data may influence confidence calibration or parameter selection,
    whereas held-out test data is reserved for final evaluation.

    This function prepares data only; it does not train or fine-tune the QA model.
    """

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at: {RAW_DATA_DIR}\n"
            "Run python -m src.data.download_data first."
        )

    if OUTPUT_DIR.exists():
        # Protect an existing processed dataset unless replacement is explicitly requested.
        if not overwrite:
            raise FileExistsError(f"Processed dataset already exists at: {OUTPUT_DIR}")

        # Remove the previous processed dataset before rebuilding it.
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

    # Preserve the original training split and attach answerability metadata.
    # This preprocessing step does not train or fine-tune the QA model.
    train_dataset = add_answerability_column(raw_dataset["train"])

    # SQuAD v2 does not expose gold labels for its official test split.
    # Therefore, the original validation split is divided into:
    # calibration -> ayar/kalibrasyon için kullanılabilecek veri
    # held-out test -> yalnızca nihai değerlendirme için saklanan veri
    validation_dataset = add_answerability_column(raw_dataset["validation"])

    # Stratified splitting keeps approximately the same answerable/unanswerable
    # distribution in both subsets while SEED makes the split reproducible.
    calibration_indices, test_indices = stratified_split_indices(
        labels=validation_dataset["is_answerable"],
        calibration_fraction=CALIBRATION_FRACTION,
        seed=SEED,
    )

    # Calibration split: may be used to fit calibration parameters or select
    # experiment settings.
    calibration_dataset = validation_dataset.select(calibration_indices)

    # Held-out test split (nihai değerlendirme verisi): kept separate so its labels
    # cannot influence calibration or parameter selection.
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

    # Save dataset composition for quick integrity and reproducibility checks.
    save_json(statistics, OUTPUT_DIR / "statistics.json")

    # Record the exact split configuration needed to reproduce this dataset setup.
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
