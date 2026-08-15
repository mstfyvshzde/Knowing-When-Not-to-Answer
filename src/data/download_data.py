"""
Download and preserve the raw SQuAD v2 dataset used by this project.

The dataset is loaded from Hugging Face and saved locally under data/raw/
without project-specific transformations.

Keeping an untouched raw copy provides a stable starting point for later
preprocessing (ön işleme) and makes it possible to rebuild the processed
dataset from the same source data.

Existing raw data is protected from accidental replacement unless
overwrite=True is explicitly requested.
"""


import shutil
from pathlib import Path

from datasets import DatasetDict, load_dataset

# Hugging Face dataset identifier used throughout the project.
DATASET_NAME = "rajpurkar/squad_v2"

OUTPUT_DIR = Path("data/raw/squad_v2")


def download_dataset(overwrite: bool = False) -> DatasetDict:
    """
    Download, validate, and save the raw SQuAD v2 dataset.

    By default, an existing local copy is preserved. Setting overwrite=True
    explicitly removes that copy before downloading and saving the dataset
    again.

    No answerability labels, calibration splits, or other project-specific
    transformations are created here; those belong to the preprocessing step.
    """

    if OUTPUT_DIR.exists():
        # Protect an existing raw dataset unless replacement is explicitly requested.
        if not overwrite:
            raise FileExistsError(
                f"Dataset already exists at: {OUTPUT_DIR}\n"
                "Delete it manually or use overwrite=True."
            )

        # Remove the existing dataset before saving a fresh copy.
        shutil.rmtree(OUTPUT_DIR)

    # Ensure the raw-data directory exists before saving the dataset.
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset: {DATASET_NAME}")

    # Load SQuAD v2 using the Hugging Face dataset identifier defined above.
    # Project-specific preprocessing is deliberately performed later so this
    # directory remains an untouched raw-data source.
    dataset = load_dataset(DATASET_NAME)

    # The project expects named splits such as train and validation.
    if not isinstance(dataset, DatasetDict):
        raise TypeError("Expected load_dataset() to return a DatasetDict.")

    # Save the untouched raw dataset locally for reproducible preprocessing.
    dataset.save_to_disk(str(OUTPUT_DIR))

    print(f"\nDataset saved to: {OUTPUT_DIR}")

    # Report the downloaded split sizes for a quick integrity check.
    for split_name, split_data in dataset.items():
        print(f"{split_name}: {len(split_data):,} examples")

    return dataset


if __name__ == "__main__":
    download_dataset()
