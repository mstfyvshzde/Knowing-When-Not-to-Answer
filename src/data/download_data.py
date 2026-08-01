"""
To download the SQuAD v2 dataset from Hugging Face, verify that it contains dataset splits, save it locally, and prevent accidental overwriting of existing data.
"""

# shutil is used for high-level file and folder operations, such as copying, moving, or deleting them.
import shutil

# Create and manage file paths safely across operating systems.
from pathlib import Path

# DatasetDict represents datasets split into parts like train and validation.
# load_dataset downloads or loads a dataset, usually from Hugging Face.
from datasets import DatasetDict, load_dataset

# This constant stores the Hugging Face dataset identifier. Here, it tells load_dataset() to use the SQuAD v2 dataset.
DATASET_NAME = "rajpurkar/squad_v2"

OUTPUT_DIR = Path("data/raw/squad_v2")


def download_dataset(
    overwrite: bool = False,  # overwrite=False means the function should not replace an existing dataset folder by default.
) -> DatasetDict:
    if OUTPUT_DIR.exists():
        # stops replacement when overwrite permission is not given.
        if not overwrite:
            raise FileExistsError(
                f"Dataset already exists at: {OUTPUT_DIR}\n"
                "Delete it manually or use overwrite=True."
            )

        # deletes the old dataset folder completely.
        shutil.rmtree(OUTPUT_DIR)

    # creates the parent folder if it does not exist.
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset: {DATASET_NAME}")

    dataset = load_dataset(DATASET_NAME)

    # checks whether the downloaded object has dataset splits like train and validation.
    if not isinstance(dataset, DatasetDict):
        raise TypeError("Expected load_dataset() to return a DatasetDict.")

    dataset.save_to_disk(str(OUTPUT_DIR))

    print(f"\nDataset saved to: {OUTPUT_DIR}")

    for split_name, split_data in dataset.items():
        print(f"{split_name}: {len(split_data):,} examples")

    return dataset


if __name__ == "__main__":
    download_dataset()
