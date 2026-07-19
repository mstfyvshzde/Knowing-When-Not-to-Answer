"""
SQuAD 2.0 dataset download utility.

Bu dosyanın görevi:
1. Dataset'i Hugging Face Hub'dan indirmek.
2. Orijinal veriyi data/raw/ klasörüne kaydetmek.
3. Train ve validation örnek sayılarını göstermek.

Bu dosya veriyi değiştirmez.
Yalnızca indirir ve orijinal hâliyle saklar.
"""

import shutil
from pathlib import Path

# DatasetDict:
# train ve validation gibi birden fazla split içeren veri yapısıdır.
#
# load_dataset:
# Hugging Face Hub'dan dataset indirmek için kullanılır.
from datasets import DatasetDict, load_dataset


# Hugging Face üzerindeki dataset kimliği.
DATASET_NAME = "rajpurkar/squad_v2"

# Dataset'in bilgisayarda kaydedileceği klasör.
#
# Daha önce klasörümüzün adı "datasets" idi.
# Python package ile çakışmaması için "data" yaptık.
OUTPUT_DIR = Path("data/raw/squad_v2")


def download_dataset(
    overwrite: bool = False,
) -> DatasetDict:
    """
    Download SQuAD 2.0 and save it locally.

    Args:
        overwrite:
            False ise mevcut dataset'in üzerine yazılmaz.

            True ise eski klasör silinir ve dataset
            yeniden indirilir.

    Returns:
        Train ve validation splitlerini içeren DatasetDict.
    """

    # Dataset klasörü zaten varsa ne yapacağımıza karar veriyoruz.
    if OUTPUT_DIR.exists():

        # overwrite=False ise yanlışlıkla tekrar indirmeyi engelliyoruz.
        if not overwrite:
            raise FileExistsError(
                f"Dataset already exists at: {OUTPUT_DIR}\n"
                "Delete it manually or use overwrite=True."
            )

        # overwrite=True ise eski dataset klasörünü tamamen siliyoruz.
        shutil.rmtree(OUTPUT_DIR)

    # data/raw klasörü yoksa oluşturuyoruz.
    OUTPUT_DIR.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Downloading dataset: {DATASET_NAME}")

    # Hugging Face Hub'dan dataset'i indiriyoruz.
    #
    # Beklenen yapı:
    # DatasetDict({
    #     train: Dataset(...)
    #     validation: Dataset(...)
    # })
    dataset = load_dataset(DATASET_NAME)

    # Yanlış veya beklenmeyen bir veri yapısı gelirse
    # programı anlaşılır bir hatayla durduruyoruz.
    if not isinstance(dataset, DatasetDict):
        raise TypeError(
            "Expected load_dataset() to return a DatasetDict."
        )

    # İndirilen dataset'i diske kaydediyoruz.
    #
    # Daha sonra internet olmadan şu şekilde açabileceğiz:
    # load_from_disk("data/raw/squad_v2")
    dataset.save_to_disk(str(OUTPUT_DIR))

    print(f"\nDataset saved to: {OUTPUT_DIR}")

    # Her split'in örnek sayısını gösteriyoruz.
    for split_name, split_data in dataset.items():
        print(
            f"{split_name}: "
            f"{len(split_data):,} examples"
        )

    return dataset


if __name__ == "__main__":
    """
    Bu bölüm yalnızca dosya doğrudan çalıştırıldığında çalışır.

    Terminal:
        python -m src.data.download_data
    """

    download_dataset()