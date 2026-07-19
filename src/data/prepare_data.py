"""
SQuAD 2.0 dataset preparation pipeline.

Bu dosyanın görevi:
1. Ham dataset'i data/raw/squad_v2 klasöründen yüklemek.
2. Her soruya answerable/unanswerable etiketi eklemek.
3. Validation split'ini calibration ve test olarak bölmek.
4. Bölünmüş veriyi data/processed/squad_v2 içine kaydetmek.
5. Split bilgilerini ve istatistikleri JSON olarak saklamak.

Önemli:
- Calibration split threshold seçmek için kullanılır.
- Test split yalnızca final değerlendirmede kullanılmalıdır.
"""

import shutil
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk

from src.utils.io import save_json


# İndirdiğimiz orijinal dataset'in konumu.
RAW_DATA_DIR = Path("data/raw/squad_v2")

# Hazırlanmış dataset'in kaydedileceği konum.
OUTPUT_DIR = Path("data/processed/squad_v2")

# Split işleminin her çalıştırmada aynı sonucu vermesi için seed.
SEED = 17

# Validation verisinin %50'si calibration için kullanılacak.
# Geri kalan %50 final test seti olacak.
CALIBRATION_FRACTION = 0.50
# Kalibrasyon, modelin verdiği tahminlerin ne kadar güvenilir olduğunu kontrol edip ayarlama işlemidir.
# Mesela model “Bu tahmin %80 doğru” diyorsa, kalibrasyon verileri bu %80’in gerçekten güvenilir olup olmadığını kontrol eder


def is_answerable(example: dict[str, Any]) -> bool:
    """
    Bir sorunun context kullanılarak cevaplanabilir olup
    olmadığını kontrol eder.

    SQuAD 2.0 formatında:

        answers["text"] boş değilse → answerable
        answers["text"] boşsa      → unanswerable

    Returns:
        True  → soru cevaplanabilir
        False → context içinde cevap yok
    """

    # Örneğin reference cevaplarını alıyoruz.
    answers = example.get("answers", {})

    # answers içindeki metin listesini alıyoruz.
    answer_texts = answers.get("text", [])

    # En az bir reference cevap varsa soru answerable kabul edilir.
    return len(answer_texts) > 0


def add_answerability_column(
    dataset: Dataset,
) -> Dataset:
    """
    Dataset'e is_answerable isimli yeni bir kolon ekler.

    Örnek:

        answerable soru   → is_answerable = 1
        unanswerable soru → is_answerable = 0
    """

    # map(), dataset içindeki her örneğe aynı fonksiyonu uygular.
    return dataset.map(
        lambda example: {
            "is_answerable": int(
                is_answerable(example)
            )
        },
        desc="Adding answerability labels",
    )


def stratified_split_indices(
    labels: list[int],
    calibration_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """
    Validation verisini stratified olarak ikiye böler.

    Stratified split:
        Answerable/unanswerable oranını calibration ve
        test setlerinde mümkün olduğunca aynı tutar.

    Returns:
        calibration_indices
        test_indices
    """

    # Geçersiz oranları engelliyoruz.
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError(
            "calibration_fraction must be between 0 and 1."
        )

    # Modern NumPy random generator oluşturuyoruz.
    # Aynı seed kullanıldığı için split tekrar üretilebilir.
    rng = np.random.default_rng(seed)

    calibration_indices: list[int] = []
    test_indices: list[int] = []

    # Python listesini NumPy array'e çeviriyoruz.
    labels_array = np.asarray(labels)

    # Unique labels burada 0 ve 1 olacaktır.
    for label in np.unique(labels_array):

        # Bu label'a ait örneklerin indexlerini buluyoruz.
        class_indices = np.where(
            labels_array == label
        )[0]

        # Index sırasını rastgele karıştırıyoruz.
        rng.shuffle(class_indices)

        # Bu sınıftaki örneklerin kaç tanesi calibration'a gidecek?
        split_point = int(
            len(class_indices) * calibration_fraction
        )

        # İlk bölüm calibration setine gider.
        calibration_indices.extend(
            class_indices[:split_point].tolist()
        )

        # Kalan bölüm test setine gider.
        test_indices.extend(
            class_indices[split_point:].tolist()
        )

    # Calibration ve test içindeki örnek sırasını da karıştırıyoruz.
    rng.shuffle(calibration_indices)
    rng.shuffle(test_indices)

    return calibration_indices, test_indices


def build_statistics(
    dataset: DatasetDict,
) -> dict[str, Any]:
    """
    Her split için temel istatistikleri hesaplar.

    Örnek çıktı:

        {
            "calibration": {
                "total_examples": 5900,
                "answerable_examples": 2960,
                "unanswerable_examples": 2940
            }
        }
    """

    statistics: dict[str, Any] = {}

    # Train, calibration ve test splitlerini tek tek inceliyoruz.
    for split_name, split_data in dataset.items():

        # 1 ve 0 değerlerinden oluşan answerability kolonu.
        labels = split_data["is_answerable"]

        # 1'lerin toplamı answerable soru sayısıdır.
        answerable_count = int(sum(labels))

        total_count = len(split_data)

        # Kalan sorular unanswerable'dır.
        unanswerable_count = (
            total_count - answerable_count
        )

        statistics[split_name] = {
            "total_examples": total_count,
            "answerable_examples": answerable_count,
            "unanswerable_examples": unanswerable_count,
            "answerable_fraction": (
                answerable_count / total_count
                if total_count > 0
                else 0.0
            ),
        }

    return statistics


def prepare_dataset(
    overwrite: bool = False,
) -> DatasetDict:
    """
    Ham dataset'i araştırmaya hazırlar.

    Oluşturulan splitler:
        train
        calibration
        test
    """

    # Dataset daha önce indirilmemişse programı durduruyoruz.
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at: {RAW_DATA_DIR}\n"
            "Run python -m src.data.download_data first."
        )

    # Hazırlanmış dataset zaten varsa yanlışlıkla silmeyelim.
    if OUTPUT_DIR.exists():

        if not overwrite:
            raise FileExistsError(
                f"Processed dataset already exists at: "
                f"{OUTPUT_DIR}"
            )

        # overwrite=True verilirse eski processed veri silinir.
        shutil.rmtree(OUTPUT_DIR)

    print(f"Loading raw dataset from: {RAW_DATA_DIR}")

    # save_to_disk() ile kaydedilen dataset'i tekrar açıyoruz.
    raw_dataset = load_from_disk(
        str(RAW_DATA_DIR)
    )

    if not isinstance(raw_dataset, DatasetDict):
        raise TypeError(
            "Expected the raw dataset to be a DatasetDict."
        )

    # Gerekli splitlerin mevcut olduğunu kontrol ediyoruz.
    required_splits = {"train", "validation"}

    missing_splits = required_splits.difference(
        raw_dataset.keys()
    )

    if missing_splits:
        raise ValueError(
            f"Missing dataset splits: {missing_splits}"
        )

    # Train verisine answerability kolonu ekle.
    train_dataset = add_answerability_column(
        raw_dataset["train"]
    )

    # Validation verisine answerability kolonu ekle.
    validation_dataset = add_answerability_column(
        raw_dataset["validation"]
    )

    # Validation setini calibration ve test olarak böl.
    calibration_indices, test_indices = (
        stratified_split_indices(
            labels=validation_dataset["is_answerable"],
            calibration_fraction=CALIBRATION_FRACTION,
            seed=SEED,
        )
    )

    # select(), verilen indexlerdeki örnekleri seçer.
    calibration_dataset = validation_dataset.select(
        calibration_indices
    )

    test_dataset = validation_dataset.select(
        test_indices
    )

    # Yeni dataset yapısını oluşturuyoruz.
    prepared_dataset = DatasetDict(
        {
            "train": train_dataset,
            "calibration": calibration_dataset,
            "test": test_dataset,
        }
    )

    # data/processed klasörü yoksa oluşturuyoruz.
    OUTPUT_DIR.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Hazırlanmış dataset'i diske kaydediyoruz.
    prepared_dataset.save_to_disk(
        str(OUTPUT_DIR)
    )

    # Split istatistiklerini hesaplıyoruz.
    statistics = build_statistics(
        prepared_dataset
    )

    # İstatistikleri JSON olarak kaydediyoruz.
    save_json(
        statistics,
        OUTPUT_DIR / "statistics.json",
    )

    # Split'in nasıl oluşturulduğunu kaydediyoruz.
    save_json(
        {
            "source_dataset": "rajpurkar/squad_v2",
            "seed": SEED,
            "calibration_fraction": CALIBRATION_FRACTION,
            "splits": {
                split_name: len(split_data)
                for split_name, split_data
                in prepared_dataset.items()
            },
        },
        OUTPUT_DIR / "split_manifest.json",
    )

    print(
        f"\nPrepared dataset saved to: {OUTPUT_DIR}"
    )

    # Sonuçları Terminal'de gösteriyoruz.
    for split_name, split_stats in statistics.items():
        print(
            f"{split_name}: "
            f"{split_stats['total_examples']:,} total | "
            f"{split_stats['answerable_examples']:,} answerable | "
            f"{split_stats['unanswerable_examples']:,} unanswerable"
        )

    return prepared_dataset


if __name__ == "__main__":
    """
    Terminal:

        python -m src.data.prepare_data
    """

    prepare_dataset()