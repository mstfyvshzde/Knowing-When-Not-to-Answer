"""
Input/output utilities for experiment files.

Bu dosyanın görevi:
- Python verilerini JSON olarak kaydetmek
- Her prediction'ı ayrı satırda JSONL olarak kaydetmek
- Sonuç tablolarını CSV olarak kaydetmek
- Kaydedilmiş JSON ve JSONL dosyalarını yeniden okumak
"""

import csv
import json
from pathlib import Path
from typing import Any


def ensure_parent_directory(path: str | Path) -> Path:
    """
    Dosyanın kaydedileceği üst klasörleri oluşturur.

    Örnek:
        path = "outputs/predictions/results.json"

    Eğer outputs/predictions klasörü yoksa otomatik oluşturur.

    Returns:
        Path nesnesi olarak dosya yolunu döndürür.
    """

    # String yolu Path nesnesine çeviriyoruz.
    file_path = Path(path)

    # Dosyanın kendisini değil, bulunduğu üst klasörü alıyoruz.
    # Örnek:
    # outputs/predictions/results.json
    # parent → outputs/predictions
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return file_path


def save_json(
    data: Any,
    path: str | Path,
    indent: int = 2,
) -> None:
    """
    Python verisini JSON dosyası olarak kaydeder.

    Kullanılabilecek veri örnekleri:
    - dictionary
    - list
    - string
    - number

    Örnek:
        save_json(
            {"accuracy": 0.82},
            "outputs/tables/metrics.json"
        )
    """

    # Üst klasör yoksa oluşturuyoruz.
    file_path = ensure_parent_directory(path)

    # Dosyayı yazma modunda açıyoruz.
    with file_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        # Python nesnesini JSON formatına çevirip kaydediyoruz.
        json.dump(
            data,
            file,
            indent=indent,

            # Türkçe veya başka Unicode karakterlerin
            # bozulmadan kaydedilmesini sağlar.
            ensure_ascii=False,
        )


def load_json(path: str | Path) -> Any:
    """
    JSON dosyasını okuyup Python verisine dönüştürür.

    Örnek:
        metrics = load_json(
            "outputs/tables/metrics.json"
        )
    """

    file_path = Path(path)

    # Dosya yoksa anlaşılır hata mesajı veriyoruz.
    if not file_path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {file_path}"
        )

    # Dosyayı okuma modunda açıyoruz.
    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        # JSON içeriğini Python dictionary veya list'e çevirir.
        return json.load(file)


def save_jsonl(
    records: list[dict[str, Any]],
    path: str | Path,
) -> None:
    """
    Prediction kayıtlarını JSON Lines formatında kaydeder.

    JSONL formatında her satır ayrı bir JSON nesnesidir.

    Örnek dosya:

        {"id": "1", "answer": "Paris"}
        {"id": "2", "answer": "London"}

    Bu format büyük prediction dosyaları için kullanışlıdır.
    """

    file_path = ensure_parent_directory(path)

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        # Her prediction kaydını tek tek işliyoruz.
        for record in records:

            # Dictionary'yi JSON metnine çeviriyoruz.
            json_line = json.dumps(
                record,
                ensure_ascii=False,
            )

            # Her JSON kaydını ayrı satıra yazıyoruz.
            file.write(json_line + "\n")


def load_jsonl(
    path: str | Path,
) -> list[dict[str, Any]]:
    """
    JSONL dosyasındaki bütün kayıtları okur.

    Returns:
        Her satırdaki JSON nesnelerini içeren bir liste.
    """

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"JSONL file not found: {file_path}"
        )

    # Okunan bütün prediction'lar burada tutulacak.
    records: list[dict[str, Any]] = []

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        # Satır numarasını da takip ediyoruz.
        # Hatalı JSON varsa hangi satırda olduğunu söyleyebiliriz.
        for line_number, line in enumerate(
            file,
            start=1,
        ):

            # Satır başı ve sonundaki boşlukları temizliyoruz.
            stripped_line = line.strip()

            # Boş satırları görmezden geliyoruz.
            if not stripped_line:
                continue

            try:
                # JSON satırını Python dictionary'ye çeviriyoruz.
                record = json.loads(stripped_line)
                records.append(record)

            except json.JSONDecodeError as error:
                # JSON bozuksa hangi satırda hata olduğunu gösteriyoruz.
                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number}: {file_path}"
                ) from error

    return records


def save_csv(
    rows: list[dict[str, Any]],
    path: str | Path,
) -> None:
    """
    Dictionary listesini CSV tablosu olarak kaydeder.

    Örnek:

        rows = [
            {
                "system": "raw_baseline",
                "accuracy": 0.70
            },
            {
                "system": "verified",
                "accuracy": 0.82
            }
        ]
    """

    # Boş listeyle CSV kolonlarını belirleyemeyiz.
    if not rows:
        raise ValueError(
            "Cannot save an empty list to CSV."
        )

    file_path = ensure_parent_directory(path)

    # İlk dictionary'nin key'lerini kolon isimleri kabul ediyoruz.
    fieldnames = list(rows[0].keys())

    with file_path.open(
        "w",
        encoding="utf-8",

        # CSV dosyasında gereksiz boş satır oluşmasını önler.
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        # İlk satıra kolon başlıklarını yazar.
        writer.writeheader()

        # Listedeki bütün dictionary'leri tablo satırı olarak yazar.
        writer.writerows(rows)


if __name__ == "__main__":
    """
    Dosyanın fonksiyonlarını test etmek için örnek bölüm.

    Terminal:
        python -m src.utils.io
    """

    example_predictions = [
        {
            "question_id": "example-1",
            "answer": "Albert Einstein",
            "confidence": 0.91,
            "decision": "ANSWER",
        },
        {
            "question_id": "example-2",
            "answer": "",
            "confidence": 0.24,
            "decision": "ABSTAIN",
        },
    ]

    # Normal JSON dosyası oluştur.
    save_json(
        example_predictions,
        "outputs/predictions/example.json",
    )

    # Her prediction'ı ayrı satıra kaydet.
    save_jsonl(
        example_predictions,
        "outputs/predictions/example.jsonl",
    )

    # Prediction'ları tablo olarak kaydet.
    save_csv(
        example_predictions,
        "outputs/tables/example.csv",
    )

    # Kaydedilen JSON dosyasını yeniden oku.
    loaded_predictions = load_json(
        "outputs/predictions/example.json"
    )

    print("Loaded predictions:")
    print(loaded_predictions)