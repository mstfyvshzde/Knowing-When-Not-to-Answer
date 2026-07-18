"""
Bu dosya; verileri JSON, JSONL ve CSV formatlarında kaydetmek ve JSON/JSONL dosyalarını geri okumak için hazırlanmış yardımcı fonksiyonlar içerir. 📁
save_json() veriyi .json dosyasına kaydeder.
load_json() JSON dosyasını okuyup Python verisine çevirir.
save_jsonl() her kaydı ayrı satıra yazar.
load_jsonl() JSONL dosyasını satır satır okur.
save_csv() sözlük listesini CSV tablosu olarak kaydeder.
Ana bölümde örnek tahminler üç farklı formatta kaydedilir ve JSON dosyası tekrar okunur.
"""
import csv
import json
from pathlib import Path
from typing import Any


def ensure_parent_directory(path: str | Path) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    return file_path


def save_json(
    data: Any,
    path: str | Path,
    indent: int = 2
) -> None:
    file_path = ensure_parent_directory(path)

    with file_path.open('w', encoding='utf-8') as file:
        json.dump(
            data, 
            file,
            indent=indent,
            ensure_ascii=False
        )


def load_json(path: str | Path) -> Any:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f'JSON file not found')
    
    with file_path.open('r', encoding='utf-8') as file:
        return json.load(file)
    

def save_jsonl(
    records: list[dict[str, Any]],
    path: str | Path
) -> None:
    file_path = ensure_parent_directory(path)

    with file_path.open('w', encoding='utf-8') as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + '\n'
            )





def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load records from a JSON Lines file."""
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    records = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.strip()

            if not stripped_line:
                continue

            try:
                records.append(json.loads(stripped_line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {file_path}"
                ) from error

    return records


def save_csv(
    rows: list[dict[str, Any]],
    path: str | Path,
) -> None:
    """Save a list of dictionaries as a CSV file."""
    if not rows:
        raise ValueError("Cannot save an empty list to CSV.")

    file_path = ensure_parent_directory(path)
    fieldnames = list(rows[0].keys())

    with file_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    example_predictions = [
        {
            "question_id": "example-1",
            "answer": "Albert Einstein",
            "confidence": 0.91,
            "decision": "ANSWER",
        }
    ]

    save_json(
        example_predictions,
        "outputs/predictions/example.json",
    )

    save_jsonl(
        example_predictions,
        "outputs/predictions/example.jsonl",
    )

    save_csv(
        example_predictions,
        "outputs/tables/example.csv",
    )

    loaded_predictions = load_json(
        "outputs/predictions/example.json"
    )

    print(loaded_predictions)