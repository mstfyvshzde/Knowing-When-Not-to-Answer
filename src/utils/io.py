"""
This file manages input and output operations for the whole project.
In simple terms, it helps the project:
- create folders when needed
- save Python data as JSON
- load JSON files
- save multiple records as JSONL
- load JSONL records
- save table-like data as CSV
So instead of rewriting file-handling code in every part of the project, other files can simply import these helper functions.
"""

# Read and write CSV files.
import csv

# Convert Python objects to and from JSON.
import json

# Create and manage file paths safely across operating systems.
from pathlib import Path

# Allow a function to accept values of any data type.
from typing import Any


# Write a utility function that guarantees the parent directory exists before saving a file.
def ensure_parent_directory(path: str | Path) -> Path:
    # Converts a string path into a Path object with useful file methods.
    file_path = Path(path)
    main_directory = file_path.parent

    main_directory.mkdir(parents=True, exist_ok=True)

    return file_path


# Save a Python object as a JSON file.
def save_json(data: str | Any, path: str | Path, indent: int = 4) -> None:
    file_path = ensure_parent_directory(path)

    # Opens the file safely and automatically closes it afterwards.
    with file_path.open("w", encoding="utf-8") as file:
        # Writes a Python object into a JSON file.
        json.dump(
            data,
            file,
            # Controls the number of spaces used for indentation, making JSON
            indent=indent,
            # Preserves Unicode characters (e.g., Ş, ə, ğ) instead of converting them to escape sequences.
            ensure_ascii=False,
        )


# Load data from a JSON file and convert it into a Python object.
def load_json(path: str | Path) -> object:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        # Reads JSON content from an opened file and converts it into a Python object.
        data = json.load(file)

    return data


# Save multiple dictionaries in JSON Lines format.
def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    file_path = ensure_parent_directory(path)

    with file_path.open("w", encoding="utf-8") as file:
        for record in records:
            # converts a Python dictionary into a JSON string.
            json_line = json.dumps(record)
            file.write(json_line + "\n")


# This function reads a JSONL file and returns all JSON objects as Python dictionaries.
def load_jsonl(
    path: str | Path,
) -> list[dict[str, Any]]:
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    records: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            stripped_line = line.strip()

            if not stripped_line:
                continue

            # does not simply store the line. It converts a JSON string into a Python object
            try:
                record = json.loads(stripped_line)

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object.")

            records.append(record)

    if not records:
        raise ValueError(f"No records were found in {input_path}")

    return records


# It converts a list of dictionaries into a CSV table.
def save_csv(
    rows: list[dict[str, Any]],
    path: str | Path,
) -> None:

    if not rows:
        raise ValueError("Cannot save an empty list to CSV.")

    file_path = ensure_parent_directory(path)

    field_names = list(rows[0].keys())

    with file_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        # Creates a writer that converts dictionaries into CSV rows.
        writer = csv.DictWriter(
            file,
            field_names=field_names,
        )

        # Writes column labels, not actual data.
        writer.writeheader()

        # Writes all dictionaries as data rows.
        writer.writerows(rows)


# It tests whether fnctions work
if __name__ == "__main__":
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

    loaded_predictions = load_json("outputs/predictions/example.json")

    print("Loaded predictions:")
    print(loaded_predictions)
