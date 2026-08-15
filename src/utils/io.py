"""
Provide shared file input/output utilities for the project.

These helpers centralize the way experiment artifacts are read and written,
so individual modules do not implement their own file-handling logic.

The module supports:

- JSON: one complete Python object stored in a file
- JSONL (JSON Lines): one JSON record per line, used mainly for predictions
- CSV: tabular experiment summaries

Parent directories are created automatically before writing files.
All text files use UTF-8 so Unicode characters are preserved consistently.
"""

import csv
import json
from pathlib import Path
from typing import Any


def ensure_parent_directory(path: str | Path) -> Path:
    """
    Ensure that the parent directory of a target file exists.

    This prevents save operations from failing when an output directory has
    not been created yet. The original file path is returned as a Path object
    so callers can immediately use it for writing.
    """

    file_path = Path(path)
    parent_directory = file_path.parent

    parent_directory.mkdir(parents=True, exist_ok=True)

    return file_path


def save_json(data: Any, path: str | Path, indent: int = 4) -> None:
    """
    Save a Python object as human-readable UTF-8 JSON.

    JSON is used for structured experiment metadata and summary results where
    the entire object is naturally stored and loaded as one document.
    """

    file_path = ensure_parent_directory(path)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            # Controls the number of spaces used for indentation, making JSON
            indent=indent,
            # Preserves Unicode characters (e.g., Ş, ə, ğ) instead of converting them to escape sequences.
            ensure_ascii=False,
        )


def load_json(path: str | Path) -> Any:
    """
    Load a UTF-8 JSON file and return the decoded Python object.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    """
    Save prediction records in JSON Lines format.

    JSONL (satır-bazlı JSON) stores exactly one JSON object per line. This is
    useful for experiment predictions because each example remains an
    independent record and large files can be processed line by line.
    """

    file_path = ensure_parent_directory(path)

    with file_path.open("w", encoding="utf-8") as file:
        for record in records:
            json_line = json.dumps(record, ensure_ascii=False)
            file.write(json_line + "\n")


def load_jsonl(
    path: str | Path,
) -> list[dict[str, Any]]:
    """
    Load a JSONL file as a list of prediction dictionaries.

    Blank lines are ignored. Each non-empty line must contain one valid JSON
    object. Invalid JSON reports its exact line number so corrupted experiment
    artifacts can be diagnosed quickly.
    """

    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    records: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            stripped_line = line.strip()

            if not stripped_line:
                continue

            # Parse each line independently so malformed records can be reported with
            # their exact line number.
            try:
                record = json.loads(stripped_line)

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise TypeError(f"Line {line_number} must contain a JSON object.")

            records.append(record)

    if not records:
        raise ValueError(f"No records were found in {input_path}")

    return records



def save_csv(
    rows: list[dict[str, Any]],
    path: str | Path,
) -> None:
    """
    Save a list of dictionaries as a UTF-8 CSV table.

    Dictionary keys from the first row define the CSV columns. CSV is mainly
    used for experiment summaries that are easier to inspect as tables.
    """

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

        # Write the header first, followed by one table row per dictionary.
        writer.writeheader()
        writer.writerows(rows)
