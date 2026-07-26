"""
Prepare high-entailment incorrect predictions for qualitative error analysis.

The script reads:

outputs/analysis/question_aware_verifier/high_entailment_incorrect.jsonl

and produces:

outputs/analysis/high_entailment_investigation/
├── high_entailment_cases.csv
├── high_entailment_cases.md
├── annotation_template.csv
└── summary.json

The annotation template intentionally leaves error-category fields empty.
Researchers should fill them manually after inspecting each case.

Suggested manual categories
---------------------------
- RETRIEVAL_ERROR
- QA_PREDICTION_ERROR
- CLAIM_GENERATION_ERROR
- NLI_ERROR
- ANNOTATION_AMBIGUITY
- EVIDENCE_SUPPORTS_WRONG_ANSWER
- MULTIPLE_FAILURES
- OTHER
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

DEFAULT_INPUT_PATH = Path(
    "outputs/analysis/question_aware_verifier/high_entailment_incorrect.jsonl"
)

DEFAULT_OUTPUT_DIRECTORY = Path("outputs/analysis/high_entailment_investigation")


QUESTION_FIELDS = (
    "question",
    "query",
    "input_question",
)

PREDICTION_FIELDS = (
    "predicted_answer",
    "prediction_text",
    "prediction_answer",
    "prediction",
    "model_answer",
    "generated_answer",
    "answer",
)

REFERENCE_FIELDS = (
    "reference_answers",
    "gold_answers",
    "accepted_answers",
    "references",
    "reference_answer",
    "gold_answer",
    "target_answer",
    "ground_truth",
    "answers",
    "answer_texts",
    "gold",
    "annotations",
    "targets",
    "labels",
)

CONFIDENCE_FIELDS = (
    "confidence",
    "calibrated_confidence",
    "temperature_scaled_confidence",
    "qa_calibrated_confidence",
    "qa_confidence",
    "prediction_confidence",
    "max_probability",
    "probability",
)

ENTAILMENT_FIELDS = (
    "entailment_probability",
    "qa_entailment_probability",
    "question_aware_entailment_probability",
    "question_aware_semantic_score",
    "qa_semantic_score",
)

CONTRADICTION_FIELDS = (
    "contradiction_probability",
    "qa_contradiction_probability",
    "question_aware_contradiction_probability",
)

NEUTRAL_FIELDS = (
    "neutral_probability",
    "qa_neutral_probability",
    "question_aware_neutral_probability",
)

LABEL_FIELDS = (
    "nli_label",
    "qa_nli_label",
    "question_aware_nli_label",
    "qa_verification_label",
    "semantic_label",
)

CLAIM_FIELDS = (
    "claim",
    "qa_claim",
    "generated_claim",
    "question_aware_claim",
)

EVIDENCE_FIELDS = (
    "evidence",
    "hybrid_evidence",
    "retrieved_evidence",
    "evidence_text",
    "context",
    "passage",
    "supporting_evidence",
)

CLAIM_VALIDITY_FIELDS = (
    "claim_valid",
    "qa_claim_valid",
    "question_aware_claim_valid",
)

INVALID_REASON_FIELDS = (
    "invalid_claim_reasons",
    "qa_invalid_claim_reasons",
    "claim_invalid_reasons",
    "qa_claim_invalid_reasons",
)

RETRIEVER_SCORE_FIELDS = (
    "retriever_score",
    "retrieval_score",
    "hybrid_retrieval_score",
    "dense_score",
    "reranker_score",
    "evidence_score",
)

SOURCE_FIELDS = (
    "source",
    "document_id",
    "doc_id",
    "passage_id",
    "evidence_source",
)

CORRECTNESS_FIELDS = (
    "correct",
    "is_correct",
    "exact_match",
    "em",
    "prediction_correct",
    "is_exact_match",
    "exact_match_score",
    "em_score",
    "answer_correct",
    "prediction_is_correct",
    "qa_is_correct",
)

ANSWERABLE_FIELDS = (
    "is_answerable",
    "answerable",
    "has_answer",
)

OUTPUT_COLUMNS = (
    "case_id",
    "record_index",
    "question",
    "gold_answers",
    "is_answerable",
    "prediction",
    "correct",
    "confidence",
    "entailment_probability",
    "contradiction_probability",
    "neutral_probability",
    "nli_label",
    "claim",
    "evidence",
    "retriever_score",
    "evidence_source",
    "claim_valid",
    "invalid_claim_reasons",
)


ANNOTATION_COLUMNS = (
    *OUTPUT_COLUMNS,
    "primary_error_category",
    "secondary_error_category",
    "retrieval_relevant",
    "evidence_supports_prediction",
    "claim_preserves_answer",
    "claim_preserves_question_meaning",
    "nli_label_reasonable",
    "gold_annotation_clear",
    "notes",
)


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    return re.sub(r"\s+", " ", str(value)).strip()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    records: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                record = json.loads(stripped)
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


def save_json(data: Any, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            data,
            output_file,
            indent=2,
            ensure_ascii=False,
        )


def save_csv(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
    fieldnames: Sequence[str],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def get_first_value(
    record: dict[str, Any],
    field_names: Sequence[str],
    default: Any = None,
) -> Any:
    for field_name in field_names:
        if field_name in record and record[field_name] is not None:
            return record[field_name]

    return default


def recursive_field_search(
    value: Any,
    field_names: Sequence[str],
) -> Any:
    """
    Recursively search dictionaries and lists for a matching field.

    Top-level and nearer fields are preferred over deeper fields.
    """

    if isinstance(value, dict):
        for field_name in field_names:
            if field_name in value:
                return value[field_name]

        for nested_value in value.values():
            result = recursive_field_search(
                nested_value,
                field_names,
            )

            if result is not None:
                return result

    elif isinstance(value, (list, tuple)):
        for item in value:
            if not isinstance(item, (dict, list, tuple)):
                continue

            result = recursive_field_search(
                item,
                field_names,
            )

            if result is not None:
                return result

    return None


def extract_nested_value(
    record: dict[str, Any],
    field_names: Sequence[str],
    nested_containers: Sequence[str] = (
        "source_record",
        "metadata",
        "verification",
        "semantic_verification",
        "question_aware_verification",
        "retrieval",
        "example",
        "sample",
        "data",
    ),
    default: Any = None,
) -> Any:
    """
    Extract a field from the current record.

    Search order:
    1. Top-level fields
    2. Known nested containers
    3. Fully recursive fallback
    """

    direct_value = get_first_value(
        record,
        field_names,
        default=None,
    )

    if direct_value is not None:
        return direct_value

    for container_name in nested_containers:
        container = record.get(container_name)

        if not isinstance(container, (dict, list, tuple)):
            continue

        nested_value = recursive_field_search(
            container,
            field_names,
        )

        if nested_value is not None:
            return nested_value

    recursive_value = recursive_field_search(
        record,
        field_names,
    )

    if recursive_value is not None:
        return recursive_value

    return default


def flatten_references(value: Any) -> list[str]:
    """
    Convert different reference-answer structures into a flat text list.
    """

    if value is None:
        return []

    if isinstance(value, str):
        cleaned = clean_text(value)
        return [cleaned] if cleaned else []

    if isinstance(value, (int, float, bool)):
        return [clean_text(value)]

    if isinstance(value, dict):
        preferred_text_fields = (
            "text",
            "answer",
            "answers",
            "answer_text",
            "answer_texts",
            "reference_answer",
            "reference_answers",
            "gold_answer",
            "gold_answers",
            "value",
            "label",
        )

        for key in preferred_text_fields:
            if key in value:
                extracted = flatten_references(value[key])

                if extracted:
                    return extracted

        # Annotation-style structures may contain nested answer objects.
        collected: list[str] = []

        for key, nested_value in value.items():
            # Avoid treating offsets, IDs and scores as answers.
            if key.lower() in {
                "answer_start",
                "start",
                "end",
                "offset",
                "offsets",
                "score",
                "probability",
                "id",
            }:
                continue

            if isinstance(
                nested_value,
                (dict, list, tuple),
            ):
                collected.extend(flatten_references(nested_value))

        return list(dict.fromkeys(collected))

    if isinstance(value, (list, tuple, set)):
        references: list[str] = []

        for item in value:
            references.extend(flatten_references(item))

        return list(dict.fromkeys(reference for reference in references if reference))

    return []


def stringify_list(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return clean_text(value)

    if isinstance(value, dict):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )

    if isinstance(value, Iterable):
        cleaned = [clean_text(item) for item in value if clean_text(item)]
        return " | ".join(cleaned)

    return clean_text(value)


def coerce_boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        if value == 1:
            return True
        if value == 0:
            return False

    if isinstance(value, float):
        if value == 1.0:
            return True
        if value == 0.0:
            return False

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {
            "true",
            "yes",
            "correct",
            "1",
            "1.0",
            "valid",
        }:
            return True

        if normalized in {
            "false",
            "no",
            "incorrect",
            "0",
            "0.0",
            "invalid",
        }:
            return False

    return None


def format_boolean(value: Any) -> str:
    boolean = coerce_boolean(value)

    if boolean is True:
        return "true"

    if boolean is False:
        return "false"

    return ""


def format_number(value: Any) -> str:
    if value is None or value == "":
        return ""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return clean_text(value)

    return f"{number:.6f}"


def normalise_record(
    record: dict[str, Any],
    case_id: int,
) -> dict[str, Any]:
    references = flatten_references(
        extract_nested_value(
            record,
            REFERENCE_FIELDS,
            default=[],
        )
    )

    is_answerable = coerce_boolean(
        extract_nested_value(
            record,
            ANSWERABLE_FIELDS,
            default=None,
        )
    )

    if not references and is_answerable is False:
        gold_answers = "[UNANSWERABLE]"
    else:
        gold_answers = " | ".join(references)

    invalid_reasons = extract_nested_value(
        record,
        INVALID_REASON_FIELDS,
        default=[],
    )

    row = {
        "case_id": case_id,
        "record_index": record.get(
            "index",
            record.get(
                "_analysis_index",
                record.get("record_index", ""),
            ),
        ),
        "question": clean_text(
            extract_nested_value(
                record,
                QUESTION_FIELDS,
                default="",
            )
        ),
        "gold_answers": gold_answers,
        "is_answerable": (format_boolean(is_answerable)),
        "prediction": clean_text(
            extract_nested_value(
                record,
                PREDICTION_FIELDS,
                default="",
            )
        ),
        "correct": format_boolean(
            extract_nested_value(
                record,
                CORRECTNESS_FIELDS,
                default=False,
            )
        ),
        "confidence": format_number(
            extract_nested_value(
                record,
                CONFIDENCE_FIELDS,
                default=None,
            )
        ),
        "entailment_probability": format_number(
            extract_nested_value(
                record,
                ENTAILMENT_FIELDS,
                default=None,
            )
        ),
        "contradiction_probability": format_number(
            extract_nested_value(
                record,
                CONTRADICTION_FIELDS,
                default=None,
            )
        ),
        "neutral_probability": format_number(
            extract_nested_value(
                record,
                NEUTRAL_FIELDS,
                default=None,
            )
        ),
        "nli_label": clean_text(
            extract_nested_value(
                record,
                LABEL_FIELDS,
                default="",
            )
        ).upper(),
        "claim": clean_text(
            extract_nested_value(
                record,
                CLAIM_FIELDS,
                default="",
            )
        ),
        "evidence": clean_text(
            extract_nested_value(
                record,
                EVIDENCE_FIELDS,
                default="",
            )
        ),
        "retriever_score": format_number(
            extract_nested_value(
                record,
                RETRIEVER_SCORE_FIELDS,
                default=None,
            )
        ),
        "evidence_source": clean_text(
            extract_nested_value(
                record,
                SOURCE_FIELDS,
                default="",
            )
        ),
        "claim_valid": format_boolean(
            extract_nested_value(
                record,
                CLAIM_VALIDITY_FIELDS,
                default=None,
            )
        ),
        "invalid_claim_reasons": stringify_list(invalid_reasons),
    }

    return row


def escape_markdown(value: Any) -> str:
    text = clean_text(value)

    text = text.replace("\\", "\\\\")
    text = text.replace("|", "\\|")
    text = text.replace("\n", "<br>")

    return text


def truncate_text(
    value: str,
    maximum_characters: int,
) -> str:
    text = clean_text(value)

    if maximum_characters <= 0:
        return text

    if len(text) <= maximum_characters:
        return text

    return text[: maximum_characters - 1].rstrip() + "…"


def render_markdown(
    rows: Sequence[dict[str, Any]],
    maximum_evidence_characters: int,
) -> str:
    lines: list[str] = [
        "# High-Entailment Incorrect Predictions",
        "",
        (
            "These cases were automatically selected because the QA "
            "prediction was incorrect while the question-aware verifier "
            "assigned high entailment. Error categories must be assigned "
            "manually."
        ),
        "",
        "## Annotation categories",
        "",
        "- `RETRIEVAL_ERROR`",
        "- `QA_PREDICTION_ERROR`",
        "- `CLAIM_GENERATION_ERROR`",
        "- `NLI_ERROR`",
        "- `ANNOTATION_AMBIGUITY`",
        "- `EVIDENCE_SUPPORTS_WRONG_ANSWER`",
        "- `MULTIPLE_FAILURES`",
        "- `OTHER`",
        "",
    ]

    for row in rows:
        lines.extend(
            [
                f"## Case {row['case_id']}",
                "",
                "| Field | Value |",
                "|---|---|",
                (f"| Record index | {escape_markdown(row['record_index'])} |"),
                (f"| Question | {escape_markdown(row['question'])} |"),
                (f"| Gold answer(s) | {escape_markdown(row['gold_answers'])} |"),
                (f"| Answerable | {escape_markdown(row['is_answerable'])} |"),
                (f"| Prediction | {escape_markdown(row['prediction'])} |"),
                (f"| Confidence | {escape_markdown(row['confidence'])} |"),
                (
                    "| Entailment probability | "
                    f"{escape_markdown(row['entailment_probability'])} |"
                ),
                (
                    "| Contradiction probability | "
                    f"{escape_markdown(row['contradiction_probability'])} |"
                ),
                (
                    "| Neutral probability | "
                    f"{escape_markdown(row['neutral_probability'])} |"
                ),
                (f"| NLI label | {escape_markdown(row['nli_label'])} |"),
                (f"| Claim | {escape_markdown(row['claim'])} |"),
                (
                    "| Evidence | "
                    f"{escape_markdown(truncate_text(row['evidence'], maximum_evidence_characters))} |"
                ),
                (f"| Retriever score | {escape_markdown(row['retriever_score'])} |"),
                (f"| Evidence source | {escape_markdown(row['evidence_source'])} |"),
                (f"| Claim valid | {escape_markdown(row['claim_valid'])} |"),
                (
                    "| Invalid reasons | "
                    f"{escape_markdown(row['invalid_claim_reasons'])} |"
                ),
                "| Primary error category |  |",
                "| Secondary error category |  |",
                "| Notes |  |",
                "",
            ]
        )

    return "\n".join(lines)


def build_summary(
    rows: Sequence[dict[str, Any]],
    input_path: str | Path,
) -> dict[str, Any]:
    label_counts = Counter(row["nli_label"] or "UNKNOWN" for row in rows)

    valid_counts = Counter(row["claim_valid"] or "UNKNOWN" for row in rows)

    missing_fields = Counter()

    fields_to_check = (
        "question",
        "gold_answers",
        "prediction",
        "confidence",
        "entailment_probability",
        "claim",
        "evidence",
    )

    for row in rows:
        for field_name in fields_to_check:
            if not row[field_name]:
                missing_fields[field_name] += 1

    entailment_scores: list[float] = []
    confidence_scores: list[float] = []

    for row in rows:
        if row["entailment_probability"]:
            try:
                entailment_scores.append(float(row["entailment_probability"]))
            except ValueError:
                pass

        if row["confidence"]:
            try:
                confidence_scores.append(float(row["confidence"]))
            except ValueError:
                pass

    return {
        "input_path": str(input_path),
        "total_cases": len(rows),
        "nli_label_counts": dict(label_counts),
        "claim_validity_counts": dict(valid_counts),
        "missing_field_counts": dict(missing_fields),
        "mean_entailment_probability": (
            sum(entailment_scores) / len(entailment_scores)
            if entailment_scores
            else None
        ),
        "mean_confidence": (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores
            else None
        ),
        "manual_annotation_categories": [
            "RETRIEVAL_ERROR",
            "QA_PREDICTION_ERROR",
            "CLAIM_GENERATION_ERROR",
            "NLI_ERROR",
            "ANNOTATION_AMBIGUITY",
            "EVIDENCE_SUPPORTS_WRONG_ANSWER",
            "MULTIPLE_FAILURES",
            "OTHER",
        ],
    }


def investigate_high_entailment_errors(
    input_path: str | Path,
    output_directory: str | Path,
    maximum_evidence_characters: int,
) -> None:
    records = load_jsonl(input_path)

    rows = [
        normalise_record(
            record,
            case_id=index,
        )
        for index, record in enumerate(records, start=1)
    ]

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    save_csv(
        rows,
        output_path / "high_entailment_cases.csv",
        fieldnames=OUTPUT_COLUMNS,
    )

    annotation_rows = []

    for row in rows:
        annotation_row = dict(row)

        annotation_row.update(
            {
                "primary_error_category": "",
                "secondary_error_category": "",
                "retrieval_relevant": "",
                "evidence_supports_prediction": "",
                "claim_preserves_answer": "",
                "claim_preserves_question_meaning": "",
                "nli_label_reasonable": "",
                "gold_annotation_clear": "",
                "notes": "",
            }
        )

        annotation_rows.append(annotation_row)

    save_csv(
        annotation_rows,
        output_path / "annotation_template.csv",
        fieldnames=ANNOTATION_COLUMNS,
    )

    markdown = render_markdown(
        rows,
        maximum_evidence_characters=maximum_evidence_characters,
    )

    with (output_path / "high_entailment_cases.md").open(
        "w", encoding="utf-8"
    ) as markdown_file:
        markdown_file.write(markdown)

    summary = build_summary(
        rows,
        input_path=input_path,
    )

    save_json(
        summary,
        output_path / "summary.json",
    )

    print("\n" + "=" * 92)
    print("HIGH-ENTAILMENT ERROR INVESTIGATION")
    print("=" * 92)

    print(f"Input cases: {len(rows)}")

    print("\nNLI labels:")
    for label, count in sorted(summary["nli_label_counts"].items()):
        print(f"  {label:<20} {count}")

    print("\nClaim validity:")
    for validity, count in sorted(summary["claim_validity_counts"].items()):
        print(f"  {validity:<20} {count}")

    print("\nMissing fields:")
    if summary["missing_field_counts"]:
        for field_name, count in sorted(summary["missing_field_counts"].items()):
            print(f"  {field_name:<28} {count}")
    else:
        print("  None")

    print("\n" + "-" * 92)
    print("OUTPUT FILES")
    print("-" * 92)

    for filename in (
        "high_entailment_cases.csv",
        "high_entailment_cases.md",
        "annotation_template.csv",
        "summary.json",
    ):
        print(output_path / filename)


def non_negative_integer(value: str) -> int:
    number = int(value)

    if number < 0:
        raise argparse.ArgumentTypeError("Value must be zero or greater.")

    return number


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare high-entailment incorrect predictions for "
            "manual qualitative error analysis."
        )
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help=("Input JSONL file containing high-entailment incorrect predictions."),
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIRECTORY),
        help="Directory where analysis files will be written.",
    )

    parser.add_argument(
        "--max-evidence-characters",
        type=non_negative_integer,
        default=2500,
        help=(
            "Maximum evidence length in the Markdown report. "
            "Use 0 to disable truncation. CSV output is never truncated."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    investigate_high_entailment_errors(
        input_path=arguments.input,
        output_directory=arguments.output_dir,
        maximum_evidence_characters=(arguments.max_evidence_characters),
    )
