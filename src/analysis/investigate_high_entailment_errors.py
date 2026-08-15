"""
Prepare high-entailment verifier failures for manual qualitative analysis.

This module consumes the `high_entailment_incorrect.jsonl` artifact produced by
`analyze_question_aware_verifier_errors.py`.

Those examples are interesting because:

- the underlying forced-answer QA candidate is incorrect;
- the question-aware NLI verifier nevertheless assigns high entailment.

The goal here is not to calculate another performance metric. Instead, this
script transforms those automatically selected cases into human-readable files
for qualitative diagnosis.

Generated outputs include:

- a compact CSV containing the main diagnostic fields;
- a Markdown report for manual reading;
- an annotation-template CSV for assigning failure categories;
- a JSON summary describing the selected case set.

Possible manual categories include retrieval errors, QA prediction errors,
claim-generation errors, NLI errors, annotation ambiguity, and cases where the
evidence genuinely supports a wrong answer.

This is a post-hoc diagnostic tool. Manual annotations created here must not be
used to tune parameters on the held-out test set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from src.utils.io import (
    load_jsonl,
    save_json,
)

DEFAULT_INPUT_PATH = Path(
    "outputs/analysis/question_aware_verifier/"
    "high_entailment_incorrect.jsonl"
)

DEFAULT_OUTPUT_DIRECTORY = Path(
    "outputs/analysis/high_entailment_investigation"
)


# Historical artifacts have used several names for the same conceptual field.
# These alias groups keep the qualitative-analysis script compatible with old
# and new diagnostic outputs without changing their meaning.
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


# These are the fields needed to understand one high-entailment failure without
# carrying the complete raw prediction artifact into the annotation table.
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


# The annotation template adds human judgments that help separate verifier
# failures from upstream QA, retrieval, claim-generation, or annotation issues.
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


MANUAL_ANNOTATION_CATEGORIES = (
    "RETRIEVAL_ERROR",
    "QA_PREDICTION_ERROR",
    "CLAIM_GENERATION_ERROR",
    "NLI_ERROR",
    "ANNOTATION_AMBIGUITY",
    "EVIDENCE_SUPPORTS_WRONG_ANSWER",
    "MULTIPLE_FAILURES",
    "OTHER",
)


def clean_text(
    value: Any,
) -> str:
    """Collapse whitespace while preserving the original textual content."""

    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def save_csv(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
    fieldnames: Sequence[str],
) -> None:
    """
    Save an analysis table with a fixed schema.

    Stable columns make the manually annotated files easier to compare across
    reruns of the diagnostic pipeline.
    """

    output_path = Path(
        path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(
                fieldnames
            ),
            extrasaction="ignore",
            lineterminator="\n",
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def get_first_value(
    record: dict[str, Any],
    field_names: Sequence[str],
    default: Any = None,
) -> Any:
    """Return the first available non-None value from ordered aliases."""

    for field_name in field_names:
        if (
            field_name in record
            and record[
                field_name
            ]
            is not None
        ):
            return record[
                field_name
            ]

    return default


def recursive_field_search(
    value: Any,
    field_names: Sequence[str],
) -> Any:
    """
    Search nested historical artifact structures for one requested field.

    The upstream diagnostic records can contain an embedded `source_record`.
    Recursive lookup allows this investigation script to recover useful context
    without depending on one exact historical JSON schema.
    """

    if isinstance(
        value,
        dict,
    ):
        for field_name in field_names:
            if field_name in value:
                return value[
                    field_name
                ]

        for nested_value in value.values():
            result = (
                recursive_field_search(
                    nested_value,
                    field_names,
                )
            )

            if result is not None:
                return result

    elif isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        for item in value:
            if not isinstance(
                item,
                (
                    dict,
                    list,
                    tuple,
                ),
            ):
                continue

            result = (
                recursive_field_search(
                    item,
                    field_names,
                )
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
    Retrieve a diagnostic field from direct or nested representations.

    Direct fields are preferred because the compact error-analysis artifact
    contains the values actually used by the upstream analysis. Nested source
    data is only a compatibility fallback.
    """

    direct_value = (
        get_first_value(
            record,
            field_names,
            default=None,
        )
    )

    if direct_value is not None:
        return direct_value

    for container_name in nested_containers:
        container = record.get(
            container_name
        )

        if not isinstance(
            container,
            (
                dict,
                list,
                tuple,
            ),
        ):
            continue

        nested_value = (
            recursive_field_search(
                container,
                field_names,
            )
        )

        if nested_value is not None:
            return nested_value

    recursive_value = (
        recursive_field_search(
            record,
            field_names,
        )
    )

    if recursive_value is not None:
        return recursive_value

    return default


def flatten_references(
    value: Any,
) -> list[str]:
    """
    Extract human-readable gold answers from common annotation structures.

    Numeric offsets, IDs, scores, and other metadata are intentionally ignored
    so they cannot accidentally appear as reference answers in the qualitative
    report.
    """

    if value is None:
        return []

    if isinstance(
        value,
        str,
    ):
        cleaned = clean_text(
            value
        )

        return (
            [
                cleaned
            ]
            if cleaned
            else []
        )

    if isinstance(
        value,
        dict,
    ):
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
            if key not in value:
                continue

            extracted = (
                flatten_references(
                    value[
                        key
                    ]
                )
            )

            if extracted:
                return extracted

        collected: list[
            str
        ] = []

        ignored_metadata_fields = {
            "answer_start",
            "start",
            "end",
            "offset",
            "offsets",
            "score",
            "probability",
            "id",
        }

        for (
            key,
            nested_value,
        ) in value.items():
            if (
                key.lower()
                in ignored_metadata_fields
            ):
                continue

            if isinstance(
                nested_value,
                (
                    dict,
                    list,
                    tuple,
                ),
            ):
                collected.extend(
                    flatten_references(
                        nested_value
                    )
                )

        return list(
            dict.fromkeys(
                collected
            )
        )

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        references: list[
            str
        ] = []

        for item in value:
            references.extend(
                flatten_references(
                    item
                )
            )

        return list(
            dict.fromkeys(
                reference
                for reference
                in references
                if reference
            )
        )

    return []


def stringify_list(
    value: Any,
) -> str:
    """
    Convert variable-format metadata into one readable annotation-table field.
    """

    if value is None:
        return ""

    if isinstance(
        value,
        str,
    ):
        return clean_text(
            value
        )

    if isinstance(
        value,
        dict,
    ):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )

    if isinstance(
        value,
        Iterable,
    ):
        cleaned = [
            clean_text(
                item
            )
            for item in value
            if clean_text(
                item
            )
        ]

        return " | ".join(
            cleaned
        )

    return clean_text(
        value
    )


def coerce_boolean(
    value: Any,
) -> bool | None:
    """Interpret common Boolean representations found in stored artifacts."""

    if isinstance(
        value,
        bool,
    ):
        return value

    if (
        isinstance(
            value,
            int,
        )
        and not isinstance(
            value,
            bool,
        )
    ):
        if value == 1:
            return True

        if value == 0:
            return False

    if isinstance(
        value,
        float,
    ):
        if value == 1.0:
            return True

        if value == 0.0:
            return False

    if isinstance(
        value,
        str,
    ):
        normalized = (
            value
            .strip()
            .lower()
        )

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


def format_boolean(
    value: Any,
) -> str:
    """
    Store optional Boolean diagnostics as stable lowercase CSV values.
    """

    boolean = coerce_boolean(
        value
    )

    if boolean is True:
        return "true"

    if boolean is False:
        return "false"

    return ""


def format_number(
    value: Any,
) -> str:
    """
    Format optional numeric diagnostics consistently for manual inspection.

    Non-numeric values are preserved as cleaned text rather than silently
    discarded because the investigation report should expose malformed data.
    """

    if (
        value is None
        or value == ""
    ):
        return ""

    try:
        number = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return clean_text(
            value
        )

    if not math.isfinite(
        number
    ):
        return clean_text(
            value
        )

    return f"{number:.6f}"


def normalise_record(
    record: dict[str, Any],
    case_id: int,
) -> dict[str, Any]:
    """
    Convert one selected verifier failure into a stable annotation row.

    The row combines QA information, NLI signals, generated claim, evidence,
    retrieval metadata, and claim-validity diagnostics in one human-readable
    representation.
    """

    references = (
        flatten_references(
            extract_nested_value(
                record,
                REFERENCE_FIELDS,
                default=[],
            )
        )
    )

    is_answerable = (
        coerce_boolean(
            extract_nested_value(
                record,
                ANSWERABLE_FIELDS,
                default=None,
            )
        )
    )

    if (
        not references
        and is_answerable is False
    ):
        gold_answers = (
            "[UNANSWERABLE]"
        )

    else:
        gold_answers = (
            " | ".join(
                references
            )
        )

    invalid_reasons = (
        extract_nested_value(
            record,
            INVALID_REASON_FIELDS,
            default=[],
        )
    )

    return {
        "case_id": (
            case_id
        ),
        "record_index": record.get(
            "index",
            record.get(
                "_analysis_index",
                record.get(
                    "record_index",
                    "",
                ),
            ),
        ),
        "question": (
            clean_text(
                extract_nested_value(
                    record,
                    QUESTION_FIELDS,
                    default="",
                )
            )
        ),
        "gold_answers": (
            gold_answers
        ),
        "is_answerable": (
            format_boolean(
                is_answerable
            )
        ),
        "prediction": (
            clean_text(
                extract_nested_value(
                    record,
                    PREDICTION_FIELDS,
                    default="",
                )
            )
        ),
        "correct": (
            format_boolean(
                extract_nested_value(
                    record,
                    CORRECTNESS_FIELDS,
                    default=None,
                )
            )
        ),
        "confidence": (
            format_number(
                extract_nested_value(
                    record,
                    CONFIDENCE_FIELDS,
                    default=None,
                )
            )
        ),
        "entailment_probability": (
            format_number(
                extract_nested_value(
                    record,
                    ENTAILMENT_FIELDS,
                    default=None,
                )
            )
        ),
        "contradiction_probability": (
            format_number(
                extract_nested_value(
                    record,
                    CONTRADICTION_FIELDS,
                    default=None,
                )
            )
        ),
        "neutral_probability": (
            format_number(
                extract_nested_value(
                    record,
                    NEUTRAL_FIELDS,
                    default=None,
                )
            )
        ),
        "nli_label": (
            clean_text(
                extract_nested_value(
                    record,
                    LABEL_FIELDS,
                    default="",
                )
            ).upper()
        ),
        "claim": (
            clean_text(
                extract_nested_value(
                    record,
                    CLAIM_FIELDS,
                    default="",
                )
            )
        ),
        "evidence": (
            clean_text(
                extract_nested_value(
                    record,
                    EVIDENCE_FIELDS,
                    default="",
                )
            )
        ),
        "retriever_score": (
            format_number(
                extract_nested_value(
                    record,
                    RETRIEVER_SCORE_FIELDS,
                    default=None,
                )
            )
        ),
        "evidence_source": (
            clean_text(
                extract_nested_value(
                    record,
                    SOURCE_FIELDS,
                    default="",
                )
            )
        ),
        "claim_valid": (
            format_boolean(
                extract_nested_value(
                    record,
                    CLAIM_VALIDITY_FIELDS,
                    default=None,
                )
            )
        ),
        "invalid_claim_reasons": (
            stringify_list(
                invalid_reasons
            )
        ),
    }


def validate_investigation_rows(
    rows: Sequence[
        dict[str, Any]
    ],
) -> None:
    """
    Validate that the input still represents the intended diagnostic subset.

    This investigation is specifically for incorrect QA candidates selected for
    high verifier entailment. If an upstream artifact unexpectedly contains a
    record marked correct, the script stops instead of producing a misleading
    qualitative report.
    """

    if not rows:
        raise ValueError(
            "Investigation row list "
            "cannot be empty."
        )

    for row in rows:
        correctness = (
            row[
                "correct"
            ]
        )

        if correctness == "true":
            raise ValueError(
                "High-entailment investigation "
                "contains a record marked correct: "
                f"case_id={row['case_id']}."
            )

        if correctness == "":
            raise ValueError(
                "High-entailment investigation "
                "contains a record with unknown "
                "correctness: "
                f"case_id={row['case_id']}."
            )


def escape_markdown(
    value: Any,
) -> str:
    """Escape table-sensitive characters before inserting text into Markdown."""

    text = clean_text(
        value
    )

    text = text.replace(
        "\\",
        "\\\\",
    )

    text = text.replace(
        "|",
        "\\|",
    )

    return text


def truncate_text(
    value: str,
    maximum_characters: int,
) -> str:
    """
    Shorten evidence only in the Markdown view.

    CSV output keeps the complete evidence so qualitative inspection never
    loses the original text.
    """

    text = clean_text(
        value
    )

    if maximum_characters <= 0:
        return text

    if len(
        text
    ) <= maximum_characters:
        return text

    return (
        text[
            : maximum_characters - 1
        ].rstrip()
        + "…"
    )


def render_markdown(
    rows: Sequence[
        dict[str, Any]
    ],
    maximum_evidence_characters: int,
) -> str:
    """
    Render selected verifier failures as a manual-review document.

    The report intentionally leaves annotation fields empty. Failure categories
    must be assigned by inspection rather than inferred automatically from the
    same verifier signals being investigated.
    """

    lines: list[
        str
    ] = [
        "# High-Entailment Incorrect Predictions",
        "",
        (
            "These cases were automatically selected because the underlying "
            "forced-answer QA candidate was incorrect while the question-aware "
            "verifier assigned high entailment."
        ),
        "",
        (
            "The categories below are for qualitative diagnosis only. "
            "They must be assigned manually and are not used to tune the "
            "held-out evaluation."
        ),
        "",
        "## Annotation categories",
        "",
    ]

    lines.extend(
        f"- `{category}`"
        for category
        in MANUAL_ANNOTATION_CATEGORIES
    )

    lines.append(
        ""
    )

    for row in rows:
        evidence = (
            truncate_text(
                row[
                    "evidence"
                ],
                maximum_evidence_characters,
            )
        )

        lines.extend(
            [
                f"## Case {row['case_id']}",
                "",
                "| Field | Value |",
                "|---|---|",
                (
                    "| Record index | "
                    f"{escape_markdown(row['record_index'])} |"
                ),
                (
                    "| Question | "
                    f"{escape_markdown(row['question'])} |"
                ),
                (
                    "| Gold answer(s) | "
                    f"{escape_markdown(row['gold_answers'])} |"
                ),
                (
                    "| Answerable | "
                    f"{escape_markdown(row['is_answerable'])} |"
                ),
                (
                    "| Prediction | "
                    f"{escape_markdown(row['prediction'])} |"
                ),
                (
                    "| Confidence | "
                    f"{escape_markdown(row['confidence'])} |"
                ),
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
                (
                    "| NLI label | "
                    f"{escape_markdown(row['nli_label'])} |"
                ),
                (
                    "| Claim | "
                    f"{escape_markdown(row['claim'])} |"
                ),
                (
                    "| Evidence | "
                    f"{escape_markdown(evidence)} |"
                ),
                (
                    "| Retriever score | "
                    f"{escape_markdown(row['retriever_score'])} |"
                ),
                (
                    "| Evidence source | "
                    f"{escape_markdown(row['evidence_source'])} |"
                ),
                (
                    "| Claim valid | "
                    f"{escape_markdown(row['claim_valid'])} |"
                ),
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

    return "\n".join(
        lines
    )


def parse_optional_numeric_column(
    rows: Sequence[
        dict[str, Any]
    ],
    field_name: str,
) -> list[float]:
    """
    Collect finite numeric values from one diagnostic output column.

    Missing textual diagnostics are ignored in summary means rather than
    silently converted to zero.
    """

    values: list[
        float
    ] = []

    for row in rows:
        value = row[
            field_name
        ]

        if not value:
            continue

        try:
            number = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if math.isfinite(
            number
        ):
            values.append(
                number
            )

    return values


def build_summary(
    rows: Sequence[
        dict[str, Any]
    ],
    input_path: str | Path,
) -> dict[str, Any]:
    """
    Summarize the qualitative-review set before manual annotation.

    These statistics describe only the automatically selected high-entailment
    failure subset. They are not whole-dataset verifier performance metrics.
    """

    label_counts = Counter(
        row[
            "nli_label"
        ]
        or "UNKNOWN"
        for row in rows
    )

    valid_counts = Counter(
        row[
            "claim_valid"
        ]
        or "UNKNOWN"
        for row in rows
    )

    missing_fields: Counter[
        str
    ] = Counter()

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
            if not row[
                field_name
            ]:
                missing_fields[
                    field_name
                ] += 1

    entailment_scores = (
        parse_optional_numeric_column(
            rows,
            "entailment_probability",
        )
    )

    confidence_scores = (
        parse_optional_numeric_column(
            rows,
            "confidence",
        )
    )

    return {
        "analysis_type": (
            "high_entailment_incorrect_qualitative_investigation"
        ),
        "analysis_scope": (
            "post-hoc manual diagnostic subset; "
            "not used for parameter tuning"
        ),
        "input_path": (
            str(
                input_path
            )
        ),
        "total_cases": (
            len(
                rows
            )
        ),
        "nli_label_counts": dict(
            label_counts
        ),
        "claim_validity_counts": dict(
            valid_counts
        ),
        "missing_field_counts": dict(
            missing_fields
        ),
        "mean_entailment_probability": (
            sum(
                entailment_scores
            )
            / len(
                entailment_scores
            )
            if entailment_scores
            else None
        ),
        "mean_confidence": (
            sum(
                confidence_scores
            )
            / len(
                confidence_scores
            )
            if confidence_scores
            else None
        ),
        "manual_annotation_categories": list(
            MANUAL_ANNOTATION_CATEGORIES
        ),
    }


def build_annotation_rows(
    rows: Sequence[
        dict[str, Any]
    ],
) -> list[
    dict[str, Any]
]:
    """
    Add blank human-annotation fields to each automatically extracted case.

    Automated fields describe what the models produced; the added fields capture
    the researcher's independent qualitative judgment about where the failure
    occurred.
    """

    annotation_rows: list[
        dict[str, Any]
    ] = []

    for row in rows:
        annotation_row = dict(
            row
        )

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

        annotation_rows.append(
            annotation_row
        )

    return annotation_rows


def investigate_high_entailment_errors(
    input_path: str | Path,
    output_directory: str | Path,
    maximum_evidence_characters: int,
) -> None:
    """
    Generate the complete manual-review package for verifier failures.

    No verifier scores, model parameters, or evaluation thresholds are changed
    here. The function only reorganizes already selected diagnostic cases into
    formats that are easier to inspect and annotate.
    """

    records = load_jsonl(
        input_path
    )

    if not records:
        raise ValueError(
            "Input high-entailment "
            "case file is empty."
        )

    rows = [
        normalise_record(
            record,
            case_id=index,
        )
        for index, record
        in enumerate(
            records,
            start=1,
        )
    ]

    validate_investigation_rows(
        rows
    )

    output_path = Path(
        output_directory
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_csv(
        rows,
        output_path
        / "high_entailment_cases.csv",
        fieldnames=(
            OUTPUT_COLUMNS
        ),
    )

    annotation_rows = (
        build_annotation_rows(
            rows
        )
    )

    save_csv(
        annotation_rows,
        output_path
        / "annotation_template.csv",
        fieldnames=(
            ANNOTATION_COLUMNS
        ),
    )

    markdown = (
        render_markdown(
            rows,
            maximum_evidence_characters=(
                maximum_evidence_characters
            ),
        )
    )

    (
        output_path
        / "high_entailment_cases.md"
    ).write_text(
        markdown,
        encoding="utf-8",
    )

    summary = (
        build_summary(
            rows,
            input_path=(
                input_path
            ),
        )
    )

    save_json(
        summary,
        output_path
        / "summary.json",
    )

    print(
        "\n"
        + "=" * 92
    )

    print(
        "HIGH-ENTAILMENT ERROR INVESTIGATION"
    )

    print(
        "=" * 92
    )

    print(
        f"Input cases: "
        f"{len(rows)}"
    )

    print(
        "\nNLI labels:"
    )

    for (
        label,
        count,
    ) in sorted(
        summary[
            "nli_label_counts"
        ].items()
    ):
        print(
            f"  {label:<20} "
            f"{count}"
        )

    print(
        "\nClaim validity:"
    )

    for (
        validity,
        count,
    ) in sorted(
        summary[
            "claim_validity_counts"
        ].items()
    ):
        print(
            f"  {validity:<20} "
            f"{count}"
        )

    print(
        "\nMissing fields:"
    )

    if summary[
        "missing_field_counts"
    ]:
        for (
            field_name,
            count,
        ) in sorted(
            summary[
                "missing_field_counts"
            ].items()
        ):
            print(
                f"  {field_name:<28} "
                f"{count}"
            )

    else:
        print(
            "  None"
        )

    print(
        "\n"
        + "-" * 92
    )

    print(
        "OUTPUT FILES"
    )

    print(
        "-" * 92
    )

    for filename in (
        "high_entailment_cases.csv",
        "high_entailment_cases.md",
        "annotation_template.csv",
        "summary.json",
    ):
        print(
            output_path
            / filename
        )


def non_negative_integer(
    value: str,
) -> int:
    """
    Parse a non-negative integer.

    A value of zero disables Markdown evidence truncation.
    """

    number = int(
        value
    )

    if number < 0:
        raise argparse.ArgumentTypeError(
            "Value must be zero or greater."
        )

    return number


def parse_arguments() -> argparse.Namespace:
    """Parse paths and display settings for qualitative investigation."""

    parser = argparse.ArgumentParser(
        description=(
            "Prepare high-entailment incorrect "
            "question-aware verifier cases for "
            "manual qualitative analysis."
        )
    )

    parser.add_argument(
        "--input",
        default=str(
            DEFAULT_INPUT_PATH
        ),
        help=(
            "JSONL containing automatically "
            "selected high-entailment incorrect "
            "predictions."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIRECTORY
        ),
        help=(
            "Directory for qualitative-analysis "
            "CSV, Markdown, and JSON outputs."
        ),
    )

    parser.add_argument(
        "--max-evidence-characters",
        type=non_negative_integer,
        default=2500,
        help=(
            "Maximum evidence length in the "
            "Markdown report. Use 0 to disable "
            "truncation. CSV evidence is never "
            "truncated."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the qualitative high-entailment failure investigation."""

    arguments = (
        parse_arguments()
    )

    investigate_high_entailment_errors(
        input_path=(
            arguments.input
        ),
        output_directory=(
            arguments.output_dir
        ),
        maximum_evidence_characters=(
            arguments.max_evidence_characters
        ),
    )


if __name__ == "__main__":
    main()