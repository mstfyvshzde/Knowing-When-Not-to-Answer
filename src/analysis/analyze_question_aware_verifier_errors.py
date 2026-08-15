"""
Analyze failure modes of the question-aware NLI verifier.

The input is the V2 question-aware verifier output. This diagnostic analysis
measures how verifier signals relate to the correctness of forced-answer QA
candidates and extracts suspicious cases for manual inspection.

Reported diagnostics include correctness by NLI label and claim validity,
entailment-score bins, invalid-claim reasons, confidence/entailment
correlations, high/low disagreement groups, and a scatter plot.

Correctness is recomputed from prediction/reference text instead of trusting
stored correctness fields. In the project's forced-answer setup, unanswerable
examples are always incorrect candidates.

The high/low thresholds below are used only for qualitative example extraction.
They are not tuned decision thresholds and do not affect the final AURC results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.evaluation.metrics import normalize_answer, parse_answerability
from src.utils.io import load_jsonl, save_json

INPUT = Path(
    "outputs/predictions/"
    "calibration_with_question_aware_semantic_evidence_v2.jsonl"
)

OUT = Path(
    "outputs/analysis/question_aware_verifier"
)


# Only explicitly calibrated confidence aliases are accepted. Raw QA confidence
# is intentionally excluded because this analysis interprets the signal as a
# calibrated probability.
CONFIDENCE = (
    "calibrated_confidence",
    "temperature_scaled_confidence",
    "qa_calibrated_confidence",
    "confidence_calibrated",
    "calibrated_probability",
)

ENTAILMENT = (
    "qa_entailment_probability",
    "question_aware_entailment_probability",
    "question_aware_semantic_score",
    "qa_semantic_score",
)

LABEL = (
    "qa_nli_label",
    "question_aware_nli_label",
    "qa_verification_label",
    "semantic_label",
    "nli_label",
)

VALID = (
    "qa_claim_valid",
    "claim_valid",
    "question_aware_claim_valid",
)

REASONS = (
    "qa_invalid_claim_reasons",
    "invalid_claim_reasons",
    "claim_invalid_reasons",
    "qa_claim_invalid_reasons",
)

QUESTION = (
    "question",
    "query",
    "input_question",
)

PREDICTION = (
    "prediction_text",
    "predicted_answer",
    "prediction_answer",
    "prediction",
    "model_answer",
    "generated_answer",
    "answer",
)

REFERENCE = (
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
)

ANSWERABILITY = (
    "is_answerable",
    "answerable",
    "gold_is_answerable",
)

CLAIM = (
    "qa_claim",
    "generated_claim",
    "claim",
    "question_aware_claim",
)

EVIDENCE = (
    "hybrid_evidence",
    "evidence",
    "retrieved_evidence",
    "evidence_text",
    "context",
    "passage",
    "supporting_evidence",
)

CONTRADICTION = (
    "qa_contradiction_probability",
    "question_aware_contradiction_probability",
)

NEUTRAL = (
    "qa_neutral_probability",
    "question_aware_neutral_probability",
)

BINS = tuple(
    (
        index / 10,
        (index + 1) / 10,
    )
    for index in range(10)
)


def first(
    record: dict[str, Any],
    names: Sequence[str],
    default: Any = None,
) -> Any:
    """Return the first non-None value from ordered field aliases."""

    for name in names:
        if (
            name in record
            and record[name] is not None
        ):
            return record[name]

    return default


def search_nested_field(
    value: Any,
    candidate_fields: Sequence[str],
) -> Any:
    """
    Search nested dictionaries/lists for the first matching field.

    Historical artifacts used slightly different schemas, so this helper keeps
    the analysis tolerant of nested representations without changing which
    semantic signal is requested.
    """

    if isinstance(
        value,
        dict,
    ):
        for field_name in candidate_fields:
            if field_name in value:
                return value[
                    field_name
                ]

        for nested_value in value.values():
            result = search_nested_field(
                nested_value,
                candidate_fields,
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
            if isinstance(
                item,
                (
                    dict,
                    list,
                    tuple,
                ),
            ):
                result = search_nested_field(
                    item,
                    candidate_fields,
                )

                if result is not None:
                    return result

    return None


def first_or_nested(
    record: dict[str, Any],
    names: Sequence[str],
    default: Any = None,
) -> Any:
    """Prefer a direct field and fall back to a nested historical alias."""

    direct_value = first(
        record,
        names,
        default=None,
    )

    if direct_value is not None:
        return direct_value

    nested_value = search_nested_field(
        record,
        names,
    )

    if nested_value is not None:
        return nested_value

    return default


def to_finite_float(
    value: Any,
) -> float | None:
    """Convert a value to a finite float, returning None when unavailable."""

    try:
        number = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(
        number
    ):
        return None

    return number


def to_probability(
    value: Any,
    field_name: str,
) -> float | None:
    """
    Convert an optional verifier score to a validated probability.

    Out-of-range values raise instead of being silently clipped because clipping
    could hide malformed verifier artifacts and alter the diagnostics.
    """

    number = to_finite_float(
        value
    )

    if number is None:
        return None

    if not (
        0.0
        <= number
        <= 1.0
    ):
        raise ValueError(
            f"{field_name} must lie in "
            f"[0, 1], received {number}."
        )

    return number


def to_boolean(
    value: Any,
) -> bool | None:
    """Convert common Boolean representations; unknown values return None."""

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        (
            int,
            float,
        ),
    ):
        if value == 1:
            return True

        if value == 0:
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


def clean(
    value: Any,
) -> str:
    """Collapse repeated whitespace while preserving textual content."""

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def extract_reference_values(
    value: Any,
) -> list[str]:
    """
    Extract answer strings from common reference-answer structures.

    Unknown dictionaries are not traversed blindly so metadata such as answer
    offsets cannot become accidental gold-answer text.
    """

    if value is None:
        return []

    if isinstance(
        value,
        str,
    ):
        return [
            value
        ]

    if isinstance(
        value,
        dict,
    ):
        for key in (
            "text",
            "answer",
            "answers",
            "value",
            "label",
        ):
            if key in value:
                return extract_reference_values(
                    value[
                        key
                    ]
                )

        return []

    if isinstance(
        value,
        Iterable,
    ):
        output: list[
            str
        ] = []

        for item in value:
            output.extend(
                extract_reference_values(
                    item
                )
            )

        return output

    return [
        str(
            value
        )
    ]


def extract_reference_answers(
    record: dict[str, Any],
) -> list[str]:
    """Extract reference answers from direct or nested historical fields."""

    for field_name in REFERENCE:
        if field_name in record:
            return extract_reference_values(
                record[
                    field_name
                ]
            )

    nested_value = search_nested_field(
        record,
        REFERENCE,
    )

    return extract_reference_values(
        nested_value
    )


def infer_answerability(
    record: dict[str, Any],
    references: Sequence[str],
) -> bool:
    """
    Retrieve explicit answerability metadata when available.

    Reference presence is used only as a compatibility fallback for old
    diagnostic artifacts that predate an explicit `is_answerable` field.
    """

    raw_value = first_or_nested(
        record,
        ANSWERABILITY,
        default=None,
    )

    if raw_value is not None:
        try:
            return parse_answerability(
                raw_value
            )

        except (
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                "Invalid answerability at "
                f"record {record['_index']}: "
                f"{raw_value!r}."
            ) from error

    return any(
        normalize_answer(
            reference
        )
        for reference in references
    )


def infer_prediction_correctness(
    record: dict[str, Any],
) -> bool:
    """
    Recompute forced-answer Exact-Match correctness from raw QA fields.

    Stored `is_correct` or `exact_match` fields are deliberately ignored because
    they may belong to an older evaluation stage or another metric definition.

    Unanswerable examples are always incorrect forced-answer candidates.
    """

    prediction = first_or_nested(
        record,
        PREDICTION,
        default=None,
    )

    if prediction is None:
        raise ValueError(
            "No prediction field at "
            f"record {record['_index']}; "
            f"keys={sorted(record)}"
        )

    references = extract_reference_answers(
        record
    )

    is_answerable = infer_answerability(
        record,
        references,
    )

    normalized_references = [
        normalize_answer(
            reference
        )
        for reference in references
        if normalize_answer(
            reference
        )
    ]

    if not is_answerable:
        if normalized_references:
            raise ValueError(
                "Unanswerable record "
                f"{record['_index']} contains "
                "non-empty references."
            )

        return False

    if not normalized_references:
        raise ValueError(
            "Answerable record "
            f"{record['_index']} has no "
            "usable reference."
        )

    normalized_prediction = (
        normalize_answer(
            str(
                prediction
            )
        )
    )

    return (
        normalized_prediction
        in normalized_references
    )


def find_best_available_field(
    records: Sequence[
        dict[str, Any]
    ],
    names: Sequence[str],
    numeric: bool = False,
) -> str | None:
    """
    Choose the alias populated in the largest number of records.

    One field is selected for the whole analysis so one statistic does not mix
    multiple score definitions across examples.
    """

    best_field: str | None = None
    best_count = 0

    for name in names:
        if numeric:
            count = sum(
                to_finite_float(
                    record.get(
                        name
                    )
                )
                is not None
                for record
                in records
            )

        else:
            count = sum(
                name in record
                and record[
                    name
                ]
                is not None
                for record
                in records
            )

        if count > best_count:
            best_field = name
            best_count = count

    return best_field


def normalize_invalid_claim_reasons(
    value: Any,
) -> list[str]:
    """
    Normalize invalid-claim reason formats into stable uppercase labels.

    The verifier has historically stored reasons as strings, JSON-encoded
    strings, lists, or Boolean dictionaries.
    """

    if value is None:
        return []

    if isinstance(
        value,
        str,
    ):
        stripped = (
            value.strip()
        )

        if not stripped:
            return []

        try:
            decoded = json.loads(
                stripped
            )

            if decoded != value:
                return normalize_invalid_claim_reasons(
                    decoded
                )

        except json.JSONDecodeError:
            pass

        return [
            clean(
                item
            )
            .upper()
            .replace(
                " ",
                "_",
            )
            for item in re.split(
                r"[,;|]",
                stripped,
            )
            if clean(
                item
            )
        ]

    if isinstance(
        value,
        dict,
    ):
        return [
            clean(
                key
            )
            .upper()
            .replace(
                " ",
                "_",
            )
            for key, active
            in value.items()
            if to_boolean(
                active
            )
            is True
        ]

    if isinstance(
        value,
        Iterable,
    ):
        output: list[
            str
        ] = []

        for item in value:
            output.extend(
                normalize_invalid_claim_reasons(
                    item
                )
            )

        return output

    return [
        clean(
            value
        )
        .upper()
        .replace(
            " ",
            "_",
        )
    ]


def infer_nli_label(
    record: dict[str, Any],
    label_field: str | None,
    valid: bool,
    entailment: float | None,
) -> str:
    """
    Return the diagnostic NLI label for one verifier record.

    Invalid generated claims are separated from normal NLI outcomes. If no
    explicit label exists, the largest available NLI probability reconstructs
    the label for diagnostic purposes only.
    """

    if not valid:
        return "INVALID_CLAIM"

    raw_label = ""

    if label_field is not None:
        raw_label = clean(
            record.get(
                label_field,
                "",
            )
        ).upper()

    for label in (
        "ENTAILMENT",
        "CONTRADICTION",
        "NEUTRAL",
        "EMPTY_ANSWER",
    ):
        if label in raw_label:
            return label

    candidates: list[
        tuple[
            str,
            float,
        ]
    ] = []

    if entailment is not None:
        candidates.append(
            (
                "ENTAILMENT",
                entailment,
            )
        )

    contradiction = (
        to_probability(
            first_or_nested(
                record,
                CONTRADICTION,
                default=None,
            ),
            "contradiction probability",
        )
    )

    neutral = (
        to_probability(
            first_or_nested(
                record,
                NEUTRAL,
                default=None,
            ),
            "neutral probability",
        )
    )

    if contradiction is not None:
        candidates.append(
            (
                "CONTRADICTION",
                contradiction,
            )
        )

    if neutral is not None:
        candidates.append(
            (
                "NEUTRAL",
                neutral,
            )
        )

    if not candidates:
        return "UNKNOWN"

    return max(
        candidates,
        key=lambda item: (
            item[
                1
            ]
        ),
    )[
        0
    ]


def mean(
    values: Sequence[float],
) -> float | None:
    """Return the arithmetic mean, or None for an empty sequence."""

    if not values:
        return None

    return (
        sum(
            values
        )
        / len(
            values
        )
    )


def calculate_pearson_correlation(
    first_values: Sequence[float],
    second_values: Sequence[float],
) -> float | None:
    """
    Calculate Pearson correlation between paired score sequences.

    This is descriptive only: it measures whether calibrated QA confidence and
    NLI entailment move together, not whether either score is calibrated or
    causally related to the other.
    """

    if (
        len(
            first_values
        )
        != len(
            second_values
        )
    ):
        raise ValueError(
            "Correlation inputs must have "
            "equal length."
        )

    if len(
        first_values
    ) < 2:
        return None

    first_mean = mean(
        first_values
    )

    second_mean = mean(
        second_values
    )

    if (
        first_mean is None
        or second_mean is None
    ):
        return None

    numerator = sum(
        (
            first
            - first_mean
        )
        * (
            second
            - second_mean
        )
        for first, second
        in zip(
            first_values,
            second_values,
        )
    )

    denominator = math.sqrt(
        sum(
            (
                value
                - first_mean
            )
            ** 2
            for value
            in first_values
        )
        * sum(
            (
                value
                - second_mean
            )
            ** 2
            for value
            in second_values
        )
    )

    if math.isclose(
        denominator,
        0.0,
        abs_tol=1e-15,
    ):
        return None

    return (
        numerator
        / denominator
    )


def ranks(
    values: Sequence[float],
) -> list[float]:
    """
    Convert values to average ranks for Spearman correlation.

    Equal score values receive the average of their occupied rank positions.
    """

    ordered = sorted(
        enumerate(
            values
        ),
        key=lambda item: (
            item[
                1
            ]
        ),
    )

    output = [
        0.0
    ] * len(
        values
    )

    position = 0

    while position < len(
        ordered
    ):
        end = (
            position
            + 1
        )

        while (
            end
            < len(
                ordered
            )
            and math.isclose(
                ordered[
                    end
                ][
                    1
                ],
                ordered[
                    position
                ][
                    1
                ],
                abs_tol=1e-15,
            )
        ):
            end += 1

        average_rank = (
            position
            + 1
            + end
        ) / 2

        for index in range(
            position,
            end,
        ):
            original_index = (
                ordered[
                    index
                ][
                    0
                ]
            )

            output[
                original_index
            ] = average_rank

        position = end

    return output


def write_csv(
    rows: Sequence[
        dict[str, Any]
    ],
    path: Path,
    fields: Sequence[str] | None = None,
) -> None:
    """Write a diagnostic table as UTF-8 CSV with Unix line endings."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if fields is None:
        fields = (
            list(
                rows[
                    0
                ]
            )
            if rows
            else []
        )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(
                fields
            ),
            lineterminator="\n",
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def write_jsonl(
    rows: Sequence[
        dict[str, Any]
    ],
    path: Path,
) -> None:
    """Write qualitative diagnostic examples as one JSON object per line."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for row in rows:
            output_file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_compact_diagnostic_record(
    record: dict[str, Any],
    correct: bool,
    confidence: float | None,
    entailment: float | None,
    valid: bool,
    label: str,
    reasons: Sequence[str],
) -> dict[str, Any]:
    """
    Keep the fields needed for manual verifier inspection.

    The untouched source record is retained so an unusual case can be traced
    back without rerunning the verifier.
    """

    return {
        "index": (
            record[
                "_index"
            ]
        ),
        "correct": (
            correct
        ),
        "confidence": (
            confidence
        ),
        "entailment_probability": (
            entailment
        ),
        "claim_valid": (
            valid
        ),
        "nli_label": (
            label
        ),
        "invalid_claim_reasons": list(
            reasons
        ),
        "question": (
            first_or_nested(
                record,
                QUESTION,
                default=None,
            )
        ),
        "predicted_answer": (
            first_or_nested(
                record,
                PREDICTION,
                default=None,
            )
        ),
        "reference_answers": (
            extract_reference_answers(
                record
            )
        ),
        "claim": (
            first_or_nested(
                record,
                CLAIM,
                default=None,
            )
        ),
        "evidence": (
            first_or_nested(
                record,
                EVIDENCE,
                default=None,
            )
        ),
        "contradiction_probability": (
            to_probability(
                first_or_nested(
                    record,
                    CONTRADICTION,
                    default=None,
                ),
                "contradiction probability",
            )
        ),
        "neutral_probability": (
            to_probability(
                first_or_nested(
                    record,
                    NEUTRAL,
                    default=None,
                ),
                "neutral probability",
            )
        ),
        "source_record": {
            key: value
            for key, value
            in record.items()
            if key
            != "_index"
        },
    }


def select_diagnostic_groups(
    data: Sequence[
        dict[str, Any]
    ],
    high_ent: float,
    low_ent: float,
    high_conf: float,
    low_conf: float,
) -> dict[
    str,
    list[
        dict[str, Any]
    ],
]:
    """
    Build complete diagnostic groups before display truncation.

    Group counts must reflect all matching records. `max_examples` limits only
    how many qualitative examples are later saved.
    """

    return {
        "high_entailment_incorrect": [
            item
            for item in data
            if (
                not item[
                    "correct"
                ]
                and item[
                    "entailment_probability"
                ]
                is not None
                and item[
                    "entailment_probability"
                ]
                >= high_ent
            )
        ],
        "low_entailment_correct": [
            item
            for item in data
            if (
                item[
                    "correct"
                ]
                and item[
                    "entailment_probability"
                ]
                is not None
                and item[
                    "entailment_probability"
                ]
                <= low_ent
            )
        ],
        "confidence_high_semantic_low": [
            item
            for item in data
            if (
                item[
                    "confidence"
                ]
                is not None
                and item[
                    "entailment_probability"
                ]
                is not None
                and item[
                    "confidence"
                ]
                >= high_conf
                and item[
                    "entailment_probability"
                ]
                <= low_ent
            )
        ],
        "confidence_low_semantic_high": [
            item
            for item in data
            if (
                item[
                    "confidence"
                ]
                is not None
                and item[
                    "entailment_probability"
                ]
                is not None
                and item[
                    "confidence"
                ]
                <= low_conf
                and item[
                    "entailment_probability"
                ]
                >= high_ent
            )
        ],
    }


def select_display_examples(
    groups: dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
    max_examples: int,
) -> dict[
    str,
    list[
        dict[str, Any]
    ],
]:
    """
    Sort diagnostic groups and keep only examples intended for inspection.

    `max_examples` affects saved qualitative examples, not aggregate counts.
    """

    return {
        "high_entailment_incorrect": sorted(
            groups[
                "high_entailment_incorrect"
            ],
            key=lambda item: (
                -float(
                    item[
                        "entailment_probability"
                    ]
                ),
                -float(
                    item[
                        "confidence"
                    ]
                    or 0.0
                ),
                int(
                    item[
                        "index"
                    ]
                ),
            ),
        )[
            :max_examples
        ],
        "low_entailment_correct": sorted(
            groups[
                "low_entailment_correct"
            ],
            key=lambda item: (
                float(
                    item[
                        "entailment_probability"
                    ]
                ),
                -float(
                    item[
                        "confidence"
                    ]
                    or 0.0
                ),
                int(
                    item[
                        "index"
                    ]
                ),
            ),
        )[
            :max_examples
        ],
        "confidence_high_semantic_low": sorted(
            groups[
                "confidence_high_semantic_low"
            ],
            key=lambda item: -(
                float(
                    item[
                        "confidence"
                    ]
                )
                - float(
                    item[
                        "entailment_probability"
                    ]
                )
            ),
        )[
            :max_examples
        ],
        "confidence_low_semantic_high": sorted(
            groups[
                "confidence_low_semantic_high"
            ],
            key=lambda item: -(
                float(
                    item[
                        "entailment_probability"
                    ]
                )
                - float(
                    item[
                        "confidence"
                    ]
                )
            ),
        )[
            :max_examples
        ],
    }


def validate_runtime_settings(
    high_ent: float,
    low_ent: float,
    high_conf: float,
    low_conf: float,
    max_examples: int,
) -> None:
    """Validate thresholds used only for diagnostic example extraction."""

    for name, value in (
        (
            "high_entailment",
            high_ent,
        ),
        (
            "low_entailment",
            low_ent,
        ),
        (
            "high_confidence",
            high_conf,
        ),
        (
            "low_confidence",
            low_conf,
        ),
    ):
        if (
            not math.isfinite(
                value
            )
            or not (
                0.0
                <= value
                <= 1.0
            )
        ):
            raise ValueError(
                f"{name} must be a finite "
                "probability in [0, 1]."
            )

    if low_ent > high_ent:
        raise ValueError(
            "low_entailment cannot exceed "
            "high_entailment."
        )

    if low_conf > high_conf:
        raise ValueError(
            "low_confidence cannot exceed "
            "high_confidence."
        )

    if max_examples <= 0:
        raise ValueError(
            "max_examples must be positive."
        )


def format_optional(
    value: float | None,
) -> str:
    """Format an optional statistic for terminal output."""

    if value is None:
        return "N/A"

    return f"{value:.4f}"


def run(
    input_path: str | Path,
    output_dir: str | Path,
    high_ent: float,
    low_ent: float,
    high_conf: float,
    low_conf: float,
    max_examples: int,
) -> None:
    """
    Run the complete question-aware verifier diagnostic analysis.

    This function does not tune verifier parameters. It reads existing verifier
    outputs, recomputes QA correctness, summarizes verifier behavior, extracts
    suspicious examples, and writes tables plus one scatter plot.
    """

    validate_runtime_settings(
        high_ent=(
            high_ent
        ),
        low_ent=(
            low_ent
        ),
        high_conf=(
            high_conf
        ),
        low_conf=(
            low_conf
        ),
        max_examples=(
            max_examples
        ),
    )

    loaded_records = load_jsonl(
        input_path
    )

    if not loaded_records:
        raise ValueError(
            "Input JSONL is empty."
        )

    records: list[
        dict[str, Any]
    ] = []

    for index, original_record in enumerate(
        loaded_records
    ):
        if not isinstance(
            original_record,
            dict,
        ):
            raise TypeError(
                f"Record {index} must "
                "be a JSON object."
            )

        record = dict(
            original_record
        )

        record[
            "_index"
        ] = index

        records.append(
            record
        )

    confidence_field = (
        find_best_available_field(
            records,
            CONFIDENCE,
            numeric=True,
        )
    )

    entailment_field = (
        find_best_available_field(
            records,
            ENTAILMENT,
            numeric=True,
        )
    )

    label_field = (
        find_best_available_field(
            records,
            LABEL,
        )
    )

    validity_field = (
        find_best_available_field(
            records,
            VALID,
        )
    )

    if not confidence_field:
        raise ValueError(
            "No calibrated confidence "
            "field found. "
            f"Checked: {CONFIDENCE}"
        )

    if not entailment_field:
        raise ValueError(
            "No entailment field found. "
            f"Checked: {ENTAILMENT}"
        )

    data: list[
        dict[str, Any]
    ] = []

    for record in records:
        correct = (
            infer_prediction_correctness(
                record
            )
        )

        confidence = (
            to_probability(
                record.get(
                    confidence_field
                ),
                confidence_field,
            )
        )

        entailment = (
            to_probability(
                record.get(
                    entailment_field
                ),
                entailment_field,
            )
        )

        raw_validity = (
            record.get(
                validity_field
            )
            if validity_field
            else None
        )

        if raw_validity is not None:
            parsed_validity = (
                to_boolean(
                    raw_validity
                )
            )

            if parsed_validity is None:
                raise ValueError(
                    "Invalid claim-validity "
                    f"at record {record['_index']}: "
                    f"{raw_validity!r}."
                )

            valid = (
                parsed_validity
            )

        else:
            raw_label = clean(
                first_or_nested(
                    record,
                    LABEL,
                    default="",
                )
            ).upper()

            valid = (
                "INVALID_CLAIM"
                not in raw_label
            )

        nli_label = (
            infer_nli_label(
                record,
                label_field,
                valid,
                entailment,
            )
        )

        invalid_reasons = (
            normalize_invalid_claim_reasons(
                first_or_nested(
                    record,
                    REASONS,
                    default=None,
                )
            )
        )

        data.append(
            build_compact_diagnostic_record(
                record=(
                    record
                ),
                correct=(
                    correct
                ),
                confidence=(
                    confidence
                ),
                entailment=(
                    entailment
                ),
                valid=(
                    valid
                ),
                label=(
                    nli_label
                ),
                reasons=(
                    invalid_reasons
                ),
            )
        )

    output_path = Path(
        output_dir
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = len(
        data
    )

    correct_count = sum(
        int(
            item[
                "correct"
            ]
        )
        for item in data
    )

    # These tables ask whether each verifier state is associated with better or
    # worse underlying QA candidates. They are diagnostics, not routing-policy
    # accuracy metrics.
    by_label: dict[
        str,
        Counter[str],
    ] = defaultdict(
        Counter
    )

    for item in data:
        correctness_key = (
            "correct"
            if item[
                "correct"
            ]
            else "incorrect"
        )

        by_label[
            item[
                "nli_label"
            ]
        ][
            correctness_key
        ] += 1

    label_rows: list[
        dict[str, Any]
    ] = []

    for label in sorted(
        by_label
    ):
        correct = (
            by_label[
                label
            ][
                "correct"
            ]
        )

        incorrect = (
            by_label[
                label
            ][
                "incorrect"
            ]
        )

        count = (
            correct
            + incorrect
        )

        label_rows.append(
            {
                "nli_label": (
                    label
                ),
                "count": (
                    count
                ),
                "correct": (
                    correct
                ),
                "incorrect": (
                    incorrect
                ),
                "accuracy": (
                    correct
                    / count
                ),
                "error_rate": (
                    incorrect
                    / count
                ),
                "share_of_all_records": (
                    count
                    / total
                ),
            }
        )

    by_validity: dict[
        str,
        Counter[str],
    ] = defaultdict(
        Counter
    )

    for item in data:
        validity_key = (
            "VALID"
            if item[
                "claim_valid"
            ]
            else "INVALID"
        )

        correctness_key = (
            "correct"
            if item[
                "correct"
            ]
            else "incorrect"
        )

        by_validity[
            validity_key
        ][
            correctness_key
        ] += 1

    validity_rows: list[
        dict[str, Any]
    ] = []

    for key in (
        "VALID",
        "INVALID",
    ):
        correct = (
            by_validity[
                key
            ][
                "correct"
            ]
        )

        incorrect = (
            by_validity[
                key
            ][
                "incorrect"
            ]
        )

        count = (
            correct
            + incorrect
        )

        validity_rows.append(
            {
                "claim_validity": (
                    key
                ),
                "count": (
                    count
                ),
                "correct": (
                    correct
                ),
                "incorrect": (
                    incorrect
                ),
                "accuracy": (
                    correct
                    / count
                    if count
                    else None
                ),
                "error_rate": (
                    incorrect
                    / count
                    if count
                    else None
                ),
                "share_of_all_records": (
                    count
                    / total
                ),
            }
        )

    # Entailment bins show whether larger semantic-support scores are associated
    # with a larger share of correct QA candidates. This is not a calibration
    # metric.
    bin_counts: dict[
        tuple[
            float,
            float,
        ],
        Counter[str],
    ] = defaultdict(
        Counter
    )

    bin_scores: dict[
        tuple[
            float,
            float,
        ],
        list[float],
    ] = defaultdict(
        list
    )

    for item in data:
        score = item[
            "entailment_probability"
        ]

        if score is None:
            continue

        bin_index = min(
            int(
                score
                * 10
            ),
            9,
        )

        bin_key = (
            BINS[
                bin_index
            ]
        )

        correctness_key = (
            "correct"
            if item[
                "correct"
            ]
            else "incorrect"
        )

        bin_counts[
            bin_key
        ][
            correctness_key
        ] += 1

        bin_scores[
            bin_key
        ].append(
            score
        )

    bin_rows: list[
        dict[str, Any]
    ] = []

    for lower, upper in BINS:
        correct = (
            bin_counts[
                (
                    lower,
                    upper,
                )
            ][
                "correct"
            ]
        )

        incorrect = (
            bin_counts[
                (
                    lower,
                    upper,
                )
            ][
                "incorrect"
            ]
        )

        count = (
            correct
            + incorrect
        )

        closing_bracket = (
            "]"
            if upper == 1
            else ")"
        )

        bin_rows.append(
            {
                "score_bin": (
                    f"[{lower:.1f}, "
                    f"{upper:.1f}"
                    f"{closing_bracket}"
                ),
                "lower_bound": (
                    lower
                ),
                "upper_bound": (
                    upper
                ),
                "count": (
                    count
                ),
                "correct": (
                    correct
                ),
                "incorrect": (
                    incorrect
                ),
                "accuracy": (
                    correct
                    / count
                    if count
                    else None
                ),
                "error_rate": (
                    incorrect
                    / count
                    if count
                    else None
                ),
                "mean_entailment_score": (
                    mean(
                        bin_scores[
                            (
                                lower,
                                upper,
                            )
                        ]
                    )
                ),
            }
        )

    reason_counts: Counter[
        str
    ] = Counter()

    reason_correct: Counter[
        str
    ] = Counter()

    reason_incorrect: Counter[
        str
    ] = Counter()

    for item in data:
        if item[
            "claim_valid"
        ]:
            continue

        reasons = (
            item[
                "invalid_claim_reasons"
            ]
            or [
                "UNSPECIFIED"
            ]
        )

        for reason in reasons:
            reason_counts[
                reason
            ] += 1

            if item[
                "correct"
            ]:
                reason_correct[
                    reason
                ] += 1

            else:
                reason_incorrect[
                    reason
                ] += 1

    reason_rows = [
        {
            "reason": (
                reason
            ),
            "count": (
                count
            ),
            "correct": (
                reason_correct[
                    reason
                ]
            ),
            "incorrect": (
                reason_incorrect[
                    reason
                ]
            ),
            "accuracy": (
                reason_correct[
                    reason
                ]
                / count
            ),
        }
        for reason, count
        in reason_counts.most_common()
    ]

    paired = [
        (
            item[
                "confidence"
            ],
            item[
                "entailment_probability"
            ],
        )
        for item in data
        if (
            item[
                "confidence"
            ]
            is not None
            and item[
                "entailment_probability"
            ]
            is not None
        )
    ]

    first_scores = [
        first_score
        for first_score, _
        in paired
    ]

    second_scores = [
        second_score
        for _, second_score
        in paired
    ]

    diagnostic_groups = (
        select_diagnostic_groups(
            data=(
                data
            ),
            high_ent=(
                high_ent
            ),
            low_ent=(
                low_ent
            ),
            high_conf=(
                high_conf
            ),
            low_conf=(
                low_conf
            ),
        )
    )

    display_examples = (
        select_display_examples(
            groups=(
                diagnostic_groups
            ),
            max_examples=(
                max_examples
            ),
        )
    )

    correct_entailment = [
        item[
            "entailment_probability"
        ]
        for item in data
        if (
            item[
                "correct"
            ]
            and item[
                "entailment_probability"
            ]
            is not None
        )
    ]

    incorrect_entailment = [
        item[
            "entailment_probability"
        ]
        for item in data
        if (
            not item[
                "correct"
            ]
            and item[
                "entailment_probability"
            ]
            is not None
        )
    ]

    pearson_value = (
        calculate_pearson_correlation(
            first_scores,
            second_scores,
        )
    )

    spearman_value = (
        calculate_pearson_correlation(
            ranks(
                first_scores
            ),
            ranks(
                second_scores
            ),
        )
        if len(
            paired
        )
        >= 2
        else None
    )

    summary = {
        "analysis_type": (
            "question_aware_verifier_error_analysis"
        ),
        "analysis_scope": (
            "diagnostic analysis only; "
            "example thresholds are not "
            "tuned decision thresholds"
        ),
        "correctness_definition": (
            "forced-answer normalized Exact Match; "
            "unanswerable candidates are incorrect"
        ),
        "input_path": (
            str(
                input_path
            )
        ),
        "total_records": (
            total
        ),
        "correct_predictions": (
            correct_count
        ),
        "incorrect_predictions": (
            total
            - correct_count
        ),
        "full_accuracy": (
            correct_count
            / total
        ),
        "detected_fields": {
            "confidence": (
                confidence_field
            ),
            "entailment": (
                entailment_field
            ),
            "nli_label": (
                label_field
            ),
            "claim_validity": (
                validity_field
            ),
        },
        "thresholds_used_for_example_extraction_only": {
            "high_entailment": (
                high_ent
            ),
            "low_entailment": (
                low_ent
            ),
            "high_confidence": (
                high_conf
            ),
            "low_confidence": (
                low_conf
            ),
        },
        "claim_validity": {
            "valid_claims": sum(
                int(
                    item[
                        "claim_valid"
                    ]
                )
                for item in data
            ),
            "invalid_claims": sum(
                int(
                    not item[
                        "claim_valid"
                    ]
                )
                for item in data
            ),
        },
        "mean_entailment": {
            "correct_predictions": (
                mean(
                    correct_entailment
                )
            ),
            "incorrect_predictions": (
                mean(
                    incorrect_entailment
                )
            ),
        },
        "confidence_entailment_relationship": {
            "paired_records": (
                len(
                    paired
                )
            ),
            "pearson_correlation": (
                pearson_value
            ),
            "spearman_correlation": (
                spearman_value
            ),
        },

        # These are counts of the complete matching groups. `max_examples`
        # affects only how many examples are saved to the JSONL files.
        "diagnostic_group_counts": {
            name: len(
                group
            )
            for name, group
            in diagnostic_groups.items()
        },
        "saved_examples_per_group": {
            name: len(
                examples
            )
            for name, examples
            in display_examples.items()
        },

        "correctness_by_nli_label": (
            label_rows
        ),
        "correctness_by_claim_validity": (
            validity_rows
        ),
        "entailment_score_bins": (
            bin_rows
        ),
        "invalid_claim_reasons": (
            reason_rows
        ),
    }

    save_json(
        summary,
        output_path
        / "summary.json",
    )

    write_csv(
        label_rows,
        output_path
        / "correctness_by_nli_label.csv",
    )

    write_csv(
        validity_rows,
        output_path
        / "correctness_by_claim_validity.csv",
    )

    write_csv(
        bin_rows,
        output_path
        / "entailment_score_bins.csv",
    )

    write_csv(
        reason_rows,
        output_path
        / "invalid_claim_reasons.csv",
        (
            "reason",
            "count",
            "correct",
            "incorrect",
            "accuracy",
        ),
    )

    write_jsonl(
        display_examples[
            "high_entailment_incorrect"
        ],
        output_path
        / "high_entailment_incorrect.jsonl",
    )

    write_jsonl(
        display_examples[
            "low_entailment_correct"
        ],
        output_path
        / "low_entailment_correct.jsonl",
    )

    write_jsonl(
        display_examples[
            "confidence_high_semantic_low"
        ],
        output_path
        / "confidence_high_semantic_low.jsonl",
    )

    write_jsonl(
        display_examples[
            "confidence_low_semantic_high"
        ],
        output_path
        / "confidence_low_semantic_high.jsonl",
    )

    # This plot visualizes agreement/disagreement between calibrated QA
    # confidence and the verifier's semantic-support score. It does not itself
    # evaluate selective ranking quality.
    figure = plt.figure(
        figsize=(
            8,
            6,
        )
    )

    axis = figure.add_subplot(
        111
    )

    for correctness, label in (
        (
            True,
            "Correct",
        ),
        (
            False,
            "Incorrect",
        ),
    ):
        points = [
            item
            for item in data
            if (
                item[
                    "correct"
                ]
                == correctness
                and item[
                    "confidence"
                ]
                is not None
                and item[
                    "entailment_probability"
                ]
                is not None
            )
        ]

        if points:
            axis.scatter(
                [
                    item[
                        "confidence"
                    ]
                    for item in points
                ],
                [
                    item[
                        "entailment_probability"
                    ]
                    for item in points
                ],
                alpha=0.65,
                label=(
                    label
                ),
            )

    axis.set(
        xlabel=(
            "Calibrated confidence"
        ),
        ylabel=(
            "Question-aware entailment probability"
        ),
        title=(
            "Confidence vs Question-Aware Entailment"
        ),
        xlim=(
            0,
            1,
        ),
        ylim=(
            0,
            1,
        ),
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path
        / "confidence_entailment_scatter.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    correct_mean = mean(
        correct_entailment
    )

    incorrect_mean = mean(
        incorrect_entailment
    )

    print(
        "\n"
        + "=" * 92
        + "\n"
        + "QUESTION-AWARE VERIFIER ERROR ANALYSIS"
        + "\n"
        + "=" * 92
    )

    print(
        f"Total: {total} | "
        f"Correct: {correct_count} | "
        f"Incorrect: {total - correct_count} | "
        f"Accuracy: {correct_count / total:.4f}"
    )

    print(
        "Valid claims: "
        f"{summary['claim_validity']['valid_claims']} | "
        "Invalid claims: "
        f"{summary['claim_validity']['invalid_claims']}"
    )

    print(
        "Mean entailment — correct: "
        f"{format_optional(correct_mean)} | "
        "incorrect: "
        f"{format_optional(incorrect_mean)}"
    )

    print(
        "Pearson: "
        f"{format_optional(pearson_value)} | "
        "Spearman: "
        f"{format_optional(spearman_value)}"
    )

    print(
        "\nCORRECTNESS BY NLI LABEL"
    )

    for row in label_rows:
        print(
            f"{row['nli_label']:<20} "
            f"count={row['count']:<4} "
            f"accuracy={row['accuracy']:.4f}"
        )

    print(
        "\nDIAGNOSTIC GROUPS"
    )

    for key, value in (
        summary[
            "diagnostic_group_counts"
        ].items()
    ):
        saved = (
            summary[
                "saved_examples_per_group"
            ][
                key
            ]
        )

        print(
            f"{key:<35} "
            f"count={value:<5} "
            f"saved={saved}"
        )

    print(
        f"\nOutputs: "
        f"{output_path}"
    )


def probability_argument(
    value: str,
) -> float:
    """Parse a command-line probability in the closed interval [0, 1]."""

    number = float(
        value
    )

    if not (
        0.0
        <= number
        <= 1.0
    ):
        raise argparse.ArgumentTypeError(
            "Probability must be "
            "between 0 and 1."
        )

    return number


def positive_integer_argument(
    value: str,
) -> int:
    """Parse a strictly positive integer command-line argument."""

    number = int(
        value
    )

    if number <= 0:
        raise argparse.ArgumentTypeError(
            "Value must be positive."
        )

    return number


def main() -> None:
    """Run question-aware verifier diagnostics from command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Diagnose question-aware NLI "
            "verifier behavior on existing "
            "prediction artifacts."
        )
    )

    parser.add_argument(
        "--input",
        default=str(
            INPUT
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            OUT
        ),
    )

    parser.add_argument(
        "--high-entailment",
        type=probability_argument,
        default=0.8,
    )

    parser.add_argument(
        "--low-entailment",
        type=probability_argument,
        default=0.2,
    )

    parser.add_argument(
        "--high-confidence",
        type=probability_argument,
        default=0.8,
    )

    parser.add_argument(
        "--low-confidence",
        type=probability_argument,
        default=0.2,
    )

    parser.add_argument(
        "--max-examples",
        type=positive_integer_argument,
        default=50,
    )

    arguments = parser.parse_args()

    run(
        arguments.input,
        arguments.output_dir,
        arguments.high_entailment,
        arguments.low_entailment,
        arguments.high_confidence,
        arguments.low_confidence,
        arguments.max_examples,
    )


if __name__ == "__main__":
    main()