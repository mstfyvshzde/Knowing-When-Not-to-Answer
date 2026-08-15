"""
Analyze how lexical evidence verification changes prototype routing decisions.

The input is expected to come from the rule-based selective-QA pipeline:

    calibrated confidence
        -> threshold routing: ANSWER / VERIFY / ABSTAIN
        -> lexical evidence verification
        -> final routing: ANSWER / VERIFY / ABSTAIN

The decision engine preserves direct threshold decisions:

- threshold ANSWER  -> final ANSWER
- threshold ABSTAIN -> final ABSTAIN

Evidence affects only the intermediate VERIFY region:

- VERIFY + SUPPORTED   -> ANSWER
- VERIFY + WEAK        -> VERIFY
- VERIFY + UNSUPPORTED -> ABSTAIN

This module measures which underlying QA candidates were promoted, blocked, or
left unresolved by evidence-aware routing.

Important
---------
Underlying QA correctness must be independent of the later routing action.
Therefore this analysis reuses the same correctness function used during
confidence-threshold selection: `calibration_metrics.is_prediction_correct`.

Under that forced-answer correctness definition, answerable examples are
correct only when the predicted span exactly matches a reference after project
normalization, while unanswerable forced-answer candidates are incorrect.

This is a diagnostic analysis of the earlier rule-based prototype, not the
project's final score-ranking/AURC evaluation.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.calibration.calibration_metrics import (
    is_prediction_correct,
)
from src.calibration.calibration_metrics import (
    normalize_answer as calibration_normalize_answer,
)
from src.evaluation.metrics import parse_answerability
from src.utils.io import (
    load_jsonl,
    save_jsonl,
)
from src.utils.io import (
    save_json as shared_save_json,
)

DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/calibration_final_decisions.jsonl"
)

DEFAULT_METRICS_OUTPUT_PATH = Path(
    "outputs/tables/evidence_impact_analysis.json"
)

DEFAULT_CASES_OUTPUT_PATH = Path(
    "outputs/predictions/evidence_impact_cases.jsonl"
)


VALID_DECISIONS = {
    "ANSWER",
    "VERIFY",
    "ABSTAIN",
}

VALID_EVIDENCE_LABELS = {
    "SUPPORTED",
    "WEAK",
    "UNSUPPORTED",
}

EXPECTED_VERIFY_RESOLUTION = {
    "SUPPORTED": "ANSWER",
    "WEAK": "VERIFY",
    "UNSUPPORTED": "ABSTAIN",
}


CALIBRATED_CONFIDENCE_FIELDS = (
    "calibrated_confidence",
    "temperature_scaled_confidence",
    "qa_calibrated_confidence",
)

EVIDENCE_SCORE_FIELDS = (
    "combined_evidence_score",
    "evidence_score",
    "lexical_evidence_score",
)


def normalize_boolean(
    value: Any,
) -> bool:
    """
    Convert common stored Boolean representations into a strict Boolean.

    This helper is retained for compatibility with historical callers. It is
    not used to derive QA correctness in this analysis.
    """

    if isinstance(value, bool):
        return value

    if (
        isinstance(value, int)
        and not isinstance(value, bool)
    ):
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
        normalized_value = (
            value.strip().lower()
        )

        if normalized_value in {
            "true",
            "1",
            "correct",
            "yes",
        }:
            return True

        if normalized_value in {
            "false",
            "0",
            "incorrect",
            "no",
        }:
            return False

    raise ValueError(
        "Could not convert value to Boolean: "
        f"{value!r}"
    )


def normalize_answer(
    text: Any,
) -> str:
    """
    Normalize answer text using the same rule as calibration correctness.

    The wrapper preserves this module's historical public helper.
    """

    return calibration_normalize_answer(
        str(text or "")
    )


def extract_reference_answers(
    prediction: dict[str, Any],
) -> list[str]:
    """Extract reference answers for diagnostics and input validation."""

    reference_answers = prediction.get(
        "reference_answers",
        [],
    )

    if reference_answers is None:
        return []

    if isinstance(
        reference_answers,
        str,
    ):
        return [
            reference_answers
        ]

    if isinstance(
        reference_answers,
        dict,
    ):
        for field_name in (
            "text",
            "answers",
            "answer_text",
        ):
            values = reference_answers.get(
                field_name
            )

            if values is None:
                continue

            if isinstance(
                values,
                str,
            ):
                return [
                    values
                ]

            if isinstance(
                values,
                list,
            ):
                extracted: list[str] = []

                for value in values:
                    if isinstance(
                        value,
                        dict,
                    ):
                        answer_text = (
                            value.get("text")
                            or value.get("answer")
                            or value.get(
                                "answer_text"
                            )
                        )

                        if answer_text is not None:
                            extracted.append(
                                str(answer_text)
                            )

                    elif value is not None:
                        extracted.append(
                            str(value)
                        )

                return extracted

        return []

    if isinstance(
        reference_answers,
        list,
    ):
        extracted: list[str] = []

        for item in reference_answers:
            if isinstance(
                item,
                str,
            ):
                extracted.append(
                    item
                )

            elif isinstance(
                item,
                dict,
            ):
                answer_text = (
                    item.get("text")
                    or item.get("answer")
                    or item.get(
                        "answer_text"
                    )
                )

                if answer_text is not None:
                    extracted.append(
                        str(answer_text)
                    )

            elif item is not None:
                extracted.append(
                    str(item)
                )

        return extracted

    return [
        str(reference_answers)
    ]


def validate_correctness_inputs(
    prediction: dict[str, Any],
) -> None:
    """
    Validate the fields used by calibration correctness.

    This protects the shared correctness function from malformed answerability
    or reference data that could otherwise create silent label errors.
    """

    if (
        "is_answerable"
        not in prediction
    ):
        raise ValueError(
            "Prediction does not contain "
            "is_answerable."
        )

    try:
        is_answerable = (
            parse_answerability(
                prediction[
                    "is_answerable"
                ]
            )
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            "Invalid is_answerable value: "
            f"{prediction.get('is_answerable')!r}."
        ) from error

    if (
        "prediction_text"
        not in prediction
    ):
        raise ValueError(
            "Prediction does not contain "
            "prediction_text."
        )

    references = (
        extract_reference_answers(
            prediction
        )
    )

    usable_references = [
        reference
        for reference in references
        if normalize_answer(
            reference
        )
    ]

    if (
        is_answerable
        and not usable_references
    ):
        raise ValueError(
            "Answerable prediction does not "
            "contain a usable reference answer."
        )

    if (
        not is_answerable
        and usable_references
    ):
        raise ValueError(
            "Unanswerable prediction contains "
            "non-empty reference answers."
        )


def get_correctness(
    prediction: dict[str, Any],
) -> bool:
    """
    Return underlying forced-answer QA correctness.

    The same function used by confidence-threshold selection is reused here.
    Threshold and final routing decisions are intentionally ignored.
    """

    validate_correctness_inputs(
        prediction
    )

    return bool(
        is_prediction_correct(
            prediction
        )
    )


def get_threshold_decision(
    prediction: dict[str, Any],
) -> str:
    """
    Retrieve the confidence-threshold routing decision.

    Current threshold-selection artifacts store this value in `decision`.
    Explicit threshold-specific aliases are preferred when available.
    """

    possible_fields = (
        "threshold_decision",
        "confidence_decision",
        "selective_decision",
        "decision",
    )

    for field_name in possible_fields:
        value = prediction.get(
            field_name
        )

        if value is None:
            continue

        decision = (
            str(value)
            .strip()
            .upper()
        )

        if (
            decision
            in VALID_DECISIONS
        ):
            return decision

    raise ValueError(
        "Prediction does not contain a valid "
        "threshold decision."
    )


def get_final_decision(
    prediction: dict[str, Any],
) -> str:
    """Retrieve and validate the decision-engine output."""

    value = prediction.get(
        "final_decision"
    )

    if value is None:
        raise ValueError(
            "Prediction does not contain "
            "final_decision."
        )

    decision = (
        str(value)
        .strip()
        .upper()
    )

    if (
        decision
        not in VALID_DECISIONS
    ):
        raise ValueError(
            f"Invalid final decision: "
            f"{decision!r}."
        )

    return decision


def get_evidence_support(
    prediction: dict[str, Any],
) -> str:
    """Retrieve and validate the lexical evidence-support label."""

    value = prediction.get(
        "evidence_support"
    )

    if value is None:
        raise ValueError(
            "Prediction does not contain "
            "evidence_support."
        )

    evidence_support = (
        str(value)
        .strip()
        .upper()
    )

    if (
        evidence_support
        not in VALID_EVIDENCE_LABELS
    ):
        raise ValueError(
            "Invalid evidence support label: "
            f"{evidence_support!r}."
        )

    return evidence_support


def get_numeric_value(
    prediction: dict[str, Any],
    field: str,
) -> float | None:
    """
    Read one optional numeric diagnostic field.

    Missing values return None. Present malformed or non-finite values raise an
    error instead of silently disappearing from averages.
    """

    if (
        field not in prediction
        or prediction[field] is None
    ):
        return None

    value = prediction[
        field
    ]

    if isinstance(
        value,
        bool,
    ):
        raise ValueError(  # noqa: TRY004
            "Boolean value is invalid for "
            f"numeric field {field!r}."
        )

    try:
        numeric_value = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            f"Invalid numeric value in "
            f"{field!r}: {value!r}."
        ) from error

    if not math.isfinite(
        numeric_value
    ):
        raise ValueError(
            f"Non-finite numeric value in "
            f"{field!r}: {numeric_value}."
        )

    return numeric_value


def get_first_numeric_value(
    prediction: dict[str, Any],
    fields: tuple[str, ...],
) -> float | None:
    """Return the first available validated numeric value from aliases."""

    for field_name in fields:
        value = (
            get_numeric_value(
                prediction,
                field_name,
            )
        )

        if value is not None:
            return value

    return None


def get_calibrated_confidence(
    prediction: dict[str, Any],
) -> float | None:
    """
    Retrieve calibrated confidence for diagnostic averages.

    Generic `confidence` is accepted only when explicitly marked calibrated.
    """

    value = (
        get_first_numeric_value(
            prediction,
            CALIBRATED_CONFIDENCE_FIELDS,
        )
    )

    if (
        value is None
        and prediction.get(
            "confidence_is_calibrated"
        )
        is True
    ):
        value = (
            get_numeric_value(
                prediction,
                "confidence",
            )
        )

    if value is None:
        return None

    if not (
        0.0
        <= value
        <= 1.0
    ):
        raise ValueError(
            "Calibrated confidence must lie "
            "in [0, 1], "
            f"received {value}."
        )

    return value


def get_evidence_score(
    prediction: dict[str, Any],
) -> float | None:
    """Retrieve the lexical evidence score across old/new field aliases."""

    value = (
        get_first_numeric_value(
            prediction,
            EVIDENCE_SCORE_FIELDS,
        )
    )

    if value is None:
        return None

    if not (
        0.0
        <= value
        <= 1.0
    ):
        raise ValueError(
            "Evidence score must lie in [0, 1], "
            f"received {value}."
        )

    return value


def safe_divide(
    numerator: float,
    denominator: float,
) -> float | None:
    """Return numerator / denominator, or None when undefined."""

    if denominator == 0:
        return None

    return float(
        numerator
        / denominator
    )


def calculate_group_metrics(
    predictions: list[dict[str, Any]],
    total_predictions: int | None = None,
) -> dict[str, Any]:
    """
    Summarize underlying QA quality and diagnostics for one group.

    `accuracy` is forced-answer QA candidate accuracy inside the group, not
    correctness of the routing action itself.
    """

    count = len(
        predictions
    )

    correct_count = sum(
        int(
            get_correctness(
                prediction
            )
        )
        for prediction in predictions
    )

    incorrect_count = (
        count
        - correct_count
    )

    accuracy = (
        safe_divide(
            correct_count,
            count,
        )
    )

    risk = (
        None
        if accuracy is None
        else 1.0 - accuracy
    )

    group_rate = (
        None
        if total_predictions is None
        else safe_divide(
            count,
            total_predictions,
        )
    )

    signal_extractors = {
        "calibrated_confidence": (
            get_calibrated_confidence
        ),
        "evidence_score": (
            get_evidence_score
        ),
        "answer_context_score": (
            lambda item: get_numeric_value(
                item,
                "answer_context_score",
            )
        ),
        "question_evidence_overlap": (
            lambda item: get_numeric_value(
                item,
                "question_evidence_overlap",
            )
        ),
    }

    averages: dict[
        str,
        float | None,
    ] = {}

    for (
        signal_name,
        extractor,
    ) in signal_extractors.items():
        values = [
            float(value)
            for prediction in predictions
            if (
                value := extractor(
                    prediction
                )
            )
            is not None
        ]

        averages[
            f"average_{signal_name}"
        ] = (
            None
            if not values
            else sum(values)
            / len(values)
        )

    return {
        "count": count,
        "group_rate": (
            group_rate
        ),
        "correct_count": (
            correct_count
        ),
        "incorrect_count": (
            incorrect_count
        ),
        "accuracy": (
            accuracy
        ),
        "risk": (
            risk
        ),
        **averages,
    }


def build_transition_name(
    threshold_decision: str,
    final_decision: str,
) -> str:
    """Combine threshold and final labels into one transition name."""

    return (
        f"{threshold_decision}"
        f"_TO_"
        f"{final_decision}"
    )


def validate_transition(
    threshold_decision: str,
    evidence_support: str,
    final_decision: str,
) -> None:
    """
    Verify that one record obeys the configured decision-engine policy.

    Evidence may change only predictions from the VERIFY region.
    """

    if (
        threshold_decision
        == "ANSWER"
    ):
        expected_final = (
            "ANSWER"
        )

    elif (
        threshold_decision
        == "ABSTAIN"
    ):
        expected_final = (
            "ABSTAIN"
        )

    else:
        expected_final = (
            EXPECTED_VERIFY_RESOLUTION[
                evidence_support
            ]
        )

    if (
        final_decision
        != expected_final
    ):
        raise ValueError(
            "Transition is inconsistent with "
            "the rule-based decision engine: "
            f"threshold={threshold_decision}, "
            f"evidence={evidence_support}, "
            f"expected_final={expected_final}, "
            f"observed_final={final_decision}."
        )


def classify_case(
    threshold_decision: str,
    final_decision: str,
    is_correct: bool,
) -> str:
    """Classify how evidence resolved an example from the VERIFY region."""

    if (
        threshold_decision
        == "VERIFY"
        and final_decision
        == "ANSWER"
    ):
        return (
            "verify_correct_promoted_to_answer"
            if is_correct
            else "verify_incorrect_promoted_to_answer"
        )

    if (
        threshold_decision
        == "VERIFY"
        and final_decision
        == "ABSTAIN"
    ):
        return (
            "verify_correct_blocked"
            if is_correct
            else "verify_incorrect_blocked"
        )

    if (
        threshold_decision
        == "VERIFY"
        and final_decision
        == "VERIFY"
    ):
        return (
            "verify_correct_preserved"
            if is_correct
            else "verify_incorrect_preserved"
        )

    return "other_transition"


def calculate_verify_impact_rates(
    verify_metrics: dict[str, Any],
    analyzed_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Calculate interpretable rates within the original VERIFY region.

    Beneficial blocking means an incorrect candidate was routed to ABSTAIN.
    Harmful blocking means a correct candidate was routed to ABSTAIN.
    """

    counts = Counter(
        case[
            "diagnostic_category"
        ]
        for case in analyzed_cases
    )

    correct_promoted = counts[
        "verify_correct_promoted_to_answer"
    ]

    incorrect_promoted = counts[
        "verify_incorrect_promoted_to_answer"
    ]

    correct_blocked = counts[
        "verify_correct_blocked"
    ]

    incorrect_blocked = counts[
        "verify_incorrect_blocked"
    ]

    correct_preserved = counts[
        "verify_correct_preserved"
    ]

    incorrect_preserved = counts[
        "verify_incorrect_preserved"
    ]

    verify_count = int(
        verify_metrics[
            "count"
        ]
    )

    verify_correct_count = int(
        verify_metrics[
            "correct_count"
        ]
    )

    verify_incorrect_count = int(
        verify_metrics[
            "incorrect_count"
        ]
    )

    categorized_verify_count = (
        correct_promoted
        + incorrect_promoted
        + correct_blocked
        + incorrect_blocked
        + correct_preserved
        + incorrect_preserved
    )

    if (
        categorized_verify_count
        != verify_count
    ):
        raise RuntimeError(
            "VERIFY diagnostic categories do "
            "not sum to the threshold VERIFY count."
        )

    promoted_count = (
        correct_promoted
        + incorrect_promoted
    )

    blocked_count = (
        correct_blocked
        + incorrect_blocked
    )

    preserved_count = (
        correct_preserved
        + incorrect_preserved
    )

    return {
        "verify_predictions": (
            verify_count
        ),
        "verify_correct_candidates": (
            verify_correct_count
        ),
        "verify_incorrect_candidates": (
            verify_incorrect_count
        ),
        "verify_correct_promoted_to_answer": (
            correct_promoted
        ),
        "verify_incorrect_promoted_to_answer": (
            incorrect_promoted
        ),
        "verify_correct_blocked": (
            correct_blocked
        ),
        "verify_incorrect_blocked": (
            incorrect_blocked
        ),
        "verify_correct_preserved": (
            correct_preserved
        ),
        "verify_incorrect_preserved": (
            incorrect_preserved
        ),
        "promotion_precision": (
            safe_divide(
                correct_promoted,
                promoted_count,
            )
        ),
        "beneficial_block_rate_among_incorrect_verify": (
            safe_divide(
                incorrect_blocked,
                verify_incorrect_count,
            )
        ),
        "harmful_block_rate_among_correct_verify": (
            safe_divide(
                correct_blocked,
                verify_correct_count,
            )
        ),
        "resolved_verify_rate": (
            safe_divide(
                promoted_count
                + blocked_count,
                verify_count,
            )
        ),
        "preserved_verify_rate": (
            safe_divide(
                preserved_count,
                verify_count,
            )
        ),
    }


def analyze_predictions(
    predictions: list[dict[str, Any]],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    """
    Analyze evidence effects on threshold routing.

    Every record is validated against the expected decision-engine transition
    before aggregate statistics are calculated.
    """

    if not predictions:
        raise ValueError(
            "Prediction list cannot be empty."
        )

    total_predictions = len(
        predictions
    )

    transition_groups: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(
        list
    )

    evidence_groups: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(
        list
    )

    diagnostic_groups: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(
        list
    )

    analyzed_cases: list[
        dict[str, Any]
    ] = []

    for index, prediction in enumerate(
        predictions,
        start=1,
    ):
        try:
            threshold_decision = (
                get_threshold_decision(
                    prediction
                )
            )

            final_decision = (
                get_final_decision(
                    prediction
                )
            )

            evidence_support = (
                get_evidence_support(
                    prediction
                )
            )

            is_correct = (
                get_correctness(
                    prediction
                )
            )

            validate_transition(
                threshold_decision=(
                    threshold_decision
                ),
                evidence_support=(
                    evidence_support
                ),
                final_decision=(
                    final_decision
                ),
            )

            calibrated_confidence = (
                get_calibrated_confidence(
                    prediction
                )
            )

            evidence_score = (
                get_evidence_score(
                    prediction
                )
            )

            answer_context_score = (
                get_numeric_value(
                    prediction,
                    "answer_context_score",
                )
            )

            question_evidence_overlap = (
                get_numeric_value(
                    prediction,
                    "question_evidence_overlap",
                )
            )

        except (
            ValueError,
            TypeError,
            KeyError,
        ) as error:
            raise ValueError(
                f"Prediction {index} "
                f"failed analysis: {error}"
            ) from error

        transition = (
            build_transition_name(
                threshold_decision,
                final_decision,
            )
        )

        diagnostic_category = (
            classify_case(
                threshold_decision=(
                    threshold_decision
                ),
                final_decision=(
                    final_decision
                ),
                is_correct=(
                    is_correct
                ),
            )
        )

        transition_groups[
            transition
        ].append(
            prediction
        )

        evidence_groups[
            evidence_support
        ].append(
            prediction
        )

        diagnostic_groups[
            diagnostic_category
        ].append(
            prediction
        )

        analyzed_cases.append(
            {
                "index": index,
                "id": prediction.get(
                    "id"
                ),
                "question": prediction.get(
                    "question"
                ),
                "prediction_text": prediction.get(
                    "prediction_text"
                ),
                "reference_answers": (
                    extract_reference_answers(
                        prediction
                    )
                ),
                "is_answerable": prediction.get(
                    "is_answerable"
                ),
                "is_correct": (
                    is_correct
                ),
                "calibrated_confidence": (
                    calibrated_confidence
                ),
                "evidence_score": (
                    evidence_score
                ),
                "answer_context_score": (
                    answer_context_score
                ),
                "question_evidence_overlap": (
                    question_evidence_overlap
                ),
                "evidence_support": (
                    evidence_support
                ),
                "threshold_decision": (
                    threshold_decision
                ),
                "final_decision": (
                    final_decision
                ),
                "decision_reason": prediction.get(
                    "decision_reason"
                ),
                "transition": (
                    transition
                ),
                "diagnostic_category": (
                    diagnostic_category
                ),
                "evidence_text": prediction.get(
                    "evidence_text"
                ),
            }
        )

    transition_metrics = {
        transition: calculate_group_metrics(
            group_predictions,
            total_predictions=(
                total_predictions
            ),
        )
        for (
            transition,
            group_predictions,
        )
        in sorted(
            transition_groups.items()
        )
    }

    evidence_metrics = {
        evidence_label: calculate_group_metrics(
            group_predictions,
            total_predictions=(
                total_predictions
            ),
        )
        for (
            evidence_label,
            group_predictions,
        )
        in sorted(
            evidence_groups.items()
        )
    }

    diagnostic_metrics = {
        category: calculate_group_metrics(
            group_predictions,
            total_predictions=(
                total_predictions
            ),
        )
        for (
            category,
            group_predictions,
        )
        in sorted(
            diagnostic_groups.items()
        )
    }

    threshold_answer_predictions = [
        prediction
        for prediction in predictions
        if (
            get_threshold_decision(
                prediction
            )
            == "ANSWER"
        )
    ]

    threshold_verify_predictions = [
        prediction
        for prediction in predictions
        if (
            get_threshold_decision(
                prediction
            )
            == "VERIFY"
        )
    ]

    final_answer_predictions = [
        prediction
        for prediction in predictions
        if (
            get_final_decision(
                prediction
            )
            == "ANSWER"
        )
    ]

    threshold_answer_metrics = (
        calculate_group_metrics(
            threshold_answer_predictions,
            total_predictions=(
                total_predictions
            ),
        )
    )

    threshold_verify_metrics = (
        calculate_group_metrics(
            threshold_verify_predictions,
            total_predictions=(
                total_predictions
            ),
        )
    )

    final_answer_metrics = (
        calculate_group_metrics(
            final_answer_predictions,
            total_predictions=(
                total_predictions
            ),
        )
    )

    verify_impact = (
        calculate_verify_impact_rates(
            verify_metrics=(
                threshold_verify_metrics
            ),
            analyzed_cases=(
                analyzed_cases
            ),
        )
    )

    threshold_coverage = (
        threshold_answer_metrics[
            "group_rate"
        ]
    )

    final_coverage = (
        final_answer_metrics[
            "group_rate"
        ]
    )

    threshold_risk = (
        threshold_answer_metrics[
            "risk"
        ]
    )

    final_risk = (
        final_answer_metrics[
            "risk"
        ]
    )

    operating_point_comparison = {
        "threshold_direct_answer_coverage": (
            threshold_coverage
        ),
        "final_answer_coverage": (
            final_coverage
        ),
        "coverage_change": (
            None
            if (
                threshold_coverage is None
                or final_coverage is None
            )
            else (
                final_coverage
                - threshold_coverage
            )
        ),
        "threshold_direct_answer_risk": (
            threshold_risk
        ),
        "final_answer_risk": (
            final_risk
        ),
        "risk_change": (
            None
            if (
                threshold_risk is None
                or final_risk is None
            )
            else (
                final_risk
                - threshold_risk
            )
        ),
        "interpretation_note": (
            "This is an operating-point comparison. "
            "Coverage changes when VERIFY examples "
            "are promoted to ANSWER, so risk change "
            "is not a matched-coverage effect."
        ),
    }

    summary = {
        "analysis_type": (
            "prototype_lexical_evidence_impact"
        ),
        "total_predictions": (
            total_predictions
        ),
        "correctness_definition": (
            "same forced-answer Exact-Match "
            "correctness used by calibration "
            "threshold selection; unanswerable "
            "forced-answer candidates are incorrect"
        ),
        "threshold_answer_policy": (
            threshold_answer_metrics
        ),
        "threshold_verify_policy": (
            threshold_verify_metrics
        ),
        "final_answer_policy": (
            final_answer_metrics
        ),
        "operating_point_comparison": (
            operating_point_comparison
        ),
        "evidence_label_metrics": (
            evidence_metrics
        ),
        "transition_metrics": (
            transition_metrics
        ),
        "diagnostic_metrics": (
            diagnostic_metrics
        ),
        "evidence_impact_summary": (
            verify_impact
        ),
        "diagnostic_category_counts": (
            dict(
                Counter(
                    case[
                        "diagnostic_category"
                    ]
                    for case in analyzed_cases
                )
            )
        ),
    }

    return (
        summary,
        analyzed_cases,
    )


def save_json(
    data: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save formatted JSON through the shared repository helper."""

    shared_save_json(
        data,
        output_path,
    )


def format_metric(
    value: float | None,
) -> str:
    """Format an optional metric to four decimal places."""

    if value is None:
        return "N/A"

    return f"{value:.4f}"


def print_summary(
    analysis: dict[str, Any],
) -> None:
    """Print the main evidence-impact diagnostics."""

    impact = analysis[
        "evidence_impact_summary"
    ]

    threshold_answer = analysis[
        "threshold_answer_policy"
    ]

    threshold_verify = analysis[
        "threshold_verify_policy"
    ]

    final_answer = analysis[
        "final_answer_policy"
    ]

    operating_point = analysis[
        "operating_point_comparison"
    ]

    print(
        "\nEvidence impact analysis completed."
    )

    print(
        f"Total predictions: "
        f"{analysis['total_predictions']}"
    )

    print(
        "\nThreshold direct-ANSWER region:"
    )

    print(
        f"Answer count: "
        f"{threshold_answer['count']}"
    )

    print(
        "Coverage: "
        f"{format_metric(
            threshold_answer['group_rate']
        )}"
    )

    print(
        "Underlying QA accuracy: "
        f"{format_metric(
            threshold_answer['accuracy']
        )}"
    )

    print(
        "Selective risk: "
        f"{format_metric(
            threshold_answer['risk']
        )}"
    )

    print(
        "\nThreshold VERIFY region:"
    )

    print(
        f"VERIFY count: "
        f"{threshold_verify['count']}"
    )

    print(
        "Underlying QA accuracy: "
        f"{format_metric(
            threshold_verify['accuracy']
        )}"
    )

    print(
        "\nFinal evidence-aware ANSWER region:"
    )

    print(
        f"Answer count: "
        f"{final_answer['count']}"
    )

    print(
        "Coverage: "
        f"{format_metric(
            final_answer['group_rate']
        )}"
    )

    print(
        "Underlying QA accuracy: "
        f"{format_metric(
            final_answer['accuracy']
        )}"
    )

    print(
        "Selective risk: "
        f"{format_metric(
            final_answer['risk']
        )}"
    )

    print(
        "\nOperating-point change "
        "(not matched coverage):"
    )

    print(
        "Coverage change: "
        f"{format_metric(
            operating_point['coverage_change']
        )}"
    )

    print(
        "Risk change: "
        f"{format_metric(
            operating_point['risk_change']
        )}"
    )

    print(
        "\nEvidence impact inside VERIFY:"
    )

    print(
        f"VERIFY predictions: "
        f"{impact['verify_predictions']}"
    )

    print(
        "Correct VERIFY promoted to ANSWER: "
        f"{impact[
            'verify_correct_promoted_to_answer'
        ]}"
    )

    print(
        "Incorrect VERIFY promoted to ANSWER: "
        f"{impact[
            'verify_incorrect_promoted_to_answer'
        ]}"
    )

    print(
        "Correct VERIFY blocked: "
        f"{impact[
            'verify_correct_blocked'
        ]}"
    )

    print(
        "Incorrect VERIFY blocked: "
        f"{impact[
            'verify_incorrect_blocked'
        ]}"
    )

    print(
        "Correct VERIFY preserved: "
        f"{impact[
            'verify_correct_preserved'
        ]}"
    )

    print(
        "Incorrect VERIFY preserved: "
        f"{impact[
            'verify_incorrect_preserved'
        ]}"
    )

    print(
        "Promotion precision: "
        f"{format_metric(
            impact['promotion_precision']
        )}"
    )

    print(
        "Beneficial block rate among incorrect "
        "VERIFY candidates: "
        f"{format_metric(
            impact[
                'beneficial_block_rate_among_incorrect_verify'
            ]
        )}"
    )

    print(
        "Harmful block rate among correct "
        "VERIFY candidates: "
        f"{format_metric(
            impact[
                'harmful_block_rate_among_correct_verify'
            ]
        )}"
    )

    print(
        "\nEvidence-label underlying QA accuracy:"
    )

    evidence_metrics = analysis[
        "evidence_label_metrics"
    ]

    for evidence_label in (
        "SUPPORTED",
        "WEAK",
        "UNSUPPORTED",
    ):
        metrics = (
            evidence_metrics.get(
                evidence_label
            )
        )

        if metrics is None:
            continue

        print(
            f"{evidence_label}: "
            f"count={metrics['count']} | "
            "accuracy="
            f"{format_metric(
                metrics['accuracy']
            )}"
        )

    print(
        "\nDecision transitions:"
    )

    for (
        transition,
        metrics,
    ) in analysis[
        "transition_metrics"
    ].items():
        print(
            f"{transition}: "
            f"count={metrics['count']} | "
            "underlying QA accuracy="
            f"{format_metric(
                metrics['accuracy']
            )}"
        )


def run_analysis(
    input_path: str | Path,
    metrics_output_path: str | Path,
    cases_output_path: str | Path,
) -> dict[str, Any]:
    """Run evidence-impact analysis and save aggregate/per-case outputs."""

    predictions = (
        load_jsonl(
            input_path
        )
    )

    (
        analysis,
        analyzed_cases,
    ) = analyze_predictions(
        predictions
    )

    save_json(
        data=analysis,
        output_path=(
            metrics_output_path
        ),
    )

    save_jsonl(
        analyzed_cases,
        cases_output_path,
    )

    print_summary(
        analysis
    )

    print(
        f"\nMetrics saved to: "
        f"{metrics_output_path}"
    )

    print(
        "Diagnostic cases saved to: "
        f"{cases_output_path}"
    )

    return analysis


def parse_arguments() -> argparse.Namespace:
    """Parse evidence-impact analysis paths."""

    parser = argparse.ArgumentParser(
        description=(
            "Analyze how lexical evidence "
            "verification changes prototype "
            "selective-QA routing."
        )
    )

    parser.add_argument(
        "--input",
        default=str(
            DEFAULT_INPUT_PATH
        ),
        help=(
            "JSONL file containing final "
            "decision-engine predictions."
        ),
    )

    parser.add_argument(
        "--metrics-output",
        default=str(
            DEFAULT_METRICS_OUTPUT_PATH
        ),
        help=(
            "JSON output path for aggregate "
            "evidence-impact metrics."
        ),
    )

    parser.add_argument(
        "--cases-output",
        default=str(
            DEFAULT_CASES_OUTPUT_PATH
        ),
        help=(
            "JSONL output path for "
            "per-prediction diagnostic cases."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_analysis(
        input_path=args.input,
        metrics_output_path=(
            args.metrics_output
        ),
        cases_output_path=(
            args.cases_output
        ),
    )