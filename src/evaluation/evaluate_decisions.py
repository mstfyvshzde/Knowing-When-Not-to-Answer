"""
Evaluate the prototype ANSWER / VERIFY / ABSTAIN decision policy.

This module analyzes the earlier rule-based selective-QA pipeline after
confidence-threshold routing and lexical evidence verification.

Pipeline:
    calibrated QA confidence
        -> threshold routing: ANSWER / VERIFY / ABSTAIN
        -> lexical evidence verification
        -> final routing: ANSWER / VERIFY / ABSTAIN

Important
---------
Underlying QA correctness is independent of later routing actions. This module
therefore reuses `calibration_metrics.is_prediction_correct`, the same
forced-answer correctness definition used during confidence-threshold
selection.

Under that definition:
- answerable examples are correct only when the predicted answer exactly
  matches a reference after project normalization;
- unanswerable forced-answer candidates are incorrect.

Accordingly, VERIFY/ABSTAIN group accuracy describes the underlying QA
candidate quality inside those groups. It does not measure whether VERIFY or
ABSTAIN itself was the correct action.

Selective risk is defined only among final ANSWER predictions:

    selective_risk = 1 - answer_accuracy

This is an operating-point analysis of an earlier rule-based prototype, not
the project's final score-ranking/AURC evaluation.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

from src.calibration.calibration_metrics import is_prediction_correct
from src.evaluation.metrics import parse_answerability
from src.utils.io import load_jsonl, save_json

DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/calibration_final_decisions.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "outputs/tables/final_decision_metrics.json"
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


def normalize_boolean(
    value: Any,
) -> bool:
    """
    Convert common stored Boolean representations into a strict Boolean.

    Retained for compatibility with historical callers. QA correctness in this
    evaluator does not depend on stored correctness fields.
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
        normalized = (
            value.strip().lower()
        )

        if normalized in {
            "true",
            "1",
            "correct",
            "yes",
        }:
            return True

        if normalized in {
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


def extract_reference_answers(
    prediction: dict[str, Any],
) -> list[str]:
    """
    Extract reference-answer strings from supported historical formats.

    The extracted list is also used to canonicalize input before calling the
    shared calibration correctness function.
    """

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
        for field in (
            "text",
            "answers",
            "answer_text",
        ):
            values = (
                reference_answers.get(
                    field
                )
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
                extracted: list[
                    str
                ] = []

                for value in values:
                    if isinstance(
                        value,
                        dict,
                    ):
                        answer_text = (
                            value.get(
                                "text"
                            )
                            or value.get(
                                "answer"
                            )
                            or value.get(
                                "answer_text"
                            )
                        )

                        if (
                            answer_text
                            is not None
                        ):
                            extracted.append(
                                str(
                                    answer_text
                                )
                            )

                    elif (
                        value
                        is not None
                    ):
                        extracted.append(
                            str(
                                value
                            )
                        )

                return extracted

        return []

    if isinstance(
        reference_answers,
        list,
    ):
        extracted: list[
            str
        ] = []

        for item in (
            reference_answers
        ):
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
                    item.get(
                        "text"
                    )
                    or item.get(
                        "answer"
                    )
                    or item.get(
                        "answer_text"
                    )
                )

                if (
                    answer_text
                    is not None
                ):
                    extracted.append(
                        str(
                            answer_text
                        )
                    )

            elif item is not None:
                extracted.append(
                    str(
                        item
                    )
                )

        return extracted

    return [
        str(
            reference_answers
        )
    ]


def validate_correctness_inputs(
    prediction: dict[str, Any],
) -> None:
    """
    Validate fields required by forced-answer QA correctness.

    Strict answerability parsing prevents values such as the string "False"
    from being interpreted via normal Python truthiness.
    """

    required_fields = {
        "prediction_text",
        "reference_answers",
        "is_answerable",
    }

    missing_fields = (
        required_fields
        - prediction.keys()
    )

    if missing_fields:
        raise ValueError(
            "Missing correctness fields: "
            f"{sorted(missing_fields)}."
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

    references = (
        extract_reference_answers(
            prediction
        )
    )

    non_empty_references = [
        reference
        for reference
        in references
        if str(
            reference
        ).strip()
    ]

    if (
        is_answerable
        and not non_empty_references
    ):
        raise ValueError(
            "Answerable prediction does not "
            "contain a usable reference answer."
        )

    if (
        not is_answerable
        and non_empty_references
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

    `decision`, `final_decision`, and stored correctness fields are deliberately
    ignored. The record is first canonicalized, then evaluated using the same
    correctness function used during threshold selection.
    """

    validate_correctness_inputs(
        prediction
    )

    prediction_text = (
        prediction.get(
            "prediction_text",
            "",
        )
    )

    canonical_prediction = dict(
        prediction
    )

    canonical_prediction[
        "is_answerable"
    ] = parse_answerability(
        prediction[
            "is_answerable"
        ]
    )

    canonical_prediction[
        "prediction_text"
    ] = (
        ""
        if prediction_text is None
        else str(
            prediction_text
        )
    )

    canonical_prediction[
        "reference_answers"
    ] = extract_reference_answers(
        prediction
    )

    return bool(
        is_prediction_correct(
            canonical_prediction
        )
    )


def normalize_decision_value(
    value: Any,
    field_name: str,
) -> str:
    """Normalize and validate one routing decision."""

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
            f"Invalid {field_name} value: "
            f"{decision!r}."
        )

    return decision


def get_final_decision(
    prediction: dict[str, Any],
) -> str:
    """Retrieve and validate the final rule-based routing decision."""

    value = prediction.get(
        "final_decision"
    )

    if value is None:
        raise ValueError(
            "Prediction does not contain "
            "final_decision."
        )

    return normalize_decision_value(
        value,
        "final_decision",
    )


def get_threshold_decision(
    prediction: dict[str, Any],
) -> str | None:
    """
    Retrieve the original confidence-threshold routing decision.

    Current threshold-selection artifacts store this value in `decision`.
    Older explicit aliases are retained for compatibility. If multiple aliases
    exist, they must agree.
    """

    fields = (
        "threshold_decision",
        "confidence_decision",
        "selective_decision",
        "decision",
    )

    observed: dict[
        str,
        str,
    ] = {}

    for field in fields:
        value = prediction.get(
            field
        )

        if value is None:
            continue

        observed[
            field
        ] = normalize_decision_value(
            value,
            field,
        )

    if not observed:
        return None

    unique_decisions = set(
        observed.values()
    )

    if (
        len(
            unique_decisions
        )
        != 1
    ):
        raise ValueError(
            "Conflicting threshold-decision "
            f"fields: {observed}."
        )

    return next(
        iter(
            unique_decisions
        )
    )


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

    support = (
        str(value)
        .strip()
        .upper()
    )

    if (
        support
        not in VALID_EVIDENCE_LABELS
    ):
        raise ValueError(
            "Invalid evidence_support value: "
            f"{support!r}."
        )

    return support


def validate_transition(
    threshold_decision: str,
    evidence_support: str,
    final_decision: str,
) -> None:
    """
    Verify that a record obeys the rule-based decision engine.

    Evidence changes only records originally routed to VERIFY.
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
            "Decision transition is "
            "inconsistent with the prototype "
            "policy: "
            f"threshold={threshold_decision}, "
            f"evidence={evidence_support}, "
            f"expected_final={expected_final}, "
            f"observed_final={final_decision}."
        )


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


def evaluate_decision_group(
    predictions: list[dict[str, Any]],
    decision: str,
) -> dict[str, Any]:
    """
    Describe one final routing group.

    `accuracy` is underlying forced-answer QA candidate accuracy in the group.
    Only the ANSWER-group error rate is interpreted as selective risk.
    """

    normalized_decision = (
        normalize_decision_value(
            decision,
            "decision group",
        )
    )

    selected_predictions = [
        prediction
        for prediction
        in predictions
        if (
            get_final_decision(
                prediction
            )
            == normalized_decision
        )
    ]

    count = len(
        selected_predictions
    )

    total = len(
        predictions
    )

    correct_count = sum(
        int(
            get_correctness(
                prediction
            )
        )
        for prediction
        in selected_predictions
    )

    incorrect_count = (
        count
        - correct_count
    )

    accuracy = safe_divide(
        correct_count,
        count,
    )

    error_rate = (
        None
        if accuracy is None
        else 1.0 - accuracy
    )

    return {
        "count": count,
        "rate": safe_divide(
            count,
            total,
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
            error_rate
        ),
    }


def evaluate_threshold_policy(
    predictions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Reconstruct the original three-way confidence-threshold operating point.

    Selective risk applies only to the direct threshold ANSWER region.
    """

    threshold_predictions: list[
        dict[str, Any]
    ] = []

    for prediction in predictions:
        threshold_decision = (
            get_threshold_decision(
                prediction
            )
        )

        if (
            threshold_decision
            is None
        ):
            return None

        threshold_predictions.append(
            {
                "decision": (
                    threshold_decision
                ),
                "is_correct": (
                    get_correctness(
                        prediction
                    )
                ),
            }
        )

    total = len(
        threshold_predictions
    )

    counts = Counter(
        item[
            "decision"
        ]
        for item
        in threshold_predictions
    )

    answer_predictions = [
        item
        for item
        in threshold_predictions
        if (
            item[
                "decision"
            ]
            == "ANSWER"
        )
    ]

    answer_correct_count = sum(
        int(
            item[
                "is_correct"
            ]
        )
        for item
        in answer_predictions
    )

    answer_accuracy = (
        safe_divide(
            answer_correct_count,
            len(
                answer_predictions
            ),
        )
    )

    selective_risk = (
        None
        if answer_accuracy is None
        else 1.0 - answer_accuracy
    )

    return {
        "total": total,
        "answer_count": (
            counts[
                "ANSWER"
            ]
        ),
        "verify_count": (
            counts[
                "VERIFY"
            ]
        ),
        "abstain_count": (
            counts[
                "ABSTAIN"
            ]
        ),
        "answer_coverage": (
            safe_divide(
                counts[
                    "ANSWER"
                ],
                total,
            )
        ),
        "verify_rate": (
            safe_divide(
                counts[
                    "VERIFY"
                ],
                total,
            )
        ),
        "abstain_rate": (
            safe_divide(
                counts[
                    "ABSTAIN"
                ],
                total,
            )
        ),
        "answer_accuracy": (
            answer_accuracy
        ),
        "selective_risk": (
            selective_risk
        ),
    }


def evaluate_reason_distribution(
    predictions: list[dict[str, Any]],
) -> dict[str, int]:
    """Count decision-engine reason codes."""

    return dict(
        Counter(
            str(
                prediction.get(
                    "decision_reason",
                    "unknown",
                )
            )
            for prediction
            in predictions
        )
    )


def evaluate_evidence_distribution(
    predictions: list[dict[str, Any]],
) -> dict[str, int]:
    """Count validated lexical evidence-support labels."""

    return dict(
        Counter(
            get_evidence_support(
                prediction
            )
            for prediction
            in predictions
        )
    )


def calculate_final_metrics(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Calculate the complete prototype decision-policy evaluation.

    The final evidence-aware ANSWER set is compared with the original direct
    threshold ANSWER set. Those changes are operating-point differences, not
    matched-coverage effects.
    """

    if not predictions:
        raise ValueError(
            "Prediction list cannot be empty."
        )

    total = len(
        predictions
    )

    total_correct = sum(
        int(
            get_correctness(
                prediction
            )
        )
        for prediction
        in predictions
    )

    baseline_accuracy = (
        safe_divide(
            total_correct,
            total,
        )
    )

    answer_metrics = (
        evaluate_decision_group(
            predictions,
            "ANSWER",
        )
    )

    verify_metrics = (
        evaluate_decision_group(
            predictions,
            "VERIFY",
        )
    )

    abstain_metrics = (
        evaluate_decision_group(
            predictions,
            "ABSTAIN",
        )
    )

    threshold_policy = (
        evaluate_threshold_policy(
            predictions
        )
    )

    metrics: dict[
        str,
        Any,
    ] = {
        "total_predictions": (
            total
        ),
        "baseline_full_coverage_accuracy": (
            baseline_accuracy
        ),
        "correctness_definition": (
            "forced-answer Exact-Match "
            "correctness shared with "
            "calibration threshold selection; "
            "unanswerable candidates are incorrect"
        ),
        "final_policy": {
            "answer": (
                answer_metrics
            ),
            "verify": (
                verify_metrics
            ),
            "abstain": (
                abstain_metrics
            ),
            "answer_coverage": (
                answer_metrics[
                    "rate"
                ]
            ),
            "selective_risk": (
                answer_metrics[
                    "risk"
                ]
            ),
            "verify_rate": (
                verify_metrics[
                    "rate"
                ]
            ),
            "abstain_rate": (
                abstain_metrics[
                    "rate"
                ]
            ),
        },
        "threshold_only_policy": (
            threshold_policy
        ),
        "evidence_distribution": (
            evaluate_evidence_distribution(
                predictions
            )
        ),
        "decision_reason_distribution": (
            evaluate_reason_distribution(
                predictions
            )
        ),
    }

    if (
        threshold_policy
        is not None
    ):
        final_risk = (
            answer_metrics[
                "risk"
            ]
        )

        threshold_risk = (
            threshold_policy[
                "selective_risk"
            ]
        )

        final_coverage = (
            answer_metrics[
                "rate"
            ]
        )

        threshold_coverage = (
            threshold_policy[
                "answer_coverage"
            ]
        )

        metrics[
            "policy_comparison"
        ] = {
            "risk_change": (
                None
                if (
                    final_risk
                    is None
                    or threshold_risk
                    is None
                )
                else (
                    final_risk
                    - threshold_risk
                )
            ),
            "coverage_change": (
                None
                if (
                    final_coverage
                    is None
                    or threshold_coverage
                    is None
                )
                else (
                    final_coverage
                    - threshold_coverage
                )
            ),
            "comparison_note": (
                "Operating-point comparison "
                "only; final ANSWER coverage "
                "can differ from threshold "
                "direct-ANSWER coverage."
            ),
        }

    return metrics


def validate_metric_invariants(
    metrics: dict[str, Any],
) -> None:
    """Check structural invariants in the calculated routing metrics."""

    total = int(
        metrics[
            "total_predictions"
        ]
    )

    final_policy = (
        metrics[
            "final_policy"
        ]
    )

    answer = (
        final_policy[
            "answer"
        ]
    )

    verify = (
        final_policy[
            "verify"
        ]
    )

    abstain = (
        final_policy[
            "abstain"
        ]
    )

    routed_total = (
        int(
            answer[
                "count"
            ]
        )
        + int(
            verify[
                "count"
            ]
        )
        + int(
            abstain[
                "count"
            ]
        )
    )

    if (
        routed_total
        != total
    ):
        raise RuntimeError(
            "Final routing counts do not sum "
            "to total predictions."
        )

    rates = (
        final_policy[
            "answer_coverage"
        ],
        final_policy[
            "verify_rate"
        ],
        final_policy[
            "abstain_rate"
        ],
    )

    if any(
        rate is None
        for rate
        in rates
    ):
        raise RuntimeError(
            "Final routing rates must be "
            "defined for a non-empty dataset."
        )

    if not math.isclose(
        sum(
            float(
                rate
            )
            for rate
            in rates
        ),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Final ANSWER, VERIFY, and ABSTAIN "
            "rates do not sum to 1.0."
        )

    answer_count = int(
        answer[
            "count"
        ]
    )

    answer_accuracy = (
        answer[
            "accuracy"
        ]
    )

    selective_risk = (
        final_policy[
            "selective_risk"
        ]
    )

    if answer_count == 0:
        if (
            answer_accuracy
            is not None
            or selective_risk
            is not None
        ):
            raise RuntimeError(
                "ANSWER accuracy and selective "
                "risk must be undefined when "
                "no records are answered."
            )

        return

    if (
        answer_accuracy
        is None
        or selective_risk
        is None
    ):
        raise RuntimeError(
            "ANSWER accuracy and selective "
            "risk must be defined when ANSWER "
            "count is positive."
        )

    if not math.isclose(
        float(
            selective_risk
        ),
        1.0
        - float(
            answer_accuracy
        ),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Selective risk is inconsistent "
            "with ANSWER accuracy."
        )


def save_metrics(
    metrics: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save decision-policy metrics with shared JSON I/O."""

    save_json(
        metrics,
        output_path,
    )


def format_optional_metric(
    value: float | None,
) -> str:
    """Format an optional metric to four decimal places."""

    if value is None:
        return "N/A"

    return f"{value:.4f}"


def print_metrics(
    metrics: dict[str, Any],
) -> None:
    """Print prototype routing metrics with precise labels."""

    final_policy = (
        metrics[
            "final_policy"
        ]
    )

    answer = (
        final_policy[
            "answer"
        ]
    )

    verify = (
        final_policy[
            "verify"
        ]
    )

    abstain = (
        final_policy[
            "abstain"
        ]
    )

    print(
        "\nPrototype selective-QA "
        "evaluation completed."
    )

    print(
        f"Total predictions: "
        f"{metrics['total_predictions']}"
    )

    print(
        "Full-coverage forced-answer "
        "QA accuracy: "
        f"{format_optional_metric(
            metrics[
                'baseline_full_coverage_accuracy'
            ]
        )}"
    )

    print(
        "\nFinal evidence-aware routing:"
    )

    print(
        f"ANSWER count: "
        f"{answer['count']}"
    )

    print(
        "Answer coverage: "
        f"{format_optional_metric(
            final_policy[
                'answer_coverage'
            ]
        )}"
    )

    print(
        "ANSWER-group underlying QA "
        "accuracy: "
        f"{format_optional_metric(
            answer[
                'accuracy'
            ]
        )}"
    )

    print(
        "Selective risk: "
        f"{format_optional_metric(
            final_policy[
                'selective_risk'
            ]
        )}"
    )

    print(
        f"VERIFY count: "
        f"{verify['count']}"
    )

    print(
        "Verify rate: "
        f"{format_optional_metric(
            final_policy[
                'verify_rate'
            ]
        )}"
    )

    print(
        "VERIFY-group underlying QA "
        "accuracy: "
        f"{format_optional_metric(
            verify[
                'accuracy'
            ]
        )}"
    )

    print(
        f"ABSTAIN count: "
        f"{abstain['count']}"
    )

    print(
        "Abstain rate: "
        f"{format_optional_metric(
            final_policy[
                'abstain_rate'
            ]
        )}"
    )

    print(
        "ABSTAIN-group underlying QA "
        "accuracy: "
        f"{format_optional_metric(
            abstain[
                'accuracy'
            ]
        )}"
    )

    threshold_policy = (
        metrics.get(
            "threshold_only_policy"
        )
    )

    if (
        threshold_policy
        is not None
    ):
        print(
            "\nOriginal confidence-threshold "
            "routing:"
        )

        print(
            "Direct-ANSWER coverage: "
            f"{format_optional_metric(
                threshold_policy[
                    'answer_coverage'
                ]
            )}"
        )

        print(
            "Direct-ANSWER underlying QA "
            "accuracy: "
            f"{format_optional_metric(
                threshold_policy[
                    'answer_accuracy'
                ]
            )}"
        )

        print(
            "Direct-ANSWER selective risk: "
            f"{format_optional_metric(
                threshold_policy[
                    'selective_risk'
                ]
            )}"
        )

        print(
            "VERIFY rate: "
            f"{format_optional_metric(
                threshold_policy[
                    'verify_rate'
                ]
            )}"
        )

        print(
            "ABSTAIN rate: "
            f"{format_optional_metric(
                threshold_policy[
                    'abstain_rate'
                ]
            )}"
        )

    comparison = (
        metrics.get(
            "policy_comparison"
        )
    )

    if comparison is not None:
        print(
            "\nEvidence-aware operating-point "
            "change (not matched coverage):"
        )

        print(
            "Risk change: "
            f"{format_optional_metric(
                comparison[
                    'risk_change'
                ]
            )}"
        )

        print(
            "Coverage change: "
            f"{format_optional_metric(
                comparison[
                    'coverage_change'
                ]
            )}"
        )


def validate_predictions(
    predictions: list[dict[str, Any]],
) -> None:
    """
    Validate correctness inputs, routing labels, evidence labels, and transitions.

    Malformed or policy-inconsistent records fail before aggregate metrics are
    calculated.
    """

    if not predictions:
        raise ValueError(
            "Prediction list cannot be empty."
        )

    for index, prediction in enumerate(
        predictions,
        start=1,
    ):
        try:
            get_correctness(
                prediction
            )

            threshold_decision = (
                get_threshold_decision(
                    prediction
                )
            )

            if (
                threshold_decision
                is None
            ):
                raise ValueError(
                    "Prediction does not contain "
                    "a threshold-routing decision."
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

        except (
            ValueError,
            TypeError,
            KeyError,
        ) as error:
            raise ValueError(
                f"Prediction {index} failed "
                f"validation: {error}"
            ) from error


def run_evaluation(
    input_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Run the complete prototype decision-policy evaluation."""

    predictions = (
        load_jsonl(
            input_path
        )
    )

    validate_predictions(
        predictions
    )

    metrics = (
        calculate_final_metrics(
            predictions
        )
    )

    validate_metric_invariants(
        metrics
    )

    save_metrics(
        metrics=metrics,
        output_path=output_path,
    )

    print_metrics(
        metrics
    )

    print(
        f"\nMetrics saved to: "
        f"{output_path}"
    )

    return metrics


def parse_arguments() -> argparse.Namespace:
    """Parse prototype decision-policy evaluation paths."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate final ANSWER, VERIFY, "
            "and ABSTAIN routing decisions for "
            "the rule-based selective-QA prototype."
        )
    )

    parser.add_argument(
        "--input",
        default=str(
            DEFAULT_INPUT_PATH
        ),
        help=(
            "JSONL file containing final "
            "prototype routing decisions."
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT_PATH
        ),
        help=(
            "JSON output path for aggregate "
            "prototype decision metrics."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_evaluation(
        input_path=args.input,
        output_path=args.output,
    )