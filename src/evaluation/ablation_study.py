"""
Run component ablations for the prototype hybrid evidence verifier.

The hybrid verifier combines three weighted signals:

1. calibrated QA confidence,
2. lexical evidence support,
3. semantic NLI entailment.

It also uses semantic NLI contradiction as a rejection override.

This module evaluates four fixed configurations:

- full_hybrid
- without_confidence
- without_lexical
- without_semantic

When one weighted component is removed, the remaining positive weights are
renormalized by the shared hybrid-score implementation.

The `without_semantic` ablation removes both:

- semantic entailment from the weighted score, and
- the contradiction-based semantic rejection override.

This is intentional because entailment and contradiction originate from the
same semantic NLI component.

Important
---------
This is an operating-point ablation of the earlier rule-based hybrid prototype.

All configurations use the same support thresholds, but their resulting
coverage can differ. Therefore direct changes in selective risk or selective
accuracy are NOT matched-coverage effects.

For example, a lower risk can partly result from answering fewer examples.

These results should not be mixed with the project's final score-ranking/AURC
ablation, which compares methods at matched coverage and across complete
risk-coverage curves.

Underlying answer correctness follows the prototype evaluator's relaxed rule:

    Exact Match == 1
        OR
    token F1 >= relaxed_f1_threshold

with a default relaxed F1 threshold of 0.80.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from src.evaluation.evaluate_hybrid_verifier import (
    DEFAULT_RELAXED_F1_THRESHOLD,
    evaluate_system,
)
from src.utils.io import (
    load_jsonl,
    save_json,
)
from src.verification.hybrid_verifier import (
    DEFAULT_CONFIDENCE_WEIGHT,
    DEFAULT_CONTRADICTION_THRESHOLD,
    DEFAULT_LEXICAL_WEIGHT,
    DEFAULT_SEMANTIC_WEIGHT,
    DEFAULT_SUPPORTED_THRESHOLD,
    DEFAULT_WEAK_THRESHOLD,
    calculate_hybrid_score,
    classify_hybrid_support,
    get_calibrated_confidence,
    get_contradiction_probability,
    get_entailment_probability,
    get_lexical_score,
)

DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/calibration_with_hybrid_evidence.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "outputs/tables/hybrid_ablation_study.json"
)


ABLATION_CONFIGS: dict[
    str,
    dict[str, float | bool],
] = {
    "full_hybrid": {
        "confidence_weight": (
            DEFAULT_CONFIDENCE_WEIGHT
        ),
        "lexical_weight": (
            DEFAULT_LEXICAL_WEIGHT
        ),
        "semantic_weight": (
            DEFAULT_SEMANTIC_WEIGHT
        ),
        "use_semantic_contradiction": True,
    },
    "without_confidence": {
        "confidence_weight": 0.0,
        "lexical_weight": (
            DEFAULT_LEXICAL_WEIGHT
        ),
        "semantic_weight": (
            DEFAULT_SEMANTIC_WEIGHT
        ),
        "use_semantic_contradiction": True,
    },
    "without_lexical": {
        "confidence_weight": (
            DEFAULT_CONFIDENCE_WEIGHT
        ),
        "lexical_weight": 0.0,
        "semantic_weight": (
            DEFAULT_SEMANTIC_WEIGHT
        ),
        "use_semantic_contradiction": True,
    },
    "without_semantic": {
        "confidence_weight": (
            DEFAULT_CONFIDENCE_WEIGHT
        ),
        "lexical_weight": (
            DEFAULT_LEXICAL_WEIGHT
        ),
        "semantic_weight": 0.0,
        "use_semantic_contradiction": False,
    },
}


REQUIRED_CONFIGURATION_FIELDS = {
    "confidence_weight",
    "lexical_weight",
    "semantic_weight",
    "use_semantic_contradiction",
}


def validate_runtime_settings(
    relaxed_f1_threshold: float,
) -> None:
    """Validate experiment-level evaluation settings."""

    if (
        not math.isfinite(
            relaxed_f1_threshold
        )
        or not (
            0.0
            <= relaxed_f1_threshold
            <= 1.0
        )
    ):
        raise ValueError(
            "relaxed_f1_threshold must be "
            "finite and between 0 and 1."
        )


def validate_configuration(
    configuration_name: str,
    configuration: dict[str, Any],
) -> None:
    """
    Validate one hybrid ablation configuration.

    At least one weighted signal must remain active.
    """

    missing_fields = (
        REQUIRED_CONFIGURATION_FIELDS
        - configuration.keys()
    )

    if missing_fields:
        raise ValueError(
            f"Ablation configuration "
            f"{configuration_name!r} is missing "
            f"fields: {sorted(missing_fields)}."
        )

    weight_fields = (
        "confidence_weight",
        "lexical_weight",
        "semantic_weight",
    )

    weights: list[float] = []

    for field_name in weight_fields:
        value = configuration[
            field_name
        ]

        if isinstance(value, bool):
            raise ValueError(  # noqa: TRY004
                f"{configuration_name!r}: "
                f"{field_name} must be numeric, "
                f"not Boolean."
            )

        try:
            weight = float(value)

        except (
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                f"{configuration_name!r}: "
                f"invalid {field_name}: "
                f"{value!r}."
            ) from error

        if not math.isfinite(
            weight
        ):
            raise ValueError(
                f"{configuration_name!r}: "
                f"{field_name} must be finite."
            )

        if weight < 0.0:
            raise ValueError(
                f"{configuration_name!r}: "
                f"{field_name} cannot be negative."
            )

        weights.append(
            weight
        )

    if not any(
        weight > 0.0
        for weight in weights
    ):
        raise ValueError(
            f"{configuration_name!r} disables "
            "all weighted hybrid signals."
        )

    contradiction_flag = configuration[
        "use_semantic_contradiction"
    ]

    if not isinstance(
        contradiction_flag,
        bool,
    ):
        raise ValueError(  # noqa: TRY004
            f"{configuration_name!r}: "
            "use_semantic_contradiction "
            "must be Boolean."
        )


def validate_ablation_configs() -> None:
    """Validate every predefined hybrid ablation configuration."""

    required_names = {
        "full_hybrid",
        "without_confidence",
        "without_lexical",
        "without_semantic",
    }

    missing_names = (
        required_names
        - ABLATION_CONFIGS.keys()
    )

    if missing_names:
        raise ValueError(
            "Missing required ablation "
            f"configurations: "
            f"{sorted(missing_names)}."
        )

    for (
        configuration_name,
        configuration,
    ) in ABLATION_CONFIGS.items():
        validate_configuration(
            configuration_name=(
                configuration_name
            ),
            configuration=configuration,
        )


def validate_predictions(
    predictions: list[dict[str, Any]],
) -> None:
    """
    Validate prediction records required by the hybrid ablation.

    The check verifies both basic QA metadata and all signals required to
    recompute the hybrid score.

    The shared hybrid-verifier accessors perform their own detailed numeric
    validation, so malformed or missing scores fail before the ablation study
    begins.
    """

    if not predictions:
        raise ValueError(
            "Prediction list cannot be empty."
        )

    required_fields = {
        "prediction_text",
        "is_answerable",
    }

    for index, prediction in enumerate(
        predictions,
        start=1,
    ):
        missing_fields = (
            required_fields
            - prediction.keys()
        )

        if missing_fields:
            raise ValueError(
                f"Prediction {index} is missing "
                "required fields: "
                f"{sorted(missing_fields)}."
            )

        try:
            get_calibrated_confidence(
                prediction
            )

            get_lexical_score(
                prediction
            )

            get_entailment_probability(
                prediction
            )

            get_contradiction_probability(
                prediction
            )

        except (
            ValueError,
            TypeError,
            KeyError,
        ) as error:
            raise ValueError(
                f"Prediction {index} contains "
                "invalid hybrid-verifier input: "
                f"{error}"
            ) from error


def calculate_ablation_support(
    prediction: dict[str, Any],
    confidence_weight: float,
    lexical_weight: float,
    semantic_weight: float,
    use_semantic_contradiction: bool,
) -> tuple[float, str]:
    """
    Recompute hybrid score and support label for one ablation configuration.

    The weighted score contains:

        calibrated confidence
        lexical evidence score
        semantic entailment probability

    Semantic contradiction is not another weighted score. When enabled, it is
    passed to the hybrid classifier as a rejection override.

    Weight normalization is handled by `calculate_hybrid_score`.
    """

    confidence = (
        get_calibrated_confidence(
            prediction
        )
    )

    lexical_score = (
        get_lexical_score(
            prediction
        )
    )

    entailment_probability = (
        get_entailment_probability(
            prediction
        )
    )

    if use_semantic_contradiction:
        contradiction_probability = (
            get_contradiction_probability(
                prediction
            )
        )

    else:
        contradiction_probability = 0.0

    hybrid_score = (
        calculate_hybrid_score(
            calibrated_confidence=(
                confidence
            ),
            lexical_score=(
                lexical_score
            ),
            entailment_probability=(
                entailment_probability
            ),
            confidence_weight=(
                confidence_weight
            ),
            lexical_weight=(
                lexical_weight
            ),
            semantic_weight=(
                semantic_weight
            ),
        )
    )

    support_label = (
        classify_hybrid_support(
            hybrid_score=(
                hybrid_score
            ),
            contradiction_probability=(
                contradiction_probability
            ),
            supported_threshold=(
                DEFAULT_SUPPORTED_THRESHOLD
            ),
            weak_threshold=(
                DEFAULT_WEAK_THRESHOLD
            ),
            contradiction_threshold=(
                DEFAULT_CONTRADICTION_THRESHOLD
            ),
        )
    )

    return (
        hybrid_score,
        support_label,
    )


def evaluate_ablation(
    predictions: list[dict[str, Any]],
    configuration: dict[str, Any],
    relaxed_f1_threshold: float,
) -> dict[str, Any]:
    """
    Evaluate one fixed hybrid component ablation.

    The hybrid score and support label are recomputed for every record.

    Only SUPPORTED predictions are answered.

    WEAK and UNSUPPORTED predictions are treated as rejections by the prototype
    evaluator.

    Risk and accuracy therefore describe one operating point and can change
    partly because coverage changes.
    """

    validate_configuration(
        configuration_name=(
            "runtime_configuration"
        ),
        configuration=configuration,
    )

    answer_decisions: list[
        bool
    ] = []

    support_counts = {
        "SUPPORTED": 0,
        "WEAK": 0,
        "UNSUPPORTED": 0,
    }

    scores: list[
        float
    ] = []

    for prediction in predictions:
        (
            score,
            support,
        ) = calculate_ablation_support(
            prediction=prediction,
            confidence_weight=float(
                configuration[
                    "confidence_weight"
                ]
            ),
            lexical_weight=float(
                configuration[
                    "lexical_weight"
                ]
            ),
            semantic_weight=float(
                configuration[
                    "semantic_weight"
                ]
            ),
            use_semantic_contradiction=bool(
                configuration[
                    "use_semantic_contradiction"
                ]
            ),
        )

        if support not in support_counts:
            raise RuntimeError(
                "Hybrid classifier returned "
                f"unexpected label: {support!r}."
            )

        scores.append(
            score
        )

        support_counts[
            support
        ] += 1

        answer_decisions.append(
            support == "SUPPORTED"
        )

    metrics = evaluate_system(
        predictions=predictions,
        answer_decisions=(
            answer_decisions
        ),
        relaxed_f1_threshold=(
            relaxed_f1_threshold
        ),
    )

    metrics[
        "average_ablation_score"
    ] = (
        sum(scores)
        / len(scores)
    )

    metrics[
        "support_distribution"
    ] = support_counts

    return metrics


def validate_full_hybrid_consistency(
    predictions: list[dict[str, Any]],
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """
    Compare recomputed full-hybrid values with stored hybrid outputs when present.

    This check is diagnostic only. Historical prediction artifacts may not store
    hybrid score/support fields at all, in which case those comparisons are
    reported as unavailable.

    A mismatch indicates that the stored artifact and current hybrid-verifier
    implementation are not equivalent and should be investigated before using
    the ablation output.
    """

    full_configuration = (
        ABLATION_CONFIGS[
            "full_hybrid"
        ]
    )

    checked_scores = 0
    score_mismatches = 0

    checked_labels = 0
    label_mismatches = 0

    for prediction in predictions:
        (
            recomputed_score,
            recomputed_label,
        ) = calculate_ablation_support(
            prediction=prediction,
            confidence_weight=float(
                full_configuration[
                    "confidence_weight"
                ]
            ),
            lexical_weight=float(
                full_configuration[
                    "lexical_weight"
                ]
            ),
            semantic_weight=float(
                full_configuration[
                    "semantic_weight"
                ]
            ),
            use_semantic_contradiction=bool(
                full_configuration[
                    "use_semantic_contradiction"
                ]
            ),
        )

        stored_score = (
            prediction.get(
                "hybrid_evidence_score"
            )
        )

        if stored_score is not None:
            try:
                stored_score_value = float(
                    stored_score
                )

            except (
                TypeError,
                ValueError,
            ) as error:
                raise ValueError(
                    "Stored hybrid_evidence_score "
                    f"is non-numeric: "
                    f"{stored_score!r}."
                ) from error

            if not math.isfinite(
                stored_score_value
            ):
                raise ValueError(
                    "Stored hybrid_evidence_score "
                    "must be finite."
                )

            checked_scores += 1

            if not math.isclose(
                recomputed_score,
                stored_score_value,
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                score_mismatches += 1

        stored_label = (
            prediction.get(
                "hybrid_evidence_support"
            )
        )

        if stored_label is not None:
            normalized_stored_label = (
                str(stored_label)
                .strip()
                .upper()
            )

            checked_labels += 1

            if (
                normalized_stored_label
                != recomputed_label
            ):
                label_mismatches += 1

    return {
        "score_records_checked": (
            checked_scores
        ),
        "score_mismatches": (
            score_mismatches
        ),
        "label_records_checked": (
            checked_labels
        ),
        "label_mismatches": (
            label_mismatches
        ),
        "tolerance": tolerance,
    }


def build_comparison(
    results: dict[
        str,
        dict[str, Any],
    ],
) -> dict[str, Any]:
    """
    Compare each operating-point ablation with the full hybrid system.

    These differences are descriptive operating-point changes.

    Because coverage can change between configurations, `risk_change_vs_full`
    and `accuracy_change_vs_full` must not be interpreted as matched-coverage
    effects.
    """

    if "full_hybrid" not in results:
        raise ValueError(
            "Ablation results must contain "
            "'full_hybrid'."
        )

    full_metrics = results[
        "full_hybrid"
    ]

    comparison: dict[
        str,
        Any,
    ] = {}

    for (
        name,
        metrics,
    ) in results.items():
        comparison[
            name
        ] = {
            "coverage": (
                metrics[
                    "coverage"
                ]
            ),
            "selective_accuracy": (
                metrics[
                    "selective_accuracy"
                ]
            ),
            "selective_risk": (
                metrics[
                    "selective_risk"
                ]
            ),
            "wrong_answered": (
                metrics[
                    "wrong_answered"
                ]
            ),
            "correct_rejected": (
                metrics[
                    "correct_rejected"
                ]
            ),
            "coverage_change_vs_full": (
                metrics[
                    "coverage"
                ]
                - full_metrics[
                    "coverage"
                ]
            ),
            "risk_change_vs_full": (
                metrics[
                    "selective_risk"
                ]
                - full_metrics[
                    "selective_risk"
                ]
            ),
            "accuracy_change_vs_full": (
                metrics[
                    "selective_accuracy"
                ]
                - full_metrics[
                    "selective_accuracy"
                ]
            ),
        }

    return comparison


def run_ablation_study(
    input_path: str | Path,
    output_path: str | Path,
    relaxed_f1_threshold: float,
) -> dict[str, Any]:
    """
    Run the complete hybrid component-ablation study.

    The study:

    1. validates input records and configurations,
    2. recomputes every ablation from the same underlying records,
    3. evaluates each fixed support-policy operating point,
    4. compares each ablation with the full hybrid configuration,
    5. optionally checks stored full-hybrid artifact consistency,
    6. saves the complete result as JSON.
    """

    validate_runtime_settings(
        relaxed_f1_threshold=(
            relaxed_f1_threshold
        )
    )

    validate_ablation_configs()

    predictions = load_jsonl(
        input_path
    )

    validate_predictions(
        predictions
    )

    consistency_check = (
        validate_full_hybrid_consistency(
            predictions
        )
    )

    if (
        consistency_check[
            "score_mismatches"
        ]
        > 0
        or consistency_check[
            "label_mismatches"
        ]
        > 0
    ):
        raise RuntimeError(
            "Recomputed full-hybrid results do "
            "not match stored hybrid outputs. "
            f"Score mismatches: "
            f"{consistency_check['score_mismatches']}; "
            f"label mismatches: "
            f"{consistency_check['label_mismatches']}."
        )

    results: dict[
        str,
        dict[str, Any],
    ] = {}

    for (
        name,
        configuration,
    ) in ABLATION_CONFIGS.items():
        results[
            name
        ] = evaluate_ablation(
            predictions=(
                predictions
            ),
            configuration=(
                configuration
            ),
            relaxed_f1_threshold=(
                relaxed_f1_threshold
            ),
        )

    comparison = (
        build_comparison(
            results
        )
    )

    output = {
        "evaluation_type": (
            "hybrid_component_ablation"
        ),
        "analysis_scope": (
            "prototype fixed-threshold "
            "operating-point ablation"
        ),
        "input_path": (
            str(input_path)
        ),
        "total_predictions": (
            len(predictions)
        ),
        "relaxed_f1_threshold": (
            relaxed_f1_threshold
        ),
        "correctness_definition": (
            "answerable prediction is correct "
            "when exact match == 1 or token F1 "
            ">= relaxed_f1_threshold"
        ),
        "comparison_note": (
            "Risk and accuracy changes are "
            "operating-point differences. "
            "Coverage can differ across ablations; "
            "these are not matched-coverage effects."
        ),
        "semantic_ablation_note": (
            "without_semantic removes semantic "
            "entailment from the weighted score "
            "and disables the contradiction "
            "override because both originate "
            "from the semantic NLI component."
        ),
        "hybrid_thresholds": {
            "supported_threshold": (
                DEFAULT_SUPPORTED_THRESHOLD
            ),
            "weak_threshold": (
                DEFAULT_WEAK_THRESHOLD
            ),
            "contradiction_threshold": (
                DEFAULT_CONTRADICTION_THRESHOLD
            ),
        },
        "configurations": (
            ABLATION_CONFIGS
        ),
        "full_hybrid_consistency_check": (
            consistency_check
        ),
        "results": results,
        "comparison": comparison,
    }

    save_json(
        output,
        output_path,
    )

    print(
        "\nHybrid verifier ablation "
        "study completed."
    )

    print(
        f"Total predictions: "
        f"{len(predictions)}"
    )

    print(
        "\n"
        f"{'Configuration':<22}"
        f"{'Coverage':>10}"
        f"{'Accuracy':>12}"
        f"{'Risk':>10}"
        f"{'Wrong':>8}"
        f"{'Correct rej.':>14}"
    )

    print(
        "-" * 76
    )

    for (
        name,
        metrics,
    ) in results.items():
        print(
            f"{name:<22}"
            f"{metrics['coverage']:>10.4f}"
            f"{metrics['selective_accuracy']:>12.4f}"
            f"{metrics['selective_risk']:>10.4f}"
            f"{metrics['wrong_answered']:>8}"
            f"{metrics['correct_rejected']:>14}"
        )

    print(
        "\nOperating-point change "
        "relative to full hybrid "
        "(coverage may differ):"
    )

    for (
        name,
        metrics,
    ) in comparison.items():
        if name == "full_hybrid":
            continue

        print(
            f"{name}: "
            "coverage Δ="
            f"{metrics['coverage_change_vs_full']:+.4f} | "
            "risk Δ="
            f"{metrics['risk_change_vs_full']:+.4f} | "
            "accuracy Δ="
            f"{metrics['accuracy_change_vs_full']:+.4f}"
        )

    print(
        "\nFull-hybrid artifact consistency:"
    )

    print(
        "Stored scores checked: "
        f"{consistency_check['score_records_checked']} | "
        "mismatches: "
        f"{consistency_check['score_mismatches']}"
    )

    print(
        "Stored labels checked: "
        f"{consistency_check['label_records_checked']} | "
        "mismatches: "
        f"{consistency_check['label_mismatches']}"
    )

    print(
        f"\nResults saved to: "
        f"{output_path}"
    )

    return output


def parse_arguments() -> argparse.Namespace:
    """Parse hybrid component-ablation settings."""

    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-policy component "
            "ablations for the prototype "
            "hybrid evidence verifier."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=(
            DEFAULT_INPUT_PATH
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_OUTPUT_PATH
        ),
    )

    parser.add_argument(
        "--relaxed-f1-threshold",
        type=float,
        default=(
            DEFAULT_RELAXED_F1_THRESHOLD
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the hybrid component ablation from command-line arguments."""

    args = parse_arguments()

    run_ablation_study(
        input_path=args.input,
        output_path=args.output,
        relaxed_f1_threshold=(
            args.relaxed_f1_threshold
        ),
    )


if __name__ == "__main__":
    main()