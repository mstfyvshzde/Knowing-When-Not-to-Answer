"""
An ablation study is an experiment where you remove or disable one part of a system to see how important that part really is.

In this project, for example:
Full system:
Confidence + Lexical + Semantic + Contradiction
Then test versions like:
Ablation 1:
Confidence + Lexical
(no Semantic)

Run component ablations for the hybrid evidence verifier.

The study measures how selective-QA performance changes when confidence,
lexical evidence, or semantic evidence is removed from the hybrid verifier.

All remaining component weights are automatically normalized so that each
ablation differs from the full hybrid system only by the removed signal.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from src.evaluation.evaluate_hybrid_verifier import (
    DEFAULT_RELAXED_F1_THRESHOLD,
    evaluate_system,
)
from src.utils.io import load_jsonl
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


ABLATION_CONFIGS = {
    "full_hybrid": {
        "confidence_weight": DEFAULT_CONFIDENCE_WEIGHT,
        "lexical_weight": DEFAULT_LEXICAL_WEIGHT,
        "semantic_weight": DEFAULT_SEMANTIC_WEIGHT,
        "use_semantic_contradiction": True
    },
    "without_confidence": {
        "confidence_weight": 0.0,
        "lexical_weight": DEFAULT_LEXICAL_WEIGHT,
        "semantic_weight": DEFAULT_SEMANTIC_WEIGHT,
        "use_semantic_contradiction": True
    },
    "without_lexical": {
        "confidence_weight": DEFAULT_CONFIDENCE_WEIGHT,
        "lexical_weight": 0.0,
        "semantic_weight": DEFAULT_SEMANTIC_WEIGHT,
        "use_semantic_contradiction": True
    },
    "without_semantic": {
        "confidence_weight": DEFAULT_CONFIDENCE_WEIGHT,
        "lexical_weight": DEFAULT_LEXICAL_WEIGHT,
        "semantic_weight": 0.0,
        "use_semantic_contradiction": False
    }
}


# check predictions exist -> verify each one has prediction_text and is_answerable -> raise an error if anything is missing.
def validate_predictions(
    predictions: list[dict[str, Any]]
) -> None:
    if not predictions:
        raise ValueError(
            "Prediction list cannot be empty."
        )

    required_fields = {
        "prediction_text",
        "is_answerable"
    }

    for index, prediction in enumerate(
        predictions,
        start=1
    ):
        missing_fields = (
            required_fields - prediction.keys()
        )

        if missing_fields:
            missing_text = ", ".join(
                sorted(missing_fields)
            )

            raise ValueError(
                f"Prediction {index} is missing "
                f"required fields: {missing_text}"
            )


# get the three signals -> optionally include contradiction -> calculate a new hybrid score -> classify it as SUPPORTED / WEAK / UNSUPPORTED.
def calculate_ablation_support(
    prediction: dict[str, Any],
    confidence_weight: float,
    lexical_weight: float,
    semantic_weight: float,
    use_semantic_contradiction: bool
) -> tuple[float, str]:
    confidence = get_calibrated_confidence(
        prediction
    )

    lexical_score = get_lexical_score(
        prediction
    )

    entailment_probability = (
        get_entailment_probability(
            prediction
        )
    )

    contradiction_probability = (
        get_contradiction_probability(
            prediction
        )
        if use_semantic_contradiction
        else 0.0
    )

    hybrid_score = calculate_hybrid_score(
        calibrated_confidence=confidence,
        lexical_score=lexical_score,
        entailment_probability=(
            entailment_probability
        ),
        confidence_weight=confidence_weight,
        lexical_weight=lexical_weight,
        semantic_weight=semantic_weight
    )

    support_label = classify_hybrid_support(
        hybrid_score=hybrid_score,
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
        )
    )

    return (
        hybrid_score,
        support_label,
    )


# apply one ablation setup -> recompute SUPPORTED / WEAK / UNSUPPORTED -> answer only SUPPORTED cases -> evaluate accuracy/risk/coverage -> return the metrics.
# Ablation means removing or disabling one part of a system to test how important that part is.
def evaluate_ablation(
    predictions: list[dict[str, Any]],
    configuration: dict[str, Any],
    relaxed_f1_threshold: float
) -> dict[str, Any]:
    answer_decisions: list[bool] = []

    support_counts = {
        "SUPPORTED": 0,
        "WEAK": 0,
        "UNSUPPORTED": 0
    }

    scores: list[float] = []

    for prediction in predictions:
        score, support = (
            calculate_ablation_support(
                prediction=prediction,
                confidence_weight=(
                    configuration[
                        "confidence_weight"
                    ]
                ),
                lexical_weight=(
                    configuration[
                        "lexical_weight"
                    ]
                ),
                semantic_weight=(
                    configuration[
                        "semantic_weight"
                    ]
                ),
                use_semantic_contradiction=(
                    configuration[
                        "use_semantic_contradiction"
                    ]
                )
            )
        )

        scores.append(score)

        support_counts[support] += 1

        answer_decisions.append(
            support == "SUPPORTED"
        )

    metrics = evaluate_system(
        predictions=predictions,
        answer_decisions=answer_decisions,
        relaxed_f1_threshold=(
            relaxed_f1_threshold
        )
    )

    metrics["average_ablation_score"] = (
        sum(scores) / len(scores)
        if scores
        else 0.0
    )

    metrics["support_distribution"] = (
        support_counts
    )

    return metrics


# take all ablation results -> use full_hybrid as the baseline -> calculate how much each configuration improves or worsens coverage, risk, and accuracy
def build_comparison(
    results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    full_metrics = results["full_hybrid"]

    comparison: dict[str, Any] = {}

    for name, metrics in results.items():
        comparison[name] = {
            "coverage": metrics["coverage"],
            "selective_accuracy": (
                metrics["selective_accuracy"]
            ),
            "selective_risk": (
                metrics["selective_risk"]
            ),
            "wrong_answered": (
                metrics["wrong_answered"]
            ),
            "correct_rejected": (
                metrics["correct_rejected"]
            ),
            "coverage_change_vs_full": (
                metrics["coverage"]
                - full_metrics["coverage"]
            ),
            "risk_change_vs_full": (
                metrics["selective_risk"]
                - full_metrics[
                    "selective_risk"
                ]
            ),
            "accuracy_change_vs_full": (
                metrics["selective_accuracy"]
                - full_metrics[
                    "selective_accuracy"
                ]
            )
        }

    return comparison



# load predictions -> test every ablation configuration -> compare each against full_hybrid -> measure coverage/accuracy/risk changes -> save and print the results.
def run_ablation_study(
    input_path: str | Path,
    output_path: str | Path,
    relaxed_f1_threshold: float
) -> dict[str, Any]:
    predictions = load_jsonl(
        input_path
    )

    validate_predictions(
        predictions
    )

    results: dict[
        str,
        dict[str, Any]
    ] = {}

    for name, configuration in (
        ABLATION_CONFIGS.items()
    ):
        results[name] = evaluate_ablation(
            predictions=predictions,
            configuration=configuration,
            relaxed_f1_threshold=(
                relaxed_f1_threshold
            )
        )

    comparison = build_comparison(
        results
    )

    output = {
        "evaluation_type": (
            "hybrid_component_ablation"
        ),
        "input_path": str(input_path),
        "total_predictions": len(predictions),
        "relaxed_f1_threshold": (
            relaxed_f1_threshold
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
            )
        },
        "configurations": (
            ABLATION_CONFIGS
        ),
        "results": results,
        "comparison": comparison
    }

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as output_file:
        json.dump(
            output,
            output_file,
            indent=2,
            ensure_ascii=False
        )

    print(
        "\nHybrid verifier ablation study completed."
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
        f"{'Rejected':>12}"
    )

    print("-" * 74)

    for name, metrics in results.items():
        print(
            f"{name:<22}"
            f"{metrics['coverage']:>10.4f}"
            f"{metrics[
                'selective_accuracy'
            ]:>12.4f}"
            f"{metrics[
                'selective_risk'
            ]:>10.4f}"
            f"{metrics[
                'wrong_answered'
            ]:>8}"
            f"{metrics[
                'correct_rejected'
            ]:>12}"
        )

    print(
        "\nChange relative to full hybrid:"
    )

    for name, metrics in (
        comparison.items()
    ):
        if name == "full_hybrid":
            continue

        print(
            f"{name}: "
            f"coverage Δ="
            f"{metrics[
                'coverage_change_vs_full'
            ]:+.4f} | "
            f"risk Δ="
            f"{metrics[
                'risk_change_vs_full'
            ]:+.4f} | "
            f"accuracy Δ="
            f"{metrics[
                'accuracy_change_vs_full'
            ]:+.4f}"
        )

    print(
        f"\nResults saved to: "
        f"{output_path}"
    )

    return output



def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run component ablations for "
            "the hybrid evidence verifier."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH
    )

    parser.add_argument(
        "--relaxed-f1-threshold",
        type=float,
        default=(
            DEFAULT_RELAXED_F1_THRESHOLD
        )
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    run_ablation_study(
        input_path=args.input,
        output_path=args.output,
        relaxed_f1_threshold=(
            args.relaxed_f1_threshold
        )
    )


if __name__ == "__main__":
    main()