"""
Tune confidence/verifier fusion weights using calibration data only.

This experiment asks whether the verifier should receive any weight once a
strong calibrated QA-confidence signal is already available.

For each verifier, the fused ranking score is

    confidence^alpha * verifier^(1 - alpha)

where:

    alpha = 1.0 -> confidence only
    alpha = 0.0 -> verifier only
    0 < alpha < 1 -> weighted geometric fusion

The script searches alpha from 0.00 to 1.00 in steps of 0.01 and selects the
value with the lowest AURC on the calibration split.

The selected alpha must be frozen before held-out test evaluation. Test labels
must never be used to choose the fusion weight.

When two alpha values have exactly the same calibration AURC, the larger alpha
is preferred. This deterministic tie-break favors the simpler confidence-heavy
solution rather than assigning unnecessary weight to the verifier.

This experiment tunes two verifier fusions independently:

- confidence + question-aware semantic V2;
- confidence + answer-support/self verifier.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from src.evaluation.evaluate_question_aware_ablation import (
    build_risk_coverage_curve,
    calculate_aurc,
    extract_question_aware_score,
    extract_self_verification_score,
    infer_correctness,
    load_jsonl,
)
from src.utils.io import save_json

DEFAULT_INPUT = Path(
    "outputs/predictions/"
    "calibration_with_question_aware_v2_and_self_verification.jsonl"
)

DEFAULT_OUTPUT = Path(
    "outputs/tables/fusion_weight_tuning.json"
)

GRID_STEP = 0.01
GRID_STEPS = 100


def validate_probability(
    value: float,
    name: str,
) -> float:
    """Require a finite probability-like fusion input in [0, 1]."""

    if (
        not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(
            f"{name} must be a finite value "
            f"in [0, 1], received {value!r}."
        )

    return value


def weighted_geometric_mean(
    confidence: float,
    verifier: float,
    alpha: float,
) -> float:
    """
    Combine confidence and verifier scores with a weighted geometric mean.

    Explicit endpoint handling is important:

        alpha = 1 -> return confidence exactly
        alpha = 0 -> return verifier exactly

    For interior alpha values, a zero component keeps the fused score at zero.
    This preserves the intended behavior of invalid question-aware claims,
    whose verifier score is defined as zero.
    """

    confidence = validate_probability(
        float(confidence),
        "confidence",
    )

    verifier = validate_probability(
        float(verifier),
        "verifier",
    )

    if (
        not math.isfinite(alpha)
        or not 0.0 <= alpha <= 1.0
    ):
        raise ValueError(
            "alpha must be a finite value "
            f"in [0, 1], received {alpha!r}."
        )

    if alpha == 0.0:
        return verifier

    if alpha == 1.0:
        return confidence

    if (
        confidence == 0.0
        or verifier == 0.0
    ):
        return 0.0

    return math.exp(
        alpha
        * math.log(
            confidence
        )
        + (
            1.0
            - alpha
        )
        * math.log(
            verifier
        )
    )


def validate_score_vectors(
    correctness: list[bool],
    confidence_scores: list[float],
    verifier_scores: list[float],
) -> None:
    """Validate aligned calibration labels and ranking-score vectors."""

    if not correctness:
        raise ValueError(
            "Calibration data cannot be empty."
        )

    if not (
        len(correctness)
        == len(confidence_scores)
        == len(verifier_scores)
    ):
        raise ValueError(
            "Correctness, confidence, and "
            "verifier score lengths must match."
        )

    if (
        not any(correctness)
        or all(correctness)
    ):
        raise ValueError(
            "Calibration data must contain both "
            "correct and incorrect QA candidates."
        )

    for index, score in enumerate(
        confidence_scores
    ):
        validate_probability(
            score,
            f"confidence_scores[{index}]",
        )

    for index, score in enumerate(
        verifier_scores
    ):
        validate_probability(
            score,
            f"verifier_scores[{index}]",
        )


def tune_one(
    correctness: list[bool],
    confidence_scores: list[float],
    verifier_scores: list[float],
) -> dict[str, Any]:
    """
    Select one fusion weight by minimizing calibration AURC.

    Every alpha is evaluated on exactly the same calibration examples.

    The final `min` tie-break uses `-alpha`, so if two configurations have
    numerically identical AURC, the one with more confidence weight is chosen.
    """

    validate_score_vectors(
        correctness,
        confidence_scores,
        verifier_scores,
    )

    rows: list[
        dict[str, float]
    ] = []

    for step in range(
        GRID_STEPS + 1
    ):
        alpha = (
            step
            / GRID_STEPS
        )

        scores = [
            weighted_geometric_mean(
                confidence,
                verifier,
                alpha,
            )
            for confidence, verifier
            in zip(
                confidence_scores,
                verifier_scores,
            )
        ]

        aurc = calculate_aurc(
            build_risk_coverage_curve(
                correctness,
                scores,
            )
        )

        rows.append(
            {
                "alpha": (
                    alpha
                ),
                "verifier_weight": (
                    1.0
                    - alpha
                ),
                "aurc": (
                    aurc
                ),
            }
        )

    best = min(
        rows,
        key=lambda row: (
            row[
                "aurc"
            ],
            -row[
                "alpha"
            ],
        ),
    )

    return {
        "best_alpha": (
            best[
                "alpha"
            ]
        ),
        "best_verifier_weight": (
            1.0
            - best[
                "alpha"
            ]
        ),
        "best_aurc": (
            best[
                "aurc"
            ]
        ),
        "grid_step": (
            GRID_STEP
        ),
        "tie_break_rule": (
            "minimum AURC, then larger alpha "
            "if AURC is exactly tied"
        ),
        "grid": (
            rows
        ),
    }


def validate_calibration_input(
    records: list[dict[str, Any]],
    input_path: Path,
) -> None:
    """
    Guard the parameter-selection stage against obvious test-split misuse.

    Fusion weights are model-selection parameters, so this script must operate
    on calibration data only.

    If split metadata exists in the records, every record must explicitly belong
    to the calibration split. The default artifact name also identifies the
    intended calibration-stage input.
    """

    if not records:
        raise ValueError(
            "Calibration prediction file "
            "cannot be empty."
        )

    observed_splits = {
        str(
            record[
                "split"
            ]
        )
        .strip()
        .lower()
        for record in records
        if (
            record.get(
                "split"
            )
            is not None
        )
    }

    if (
        observed_splits
        and observed_splits
        != {
            "calibration"
        }
    ):
        raise ValueError(
            "Fusion-weight tuning must use "
            "calibration data only; observed "
            f"splits: {sorted(observed_splits)}."
        )

    if (
        not observed_splits
        and "calibration"
        not in input_path.name.lower()
    ):
        raise ValueError(
            "Input records contain no split metadata "
            "and the filename does not identify a "
            "calibration artifact. Refusing to tune "
            "fusion weights because this could leak "
            "held-out test labels."
        )


def main() -> None:
    """Tune both verifier fusion weights on calibration data and save them."""

    parser = argparse.ArgumentParser(
        description=(
            "Tune confidence/verifier fusion "
            "weights using calibration AURC only."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=(
            DEFAULT_INPUT
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_OUTPUT
        ),
    )

    args = parser.parse_args()

    records = load_jsonl(
        args.input
    )

    validate_calibration_input(
        records,
        args.input,
    )

    # Correctness labels are used only on the calibration split to select alpha.
    # Once selected, alpha must be frozen before any held-out evaluation.
    correctness = [
        infer_correctness(
            record
        )
        for record in records
    ]

    confidence_scores = [
        validate_probability(
            float(
                record[
                    "calibrated_confidence"
                ]
            ),
            "calibrated_confidence",
        )
        for record in records
    ]

    # Invalid question-aware claims are assigned score 0 by the shared extractor,
    # matching the score definition used in the main ablation evaluator.
    qa_scores = [
        extract_question_aware_score(
            record,
            "qa_entailment_probability",
        )
        for record in records
    ]

    self_scores: list[
        float
    ] = []

    for record in records:
        score = (
            extract_self_verification_score(
                record,
                "self_verification_score",
            )
        )

        if score is None:
            raise ValueError(
                "Missing self_verification_score "
                "in calibration data."
            )

        self_scores.append(
            validate_probability(
                score,
                "normalized self-verification score",
            )
        )

    confidence_aurc = calculate_aurc(
        build_risk_coverage_curve(
            correctness,
            confidence_scores,
        )
    )

    question_aware_result = (
        tune_one(
            correctness,
            confidence_scores,
            qa_scores,
        )
    )

    self_verifier_result = (
        tune_one(
            correctness,
            confidence_scores,
            self_scores,
        )
    )

    result = {
        "analysis_type": (
            "calibration_only_fusion_weight_tuning"
        ),
        "input": (
            str(
                args.input
            )
        ),
        "split": (
            "calibration"
        ),
        "records": (
            len(
                records
            )
        ),
        "correct": (
            sum(
                correctness
            )
        ),
        "incorrect": (
            len(
                records
            )
            - sum(
                correctness
            )
        ),
        "selection_rule": (
            "Choose alpha with minimum calibration "
            "AURC; if exactly tied, choose the "
            "larger alpha; freeze before test."
        ),
        "formula": (
            "confidence^alpha * "
            "verifier^(1-alpha)"
        ),
        "alpha_interpretation": {
            "0.0": (
                "verifier only"
            ),
            "1.0": (
                "confidence only"
            ),
        },
        "confidence_only_aurc": (
            confidence_aurc
        ),
        "question_aware": (
            question_aware_result
        ),
        "self_verifier": (
            self_verifier_result
        ),
    }

    save_json(
        result,
        args.output,
    )

    print(
        f"Calibration records: "
        f"{len(records)}"
    )

    print(
        f"Confidence-only AURC: "
        f"{confidence_aurc:.6f}"
    )

    print(
        "QA fusion: "
        f"alpha="
        f"{question_aware_result['best_alpha']:.2f}, "
        f"verifier weight="
        f"{question_aware_result['best_verifier_weight']:.2f}, "
        f"AURC="
        f"{question_aware_result['best_aurc']:.6f}"
    )

    print(
        "Self fusion: "
        f"alpha="
        f"{self_verifier_result['best_alpha']:.2f}, "
        f"verifier weight="
        f"{self_verifier_result['best_verifier_weight']:.2f}, "
        f"AURC="
        f"{self_verifier_result['best_aurc']:.6f}"
    )

    print(
        f"Saved: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()