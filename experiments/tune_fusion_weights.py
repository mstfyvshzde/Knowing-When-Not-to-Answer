from __future__ import annotations

import argparse
import json
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

DEFAULT_INPUT = Path(
    "outputs/predictions/calibration_with_question_aware_v2_and_self_verification.jsonl"
)
DEFAULT_OUTPUT = Path("outputs/tables/fusion_weight_tuning.json")


def weighted_geometric_mean(
    confidence: float,
    verifier: float,
    alpha: float,
) -> float:
    if alpha <= 0.0:
        return verifier
    if alpha >= 1.0:
        return confidence

    confidence = max(0.0, min(1.0, confidence))
    verifier = max(0.0, min(1.0, verifier))

    if confidence == 0.0 or verifier == 0.0:
        return 0.0

    return math.exp(
        alpha * math.log(confidence)
        + (1.0 - alpha) * math.log(verifier)
    )


def tune_one(
    correctness: list[bool],
    confidence_scores: list[float],
    verifier_scores: list[float],
) -> dict[str, Any]:
    rows: list[dict[str, float]] = []

    for step in range(101):
        alpha = step / 100.0
        scores = [
            weighted_geometric_mean(confidence, verifier, alpha)
            for confidence, verifier in zip(confidence_scores, verifier_scores)
        ]
        aurc = calculate_aurc(build_risk_coverage_curve(correctness, scores))
        rows.append({"alpha": alpha, "aurc": aurc})

    best = min(rows, key=lambda row: (row["aurc"], -row["alpha"]))

    return {
        "best_alpha": best["alpha"],
        "best_aurc": best["aurc"],
        "grid_step": 0.01,
        "grid": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune confidence/verifier fusion weights on calibration data only."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    correctness = [infer_correctness(record) for record in records]
    confidence_scores = [
        float(record["calibrated_confidence"]) for record in records
    ]

    qa_scores = [
        extract_question_aware_score(record, "qa_entailment_probability")
        for record in records
    ]

    self_scores: list[float] = []
    for record in records:
        score = extract_self_verification_score(record, "self_verification_score")
        if score is None:
            raise ValueError("Missing self_verification_score in calibration data.")
        self_scores.append(score)

    confidence_aurc = calculate_aurc(
        build_risk_coverage_curve(correctness, confidence_scores)
    )

    result = {
        "input": str(args.input),
        "split": "calibration",
        "records": len(records),
        "selection_rule": "Choose alpha with minimum calibration AURC; freeze before test.",
        "formula": "confidence^alpha * verifier^(1-alpha)",
        "confidence_only_aurc": confidence_aurc,
        "question_aware": tune_one(
            correctness,
            confidence_scores,
            qa_scores,
        ),
        "self_verifier": tune_one(
            correctness,
            confidence_scores,
            self_scores,
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Calibration records: {len(records)}")
    print(f"Confidence-only AURC: {confidence_aurc:.6f}")
    print(
        "QA fusion: "
        f"alpha={result['question_aware']['best_alpha']:.2f}, "
        f"AURC={result['question_aware']['best_aurc']:.6f}"
     )
    print(
        "Self fusion: "
        f"alpha={result['self_verifier']['best_alpha']:.2f}, "
        f"AURC={result['self_verifier']['best_aurc']:.6f}"
    )
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
