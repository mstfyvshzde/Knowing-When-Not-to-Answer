from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluation.evaluate_question_aware_ablation import (
    build_risk_coverage_curve,
    calculate_aurc,
    create_ranked_indices,
    extract_question_aware_score,
    extract_self_verification_score,
    geometric_mean_score,
    infer_correctness,
    load_jsonl,
)

DEFAULT_INPUT = Path(
    "outputs/predictions/test_with_question_aware_v2_and_self_verification.jsonl"
)
DEFAULT_OUTPUT = Path("outputs/tables/rank_flip_analysis.json")


def rank_flip_summary(
    correctness: list[bool],
    confidence_scores: list[float],
    alternative_scores: list[float],
) -> dict[str, Any]:
    confidence_order = create_ranked_indices(confidence_scores)
    alternative_order = create_ranked_indices(alternative_scores)

    confidence_rank = [0] * len(correctness)
    alternative_rank = [0] * len(correctness)

    for rank,x in enumerate(confidence_order):
        confidence_rank[x] = rank

    for rank, x in enumerate(alternative_order):
        alternative_rank[x] = rank

    correct_indices = [i for i, correct in enumerate(correctness) if correct]
    incorrect_indices = [i for i, correct in enumerate(correctness) if not correct]

    fixes = 0
    harms = 0
    unchanged_good = 0
    unchanged_bad = 0

    for correct_index in correct_indices:
        for incorrect_index in incorrect_indices:
            confidence_good = (
                confidence_rank[correct_index] < confidence_rank[incorrect_index]
            )
            alternative_good = (
                alternative_rank[correct_index] < alternative_rank[incorrect_index]
            )

            if not confidence_good and alternative_good:
                fixes += 1
            elif confidence_good and not alternative_good:
                harms += 1
            elif confidence_good:
                unchanged_good += 1
            else:
                unchanged_bad += 1

    total_pairs = len(correct_indices) * len(incorrect_indices)

    return {
        "correct_incorrect_pairs": total_pairs,
        "fixes": fixes,
        "harms": harms,
        "net_fixes_minus_harms": fixes - harms,
        "fix_rate": fixes / total_pairs,
      "harm_rate": harms / total_pairs,
        "unchanged_correct_order": unchanged_good,
        "unchanged_incorrect_order": unchanged_bad,
        "confidence_aurc": calculate_aurc(
            build_risk_coverage_curve(correctness, confidence_scores)
        ),
        "alternative_aurc": calculate_aurc(
            build_risk_coverage_curve(correctness, alternative_scores)
        ),
    }


def main() -> None:
    records = load_jsonl(DEFAULT_INPUT)
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
        score = extract_self_verification_score(
            record, "self_verification_score"
        )
        if score is None:
            raise ValueError("Missing self_verification_score.")
        self_scores.append(score)

    qa_fusion = [
        geometric_mean_score(confidence, verifier)
        for confidence, verifier in zip(confidence_scores, qa_scores)
    ]
    self_fusion = [
        geometric_mean_score(confidence, verifier)
        for confidence, verifier in zip(confidence_scores, self_scores)
    ]

    result = {
        "records": len(records),
        "correct": sum(correctness),
        "incorrect": len(records) - sum(correctness),
        "question_aware_fixed_50_50": rank_flip_summary(
            correctness, confidence_scores, qa_fusion
        ),
        "self_verifier_fixed_50_50": rank_flip_summary(
            correctness, confidence_scores, self_fusion
        ),
        "tuned_fusion_note": (
            "Calibration selected alpha=1.0 for both verifier fusions, "
            "so tuned fusion is identical to confidence-only and causes zero rank flips."
        ),
    }

    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Records: {result['records']}")
    print(f"Correct / incorrect: {result['correct']} / {result['incorrect']}")

    for name in ("question_aware_fixed_50_50", "self_verifier_fixed_50_50"):
        row = result[name]
        print(f"\n{name}")
        print(f"Fixes: {row['fixes']}")
        print(f"Harms: {row['harms']}")
        print(f"Net fixes-harms: {row['net_fixes_minus_harms']}")
        print(f"Confidence AURC: {row['confidence_aurc']:.6f}")
        print(f"Fusion AURC: {row['alternative_aurc']:.6f}")

    print(f"\nSaved: {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
