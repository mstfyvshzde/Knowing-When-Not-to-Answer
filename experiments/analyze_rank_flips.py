"""
Analyze how verifier fusion changes correct-vs-incorrect pair ordering.

This experiment explains one mechanism behind the AURC differences observed
between confidence-only ranking and the two fixed 50/50 verifier fusions:

- confidence + question-aware semantic V2;
- confidence + self-verification.

The analysis considers every pair containing one correct and one incorrect QA
candidate.

For each method, the score relation is classified as:

    +1 -> correct candidate scores above incorrect candidate
     0 -> both candidates receive exactly the same score
    -1 -> correct candidate scores below incorrect candidate

Treating equal scores as genuine ties is important. Verifier scores can contain
many exact ties, especially when invalid question-aware claims are assigned
score 0.0. Arbitrarily breaking those ties by record index would make some
pairs appear to be "fixed" or "harmed" even though the score itself expresses
no preference.

Pairwise categories therefore distinguish:

- strict fixes
- strict harms
- good/bad ordering changed into a tie
- ties resolved in a good/bad direction
- unchanged good, bad, or tied ordering

AURC is also reported for context. Unlike the pairwise diagnostic, AURC requires
a deterministic order for equal scores. Records are therefore placed in the
same seed-17 deterministic order used by the nested held-out evaluation before
the standard evaluator's score-descending/original-index tie rule is applied.

This script is diagnostic only. It does not tune fusion weights or use test
labels to choose model parameters.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

from src.evaluation.evaluate_question_aware_ablation import (
    build_risk_coverage_curve,
    calculate_aurc,
    extract_question_aware_score,
    extract_self_verification_score,
    geometric_mean_score,
    infer_correctness,
    load_jsonl,
)
from src.utils.io import save_json

DEFAULT_INPUT = Path(
    "outputs/predictions/"
    "test_with_question_aware_v2_and_self_verification.jsonl"
)

DEFAULT_OUTPUT = Path(
    "outputs/tables/rank_flip_analysis.json"
)

DEFAULT_ORDER_SEED = 17


def deterministic_record_order(
    records: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    """
    Return a reproducible shuffled copy of the held-out records.

    The same seed-17 ordering is used by the nested sample-size experiment.
    Keeping that order here ensures that AURC tie-breaking is comparable with
    the project's final held-out evaluation.
    """

    ordered_records = list(
        records
    )

    random.Random(
        seed
    ).shuffle(
        ordered_records
    )

    return ordered_records


def validate_scores(
    scores: list[float],
    score_name: str,
) -> None:
    """Require one finite [0, 1] ranking score for every evaluated record."""

    if not scores:
        raise ValueError(
            f"{score_name} cannot be empty."
        )

    for index, score in enumerate(
        scores
    ):
        if (
            not math.isfinite(
                score
            )
            or not (
                0.0
                <= score
                <= 1.0
            )
        ):
            raise ValueError(
                f"{score_name} contains invalid "
                f"score at index {index}: "
                f"{score!r}."
            )


def pair_relation(
    correct_score: float,
    incorrect_score: float,
) -> int:
    """
    Compare one correct candidate with one incorrect candidate.

    Returns:
        +1 if the correct candidate is ranked above the incorrect candidate;
         0 if both scores are exactly tied;
        -1 if the incorrect candidate is ranked above the correct candidate.

    No record-index tie-break is used here because this function measures what
    the score itself says about the pair.
    """

    if correct_score > incorrect_score:
        return 1

    if correct_score < incorrect_score:
        return -1

    return 0


def rank_flip_summary(
    correctness: list[bool],
    confidence_scores: list[float],
    alternative_scores: list[float],
) -> dict[str, Any]:
    """
    Compare pairwise ordering under confidence and one alternative ranking.

    Pairwise analysis is tie-aware, while the accompanying AURCs use the
    evaluator's deterministic tie-breaking rule.
    """

    if not (
        len(
            correctness
        )
        == len(
            confidence_scores
        )
        == len(
            alternative_scores
        )
    ):
        raise ValueError(
            "Correctness and score lengths "
            "must match."
        )

    if not correctness:
        raise ValueError(
            "Cannot analyze an empty dataset."
        )

    validate_scores(
        confidence_scores,
        "confidence_scores",
    )

    validate_scores(
        alternative_scores,
        "alternative_scores",
    )

    correct_indices = [
        index
        for index, correct
        in enumerate(
            correctness
        )
        if correct
    ]

    incorrect_indices = [
        index
        for index, correct
        in enumerate(
            correctness
        )
        if not correct
    ]

    if (
        not correct_indices
        or not incorrect_indices
    ):
        raise ValueError(
            "Rank-flip analysis requires at "
            "least one correct and one "
            "incorrect prediction."
        )

    strict_fixes = 0
    strict_harms = 0

    good_to_tie = 0
    bad_to_tie = 0

    tie_resolved_good = 0
    tie_resolved_bad = 0

    unchanged_good = 0
    unchanged_bad = 0
    unchanged_tie = 0

    # Every correct/incorrect pair asks the ranking question that matters for
    # selective QA: does the scoring method place the correct candidate first?
    for correct_index in correct_indices:
        for incorrect_index in incorrect_indices:
            confidence_relation = (
                pair_relation(
                    confidence_scores[
                        correct_index
                    ],
                    confidence_scores[
                        incorrect_index
                    ],
                )
            )

            alternative_relation = (
                pair_relation(
                    alternative_scores[
                        correct_index
                    ],
                    alternative_scores[
                        incorrect_index
                    ],
                )
            )

            if (
                confidence_relation == -1
                and alternative_relation == 1
            ):
                strict_fixes += 1

            elif (
                confidence_relation == 1
                and alternative_relation == -1
            ):
                strict_harms += 1

            elif (
                confidence_relation == 1
                and alternative_relation == 0
            ):
                good_to_tie += 1

            elif (
                confidence_relation == -1
                and alternative_relation == 0
            ):
                bad_to_tie += 1

            elif (
                confidence_relation == 0
                and alternative_relation == 1
            ):
                tie_resolved_good += 1

            elif (
                confidence_relation == 0
                and alternative_relation == -1
            ):
                tie_resolved_bad += 1

            elif (
                confidence_relation == 1
                and alternative_relation == 1
            ):
                unchanged_good += 1

            elif (
                confidence_relation == -1
                and alternative_relation == -1
            ):
                unchanged_bad += 1

            elif (
                confidence_relation == 0
                and alternative_relation == 0
            ):
                unchanged_tie += 1

            else:
                raise RuntimeError(
                    "Unexpected pairwise rank "
                    "transition."
                )

    total_pairs = (
        len(
            correct_indices
        )
        * len(
            incorrect_indices
        )
    )

    categorized_pairs = (
        strict_fixes
        + strict_harms
        + good_to_tie
        + bad_to_tie
        + tie_resolved_good
        + tie_resolved_bad
        + unchanged_good
        + unchanged_bad
        + unchanged_tie
    )

    if categorized_pairs != total_pairs:
        raise RuntimeError(
            "Pairwise categories do not sum "
            "to the total number of "
            "correct/incorrect pairs."
        )

    confidence_aurc = calculate_aurc(
        build_risk_coverage_curve(
            correctness,
            confidence_scores,
        )
    )

    alternative_aurc = calculate_aurc(
        build_risk_coverage_curve(
            correctness,
            alternative_scores,
        )
    )

    return {
        "correct_incorrect_pairs": (
            total_pairs
        ),

        # These preserve the intuitive meaning of a true rank flip:
        # wrong order -> correct order, or correct order -> wrong order.
        "strict_fixes": (
            strict_fixes
        ),
        "strict_harms": (
            strict_harms
        ),
        "net_strict_fixes_minus_harms": (
            strict_fixes
            - strict_harms
        ),
        "strict_fix_rate": (
            strict_fixes
            / total_pairs
        ),
        "strict_harm_rate": (
            strict_harms
            / total_pairs
        ),

        # Tie transitions are reported separately rather than being converted
        # into artificial wins/losses by an arbitrary index tie-break.
        "good_to_tie": (
            good_to_tie
        ),
        "bad_to_tie": (
            bad_to_tie
        ),
        "tie_resolved_good": (
            tie_resolved_good
        ),
        "tie_resolved_bad": (
            tie_resolved_bad
        ),

        "unchanged_good": (
            unchanged_good
        ),
        "unchanged_bad": (
            unchanged_bad
        ),
        "unchanged_tie": (
            unchanged_tie
        ),

        # AURC still uses deterministic tie-breaking because a complete ranked
        # list is required to construct the risk-coverage curve.
        "confidence_aurc": (
            confidence_aurc
        ),
        "alternative_aurc": (
            alternative_aurc
        ),
        "aurc_difference": (
            alternative_aurc
            - confidence_aurc
        ),
    }


def main() -> None:
    """
    Run pairwise rank diagnostics for the two fixed 50/50 verifier fusions.
    """

    records = load_jsonl(
        DEFAULT_INPUT
    )

    if not records:
        raise ValueError(
            "Rank-flip input is empty."
        )

    # The shuffled order affects only deterministic tie resolution used by
    # AURC. Pairwise tie-aware counts themselves are invariant to record order.
    records = (
        deterministic_record_order(
            records,
            DEFAULT_ORDER_SEED,
        )
    )

    correctness = [
        infer_correctness(
            record
        )
        for record in records
    ]

    confidence_scores = [
        float(
            record[
                "calibrated_confidence"
            ]
        )
        for record in records
    ]

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
                "Missing "
                "self_verification_score."
            )

        self_scores.append(
            score
        )

    validate_scores(
        confidence_scores,
        "confidence_scores",
    )

    validate_scores(
        qa_scores,
        "question_aware_scores",
    )

    validate_scores(
        self_scores,
        "self_verification_scores",
    )

    # These are the historical fixed equal-weight combinations. They are
    # analyzed to explain why verification can change the ranking even though
    # calibration-only tuning later selected alpha=1.0.
    qa_fusion = [
        geometric_mean_score(
            confidence,
            verifier,
        )
        for confidence, verifier
        in zip(
            confidence_scores,
            qa_scores,
        )
    ]

    self_fusion = [
        geometric_mean_score(
            confidence,
            verifier,
        )
        for confidence, verifier
        in zip(
            confidence_scores,
            self_scores,
        )
    ]

    result = {
        "analysis_type": (
            "tie_aware_correct_incorrect_pair_rank_analysis"
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
        "ordering_seed": (
            DEFAULT_ORDER_SEED
        ),
        "pairwise_tie_handling": (
            "Equal scores are treated as ties; "
            "record index is not used to classify "
            "pairwise fixes or harms."
        ),
        "aurc_tie_break_rule": (
            "score descending, then seed-17 "
            "deterministic input order"
        ),
        "question_aware_fixed_50_50": (
            rank_flip_summary(
                correctness,
                confidence_scores,
                qa_fusion,
            )
        ),
        "self_verifier_fixed_50_50": (
            rank_flip_summary(
                correctness,
                confidence_scores,
                self_fusion,
            )
        ),
        "tuned_fusion_note": (
            "Calibration selected alpha=1.0 "
            "for both verifier fusions, so the "
            "tuned fusion score is identical "
            "to confidence-only and therefore "
            "produces no score-based rank changes."
        ),
    }

    save_json(
        result,
        DEFAULT_OUTPUT,
    )

    print(
        f"Records: "
        f"{result['records']}"
    )

    print(
        "Correct / incorrect: "
        f"{result['correct']} / "
        f"{result['incorrect']}"
    )

    for name in (
        "question_aware_fixed_50_50",
        "self_verifier_fixed_50_50",
    ):
        row = result[
            name
        ]

        print(
            f"\n{name}"
        )

        print(
            "Strict fixes: "
            f"{row['strict_fixes']}"
        )

        print(
            "Strict harms: "
            f"{row['strict_harms']}"
        )

        print(
            "Net strict fixes-harms: "
            f"{row[
                'net_strict_fixes_minus_harms'
            ]}"
        )

        print(
            "Good -> tie: "
            f"{row['good_to_tie']}"
        )

        print(
            "Bad -> tie: "
            f"{row['bad_to_tie']}"
        )

        print(
            "Tie -> good: "
            f"{row['tie_resolved_good']}"
        )

        print(
            "Tie -> bad: "
            f"{row['tie_resolved_bad']}"
        )

        print(
            "Confidence AURC: "
            f"{row['confidence_aurc']:.6f}"
        )

        print(
            "Fusion AURC: "
            f"{row['alternative_aurc']:.6f}"
        )

        print(
            "AURC difference: "
            f"{row['aurc_difference']:+.6f}"
        )

    print(
        f"\nSaved: "
        f"{DEFAULT_OUTPUT}"
    )


if __name__ == "__main__":
    main()