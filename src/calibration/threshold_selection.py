"""
Select confidence thresholds for the prototype three-way decision policy.

This module divides calibrated QA predictions into three confidence regions:

- ANSWER: confidence is high enough to answer directly
- VERIFY: confidence lies between the two thresholds and requires an additional check
- ABSTAIN: confidence is low enough to avoid answering

Two thresholds are selected using the calibration split only:

    confidence >= answer_threshold  -> ANSWER
    confidence <= abstain_threshold -> ABSTAIN
    otherwise                       -> VERIFY

Threshold selection is constrained by answer risk, unnecessary abstention,
and minimum decision-rate requirements.

Important: VERIFY here is a routing decision (doğrulamaya gönderme kararı).
It does not mean that verification has already succeeded.

This threshold-based policy is an earlier prototype component. The project's
final selective-QA evaluation ranks examples by scoring signals and measures
risk across coverage levels instead of using these fixed thresholds.

Held-out test labels are never used to select the thresholds.
"""

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.calibration.calibration_metrics import is_prediction_correct
from src.utils.io import load_jsonl, save_json, save_jsonl

# Thresholds are selected from temperature-calibrated confidence values on the
# calibration split. The held-out test split must not influence this search.
DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/raw_baseline_calibrated_calibration.jsonl"
)

DEFAULT_OUTPUT_PATH = Path("outputs/tables/decision_thresholds.json")

DEFAULT_ANNOTATED_OUTPUT_PATH = Path(
    "outputs/predictions/calibration_with_decisions.jsonl"
)


def validate_predictions(
    predictions: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate that prediction records are suitable for threshold selection.

    The search requires:

    - a confidence value for every prediction
    - confidence values already calibrated by temperature scaling
    - probabilities inside [0, 1]
    - both correct and incorrect examples
    - calibration-split data only

    These checks prevent malformed inputs and data leakage
    (test bilgisinin parameter selection sürecine sızması).
    """

    if not predictions:
        raise ValueError("Prediction list cannot be empty.")

    # Detect malformed records before threshold search.
    missing_confidence = [
        prediction.get("id", "unknown")
        for prediction in predictions
        if "confidence" not in prediction
    ]

    if missing_confidence:
        raise ValueError(
            "Some predictions do not contain confidence. "
            f"Missing IDs: {missing_confidence[:5]}"
        )

    # Threshold selection requires temperature-calibrated confidence values.
    uncalibrated_examples = [
        prediction.get("id", "unknown")
        for prediction in predictions
        if not prediction.get("confidence_is_calibrated", False)
    ]

    if uncalibrated_examples:
        raise ValueError(
            "Some confidence values are not calibrated. "
            "Run temperature_scaling.py first. "
            f"Example IDs: {uncalibrated_examples[:5]}"
        )

    # Thresholds are model-selection parameters. Using held-out test labels to
    # choose them would cause data leakage and make the final evaluation biased.
    observed_splits = {
        prediction.get("split")
        for prediction in predictions
        if prediction.get("split") is not None
    }

    if observed_splits and observed_splits != {"calibration"}:
        raise ValueError(
            "Threshold selection must use only the "
            "calibration split. "
            f"Observed splits: {observed_splits}"
        )

    confidences = np.asarray(
        [float(prediction["confidence"]) for prediction in predictions],
        dtype=np.float64,
    )

    if np.any(~np.isfinite(confidences)):
        raise ValueError("Confidence values must be finite.")

    if np.any(confidences < 0.0) or np.any(confidences > 1.0):
        raise ValueError("Confidence values must be between zero and one.")

    correct_labels = np.asarray(
        [is_prediction_correct(prediction) for prediction in predictions],
        dtype=np.int64,
    )

    if len(np.unique(correct_labels)) < 2:
        raise ValueError(
            "Threshold selection requires both correct and incorrect predictions."
        )

    return confidences, correct_labels


def assign_decision(
    confidence: float, abstain_threshold: float, answer_threshold: float
) -> str:
    """
    Map one calibrated confidence value to ANSWER, VERIFY, or ABSTAIN.

    answer_threshold (cevap eşiği):
        minimum confidence required for a direct ANSWER.

    abstain_threshold (kaçınma eşiği):
        maximum confidence allowed for a direct ABSTAIN.

    Values between the two thresholds enter the VERIFY region.

    VERIFY is only a routing label at this stage; no verifier result is evaluated
    inside this function.
    """

    if abstain_threshold >= answer_threshold:
        raise ValueError("abstain_threshold must be smaller than answer_threshold.")

    if confidence >= answer_threshold:
        return "ANSWER"

    if confidence <= abstain_threshold:
        return "ABSTAIN"

    return "VERIFY"


def evaluate_threshold_pair(
    confidences: np.ndarray,
    correct_labels: np.ndarray,
    abstain_threshold: float,
    answer_threshold: float,
) -> dict[str, Any]:
    """
    Evaluate one candidate abstain/answer threshold pair.

    The thresholds divide calibration examples into three regions and measure how
    safe and useful those regions would be.

    Important quantities include:

    - answer_risk:
        fraction of ANSWER examples whose forced-answer prediction is incorrect

    - answer_coverage:
        fraction of all examples assigned directly to ANSWER

    - abstain_correct_rate:
        fraction of ABSTAIN examples whose underlying forced-answer prediction was
        actually correct; high values indicate unnecessary abstention

    - verify_rate:
        fraction routed to the intermediate VERIFY region

    - direct_decision_rate:
        fraction handled directly by ANSWER or ABSTAIN without VERIFY
    """

    if abstain_threshold >= answer_threshold:
        raise ValueError("Invalid threshold ordering.")

    total_examples = len(confidences)

    # Partition the calibration examples according to the two thresholds.
    # The VERIFY mask contains only examples that are neither confident enough for
    # ANSWER nor low-confidence enough for ABSTAIN.

    answer_mask = confidences >= answer_threshold
    abstain_mask = confidences <= abstain_threshold
    verify_mask = ~answer_mask & ~abstain_mask

    answer_count = int(np.sum(answer_mask))
    abstain_count = int(np.sum(abstain_mask))
    verify_count = int(np.sum(verify_mask))

    answer_correct = int(np.sum(correct_labels[answer_mask] == 1))
    answer_incorrect = int(np.sum(correct_labels[answer_mask] == 0))

    # A correct prediction placed in the ABSTAIN region represents an unnecessary
    # refusal: the underlying forced-answer model would have answered correctly.
    abstain_correct = int(np.sum(correct_labels[abstain_mask] == 1))
    abstain_incorrect = int(np.sum(correct_labels[abstain_mask] == 0))

    if answer_count > 0:
        answer_accuracy = answer_correct / answer_count
        answer_risk = answer_incorrect / answer_count

    else:
        answer_accuracy = None
        answer_risk = None

    if abstain_count > 0:
        abstain_correct_rate = abstain_correct / abstain_count

    else:
        abstain_correct_rate = None


    answer_coverage = answer_count / total_examples
    abstain_rate = abstain_count / total_examples
    verify_rate = verify_count / total_examples
    direct_decision_rate = (answer_count + abstain_count) / total_examples
    false_answer_rate = answer_incorrect / total_examples
    unnecessary_abstention_rate = abstain_correct / total_examples

    return {
        "abstain_threshold": float(abstain_threshold),
        "answer_threshold": float(answer_threshold),
        "total_examples": total_examples,
        "answer_count": answer_count,
        "verify_count": verify_count,
        "abstain_count": abstain_count,
        "answer_correct": answer_correct,
        "answer_incorrect": answer_incorrect,
        "abstain_correct": abstain_correct,
        "abstain_incorrect": abstain_incorrect,
        "answer_accuracy": answer_accuracy,
        "answer_risk": answer_risk,
        "abstain_correct_rate": abstain_correct_rate,
        "answer_coverage": answer_coverage,
        "verify_rate": verify_rate,
        "abstain_rate": abstain_rate,
        "direct_decision_rate": direct_decision_rate,
        "false_answer_rate": false_answer_rate,
        "unnecessary_abstention_rate": unnecessary_abstention_rate,
    }


def create_threshold_candidates(
    confidences: np.ndarray, grid_size: int
) -> np.ndarray:
    """
    Create candidate threshold values from confidence quantiles.

    Quantile (yüzdelik konum) based search places candidate thresholds according
    to the actual calibration confidence distribution instead of testing only
    uniformly spaced probability values.

    For example, if many predictions are concentrated around confidence 0.6,
    quantile-based candidates place more useful search points around that region.

    Duplicate confidence values are removed and 0.0 / 1.0 are included as
    probability boundaries.
    """

    if grid_size < 3:
        raise ValueError("grid_size must be at least 3.")

    # Generate evenly spaced quantile positions.
    quantiles = np.linspace(0.0, 1.0, grid_size)

    # Convert quantiles into observed confidence-scale thresholds.
    candidates = np.quantile(confidences, quantiles)

    # Remove duplicates and include both probability boundaries.
    candidates = np.unique(candidates)

    candidates = np.unique(
        np.concatenate([np.asarray([0.0]), candidates, np.asarray([1.0])])
    )

    return candidates


def select_thresholds(
    confidences: np.ndarray,
    correct_labels: np.ndarray,
    max_answer_risk: float,
    max_abstain_correct_rate: float,
    min_answer_rate: float,
    min_abstain_rate: float,
    grid_size: int,
) -> dict[str, Any]:
    """
    Search calibration data for the best ANSWER/ABSTAIN threshold pair.

    A candidate pair must first provide at least the required minimum number of
    ANSWER and ABSTAIN examples.

    The main constraints are:

    - max_answer_risk:
        maximum tolerated error rate inside the ANSWER region

    - max_abstain_correct_rate:
        maximum tolerated fraction of correct predictions unnecessarily placed
        inside the ABSTAIN region

    - min_answer_rate:
        prevents the policy from achieving low risk simply by refusing almost
        everything

    - min_abstain_rate:
        ensures the search actually creates a meaningful abstention region

    If one or more threshold pairs satisfy every constraint, the search prefers
    higher answer coverage and higher direct-decision rate.

    If no pair satisfies every constraint, the least-violating eligible pair is
    returned as a fallback and `constraints_satisfied` is set to False.
    """


    if not 0.0 <= max_answer_risk <= 1.0:
        raise ValueError("max_answer_risk must be between 0 and 1.")

    if not 0.0 <= max_abstain_correct_rate <= 1.0:
        raise ValueError("max_abstain_correct_rate must be between 0 and 1.")

    if not 0.0 <= min_answer_rate <= 1.0:
        raise ValueError("min_answer_rate must be between 0 and 1.")

    if not 0.0 <= min_abstain_rate <= 1.0:
        raise ValueError("min_abstain_rate must be between 0 and 1.")

    total_examples = len(confidences)

    # Convert minimum decision rates into required example counts. ceil ensures
    # the requested minimum rate is never violated because of integer rounding.
    minimum_answer_count = max(
        1,
        math.ceil(total_examples * min_answer_rate),
    )

    # minimum number of examples on which the model must choose ABSTAIN.
    minimum_abstain_count = max(1, math.ceil(total_examples * min_abstain_rate))

    candidates = create_threshold_candidates(
        confidences=confidences, grid_size=grid_size
    )

    # Feasible results satisfy all configured risk constraints.
    feasible_results: list[dict[str, Any]] = []

    # Fallback results satisfy the minimum ANSWER/ABSTAIN size requirements but
    # may violate one or more risk constraints.
    fallback_results: list[dict[str, Any]] = []

    for abstain_threshold in candidates:
        for answer_threshold in candidates:
            if abstain_threshold >= answer_threshold:
                continue

            result = evaluate_threshold_pair(
                confidences=confidences,
                correct_labels=correct_labels,
                abstain_threshold=float(abstain_threshold),
                answer_threshold=float(answer_threshold),
            )

            if result["answer_count"] < minimum_answer_count:
                continue

            if result["abstain_count"] < minimum_abstain_count:
                continue

            # Answer risk (cevap riski): among examples that would be answered directly,
            # what fraction of the underlying forced-answer predictions are incorrect?
            answer_risk = result["answer_risk"]

            # Unnecessary-abstention signal: among examples the system would refuse,
            # what fraction actually had a correct forced-answer prediction?
            abstain_correct_rate = result["abstain_correct_rate"]

            if answer_risk is None or abstain_correct_rate is None:
                continue

            # Constraint violation (kısıt ihlali) measures how far a candidate exceeds
            # the allowed maximum. A value of 0 means that constraint is satisfied.
            answer_risk_violation = max(0.0, answer_risk - max_answer_risk)

            abstain_violation = max(
                0.0, abstain_correct_rate - max_abstain_correct_rate
            )

            result["answer_risk_violation"] = answer_risk_violation
            result["abstain_violation"] = abstain_violation
            result["total_constraint_violation"] = (
                answer_risk_violation + abstain_violation
            )

            # Retain every eligible pair in case no fully feasible solution exists.
            fallback_results.append(result)

            constraints_satisfied = (
                answer_risk <= max_answer_risk
                and abstain_correct_rate <= max_abstain_correct_rate
            )

            if constraints_satisfied:
                feasible_results.append(result)

    # For fully feasible candidates, prioritize:
    # 1. higher direct ANSWER coverage
    # 2. higher ANSWER + ABSTAIN direct-decision rate
    # 3. lower answer risk
    # 4. lower unnecessary-abstention rate
    if feasible_results:
        # Select the best threshold pair according to the comparison rules in key
        best_result = max(
            feasible_results,
            key=lambda result: (
                result["answer_coverage"],
                result["direct_decision_rate"],
                -result["answer_risk"],
                -result["abstain_correct_rate"],
            ),
        )

        # Indicate that the selected threshold pair satisfies all constraints.
        constraints_satisfied = True

    else:
        if not fallback_results:
            raise ValueError(
                "No threshold pair produced enough ANSWER and ABSTAIN examples."
            )

        # If every candidate violates at least one constraint, choose the candidate
        # with the smallest total violation. Coverage is used as a secondary preference.
        best_result = min(
            fallback_results,
            key=lambda result: (
                result["total_constraint_violation"],
                -result["answer_coverage"],
                -result["direct_decision_rate"],
            ),
        )

        # Indicate that the selected threshold pair does not satisfy all constraints but is the best available.
        constraints_satisfied = False

    # Save both the selected policy and its provenance (hangi veri ve kurallarla
    # seçildiğini gösteren metadata) so the threshold choice can be audited later.
    selection_result = {
        # Kullanılan seçim yöntemi
        "method": (
            # Model üç karar verebilir: cevapla, kaçın/abstain, veya aradaki başka bir karar sınıfı.
            "constrained_three_way_"
            # Farklı eşik çiftleri denenir; yalnızca belirlenen risk, cevap oranı ve örnek sayısı koşullarını sağlayanlar kabul edilir.
            "threshold_search"
        ),
        # Modelin karar eşikleri calibration verisi üzerinde ayarlanmış. Test verisi bu ayarlama sırasında kullanılmamış.
        "fit_split": "calibration",
        # test seti tarafsız değerlendirme için saklanmış.
        "test_set_used_for_selection": False,
        # constraints_satisfied, seçilen sonucun belirlenen tüm koşulları sağlayıp sağlamadığını gösterir.
        "constraints_satisfied": (constraints_satisfied),
        # Bu değerler, seçimin uyması gereken sınırları temsil eder:
        "constraints": {
            # Cevap verirken izin verilen en yüksek hata/risk oranı
            "max_answer_risk": (max_answer_risk),
            # Kaçınma ile ilgili izin verilen en yüksek oran
            "max_abstain_correct_rate": (max_abstain_correct_rate),
            # Modelin en az cevap vermesi gereken oran
            "min_answer_rate": (min_answer_rate),
            # Modelin en az kaçınması gereken oran
            "min_abstain_rate": (min_abstain_rate),
            # En az kaç örneğe cevap verilmesi gerektiği
            "minimum_answer_count": (minimum_answer_count),
            # En az kaç örnekte cevap vermekten kaçınılması gerektiği
            "minimum_abstain_count": (minimum_abstain_count),
        },
        # Bu kısım, arama sürecinin özetini ve seçilen en iyi sonucu saklıyor.
        "search": {
            # Kaç farklı eşik noktası üretileceğini gösterir. Örneğin 101 ise yaklaşık 101 farklı threshold adayı oluşturulur.
            "grid_size": grid_size,
            # Tekrar eden değerler silindikten sonra gerçekte kaç threshold adayı kaldığını gösterir.
            "candidate_count": len(candidates),
            # Kaç farklı abstain_threshold + answer_threshold çifti değerlendirmeye alınmış, onu gösterir.
            "evaluated_pairs": (
                len(feasible_results)
                + len(
                    [
                        result
                        for result in fallback_results
                        if result not in feasible_results
                    ]
                )
            ),
            # Değerlendirilen çiftlerden kaç tanesi bütün güvenlik koşullarını sağlamış, onu gösterir.
            "feasible_pairs": len(feasible_results),
        },
        # Sonunda seçilen en iyi threshold çiftini ve onun bütün sonuçlarını saklar.
        "selected": best_result,
    }

    return selection_result


def annotate_predictions(
    predictions: list[dict[str, Any]],
    abstain_threshold: float,
    answer_threshold: float,
) -> list[dict[str, Any]]:
    """
    Apply the selected thresholds to calibration prediction records.

    Each prediction receives its prototype ANSWER, VERIFY, or ABSTAIN routing
    decision together with the thresholds that produced that decision.

    No new threshold fitting occurs here; the already selected values are reused.
    """

    annotated_predictions: list[dict[str, Any]] = []

    for prediction in predictions:
        confidence = float(prediction["confidence"])

        decision = assign_decision(
            confidence=confidence,
            abstain_threshold=abstain_threshold,
            answer_threshold=answer_threshold,
        )

        # Preserve the calibrated prediction and attach the prototype routing decision
        # together with its threshold provenance.
        updated_prediction = prediction.copy()

        updated_prediction.update(
            {
                "decision": decision,
                "abstain_threshold": abstain_threshold,
                "answer_threshold": answer_threshold,
                "threshold_source": "calibration_split",
            }
        )

        annotated_predictions.append(updated_prediction)

    return annotated_predictions


def run_threshold_selection(
    input_path: str | Path,
    output_path: str | Path,
    annotated_output_path: str | Path,
    max_answer_risk: float,
    max_abstain_correct_rate: float,
    min_answer_rate: float,
    min_abstain_rate: float,
    grid_size: int,
) -> dict[str, Any]:
    """
    Run the complete calibration-only threshold-selection workflow.

    The function:

    1. loads calibrated calibration predictions
    2. validates that test data is absent
    3. searches candidate threshold pairs
    4. selects the best feasible or fallback pair
    5. annotates predictions with three-way decisions
    6. saves both the policy metadata and annotated predictions

    This workflow belongs to the prototype threshold policy and is not the final
    risk-coverage ranking evaluation.
    """

    predictions = load_jsonl(input_path)

    confidences, correct_labels = validate_predictions(predictions)

    selection_result = select_thresholds(
        confidences=confidences,
        correct_labels=correct_labels,
        max_answer_risk=max_answer_risk,
        max_abstain_correct_rate=max_abstain_correct_rate,
        min_answer_rate=min_answer_rate,
        min_abstain_rate=min_abstain_rate,
        grid_size=grid_size,
    )

    selected = selection_result["selected"]

    abstain_threshold = float(selected["abstain_threshold"])

    answer_threshold = float(selected["answer_threshold"])

    annotated_predictions = annotate_predictions(
        predictions=predictions,
        abstain_threshold=abstain_threshold,
        answer_threshold=answer_threshold,
    )

    save_json(selection_result, output_path)

    save_jsonl(annotated_predictions, annotated_output_path)

    print("\nThreshold selection completed.")

    print(f"Constraints satisfied: {selection_result['constraints_satisfied']}")

    print(f"Abstain threshold: {abstain_threshold:.6f}")

    print(f"Answer threshold: {answer_threshold:.6f}")

    print(f"Answer coverage: {selected['answer_coverage']:.4f}")

    print(f"Answer risk: {selected['answer_risk']:.4f}")

    print(f"Verify rate: {selected['verify_rate']:.4f}")

    print(f"Abstain rate: {selected['abstain_rate']:.4f}")

    print(f"Abstain correct rate: {selected['abstain_correct_rate']:.4f}")

    print(f"Thresholds saved to: {output_path}")

    print(f"Annotated predictions saved to: {annotated_output_path}")

    return selection_result


def parse_arguments() -> argparse.Namespace:
    """Parse calibration threshold-search settings."""

    parser = argparse.ArgumentParser(
        description=(
            "Select ANSWER, VERIFY and ABSTAIN thresholds on calibration data."
        )
    )

    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))

    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))

    parser.add_argument(
        "--annotated-output", default=str(DEFAULT_ANNOTATED_OUTPUT_PATH)
    )

    parser.add_argument(
        "--max-answer-risk",
        type=float,
        default=0.10,
        help=("Maximum allowed error rate among direct ANSWER decisions."),
    )

    parser.add_argument(
        "--max-abstain-correct-rate",
        dest="max_abstain_correct_rate",
        type=float,
        default=0.25,
        help=(
            "Maximum allowed fraction of correct predictions inside the ABSTAIN region."
        ),
    )

    parser.add_argument("--min-answer-rate", type=float, default=0.05)

    parser.add_argument("--min-abstain-rate", type=float, default=0.05)

    parser.add_argument(
        "--grid_size",
        type=int,
        default=101,
        help="Number of quantile positions used to generate threshold candidates.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_threshold_selection(
        input_path=args.input,
        output_path=args.output,
        annotated_output_path=args.annotated_output,
        max_answer_risk=args.max_answer_risk,
        max_abstain_correct_rate=args.max_abstain_correct_rate,
        min_answer_rate=args.min_answer_rate,
        min_abstain_rate=args.min_abstain_rate,
        grid_size=args.grid_size,
    )
