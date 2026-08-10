"""
The purpose of this file is to search for the best decision thresholds on the calibration set and use them to assign the final ANSWER, VERIFY, or ABSTAIN decision to every prediction.
"""

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.calibration.calibration_metrics import is_prediction_correct
from src.utils.io import load_jsonl, save_json, save_jsonl

DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/raw_baseline_calibrated_calibration.jsonl"
)

DEFAULT_OUTPUT_PATH = Path("outputs/tables/decision_thresholds.json")

DEFAULT_ANNOTATED_OUTPUT_PATH = Path(
    "outputs/predictions/calibration_with_decisions.jsonl"
)


# validates that the prediction file is suitable for threshold selection and returns the confidence scores and correctness labels as NumPy arrays.
def validate_predictions(
    predictions: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    if not predictions:
        raise ValueError("Prediction list cannot be empty.")

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


# assigns one of three decisions (ANSWER, ABSTAIN, or VERIFY) based on the model's confidence score and two thresholds.
def assign_decision(
    confidence: float, abstain_threshold: float, answer_threshold: float
) -> str:
    # Abstain threshold is the maximum confidence at which the system decides not to answer.
    # Answer threshold is the minimum confidence required for the system to answer directly.
    if abstain_threshold >= answer_threshold:
        raise ValueError("abstain_threshold must be smaller than answer_threshold.")

    if confidence >= answer_threshold:
        return "ANSWER"

    if confidence <= abstain_threshold:
        return "ABSTAIN"

    return "VERIFY"


# valuates a single pair of abstain and answer thresholds by assigning decisions (ANSWER, VERIFY, ABSTAIN), computing their performance statistics, and returning all metrics needed to compare threshold pairs.
def evaluate_threshold_pair(
    confidences: np.ndarray,
    correct_labels: np.ndarray,
    abstain_threshold: float,
    answer_threshold: float,
) -> dict[str, Any]:

    if abstain_threshold >= answer_threshold:
        raise ValueError("Invalid threshold ordering.")

    total_examples = len(confidences)

    # answer_mask -> marks predictions whose confidence is high enough to ANSWER.
    # abstain_mask -> marks predictions whose confidence is low enough to ABSTAIN.
    # verify_mask -> marks predictions that are between the two thresholds, so they should be VERIFY.
    answer_mask = confidences >= answer_threshold
    abstain_mask = confidences <= abstain_threshold
    verify_mask = ~answer_mask & ~abstain_mask

    answer_count = int(np.sum(answer_mask))
    abstain_count = int(np.sum(abstain_mask))
    verify_count = int(np.sum(verify_mask))

    answer_correct = int(np.sum(correct_labels[answer_mask] == 1))
    answer_incorrect = int(np.sum(correct_labels[answer_mask] == 0))

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

    if verify_count > 0:
        verify_accuracy = verify_count / total_examples

    else:
        verify_accuracy = None

    answer_coverage = answer_count / total_examples
    abstain_rate = abstain_count / total_examples
    verify_rate = verify_count / total_examples
    direct_decision_rate = (answer_count + abstain_count) / total_examples
    false_answer_rate = answer_incorrect / total_examples
    unnecessary_abstention_rate = abstain_correct / total_examples

    return {
        "abstain_threshold": (float(abstain_threshold)),
        "answer_threshold": (float(answer_threshold)),
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
        "abstain_correct_rate": (abstain_correct_rate),
        "verify_accuracy": verify_accuracy,
        "answer_coverage": answer_coverage,
        "verify_rate": verify_rate,
        "abstain_rate": abstain_rate,
        "direct_decision_rate": (direct_decision_rate),
        "false_answer_rate": (false_answer_rate),
        "unnecessary_abstention_rate": (unnecessary_abstention_rate),
    }


# generates a unique set of candidate threshold values from the confidence score distribution, allowing the system to efficiently search for the best abstain and answer thresholds.
def create_threshold_candidates(confidences: np.ndarray, grid_size: int) -> np.ndarray:
    # controls how many candidate threshold values are generated from the confidence distribution for the threshold search.
    # grid_size = 5 -> [0.12, 0.30, 0.54, 0.74, 0.95]
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3.")

    # Creates evenly spaced percentile positions between 0 and 1.
    # grid_size = 5 -> [0.00, 0.25, 0.50, 0.75, 1.00]
    quantiles = np.linspace(0.0, 1.0, grid_size)

    # Converts those percentile positions into actual confidence values.
    # confidences = [0.10, 0.20, 0.40, 0.60, 0.90]
    # quantiles = [0.00, 0.25, 0.50, 0.75, 1.00]
    # -> candidates = [0.10, 0.20, 0.40, 0.60, 0.90]
    candidates = np.quantile(confidences, quantiles)

    # Removes duplicate threshold values.
    candidates = np.unique(candidates)

    # Adds the extreme threshold values (0.0 and 1.0) to the candidate list, then removes duplicates.
    candidates = np.unique(
        np.concatenate([np.asarray([0.0]), candidates, np.asarray([1.0])])
    )

    return candidates


# searches through many possible abstain and answer threshold pairs, evaluates each pair, and selects the best one according to the given safety and performance constraints.
def select_thresholds(
    confidences: np.ndarray,
    correct_labels: np.ndarray,
    max_answer_risk: float,
    max_abstain_correct_rate: float,
    min_answer_rate: float,
    min_abstain_rate: float,
    grid_size: int,
) -> dict[str, Any]:
    # We need these 3 variables to balance safety and usefulness:
    # max_answer_risk limits how many wrong answers are allowed.
    # max_abstain_correct_rate limits unnecessary refusals.
    # min_answer_rate prevents the model from avoiding too many questions

    # max_answer_risk -> the maximum mistake rate allowed when the model answers
    if not (0.0 <= max_answer_risk <= 1.0):
        raise ValueError("max_answer_risk must be between 0 and 1.")

    # ax_abstain_correct_rate -> maximum allowed rate of unnecessary abstentions.
    if not (0.0 <= max_abstain_correct_rate <= 1.0):
        raise ValueError("max_abstain_correct_rate must be between 0 and 1.")

    # the minimum proportion of examples that the model must answer directly
    if not (0.0 <= min_answer_rate <= 1.0):
        raise ValueError("min_answer_rate must be between 0 and 1.")

    total_examples = len(confidences)

    # the minimum number of examples the model must answer.
    minimum_answer_count = max(
        1,
        # math.ceil() rounds a decimal up to the next whole number.
        math.ceil(total_examples * min_answer_rate),
    )

    # minimum number of examples on which the model must choose ABSTAIN.
    minimum_abstain_count = max(1, math.ceil(total_examples * min_abstain_rate))

    candidates = create_threshold_candidates(
        confidences=confidences, grid_size=grid_size
    )

    # threshold pairs that meet all requirements.
    feasible_results: list[dict[str, Any]] = []

    # threshold pairs that don't meet the requirements but are saved as backups.
    falback_results: list[dict[str, Any]] = []

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

            # the proportion of wrong answers among the predictions marked ANSWER.
            answer_risk = result["answer_risk"]

            # the proportion of actually correct predictions among those marked ABSTAIN, meaning unnecessary refusals.
            abstain_correct_rate = result["abstain_correct_rate"]

            if answer_risk is None or abstain_correct_rate is None:
                continue

            # how far the actual answer risk exceeds the allowed limit.
            answer_risk_violation = max(0.0, answer_risk - max_answer_risk)

            # how far unnecessary abstentions exceed the allowed limit.
            abstain_violation = max(
                0.0, abstain_correct_rate - max_abstain_correct_rate
            )

            result["anser_risk_violation"] = answer_risk_violation
            result["abstain_violation"] = abstain_violation
            result["total_constraint_violation"] = (
                answer_risk_violation + abstain_violation
            )

            # stores an infeasible threshold pair as a backup in case no feasible solution exists
            falback_results.append(result)

            constraints_satisfied = (
                answer_risk <= max_answer_risk
                and abstain_correct_rate <= max_abstain_correct_rate
            )

            if constraints_satisfied:
                feasible_results.append(result)

    # Check whether there is at least one threshold pair that satisfies all constraints
    # Priority order:
    # Highest answer_coverage
    # Highest direct_decision_rate
    # Lowest answer_risk
    # Lowest abstain_correct_rate
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
        if not falback_results:
            raise ValueError(
                "No threshold pair produced enough ANSWER and ABSTAIN examples."
            )

        # Select the least bad threshold pair according to the comparison rules in key.
        best_result = min(
            falback_results,
            key=lambda result: (
                result["total_constraint_violation"],
                -result["answer_coverage"],
                -result["direct_decision_rate"],
            ),
        )

        # Indicate that the selected threshold pair does not satisfy all constraints but is the best available.
        constraints_satisfied = False

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
                        for result in falback_results
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


# abels every prediction with its final decision and records the thresholds used to make that decision.
def annotate_predictions(
    predictions: list[dict[str, Any]],
    abstain_threshold: float,
    answer_threshold: float,
) -> list[dict[str, Any]]:
    annotated_predictions: list[dict[str, Any]] = []

    for prediction in predictions:
        confidence = float(prediction["confidence"])

        decision = assign_decision(
            confidence=confidence,
            abstain_threshold=abstain_threshold,
            answer_threshold=answer_threshold,
        )

        updated_prediction = prediction.copy()

        updated_prediction.update(
            {
                "decision": decision,
                "abstain_threshold": (abstain_threshold),
                "answer_threshold": (answer_threshold),
                "threshold_source": ("calibration_split"),
            }
        )

        annotated_predictions.append(updated_prediction)

    return annotated_predictions


# orchestrates the entire threshold selection workflow from loading predictions to saving the final thresholds and annotated predictions.
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
        abstain_threshold=(abstain_threshold),
        answer_threshold=(answer_threshold),
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
    """Terminal argümanlarını okur."""

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
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_threshold_selection(
        input_path=args.input,
        output_path=args.output,
        annotated_output_path=(args.annotated_output),
        max_answer_risk=(args.max_answer_risk),
        max_abstain_correct_rate=(args.max_abstain_correct_rate),
        min_answer_rate=(args.min_answer_rate),
        min_abstain_rate=(args.min_abstain_rate),
        grid_size=args.grid_size,
    )
