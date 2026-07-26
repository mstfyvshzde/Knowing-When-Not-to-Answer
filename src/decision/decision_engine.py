"""
Final decision engine for selective question answering.

Bu dosyanın görevi:

1. Calibration aşamasından gelen confidence kararını okumak.
2. Evidence verifier sonucunu okumak.
3. Bu iki sinyali birleştirerek final karar üretmek.

Final kararlar:

    ANSWER
        Sistem cevabı doğrudan kullanıcıya gösterir.

    VERIFY
        Sistem cevabı göstermeden önce ek doğrulama ister.

    ABSTAIN
        Sistem güvenilir bir cevap veremediğini söyler.

Bu karar motoru deterministic ve açıklanabilirdir.
Her karar için decision_reason alanı kaydedilir.
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.io import load_jsonl, save_jsonl

DEFAULT_INPUT_PATH = Path("outputs/predictions/calibration_with_evidence.jsonl")

DEFAULT_OUTPUT_PATH = Path("outputs/predictions/calibration_final_decisions.jsonl")


VALID_THRESHOLD_DECISIONS = {
    "ANSWER",
    "VERIFY",
    "ABSTAIN",
}

VALID_EVIDENCE_LABELS = {
    "SUPPORTED",
    "WEAK",
    "UNSUPPORTED",
}


def get_threshold_decision(prediction: dict[str, Any]) -> str:
    """
    Threshold selection aşamasında üretilen
    confidence-based kararı bulur.

    Farklı dosya sürümlerinde alan adı değişmiş
    olabilir. Bu nedenle birkaç olası alan
    desteklenir.
    """

    possible_fields = (
        "threshold_decision",  # Eşik değerine göre verilen karar.
        "confidence_decision",  # Güven skoruna göre verilen karar.
        "decision",  # Genel/standart karar alanı.
        "selective_decision",  # Yalnızca yeterince güvenilen durumlarda verilen seçici karar.
    )

    for field in possible_fields:
        value = prediction.get(field)

        if value is None:
            continue

        normalized_value = str(value).strip().upper()

        if normalized_value in (VALID_THRESHOLD_DECISIONS):
            return normalized_value

    raise ValueError(
        "Prediction does not contain a valid "
        "threshold decision. Expected one of "
        f"{possible_fields}. Available keys: "
        f"{list(prediction.keys())}"
    )


def get_evidence_support(prediction: dict[str, Any]) -> str:
    """
    Evidence verifier tarafından üretilen
    support label'ını okur.
    """

    evidence_support = str(prediction.get("evidence_support", "")).strip().upper()

    if evidence_support not in (VALID_EVIDENCE_LABELS):
        raise ValueError(
            "Prediction contains an invalid "
            f"evidence_support value: "
            f"{evidence_support!r}."
        )

    return evidence_support


def get_confidence(prediction: dict[str, Any]) -> float:
    """
    Prediction kaydından calibrated confidence
    değerini okur.

    Öncelik calibrated confidence alanındadır.
    Eski dosya sürümleri için confidence alanı
    fallback olarak desteklenir.
    """

    possible_fields = (
        "calibrated_confidence",
        "confidence",
    )

    for field in possible_fields:
        value = prediction.get(field)

        if value is None:
            continue

        try:
            confidence = float(value)

        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid confidence value in field {field!r}: {value!r}."
            ) from error

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0 and 1, received {confidence}."
            )

        return confidence

    raise ValueError("Prediction does not contain calibrated_confidence or confidence.")


def combine_decisions(
    threshold_decision: str, evidence_support: str
) -> tuple[str, str]:
    """
    Confidence-based karar ile evidence desteğini
    birleştirerek final karar üretir.

    Yeni strateji:

    1. ANSWER kararları korunur.
    2. ABSTAIN kararları korunur.
    3. Evidence verifier yalnızca VERIFY
       bölgesinde kullanılır.
    """

    # Model yüksek güvenli ANSWER bölgesindeyse,
    # lexical evidence verifier bu kararı düşürmez.
    if threshold_decision == "ANSWER":
        return (
            "ANSWER",
            "high_confidence_answer_preserved",
        )

    # Modelin güveni çok düşükse evidence,
    # cevabı doğrudan yükseltmemelidir.
    if threshold_decision == "ABSTAIN":
        return (
            "ABSTAIN",
            "confidence_below_abstain_threshold",
        )

    # Belirsiz confidence bölgesinde güçlü evidence,
    # cevabı ANSWER seviyesine yükseltebilir.
    if threshold_decision == "VERIFY" and evidence_support == "SUPPORTED":
        return (
            "ANSWER",
            "medium_confidence_with_supported_evidence",
        )

    # Confidence ve evidence birlikte belirsizse
    # ek doğrulama gerekir.
    if threshold_decision == "VERIFY" and evidence_support == "WEAK":
        return (
            "VERIFY",
            "medium_confidence_and_weak_evidence",
        )

    # Belirsiz confidence ve desteklenmeyen evidence
    # birlikteyse cevap verilmez.
    if threshold_decision == "VERIFY" and evidence_support == "UNSUPPORTED":
        return (
            "ABSTAIN",
            "medium_confidence_and_unsupported_evidence",
        )

    return (
        "ABSTAIN",
        "unhandled_decision_combination",
    )


def process_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    """
    Tek bir prediction kaydı için final karar üretir.
    """

    threshold_decision = get_threshold_decision(prediction)

    evidence_support = get_evidence_support(prediction)

    confidence = get_confidence(prediction)

    fnal_decision, decision_reason = combine_decisions(
        threshold_decision=(threshold_decision), evidence_support=(evidence_support)
    )

    updated_prediction = prediction.copy()

    updated_prediction.update(
        {
            "threshold_decision": (threshold_decision),
            "final_decision": (fnal_decision),
            "decision_reason": (decision_reason),
            "decision_engine": ("confidence_evidence_rule_based"),
            "decision_confidence": (confidence),
        }
    )

    return updated_prediction


def validate_predictions(predictions: list[dict[str, Any]]) -> None:
    """
    Input prediction listesinin boş olmadığını ve
    gerekli evidence alanını içerdiğini kontrol eder.
    """

    if not predictions:
        raise ValueError("Prediction list cannot be empty.")

    for index, prediction in enumerate(predictions, start=1):
        if "evidence_support" not in prediction:
            raise ValueError(
                f"Prediction {index} does not "
                "contain evidence_support. "
                "Run evidence_verifier.py first."
            )


def calculate_decision_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Final karar dağılımını ve mümkünse her karar
    bölgesindeki doğruluk oranını hesaplar.
    """

    total = len(predictions)

    decision_counts = Counter(
        prediction["final_decision"] for prediction in predictions
    )

    metrics: dict[str, Any] = {
        "total": total,
        "answer_count": decision_counts["ANSWER"],
        "verify_count": decision_counts["VERIFY"],
        "abstain_count": decision_counts["ABSTAIN"],
        "answer_rate": (decision_counts["ANSWER"] / total),
        "verify_rate": (decision_counts["VERIFY"] / total),
        "abstain_rate": (decision_counts["ABSTAIN"] / total),
    }

    # Prediction kayıtlarında is_correct alanı
    # bulunuyorsa karar bölgelerinin doğruluğu da
    # hesaplanır.
    if all("is_correct" in prediction for prediction in predictions):
        for decision in (
            "ANSWER",
            "VERIFY",
            "ABSTAIN",
        ):
            decision_predictions = [
                prediction
                for prediction in predictions
                if prediction["final_decision"] == decision
            ]

            if decision_predictions:
                correct_count = sum(
                    bool(prediction["is_correct"])
                    for prediction in decision_predictions
                )

                accuracy = correct_count / len(decision_predictions)

            else:
                accuracy = None

            metrics[f"{decision.lower()}_accuracy"] = accuracy

        answer_accuracy = metrics.get("answer_accuracy")

        if answer_accuracy is not None:
            metrics["answer_risk"] = 1.0 - answer_accuracy

    return metrics


def print_decision_summary(
    metrics: dict[str, Any],
) -> None:
    """
    Decision engine sonuçlarını terminale yazdırır.
    """

    print("\nFinal decision generation completed.")

    print(f"Total predictions: {metrics['total']}")

    print(f"ANSWER: {metrics['answer_count']} ({metrics['answer_rate']:.4f})")

    print(f"VERIFY: {metrics['verify_count']} ({metrics['verify_rate']:.4f})")

    print(f"ABSTAIN: {metrics['abstain_count']} ({metrics['abstain_rate']:.4f})")

    if "answer_risk" in metrics:
        print(f"Final answer risk: {metrics['answer_risk']:.4f}")

    for decision in (
        "answer",
        "verify",
        "abstain",
    ):
        metric_name = f"{decision}_accuracy"

        accuracy = metrics.get(metric_name)

        if accuracy is not None:
            print(f"{decision.upper()} accuracy: {accuracy:.4f}")


def run_decision_engine(
    input_path: str | Path,
    output_path: str | Path,
) -> list[dict[str, Any]]:
    """
    Tüm prediction kayıtları için final karar üretir.
    """

    predictions = load_jsonl(input_path)

    validate_predictions(predictions)

    final_predictions: list[dict[str, Any]] = []

    for index, prediction in enumerate(
        predictions,
        start=1,
    ):
        final_prediction = process_prediction(prediction)

        final_predictions.append(final_prediction)

        print(
            f"{index}/{len(predictions)} | "
            f"threshold="
            f"{final_prediction['threshold_decision']} | "
            f"evidence="
            f"{final_prediction['evidence_support']} | "
            f"final="
            f"{final_prediction['final_decision']}"
        )

    save_jsonl(
        final_predictions,
        output_path,
    )

    metrics = calculate_decision_metrics(final_predictions)

    print_decision_summary(metrics)

    print(f"Results saved to: {output_path}")

    return final_predictions


def parse_arguments() -> argparse.Namespace:
    """
    Terminal argümanlarını okur.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Combine calibrated confidence "
            "decisions and evidence verification "
            "into final ANSWER, VERIFY or ABSTAIN "
            "decisions."
        )
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help=(
            "JSONL file containing calibrated "
            "confidence decisions and evidence "
            "verification results."
        ),
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=("Output JSONL file for final decisions."),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_decision_engine(
        input_path=args.input,
        output_path=args.output,
    )
