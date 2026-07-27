"""
Evaluation metrics for selective question answering.

Bu dosyanın görevleri:
1. Model cevabını reference cevaplarla karşılaştırmak.
2. ANSWER ve ABSTAIN kararlarının doğruluğunu ölçmek.
3. Exact Match ve Token F1 hesaplamak.
4. Accuracy, coverage ve selective risk hesaplamak.
5. Değerlendirilen prediction'ları JSONL olarak kaydetmek.
"""

import argparse
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.io import load_jsonl, save_jsonl


def normalize_answer(text: str) -> str:
    """
    Cevapları adil şekilde karşılaştırmak için normalize eder.

    Yapılan işlemler:
    - küçük harfe çevirme
    - noktalama işaretlerini kaldırma
    - a, an, the article'larını kaldırma
    - gereksiz boşlukları temizleme

    Örnek:
        "The James Watt!" -> "james watt"
    """

    if not isinstance(text, str):
        text = str(text)

    text = text.lower()

    text = "".join(
        character
        for character in text
        if character not in string.punctuation
    )

    text = re.sub(r"\b(a|an|the)\b", " ", text)

    return " ".join(text.split())


def exact_match_score(
    prediction: str,
    references: list[str],
) -> float:
    """
    Prediction, reference cevaplardan biriyle tamamen aynı mı?

    Normalize edilmiş prediction herhangi bir normalize edilmiş
    reference ile eşleşirse 1.0, aksi hâlde 0.0 döndürür.
    """

    if not references:
        return 0.0

    normalized_prediction = normalize_answer(prediction)

    return float(
        any(
            normalized_prediction == normalize_answer(reference)
            for reference in references
        )
    )


def token_f1_score(
    prediction: str,
    references: list[str],
) -> float:
    """
    Prediction ve reference cevaplar arasındaki kelime örtüşmesini ölçer.

    Birden fazla reference varsa en yüksek F1 skoru kullanılır.

    Örnek:
        prediction = "viral"
        reference = "viral antigens"

        Exact Match = 0.0
        Token F1 > 0.0
    """

    if not references:
        return 0.0

    prediction_tokens = normalize_answer(prediction).split()

    if not prediction_tokens:
        return 0.0

    best_f1 = 0.0

    for reference in references:
        reference_tokens = normalize_answer(reference).split()

        if not reference_tokens:
            continue

        common_tokens = (
            Counter(prediction_tokens)
            & Counter(reference_tokens)
        )

        overlap = sum(common_tokens.values())

        if overlap == 0:
            continue

        precision = overlap / len(prediction_tokens)
        recall = overlap / len(reference_tokens)

        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

        best_f1 = max(best_f1, f1)

    return best_f1


def evaluate_single_prediction(
    prediction: dict[str, Any],
) -> dict[str, Any]:
    """
    Tek bir prediction'ın doğruluğunu değerlendirir.

    Karar mantığı:

    ANSWER + answerable:
        Cevap reference cevaplarla karşılaştırılır.

    ANSWER + unanswerable:
        Model cevap vermemeliydi; karar yanlış kabul edilir.

    ABSTAIN + unanswerable:
        Doğru abstention kabul edilir.

    ABSTAIN + answerable:
        Gereksiz abstention kabul edilir.
    """

    required_fields = {
        "decision",
        "is_answerable",
    }

    missing_fields = required_fields - prediction.keys()

    if missing_fields:
        missing_text = ", ".join(sorted(missing_fields))

        raise KeyError(
            f"Prediction is missing required fields: {missing_text}"
        )

    decision = str(prediction["decision"]).upper()
    is_answerable = bool(prediction["is_answerable"])

    predicted_answer = str(
        prediction.get("prediction_text", "")
    )

    references = prediction.get("reference_answers", [])

    if references is None:
        references = []

    if not isinstance(references, list):
        raise TypeError(
            "'reference_answers' must be a list of strings."
        )

    references = [
        str(reference)
        for reference in references
    ]

    if decision == "ANSWER":
        if not is_answerable:
            exact_match = 0.0
            token_f1 = 0.0
            is_correct = False

        else:
            exact_match = exact_match_score(
                predicted_answer,
                references,
            )

            token_f1 = token_f1_score(
                predicted_answer,
                references,
            )

            # Strict correctness için Exact Match kullanılır.
            is_correct = exact_match == 1.0

    elif decision == "ABSTAIN":
        exact_match = 0.0
        token_f1 = 0.0

        # Unanswerable soruda abstain etmek doğrudur.
        is_correct = not is_answerable

    else:
        raise ValueError(
            f"Unknown decision: {decision}. "
            "Expected 'ANSWER' or 'ABSTAIN'."
        )

    evaluated = prediction.copy()

    evaluated.update(
        {
            "decision": decision,
            "exact_match": exact_match,
            "token_f1": token_f1,
            "is_correct": is_correct,
        }
    )

    return evaluated


def calculate_metrics(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Bütün prediction'lar için selective QA metriklerini hesaplar.
    """

    if not predictions:
        raise ValueError(
            "Prediction list cannot be empty."
        )

    evaluated_predictions = [
        evaluate_single_prediction(prediction)
        for prediction in predictions
    ]

    total = len(evaluated_predictions)

    answered_predictions = [
        prediction
        for prediction in evaluated_predictions
        if prediction["decision"] == "ANSWER"
    ]

    abstained_predictions = [
        prediction
        for prediction in evaluated_predictions
        if prediction["decision"] == "ABSTAIN"
    ]

    answered = len(answered_predictions)
    abstained = len(abstained_predictions)

    total_correct = sum(
        int(prediction["is_correct"])
        for prediction in evaluated_predictions
    )

    answered_correct = sum(
        int(prediction["is_correct"])
        for prediction in answered_predictions
    )

    answerable_count = sum(
        int(bool(prediction["is_answerable"]))
        for prediction in evaluated_predictions
    )

    unanswerable_count = total - answerable_count

    unnecessary_abstentions = sum(
        int(
            prediction["decision"] == "ABSTAIN"
            and bool(prediction["is_answerable"])
        )
        for prediction in evaluated_predictions
    )

    answered_unanswerable = sum(
        int(
            prediction["decision"] == "ANSWER"
            and not bool(prediction["is_answerable"])
        )
        for prediction in evaluated_predictions
    )

    correct_abstentions = sum(
        int(
            prediction["decision"] == "ABSTAIN"
            and not bool(prediction["is_answerable"])
        )
        for prediction in evaluated_predictions
    )

    exact_match_total = sum(
        float(prediction["exact_match"])
        for prediction in evaluated_predictions
    )

    token_f1_total = sum(
        float(prediction["token_f1"])
        for prediction in evaluated_predictions
    )

    answered_exact_match_total = sum(
        float(prediction["exact_match"])
        for prediction in answered_predictions
    )

    answered_token_f1_total = sum(
        float(prediction["token_f1"])
        for prediction in answered_predictions
    )

    accuracy = total_correct / total

    coverage = answered / total

    abstention_rate = abstained / total

    answered_accuracy = (
        answered_correct / answered
        if answered > 0
        else 0.0
    )

    selective_risk = (
        1.0 - answered_accuracy
        if answered > 0
        else 0.0
    )

    exact_match = exact_match_total / total
    token_f1 = token_f1_total / total

    answered_exact_match = (
        answered_exact_match_total / answered
        if answered > 0
        else 0.0
    )

    answered_token_f1 = (
        answered_token_f1_total / answered
        if answered > 0
        else 0.0
    )

    unnecessary_abstention_rate = (
        unnecessary_abstentions / answerable_count
        if answerable_count > 0
        else 0.0
    )

    unanswerable_answer_rate = (
        answered_unanswerable / unanswerable_count
        if unanswerable_count > 0
        else 0.0
    )

    correct_abstention_rate = (
        correct_abstentions / unanswerable_count
        if unanswerable_count > 0
        else 0.0
    )

    return {
        "total": total,
        "answered": answered,
        "abstained": abstained,
        "answerable_count": answerable_count,
        "unanswerable_count": unanswerable_count,
        "total_correct": total_correct,
        "answered_correct": answered_correct,
        "correct_abstentions": correct_abstentions,
        "unnecessary_abstentions": unnecessary_abstentions,
        "answered_unanswerable": answered_unanswerable,
        "accuracy": accuracy,
        "coverage": coverage,
        "abstention_rate": abstention_rate,
        "answered_accuracy": answered_accuracy,
        "selective_risk": selective_risk,
        "exact_match": exact_match,
        "token_f1": token_f1,
        "answered_exact_match": answered_exact_match,
        "answered_token_f1": answered_token_f1,
        "unnecessary_abstention_rate": (
            unnecessary_abstention_rate
        ),
        "unanswerable_answer_rate": (
            unanswerable_answer_rate
        ),
        "correct_abstention_rate": (
            correct_abstention_rate
        ),
    }


def print_metrics(metrics: dict[str, Any]) -> None:
    """
    Metrikleri terminalde okunabilir şekilde gösterir.
    """

    print("\nEvaluation Results")
    print("=" * 40)

    for name, value in metrics.items():
        readable_name = name.replace("_", " ").title()

        if isinstance(value, float):
            print(f"{readable_name}: {value:.4f}")
        else:
            print(f"{readable_name}: {value}")


def parse_args() -> argparse.Namespace:
    """
    Terminal argümanlarını tanımlar.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate predictions for selective "
            "question answering."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input predictions JSONL file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output JSONL path for evaluated "
            "predictions."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Evaluation komut satırı giriş noktası.
    """

    args = parse_args()

    predictions = load_jsonl(args.input)

    evaluated_predictions = [
        evaluate_single_prediction(prediction)
        for prediction in predictions
    ]

    metrics = calculate_metrics(predictions)

    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_jsonl(
            evaluated_predictions,
            args.output,
        )

        print(
            f"Evaluated predictions saved to: {args.output}"
        )

    print_metrics(metrics)


if __name__ == "__main__":
    main()