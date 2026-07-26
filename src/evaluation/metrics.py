"""
Evaluation metrics for selective question answering.

Bu dosyanın görevi:
1. Model cevabını reference cevaplarla karşılaştırmak.
2. ANSWER ve ABSTAIN kararlarının doğru olup olmadığını ölçmek.
3. Accuracy, coverage ve selective risk hesaplamak.

Bu dosya research projesinin önemli AI/ML evaluation kısmıdır.
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

    Örnek:
        "The James Watt!" → "james watt"

    Yapılan işlemler:
    - küçük harfe çevirme
    - noktalama işaretlerini kaldırma
    - a, an, the gibi article'ları kaldırma
    - gereksiz boşlukları temizleme
    """

    text = text.lower()

    # Noktalama işaretlerini kaldır.
    text = "".join(
        character for character in text if character not in string.punctuation
    )

    # İngilizce article'ları kaldır.
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # Birden fazla boşluğu tek boşluğa dönüştür.
    return " ".join(text.split())


def exact_match_score(prediction: str, references: list[str]) -> float:
    """
    Prediction reference cevaplardan biriyle tamamen aynı mı?

    Örnek:
        prediction = "James Watt"
        reference = "James Watt"
        -> 1.0

        prediction = "Watt"
        reference = "James Watt"
        -> 0.0
    """

    if not references:
        return 0.0

    normalized_prediction = normalize_answer(prediction)

    # any(...), içindeki karşılaştırmalardan en az biri True ise True, hiçbiri eşleşmezse False döndürür.
    # Burada model cevabı, references içindeki doğru cevaplardan herhangi biriyle eşleşiyor mu diye kontrol eder; dışındaki float(...) da sonucu True → 1.0, False → 0.0 yapar.
    return float(
        any(
            normalized_prediction == normalize_answer(reference)
            for reference in references
        )
    )


def toekn_f1_score(prediction: str, references: list[str]) -> float:
    """
    Prediction ve reference arasındaki kelime örtüşmesini ölçer.

    Exact Match tamamen aynı cevap ister.
    Token F1 ise kısmen doğru cevaplara da puan verebilir.

    Örnek:
        prediction = "viral"
        reference = "viral antigens"

    Exact Match = 0
    Token F1 > 0
    """

    if not references:
        return 0.0

    prediction_tokens = normalize_answer(prediction).split()

    if not prediction_tokens:
        return 0.0

    best_f1 = 0.0

    # Bir sorunun birden fazla kabul edilen cevabı olabilir.
    for reference in references:
        reference_tokens = normalize_answer(reference).split()

        if not reference_tokens:
            continue

        # Prediction ve reference içindeki ortak kelimeleri bul.
        # Counter(prediction_tokens) & Counter(reference_tokens), tahmin ile doğru cevaptaki ortak kelimeleri ve kaç kez ortak olduklarını bulur.
        common_tokens = Counter(prediction_tokens) & Counter(reference_tokens)

        # overlap ise bu ortak kelimelerin toplam sayısıdır.
        overlap = sum(common_tokens.values())

        if overlap == 0:
            continue

        # prediction_tokens, modelin ürettiği cevabın kelimelere ayrılmış hâlidir;
        # reference_tokens ise doğru cevabın kelimelere ayrılmış hâlidir.
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(reference_tokens)

        f1 = 2 * precision * recall / (precision + recall)

        best_f1 = max(best_f1, f1)

    return best_f1


def evaluate_single_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    """
    Tek bir prediction'ın doğru olup olmadığını değerlendirir.

    Karar mantığı:

    ANSWER + answerable:
        Model cevabı reference ile karşılaştırılır.

    ANSWER + unanswerable:
        Yanlış kabul edilir çünkü context'te cevap yoktur.

    ABSTAIN + unanswerable:
        Doğru karar kabul edilir.

    ABSTAIN + answerable:
        Gereksiz abstention kabul edilir.
    """

    decision = prediction["decision"]
    is_answerable = bool(prediction["is_answerable"])

    predicted_answer = prediction.get("prediction_text", "")

    references = prediction.get("reference_answers", [])

    # Model cevap verdiyse cevabın doğruluğunu ölç.
    if decision == "ANSWER":
        # Context'te cevap yokken cevap vermek yanlıştır.
        if not is_answerable:
            exact_match = 0.0
            token_f1 = 0.0
            is_correct = False

        else:
            exact_match = exact_match_score(predicted_answer, references)

            # Burada strict correctness için Exact Match kullanıyoruz.
            is_correct = exact_match == 1.0

    # Model abstain ettiyse cevap metnini değerlendirmiyoruz.
    elif decision == "ABSTAIN":
        exact_match = 0.0
        token_f1 = 0.0

        # Unanswerable soruda abstain etmek doğru karardır.
        is_correct = not is_answerable

    else:
        raise ValueError(f"Unknown decision: {decision}")

    evaluated = prediction.copy()

    evaluated.update(
        {"exact_match": exact_match, "token_f1": token_f1, "is_correct": is_correct}
    )

    return evaluated


def calculate_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Bütün prediction'lar için araştırma metriklerini hesaplar.
    """

    if not predictions:
        raise ValueError("Prediction list cannot be empty.")

    evaluated_predictions = [
        evaluate_single_prediction(prediction) for prediction in predictions
    ]

    total = len(evaluated_predictions)

    answered_predictions = [
        prediction for prediction in predictions if prediction["decision"] == "ANSWER"
    ]

    abstained_predictions = [
        prediction for prediction in predictions if prediction["decision"] == "ABSTAIN"
    ]

    answered = len(answered_predictions)
    abstained = len(abstained_predictions)

    total_correct = sum(
        prediction["is_correct"] for prediction in evaluated_predictions
    )

    answered_correct = sum(
        prediction["is_correct"] for prediction in answered_predictions
    )

    answerable_count = sum(
        prediction["is_answerable"] for prediction in evaluated_predictions
    )

    unanswerable_count = total - answerable_count

    # Answerable olduğu hâlde model abstain etmiş.
    unnecessary_abstantions = sum(
        prediction["decision"] == "ABSTAIN" and prediction["is_answerable"]
        for prediction in evaluated_predictions
    )

    # Context'te cevap olmadığı hâlde model cevap vermiş.
    answered_unanswerable = sum(
        prediction["decision"] == "ANSWER" and not prediction["is_answerble"]
        for prediction in evaluated_predictions
    )

    coverage = answered / total
    abstantion_rate = abstained / total

    # Selective accuracy yalnızca cevap verilen örneklerde ölçülür.
    selective_accuracy = answered_correct / answered if answered > 0 else 0.0

    # Selective risk, cevap verilen örneklerdeki hata oranıdır.
    selective_risk = 1.0 - selective_accuracy if answered > 0 else 0.0

    mean_exact_match = (
        sum(prediction["exact_match"] for prediction in answered_predictions) / answered
        if answered > 0
        else 0.0
    )

    mean_token_f1 = (
        sum(prediction["token_f1"] for prediction in answered_predictions) / answered
        if answered > 0
        else 0.0
    )

    return {
        "system": predictions[0].get("syste", "unknown"),
        "total_examples": total,
        "answered_examples": answered,
        "abstained_examples": abstained,
        # Bütün örneklerde doğru sistem kararı oranı.
        "overall_accuracy": total_correct / total,
        # Sistemin cevap verdiği örneklerin oranı.
        "coverage": coverage,
        # Sistemin cevap vermediği örneklerin oranı.
        "abstention_rate": abstantion_rate,
        # Yalnızca cevap verilen örneklerde doğruluk.
        "selective_accuracy": selective_accuracy,
        # Yalnızca cevap verilen örneklerde hata oranı.
        "selective_risk": selective_risk,
        "mean_exact_match_answered": mean_exact_match,
        "mean_token_f1_answered": mean_token_f1,
        "unnecessary_abstentions": (unnecessary_abstantions),
        "unnecessary_abstention_rate": (
            unnecessary_abstantions / answerable_count if answerable_count > 0 else 0.0
        ),
        "answered_unanswerable_examples": (answered_unanswerable),
        "unsupported_answer_rate": (
            answered_unanswerable / unanswerable_count
            if unanswerable_count > 0
            else 0.0
        ),
    }


def run_avluation(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """
    Prediction dosyasını okur ve metrikleri kaydeder.
    """

    predictions = load_jsonl(input_path)

    metrics = calculate_metrics(predictions)

    save_jsonl(metrics, output_path)

    print("\nEvaluation completed.")

    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            print(f"{metric_name}: {metric_value:.4f}")
        else:
            print(f"{metric_name}: {metric_value}")

    print(f"\nMetrics saved to: {output_path}")

    return metrics


def parse_arguments() -> argparse.Namespace:
    """Terminal argümanlarını okur."""

    parser = argparse.ArgumentParser(description=("Evaluate selective QA predictions."))

    parser.add_argument(
        "--imput",
        required=True,
        help="Prediction JSONL file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Metrics JSON file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_avluation(
        input_path=args.input,
        output_path=args.output,
    )
