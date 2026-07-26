"""
Evidence verifier for extractive question answering.

Bu dosyanın görevi:

1. Model cevabının context içinde bulunup bulunmadığını kontrol etmek.
2. Cevabın çevresindeki evidence span'i çıkarmak.
3. Question ile evidence arasındaki kelime örtüşmesini ölçmek.
4. Evidence desteğini SUPPORTED, WEAK veya UNSUPPORTED
   olarak sınıflandırmak.

Bu ilk sürüm açıklanabilir ve deterministic bir baseline'dır.
Daha sonra NLI veya learned verifier ile karşılaştırılabilir.
"""

import argparse
import re
import string
from pathlib import Path
from typing import Any

from src.utils.io import load_jsonl, save_jsonl

DEFAULT_INPUT_PATH = Path("outputs/predictions/calibration_with_decisions.jsonl")

DEFAULT_OUTPUT_PATH = Path("outputs/predictions/calibration_with_evidence.jsonl")


# Question içinde çok sık görülen ve evidence ölçümüne fazla katkı sağlamayan kelimeler.
STOP_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "from",
    "with",
    "by",
    "and",
    "or",
    "but",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "when",
    "where",
    "why",
    "how",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "as",
    "about",
    "into",
    "than",
    "then",
    "also",
    "according",
    "name",
}


def normalize_text(text: str) -> str:
    """
    Metni karşılaştırma için normalize eder.

    İşlemler:
    - Küçük harfe çevirir.
    - Noktalama işaretlerini kaldırır.
    - Fazla boşlukları temizler.
    """

    text = text.lower()

    text = text.translate(str.maketrans("", "", string.punctuation))

    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_content_words(text: str) -> set[str]:
    """
    Metindeki anlamlı kelimeleri çıkarır.

    Stop word'ler çıkarılır.
    Çok kısa token'lar kullanılmaz.
    """

    normalized = normalize_text(text)

    tokens = normalized.split()

    content_words = {
        token for token in tokens if token not in STOP_WORDS and len(token) > 1
    }

    return content_words


def find_answer_position(context: str, answer: str) -> tuple[int, int] | None:
    """
    Predicted answer'ın context içindeki yerini bulur.

    Returns:
        (start_index, end_index)

    Answer bulunamazsa:
        None
    """

    if not answer.strip():
        return None

    context_lower = context.lower()
    answer_lower = answer.lower().strip()

    start_index = context_lower.find(answer_lower)

    if start_index == -1:
        return None

    end_index = start_index + len(answer_lower)

    return start_index, end_index


# Bu fonksiyon, metin içindeki cevabın başlangıç ve bitiş konumlarını kullanarak cevabın sağından ve solundan belirli sayıda karakter alıp bir kanıt metni (evidence window) oluşturmayı amaçlar.
def extract_evidence_window(
    context: str, answer_start: int, answer_end: int, window_size: int = 120
) -> str:
    """
    Cevabın çevresinden evidence window çıkarır.

    Cevabın hem solundan hem sağından belirli
    sayıda karakter alınır.
    """
    # window_start: alınacak metnin başladığı konum
    window_start = max(0, answer_start - window_size)

    # window_end: alınacak metnin bittiği konum
    window_end = min(len(context), answer_end + window_size)

    evidence = context[window_start:window_end].strip()

    return evidence


# Burada fonksiyon, sorudaki önemli kelimelerle evidence içindeki önemli kelimelerin ortak olanlarını bulur
def calculate_question_evidence_overlap(question: str, evidence: str) -> float:
    """
    Question ile evidence arasındaki content-word
    örtüşmesini hesaplar.

    Örnek:

        Question words:
            {"invented", "telephone"}

        Evidence words:
            {"alexander", "bell", "invented", "telephone"}

        overlap:
            2 / 2 = 1.0
    """

    question_words = tokenize_content_words(question)

    evidence_words = tokenize_content_words(evidence)

    if not question_words:
        return 0.0

    shared_words = question_words & evidence_words
    # sorudaki kelimelerin ne kadarının evidence içinde geçtiğini hesaplar.
    overlap = len(shared_words) / len(question_words)

    return float(overlap)


# Bu fonksiyon, cevabın context tarafından ne kadar desteklendiğini gösteren 0.0–1.0 arasında bir skor döndürür.
# İlk fonksiyon, cevabın çevresinden alınan kısa metin parçası olan **evidence’ın soruyla ne kadar örtüştüğünü** ölçer; ikinci fonksiyon ise tüm kaynak metin olan **context’in cevabı ne kadar desteklediğini** hesaplar.
def calculate_answer_context_score(answer: str, context: str) -> float:
    """
    Answer'ın context tarafından doğrudan
    desteklenip desteklenmediğini ölçer.

    Extractive QA sisteminde cevap context içinde
    tam olarak bulunuyorsa güçlü evidence vardır.
    """

    normalized_answer = normalize_text(answer)

    normalized_context = normalize_text(context)

    if not normalized_answer:
        return 0.0

    if normalized_answer in normalized_context:
        return 1.0

    answer_words = tokenize_content_words(answer)

    context_words = tokenize_content_words(context)

    if not answer_words:
        return 0.0

    shared_words = answer_words & context_words

    return float(len(shared_words) / len(answer_words))


# question_overlap, bulunan cevabın çevresindeki evidence’ın gerçekten soruyla ilgili olup olmadığını kontrol etmek için vardır. Çünkü cevap context içinde geçse bile, yanlış veya alakasız bir bölümden alınmış olabilir.
def classify_evidence_support(
    answer_context_score: float,  # Cevabın, tüm kaynak metin olan context içinde ne kadar desteklendiğini gösterir.
    question_overlap: float,  # Sorudaki önemli kelimelerin, cevabın çevresinden alınan kısa parça olan evidence içinde ne kadar geçtiğini gösterir.
    supported_threshold: float,
    weak_threshold: float,
) -> str:
    """
    Evidence score'larını üç sınıfa dönüştürür.

    SUPPORTED:
        Cevap context içinde açıkça bulunur ve
        question ile evidence uyumludur.

    WEAK:
        Kısmi evidence vardır fakat güçlü değildir.

    UNSUPPORTED:
        Cevap context tarafından yeterince
        desteklenmez.
    """

    # Combined score, answer_context_score ile question_overlap değerlerinin ağırlıklı birleşimidir. Yani cevabın context desteğini ve evidence’ın soruyla uyumunu tek bir puanda toplar.
    combined_score = 0.65 * answer_context_score + 0.35 * question_overlap

    if (
        answer_context_score >= 1.0
        and question_overlap >= 0.45
        and combined_score >= supported_threshold
    ):
        return "SUPPORTED"

    if answer_context_score >= 0.50 and combined_score >= weak_threshold:
        return "WEAK"

    return "UNSUPPORTED"


# Bu fonksiyonun amacı, tek bir QA tahminini inceleyip cevabın context içinde bulunup bulunmadığını, evidence’ın soruyla uyumunu ve birleşik destek skorunu hesaplamaktır. Sonunda tahmine SUPPORTED, WEAK veya UNSUPPORTED etiketi ekleyerek güncellenmiş sözlüğü döndürür
def verify_prediction(
    prediction: dict[str, Any],
    evidence_window_size: int,
    supported_threshold: float,
    weak_threshold: float,
) -> dict[str, Any]:
    """
    Tek bir QA prediction için evidence verification yapar.
    """

    question = str(prediction.get("question", ""))

    context = str(prediction.get("context", ""))

    predicted_answer = str(
        prediction.get(
            "predicted_answer",
            prediction.get("prediction_text", prediction.get("answer", "")),
        )
    )

    answer_position = find_answer_position(context=context, answer=predicted_answer)

    if answer_position is None:
        evidence_text = ""

        # answer_context_score, tahmin edilen cevabın context içinde ne kadar bulunduğunu veya desteklendiğini gösteren puandır. Genellikle 0.0 ile 1.0 arasındadır; 1.0 tam destek, 0.0 ise destek yok demektir.
        answer_context_score = calculate_answer_context_score(
            answer=predicted_answer, context=context
        )

        # question_overlap = 0.0, soru ile evidence arasında ortak önemli kelime bulunmadığını veya evidence olmadığı için örtüşmenin hesaplanamadığını gösterir. Yani uyum skoru sıfırdır
        question_overlap = 0.0

        combined_evidence_score = 0.65 * answer_context_score + 0.35 * question_overlap

    else:
        answer_start, answer_end = answer_position

        # evidence_text, tahmin edilen cevabın context içindeki bulunduğu kısmın çevresinden alınan kısa metin parçasıdır. Cevabı destekleyen cümleleri veya yakın bağlamı gösterir
        evidence_text = extract_evidence_window(
            context=context,
            answer_start=answer_start,
            answer_end=answer_end,
            window_size=(evidence_window_size),
        )

        answer_context_score = calculate_answer_context_score(
            answer=predicted_answer, context=context
        )

        question_overlap = calculate_question_evidence_overlap(
            question=question, evidence=evidence_text
        )

        combined_evidence_score = 0.65 * answer_context_score + 0.35 * question_overlap

    # Bu kod, answer_context_score ve question_overlap değerlerini eşiklerle karşılaştırıp sonucu SUPPORTED, WEAK veya UNSUPPORTED olarak belirler. Bu etiketi de support_label değişkenine kaydeder.
    support_label = classify_evidence_support(
        answer_context_score=(answer_context_score),
        question_overlap=(question_overlap),
        supported_threshold=(supported_threshold),
        weak_threshold=(weak_threshold),
    )

    updated_prediction = prediction.copy()

    updated_prediction.update(
        {
            "evidence_text": evidence_text,
            "answer_context_score": (answer_context_score),
            "question_evidence_overlap": (question_overlap),
            "evidence_score": (combined_evidence_score),
            "evidence_support": (support_label),
            "evidence_verifier": ("lexical_extractive_baseline"),
        }
    )

    return updated_prediction


def validate_predictions(predictions: list[dict[str, Any]]) -> None:
    """
    Gerekli alanların prediction kayıtlarında
    bulunduğunu kontrol eder.
    """

    if not predictions:
        raise ValueError("Prediction list cannot be empty.")

    required_fields = {"question", "context"}

    for index, prediction in enumerate(predictions, start=1):
        missing_fields = [field for field in required_fields if field not in prediction]

        has_answer_field = any(
            field in prediction
            for field in ("prediction_answer", "prediction_text", "answer")
        )

        if not has_answer_field:
            missing_fields.append("predicted_answer")

        if missing_fields:
            raise ValueError(
                f"Prediction {index} is missing "
                f"fields: {missing_fields}. "
                f"Available keys: "
                f"{list(prediction.keys())}"
            )


def run_evidence_verification(
    input_path: str | Path,
    output_path: str | Path,
    evidence_window_size: int,
    supported_threshold: float,
    weak_threshold: float,
) -> list[dict[str, Any]]:
    """
    Tüm prediction kayıtlarında evidence verification
    çalıştırır.
    """
    if not (0.0 <= weak_threshold < supported_threshold <= 1.0):
        raise ValueError("Thresholds must satisfy: 0 <= weak < supported <= 1.")

    predictions = load_jsonl(input_path)

    validate_predictions(predictions)

    verified_predictions: list[dict[str, Any]] = []

    for index, prediction in enumerate(predictions, start=1):
        verified_prediction = verify_prediction(
            prediction=prediction,
            evidence_window_size=(evidence_window_size),
            supported_threshold=(supported_threshold),
            weak_threshold=(weak_threshold),
        )

        verified_predictions.append(verified_prediction)

        print(
            f"{index}/{len(predictions)} | "
            f"evidence="
            f"{verified_prediction['evidence_score']:.4f} | "
            f"support="
            f"{verified_prediction['evidence_support']}"
        )

    save_jsonl(verified_predictions, output_path)

    support_counts = {
        "SUPPORTED": 0,
        "WEAK": 0,
        "UNSUPPORTED": 0,
    }

    for prediction in verified_predictions:
        support_counts[prediction["evidence_support"]] += 1

    print("\nEvidence verification completed.")

    print(f"SUPPORTED: {support_counts['SUPPORTED']}")

    print(f"WEAK: {support_counts['WEAK']}")

    print(f"UNSUPPORTED: {support_counts['UNSUPPORTED']}")

    print(f"Results saved to: {output_path}")

    return verified_predictions


def parse_arguments() -> argparse.Namespace:
    """
    Terminal argümanlarını okur.
    """

    parser = argparse.ArgumentParser(
        description=("Verify whether QA predictions are supported by their context.")
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
    )

    parser.add_argument(
        "--evidence-window-size",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--supported-threshold",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--weak-threshold",
        type=float,
        default=0.40,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_evidence_verification(
        input_path=args.input,
        output_path=args.output,
        evidence_window_size=(args.evidence_window_size),
        supported_threshold=(args.supported_threshold),
        weak_threshold=(args.weak_threshold),
    )
