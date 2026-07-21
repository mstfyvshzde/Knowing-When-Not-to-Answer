"""
Independent evidence checker for generated QA answers.

Bu dosyanın görevi:
1. Raw baseline tarafından üretilen cevapları okumak.
2. Aynı question + context çiftini farklı bir QA modeline vermek.
3. Verifier modelinin cevabını raw model cevabıyla karşılaştırmak.
4. Cevabı SUPPORTED, UNSUPPORTED veya UNCERTAIN olarak etiketlemek.

Önemli:
Bu verifier kusursuz bir truth detector değildir.

Sadece şu soruyu araştırır:

"Bağımsız ikinci bir model, ilk modelin cevabını aynı evidence
üzerinde doğruluyor mu?"
"""

import argparse
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import pipeline

from src.utils.io import load_jsonl, save_jsonl


# Raw cevapları üreten modelden farklı bir model kullanıyoruz.

# Farklı model kullanmak iki sistemin tamamen aynı hataları tekrarlama ihtimalini bir miktar azaltır.
VERIFIER_MODEL_NAME = "deepset/deberta-v3-base-squad2"


DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/raw_baseline_calibration.jsonl"
)

OUTPUT_PATH = Path(
    "outputs/predictions/evidence_verified_calibration.jsonl"
)


def normalize_answer(text: str) -> str:
    """
    İki cevabı karşılaştırmadan önce normalize eder.

    Örnek:
        "The James Watt!" → "james watt"

    İşlemler:
    - küçük harfe çevirme
    - punctuation kaldırma
    - a, an, the article'larını kaldırma
    - gereksiz boşlukları temizleme
    """

    text = text.lower()

    text = ''.join(
        character
        for character in text
        if character not in string.punctuation #string.punctuation, Python’daki yaygın noktalama işaretlerini içeren bir metindir.
    )

    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text,
    )

    return " ".join(text.split())


# overlap, iki cevapta ortak bulunan tokenların toplam sayısıdır.
def answer_overlap_f1(
    first_answer: str,
    second_answer: str
) -> float:
    """
    İki answer arasındaki token örtüşmesini ölçer.

    Neden Exact Match kullanmıyoruz?

    Çünkü iki model aynı doğru cevabı farklı uzunlukta verebilir.

    Örnek:
        İlk model:     "James Watt"
        Verifier:      "Watt"

    Exact Match:
        0

    Token overlap F1:
        0'dan büyük

    Returns:
        0.0 ile 1.0 arasında benzerlik değeri.
    """

    first_tokens = normalize_answer(
        first_answer
    ).split()

    second_tokens = normalize_answer(
        second_answer
    ).split()


    # Cevaplardan biri boşsa örtüşme yoktur.
    if not first_answer or not second_answer:
        return 0.0

    # İki cevapta ortak bulunan kelimeleri sayıyoruz.
    common_tokens = (
        Counter(first_answer)
        & Counter(second_answer)
    )

    overlap = sum(common_tokens.values())

    if overlap == 0:
        return 0.0
    
    precision = overlap / len(first_tokens)
    recall = overlap / len(second_tokens)

    return (
        2* precision * recall
        / (precision + recall)
    )



def select_device(
    device_name: str
) -> torch.device:
    """
    Verifier modelinin çalışacağı cihazı seçer.

    cpu:
        Normal işlemci.

    mps:
        Apple Silicon GPU.

    cuda:
        NVIDIA GPU.
    """

    if device_name == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested but is not available."
            )
        
        return torch.device('mps')
    

    if device_name == 'cude':
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )
        
        return torch.device('cude')
    
    return torch.device('cpu')



def classify_evidence(
    generated_answer: str,
    verifier_answer: str,
    verifier_score: float,
    support_threshold: float,
    match_threshold: float,
    contradiction_threshold: float
) -> tuple[str, str, float]:
    """
    Verifier sonucunu evidence label'a dönüştürür.

    Returns:
        evidence_label
        evidence_reason
        answer_match_f1

    Label mantığı:

    1. Verifier güçlü şekilde aynı cevabı bulursa:
       SUPPORTED

    2. Verifier yüksek skorla farklı cevap bulursa
       veya yüksek skorla cevap olmadığını söylerse:
       UNSUPPORTED

    3. Sonuç yeterince net değilse:
       UNCERTAIN
    """

    answer_match = answer_overlap_f1(
        generated_answer,
        verifier_answer
    )


    # Verifier boş cevap döndürdüyse: 
    # context içinde güvenilir bir cevap bulamamış olabilir.
    if not verifier_answer.strip(): # strip(), metnin başındaki ve sonundaki boşlukları siler.
        if verifier_score >= contradiction_threshold: # contradiction_threshold, modelin “bu cevap context tarafından desteklenmiyor” demesi için gereken sınır değerdir.
            return (
                'UNSUPPORTED',
                "Verifier predicted that the question "
                "is unanswerable from the context.",
                answer_match,
            )
        
        return (
            "UNCERTAIN",
            "Verifier returned no answer but its score "
            "was not high enough for rejection.",
            answer_match,
        )
    

    # İkinci model de benzer cevabı bulduysa
    # ilk cevabı supported kabul ediyoruz.
    if (
        answer_match >= match_threshold
        and verifier_score >= support_threshold
    ):
        return (
            "SUPPORTED",
            "Independent verifier produced a matching answer.",
            answer_match,
        )

    # Verifier güçlü ama tamamen farklı bir cevap verdiyse
    # ilk cevabın desteklenmediğini düşünüyoruz.
    if (
        answer_match < 0.20
        and verifier_score >= contradiction_threshold
    ):
        return (
            "UNSUPPORTED",
            "Independent verifier confidently produced "
            "a different answer.",
            answer_match,
        )
    

    # Diğer bütün belirsiz durumlar.
    return (
        "UNCERTAIN",
        "Verifier evidence was not decisive.",
        answer_match,
    )


def verity_predictions(
    predictions: list[dict[str, Any]],
    device_name: str = 'cpu',
    support_threshold: float = 0.30,
    match_threshold: float = 0.80,
    contradiction_threshold: float = 0.50
) -> list[dict[str, Any]]:
    """
    Bu fonksiyon, modelin ürettiği bütün cevapları verilen kaynak metne göre kontrol eder.

    Kullanılan eşik değerleri şimdilik geçicidir.

    Daha sonra bu değerler, ayrı bir doğrulama veri seti üzerinde test edilerek en uygun şekilde belirlenecektir.
    """


    device = select_device(
        device_name
    )

    print(
        f"Loading verifier model: "
        f"{VERIFIER_MODEL_NAME}"
    )

    print(
        f"Using device: {device}"
    )

    # İkinci, bağımsız QA modelini yüklüyoruz.
    verifier = pipeline(
        task='question-answering',
        model=VERIFIER_MODEL_NAME,
        tokenizer=VERIFIER_MODEL_NAME,
        device=device
    )

    verified_predictions: list[dict[str, Any]] = []

    for index, prediction in enumerate(
        predictions,
        start=1
    ):
        # Evidence checker'ın çalışması için
        # question ve context zorunludur.
        question = prediction['question']
        context = prediction['context'] # context, sorunun cevabını bulmak için modele verilen kaynak metindir

        generated_answer = prediction[
            'prediction_text' # modelin ürettiği cevap metni
        ]

        # Verifier modeline aynı question ve context'i veriyoruz.
        
        # handle_impossible_answer=True:
        # Verifier gerektiğinde boş cevap döndürebilir.
        verifier_result = verifier(
            question=question,
            context=context,
            top_k=1,
            handle_impossible_anser=True
        )

        verifier_answer = str(
            verifier_result['answer']
        )

        verifier_score = float(
            verifier_result['score']
        )

        # Verifier çıktısını üç evidence label'dan
        # birine dönüştürüyoruz.
        (
            evidence_label,
            evidence_reason,
            answer_match
        ) = classify_evidence(
            generated_answer=generated_answer,
            verifier_answer=verifier_answer,
            verifier_score=verifier_score,
            support_threshold=support_threshold,
            match_threshold=match_threshold,
            contradiction_threshold=(
                contradiction_threshold
            )
        )

        # Raw prediction'ı kaybetmemek için kopyalıyoruz.
        verified_prediction = prediction.copy()

        verified_prediction.update(
            {
                "verifier_answer": verifier_answer,
                'verifier_score': verifier_score,
                'answer_match_f1': answer_match,
                'evidence_label': evidence_label,
                'evidence_reason': evidence_reason,
                'verifier_model': VERIFIER_MODEL_NAME
            }
        )

        verified_predictions.append(
            verified_prediction
        )

        print(
            f"\nExample {index}/{len(predictions)}"
        )

        print(
            f"Question: {question}"
        )

        print(
            f"Generated answer: {generated_answer}"
        )

        print(
            f"Verifier answer: "
            f"{verifier_answer or '[NO ANSWER]'}"
        )

        print(
            f"Verifier score: {verifier_score:.4f}"
        )

        print(
            f"Answer match F1: {answer_match:.4f}"
        )

        print(
            f"Evidence label: {evidence_label}"
        )

    return verified_predictions


def summarize_evidence(
    predictions: list[dict[str, Any]]
) -> dict[str, int | float]:
    """
    Evidence label dağılımını hızlıca özetler.
    """

    total = len(predictions)

    if total == 0:
        raise ValueError(
            "Prediction list cannot be empty."
        )
    
    supported = sum(
        prediction['evidence_label'] == 'SUPPORTED'
        for prediction in predictions
    )

    unsupported = sum(
        prediction['evidence_label'] == 'UNSUPPORTED'
        for prediction in predictions
    )

    uncertain = total - supported - unsupported


    return {
        "total": total,
        "supported": supported,
        "unsupported": unsupported,
        "uncertain": uncertain,
        "supported_rate": supported / total,
        "unsupported_rate": unsupported / total,
        "uncertain_rate": uncertain / total,
    }


def run_evidence_checker(
    input_path: str | Path,
    output_path: str | Path,
    device_name: str
) -> list[dict[str, Any]]:
    """
    Evidence checker'ın ana çalışma fonksiyonu.
    """

    # Raw baseline prediction'larını yükle.
    predictions = load_jsonl(
        input_path
    )

    
    # Context alanı eklenmeden oluşturulan eski
    # prediction dosyalarını engelliyoruz.
    missing_context = any(
        'context' not in prediction
        for prediction in predictions
    )

    if missing_context:
        raise ValueError(
            "Some predictions do not contain 'context'. "
            "Run raw_answer_baseline.py again after adding "
            "the context field."
        )
    
    verified_predictions = verity_predictions(
        predictions=predictions,
        device_name=device_name
    )

    save_jsonl(
    verified_predictions,
    output_path
    )

    summary = summarize_evidence(
        verified_predictions,
    )

    print("\nEvidence verification completed.")

    print(
        f"SUPPORTED: {summary['supported']}"
    )

    print(
        f"UNSUPPORTED: {summary['unsupported']}"
    )

    print(
        f"UNCERTAIN: {summary['uncertain']}"
    )

    print(
        f"Results saved to: {output_path}"
    )

    return verified_predictions



def parse_arguments() -> argparse.Namespace:
    """
    Terminal argümanlarını okur.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Verify generated answers using an "
            "independent QA model."
        )
    )

    parser.add_argument(
        '--input',
        default=str(DEFAULT_INPUT_PATH)
    )

    parser.add_argument(
        '--output',
        default=str(OUTPUT_PATH)
    )

    parser.add_argument(
        '--device',
        choices=[
            'cpu',
            'mps',
            'cuda'
        ],
        default='cpu'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_evidence_checker(
        input_path=args.input,
        output_path=args.output,
        device_name=args.device,
    )