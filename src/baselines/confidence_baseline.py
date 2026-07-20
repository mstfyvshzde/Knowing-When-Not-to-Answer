"""
Confidence-based abstention baseline.

Bu dosyanın görevi:
1. Raw baseline prediction'larını okumak.
2. Her prediction'ın confidence score'una bakmak.
3. Confidence threshold üstündeyse ANSWER kararı vermek.
4. Threshold altındaysa ABSTAIN etmek.
5. Yeni kararları ayrı bir JSONL dosyasına kaydetmek.

Bu sistem evidence kullanmaz.
Yalnızca model confidence score'una güvenir.
"""


import argparse
from pathlib import Path
from typing import Any

from src.utils.io import load_jsonl, save_jsonl


# Raw baseline tarafından oluşturulan prediction dosyası.
DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/raw_baseline_calibration.jsonl"
)

# Confidence baseline sonuçlarının kaydedileceği klasör.
OUTPUT_DIR = Path("outputs/predictions")


def apply_confidence_threshold(
    predictions: list[dict[str, Any]],
    threshold: float
) -> list[dict[str, Any]]:
    """
    Confidence threshold kullanarak ANSWER veya ABSTAIN kararı verir.

    Args:
        predictions:
            Raw baseline prediction kayıtları.

        threshold:
            Cevap verebilmek için gereken minimum confidence.

            Örnek:
                confidence = 0.82
                threshold = 0.60
                -> ANSWER

                confidence = 0.35
                threshold = 0.60
                -> ABSTAIN

    Returns:
        Yeni kararlar eklenmiş prediction listesi.
    """


    # Confidence değeri 0 ile 1 arasında olmalıdır.
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )
    
    updated_predictions: list[dict[str, Any]] = []

    for prediction in predictions:
        # Raw modelin ürettiği confidence değerini alıyoruz.
        confidence = float(
            prediction['confidence']
        )

        # -----------------------------------------------
        # CORE AI/ML KARAR MANTIĞI
        # -----------------------------------------------
        
        # Confidence yeterince yüksekse cevap ver.
        # Değilse cevap vermekten kaçın.
        if confidence >= threshold:
            decision = 'ANSWER'
            final_answer = prediction['prediction_text']

        else:
            decision = 'ABSTAIN'
            final_answer = 'I do not know'


        # Orijinal prediction'ı değiştirmemek için kopyalıyoruz.
        updated_prediction = prediction.copy()
        

        # Yeni sistemin kararlarını kaydediyoruz.
        updated_prediction.update(
            {
                'final_answer': final_answer,
                'decision': decision,
                'threshold': threshold,
                'system': 'confidence_baseline'
            }
        )

        updated_predictions.append(
            updated_prediction
        )

    return updated_predictions



def summarize_decisions(
    predictions: list[dict[str, Any]]
) -> dict[str, float | int]:
    """
    Sistemin kaç soruya cevap verdiğini özetler.

    Bu henüz tam evaluation değildir.
    Yalnızca hızlı kontrol içindir.
    """

    total = len(predictions)

    if total == 0:
        raise ValueError(
            "Prediction list cannot be empty."
        )
    
    answered = sum(
        prediction['decision'] == 'ANSWER'
        for prediction in predictions
    )

    abstained = total - answered

    return {
        'total_examples': total,
        'answered_examples': answered,
        'abstained-examples': abstained,

        # Coverage:
        # Sistemin cevap verdiği örneklerin oranı.
        'coverage': answered / total,


        # Abstention rate:
        # Sistemin cevap vermediği örneklerin oranı.
        'abstention_rate': abstained / total
    }



def run_confidence_baseline(
    input_path: str | Path,
    threshold: float
) -> list[dict[str, Any]]:
    """
    Confidence baseline'ın ana çalışma fonksiyonu.
    """
    input_path = Path(input_path)

    # Raw prediction dosyasını oku.
    raw_predictions = load_jsonl(
        input_path
    )

    # Confidence threshold kararını uygula.
    predictions = apply_confidence_threshold(
        predictions=raw_predictions,
        threshold=threshold
    )

    # Hızlı karar özeti oluştur.
    summary = summarize_decisions(
        predictions
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Threshold'u dosya adına ekliyoruz.
    # Örnek: 0.60 -> 0_60
    threshold_name = str(threshold).replace(
        '.',
        '-'
    )

    output_path = (
        OUTPUT_DIR
        / f"confidence_baseline_{threshold_name}.jsonl"
    )

    save_jsonl(
        predictions,
        output_path,
    )

    print("\nConfidence baseline completed.")
    print(f"Threshold: {threshold:.2f}")
    print(
        f"Answered: "
        f"{summary['answered_examples']}/"
        f"{summary['total_examples']}"
    )
    print(
        f"Coverage: "
        f"{summary['coverage']:.4f}"
    )
    print(
        f"Abstention rate: "
        f"{summary['abstention_rate']:.4f}"
    )
    print(f"Saved to: {output_path}")

    return predictions


def parse_arguments() -> argparse.Namespace:
    """
    Terminal seçeneklerini okur.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Apply confidence-based abstention "
            "to raw QA predictions."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_confidence_baseline(
        input_path=args.input,
        threshold=args.threshold,
    )