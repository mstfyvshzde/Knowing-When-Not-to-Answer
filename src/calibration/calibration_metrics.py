"""
Calibration metrics for question-answering confidence scores.

Bu dosyanın görevi:
1. Model prediction'ının doğru olup olmadığını belirlemek.
2. Raw confidence ile gerçek correctness'i karşılaştırmak.
3. ECE, MCE, Brier score ve NLL hesaplamak.
4. Her confidence aralığının doğruluğunu kaydetmek.

Bu dosya confidence'ı henüz değiştirmez.
Yalnızca confidence'ın ne kadar güvenilir olduğunu ölçer.
"""


import argparse
import math
import re
import string
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.io import load_jsonl, save_json


def normalize_answer(text: str) -> str:
    """
    Prediction ve reference cevapları adil karşılaştırmak için
    normalize eder.

    Örnek:
        "The James Watt!" -> "james watt"
    """

    # Küçük harfe çevir.
    text = text.lower()


    # Noktalama işaretlerini kaldır.
    text = ''.join(
        character
        for character in text
        if character not in string.punctuation
    )


    # İngilizce article'ları kaldır.
    text = re.sub(
        r"\b(a|an|the)\b",
        ' ',
        text
    )

    # Birden fazla boşluğu tek boşluğa indir.
    return ' '.join(text.split())




def is_prediction_correct(
    prediction: dict[str, Any]
) -> int:
    """
    Prediction'ın doğru olup olmadığını 0 veya 1 olarak döndürür.

    Returns:
        1 -> doğru
        0 -> yanlış

    Raw baseline her zaman ANSWER verdiği için:

    - Unanswerable soruya verilen cevap otomatik olarak yanlıştır.
    - Answerable soruda prediction reference cevaplardan
      biriyle eşleşiyorsa doğrudur.
    """

    is_answerable = bool(
        prediction['is_answerable']
    )

    # Context içinde cevap yoksa raw modelin verdiği
    # herhangi bir cevap yanlış kabul edilir.
    if not is_answerable:
        return 0.0
    

    predicted_answer = normalize_answer(
        prediction.get(
            'prediction_text',
            ''
        )
    ) 

    
    reference_answer = prediction.get(
        'reference_answers',
        []
    )


    normalized_references = [
        normalize_answer(reference)
        for reference in reference_answer
    ]

    return int(
        predicted_answer in normalized_references
    )



def validate_confidences(
    confidences: np.ndarray
) -> None:
    """
    Confidence değerlerinin 0 ile 1 arasında olduğunu kontrol eder.
    """

    if confidences.size == 0:
        raise ValueError(
            "Confidence array cannot be empty."
        )
    
    if np.any(confidences < 0.0):
        raise ValueError(
            "Confidence values cannot be below zero."
        )
    
    if np.any(confidences > 1.0):
        raise ValueError(
            "Confidence values cannot exceed one."
        )



def calculate_brier_score(
    confidences: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Brier score confidence ile gerçek label arasındaki
    karesel hatayı ölçer.

    Mantık:

        Brier = Ortalama((confidence - correctness)^2)

    Örnek:

        confidence = 0.90
        correctness = 1
        hata küçük

        confidence = 0.90
        correctness = 0
        hata büyük

    Düşük değer daha iyidir.
    """

    return float(
        np.mean(
            (confidences - labels) ** 2
        )
    )



def calculate_negative_log_likelihood(
    confidences: np.ndarray,
    labels: np.ndarray,
    epsilon: float = 1e-12
) -> float:
    """
    Binary Negative Log-Likelihood hesaplar.

    NLL özellikle yanlış fakat yüksek confidence verilen
    tahminleri güçlü biçimde cezalandırır.

    Confidence tam 0 veya 1 olduğunda log(0) oluşmaması için
    değerleri epsilon ile sınırlandırıyoruz.

    Düşük değer daha iyidir.
    """

    clipped_confidences = np.clip(
        confidences, epsilon,
        1.0 - epsilon
    )

    # losses, her tahmin için ayrı ayrı hesaplanan hata değerlerini tutar.
    losses = -(
        labels
        * np.log(clipped_confidences)
        + (1 - labels)
        * np.log(
            1.0 - clipped_confidences
        )
    )

    return float(
        np.mean(losses)
    )


# Kalibrasyon, modelin güven skorlarının gerçek doğrulukla ne kadar uyuştuğunu gösterir. Fark küçükse model iyi kalibre edilmiştir.
# Bu fonksiyon, modelin güven skorlarını belirli aralıklara, yani binlere ayırır. Her bin içinde bulunan tahminlerin ortalama güven skoru ve gerçek doğruluk oranı ayrı ayrı hesaplanır. Daha sonra bu iki değer arasındaki fark bulunarak modelin ne kadar iyi kalibre edildiği ölçülür.
# Örneğin 10 bin kullanıldığında güven skorları 0.0–0.1, 0.1–0.2, …, 0.9–1.0 şeklinde ayrılır. Modelin 0.7–0.8 bininde 20 tahmin varsa, bu 20 tahminin güven skorlarının ortalaması hesaplanır. Ortalama güven %75 ve gerçek doğruluk %65 ise kalibrasyon farkı %10 olur. Bu da modelin bu binde kendine fazla güvendiğini gösterir.

def calculate_calibration_bins(
    confidences: np.ndarray,
    labels:np.ndarray,
    number_of_bins: int = 10
) -> list[dict[str, Any]]:
    """
    Confidence değerlerini aralıklara ayırır.

    Örnek olarak 10 bin kullanılırsa:

        Bin 1: 0.0–0.1
        Bin 2: 0.1–0.2
        ...
        Bin 10: 0.9–1.0

    Her bin için ölçülenler:

    - kaç prediction var?
    - ortalama confidence ne?
    - gerçek accuracy ne?
    - aralarındaki fark ne?
    """

    if number_of_bins <= 0:
        raise ValueError(
            "number_of_bins must be greater than zero."
        )
    
    validate_confidences(confidences)

    
    # 0 ile 1 arasında eşit aralıklı sınırlar oluştur.
    bin_edges = np.linspace(
        0.0, 
        1.0,
        number_of_bins + 1
    )

    calibration_bins: list[dict[str, Any]] = []

    for bin_index in range(number_of_bins):
        lower_bound = float(
            bin_edges[bin_index]
        )

        upper_bound = float(
            bin_edges[bin_index + 1]
        )

        # Son bin 1.0 confidence değerini de içermelidir.
        if bin_index == number_of_bins - 1:
            in_bin = (
                (confidences >= lower_bound)
                & (confidences <= upper_bound)
            )
        
        else:
            in_bin = (
                (confidences >= lower_bound)
                & (confidences < upper_bound)
            )

        count = int(
            np.sum(in_bin)
        )

        # Bu aralıkta prediction yoksa boş bin kaydediyoruz.
        if count == 0:
            calibration_bins.append(
                {
                    'bin_index': bin_index,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'count': 0,
                    'mean_confidence': None,
                    'accuracy': None,
                    'calibration_gap': None
                }
            )

            continue

        mean_confidence = float(
            np.mean(confidences[in_bin])
        )

        accuracy = float(
            np.mean(labels[in_bin])
        )

        calibration_gap = abs(
            mean_confidence - accuracy
        )

        calibration_bins.append(
            {
                "bin_index": bin_index,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "calibration_gap": calibration_gap,
            }
        )

    return calibration_bins




# Bu fonksiyon, tüm binlerdeki kalibrasyon farklarını örnek sayılarına göre ağırlıklandırarak tek bir ECE değeri hesaplar. ECE ne kadar düşükse model o kadar iyi kalibre edilmiştir.
# Her binin gap değeri, o bindeki tahminlerin toplam tahminlere oranıyla çarpılır ve hepsi toplanır.
# Böylece çok örnek bulunan binler ECE sonucunu daha fazla etkiler. Sonuç küçükse modelin güveni ile doğruluğu birbirine yakındır.
def calculate_ece(
    calibration_bins: list[dict[str, Any]],
    total_examples: int
) -> float:
    """
    Expected Calibration Error hesaplar.

    Her confidence binindeki calibration farkı,
    o bindeki örnek sayısıyla ağırlıklandırılır.

    Düşük ECE daha iyi calibration anlamına gelir.
    """

    if total_examples <= 0:
        raise ValueError(
            "total_examples must be greater than zero."
        )

    ece = 0.0

    for calibration_bin in calibration_bins:
        count = calibration_bin['count']
        gap = calibration_bin['calibration_gap'] # gap, bir bindeki ortalama güven skoru ile gerçek doğruluk arasındaki farktır.

        if count == 0 or gap is None:
            continue

        bin_weight = count / total_examples

        ece += bin_weight * gap

    return float(ece)



# Bu fonksiyon, tüm binler arasındaki en büyük kalibrasyon farkını bulur. Yani modelin en kötü kalibre olduğu güven aralığını gösterir.
# MCE, modelin hangi binde en fazla hata yaptığını görmek için kullanılır. Değer ne kadar küçükse o kadar iyidir.
def calculate_mce(
    calibration_bins: list[dict[str, Any]]
) -> float:
    """
    Maximum Calibration Error hesaplar.

    Confidence binleri arasındaki en büyük
    confidence-accuracy farkını döndürür.
    """

    gaps = [
        calibration_bin['calibration_gap']
        for calibration_bin in calibration_bins
        if calibration_bin['calibration_gap'] is not None
    ]

    if not gaps:
        return 0.0

    return float(max(gaps))



def calculate_calibration_metrcis(
    predictions: list[dict[str, Any]],
    number_of_bins: int = 10
) -> dict[str, Any]:
    """
    Prediction listesinden bütün calibration metriklerini hesaplar.
    """

    if not predictions:
        raise ValueError(
            "Prediction list cannot be empty."
        )

    # Raw confidence değerlerini NumPy array'e dönüştür.
    confidences = np.asarray(
        [
            float(prediction['confidence'])
            for prediction in predictions
        ],
        dtype=np.float64
    )


    # Her prediction için gerçek correctness label'ı oluştur.
    #
    # 1 -> doğru
    # 0 -> yanlış
    labels = np.asarray(
        [
            is_prediction_correct(prediction)
            for prediction in predictions
        ],
        dtype=np.float64
    )

    validate_confidences(confidences)

    calibration_bins = calculate_calibration_bins(
        confidences=confidences,
        labels=labels,
        number_of_bins=number_of_bins
    )

    ece = calculate_ece(
        calibration_bins=calibration_bins,
        total_examples=len(predictions)
    )

    mce = calculate_mce(
        calibration_bins
    )

    brier_score = calculate_brier_score(
        confidences=confidences,
        labels=labels
    )

    negative_log_likelihood = (
        calculate_negative_log_likelihood(
            confidences=confidences,
            labels=labels
        )
    )

    accuracy = float(
        np.mean(labels)
    )


    mean_confidence = float(
        np.mean(confidences)
    )

    return {
        'system': predictions[0].get(
            'system',
            'unknown'
        ),

        'total_examples': len(predictions),

        'correct_examples': int(
            np.sum(labels)
        ),

        'accuracy': accuracy,

        'mean_confidence': mean_confidence,
    

        # Confidence ile doğruluk arasındaki genel fark.
        'confidence_accuracy_gap': abs(
            mean_confidence - accuracy
        ),

        'expected_calibration_error': ece,

        'maximum_calibration_error': mce,

        'brier_score': brier_score,

        'negative_log_likelihood': (
            negative_log_likelihood
        ),

        'number_of_bins': number_of_bins,

        'calibration_bins': calibration_bins

    }



# Examples: Değerlendirilen toplam örnek sayısıdır. Sonuçların kaç veri üzerinden hesaplandığını görmek için vardır.
# Accuracy: Doğru tahmin oranıdır. Modelin genel başarısını ölçmek için kullanılır.
# Mean confidence: Modelin ortalama güven seviyesidir. Modelin kendine ne kadar güvendiğini görmek için vardır.
# ECE: Güven ile doğruluk arasındaki genel farkı ölçer. Modelin iyi kalibre edilip edilmediğini anlamak için kullanılır.
# MCE: En büyük kalibrasyon hatasını gösterir. Modelin en kötü olduğu güven aralığını bulmak için vardır.
# Brier score: Tahmin olasılıkları ile gerçek sonuç arasındaki farkı ölçer. Olasılık tahminlerinin kalitesini değerlendirmek için kullanılır.
# NLL: Gerçek sınıfa düşük olasılık veren tahminleri cezalandırır. Özellikle yanlış ve aşırı güvenli tahminleri görmek için kullanılır.
def run_calibration_analysis(
    input_path: str | Path,
    output_path: str | Path,
    number_of_bins: int,
) -> dict[str, Any]:
    """
    Prediction dosyasını yükler, calibration metriklerini
    hesaplar ve JSON olarak kaydeder.
    """

    predictions = load_jsonl(
        input_path
    )

    metrics = calculate_calibration_metrcis(
        predictions=predictions,
        number_of_bins=number_of_bins,
    )

    save_json(
        metrics,
        output_path,
    )

    print("\nCalibration analysis completed.")

    print(
        f"Examples: "
        f"{metrics['total_examples']}"
    )

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Mean confidence: "
        f"{metrics['mean_confidence']:.4f}"
    )

    print(
        f"ECE: "
        f"{metrics['expected_calibration_error']:.4f}"
    )

    print(
        f"MCE: "
        f"{metrics['maximum_calibration_error']:.4f}"
    )

    print(
        f"Brier score: "
        f"{metrics['brier_score']:.4f}"
    )

    print(
        f"NLL: "
        f"{metrics['negative_log_likelihood']:.4f}"
    )

    print(
        f"Saved to: {output_path}"
    )

    return metrics


def parse_arguments() -> argparse.Namespace:
    """
    Terminal argümanlarını okur.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Measure calibration of raw QA confidence scores."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Raw prediction JSONL file.",
    )

    parser.add_argument(
        "--output",
        default=(
            "outputs/tables/"
            "raw_confidence_calibration.json"
        ),
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=10,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_calibration_analysis(
        input_path=args.input,
        output_path=args.output,
        number_of_bins=args.bins,
    )