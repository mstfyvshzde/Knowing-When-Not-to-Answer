"""
Temperature scaling for QA confidence calibration.

Bu dosyanın görevi:
1. Confidence estimator tarafından üretilen margin değerlerini okumak.
2. Her cevabın doğru veya yanlış olduğunu belirlemek.
3. Calibration verisi üzerinde tek bir temperature değeri öğrenmek.
4. Margin değerlerini temperature ile ölçeklemek.
5. Calibrated confidence değerlerini ayrı dosyaya kaydetmek.

Temel formül:

    calibrated_logit = margin / temperature
    calibrated_confidence = sigmoid(calibrated_logit)

Önemli:
- Temperature yalnızca calibration split üzerinde öğrenilir.
- Test split temperature öğrenmek için kullanılmaz.
- Temperature scaling tahmin sıralamasını değiştirmez.
- Yalnızca confidence değerlerinin keskinliğini değiştirir.
"""

import argparse
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.calibration.calibration_metrics import is_prediction_correct
from src.utils.io import load_jsonl, save_json, save_jsonl

DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/raw_baseline_with_confidence_calibration.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "outputs/predictions/raw_baseline_calibrated_calibration.jsonl"
)

DEFAULT_PARAMETERS_PATH = Path("outputs/tables/temperature_scaling_parameters.json")


def stable_sigmoid(value: float) -> float:
    """
    Bir logit değerini güvenli biçimde 0–1 aralığına çevirir.

    Pozitif büyük sayı:
        confidence 1'e yaklaşır.

    Negatif büyük sayı:
        confidence 0'a yaklaşır.
    """

    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))

    exp_value = math.exp(value)

    return exp_value / (1.0 + exp_value)


def prepare_calibration_data(
    predictions: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prediction kayıtlarını PyTorch tensorlarına dönüştürür.

    Returns:
        margins:
            Answer-vs-null logit margin değerleri.

        labels:
            1 -> prediction doğru
            0 -> prediction yanlış
    """

    if not predictions:
        raise ValueError("Prediction list cannot be empty.")

    missing_margin = [
        prediction.get("id", "unknown")
        for prediction in predictions
        if "answer_null_margin" not in prediction
    ]

    print(
        "First prediction has margin:",
        "answer_null_margin" in predictions[0],
    )

    print(f"Missing answer_null_margin: {len(missing_margin)}")

    if len(missing_margin) > 0:
        raise ValueError(
            "Some predictions do not contain "
            "'answer_null_margin'. "
            f"Missing IDs: {missing_margin[:5]}"
        )

    # Confidence estimator tarafından oluşturulan
    # answer-vs-null margin değerlerini alıyoruz.
    margins = torch.tensor(
        [float(prediction["answer_null_margin"]) for prediction in predictions],
        dtype=torch.float64,
    )

    # Her tahmin doğruysa 1.0, yanlışsa 0.0 üretir. Sonra bu değerleri bir PyTorch tensorunda toplar.
    labels = torch.tensor(
        [float(is_prediction_correct(prediction)) for prediction in predictions],
        dtype=torch.float64,
    )

    # torch.unique(labels), etiketlerde kaç farklı değer olduğunu bulur. Sadece 0 ya da sadece 1 varsa hata verir; hem doğru hem yanlış örnek varsa margins ve labels değerlerini döndürür.
    unique_labels = torch.unique(labels)

    if (
        unique_labels.numel() < 2
    ):  # unique_labels içinde 2’den az farklı değer varsa, yani sadece 0 veya sadece 1 bulunuyorsa koşul çalışır ve hata verir.
        raise ValueError(
            "Temperature fitting requires both correct and incorrect predictions."
        )

    return margins, labels


# NLL (Negative Log-Likelihood), modelin tahminlerinin gerçek etiketlere ne kadar uyduğunu ölçen hata değeridir. Düşük NLL daha iyi, yüksek NLL ise tahminlerin daha kötü veya aşırı güvensiz olduğunu gösterir.
def calculate_nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Binary Negative Log-Likelihood hesaplar.

    Burada BCEWithLogitsLoss kullanılır.

    Bu fonksiyon:
        sigmoid + binary cross entropy

    işlemlerini numerik olarak güvenli şekilde birleştirir.

    Düşük NLL daha iyidir.
    """

    # nn.BCEWithLogitsLoss(), ikili sınıflandırmada kullanılan kayıp fonksiyonudur. Logit değerlerine sigmoid uygular ve tahminlerle gerçek 0/1 etiketler arasındaki hatayı hesaplar.
    criterion = nn.BCEWithLogitsLoss()

    loss = criterion(logits, labels)

    return float(loss.item())


def fit_temperature(
    margins: torch.Tensor, labels: torch.Tensor
) -> tuple[float, float, float]:
    """
    Calibration verisi üzerinde temperature değerini öğrenir.

    Returns:
        temperature
        nll_before
        nll_after

    Mantık:

        T = 1
            Orijinal margin değişmez.

        T > 1
            Confidence değerleri yumuşar.
            0 ve 1 uçlarından uzaklaşır.

        T < 1
            Confidence değerleri daha keskin olur.
    """

    # Bu satır, başlangıç değeri 0 olan ve eğitim sırasında öğrenilecek bir log-temperature parametresi oluşturur. nn.Parameter sayesinde optimizer bu değeri günceller; gerçek temperature değeri genelde torch.exp(log_temperature) ile hesaplanır
    log_tempperature = nn.Parameter(torch.zeros(1, dtype=torch.float64))

    criterion = nn.BCEWithLogitsLoss()

    # Bu kod, log_tempperature değerini en iyi hale getirmek için LBFGS optimizer oluşturur:
    # Amaç, kayıp değerini düşüren en iyi temperature değerini bulmaktı
    optimizer = torch.optim.LBFGS(
        [log_tempperature],  # Güncellenecek parametre.
        lr=0.1,  # Her güncellemenin adım büyüklüğü.
        max_iter=100,  # En fazla 100 optimizasyon adımı yapar.
        line_search_fn="strong_wolfe",  # En uygun ve güvenli adım büyüklüğünü arar.
    )

    # Calibration öncesi NLL.
    nll_before = calculate_nll(logits=margins, labels=labels)

    # Genel olarak bu fonksiyon, en uygun temperature değerini bulmak için kaybı hesaplıyor. Optimizer da bu kayba bakarak temperature değerini güncelliyor.
    def closure() -> torch.Tensor:
        """
        Optimizer'ın her adımda temperature değerini
        değerlendirmek için çağırdığı fonksiyon.
        """

        # Bu satır, önceki adımdan kalan gradientleri temizler. Böylece yeni hesaplama sıfırdan yapılır.
        optimizer.zero_grad()

        # Temperature, modelin güven skorlarını yumuşatan veya keskinleştiren değerdir.
        # Log temperature ise temperature’ın logaritmasıdır; exp(log_temperature) yapınca gerçek ve her zaman pozitif temperature elde edilir.
        # Keskin: Model olasılıkları uçlara yaklaştırır; örneğin 0.70 -> 0.90. Model daha emin görünür.
        # Yumuşak: Olasılıkları 0.5’e yaklaştırır; örneğin 0.90 -> 0.70. Model daha az emin görünür.
        temperature = torch.exp(log_tempperature)

        # Bu satır, margins değerlerini temperature’a bölerek modelin güvenini ayarlar.
        # temperature > 1 ise değerler küçülür, tahminler yumuşar.
        # temperature < 1 ise değerler büyür, tahminler keskinleşir.
        sclaed_logits = margins / temperature

        # Ölçeklenmiş tahminlerle gerçek labels değerlerini karşılaştırıp hata (loss) hesaplar. Hata ne kadar düşükse tahminler o kadar iyidir.
        loss = criterion(sclaed_logits, labels)

        # loss.backward(), hatanın log_temperature parametresine göre gradientini hesaplar. Optimizer da bu bilgiyi kullanarak temperature değerini günceller.
        loss.backward()

        return loss

    # closure, kaybı yeniden hesaplayan fonksiyondur. optimizer.step(closure) ise bu fonksiyonu çağırıp gradientlere bakarak log_temperature değerini günceller.
    optimizer.step(closure)

    learned_temperature = float(
        torch.exp(
            log_tempperature.detach()  # detach(), tensörü hesaplama grafiğinden ayırır. Yani bundan sonra bu değer için gradient hesaplanmaz; sadece sonucu okumak için kullanılır.
        ).item()
    )

    if not math.isfinite(learned_temperature) or learned_temperature <= 0.0:
        raise ValueError(
            f"Temperature optimization produced an invalid value: {learned_temperature}"
        )

    scaled_logits = margins / learned_temperature

    # Calibration sonrası NLL.
    nll_after = calculate_nll(logits=scaled_logits, labels=labels)

    return (learned_temperature, nll_before, nll_after)


def apply_temperature(
    predictions: list[dict[str, Any]], temperature: float
) -> list[dict[str, Any]]:
    """
    Öğrenilen temperature değerini prediction'lara uygular.
    """
    if temperature <= 0.0:
        raise ValueError("Temperature must be positive.")

    calibrated_predictions: list[dict[str, Any]] = []

    for prediction in predictions:
        margin = float(prediction["answer_null_margin"])

        calibrated_logit = margin / temperature

        calibrated_confidence = stable_sigmoid(calibrated_logit)

        updated_prediction = prediction.copy()

        # Eski confidence değerini kaybetmiyoruz.
        updated_prediction["uncalibrated_confidence"] = float(prediction["confidence"])

        updated_prediction.update(
            {
                "calibrated_logit": (calibrated_logit),
                # Bundan sonraki karar sistemleri
                # bu confidence alanını kullanacak.
                "confidence": (calibrated_confidence),
                "calibrated_confidence": (calibrated_confidence),
                "temperature": temperature,
                "confidence_type": ("temperature_scaled_answer_vs_null"),
                "confidence_is_calibrated": True,
            }
        )

        calibrated_predictions.append(updated_prediction)

    return calibrated_predictions


def run_temperature_scaling(
    input_path: str | Path, output_path: str | Path, parameters_path: str | Path
) -> list[dict[str, Any]]:
    """
    Temperature fitting ve uygulama sürecini çalıştırır.
    """
    predictions = load_jsonl(input_path)

    margins, labels = prepare_calibration_data(predictions)

    temperature, nll_before, nll_after = fit_temperature(margins=margins, labels=labels)

    calibrated_predictions = apply_temperature(
        predictions=predictions, temperature=temperature
    )

    save_jsonl(calibrated_predictions, output_path)

    parameter_data = {
        "method": "temperature_scaling",  # Kullanılan kalibrasyon yöntemi.
        "temperature": temperature,  # Öğrenilen temperature değeri.
        "fit_examples": len(
            predictions
        ),  # Kalibrasyonda kullanılan toplam örnek sayısı.
        "correct_examples": int(labels.sum().item()),  # Doğru tahmin sayısı.
        "incorrect_examples": int(
            len(labels) - labels.sum().item()
        ),  # Yanlış tahmin sayısı.
        "nll_before": nll_before,  # Kalibrasyondan önceki hata değeri.
        "nll_after": nll_after,  # Kalibrasyondan sonraki hata değeri.
        "fit_split": predictions[0].get(
            "split",
            "unknown",
        ),  # Verinin hangi bölümden geldiği, örneğin train, validation.
        "input_signal": (
            "answer_null_margin"
        ),  # Kalibrasyonda kullanılan skorun adı: answer_null_margin.
        "test_set_used_for_fitting": False,  # Test verisinin temperature öğrenmek için kullanılıp kullanılmadığını belirtir. Genelde False olmalıdır.
    }

    save_json(
        parameter_data,
        parameters_path,
    )

    print("\nTemperature scaling completed.")

    print(f"Examples used: {len(predictions)}")

    print(f"Temperature: {temperature:.6f}")

    print(f"NLL before: {nll_before:.6f}")

    print(f"NLL after: {nll_after:.6f}")

    print(f"Predictions saved to: {output_path}")

    print(f"Parameters saved to: {parameters_path}")

    return calibrated_predictions


def parse_arguments() -> argparse.Namespace:
    """Terminal argümanlarını okur."""

    parser = argparse.ArgumentParser(
        description=("Calibrate QA confidence using temperature scaling.")
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
        "--parameters",
        default=str(DEFAULT_PARAMETERS_PATH),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_temperature_scaling(
        input_path=args.input,
        output_path=args.output,
        parameters_path=args.parameters,
    )
