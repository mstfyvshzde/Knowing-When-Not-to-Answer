"""
Select ANSWER, VERIFY and ABSTAIN thresholds.

Bu dosyanın görevi:
1. Calibrated confidence değerlerini okumak.
2. Her prediction'ın doğru veya yanlış olduğunu belirlemek.
3. Olası threshold çiftlerini calibration verisinde denemek.
4. Güvenlik kısıtlarını karşılayan en iyi threshold'ları seçmek.
5. Threshold sonuçlarını JSON olarak kaydetmek.

Karar sistemi:

    confidence <= abstain_threshold
        -> ABSTAIN

    abstain_threshold < confidence < answer_threshold
        -> VERIFY

    confidence >= answer_threshold
        -> ANSWER

Önemli:
- Threshold'lar yalnızca calibration split üzerinde seçilir.
- Test split threshold seçmek için kullanılmaz.
- Test split yalnızca final evaluation için kullanılır.
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


def validate_predictions(
    predictions: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prediction kayıtlarını kontrol eder.

    Returns:
        confidences:
            Calibrated confidence değerleri.

        labels:
            1 -> prediction doğru
            0 -> prediction yanlış
    """

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

    # Threshold selection calibrated confidence üzerinde yapılmalıdır.
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

    # observed_splits, predictions içindeki hangi veri bölümlerinin kullanıldığını toplar.
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

    labels = np.asarray(
        [is_prediction_correct(prediction) for prediction in predictions],
        dtype=np.int64,
    )

    if len(np.unique(labels)) < 2:
        raise ValueError(
            "Threshold selection requires both correct and incorrect predictions."
        )

    return confidences, labels


def assign_decision(
    confidence: float, abstain_threshold: float, answer_threshold: float
) -> str:
    """
    Tek bir confidence değerini karara dönüştürür.
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
    labels: np.ndarray,
    abstain_threshold: float,
    answer_threshold: float,
) -> dict[str, Any]:
    """
    Tek bir threshold çiftinin davranışını ölçer.

    Burada önemli ölçümler:

    answer_risk:
        Modelin doğrudan cevap verdiği örneklerdeki
        yanlış cevap oranı.

    abstain_correct_rate:
        Modelin sustuğu örneklerin ne kadarının aslında
        doğru tahmin olduğu.

    verify_rate:
        Ek doğrulama isteyen örnek oranı.
    """

    if abstain_threshold >= answer_threshold:
        raise ValueError("Invalid threshold ordering.")

    total_examples = len(confidences)

    answer_mask = confidences >= answer_threshold

    abstain_mask = confidences <= abstain_threshold

    verify_mask = ~answer_mask & ~abstain_mask

    answer_count = int(np.sum(answer_mask))

    abstain_count = int(np.sum(abstain_mask))

    verify_count = int(np.sum(verify_mask))

    answer_correct = int(np.sum(labels[answer_mask] == 1))

    answer_incorrect = int(np.sum(labels[answer_mask] == 0))

    abstain_correct = int(np.sum(labels[abstain_mask] == 1))

    abstain_incorrect = int(np.sum(labels[abstain_mask] == 0))

    # Direct answers içindeki doğruluk.
    if answer_count > 0:
        answer_accuracy = answer_correct / answer_count

        answer_risk = answer_incorrect / answer_count

    else:
        answer_accuracy = None
        answer_risk = None

    # ABSTAIN edilen örneklerin ne kadarı aslında doğru cevaptı?
    # Bunun düşük olmasını isteriz.
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


def create_threshold_candidates(
    confidences: np.ndarray,
    grid_size: int,  # grid_size, kaç tane threshold adayı üretileceğini belirler. Bu threshold’lar, modelin tahminini kabul edip etmeyeceğine karar vermek için kullanılır.
) -> np.ndarray:
    """
    Confidence dağılımından threshold adayları üretir.

    Bütün unique confidence değerlerini kullanmak,
    büyük datasetlerde çok pahalı olabilir.

    Bu nedenle confidence quantile'larından
    sınırlı sayıda aday üretiriz.
    """

    if grid_size < 3:
        raise ValueError("grid_size must be at least 3.")

    # np.linspace(0.0, 1.0, grid_size), 0 ile 1 arasında eşit aralıklı quantile noktaları üretir.
    quantiles = np.linspace(0.0, 1.0, grid_size)

    # np.quantile(confidences, quantiles) ise confidence değerlerinin bu yüzdeliklere karşılık gelen skorlarını bulur.
    candidates = np.quantile(confidences, quantiles)

    # Tekrarlanan threshold değerlerini kaldır.
    candidates = np.unique(candidates)

    # candidates listesinin başına 0.0, sonuna 1.0 ekler ve tekrar eden değerleri siler.
    candidates = np.unique(
        # np.concatenate, birden fazla NumPy dizisini tek bir dizi halinde birleştirir.
        np.concatenate([np.asarray([0.0]), candidates, np.asarray([1.0])])
    )

    return candidates


def select_thresholds(
    confidences: np.ndarray,
    labels: np.ndarray,
    max_answer_risk: float,
    max_abstain_corrrect_rate: float,
    min_answer_rate: float,
    min_abstain_rate: float,
    grid_size: int,
) -> dict[str, Any]:
    """
    Calibration verisi üzerinde en iyi threshold çiftini seçer.

    Güvenlik kısıtları:

    1. ANSWER bölgesindeki yanlış oranı,
       max_answer_risk değerini geçmemeli.

    2. ABSTAIN bölgesindeki doğru tahmin oranı,
       max_abstain_correct_rate değerini geçmemeli.

    Seçim öncelikleri:

    1. Daha fazla güvenli ANSWER üret.
    2. Daha az örneği VERIFY'a gönder.
    3. ANSWER riskini düşür.
    4. Gereksiz ABSTAIN oranını düşür.
    """

    if (
        not 0.0 <= max_answer_risk <= 1.0
    ):  # max_answer_risk, modelin kabul edilen cevaplarında izin verilen en yüksek hata oranıdır.
        raise ValueError("max_answer_risk must be between 0 and 1.")

    if not (
        0.0
        <= max_abstain_corrrect_rate  # max_abstain_correct_rate, modelin cevap vermediği örnekler içinde aslında doğru tahmin ettiği örneklerin izin verilen en yüksek oranıdır.
        <= 1.0
    ):
        raise ValueError("max_abstain_correct_rate must be between 0 and 1.")

    if (
        not 0.0 <= min_answer_rate <= 1.0
    ):  # min_answer_rate, modelin tüm örneklerin en az ne kadarına doğrudan ANSWER vermesi gerektiğini belirtir.
        raise ValueError("min_answer_rate must be between 0 and 1.")

    total_examples = len(confidences)

    # Bu kod, modelin en az kaç örneğe ANSWER vermesi gerektiğini hesaplar.
    minimum_answer_count = max(
        1,
        math.ceil(
            total_examples
            * min_answer_rate  # math.ceil(...), çıkan sayıyı bir üst tam sayıya yuvarlar.
        ),
    )

    minimum_abstain_count = max(1, math.ceil(total_examples * min_abstain_rate))

    candidates = create_threshold_candidates(
        confidences=confidences, grid_size=grid_size
    )

    # Uygun sonuçları saklamak için oluşturulmuş boş bir listedir. İçine sözlükler eklenir.
    feasible_results: list[dict[str, Any]] = []

    # fallback_results, uygun sonuç bulunamazsa kullanılacak yedek sonuçları saklayan boş listedir.
    fallback_results: list[dict[str, Any]] = []

    # abstain_threshold: Güven skoru bunun altındaysa model cevap vermez (ABSTAIN).
    # answer_threshold: Güven skoru bunun üstündeyse model doğrudan cevap verir (ANSWER).
    # İkisinin arasında kalırsa VERIFY olur.

    # Amaç, farklı abstain_threshold ve answer_threshold çiftlerini deneyerek model için en güvenli ve en verimli eşikleri bulmaktır. Sonunda şartları en iyi karşılayan threshold çifti seçilir.
    # Önce abstain_threshold yazılması sadece döngü sırasıdır; teknik olarak zorunlu değildir. Önce düşük sınırı belirleyip sonra ona uygun daha yüksek answer_threshold değerlerini denemek, kodu daha anlaşılır yapar.
    for abstain_threshold in candidates:
        for answer_threshold in candidates:
            if abstain_threshold >= answer_threshold:
                continue

            # Bu satır, seçilen abstain_threshold ve answer_threshold çiftini veri üzerinde test eder. Sonuç olarak kaç tane ANSWER, VERIFY, ABSTAIN çıktığını ve risk oranlarını hesaplayıp result içine koyar.
            result = evaluate_threshold_pair(
                confidences=confidences,
                labels=labels,
                abstain_threshold=float(abstain_threshold),
                answer_threshold=float(answer_threshold),
            )

            if result["answer_count"] < minimum_answer_count:
                continue

            if result["abstain_count"] < minimum_abstain_count:
                continue

            # answer_risk: Modelin ANSWER verdiği örneklerdeki yanlış cevap oranı.
            answer_risk = result["answer_risk"]

            # abstain_correct_rate: Modelin ABSTAIN dediği örneklerde aslında doğru tahmin etmiş olma oranı.
            abstain_correct_rate = result["abstain_correct_rate"]

            if answer_risk is None or abstain_correct_rate is None:
                continue

            # Bu, answer_risk değerinin izin verilen sınırı ne kadar aştığını hesaplar.
            answer_risk_violation = max(0.0, answer_risk - max_answer_risk)

            # Bu, abstain_correct_rate değerinin izin verilen maksimum oranı ne kadar aştığını hesaplar.
            abstain_violation = max(
                0.0, abstain_correct_rate - max_abstain_corrrect_rate
            )

            # "answer_risk_violation" -> ANSWER risk sınırı ne kadar aşılmış
            result["answer_risk_violation"] = answer_risk_violation

            # "abstain_constraint_violation" -> ABSTAIN sınırı ne kadar aşılmış
            result["abstain_constraint_violation"] = abstain_violation

            # threshold çiftinin kuralları toplamda ne kadar aştığını gösterir. Değer ne kadar küçükse sonuç o kadar iyidir
            result["total_constraint_violation"] = (
                answer_risk_violation + abstain_violation
            )

            # her threshold çiftinin sonucu, gerektiğinde yedek seçenek olarak kullanılmak üzere saklanır.
            fallback_results.append(result)

            # Bu kod, iki güvenlik şartının da sağlanıp sağlanmadığını kontrol eder:
            constraints_satisfied = (
                answer_risk <= max_answer_risk
                and abstain_correct_rate <= max_abstain_corrrect_rate
            )

            # Eğer tüm güvenlik şartları sağlandıysa, o threshold çiftinin sonucunu feasible_results listesine ekler. Yani uygun ve güvenli sonuçları saklar
            if constraints_satisfied:
                feasible_results.append(result)

    # feasible_results boş değilse, içindeki en iyi sonucu seçer.
    # Öncelik sırası:
    # En yüksek answer_coverage
    # En yüksek direct_decision_rate
    # En düşük answer_risk
    # En düşük abstain_correct_rate
    # Sonra güvenlik koşullarının sağlandığını belirtmek için:
    # constraints_satisfied = True yapar
    if feasible_results:
        best_result = max(
            feasible_results,
            key=lambda result: (
                result[
                    "answer_coverage"
                ],  # Modelin kaç soruya cevap verdiği. Yüksek olması, daha az “cevap veremiyorum” demesi demektir.
                result[
                    "direct_decision_rate"
                ],  # Modelin kaç durumda doğrudan karar verdiği.
                -result[
                    "answer_risk"
                ],  # Verilen cevapların hatalı veya riskli olma oranı. Başındaki - nedeniyle düşük olması tercih ediliyor.
                -result[
                    "abstain_correct_rate"  # Model cevap vermediğinde, bu kaçınmanın ne kadar doğru olduğu. Örneğin gerçekten belirsiz bir soruda cevap vermemesi.
                ],
            ),
        )

        constraints_satisfied = True

    else:
        # Pilot veri küçük olduğunda bazen hiçbir threshold tüm kısıtları karşılamayabilir.
        # Bu durumda programı tamamen durdurmak yerine,
        # en az constraint ihlali yapan sonucu kaydederiz.
        if not fallback_results:
            raise ValueError(
                "No threshold pair produced enough ANSWER and ABSTAIN examples."
            )

        # Bu kod, fallback_results içinden kuralları en az ihlal eden sonucu seçmek içindir.
        # Öncelik sırası:
        # En düşük total_constraint_violation
        # En yüksek answer_coverage
        # En yüksek direct_decision_rate
        best_result = min(
            fallback_results,
            key=lambda result: (
                result["total_constraint_violation"],
                -result["answer_coverage"],
                -result["direct_decision_rate"],
            ),
        )

        constraints_satisfied = False

    # Bu sözlük, eşik araması sonucunda seçilen en iyi ayarın özetini saklıyor.
    selection_result = {
        # Kullanılan seçim yöntemi
        "method": (
            "constrained_three_way_"  # Model üç karar verebilir: cevapla, kaçın/abstain, veya aradaki başka bir karar sınıfı.
            "threshold_search"  # Farklı eşik çiftleri denenir; yalnızca belirlenen risk, cevap oranı ve örnek sayısı koşullarını sağlayanlar kabul edilir.
        ),
        # Modelin karar eşikleri calibration verisi üzerinde ayarlanmış. Test verisi bu ayarlama sırasında kullanılmamış.
        "fit_split": "calibration",
        # # test seti tarafsız değerlendirme için saklanmış.
        "test_set_used_for_selection": False,
        # constraints_satisfied, seçilen sonucun belirlenen tüm koşulları sağlayıp sağlamadığını gösterir.
        "constraints_satisfied": (constraints_satisfied),
        # Bu değerler, seçimin uyması gereken sınırları temsil eder:
        "constraints": {
            # Cevap verirken izin verilen en yüksek hata/risk oranı
            "max_answer_risk": (max_answer_risk),
            # Kaçınma ile ilgili izin verilen en yüksek oran
            "max_abstain_correct_rate": (max_abstain_corrrect_rate),
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


# Bu fonksiyon, tahmin listesindeki (predictions) her bir model çıktısının güven skorunu (confidence) inceleyerek belirlenen iki eşik değerine (abstain_threshold ve answer_threshold) göre bir karar (ANSWER, VERIFY veya ABSTAIN) atar. Daha sonra bu kararı ve kullanılan eşik bilgilerini tahmin sözlüklerine ekleyerek güncellenmiş yeni bir liste döndürür.
def annotate_predictions(
    predictions: list[dict[str, Any]], abstain_threshold: float, answer_threshold: float
) -> list[dict[str, Any]]:
    """
    Seçilen threshold'larla calibration kayıtlarına
    ANSWER, VERIFY veya ABSTAIN kararı ekler.
    """

    annotated_predictions: list[dict[str, Any]] = []

    for prediction in predictions:
        confidence = float(prediction["confidence"])

        decision = assign_decision(
            confidence=confidence,
            abstain_threshold=(abstain_threshold),
            answer_threshold=(answer_threshold),
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


# Bu fonksiyon, model tahminlerinin güven skorlarını analiz ederek güvenli bir karar verme mekanizması kurar. Belirlenen başarı/risk kısıtlarına göre iki farklı eşik değeri (vazgeçme ve yanıtlama) hesaplar, tahminleri bu eşiklere göre etiketleyip sonuçları dosyalara kaydeder.
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
    Threshold selection sürecini çalıştırır.
    """

    predictions = load_jsonl(input_path)

    confidences, labels = validate_predictions(predictions)

    selection_result = select_thresholds(
        confidences=confidences,
        labels=labels,
        max_answer_risk=max_answer_risk,
        max_abstain_corrrect_rate=(max_abstain_correct_rate),
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
