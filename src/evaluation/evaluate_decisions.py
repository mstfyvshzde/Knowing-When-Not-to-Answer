"""
Nihai seçici soru-cevap karar sistemi için değerlendirme araçları.

Bu modül şunları değerlendirir:

1. Tam kapsamlı soru-cevap doğruluğu.
2. Nihai ANSWER / VERIFY / ABSTAIN kararları.
3. Cevap kapsamı ve seçici risk.
4. Yalnızca güven skoruna dayalı eşik politikası.
5. Kanıt farkındalıklı nihai politika.
6. Karar nedeni ve kanıt dağılımları.
"""

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.io import load_jsonl

DEFAULT_INPUT_PATH = Path("outputs/predictions/calibration_final_decisions.jsonl")

DEFAULT_OUTPUT_PATH = Path("outputs/tables/final_decision_metrics.json")


VALID_DECISIONS = {
    "ANSWER",
    "VERIFY",
    "ABSTAIN",
}


def normalize_boolean(
    value: Any,
) -> bool:
    """
    Farklı doğruluk gösterimlerini Python boolean
    değerine dönüştürür.
    """

    # Değer zaten boolean ise doğrudan döndürülür.
    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        if value == 1:
            return True

        if value == 0:
            return False

    if isinstance(value, float):
        if value == 1.0:
            return True

        if value == 0.0:
            return False

    # Metinsel değerler boşluk ve büyük/küçük harf farkından arındırılır.
    if isinstance(value, str):
        normalized_value = value.strip().lower()

        true_values = {
            "true",
            "1",
            "correct",
            "yes",
        }

        false_values = {
            "false",
            "0",
            "incorrect",
            "no",
        }

        if normalized_value in true_values:
            return True

        if normalized_value in false_values:
            return False

    raise ValueError(f"Could not convert correctness value to boolean: {value!r}")


def normalize_answer(
    text: Any,
) -> str:
    """
    SQuAD tarzı cevap normalizasyonu uygular.

    İşlemler:

    1. Küçük harfe dönüştürür.
    2. Noktalama işaretlerini kaldırır.
    3. İngilizce artikelleri kaldırır.
    4. Fazladan boşlukları kaldırır.
    """

    normalized_text = str(text or "").lower()

    # Noktalama karakterleri tek tek filtrelenir.
    normalized_text = "".join(
        character
        for character in normalized_text
        if character not in string.punctuation
    )

    # İngilizce a/an/the artikelleri kelime sınırlarıyla kaldırılır.
    normalized_text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        normalized_text,
    )

    normalized_text = " ".join(normalized_text.split())

    return normalized_text


def extract_reference_answers(
    prediction: dict[str, Any],
) -> list[str]:
    """
    reference_answers değerini standart bir cevap
    metinleri listesine dönüştürür.

    Desteklenen biçimler:

    ["answer one", "answer two"]

    {
        "text": ["answer one", "answer two"],
        "answer_start": [10, 25]
    }

    "single answer"
    """

    reference_answers = prediction.get(
        "reference_answers",
        [],
    )

    if reference_answers is None:
        return []

    if isinstance(
        reference_answers,
        str,
    ):
        return [reference_answers]

    # Sözlük biçiminde farklı veri sürümlerinde kullanılan alan adları aranır.
    if isinstance(
        reference_answers,
        dict,
    ):
        possible_fields = (
            "text",
            "answers",
            "answer_text",
        )

        for field in possible_fields:
            values = reference_answers.get(field)

            if values is None:
                continue

            if isinstance(
                values,
                str,
            ):
                return [values]

            if isinstance(
                values,
                list,
            ):
                extracted_values: list[str] = []

                # Liste elemanı sözlükse cevap metni olası anahtarlardan alınır.
                for value in values:
                    if isinstance(value, dict):
                        answer_text = (
                            value.get("text")
                            or value.get("answer")
                            or value.get("answer_text")
                        )

                        if answer_text is not None:
                            extracted_values.append(str(answer_text))

                    else:
                        extracted_values.append(str(value))

                return extracted_values

        return []

    if isinstance(
        reference_answers,
        list,
    ):
        extracted_answers: list[str] = []

        for item in reference_answers:
            if isinstance(
                item,
                str,
            ):
                extracted_answers.append(item)

            elif isinstance(
                item,
                dict,
            ):
                answer_text = (
                    item.get("text") or item.get("answer") or item.get("answer_text")
                )

                if answer_text is not None:
                    extracted_answers.append(str(answer_text))

            else:
                extracted_answers.append(str(item))

        return extracted_answers

    return [str(reference_answers)]


def get_correctness(
    prediction: dict[str, Any],
) -> bool:
    """
    Bir soru-cevap tahmininin doğru olup olmadığını belirler.

    Önce mevcut bir doğruluk alanı aranır.

    Doğruluk alanı yoksa prediction_text ve
    reference_answers kullanılarak normalize edilmiş
    Exact Match hesaplanır.

    Cevaplanamaz sorularda boş tahmin doğru kabul edilir.
    """

    # Önceden hesaplanmış doğruluk bilgisi farklı alan adlarıyla gelebilir.
    possible_fields = (
        "is_correct",
        "correct",
        "prediction_correct",
        "exact_match",
        "em",
    )

    for field in possible_fields:
        if field not in prediction:
            continue

        value = prediction[field]

        # Exact Match bazı dosyalarda 1/100 veya metinsel değer olarak saklanabilir.
        if field in {
            "exact_match",
            "em",
        }:
            try:
                numeric_value = float(value)

            except (
                TypeError,
                ValueError,
            ):
                return normalize_boolean(value)

            return numeric_value in {
                1.0,
                100.0,
            }

        return normalize_boolean(value)

    # Hazır doğruluk alanı yoksa normalize edilmiş Exact Match hesaplanır.
    prediction_text = normalize_answer(
        prediction.get(
            "prediction_text",
            "",
        )
    )

    reference_answers = extract_reference_answers(prediction)

    normalized_references = [
        normalize_answer(reference_answer) for reference_answer in reference_answers
    ]

    normalized_references = [
        reference_answer
        for reference_answer in normalized_references
        if reference_answer
    ]

    is_answerable_value = prediction.get(
        "is_answerable",
        True,
    )

    try:
        is_answerable = normalize_boolean(is_answerable_value)

    except ValueError:
        is_answerable = bool(is_answerable_value)

    # Cevaplanamaz soruda modelin boş cevap vermesi doğru kabul edilir.
    if not is_answerable:
        return prediction_text == ""

    if not normalized_references:
        raise ValueError(
            "Answerable prediction does not contain "
            "a usable reference answer. "
            f"Prediction id: {prediction.get('id')!r}"
        )

    return prediction_text in normalized_references


def get_final_decision(
    prediction: dict[str, Any],
) -> str:
    """
    Nihai kararı okur ve doğrular.
    """

    value = prediction.get("final_decision")

    if value is None:
        raise ValueError("Prediction does not contain final_decision.")

    # Karşılaştırmanın tutarlı olması için karar metni standartlaştırılır.
    decision = str(value).strip().upper()

    if decision not in VALID_DECISIONS:
        raise ValueError(f"Invalid final_decision value: {decision!r}.")

    return decision


def get_threshold_decision(
    prediction: dict[str, Any],
) -> str | None:
    """
    Yalnızca güven skoruna dayalı eşik kararını okur.

    Eşik kararı mevcut değilse None döndürür.
    """

    # Alan adı dosya sürümüne göre değişebildiği için olası adlar sırayla denenir.
    possible_fields = (
        "threshold_decision",
        "confidence_decision",
        "selective_decision",
    )

    for field in possible_fields:
        value = prediction.get(field)

        if value is None:
            continue

        decision = str(value).strip().upper()

        if decision in VALID_DECISIONS:
            return decision

    return None


def safe_divide(
    numerator: float,
    denominator: float,
) -> float | None:
    """
    ZeroDivisionError oluşturmadan değerleri güvenli
    biçimde böler.
    """

    # Sıfıra bölme hatası yerine hesaplanamayan metrik için None kullanılır.
    if denominator == 0:
        return None

    return float(numerator / denominator)


def evaluate_decision_group(
    predictions: list[dict[str, Any]],
    decision: str,
) -> dict[str, Any]:
    """
    Tek bir karar grubunu değerlendirir.

    Hesaplanan değerler:

    count
    rate
    correct_count
    incorrect_count
    accuracy
    risk
    """

    # Yalnızca istenen nihai karara sahip tahminler seçilir.
    selected_predictions = [
        prediction
        for prediction in predictions
        if get_final_decision(prediction) == decision
    ]

    count = len(selected_predictions)

    total = len(predictions)

    correct_count = sum(
        int(get_correctness(prediction)) for prediction in selected_predictions
    )

    incorrect_count = count - correct_count

    accuracy = safe_divide(
        correct_count,
        count,
    )

    # Seçici risk, cevaplanan örneklerdeki hata oranıdır: 1 - doğruluk.
    risk = None if accuracy is None else 1.0 - accuracy

    return {
        "count": count,
        "rate": safe_divide(
            count,
            total,
        ),
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "accuracy": accuracy,
        "risk": risk,
    }


def evaluate_threshold_policy(
    predictions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Yalnızca güven skoruna dayalı eşik politikasını değerlendirir.

    Bu, kanıt doğrulaması uygulanmadan önceki sistemi temsil eder.
    """

    threshold_predictions: list[dict[str, Any]] = []

    for prediction in predictions:
        threshold_decision = get_threshold_decision(prediction)

        # Tek bir kayıtta bile eşik kararı yoksa bu politika değerlendirilemez.
        if threshold_decision is None:
            return None

        threshold_predictions.append(
            {
                "decision": threshold_decision,
                "is_correct": get_correctness(prediction),
            }
        )

    total = len(threshold_predictions)

    # Her karar türünün kaç kez üretildiği sayılır.
    counts = Counter(item["decision"] for item in threshold_predictions)

    answer_predictions = [
        item for item in threshold_predictions if item["decision"] == "ANSWER"
    ]

    answer_correct_count = sum(int(item["is_correct"]) for item in answer_predictions)

    answer_accuracy = safe_divide(
        answer_correct_count,
        len(answer_predictions),
    )

    selective_risk = None if answer_accuracy is None else 1.0 - answer_accuracy

    return {
        "total": total,
        "answer_count": counts["ANSWER"],
        "verify_count": counts["VERIFY"],
        "abstain_count": counts["ABSTAIN"],
        "answer_coverage": safe_divide(
            counts["ANSWER"],
            total,
        ),
        "verify_rate": safe_divide(
            counts["VERIFY"],
            total,
        ),
        "abstain_rate": safe_divide(
            counts["ABSTAIN"],
            total,
        ),
        "answer_accuracy": answer_accuracy,
        "selective_risk": selective_risk,
    }


def evaluate_reason_distribution(
    predictions: list[dict[str, Any]],
) -> dict[str, int]:
    """
    Karar nedenlerini sayar.
    """

    reason_counts = Counter(
        str(
            prediction.get(
                "decision_reason",
                "unknown",
            )
        )
        for prediction in predictions
    )

    return dict(reason_counts)


def evaluate_evidence_distribution(
    predictions: list[dict[str, Any]],
) -> dict[str, int]:
    """
    Kanıt destek etiketlerini sayar.
    """

    evidence_counts = Counter(
        str(
            prediction.get(
                "evidence_support",
                "UNKNOWN",
            )
        )
        .strip()
        .upper()
        for prediction in predictions
    )

    return dict(evidence_counts)


def calculate_final_metrics(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Tüm nihai seçici soru-cevap metriklerini hesaplar.
    """

    if not predictions:
        raise ValueError("Prediction list cannot be empty.")

    total = len(predictions)

    total_correct = sum(int(get_correctness(prediction)) for prediction in predictions)

    # Tüm sorular cevaplanmış kabul edilerek temel doğruluk hesaplanır.
    baseline_accuracy = safe_divide(
        total_correct,
        total,
    )

    # Her nihai karar grubu ayrı ayrı değerlendirilir.
    answer_metrics = evaluate_decision_group(
        predictions=predictions,
        decision="ANSWER",
    )

    verify_metrics = evaluate_decision_group(
        predictions=predictions,
        decision="VERIFY",
    )

    abstain_metrics = evaluate_decision_group(
        predictions=predictions,
        decision="ABSTAIN",
    )

    threshold_policy = evaluate_threshold_policy(predictions)

    metrics: dict[str, Any] = {
        "total_predictions": total,
        "baseline_full_coverage_accuracy": (baseline_accuracy),
        "final_policy": {
            "answer": answer_metrics,
            "verify": verify_metrics,
            "abstain": abstain_metrics,
            "answer_coverage": (answer_metrics["rate"]),
            "selective_risk": (answer_metrics["risk"]),
            "verify_rate": (verify_metrics["rate"]),
            "abstain_rate": (abstain_metrics["rate"]),
        },
        "threshold_only_policy": (threshold_policy),
        "evidence_distribution": (evaluate_evidence_distribution(predictions)),
        "decision_reason_distribution": (evaluate_reason_distribution(predictions)),
    }

    # Kanıt doğrulaması öncesi ve sonrası risk/kapsam farkları hesaplanır.
    if threshold_policy is not None:
        final_risk = answer_metrics["risk"]

        threshold_risk = threshold_policy["selective_risk"]

        final_coverage = answer_metrics["rate"]

        threshold_coverage = threshold_policy["answer_coverage"]

        metrics["policy_comparison"] = {
            "risk_change": (
                None
                if (final_risk is None or threshold_risk is None)
                else (final_risk - threshold_risk)
            ),
            "coverage_change": (
                None
                if (final_coverage is None or threshold_coverage is None)
                else (final_coverage - threshold_coverage)
            ),
        }

    return metrics


def save_metrics(
    metrics: dict[str, Any],
    output_path: str | Path,
) -> None:
    """
    Değerlendirme metriklerini JSON dosyasına kaydeder.
    """

    output_path = Path(output_path)

    # Hedef klasör yoksa üst klasörleriyle birlikte oluşturulur.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
            ensure_ascii=False,
        )


def format_optional_metric(
    value: float | None,
) -> str:
    """
    İsteğe bağlı sayısal metrikleri biçimlendirir.
    """

    if value is None:
        return "N/A"

    return f"{value:.4f}"


def print_metrics(
    metrics: dict[str, Any],
) -> None:
    """
    En önemli değerlendirme metriklerini ekrana yazdırır.
    """

    final_policy = metrics["final_policy"]

    answer = final_policy["answer"]

    verify = final_policy["verify"]

    abstain = final_policy["abstain"]

    print("\nSelective QA evaluation completed.")

    print(f"Total predictions: {metrics['total_predictions']}")

    print(
        "Full-coverage baseline accuracy: "
        f"{format_optional_metric(metrics['baseline_full_coverage_accuracy'])}"
    )

    print("\nFinal evidence-aware policy:")

    print(f"ANSWER count: {answer['count']}")

    print(f"Answer coverage: {format_optional_metric(final_policy['answer_coverage'])}")

    print(f"Answer accuracy: {format_optional_metric(answer['accuracy'])}")

    print(f"Selective risk: {format_optional_metric(final_policy['selective_risk'])}")

    print(f"VERIFY count: {verify['count']}")

    print(f"Verify rate: {format_optional_metric(final_policy['verify_rate'])}")

    print(f"VERIFY accuracy: {format_optional_metric(verify['accuracy'])}")

    print(f"ABSTAIN count: {abstain['count']}")

    print(f"Abstain rate: {format_optional_metric(final_policy['abstain_rate'])}")

    print(f"ABSTAIN correct rate: {format_optional_metric(abstain['accuracy'])}")

    threshold_policy = metrics.get("threshold_only_policy")

    if threshold_policy is not None:
        print("\nConfidence-only threshold policy:")

        print(
            "Threshold answer coverage: "
            f"{format_optional_metric(threshold_policy['answer_coverage'])}"
        )

        print(
            "Threshold answer accuracy: "
            f"{format_optional_metric(threshold_policy['answer_accuracy'])}"
        )

        print(
            "Threshold selective risk: "
            f"{format_optional_metric(threshold_policy['selective_risk'])}"
        )

    comparison = metrics.get("policy_comparison")

    if comparison is not None:
        print("\nEvidence-aware policy change:")

        print(f"Risk change: {format_optional_metric(comparison['risk_change'])}")

        print(
            f"Coverage change: {format_optional_metric(comparison['coverage_change'])}"
        )


def validate_predictions(
    predictions: list[dict[str, Any]],
) -> None:
    """
    Gerekli değerlendirme alanlarını doğrular.
    """

    if not predictions:
        raise ValueError("Prediction list cannot be empty.")

    for index, prediction in enumerate(
        predictions,
        start=1,
    ):
        try:
            get_final_decision(prediction)

            get_correctness(prediction)

        # Hatanın hangi tahmin kaydında oluştuğu ek bağlamla yeniden bildirilir.
        except ValueError as error:
            raise ValueError(
                f"Prediction {index} failed validation: {error}"
            ) from error


def run_evaluation(
    input_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """
    Tüm değerlendirme işlem hattını çalıştırır.
    """

    # Veri yükleme, doğrulama, hesaplama, kaydetme ve yazdırma sırasıyla yürütülür.
    predictions = load_jsonl(input_path)

    validate_predictions(predictions)

    metrics = calculate_final_metrics(predictions)

    save_metrics(
        metrics=metrics,
        output_path=output_path,
    )

    print_metrics(metrics)

    print(f"\nMetrics saved to: {output_path}")

    return metrics


def parse_arguments() -> argparse.Namespace:
    """
    Komut satırı argümanlarını ayrıştırır.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate final ANSWER, VERIFY and ABSTAIN decisions for selective QA."
        )
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help=("JSONL file containing final selective QA decisions."),
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=("JSON output path for evaluation metrics."),
    )

    return parser.parse_args()


# Dosya doğrudan çalıştırıldığında komut satırı akışı başlatılır.
if __name__ == "__main__":
    args = parse_arguments()

    run_evaluation(
        input_path=args.input,
        output_path=args.output,
    )
