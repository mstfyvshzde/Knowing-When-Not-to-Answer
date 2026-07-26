"""
Confidence estimation from question-answering model logits.

Bu dosyanın görevi:
1. Raw baseline prediction'larını okumak.
2. Question + context'i doğrudan QA modeline vermek.
3. Raw cevabın start ve end logit skorlarını bulmak.
4. Modelin "cevap yok" skorunu hesaplamak.
5. Answer-vs-null margin üretmek.
6. Bu margin'i 0–1 aralığında uncalibrated confidence'a çevirmek.

Önemli:
Bu dosyanın ürettiği confidence henüz calibrated değildir.

Yani:
    confidence = 0.80

doğrudan:
    cevap %80 ihtimalle doğrudur

anlamına gelmez.

Daha sonra calibration split üzerinde temperature scaling
veya başka bir calibration yöntemi uygulayacağız.
"""

import argparse
import math
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from src.utils.io import load_jsonl, save_jsonl

# Raw baseline'da kullandığımız aynı QA modeli.
MODEL_NAME = "deepset/roberta-base-squad2"

# Raw baseline prediction dosyası.
DEFAULT_INPUT_PATH = Path("outputs/predictions/raw_baseline_calibration.jsonl")

# Confidence eklenmiş prediction dosyası.
DEFAULT_OUTPUT_PATH = Path(
    "outputs/predictions/raw_baseline_with_confidence_calibration.jsonl"
)


# Modelin tek seferde işleyebileceği toplam maksimum token sayısıdır.
# Bu sayı; soru, context ve özel tokenları birlikte kapsar.
MAX_LENGTH = 384

# Uzun bir context parçalara ayrıldığında, ardışık parçalar arasında
# kaç tokenın tekrar kullanılacağını belirtir.
# Böylece cevap iki parçanın sınırında kalırsa bilgi kaybolmaz.
DOCUMENT_STRIDE = 128

# Context çok uzunsa tokenizer onu parçalara ayırır. İlk parça 384 tokena kadar işlenir. Sonraki parça, önceki parçanın son **128 tokenını tekrar içerir**. Böylece parçalar arasında bağlantı korunur ve sınırdaki cevabın kaçırılma ihtimali azalır.


def select_device(device_name: str) -> torch.device:
    """
    Modelin çalışacağı cihazı seçer.

    cpu:
        Normal işlemci.

    mps:
        Apple Silicon GPU.

    cuda:
        NVIDIA GPU.
    """

    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")

        return torch.device("mps")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

        return torch.device("cuda")

    return torch.device("cpu")


def sigmoid(value: float) -> float:
    """
    Herhangi bir sayıyı 0–1 aralığına dönüştürür.

    Büyük pozitif margin:
        confidence 1'e yaklaşır.

    Büyük negatif margin:
        confidence 0'a yaklaşır.

    Margin 0:
        confidence 0.5 olur.
    """

    # Büyük negatif değerlerde math.exp taşmasını önler.
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))

    exp_value = math.exp(value)

    return exp_value / (1.0 + exp_value)


def find_cls_index(input_ids: torch.Tensor, cls_token_id: int) -> int:
    """
    Input içindeki CLS token pozisyonunu bulur.

    SQuAD 2.0 modellerinde CLS token genellikle
    "no answer" seçeneğini temsil eder.
    """

    # .nonzero() koşulun True olduğu indeksleri döndürür.
    # Burada input_ids == cls_token_id sonucu True/False üretir; .nonzero() ise CLS tokenının bulunduğu konumları verir.
    positions = (input_ids == cls_token_id).nonzero(as_tuple=False)

    if len(positions) == 0:
        raise ValueError("CLS token was not found in model input.")

    return int(positions[0].item())


# Karakter konumunu token konumuna çevirir.
# text = "Ali Bakü'de yaşıyor"
# answer = "Bakü"
# answer_start = 4
# answer_end = 8
# "Ali"  -> token 0
# "Bakü" -> token 1
# Burada "Bakü" cevabı karakter olarak 4–8 aralığındadır.
def find_answer_token_span(
    offsets: list[list[int]],  # offset, tokenın metindeki karakter aralığını;
    sequence_ids: list[
        int | None
    ],  # sequence_id ise tokenın soru mu, context mi olduğunu gösterir.
    answer_start: int,
    answer_end: int,
) -> tuple[int, int] | None:
    """
    Character-level answer span'ını token span'ına çevirir.

    Raw baseline bize şunu verir:

        start = cevabın context içindeki başlangıç karakteri
        end   = cevabın context içindeki bitiş karakteri

    QA modeli ise token pozisyonlarıyla çalışır.

    Bu fonksiyon:
        character span -> token span

    dönüşümünü yapar.

    Returns:
        (start_token_index, end_token_index)

    Cevap bu chunk içinde değilse:
        None
    """

    token_start: int | None = None
    token_end: int | None = None

    for token_index, (offset, sequence_id) in enumerate(zip(offsets, sequence_ids)):
        # sequence_id == 1 context token'ını gösterir.
        # Question ve special tokenları kullanmıyoruz.
        if sequence_id != 1:
            continue

        offset_start, offset_end = offset

        #         if sequence_id != 1:
        #             continue

        #         offset_start, offset_end = offset

        # Special veya padding tokenlarında offset [0, 0]
        # olabilir.
        if offset_start == offset_end:
            continue

        # Answer'ın başladığı token.
        if token_start is None and offset_start <= answer_start < offset_end:
            token_start = token_index

        # answer_end exclusive character index'tir.
        # Yani cevabın son karakterinden bir sonrasını gösterir.
        if offset_start < answer_end <= offset_end:
            token_end = token_index

    if token_start is None or token_end is None or token_end < token_start:
        return None

    return token_start, token_end


def estimate_single_confidence(
    prediction: dict[str, Any], tokenizer: Any, model: Any, device: torch.device
) -> dict[str, Any]:
    """
    Tek bir raw prediction için confidence signal üretir.

    Core fikir:

        answer score =
            start logit + end logit

        null score =
            CLS start logit + CLS end logit

        margin =
            answer score - null score

    Margin pozitifse:
        model answer'ı null seçeneğine tercih ediyor.

    Margin negatifse:
        model "cevap yok" seçeneğine daha yakın.
    """

    required_fields = [
        "question",
        "context",
        "prediction_text",
        "start",
        "end",
    ]

    missing_fields = [field for field in required_fields if field not in prediction]

    if missing_fields:
        raise ValueError(f"Prediction is missing fields: {missing_fields}")

    question = prediction["question"]
    context = prediction["context"]

    answer_start = int(prediction["start"])

    answer_end = int(prediction["end"])

    # Uzun context'leri overlapping chunk'lara ayırıyoruz.
    # Overlapping chunk, uzun metni parçalarken parçaların bir kısmının ortak olmasıdır.
    # Örnek:
    # Chunk 1: Ali Bakü'de yaşıyor ve
    # Chunk 2: yaşıyor ve üniversiteye gidiyor

    # encoded, tokenizer’ın ürettiği model girdilerinin tamamıdır; yani token ID’leri, attention mask ve offset bilgileri burada tutulur.
    encoded = tokenizer(
        question,  # question soru metnidir,
        context,  #  context ise cevabın aranacağı metindir.
        truncation="only_second",  # truncation="only_second": Uzunluk aşılırsa yalnızca ikinci metni, yani contexti kısaltır; soru korunur.
        max_length=MAX_LENGTH,  # max_length=MAX_LENGTH: Her chunk’ın maksimum token sayısını belirler.
        stride=DOCUMENT_STRIDE,  # stride=DOCUMENT_STRIDE: Chunk’lar arasındaki ortak token miktarıdır.
        return_overflowing_tokens=True,  # return_overflowing_tokens=True: Uzun context’i birden fazla overlapping chunk’a böler.
        return_offsets_mapping=True,  # return_offsets_mapping=True: Her tokenın metindeki karakter başlangıç-bitiş konumunu verir.
        padding=True,  # padding=True: Kısa chunk’ları aynı uzunluğa tamamlar.
        return_tensors="pt",  # return_tensors="pt": Sonuçları PyTorch tensoru olarak döndürür.
    )

    # Offset mapping model input'u değildir.
    # Yalnızca token-character eşleştirmesi için kullanılır.
    offset_mapping = encoded.pop("offset_mapping")

    # Uzun context kaç chunk’a bölündüyse, her chunk’ın hangi örneğe ait olduğunu gösteren mapping’i siler. Burada tek örnek olduğu için gereksizdir.
    encoded.pop(
        "overflow_to_sample_mapping",
        None,
    )

    # Her chunk’taki tokenların:
    # 0 -> soruya,
    # 1 -> context’e,
    # None → special tokenlara ait olduğunu kaydeder.
    all_sequence_ids = [
        encoded.sequence_ids(chunk_index)
        for chunk_index in range(encoded["input_ids"].shape[0])
    ]

    # Şuna benzer bir sözlük döndürür:
    # {
    #     "input_ids": tensor(...),
    #     "attention_mask": tensor(...),
    #     "token_type_ids": tensor(...)
    # }
    model_inputs = {key: value.to(device) for key, value in encoded.items()}

    # Inference sırasında gradient hesaplamıyoruz. Inference, eğitilmiş modelin yeni bir veri üzerinde tahmin yapmasıdır.
    # torch.no_grad() gradient hesaplamayı kapatır; model(**model_inputs) ise girdileri modele verip start_logits ve end_logits gibi tahmin sonuçlarını outputs içine koyar.
    with torch.no_grad():
        outputs = model(**model_inputs)

    # Model outputları:
    # [chunk_count, sequence_length]
    # Modelin başlangıç skorlarını alır, gradient bağlantısını keser (detach) ve CPU’ya taşır (cpu).
    start_logits = outputs.start_logits.detach().cpu()

    # end_logits: Cevabın hangi tokenda biteceğine ait skorları alır, gradient bağlantısını keser ve CPU’ya taşır.
    end_logits = outputs.end_logits.detach().cpu()

    # input_ids: Token ID’lerini alır, gradient bağlantısını keser ve CPU’ya taşır.
    input_ids = encoded["input_ids"].detach().cpu()

    # offset_by_chunk: Her tokenın karakter aralığını alır, CPU’ya taşır ve Python listesine çevirir.
    offset_by_chunk = offset_mapping.detach().cpu().tolist()

    # Aynı answer span'i overlapping birden fazla chunk içinde bulunabilir.
    answer_scores: list[float] = []

    # Her chunk için no-answer skoru hesaplanacak.
    null_scores: list[float] = []

    selected_span: tuple[int, int] | None = (
        None  # Span, metindeki bir parçanın başlangıç ve bitiş aralığıdır.
    )
    selected_chunk: int | None = None

    for chunk_index in range(input_ids.shape[0]):
        cls_index = find_cls_index(
            input_ids=input_ids[chunk_index], cls_token_id=tokenizer.cls_token_id
        )

        # No-answer skoru:
        # CLS token'ın start ve end logit toplamı.
        # QA modeli cevabın başlangıç ve bitiş konumlarını ayrı ayrı tahmin eder. Bu nedenle bir cevabın toplam skoru, başlangıç ve bitiş logitlerinin toplanmasıyla hesaplanır. “Cevap yok” durumunda model hem başlangıç hem de bitiş için `CLS` tokenını seçtiği için `null_score`, `CLS start logit + CLS end logit` şeklinde bulunur.
        null_score = float(
            start_logits[chunk_index, cls_index].item()
            + end_logits[chunk_index, cls_index].item()
        )

        null_scores.append(null_score)

        token_span = find_answer_token_span(
            offsets=offset_by_chunk[  # Offset, bir tokenın metinde başladığı ve bittiği karakter konumudur.
                chunk_index
            ],
            sequence_ids=all_sequence_ids[chunk_index],
            answer_start=answer_start,
            answer_end=answer_end,
        )

        # Raw cevap bu chunk içinde değilse geç.
        if token_span is None:
            continue

        token_start, token_end = token_span

        # Raw answer'ın model logit skoru.
        answer_score = float(
            start_logits[chunk_index, token_start].item()
            + end_logits[chunk_index, token_end].item()
        )

        answer_scores.append(answer_score)

        # Aynı cevap birden fazla chunk'ta varsa
        # en güçlü span'i kaydediyoruz.
        if selected_chunk is None or answer_score > max(
            answer_scores[:-1], default=math.inf
        ):
            selected_chunk = chunk_index
            selected_span = (token_start, token_end)

    if not answer_scores:
        raise ValueError(
            "The raw answer span could not be mapped "
            f"to model tokens. Example ID: "
            f"{prediction.get('id', 'unknown')}"
        )

    if not null_scores:
        raise ValueError("No null scores were produced.")

    # answer_scores için max alınır; çünkü yüksek logit, modelin cevaba daha çok güvendiğini gösterir. Yani en güçlü cevap seçilir.
    best_answer_score = max(answer_scores)

    # null_scores için min alınır; çünkü düşük null skoru, modelin “cevap yok” seçeneğine daha az güvendiğini gösterir. Böylece cevabın bulunduğu en uygun chunk esas alınır.
    best_null_score = min(null_scores)

    # CORE UNCERTAINTY SIGNAL
    # Pozitif:
    # answer, no-answer'dan güçlü.
    #
    # Negatif:
    # no-answer daha güçlü.
    answer_null_margon = best_answer_score - best_null_score

    # null, modelin “cevap yok” seçeneğine verdiği skordur.
    # margin ise cevap skoru ile null skoru arasındaki farktır:
    # Margin pozitifse cevap var, negatifse model cevap yok seçeneğine daha yakındır.
    uncalibrated_confidence = sigmoid(answer_null_margon)

    updated_prediction = prediction.copy()

    # confidence_token_span: Cevabın başlangıç ve bitiş token indeksleri.
    updated_prediction.update(
        {
            # answer_span_logit: Modelin seçilen cevaba verdiği en yüksek skor.
            "answer_span_logit": (best_answer_score),
            # null_logit: Modelin “cevap yok” seçeneğine verdiği skor.
            "null_logit": best_null_score,
            # answer_null_margin: Cevap skoru ile null skoru arasındaki fark.
            "answer_null_margin": (answer_null_margon),
            # confidence: Margin’den üretilen 0–1 arası güven değeri.
            "confidence": (uncalibrated_confidence),
            # confidence_type: Güvenin cevap ve null karşılaştırmasından üretildiğini belirtir.
            "confidence_type": ("uncalibrated_answer_vs_null"),
            # confidence_is_calibrated: Güven değerinin kalibre edilip edilmediğini gösterir. Burada False.
            "confidence_is_calibrated": False,
            # confidence_model: Kullanılan modelin adı.
            "confidence_model": MODEL_NAME,
            # confidence_chunk: En güçlü cevabın bulunduğu chunk’ın numarası.
            "confidence_chunk": selected_chunk,
            # confidence_chunk: En güçlü cevabın bulunduğu chunk’ın numarası.
            "confidence_token_span": (
                list(selected_span) if selected_span is not None else None
            ),
        }
    )

    return updated_prediction


def estimate_confidences(
    predictions: list[dict[str, Any]], device_name: str
) -> list[dict[str, Any]]:
    """
    Bütün raw prediction'lar için confidence hesaplar.
    """

    if not predictions:
        raise ValueError("Prediction list cannot be empty.")

    device = select_device(device_name)

    print(f"Loading model: {MODEL_NAME}")

    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)

    # Dropout gibi training davranışlarını kapatır.
    model.eval()

    updated_predictions: list[dict[str, Any]] = []

    for index, prediction in enumerate(predictions, start=1):
        updated_prediction = estimate_single_confidence(
            prediction=prediction, tokenizer=tokenizer, model=model, device=device
        )

        updated_predictions.append(updated_prediction)

        print(
            f"{index}/{len(predictions)} | "
            f"margin="
            f"{updated_prediction['answer_null_margin']:.4f} | "
            f"confidence="
            f"{updated_prediction['confidence']:.4f}"
        )

    return updated_predictions


def run_confidence_estimator(
    input_path: str | Path, output_path: str | Path, device_name: str
) -> list[dict[str, Any]]:
    """
    Confidence estimator'ın ana çalışma fonksiyonu.
    """

    predictions = load_jsonl(input_path)

    updated_predictions = estimate_confidences(
        predictions=predictions, device_name=device_name
    )

    save_jsonl(updated_predictions, output_path)

    print(f"\nConfidence predictions saved to: {output_path}")

    return updated_predictions


def parse_arguments() -> argparse.Namespace:
    """
    Terminal argümanlarını okur.
    """
    parser = argparse.ArgumentParser(
        description=("Estimate QA confidence from answer-vs-null logits.")
    )

    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))

    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))

    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_confidence_estimator(
        input_path=args.input,
        output_path=args.output,
        device_name=args.device,
    )
