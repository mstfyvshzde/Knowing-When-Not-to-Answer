"""
Raw question-answering baseline.

Bu dosyanın görevi:
1. Hazırladığımız SQuAD 2.0 verisini yüklemek.
2. Hazır bir question-answering modelini açmak.
3. Her question + context çiftini modele vermek.
4. Answer ve confidence değerini kaydetmek.
5. Confidence düşük olsa bile her zaman ANSWER kararı vermek.

Bu sistem daha sonra geliştireceğimiz verification sisteminin
karşılaştırma noktasıdır.
"""

import argparse
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import pipeline

from src.utils.io import save_jsonl

# Kullanacağımız hazır question-answering modeli.

# Bu model SQuAD 2.0 üzerinde eğitilmiştir.
# Context içindeki bir answer span'ını bulmaya çalışır.
MODEL_NAME = "deepset/roberta-base-squad2"


# prepare_data.py tarafından oluşturulan dataset yolu.
DATASET_PATH = Path("data/processed/squad_v2")


# Model tahminlerinin kaydedileceği klasör.
OUTPUT_DIR = Path("outputs/predictions")


def select_device(device_name: str) -> torch.device:
    """
    Modelin hangi donanım üzerinde çalışacağını seçer.

    Seçenekler:
        cpu  → işlemci
        mps  → Apple Silicon GPU
        cuda → NVIDIA GPU

    Returns:
        PyTorch device nesnesi.
    """

    if device_name == "cuda":
        # Bilgisayarda NVIDIA GPU desteği var mı?
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but isnt avalable yet")

        return torch.device("cuda")

    if device_name == "mps":
        # MacBook'taki Apple GPU kullanılabilir mi?
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available yet")

        return torch.device("mps")

    # GPU seçilmemişse CPU kullanılır.
    return torch.device("cpu")


def load_split(split_name: str) -> Dataset:
    """
    Hazırlanmış dataset içinden tek bir split yükler.

    Örnek:
        load_split("calibration")

    Returns:
        Seçilen Hugging Face Dataset nesnesi.
    """

    # Dataset henüz hazırlanmamışsa programı durdurur.
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at: {DATASET_PATH}\n"
            "Run python -m src.data.prepare_data first."
        )

    # save_to_disk ile kaydettiğimiz dataset'i açıyoruz.
    dataset = load_from_disk(str(DATASET_PATH))

    # Train, calibration ve test birlikte olduğu için
    # DatasetDict bekliyoruz.
    if not isinstance(dataset, DatasetDict):
        raise TypeError("Expected the processed data to be a DatasetDict.")

    # Yanlış split adı verilmesini engelliyoruz.
    if split_name not in dataset:
        raise ValueError(
            f"Unknown split: {split_name}. Available splits: {list(dataset.keys())}"
        )

    return dataset[split_name]


def build_reference_answers(example: dict[str, Any]) -> list[str]:
    """
    Dataset içindeki doğru reference cevapları çıkarır.

    Answerable bir soru örneği:
        ["Albert Einstein", "Einstein"]

    Unanswerable bir soru:
        []

    Bu cevaplar daha sonra model prediction'ını
    değerlendirmek için kullanılacak.
    """

    answers = example.get("answers", {})

    return list(answers.get("text", []))


def run_eaw_baseline(
    split_name: str = "calibration",
    limit: int | None = 10,
    device_name: str = "cpu",
) -> list[dict[str, Any]]:
    """
    Raw baseline sistemini çalıştırır.

    Args:
        split_name:
            Kullanılacak dataset split'i.

            Şimdilik calibration kullanacağız.
            Test split'ine henüz dokunmayacağız.

        limit:
            Kaç örnek çalıştırılacağını belirler.

            10 -> yalnızca 10 soru
            None -> split içindeki tüm sorular

        device_name:
            cpu, mps veya cuda

    Returns:
        Her soru için prediction kayıtlarının listesi.
    """

    # Seçilen dataset split'ini yükle.
    dataset = load_split(split_name)

    # İlk testlerde tüm dataset'i çalıştırmak yerine küçük bir örnek grubu seçiyoruz.
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        # Dataset'in ilk limit adet örneğini seçer.
        dataset = dataset.select(range(min(limit, len(dataset))))

    # CPU, MPS veya CUDA cihazını seç.
    device = select_device(device_name)

    print(f"Loading model: {MODEL_NAME}")
    print(f"Using device: {device}")
    print(f"Number of examples: {len(dataset)}")

    # Hugging Face pipeline:
    # Model yükleme, tokenization ve inference (eğitilmiş bir yapay zekâ modelinin yeni bir veriyi kullanarak tahmin veya sonuç üretmesi) işlemlerini daha kolay kullanmamızı sağlar.
    qa_model = pipeline(
        task="question-answering", model=MODEL_NAME, tokenizer=MODEL_NAME, device=device
    )

    # Bütün model çıktıları burada toplanacak.
    predictions: list[dict[str, Any]] = []

    # Dataset içindeki örnekleri tek tek modele veriyoruz.
    for index, example in enumerate(dataset, start=1):
        # Model iki temel input alır:
        # 1. question
        # 2. context
        result = qa_model(
            question=example["question"],
            context=example["context"],
            # En iyi tek cevabı döndür.
            top_k=1,
            # Çok önemli:
            # False olduğu için model zorla bir cevap span'ı seçer.
            # Yani context'te cevap olmasa bile cevap üretir.
            # Raw baseline'ın amacı da budur:
            # her soruya cevap vermek.
            handle_impossible_answer=False,
        )

        # Tek bir soru için saklayacağımız kayıt.
        prediction = {
            # Dataset örneğinin benzersiz kimliği.
            "id": example["id"],
            # Kullanıcı sorusu.
            "question": example["question"],
            # Modelin bulduğu cevap.
            "prediction_text": result["answer"],
            # Modein seçtiği answer span'ına verdiği skor.
            # Bu değer gerçek doğruluk olasılığı değildir.
            # Daha sonra calibration ile bunu inceleyeceğiz.
            # Bu skor gerçek confidence değildir.
            "pipeline_score": float(result["score"]),
            # Cevabın context içinde başladığı karakter.
            "start": int(result["start"]),
            # Cevabın bittiği karakter.
            "end": int(result["end"]),
            # Dataset tarafından verilen doğru cevaplar.
            "reference_answers": build_reference_answers(example),
            # Context içinde gerçekten cevap var mı?
            "is_answerable": bool(example["is_answerable"]),
            # Raw baseline hiçbir zaman abstain etmez.
            "decision": "ANSWER",
            # Sonucun hangi sistemden geldiğini kaydediyoruz.
            "system": "raw_baseline",
            # Kullanılan modeli kaydediyoruz.
            "model": MODEL_NAME,
            # Hangi split üzerinde çalıştığını kaydediyoruz.
            "split": split_name,
            # Evidence verification aşamasında kullanılacak kaynak metin.
            "context": example["context"],
        }

        predictions.append(prediction)

        print(f"\nExample {index}/{len(dataset)}")
        print(f"Question: {example['question']}")
        print(f"Prediction: {result['answer']}")
        print(f"Confidence: {result['score']:.4f}")
        print(f"Answerable: {example['is_answerable']}")
        print("Decision: ANSWER")

    # Output klasörü yoksa oluştur.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Dosya adı kullanılan split'e göre oluşturulur.
    # Split, veri setinin train, calibration ve test gibi ayrı bölümlere ayrılmış hâlidir; örneğin split_name="calibration" seçilirse model yalnızca calibration bölümündeki sorular üzerinde çalışır.
    output_path = OUTPUT_DIR / f"raw_baseline_{split_name}.jsonl"

    # Prediction listesini JSONL dosyasına kaydet.
    save_jsonl(
        predictions,
        output_path,
    )

    print(f"\nPredictions saved to: {output_path}")

    return predictions


def parse_arguments() -> argparse.Namespace:
    """
    Terminal'den verilen seçenekleri okur.

    Örnek:

        python -m src.baselines.raw_answer_baseline \
            --split calibration \
            --limit 10 \
            --device mps
    """

    parser = argparse.ArgumentParser(
        description=("Run the raw question-answering baseline.")
    )

    # Kullanılacak dataset split'i.
    # parser, terminalden gelen --split, --limit ve --device seçeneklerini okuyup saklayan ArgumentParser nesnesidir. Örneğin parser.add_argument('--split', ...) ile --split seçeneği tanımlanır.    # ArgumentParser, Python programına terminalden verilen --split, --limit gibi seçenekleri okuyup anlamlandıran argparse sınıfıdır
    # argparse, Python programlarına terminalden verilen seçenekleri ve değerleri okumayı kolaylaştıran yerleşik bir kütüphanedir.
    # Örnek: python program.py --limit 10 komutundaki --limit 10 kısmını argparse okur.
    parser.add_argument(
        "--split",
        choices=[
            "train",
            "calibration",  # Calibration, modelin hangi confidence değerinin altında cevaba güvenmemesi gerektiğini belirlemek için kullanılan veri bölümüdür; örneğin bu sınır 0.60 seçilebilir.
            "test",
        ],
        default="calibration",
    )

    # Kaç örnek çalıştırılacak?
    parser.add_argument("--limit", type=int, default=10)

    # Model hangi cihazda çalışacak?
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")

    return parser.parse_args()


if __name__ == "__main__":
    """
    Dosya doğrudan çalıştırıldığında:

    1. Terminal seçeneklerini oku.
    2. Raw baseline'ı çalıştır.
    """

    args = parse_arguments()

    run_eaw_baseline(
        split_name=args.split,
        limit=args.limit,
        device_name=args.device,
    )
