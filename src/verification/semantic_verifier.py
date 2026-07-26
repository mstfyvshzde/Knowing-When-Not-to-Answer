"""
Semantic evidence verifier based on Natural Language Inference.

Bu verifier herhangi bir özel soru, cevap, kişi, sayı veya
relation listesine bağlı değildir.

İşlem:

    Evidence / context
            +
    Question-answer hypothesis
            ↓
        NLI model
            ↓
    ENTAILMENT / NEUTRAL / CONTRADICTION

Bu dosya lexical evidence verifier'ın yerine geçmez.
Bağımsız bir semantic verification katmanı oluşturur.
"""

import argparse
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.utils.io import (
    load_jsonl,
    save_jsonl,
)

DEFAULT_INPUT_PATH = Path("outputs/predictions/calibration_with_evidence.jsonl")

DEFAULT_OUTPUT_PATH = Path(
    "outputs/predictions/calibration_with_semantic_evidence.jsonl"
)

DEFAULT_MODEL_NAME = "FacebookAI/roberta-large-mnli"


def get_predicted_answer(prediction: dict[str, Any]) -> str:
    """
    Prediction kaydındaki tahmin edilen cevabı çıkarır.
    """

    answer_fields = (
        "predicted_answer",
        "prediction_text",
        "prediction_answer",
        "answer",
    )

    for field in answer_fields:
        value = prediction.get(field)

        if value is not None:
            return str(value).strip()

    return ""


def get_evidence_text(prediction: dict[str, Any]) -> str:
    """
    Semantic verifier için kullanılacak evidence metnini seçer.

    Öncelik:
        1. Lexical verifier tarafından çıkarılan evidence_text
        2. Tam context
    """

    evidence_text = str(
        prediction.get(
            "evidence_text",
            "",
        )
    ).strip()

    if evidence_text:
        return evidence_text

    return str(
        prediction.get(
            "context",
            "",
        )
    ).strip()


def build_qa_hypothesis(question: str, predicted_answer: str) -> str:
    """
    Question ve predicted answer'dan genel bir NLI
    hypothesis oluşturur.

    Bu fonksiyon soru türlerine veya belirli relation
    kalıplarına özel kurallar kullanmaz.
    """

    question = question.strip()
    predicted_answer = predicted_answer.strip()

    if not question or not predicted_answer:
        return ""

    return f'The answer to the question "{question}" is "{predicted_answer}".'


def resolve_device() -> torch.device:
    """
    Kullanılabilecek en uygun PyTorch device'ı seçer.

    Öncelik:
        CUDA → Apple MPS → CPU
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def normalize_label_name(label: str) -> str:
    """
    Model label isimlerini ortak formata dönüştürür.
    """

    normalized = str(label).upper().strip()

    aliases = {
        "ENTAILMENT": "ENTAILMENT",
        "NEUTRAL": "NEUTRAL",
        "CONTRADICTION": "CONTRADICTION",
    }

    return aliases.get(
        normalized,
        normalized,
    )


def find_label_id(id_to_label: dict[int, str], target_label: str) -> int:
    """
    Model config içindeki target label'ın ID'sini bulur.
    """

    target_label = target_label.upper()

    for label_id, label_name in id_to_label.items():
        normalized_label = normalize_label_name(label_name)

        if normalized_label == target_label:
            return int(label_id)

    raise ValueError(
        f"Could not find NLI label {target_label!r}. Available labels: {id_to_label}"
    )


class SemanticEvidenceVerifier:
    """
    NLI modelini yükleyen ve semantic evidence
    doğrulaması yapan sınıf.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.device = resolve_device()

        print(f"Loading semantic verifier: {model_name}")

        print(f"Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.model.to(self.device)

        self.model.eval()

        raw_id_to_label = self.model.config.id2label

        self.id_to_label = {
            int(label_id): str(label_name)
            for label_id, label_name in raw_id_to_label.items()
        }

        self.entailment_id = find_label_id(
            self.id_to_label,
            "ENTAILMENT",
        )

        self.neutral_id = find_label_id(
            self.id_to_label,
            "NEUTRAL",
        )

        self.contradiction_id = find_label_id(
            self.id_to_label,
            "CONTRADICTION",
        )

    def predict_batch(
        self,
        premises: list[str],
        hypotheses: list[str],
        batch_size: int = 8,
    ) -> list[dict[str, Any]]:
        """
        Evidence-hypothesis çiftlerinde toplu NLI inference yapar.
        """

        if len(premises) != len(hypotheses):
            raise ValueError("Premises and hypotheses must have the same length.")

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        results: list[dict[str, Any]] = []

        for start_index in range(
            0,
            len(premises),
            batch_size,
        ):
            end_index = start_index + batch_size

            premise_batch = premises[start_index:end_index]

            hypothesis_batch = hypotheses[start_index:end_index]

            processed_end = min(
                end_index,
                len(premises),
            )

            print(
                f"Processing semantic batch: "
                f"{start_index + 1}-{processed_end}"
                f"/{len(premises)}",
                flush=True,
            )

            encoded_inputs = self.tokenizer(
                premise_batch,
                hypothesis_batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            encoded_inputs = {
                key: value.to(self.device) for key, value in encoded_inputs.items()
            }

            with torch.inference_mode():
                outputs = self.model(**encoded_inputs)

                probabilities = torch.softmax(
                    outputs.logits,
                    dim=-1,
                )

            probabilities = probabilities.detach().cpu()

            for probability_vector in probabilities:
                entailment_probability = float(
                    probability_vector[self.entailment_id].item()
                )

                neutral_probability = float(probability_vector[self.neutral_id].item())

                contradiction_probability = float(
                    probability_vector[self.contradiction_id].item()
                )

                label_probabilities = {
                    "ENTAILMENT": (entailment_probability),
                    "NEUTRAL": (neutral_probability),
                    "CONTRADICTION": (contradiction_probability),
                }

                predicted_label = max(
                    label_probabilities,
                    key=label_probabilities.get,
                )

                results.append(
                    {
                        "semantic_label": (predicted_label),
                        "entailment_probability": (entailment_probability),
                        "neutral_probability": (neutral_probability),
                        "contradiction_probability": (contradiction_probability),
                        "semantic_confidence": (label_probabilities[predicted_label]),
                    }
                )

        return results


def validate_predictions(predictions: list[dict[str, Any]]) -> None:
    """
    Prediction kayıtlarının gerekli alanlarını kontrol eder.
    """

    if not predictions:
        raise ValueError("Prediction list cannot be empty.")

    for index, prediction in enumerate(
        predictions,
        start=1,
    ):
        missing_fields: list[str] = []

        if "question" not in prediction:
            missing_fields.append("question")

        if "context" not in prediction and "evidence_text" not in prediction:
            missing_fields.append("context or evidence_text")

        has_answer = any(
            field in prediction
            for field in (
                "predicted_answer",
                "prediction_text",
                "prediction_answer",
                "answer",
            )
        )

        if not has_answer:
            missing_fields.append("predicted answer")

        if missing_fields:
            raise ValueError(
                f"Prediction {index} is missing "
                f"fields: {missing_fields}. "
                f"Available keys: "
                f"{list(prediction.keys())}"
            )


def build_semantic_inputs(
    predictions: list[dict[str, Any]],
) -> tuple[
    list[str],
    list[str],
]:
    """
    Tüm prediction kayıtları için premise ve
    hypothesis listelerini oluşturur.
    """

    premises: list[str] = []
    hypotheses: list[str] = []

    for prediction in predictions:
        question = str(
            prediction.get(
                "question",
                "",
            )
        ).strip()

        predicted_answer = get_predicted_answer(prediction)

        evidence_text = get_evidence_text(prediction)

        hypothesis = build_qa_hypothesis(
            question=question,
            predicted_answer=(predicted_answer),
        )

        premises.append(evidence_text)

        hypotheses.append(hypothesis)

    return premises, hypotheses


def run_semantic_verification(
    input_path: str | Path,
    output_path: str | Path,
    model_name: str,
    batch_size: int,
    max_length: int,
) -> list[dict[str, Any]]:
    """
    Tüm predictions üzerinde semantic verification çalıştırır.
    """

    predictions = load_jsonl(input_path)

    validate_predictions(predictions)

    premises, hypotheses = build_semantic_inputs(predictions)

    verifier = SemanticEvidenceVerifier(
        model_name=model_name,
        max_length=max_length,
    )

    semantic_results = verifier.predict_batch(
        premises=premises,
        hypotheses=hypotheses,
        batch_size=batch_size,
    )

    if len(semantic_results) != len(predictions):
        raise RuntimeError("Semantic result count does not match prediction count.")

    verified_predictions: list[dict[str, Any]] = []

    label_counts = {
        "ENTAILMENT": 0,
        "NEUTRAL": 0,
        "CONTRADICTION": 0,
    }

    for index, (
        prediction,
        hypothesis,
        semantic_result,
    ) in enumerate(
        zip(
            predictions,
            hypotheses,
            semantic_results,
        ),
        start=1,
    ):
        updated_prediction = prediction.copy()

        updated_prediction.update(
            {
                "semantic_hypothesis": (hypothesis),
                "semantic_label": (semantic_result["semantic_label"]),
                "entailment_probability": (semantic_result["entailment_probability"]),
                "neutral_probability": (semantic_result["neutral_probability"]),
                "contradiction_probability": (
                    semantic_result["contradiction_probability"]
                ),
                "semantic_confidence": (semantic_result["semantic_confidence"]),
                "semantic_verifier_model": (model_name),
            }
        )

        verified_predictions.append(updated_prediction)

        semantic_label = semantic_result["semantic_label"]

        label_counts[semantic_label] += 1

        print(
            f"{index}/{len(predictions)} | "
            f"semantic={semantic_label} | "
            f"entailment="
            f"{semantic_result['entailment_probability']:.4f} | "
            f"neutral="
            f"{semantic_result['neutral_probability']:.4f} | "
            f"contradiction="
            f"{semantic_result['contradiction_probability']:.4f}"
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_jsonl(
        verified_predictions,
        output_path,
    )

    print("\nSemantic verification completed.")

    print(f"ENTAILMENT: {label_counts['ENTAILMENT']}")

    print(f"NEUTRAL: {label_counts['NEUTRAL']}")

    print(f"CONTRADICTION: {label_counts['CONTRADICTION']}")

    print(f"Results saved to: {output_path}")

    return verified_predictions


def parse_arguments() -> argparse.Namespace:
    """
    Command-line argümanlarını okur.
    """

    parser = argparse.ArgumentParser(
        description=("Run general NLI-based semantic evidence verification.")
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
        "--model-name",
        default=DEFAULT_MODEL_NAME,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_semantic_verification(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
