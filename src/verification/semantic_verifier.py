"""
To check whether a predicted answer is semantically supported by the available evidence/context using an NLI model. It turns each question + predicted answer into a hypothesis, compares it with the evidence, and outputs ENTAILMENT, NEUTRAL, or CONTRADICTION with probabilities.
"""


import argparse 
from pathlib import Path
from typing import Any

import torch

# AutoTokenizer -> converts text into tokens/numbers that the transformer model can understand.
# AutoModelForSequenceClassification -> loads a transformer model that classifies text into labels/classes.
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)

from src.utils.io import (
    load_jsonl,
    save_jsonl
)

DEFAULT_INPUT_PATH = Path("outputs/predictions/calibration_with_evidence.jsonl")

DEFAULT_OUTPUT_PATH = Path("outputs/predictions/calibration_with_semantic_evidence.jsonl")


# This is the default Hugging Face model used for semantic/NLI classification.
DEFAULT_MODEL_NAME = "FacebookAI/roberta-large-mnli"


# find and return the predicted answer text from a prediction dictionary, even if different files use different field names.
def get_predicted_answer(
    prediction: dict[str, Any]
) -> str:
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

    # If none of those fields exist, it returns:
    return ''


# To get the evidence text for a prediction.
def get_evidence_text(prediction: dict[str, Any]) -> str:
    evidence_text = str(
        prediction.get(
            'evidence_text',
            ''
        )
    ).strip()

    if evidence_text:
        return evidence_text

    return str(
        prediction.get(
            'context',
            ''
        )
    ).strip()


# To convert the question + predicted answer into a clear statement that the NLI model can evaluate.
def build_qa_hypothesis(question: str, predicted_answer: str) -> str:
    question = question.strip()
    predicted_answer = predicted_answer.strip()

    if not question or not predicted_answer:
        return ''

    return f'The answer to the question "{question}" is "{predicted_answer}".'


# To choose the best available hardware device for running the model
def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device('cpu')



# To standardize model label names into a consistent format.
def normalize_label_name(label: str) -> str:
    normalized = str(label).upper().strip()

    # The aliases dictionary also gives a place to map alternative label names later if needed.
    aliases = {
        "ENTAILMENT": "ENTAILMENT",
        "NEUTRAL": "NEUTRAL",
        "CONTRADICTION": "CONTRADICTION",
    }

    # “Try to find normalized inside the aliases dictionary. If it exists, return the mapped value. If it does not exist, return normalized itself.”
    return aliases.get(
        normalized,
        normalized,
    )


# To find the numeric ID that the NLI model uses for a specific label such as ENTAILMENT, NEUTRAL, or CONTRADICTION.
def find_label_id(id_to_label: dict[int, str], target_label: str) -> int:
    target_label = target_label.upper()

    for label_id, label_name in id_to_label.items():
        normalized_label = normalize_label_name(label_name)

        if normalized_label == target_label:
            return int(label_id)

    raise ValueError(
        f"Could not find NLI label {target_label!r}. Available labels: {id_to_label}"
    )


# use an NLI model to compare evidence with a question-answer statement and classify the relationship as ENTAILMENT, NEUTRAL, or CONTRADICTION.
class SemanticEvidenceVerifier:
    # loads the tokenizer/model, selects the device, and finds the label IDs.
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        # the tokenizer will use at most 512 tokens for each input pair.
        max_length: int = 512
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.device = resolve_device()

        print(f"Loading semantic verifier: {model_name}")

        print(f"Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # moves the model to the selected hardware: GPU, Apple MPS, or CPU.
        self.model.to(self.device)
        # puts the model in prediction mode and disables training behaviors like dropout, making outputs more stable and consistent.
        self.model.eval()

        # ets the model’s mapping from numeric class IDs to label names.
        # Example: {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
        raw_id_to_label = self.model.config.id2label

        self.id_to_label = {
            int(label_id): str(label_name)
            for label_id, label_name in raw_id_to_label.items()
        }


        # These lines find and store the numeric class ID for each NLI label: ENTAILMENT, NEUTRAL, and CONTRADICTION.
        # For example, if ENTAILMENT = 2, then self.entailment_id becomes 2, so later the code can read the correct probability from the model output.
        self.entailment_id = find_label_id(
            self.id_to_label,
            'ENTAILMENT'
        )

        self.neutral_id = find_label_id(
            self.id_to_label,
            'NEUTRAL'
        )

        self.contradiction_id = find_label_id(
            self.id_to_label,
            'CONTRADICTION'
        )

    # It processes many evidence–hypothesis pairs in batches, runs them through the NLI model, and returns ENTAILMENT, NEUTRAL, or CONTRADICTION probabilities plus the final semantic label.
    def predict_batch(
        self, 
        # the evidence/context texts the model will check.
        premises: list[str],
        # the statements built from the question + predicted answer that are tested against the premises.
        hypotheses: list[str],
        # process 8 premise–hypothesis pairs at a time.
        batch_size: int = 8
    ) -> list[dict[str, Any]]:
        # Example:
        # premises = ["James Watt improved the steam engine."]
        # hypotheses = ['The answer to the question "Who improved the steam engine?" is "James Watt".']
        # Then the NLI model checks whether the hypothesis is supported by the premise. ✅
        
        
        if len(premises) != len(hypotheses):
            raise ValueError("Premises and hypotheses must have the same length.")

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        results: list[dict[str, Any]] = []

        for start_index in range(
            0,
            len(premises),
            batch_size
        ):
            end_index = start_index + batch_size
            premise_batch = premises[start_index: end_index]
            hypothesis_batch = hypotheses[start_index: end_index]

            processed_end = min(
                end_index,
                len(premises)
            )

            print(
                f"Processing semantic batch: "
                f"{start_index + 1}-{processed_end}"
                f"/{len(premises)}",
                flush=True,
            )

            # This part converts the text pairs into numeric tensors that the transformer model can understand.
            encoded_inputs = self.tokenizer(
                premise_batch,
                hypothesis_batch,
                # makes sequences in the batch the same length
                padding=True,
                truncation=True,
                # limits input length
                max_length=self.max_length,
                # returns PyTorch tensors
                return_tensors='pt'
            )

            # This part moves every tokenizer output tensor to the same device as the model — GPU, MPS, or CPU.
            encoded_inputs = {
                key: value.to(self.device) for key, value in encoded_inputs.items()
            }

            # This block runs the model on the encoded inputs without training and gets the raw prediction scores (logits).
            with torch.inference_mode():
                # We use ** to unpack a dictionary into keyword arguments
                outputs = self.model(**encoded_inputs)

                probabilities = torch.softmax(
                    outputs.logits,
                    dim=-1
                )

            # detach() → disconnects the probabilities from PyTorch’s computation graph, because we are not training.
            # .cpu() → moves them from GPU/MPS back to the CPU so they are easier to process and convert to normal Python values.
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
                    "CONTRADICTION": (contradiction_probability)
                }

                predicted_label = max(
                    label_probabilities,
                    key=label_probabilities.get
                )


                results.append(
                    {
                        "semantic_label": (predicted_label),
                        "entailment_probability": (entailment_probability),
                        "neutral_probability": (neutral_probability),
                        "contradiction_probability": (contradiction_probability),
                        "semantic_confidence": (label_probabilities[predicted_label])
                    }
                )

        return results


# To validate that every prediction has the minimum required data before semantic verification starts
def validate_predictions(predictions: list[dict[str, Any]]) -> None:
    if not predictions: 
        raise ValueError("Prediction list cannot be empty.")

    for index, prediction in enumerate(
        predictions, start=1
    ):
        missing_fields: list[str] = []

        if 'question' not in prediction:
            missing_fields.append('question')

        if 'context' not in prediction and 'evidence_text' not in prediction:
            missing_fields.append('context and evidence_text')  

        has_answer = any(
            field in prediction
            for field in (
                "predicted_answer",
                "prediction_text",
                "prediction_answer",
                "answer"
            )
        )

        if not has_answer:
            missing_fields.append('predicted aswer')

        if missing_fields:
            raise ValueError(
                f"Prediction {index} is missing "
                f"fields: {missing_fields}. "
                f"Available keys: "
                f"{list(prediction.keys())}"
            )


# To convert the raw predictions into two lists that the NLI model can process: premises and hypotheses. ✅
# premises -> evidence/context
# hypotheses -> question + predicted answer statemen
def build_semantic_inputs(
    predictions: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    premises: list[str] = []
    hypotheses: list[str] =[]

    for prediction in predictions:
        question = str(
            prediction.get(
                'question',
                ''
            )
        ).strip()

        predicted_answer = get_predicted_answer(prediction)
        evidence_text = get_evidence_text(prediction)
        hypothesis = build_qa_hypothesis(
            question=question,
            predicted_answer=predicted_answer
        )

        premises.append(evidence_text)
        hypotheses.append(hypothesis)

    return premises, hypotheses


# uns the complete semantic-verification pipeline on all predictions and saves the results.
def run_semantic_verification(
    input_path: str | Path,
    output_path: str | Path,
    model_name: str,
    batch_size: int,
    max_length: int
) -> list[dict[str, Any]]:
    predictions = load_jsonl(input_path)

    validate_predictions(predictions)

    premises, hypotheses = build_semantic_inputs(predictions)

    verifier = SemanticEvidenceVerifier(
        model_name=model_name,
        max_length=max_length
    )

    semantic_results = verifier.predict_batch(
        premises=premises,
        hypotheses=hypotheses,
        batch_size=batch_size
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
        semantic_result
    ) in enumerate(
        zip(
            predictions,
            hypotheses,
            semantic_results
        ),
        start=1
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

        semantic_label = semantic_result['semantic_label']
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
        exist_ok=True
    )

    save_jsonl(
        verified_predictions,
        output_path
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
        default=str(DEFAULT_INPUT_PATH)
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH)
    )

    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512
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
