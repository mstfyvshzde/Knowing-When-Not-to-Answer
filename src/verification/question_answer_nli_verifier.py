"""
Question-aware semantic evidence verifier.

Pipeline
--------
1. Convert a question-answer pair into a declarative claim.
2. Compare the generated claim against the context using NLI.
3. Store entailment, neutral, and contradiction probabilities.
4. Assign an evidence label:
       ENTAILMENT
       NEUTRAL
       CONTRADICTION

Example
-------
Question:
    What is stainless steel's theoretical Carnot efficiency?

Predicted answer:
    63%

Generated claim:
    Stainless steel's theoretical Carnot efficiency is 63%.

The generated claim is then evaluated against the context.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

DEFAULT_INPUT_PATH = Path("outputs/predictions/calibration_with_hybrid_evidence.jsonl")

DEFAULT_OUTPUT_PATH = Path(
    "outputs/predictions/calibration_with_question_aware_semantic_evidence.jsonl"
)

DEFAULT_QA2D_MODEL = "google/flan-t5-base"
DEFAULT_NLI_MODEL = "FacebookAI/roberta-large-mnli"

DEFAULT_MAX_CONTEXT_TOKENS = 384
DEFAULT_MAX_CLAIM_TOKENS = 96
DEFAULT_GENERATION_MAX_NEW_TOKENS = 64

DEFAULT_ENTAILMENT_THRESHOLD = 0.50
DEFAULT_CONTRADICTION_THRESHOLD = 0.50

DEFAULT_BATCH_SIZE = 8


def load_jsonl(
    path: str | Path,
) -> list[dict[str, Any]]:
    """
    Load a JSON Lines file.
    """

    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    records: list[dict[str, Any]] = []

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        for line_number, line in enumerate(
            input_file,
            start=1,
        ):
            stripped_line = line.strip()

            if not stripped_line:
                continue

            try:
                record = json.loads(stripped_line)

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number} in {input_path}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise TypeError(f"Line {line_number} must contain a JSON object.")

            records.append(record)

    return records


def save_jsonl(
    records: Iterable[dict[str, Any]],
    path: str | Path,
) -> None:
    """
    Save records as JSON Lines.
    """

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for record in records:
            output_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def get_first_value(
    record: dict[str, Any],
    field_names: tuple[str, ...],
    default: Any = None,
) -> Any:
    """
    Return the first available non-None value.
    """

    for field_name in field_names:
        value = record.get(field_name)

        if value is not None:
            return value

    return default


def get_question(
    record: dict[str, Any],
) -> str:
    """
    Extract the question.
    """

    value = get_first_value(
        record=record,
        field_names=(
            "question",
            "question_text",
            "query",
        ),
        default="",
    )

    return clean_text(str(value))


def get_predicted_answer(
    record: dict[str, Any],
) -> str:
    """
    Extract the model's predicted answer.
    """

    value = get_first_value(
        record=record,
        field_names=(
            "predicted_answer",
            "prediction_text",
            "prediction_answer",
            "answer",
        ),
        default="",
    )

    return clean_text(str(value))


def get_context(
    record: dict[str, Any],
) -> str:
    """
    Extract the evidence context.
    """

    value = get_first_value(
        record=record,
        field_names=(
            "context",
            "passage",
            "evidence_context",
            "source_context",
        ),
        default="",
    )

    return clean_text(str(value))


def clean_text(
    text: str,
) -> str:
    """
    Normalise whitespace without changing content.
    """

    return re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip()


def normalise_generated_claim(
    claim: str,
) -> str:
    """
    Clean a generated declarative claim.
    """

    cleaned_claim = clean_text(claim)

    prefixes = (
        "statement:",
        "claim:",
        "declarative statement:",
        "answer:",
    )

    lowered_claim = cleaned_claim.lower()

    for prefix in prefixes:
        if lowered_claim.startswith(prefix):
            cleaned_claim = cleaned_claim[len(prefix) :].strip()

            break

    cleaned_claim = cleaned_claim.strip("\"' ")

    if cleaned_claim and cleaned_claim[-1] not in ".!?":
        cleaned_claim += "."

    return cleaned_claim


def build_qa2d_prompt(
    question: str,
    answer: str,
) -> str:
    """
    Build the instruction used to convert a QA pair
    into one self-contained declarative statement.
    """

    return (
        "Convert the following question and answer into "
        "one concise, self-contained declarative statement. "
        "Preserve the meaning exactly. Do not add facts, "
        "explanations, or uncertainty.\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        "Statement:"
    )


def fallback_claim(
    question: str,
    answer: str,
) -> str:
    """
    Safe fallback when claim generation returns empty text.

    This representation still keeps the question and answer
    together instead of verifying the answer alone.
    """

    return f'The answer to the question "{question}" is "{answer}".'


def select_device() -> torch.device:
    """
    Select CUDA, Apple Silicon MPS, or CPU.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class QuestionToClaimConverter:
    """
    Convert question-answer pairs into declarative claims
    using an instruction-tuned sequence-to-sequence model.
    """

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        max_input_tokens: int,
        max_new_tokens: int,
    ) -> None:
        self.device = device
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens

        print(f"Loading QA-to-claim model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def convert_batch(
        self,
        questions: list[str],
        answers: list[str],
    ) -> list[str]:
        """
        Generate claims for a batch of QA pairs.
        """

        if len(questions) != len(answers):
            raise ValueError("Question and answer batch sizes must match.")

        prompts = [
            build_qa2d_prompt(
                question=question,
                answer=answer,
            )
            for question, answer in zip(
                questions,
                answers,
            )
        ]

        encoded_inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
            return_tensors="pt",
        )

        encoded_inputs = {
            key: value.to(self.device) for key, value in encoded_inputs.items()
        }

        generated_ids = self.model.generate(
            **encoded_inputs,
            max_new_tokens=self.max_new_tokens,
            num_beams=4,
            do_sample=False,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

        generated_claims = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        final_claims: list[str] = []

        for question, answer, claim in zip(
            questions,
            answers,
            generated_claims,
        ):
            normalised_claim = normalise_generated_claim(claim)

            if not normalised_claim:
                normalised_claim = fallback_claim(
                    question=question,
                    answer=answer,
                )

            final_claims.append(normalised_claim)

        return final_claims


class QuestionAwareNLIVerifier:
    """
    Verify generated question-answer claims using an
    MNLI sequence-classification model.
    """

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        max_context_tokens: int,
        max_claim_tokens: int,
        entailment_threshold: float,
        contradiction_threshold: float,
    ) -> None:
        self.device = device

        self.max_context_tokens = max_context_tokens

        self.max_claim_tokens = max_claim_tokens

        self.entailment_threshold = entailment_threshold

        self.contradiction_threshold = contradiction_threshold

        print(f"Loading NLI model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

        self.label_to_id = self._resolve_label_ids()

    def _resolve_label_ids(
        self,
    ) -> dict[str, int]:
        """
        Resolve model-specific NLI label indices.
        """

        resolved_labels: dict[str, int] = {}

        for raw_id, raw_label in self.model.config.id2label.items():
            label = str(raw_label).strip().lower()

            label_id = int(raw_id)

            if "entail" in label:
                resolved_labels["entailment"] = label_id

            elif "neutral" in label:
                resolved_labels["neutral"] = label_id

            elif "contrad" in label:
                resolved_labels["contradiction"] = label_id

        required_labels = {
            "entailment",
            "neutral",
            "contradiction",
        }

        missing_labels = required_labels - set(resolved_labels)

        if missing_labels:
            raise ValueError(
                "Could not resolve NLI label IDs. "
                f"Missing: {sorted(missing_labels)}. "
                f"Model labels: "
                f"{self.model.config.id2label}"
            )

        return resolved_labels

    def _truncate_text(
        self,
        text: str,
        max_tokens: int,
    ) -> str:
        """
        Truncate text using the NLI tokenizer.
        """

        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_tokens,
        )

        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    def _assign_label(
        self,
        entailment_probability: float,
        neutral_probability: float,
        contradiction_probability: float,
    ) -> str:
        """
        Assign a semantic evidence label.

        Contradiction receives priority when its
        probability reaches the configured threshold.
        """

        if contradiction_probability >= self.contradiction_threshold:
            return "CONTRADICTION"

        if entailment_probability >= self.entailment_threshold:
            return "ENTAILMENT"

        largest_probability = max(
            entailment_probability,
            neutral_probability,
            contradiction_probability,
        )

        if largest_probability == contradiction_probability:
            return "CONTRADICTION"

        if largest_probability == entailment_probability:
            return "ENTAILMENT"

        return "NEUTRAL"

    @torch.inference_mode()
    def verify_batch(
        self,
        contexts: list[str],
        claims: list[str],
    ) -> list[dict[str, Any]]:
        """
        Verify a batch of context-claim pairs.
        """

        if len(contexts) != len(claims):
            raise ValueError("Context and claim batch sizes must match.")

        truncated_contexts = [
            self._truncate_text(
                text=context,
                max_tokens=(self.max_context_tokens),
            )
            for context in contexts
        ]

        truncated_claims = [
            self._truncate_text(
                text=claim,
                max_tokens=(self.max_claim_tokens),
            )
            for claim in claims
        ]

        encoded_inputs = self.tokenizer(
            truncated_contexts,
            truncated_claims,
            padding=True,
            truncation="only_first",
            max_length=512,
            return_tensors="pt",
        )

        encoded_inputs = {
            key: value.to(self.device) for key, value in encoded_inputs.items()
        }

        outputs = self.model(**encoded_inputs)

        probabilities = (
            torch.softmax(
                outputs.logits,
                dim=-1,
            )
            .detach()
            .cpu()
        )

        results: list[dict[str, Any]] = []

        entailment_id = self.label_to_id["entailment"]

        neutral_id = self.label_to_id["neutral"]

        contradiction_id = self.label_to_id["contradiction"]

        for row in probabilities:
            entailment_probability = float(row[entailment_id].item())

            neutral_probability = float(row[neutral_id].item())

            contradiction_probability = float(row[contradiction_id].item())

            semantic_label = self._assign_label(
                entailment_probability=(entailment_probability),
                neutral_probability=(neutral_probability),
                contradiction_probability=(contradiction_probability),
            )

            semantic_confidence = max(
                entailment_probability,
                neutral_probability,
                contradiction_probability,
            )

            results.append(
                {
                    "qa_nli_label": (semantic_label),
                    "qa_nli_confidence": (semantic_confidence),
                    "qa_entailment_probability": (entailment_probability),
                    "qa_neutral_probability": (neutral_probability),
                    "qa_contradiction_probability": (contradiction_probability),
                }
            )

        return results


def batched_indices(
    total_size: int,
    batch_size: int,
) -> Iterable[tuple[int, int]]:
    """
    Yield start and end indices for batches.
    """

    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    for start_index in range(
        0,
        total_size,
        batch_size,
    ):
        end_index = min(
            start_index + batch_size,
            total_size,
        )

        yield (
            start_index,
            end_index,
        )


def validate_records(
    records: list[dict[str, Any]],
) -> None:
    """
    Validate required data before model inference.
    """

    if not records:
        raise ValueError("Input file contains no predictions.")

    missing_questions = 0
    missing_answers = 0
    missing_contexts = 0

    for record in records:
        if not get_question(record):
            missing_questions += 1

        if not get_predicted_answer(record):
            missing_answers += 1

        if not get_context(record):
            missing_contexts += 1

    if missing_questions:
        raise ValueError(f"{missing_questions} records do not contain a question.")

    if missing_contexts:
        raise ValueError(f"{missing_contexts} records do not contain a context.")

    if missing_answers:
        print(
            f"Warning: {missing_answers} records have "
            "empty predicted answers. They will receive "
            "EMPTY_ANSWER labels."
        )


def verify_predictions(
    input_path: str | Path,
    output_path: str | Path,
    qa2d_model_name: str,
    nli_model_name: str,
    batch_size: int,
    max_context_tokens: int,
    max_claim_tokens: int,
    generation_max_new_tokens: int,
    entailment_threshold: float,
    contradiction_threshold: float,
) -> list[dict[str, Any]]:
    """
    Run question-aware claim generation and NLI
    verification over all predictions.
    """

    records = load_jsonl(input_path)

    validate_records(records)

    if not (0.0 <= entailment_threshold <= 1.0):
        raise ValueError("Entailment threshold must be between 0 and 1.")

    if not (0.0 <= contradiction_threshold <= 1.0):
        raise ValueError("Contradiction threshold must be between 0 and 1.")

    device = select_device()

    print(f"Using device: {device}")

    claim_converter = QuestionToClaimConverter(
        model_name=qa2d_model_name,
        device=device,
        max_input_tokens=256,
        max_new_tokens=(generation_max_new_tokens),
    )

    nli_verifier = QuestionAwareNLIVerifier(
        model_name=nli_model_name,
        device=device,
        max_context_tokens=(max_context_tokens),
        max_claim_tokens=(max_claim_tokens),
        entailment_threshold=(entailment_threshold),
        contradiction_threshold=(contradiction_threshold),
    )

    verified_records: list[dict[str, Any]] = []

    total_records = len(records)

    for batch_number, (
        start_index,
        end_index,
    ) in enumerate(
        batched_indices(
            total_size=total_records,
            batch_size=batch_size,
        ),
        start=1,
    ):
        batch_records = records[start_index:end_index]

        questions = [get_question(record) for record in batch_records]

        answers = [get_predicted_answer(record) for record in batch_records]

        contexts = [get_context(record) for record in batch_records]

        claims: list[str] = ["" for _ in batch_records]

        non_empty_indices = [index for index, answer in enumerate(answers) if answer]

        if non_empty_indices:
            generated_claims = claim_converter.convert_batch(
                questions=[questions[index] for index in non_empty_indices],
                answers=[answers[index] for index in non_empty_indices],
            )

            for local_index, claim in zip(
                non_empty_indices,
                generated_claims,
            ):
                claims[local_index] = claim

        nli_results: list[dict[str, Any]] = [
            {
                "qa_nli_label": ("EMPTY_ANSWER"),
                "qa_nli_confidence": 1.0,
                "qa_entailment_probability": 0.0,
                "qa_neutral_probability": 1.0,
                "qa_contradiction_probability": 0.0,
            }
            for _ in batch_records
        ]

        if non_empty_indices:
            verified_non_empty = nli_verifier.verify_batch(
                contexts=[contexts[index] for index in non_empty_indices],
                claims=[claims[index] for index in non_empty_indices],
            )

            for local_index, result in zip(
                non_empty_indices,
                verified_non_empty,
            ):
                nli_results[local_index] = result

        for (
            record,
            question,
            answer,
            claim,
            result,
        ) in zip(
            batch_records,
            questions,
            answers,
            claims,
            nli_results,
        ):
            updated_record = dict(record)

            updated_record.update(
                {
                    "qa_claim": (claim if claim else None),
                    "qa_claim_question": (question),
                    "qa_claim_answer": (answer),
                    **result,
                    "qa_nli_model": (nli_model_name),
                    "qa_claim_generator_model": (qa2d_model_name),
                }
            )

            verified_records.append(updated_record)

        print(f"Processed {end_index}/{total_records} records (batch {batch_number}).")

    save_jsonl(
        records=verified_records,
        path=output_path,
    )

    label_counts = Counter(
        str(
            record.get(
                "qa_nli_label",
                "UNKNOWN",
            )
        )
        for record in verified_records
    )

    print("\nQuestion-aware semantic verification completed.")

    print(f"Input:  {input_path}")

    print(f"Output: {output_path}")

    print("\nLabel distribution:")

    for label in (
        "ENTAILMENT",
        "NEUTRAL",
        "CONTRADICTION",
        "EMPTY_ANSWER",
    ):
        print(f"{label:<16}: {label_counts.get(label, 0)}")

    return verified_records


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Generate question-answer claims and verify them against context using NLI."
        )
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
        "--qa2d-model",
        default=DEFAULT_QA2D_MODEL,
    )

    parser.add_argument(
        "--nli-model",
        default=DEFAULT_NLI_MODEL,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=(DEFAULT_MAX_CONTEXT_TOKENS),
    )

    parser.add_argument(
        "--max-claim-tokens",
        type=int,
        default=(DEFAULT_MAX_CLAIM_TOKENS),
    )

    parser.add_argument(
        "--generation-max-new-tokens",
        type=int,
        default=(DEFAULT_GENERATION_MAX_NEW_TOKENS),
    )

    parser.add_argument(
        "--entailment-threshold",
        type=float,
        default=(DEFAULT_ENTAILMENT_THRESHOLD),
    )

    parser.add_argument(
        "--contradiction-threshold",
        type=float,
        default=(DEFAULT_CONTRADICTION_THRESHOLD),
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    verify_predictions(
        input_path=arguments.input,
        output_path=arguments.output,
        qa2d_model_name=(arguments.qa2d_model),
        nli_model_name=(arguments.nli_model),
        batch_size=arguments.batch_size,
        max_context_tokens=(arguments.max_context_tokens),
        max_claim_tokens=(arguments.max_claim_tokens),
        generation_max_new_tokens=(arguments.generation_max_new_tokens),
        entailment_threshold=(arguments.entailment_threshold),
        contradiction_threshold=(arguments.contradiction_threshold),
    )
