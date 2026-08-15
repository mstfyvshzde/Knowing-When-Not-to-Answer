"""
Question-aware semantic verifier V2.

This verifier converts a question-answer pair into a declarative claim,
validates the generated claim using dataset-independent structural checks,
and verifies only valid claims against the source context using Natural
Language Inference (NLI).

Pipeline
--------
1. Convert question + predicted answer into a declarative claim.
2. Validate whether the claim reliably preserves the QA information.
3. Mark unreliable generated claims as INVALID_CLAIM.
4. Run NLI only for valid claims.
5. Store claim-quality diagnostics and NLI probabilities.

The NLI model treats:

    context -> premise
    generated QA claim -> hypothesis

and produces probabilities for:

- ENTAILMENT
- NEUTRAL
- CONTRADICTION

Important
---------
- No dataset-specific examples are used in the QA-to-claim input.
- Gold/reference answers and correctness labels are never used.
- Invalid claims are not passed to the NLI model.
- NLI probabilities are model classification probabilities, not calibrated
  probabilities that the original QA prediction is correct.
- Claim generation can itself introduce errors, so claim validity is checked
  before semantic verification.
"""

from __future__ import annotations

import argparse
import re
import string
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

from src.utils.io import load_jsonl, save_jsonl

DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/calibration_with_hybrid_evidence.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "outputs/predictions/"
    "calibration_with_question_aware_semantic_evidence_v2.jsonl"
)

# Specialized QA-to-declarative-statement model.
DEFAULT_QA2D_MODEL = "domenicrosati/QA2D-t5-base"

# NLI model used to compare generated claims with their source contexts.
DEFAULT_NLI_MODEL = "FacebookAI/roberta-large-mnli"

DEFAULT_BATCH_SIZE = 4

DEFAULT_QA2D_MAX_INPUT_TOKENS = 256
DEFAULT_GENERATION_MAX_NEW_TOKENS = 80

DEFAULT_MAX_CONTEXT_TOKENS = 384
DEFAULT_MAX_CLAIM_TOKENS = 96
NLI_MAX_TOTAL_TOKENS = 512

DEFAULT_ENTAILMENT_THRESHOLD = 0.50
DEFAULT_CONTRADICTION_THRESHOLD = 0.50

# Claim-quality requirements used before NLI inference.
DEFAULT_MIN_ANSWER_TOKEN_COVERAGE = 0.80
DEFAULT_MIN_CLAIM_TOKEN_COUNT = 4
DEFAULT_MAX_QUESTION_COPY_RATIO = 0.98


# Common prefixes suggesting that generated text is still phrased as a
# question rather than as the declarative claim required by NLI.
QUESTION_PREFIXES = (
    "what ",
    "who ",
    "when ",
    "where ",
    "why ",
    "how ",
    "which ",
    "whose ",
    "whom ",
    "is ",
    "are ",
    "was ",
    "were ",
    "do ",
    "does ",
    "did ",
    "can ",
    "could ",
    "will ",
    "would ",
    "should ",
    "has ",
    "have ",
    "had ",
)

# Common low-information words ignored by some lexical claim-quality checks.
STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "for",
    "from",
    "with",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "has",
    "have",
    "had",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "their",
    "his",
    "her",
    "them",
}


def get_first_value(
    record: dict[str, Any],
    field_names: tuple[str, ...],
    default: Any = None,
) -> Any:
    """
    Return the first available non-None value from a group of field aliases.

    Earlier pipeline stages may store equivalent information under slightly
    different field names, so aliases are checked from left to right.
    """

    for field_name in field_names:
        value = record.get(field_name)

        if value is not None:
            return value

    return default


def clean_text(text: str) -> str:
    """
    Normalize whitespace without changing the textual content.

    Repeated spaces, tabs, and line breaks are collapsed into one space.
    """

    return re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip()


def get_question(record: dict[str, Any]) -> str:
    """Extract and clean the question from one prediction record."""

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


def get_predicted_answer(record: dict[str, Any]) -> str:
    """Extract and clean the predicted QA answer from one record."""

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


def get_context(record: dict[str, Any]) -> str:
    """Extract and clean the evidence context from one prediction record."""

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


def normalize_for_comparison(text: str) -> str:
    """
    Normalize text for lexical claim-quality comparisons.

    Text is lowercased, punctuation is replaced with spaces, and repeated
    whitespace is collapsed.
    """

    normalized_text = clean_text(text).lower()

    normalized_text = "".join(
        character if character not in string.punctuation else " "
        for character in normalized_text
    )

    return " ".join(normalized_text.split())


def tokenize_text(
    text: str,
    remove_stopwords: bool = False,
) -> list[str]:
    """
    Convert normalized text into lexical tokens.

    Stop words can optionally be removed when the comparison should emphasize
    content-bearing words rather than common grammatical words.
    """

    tokens = normalize_for_comparison(text).split()

    if remove_stopwords:
        tokens = [
            token
            for token in tokens
            if token not in STOPWORDS
        ]

    return tokens


def normalise_generated_claim(claim: str) -> str:
    """
    Clean the claim produced by the QA2D generation model.

    Common generation prefixes and surrounding quotation marks are removed.
    Terminal punctuation is added when necessary.
    """

    cleaned_claim = clean_text(claim)

    prefixes = (
        "statement:",
        "claim:",
        "declarative statement:",
        "declarative claim:",
        "final statement:",
        "answer:",
    )

    lowered_claim = cleaned_claim.lower()

    for prefix in prefixes:
        if lowered_claim.startswith(prefix):
            cleaned_claim = cleaned_claim[
                len(prefix) :
            ].strip()
            break

    cleaned_claim = cleaned_claim.strip("\"' ")

    if (
        cleaned_claim
        and cleaned_claim[-1] not in ".!?"
    ):
        cleaned_claim += "."

    return cleaned_claim


def normalize_qa2d_input(text: str) -> str:
    """
    Normalize text according to the QA2D model's expected input format.

    Input is lowercased and punctuation is removed before the question and
    answer are combined into the model-specific `question. answer` format.
    """

    normalized_text = clean_text(text).lower()

    normalized_text = normalized_text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation,
        )
    )

    return clean_text(normalized_text)


def calculate_token_coverage(
    source_text: str,
    target_text: str,
) -> float:
    """
    Measure how much source information is preserved in a target text.

    The score is the proportion of source tokens that also appear in the
    target. Content words are preferred; if no content words remain, the
    comparison falls back to all normalized tokens.

    This is used mainly to check whether the generated claim preserved the
    predicted answer.
    """

    source_tokens = tokenize_text(
        source_text,
        remove_stopwords=True,
    )

    target_tokens = tokenize_text(
        target_text,
        remove_stopwords=True,
    )

    if not source_tokens:
        source_tokens = tokenize_text(
            source_text,
            remove_stopwords=False,
        )

        target_tokens = tokenize_text(
            target_text,
            remove_stopwords=False,
        )

    if not source_tokens:
        return 1.0

    source_counter = Counter(source_tokens)
    target_counter = Counter(target_tokens)

    matched_count = sum(
        (
            source_counter
            & target_counter
        ).values()
    )

    total_source_count = sum(
        source_counter.values()
    )

    return matched_count / total_source_count


def calculate_question_copy_ratio(
    question: str,
    claim: str,
) -> float:
    """
    Measure how much of the original question is copied into the claim.

    Very high copying can indicate that the QA2D model failed to transform
    the interrogative question into a proper declarative statement.
    """

    question_tokens = tokenize_text(
        question,
        remove_stopwords=False,
    )

    claim_tokens = tokenize_text(
        claim,
        remove_stopwords=False,
    )

    if not question_tokens:
        return 0.0

    question_counter = Counter(
        question_tokens
    )

    claim_counter = Counter(
        claim_tokens
    )

    shared_count = sum(
        (
            question_counter
            & claim_counter
        ).values()
    )

    return (
        shared_count
        / sum(question_counter.values())
    )


def looks_like_question(claim: str) -> bool:
    """
    Detect whether generated text still appears to be a question.

    A trailing question mark or a common interrogative/auxiliary prefix is
    treated as evidence that QA-to-declarative conversion may have failed.
    """

    cleaned_claim = clean_text(claim)
    lowered_claim = cleaned_claim.lower()

    if cleaned_claim.endswith("?"):
        return True

    return any(
        lowered_claim.startswith(prefix)
        for prefix in QUESTION_PREFIXES
    )


def is_answer_only_fragment(
    claim: str,
    answer: str,
) -> bool:
    """
    Detect a generation that contains only the answer.

    An answer-only fragment is not a self-contained declarative statement and
    therefore is not considered reliable input for the NLI verifier.
    """

    normalized_claim = normalize_for_comparison(
        claim
    )

    normalized_answer = normalize_for_comparison(
        answer
    )

    if not normalized_answer:
        return False

    return normalized_claim == normalized_answer


def is_question_repetition(
    question: str,
    claim: str,
) -> bool:
    """
    Detect an unchanged question returned as the generated claim.

    This identifies a clear failure of the QA-to-declarative transformation.
    """

    normalized_question = normalize_for_comparison(
        question
    )

    normalized_claim = normalize_for_comparison(
        claim
    )

    if not normalized_question:
        return False

    return normalized_question == normalized_claim


def extract_numbers(text: str) -> list[str]:
    """
    Extract simple numeric expressions used by the preservation check.

    Examples include integer/decimal values and optional percentage symbols.
    """

    return re.findall(
        r"\b\d+(?:[.,]\d+)?%?\b",
        text,
    )


def validate_number_preservation(
    question: str,
    answer: str,
    claim: str,
) -> bool:
    """
    Check whether numeric values from the predicted answer survive generation.

    Numbers that occur only in the question are not required to be duplicated.
    Gold/reference answers are not used.
    """

    del question

    answer_numbers = Counter(
        extract_numbers(answer)
    )

    if not answer_numbers:
        return True

    claim_numbers = Counter(
        extract_numbers(claim)
    )

    return all(
        claim_numbers[number] >= count
        for number, count
        in answer_numbers.items()
    )


def validate_negation_preservation(
    question: str,
    answer: str,
    claim: str,
) -> bool:
    """
    Check whether explicit negation survives QA-to-claim generation.

    This is intentionally a conservative lexical diagnostic rather than
    semantic reasoning. If the source QA pair contains explicit negation,
    the generated claim must also contain an explicit negation signal.
    """

    negation_terms = {
        "not",
        "no",
        "never",
        "neither",
        "nor",
        "without",
        "cannot",
        "can't",
        "didn't",
        "doesn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "won't",
        "wouldn't",
        "shouldn't",
        "couldn't",
    }

    source_tokens = set(
        tokenize_text(
            f"{question} {answer}",
            remove_stopwords=False,
        )
    )

    claim_tokens = set(
        tokenize_text(
            claim,
            remove_stopwords=False,
        )
    )

    source_negations = (
        source_tokens
        & negation_terms
    )

    if not source_negations:
        return True

    claim_negations = (
        claim_tokens
        & negation_terms
    )

    return bool(claim_negations)


def validate_generated_claim(
    question: str,
    answer: str,
    claim: str,
    min_answer_token_coverage: float,
    min_claim_token_count: int,
    max_question_copy_ratio: float,
) -> dict[str, Any]:
    """
    Validate generated claim quality using dataset-independent rules.

    The checks detect structural or information-preservation failures before
    the claim reaches the NLI model.

    Validation considers:

    - empty claims
    - outputs that still look like questions
    - answer-only fragments
    - repeated questions
    - claims that are too short
    - loss of answer tokens
    - loss of numeric information
    - loss of explicit negation
    - excessive copying of a question that remains question-like

    Gold answers and prediction-correctness labels are never used.
    """

    reasons: list[str] = []

    cleaned_claim = clean_text(claim)

    claim_tokens = tokenize_text(
        cleaned_claim,
        remove_stopwords=False,
    )

    claim_content_tokens = tokenize_text(
        cleaned_claim,
        remove_stopwords=True,
    )

    answer_token_coverage = (
        calculate_token_coverage(
            source_text=answer,
            target_text=cleaned_claim,
        )
    )

    question_copy_ratio = (
        calculate_question_copy_ratio(
            question=question,
            claim=cleaned_claim,
        )
    )

    number_preserved = (
        validate_number_preservation(
            question=question,
            answer=answer,
            claim=cleaned_claim,
        )
    )

    negation_preserved = (
        validate_negation_preservation(
            question=question,
            answer=answer,
            claim=cleaned_claim,
        )
    )

    if not cleaned_claim:
        reasons.append("EMPTY_CLAIM")

    if looks_like_question(cleaned_claim):
        reasons.append("QUESTION_FORM")

    if is_answer_only_fragment(
        claim=cleaned_claim,
        answer=answer,
    ):
        reasons.append(
            "ANSWER_ONLY_FRAGMENT"
        )

    if is_question_repetition(
        question=question,
        claim=cleaned_claim,
    ):
        reasons.append(
            "QUESTION_REPEATED"
        )

    if (
        len(claim_tokens)
        < min_claim_token_count
    ):
        reasons.append("TOO_SHORT")

    if (
        answer_token_coverage
        < min_answer_token_coverage
    ):
        reasons.append(
            "ANSWER_NOT_PRESERVED"
        )

    if not number_preserved:
        reasons.append(
            "NUMBER_NOT_PRESERVED"
        )

    if not negation_preserved:
        reasons.append(
            "NEGATION_NOT_PRESERVED"
        )

    if (
        question_copy_ratio
        >= max_question_copy_ratio
        and looks_like_question(
            cleaned_claim
        )
    ):
        reasons.append(
            "EXCESSIVE_QUESTION_COPY"
        )

    return {
        "qa_claim_valid": len(reasons) == 0,
        "qa_claim_validation_reasons": reasons,
        "qa_claim_answer_token_coverage": (
            answer_token_coverage
        ),
        "qa_claim_question_copy_ratio": (
            question_copy_ratio
        ),
        "qa_claim_token_count": (
            len(claim_tokens)
        ),
        "qa_claim_content_token_count": (
            len(claim_content_tokens)
        ),
        "qa_claim_number_preserved": (
            number_preserved
        ),
        "qa_claim_negation_preserved": (
            negation_preserved
        ),
    }


def select_device() -> torch.device:
    """
    Select the best available inference hardware.

    CUDA is preferred when available, followed by Apple's MPS backend.
    CPU is used otherwise.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def build_qa2d_prompt(
    question: str,
    answer: str,
) -> str:
    """
    Build the model-specific input for the QA2D converter.

    The specialized QA2D model expects a compact:

        question. answer

    representation rather than the instruction-style prompt used by V1.
    """

    normalized_question = normalize_qa2d_input(
        question
    )

    normalized_answer = normalize_qa2d_input(
        answer
    )

    return (
        f"{normalized_question}. "
        f"{normalized_answer}"
    )


class QuestionToClaimConverter:
    """
    Convert QA pairs into declarative claims using a specialized QA2D model.

    The pretrained sequence-to-sequence model is used only for inference.
    Its model weights are not trained or fine-tuned by this pipeline.
    """

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        max_input_tokens: int,
        max_new_tokens: int,
    ) -> None:
        if max_input_tokens <= 0:
            raise ValueError(
                "max_input_tokens must be greater than zero."
            )

        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than zero."
            )

        self.device = device
        self.max_input_tokens = (
            max_input_tokens
        )
        self.max_new_tokens = (
            max_new_tokens
        )

        print(
            f"Loading QA-to-claim model: "
            f"{model_name}"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                model_name
            )
        )

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def convert_batch(
        self,
        questions: list[str],
        answers: list[str],
    ) -> list[str]:
        """
        Generate declarative claims for one batch of QA pairs.

        Beam search is deterministic because sampling is disabled.
        """

        if len(questions) != len(answers):
            raise ValueError(
                "Question and answer batch sizes must match."
            )

        if not questions:
            return []

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
            key: value.to(self.device)
            for key, value
            in encoded_inputs.items()
        }

        generated_ids = self.model.generate(
            **encoded_inputs,
            max_new_tokens=self.max_new_tokens,
            num_beams=4,
            do_sample=False,
            early_stopping=True,
        )

        generated_claims = (
            self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        )

        return [
            normalise_generated_claim(claim)
            for claim in generated_claims
        ]


class QuestionAwareNLIVerifier:
    """
    Verify generated QA claims against contexts using MNLI.

    The context is the premise and the generated declarative claim is the
    hypothesis.

    The verifier returns entailment, neutral, and contradiction probabilities
    together with a categorical NLI label.

    These probabilities describe the NLI classifier's belief over its own
    classes; they are not calibrated probabilities that the QA answer is correct.
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
        if max_context_tokens <= 0:
            raise ValueError(
                "max_context_tokens must be greater than zero."
            )

        if max_claim_tokens <= 0:
            raise ValueError(
                "max_claim_tokens must be greater than zero."
            )

        if not (
            0.0
            <= entailment_threshold
            <= 1.0
        ):
            raise ValueError(
                "Entailment threshold must be between 0 and 1."
            )

        if not (
            0.0
            <= contradiction_threshold
            <= 1.0
        ):
            raise ValueError(
                "Contradiction threshold must be between 0 and 1."
            )

        self.device = device
        self.max_context_tokens = (
            max_context_tokens
        )
        self.max_claim_tokens = (
            max_claim_tokens
        )
        self.entailment_threshold = (
            entailment_threshold
        )
        self.contradiction_threshold = (
            contradiction_threshold
        )

        print(
            f"Loading NLI model: "
            f"{model_name}"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name
            )
        )

        self.model.to(self.device)
        self.model.eval()

        self.label_to_id = (
            self._resolve_label_ids()
        )

    def _resolve_label_ids(
        self,
    ) -> dict[str, int]:
        """
        Resolve semantic NLI labels to model-specific class IDs.

        The implementation reads the model configuration instead of assuming
        a fixed label order.
        """

        resolved_labels: dict[str, int] = {}

        for (
            raw_id,
            raw_label,
        ) in self.model.config.id2label.items():
            label = (
                str(raw_label)
                .strip()
                .lower()
            )

            label_id = int(raw_id)

            if "entail" in label:
                resolved_labels[
                    "entailment"
                ] = label_id

            elif "neutral" in label:
                resolved_labels[
                    "neutral"
                ] = label_id

            elif "contrad" in label:
                resolved_labels[
                    "contradiction"
                ] = label_id

        required_labels = {
            "entailment",
            "neutral",
            "contradiction",
        }

        missing_labels = (
            required_labels
            - set(resolved_labels)
        )

        if missing_labels:
            raise ValueError(
                "Could not resolve NLI labels. "
                f"Missing: {sorted(missing_labels)}. "
                "Available labels: "
                f"{self.model.config.id2label}"
            )

        return resolved_labels

    def _truncate_text(
        self,
        text: str,
        max_tokens: int,
    ) -> str:
        """
        Truncate one text using the NLI model's own tokenizer.

        This keeps the context/claim pair within the model's input limits.
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
        Convert NLI probabilities into a semantic evidence label.

        Contradiction and entailment thresholds receive priority. If neither
        threshold is reached, the class with the largest probability is used.
        """

        if (
            contradiction_probability
            >= self.contradiction_threshold
        ):
            return "CONTRADICTION"

        if (
            entailment_probability
            >= self.entailment_threshold
        ):
            return "ENTAILMENT"

        probabilities = {
            "ENTAILMENT": (
                entailment_probability
            ),
            "NEUTRAL": (
                neutral_probability
            ),
            "CONTRADICTION": (
                contradiction_probability
            ),
        }

        return max(
            probabilities,
            key=probabilities.get,
        )

    @torch.inference_mode()
    def verify_batch(
        self,
        contexts: list[str],
        claims: list[str],
    ) -> list[dict[str, Any]]:
        """
        Run NLI inference for a batch of valid context-claim pairs.

        Invalid claims are filtered before this function is called.
        """

        if len(contexts) != len(claims):
            raise ValueError(
                "Context and claim batch sizes must match."
            )

        if not contexts:
            return []

        truncated_contexts = [
            self._truncate_text(
                text=context,
                max_tokens=self.max_context_tokens,
            )
            for context in contexts
        ]

        truncated_claims = [
            self._truncate_text(
                text=claim,
                max_tokens=self.max_claim_tokens,
            )
            for claim in claims
        ]

        encoded_inputs = self.tokenizer(
            truncated_contexts,
            truncated_claims,
            padding=True,
            truncation="only_first",
            max_length=NLI_MAX_TOTAL_TOKENS,
            return_tensors="pt",
        )

        encoded_inputs = {
            key: value.to(self.device)
            for key, value
            in encoded_inputs.items()
        }

        outputs = self.model(
            **encoded_inputs
        )

        probabilities = (
            torch.softmax(
                outputs.logits,
                dim=-1,
            )
            .detach()
            .cpu()
        )

        entailment_id = self.label_to_id[
            "entailment"
        ]

        neutral_id = self.label_to_id[
            "neutral"
        ]

        contradiction_id = self.label_to_id[
            "contradiction"
        ]

        results: list[
            dict[str, Any]
        ] = []

        for row in probabilities:
            entailment_probability = float(
                row[entailment_id].item()
            )

            neutral_probability = float(
                row[neutral_id].item()
            )

            contradiction_probability = float(
                row[
                    contradiction_id
                ].item()
            )

            label = self._assign_label(
                entailment_probability=(
                    entailment_probability
                ),
                neutral_probability=(
                    neutral_probability
                ),
                contradiction_probability=(
                    contradiction_probability
                ),
            )

            confidence = max(
                entailment_probability,
                neutral_probability,
                contradiction_probability,
            )

            results.append(
                {
                    "qa_nli_label": label,
                    "qa_nli_confidence": confidence,
                    "qa_entailment_probability": (
                        entailment_probability
                    ),
                    "qa_neutral_probability": (
                        neutral_probability
                    ),
                    "qa_contradiction_probability": (
                        contradiction_probability
                    ),
                }
            )

        return results


def batched_indices(
    total_size: int,
    batch_size: int,
) -> Iterable[tuple[int, int]]:
    """
    Yield deterministic start/end ranges for mini-batch processing.
    """

    if total_size < 0:
        raise ValueError(
            "total_size cannot be negative."
        )

    if batch_size <= 0:
        raise ValueError(
            "Batch size must be positive."
        )

    for start_index in range(
        0,
        total_size,
        batch_size,
    ):
        yield (
            start_index,
            min(
                start_index + batch_size,
                total_size,
            ),
        )


def validate_records(
    records: list[dict[str, Any]],
) -> None:
    """
    Validate mandatory input fields before claim generation.

    Question and context are required. Empty predicted answers are allowed and
    will be assigned EMPTY_ANSWER without entering claim generation or NLI.
    """

    if not records:
        raise ValueError(
            "Input file contains no records."
        )

    missing_questions = 0
    missing_contexts = 0

    for record in records:
        if not get_question(record):
            missing_questions += 1

        if not get_context(record):
            missing_contexts += 1

    if missing_questions:
        raise ValueError(
            f"{missing_questions} records "
            "do not contain a question."
        )

    if missing_contexts:
        raise ValueError(
            f"{missing_contexts} records "
            "do not contain a context."
        )


def create_invalid_result(
    label: str,
) -> dict[str, Any]:
    """
    Create the semantic result for a record that does not enter NLI.

    INVALID_CLAIM and EMPTY_ANSWER are bookkeeping states, not NLI classes.
    Their NLI probabilities are deliberately stored as zero because no NLI
    inference was performed.
    """

    if label not in {
        "INVALID_CLAIM",
        "EMPTY_ANSWER",
    }:
        raise ValueError(
            f"Unsupported non-NLI label: {label}"
        )

    return {
        "qa_nli_label": label,
        "qa_nli_confidence": 0.0,
        "qa_entailment_probability": 0.0,
        "qa_neutral_probability": 0.0,
        "qa_contradiction_probability": 0.0,
    }


def validate_runtime_settings(
    batch_size: int,
    qa2d_max_input_tokens: int,
    generation_max_new_tokens: int,
    max_context_tokens: int,
    max_claim_tokens: int,
    entailment_threshold: float,
    contradiction_threshold: float,
    min_answer_token_coverage: float,
    min_claim_token_count: int,
    max_question_copy_ratio: float,
) -> None:
    """
    Validate runtime and claim-quality configuration before model inference.

    These checks reject malformed experiment settings without changing the
    configured V2 verification policy.
    """

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    if qa2d_max_input_tokens <= 0:
        raise ValueError(
            "qa2d_max_input_tokens must be greater than zero."
        )

    if generation_max_new_tokens <= 0:
        raise ValueError(
            "generation_max_new_tokens must be greater than zero."
        )

    if max_context_tokens <= 0:
        raise ValueError(
            "max_context_tokens must be greater than zero."
        )

    if max_claim_tokens <= 0:
        raise ValueError(
            "max_claim_tokens must be greater than zero."
        )

    if not (
        0.0
        <= entailment_threshold
        <= 1.0
    ):
        raise ValueError(
            "entailment_threshold must be between 0 and 1."
        )

    if not (
        0.0
        <= contradiction_threshold
        <= 1.0
    ):
        raise ValueError(
            "contradiction_threshold must be between 0 and 1."
        )

    if not (
        0.0
        <= min_answer_token_coverage
        <= 1.0
    ):
        raise ValueError(
            "min_answer_token_coverage must be between 0 and 1."
        )

    if min_claim_token_count <= 0:
        raise ValueError(
            "min_claim_token_count must be greater than zero."
        )

    if not (
        0.0
        <= max_question_copy_ratio
        <= 1.0
    ):
        raise ValueError(
            "max_question_copy_ratio must be between 0 and 1."
        )


def verify_predictions(
    input_path: str | Path,
    output_path: str | Path,
    qa2d_model_name: str,
    nli_model_name: str,
    batch_size: int,
    qa2d_max_input_tokens: int,
    generation_max_new_tokens: int,
    max_context_tokens: int,
    max_claim_tokens: int,
    entailment_threshold: float,
    contradiction_threshold: float,
    min_answer_token_coverage: float,
    min_claim_token_count: int,
    max_question_copy_ratio: float,
) -> list[dict[str, Any]]:
    """
    Run the complete question-aware semantic verification V2 pipeline.

    Predictions are processed in batches. Non-empty answers are converted into
    declarative claims and validated using dataset-independent checks.

    Only valid claims are sent to the NLI model. Invalid claims and empty answers
    receive explicit non-NLI labels.

    Original prediction fields are preserved and enriched with claim-quality,
    semantic-verification, model, and version metadata.
    """

    records = load_jsonl(
        input_path
    )

    validate_records(
        records
    )

    validate_runtime_settings(
        batch_size=batch_size,
        qa2d_max_input_tokens=(
            qa2d_max_input_tokens
        ),
        generation_max_new_tokens=(
            generation_max_new_tokens
        ),
        max_context_tokens=(
            max_context_tokens
        ),
        max_claim_tokens=(
            max_claim_tokens
        ),
        entailment_threshold=(
            entailment_threshold
        ),
        contradiction_threshold=(
            contradiction_threshold
        ),
        min_answer_token_coverage=(
            min_answer_token_coverage
        ),
        min_claim_token_count=(
            min_claim_token_count
        ),
        max_question_copy_ratio=(
            max_question_copy_ratio
        ),
    )

    device = select_device()

    print(
        f"Using device: {device}"
    )

    claim_converter = (
        QuestionToClaimConverter(
            model_name=qa2d_model_name,
            device=device,
            max_input_tokens=(
                qa2d_max_input_tokens
            ),
            max_new_tokens=(
                generation_max_new_tokens
            ),
        )
    )

    nli_verifier = (
        QuestionAwareNLIVerifier(
            model_name=nli_model_name,
            device=device,
            max_context_tokens=(
                max_context_tokens
            ),
            max_claim_tokens=(
                max_claim_tokens
            ),
            entailment_threshold=(
                entailment_threshold
            ),
            contradiction_threshold=(
                contradiction_threshold
            ),
        )
    )

    verified_records: list[
        dict[str, Any]
    ] = []

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
        batch_records = records[
            start_index:end_index
        ]

        questions = [
            get_question(record)
            for record in batch_records
        ]

        answers = [
            get_predicted_answer(record)
            for record in batch_records
        ]

        contexts = [
            get_context(record)
            for record in batch_records
        ]

        claims: list[
            str | None
        ] = [
            None
            for _ in batch_records
        ]

        # Empty answers cannot form a meaningful QA claim. These default
        # validation records are replaced below for non-empty answers.
        validations: list[
            dict[str, Any]
        ] = [
            {
                "qa_claim_valid": False,
                "qa_claim_validation_reasons": [
                    "EMPTY_ANSWER"
                ],
                "qa_claim_answer_token_coverage": 0.0,
                "qa_claim_question_copy_ratio": 0.0,
                "qa_claim_token_count": 0,
                "qa_claim_content_token_count": 0,
                "qa_claim_number_preserved": True,
                "qa_claim_negation_preserved": True,
            }
            for _ in batch_records
        ]

        non_empty_answer_indices = [
            index
            for index, answer
            in enumerate(answers)
            if answer
        ]

        if non_empty_answer_indices:
            generated_claims = (
                claim_converter.convert_batch(
                    questions=[
                        questions[index]
                        for index
                        in non_empty_answer_indices
                    ],
                    answers=[
                        answers[index]
                        for index
                        in non_empty_answer_indices
                    ],
                )
            )

            for (
                local_index,
                claim,
            ) in zip(
                non_empty_answer_indices,
                generated_claims,
            ):
                claims[
                    local_index
                ] = claim

                validations[
                    local_index
                ] = validate_generated_claim(
                    question=questions[
                        local_index
                    ],
                    answer=answers[
                        local_index
                    ],
                    claim=claim,
                    min_answer_token_coverage=(
                        min_answer_token_coverage
                    ),
                    min_claim_token_count=(
                        min_claim_token_count
                    ),
                    max_question_copy_ratio=(
                        max_question_copy_ratio
                    ),
                )

        valid_claim_indices = [
            index
            for index, validation
            in enumerate(validations)
            if validation[
                "qa_claim_valid"
            ]
        ]

        # INVALID_CLAIM and EMPTY_ANSWER records bypass NLI. In particular,
        # INVALID_CLAIM therefore receives zero entailment probability rather
        # than an unreliable semantic score from a malformed generated claim.
        nli_results = [
            create_invalid_result(

                    "EMPTY_ANSWER"
                    if not answers[index]
                    else "INVALID_CLAIM"

            )
            for index
            in range(len(batch_records))
        ]

        if valid_claim_indices:
            valid_nli_results = (
                nli_verifier.verify_batch(
                    contexts=[
                        contexts[index]
                        for index
                        in valid_claim_indices
                    ],
                    claims=[
                        str(claims[index])
                        for index
                        in valid_claim_indices
                    ],
                )
            )

            for (
                local_index,
                result,
            ) in zip(
                valid_claim_indices,
                valid_nli_results,
            ):
                nli_results[
                    local_index
                ] = result

        for (
            record,
            question,
            answer,
            claim,
            validation,
            nli_result,
        ) in zip(
            batch_records,
            questions,
            answers,
            claims,
            validations,
            nli_results,
        ):
            updated_record = dict(
                record
            )

            updated_record.update(
                {
                    "qa_claim": claim,
                    "qa_claim_question": question,
                    "qa_claim_answer": answer,
                    **validation,
                    **nli_result,
                    "qa_claim_generator_model": (
                        qa2d_model_name
                    ),
                    "qa_nli_model": (
                        nli_model_name
                    ),
                    "qa_verifier_version": (
                        "question_aware_v2"
                    ),
                }
            )

            verified_records.append(
                updated_record
            )

        print(
            f"Processed "
            f"{end_index}/{total_records} "
            f"records "
            f"(batch {batch_number})."
        )

    save_jsonl(
        verified_records,
        output_path,
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

    valid_claim_count = sum(
        bool(
            record.get(
                "qa_claim_valid",
                False,
            )
        )
        for record in verified_records
    )

    invalid_reason_counts: Counter[
        str
    ] = Counter()

    for record in verified_records:
        for reason in record.get(
            "qa_claim_validation_reasons",
            [],
        ):
            invalid_reason_counts[
                str(reason)
            ] += 1

    print(
        "\nQuestion-aware semantic "
        "verification V2 completed."
    )

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    print(
        f"\nValid claims:   "
        f"{valid_claim_count}"
    )

    print(
        f"Invalid claims: "
        f"{total_records - valid_claim_count}"
    )

    print("\nLabel distribution:")

    for label in (
        "ENTAILMENT",
        "NEUTRAL",
        "CONTRADICTION",
        "INVALID_CLAIM",
        "EMPTY_ANSWER",
    ):
        print(
            f"{label:<16}: "
            f"{label_counts.get(label, 0)}"
        )

    print(
        "\nInvalid-claim reasons:"
    )

    if invalid_reason_counts:
        for (
            reason,
            count,
        ) in invalid_reason_counts.most_common():
            print(
                f"{reason:<28}: "
                f"{count}"
            )
    else:
        print("None")

    return verified_records


def parse_arguments() -> argparse.Namespace:
    """Parse question-aware V2 verification settings."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate and validate question-aware claims, "
            "then verify valid claims using NLI."
        )
    )

    parser.add_argument(
        "--input",
        default=str(
            DEFAULT_INPUT_PATH
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT_PATH
        ),
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
        "--qa2d-max-input-tokens",
        type=int,
        default=(
            DEFAULT_QA2D_MAX_INPUT_TOKENS
        ),
    )

    parser.add_argument(
        "--generation-max-new-tokens",
        type=int,
        default=(
            DEFAULT_GENERATION_MAX_NEW_TOKENS
        ),
    )

    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=(
            DEFAULT_MAX_CONTEXT_TOKENS
        ),
    )

    parser.add_argument(
        "--max-claim-tokens",
        type=int,
        default=(
            DEFAULT_MAX_CLAIM_TOKENS
        ),
    )

    parser.add_argument(
        "--entailment-threshold",
        type=float,
        default=(
            DEFAULT_ENTAILMENT_THRESHOLD
        ),
    )

    parser.add_argument(
        "--contradiction-threshold",
        type=float,
        default=(
            DEFAULT_CONTRADICTION_THRESHOLD
        ),
    )

    parser.add_argument(
        "--min-answer-token-coverage",
        type=float,
        default=(
            DEFAULT_MIN_ANSWER_TOKEN_COVERAGE
        ),
    )

    parser.add_argument(
        "--min-claim-token-count",
        type=int,
        default=(
            DEFAULT_MIN_CLAIM_TOKEN_COUNT
        ),
    )

    parser.add_argument(
        "--max-question-copy-ratio",
        type=float,
        default=(
            DEFAULT_MAX_QUESTION_COPY_RATIO
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    verify_predictions(
        input_path=arguments.input,
        output_path=arguments.output,
        qa2d_model_name=(
            arguments.qa2d_model
        ),
        nli_model_name=(
            arguments.nli_model
        ),
        batch_size=(
            arguments.batch_size
        ),
        qa2d_max_input_tokens=(
            arguments.qa2d_max_input_tokens
        ),
        generation_max_new_tokens=(
            arguments.generation_max_new_tokens
        ),
        max_context_tokens=(
            arguments.max_context_tokens
        ),
        max_claim_tokens=(
            arguments.max_claim_tokens
        ),
        entailment_threshold=(
            arguments.entailment_threshold
        ),
        contradiction_threshold=(
            arguments.contradiction_threshold
        ),
        min_answer_token_coverage=(
            arguments.min_answer_token_coverage
        ),
        min_claim_token_count=(
            arguments.min_claim_token_count
        ),
        max_question_copy_ratio=(
            arguments.max_question_copy_ratio
        ),
    )
