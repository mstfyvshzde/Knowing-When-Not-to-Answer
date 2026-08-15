"""
Verify QA predictions with a fixed-template semantic NLI heuristic.

For each prediction, this prototype verifier:

1. retrieves the predicted answer,
2. selects local lexical evidence when available, otherwise the full context,
3. converts the question-answer pair into a fixed declarative hypothesis,
4. evaluates the evidence/hypothesis pair with a pretrained MNLI model,
5. stores entailment, neutral, and contradiction probabilities.

The NLI setup is:

    premise:
        local evidence window or source context

    hypothesis:
        The answer to the question "<question>" is "<predicted answer>".

The class with the highest NLI probability becomes the semantic label:

- ENTAILMENT
- NEUTRAL
- CONTRADICTION

Important
---------
This component is a prototype semantic-evidence heuristic.

The NLI probabilities describe the classifier's belief over its own semantic
classes. They must not be interpreted as calibrated probabilities that the
original QA prediction is correct.

The underlying RoBERTa-large-MNLI model is also used by other verification
components in this project, so these signals should not be described as
independent model evidence.
"""

import argparse
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.utils.io import load_jsonl, save_jsonl

DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/calibration_with_evidence.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "outputs/predictions/calibration_with_semantic_evidence.jsonl"
)

DEFAULT_MODEL_NAME = "FacebookAI/roberta-large-mnli"

DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_LENGTH = 512


def clean_text(value: Any) -> str:
    """
    Normalize whitespace while preserving textual meaning.

    Missing values become an empty string.
    """

    if value is None:
        return ""

    return " ".join(
        str(value).split()
    )


def get_first_value(
    prediction: dict[str, Any],
    field_names: tuple[str, ...],
    default: Any = None,
) -> Any:
    """
    Return the first available non-None value from a group of field aliases.

    Different pipeline stages may store equivalent information under slightly
    different field names, so aliases are checked from left to right.
    """

    for field_name in field_names:
        value = prediction.get(field_name)

        if value is not None:
            return value

    return default


def get_question(
    prediction: dict[str, Any],
) -> str:
    """Extract and clean the question from one prediction record."""

    value = get_first_value(
        prediction=prediction,
        field_names=(
            "question",
            "question_text",
            "query",
        ),
        default="",
    )

    return clean_text(value)


def get_predicted_answer(
    prediction: dict[str, Any],
) -> str:
    """Extract and clean the predicted QA answer from one record."""

    value = get_first_value(
        prediction=prediction,
        field_names=(
            "predicted_answer",
            "prediction_text",
            "prediction_answer",
            "answer",
        ),
        default="",
    )

    return clean_text(value)


def get_evidence_text(
    prediction: dict[str, Any],
) -> str:
    """
    Retrieve the premise used for semantic verification.

    The lexical verifier may already have extracted a local `evidence_text`
    window around the predicted answer. That local evidence is preferred.

    If no usable evidence window exists, the verifier falls back to the full
    source context.
    """

    evidence_text = clean_text(
        prediction.get(
            "evidence_text",
            "",
        )
    )

    if evidence_text:
        return evidence_text

    return clean_text(
        prediction.get(
            "context",
            "",
        )
    )


def build_qa_hypothesis(
    question: str,
    predicted_answer: str,
) -> str:
    """
    Build the fixed declarative hypothesis evaluated by the NLI model.

    Unlike the question-aware V2 verifier, this method does not use a
    generative QA-to-declarative model. Every prediction uses the same
    deterministic template.
    """

    question = clean_text(question)
    predicted_answer = clean_text(
        predicted_answer
    )

    if not question or not predicted_answer:
        return ""

    return (
        f'The answer to the question '
        f'"{question}" is "{predicted_answer}".'
    )


def resolve_device() -> torch.device:
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


def normalize_label_name(
    label: str,
) -> str:
    """
    Normalize an NLI model label for consistent comparison.
    """

    return str(label).strip().upper()


def find_label_id(
    id_to_label: dict[int, str],
    target_label: str,
) -> int:
    """
    Resolve the numerical model class ID for one semantic NLI label.

    The model configuration is inspected rather than assuming a fixed class
    ordering.
    """

    target_label = normalize_label_name(
        target_label
    )

    for (
        label_id,
        label_name,
    ) in id_to_label.items():
        normalized_label = (
            normalize_label_name(
                label_name
            )
        )

        if normalized_label == target_label:
            return int(label_id)

    raise ValueError(
        f"Could not find NLI label "
        f"{target_label!r}. "
        f"Available labels: {id_to_label}"
    )


class SemanticEvidenceVerifier:
    """
    Evaluate evidence/hypothesis pairs using a pretrained MNLI classifier.

    The evidence is treated as the premise and the fixed QA statement as the
    hypothesis.

    The verifier records probabilities for entailment, neutral, and
    contradiction, then assigns the highest-probability class as the semantic
    label.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        if max_length <= 0:
            raise ValueError(
                "max_length must be greater than zero."
            )

        self.model_name = model_name
        self.max_length = max_length
        self.device = resolve_device()

        print(
            f"Loading semantic verifier: "
            f"{model_name}"
        )

        print(
            f"Device: {self.device}"
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

        self.model.to(
            self.device
        )

        self.model.eval()

        raw_id_to_label = (
            self.model.config.id2label
        )

        self.id_to_label = {
            int(label_id): str(label_name)
            for label_id, label_name
            in raw_id_to_label.items()
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

    @torch.inference_mode()
    def predict_batch(
        self,
        premises: list[str],
        hypotheses: list[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> list[dict[str, Any]]:
        """
        Run semantic NLI inference over premise/hypothesis pairs.

        Inputs are processed in deterministic mini-batches. Softmax converts
        model logits into probabilities over the three NLI classes.

        `semantic_confidence` is the probability of the selected NLI class,
        not a calibrated probability that the QA answer is correct.
        """

        if len(premises) != len(hypotheses):
            raise ValueError(
                "Premises and hypotheses must have the same length."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if not premises:
            return []

        results: list[
            dict[str, Any]
        ] = []

        total = len(premises)

        for start_index in range(
            0,
            total,
            batch_size,
        ):
            end_index = min(
                start_index + batch_size,
                total,
            )

            premise_batch = premises[
                start_index:end_index
            ]

            hypothesis_batch = hypotheses[
                start_index:end_index
            ]

            print(
                "Processing semantic batch: "
                f"{start_index + 1}-{end_index}"
                f"/{total}",
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

            for probability_vector in probabilities:
                entailment_probability = float(
                    probability_vector[
                        self.entailment_id
                    ].item()
                )

                neutral_probability = float(
                    probability_vector[
                        self.neutral_id
                    ].item()
                )

                contradiction_probability = float(
                    probability_vector[
                        self.contradiction_id
                    ].item()
                )

                label_probabilities = {
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

                predicted_label = max(
                    label_probabilities,
                    key=lambda label: (
                        label_probabilities[label]
                    ),
                )

                results.append(
                    {
                        "semantic_label": (
                            predicted_label
                        ),
                        "entailment_probability": (
                            entailment_probability
                        ),
                        "neutral_probability": (
                            neutral_probability
                        ),
                        "contradiction_probability": (
                            contradiction_probability
                        ),
                        "semantic_confidence": (
                            label_probabilities[
                                predicted_label
                            ]
                        ),
                    }
                )

        return results


def validate_predictions(
    predictions: list[dict[str, Any]],
) -> None:
    """
    Validate semantic-verification inputs before loading the NLI model.

    Every prediction must contain a usable question, predicted answer, and
    evidence source.

    Local `evidence_text` is preferred when available; otherwise `context`
    must provide the semantic premise.
    """

    if not predictions:
        raise ValueError(
            "Prediction list cannot be empty."
        )

    for index, prediction in enumerate(
        predictions,
        start=1,
    ):
        missing_fields: list[str] = []

        if not get_question(prediction):
            missing_fields.append(
                "question"
            )

        if not get_predicted_answer(
            prediction
        ):
            missing_fields.append(
                "predicted_answer"
            )

        if not get_evidence_text(
            prediction
        ):
            missing_fields.append(
                "evidence_text/context"
            )

        if missing_fields:
            raise ValueError(
                f"Prediction {index} is missing "
                f"usable fields: {missing_fields}. "
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
    Build NLI premises and hypotheses from QA prediction records.

    Premise:
        local evidence window when available, otherwise full context

    Hypothesis:
        fixed question-answer declarative statement
    """

    premises: list[str] = []
    hypotheses: list[str] = []

    for prediction in predictions:
        question = get_question(
            prediction
        )

        predicted_answer = (
            get_predicted_answer(
                prediction
            )
        )

        evidence_text = (
            get_evidence_text(
                prediction
            )
        )

        hypothesis = (
            build_qa_hypothesis(
                question=question,
                predicted_answer=(
                    predicted_answer
                ),
            )
        )

        premises.append(
            evidence_text
        )

        hypotheses.append(
            hypothesis
        )

    return premises, hypotheses


def validate_runtime_settings(
    batch_size: int,
    max_length: int,
) -> None:
    """
    Validate semantic-verifier runtime settings.
    """

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    if max_length <= 0:
        raise ValueError(
            "max_length must be greater than zero."
        )


def run_semantic_verification(
    input_path: str | Path,
    output_path: str | Path,
    model_name: str,
    batch_size: int,
    max_length: int,
) -> list[dict[str, Any]]:
    """
    Run the complete fixed-template semantic NLI verification pipeline.

    Predictions are validated, converted into NLI premise/hypothesis pairs,
    evaluated in batches, enriched with semantic probabilities, and saved for
    later prototype verification stages.
    """

    predictions = load_jsonl(
        input_path
    )

    validate_predictions(
        predictions
    )

    validate_runtime_settings(
        batch_size=batch_size,
        max_length=max_length,
    )

    premises, hypotheses = (
        build_semantic_inputs(
            predictions
        )
    )

    verifier = (
        SemanticEvidenceVerifier(
            model_name=model_name,
            max_length=max_length,
        )
    )

    semantic_results = (
        verifier.predict_batch(
            premises=premises,
            hypotheses=hypotheses,
            batch_size=batch_size,
        )
    )

    if (
        len(semantic_results)
        != len(predictions)
    ):
        raise RuntimeError(
            "Semantic result count does not "
            "match prediction count."
        )

    verified_predictions: list[
        dict[str, Any]
    ] = []

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
        updated_prediction = (
            prediction.copy()
        )

        updated_prediction.update(
            {
                "semantic_hypothesis": (
                    hypothesis
                ),
                "semantic_label": (
                    semantic_result[
                        "semantic_label"
                    ]
                ),
                "entailment_probability": (
                    semantic_result[
                        "entailment_probability"
                    ]
                ),
                "neutral_probability": (
                    semantic_result[
                        "neutral_probability"
                    ]
                ),
                "contradiction_probability": (
                    semantic_result[
                        "contradiction_probability"
                    ]
                ),
                "semantic_confidence": (
                    semantic_result[
                        "semantic_confidence"
                    ]
                ),
                "semantic_verifier_model": (
                    model_name
                ),
            }
        )

        verified_predictions.append(
            updated_prediction
        )

        semantic_label = (
            semantic_result[
                "semantic_label"
            ]
        )

        label_counts[
            semantic_label
        ] += 1

        print(
            f"{index}/{len(predictions)} | "
            f"semantic={semantic_label} | "
            "entailment="
            f"{semantic_result['entailment_probability']:.4f} | "
            "neutral="
            f"{semantic_result['neutral_probability']:.4f} | "
            "contradiction="
            f"{semantic_result['contradiction_probability']:.4f}"
        )

    save_jsonl(
        verified_predictions,
        output_path,
    )

    print(
        "\nSemantic verification completed."
    )

    print(
        f"ENTAILMENT: "
        f"{label_counts['ENTAILMENT']}"
    )

    print(
        f"NEUTRAL: "
        f"{label_counts['NEUTRAL']}"
    )

    print(
        f"CONTRADICTION: "
        f"{label_counts['CONTRADICTION']}"
    )

    print(
        f"Results saved to: "
        f"{output_path}"
    )

    return verified_predictions


def parse_arguments() -> argparse.Namespace:
    """Parse semantic NLI verification settings."""

    parser = argparse.ArgumentParser(
        description=(
            "Verify QA predictions using a fixed "
            "question-answer hypothesis and NLI."
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
        "--model-name",
        default=DEFAULT_MODEL_NAME,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
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