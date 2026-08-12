"""
Independent self-verification for selective question answering.

The verifier checks whether a predicted answer is supported by the
question context using an independent NLI model.

It does not use gold answers or correctness labels at inference time.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.utils.io import load_jsonl, save_jsonl


DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/calibration_with_hybrid_evidence.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "outputs/predictions/calibration_with_self_verification.jsonl"
)

DEFAULT_MODEL_NAME = "FacebookAI/roberta-large-mnli"

DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_CONTEXT_TOKENS = 384
DEFAULT_MAX_CLAIM_TOKENS = 96

DEFAULT_SUPPORTED_THRESHOLD = 0.70
DEFAULT_REJECT_THRESHOLD = 0.70


def clean_text(value: Any) -> str:
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
    for field_name in field_names:
        value = prediction.get(field_name)

        if value is not None:
            return value

    return default


def get_question(
    prediction: dict[str, Any],
) -> str:
    value = get_first_value(
        prediction,
        (
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
    value = get_first_value(
        prediction,
        (
            "prediction_text",
            "predicted_answer",
            "prediction_answer",
            "answer",
        ),
        default="",
    )

    return clean_text(value)


def get_context(
    prediction: dict[str, Any],
) -> str:
    value = get_first_value(
        prediction,
        (
            "context",
            "passage",
            "evidence_context",
            "source_context",
        ),
        default="",
    )

    return clean_text(value)


def build_self_verification_claim(
    question: str,
    answer: str,
) -> str:
    return (
        f'The answer to the question '
        f'"{question}" is "{answer}".'
    )


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(
            torch.backends,
            "mps",
        )
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def batched_indices(
    total_size: int,
    batch_size: int,
):
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


class SelfVerifier:
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        max_context_tokens: int,
        max_claim_tokens: int,
        supported_threshold: float,
        reject_threshold: float,
    ) -> None:
        self.device = device

        self.max_context_tokens = (
            max_context_tokens
        )

        self.max_claim_tokens = (
            max_claim_tokens
        )

        self.supported_threshold = (
            supported_threshold
        )

        self.reject_threshold = (
            reject_threshold
        )

        print(
            f"Loading self-verification model: "
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

        self.model.to(
            self.device
        )

        self.model.eval()

        self.label_to_id = (
            self._resolve_label_ids()
        )

    def _resolve_label_ids(
        self,
    ) -> dict[str, int]:
        resolved: dict[
            str,
            int,
        ] = {}

        for raw_id, raw_label in (
            self.model.config.id2label.items()
        ):
            label = (
                str(raw_label)
                .strip()
                .lower()
            )

            label_id = int(
                raw_id
            )

            if "entail" in label:
                resolved[
                    "entailment"
                ] = label_id

            elif "neutral" in label:
                resolved[
                    "neutral"
                ] = label_id

            elif "contrad" in label:
                resolved[
                    "contradiction"
                ] = label_id

        required = {
            "entailment",
            "neutral",
            "contradiction",
        }

        missing = (
            required
            - set(resolved)
        )

        if missing:
            raise ValueError(
                "Could not resolve NLI labels. "
                f"Missing: {sorted(missing)}"
            )

        return resolved

    def _truncate_text(
        self,
        text: str,
        max_tokens: int,
    ) -> str:
        token_ids = (
            self.tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_tokens,
            )
        )

        return (
            self.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        )

    def _assign_label(
        self,
        entailment_probability: float,
        neutral_probability: float,
        contradiction_probability: float,
    ) -> str:
        if (
            contradiction_probability
            >= self.reject_threshold
        ):
            return "REJECTED"

        if (
            entailment_probability
            >= self.supported_threshold
        ):
            return "SUPPORTED"

        return "UNCERTAIN"

    @torch.inference_mode()
    def verify_batch(
        self,
        contexts: list[str],
        claims: list[str],
    ) -> list[dict[str, Any]]:
        if len(contexts) != len(claims):
            raise ValueError(
                "Context and claim batch sizes "
                "must match."
            )

        truncated_contexts = [
            self._truncate_text(
                context,
                self.max_context_tokens,
            )
            for context in contexts
        ]

        truncated_claims = [
            self._truncate_text(
                claim,
                self.max_claim_tokens,
            )
            for claim in claims
        ]

        encoded = self.tokenizer(
            truncated_contexts,
            truncated_claims,
            padding=True,
            truncation="only_first",
            max_length=512,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(
                self.device
            )
            for key, value
            in encoded.items()
        }

        outputs = self.model(
            **encoded
        )

        probabilities = (
            torch.softmax(
                outputs.logits,
                dim=-1,
            )
            .detach()
            .cpu()
        )

        entailment_id = (
            self.label_to_id[
                "entailment"
            ]
        )

        neutral_id = (
            self.label_to_id[
                "neutral"
            ]
        )

        contradiction_id = (
            self.label_to_id[
                "contradiction"
            ]
        )

        results: list[
            dict[str, Any]
        ] = []

        for row in probabilities:
            entailment_probability = float(
                row[
                    entailment_id
                ].item()
            )

            neutral_probability = float(
                row[
                    neutral_id
                ].item()
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

            score = (
                entailment_probability
                - contradiction_probability
            )

            score = max(
                -1.0,
                min(
                    1.0,
                    score,
                ),
            )

            results.append(
                {
                    "self_verification_label": (
                        label
                    ),
                    "self_verification_score": (
                        score
                    ),
                    "self_entailment_probability": (
                        entailment_probability
                    ),
                    "self_neutral_probability": (
                        neutral_probability
                    ),
                    "self_contradiction_probability": (
                        contradiction_probability
                    ),
                }
            )

        return results


def validate_predictions(
    predictions: list[dict[str, Any]],
) -> None:
    if not predictions:
        raise ValueError(
            "Prediction list cannot be empty."
        )

    for index, prediction in enumerate(
        predictions,
        start=1,
    ):
        if not get_question(
            prediction
        ):
            raise ValueError(
                f"Prediction {index} "
                "has no question."
            )

        if not get_context(
            prediction
        ):
            raise ValueError(
                f"Prediction {index} "
                "has no context."
            )


def run_self_verification(
    input_path: str | Path,
    output_path: str | Path,
    model_name: str,
    batch_size: int,
    max_context_tokens: int,
    max_claim_tokens: int,
    supported_threshold: float,
    reject_threshold: float,
) -> list[dict[str, Any]]:
    predictions = load_jsonl(
        input_path
    )

    validate_predictions(
        predictions
    )

    if not (
        0.0
        <= supported_threshold
        <= 1.0
    ):
        raise ValueError(
            "supported_threshold must be "
            "between 0 and 1."
        )

    if not (
        0.0
        <= reject_threshold
        <= 1.0
    ):
        raise ValueError(
            "reject_threshold must be "
            "between 0 and 1."
        )

    device = select_device()

    print(
        f"Using device: "
        f"{device}"
    )

    verifier = SelfVerifier(
        model_name=model_name,
        device=device,
        max_context_tokens=(
            max_context_tokens
        ),
        max_claim_tokens=(
            max_claim_tokens
        ),
        supported_threshold=(
            supported_threshold
        ),
        reject_threshold=(
            reject_threshold
        ),
    )

    verified_predictions: list[
        dict[str, Any]
    ] = []

    label_counts: Counter[
        str
    ] = Counter()

    total = len(
        predictions
    )

    for batch_number, (
        start_index,
        end_index,
    ) in enumerate(
        batched_indices(
            total_size=total,
            batch_size=batch_size,
        ),
        start=1,
    ):
        batch = predictions[
            start_index:end_index
        ]

        contexts: list[str] = []
        claims: list[str] = []
        answer_available: list[
            bool
        ] = []

        for prediction in batch:
            question = get_question(
                prediction
            )

            answer = get_predicted_answer(
                prediction
            )

            context = get_context(
                prediction
            )

            contexts.append(
                context
            )

            answer_available.append(
                bool(answer)
            )

            claims.append(
                build_self_verification_claim(
                    question=question,
                    answer=answer,
                )
                if answer
                else ""
            )

        results: list[
            dict[str, Any]
        ] = [
            {
                "self_verification_label": (
                    "REJECTED"
                ),
                "self_verification_score": (
                    -1.0
                ),
                "self_entailment_probability": (
                    0.0
                ),
                "self_neutral_probability": (
                    0.0
                ),
                "self_contradiction_probability": (
                    1.0
                ),
            }
            for _ in batch
        ]

        valid_indices = [
            index
            for index, available
            in enumerate(
                answer_available
            )
            if available
        ]

        if valid_indices:
            verified_results = (
                verifier.verify_batch(
                    contexts=[
                        contexts[index]
                        for index
                        in valid_indices
                    ],
                    claims=[
                        claims[index]
                        for index
                        in valid_indices
                    ],
                )
            )

            for local_index, result in zip(
                valid_indices,
                verified_results,
            ):
                results[
                    local_index
                ] = result

        for prediction, claim, result in zip(
            batch,
            claims,
            results,
        ):
            updated = dict(
                prediction
            )

            updated.update(
                {
                    "self_verification_claim": (
                        claim
                        if claim
                        else None
                    ),
                    **result,
                    "self_verification_model": (
                        model_name
                    ),
                    "self_supported_threshold": (
                        supported_threshold
                    ),
                    "self_reject_threshold": (
                        reject_threshold
                    ),
                }
            )

            verified_predictions.append(
                updated
            )

            label_counts[
                result[
                    "self_verification_label"
                ]
            ] += 1

        print(
            f"Processed "
            f"{end_index}/{total} "
            f"records "
            f"(batch {batch_number})."
        )

    save_jsonl(
        verified_predictions,
        output_path,
    )

    print(
        "\nSelf verification completed."
    )

    for label in (
        "SUPPORTED",
        "UNCERTAIN",
        "REJECTED",
    ):
        print(
            f"{label:<12}: "
            f"{label_counts[label]}"
        )

    print(
        f"\nResults saved to: "
        f"{output_path}"
    )

    return verified_predictions


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify predicted answers "
            "against their contexts "
            "using an independent NLI model."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=(
            DEFAULT_INPUT_PATH
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_OUTPUT_PATH
        ),
    )

    parser.add_argument(
        "--model",
        default=(
            DEFAULT_MODEL_NAME
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=(
            DEFAULT_BATCH_SIZE
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
        "--supported-threshold",
        type=float,
        default=(
            DEFAULT_SUPPORTED_THRESHOLD
        ),
    )

    parser.add_argument(
        "--reject-threshold",
        type=float,
        default=(
            DEFAULT_REJECT_THRESHOLD
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    run_self_verification(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        max_context_tokens=(
            args.max_context_tokens
        ),
        max_claim_tokens=(
            args.max_claim_tokens
        ),
        supported_threshold=(
            args.supported_threshold
        ),
        reject_threshold=(
            args.reject_threshold
        ),
    )


if __name__ == "__main__":
    main()