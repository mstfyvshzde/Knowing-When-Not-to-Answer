"""
Estimate QA confidence from answer-vs-null logits.

This module reconstructs a confidence signal for each forced-answer prediction
using the same pretrained SQuAD v2 QA backbone.

For each prediction, it:

1. tokenizes the original question and context,
2. locates the forced-answer span in model token space,
3. computes the answer-span logit score,
4. computes the model's null/no-answer logit score,
5. forms an answer-vs-null margin,
6. maps that margin into [0, 1] with a sigmoid.

The core signal is:

    answer_score = start_logit(answer_start) + end_logit(answer_end)

    null_score = start_logit(CLS) + end_logit(CLS)

    answer_null_margin = answer_score - null_score

A positive margin means the forced answer is favored over the null option,
while a negative margin means the model favors no-answer more strongly.

The sigmoid output produced here is an uncalibrated confidence
(kalibre edilmemiş güven skoru). A value such as 0.80 must not yet be
interpreted as an 80% probability of correctness.

Temperature scaling is fitted later on the calibration split to improve that
probabilistic interpretation.
"""

import argparse
import math
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from src.utils.io import load_jsonl, save_jsonl

# Use the same pretrained QA backbone as the forced-answer baseline so the
# confidence signal is derived from the model that produced the answer.
MODEL_NAME = "deepset/roberta-base-squad2"

DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/raw_baseline_calibration.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "outputs/predictions/raw_baseline_with_confidence_calibration.jsonl"
)


# Maximum number of tokens processed in one QA feature, including the question,
# context portion, and special tokens.
MAX_LENGTH = 384

# Long contexts are split into overlapping chunks. The stride (örtüşme miktarı)
# reuses 128 tokens between neighboring chunks so an answer near a chunk
# boundary is less likely to be lost.
DOCUMENT_STRIDE = 128


def select_device(device_name: str) -> torch.device:
    """
    Select the hardware device used for QA inference.

    CUDA refers to NVIDIA GPU execution, while MPS is Apple's GPU backend.
    If the requested accelerator is unavailable, an error is raised instead of
    silently changing the experiment hardware.
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
    Map an answer-vs-null logit margin into the interval [0, 1].

    Large positive margins approach 1.0, large negative margins approach 0.0,
    and a margin of zero maps to 0.5.

    This numerical transformation does not by itself make the result a calibrated
    probability of correctness.
    """

    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))

    exp_value = math.exp(value)

    return exp_value / (1.0 + exp_value)


def find_cls_index(input_ids: torch.Tensor, cls_token_id: int) -> int:
    """
    Find the CLS token position in one tokenized QA feature.

    For this SQuAD v2 QA model, the CLS position is used as the null/no-answer
    position. Its start and end logits therefore provide the model's null score.
    """

    positions = (input_ids == cls_token_id).nonzero(as_tuple=False)

    if len(positions) == 0:
        raise ValueError("CLS token was not found in model input.")

    return int(positions[0].item())


def find_answer_token_span(
    offsets: list[list[int]],
    sequence_ids: list[int | None],
    answer_start: int,
    answer_end: int,
) -> tuple[int, int] | None:
    """
    Map a character-level answer span to token indices inside one QA chunk.

    The forced-answer pipeline stores answer boundaries as character positions in
    the original context, but the QA model produces logits for token positions.

    offsets map each token back to its character interval, while sequence_ids
    identify whether a token belongs to the question, context, or special tokens.

    Only context tokens are considered.

    Returns:
        (start_token_index, end_token_index) when the complete answer is present
        inside this chunk, otherwise None.
    """

    token_start: int | None = None
    token_end: int | None = None

    for token_index, (offset, sequence_id) in enumerate(zip(offsets, sequence_ids)):
        # sequence_id == 1 identifies context tokens. Question and special tokens
        # cannot represent the extracted answer span.
        if sequence_id != 1:
            continue

        offset_start, offset_end = offset

        # Special or padding tokens may have an empty [0, 0] offset.
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
    Estimate the answer-vs-null confidence signal for one forced-answer prediction.

    The forced-answer prediction provides a character-level answer span. This
    function runs the same QA model again to recover the logits associated with
    that span and with the model's null/no-answer position.

    For each relevant chunk:

    answer_score =
        start_logit(answer_start_token) + end_logit(answer_end_token)

    null_score =
        start_logit(CLS) + end_logit(CLS)

    Across overlapping chunks, the strongest representation of the forced answer
    and the strongest evidence against the null option are retained.

    The final uncertainty signal is:

    answer_null_margin = best_answer_score - best_null_score

    The margin is then passed through sigmoid to obtain an uncalibrated confidence.
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

    # Split long contexts into overlapping QA features while preserving character
    # offsets so the raw answer span can later be mapped back to model tokens.
    encoded = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOCUMENT_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=True,
        return_tensors="pt",
    )

    # Offset mappings are needed only for character-to-token alignment and are not
    # valid model inputs.
    offset_mapping = encoded.pop("offset_mapping")

    # This function processes one original example at a time, so the tokenizer's
    # overflow-to-sample mapping is unnecessary after chunk creation.
    encoded.pop("overflow_to_sample_mapping", None)

    # Record which tokens belong to the question, context, or special-token region
    # for every overlapping chunk.
    all_sequence_ids = [
        encoded.sequence_ids(chunk_index)
        for chunk_index in range(encoded["input_ids"].shape[0])
    ]


    model_inputs = {key: value.to(device) for key, value in encoded.items()}


    # Confidence extraction is inference only. Gradients are unnecessary because
    # the pretrained QA model is not being trained or fine-tuned here.
    with torch.no_grad():
        outputs = model(**model_inputs)

    # Move logits and alignment information to CPU for deterministic bookkeeping
    # and Python-side span calculations.
    start_logits = outputs.start_logits.detach().cpu()
    end_logits = outputs.end_logits.detach().cpu()
    input_ids = encoded["input_ids"].detach().cpu()
    offset_by_chunk = offset_mapping.detach().cpu().tolist()

    answer_scores: list[float] = []
    null_scores: list[float] = []

    selected_span: tuple[int, int] | None = None
    selected_chunk: int | None = None
    selected_answer_score = -math.inf

    for chunk_index in range(input_ids.shape[0]):
        cls_index = find_cls_index(
            input_ids=input_ids[chunk_index], cls_token_id=tokenizer.cls_token_id
        )

        # For SQuAD v2, selecting CLS for both start and end represents the null
        # prediction. Their summed logits form this chunk's no-answer score.
        null_score = float(
            start_logits[chunk_index, cls_index].item()
            + end_logits[chunk_index, cls_index].item()
        )

        null_scores.append(null_score)

        token_span = find_answer_token_span(
            offsets=offset_by_chunk[chunk_index],
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

        # The same character span may appear in multiple overlapping chunks.
        # Keep the chunk where that span receives its highest model logit score.
        if answer_score > selected_answer_score:
            selected_answer_score = answer_score
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

    # The forced answer may occur in several overlapping chunks. Use its highest
    # observed span score as the strongest model support for that answer.
    best_answer_score = max(answer_scores)

    # Following the SQuAD-style multi-feature setup, use the lowest null score
    # observed across chunks as the strongest feature-level evidence against
    # the no-answer option.
    best_null_score = min(null_scores)

    # Core uncertainty signal:
    # positive margin -> forced answer is stronger than the null option
    # negative margin -> null/no-answer option is stronger
    answer_null_margin = best_answer_score - best_null_score

    # Sigmoid makes the margin easier to use on a [0, 1] scale, but this value is
    # still uncalibrated and must not yet be interpreted as correctness probability.
    uncalibrated_confidence = sigmoid(answer_null_margin)

    # Preserve the original forced-answer record and attach the uncertainty signal
    # together with provenance needed by later calibration stages.
    updated_prediction = prediction.copy()

    updated_prediction.update(
        {
            "answer_span_logit": best_answer_score,
            "null_logit": best_null_score,
            "answer_null_margin": answer_null_margin,
            "confidence": uncalibrated_confidence,
            "confidence_type": "uncalibrated_answer_vs_null",
            "confidence_is_calibrated": False,
            "confidence_model": MODEL_NAME,
            "confidence_chunk": selected_chunk,
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
    Estimate answer-vs-null confidence for a collection of raw QA predictions.

    The tokenizer and pretrained QA model are loaded once and reused for every
    prediction. Model weights remain frozen throughout the procedure.
    """

    if not predictions:
        raise ValueError("Prediction list cannot be empty.")

    device = select_device(device_name)

    print(f"Loading model: {MODEL_NAME}")

    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)

    # Disable training-specific behavior such as dropout so repeated inference
    # uses the model in evaluation mode.
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
            f"uncalibrated_confidence="
            f"{updated_prediction['confidence']:.4f}"
        )

    return updated_predictions


def run_confidence_estimator(
    input_path: str | Path, output_path: str | Path, device_name: str
) -> list[dict[str, Any]]:
    """
    Load forced-answer predictions, estimate confidence signals, and save them.

    The resulting JSONL preserves the original QA outputs while adding the
    answer-vs-null margin and its uncalibrated sigmoid transformation for later
    temperature scaling.
    """

    predictions = load_jsonl(input_path)

    updated_predictions = estimate_confidences(
        predictions=predictions, device_name=device_name
    )

    save_jsonl(updated_predictions, output_path)

    print(f"\nConfidence predictions saved to: {output_path}")

    return updated_predictions


def parse_arguments() -> argparse.Namespace:
    """Parse confidence-estimation paths and inference-device settings."""

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
