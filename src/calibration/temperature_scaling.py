"""
The file learns the best temperature using the calibration set and applies it to make the model’s confidence scores more realistic.
"""

import argparse
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.calibration.calibration_metrics import is_prediction_correct
from src.utils.io import load_jsonl, save_json, save_jsonl

DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/raw_baseline_with_confidence_calibration.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "outputs/predictions/raw_baseline_calibrated_calibration.jsonl"
)

DEFAULT_PARAMETERS_PATH = Path("outputs/tables/temperature_scaling_parameters.json")


# converts any real number into a probability between 0 and 1 in a numerically stable way.
# It is typically used after temperature scaling to convert logits into calibrated confidence scores.
def stable_sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))

    exp_value = math.exp(value)

    return exp_value / (1.0 + exp_value)


# prepares the data needed for temperature scaling by extracting the model's answer–null margins and the corresponding correctness labels as PyTorch tensors.
def prepare_calibration_data(
    predictions: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not predictions:
        raise ValueError("Prediction list cannot be empty.")

    # is a list of prediction IDs that do not contain the "answer_null_margin" field.
    missing_margin = [
        prediction.get("id", "unknown")
        for prediction in predictions
        if "answer_null_margin" not in prediction
    ]

    print(
        "First prediction has margin:",
        "answer_null_margin" in predictions[0],
    )

    print(f"Missing answer_null_margin: {len(missing_margin)}")

    if len(missing_margin) > 0:
        raise ValueError(
            "Some predictions do not contain "
            "'answer_null_margin'. "
            f"Missing IDs: {missing_margin[:5]}"
        )

    # This line collects every prediction’s answer_null_margin and converts them into a PyTorch tensor.
    #  answer_null_margin -> shows whether the model prefers giving an answer or saying “no answer.”
    # Positive -> answer
    # Negative -> no answer
    # Bigger number -> stronger confidence
    # Example:
    # [2.1, -0.8, 1.4]
    margins = torch.tensor(
        [float(prediction["answer_null_margin"]) for prediction in predictions],
        dtype=torch.float64,
    )

    # This line checks every prediction with is_prediction_correct() and converts the results into a PyTorch tensor.
    # Example:
    # [1, 0, 1, 0]
    # Here, 1 means correct and 0 means incorrect.
    correct_labels = torch.tensor(
        [float(is_prediction_correct(prediction)) for prediction in predictions],
        dtype=torch.float,
    )

    # It returns all unique values in correct_labels.
    # Example:
    # [1, 0, 1, 0] -> [1, 0]
    unique_labels = torch.unique(correct_labels)

    # numel() returns the number of elements in a tensor.
    if unique_labels.numel() < 2:
        raise ValueError(
            "Temperature fitting requires both correct and incorrect predictions."
        )

    return margins, correct_labels


# It measures how well the model’s confidence matches whether predictions are actually correct.
def calculate_nll(logits: torch.Tensor, correct_labels: torch.Tensor) -> float:
    criterion = nn.BCEWithLogitsLoss()

    loss = criterion(logits, correct_labels)

    return float(loss.item())


# finds the optimal temperature that produces the lowest NLL, thereby improving the calibration of the model's confidence scores.
# Here, temperature is a positive number that controls how sharp or soft the model’s confidence scores are
# Sharp confidence -> probabilities are more extreme, closer to 0 or 1.
# Soft confidence -> probabilities are less extreme, closer to 0.5.
# We use temperature to fix confidence:
# If the model is too confident, use a higher temperature to make confidence lower.
# If the model is not confident enough, use a lower temperature to make confidence higher.
def fit_temperature(
    margins: torch.Tensor, correct_labels: torch.Tensor
) -> tuple[float, float, float]:
    # is NOT the actual temperature.
    # It is the logarithm of the temperature that the optimizer learns.
    # Initially: log_temperature = 0
    # Then: temperature = exp(log_temperature) -> temperature = exp(0) = 1
    log_temperature = nn.Parameter(torch.zeros(1, dtype=torch.float64))

    # creates a loss function for binary labels like 0 and 1.
    # It directly takes raw logits, converts them internally into probabilities, and measures how far they are from correct_labels.
    criterion = nn.BCEWithLogitsLoss()

    # This creates an optimizer that will automatically find the best value of log_temperature by minimizing the NLL.
    # LBFGS is an optimization algorithm. Its job is to repeatedly update log_temperature until the loss cannot be reduced anymore
    # It tries different log_temperature values, converts each one into a positive temperature, scales the margins, and checks the NLL.
    # Then loss.backward() tells the optimizer which direction would reduce the loss, and LBFGS keeps updating log_temperature until it finds the value with the lowest NLL.
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe"
    )

    nll_before = calculate_nll(logits=margins, correct_labels=correct_labels)

    # it computes the current loss and gradients so that the optimizer knows how to update log_temperature.
    # closure() is defined inside fit_temperature() because it is only needed by the LBFGS optimizer. Keeping it inside the function makes the code cleaner and follows PyTorch's recommended LBFGS usage.
    def closure() -> torch.Tensor:
        # Removes gradients from the previous attempt.
        optimizer.zero_grad()

        # The reason is temperature must always be positive.
        temperature = torch.exp(log_temperature)

        # We use scaled_logits because temperature scaling modifies the original logits before computing the loss, allowing the optimizer to find the temperature that produces the best-calibrated confidence scores.
        scaled_logits = margins / temperature

        loss = criterion(scaled_logits, correct_labels)

        # loss.backward() computes the information the optimizer needs to update the temperature and minimize NLL.
        loss.backward()

        return loss

    # optimizer.step(closure) runs the optimization process, repeatedly evaluating the loss through closure() and updating log_temperature until the NLL is minimized.
    optimizer.step(closure)

    # detach -> creates a version of the tensor that is no longer connected to PyTorch’s gradient-tracking graph.
    learned_temperature = float(torch.exp(log_temperature.detach()).item())

    if not math.isfinite(learned_temperature) or learned_temperature <= 0.0:
        raise ValueError(
            f"Temperature optimization produced an invalid value: {learned_temperature}"
        )

    scaled_logits = margins / learned_temperature

    nll_after = calculate_nll(logits=scaled_logits, correct_labels=correct_labels)

    return (learned_temperature, nll_before, nll_after)


# applies the learned temperature to every prediction, recalculates the confidence scores using the calibrated logits, stores both the original and calibrated confidences, and returns the updated predictions.
def apply_temperature(
    predictions: list[dict[str, Any]], temperature: float
) -> list[dict[str, Any]]:
    if temperature <= 0.0:
        raise ValueError("Temperature must be positive.")

    calibrated_predictions: list[dict[str, Any]] = []

    for prediction in predictions:
        margin = float(prediction["answer_null_margin"])

        calibrated_logit = margin / temperature

        calibrated_confidence = stable_sigmoid(calibrated_logit)

        updated_prediction = prediction.copy()

        updated_prediction["uncalibrated_confidence"] = float(prediction["confidence"])

        updated_prediction.update(
            {
                "calibrated_logit": (calibrated_logit),
                "confidence": (calibrated_confidence),
                "calibrated_confidence": (calibrated_confidence),
                "temperature": temperature,
                "confidence_type": ("temperature_scaled_answer_vs_null"),
                "confidence_is_calibrated": True,
            }
        )

        calibrated_predictions.append(updated_prediction)

    return calibrated_predictions


def run_temperature_scaling(
    input_path: str | Path, output_path: str | Path, parameters_path: str | Path
) -> list[dict[str, Any]]:
    predictions = load_jsonl(input_path)

    margins, correct_labels = prepare_calibration_data(predictions)

    temperature, nll_before, nll_after = fit_temperature(
        margins=margins, correct_labels=correct_labels
    )

    calibrated_predictions = apply_temperature(
        predictions=predictions, temperature=temperature
    )

    save_jsonl(calibrated_predictions, output_path)

    parameter_data = {
        "method": "temperature_scaling",
        "temperature": temperature,
        "fit_examples": len(predictions),
        "correct_examples": int(correct_labels.sum().item()),
        "incorrect_examples": int(len(correct_labels) - correct_labels.sum().item()),
        "nll_before": nll_before,
        "nll_after": nll_after,
        # gets the dataset split name from the first prediction, such as "calibration" or "test".
        # If "split" does not exist, it stores "unknown" instead.
        "fit_split": predictions[0].get(
            "split",
            "unknown",
        ),
        # it tells us which signal was calibrated.
        "input_signal": ("answer_null_margin"),
        # records that the test set was not used to learn the temperature.
        "test_set_used_for_fitting": False,
    }

    save_json(
        parameter_data,
        parameters_path,
    )

    print("\nTemperature scaling completed.")

    print(f"Examples used: {len(predictions)}")

    print(f"Temperature: {temperature:.6f}")

    print(f"NLL before: {nll_before:.6f}")

    print(f"NLL after: {nll_after:.6f}")

    print(f"Predictions saved to: {output_path}")

    print(f"Parameters saved to: {parameters_path}")

    return calibrated_predictions


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate QA confidence using temperature scaling."
    )

    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))

    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))

    parser.add_argument("--parameters", default=str(DEFAULT_PARAMETERS_PATH))

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_temperature_scaling(
        input_path=args.input,
        output_path=args.output,
        parameters_path=args.parameters,
    )
