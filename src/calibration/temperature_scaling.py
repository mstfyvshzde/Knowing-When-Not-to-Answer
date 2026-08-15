"""
Fit temperature scaling on the calibration split and recalibrate QA confidence.

Temperature scaling (sıcaklık ölçekleme) learns one positive scalar T using
the calibration split. The project's answer-vs-null margin is divided by T
before being converted into a probability with the sigmoid function.

The temperature is selected by minimizing binary Negative Log-Likelihood (NLL)
between calibrated confidence and forced-answer correctness.

Interpretation:

- T > 1 -> confidence becomes softer and moves closer to 0.5
- T < 1 -> confidence becomes sharper and more extreme
- T = 1 -> confidence remains unchanged

Because T is always positive, dividing margins by T and then applying sigmoid
is a monotonic transformation (sıralamayı bozmayan dönüşüm). Therefore,
temperature scaling changes probability calibration but does not change the
confidence-only ranking of examples.

The temperature is fitted on calibration data only. Held-out test labels are
never used to estimate this parameter.
"""


import argparse
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.calibration.calibration_metrics import is_prediction_correct
from src.utils.io import load_jsonl, save_json, save_jsonl

# Calibration predictions used to fit the temperature parameter.
DEFAULT_INPUT_PATH = Path(
    "outputs/predictions/raw_baseline_with_confidence_calibration.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "outputs/predictions/raw_baseline_calibrated_calibration.jsonl"
)

# Store the learned parameter separately so it can be frozen and reused.
DEFAULT_PARAMETERS_PATH = Path("outputs/tables/temperature_scaling_parameters.json")

def stable_sigmoid(value: float) -> float:
    """
    Convert a logit into a probability using a numerically stable sigmoid.

    A logit (olasılıktan önceki sınırsız skor) can take any real value, while
    sigmoid maps it into the probability range [0, 1].

    The two-branch implementation avoids numerical overflow for very large
    positive or negative values.
    """

    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))

    exp_value = math.exp(value)

    return exp_value / (1.0 + exp_value)


def prepare_calibration_data(
    predictions: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract the signal and target required for temperature fitting.

    answer_null_margin is the signed answer-vs-null score difference
    (cevap ile no-answer seçeneği arasındaki skor farkı).

    Generally:

    - positive margin -> model favors returning an answer
    - negative margin -> model favors the null/no-answer option
    - larger absolute magnitude -> stronger preference

    The corresponding correctness label is binary:

    - 1 -> forced-answer prediction is correct
    - 0 -> forced-answer prediction is incorrect

    Temperature scaling learns how these margins should map to probabilities of
    prediction correctness.
    """

    if not predictions:
        raise ValueError("Prediction list cannot be empty.")

    # Identify malformed records before attempting temperature fitting.
    missing_margin = [
        prediction.get("id", "unknown")
        for prediction in predictions
        if "answer_null_margin" not in prediction
    ]

    if missing_margin:
        raise ValueError(
            "Some predictions do not contain "
            "'answer_null_margin'. "
            f"Missing IDs: {missing_margin[:5]}"
        )

    # answer_null_margin represents the model's preference between returning an
    # answer and choosing the null/no-answer option. Its raw scale is not yet a
    # calibrated probability, which is why temperature scaling is applied later.
    margins = torch.tensor(
        [float(prediction["answer_null_margin"]) for prediction in predictions],
        dtype=torch.float64,
    )

    # Build binary correctness targets using the project's forced-answer metric.
    correct_labels = torch.tensor(
        [float(is_prediction_correct(prediction)) for prediction in predictions],
        dtype=torch.float64,
    )

    # Both classes are required to fit a meaningful binary calibration parameter.
    unique_labels = torch.unique(correct_labels)

    # Temperature fitting needs both outcomes; otherwise there is no meaningful
    # correct-vs-incorrect probability calibration problem to optimize.
    if unique_labels.numel() < 2:
        raise ValueError(
            "Temperature fitting requires both correct and incorrect predictions."
        )

    return margins, correct_labels


def calculate_nll(logits: torch.Tensor, correct_labels: torch.Tensor) -> float:
    """
    Calculate binary Negative Log-Likelihood (NLL) directly from logits.

    NLL measures how well predicted probabilities agree with actual correctness.
    It penalizes confident mistakes especially strongly, making it suitable as
    the optimization objective for temperature scaling.

    Lower NLL means better probabilistic calibration.
    """

    criterion = nn.BCEWithLogitsLoss()

    loss = criterion(logits, correct_labels)

    return float(loss.item())


def fit_temperature(
    margins: torch.Tensor, correct_labels: torch.Tensor
) -> tuple[float, float, float]:
    """
    Learn the temperature that minimizes calibration-set NLL.

    Instead of optimizing temperature directly, the function optimizes
    log_temperature and then computes:

        temperature = exp(log_temperature)

    Exponentiation guarantees T > 0 throughout optimization.

    The calibrated logit is:

        calibrated_logit = margin / temperature

    A larger temperature softens confidence values, while a smaller temperature
    makes them more extreme.

    Only one scalar parameter is learned; model weights and predicted answer spans
    remain unchanged.
    """

    # Optimize log-temperature so the actual temperature is always positive.
    log_temperature = nn.Parameter(torch.zeros(1, dtype=torch.float64))

    criterion = nn.BCEWithLogitsLoss()

    # LBFGS is an optimization algorithm used here to find the single temperature
    # parameter that minimizes calibration NLL. It changes only log_temperature;
    # the underlying QA model remains frozen.
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe"
    )

    nll_before = calculate_nll(logits=margins, correct_labels=correct_labels)


    # LBFGS requires a closure (optimizerın tekrar çağırabildiği loss fonksiyonu)
    # because it may evaluate the objective several times while searching for a
    # better temperature value.
    def closure() -> torch.Tensor:
        optimizer.zero_grad()

        temperature = torch.exp(log_temperature)
        scaled_logits = margins / temperature

        loss = criterion(scaled_logits, correct_labels)

        loss.backward()

        return loss

    # Fit the temperature using calibration-set correctness only.
    optimizer.step(closure)

    learned_temperature = float(torch.exp(log_temperature.detach()).item())

    if not math.isfinite(learned_temperature) or learned_temperature <= 0.0:
        raise ValueError(
            f"Temperature optimization produced an invalid value: {learned_temperature}"
        )

    scaled_logits = margins / learned_temperature

    nll_after = calculate_nll(logits=scaled_logits, correct_labels=correct_labels)

    return learned_temperature, nll_before, nll_after


def apply_temperature(
    predictions: list[dict[str, Any]], temperature: float
) -> list[dict[str, Any]]:
    """
    Apply an already learned temperature without fitting any new parameter.

    For every prediction:

        calibrated_logit = answer_null_margin / temperature
        calibrated_confidence = sigmoid(calibrated_logit)

    The original confidence is preserved in `uncalibrated_confidence`, while the
    main `confidence` field is replaced with the calibrated probability.

    Because temperature is positive and sigmoid is monotonic (artan sıralamayı
    koruyan fonksiyon), this transformation preserves example ranking.

    Therefore, temperature scaling can improve ECE, Brier score, and NLL without
    changing confidence-only risk-coverage ranking or AURC.
    """

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("Temperature must be finite and positive.")

    calibrated_predictions: list[dict[str, Any]] = []

    for prediction in predictions:
        # Apply the frozen temperature (önceden öğrenilmiş ve artık değiştirilmemiş T)
        # to each answer-vs-null margin. No test label or correctness information is
        # needed during this transformation.
        margin = float(prediction["answer_null_margin"])

        calibrated_logit = margin / temperature

        calibrated_confidence = stable_sigmoid(calibrated_logit)

        # Preserve both the original and calibrated confidence values.
        updated_prediction = prediction.copy()

        updated_prediction["uncalibrated_confidence"] = float(prediction["confidence"])

        updated_prediction.update(
            {
                "calibrated_logit": calibrated_logit,
                "confidence": calibrated_confidence,
                "calibrated_confidence": calibrated_confidence,
                "temperature": temperature,
                "confidence_type": "temperature_scaled_answer_vs_null",
                "confidence_is_calibrated": True,
            }
        )

        calibrated_predictions.append(updated_prediction)

    return calibrated_predictions


def run_temperature_scaling(
    input_path: str | Path,
    output_path: str | Path,
    parameters_path: str | Path,
) -> list[dict[str, Any]]:
    """
    Fit temperature scaling using calibration predictions and save the result.

    The function verifies that all fitting records belong to the calibration split,
    learns one temperature from calibration correctness labels, applies that
    temperature to the same calibration records, and stores the learned parameter
    separately for reproducibility and later reuse.

    The held-out test split must never enter this fitting procedure.
    """

    predictions = load_jsonl(input_path)

    # Data leakage (test bilgisinin ayar sürecine sızması) would occur if held-out
    # test correctness influenced the learned temperature. Enforce calibration-only
    # fitting before optimization starts.
    fit_splits = {prediction.get("split", "unknown") for prediction in predictions}

    if fit_splits != {"calibration"}:
        raise ValueError(
            "Temperature must be fitted on the calibration split only. "
            f"Found splits: {sorted(fit_splits)}"
        )

    margins, correct_labels = prepare_calibration_data(predictions)

    temperature, nll_before, nll_after = fit_temperature(
        margins=margins, correct_labels=correct_labels
    )

    calibrated_predictions = apply_temperature(
        predictions=predictions, temperature=temperature
    )

    save_jsonl(calibrated_predictions, output_path)


    # Save calibration provenance (parametrenin nereden/nasıl öğrenildiği bilgisi)
    # so the learned temperature can be audited and reproduced later.
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

    save_json(parameter_data, parameters_path)

    print("\nTemperature scaling completed.")

    print(f"Examples used: {len(predictions)}")

    print(f"Temperature: {temperature:.6f}")

    print(f"NLL before: {nll_before:.6f}")

    print(f"NLL after: {nll_after:.6f}")

    print(f"Predictions saved to: {output_path}")

    print(f"Parameters saved to: {parameters_path}")

    return calibrated_predictions


def parse_arguments() -> argparse.Namespace:
    """Parse paths used for calibration fitting and saved parameters."""

    parser = argparse.ArgumentParser(
        description="Fit temperature scaling on QA calibration predictions."
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Calibration prediction JSONL containing answer-vs-null margins.",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path for temperature-calibrated calibration predictions.",
    )

    parser.add_argument(
        "--parameters",
        default=str(DEFAULT_PARAMETERS_PATH),
        help="Path used to save the learned temperature and fitting metadata.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_temperature_scaling(
        input_path=args.input,
        output_path=args.output,
        parameters_path=args.parameters,
    )
