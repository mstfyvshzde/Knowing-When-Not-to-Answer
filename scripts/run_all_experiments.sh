#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT"

DEVICE="${DEVICE:-cpu}"
LIMIT="${LIMIT:-3000}"

if [[ "$LIMIT" == "3000" ]]; then
    PRED_DIR="outputs/predictions"
    TABLE_DIR="outputs/tables"
    FINAL_EVAL_DIR="outputs/evaluation/final_sample_size_comparison"
else
    PRED_DIR="outputs/predictions/smoke_limit_${LIMIT}"
    TABLE_DIR="outputs/tables/smoke_limit_${LIMIT}"
    FINAL_EVAL_DIR="outputs/evaluation/smoke_limit_${LIMIT}"
fi

mkdir -p "$PRED_DIR" "$TABLE_DIR" "$FINAL_EVAL_DIR"

run_if_missing() {
    local output="$1"
    shift

    if [[ -f "$output" ]]; then
        local line_count
        line_count="$(wc -l < "$output" | tr -d "[:space:]")"

        if [[ "$line_count" == "$LIMIT" ]]; then
            echo "✓ Exists with $line_count rows, skipping: $output"
            return
        fi

        echo "↻ Stale row count ($line_count != $LIMIT), regenerating: $output"
    else
        echo "→ Creating: $output"
    fi

    "$@"
}

echo "========================================"
echo "Final selective-QA experiment pipeline"
echo "Device: $DEVICE"
echo "Examples per split: $LIMIT"
echo "========================================"

# 1. Raw predictions
run_if_missing \
    "$PRED_DIR/raw_baseline_calibration.jsonl" \
    python -m src.baselines.raw_answer_baseline \
        --split calibration \
        --limit "$LIMIT" \
        --device "$DEVICE"

run_if_missing \
    "$PRED_DIR/raw_baseline_test.jsonl" \
    python -m src.baselines.raw_answer_baseline \
        --split test \
        --limit "$LIMIT" \
        --device "$DEVICE"

# 2. Uncalibrated confidence
run_if_missing \
    "$PRED_DIR/raw_baseline_with_confidence_calibration.jsonl" \
    python -m src.verification.confidence_estimator \
        --input "$PRED_DIR/raw_baseline_calibration.jsonl" \
        --output "$PRED_DIR/raw_baseline_with_confidence_calibration.jsonl" \
        --device "$DEVICE"

run_if_missing \
    "$PRED_DIR/raw_baseline_with_confidence_test.jsonl" \
    python -m src.verification.confidence_estimator \
        --input "$PRED_DIR/raw_baseline_test.jsonl" \
        --output "$PRED_DIR/raw_baseline_with_confidence_test.jsonl" \
        --device "$DEVICE"

# 3. Fit temperature on CALIBRATION ONLY
CALIBRATED_CALIBRATION="$PRED_DIR/raw_baseline_calibrated_calibration.jsonl"

if [[ -f "$CALIBRATED_CALIBRATION" && -f "$TABLE_DIR/temperature_scaling_parameters.json" ]]; then
    calibration_rows="$(wc -l < "$CALIBRATED_CALIBRATION" | tr -d "[:space:]")"
else
    calibration_rows=0
fi

if [[ "$calibration_rows" == "$LIMIT" ]]; then
    echo "Calibration parameters exist for $calibration_rows rows, skipping temperature fitting."
else
    echo "Fitting temperature on calibration split ($LIMIT rows)."
    python -m src.calibration.temperature_scaling         --input "$PRED_DIR/raw_baseline_with_confidence_calibration.jsonl"         --output "$CALIBRATED_CALIBRATION"         --parameters "$TABLE_DIR/temperature_scaling_parameters.json"
fi


# 4. Apply the frozen calibration temperature to TEST.
#    Test labels are never used for fitting.
run_if_missing \
    "$PRED_DIR/raw_baseline_calibrated_test.jsonl" \
    env PRED_DIR="$PRED_DIR" TABLE_DIR="$TABLE_DIR" python -c '
import json
import os
from pathlib import Path

from src.calibration.temperature_scaling import apply_temperature
from src.utils.io import load_jsonl, save_jsonl

parameter_path = Path(os.environ["TABLE_DIR"]) / "temperature_scaling_parameters.json"
input_path = Path(os.environ["PRED_DIR"]) / "raw_baseline_with_confidence_test.jsonl"
output_path = Path(os.environ["PRED_DIR"]) / "raw_baseline_calibrated_test.jsonl"

with parameter_path.open("r", encoding="utf-8") as file:
    parameters = json.load(file)

temperature = float(parameters["temperature"])
predictions = load_jsonl(input_path)
calibrated = apply_temperature(predictions, temperature)
save_jsonl(calibrated, output_path)

print(f"Applied frozen temperature: {temperature:.6f}")
print(f"Test examples: {len(calibrated)}")
print(f"Saved: {output_path}")
'

# 5. Question-aware semantic verifier V2
run_if_missing \
    "$PRED_DIR/test_with_question_aware_semantic_v2.jsonl" \
    python -m src.verification.question_answer_nli_verifier_v2 \
        --input "$PRED_DIR/raw_baseline_calibrated_test.jsonl" \
        --output "$PRED_DIR/test_with_question_aware_semantic_v2.jsonl" \
        --batch-size 4

# 6. Self-verification
run_if_missing \
    "$PRED_DIR/test_with_question_aware_v2_and_self_verification.jsonl" \
    python -m src.verification.self_verifier \
        --input "$PRED_DIR/test_with_question_aware_semantic_v2.jsonl" \
        --output "$PRED_DIR/test_with_question_aware_v2_and_self_verification.jsonl" \
        --batch-size 4

# 7. Final deterministic nested held-out evaluation

if [[ "$LIMIT" == "3000" ]]; then
    SAMPLE_SIZES="200,500,1000,2000,3000"
else
    SAMPLE_SIZES="$LIMIT"
fi

echo "Running deterministic evaluation for sample sizes: $SAMPLE_SIZES"

python experiments/compare_question_aware_ablation_sample_sizes.py     --input "$PRED_DIR/test_with_question_aware_v2_and_self_verification.jsonl"     --output-dir "$FINAL_EVAL_DIR"     --sample-sizes "$SAMPLE_SIZES"     --seed 17     --save-subsets


# 8. Paired bootstrap uncertainty analysis
echo
echo "Running paired bootstrap AURC uncertainty analysis..."

python experiments/bootstrap_aurc_uncertainty.py \
    --input "$PRED_DIR/test_with_question_aware_v2_and_self_verification.jsonl" \
    --output-dir "$FINAL_EVAL_DIR/bootstrap" \
    --bootstrap-samples 5000 \
    --seed 17 \
    --order-seed 17


# 9. Final publication-strength analyses
if [[ "$LIMIT" == "3000" ]]; then
    echo
    echo "Running native SQuAD2 no-answer baseline..."
    python -m src.baselines.native_no_answer_baseline \
        --split test \
        --limit "$LIMIT" \
        --device "$DEVICE"

    python -m src.evaluation.evaluate_native_no_answer_baseline

    echo
    echo "Running verifier signals on calibration split for fusion tuning..."
    run_if_missing \
        "$PRED_DIR/calibration_with_question_aware_semantic_evidence_v2.jsonl" \
        python -m src.verification.question_answer_nli_verifier_v2 \
            --input "$PRED_DIR/raw_baseline_calibrated_calibration.jsonl" \
            --output "$PRED_DIR/calibration_with_question_aware_semantic_evidence_v2.jsonl" \
            --batch-size 4

    run_if_missing \
        "$PRED_DIR/calibration_with_question_aware_v2_and_self_verification.jsonl" \
        python -m src.verification.self_verifier \
            --input "$PRED_DIR/calibration_with_question_aware_semantic_evidence_v2.jsonl" \
            --output "$PRED_DIR/calibration_with_question_aware_v2_and_self_verification.jsonl" \
            --batch-size 4

    echo
    echo "Tuning fusion weights on calibration split only..."
    PYTHONPATH=. python experiments/tune_fusion_weights.py

    echo
    echo "Evaluating held-out calibration quality..."
    python -m src.calibration.calibration_metrics \
        --input "$PRED_DIR/raw_baseline_with_confidence_test.jsonl" \
        --output "$TABLE_DIR/test_calibration_before.json" \
        --bins 10

    python -m src.calibration.calibration_metrics \
        --input "$PRED_DIR/raw_baseline_calibrated_test.jsonl" \
        --output "$TABLE_DIR/test_calibration_after.json" \
        --bins 10

    echo
    echo "Running rank-flip analysis..."
    PYTHONPATH=. python experiments/analyze_rank_flips.py
fi

echo
echo "========================================"
echo "FINAL EXPERIMENT PIPELINE COMPLETE"
echo "========================================"
echo "Final predictions:"
echo "$PRED_DIR/test_with_question_aware_v2_and_self_verification.jsonl"
echo
echo "Final evaluation:"
echo "$FINAL_EVAL_DIR"
