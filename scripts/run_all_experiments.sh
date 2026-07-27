#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
    source ".venv/bin/activate"
fi

export PYTHONPATH="$PROJECT_ROOT"

INPUT_FILE="outputs/predictions/calibration_with_question_aware_semantic_evidence_v2.jsonl"

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: required input file not found:"
    echo "$INPUT_FILE"
    exit 1
fi

RECORD_COUNT=$(wc -l < "$INPUT_FILE" | tr -d ' ')

CANDIDATE_SIZES=(200 500 1000 2000 5000)
AVAILABLE_SIZES=()

for SIZE in "${CANDIDATE_SIZES[@]}"; do
    if (( SIZE <= RECORD_COUNT )); then
        AVAILABLE_SIZES+=("$SIZE")
    fi
done

if (( ${#AVAILABLE_SIZES[@]} == 0 )); then
    AVAILABLE_SIZES=("$RECORD_COUNT")
fi

SAMPLE_SIZES=$(IFS=,; echo "${AVAILABLE_SIZES[*]}")

echo "Found $RECORD_COUNT processed prediction records."
echo "Evaluating sample sizes: $SAMPLE_SIZES"
echo "Running question-aware ablation evaluation..."

python experiments/compare_question_aware_ablation_sample_sizes.py \
    --input "$INPUT_FILE" \
    --sample-sizes "$SAMPLE_SIZES" \
    --save-subsets

echo
echo "All available experiments completed successfully."
