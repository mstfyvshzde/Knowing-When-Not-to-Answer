#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT"

DEVICE="${DEVICE:-cpu}"
LIMIT="${LIMIT:-3000}"

echo "========================================"
echo "Knowing When Not to Answer"
echo "Full reproducibility pipeline"
echo "========================================"
echo "Device: $DEVICE"
echo "Examples per split: $LIMIT"
echo

echo "[1/3] Preparing dataset..."
bash scripts/download_dataset.sh

echo
echo "[2/3] Running final experiments..."
DEVICE="$DEVICE" LIMIT="$LIMIT" bash scripts/run_all_experiments.sh

echo
echo "[3/3] Running quality checks..."
python -m ruff check .
python -m pytest -q

echo
echo "========================================"
echo "REPRODUCTION COMPLETED SUCCESSFULLY"
echo "========================================"
echo "Final evaluation directory:"
echo "outputs/evaluation/final_sample_size_comparison"
