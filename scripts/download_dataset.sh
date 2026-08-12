#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT"

RAW_DIR="data/raw/squad_v2"
PROCESSED_DIR="data/processed/squad_v2"

echo "========================================"
echo "Knowing When Not to Answer"
echo "Dataset preparation"
echo "========================================"

if [[ -d "$RAW_DIR" ]]; then
    echo "Raw SQuAD v2 dataset already exists: $RAW_DIR"
    echo "Skipping download."
else
    echo "Downloading SQuAD v2..."
    python -m src.data.download_data
fi

echo

if [[ -d "$PROCESSED_DIR" ]]; then
    echo "Processed dataset already exists: $PROCESSED_DIR"
    echo "Skipping preprocessing."
else
    echo "Preparing calibration and test splits..."
    python -m src.data.prepare_data
fi

echo
echo "Dataset setup completed successfully."
echo "Raw data:       $RAW_DIR"
echo "Processed data: $PROCESSED_DIR"
