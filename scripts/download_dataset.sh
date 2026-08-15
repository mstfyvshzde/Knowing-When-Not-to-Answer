#!/usr/bin/env bash

# Prepare the SQuAD v2 data used by the project.
#
# The dataset pipeline has two stages:
#
#   1. download the source SQuAD v2 data;
#   2. deterministically prepare the project's calibration and held-out
#      evaluation splits.
#
# Dataset preparation does not train or fine-tune the QA model. It only creates
# the data artifacts consumed by later confidence-calibration, verification,
# and selective-QA experiments.


# Stop immediately if a command fails, an undefined variable is referenced, or
# a command inside a pipeline fails. Dataset preparation should not continue
# after a partial upstream failure.
set -euo pipefail


# Resolve paths relative to the repository rather than the user's current
# working directory, so this script can be launched from anywhere.
PROJECT_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"

cd "$PROJECT_ROOT"


# Make repository modules importable by the Python commands below without
# requiring a global package installation.
export PYTHONPATH="$PROJECT_ROOT"


RAW_DIR="data/raw/squad_v2"
PROCESSED_DIR="data/processed/squad_v2"


echo "========================================"
echo "Knowing When Not to Answer"
echo "Dataset preparation"
echo "========================================"


# ---------------------------------------------------------------------------
# 1. Raw dataset
# ---------------------------------------------------------------------------
#
# `src.data.download_data` is responsible only for obtaining the source dataset.
# No calibration/test partitioning or model-dependent processing occurs here.
#
# Directory existence is used as a lightweight resume check. It does not prove
# that an interrupted or manually modified dataset is complete; remove the raw
# directory before a clean reproduction if its integrity is uncertain.

if [[ -d "$RAW_DIR" ]]; then

    echo \
        "Raw SQuAD v2 dataset already exists: " \
        "$RAW_DIR"

    echo "Skipping download."

else

    echo "Downloading SQuAD v2..."

    python -m src.data.download_data

fi


echo


# ---------------------------------------------------------------------------
# 2. Deterministic project splits
# ---------------------------------------------------------------------------
#
# `src.data.prepare_data` converts the raw dataset into the project's processed
# calibration and held-out evaluation artifacts.
#
# The calibration split is later allowed to fit quantities such as temperature
# and fusion weights. The held-out split is reserved for final evaluation.
# Keeping this separation fixed is part of the project's leakage-prevention
# protocol.
#
# As above, directory existence is only a resume shortcut. For a clean
# end-to-end reproduction after preprocessing logic changes, remove the
# processed directory and regenerate it.

if [[ -d "$PROCESSED_DIR" ]]; then

    echo \
        "Processed dataset already exists: " \
        "$PROCESSED_DIR"

    echo "Skipping preprocessing."

else

    echo \
        "Preparing deterministic calibration " \
        "and held-out evaluation splits..."

    python -m src.data.prepare_data

fi


echo
echo "Dataset setup completed successfully."
echo "Raw data:       $RAW_DIR"
echo "Processed data: $PROCESSED_DIR"