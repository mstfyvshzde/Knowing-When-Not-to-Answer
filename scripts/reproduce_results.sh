#!/usr/bin/env bash

# Reproduce the complete project from dataset preparation through final
# experiments and repository-level quality checks.
#
# This script is intentionally a thin orchestration layer. Scientific choices
# such as calibration fitting, frozen test application, verifier execution,
# nested evaluation, bootstrap uncertainty, and rank diagnostics live in
# `scripts/run_all_experiments.sh` and the corresponding Python modules.
#
# Optional environment variables:
#
#   DEVICE=cpu|...   Device forwarded to model-running stages.
#   LIMIT=N          Number of examples processed per split.
#
# The canonical final evaluation uses LIMIT=3000. Smaller LIMIT values are
# useful for smoke tests but should not be reported as final project results.


# Fail immediately when:
# - a command returns a non-zero status (`-e`);
# - an undefined variable is used (`-u`);
# - any command inside a pipeline fails (`pipefail`).
#
# A reproduction run should stop at the first failure rather than continue and
# produce a mixture of complete and incomplete artifacts.
set -euo pipefail


# Resolve the repository root from this script's own location. This makes the
# pipeline independent of the directory from which the user launches it.
PROJECT_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"

cd "$PROJECT_ROOT"


# Make project modules importable by Python commands launched from the shell
# scripts without requiring the package to be installed globally.
export PYTHONPATH="$PROJECT_ROOT"


# Environment overrides make the same pipeline usable for both the canonical
# run and smaller development/smoke runs.
DEVICE="${DEVICE:-cpu}"
LIMIT="${LIMIT:-3000}"


echo "========================================"
echo "Knowing When Not to Answer"
echo "Full reproducibility pipeline"
echo "========================================"
echo "Device: $DEVICE"
echo "Examples per split: $LIMIT"
echo


# Stage 1 prepares the deterministic project dataset artifacts required by all
# later prediction and evaluation steps.
echo "[1/4] Preparing dataset..."
bash scripts/download_dataset.sh


# Stage 2 delegates the complete scientific experiment pipeline:
# raw QA predictions, confidence estimation, calibration-only temperature
# fitting, frozen test calibration, verifier inference, held-out ranking
# evaluation, bootstrap uncertainty, and final diagnostic analyses.
echo
echo "[2/4] Running final experiments..."
DEVICE="$DEVICE" \
LIMIT="$LIMIT" \
bash scripts/run_all_experiments.sh


# Publication-facing figures are regenerated from the canonical retained
# artifacts after the scientific pipeline completes.
echo
echo "[3/4] Regenerating canonical figures..."
python scripts/generate_figures.py


# Reproducibility includes software integrity as well as numerical outputs.
# Ruff checks repository code quality, while pytest verifies the implemented
# invariants and evaluation behavior.
echo
echo "[4/4] Running quality checks..."
python -m ruff check .
python -m pytest -q
bash -n scripts/*.sh


# Reaching this point means every required stage exited successfully because
# `set -euo pipefail` prevents the script from hiding an upstream failure.
echo
echo "========================================"
echo "REPRODUCTION COMPLETED SUCCESSFULLY"
echo "========================================"
echo "Final evaluation directory:"
echo "outputs/evaluation/final_sample_size_comparison"