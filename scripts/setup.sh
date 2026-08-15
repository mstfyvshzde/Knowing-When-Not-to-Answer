#!/usr/bin/env bash

# Create a local Python environment for running the project.
#
# This script handles software setup only. It does not download the dataset,
# generate model predictions, or run experiments.
#
# Optional environment variables:
#
#   PYTHON_BIN=/path/to/python
#       Python interpreter used to create the virtual environment.
#
#   VENV_DIR=/path/to/venv
#       Location of the virtual environment.
#
# Defaults:
#
#   PYTHON_BIN=python3
#   VENV_DIR=.venv


# Stop immediately if a command fails, an undefined variable is referenced,
# or a command inside a pipeline fails. A partially installed environment
# should not be reported as a successful project setup.
set -euo pipefail


# Resolve the repository root from this script's location so setup behaves the
# same way regardless of the directory from which it is launched.
PROJECT_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"

cd "$PROJECT_ROOT"


# These overrides make the setup script usable with another Python installation
# or a virtual-environment location outside the repository when needed.
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"


# ---------------------------------------------------------------------------
# 1. Virtual environment
# ---------------------------------------------------------------------------
#
# Keeping project dependencies inside an isolated environment reduces
# interference from unrelated packages installed on the host machine.

echo \
    "Creating virtual environment in " \
    "$VENV_DIR..."

"$PYTHON_BIN" -m venv \
    "$VENV_DIR"


# Activate the environment so every following pip command targets the newly
# created project environment rather than the system Python installation.
echo "Activating virtual environment..."

source "$VENV_DIR/bin/activate"


# ---------------------------------------------------------------------------
# 2. Package installer
# ---------------------------------------------------------------------------
#
# Upgrade pip inside the virtual environment before installing project
# dependencies. This does not modify the system-wide pip installation.

echo "Upgrading pip..."

python -m pip install \
    --upgrade pip


# ---------------------------------------------------------------------------
# 3. Project dependencies
# ---------------------------------------------------------------------------
#
# Install the dependency set declared by the repository. Environment creation
# is kept separate from experiment execution so a user can inspect or validate
# the software environment before running expensive model inference.

echo "Installing project dependencies..."

python -m pip install \
    -r requirements.txt


echo
echo "Setup completed successfully."
echo "Virtual environment: $VENV_DIR"