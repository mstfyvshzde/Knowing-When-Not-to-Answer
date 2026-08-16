# Common development and validation commands for the project.
#
# These targets keep local validation aligned with the repository's
# scientific-code and continuous-integration checks.


.PHONY: lint test shell-check check


# ---------------------------------------------------------------------------
# Linting
# ---------------------------------------------------------------------------

# make lint
#
# Runs Ruff across the complete repository, including src/, tests/,
# experiments/, and Python scripts.
lint:
	python -m ruff check .


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# make test
#
# Runs the complete pytest suite.
# Coverage settings are defined centrally in pyproject.toml.
test:
	python -m pytest -q


# ---------------------------------------------------------------------------
# Shell-script syntax
# ---------------------------------------------------------------------------

# make shell-check
#
# Checks Bash scripts for syntax errors without executing experiments
# or modifying scientific artifacts.
shell-check:
	bash -n scripts/*.sh


# ---------------------------------------------------------------------------
# Full repository validation
# ---------------------------------------------------------------------------

# make check
#
# Runs the non-destructive validation suite used before committing:
#
#   1. Ruff linting
#   2. Pytest + coverage
#   3. Bash syntax validation
#
# It intentionally does not auto-format source files or rerun expensive
# model inference.
check: lint test shell-check