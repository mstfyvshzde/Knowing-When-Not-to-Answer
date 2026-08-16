# Contributing

Thank you for your interest in contributing to **Knowing When Not to Answer**.

Contributions that improve the correctness, reproducibility, clarity, testing,
or documentation of the project are welcome.

Because this repository contains research code and reported experimental
results, changes to evaluation logic, data handling, calibration, ranking,
or verification require particular care.

## Development Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the runtime dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the development tools used by the repository checks:

```bash
python -m pip install pytest pytest-cov ruff
```

## How to Contribute

1. Fork the repository.

2. Create a feature or fix branch:

```bash
git checkout -b feature/your-feature
```

or:

```bash
git checkout -b fix/your-fix
```

3. Make the intended changes.

4. Add or update tests when behavior changes.

5. Run the repository checks:

```bash
make check
```

If `make` is unavailable, run the checks directly:

```bash
python -m ruff check .
python -m pytest -q
bash -n scripts/*.sh
```

6. Review the final diff before committing:

```bash
git diff --check
```

7. Commit with a clear and specific message:

```bash
git commit -m "Improve confidence calibration validation"
```

8. Push the branch and open a Pull Request.

## Coding Guidelines

- Follow clear, idiomatic Python conventions.
- Keep functions focused and modular.
- Use meaningful variable and function names.
- Add type hints where they improve clarity.
- Add docstrings for non-trivial functions and research logic.
- Prefer explicit validation over silent fallback behavior.
- Preserve deterministic behavior where the experiment requires it.
- Avoid unrelated refactoring in scientific evaluation code.
- Include tests for bug fixes and behavior changes.
- Update documentation and configuration files when implementation semantics change.

## Research Integrity and Reproducibility

Changes must preserve the distinction between development, calibration, and
held-out evaluation.

In particular:

- Do not use held-out test labels to tune temperatures, thresholds, fusion
  weights, prompts, or other model parameters.
- Do not manually modify final test predictions after inspecting correctness.
- Do not move examples between calibration and test after examining final
  results.
- Preserve documented deterministic ordering and random seeds unless a change
  is explicitly justified.
- Clearly document any change that can alter reported experimental outputs.
- Do not silently change correctness definitions, score normalization,
  tie-breaking rules, or risk-coverage calculations.
- Keep candidate correctness separate from routing-action correctness where
  those concepts differ.
- Treat NLI probabilities as verifier-model class probabilities rather than
  calibrated probabilities that a QA answer is correct.
- Do not describe verification components as statistically independent when
  they share the same underlying model backbone.

If a contribution intentionally changes the experimental protocol, the Pull
Request should explain:

1. what changed,
2. why it changed,
3. which outputs may change,
4. which experiments must be rerun,
5. whether existing reported results remain comparable.

## Tests

Tests should cover the behavior being changed.

For evaluation or metric code, useful cases include:

- answerable and unanswerable examples,
- ANSWER and ABSTAIN decisions,
- empty or malformed predictions,
- score ties,
- score-range validation,
- deterministic ordering,
- missing metadata,
- boundary values.

Run:

```bash
python -m pytest -q
```

before submitting a Pull Request.

## Linting

Ruff is the repository's canonical linting tool.

Run:

```bash
python -m ruff check .
```

Do not introduce a repository-wide formatting change as part of an unrelated
research or bug-fix contribution.

## Shell Scripts

Changes to Bash scripts should at minimum pass syntax validation:

```bash
bash -n scripts/*.sh
```

This command checks syntax without executing expensive model inference.

## Experimental Artifacts

Large datasets and prediction-level artifacts should not normally be committed.

Small canonical summaries, evaluation tables, metadata, and
publication-facing figures may be version-controlled when they are required to
support reported results.

When updating a tracked experimental result:

- identify the script that generated it,
- record the relevant seed and configuration,
- verify that the source prediction artifact is correct,
- avoid replacing canonical outputs with smoke-test or partial-run results.

## Documentation Changes

Documentation should distinguish clearly between:

- the current final evaluation protocol,
- historical or prototype components,
- diagnostic analyses,
- publication-facing results.

Avoid presenting prototype thresholds or routing policies as part of the final
held-out ranking protocol unless they are actually used there.

## Reporting Issues

Please include, where relevant:

- Python version,
- operating system,
- hardware/backend (`cpu`, `cuda`, or `mps`),
- dependency versions,
- command that produced the problem,
- steps to reproduce,
- expected behavior,
- actual behavior,
- traceback or error logs,
- relevant configuration or random seed.

Do not include API keys, credentials, private data, or other secrets in an
issue.

## Feature Requests

Feature requests are welcome.

Please describe:

- the motivation,
- the proposed change,
- possible alternatives,
- expected scientific or engineering impact,
- whether the proposal would alter the current experimental protocol.

## Pull Request Scope

Prefer small, reviewable Pull Requests.

A Pull Request should avoid mixing unrelated changes such as:

- evaluation logic changes,
- repository-wide formatting,
- documentation rewrites,
- dependency upgrades,

unless those changes genuinely depend on one another.

Thank you for helping improve the quality and reproducibility of the project.