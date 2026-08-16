# Reproducibility

## Reproducibility Goal

This repository is designed to make the final selective-QA experiments
auditable and reproducible from the recorded dataset protocol, model
identifiers, configuration files, deterministic seeds, scripts, and retained
evaluation artifacts.

The project distinguishes between:

- exact reference-environment reproduction;
- deterministic experimental design;
- lightweight tracked result artifacts;
- large intermediate prediction files that can be regenerated locally.

## Reference Environment

The reference environment uses:

- Python **3.12.13**
- exact package versions in `requirements-lock.txt`
- supported dependency ranges in `requirements.txt`
- project and test configuration in `pyproject.toml`

For the closest reproduction of the reference environment, install:

```bash
pip install -r requirements-lock.txt
```

For a compatible development environment using supported version ranges:

```bash
pip install -r requirements.txt
```

Exact numerical behavior can still vary slightly across hardware and PyTorch
backends.

## Dataset

The project uses **SQuAD v2**.

Dataset preparation is automated with:

```bash
bash scripts/download_dataset.sh
```

The source validation set is divided into separate calibration and held-out test
partitions using a deterministic:

```text
50/50 stratified split
seed = 17
```

Stratification preserves answerability proportions.

The split is created before final parameter selection and held-out evaluation.

## Calibration / Test Separation

The calibration partition is used for procedures that require parameter
selection before final test evaluation.

These include:

- temperature-scaling fitting;
- calibration diagnostics;
- auxiliary fusion-weight tuning.

The held-out test partition is reserved for final evaluation after these
parameters and experimental rules are fixed.

Held-out test labels are not used to:

- fit the temperature parameter;
- select fusion weights;
- revise verifier prompts;
- revise claim-validation rules;
- choose final ranking formulas;
- select bootstrap parameters;
- alter the deterministic evaluation ordering.

This separation is an important leakage control.

## Temperature Scaling

Temperature scaling is fitted on the calibration partition only.

The fitted temperature used by the final experiment is approximately:

```text
T = 4.604539
```

After fitting, the temperature is frozen before application to held-out test
predictions.

Temperature scaling is monotonic and therefore does not change the
confidence-only ranking order.

It changes the probability scale rather than the ordering used to compute
confidence-only AURC.

## Calibration-Only Fusion Tuning

Weighted geometric fusion is evaluated separately on the calibration
partition.

The searched rule is:

```text
score =
    confidence^alpha
    * verifier_score^(1 - alpha)
```

with:

```text
alpha = 0.00, 0.01, ..., 1.00
```

The optimization target is calibration-set AURC.

No fusion weight is selected using held-out test labels.

For both verifier combinations, the calibration-only search selects:

```text
alpha = 1.00
```

corresponding to the confidence-only endpoint.

This tuning experiment is auxiliary to the predefined equal-weight fusion
methods used in the primary held-out comparison.

## Deterministic Held-Out Evaluation

The final evaluation uses:

```text
N = 3000
```

held-out examples.

A single deterministic shuffled ordering with seed `17` defines the nested
evaluation subsets:

```text
200 < 500 < 1000 < 2000 < 3000
```

Each larger evaluation set extends the same ordering rather than drawing an
independent sample.

The same examples and ordering are used across all five ranking methods.

## Deterministic Ranking

Predictions are sorted by ranking score in descending order.

When scores are tied, the original deterministic evaluation index is used as
the secondary key:

```text
higher score first
then smaller original index
```

This is particularly important for methods that can produce large tied score
groups.

## Canonical Correctness

The final forced-answer selective-ranking evaluation uses:

```text
src.calibration.calibration_metrics.is_prediction_correct
```

as the canonical correctness implementation.

Under this definition:

- an answerable prediction is correct when its normalized text exactly matches
  at least one normalized reference answer;
- an unanswerable forced-answer candidate is incorrect.

This prevents punctuation-only or normalization-empty forced answers from being
mistaken for correct abstentions.

For the final 3,000-example prediction set:

```text
correct   = 1267
incorrect = 1733
accuracy  = 0.422333
```

All five primary ranking methods operate over the same forced-answer QA
candidates and therefore share this full-coverage accuracy.

## Primary Reproduction Command

Run the canonical experiment pipeline with:

```bash
DEVICE=cpu LIMIT=3000 bash scripts/run_all_experiments.sh
```

CPU is the reference execution mode for the canonical repository workflow.

When supported by the local PyTorch installation, other devices can be used
for exploratory reproduction, for example:

```text
mps
cuda
```

Hardware changes can introduce small numerical differences, so CPU remains the
preferred mode when auditing the reference experiment.

## Repository-Level Reproduction

Run the broader repository reproduction workflow with:

```bash
DEVICE=cpu LIMIT=3000 bash scripts/reproduce_results.sh
```

This workflow coordinates dataset preparation and the main experimental
stages.

Individual scripts can also be executed separately when auditing a particular
stage.

## Canonical Final Error Analysis

The final verifier error-analysis summary is generated from the saved held-out
prediction file with:

```bash
python -m src.analysis.generate_final_error_analysis
```

Input:

```text
outputs/predictions/
test_with_question_aware_v2_and_self_verification.jsonl
```

Output:

```text
outputs/analysis/final_error_analysis/summary.json
```

The generator imports the same canonical correctness function used by the final
evaluation.

It performs no model inference and does not tune any experimental parameter.

This keeps the diagnostic summary synchronized with the final evaluation
semantics.

## Bootstrap Reproduction

The final N=3,000 comparison uses:

```text
bootstrap samples      = 5000
bootstrap seed         = 17
evaluation-order seed  = 17
confidence interval    = 95% percentile
```

Bootstrap resampling is paired across methods.

The analysis is run only after predictions and ranking rules are fixed.

It quantifies uncertainty and is not used for model or parameter selection.

## Rank Diagnostics

The final rank diagnostic uses the same deterministic seed-17 evaluation
ordering.

Correct-incorrect prediction pairs are analyzed using tie-aware comparisons
rather than arbitrary integer ranks inside tied score groups.

This is important because some verifier-based scores create substantial tied
groups.

The rank analysis is diagnostic only and does not modify final evaluation
scores.

## Retained Artifacts

Canonical lightweight evaluation results are retained under:

```text
outputs/evaluation/final_sample_size_comparison/
```

Additional compact analysis and parameter artifacts are retained under
locations such as:

```text
outputs/tables/
outputs/analysis/
```

Large intermediate prediction files and reproducible nested subset files can be
excluded from version control when they can be regenerated from the recorded
pipeline.

Tracked compact artifacts are intended to provide enough information to audit
the reported final results without committing every large intermediate file.

## Quality Checks

Before a release or paper freeze, run the repository quality checks from the
project root:

```bash
make check
```

The equivalent individual commands are:

```bash
python -m ruff check .
python -m pytest -q
bash -n scripts/*.sh
```

A repository-diff whitespace check should also be run before the final commit:

```bash
git diff --check
```

Exact test-pass and coverage counts should be recorded only after the final
repository validation run.

They are intentionally not hard-coded here while the repository is still being
cleaned and validated.

## Configuration Audit

The final protocol is recorded across configuration files including:

```text
configs/base.yaml
configs/dataset.yaml
configs/evaluation.yaml
configs/model.yaml
configs/verification.yaml
```

These files document important settings such as:

- dataset split behavior;
- model identifiers;
- calibration semantics;
- verifier thresholds;
- ranking methods;
- nested evaluation sizes;
- bootstrap settings;
- deterministic seeds.

Configuration values should remain consistent with the executable scripts and
reported final artifacts.

## Randomness Control

The final protocol uses seed `17` for the principal deterministic operations,
including:

- dataset splitting;
- held-out evaluation ordering;
- nested sample construction;
- bootstrap configuration where specified.

The same recorded ordering is reused across compared ranking methods.

This avoids introducing method-specific sampling differences into the final
comparison.

## External Model Dependencies

The experiment depends on pretrained model resources including:

```text
deepset/roberta-base-squad2
domenicrosati/QA2D-t5-base
FacebookAI/roberta-large-mnli
```

Reproduction therefore requires access to the corresponding model files unless
they are already available in a local cache.

External model hosting and availability are outside the control of this
repository.

## Hardware and Numerical Reproducibility

Deterministic seeds reduce experimental randomness but do not guarantee
bit-for-bit equality across all execution environments.

Potential sources of small numerical differences include:

- CPU architecture;
- GPU or Apple MPS execution;
- PyTorch backend behavior;
- low-level numerical kernels;
- dependency versions;
- external model revisions.

For this reason, the repository records both exact reference package versions
and final lightweight result artifacts.

The experimental conclusions should be audited using the complete set of
metrics and ranking relationships rather than requiring bit-identical floating
point values on every platform.

## Reproducibility Boundary

The reproducibility claims in this repository apply to the implemented
experimental setting:

- SQuAD v2;
- the recorded deterministic split;
- one pretrained extractive QA backbone;
- the recorded QA2D model;
- the shared MNLI verifier backbone;
- the implemented confidence and verification scores;
- the fixed primary fusion rules;
- calibration-only parameter fitting;
- the final forced-answer ranking protocol.

They do not imply that the results reproduce universally across different
datasets, QA architectures, verifier models, or selective-prediction systems.