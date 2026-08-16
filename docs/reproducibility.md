# Reproducibility

## Reference Environment

The final experiments were executed with:

- Python **3.12.13**
- exact package versions in `requirements-lock.txt`
- supported dependency ranges in `requirements.txt`
- project configuration in `pyproject.toml`

The exact reference environment can be recreated with:

```bash
pip install -r requirements-lock.txt
```

## Dataset and Split Control

The project uses SQuAD v2. The original validation data is divided into separate calibration and held-out test partitions using a deterministic 50/50 stratified split with seed **17**.

Calibration data is used for temperature scaling, calibration diagnostics, and auxiliary fusion-weight tuning. The learned temperature and selected fusion parameters are frozen before held-out test evaluation. Test labels are not used to fit calibration parameters, select score-combination weights, or revise verifier rules.

Dataset preparation is automated with:

```bash
bash scripts/download_dataset.sh
```

## Deterministic Final Evaluation

The held-out evaluation uses **3,000** test examples.

A deterministic ordering with seed **17** defines nested subsets:

```text
200 < 500 < 1000 < 2000 < 3000
```

The same ordering is reused across all methods and sample sizes. Score ties are resolved by original deterministic index.

## Statistical Uncertainty

The final comparison uses **5,000 paired bootstrap resamples**, bootstrap seed **17**, evaluation-order seed **17**, and 95% percentile confidence intervals.

Bootstrap analysis is performed only after final predictions are fixed and is used to quantify uncertainty rather than tune the system.

## Reproduction Commands

Run the final experiment pipeline with:

```bash
DEVICE=cpu LIMIT=3000 bash scripts/run_all_experiments.sh
```

Run the complete reproducibility workflow with:

```bash
DEVICE=cpu LIMIT=3000 bash scripts/reproduce_results.sh
```

`DEVICE` can be changed to `mps` or `cuda` when supported by the local PyTorch installation.

## Saved Artifacts

Final lightweight evaluation artifacts are stored under:

```text
outputs/evaluation/final_sample_size_comparison/
```

Large intermediate predictions and reproducible nested `subset.jsonl` files are intentionally excluded from version control.

## Quality Checks

Before a release or paper freeze, repository quality is validated with `make check`, which runs linting, tests, and shell-script syntax checks. Exact test-pass and coverage counts should be recorded from the final validation run rather than hard-coded while the repository is still being modified.

## Scope

Exact numerical reproduction can still depend on hardware, PyTorch backend behavior, and external model availability. The repository therefore records the software environment, deterministic seeds, model identifiers, execution scripts, and final evaluation artifacts needed to audit the reported results.
