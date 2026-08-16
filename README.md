![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![Research](https://img.shields.io/badge/Research-Selective%20QA-orange)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen)

# Knowing When Not to Answer

[![Python Quality Checks](https://github.com/mstfyvshzde/Knowing-When-Not-to-Answer/actions/workflows/tests.yml/badge.svg)](https://github.com/mstfyvshzde/Knowing-When-Not-to-Answer/actions/workflows/tests.yml)

> Research software for selective extractive question answering, confidence
> calibration, abstention, and NLI-based answer-support verification.

---

## Overview

**Knowing When Not to Answer** studies selective question answering with a
pretrained extractive QA model.

The central question is:

> Does adding semantic answer-verification signals improve selective ranking
> over a strong confidence baseline?

The final experiments use the pretrained
`deepset/roberta-base-squad2` QA model on SQuAD v2.

Rather than training a new QA model, the project studies how existing QA
candidates should be ranked according to:

- calibrated QA confidence,
- question-aware semantic verification,
- answer-support NLI verification,
- and fixed confidence-verifier fusions.

Selective performance is evaluated primarily with
**Area Under the Risk-Coverage Curve (AURC)**, where lower values are better.

The repository also evaluates the QA model's native no-answer behavior as a
separate operational abstention baseline.

## Main Finding

On the final held-out evaluation with 3,000 examples, **confidence-only ranking
achieved the lowest AURC** among the five evaluated ranking methods.

The result supports a deliberately narrow conclusion:

> Adding semantic or answer-support verification signals does not necessarily
> improve selective QA ranking over a strong confidence baseline.

This result should not be interpreted as evidence that verification is
universally ineffective. It applies to the models, signals, benchmark, and
evaluation protocol studied in this repository.

## Research Scope

This project focuses on:

- selective extractive question answering,
- answer-vs-null confidence estimation,
- temperature scaling,
- NLI-based answer-support signals,
- risk-coverage evaluation,
- abstention behavior,
- paired bootstrap uncertainty,
- and ranking diagnostics.

It is **not** a general-purpose LLM safety framework.

The QA backbone is pretrained and is not fine-tuned as part of the final
experimental pipeline.

## Experimental Protocol

### Dataset

The benchmark is **SQuAD v2** using the Hugging Face dataset:

```text
rajpurkar/squad_v2
```

Because public gold labels for the official SQuAD v2 test set are unavailable,
the original validation split is deterministically divided into:

- a calibration split,
- a held-out test split.

The partition is:

- 50% calibration,
- 50% held-out test,
- stratified by answerability,
- seed `17`.

The calibration split may be used to fit parameters.

The held-out test split is reserved for final evaluation.

Test labels are not used to tune:

- temperature,
- fusion weights,
- thresholds,
- verifier prompts,
- or ranking rules.

### Final Held-Out Sample

The canonical final evaluation uses:

```text
N = 3000
```

A single deterministic seed-17 ordering also defines nested subsets:

```text
200 ⊂ 500 ⊂ 1000 ⊂ 2000 ⊂ 3000
```

The smaller evaluations are therefore prefixes of one fixed order rather than
independent random samples.

## Models and Ranking Signals

### QA Backbone

```text
deepset/roberta-base-squad2
```

The same pretrained QA backbone is used for the main forced-answer experiments
and the native no-answer baseline.

### Confidence

The confidence estimator derives an answer-vs-null logit margin from the QA
model.

Temperature scaling is then fitted on the calibration split only.

### Question-Aware Semantic Verifier V2

The question-aware verifier first converts the question-answer pair into a
declarative claim using:

```text
domenicrosati/QA2D-t5-base
```

The generated claim is then evaluated against the context with:

```text
FacebookAI/roberta-large-mnli
```

The ranking signal is the NLI entailment probability.

Invalid generated claims receive a ranking score of `0.0`.

### Answer-Support / Self Verifier

The historically named **self-verifier** constructs a fixed claim containing
the question and proposed answer and evaluates that claim against the context.

It also uses:

```text
FacebookAI/roberta-large-mnli
```

Therefore, the two verifier components should **not** be interpreted as
independent verifier models: they share the same NLI backbone.

The raw self-verification score is based on:

```text
entailment probability - contradiction probability
```

and is normalized from `[-1, 1]` into `[0, 1]` before final ranking.

## Final Ranking Methods

Five ranking methods are evaluated:

| Method | Ranking score |
|---|---|
| Confidence only | calibrated confidence |
| Question-aware semantic V2 | QA entailment probability |
| Confidence + question-aware semantic V2 | `sqrt(confidence × QA entailment)` |
| Self-verifier only | normalized self-verification score |
| Confidence + self-verifier | `sqrt(confidence × normalized self-verification)` |

The two reported fusions use fixed equal-weight geometric means.

Fusion-weight tuning is performed separately on the calibration split and does
not use held-out test labels.

## Correctness and Ranking

The final five-method comparison evaluates different rankings of the same
forced-answer QA candidates.

Candidate correctness is fixed before ranking:

- an answerable example is correct when the predicted answer matches a gold
  answer under the project's normalized Exact Match definition;
- an unanswerable example with a forced non-empty answer is incorrect.

Changing a ranking score changes the **order** in which candidates are accepted;
it does not change their correctness labels.

This is why all five ranking methods have the same full-sample candidate
accuracy while producing different risk-coverage curves.

## Experimental Results

### Final Held-Out Evaluation — N=3000

Lower AURC and normalized AURC are better.

| Method | AURC | Normalized AURC | Δ AURC vs. Confidence | 95% Paired Bootstrap CI for Δ |
|---|---:|---:|---:|---:|
| **Confidence only** | **0.292379** | **0.216109** | — | — |
| Confidence + self-verifier | 0.309120 | 0.262110 | +0.016741 | [0.008561, 0.024773] |
| Confidence + question-aware semantic V2 | 0.339417 | 0.345358 | +0.047038 | [0.036312, 0.057263] |
| Question-aware semantic V2 | 0.394644 | 0.497105 | +0.102265 | [0.086357, 0.118286] |
| Self-verifier only | 0.433892 | 0.604947 | +0.141513 | [0.124280, 0.159087] |

The final candidate accuracy is:

```text
1267 / 3000 = 0.422333
```

All four paired AURC-difference confidence intervals are above zero on the final
N=3000 evaluation.

Positive Δ AURC means that the alternative method performs worse than
confidence-only because lower AURC is better.

![Final N=3000 ablation results](assets/figures/ablation_results.png)

### Nested Sample-Size Evaluation

Confidence-only ranking achieved the lowest AURC at:

- N=500,
- N=1000,
- N=2000,
- N=3000.

At N=200, confidence + self-verifier was slightly better:

```text
Confidence + self-verifier: 0.284548
Confidence only:            0.292532
```

The nested result therefore supports the final conclusion at larger sample
sizes while also showing that small-sample rankings can vary.

![AURC by sample size](outputs/evaluation/final_sample_size_comparison/aurc_by_sample_size.png)

![Normalized AURC by sample size](outputs/evaluation/final_sample_size_comparison/normalized_aurc_by_sample_size.png)

## Paired Bootstrap Uncertainty

The final uncertainty analysis uses:

```text
Bootstrap replicates: 5000
Bootstrap seed:       17
Ordering seed:        17
```

Every bootstrap replicate samples one set of example indices and evaluates all
five ranking methods on that same sample.

This paired design directly estimates uncertainty in AURC differences relative
to confidence-only ranking.

The confidence-only AURC itself has a 95% bootstrap interval of:

```text
[0.273114, 0.312520]
```

## Confidence Calibration

Temperature scaling is fitted **only on the calibration split**.

The learned temperature is:

```text
T = 4.604539
```

On the calibration split, negative log-likelihood changed from:

```text
Before temperature scaling: 0.966913
After temperature scaling:  0.412469
```

The fitted temperature is then frozen before application to held-out test
confidence values.

Temperature scaling is monotonic for a positive temperature, so it improves
probability calibration without changing the confidence-only ranking order.
Its scale can still matter when calibrated confidence is combined numerically
with verifier scores.

![Calibration curve](assets/figures/calibration_curve.png)

## Native No-Answer Baseline

The repository separately evaluates the QA model using its native SQuAD v2
no-answer behavior.

Unlike the main forced-answer ranking experiment, this baseline can directly
produce:

```text
ANSWER
ABSTAIN
```

This comparison helps distinguish two questions:

1. How well does the pretrained model natively decide whether to abstain?
2. How well can forced-answer candidates be ordered for selective prediction?

The native-baseline lexical Exact Match and token F1 values use this
repository's evaluation convention and should not be presented as official
SQuAD v2 leaderboard EM/F1 scores.

## Rank Diagnostics

The final pipeline includes a tie-aware correct-vs-incorrect pair analysis.

For every pair containing one correct and one incorrect candidate, the analysis
asks whether adding a verifier signal:

- strictly fixes the ordering,
- strictly harms the ordering,
- turns an ordering into a tie,
- resolves a tie,
- or leaves the relation unchanged.

Verifier-score ties are treated as genuine ties during this diagnostic rather
than being converted into artificial wins or losses by an arbitrary record
index.

The diagnostic is explanatory only and does not tune parameters on the held-out
test set.

## Historical Decision-Engine Prototype

Earlier versions of the project explored an explicit:

```text
ANSWER / VERIFY / ABSTAIN
```

decision engine.

That component remains in the repository for historical and prototype
experiments, but it is **not the final held-out evaluation protocol** reported
above.

The final experiment compares ranking signals directly through risk-coverage
evaluation.

## Final Pipeline

The canonical scientific workflow is:

```text
SQuAD v2
    │
    ├── deterministic calibration split
    │       │
    │       ├── forced-answer QA candidates
    │       ├── answer-vs-null confidence
    │       ├── temperature fitting
    │       └── verifier signals for calibration-only parameter selection
    │
    └── held-out test split
            │
            ├── forced-answer QA candidates
            ├── answer-vs-null confidence
            ├── frozen temperature application
            ├── question-aware semantic verification
            ├── answer-support / self verification
            │
            ├── five ranking methods
            │       │
            │       ├── risk-coverage curves
            │       ├── AURC
            │       ├── normalized AURC
            │       └── matched-coverage evaluation
            │
            ├── paired bootstrap uncertainty
            └── tie-aware ranking diagnostics
```

The native no-answer baseline is evaluated separately from this forced-answer
ranking pipeline.

## Repository Structure

```text
Knowing-When-Not-to-Answer/
│
├── .github/
│   └── workflows/              # Continuous-integration checks
│
├── assets/
│   └── figures/                # Publication-facing and documentation figures
│
├── configs/                    # Experiment and protocol configuration
├── data/                       # Raw and processed dataset locations
├── docs/                       # Research documentation and analysis notes
├── experiments/                # Evaluation and statistical-analysis entry points
├── notebooks/                  # Exploratory notebooks
├── outputs/
│   ├── analysis/               # Diagnostic summaries and intermediate analysis
│   ├── evaluation/             # Canonical evaluation artifacts
│   ├── logs/                   # Runtime logs
│   ├── predictions/            # Large generated prediction artifacts
│   └── tables/                 # Lightweight experiment summaries
│
├── scripts/                    # Reproduction and experiment orchestration
│
├── src/
│   ├── analysis/               # Analysis utilities
│   ├── baselines/              # Forced-answer and native no-answer baselines
│   ├── calibration/            # Temperature scaling and calibration metrics
│   ├── data/                   # Dataset download and deterministic preparation
│   ├── decision/               # Historical/prototype routing components
│   ├── evaluation/             # Metrics and final selective evaluation
│   ├── utils/                  # Shared utilities
│   └── verification/           # Confidence and NLI verification modules
│
├── tests/                      # Automated test suite
│
├── CHANGELOG.md
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── environment.yml
├── LICENSE
├── Makefile
├── pyproject.toml
├── requirements-lock.txt
├── requirements.txt
├── SECURITY.md
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/mstfyvshzde/Knowing-When-Not-to-Answer.git
cd Knowing-When-Not-to-Answer
```

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install runtime dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install development tools when running repository checks locally:

```bash
python -m pip install pytest pytest-cov ruff
```

The project targets Python 3.12.

`requirements.txt` defines supported dependency ranges, while
`requirements-lock.txt` records the reference package snapshot used for
reproducibility.

## Reproducibility

### 1. Prepare the Dataset

```bash
bash scripts/download_dataset.sh
```

This downloads SQuAD v2 and creates the deterministic calibration and held-out
splits.

Dataset preparation does not train or fine-tune the QA model.

### 2. Run the Canonical Experiment

```bash
DEVICE=cpu LIMIT=3000 bash scripts/run_all_experiments.sh
```

The canonical defaults are:

```text
DEVICE=cpu
LIMIT=3000
ORDER_SEED=17
BOOTSTRAP_SAMPLES=5000
```

`DEVICE` may be changed to `cuda` or `mps` when supported by the local PyTorch
installation.

Smaller `LIMIT` values are intended for smoke testing and are written to
separate output directories.

### 3. Run the End-to-End Reproduction Workflow

```bash
DEVICE=cpu LIMIT=3000 bash scripts/reproduce_results.sh
```

This performs:

1. dataset preparation,
2. the final scientific experiment pipeline,
3. repository-level linting,
4. the automated test suite.

### 4. Run Repository Checks Only

```bash
make check
```

The repository checks include:

```text
Ruff linting
pytest
Bash syntax validation
```

### Resume Behavior

The experiment scripts can skip expensive generated artifacts when an existing
file has the expected number of records.

This is a convenience mechanism, not cryptographic provenance.

After changing model, preprocessing, verification, or evaluation logic, stale
generated artifacts should be removed before performing a clean reproduction.

## Output Artifacts

Canonical lightweight held-out results are stored under:

```text
outputs/evaluation/final_sample_size_comparison/
```

Important final artifacts include:

```text
sample_size_comparison.csv

n_3000/
├── ablation_summary.csv
├── matched_coverage.csv
├── risk_coverage_curves.csv
└── risk_coverage_curves.png

bootstrap/
├── bootstrap_aurc_summary.csv
└── bootstrap_aurc_summary.json
```

Large prediction-level JSONL files and deterministic nested `subset.jsonl`
files are intentionally excluded from version control.

Publication-facing figures are stored in:

```text
assets/figures/
```

## Limitations

The conclusions of this repository are intentionally limited to the evaluated
setting.

Important limitations include:

- one primary benchmark: SQuAD v2,
- one pretrained extractive QA backbone,
- no QA fine-tuning in the final experiment,
- verifier signals built around one shared RoBERTa-large-MNLI backbone,
- fixed final equal-weight fusion rules,
- and a held-out evaluation of 3,000 examples.

The results should therefore not be generalized automatically to:

- generative LLM question answering,
- other domains,
- other QA architectures,
- independently trained verifier ensembles,
- or other abstention objectives.

## Contributing

Contributions are welcome.

Please read:

```text
CONTRIBUTING.md
CODE_OF_CONDUCT.md
```

before submitting a Pull Request.

## Citation

If you use this repository or build upon this research artifact, please use the
metadata in:

```text
CITATION.cff
```

Until an academic publication is available, the repository may also be cited
as:

```bibtex
@misc{mustafayev2026knowing,
  author    = {Shahzada Mustafayev},
  title     = {Knowing When Not to Answer},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/mstfyvshzde/Knowing-When-Not-to-Answer}
}
```

## License

This project is released under the **MIT License**.

See `LICENSE` for the full license text.