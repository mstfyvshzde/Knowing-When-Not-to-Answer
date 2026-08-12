# Experimental Design

## Objective

The final experiments evaluate whether semantic or self-verification signals improve selective QA ranking over a calibrated-confidence baseline under a fixed held-out protocol.

## Compared Methods

Five ranking methods are evaluated:

1. **Confidence only**
2. **Question-aware semantic V2**
3. **Confidence + question-aware semantic V2**
4. **Self-verifier only**
5. **Confidence + self-verifier**

The two combined methods use a fixed equal-weight geometric mean:

```text
combined_score = sqrt(score_a * score_b)
```

The combination rule is fixed in advance and is not tuned on held-out test labels.

## Controlled Variables

All primary methods use the same:

- SQuAD v2 calibration/test split
- extractive QA backbone
- underlying QA predictions
- held-out evaluation examples
- correctness labels
- deterministic sample ordering
- tie-breaking rule

The compared methods differ only in the ranking score assigned to each prediction.

## Dataset Partitioning

The original SQuAD v2 validation split is divided into separate calibration and held-out test partitions using a deterministic 50/50 stratified split with seed `17`.

The calibration partition is used to fit temperature scaling.

The held-out test partition is reserved for final evaluation.

## Calibration Control

Temperature scaling is fitted on calibration data only.

The fitted temperature is frozen before application to the held-out test predictions.

Held-out test labels are not used for:

- temperature fitting
- score-combination weight selection
- verifier-rule revision
- prompt revision
- ranking-rule tuning

## Verification Signals

### Question-Aware Semantic V2

Question-answer pairs are converted into declarative claims and evaluated against the supplied context.

Invalid claims receive semantic score `0`.

### Self-Verifier

The self-verification path produces a raw score on `[-1, 1]`, which is mapped to `[0, 1]` as:

```text
normalized_self = (raw_self + 1) / 2
```

The semantic and self-verification paths are treated as separate verification signals, not statistically independent models, because both use the same NLI backbone.

## Final Sample-Size Design

The final held-out evaluation uses 3,000 examples.

A deterministic order with seed `17` defines nested subsets:

```text
200 ⊂ 500 ⊂ 1000 ⊂ 2000 ⊂ 3000
```

Using nested subsets ensures that larger evaluations contain all examples from smaller evaluations.

## Deterministic Ranking

Predictions are ranked by score in descending order.

Score ties are broken by the smaller original deterministic index.

This makes AURC reproducible even when methods produce identical scores.

## Evaluation Metrics

The primary metric is **AURC**, where lower values indicate better selective ranking.

The evaluation also reports:

- normalized AURC
- risk-coverage curves
- matched-coverage selective risk
- matched-coverage selective accuracy

## Statistical Uncertainty

The final 3,000-example comparison uses paired bootstrap analysis with:

- 5,000 resamples
- bootstrap seed `17`
- evaluation-order seed `17`
- 95% percentile confidence intervals

For each non-baseline method:

```text
Delta AURC = method AURC - confidence-only AURC
```

Because lower AURC is better, a positive Delta AURC favors the confidence-only baseline.

Bootstrap analysis is performed after final predictions are fixed and is used for uncertainty quantification, not model tuning.

## Hypothesis and Observed Outcome

The working hypothesis was that adding verification signals could improve the risk-coverage trade-off over confidence alone.

The observed final result does not support that hypothesis in this experimental setting.

Confidence only achieves the lowest AURC at every tested sample size from 200 through 3,000 examples.

This negative result is reported as an experimental finding rather than reframed as a successful combined-verification method.

## Scope

The design supports conclusions only for the evaluated setting: SQuAD v2, the extractive QA backbone, and the verification signals implemented in this repository.

It does not establish a general result for large language models, other datasets, or high-stakes deployment settings.
