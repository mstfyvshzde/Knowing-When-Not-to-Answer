# Evaluation Plan

## Evaluation Goal

The final evaluation tests whether question-aware semantic or answer-support
verification improves selective ranking over a strong confidence baseline.

The primary comparison evaluates five different ranking scores over the same
forced-answer QA candidates.

## Evaluation Data

The project uses SQuAD v2.

The original validation data is divided into separate calibration and held-out
test partitions using:

```text
50/50 stratified split
seed = 17
```

The calibration partition is used for:

- temperature-scaling fitting;
- calibration diagnostics;
- auxiliary fusion-weight tuning.

The held-out test partition is reserved for final evaluation.

## Final Evaluation Set

The final comparison uses:

```text
N = 3000
```

held-out examples.

A deterministic order using seed `17` defines nested subsets:

```text
200 < 500 < 1000 < 2000 < 3000
```

Every larger subset extends the same deterministic ordering.

## Canonical Correctness

The five primary methods rank forced-answer QA candidates.

Correctness is computed using:

```text
src.calibration.calibration_metrics.is_prediction_correct
```

For answerable examples, the normalized prediction must exactly match at least
one normalized reference answer.

For unanswerable examples, the forced-answer candidate is incorrect.

The final N=3,000 candidate set contains:

```text
1267 correct
1733 incorrect
```

All five primary methods therefore have identical full-coverage accuracy:

```text
0.422333
```

## Compared Ranking Methods

Five methods are evaluated:

1. **Confidence only**
2. **Question-aware semantic V2**
3. **Confidence + question-aware semantic V2**
4. **Self-verifier only**
5. **Confidence + self-verifier**

Invalid question-aware claims receive semantic ranking score:

```text
0.0
```

The raw self-verifier score is mapped from `[-1, 1]` into `[0, 1]`:

```text
normalized_self =
    (raw_self + 1) / 2
```

The primary combined methods use:

```text
combined_score =
    sqrt(score_a * score_b)
```

These predefined equal-weight rules are not tuned using held-out test labels.

## Auxiliary Fusion-Weight Analysis

A separate calibration-only analysis searches:

```text
alpha = 0.00, 0.01, ..., 1.00
```

using weighted geometric fusion:

```text
score =
    confidence^alpha
    * verifier_score^(1 - alpha)
```

Calibration AURC is the objective.

The selected weights are frozen before held-out evaluation.

For both verifier signals, calibration-only tuning selects:

```text
alpha = 1.00
```

which is the confidence-only endpoint.

This experiment is auxiliary and does not replace the predefined equal-weight
primary comparison.

## Primary Metric

The principal metric is:

```text
Area Under the Risk-Coverage Curve (AURC)
```

Lower AURC indicates better global selective ranking.

Selective risk is:

```text
incorrect answered examples
/
answered examples
```

Coverage is:

```text
answered examples
/
total examples
```

## Complementary Metrics

The evaluation additionally reports:

- normalized AURC;
- matched-coverage selective accuracy;
- matched-coverage selective risk;
- answered counts;
- abstained counts;
- risk-coverage curves.

These metrics complement rather than replace global AURC.

## Deterministic Ranking

Candidates are sorted by ranking score in descending order.

Ties are resolved deterministically:

```text
higher score first
then smaller original index
```

This rule is particularly important for verifier scores that may create large
tied groups.

## Final N=3,000 Reference Results

The canonical final comparison is:

| Method | AURC | Normalized AURC |
|---|---:|---:|
| Confidence only | **0.292379** | **0.216109** |
| Confidence + self-verifier | 0.309120 | 0.262110 |
| Confidence + question-aware semantic V2 | 0.339417 | 0.345358 |
| Question-aware semantic V2 | 0.394644 | 0.497105 |
| Self-verifier only | 0.433892 | 0.604947 |

Lower values are better.

## Sample-Size Interpretation

Confidence only is best at:

```text
N = 500
N = 1000
N = 2000
N = 3000
```

At N=200:

```text
Confidence + self-verifier = 0.284548
Confidence only            = 0.292532
```

The N=200 result is therefore treated as an explicit exception rather than
claiming that confidence wins at every evaluated sample size.

## Statistical Uncertainty

The final N=3,000 comparison uses:

```text
paired bootstrap resamples = 5000
bootstrap seed             = 17
evaluation-order seed      = 17
confidence interval        = 95% percentile
```

For each non-baseline method:

```text
Delta AURC =
    method AURC
    - confidence-only AURC
```

The canonical paired results are:

| Method | Delta AURC | 95% CI |
|---|---:|---:|
| Confidence + self-verifier | +0.016741 | [0.008561, 0.024773] |
| Confidence + question-aware semantic V2 | +0.047038 | [0.036312, 0.057263] |
| Question-aware semantic V2 | +0.102265 | [0.086357, 0.118286] |
| Self-verifier only | +0.141513 | [0.124280, 0.159087] |

Because lower AURC is better, positive Delta AURC favors confidence only.

Bootstrap analysis is performed after final predictions are fixed and is not a
parameter-selection stage.

## Error Analysis

Canonical final error analysis is regenerated with:

```bash
python -m src.analysis.generate_final_error_analysis
```

It examines:

- QA correctness;
- question-aware NLI labels;
- claim validity;
- invalid-claim reasons;
- self-verification labels;
- high-entailment incorrect cases;
- low-entailment correct cases.

Error analysis is descriptive.

Held-out failures are not used to revise final ranking rules.

## Rank Diagnostics

The final analysis additionally performs tie-aware comparisons of
correct-incorrect candidate pairs.

This diagnostic measures beneficial and harmful strict reorderings introduced
by verifier-based fusion.

It is not used for tuning.

## Leakage Controls

The final protocol enforces the following controls:

- calibration and held-out test partitions are separate;
- temperature scaling is fitted on calibration data only;
- fusion-weight tuning is performed on calibration data only;
- fitted parameters are frozen before test evaluation;
- held-out labels do not select fusion weights;
- held-out labels do not revise verifier prompts;
- held-out errors do not revise claim-validation rules;
- all primary methods use identical held-out candidates;
- sample ordering is deterministic;
- tie-breaking is deterministic;
- bootstrap settings are fixed;
- error and rank analyses are diagnostic.

## Interpretation Rule

The supported conclusion is restricted to this experimental setting:

> Adding semantic or answer-support verification signals does not necessarily
> improve selective QA ranking over a strong confidence baseline.

The evaluation does not establish a universal result for other datasets,
architectures, verification systems, or deployment settings.