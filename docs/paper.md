# Knowing When Not to Answer

## Abstract

Selective question answering allows a system to abstain on predictions likely to be wrong. This study tests whether semantic or self-verification signals improve selective ranking over a calibrated-confidence baseline in extractive QA. Experiments use SQuAD v2 with `deepset/roberta-base-squad2`. The original validation data is split into separate calibration and held-out test partitions; temperature scaling is fitted only on calibration data and frozen before test evaluation. Five ranking methods are compared on deterministic nested held-out subsets up to 3,000 examples. The primary metric is area under the risk-coverage curve (AURC), where lower is better. Confidence only achieves the best final AURC (`0.292378`) and the lowest AURC at every tested sample size. Adding self-verification increases AURC to `0.309103`, while adding question-aware semantic verification increases it to `0.339439`. Paired bootstrap analysis with 5,000 resamples supports the stability of the confidence-only advantage on this held-out set. The result is a negative but informative finding: in this setting, the evaluated verification signals do not improve global selective ranking over strong calibrated confidence.

## 1. Research Question

The study asks whether adding semantic or self-verification improves selective QA ranking over calibrated model confidence. The claim is deliberately narrow and does not address general LLM reliability or safety.

## 2. Related Work

Selective classification studies the trade-off between prediction coverage and risk when a model is allowed to reject uncertain cases (Geifman & El-Yaniv, 2017). Confidence calibration addresses whether predictive confidence reflects empirical correctness; temperature scaling is a simple post-hoc calibration method for neural networks (Guo et al., 2017).

SQuAD v2 extends extractive question answering with adversarially written unanswerable questions, requiring systems to recognize when an answer is not supported by the context (Rajpurkar et al., 2018). Selective question answering has also been studied explicitly under domain shift, where abstention policies are evaluated by how much coverage they retain at a desired accuracy level (Kamath et al., 2020).

This study does not claim novelty for abstention, calibration, or selective QA themselves. Its narrower contribution is a controlled held-out comparison of question-aware semantic and self-verification signals against a calibrated-confidence baseline under the same extractive-QA evaluation protocol.

## 3. Experimental Setup

Experiments use SQuAD v2 (Rajpurkar et al., 2018). The original validation set is divided into calibration and held-out test partitions with a deterministic 50/50 stratified split using seed `17`.

The extractive QA backbone is `deepset/roberta-base-squad2`. Confidence is derived from the answer-vs-null margin and calibrated with temperature scaling (Guo et al., 2017). The learned temperature is `4.754804205196784`; calibration negative log-likelihood changes from `1.011796` before scaling to `0.422054` after scaling. The fitted temperature is frozen before application to the held-out test set.

Question-aware semantic verification converts each question-answer pair into a declarative claim using `domenicrosati/QA2D-t5-base` and evaluates the claim against the context with `FacebookAI/roberta-large-mnli`. Structurally invalid claims receive semantic score `0`.

The self-verifier separately evaluates whether a predicted answer is supported by context, also using `FacebookAI/roberta-large-mnli`. Its raw score is mapped from `[-1, 1]` to `[0, 1]`. Because both verification paths share the same NLI backbone, they are treated as separate signals rather than independent models.

## 4. Compared Methods

Five ranking methods are evaluated:

1. Confidence only
2. Question-aware semantic V2
3. Confidence + question-aware semantic V2
4. Self-verifier only
5. Confidence + self-verifier

Combined methods use the fixed equal-weight geometric mean `sqrt(score_a * score_b)`. The combination rule is not tuned on held-out test labels.

## 5. Evaluation Protocol

The final held-out evaluation uses 3,000 examples. A deterministic order with seed `17` defines nested subsets:

`200 ⊂ 500 ⊂ 1000 ⊂ 2000 ⊂ 3000`

Predictions are ranked by score in descending order, with the original deterministic index used to break ties.

The primary metric is AURC. Lower AURC indicates better selective ranking. Normalized AURC and matched-coverage accuracy and risk are complementary metrics.

## 6. Results

Final held-out exact-match accuracy is `0.4233`.

| Method | AURC | Normalized AURC |
|---|---:|---:|
| Confidence only | **0.292378** | **0.218555** |
| Confidence + self-verifier | 0.309103 | 0.264529 |
| Confidence + question-aware semantic V2 | 0.339439 | 0.347915 |
| Question-aware semantic V2 | 0.394397 | 0.498982 |
| Self-verifier only | 0.433766 | 0.607200 |

Confidence only has the lowest AURC at every nested sample size.

Small local reversals occur. At approximately 20% coverage on the 3,000-example evaluation, confidence + question-aware semantic V2 has risk around `0.1450`, compared with `0.1467` for confidence only. This does not reverse the global AURC ordering.

## 7. Statistical Uncertainty

The final comparison uses 5,000 paired bootstrap resamples with bootstrap seed `17`, evaluation-order seed `17`, and 95% percentile confidence intervals.

For each non-baseline method, `Delta AURC = method AURC - confidence-only AURC`.

| Method | Delta AURC | 95% CI |
|---|---:|---:|
| Confidence + self-verifier | +0.016725 | [0.008614, 0.024690] |
| Confidence + question-aware semantic V2 | +0.047061 | [0.036347, 0.057295] |
| Question-aware semantic V2 | +0.102019 | [0.086097, 0.118102] |
| Self-verifier only | +0.141389 | [0.124188, 0.158985] |

All intervals remain above zero. Since lower AURC is better, this supports the stability of the confidence-only advantage on this held-out test set. It is not a universal significance claim beyond this experimental setting.

## 8. Error Analysis

The final prediction set contains 1,270 exact-match correct and 1,730 incorrect predictions.

Question-aware labels are distributed as 1,622 `ENTAILMENT`, 713 `CONTRADICTION`, 332 `NEUTRAL`, and 333 `INVALID_CLAIM`. Despite the diagnostic value of these labels, 686 entailment-labeled examples are incorrect. At entailment probability at least `0.8`, 464 incorrect predictions remain. Conversely, 276 correct predictions have entailment probability at most `0.2`.

Self-verification produces 886 `SUPPORTED`, 1,791 `UNCERTAIN`, and 323 `REJECTED` examples. However, 344 incorrect predictions are still labeled `SUPPORTED`, while 712 correct predictions are labeled `UNCERTAIN`.

The question-aware path produces 333 invalid claims. The most frequent validation reason is `ANSWER_NOT_PRESERVED` with 309 occurrences. Validation reasons are not mutually exclusive.

These overlaps help explain why the verification signals can be diagnostically informative without improving the global ranking.

## 9. Discussion

The central result is negative but methodologically useful. A verification signal can contain information about correctness without improving the complete risk-coverage ordering. In this experiment, calibrated confidence already provides a strong ranking, while adding noisier verification signals through a fixed geometric mean reorders some predictions in ways that worsen global AURC.

The result also emphasizes baseline strength: verification methods should be compared against calibrated confidence, not only raw confidence or an always-answer system.

Local operating-point improvements suggest that verification may still be useful for specific decision policies even when it does not improve global AURC. That question is distinct from global ranking performance.

## 10. Limitations

The study evaluates one benchmark and one extractive QA backbone. Both verification paths use the same MNLI backbone. The question-aware path depends on QA-to-declarative claim generation, and its invalid-claim rule can downrank correct predictions. The combined methods use a fixed equal-weight geometric mean rather than a learned fusion model. Bootstrap intervals quantify uncertainty conditional on the observed held-out sample. The study therefore does not establish general LLM reliability, safety, or cross-domain performance.

## 11. Reproducibility

The reference environment uses Python `3.12.13`, with exact package versions in `requirements-lock.txt`. The complete workflow is executed with:

`DEVICE=cpu LIMIT=3000 bash scripts/reproduce_results.sh`

Final lightweight evaluation and bootstrap artifacts are retained under `outputs/evaluation/final_sample_size_comparison/`.

## 12. Conclusion

Adding semantic or self-verification signals does not necessarily improve selective QA ranking over a strong calibrated-confidence baseline.

In this SQuAD v2 extractive-QA experiment, confidence only achieves the best global AURC across all tested sample sizes, and paired bootstrap analysis supports the stability of that result on the held-out test set. The broader lesson is methodological: additional verification complexity should be judged by whether it improves the final selective ranking, not merely by whether the verifier appears informative on individual examples.

## 13. References

- Geifman, Y., & El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks. Advances in Neural Information Processing Systems 30 (NIPS 2017).
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. Proceedings of the 34th International Conference on Machine Learning, PMLR 70, 1321-1330.
- Kamath, A., Jia, R., & Liang, P. (2020). Selective Question Answering under Domain Shift. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 5684-5696.
- Rajpurkar, P., Jia, R., & Liang, P. (2018). Know What You Don’t Know: Unanswerable Questions for SQuAD. Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), 784-789.
