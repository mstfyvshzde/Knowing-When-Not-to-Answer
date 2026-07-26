# Literature Review

## Scope

This review examines research related to:

* selective prediction and abstention
* confidence calibration
* answerability detection
* evidence-based verification
* truthfulness in language models

## 1. Selective Prediction and Abstention

Selective prediction allows a model to reject uncertain predictions instead of answering every input. The central evaluation problem is the trade-off between **coverage** and **risk**.

Geifman and El-Yaniv introduced selective prediction methods for deep neural networks and later proposed SelectiveNet, which integrates prediction and rejection into a single model.

For this project, selective prediction provides the theoretical basis for measuring whether abstention reduces errors without making the system unusably conservative.

## 2. Confidence Calibration

A model's confidence score does not always represent its actual probability of being correct.

Guo et al. showed that modern neural networks can be poorly calibrated and demonstrated that temperature scaling can improve the relationship between confidence and correctness.

This project will therefore distinguish between:

* raw confidence
* calibrated confidence
* actual correctness

## 3. Answerability Detection

SQuAD 2.0 introduced adversarially written unanswerable questions alongside answerable questions.

A successful system must both locate supported answers and recognize when the supplied context does not contain sufficient information.

This benchmark provides a controlled starting point for evaluating abstention.

## 4. Evidence Verification

FEVER introduced claims labeled as:

* `SUPPORTED`
* `REFUTED`
* `NOT ENOUGH INFO`

Supported and refuted claims are accompanied by evidence sentences.

This structure is relevant to the proposed evidence-checking component because it separates answer generation from evidence-based verification.

## 5. Truthfulness and LLM Abstention

TruthfulQA evaluates whether language models generate truthful answers instead of reproducing common human misconceptions.

Recent work on LLM abstention shows that deciding when not to answer remains difficult and lacks a universally accepted evaluation framework.

These findings suggest that confidence alone may not be sufficient for reliable abstention.

## Preliminary Synthesis

The literature already contains methods for confidence calibration, selective prediction, answerability detection, and evidence verification.

Therefore, this project must not claim that abstention or self-verification is entirely new.

The intended research contribution is a controlled and reproducible comparison of:

1. raw answer generation
2. confidence-based abstention
3. evidence-based verification
4. a combined self-verification framework

The study will test whether combining calibrated confidence and evidence support produces a better risk–coverage trade-off than either component alone.

## Working Research Gap

The provisional research gap is:

> It remains unclear whether a simple, modular combination of calibrated confidence and evidence verification can consistently reduce unsupported and overconfident answers while preserving useful answer coverage across controlled question-answering settings.

This gap must be revised after a broader and more systematic literature search.

## Key References

1. Geifman, Y., & El-Yaniv, R. (2017). *Selective Classification for Deep Neural Networks*.
2. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*.
3. Rajpurkar, P., Jia, R., & Liang, P. (2018). *Know What You Don't Know: Unanswerable Questions for SQuAD*.
4. Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018). *FEVER: A Large-Scale Dataset for Fact Extraction and Verification*.
5. Geifman, Y., & El-Yaniv, R. (2019). *SelectiveNet: A Deep Neural Network with an Integrated Reject Option*.
6. Lin, S., Hilton, J., & Evans, O. (2021). *TruthfulQA: Measuring How Models Mimic Human Falsehoods*.
7. Wen, B., et al. (2024). *Know Your Limits: A Survey of Abstention in Large Language Models*.
8. Madhusudhan, N., et al. (2024). *Do LLMs Know When to NOT Answer?*
