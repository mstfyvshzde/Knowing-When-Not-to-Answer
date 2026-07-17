# Research Decision Log

This document records important research and engineering decisions made during the project.

## Decision 001 — Research Focus

**Decision:**
The project will focus on selective question answering and abstention rather than general chatbot development.

**Reason:**
A controlled question-answering setting allows reliability, confidence, evidence support, and abstention to be measured objectively.

**Alternatives Considered:**

* general-purpose chatbot evaluation
* prompt-only hallucination reduction
* open-domain RAG system

**Why Rejected:**
These alternatives introduce too many uncontrolled variables during the initial experiments.

---

## Decision 002 — Decision Space

**Decision:**
The framework will use three possible actions:

* `ANSWER`
* `VERIFY`
* `ABSTAIN`

**Reason:**
A binary answer-or-refuse policy does not represent cases where additional verification could resolve uncertainty.

---

## Decision 003 — Primary Benchmark

**Decision:**
SQuAD 2.0 is the initial benchmark candidate.

**Reason:**
It contains both answerable and unanswerable questions and supports controlled abstention experiments.

**Limitation:**
Results may not generalize to open-domain or conversational settings.

---

## Decision 004 — Evaluation Principle

**Decision:**
Systems will not be compared using accuracy alone.

**Required Measures:**

* accuracy
* coverage
* selective risk
* abstention rate
* unsupported answer rate
* false confidence rate
* calibration

**Reason:**
A system can reduce errors simply by refusing most questions. Reliability must therefore be evaluated together with usefulness.

---

## Decision 005 — Test-Set Protection

**Decision:**
Thresholds, calibration parameters, prompts, and verification rules will be selected using development data only.

**Reason:**
Using test results to modify the system would create evaluation leakage and produce misleading performance estimates.

---

## Decision 006 — Research Claims

**Decision:**
The project will not claim to solve hallucination or guarantee AI safety.

**Reason:**
Conclusions must remain limited to the selected models, datasets, and experimental conditions.

## Status

These decisions are provisional and may be revised when supported by literature review findings or experimental evidence.
