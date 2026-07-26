# Problem Definition

## Research Problem

Large language models often generate incorrect or unsupported answers with high confidence. Standard question-answering systems are usually optimized to produce an answer, even when the available evidence is insufficient.

This project investigates whether an AI system can evaluate the reliability of its own response before presenting it to the user.

## Core Research Question

Can an AI system detect when its answer is unsupported, uncertain, or unsafe to trust?

## Proposed Decision Space

The system will select one of three actions:

1. **Answer** — provide the response when sufficient evidence and confidence are available.
2. **Verify** — perform an additional verification step when reliability is unclear.
3. **Abstain** — return `I don't know` when the answer cannot be supported.

## Scope

The study focuses on controlled question-answering benchmarks containing both answerable and unanswerable examples.

The project does not claim to solve hallucination across all language models or real-world domains.

## Research Objective

The objective is to determine whether confidence estimation and evidence verification can reduce unsupported and overconfident answers while maintaining useful answer coverage.
