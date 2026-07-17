# Ethics Statement

## Research Purpose

This project investigates whether AI systems can reduce unreliable responses by deciding when to answer, verify, or abstain.

The project is intended for academic research and does not claim to create a universally safe or trustworthy AI system.

## Potential Benefits

A reliable abstention mechanism may help reduce:

* unsupported answers
* high-confidence errors
* misleading responses
* inappropriate certainty

The research may also provide clearer evaluation methods for studying reliability beyond answer accuracy.

## Potential Risks

Abstention mechanisms can introduce new risks.

A system may:

* refuse answerable questions unnecessarily
* behave differently across domains or demographic groups
* create a false impression of safety
* rely on incomplete or biased evidence
* assign misleading confidence scores

## User Trust

An abstaining system should communicate uncertainty clearly.

The message `I don't know` must not imply that the system has performed a complete or authoritative safety assessment.

Users should be informed that:

* confidence is imperfect
* evidence may be incomplete
* verification can fail
* abstention does not guarantee safety

## Fairness

The project will examine whether abstention and error rates differ across available dataset categories.

If demographic or domain-based analysis is possible, performance differences will be reported rather than hidden within aggregate metrics.

The project will not make broad fairness claims without appropriate data.

## Data Use

Only datasets with documented research usage conditions will be used.

The repository will document:

* dataset source
* license or usage terms
* preprocessing steps
* excluded examples
* known dataset limitations

Sensitive personal data will not be intentionally collected.

## High-Stakes Applications

The framework will not be presented as suitable for medical, legal, financial, or safety-critical decision-making.

Controlled benchmark improvements do not establish real-world safety.

## Transparency

The final report will disclose:

* selected models
* benchmark limitations
* verification failures
* threshold choices
* computational constraints
* negative or inconclusive results

Failure cases will be reported alongside successful examples.

## Responsible Claims

The project will not claim to:

* solve hallucination
* guarantee factual correctness
* make language models safe
* replace human verification
* generalize beyond the tested conditions

All conclusions will remain limited to the experiments conducted.
