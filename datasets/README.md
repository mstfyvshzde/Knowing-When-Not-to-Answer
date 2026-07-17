# Dataset Documentation

## Purpose

This directory contains documentation and scripts related to the datasets used in the project.

Large dataset files should not be committed directly unless their license explicitly permits redistribution.

## Directory Structure

```text
datasets/
├── raw/          Original downloaded data
├── processed/    Cleaned and transformed data
├── external/     Additional benchmark resources
└── README.md
```

## Initial Benchmark Candidate

The initial benchmark candidate is **SQuAD 2.0**.

It contains:

* answerable questions
* unanswerable questions
* evidence passages
* reference answers

The final dataset selection will be confirmed after dataset inspection and literature review.

## Dataset Usage

The project will preserve official dataset splits whenever possible.

The splits will be used as follows:

* **Training split:** model or verifier development
* **Development split:** calibration and threshold selection
* **Test split:** final evaluation only

Test labels must not influence system design decisions.

## Processing Rules

Raw data will remain unchanged.

Processed data will be generated using scripts in:

```text
src/data/
```

All processing steps must document:

* removed examples
* filtering conditions
* normalization rules
* label conversions
* random seeds
* final sample counts

## Planned Labels

Each evaluated response may include:

* `ANSWERABLE`
* `UNANSWERABLE`
* `SUPPORTED`
* `UNSUPPORTED`
* `UNCERTAIN`

The exact label definitions will be documented before annotation or evaluation begins.

## Data Integrity

The project will check for:

* duplicate examples
* missing fields
* malformed records
* train–test overlap
* annotation inconsistencies
* class imbalance

## Licensing

Each dataset must have its source, version, citation, and usage conditions documented before use.

Dataset files that cannot legally be redistributed will be downloaded through reproducible scripts.

## Leakage Prevention

The project will not:

* tune thresholds using test labels
* modify prompts after inspecting test errors
* manually correct final test predictions
* combine official splits without documentation

## Current Status

Dataset selection and inspection have not yet been completed.
