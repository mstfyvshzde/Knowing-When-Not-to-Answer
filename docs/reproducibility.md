# Reproducibility

## Goal

This project will be designed so that another researcher can reproduce the main experiments using the published code, configurations, and documented procedures.

## Environment

The repository will record:

* Python version
* package versions
* operating system information
* hardware information where relevant
* model and tokenizer versions

Dependencies will be stored in:

* `requirements.txt`
* `pyproject.toml`
* `environment.yml`

## Randomness Control

All experiments involving randomness will use fixed seeds.

Seeds will be applied to:

* Python
* NumPy
* PyTorch
* dataset shuffling
* sampling procedures

The selected seed values will be recorded in configuration files.

## Dataset Tracking

The repository will document:

* dataset name and version
* original source
* license or usage terms
* preprocessing steps
* filtering rules
* dataset splits
* excluded examples

Raw datasets will not be modified directly.

Processed datasets will be generated using reproducible scripts.

## Configuration Management

Experiment settings will be stored in versioned YAML configuration files.

Configurations will include:

* model settings
* dataset settings
* confidence thresholds
* calibration settings
* verification rules
* evaluation parameters

No important experiment setting should exist only inside a notebook.

## Experiment Logging

Each experiment will save:

* configuration
* random seed
* model version
* dataset split
* start and completion time
* predictions
* confidence scores
* final decisions
* evaluation metrics

## Saved Outputs

The following outputs will be preserved:

```text
outputs/
├── figures/
├── logs/
├── predictions/
└── tables/
```

Saved predictions will allow evaluation metrics to be recalculated without rerunning the model.

## Reproduction Scripts

The repository will provide scripts for:

* downloading datasets
* preparing data
* running baselines
* running the full framework
* calculating metrics
* reproducing tables and figures

The main reproduction command will be documented after the pipeline is implemented.

## Test-Set Protection

The test set will not be used for:

* threshold selection
* calibration fitting
* prompt development
* error-driven system modification

Final test predictions will be saved before manual error analysis begins.

## Reporting

Every reported result will include:

* system name
* model version
* dataset split
* sample size
* configuration file
* random seed
* evaluation metrics

Any failed or excluded experiment will be documented when it affects the interpretation of the results.

## Reproducibility Limitations

Exact results may vary because of:

* hardware differences
* nondeterministic operations
* external model updates
* API model changes
* unavailable proprietary systems

These factors will be reported clearly rather than hidden.
