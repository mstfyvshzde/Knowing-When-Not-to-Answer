# Knowing When Not to Answer

> A modular research framework for reliable selective question answering.

---

## Overview

This project investigates **reliable selective question answering** for large language models. Instead of forcing a model to answer every question, the framework determines whether a response should be **accepted, verified, or abstained from** based on confidence calibration and evidence verification.

The project provides a modular research framework for evaluating decision policies, calibration methods, and verification strategies, with an emphasis on improving reliability while maintaining strong predictive performance.


## Motivation

Large language models often generate answers even when they are uncertain, which can lead to unreliable or misleading responses. In many real-world applications, answering incorrectly may be more harmful than not answering at all.

This project explores a reliability-first approach in which the system decides whether to answer directly, verify the generated response, or abstain when confidence is insufficient. By combining confidence calibration, evidence verification, and decision policies, the framework aims to improve the trustworthiness of question answering systems.


## Features

* Modular architecture for selective question answering research.
* Confidence calibration for more reliable decision making.
* Evidence verification before producing final responses.
* Configurable abstention policies for uncertainty-aware prediction.
* Baseline comparison and ablation study support.
* Comprehensive evaluation using risk–coverage and calibration metrics.
* Reproducible experiments with configurable settings.
* Well-organised project structure for research and future extensions.


## Repository Structure

```text
Knowing-When-Not-to-Answer/
│
├── assets/             # Images and repository assets
├── configs/            # Experiment configuration files
├── data/               # Raw, processed and external datasets
├── docs/               # Research documentation
├── experiments/        # Experiment entry points
├── notebooks/          # Exploratory analysis and visualisation
├── outputs/            # Results, figures and evaluation outputs
├── scripts/            # Utility and automation scripts
├── src/
│   ├── analysis/
│   ├── baselines/
│   ├── calibration/
│   ├── data/
│   ├── decision/
│   ├── evaluation/
│   ├── utils/
│   └── verification/
├── tests/              # Unit tests
│
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── LICENSE
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/mstfyvshzde/Knowing-When-Not-to-Answer.git

# Enter the project directory
cd Knowing-When-Not-to-Answer

# Create a virtual environment
python3 -m venv .venv

# Activate the environment
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

The project currently uses Python 3.12 and includes dependencies for PyTorch, Hugging Face Transformers, datasets, NumPy, and YAML configuration management.


## Project Pipeline

The research framework follows a modular pipeline designed to improve the reliability of large language model predictions.

1. **Data Preparation** – Load and preprocess datasets for training and evaluation.
2. **Baseline Prediction** – Generate initial answers using baseline language models.
3. **Confidence Estimation** – Estimate the confidence of each prediction.
4. **Confidence Calibration** – Apply calibration methods to improve confidence reliability.
5. **Evidence Verification** – Verify generated answers using evidence-based validation modules.
6. **Decision Engine** – Decide whether to **accept**, **verify**, or **abstain** from answering.
7. **Evaluation** – Measure performance using accuracy, risk–coverage, calibration, and reliability metrics.
8. **Analysis** – Perform error analysis and ablation studies to understand system behaviour.


## Experimental Results

## Experimental Results

The proposed framework is evaluated through a comprehensive experimental pipeline that includes baseline comparisons, confidence calibration, evidence verification, and ablation studies.

The evaluation focuses on the following aspects:

* Selective question answering performance
* Risk–coverage trade-off
* Confidence calibration quality
* Evidence verification effectiveness
* Decision policy analysis
* Ablation studies for individual framework components

Experimental figures, quantitative results, and detailed analyses are available in the `outputs/` and `docs/` directories and will be presented in future releases of this repository.


## Repository Layout

## Reproducibility

## Citation

## License
