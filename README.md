# NPGC Implementation

A clean, reusable Python implementation of the NPGC synthesizer is being prepared and will be released separately.

In the meantime, the core implementation used for Educational Data Mining conference (EDM) can be found in this repository.


# Synthetic Data Generation for Tabular UCI Datasets

This repository contains the experiment pipeline used to train, evaluate, and compare multiple tabular synthetic data generators across several UCI datasets.

The codebase is organized around **model-specific runner scripts** in `script/`, shared utilities in `src/`, and convenience shell scripts in `experiments/`.

---

## Project Goals

- Generate synthetic tabular datasets with multiple methods.
- Evaluate synthetic quality with statistical and privacy-oriented QA metrics.
- Evaluate downstream utility with TSTR (Train on Synthetic, Test on Real).
- Save all metrics to JSON reports by default, with optional Weights & Biases (W&B) logging.

---

## Repository Layout

```text
.
├── experiments/
│   ├── run_all.sh
│   └── run_non_param_only.sh
├── script/
│   ├── CTGAN.py
│   ├── CopulaGAN.py
│   ├── Gauss_corr.py
│   ├── gaussian_copula.py
│   ├── non_parametric_gaussian.py
│   ├── TVAE.py
│   └── Plot_reports.py
├── src/
│   ├── correlator.py
│   ├── image_plotter.py
│   ├── loader.py
│   ├── metrics.py
│   └── non_parametric.py
├── pyproject.toml
└── README.md
```

---

## Methods Implemented

The following synthetic data methods are available as top-level scripts:

- `script/CTGAN.py`
- `script/CopulaGAN.py`
- `script/TVAE.py`
- `script/gaussian_copula.py`
- `script/non_parametric_gaussian.py` (NPGC)
- `script/Gauss_corr.py` (custom Gaussian-correlated baseline)

All main model scripts share a similar CLI interface:

```bash
python3 script/<method>.py --dataset <dataset_name> --iterations <n> [--enable-wandb]
```

---

## Datasets

Datasets are loaded in `src/loader.py` via `ucimlrepo`.

Supported dataset names:

- `adults`
- `car_evaluation`
- `balance_scale`
- `nursery`
- `student_performance`
- `student_dropout_success`

These names should be passed to `--dataset`.

---

## Environment Setup

### 1) Python environment

This project uses Poetry metadata (`pyproject.toml` + `poetry.lock`) and keeps the lock file under version control for reproducible installs.

If you use Poetry:

```bash
# Only needed when dependencies change in pyproject.toml
# poetry lock

poetry install

# If you want to run scripts directly with `python` (outside `poetry run`)
source .venv/bin/activate
```

Recommended workflow:

- **Most users/contributors:** run `poetry install` only.
- **Only when dependencies change:** run `poetry lock` first, then `poetry install`, and commit the updated `poetry.lock`.

If you use plain pip, install equivalent dependencies listed in `pyproject.toml` (less reproducible than Poetry with a committed lock file).

### 2) GPU / CPU selection (required)

Create a `.env` file in the repository root:

```env
# Use one specific GPU
CUDA_VISIBLE_DEVICES=0

# OR CPU-only mode
# CUDA_VISIBLE_DEVICES=
```

### 3) Optional W&B logging

W&B logging is **disabled by default**. If you want online tracking, install `wandb`, authenticate, and pass `--enable-wandb` to a script.

---

## Running Experiments

### Run all models across all configured datasets

```bash
bash experiments/run_all.sh
```

### Run only the NPGC method

```bash
bash experiments/run_non_param_only.sh
```

### Run a single method manually

Example:

```bash
python3 script/CTGAN.py --dataset adults --iterations 10
```

---

## Plotting and Aggregation

To aggregate and visualize report metrics, use:

```bash
python3 script/Plot_reports.py
```

This script reads report outputs and generates comparison plots across methods and datasets.


---

## Quick Start

```bash
# 1) Install dependencies
poetry install

# 2) Activate environment (optional if you prefer `poetry run ...`)
source .venv/bin/activate

# 3) Configure runtime environment
printf "CUDA_VISIBLE_DEVICES=0\n" > .env

# Optional if you want W&B logging
# wandb login

# 4) Run one test experiment
python3 script/gaussian_copula.py --dataset car_evaluation --iterations 1
# Optional: add --enable-wandb
```

---


# Synthetic Data Generation — NPGC and baselines

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A compact, reproducible pipeline for training, evaluating, and comparing tabular synthetic data generators (NPGC and several baselines).

## Table of contents

- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Methods implemented](#methods-implemented)
- [Datasets](#datasets)
- [Environment setup](#environment-setup)
- [Running experiments](#running-experiments)
- [Plotting and aggregation](#plotting-and-aggregation)
- [Output structure](#output-structure)
- [Goals & evaluation](#goals--evaluation)
- [Contributing](#contributing)
- [License](#license)

---

## Quick start

Install dependencies and run a single test experiment:

```bash
poetry install
source .venv/bin/activate
printf "CUDA_VISIBLE_DEVICES=0\n" > .env
python3 script/gaussian_copula.py --dataset car_evaluation --iterations 1
```

Example (what to look for): a new run creates `reports/car_evaluation/<method>/1.json` and images under `images/car_evaluation/`.

---

## Repository layout

```text
.
├── experiments/
│   ├── run_all.sh
│   └── run_non_param_only.sh
├── script/
│   ├── CTGAN.py
│   ├── CopulaGAN.py
│   ├── Gauss_corr.py
│   ├── gaussian_copula.py
│   ├── non_parametric_gaussian.py
│   ├── TVAE.py
│   └── Plot_reports.py
├── src/
│   ├── correlator.py
│   ├── image_plotter.py
│   ├── loader.py
│   ├── metrics.py
│   └── non_parametric.py
├── pyproject.toml
└── README.md
```

---

## Methods implemented

Top-level scripts:

- `script/CTGAN.py`
- `script/CopulaGAN.py`
- `script/TVAE.py`
- `script/gaussian_copula.py`
- `script/non_parametric_gaussian.py` (NPGC)
- `script/Gauss_corr.py` (baseline)

CLI pattern:

```bash
python3 script/<method>.py --dataset <dataset_name> --iterations <n>
```

---

## Datasets

Datasets are loaded in `src/loader.py` via `ucimlrepo`.

Supported names: `adults`, `car_evaluation`, `balance_scale`, `nursery`, `student_performance`, `student_dropout_success`.

---

## Environment setup

1) Python environment

This project uses Poetry (`pyproject.toml` + `poetry.lock`). For reproducible installs:

```bash
poetry install
# optionally: source .venv/bin/activate
```

2) GPU / CPU selection

Create a `.env` in the repo root, for example:

```env
CUDA_VISIBLE_DEVICES=0
# or leave empty for CPU-only
```

---

## Running experiments

Run all configured experiments:

```bash
bash experiments/run_all.sh
```

Run only the NPGC method:

```bash
bash experiments/run_non_param_only.sh
```

Run a single method manually (example):

```bash
python3 script/CTGAN.py --dataset adults --iterations 10
```

---

## Plotting and aggregation

Aggregate and visualize metrics:

```bash
python3 script/Plot_reports.py
```

This reads `reports/` and generates comparison plots under `images/`.

---

## Output structure

Primary artifacts written during runs:

- `metadata/<dataset>/metadata.json`
- `models/<dataset>/<method>/<iteration>/synthesizer.pkl`
- `reports/<dataset>/<method>/<iteration>.json`
- Per-run plots written by `src/image_plotter.py` under `images/`

`images/` overview (examples):

```text
images/
├── <dataset>/
│   ├── <method>/
│   │   └── <iteration>/
│   │       └── <column>.png
│   └── metrics/
│       └── <metric>_comparison.png
└── report_average_comparison/
    └── avg_<metric>_comparison.png
```

`reports/` overview:

```text
reports/
├── <dataset>/
│   ├── <method>/
│   │   └── <iteration>.json
│   └── report_dataset.json
└── report_averages/
    └── report_averages.json
```

---

## Goals & evaluation

Goals:

- Generate high-quality synthetic tabular datasets using multiple methods.
- Measure synthetic quality with SDV diagnostics and QA metrics.
- Evaluate downstream utility using TSTR (train on synthetic, test on real).

Evaluation outputs per run include SDV quality/diagnostic reports, QA metrics, TSTR scores, and timing information — all saved to `reports/`.

---

## Contributing

Contributions welcome. Please open an issue or a pull request with a concise description and any reproduction steps. Follow the existing code style and test with a small dataset first.

---

## License

This repository is licensed under the terms in `LICENSE`.

---

## Reproducibility notes

- Scripts set a global seed (`SEED = 42`) and derive per-iteration seeds.
- Iterative training is used by most methods; iteration `i+1` may depend on synthetic data from iteration `i`.
- The first iteration typically splits real data into train/holdout for QA evaluation.
