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
- Track all runs and metrics in Weights & Biases (W&B).

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
- `script/non_parametric_gaussian.py`
- `script/Gauss_corr.py` (custom Gaussian-correlated baseline)

All main model scripts share a similar CLI interface:

```bash
python3 script/<method>.py --dataset <dataset_name> --iterations <n>
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

This project uses Poetry metadata (`pyproject.toml` + `poetry.lock`).

If you use Poetry:

```bash
poetry install
```

If you use plain pip, install equivalent dependencies listed in `pyproject.toml`.

### 2) GPU / CPU selection (required)

Create a `.env` file in the repository root:

```env
# Use one specific GPU
CUDA_VISIBLE_DEVICES=0

# OR CPU-only mode
# CUDA_VISIBLE_DEVICES=
```

### 3) W&B authentication

Because the scripts log to Weights & Biases, ensure you are authenticated:

```bash
wandb login
```

---

## Running Experiments

### Run all models across all configured datasets

```bash
bash experiments/run_all.sh
```

### Run only the non-parametric Gaussian method

```bash
bash experiments/run_non_param_only.sh
```

### Run a single method manually

Example:

```bash
python3 script/CTGAN.py --dataset adults --iterations 10
```

---

## Output Structure

As experiments run, the pipeline writes artifacts to project subfolders:

- `metadata/<dataset>/metadata.json`
- `models/<dataset>/<method>/<iteration>/synthesizer.pkl`
- `reports/<dataset>/<method>/<iteration>.json`
- model-specific plots written by `src/image_plotter.py`

If a saved model already exists for a run, the loader will reuse it instead of retraining.

---

## Evaluation Summary

Each run computes:

- SDV diagnostic and quality report fields.
- QA metrics (accuracy/similarity/distance family metrics).
- TSTR utility metrics using XGBoost classification accuracy.
- Timing metrics (training and evaluation time).

These are saved into JSON reports and logged to W&B.

---

## Plotting and Aggregation

To aggregate and visualize report metrics, use:

```bash
python3 script/Plot_reports.py
```

This script reads report outputs and generates comparison plots across methods and datasets.

---

## Reproducibility Notes

- Scripts set a global seed (`SEED = 42`) and derive per-iteration seeds.
- Most methods follow iterative training where iteration `i+1` uses synthetic data from iteration `i`.
- The first iteration typically splits real data into train/holdout for QA evaluation.

---

## Quick Start

```bash
# 1) Install dependencies
poetry install

# 2) Configure runtime environment
printf "CUDA_VISIBLE_DEVICES=0\n" > .env
wandb login

# 3) Run one test experiment
python3 script/gaussian_copula.py --dataset car_evaluation --iterations 1
```

---
