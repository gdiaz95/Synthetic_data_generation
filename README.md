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

## Output Structure

As experiments run, the pipeline writes artifacts to project subfolders:

- `metadata/<dataset>/metadata.json`
- `models/<dataset>/<method>/<iteration>/synthesizer.pkl`
- `reports/<dataset>/<method>/<iteration>.json`
- model-specific plots written by `src/image_plotter.py`

If a saved model already exists for a run, the loader will reuse it instead of retraining.

### `images/` directory guide

The `images/` folder is for browsing plots quickly. It has three main types of content:

```text
images/
├── <dataset>/
│   ├── <method>/
│   │   └── <iteration>/
│   │       └── <column>.png
│   └── metrics/
│       └── <metric>_comparison.png
├── adults_comparison/
│   └── <metric>_comparison.png
├── report_average_comparison/
│   └── avg_<metric>_comparison.png
└── report_dataset_averages/
    └── wandb.txt
```

What each part means:

- `images/<dataset>/<method>/<iteration>/<column>.png`
  These are the per-column univariate distribution plots comparing real data vs synthetic data for one run.
- `images/<dataset>/metrics/`
  These are dataset-level summary plots comparing methods on each metric.
- `images/adults_comparison/`
  These are iteration-by-iteration metric plots for the `adults` dataset only.
- `images/report_average_comparison/`
  These are global average plots across datasets.

How to browse `images/` as a new user:

- If you want to inspect one generated run visually, go to `images/<dataset>/<method>/<iteration>/`.
- If you want to compare methods on one dataset by metric, go to `images/<dataset>/metrics/`.
- If you want to study regeneration stability across repeated iterations, go to `images/adults_comparison/`.
- If you want a high-level cross-dataset summary, go to `images/report_average_comparison/`.

Important note about regeneration stability:

- The repeated-iteration stability plots are in `images/adults_comparison/`.
- In the current repository contents, `adults` is the dataset with multiple saved iterations (`1` to `10`), so it is the only dataset with that full iteration-comparison view.
- Other datasets currently only have iteration `1` saved in `images/<dataset>/<method>/1/`.

Where specific image types live:

- Regeneration stability images: `images/adults_comparison/`
- Adults per-metric comparison images: `images/adults_comparison/`
- Per-dataset per-metric method comparison images: `images/<dataset>/metrics/`
- Univariate distributions for one run: `images/<dataset>/<method>/<iteration>/<column>.png`

About the univariate distributions:

- These plots are created by `src/image_plotter.py`.
- Each image is one column from the dataset.
- Numeric columns are shown as density overlays.
- Categorical columns are shown as normalized bar plots.
- Although `adults` is the only dataset with many saved iterations in this repo, the same plotting code is used by all model scripts, so other datasets can produce the same kind of per-column plots when you run them.

### `reports/` directory guide

The `reports/` folder stores the numeric outputs behind the plots.

```text
reports/
├── <dataset>/
│   ├── <method>/
│   │   └── <iteration>.json
│   └── report_dataset.json
└── report_averages/
    └── report_averages.json
```

What each part means:

- `reports/<dataset>/<method>/<iteration>.json`
  The raw report for one run of one method on one dataset.
- `reports/<dataset>/report_dataset.json`
  The dataset-level summary across methods used for `images/<dataset>/metrics/`.
- `reports/report_averages/report_averages.json`
  The overall summary across datasets used for `images/report_average_comparison/`.

How to browse `reports/` as a new user:

- Start with `reports/<dataset>/report_dataset.json` if you want the simplest summary for one dataset.
- Open `reports/<dataset>/<method>/<iteration>.json` if you want the detailed numbers for a single run.
- Open `reports/report_averages/report_averages.json` if you want the broadest cross-dataset summary.

What is inside each per-run JSON report:

- `diagnostic_report`: SDV diagnostic checks such as validity and structure.
- `quality_report`: SDV quality scores such as overall score, column shapes, and column pair trends.
- `metrics_qa`: QA metrics such as `overall_accuracy`, `univariate_accuracy`, `bivariate_accuracy`, `discriminator_auc`, and distance/privacy-style metrics.
- `times`: training and evaluation runtime.
- `tstr_evaluation`: downstream utility scores from TSTR.

---

## Evaluation Summary

Each run computes:

- SDV diagnostic and quality report fields.
- QA metrics (accuracy/similarity/distance family metrics).
- TSTR utility metrics using XGBoost classification accuracy.
- Timing metrics (training and evaluation time).

These are saved into JSON reports. W&B logging is optional with `--enable-wandb`.

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

# Optional if you want W&B logging
# wandb login

# 3) Run one test experiment
python3 script/gaussian_copula.py --dataset car_evaluation --iterations 1
# Optional: add --enable-wandb
```

---
