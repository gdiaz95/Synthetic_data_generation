
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
SDG_MAX_BLAS_THREADS=32
# Optional safety limit for CTGAN/CopulaGAN/TVAE training threads
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


## License

This repository is licensed under the terms in `LICENSE`.

---

## Reproducibility notes

- Scripts set a global seed (`SEED = 42`) and derive per-iteration seeds.
- Iterative training is used by most methods; iteration `i+1` may depend on synthetic data from iteration `i`.
- The first iteration typically splits real data into train/holdout for QA evaluation.
