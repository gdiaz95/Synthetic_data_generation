
# Synthetic Data Generation — NPGC and baselines

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository provides code to reproduce the experiments presented in the cited paper. For using NPGC as a Python package, see the [NPGC repository](https://github.com/gdiaz95/NPGC).

A compact, reproducible pipeline for training, evaluating, and comparing tabular synthetic data generators (NPGC and several baselines).

## Citation

Gabriel Diaz Ramos, Lorenzo Luzi, Debshila Basu Mallick, and Richard Baraniuk. *Stable and Privacy-Preserving Synthetic Educational Data with Empirical Marginals: A Copula-Based Approach*. Proceedings of the 19th International Conference on Educational Data Mining, pp. 267–279. International Educational Data Mining Society, 2026. [PDF](https://educationaldatamining.org/wp-content/uploads/2026/proceedings/2026.EDM.full-papers/2026.EDM.full-papers.119.pdf) | [DOI](https://doi.org/10.5281/zenodo.21040131)

BibTeX:

```bibtex
@inproceedings{diazramos2026npgc,
  title={Stable and Privacy-Preserving Synthetic Educational Data with Empirical Marginals: A Copula-Based Approach},
  author={Diaz Ramos, Gabriel and Luzi, Lorenzo and Basu Mallick, Debshila and Baraniuk, Richard},
  booktitle={Proceedings of the 19th International Conference on Educational Data Mining},
  publisher={International Educational Data Mining Society},
  pages={267--279},
  year={2026},
  doi={10.5281/zenodo.21040131}
}
```

## Table of contents

- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Methods implemented](#methods-implemented)
- [Datasets](#datasets)
- [Environment setup](#environment-setup)
- [Running experiments](#running-experiments)
- [OpenStax release generation](#openstax-release-generation)
- [Plotting and aggregation](#plotting-and-aggregation)
- [Output structure](#output-structure)
- [Goals](#goals)
- [Reproducibility notes](#reproducibility-notes)
- [Funding](#funding)

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
│   └── run_npgc_only.sh
├── script/
│   ├── CTGAN.py
│   ├── CopulaGAN.py
│   ├── Gauss_corr.py
│   ├── gaussian_copula.py
│   ├── npgc_script.py
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
- `script/npgc_script.py` (NPGC)
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

1) GPU / CPU selection

Create a `.env` in the repo root, for example:

```env
CUDA_VISIBLE_DEVICES=0
# or leave empty for CPU-only
SDG_MAX_BLAS_THREADS=32
# Optional safety limit used only if CTGAN/CopulaGAN/TVAE hit a thread-related fit error
```

---

## Running experiments

Run all configured experiments:

```bash
bash experiments/run_all.sh
```

Run only the NPGC method:

```bash
bash experiments/run_npgc_only.sh
```

Run a single method manually (example):

```bash
python3 script/CTGAN.py --dataset adults --iterations 10
```

---

## OpenStax release generation

Generates synthetic OpenStax activity data and saves all release files to `Openstax_test_data/`.

**Step 1 — generate data and activity comparison plot:**

```bash
python3 script/generate_openstax_release.py
```

**Step 2 — generate aggregated comparison plots:**

```bash
python3 script/Plot_reports.py
```

Or run both steps at once:

```bash
bash experiments/run_openstax_release.sh
```

Output written to `Openstax_test_data/`:

- `assignable_original_release.csv` — original activity records (n=1509)
- `assignable_synthetic_release.csv` — synthetic activity records (n=1982)
- `orig_activity.csv` / `synth_activity.csv` — TikZ-ready marginal distributions
- `openstax_activity_comparison.png` — marginal distribution comparison plot

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

- `images/<dataset>/<method>/<iteration>/<column>.png` are univariate histograms for each feature at each iteration.
- `images/<dataset>/metrics/<metric>_comparison.png` compare the same metric across all methods for that dataset.
- `images/report_average_comparison/avg_<metric>_comparison.png` show aggregated metric comparisons across all datasets.
- `adults` includes a special `adults_comparison/` output with per-metric summaries across the 10 iterations.

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

- `reports/<dataset>/<method>/<iteration>.json` contains per-run metrics for that method and iteration.
- `reports/<dataset>/report_dataset.json` contains average metrics aggregated for that dataset.
- `reports/report_averages/report_averages.json` contains metrics averaged across all datasets.

---

## Goals

- Generate high-quality synthetic tabular datasets using multiple methods.
- Measure synthetic quality with SDV diagnostics and QA metrics.
- Evaluate downstream utility using TSTR (train on synthetic, test on real).

Evaluation outputs per run include SDV quality/diagnostic reports, QA metrics, TSTR scores, and timing information — all saved to `reports/`.

---

## Reproducibility notes

- Scripts set a global seed (`SEED = 42`) and derive per-iteration seeds.
- Iterative training is used by most methods; iteration `i+1` may depend on synthetic data from iteration `i`.
- The first iteration typically splits real data into train/holdout for QA evaluation.

---

## Funding

This research is supported by the National Science Foundation under Award No. 2153481.
