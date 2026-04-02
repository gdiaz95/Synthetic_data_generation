# NPGC Implementation

A clean, reusable Python implementation of the NPGC synthesizer is being prepared and will be released separately.

In the meantime, the core implementation used for Educational Data Mining conference (EDM) can be found in this repository.


# Synthetic Data Generation for Tabular UCI Datasets

This repository trains and evaluates several synthetic tabular data generators on UCI datasets.

## What you get

- Multiple generators (`CTGAN`, `TVAE`, `CopulaGAN`, `GaussianCopula`, non-parametric Gaussian, custom Gaussian-correlated baseline).
- Quality + privacy-oriented evaluation reports.
- TSTR (Train on Synthetic, Test on Real) downstream utility metrics.

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
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Supported datasets (`--dataset`)

- `adults`
- `car_evaluation`
- `balance_scale`
- `nursery`
- `student_performance`
- `student_dropout_success`

---

## Reproduce (works out of the box)

### 1) Create and activate a clean Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run a smoke test (CPU + offline W&B, no login needed)

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES="" \
python script/gaussian_copula.py --dataset adults --iterations 1
```

That command is the fastest reliable end-to-end sanity check in this repo.

### 4) (Optional) Run all configured experiments

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES="" bash experiments/run_all.sh
```

---

## Notes on W&B and GPU

- **No W&B account required for first run**: use `WANDB_MODE=offline`.
- If you want cloud logging, run `wandb login` and omit `WANDB_MODE=offline`.
- To run CPU-only, set `CUDA_VISIBLE_DEVICES=""`.
- To run on GPU `0`, set `CUDA_VISIBLE_DEVICES=0`.

---

## Single-method runs

```bash
python script/CTGAN.py --dataset adults --iterations 5
python script/TVAE.py --dataset adults --iterations 5
python script/CopulaGAN.py --dataset adults --iterations 5
python script/gaussian_copula.py --dataset adults --iterations 5
python script/non_parametric_gaussian.py --dataset adults --iterations 5
python script/Gauss_corr.py --dataset adults --iterations 5
```

## Output artifacts

Runs generate artifacts such as:

- `metadata/<dataset>/metadata.json`
- `models/<dataset>/<method>/<iteration>/synthesizer.pkl`
- `reports/<dataset>/<method>/<iteration>.json`
- plots under `images/`

## Plot report aggregation

```bash
python script/Plot_reports.py
```

## Reproducibility notes

- Global seed is fixed (`SEED = 42`) in runner scripts.
- Iterative runs use deterministic per-iteration seed offsets.
- Iteration 1 typically creates a real train/holdout split; later iterations train on prior synthetic output.
