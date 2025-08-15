# Cross-Industry Privacy-Preserving Framework for PdM based on IDA 24 Industrial Challenge




**Short description**
This repository implements a TabTransformer-based model for Remaining Useful Life (RUL) prediction on the Scania dataset, with an end-to-end preprocessing pipeline and differential-privacy-enabled training. It also contains an evaluation and Membership Inference Attack (MIA) framework so you can measure privacy/utility trade-offs.

---

## Table of contents

* [Motivation](#motivation)
* [What’s included](#whats-included)
* [Repository structure](#repository-structure)
* [Quickstart](#quickstart)
* [Data & preprocessing](#data--preprocessing)
* [Model & architecture](#model--architecture)
* [Training](#training)
* [Differential privacy (DP) details](#differential-privacy-dp-details)
* [Evaluation & Membership Inference Attack (MIA)](#evaluation--membership-inference-attack-mia)
* [Results & tips](#results--tips)
* [Reproducibility checklist](#reproducibility-checklist)
* [How to extend](#how-to-extend)
* [Cite / References](#cite--references)
* [Contributing](#contributing)
* [License & Contact](#license--contact)

---

## Motivation

This project was created to explore practical, privacy-preserving RUL forecasting on the Scania dataset. Goals:

* Use a transformer-style encoder for tabular sensor/time features (TabTransformer).
* Apply differential privacy during training to limit memorization of individual trajectories.
* Provide reproducible preprocessing, training, and attack/evaluation code suitable for research and submission.

---


## Key ideas
- Convert sliding windows of raw sensor time series into compact embeddings using a Transformer encoder (`TimeSeriesEmbedder`).  
- Combine the time-series embedding (default 128-dim) with ordinal/embedded categorical vehicle specification features (`Spec_0`..`Spec_7`) and pass them into a `TabTransformer` for final regression to predict RUL.  
- Optionally train with differential privacy using Opacus; sweep `NOISE_M` (noise multiplier) to measure privacy/utility tradeoffs.

---







## What’s included

* `Preprocess.ipynb` — Full preprocessing pipeline for the Scania dataset (feature engineering, normalization, windowing).
* `test_alpha.py` — Main training script implementing the TabTransformer model and DP-enabled training loop.
* `MIA.ipynb` — Membership Inference Attack experiments and analysis.
* `configs/` — Example config templates (hyperparameters & DP settings).
* `notebooks/` — Experiments, EDA and visualization notebooks.
* `models/` — Saved model checkpoints.
* `results/` — Experiment logs and metric summaries.
* `requirements.txt` — Python dependencies.

---

## Repository structure (example)

```
├── Preprocess.ipynb
├── test_alpha.py
├── MIA.ipynb
├── configs/
│   └── scania_dp_example.yaml
├── notebooks/
├── models/
├── results/
├── data/
├── requirements.txt
└── README.md
```

---



2. Prepare the data using `Preprocess.ipynb`:

* Run the notebook cells to download/locate the Scania dataset, generate windows, and save processed files into `data/processed/`.

3. Train (example):

```bash
python test_alpha.py \
  --config configs/scania_dp_example.yaml \
  --device cuda \
  --epochs 100 \
  --batch-size 256 \
  --save-dir models/scania_tabtransformer_dp
```

Run with differential privacy (illustrative flags — adapt to your script):

```bash
python test_alpha.py --config configs/scania_dp_example.yaml \
  --dp --noise-multiplier 1.1 --max-grad-norm 1.0 --target-eps 3.0 --delta 1e-5
```

> Note: CLI flag names depend on your implementation — check `configs/` for exact keys.

---

## Data & preprocessing

* **Dataset**: Scania RUL dataset (multivariate time series from trucks).
* Use `Preprocess.ipynb` to:
* `batch_size`, `learning_rate`, `weight_decay`
  * Clean and align sensor channels.
  * Create sliding windows / sequence samples suitable for the TabTransformer model.
  * Time series input: sliding windows of `context = 70`, `time steps × sensor_features = 105` sensor columns (default).
  * Components: positional embeddings → Transformer encoder stack → linear projector to `embedding_dim=128`.
  * Split into train / val / test by machine to avoid leakage.
  * Save scalers/encoders for reproducible inference.



## Model & architecture

Core ideas:

* Learn contextualized embeddings for categorical/ordinal tabular features using transformer encoder blocks.
* Combine learned embeddings with continuous features, then pass through an MLP regression head for RUL prediction.

Highlights:

* Per-feature embedding for discrete/categorical inputs.
* Transformer encoder blocks (multi-head attention + feed-forward network).
* Concatenation with continuous features and a small MLP head.
* Optional temporal window encoder if training on sequential windows.

---

## Training

* **Loss**: MSE is common for RUL; MAE or Huber also valid.
* **Optimizer**: Adam/AdamW.
* **Scheduler**: cosine or step LR scheduler recommended.
* **Checkpointing**: save best model by validation RMSE/MAE and checkpoint at intervals.
* Example hyperparameters:
    *`context = 70`, `sensor_features = 105`
    *`embedding_dim = 128`
    *`batch_size = 128`
    *`lr = 1e-3`
    *`epochs = 50` (use early stopping)
    *`DP: max_grad_norm = 1.0`, `noise_multiplier = <varies>`

Hyperparameters to tune:

* `batch_size`, `learning_rate`, `weight_decay`
* `num_heads`, `num_layers`, `embedding_dim`
* `dropout`, `max_grad_norm` (important for DP)

---

## Differential privacy (DP) details

Key DP hyperparameters:

* **Noise multiplier (σ)** — larger σ increases privacy (and utility loss).
* **Max grad norm (C)** — clipping threshold for per-sample gradients.
* **Target ε (epsilon)** and **δ (delta)** — the reported privacy budget.
* **clip_bound** — max per-sample grad norm.

Practical notes:

* DP training typically requires larger batch sizes or more epochs to recover utility.
* Lower ε (stronger privacy) often reduces model utility; common research values are single-digit ε.
* Log and report final `(ε, δ)` from the DP accountant after training.



## Evaluation & Membership Inference Attack (MIA)

**Metrics**: RMSE, MAE, R², and any custom RUL scoring you use.

Visualizations: predicted vs actual scatter, residuals over time, per-machine trajectory plots.

**MIA** (`MIA.ipynb`) includes:

* Shadow models and classifier-based attacks.
* Experimental comparison between DP and non-DP models.
* Metrics such as membership accuracy and AUC.

---

## Results & tips

* Run ablations comparing TabTransformer vs baseline MLP/LSTM (both with and without DP).
* Sweep DP budgets (e.g., ε ∈ {0.5, 1.0, 3.0, 8.0}) to quantify the privacy-utility trade-off.
* Ensure train/val/test splits are machine-wise to avoid leakage.
* Save preprocessing artifacts (scalers, encoders) for reproducible inference.

---

