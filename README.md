# Cross-Industry Privacy-Preserving Framework for PdM
### Based on IDA 24 Industrial Challenge (SCANIA-X Dataset)

> **Currently the best RUL prediction model on the SCANIA-X dataset â€” MSE 2725 with full differential privacy, MIA AUC 49.12% (near-random), MIA Accuracy 49.59%.**

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Architecture](#architecture)
- [Differential Privacy](#differential-privacy)
- [Federated Learning](#federated-learning)
- [Membership Inference Attack (MIA)](#membership-inference-attack-mia)
- [Repository Structure](#repository-structure)
- [Data & Preprocessing](#data--preprocessing)
- [Installation](#installation)
- [Running Experiments](#running-experiments)
- [Artifact System](#artifact-system)
- [Gallery](#gallery)

---

## Overview

This repository implements an end-to-end **privacy-preserving Predictive Maintenance (PdM)** framework designed to be **cross-industry applicable** to any dataset with numerical, categorical, or multimodal features. It was built on the **IDA 2024 Industrial Challenge** using the SCANIA-X truck telemetry dataset.

**Core contributions:**

1. **Hybrid TabTransformer Architecture** â€” A two-stage model that encodes raw time-series sensor windows into compact embeddings via a `TimeSeriesEmbedder` (Transformer encoder), then fuses them with categorical vehicle specification features through a `TabTransformer` for final RUL regression. This design minimizes information loss across both numerical and categorical modalities.

2. **Custom Differential Privacy algorithms** â€” Two independently implemented DP mechanisms: Spectral-DP (SVD-domain gradient perturbation) and DP-SGD (per-sample gradient clipping). A custom RÃ©nyi Differential Privacy (RDP) accountant based on the Moments Accountant method tracks the privacy budget (Îµ, Î´).

3. **Federated Learning with Flower (flwr)** â€” A federated training loop supporting heterogeneous clients with different compute power, batch sizes, local epochs, learning rates, and per-client DP settings â€” mirroring real-world industrial deployments.

4. **Advanced MIA evaluation** â€” A comprehensive Membership Inference Attack framework combining white-box, gray-box, black-box, and time-series-specific seasonality/trend features to rigorously quantify privacy guarantees.

---

## Key Results

| Metric | Value |
|---|---|
| Best Val MSE (NDP baseline) | ~23 (normalized scale) |
| Best Val MSE (DP â€” privacy-preserving) | **2725** |
| MIA AUC | **49.12%** (â‰ˆ random guessing) |
| MIA Accuracy | **49.59%** (â‰ˆ random guessing) |
| Dataset | SCANIA-X (IDA 2024 Industrial Challenge) |
| Sensor Features | 105 numerical |
| Categorical Features | 8 (Spec_0â€¦Spec_7) |
| Sliding Window Length | 70 time steps |

> MIA near 50% confirms the DP model does not memorize training data â€” the privacy guarantee is empirically validated.

---

## Architecture

### `TimeSeriesEmbedder`

Takes a sliding window of shape `(batch, 70, 105)` and produces a fixed-size embedding per window.

```
Input: (batch, context_length=70, num_features=105)
  â†’ Linear projection:   (batch, 70, d_model=128/256)
  â†’ TransformerEncoder:  2 layers, 8 heads, dropout=0.1
  â†’ Last-step pooling    x[:, -1, :]:  (batch, d_model)
Output: (batch, 128) embedding
```

> Last-step pooling (causal) was found to outperform mean-pooling for RUL prediction.

### `CombinedRULModel`

Fuses the time-series embedding with 8 categorical vehicle specification features.

```
Inputs:
  x_cat : (batch, 8)       â€” ordinal-encoded vehicle specs (Spec_0â€¦Spec_7)
  x_ts  : (batch, 70, 105) â€” raw sensor windows

Pipeline:
  1. ts_embedding = TimeSeriesEmbedder(x_ts)     â†’ (batch, 128)
  2. out = TabTransformer(
         categories=(3,29,21,4,2,5,17,9),
         num_continuous=128,
         dim=128, depth=6, heads=8,
         attn_dropout=0.1, ff_dropout=0.1,
         mlp_hidden_mults=(4,2)
     )(x_cat, ts_embedding)
  3. Output: (batch, 1)  â€” RUL prediction
```

**Category sizes** (cardinalities of Spec_0â€¦Spec_7): `(3, 29, 21, 4, 2, 5, 17, 9)`

### Why this design generalizes across industries

- `TimeSeriesEmbedder` accepts any `num_features` â€” drop-in replacement for other sensor sets.
- `TabTransformer` accepts any number of categorical columns with any cardinality.
- Interface: *sliding windows of sensor data + categorical metadata â†’ scalar RUL* â€” maps to aerospace, manufacturing, HVAC, and any other PdM domain.

### Model Plots

<img width="484" height="338" alt="TabTransformer model architecture" src="https://github.com/user-attachments/assets/0d11917c-2fb8-442d-b2a3-8dfb83dd5901" />
<img width="490" height="333" alt="Training convergence NDP" src="https://github.com/user-attachments/assets/051579a1-fea0-401c-b606-176f1b6c2080" />
<img width="492" height="334" alt="Val loss curve" src="https://github.com/user-attachments/assets/813f1d8d-3661-4462-95c6-f3cf89cc6293" />
<img width="483" height="295" alt="Prediction vs actual" src="https://github.com/user-attachments/assets/c634d489-8617-433f-8514-72d2e3a8632f" />
<img width="755" height="578" alt="Layer visualization" src="https://github.com/user-attachments/assets/434a0e3e-fd45-4fa3-8508-82471c5d755a" />
<img width="792" height="719" alt="Attention weights" src="https://github.com/user-attachments/assets/17975aab-e377-4273-b73a-fb90a90fbf3c" />
<img width="779" height="316" alt="Training reconstruction" src="https://github.com/user-attachments/assets/a22de429-b902-4a2a-9a3e-530bdeabedd8" />
<img width="797" height="341" alt="Model reconstruction plot" src="https://github.com/user-attachments/assets/16d3bc7f-19e0-413e-823e-6c6814daa901" />

---

## Differential Privacy

### Spectral-DP (Custom, gradient-domain SVD)

A novel custom DP mechanism that operates in the **spectral (singular value) domain** of gradient matrices. Gradient noise is injected at the SVD representation of each gradient tensor, providing better utility than standard Gaussian mechanisms.

**Algorithm per gradient tensor:**

```
1. Compute gradient G  of loss w.r.t. parameter
2. Reshape G to 2D: G_2d âˆˆ â„^{mÃ—n}
3. SVD:     G_2d = U Â· diag(S) Â· Váµ€
4. Clip:    S_clipped = clamp(S, max=C)
   (opt.)  Top-k filter: zero all but top-k singular values
5. Noise:   S_noisy = S_clipped + N(0, ÏƒÂ²CÂ²)
6. Rebuild: G_noisy = U Â· diag(S_noisy) Â· Váµ€
7. Assign:  param.grad â† G_noisy
```

Fallback for 1D gradients (biases): standard clipping + Gaussian noise.

**Dynamic noise schedule**: Ïƒ is linearly annealed from `NOISE_START=0.8` â†’ `NOISE_END=0.4` over training.

### DP-SGD (Custom per-sample clipping)

From-scratch implementation of differentially private SGD:

```
For each sample i in batch:
  1. Per-sample gradient:  g_i  via  autograd.grad
  2. Clip:  g_i_clipped = g_i Ã— min(1, C / â€–g_iâ€–)
  3. Accumulate: G_sum += g_i_clipped
After batch:
  4. Add noise: G_noisy = G_sum + N(0, ÏƒÂ²CÂ²Â·I)
  5. param.grad = G_noisy
```

> No `loss.backward()` is used in DP-SGD â€” gradients are set manually after per-sample accumulation.

### Privacy Accounting (Custom RDP Accountant)

`custom_dp.py` implements the full **Moments Accountant / RÃ©nyi Differential Privacy** accountant from scratch (based on [arXiv:1908.10530](https://arxiv.org/pdf/1908.10530.pdf)):

- Computes `RDP(Î±)` for both integer and fractional RÃ©nyi orders Î±
- Converts RDP â†’ (Îµ, Î´)-DP via the standard conversion theorem
- Sweeps Î± âˆˆ [1.1, 63] (dense low-to-mid range) for tight bounds
- `DELTA = 1 / N^1.1` where N is dataset size

### Unified Training (`all_everything_v1.py`)

All three modes are controlled by the `dp` flag:

```python
dp = "spectral"   # Spectral-DP
dp = "dp_sgd"     # DP-SGD
dp = "none"       # Baseline (no privacy)
```

| Hyperparameter | Value |
|---|---|
| `BATCH_SIZE` | 256 (local) / 1024 (server) |
| `EMBED_DIM` | 256 |
| `NUM_EPOCHS` | 20â€“100 |
| `LR` | 1e-3 |
| `MAX_GRAD_NORM` (C) | 1.0 |
| `NOISE_START` / `NOISE_END` | 0.8 / 0.4 |
| `DELTA` | 1/N^1.1 |
| Early stop patience | 100 epochs |
| TS-Encoder freeze | at `patience//2` no-improve epochs |

### SVD in Spectral-DP â€” Mathematical Derivation

<img width="261" height="205" alt="SVD decomposition" src="https://github.com/user-attachments/assets/15419292-a910-4f3b-afe3-fa95243ef0b8" />
<img width="485" height="50" alt="SVD formula" src="https://github.com/user-attachments/assets/24d514b2-eb80-4f67-9830-ecd7a194064a" />
<img width="471" height="231" alt="Spectral noise injection" src="https://github.com/user-attachments/assets/1846e37c-6261-4248-b3d4-3e72eb8db061" />
<img width="411" height="50" alt="Clipping bound" src="https://github.com/user-attachments/assets/6d3b23a6-50f1-4476-ae0f-13cdde91edb3" />
<img width="549" height="116" alt="Privacy guarantee" src="https://github.com/user-attachments/assets/bb2ba1a8-ac2f-4db4-baff-249c13b5993c" />
<img width="505" height="83" alt="RDP conversion" src="https://github.com/user-attachments/assets/5368293d-37fc-469c-b55c-903593412272" />
<img width="422" height="404" alt="SVD geometric interpretation" src="https://github.com/user-attachments/assets/e3e988be-4eb3-4d18-8e9e-e6557f3aa31b" />

---

## Federated Learning

Built on the **Flower (`flwr`)** library with two federation strategies.

### FedDiff â€” Heterogeneous Clients

Each client receives a different per-round config from the server, simulating facilities with different hardware and privacy constraints.

**Server** (`feddiff_server.py`): Custom `HeterogeneousStrategy` extends `FedAvg`. Dispatches per-client configs; saves the global model after every aggregation round.

**Clients** (`feddiff_client.py`, `feddiff_client+spectraldp.py`): `flwr.client.NumPyClient` reads config in `fit()`.

| Config Key | Example Values |
|---|---|
| `local_epochs` | 1â€“5 |
| `batch_size` | 64â€“256 |
| `lr` | 5e-4 â€“ 1e-3 |
| `use_dp` | `True` / `False` |
| `dp_sigma` | 0.1 |
| `dp_clip_bound` | 1.0 |
| `dp_spec_k` | 2 (top-k singular values) |

Data is partitioned by **vehicle ID** across clients (non-overlapping fleets).

### FedSame â€” Homogeneous Clients

All clients share the same config (`fedsame_client.py`, `fedsame_server.py`). Uses standard `FedAvg`. Suited for benchmarking.

### Running Federated Training

```bash
# Terminal 1 â€” server
python FederatedApproach/feddiff_server.py

# Terminal 2 â€” client 0
python FederatedApproach/feddiff_client+spectraldp.py --client-id 0

# Terminal 3 â€” client 1
python FederatedApproach/feddiff_client+spectraldp.py --client-id 1
```

### Federated Results

<img width="651" height="147" alt="Federated rounds" src="https://github.com/user-attachments/assets/14bbe4da-55fa-497f-8940-4c81848839be" />
<img width="524" height="319" alt="Federated convergence" src="https://github.com/user-attachments/assets/61c75505-b536-49ce-b393-4ac4d2bfacff" />
<img width="692" height="320" alt="Client training" src="https://github.com/user-attachments/assets/1608d28c-70eb-412d-8a5e-6899d87f79cb" />
<img width="708" height="166" alt="Server log" src="https://github.com/user-attachments/assets/52bb5040-415f-49bb-800e-49a6c9fda410" />
<img width="673" height="716" alt="Heterogeneous client results" src="https://github.com/user-attachments/assets/32209bdd-0f91-4ca1-b7df-8cc1314b1576" />

---

## Membership Inference Attack (MIA)

Implemented in `Important_script_part2/Membership-Inference-Attack(MIA).ipynb`. The attack was designed to **empirically validate** the privacy guarantees of the trained models. It combines:

- **Black-box features**: loss values, confidence scores, output statistics
- **Gray-box features**: gradient norms, output distributions
- **White-box features**: per-layer activation patterns
- **Time-series specific features**: seasonality components, trend decomposition from sensor windows

### MIA Results

| Metric | DP Model | NDP Model |
|---|---|---|
| AUC | **49.12%** (â‰ˆ random) | Higher (memorization present) |
| Accuracy | **49.59%** (â‰ˆ random) | Higher (memorization present) |

AUC 49.12% is effectively **below random guessing** â€” confirming no membership leakage from the DP-trained model.

<img width="362" height="58" alt="MIA results" src="https://github.com/user-attachments/assets/4a17b3a1-bbf3-4bdb-9ea7-ee3ab89f2fed" />
<img width="220" height="166" alt="MIA ROC" src="https://github.com/user-attachments/assets/fc6856da-546c-45cf-852a-b750040890d9" />
<img width="334" height="94" alt="MIA features" src="https://github.com/user-attachments/assets/db630c4e-80f6-4538-b424-7cdea0f7ab58" />
<img width="204" height="83" alt="MIA confusion matrix" src="https://github.com/user-attachments/assets/06b99139-75a1-4b4a-9557-d5bad0021ec7" />
<img width="203" height="78" alt="MIA precision-recall" src="https://github.com/user-attachments/assets/d3f20392-4fb1-4dac-b732-4db6fecc92f5" />
<img width="312" height="37" alt="MIA summary" src="https://github.com/user-attachments/assets/f7b1e683-8bfc-4375-bd2b-c506decf6c1b" />
<img width="282" height="159" alt="MIA distribution" src="https://github.com/user-attachments/assets/8306efdd-7e06-4bef-9bc5-dfaa635717b3" />




---


**Short description**
This repository implements a TabTransformer-based model for Remaining Useful Life (RUL) prediction on the Scania dataset, with an end-to-end preprocessing pipeline and differential-privacy-enabled training. It also contains an evaluation and Membership Inference Attack (MIA) framework so you can measure privacy/utility trade-offs.

---

## Repository Structure

```
.
â”œâ”€â”€ README.md
â”‚
â”œâ”€â”€ FInal_script/                         â† Production training scripts
â”‚   â”œâ”€â”€ all_everything_v1.py              â† Unified script: Spectral-DP / DP-SGD / NDP
â”‚   â”œâ”€â”€ spectral_dp+tabtf_v1.py          â† Spectral-DP only (server-optimized, batch=1024)
â”‚   â”œâ”€â”€ tabtf+dpsgd(my).py               â† DP-SGD custom implementation
â”‚   â”œâ”€â”€ tabtf+dpsgd(sirs).py             â† DP-SGD with custom_dp library
â”‚   â”œâ”€â”€ custom_dp.py                      â† RDP / Moments Accountant privacy accounting
â”‚   â”œâ”€â”€ custom_dp (1).py                  â† Extended privacy accounting utilities
â”‚   â”œâ”€â”€ future_log_generator.py           â† Artifact generation helper (log + metadata)
â”‚   â”œâ”€â”€ COMMANDS_for_server.txt           â† HPC / SLURM job commands
â”‚   â””â”€â”€ README.md
â”‚
â”œâ”€â”€ FederatedApproach/                    â† Federated learning with Flower
â”‚   â”œâ”€â”€ feddiff_server.py                 â† Heterogeneous FL server (HeterogeneousStrategy)
â”‚   â”œâ”€â”€ feddiff_client.py                 â† Heterogeneous FL client (base)
â”‚   â”œâ”€â”€ feddiff_client+spectraldp.py      â† Heterogeneous FL client + Spectral-DP
â”‚   â”œâ”€â”€ fedsame_server.py                 â† Homogeneous FL server (FedAvg)
â”‚   â”œâ”€â”€ fedsame_client.py                 â† Homogeneous FL client
â”‚   â””â”€â”€ README.md
â”‚
â”œâ”€â”€ Important_script_part1/               â† Development phase: notebooks & initial models
â”‚   â”œâ”€â”€ Data-Processing-Detailed.ipynb    â† Step-by-step data cleaning and EDA
â”‚   â”œâ”€â”€ Data-Preprocessing-Automated.ipynb â† Automated preprocessing pipeline
â”‚   â”œâ”€â”€ initial_models.ipynb              â† Baseline models: LSTM, GRU, VAE, Transformer
â”‚   â”œâ”€â”€ Modelling_part1.ipynb             â† First TabTransformer with context vector
â”‚   â”œâ”€â”€ TabTransformer_dyn-hrd-path.ipynb â† TabTransformer (dynamic + hardcoded paths)
â”‚   â”œâ”€â”€ TabTransformer+layervisualization.ipynb â† Layer-wise visualization
â”‚   â”œâ”€â”€ Basic plottings .ipynb            â† EDA visualizations
â”‚   â”œâ”€â”€ Inference_on_saved_model.py       â† Inference script for saved checkpoints
â”‚   â”œâ”€â”€ load_saved_model.ipynb            â† Interactive model loading notebook
â”‚   â”œâ”€â”€ data_windows.h5                   â† Preprocessed sliding-window data (HDF5)
â”‚   â”œâ”€â”€ spec_encoder.joblib               â† Saved OrdinalEncoder for categorical specs
â”‚   â”œâ”€â”€ artifacts/                        â† Training run outputs (DP & NDP)
â”‚   â””â”€â”€ README.md
â”‚
â”œâ”€â”€ Important_script_part2/               â† Privacy evaluation phase
â”‚   â”œâ”€â”€ spectral_dp+tabtf_v0.py          â† Initial Spectral-DP + TabTransformer script
â”‚   â”œâ”€â”€ mathematical_logic_for_spectralDP.ipynb â† Full SVD/DP mathematical derivation
â”‚   â”œâ”€â”€ Membership-Inference-Attack(MIA).ipynb â† Complete MIA framework
â”‚   â”œâ”€â”€ data_windows.h5                   â† Preprocessed data
â”‚   â”œâ”€â”€ spec_encoder.joblib               â† Spec encoder
â”‚   â”œâ”€â”€ artifacts/                        â† Training runs (multiple DP configs)
â”‚   â”œâ”€â”€ artifacts2/                       â† Additional training runs
â”‚   â””â”€â”€ README.md
â”‚
â””â”€â”€ Modular_Approach/                     â† Refactored, modular codebase
    â”œâ”€â”€ Version_1(tabtf)/                 â† Clean TabTransformer (no DP)
    â”‚   â”œâ”€â”€ models.py                     â† TimeSeriesEmbedder + CombinedRULModel
    â”‚   â”œâ”€â”€ trainer.py                    â† Training loop
    â”‚   â”œâ”€â”€ inference.py                  â† Inference script
    â”‚   â”œâ”€â”€ services.py                   â† Loss & optimizer factory
    â”‚   â”œâ”€â”€ utils.py                      â† Data loading, windowing, dataset classes
    â”‚   â””â”€â”€ README.md
    â”œâ”€â”€ Version_2(tabtf+dp+notes)/        â† TabTransformer + DP-SGD + artifact logging
    â”‚   â”œâ”€â”€ models.py, trainer.py, ...    â† Same modular structure + DP support
    â”‚   â””â”€â”€ README.md
    â””â”€â”€ Version_3(better notes+functionality)/ â† Latest modular version
        â”œâ”€â”€ models.py                     â† Final clean model definitions
        â”œâ”€â”€ trainer.py                    â† Full training loop with DP + artifact logging
        â”œâ”€â”€ inference.py                  â† Batch inference with artifact loading
        â”œâ”€â”€ services.py, utils.py         â† Utilities
        â”œâ”€â”€ loss_plotter.ipynb            â† Training curve visualization
        â”œâ”€â”€ data_windows.h5, spec_encoder.joblib
        â””â”€â”€ README.md
```

---

## Data & Preprocessing

### Dataset: SCANIA-X (IDA 2024 Industrial Challenge)

The SCANIA-X dataset contains operational telemetry logs from Scania trucks.

**Features used:**
- **105 numerical sensor features** â€” time-series operational measurements (e.g. `171_0`, `666_0`, `427_0`, `167_0`â€¦`167_9`, `272_0`â€¦`272_9`, `291_0`â€¦`291_10`, `459_0`â€¦`459_19`, `397_0`â€¦`397_35`, etc.)
- **8 categorical specification features** â€” vehicle configuration metadata (`Spec_0`â€¦`Spec_7`) with cardinalities `(3, 29, 21, 4, 2, 5, 17, 9)`

### Sliding Window Construction

Data is segmented into overlapping windows of **70 consecutive time steps** per vehicle, each labeled with the RUL at the end of the window.

- Window tensor: `(N_windows, 70, 105)`
- Labels: `(N_windows,)` â€” RUL
- Vehicle IDs: tracked per window for federated data partitioning
- Specifications: `(N_windows, 8)` â€” repeated per window

### Pipeline

1. Load raw normalized CSV (`super_same_norm.csv`)
2. Load vehicle specifications (`train_specifications.csv`)
3. Ordinal-encode 8 categorical spec columns â†’ save `spec_encoder.joblib`
4. Construct sliding windows with `context_length=70`
5. 80/20 train/val split, `random_state=42`
6. Save to HDF5 (`data_windows.h5`) for fast reloading

### HDF5 Schema

```
data_windows.h5
â”œâ”€â”€ X_windows        : float32  (N, 70, 105)
â”œâ”€â”€ y_labels         : float32  (N,)
â”œâ”€â”€ window_vids      : int64    (N,)    â€” vehicle IDs per window
â””â”€â”€ specs_per_window : int64    (N, 8)  â€” ordinal-encoded specs
```

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd Scania

# Create virtual environment (Python 3.10 recommended)
conda create -n scania-pdm python=3.10 -y
conda activate scania-pdm

# Install dependencies
pip install torch torchvision
pip install tab-transformer-pytorch
pip install flwr
pip install h5py joblib scikit-learn numpy scipy
pip install pandas matplotlib
```

> **GPU note:** Spectral-DP training is significantly faster on a CUDA GPU due to `torch.linalg.svd` per gradient. Add `torch.backends.cuda.preferred_linalg_library("magma")` for best performance.

> **HPC / SLURM:** See `FInal_script/COMMANDS_for_server.txt` for cluster job commands.

---

## Running Experiments

### 1. Preprocess data (first time)

Run `Important_script_part1/Data-Preprocessing-Automated.ipynb`, or set `use_h5 = False` in any `trainer.py` to auto-generate from CSV.

### 2. Standalone training (`FInal_script`)

```python
# In all_everything_v1.py, set:
dp = "spectral"   # Spectral-DP
dp = "dp_sgd"     # DP-SGD
dp = "none"       # Baseline

python FInal_script/all_everything_v1.py
```

### 3. Modular training (recommended)

```bash
cd "Modular_Approach/Version_3(better notes+functionality)/"

# Edit trainer.py: set use_h5, pvt (DP flag), hyperparameters
python trainer.py

# Run inference on a saved artifact
python inference.py
```

### 4. Federated training

```bash
# Terminal 1 â€” server
python FederatedApproach/feddiff_server.py

# Terminal 2 â€” client 0
python FederatedApproach/feddiff_client+spectraldp.py --client-id 0

# Terminal 3 â€” client 1
python FederatedApproach/feddiff_client+spectraldp.py --client-id 1
```

### 5. MIA evaluation

Open and run `Important_script_part2/Membership-Inference-Attack(MIA).ipynb` pointing to a trained artifact directory.

---

## Artifact System

Every training run auto-creates a timestamped artifact folder:

```
artifacts/
â””â”€â”€ CombinedRULModel-{DP|NDP}-{YYYYMMDD_HHMMSS}/
    â”œâ”€â”€ checkpoint.pth      â† Best model weights (state_dict)
    â”œâ”€â”€ metadata.json       â† Full config snapshot
    â””â”€â”€ train_val_log.txt   â† Per-epoch CSV: epoch,train_loss,val_loss,time,lr,notes
```

**`metadata.json` schema:**
```json
{
  "model_name": "CombinedRULModel",
  "num_sensor_features": 105,
  "context_length": 70,
  "continuous_dim": 128,
  "categories": [3, 29, 21, 4, 2, 5, 17, 9],
  "batch_size": 256,
  "learning_rate": 0.001,
  "num_epochs": 50,
  "privacy": "Spectral-DP",
  "dp_sigma": 0.1,
  "dp_clip_bound": 1.0,
  "total_training_time": "01:23:45"
}
```

---

## Gallery

### Artifact Generation

<img width="396" height="215" alt="Artifact folder structure" src="https://github.com/user-attachments/assets/5a8a5fba-9014-4af0-bc02-380be54b5ae2" />
<img width="348" height="218" alt="Training log CSV" src="https://github.com/user-attachments/assets/b4f6f16a-6438-4672-8d1f-7a4402fe71b0" />
<img width="452" height="509" alt="Artifact folder tree" src="https://github.com/user-attachments/assets/32e2e05c-bce7-4027-924b-73b8100c9ceb" />
<img width="589" height="528" alt="Metadata JSON view" src="https://github.com/user-attachments/assets/b255ceeb-7ece-4c5d-945c-bb47d9c23d29" />
