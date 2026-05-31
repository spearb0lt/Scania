# FInal_script — Production Training Scripts

This folder contains the final, production-ready training scripts used to generate the published results. All scripts implement the `CombinedRULModel` (TimeSeriesEmbedder + TabTransformer) on the SCANIA-X dataset.

---

## Scripts

### `all_everything_v1.py` — Unified Training Script *(primary)*

The main training script supporting all three privacy modes via the `dp` flag:

```python
dp = "spectral"   # Spectral-DP (custom SVD-domain gradient perturbation)
dp = "dp_sgd"     # DP-SGD (custom per-sample gradient clipping)
dp = "none"       # Baseline (no privacy)
```

Key features:
- `CombinedRULModel`: `TimeSeriesEmbedder` (2-layer TransformerEncoder) + `TabTransformer` (depth=6, heads=8)
- Dynamic noise annealing: σ linearly decayed from 0.8 → 0.4 over training
- Per-sample gradient clipping for DP-SGD (manual `autograd.grad` loop)
- `TimeSeriesEmbedder` layer freeze after `patience//2` non-improving epochs
- Integrated artifact generation (logs, metadata, checkpoint)
- RDP privacy budget tracking via `custom_dp.py`

### `spectral_dp+tabtf_v1.py` — Spectral-DP Only (Server-Optimized)

Spectral-DP training tuned for HPC/GPU servers:
- `BATCH_SIZE = 1024` (vs 256 for local)
- `NUM_EPOCHS = 100`
- AdamW + CosineAnnealingLR scheduler
- Uses `torch.backends.cuda.preferred_linalg_library("magma")` for faster SVD

### `tabtf+dpsgd(my).py` — DP-SGD (Custom Implementation)

Custom from-scratch DP-SGD without third-party DP libraries.

### `tabtf+dpsgd(sirs).py` — DP-SGD with custom_dp Library

DP-SGD implementation using `custom_dp.py` and `custom_dp (1).py` for privacy accounting.

### `custom_dp.py` — RDP / Moments Accountant

Full from-scratch implementation of the Rényi Differential Privacy (RDP) accountant based on [arXiv:1908.10530](https://arxiv.org/pdf/1908.10530.pdf). Computes `compute_dp_sgd_privacy(n, batch_size, noise_multiplier, epochs, delta, alphas)` to track (ε, δ)-DP budget.

### `custom_dp (1).py` — Extended Privacy Utilities

Additional privacy accounting helpers used by `tabtf+dpsgd(sirs).py`.

### `future_log_generator.py` — Artifact System

Helper for creating timestamped artifact directories with:
- `checkpoint.pth` — best model weights
- `metadata.json` — full training config
- `train_val_log.txt` — per-epoch CSV (epoch, train_loss, val_loss, time, lr, notes)

### `COMMANDS_for_server.txt` — HPC Commands

SLURM / HPC commands for submitting training jobs on the university cluster (`/csehome/p23iot002/Shubhro/`).

---

## Key Hyperparameters

| Parameter | Value |
|---|---|
| `BATCH_SIZE` | 256 (local) / 1024 (server) |
| `EMBED_DIM` | 256 |
| `NUM_EPOCHS` | 20–100 |
| `LR` | 1e-3 |
| `MAX_GRAD_NORM` (C) | 1.0 |
| `NOISE_START` / `NOISE_END` | 0.8 / 0.4 |
| `DELTA` | 1/N^1.1 |
| Early stop patience | 100 epochs |
| Rényi orders α | [1.1…63] (dense) |

---

## Running

```bash
# Set dp = "spectral" / "dp_sgd" / "none" inside the script, then:
python all_everything_v1.py

# For server (Spectral-DP, batch=1024):
python spectral_dp+tabtf_v1.py
```
