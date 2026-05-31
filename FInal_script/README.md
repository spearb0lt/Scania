# FInal_script -- Production Training Scripts

This folder contains the final, production-ready scripts used to generate the published
results. All scripts implement the `CombinedRULModel` (TimeSeriesEmbedder + TabTransformer)
on the SCANIA-X dataset and support Spectral-DP, DP-SGD, and baseline (no-DP) training.

---

## Scripts

### `all_everything_v1.py` -- Unified Training Script (primary)

The main consolidated training script. Mode is controlled by the `dp` variable:

```python
dp = "spectral"   # Spectral-DP: SVD-domain gradient perturbation
dp = "dp_sgd"     # DP-SGD: per-sample gradient clipping + Gaussian noise
dp = "none"       # Baseline: no differential privacy
```

Key features:
- `CombinedRULModel`: `TimeSeriesEmbedder` (2-layer TransformerEncoder, last-step pooling)
  fused with `TabTransformer` (depth=6, heads=8, embed_dim=256)
- Dynamic noise annealing: sigma linearly decayed from 0.8 to 0.4 over training
- DP-SGD: manual `autograd.grad` per-sample loop with per-sample clipping
- `TimeSeriesEmbedder` layer freeze triggered at `patience // 2` no-improve epochs
- Integrated artifact generation via `future_log_generator.py`
- RDP privacy budget tracking via `custom_dp.compute_dp_sgd_privacy()`

### `spectral_dp+tabtf_v1.py` -- Spectral-DP, Server-Optimized

Spectral-DP training tuned for HPC/GPU servers:
- `BATCH_SIZE = 1024` (vs 256 locally -- larger batch amplifies privacy)
- `NUM_EPOCHS = 100`
- AdamW optimizer + CosineAnnealingLR scheduler
- `torch.backends.cuda.preferred_linalg_library("magma")` for faster SVD on CUDA

### `tabtf+dpsgd(my).py` -- DP-SGD Custom Implementation

Standalone custom DP-SGD without any third-party DP libraries.

### `tabtf+dpsgd(sirs).py` -- DP-SGD with custom_dp

DP-SGD implementation using `custom_dp.py` and `custom_dp (1).py` for privacy
accounting alongside the gradient manipulation.

### `custom_dp.py` -- RDP / Moments Accountant

Full from-scratch Renyi Differential Privacy accountant based on arXiv:1908.10530.
Implements `compute_dp_sgd_privacy(n, batch_size, noise_multiplier, epochs, delta, alphas)`
which returns the (epsilon, delta)-DP guarantee for the current training configuration.

Supports integer and fractional Renyi orders alpha.

### `custom_dp (1).py` -- Extended Privacy Utilities

Additional privacy accounting helper functions used by `tabtf+dpsgd(sirs).py`.

### `future_log_generator.py` -- Artifact System Helper

Creates timestamped artifact folders and writes:
- `checkpoint.pth` -- best model weights
- `metadata.json` -- full training config
- `train_val_log.txt` -- per-epoch CSV (epoch, train_loss, val_loss, time, lr, notes)

### `COMMANDS_for_server.txt` -- HPC Commands

SLURM / HPC commands for submitting jobs on the university cluster
(`/csehome/p23iot002/Shubhro/`).

---

## Key Hyperparameters

| Parameter | Local | Server (spectral_dp+tabtf_v1) |
|---|---|---|
| `BATCH_SIZE` | 256 | 1024 |
| `EMBED_DIM` | 256 | 256 |
| `NUM_EPOCHS` | 20 | 100 |
| `LR` | 1e-3 | 1e-3 |
| `MAX_GRAD_NORM` (C) | 1.0 | 1.0 |
| `NOISE_START` | 0.8 | -- (fixed 0.1 in v1) |
| `NOISE_END` | 0.4 | -- |
| `DELTA` | 1/N^1.1 | 1/N^1.1 |
| Early stop patience | 100 | 100 |
| Renyi orders alpha | [1.1 ... 63] | [1.1 ... 63] |
| Optimizer | Adam / AdamW | AdamW |
| Scheduler | StepLR | CosineAnnealingLR |

---

## Running

```bash
# Set dp = "spectral" / "dp_sgd" / "none" inside the script, then:
python all_everything_v1.py

# Server-optimized Spectral-DP (batch=1024):
python spectral_dp+tabtf_v1.py
```
