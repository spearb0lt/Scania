# Version_1(tabtf) — Clean TabTransformer Baseline

First modular refactor of the training code. No differential privacy. Establishes the modular file structure.

---

## Files

| File | Description |
|---|---|
| `models.py` | `TimeSeriesEmbedder` (2-layer TransformerEncoder, last-step pooling) + `CombinedRULModel` (TabTransformer, depth=6, heads=8). Includes detailed inline comments explaining pooling choices. |
| `trainer.py` | Main training loop. Loads data from CSV, constructs sliding windows, runs 80/20 split, trains for `NUM_EPOCHS=20` with Adam + MSELoss. No DP. |
| `inference.py` | Inference script for a saved artifact directory. |
| `services.py` | `get_criterion()` → `MSELoss`, `get_optimizer(model, lr)` → `Adam`. |
| `utils.py` | `create_X_y()` (sliding window constructor), `load_and_encode_specs()`, `RULCombinedDataset`, `train_val_split()`. |

---

## Running

```bash
# Edit CSV_PATH and SPEC_CSV_PATH in trainer.py, then:
python trainer.py
```

---

## Model Architecture

```
TimeSeriesEmbedder:
  input_proj: Linear(105 → 128)
  encoder:    TransformerEncoder(2 layers, 8 heads, dropout=0.1)
  pooling:    x[:, -1, :]  ← last-step pooling

CombinedRULModel:
  tf:    TimeSeriesEmbedder  → (batch, 128)
  tabtf: TabTransformer(categories=(3,29,21,4,2,5,17,9), depth=6, heads=8)
         → (batch, 1)
```

<img width="247" height="214" alt="Version 1 training output" src="https://github.com/user-attachments/assets/52ed5fcd-4b4a-4b94-b00e-37ea6bed96e7" />
