# Version_1(tabtf) -- Clean TabTransformer Baseline

First modular refactor of the training code. No differential privacy. Establishes
the modular file structure used in all subsequent versions.

---

## Files

| File | Description |
|---|---|
| `models.py` | `TimeSeriesEmbedder` (2-layer TransformerEncoder, last-step pooling, d_model=128) + `CombinedRULModel` (TabTransformer, depth=6, heads=8). Includes detailed inline comments explaining pooling choices (last-step vs mean vs max pooling). |
| `trainer.py` | Main training loop. Loads data from CSV, constructs sliding windows, runs 80/20 train/val split, trains for `NUM_EPOCHS=20` with Adam + MSELoss. No DP. Saves best checkpoint. |
| `inference.py` | Inference script: loads a saved artifact and evaluates on held-out data. |
| `services.py` | `get_criterion()` returns MSELoss; `get_optimizer(model, lr)` returns Adam. |
| `utils.py` | `create_X_y()` (sliding window constructor from CSV), `load_and_encode_specs()`, `RULCombinedDataset`, `train_val_split()`. |

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
  input_proj : nn.Linear(105, 128)
  encoder    : nn.TransformerEncoder(2 layers, 8 heads, dropout=0.1)
  pooling    : x[:, -1, :]   <- last-step pooling -> (batch, 128)

CombinedRULModel:
  tf     : TimeSeriesEmbedder     -> (batch, 128)
  tabtf  : TabTransformer(
             categories=(3,29,21,4,2,5,17,9),
             depth=6, heads=8
           )                       -> (batch, 1)
```

---

<img width="247" height="214" alt="Version 1 training output" src="https://github.com/user-attachments/assets/52ed5fcd-4b4a-4b94-b00e-37ea6bed96e7" />
