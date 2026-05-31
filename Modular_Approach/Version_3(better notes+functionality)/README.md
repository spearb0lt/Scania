# Version_3(better notes+functionality) \u2014 Latest Modular Version

The most complete and production-ready version of the modular codebase. Includes comprehensive inline documentation, a self-contained dataset, and a loss plotter notebook. This is the recommended starting point for new experiments.

---

## Files

| File | Description |
|---|---|
| `models.py` | Clean `TimeSeriesEmbedder` + `CombinedRULModel` with clear docstrings. Last-step pooling. TabTransformer (depth=6, heads=8, attn_dropout=0.1, ff_dropout=0.1, mlp_hidden_mults=(4,2)). |
| `trainer.py` | Full training loop. Toggle `use_h5` (load from HDF5 vs regenerate), `pvt` (DP on/off). Includes early stopping, StepLR, artifact generation. |
| `inference.py` | Batch inference on a saved artifact directory. Auto-loads `metadata.json` to reconstruct model architecture. |
| `services.py` | `get_criterion()` → MSELoss, `get_optimizer()` → Adam. |
| `utils.py` | All data utilities: `create_X_y()`, `save_to_h5()`, `load_from_h5()`, `RULCombinedDataset`, `train_val_split()`, `make_artifact_folder()`. |
| `loss_plotter.ipynb` | Interactive notebook to load any `train_val_log.txt` and plot training/validation loss curves. |
| `data_windows.h5` | Self-contained preprocessed dataset (same format as other folders). |
| `spec_encoder.joblib` | Self-contained spec encoder. |

---

## Key Features vs Version 2

- **Better documentation**: every function and class has clear comments explaining design choices.
- **Self-contained**: includes its own `data_windows.h5` and `spec_encoder.joblib` — no path configuration needed.
- **Loss plotter**: `loss_plotter.ipynb` for visualizing training runs from any artifact folder.
- **Cleaner model code**: `models.py` is the definitive reference implementation used in the final paper.

---

## Running

```bash
cd "Modular_Approach/Version_3(better notes+functionality)/"

# Edit trainer.py: set use_h5=True (or False to regenerate), pvt=True/False
python trainer.py

# After training, run inference:
python inference.py

# Visualize training loss:
# Open loss_plotter.ipynb and point it to your artifact folder
```

---

## Training Outputs

<img width="247" height="214" alt="Version 3 training" src="https://github.com/user-attachments/assets/48fe0b29-cc43-4ef7-8426-6cf6a7b8c1e3" />
<img width="264" height="290" alt="Loss curves" src="https://github.com/user-attachments/assets/3e41ce35-b9cc-4073-800d-e43be2a9305a" />
<img width="470" height="366" alt="Train vs val MSE" src="https://github.com/user-attachments/assets/237aed64-55e2-40f6-ad18-68885e13fcda" />
<img width="392" height="465" alt="Artifact structure" src="https://github.com/user-attachments/assets/d083a50e-ce1d-43f6-b595-03becf7b7d5c" />
<img width="660" height="503" alt="Inference output" src="https://github.com/user-attachments/assets/908368ba-88de-4cdb-ba8a-2e2ca1e2ccbe" />
