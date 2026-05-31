# Version_2(tabtf+dp+notes) — TabTransformer + DP-SGD + Artifact Logging

Extends Version 1 with DP-SGD support and the artifact logging system. Training runs now save `metadata.json` and `train_val_log.txt` alongside `checkpoint.pth`.

---

## Files

| File | Description |
|---|---|
| `models.py` | Same `TimeSeriesEmbedder` + `CombinedRULModel` as Version 1 |
| `trainer.py` | Training loop with `pvt` (DP) flag, `StepLR` + early stopping (`ES_PATIENCE=11`), and full artifact generation. Sets `use_h5=True/False` to load from HDF5 or regenerate from CSV. |
| `inference.py` | Loads artifact by folder path and runs inference |
| `services.py` | Loss & optimizer factory |
| `utils.py` | Extended: adds `save_to_h5()`, `load_from_h5()`, `make_artifact_folder()`, `H5_PATH`, `ENCODER_PATH` constants |

---

## Key Additions vs Version 1

- **DP-SGD**: `pvt = True` enables per-sample gradient clipping + Gaussian noise. `max_grad_norm=1.0`, `noise_multiplier=1.0`.
- **Artifact system**: timestamped folder `CombinedRULModel-{DP|NDP}-{timestamp}/` with `checkpoint.pth`, `metadata.json`, `train_val_log.txt`.
- **H5 caching**: set `use_h5=True` to load pre-built windows from `data_windows.h5` for fast restarts.
- **LR scheduling**: `StepLR` with `LR_PATIENCE=5` and `LR_FACTOR=0.5`.
- **Early stopping**: halts at `ES_PATIENCE=11` consecutive non-improving epochs.

---

## Running

```bash
# Set use_h5, pvt, hyperparameters inside trainer.py, then:
python trainer.py
```

---

## Training Outputs

<img width="247" height="214" alt="Version 2 training" src="https://github.com/user-attachments/assets/13d6b1d8-2e8b-4ed0-a2d3-9095c6a26c00" />
<img width="363" height="138" alt="Train/val loss" src="https://github.com/user-attachments/assets/adb461a4-a260-4755-814c-f71a054a71a8" />
<img width="358" height="236" alt="LR schedule" src="https://github.com/user-attachments/assets/dacb7aca-cc69-4cbf-a89d-e0da63d9db87" />
<img width="314" height="484" alt="Metadata JSON" src="https://github.com/user-attachments/assets/4628f96c-11ff-4966-a0de-5f6cf04b0a53" />
<img width="457" height="473" alt="Artifact folder" src="https://github.com/user-attachments/assets/f0d2bc89-3d2a-4687-b10f-cc805f9800b0" />
