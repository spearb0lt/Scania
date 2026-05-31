# Modular_Approach -- Refactored Codebase

This folder contains a clean, modular refactoring of the monolithic training scripts
from `FInal_script/` and `Important_script_part1/`. The code is split into separate
files by concern (models, trainer, inference, services, utils), making it easier to
iterate, debug, and extend.

---

## Versions

### `Version_1(tabtf)/` -- Clean TabTransformer (No DP)

First modular refactor. Baseline `CombinedRULModel` (TimeSeriesEmbedder +
TabTransformer) with no differential privacy. Establishes the file structure used
in all subsequent versions.

### `Version_2(tabtf+dp+notes)/` -- TabTransformer + DP-SGD + Artifact Logging

Adds DP-SGD support via the `pvt` flag and the artifact logging system
(`make_artifact_folder`, `metadata.json`, `train_val_log.txt`). Includes detailed
inline notes explaining design decisions.

### `Version_3(better notes+functionality)/` -- Latest (recommended)

The most complete and clean version. Includes:
- Full DP-SGD support via `pvt` flag in `trainer.py`
- Comprehensive inline documentation in all files
- `loss_plotter.ipynb` for training curve visualization
- Self-contained `data_windows.h5` and `spec_encoder.joblib`
- Definitive `models.py` used as reference in the final paper

---

## Shared File Structure (all versions)

| File | Description |
|---|---|
| `models.py` | `TimeSeriesEmbedder` + `CombinedRULModel` class definitions |
| `trainer.py` | Training loop with early stopping, LR scheduling, artifact generation |
| `inference.py` | Load a saved artifact directory and run batch inference |
| `services.py` | `get_criterion()` -> MSELoss, `get_optimizer()` -> Adam |
| `utils.py` | `create_X_y()`, `save_to_h5()`, `load_from_h5()`, `RULCombinedDataset`, `train_val_split()`, `make_artifact_folder()` |

---

<img width="247" height="214" alt="Modular folder structure" src="https://github.com/user-attachments/assets/deed794a-f192-4576-943c-04243c904072" />
