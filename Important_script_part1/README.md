# Important_script_part1 -- Development Phase

This folder documents the iterative development journey from initial data exploration
to the first working privacy-preserving TabTransformer model. It contains notebooks
for EDA, preprocessing, baseline modelling, and early DP experiments, along with the
preprocessed dataset and encoder artifacts.

---

## Data Files

| File | Description |
|---|---|
| `data_windows.h5` | Preprocessed sliding-window HDF5 dataset. Schema: `X_windows (N,70,105)`, `y_labels (N,)`, `window_vids (N,)`, `specs_per_window (N,8)` |
| `spec_encoder.joblib` | Saved `sklearn.OrdinalEncoder` for 8 categorical specification columns. Category sizes: `(3,29,21,4,2,5,17,9)` |

---

## Notebooks and Scripts

### Data Processing

| File | Description |
|---|---|
| `Data-Processing-Detailed.ipynb` | Step-by-step data cleaning, feature exploration, and EDA. Documents each transformation decision. |
| `Data-Preprocessing-Automated.ipynb` | Automated end-to-end pipeline: load raw CSV -> normalize -> construct sliding windows -> encode specs -> save HDF5. |
| `Basic plottings .ipynb` | EDA visualizations: sensor distributions, RUL label histograms, vehicle lifecycle plots. |

### Baseline Models

| File | Description |
|---|---|
| `initial_models.ipynb` | First sequence-model exploration on SCANIA-X. Includes LSTM, GRU, VAE, and vanilla Transformer baselines. Saves `sequence_embeddings.npy` and `sequence_vehicle_ids.npy`. |

### TabTransformer Development

| File | Description |
|---|---|
| `Modelling_part1.ipynb` | First implementation of `TimeSeriesEmbedder` + categorical context vector fusion. Generates initial transformer embeddings. |
| `TabTransformer_dyn-hrd-path.ipynb` | TabTransformer with both dynamic (auto-detected) and hardcoded data paths. Contains model parameter plots and training graphs. |
| `TabTransformer+layervisualization.ipynb` | Layer-wise attention weight visualization and intermediate embedding analysis. |

### Differential Privacy (Early Experiments)

| File | Description |
|---|---|
| `differential_privacy_beta(wrong).ipynb` | First DP experiment combining TabTransformer + DP. Marked `(wrong)` -- used incorrect per-sample gradient accumulation. |
| `running_tf+dp+artifactgenerating(wrong).py` | Single-file script combining model + DP + artifact generation. Marked `(wrong)` -- had redundant train/val MSE recomputation after each epoch. |
| `running_tf+dp2.0+artifactgenerating(wrong).py` | Updated version with early stopping and StepLR, but still contains the redundant MSE computation bug. Superseded by `FInal_script/all_everything_v1.py`. |

### Inference

| File | Description |
|---|---|
| `Inference_on_saved_model.py` | Loads a saved artifact (`checkpoint.pth` + `metadata.json`) and runs inference on test data. Computes MSE and MAE. |
| `load_saved_model.ipynb` | Interactive notebook for loading checkpoints, exploring model structure, and visualizing predictions. |

---

## Artifacts

Training run outputs are saved under `artifacts/`. Each run folder contains:

```
artifacts/
`-- CombinedRULModel-{DP|NDP}-{YYYYMMDD_HHMMSS}/
    |-- checkpoint.pth       <- best model weights
    |-- metadata.json        <- training config
    `-- train_val_log.txt    <- per-epoch loss log
```

---

## Model Plots

<img width="484" height="338" alt="TabTransformer architecture" src="https://github.com/user-attachments/assets/0d11917c-2fb8-442d-b2a3-8dfb83dd5901" />
<img width="490" height="333" alt="Training convergence" src="https://github.com/user-attachments/assets/051579a1-fea0-401c-b606-176f1b6c2080" />
<img width="492" height="334" alt="Val loss curve" src="https://github.com/user-attachments/assets/813f1d8d-3661-4462-95c6-f3cf89cc6293" />
<img width="483" height="295" alt="Prediction vs actual" src="https://github.com/user-attachments/assets/c634d489-8617-433f-8514-72d2e3a8632f" />
<img width="755" height="578" alt="Layer visualization" src="https://github.com/user-attachments/assets/434a0e3e-fd45-4fa3-8508-82471c5d755a" />
<img width="792" height="719" alt="Attention weights" src="https://github.com/user-attachments/assets/17975aab-e377-4273-b73a-fb90a90fbf3c" />
<img width="779" height="316" alt="Training reconstruction" src="https://github.com/user-attachments/assets/a22de429-b902-4a2a-9a3e-530bdeabedd8" />
