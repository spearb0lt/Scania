# Cross-Industry Privacy-Preserving Framework for PdM based on IDA 24 Industrial Challenge


### MIA
<img width="362" height="58" alt="image" src="https://github.com/user-attachments/assets/4a17b3a1-bbf3-4bdb-9ea7-ee3ab89f2fed" />
<img width="220" height="166" alt="image" src="https://github.com/user-attachments/assets/fc6856da-546c-45cf-852a-b750040890d9" />
<img width="334" height="94" alt="image" src="https://github.com/user-attachments/assets/db630c4e-80f6-4538-b424-7cdea0f7ab58" />
<img width="204" height="83" alt="image" src="https://github.com/user-attachments/assets/06b99139-75a1-4b4a-9557-d5bad0021ec7" />
<img width="203" height="78" alt="image" src="https://github.com/user-attachments/assets/d3f20392-4fb1-4dac-b732-4db6fecc92f5" />
<img width="312" height="37" alt="image" src="https://github.com/user-attachments/assets/f7b1e683-8bfc-4375-bd2b-c506decf6c1b" />
<img width="282" height="159" alt="image" src="https://github.com/user-attachments/assets/8306efdd-7e06-4bef-9bc5-dfaa635717b3" />

### SVD in Spectral-DP

<img width="261" height="205" alt="image" src="https://github.com/user-attachments/assets/15419292-a910-4f3b-afe3-fa95243ef0b8" />
<img width="485" height="50" alt="image" src="https://github.com/user-attachments/assets/24d514b2-eb80-4f67-9830-ecd7a194064a" />
<img width="471" height="231" alt="image" src="https://github.com/user-attachments/assets/1846e37c-6261-4248-b3d4-3e72eb8db061" />
<img width="411" height="50" alt="image" src="https://github.com/user-attachments/assets/6d3b23a6-50f1-4476-ae0f-13cdde91edb3" />
<img width="549" height="116" alt="image" src="https://github.com/user-attachments/assets/bb2ba1a8-ac2f-4db4-baff-249c13b5993c" />

<img width="505" height="83" alt="image" src="https://github.com/user-attachments/assets/5368293d-37fc-469c-b55c-903593412272" />
<img width="422" height="404" alt="image" src="https://github.com/user-attachments/assets/e3e988be-4eb3-4d18-8e9e-e6557f3aa31b" />


### Artifact Generation
<img width="396" height="215" alt="image" src="https://github.com/user-attachments/assets/5a8a5fba-9014-4af0-bc02-380be54b5ae2" />
<img width="348" height="218" alt="image" src="https://github.com/user-attachments/assets/b4f6f16a-6438-4672-8d1f-7a4402fe71b0" />
<img width="452" height="509" alt="Screenshot 2025-08-16 025051" src="https://github.com/user-attachments/assets/32e2e05c-bce7-4027-924b-73b8100c9ceb" />

<img width="589" height="528" alt="Screenshot 2025-08-16 025113" src="https://github.com/user-attachments/assets/b255ceeb-7ece-4c5d-945c-bb47d9c23d29" />



<img width="797" height="341" alt="image" src="https://github.com/user-attachments/assets/16d3bc7f-19e0-413e-823e-6c6814daa901" />

<img width="779" height="316" alt="image" src="https://github.com/user-attachments/assets/a22de429-b902-4a2a-9a3e-530bdeabedd8" />
<img width="247" height="214" alt="Screenshot 2025-08-16 025301" src="https://github.com/user-attachments/assets/48fe0b29-cc43-4ef7-8426-6cf6a7b8c1e3" />
<img width="264" height="290" alt="image" src="https://github.com/user-attachments/assets/3e41ce35-b9cc-4073-800d-e43be2a9305a" />
<img width="470" height="366" alt="image" src="https://github.com/user-attachments/assets/237aed64-55e2-40f6-ad18-68885e13fcda" />
<img width="392" height="465" alt="image" src="https://github.com/user-attachments/assets/d083a50e-ce1d-43f6-b595-03becf7b7d5c" />
<img width="660" height="503" alt="image" src="https://github.com/user-attachments/assets/908368ba-88de-4cdb-ba8a-2e2ca1e2ccbe" />

**Short description**
This repository implements a TabTransformer-based model for Remaining Useful Life (RUL) prediction on the Scania dataset, with an end-to-end preprocessing pipeline and differential-privacy-enabled training. It also contains an evaluation and Membership Inference Attack (MIA) framework so you can measure privacy/utility trade-offs.

---

## Table of contents

* [Motivation](#motivation)
* [What’s included](#whats-included)
* [Repository structure](#repository-structure)

* [Data & preprocessing](#data--preprocessing)
* [Model & architecture](#model--architecture)
* [Training](#training)
* [Differential privacy (DP) details](#differential-privacy-dp-details)
* [Evaluation & Membership Inference Attack (MIA)](#evaluation--membership-inference-attack-mia)
* [Results & tips](#results--tips)

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

## Repository structure
```
.
├── README.md
├── FederatedApproach
│   ├── feddiff_client+spectraldp.py
│   ├── feddiff_client.py
│   ├── feddiff_server.py
│   ├── fedsame_client.py
│   ├── fedsame_server.py
│   └── note.md
├── FInal_script
│   ├── all_everything2(wrong).py
│   ├── all_everything_v0(wrong).py
│   ├── all_everything_v1.py
│   ├── custom_dp (1).py
│   ├── custom_dp.py
│   ├── future_log_generator.py
│   ├── note.md
│   ├── spectral_dp+tabtf_v1.py
│   ├── tabtf+dpsgd(my).py
│   └── tabtf+dpsgd(sirs).py
├── Important_script_part1
│   ├── Basic plottings .ipynb
│   ├── Data-Preprocessing-Automated.ipynb
│   ├── Data-Processing-Detailed.ipynb
│   ├── data_windows.h5
│   ├── differential_privacy_beta(wrong).ipynb
│   ├── Inference_on_saved_model.py
│   ├── initial_models.ipynb
│   ├── load_saved_model.ipynb
│   ├── Modelling_part1.ipynb
│   ├── note.md
│   ├── running_tf+dp+artifactgenerating(wrong).py
│   ├── running_tf+dp2.0+artifactgenerating(wrong).py
│   ├── spec_encoder.joblib
│   ├── TabTransformer+layervisualization.ipynb
│   ├── TabTransformer_dyn-hrd-path.ipynb
│   └── artifacts
│       ├── CombinedRULModel-DP-20250616_023852
│       │   ├── checkpoint.pth
│       │   ├── metadata.json
│       │   └── train_val_log.txt
│       └── CombinedRULModel-NDP-20250613_030711
│           ├── checkpoint.pth
│           ├── metadata.json
│           └── train_val_log.txt
├── Important_script_part2
│   ├── data_windows.h5
│   ├── mathematical_logic_for_spectralDP.ipynb
│   ├── Membership-Inference-Attack(MIA).ipynb
│   ├── spectral_dp+tabtf_v0.py
│   ├── spec_encoder.joblib
│   ├── artifacts
│   │   .
│   │   .
│   │   .
│   │   ├── CombinedRULModel-DP-20250618_160955
│   │   │   ├── checkpoint.pth
│   │   │   ├── metadata.json
│   │   │   └── train_val_log.txt
│   │   └── CombinedRULModel-NDP-20250616_163702
│   │       ├── checkpoint.pth
│   │       ├── metadata.json
│   │       └── train_val_log.txt
│   └── artifacts2
│       ├── CombinedRULModel-DP-20250704_052418
│       │   ├── checkpoint.pth
│       │   ├── metadata.json
│       │   └── train_val_log.txt
│       └── CombinedRULModel-NDP-20250707_155658
│           ├── checkpoint.pth
│           ├── metadata.json
│           └── train_val_log.txt
└── Modular_Approach
    ├── Version_1(tabtf)
    │   ├── inference.py
    │   ├── models.py
    │   ├── services.py
    │   ├── trainer.py
    │   └── utils.py
    ├── Version_2(tabtf+dp+notes)
    │   ├── inference.py
    │   ├── models.py
    │   ├── services.py
    │   ├── trainer.py
    │   ├── utils.py
    │   └── artifacts
    │       ├── CombinedRULModel-20250606_032201
    │       │   ├── checkpoint.pth
    │       │   ├── metadata.json
    │       │   └── train_val_log.txt
    │       .
    │       .
    │       .
    └── Version_3(better notes+functionality)
        ├── data_windows.h5
        ├── inference.py
        ├── loss_plotter.ipynb
        ├── models.py
        ├── services.py
        ├── spec_encoder.joblib
        ├── trainer.py
        ├── utils.py
        └── artifacts
            ├── CombinedRULModel-20250612_014443-NDP
            │   ├── checkpoint.pth
            │   ├── metadata.json
            │   └── train_val_log.txt
            ├── CombinedRULModel-20250612_020629-DP
            │   ├── checkpoint.pth
            │   ├── metadata.json
            │   └── train_val_log.txt
            .
            .
            .
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

