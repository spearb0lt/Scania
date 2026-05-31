# Important_script_part2 -- Privacy Evaluation Phase

This folder contains the advanced privacy training scripts and the complete MIA
(Membership Inference Attack) evaluation framework. It builds directly on the model
architecture from `Important_script_part1` and implements the final Spectral-DP
algorithm alongside the full mathematical derivation and empirical privacy validation.

---

## Data Files

| File | Description |
|---|---|
| `data_windows.h5` | Same preprocessed HDF5 dataset as Part 1 |
| `spec_encoder.joblib` | Same OrdinalEncoder for 8 categorical spec features |

---

## Scripts and Notebooks

### Training

| File | Description |
|---|---|
| `spectral_dp+tabtf_v0.py` | Initial Spectral-DP + TabTransformer script. Predecessor to `FInal_script/spectral_dp+tabtf_v1.py`. Uses AdamW + CosineAnnealingLR, batch=1024 for server runs. |

### Mathematical Derivation

| File | Description |
|---|---|
| `mathematical_logic_for_spectralDP.ipynb` | Complete mathematical proof for why SVD-domain gradient perturbation satisfies differential privacy. Covers: SVD decomposition, spectral clipping as a sensitivity bound, Gaussian mechanism in the singular value domain, and RDP conversion. |

### Privacy Evaluation

| File | Description |
|---|---|
| `Membership-Inference-Attack(MIA).ipynb` | Full MIA framework combining white-box (activations), gray-box (gradient norms), black-box (loss/confidence), and time-series specific (seasonality, trend) features. Achieves **AUC 49.12%, Accuracy 49.59%** against the DP model -- empirically confirming zero membership leakage. |

---

## Artifacts

### `artifacts/` -- Spectral-DP training runs (server, batch=1024)

| Run | Privacy | Notes |
|---|---|---|
| `CombinedRULModel-NDP-20250616_163702` | None | NDP baseline |
| `CombinedRULModel-DP-20250616_170035` | Spectral-DP | First server run |
| `CombinedRULModel-DP-20250617_163951` | Spectral-DP | -- |
| `CombinedRULModel-DP-20250617_164137` | Spectral-DP | -- |
| `CombinedRULModel-DP-20250617_165156` | Spectral-DP | -- |
| `CombinedRULModel-DP-20250618_154623` | Spectral-DP | sigma=0.1, clip=1.0 |
| `CombinedRULModel-DP-20250618_160955` | Spectral-DP | -- |

### `artifacts2/` -- Later runs (modular trainer)

| Run | Privacy | Notes |
|---|---|---|
| `CombinedRULModel-NDP-20250707_155658` | None | Latest NDP baseline |
| `CombinedRULModel-DP-20250704_051148` | DP | -- |
| `CombinedRULModel-DP-20250704_051501` | DP | -- |
| `CombinedRULModel-DP-20250704_051637` | DP | -- |
| `CombinedRULModel-DP-20250704_052418` | DP | -- |

Each artifact contains `checkpoint.pth`, `metadata.json`, `train_val_log.txt`.

---

## MIA Results

<img width="362" height="58" alt="MIA overall results" src="https://github.com/user-attachments/assets/4a17b3a1-bbf3-4bdb-9ea7-ee3ab89f2fed" />
<img width="220" height="166" alt="MIA ROC curve" src="https://github.com/user-attachments/assets/fc6856da-546c-45cf-852a-b750040890d9" />
<img width="334" height="94" alt="MIA feature importance" src="https://github.com/user-attachments/assets/db630c4e-80f6-4538-b424-7cdea0f7ab58" />
<img width="204" height="83" alt="MIA confusion matrix" src="https://github.com/user-attachments/assets/06b99139-75a1-4b4a-9557-d5bad0021ec7" />
<img width="203" height="78" alt="MIA precision-recall" src="https://github.com/user-attachments/assets/d3f20392-4fb1-4dac-b732-4db6fecc92f5" />
<img width="312" height="37" alt="MIA summary" src="https://github.com/user-attachments/assets/f7b1e683-8bfc-4375-bd2b-c506decf6c1b" />
<img width="282" height="159" alt="MIA distribution" src="https://github.com/user-attachments/assets/8306efdd-7e06-4bef-9bc5-dfaa635717b3" />

---

## SVD in Spectral-DP -- Mathematical Derivation

<img width="261" height="205" alt="SVD decomposition" src="https://github.com/user-attachments/assets/15419292-a910-4f3b-afe3-fa95243ef0b8" />
<img width="485" height="50" alt="SVD formula" src="https://github.com/user-attachments/assets/24d514b2-eb80-4f67-9830-ecd7a194064a" />
<img width="471" height="231" alt="Spectral noise injection" src="https://github.com/user-attachments/assets/1846e37c-6261-4248-b3d4-3e72eb8db061" />
<img width="411" height="50" alt="Clipping bound" src="https://github.com/user-attachments/assets/6d3b23a6-50f1-4476-ae0f-13cdde91edb3" />
<img width="549" height="116" alt="Privacy guarantee" src="https://github.com/user-attachments/assets/bb2ba1a8-ac2f-4db4-baff-249c13b5993c" />
<img width="505" height="83" alt="RDP conversion" src="https://github.com/user-attachments/assets/5368293d-37fc-469c-b55c-903593412272" />
<img width="422" height="404" alt="SVD geometric interpretation" src="https://github.com/user-attachments/assets/e3e988be-4eb3-4d18-8e9e-e6557f3aa31b" />

---

## Artifact Generation

<img width="396" height="215" alt="Artifact folder structure" src="https://github.com/user-attachments/assets/5a8a5fba-9014-4af0-bc02-380be54b5ae2" />
<img width="348" height="218" alt="Training log CSV" src="https://github.com/user-attachments/assets/b4f6f16a-6438-4672-8d1f-7a4402fe71b0" />
<img width="452" height="509" alt="Artifact folder tree" src="https://github.com/user-attachments/assets/32e2e05c-bce7-4027-924b-73b8100c9ceb" />
<img width="589" height="528" alt="Metadata JSON view" src="https://github.com/user-attachments/assets/b255ceeb-7ece-4c5d-945c-bb47d9c23d29" />
