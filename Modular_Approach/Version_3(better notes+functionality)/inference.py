# # inference.py

# import os, json
# import torch
# import numpy as np
# import joblib
# from models import CombinedRULModel
# from utils import ENCODER_PATH, ARTIFACT_ROOT

# # 1) Discover latest artifact folder for CombinedRULModel
# folders = [d for d in os.listdir(ARTIFACT_ROOT) if d.startswith("CombinedRULModel-")]
# # pick the latest by lex order (timestamp in name)
# artifact = sorted(folders)[-1]
# artifact_path = os.path.join(ARTIFACT_ROOT, artifact)

# # 2) Load metadata.json
# with open(os.path.join(artifact_path, "metadata.json"), "r") as f:
#     md = json.load(f)


# # in inference.py, after reading metadata.json…
# pvt_flag = md["pvt"]
# print(f"Loaded model trained with DP: {pvt_flag}")


# # 3) Load encoder + model
# encoder = joblib.load(ENCODER_PATH)
# cat_sizes = tuple(len(c) for c in encoder.categories_)
# model = CombinedRULModel(
#     num_sensor_features=md["num_sensor_features"],
#     context_length=md["context_length"],
#     categories=cat_sizes,
#     continuous_dim=md["continuous_dim"]
# )
# ckpt = torch.load(os.path.join(artifact_path, "checkpoint.pth"), map_location="cpu")
# model.load_state_dict(ckpt)
# model.eval()

# # 4) Define inference function
# def predict(window: np.ndarray, spec_row: np.ndarray):
#     x_ts  = torch.from_numpy(window[None]).float()
#     x_cat = torch.from_numpy(encoder.transform(spec_row[None]).astype(int)).long()
#     with torch.no_grad():
#         return model(x_cat, x_ts).item()

# # 5) Example usage
# if __name__=="__main__":
#     # assume new_window.npy & new_spec.npy at CWD
#     w = np.load("new_window.npy")    # shape (70,105)
#     s = np.load("new_spec.npy")      # shape (8,)
#     print("Pred RUL:", predict(w, s))
# inference.py

import os
import json
import joblib
import h5py
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error

from models import CombinedRULModel
from utils import (
    SPEC_CSV,
    ENCODER_PATH,
    ARTIFACT_ROOT
)

# === CONFIGURATION ===
# Point to your artifact folder (with -DP or -NDP suffix)
# e.g. CombinedRULModel-20250606_032201-NDP
ARTIFACT_FOLDER = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\Code\artifacts\CombinedRULModel-20250606_032201-NDP"

# Paths for test data
TEST_CSV_PATH   = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\Code\validation_super_same_norm.csv"
TEST_H5_PATH    = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\Code\validation_data.h5"

use_h5_test = False  # Set True to load TEST_H5_PATH instead of CSV

# ---------------------------------------------------
# 1. Load model metadata
# ---------------------------------------------------
meta_path = os.path.join(ARTIFACT_FOLDER, "metadata.json")
with open(meta_path, "r") as f:
    metadata = json.load(f)

# ---------------------------------------------------
# 2. Load encoder
# ---------------------------------------------------
encoder = joblib.load(ENCODER_PATH)
spec_columns = [f"Spec_{i}" for i in range(8)]

# ---------------------------------------------------
# 3. Instantiate & load model
# ---------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CombinedRULModel(
    num_sensor_features=metadata["num_sensor_features"],
    context_length=metadata["context_length"],
    categories=tuple(metadata["categories"]),
    continuous_dim=metadata["continuous_dim"]
).to(device)

ckpt_path = os.path.join(ARTIFACT_FOLDER, "checkpoint.pth")
state_dict = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

print(f"Loaded model from {ckpt_path} with DP={metadata['pvt']}")

# ---------------------------------------------------
# 4. Prepare test data
# ---------------------------------------------------
if use_h5_test:
    print(f"Loading test data from HDF5 → {TEST_H5_PATH}")
    with h5py.File(TEST_H5_PATH, "r") as f:
        X_test     = f["X_windows"][:]
        y_test     = f["y_labels"][:]
        specs_test = f["specs_per_window"][:]
else:
    print(f"Generating test windows from CSV → {TEST_CSV_PATH}")
    # 4a. Slide windows over sensor CSV
    df = pd.read_csv(TEST_CSV_PATH)
    windows, labels, vids = [], [], []
    for vid, grp in df.groupby("vehicle_id"):
        data = grp[[c for c in df.columns if c not in ("vehicle_id","RUL")]].values
        rul  = grp["RUL"].values
        # assume same context as train
        context = metadata["context_length"]
        if len(data) < context:
            continue
        for i in range(len(data) - context + 1):
            windows.append(data[i : i + context])
            labels.append(rul[i + context - 1])
            vids.append(vid)
    X_test = np.stack(windows)
    y_test = np.array(labels)

    # 4b. Build specs_per_window and ordinal-encode with saved encoder
    spec_df = pd.read_csv(SPEC_CSV)
    merged = (
        pd.DataFrame({"vehicle_id": vids})
          .merge(spec_df[["vehicle_id"] + spec_columns],
                 on="vehicle_id", how="left")
    )
    specs_raw = merged[spec_columns].values
    specs_test = encoder.transform(specs_raw).astype(int)

print(f"Test set: {X_test.shape[0]} windows, each {X_test.shape[1:]}")

# ---------------------------------------------------
# 5. Run inference in batches
# ---------------------------------------------------
batch_size = 256
preds = []
with torch.no_grad():
    for start in range(0, len(X_test), batch_size):
        end = start + batch_size
        x_ts = torch.from_numpy(X_test[start:end]).float().to(device)
        x_cat= torch.from_numpy(specs_test[start:end]).long().to(device)
        out = model(x_cat, x_ts).squeeze().cpu().numpy()
        preds.append(out)
preds = np.concatenate(preds)

# ---------------------------------------------------
# 6. Compute & print metrics
# ---------------------------------------------------
mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)
print(f"Test MSE = {mse:.4f}")
print(f"Test MAE = {mae:.4f}")
