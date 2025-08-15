# utils.py

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import joblib
import h5py
from datetime import datetime

# === Hard‐coded absolute paths ===
WINDOW_CSV     = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\Code\super_same_norm.csv"
SPEC_CSV       = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\data\train_specifications.csv"
ENCODER_PATH   = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\Code2\ModularApproach2\spec_encoder.joblib"
H5_PATH        = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\Code2\ModularApproach2\data_windows.h5"
ARTIFACT_ROOT  = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\Code2\ModularApproach2\artifacts"


def create_X_y(csv_path=WINDOW_CSV, sensor_features=None, context=70, verbose=True):
    """Read CSV→ sliding windows + ordinal‐encode specs + dump encoder."""
    df = pd.read_csv(csv_path)
    X, y, vids = [], [], []
    for vid, grp in df.groupby("vehicle_id"):
        data = grp[sensor_features].values
        rul  = grp["RUL"].values
        if len(data) < context:
            if verbose: print(f"Skipping {vid}, len<{context}")
            continue
        for i in range(len(data) - context + 1):
            X.append(data[i : i + context])
            y.append(rul[i + context - 1])
            vids.append(vid)
    X = np.stack(X); y = np.array(y); vids = np.array(vids)
    print(f"Total windows={len(X)}, window shape={X.shape[1:]}")

    # Ordinal‐encode specs
    spec_df = pd.read_csv(SPEC_CSV)
    spec_cols = [f"Spec_{i}" for i in range(8)]
    enc = OrdinalEncoder()
    spec_df[spec_cols] = enc.fit_transform(spec_df[spec_cols])
    specs = (
        pd.DataFrame({"vehicle_id": vids})
          .merge(spec_df[["vehicle_id"] + spec_cols], on="vehicle_id", how="left")
    )[spec_cols].values.astype(int)

    # Dump encoder
    joblib.dump(enc, ENCODER_PATH)
    print(f"Saved encoder → {ENCODER_PATH}")

    return X, y, vids, specs


def save_to_h5(X, y, vids, specs, h5_path=H5_PATH):
    """Save arrays into an HDF5 container."""
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("X_windows",      data=X,     compression="gzip")
        f.create_dataset("y_labels",       data=y,     compression="gzip")
        f.create_dataset("window_vids",    data=vids,  compression="gzip")
        f.create_dataset("specs_per_window", data=specs, compression="gzip")
    print(f"Saved datasets → {h5_path}")


def load_from_h5(h5_path=H5_PATH):
    """Load arrays from the HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        X     = f["X_windows"][:]
        y     = f["y_labels"][:]
        vids  = f["window_vids"][:]
        specs = f["specs_per_window"][:]
    return X, y, vids, specs


class RULCombinedDataset(Dataset):
    """PyTorch Dataset wrapping (specs, windows, labels)."""
    def __init__(self, windows, specs, labels):
        self.windows = windows
        self.specs   = specs
        self.labels  = labels.reshape(-1,1)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return (
            torch.from_numpy(self.specs[i]).long(),
            torch.from_numpy(self.windows[i]).float(),
            torch.from_numpy(self.labels[i]).float()
        )


def train_val_split(*arrays, test_size=0.2, random_state=42):
    """Wrapper around sklearn.model_selection.train_test_split."""
    return train_test_split(*arrays, test_size=test_size, random_state=random_state)


def make_artifact_folder(model_name: str, pvt: bool) -> str:
    """Create artifacts/<model_name>-<timestamp>-DP/NDP and return its path."""
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "DP" if pvt else "NDP"
    folder = f"{model_name}-{suffix}-{ts}"
    path   = os.path.join(ARTIFACT_ROOT, folder)
    os.makedirs(path, exist_ok=True)
    return path
