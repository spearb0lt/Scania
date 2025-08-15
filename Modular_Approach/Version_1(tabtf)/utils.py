# utils.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

# --------------- 1. Sliding‐Window Function ---------------
def create_X_y(
    csv_path: str,
    sensor_features: list,
    context: int = 70,
    verbose: bool = True
):
    """
    Reads a CSV with columns ['vehicle_id', 'time_step', <sensor_features…>, 'RUL'],
    then for each vehicle_id, slides a window of length `context` over time steps.
    Returns:
      - X:   np.ndarray of shape (N, context, num_sensors)
      - y:   np.ndarray of shape (N,) RUL labels, where y[i] is RUL at the last time step of window i
      - vids: np.ndarray of shape (N,) vehicle_id corresponding to each window
    """
    df = pd.read_csv(csv_path)
    X_list, y_list, vid_list = [], [], []

    for vehicle_id, group in df.groupby("vehicle_id"):
        group = group.sort_values("time_step")
        data = group[sensor_features].values
        rul = group["RUL"].values

        if len(data) < context:
            if verbose:
                print(f"Skipping vehicle {vehicle_id}: sequence length {len(data)} < {context}")
            continue

        # Slide the window (length = context)
        for i in range(len(data) - context + 1):
            X_list.append(data[i : i + context])
            y_list.append(rul[i + context - 1])
            vid_list.append(vehicle_id)

    X = np.stack(X_list)     # shape: (N, context, num_sensors)
    y = np.array(y_list)     # shape: (N,)
    vids = np.array(vid_list)
    if verbose:
        print(f"Total windows: {len(X)}, each window shape: {X.shape[1:]}")
    return X, y, vids

# --------------- 2. Load & Encode Specifications ---------------
def load_and_encode_specs(
    spec_csv_path: str,
    window_vids: np.ndarray
):
    """
    1. Reads the CSV of vehicle specifications with columns:
       ['vehicle_id', 'Spec_0', ..., 'Spec_7'].
    2. Ordinal-encodes each Spec_i column.
    3. For each window (given by window_vids), merges and returns an (N,8) integer array.

    Returns:
      - specs_per_window: np.ndarray shape (N, 8) with ordinal‐encoded Spec_i
      - encoder: the fitted OrdinalEncoder (so you can inspect categories_ if needed)
    """
    spec_df = pd.read_csv(spec_csv_path)
    spec_columns = [f"Spec_{i}" for i in range(8)]

    encoder = OrdinalEncoder()
    # Fit & transform in‐place
    spec_df[spec_columns] = encoder.fit_transform(spec_df[spec_columns])

    # Merge to get one Spec‐row per window_id
    merged = (
        pd.DataFrame({"vehicle_id": window_vids})
          .merge(spec_df[["vehicle_id"] + spec_columns], on="vehicle_id", how="left")
    )

    specs_per_window = merged[spec_columns].values.astype(int)
    return specs_per_window, encoder

# --------------- 3. Custom Dataset ---------------
class RULCombinedDataset(Dataset):
    """
    PyTorch Dataset that returns (x_cat, x_ts, y):
      - x_cat: (8,)  LongTensor of ordinal‐encoded specs
      - x_ts:  (context, num_sensors) FloatTensor of raw sensor window
      - y:     (1,)  FloatTensor RUL label
    """
    def __init__(
        self,
        windows: np.ndarray,  # shape (N, context, num_sensors)
        specs: np.ndarray,    # shape (N, 8)
        labels: np.ndarray    # shape (N,)
    ):
        super().__init__()
        self.windows = windows
        self.specs = specs
        self.labels = labels.reshape(-1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x_ts = torch.from_numpy(self.windows[idx]).float()   # → (context, num_sensors)
        x_cat = torch.from_numpy(self.specs[idx]).long()     # → (8,)
        y = torch.from_numpy(self.labels[idx]).float()       # → (1,)
        return x_cat, x_ts, y

# --------------- 4. (Optional) Train/Val Split Helper ---------------
def train_val_split(
    X: np.ndarray,
    specs: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Performs a single train/validation split on windows, specs, and labels.
    Returns train+val arrays in the order:
      X_train, X_val, specs_train, specs_val, y_train, y_val
    """
    return train_test_split(
        X, specs, y,
        test_size=test_size,
        random_state=random_state
    )
