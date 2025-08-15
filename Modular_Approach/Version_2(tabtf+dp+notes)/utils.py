# utils.py

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from models import CombinedRULModel  # for load_model()
from datetime import datetime

# ---------------------------------------------------
# Hard-coded absolute paths
# ---------------------------------------------------
WINDOW_CSV_PATH   = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\Code\super_same_norm.csv"
SPEC_CSV_PATH     = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\data\train_specifications.csv"
ARTIFACT_ROOT_DIR = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\Code2\ModularApproach\artifacts"

# ---------------------------------------------------
# 1. Sliding-Window Function
# ---------------------------------------------------
def create_X_y(
    csv_path: str = WINDOW_CSV_PATH,
    sensor_features: list = None,
    context: int = 70,
    verbose: bool = True
):
    """
    Reads a CSV with columns ['vehicle_id', 'time_step', <sensor_features…>, 'RUL'].
    For each vehicle_id, sorts by time_step and slides a window of length `context`.
    Returns:
      - X:   np.ndarray (N, context, num_sensors)
      - y:   np.ndarray (N,)           RUL at end of each window
      - vids: np.ndarray (N,)          vehicle_id for each window
    """
    if sensor_features is None:
        raise ValueError("sensor_features must be provided.")

    df = pd.read_csv(csv_path)
    X_list, y_list, vids_list = [], [], []

    for vehicle_id, group in df.groupby("vehicle_id"):
        group = group.sort_values("time_step")
        data = group[sensor_features].values
        rul = group["RUL"].values

        if len(data) < context:
            if verbose:
                print(f"Skipping vehicle {vehicle_id}: length {len(data)} < {context}")
            continue

        for i in range(len(data) - context + 1):
            X_list.append(data[i : i + context])
            y_list.append(rul[i + context - 1])
            vids_list.append(vehicle_id)

    X = np.stack(X_list)   # shape: (N, context, num_sensors)
    y = np.array(y_list)   # shape: (N,)
    vids = np.array(vids_list)
    if verbose:
        print(f"Total windows: {len(X)}, each window shape: {X.shape[1:]}")
    return X, y, vids

# ---------------------------------------------------
# 2. Load & Encode Specifications
# ---------------------------------------------------
def load_and_encode_specs(
    spec_csv_path: str = SPEC_CSV_PATH,
    window_vids: np.ndarray = None
):
    """
    1) Reads vehicle-spec CSV with columns ['vehicle_id','Spec_0',…,'Spec_7'].
    2) Ordinal-encodes each Spec_i column.
    3) For each window (given by window_vids), merges and returns an (N, 8) int array.
    Returns:
      - specs_per_window: np.ndarray (N, 8)
      - encoder: fitted OrdinalEncoder
    """
    if window_vids is None:
        raise ValueError("window_vids must be provided.")

    spec_df = pd.read_csv(spec_csv_path)
    spec_columns = [f"Spec_{i}" for i in range(8)]

    encoder = OrdinalEncoder()
    spec_df[spec_columns] = encoder.fit_transform(spec_df[spec_columns])

    merged = (
        pd.DataFrame({"vehicle_id": window_vids})
          .merge(spec_df[["vehicle_id"] + spec_columns], on="vehicle_id", how="left")
    )
    specs_per_window = merged[spec_columns].values.astype(int)
    return specs_per_window, encoder

# ---------------------------------------------------
# 3. Custom Dataset
# ---------------------------------------------------
class RULCombinedDataset(Dataset):
    """
    PyTorch Dataset returning (x_cat, x_ts, y):
      - x_cat: torch.LongTensor (8,) of ordinal-encoded specs
      - x_ts:  torch.FloatTensor (context, num_sensors)
      - y:     torch.FloatTensor (1,) of RUL
    """
    def __init__(self, windows: np.ndarray, specs: np.ndarray, labels: np.ndarray):
        super().__init__()
        self.windows = windows     # shape (N, context, num_sensors)
        self.specs = specs         # shape (N, 8)
        self.labels = labels.reshape(-1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x_ts = torch.from_numpy(self.windows[idx]).float()   # → (context, num_sensors)
        x_cat = torch.from_numpy(self.specs[idx]).long()     # → (8,)
        y    = torch.from_numpy(self.labels[idx]).float()    # → (1,)
        return x_cat, x_ts, y

# ---------------------------------------------------
# 4. Train/Validation Split Helper
# ---------------------------------------------------
def train_val_split(
    X: np.ndarray,
    specs: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Splits windows, specs, and labels into train/validation subsets.
    Returns: X_train, X_val, specs_train, specs_val, y_train, y_val
    """
    return train_test_split(X, specs, y, test_size=test_size, random_state=random_state)

# ---------------------------------------------------
# 5. Model-Loading Helper (for inference)
# ---------------------------------------------------
def load_model(
    checkpoint_path: str,
    num_sensor_features: int,
    continuous_dim: int = 128
):
    """
    1) Recreates a CombinedRULModel with exactly the same dims & categories.
    2) Loads state_dict from the given hard-coded checkpoint file path.
    Returns:
      - model (in eval mode)
      - the fitted OrdinalEncoder (so you can encode new spec rows consistently).
    """
    # (a) We must re-fit the OrdinalEncoder on the same spec CSV used at training:
    spec_df = pd.read_csv(SPEC_CSV_PATH)
    spec_columns = [f"Spec_{i}" for i in range(8)]
    from sklearn.preprocessing import OrdinalEncoder
    oe = OrdinalEncoder()
    spec_df[spec_columns] = oe.fit_transform(spec_df[spec_columns])
    category_sizes = tuple(len(oe.categories_[i]) for i in range(8))

    # (b) Instantiate CombinedRULModel
    model = CombinedRULModel(
        num_sensor_features=num_sensor_features,
        context_length=70,
        categories=category_sizes,
        continuous_dim=continuous_dim,
        cont_mean_std=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # (c) Load weights
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model, oe

# ---------------------------------------------------
# 6. Artifact-Folder Helper
# ---------------------------------------------------
def make_artifact_folder(model_name: str) -> str:
    """
    Creates a new folder under the hard-coded ARTIFACT_ROOT_DIR named "<model_name>-<TIMESTAMP>".
    Returns the absolute path to that new artifact folder.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{model_name}-{timestamp}"
    full_path = os.path.join(ARTIFACT_ROOT_DIR, folder_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path





# # utils.py
# import os
# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.model_selection import train_test_split
# from models import CombinedRULModel  # for load_model()
# from datetime import datetime

# # ---------------------------------------------------
# # 0. Dynamic “parent” directory and path helper
# # ---------------------------------------------------
# parent = os.path.abspath("")  # the project’s root folder

# def make_path(relative_path: str) -> str:
#     """
#     Helper that returns os.path.join(parent, relative_path).
#     Example: make_path("data/train_specifications.csv") will yield
#              "<project_root>/data/train_specifications.csv".
#     """
#     return os.path.join(parent, relative_path)

# # ---------------------------------------------------
# # 1. Sliding-Window Function
# # ---------------------------------------------------
# def create_X_y(
#     csv_path: str,
#     sensor_features: list,
#     context: int = 70,
#     verbose: bool = True
# ):
#     """
#     Reads a CSV with columns ['vehicle_id', 'time_step', <sensor_features…>, 'RUL'].
#     For each vehicle_id, sorts by time_step and slides a window of length `context`.
#     Returns:
#       - X:   np.ndarray (N, context, num_sensors)
#       - y:   np.ndarray (N,)           RUL at end of each window
#       - vids: np.ndarray (N,)           vehicle_id for each window
#     """
#     df = pd.read_csv(csv_path)
#     X_list, y_list, vids_list = [], [], []

#     for vehicle_id, group in df.groupby("vehicle_id"):
#         group = group.sort_values("time_step")
#         data = group[sensor_features].values
#         rul = group["RUL"].values

#         if len(data) < context:
#             if verbose:
#                 print(f"Skipping vehicle {vehicle_id}: length {len(data)} < {context}")
#             continue

#         for i in range(len(data) - context + 1):
#             X_list.append(data[i : i + context])
#             y_list.append(rul[i + context - 1])
#             vids_list.append(vehicle_id)

#     X = np.stack(X_list)   # shape: (N, context, num_sensors)
#     y = np.array(y_list)   # shape: (N,)
#     vids = np.array(vids_list)
#     if verbose:
#         print(f"Total windows: {len(X)}, each window shape: {X.shape[1:]}")
#     return X, y, vids

# # ---------------------------------------------------
# # 2. Load & Encode Specifications
# # ---------------------------------------------------
# def load_and_encode_specs(
#     spec_csv_path: str,
#     window_vids: np.ndarray
# ):
#     """
#     1) Reads vehicle-spec CSV with columns ['vehicle_id','Spec_0',…,'Spec_7'].
#     2) Ordinal-encodes each Spec_i column.
#     3) For each window (given by window_vids), merges and returns an (N, 8) int array.
#     Returns:
#       - specs_per_window: np.ndarray (N, 8)
#       - encoder: fitted OrdinalEncoder
#     """
#     spec_df = pd.read_csv(spec_csv_path)
#     spec_columns = [f"Spec_{i}" for i in range(8)]

#     encoder = OrdinalEncoder()
#     spec_df[spec_columns] = encoder.fit_transform(spec_df[spec_columns])

#     merged = (
#         pd.DataFrame({"vehicle_id": window_vids})
#           .merge(spec_df[["vehicle_id"] + spec_columns], on="vehicle_id", how="left")
#     )
#     specs_per_window = merged[spec_columns].values.astype(int)
#     return specs_per_window, encoder

# # ---------------------------------------------------
# # 3. Custom Dataset
# # ---------------------------------------------------
# class RULCombinedDataset(Dataset):
#     """
#     PyTorch Dataset returning (x_cat, x_ts, y):
#       - x_cat: torch.LongTensor (8,) of ordinal-encoded specs
#       - x_ts:  torch.FloatTensor (context, num_sensors)
#       - y:     torch.FloatTensor (1,) of RUL
#     """
#     def __init__(self, windows: np.ndarray, specs: np.ndarray, labels: np.ndarray):
#         super().__init__()
#         self.windows = windows     # shape (N, context, num_sensors)
#         self.specs = specs         # shape (N, 8)
#         self.labels = labels.reshape(-1, 1)

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         x_ts = torch.from_numpy(self.windows[idx]).float()   # → (context, num_sensors)
#         x_cat = torch.from_numpy(self.specs[idx]).long()     # → (8,)
#         y    = torch.from_numpy(self.labels[idx]).float()    # → (1,)
#         return x_cat, x_ts, y

# # ---------------------------------------------------
# # 4. Train/Validation Split Helper
# # ---------------------------------------------------
# def train_val_split(
#     X: np.ndarray,
#     specs: np.ndarray,
#     y: np.ndarray,
#     test_size: float = 0.2,
#     random_state: int = 42
# ):
#     """
#     Splits windows, specs, and labels into train/validation subsets.
#     Returns: X_train, X_val, specs_train, specs_val, y_train, y_val
#     """
#     return train_test_split(X, specs, y, test_size=test_size, random_state=random_state)

# # ---------------------------------------------------
# # 5. Model-Loading Helper (for inference)
# # ---------------------------------------------------
# def load_model(
#     model_filename: str,
#     num_sensor_features: int,
#     categories: tuple,
#     continuous_dim: int = 128
# ):
#     """
#     1) Recreates a CombinedRULModel with exactly the same dims & categories.
#     2) Loads state_dict from the given .pth (checkpoint) file.
#     Returns:
#       - model (in eval mode), 
#       - the fitted OrdinalEncoder (so you can encode new spec rows consistently).
#     """
#     # (a) We must re-fit the OrdinalEncoder on the same spec CSV used at training:
#     spec_csv_rel = os.path.join("data", "train_specifications.csv")
#     spec_df = pd.read_csv(make_path(spec_csv_rel))
#     spec_columns = [f"Spec_{i}" for i in range(8)]
#     from sklearn.preprocessing import OrdinalEncoder
#     oe = OrdinalEncoder()
#     spec_df[spec_columns] = oe.fit_transform(spec_df[spec_columns])
#     category_sizes = tuple(len(oe.categories_[i]) for i in range(8))

#     # (b) Instantiate CombinedRULModel
#     model = CombinedRULModel(
#         num_sensor_features=num_sensor_features,
#         context_length=70,
#         categories=category_sizes,
#         continuous_dim=continuous_dim,
#         cont_mean_std=None
#     ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

#     # (c) Load weights
#     state_path = make_path(os.path.join("artifacts", model_filename))
#     state_dict = torch.load(state_path, map_location=torch.device("cpu"))
#     model.load_state_dict(state_dict)
#     model.eval()
#     return model, oe

# # ---------------------------------------------------
# # 6. Artifact-Folder Helper
# # ---------------------------------------------------
# def make_artifact_folder(model_name: str) -> str:
#     """
#     Creates a new folder under <project_root>/artifacts named "<model_name>-<TIMESTAMP>".
#     Returns the absolute path to that new artifact folder.
#     """
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     folder_name = f"{model_name}-{timestamp}"
#     artifact_root = make_path("artifacts")
#     os.makedirs(artifact_root, exist_ok=True)
#     full_path = os.path.join(artifact_root, folder_name)
#     os.makedirs(full_path, exist_ok=True)
#     return full_path
