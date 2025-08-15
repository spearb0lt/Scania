# inference.py
import os
import torch
import numpy as np
import pandas as pd

from models import CombinedRULModel
from utils import load_and_encode_specs, RULCombinedDataset

# ----------------------------
# 1. Config & Paths
# ----------------------------
SENSOR_FEATURES = [
    '171_0', '666_0', '427_0', '837_0',
    '167_0', '167_1', '167_2', '167_3',
    '167_4', '167_5', '167_6', '167_7',
    '167_8', '167_9', '309_0', '272_0',
    '272_1', '272_2', '272_3', '272_4',
    '272_5', '272_6', '272_7', '272_8',
    '272_9', '835_0', '370_0', '291_0',
    '291_1', '291_2', '291_3', '291_4',
    '291_5', '291_6', '291_7', '291_8',
    '291_9', '291_10', '158_0', '158_1',
    '158_2', '158_3', '158_4', '158_5',
    '158_6', '158_7', '158_8', '158_9',
    '100_0', '459_0', '459_1', '459_2',
    '459_3', '459_4', '459_5', '459_6',
    '459_7', '459_8', '459_9', '459_10',
    '459_11', '459_12', '459_13', '459_14',
    '459_15', '459_16', '459_17', '459_18',
    '459_19', '397_0', '397_1', '397_2',
    '397_3', '397_4', '397_5', '397_6',
    '397_7', '397_8', '397_9', '397_10',
    '397_11', '397_12', '397_13', '397_14',
    '397_15', '397_16', '397_17', '397_18',
    '397_19', '397_20', '397_21', '397_22',
    '397_23', '397_24', '397_25', '397_26',
    '397_27', '397_28', '397_29', '397_30',
    '397_31', '397_32', '397_33', '397_34',
    '397_35'
]
CONTEXT_LENGTH = 70
SPEC_CSV_PATH    = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\data\train_specifications.csv"
MODEL_PATH       = "best_combined_model.pt"
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """
    1) Recreate the same CombinedRULModel instance (with identical categories & dims).
    2) Load state_dict from disk.
    """
    # 1a. We need to know how many categories each Spec_i had. 
    #     We can rebuild the encoder by reading SPEC_CSV_PATH and fitting OrdinalEncoder again.
    #     That way encoder.categories_ is the same order as at train time.
    spec_df = pd.read_csv(SPEC_CSV_PATH)
    spec_columns = [f"Spec_{i}" for i in range(8)]
    from sklearn.preprocessing import OrdinalEncoder
    oe = OrdinalEncoder()
    spec_df[spec_columns] = oe.fit_transform(spec_df[spec_columns])
    category_sizes = tuple(len(oe.categories_[i]) for i in range(8))

    # 1b. Instantiate a fresh CombinedRULModel, matching exactly the dimensions used at training.
    num_sensor_features = len(SENSOR_FEATURES)  # 105
    continuous_dim = 128

    model = CombinedRULModel(
        num_sensor_features=num_sensor_features,
        categories=category_sizes,
        continuous_dim=continuous_dim,
        cont_mean_std=None  # At inference, TabTransformer will still assume (0,1) if not provided
    ).to(DEVICE)

    # 1c. Load weights
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model, oe  # return both model and the encoder (so we can ordinal-encode new spec rows)

def infer_single_window(
    model: torch.nn.Module,
    encoder: OrdinalEncoder,
    raw_window: np.ndarray,
    raw_spec_row: np.ndarray
):
    """
    raw_window: numpy array of shape (context_length=70, num_sensors=105)
    raw_spec_row: numpy array of shape (8,) of the categorical labels (strings or original codes)
    Returns:
      - predicted RUL scalar
    """
    # 2.1 Prepare the window:
    x_ts = torch.from_numpy(raw_window.reshape(1, CONTEXT_LENGTH, -1)).float().to(DEVICE)  # (1,70,105)

    # 2.2 Ordinal-encode the single spec row:
    #     encoder was already fit on the full spec_df, so we can just do:
    x_cat_np = encoder.transform(raw_spec_row.reshape(1, -1)).astype(int)  # (1, 8)
    x_cat = torch.from_numpy(x_cat_np).long().to(DEVICE)

    # 2.3 Forward‐pass
    with torch.no_grad():
        pred = model(x_cat, x_ts)  # (1, 1)
    return pred.cpu().item()

def main():
    # 3. Example usage: 
    #    Let’s assume you have a single new CSV-entry or a single window saved as "new_window.npy"
    #    and the corresponding spec row saved as "new_spec.npy" just for demonstration.
    #    In real life, you’d load them however you obtain new data.

    # Example: assume:
    #   new_window.npy  → shape (70, 105)
    #   new_spec.npy    → shape (8,) with exact same Spec_i string categories
    if not os.path.exists("new_window.npy") or not os.path.exists("new_spec.npy"):
        print("Place your test arrays as 'new_window.npy' and 'new_spec.npy'.")
        return

    raw_window    = np.load("new_window.npy")   # shape (70, 105)
    raw_spec_row  = np.load("new_spec.npy")     # shape (8, ), containing original spec labels

    model, encoder = load_model()
    pred_rul = infer_single_window(model, encoder, raw_window, raw_spec_row)
    print(f"Predicted RUL: {pred_rul:.4f}")

if __name__ == "__main__":
    main()
