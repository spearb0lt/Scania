# inference.py

import os
import torch
import numpy as np

from utils import load_model, SPEC_CSV_PATH

# ---------------------------------------------------
# 1. Hard-coded checkpoint path
# ---------------------------------------------------
# You must set these exactly to match the folder created during training.
ARTIFACT_FOLDER = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\Code2\ModularApproach\CombinedRULModel-20250604_142012"
CHECKPOINT_PATH = os.path.join(ARTIFACT_FOLDER, "checkpoint.pth")

# ---------------------------------------------------
# 2. Inference Function
# ---------------------------------------------------
def infer_single_window(
    raw_window: np.ndarray,       # shape (70, 105)
    raw_spec_row: np.ndarray      # shape (8,) with original categorical labels (strings or codes)
):
    """
    1) Load model + encoder
    2) Ordinal-encode raw_spec_row
    3) Wrap raw_window into (1,70,105) FloatTensor
    4) Forward pass → scalar RUL
    Returns:
      - predicted RUL (float)
    """
    # 2.1 Load model & encoder
    num_sensor_features = raw_window.shape[1]  # should be 105
    continuous_dim = 128

    model, encoder = load_model(
        checkpoint_path=CHECKPOINT_PATH,
        num_sensor_features=num_sensor_features,
        continuous_dim=continuous_dim
    )
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # 2.2 Prepare inputs
    x_ts = torch.from_numpy(raw_window.reshape(1, *raw_window.shape)).float().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # Ordinal-encode spec row (1×8)
    raw_spec_row_2d = raw_spec_row.reshape(1, -1)
    x_cat_np = encoder.transform(raw_spec_row_2d).astype(int)  # (1,8)
    x_cat = torch.from_numpy(x_cat_np).long().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # 2.3 Inference
    model.eval()
    with torch.no_grad():
        pred = model(x_cat, x_ts)  # (1,1)

    return pred.cpu().item()

# ---------------------------------------------------
# 3. Example “main” (replace with your own data-loading logic)
# ---------------------------------------------------
if __name__ == "__main__":
    # Example: assume you saved test inputs as NumPy files at absolute paths
    NEW_WINDOW_PATH = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\Code\new_window.npy"  # (70,105)
    NEW_SPEC_PATH   = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\Code\new_spec.npy"    # (8,) with original spec labels

    if not (os.path.exists(NEW_WINDOW_PATH) and os.path.exists(NEW_SPEC_PATH)):
        print("Place 'new_window.npy' and 'new_spec.npy' at the specified absolute paths first.")
    else:
        raw_window   = np.load(NEW_WINDOW_PATH)    # shape (70, 105)
        raw_spec_row = np.load(NEW_SPEC_PATH)      # shape (8,)
        predicted_rul = infer_single_window(raw_window, raw_spec_row)
        print(f"Predicted RUL: {predicted_rul:.4f}")





# # inference.py
# import os
# import torch
# import numpy as np

# from utils import make_path, load_model
# from sklearn.preprocessing import OrdinalEncoder

# # ---------------------------------------------------
# # 1. Paths & Config
# # ---------------------------------------------------
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Example filenames; replace with real names or parse via argparse as needed:
# # “checkpoint_filename” should match exactly the <artifact_folder>/checkpoint.pth file name.
# # We also need to know which artifact folder holds it; for simplicity, assume you pass that name here:
# ARTIFACT_FOLDER_NAME = "CombinedRULModel-YYYYMMDD_HHMMSS"  # ← replace with actual timestamp folder
# CHECKPOINT_FILENAME = "checkpoint.pth"

# # Full checkpoint path:
# checkpoint_rel = os.path.join("artifacts", ARTIFACT_FOLDER_NAME, CHECKPOINT_FILENAME)

# # ---------------------------------------------------
# # 2. Inference Function
# # ---------------------------------------------------
# def infer_single_window(
#     raw_window: np.ndarray,       # shape (70, 105)
#     raw_spec_row: np.ndarray      # shape (8,) with original categorical labels (strings)
# ):
#     """
#     1) Load model + encoder
#     2) Ordinal-encode raw_spec_row
#     3) Wrap raw_window into (1,70,105) FloatTensor
#     4) Forward pass → scalar RUL
#     Returns:
#       - predicted RUL (float)
#     """
#     # 2.1 Load model & encoder
#     # We need to know `num_sensor_features` and `continuous_dim` to call load_model().
#     num_sensor_features = raw_window.shape[1]  # should be 105
#     continuous_dim = 128
#     categories = None  # not needed, load_model will re-derive from spec CSV

#     model, encoder = load_model(
#         model_filename=os.path.join(ARTIFACT_FOLDER_NAME, CHECKPOINT_FILENAME),
#         num_sensor_features=num_sensor_features,
#         categories=None,
#         continuous_dim=continuous_dim
#     )
#     model = model.to(DEVICE)

#     # 2.2 Prepare inputs
#     x_ts = torch.from_numpy(raw_window.reshape(1, *raw_window.shape)).float().to(DEVICE)  # (1,70,105)
#     # Ordinal-encode spec row (1×8)
#     raw_spec_row_2d = raw_spec_row.reshape(1, -1)
#     x_cat_np = encoder.transform(raw_spec_row_2d).astype(int)  # (1,8)
#     x_cat = torch.from_numpy(x_cat_np).long().to(DEVICE)

#     # 2.3 Inference
#     model.eval()
#     with torch.no_grad():
#         pred = model(x_cat, x_ts)  # (1,1)

#     return pred.cpu().item()

# # ---------------------------------------------------
# # 3. Example “main” (replace with real data-loading logic)
# # ---------------------------------------------------
# if __name__ == "__main__":
#     # Example: assume you saved test inputs as NumPy files under project root
#     #   new_window.npy  → shape (70, 105)
#     #   new_spec.npy    → shape (8,) with original strings or categories
#     new_window_path = make_path("new_window.npy")
#     new_spec_path   = make_path("new_spec.npy")

#     if not (os.path.exists(new_window_path) and os.path.exists(new_spec_path)):
#         print("Place 'new_window.npy' and 'new_spec.npy' under project root for inference.")
#     else:
#         raw_window = np.load(new_window_path)     # shape (70, 105)
#         raw_spec_row = np.load(new_spec_path)     # shape (8,)
#         predicted_rul = infer_single_window(raw_window, raw_spec_row)
#         print(f"Predicted RUL: {predicted_rul:.4f}")
