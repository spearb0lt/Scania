import os
import json
import joblib
import h5py
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
# from models import CombinedRULModel  # assuming models.py is on your PYTHONPATH
from datetime import datetime
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from tab_transformer_pytorch import TabTransformer
from torch.optim import Adam
from torch.nn import MSELoss
from sklearn.metrics import mean_squared_error, mean_absolute_error

class TimeSeriesEmbedder(nn.Module):
    def __init__(self,num_features,d_model=128,n_heads=8,num_layers=2,dropout=0.1):
        super().__init__()
        self.input_proj=nn.Linear(num_features,d_model)
        enc_layer=nn.TransformerEncoderLayer(
            d_model=d_model,nhead=n_heads,dropout=dropout,batch_first=True
        )
        self.encoder=nn.TransformerEncoder(enc_layer,num_layers=num_layers)
    def forward(self,x):
        x=self.input_proj(x)
        x=self.encoder(x)
        return x[:,-1,:]

class CombinedRULModel(nn.Module):
    def __init__(self,num_sensor_features,context_length,categories,continuous_dim,cont_mean_std=None):
        super().__init__()
        self.tf=TimeSeriesEmbedder(num_sensor_features,continuous_dim)
        if cont_mean_std is None:
            cont_mean_std=torch.stack([torch.zeros(continuous_dim),torch.ones(continuous_dim)],dim=1)
        self.tab=TabTransformer(
            categories=categories,
            num_continuous=continuous_dim,
            dim=continuous_dim,
            dim_out=1,
            depth=6,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
            mlp_hidden_mults=(4,2),
            mlp_act=nn.ReLU(),
            continuous_mean_std=cont_mean_std
        )
    def forward(self,x_cat,x_ts):
        cont=self.tf(x_ts)
        return self.tab(x_cat,cont)








# 1) Configuration: point to artifacts root and test data
# WINDOW_CSV     = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\Code\super_same_norm.csv"
# SPEC_CSV       = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\data\train_specifications.csv"
ENCODER_PATH   = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\Important_script\spec_encoder.joblib"
# H5_PATH        = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\Important_script\data_windows.h5"
ARTIFACT_ROOT  = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\Important_script\artifacts"
# VALIDATION_CSV = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\Code\validation_super_same_norm.csv"
# VALIDATION_H5  = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\Important_script\validation_data.h5"

SENSOR_FEATURES = [
    '171_0', '666_0', '427_0', '837_0', '167_0', '167_1', '167_2', '167_3', '167_4',
    '167_5', '167_6', '167_7', '167_8', '167_9', '309_0', '272_0', '272_1', '272_2',
    '272_3', '272_4', '272_5', '272_6', '272_7', '272_8', '272_9', '835_0', '370_0',
    '291_0', '291_1', '291_2', '291_3', '291_4', '291_5', '291_6', '291_7', '291_8',
    '291_9', '291_10', '158_0', '158_1', '158_2', '158_3', '158_4', '158_5', '158_6',
    '158_7', '158_8', '158_9', '100_0', '459_0', '459_1', '459_2', '459_3', '459_4',
    '459_5', '459_6', '459_7', '459_8', '459_9', '459_10', '459_11', '459_12', '459_13',
    '459_14', '459_15', '459_16', '459_17', '459_18', '459_19', '397_0', '397_1', '397_2',
    '397_3', '397_4', '397_5', '397_6', '397_7', '397_8', '397_9', '397_10', '397_11',
    '397_12', '397_13', '397_14', '397_15', '397_16', '397_17', '397_18', '397_19',
    '397_20', '397_21', '397_22', '397_23', '397_24', '397_25', '397_26', '397_27',
    '397_28', '397_29', '397_30', '397_31', '397_32', '397_33', '397_34', '397_35'
]
# Choose one:
USE_H5_TEST     = False
TEST_H5_PATH    = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\Important_script\validation_data.h5"
TEST_CSV_PATH   = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\Code\validation_super_same_norm.csv"


# 2) Locate the latest model folder
# model_name="CombinedRULModel-20250613_024224-NDP"
model_name="CombinedRULModel-NDP-20250613_030711"
# model_name=safed
# model_name=laal
# model_name=neel

   
# path=os.path.join(ARTIFACT_ROOT,model_name)
model_dir = os.path.join(ARTIFACT_ROOT, model_name)

# 3) Load metadata and encoder
with open(os.path.join(model_dir, "metadata.json")) as f:
    meta = json.load(f)
encoder = joblib.load(ENCODER_PATH)

# 4) Rebuild model and load checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CombinedRULModel(
    num_sensor_features=meta["num_sensor_features"],
    context_length=meta["context_length"],
    categories=tuple(meta["categories"]),
    continuous_dim=meta["continuous_dim"]
).to(device)
ckpt = torch.load(os.path.join(model_dir, "checkpoint.pth"), map_location=device)
model.load_state_dict(ckpt)
model.eval()

# 5) Prepare test data
if USE_H5_TEST:
    with h5py.File(TEST_H5_PATH, "r") as f:
        X_test = f["X_windows"][:]
        y_test = f["y_labels"][:]
        specs  = f["specs_per_window"][:]
else:
    df = pd.read_csv(TEST_CSV_PATH)
    windows, labels, vids = [], [], []
    for vid, grp in df.groupby("vehicle_id"):
        data = grp[SENSOR_FEATURES].values
        rul  = grp["RUL"].values
        c = meta["context_length"]
        if len(data) < c:
            continue
        for i in range(len(data) - c + 1):
            windows.append(data[i : i + c])
            labels.append(rul[i + c - 1])
            vids.append(vid)
    X_test = np.stack(windows)
    y_test = np.array(labels)
    spec_df = pd.read_csv(r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\data\validation_specifications.csv")
    spec_cols = [f"Spec_{i}" for i in range(8)]
    merged = (
        pd.DataFrame({"vehicle_id": vids})
          .merge(spec_df[["vehicle_id"] + spec_cols], on="vehicle_id", how="left")
    )
    specs_raw = merged[spec_cols].values
    specs = encoder.transform(specs_raw).astype(int)

# 6) Run inference in batches
preds = []
bs = 256
with torch.no_grad():
    for start in range(0, len(X_test), bs):
        end = start + bs
        x_ts  = torch.from_numpy(X_test[start:end]).float().to(device)
        x_cat = torch.from_numpy(specs[start:end]).long().to(device)
        out = model(x_cat, x_ts).squeeze().cpu().numpy()
        preds.append(out)
preds = np.concatenate(preds)

# 7) Compute and print metrics
mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)
print(f"Test MSE = {mse:.4f}, Test MAE = {mae:.4f}")
