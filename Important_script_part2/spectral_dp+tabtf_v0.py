import os
import sys
import json
import time
import h5py
import joblib
import torch
import pandas as pd
import numpy as np

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

# === Hard-coded paths ===
WINDOW_CSV     = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\Code\super_same_norm.csv"
SPEC_CSV       = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\data\train_specifications.csv"
ENCODER_PATH   = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\Important_script\spec_encoder.joblib"
H5_PATH        = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\Important_script\data_windows.h5"
ARTIFACT_ROOT  = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\Important_script2\artifacts2"

# === Utils ===
def create_X_y(csv_path=WINDOW_CSV, sensor_features=None, context=70, verbose=True):
    df = pd.read_csv(csv_path)
    X, y, vids = [], [], []
    for vid, grp in df.groupby("vehicle_id"):
        data = grp[sensor_features].values
        rul = grp["RUL"].values
        if len(data) < context:
            if verbose:
                print(f"Skipping {vid}, len<{context}")
            continue
        for i in range(len(data) - context + 1):
            X.append(data[i:i+context])
            y.append(rul[i+context-1])
            vids.append(vid)
    X = np.stack(X)
    y = np.array(y)
    vids = np.array(vids)
    if verbose:
        print(f"Windows: {len(X)}, shape={X.shape[1:]}")

    spec_df = pd.read_csv(SPEC_CSV)
    spec_cols = [f"Spec_{i}" for i in range(8)]
    enc = OrdinalEncoder()
    spec_df[spec_cols] = enc.fit_transform(spec_df[spec_cols])
    specs = (
        pd.DataFrame({"vehicle_id": vids})
          .merge(spec_df[["vehicle_id"] + spec_cols], on="vehicle_id")
    )[spec_cols].values.astype(int)

    joblib.dump(enc, ENCODER_PATH)
    if verbose:
        print(f"Saved encoder → {ENCODER_PATH}")

    return X, y, vids, specs


def save_to_h5(X, y, vids, specs, h5_path=H5_PATH):
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("X_windows", data=X, compression="gzip")
        f.create_dataset("y_labels", data=y, compression="gzip")
        f.create_dataset("window_vids", data=vids, compression="gzip")
        f.create_dataset("specs_per_window", data=specs, compression="gzip")
    print(f"Saved H5 → {h5_path}")


def load_from_h5(h5_path=H5_PATH):
    with h5py.File(h5_path, "r") as f:
        X = f["X_windows"][:]
        y = f["y_labels"][:]
        vids = f["window_vids"][:]
        specs = f["specs_per_window"][:]
    return X, y, vids, specs


class RULCombinedDataset(Dataset):
    def __init__(self, X, specs, y):
        self.X = X
        self.specs = specs
        self.y = y.reshape(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.specs[i]).long(),
            torch.from_numpy(self.X[i]).float(),
            torch.from_numpy(self.y[i]).float()
        )


def make_artifact_folder(model_name, suffix):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"{model_name}-{suffix}-{ts}"
    path = os.path.join(ARTIFACT_ROOT, folder)
    os.makedirs(path, exist_ok=True)
    return path

# === Models ===
class TimeSeriesEmbedder(nn.Module):
    def __init__(self, num_features, d_model=128, n_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        return x[:, -1, :]


class CombinedRULModel(nn.Module):
    def __init__(self, num_sensor_features, context_length, categories, continuous_dim, cont_mean_std=None):
        super().__init__()
        self.tf = TimeSeriesEmbedder(num_sensor_features, continuous_dim)
        if cont_mean_std is None:
            cont_mean_std = torch.stack([torch.zeros(continuous_dim), torch.ones(continuous_dim)], dim=1)
        self.tab = TabTransformer(
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

    def forward(self, x_cat, x_ts):
        cont = self.tf(x_ts)
        return self.tab(x_cat, cont)

# === Set preferred CUDA library for linalg operations ===
# use only if the file is not executing or taking too long to run
torch.backends.cuda.preferred_linalg_library("magma")



# === DP Utilities for Spectral-DP ===
# def spectral_dp_gradient_update(model, sigma, clip_bound, spec_k):
#     for name, param in model.named_parameters():
#         if param.grad is None:
#             continue
#         grad = param.grad.detach()
#         if grad.dim() >= 2:
#             U, S, Vh = torch.linalg.svd(grad, full_matrices=False)
#             S_clipped = torch.clamp(S, max=clip_bound)
#               # Spectral filtering: keep only top-k singular values if specified
#             if spec_k is not None:
#                 N = S_clipped.numel()
#                 k = min(spec_k, N)
#                 # sort indices by value
#                 _, idx = torch.sort(S_clipped, descending=True)
#                 mask = torch.zeros_like(S_clipped)
#                 mask[idx[:k]] = 1.0
#                 S_clipped = S_clipped * mask
#             # Add Gaussian noise
#             noise = torch.randn_like(S_clipped) * sigma * clip_bound
#             S_noisy = S_clipped + noise
#             param.grad = (U @ torch.diag(S_noisy) @ Vh).to(param.grad.device)
#         else:
#             norm = torch.norm(grad)
#             factor = min(1.0, clip_bound / (norm + 1e-6))
#             grad_clipped = grad * factor
#             noise = torch.randn_like(grad_clipped) * sigma * clip_bound
#             param.grad = grad_clipped + noise

# Key Features:
# Checks only if grad.dim() >= 2 for SVD, otherwise applies norm clipping and noise.
# Does NOT reshape higher-dimensional tensors to 2D before SVD. If grad is 3D or higher, SVD will fail.
# No fallback for SVD errors: If SVD fails, it will raise an error.
# Does not reshape the processed gradient back to the original shape (since it never reshapes in the first place).

def spectral_dp_gradient_update(model, sigma, clip_bound, spec_k):
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        original_shape = grad.shape

        # Handle 1D gradients (e.g., biases) separately
        if grad.dim() == 1:
            norm = torch.norm(grad)
            factor = min(1.0, clip_bound / (norm + 1e-6))
            grad_clipped = grad * factor
            noise = torch.randn_like(grad_clipped) * sigma * clip_bound
            param.grad = grad_clipped + noise
            continue

        # Reshape to 2D if needed
        if grad.dim() > 2:
            grad_2d = grad.view(grad.shape[0], -1)
        elif grad.dim() == 2:
            grad_2d = grad
        else:
            # Should not reach here, but just in case
            grad_2d = grad.view(1, -1)

        try:
            U, S, Vh = torch.linalg.svd(grad_2d, full_matrices=False)
            S_clipped = torch.clamp(S, max=clip_bound)
            if spec_k is not None:
                N = S_clipped.numel()
                k = min(spec_k, N)
                _, idx = torch.sort(S_clipped, descending=True)
                mask = torch.zeros_like(S_clipped)
                mask[idx[:k]] = 1.0
                S_clipped = S_clipped * mask
            noise = torch.randn_like(S_clipped) * sigma * clip_bound
            S_noisy = S_clipped + noise
            noisy_grad_2d = U @ torch.diag(S_noisy) @ Vh
        except torch.linalg.LinAlgError:
            norm = torch.norm(grad_2d)
            factor = min(1.0, clip_bound / (norm + 1e-6))
            grad_clipped = grad_2d * factor
            noise = torch.randn_like(grad_clipped) * sigma * clip_bound
            noisy_grad_2d = grad_clipped + noise

        # Reshape back
        param.grad = noisy_grad_2d.reshape(original_shape).to(param.grad.device)
# Key Features:
# Handles 1D gradients (biases) separately: Applies norm clipping and noise directly, never tries SVD on 1D tensors.
# Reshapes all higher-dimensional gradients to 2D before SVD, ensuring SVD always gets a valid matrix.
# Fallback for SVD errors: If SVD fails, falls back to norm clipping and noise.
# Reshapes the processed gradient back to the original shape before assigning to param.grad.

# Summary Table
# Feature	                   Improved Version (First)	            Original/Commented (Second)
# Handles 1D gradients safely	 ✔️ (never SVD)	                         ✔️ (never SVD)
# Handles >2D gradients safely	 ✔️ (reshapes to 2D)	              ❌ (SVD may fail)
# Fallback for SVD errors	      ✔️ (norm clip + noise)	            ❌ (will error)
# Reshapes back to original shape	✔️	                                ❌
# Robust to unexpected shapes	   ✔️	                                  ❌



def get_criterion(): return MSELoss()
def get_optimizer(model,lr=1e-3): return Adam(model.parameters(),lr=lr)








# === Training function ===

# you can keep spec_k=None if you want to use all singular values

def train(use_h5=True, pvt=False, sigma=0.1, clip_bound=1.0, spec_k=2):# sigma=0.5 spec_k=None
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 256
    NUM_EPOCHS = 50
    LR = 1e-3
    LR_PAT = 5
    ES_PAT = 11
    LR_F = 0.5

    # Load or create data
    if use_h5 and os.path.exists(H5_PATH):
        X, y, _, specs = load_from_h5()
    else:
        X, y, _, specs = create_X_y(sensor_features=None)
        save_to_h5(X, y, _, specs)

    encoder = joblib.load(ENCODER_PATH)
    cat_sizes = tuple(len(c) for c in encoder.categories_)

    Xtr, Xv, str_, sv, yr, yv = train_test_split(X, specs, y, test_size=0.2, random_state=42)
    tl = DataLoader(RULCombinedDataset(Xtr, str_, yr), batch_size=BATCH_SIZE, shuffle=True)
    vl = DataLoader(RULCombinedDataset(Xv, sv, yv), batch_size=BATCH_SIZE, shuffle=False)

    # Determine suffix based on DP flag
    suffix = "DP" if pvt else "NDP"

    # Artifacts
    art = make_artifact_folder("CombinedRULModel", suffix)
    logp = os.path.join(art, "train_val_log.txt")
    metap = os.path.join(art, "metadata.json")
    ckpt = os.path.join(art, "checkpoint.pth")

    # Metadata init
    meta = {
        "model_name": "CombinedRULModel",
        "num_sensor_features": X.shape[2],
        "context_length": X.shape[1],
        "continuous_dim": 128,
        "categories": list(cat_sizes),
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "num_epochs": NUM_EPOCHS,
        "privacy": "Spectral-DP" if pvt else "None",
        "dp_sigma": sigma if pvt else None,
        "dp_clip_bound": clip_bound if pvt else None
    }
    with open(metap, "w") as f:
        json.dump(meta, f, indent=4)

    # Model setup
    model = CombinedRULModel(X.shape[2], X.shape[1], cat_sizes, 128).to(DEVICE)
    crit = get_criterion()
    opt = get_optimizer(model, lr=LR)
    sched = StepLR(opt, step_size=1, gamma=LR_F)

    best = float('inf')
    noimp = 0

    with open(logp, "w") as f:
        f.write("epoch,train_loss,val_loss,epoch_time,lr,notes\n")

    start_all = time.perf_counter()
    for ep in range(1, NUM_EPOCHS + 1):
        ep_start = time.perf_counter()
        lr_cur = opt.param_groups[0]['lr']

        # Training loop
        model.train()
        tloss = 0
        for xc, xt, yb in tl:
            xc, xt, yb = xc.to(DEVICE), xt.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xc, xt), yb)
            loss.backward()
            if pvt:
                spectral_dp_gradient_update(model, sigma, clip_bound, spec_k)
                opt.step()
            else:
                opt.step()
            tloss += loss.item() * yb.size(0)
        tloss /= len(tl.dataset)

        # Validation loop
        model.eval()
        vloss = 0
        with torch.no_grad():
            for xc, xt, yb in vl:
                xc, xt, yb = xc.to(DEVICE), xt.to(DEVICE), yb.to(DEVICE)
                l = crit(model(xc, xt), yb)
                vloss += l.item() * yb.size(0)
        vloss /= len(vl.dataset)

        # Checkpointing & scheduler
        elapsed = time.perf_counter() - ep_start
        h, m, s = map(int, [elapsed//3600, (elapsed%3600)//60, elapsed%60])
        et = f"{h:02d}:{m:02d}:{s:02d}"
        notes = ""
        if vloss < best:
            best = vloss
            noimp = 0
            torch.save(model.state_dict(), ckpt)
            notes = f"Saved at epoch {ep}"
        else:
            noimp += 1
            if noimp % LR_PAT == 0:
                sched.step()
                notes += " LR stepped"
            if noimp >= ES_PAT:
                notes += " Early stopping"

        with open(logp, "a") as f:
            f.write(f"{ep},{tloss:.6f},{vloss:.6f},{et},{lr_cur:.6g},{notes.strip()}\n")
        print(f"Epoch {ep:02d} Train {tloss:.4f} Val {vloss:.4f} Time {et} LR {lr_cur:.2e} {notes}")

        if noimp >= ES_PAT:
            print("Early stop")
            break

    total = time.perf_counter() - start_all
    h, m, s = map(int, [total//3600, (total%3600)//60, total%60])
    tt = f"{h:02d}:{m:02d}:{s:02d}"

    # Update metadata
    with open(metap, "r+") as f:
        d = json.load(f)
        d["total_training_time"] = tt
        f.seek(0)
        json.dump(d, f, indent=4)
        f.truncate()

    print(f"Done. Best val MSE {best:.4f}. Total time {tt}")

# === Entry Point ===
if __name__ == "__main__":
    # Set pvt=True to enable Spectral-DP
    train(use_h5=True, pvt=False)
    # train(use_h5=True, pvt=True, sigma=0.1, clip_bound=1.0, spec_k=2)









