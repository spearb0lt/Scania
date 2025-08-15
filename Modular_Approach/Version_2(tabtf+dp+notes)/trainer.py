# trainer.py

import os
import json
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from services import get_criterion, get_optimizer
from utils import (
    create_X_y, save_to_h5, load_from_h5,
    ENCODER_PATH, RULCombinedDataset,
    train_val_split, make_artifact_folder, H5_PATH
)
import joblib
from models import CombinedRULModel

# --- OPTIONS ---
use_h5 = False        # True to load from HDF5, False to regenerate from CSV
pvt    = False        # DP flag

# --- HYPERPARAMETERS ---
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE    = 256
NUM_EPOCHS    = 5    # allow up to 50, but early stop at 11
LEARNING_RATE = 1e-3
max_grad_norm    = 1.0
noise_multiplier = 1.0

# StepLR / EarlyStopping settings
LR_PATIENCE = 5   # epochs w/o improvement → step LR
ES_PATIENCE = 11  # epochs w/o improvement → stop training
LR_FACTOR   = 0.5 # multiplicative LR reduction

# --- SENSOR FEATURES LIST (105 names) ---
sensor_features = [
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

# --- DATA LOADING / PREP ---
if use_h5 and os.path.exists(H5_PATH):
    print(f"Loading data from HDF5 → {H5_PATH}")
    X, y, vids, specs = load_from_h5()
else:
    print("Generating windows & specs from CSV")
    X, y, vids, specs = create_X_y(sensor_features=sensor_features, context=70)
    save_to_h5(X, y, vids, specs)

# Load encoder & category sizes
encoder = joblib.load(ENCODER_PATH)
category_sizes = tuple(len(c) for c in encoder.categories_)

# Train/val split & DataLoaders
X_tr, X_val, s_tr, s_val, y_tr, y_val = train_val_split(X, specs, y)
train_loader = DataLoader(RULCombinedDataset(X_tr, s_tr, y_tr),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(RULCombinedDataset(X_val, s_val, y_val),
                          batch_size=BATCH_SIZE, shuffle=False)

# --- ARTIFACT SETUP ---
model_name   = "CombinedRULModel"
artifact_dir = make_artifact_folder(model_name, pvt)  # adds -DP or -NDP
log_path     = os.path.join(artifact_dir, "train_val_log.txt")
meta_path    = os.path.join(artifact_dir, "metadata.json")
ckpt_path    = os.path.join(artifact_dir, "checkpoint.pth")

# --- Write initial metadata.json ---
metadata = {
    "model_name": model_name,
    "num_sensor_features": X.shape[2],
    "context_length": X.shape[1],
    "continuous_dim": 128,
    "categories": list(category_sizes),
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "num_epochs": NUM_EPOCHS,
    "pvt": pvt,
    # total_training_time will be added at end
}
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=4)

# --- Instantiate model, loss, optimizer, scheduler ---
model     = CombinedRULModel(
    num_sensor_features=X.shape[2],
    context_length=X.shape[1],
    categories=category_sizes,
    continuous_dim=128
).to(DEVICE)
criterion = get_criterion()
optimizer = get_optimizer(model, lr=LEARNING_RATE)

# StepLR scheduler (we will call scheduler.step() manually on patience)
scheduler = StepLR(optimizer, step_size=1, gamma=LR_FACTOR)

# --- DP microbatching helper (unchanged) ---
def train_dp_batch(x_cat, x_ts, yb):
    summed = {n: torch.zeros_like(p) for n,p in model.named_parameters()}
    for i in range(x_cat.size(0)):
        xi_cat, xi_ts, yi = x_cat[i:i+1], x_ts[i:i+1], yb[i:i+1]
        loss = criterion(model(xi_cat, xi_ts), yi)
        grads = torch.autograd.grad(loss, model.parameters())
        total_norm = torch.sqrt(sum(g.norm()**2 for g in grads))
        clip_coef = (max_grad_norm / (total_norm + 1e-6)).clamp(max=1.0)
        for (n,_), g in zip(model.named_parameters(), grads):
            summed[n] += g * clip_coef
    for n,p in model.named_parameters():
        noise = torch.randn_like(summed[n]) * (noise_multiplier * max_grad_norm)
        p.grad = (summed[n] + noise) / x_cat.size(0)

# --- Training Loop with LR stepping & EarlyStopping ---
with open(log_path, "w") as log:
    log.write("epoch,train_loss,val_loss,epoch_time,notes\n")

best_val = float("inf")
no_improve_count = 0
overall_start = time.perf_counter()

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start = time.perf_counter()

    # — Train —
    model.train()
    train_loss = 0.0
    for x_cat, x_ts, yb in train_loader:
        x_cat, x_ts, yb = x_cat.to(DEVICE), x_ts.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        if pvt:
            train_dp_batch(x_cat, x_ts, yb)
        loss = criterion(model(x_cat, x_ts), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * yb.size(0)
    train_loss /= len(train_loader.dataset)

    # — Validate —
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_cat, x_ts, yb in val_loader:
            x_cat, x_ts, yb = x_cat.to(DEVICE), x_ts.to(DEVICE), yb.to(DEVICE)
            loss = criterion(model(x_cat, x_ts), yb)
            val_loss += loss.item() * yb.size(0)
    val_loss /= len(val_loader.dataset)

    # — Epoch timing —
    epoch_end = time.perf_counter()
    elapsed = epoch_end - epoch_start
    hrs, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    epoch_time_str = f"{int(hrs):02d}:{int(mins):02d}:{int(secs):02d}"

    # — Improvement logic —
    notes = ""
    if val_loss < best_val:
        best_val = val_loss
        no_improve_count = 0
        torch.save(model.state_dict(), ckpt_path)
        notes = f"Model saved at epoch {epoch}"
    else:
        no_improve_count += 1
        # StepLR if reached LR_PATIENCE
        if no_improve_count % LR_PATIENCE == 0:
            scheduler.step()
            notes = (notes + " LR stepped").strip()
        # EarlyStopping if reached ES_PATIENCE
        if no_improve_count >= ES_PATIENCE:
            notes = (notes + " Early stopping").strip()

    # — Log & print —
    with open(log_path, "a") as log:
        log.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{epoch_time_str},{notes}\n")
    print(f"Epoch {epoch:02d} | Train {train_loss:.4f} | Val {val_loss:.4f} | Time {epoch_time_str} {notes}")

    if no_improve_count >= ES_PATIENCE:
        print(f"Stopping early at epoch {epoch} after {ES_PATIENCE} epochs without improvement.")
        break

overall_end = time.perf_counter()
total_elapsed = overall_end - overall_start
hrs, rem = divmod(total_elapsed, 3600)
mins, secs = divmod(rem, 60)
total_time_str = f"{int(hrs):02d}:{int(mins):02d}:{int(secs):02d}"

# --- Update metadata.json with total training time ---
with open(meta_path, "r+") as f:
    data = json.load(f)
    data["total_training_time"] = total_time_str
    f.seek(0)
    json.dump(data, f, indent=4)
    f.truncate()

print(f"Training complete. Best val MSE: {best_val:.4f}. Total training time: {total_time_str}")





























# import os
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import StepLR
# from datetime import datetime
# import json

# from models import CombinedRULModel
# from services import get_criterion, get_optimizer
# from utils import (
#     WINDOW_CSV_PATH,
#     SPEC_CSV_PATH,
#     ARTIFACT_ROOT_DIR,
#     create_X_y,
#     load_and_encode_specs,
#     RULCombinedDataset,
#     train_val_split,
#     make_artifact_folder
# )

# # ---------------------------------------------------
# # 1. Hyperparameters
# # ---------------------------------------------------
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SENSOR_FEATURES = [
#     '171_0', '666_0', '427_0', '837_0', '167_0', '167_1', '167_2', '167_3', '167_4',
#     '167_5', '167_6', '167_7', '167_8', '167_9', '309_0', '272_0', '272_1', '272_2',
#     '272_3', '272_4', '272_5', '272_6', '272_7', '272_8', '272_9', '835_0', '370_0',
#     '291_0', '291_1', '291_2', '291_3', '291_4', '291_5', '291_6', '291_7', '291_8',
#     '291_9', '291_10', '158_0', '158_1', '158_2', '158_3', '158_4', '158_5', '158_6',
#     '158_7', '158_8', '158_9', '100_0', '459_0', '459_1', '459_2', '459_3', '459_4',
#     '459_5', '459_6', '459_7', '459_8', '459_9', '459_10', '459_11', '459_12', '459_13',
#     '459_14', '459_15', '459_16', '459_17', '459_18', '459_19', '397_0', '397_1', '397_2',
#     '397_3', '397_4', '397_5', '397_6', '397_7', '397_8', '397_9', '397_10', '397_11',
#     '397_12', '397_13', '397_14', '397_15', '397_16', '397_17', '397_18', '397_19',
#     '397_20', '397_21', '397_22', '397_23', '397_24', '397_25', '397_26', '397_27',
#     '397_28', '397_29', '397_30', '397_31', '397_32', '397_33', '397_34', '397_35'
# ]

# CONTEXT_LENGTH = 70
# BATCH_SIZE     = 256
# NUM_EPOCHS     = 20
# LEARNING_RATE  = 1e-3

# # Scheduler hyperparameters
# STEP_SIZE = 5    # every 5 epochs, multiply LR by GAMMA
# GAMMA     = 0.5

# # Early stopping hyperparameters
# PATIENCE = 3     # stop if no val_loss improvement for 3 consecutive epochs

# # ---------------------------------------------------
# # 2. EarlyStopping Class
# # ---------------------------------------------------
# class EarlyStopping:
#     """
#     Stops training when validation loss does not improve for 'patience' epochs.
#     Saves the best_model_state when val_loss improves.
#     """
#     def __init__(self, patience=3, verbose=False):
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_loss = float("inf")
#         self.early_stop = False
#         self.best_state = None

#     def __call__(self, val_loss, model):
#         if val_loss < self.best_loss:
#             self.best_loss = val_loss
#             self.best_state = model.state_dict()
#             self.counter = 0
#             if self.verbose:
#                 print(f"Validation loss improved to {val_loss:.6f}. ")
#         else:
#             self.counter += 1
#             if self.verbose:
#                 print(f"No improvement in val_loss for {self.counter} epoch(s).")
#             if self.counter >= self.patience:
#                 self.early_stop = True

# # ---------------------------------------------------
# # 3. Main Training Flow
# # ---------------------------------------------------
# def main():
#     # 3.1 Prepare dataset
#     # (a) Create sliding-window X, y, vids
#     X_windows, y_labels, window_vids = create_X_y(
#         csv_path=WINDOW_CSV_PATH,
#         sensor_features=SENSOR_FEATURES,
#         context=CONTEXT_LENGTH,
#         verbose=True
#     )

#     # (b) Encode specs and align
#     specs_per_window, encoder = load_and_encode_specs(
#         spec_csv_path=SPEC_CSV_PATH,
#         window_vids=window_vids
#     )
#     category_sizes = tuple(len(encoder.categories_[i]) for i in range(8))  # used by model

#     # 3.2 Train/validation split
#     X_train, X_val, specs_train, specs_val, y_train, y_val = train_val_split(
#         X_windows, specs_per_window, y_labels, test_size=0.2, random_state=42
#     )

#     # Create PyTorch Datasets & DataLoaders
#     train_dataset = RULCombinedDataset(X_train, specs_train, y_train)
#     val_dataset   = RULCombinedDataset(X_val,   specs_val,   y_val)

#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
#     val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

#     # 3.3 Instantiate model, loss, optimizer, scheduler
#     num_sensor_features = len(SENSOR_FEATURES)  # 105
#     continuous_dim = 128

#     model = CombinedRULModel(
#         num_sensor_features=num_sensor_features,
#         context_length=CONTEXT_LENGTH,
#         categories=category_sizes,
#         continuous_dim=continuous_dim,
#         cont_mean_std=None
#     ).to(DEVICE)

#     criterion = get_criterion()
#     optimizer = get_optimizer(model, lr=LEARNING_RATE)
#     scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
#     early_stopper = EarlyStopping(patience=PATIENCE, verbose=True)

#     # 3.4 Prepare artifact folder and log files
#     model_name = "CombinedRULModel"
#     artifact_dir = make_artifact_folder(model_name)
#     # - training log file (loss per epoch)
#     training_log_path = os.path.join(artifact_dir, "train_val_log.txt")
#     # - metadata file for hyperparameters
#     metadata_log_path = os.path.join(artifact_dir, "metadata.json")
#     # - checkpoint file name
#     checkpoint_path = os.path.join(artifact_dir, "checkpoint.pth")

#     # 3.5 Save metadata (hyperparameters, architecture details) to metadata.json
#     metadata = {
#         "model_name": model_name,
#         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "num_sensor_features": num_sensor_features,
#         "context_length": CONTEXT_LENGTH,
#         "continuous_dim": continuous_dim,
#         "categories": category_sizes,
#         "batch_size": BATCH_SIZE,
#         "learning_rate": LEARNING_RATE,
#         "step_size": STEP_SIZE,
#         "gamma": GAMMA,
#         "num_epochs": NUM_EPOCHS,
#         "early_stopping_patience": PATIENCE
#     }
#     with open(metadata_log_path, "w") as meta_f:
#         json.dump(metadata, meta_f, indent=4)

#     # 3.6 Training loop
#     with open(training_log_path, "w") as log_f:
#         log_f.write("epoch,train_loss,val_loss\n")

#     best_val_loss = float("inf")

#     for epoch in range(1, NUM_EPOCHS + 1):
#         # --- Train one epoch ---
#         model.train()
#         running_train_loss = 0.0
#         for x_cat, x_ts, y in train_loader:
#             x_cat = x_cat.to(DEVICE)
#             x_ts  = x_ts.to(DEVICE)
#             y     = y.to(DEVICE)

#             optimizer.zero_grad()
#             preds = model(x_cat, x_ts)      # → (batch, 1)
#             loss = criterion(preds, y)
#             loss.backward()
#             optimizer.step()
#             running_train_loss += loss.item() * y.size(0)

#         train_loss = running_train_loss / len(train_loader.dataset)

#         # --- Validate one epoch ---
#         model.eval()
#         running_val_loss = 0.0
#         with torch.no_grad():
#             for x_cat, x_ts, y in val_loader:
#                 x_cat = x_cat.to(DEVICE)
#                 x_ts  = x_ts.to(DEVICE)
#                 y     = y.to(DEVICE)

#                 preds = model(x_cat, x_ts)
#                 v_loss = criterion(preds, y)
#                 running_val_loss += v_loss.item() * y.size(0)

#         val_loss = running_val_loss / len(val_loader.dataset)

#         # Step the scheduler after each epoch
#         scheduler.step()

#         # Log to file
#         with open(training_log_path, "a") as log_f:
#             log_f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f}\n")

#         print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

#         # Check early stopping
#         early_stopper(val_loss, model)
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             # Save best-so-far checkpoint
#             torch.save(model.state_dict(), checkpoint_path)

#         if early_stopper.early_stop:
#             print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
#             break

#     print("Training complete. Best validation loss:", best_val_loss)


# if __name__ == "__main__":
#     main()
