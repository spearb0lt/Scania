# trainer.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from models import CombinedRULModel
from services import get_criterion, get_optimizer
from utils import (
    create_X_y,
    load_and_encode_specs,
    RULCombinedDataset,
    train_val_split
)

# ----------------------------
# 1. Hyperparameters & Paths
# ----------------------------
CSV_PATH         = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\Code\super_same_norm.csv"
SPEC_CSV_PATH    = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\data\train_specifications.csv"

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
BATCH_SIZE     = 256
NUM_EPOCHS     = 20
LR             = 1e-3
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 2. Main Training Flow
# ----------------------------
def main():
    # 2.1 Create sliding-window data
    X_windows, y_labels, window_vids = create_X_y(
        csv_path=CSV_PATH,
        sensor_features=SENSOR_FEATURES,
        context=CONTEXT_LENGTH,
        verbose=True
    )

    # 2.2 Load & encode specs
    specs_per_window, encoder = load_and_encode_specs(
        spec_csv_path=SPEC_CSV_PATH,
        window_vids=window_vids
    )
    # specs_per_window: shape (N, 8), encoder.categories_ holds cardinalities

    # 2.3 Train/validation split
    X_train, X_val, specs_train, specs_val, y_train, y_val = train_val_split(
        X_windows, specs_per_window, y_labels, test_size=0.2, random_state=42
    )

    # 2.4 Create Datasets & DataLoaders
    train_dataset = RULCombinedDataset(X_train, specs_train, y_train)
    val_dataset   = RULCombinedDataset(X_val,   specs_val,   y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2.5 Instantiate model
    num_sensor_features = len(SENSOR_FEATURES)  # 105
    category_sizes = tuple(len(encoder.categories_[i]) for i in range(len(encoder.categories_)))
    continuous_dim = 128

    model = CombinedRULModel(
        num_sensor_features=num_sensor_features,
        categories=category_sizes,
        continuous_dim=continuous_dim,
        cont_mean_std=None  # TabTransformer will assume mean=0, std=1
    ).to(DEVICE)

    # 2.6 Loss & Optimizer
    criterion = get_criterion()
    optimizer = get_optimizer(model, lr=LR)

    # 2.7 Training & Validation loop
    best_val_loss = float("inf")
    for epoch in range(1, NUM_EPOCHS + 1):
        # --- Train one epoch ---
        model.train()
        train_running = 0.0
        for x_cat, x_ts, y in train_loader:
            x_cat = x_cat.to(DEVICE)
            x_ts  = x_ts.to(DEVICE)
            y     = y.to(DEVICE)

            optimizer.zero_grad()
            preds = model(x_cat, x_ts)          # (batch, 1)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            train_running += loss.item() * y.size(0)

        train_loss = train_running / len(train_loader.dataset)

        # --- Validate one epoch ---
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for x_cat, x_ts, y in val_loader:
                x_cat = x_cat.to(DEVICE)
                x_ts  = x_ts.to(DEVICE)
                y     = y.to(DEVICE)

                preds = model(x_cat, x_ts)
                loss = criterion(preds, y)
                val_running += loss.item() * y.size(0)

        val_loss = val_running / len(val_loader.dataset)

        print(f"Epoch {epoch:02d} → Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_combined_model.pt")

    print("Training complete. Best validation MSE:", best_val_loss)


if __name__ == "__main__":
    main()
