
# # fed_client.py
import argparse
import h5py
import joblib
import torch
import numpy as np
import flwr as fl

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
from tab_transformer_pytorch import TabTransformer

# === Paths and data loader ===
H5_PATH      = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\Important_script\data_windows.h5"
ENCODER_PATH = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\Important_script\spec_encoder.joblib"

# def load_from_h5(path=H5_PATH):
#     with h5py.File(path, "r") as f:
#         X = f["X_windows"][:]
#         y = f["y_labels"][:]
#         vids = f["window_vids"][:]
#         specs = f["specs_per_window"][:]
#     return X, y, vids, specs

# class RULCombinedDataset(Dataset):
#     def __init__(self, X, specs, y):
#         self.X = X; self.specs = specs; self.y = y.reshape(-1,1)
#     def __len__(self): return len(self.y)
#     def __getitem__(self, i):
#         return (
#             torch.from_numpy(self.specs[i]).long(),
#             torch.from_numpy(self.X[i]).float(),
#             torch.from_numpy(self.y[i]).float(),
#         )

# # === Model definitions ===
# class TimeSeriesEmbedder(nn.Module):
#     def __init__(self, num_features, d_model=128, n_heads=8, layers=2):
#         super().__init__()
#         self.proj = nn.Linear(num_features, d_model)
#         layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
#         self.enc = nn.TransformerEncoder(layer, num_layers=layers)
#     def forward(self, x): return self.enc(self.proj(x))[:, -1, :]

# class CombinedRULModel(nn.Module):
#     def __init__(self, num_features, seq_len, categories, cont_dim=128):
#         super().__init__()
#         self.tf = TimeSeriesEmbedder(num_features, cont_dim)
#         self.tab = TabTransformer(
#             categories=categories, num_continuous=cont_dim,
#             dim=cont_dim, dim_out=1, depth=6, heads=8,
#         )
#     def forward(self, x_cat, x_ts): return self.tab(x_cat, self.tf(x_ts))

# # === Flower client ===
# class RULClient(fl.client.NumPyClient):
#     def __init__(self, model, train_loader, val_loader, device):
#         self.model = model
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.device = device
#         self.criterion = MSELoss()
#         self.optimizer = Adam(self.model.parameters(), lr=1e-3)

#     # Updated signature to accept config
#     def get_parameters(self, config):
#         return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

#     def set_parameters(self, parameters, config=None):
#         state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
#         self.model.load_state_dict(state_dict, strict=True)

#     def fit(self, parameters, config):
#         # config and parameters passed by server
#         self.set_parameters(parameters)
#         self.model.train()
#         for x_cat, x_ts, y in self.train_loader:
#             x_cat, x_ts, y = x_cat.to(self.device), x_ts.to(self.device), y.to(self.device)
#             self.optimizer.zero_grad()
#             loss = self.criterion(self.model(x_cat, x_ts), y)
#             loss.backward()
#             self.optimizer.step()
#         return self.get_parameters(config), len(self.train_loader.dataset), {}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         self.model.eval()
#         loss = 0.0
#         with torch.no_grad():
#             for x_cat, x_ts, y in self.val_loader:
#                 x_cat, x_ts, y = x_cat.to(self.device), x_ts.to(self.device), y.to(self.device)
#                 loss += self.criterion(self.model(x_cat, x_ts), y).item() * y.size(0)
#         mse = loss / len(self.val_loader.dataset)
#         return float(mse), len(self.val_loader.dataset), {"mse": float(mse)}

# # === Entry point ===
# def main(client_id: int, num_clients: int = 2):
#     X, y, vids, specs = load_from_h5()
#     splits = np.array_split(np.unique(vids), num_clients)
#     mask = np.isin(vids, splits[client_id])
#     Xc, yc, sc = X[mask], y[mask], specs[mask]

#     ds = RULCombinedDataset(Xc, sc, yc)
#     val_size = max(1, int(0.1 * len(ds)))
#     train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds)-val_size, val_size])
#     train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=128)

#     encoder = joblib.load(ENCODER_PATH)
#     categories = tuple(len(c) for c in encoder.categories_)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = CombinedRULModel(X.shape[2], X.shape[1], categories).to(device)

#     client = RULClient(model, train_loader, val_loader, device)
#     fl.client.start_client(
#         server_address="127.0.0.1:8080",
#         client=client.to_client(),
#     )

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--client-id", type=int, required=True)
#     args = parser.parse_args()
#     main(args.client_id)



def load_from_h5(path=H5_PATH):
    with h5py.File(path, "r") as f:
        X = f["X_windows"][:]
        y = f["y_labels"][:]
        vids = f["window_vids"][:]
        specs = f["specs_per_window"][:]
    return X, y, vids, specs

class RULCombinedDataset(Dataset):
    def __init__(self, X, specs, y):
        self.X, self.specs, self.y = X, specs, y.reshape(-1,1)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.specs[idx]).long(),
            torch.from_numpy(self.X[idx]).float(),
            torch.from_numpy(self.y[idx]).float(),
        )

# === Model Definitions ===
class TimeSeriesEmbedder(nn.Module):
    def __init__(self, num_features, d_model=128, n_heads=8, layers=2):
        super().__init__()
        self.proj = nn.Linear(num_features, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
    def forward(self, x): return self.enc(self.proj(x))[:, -1, :]

class CombinedRULModel(nn.Module):
    def __init__(self, num_features, seq_len, categories, cont_dim=128):
        super().__init__()
        self.tf = TimeSeriesEmbedder(num_features, cont_dim)
        self.tab = TabTransformer(
            categories=categories,
            num_continuous=cont_dim,
            dim=cont_dim,
            dim_out=1,
            depth=6,
            heads=8,
        )
    def forward(self, x_cat, x_ts):
        return self.tab(x_cat, self.tf(x_ts))

# === Flower Client ===
class RULClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters, config=None):
        state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config=None):
        self.set_parameters(parameters)
        self.model.train()
        for x_cat, x_ts, y in self.train_loader:
            x_cat, x_ts, y = x_cat.to(self.device), x_ts.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(x_cat, x_ts), y)
            loss.backward()
            self.optimizer.step()
        mse = self._evaluate_local()
        return self.get_parameters(config), len(self.train_loader.dataset), {"mse": mse}

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        mse = self._evaluate_local()
        return float(mse), len(self.val_loader.dataset), {"mse": float(mse)}

    def _evaluate_local(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x_cat, x_ts, y in self.val_loader:
                x_cat, x_ts, y = x_cat.to(self.device), x_ts.to(self.device), y.to(self.device)
                total_loss += self.criterion(self.model(x_cat, x_ts), y).item() * y.size(0)
        return total_loss / len(self.val_loader.dataset)

# === Entry Point for Client ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=int, required=True)
    args = parser.parse_args()

    num_clients = 2  # Adjust as needed

    X, y, vids, specs = load_from_h5()
    splits = np.array_split(np.unique(vids), num_clients)
    mask = np.isin(vids, splits[args.client_id])
    Xc, yc, sc = X[mask], y[mask], specs[mask]

    ds = RULCombinedDataset(Xc, sc, yc)
    val_size = max(1, int(0.1 * len(ds)))
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds)-val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)

    encoder = joblib.load(ENCODER_PATH)
    categories = tuple(len(c) for c in encoder.categories_)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CombinedRULModel(X.shape[2], X.shape[1], categories).to(device)

    client = RULClient(model, train_loader, val_loader, device)
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client(),
    )