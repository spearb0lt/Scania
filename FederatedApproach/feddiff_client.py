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

# === Model ===
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
    def __init__(self, model, train_dataset, val_dataset, device, client_id):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.client_id = str(client_id)
        self.criterion = MSELoss()

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters, config=None):
        state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # Extract heterogeneous config
        epochs = config.get("local_epochs", 1)
        batch_size = config.get("batch_size", 128)
        lr = config.get("lr", 1e-3)

        # Rebuild DataLoader with new batch size
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Reset optimizer with new lr
        optimizer = Adam(self.model.parameters(), lr=lr)


        # lr = config.get("lr", 1e-3)
        # optimizer = Adam(self.model.parameters(), lr=lr)

        # Load global parameters
        self.set_parameters(parameters)
        self.model.train()
        for _ in range(epochs):
            for x_cat, x_ts, y in train_loader:
                x_cat, x_ts, y = x_cat.to(self.device), x_ts.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = self.criterion(self.model(x_cat, x_ts), y)
                loss.backward()
                optimizer.step()
        # Evaluate locally
        mse = self._evaluate_local()
        return self.get_parameters(config), len(train_loader.dataset), {"mse": mse}

    def evaluate(self, parameters, config=None):
        # Load global parameters
        self.set_parameters(parameters)
        mse = self._evaluate_local()
        return float(mse), len(self.val_dataset), {"mse": float(mse)}

    def _evaluate_local(self):
        # DataLoader for validation
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=False,
            num_workers=0,
        )
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x_cat, x_ts, y in val_loader:
                x_cat, x_ts, y = x_cat.to(self.device), x_ts.to(self.device), y.to(self.device)
                total_loss += self.criterion(self.model(x_cat, x_ts), y).item() * y.size(0)
        return total_loss / len(self.val_dataset)

# === Client Entry Point ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=int, required=True)
    args = parser.parse_args()


    num_clients=2
    # Load and split data
    X, y, vids, specs = load_from_h5()
    unique = np.unique(vids)
    splits = np.array_split(unique, num_clients)
    mask = np.isin(vids, splits[args.client_id])
    Xc, yc, sc = X[mask], y[mask], specs[mask]

    # Create datasets (not DataLoaders)
    ds = RULCombinedDataset(Xc, sc, yc)
    val_size = max(1, int(0.1 * len(ds)))
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds)-val_size, val_size])

    # Build model
    encoder = joblib.load(ENCODER_PATH)
    categories = tuple(len(c) for c in encoder.categories_)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CombinedRULModel(X.shape[2], X.shape[1], categories).to(device)

    # Start FL client, passing datasets
    client = RULClient(model, train_ds, val_ds, device, args.client_id)
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client(),
    )
