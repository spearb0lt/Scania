# below is for initial stroing of h5 files

import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OrdinalEncoder

from tab_transformer_pytorch import TabTransformer  # make sure this is installed
import torch.nn.functional as F


# ----------------------------
# 1. Prepare your raw data
# ----------------------------

def create_X_y(csv_path, sensor_features, context=70, verbose=True):
    """
    Reads the time‐series CSV, slides over each vehicle’s data to form windows of length `context`.
    Returns:
      - X:  numpy array of shape (N, context, num_features)
      - y:  numpy array of shape (N,) of RUL labels
      - vids: numpy array of shape (N,) of vehicle_ids, one per window
    """
    df = pd.read_csv(csv_path)
    X, y, vids = [], [], []
    for vehicle_id, group in df.groupby('vehicle_id'):
        # group = group.sort_values('time_step')
        data = group[sensor_features].values
        rul = group['RUL'].values
        if len(data) < context:
            if verbose:
                print(f"Skipping vehicle {vehicle_id}: {len(data)} < {context}")
            continue
        for i in range(len(data) - context + 1):
            X.append(data[i : i + context])
            y.append(rul[i + context - 1])
            vids.append(vehicle_id)
    X = np.stack(X)      # (N, context, num_features)
    y = np.array(y)      # (N,)
    vids = np.array(vids)
    print(f"Total windows: {len(X)}, window shape: {X.shape[1:]}")

    # Load & ordinal‐encode the vehicle specs
    spec_df = pd.read_csv(r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\data\train_specifications.csv")
    spec_columns = [f"Spec_{i}" for i in range(8)]
    encoder = OrdinalEncoder()
    spec_df[spec_columns] = encoder.fit_transform(spec_df[spec_columns])

    # Build a matrix of specs per window, aligned with X_windows ordering
    specs_per_window = (
        pd.DataFrame({"vehicle_id": vids})
        .merge(spec_df[["vehicle_id"] + spec_columns], on="vehicle_id", how="left")
    )[spec_columns].values.astype(int)
    # specs_per_window: shape (N, 8)

    # i am storing encoder also
    import joblib

    joblib.dump(encoder, r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\Important script\spec_encoder.joblib")
    print(f"Saved encoder to spec_encoder.joblib")
    

        



    return X, y, vids, specs_per_window

# Define sensor features exactly as you had before
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

csv_path = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\Code\super_same_norm.csv"
X_windows, y_labels, window_vids, specs_per_window = create_X_y(csv_path, sensor_features, context=70)

# storing it into h5 format here ia m doinng so
import h5py
import numpy as np

h5_path = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\Important script\data_windows.h5"
with h5py.File(h5_path, "w") as f:
    f.create_dataset("X_windows", data=X_windows, compression="gzip")
    f.create_dataset("y_labels", data=y_labels, compression="gzip")
    f.create_dataset("window_vids", data=window_vids, compression="gzip")
    f.create_dataset("specs_per_window", data=specs_per_window, compression="gzip")
print(f"Saved datasets to {h5_path}")

# with open("sensor_features.json", "w") as f:
#     json.dump(sensor_features, f)






# NOTE-> I AM ALSO STORING THE ENCODER IN JOBLIB FORMAT, SO THAT I CAN LOAD IT BACK IN

# NOTE-> STORING SENSOR FEATURES IN JSON FORMAT IS NOT NEEDED, AS I AM USING THE H5 FILE DIRECTLY


# -------------------------------------------------------------------------------------------------
# FROM HERE ONWARDS WE IMPORT H5 AND DO THE TRAINING PART
# -------------------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OrdinalEncoder

from tab_transformer_pytorch import TabTransformer  # make sure this is installed
import torch.nn.functional as F










# 2. Loading back in Step 2
import h5py
import numpy as np
h5_path = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\Important script\data_windows.h5"

with h5py.File(h5_path, "r") as f:
    X_windows       = f["X_windows"][:]        # array of shape (N, context, num_features)
    y_labels        = f["y_labels"][:]         # array of shape (N,)
    window_vids     = f["window_vids"][:]      # array of shape (N,)
    specs_per_window= f["specs_per_window"][:] # array of shape (N, 8)

print("Loaded:", X_windows.shape, y_labels.shape, window_vids.shape, specs_per_window.shape)




import joblib

encoder = joblib.load(r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\Important script\spec_encoder.joblib")
print(f"Loaded encoder with categories: {encoder.categories_}")

print(f"Specs shape: {specs_per_window.shape}, all 8:\n{specs_per_window[:8]}")

category_sizes = tuple(len(cat) for cat in encoder.categories_)
print(f"Category sizes: {category_sizes}")

# with open("sensor_features.json", "r") as f:
#     sensor_features = json.load(f)

# num_sensor_features = len(sensor_features)




# ----------------------------
# 2. Define the Dataset
# ----------------------------

class RULCombinedDataset(Dataset):
    """
    Returns, for each index:
      - x_ts: time-series window, shape (context, num_features)
      - x_categ: categorical specs, shape (8,)
      - y: RUL label (scalar)
    """
    def __init__(self, windows: np.ndarray, specs: np.ndarray, labels: np.ndarray):
        super().__init__()
        self.windows = windows            # (N, context, num_features)
        self.specs = specs                # (N, 8)
        self.labels = labels.reshape(-1, 1)  # (N, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x_ts = torch.from_numpy(self.windows[idx]).float()    # (context, num_features)
        x_categ = torch.from_numpy(self.specs[idx]).long()    # (8,)
        y = torch.from_numpy(self.labels[idx]).float()        # (1,)
        return x_categ, x_ts, y

# Split into train/validation sets
from sklearn.model_selection import train_test_split

Xc_train, Xc_val, xspec_train, xspec_val, y_train, y_val = train_test_split(
    X_windows, specs_per_window, y_labels, test_size=0.2, random_state=42
)

train_dataset = RULCombinedDataset(Xc_train, xspec_train, y_train)
val_dataset   = RULCombinedDataset(Xc_val,   xspec_val,   y_val)

BATCH_SIZE = 256
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ----------------------------
# 3. Define the Combined Model
# ----------------------------


class TimeSeriesEmbedder(nn.Module):
    def __init__(self, num_features, d_model=128, n_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        # Project raw sensor‐dim → d_model
        self.input_proj = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # We will mean‐pool over the time dimension

    def forward(self, x):
        """
        x: (batch_size, context_length, num_features)
        returns: (batch_size, d_model)
        """
        # 1) Project each time‐step’s features into d_model
        x = self.input_proj(x)         # (batch, context, d_model)
        # 2) Feed into standard PyTorch TransformerEncoder
        x = self.encoder(x)            # (batch, context, d_model)
        # 3) Mean‐pool across the context dimension
        # x = x.mean(dim=1)              # (batch, d_model)
        x=x[:, -1, :]
        # x=F.tanh(x)
        return x


class CombinedRULModel(nn.Module):
    def __init__(self, num_sensor_features, context_length, categories, continuous_dim, cont_mean_std=None):
        """
        - num_sensor_features: number of raw sensor channels (e.g., 105)
        - context_length: length of each time window (e.g., 70)
        - categories: tuple of cardinalities for each categorical spec column (length 8)
        - continuous_dim: dimensionality of the embedder’s output (e.g., 128)
        - cont_mean_std: tensor of shape (continuous_dim, 2) for TabTransformer normalization;
                         if None, we assume no normalization (i.e., mean=0, std=1).
        """
        super().__init__()

        # self.layer_norm = nn.LayerNorm(continuous_dim)

        # 3.1 TimeSeries Embedder
        self.tf = TimeSeriesEmbedder(
            num_features=num_sensor_features,
            d_model=continuous_dim,
            n_heads=8,
            num_layers=2,
            dropout=0.1
        )

        # 3.2 TabTransformer
        # If you want TabTransformer to normalize continuous features, pass cont_mean_std.
        # Otherwise, set a trivial mean/std.
        if cont_mean_std is None:
            # Create a (continuous_dim x 2) tensor: mean=0, std=1 for each embedding dimension
            cont_mean_std = torch.stack([
                torch.zeros(continuous_dim),
                torch.ones(continuous_dim)
            ], dim=1)

        self.tabtf = TabTransformer(
            categories=categories,              # e.g. (num_levels_spec0, ..., num_levels_spec7)
            num_continuous=continuous_dim,      # 128
            dim=continuous_dim,                 # internal TabTransformer dimensionality
            dim_out=1,                          # single‐value regression
            depth=6,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
            mlp_hidden_mults=(4, 2),
            mlp_act=nn.ReLU()
             ,continuous_mean_std=cont_mean_std   # tensor shape (128, 2)
        )

    def forward(self, x_cat, x_ts):
        """
        x_cat:  (batch_size, 8)              # each entry is an integer spec index
        x_ts:   (batch_size, context, F)      # raw sensor window,(256, 70, 105)
        returns: (batch_size, 1)             # predicted RUL scalar
        """
        # 1) Compute 128‐dim embedding from raw time window
        all_embs = self.tf(x_ts)    # (batch_size, 128)
        # all_embs= self.layer_norm(all_embs)  # Apply LayerNorm to the embeddings    

        # 2) Feed embeddings + categorical specs to TabTransformer
        pred = self.tabtf(x_cat, all_embs)      # (batch_size, 1)
        return pred




# ---------------------------------------------------
# 4. Instantiate, Loss, Optimizer
# ---------------------------------------------------

# i am doing this to prevent writng all the sensor features again and also so 
# i dont have to store it in json format, i can just
# work directly use the X_windows h5 file in this code

num_sensor_features = X_windows.shape[2]  # (N, context, num_features)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pvt = False # True  # Set to True for Private Aggregation of Teacher Ensembles (PATE) DP training

category_sizes = tuple(len(cats) for cats in encoder.categories_)
model = CombinedRULModel(
    num_sensor_features=num_sensor_features,# num_sensor_features=len(sensor_features),
    context_length=70,
    categories=category_sizes,
    continuous_dim=128
).to(device)
criterion = nn.MSELoss()
# DP hyperparameters
max_grad_norm    = 1.0
noise_multiplier = 1.0
optimizer        = torch.optim.Adam(model.parameters(), lr=1e-3)

# --------------------------------
# 5. DP training step
# --------------------------------
def train_dp_batch(x_cat, x_ts, y):
# Accumulators--accumulate the clipped gradients for each parameter over all microbatches.
    summed_clipped = {name: torch.zeros_like(p) for name,p in model.named_parameters()}

    # Microbatch loop
    for i in range(x_cat.size(0)):
         # 2a. Prepare microbatch
        xi_cat = x_cat[i:i+1]; xi_ts = x_ts[i:i+1]; yi = y[i:i+1]

# losses is a scalar tensor (e.g. MSE) computed on that single example.
# torch.autograd.grad computes the gradient of losses w.r.t. each model parameter, returning a tuple of gradient tensors in the same order as model.parameters().
        
        losses = criterion(model(xi_cat, xi_ts), yi)

        # 2b. Compute per-sample gradients
        grads = torch.autograd.grad(losses, list(model.parameters()), retain_graph=False)

#  total_norm is the ℓ₂ norm of the full gradient vector for this example (the square root of sum of squares over all parameters).       

# This ensures that after scaling by clip_coef, the gradient norm is at most 𝐶
        
        # 2c. Clip each grad vector to norm C
        total_norm = torch.sqrt(sum(g.norm()**2 for g in grads))
        clip_coef = (max_grad_norm / (total_norm + 1e-6)).clamp(max=1.0)


# multiply each per‐parameter gradient g by the same clip_coef, so that the entire gradient vector is clipped, then add it into our running total in summed_clipped[name].
         # 2d. Accumulate
        for (name,p), g in zip(model.named_parameters(), grads):
            summed_clipped[name] += g * clip_coef

 
 
# Noise: We draw Gaussian noise with standard deviation σ = \texttt{noise_multiplier} × C.This is the calibrated noise that ensures differential privacy.
# Average: We add the noise to the summed, clipped gradients, then divide by the batch size (𝑛 so that p.grad is the average noisy gradient.
# assigning to p.grad, the standard optimizer.step() will consume these DP‐protected gradients.   
   
    # 3. After all microbatches: add noise & average
    for name, p in model.named_parameters():
        noise = torch.randn_like(summed_clipped[name]) * (noise_multiplier * max_grad_norm)
        p.grad = (summed_clipped[name] + noise) / x_cat.size(0)
    
    
    
    # optimizer.step()
    # optimizer.zero_grad()


# Putting it all together
# Per‐example processing via microbatches → exact per‐sample gradients.
# Clipping each per‐sample gradient to norm 𝐶
# Summation of clipped gradients over the full batch.
# Noise injection proportional to 𝐶 and the noise multiplier.
# Averaging to yield final gradients in p.grad.










# --------------------------------
# 6. Training & Validation
# --------------------------------
NUM_EPOCHS = 20
best_val = float('inf')
for epoch in range(1, NUM_EPOCHS+1):
    # Training
    model.train()
    train_loss = 0.0
    for x_cat, x_ts, y in train_loader:
        optimizer.zero_grad()
        x_cat, x_ts, y = x_cat.to(device), x_ts.to(device), y.to(device)
        if pvt==True:
            train_dp_batch(x_cat, x_ts, y)
        running = criterion(model(x_cat, x_ts), y)
        running.backward()
        optimizer.step()
        train_loss += running.item() * y.size(0)
    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_cat, x_ts, y in val_loader:
            x_cat, x_ts, y = x_cat.to(device), x_ts.to(device), y.to(device)
            running = criterion(model(x_cat, x_ts), y)
            val_loss += running.item() * y.size(0)
    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch:02d}: Train MSE={train_loss:.4f} | Val MSE={val_loss:.4f}")
    # Save
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "best_dp_model.pt")

print("Training complete. Best val MSE:", best_val)