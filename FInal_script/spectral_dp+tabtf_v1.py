import os, joblib, h5py, json
import numpy as np
import sys
import torch, time, math
from torch import nn, optim, autograd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from custom_dp import compute_dp_sgd_privacy
from tab_transformer_pytorch import TabTransformer
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


# ----------------------------
# 1. Load data and encoder
# ----------------------------
parent = os.path.abspath('')
# h5_path = os.path.join(parent, 'dataset', 'train_data.h5')
# with h5py.File(h5_path, 'r') as f:
#     X_windows        = f['X_windows'][:]
#     y_labels         = f['y_labels'][:]
#     specs_per_window = f['specs_per_window'][:]
h5_path = r"/csehome/p23iot002/Shubhro/Important_script/data_windows.h5"


with h5py.File(h5_path, "r") as f:
        X = f["X_windows"][:]
        y = f["y_labels"][:]
        vids = f["window_vids"][:]
        specs = f["specs_per_window"][:]

X_windows = X
y_labels = y
specs_per_window = specs









print(f"Loaded: {X_windows.shape} {y_labels.shape} {specs_per_window.shape}")

ENCODER_PATH   = r"/csehome/p23iot002/Shubhro/Important_script/spec_encoder.joblib"
encoder = joblib.load(ENCODER_PATH)
# encoder = joblib.load(os.path.join(parent, 'dataset', 'spec_encoder.joblib'))
category_sizes = tuple(len(c) for c in encoder.categories_)
print(f"Category sizes: {category_sizes}")


# ----------------------------
# 2. Hyperparameters & settings
# ----------------------------
# Chosen batch size to balance privacy amplification and GPU memory
BATCH_SIZE      = 1024
# BATCH_SIZE      = 256

NUM_EPOCHS      = 100#2000
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
NOISE_START     = 0.8
NOISE_END       = 0.4
MAX_GRAD_NORM   = 1.0       # initial clipping bound
DELTA           = 1.0 / (len(X_windows) ** 1.1)
# Refined alphas range (dense low-to-mid orders, capped at 30)
import numpy as _np
ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)) # list(_np.linspace(1.5, 10.0, 44)) + list(range(12, 31, 2))
PATIENT         = 100
# FREEZE_EPOCH    = NUM_EPOCHS // 2
EMBED_DIM       = 256

# ----------------------------
# 3. Dataset & DataLoader
# ----------------------------
class RULDataset(Dataset):
    def __init__(self, windows, specs, labels):
        self.windows = windows
        self.specs   = specs
        self.labels  = labels.reshape(-1,1)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.specs[idx]).long(),
            torch.from_numpy(self.windows[idx]).float(),
            torch.from_numpy(self.labels[idx]).float()
        )

Xc_train, Xc_val, xs_train, xs_val, y_train, y_val = train_test_split(
    X_windows, specs_per_window, y_labels, test_size=0.2, random_state=42
)
train_loader = DataLoader(RULDataset(Xc_train, xs_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(RULDataset(Xc_val,   xs_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False)

dataset_size = len(train_loader.dataset)
sample_rate  = BATCH_SIZE / dataset_size

# ----------------------------
# 4. Model definitions
# ----------------------------
class TimeSeriesEmbedder(nn.Module):
    def __init__(self, num_features, d_model=128, n_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(num_features, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
    def forward(self, x):
        x = self.proj(x)
        # return self.enc(x)[:, -1, :]
        x= self.enc(x)
        return x[:, -1, :]  # Return the last time step's embedding

class CombinedRULModel(nn.Module):
    def __init__(self, num_features, categories, embed_dim):
        super().__init__()
        self.ts = TimeSeriesEmbedder(num_features, d_model=embed_dim)
        stats = torch.stack([torch.zeros(embed_dim), torch.ones(embed_dim)], dim=1)
        self.tab = TabTransformer(
            categories=categories, num_continuous=embed_dim,
            dim=embed_dim, dim_out=1, depth=6, heads=8,
            attn_dropout=0.1, ff_dropout=0.1,
            mlp_hidden_mults=(4,2), mlp_act=nn.ReLU(),
            continuous_mean_std=stats)
    def forward(self, x_cat, x_ts):
        emb = self.ts(x_ts)
        return self.tab(x_cat, emb)

# ----------------------------
# 5. Setup training
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CombinedRULModel(X_windows.shape[2], category_sizes, EMBED_DIM).to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
# optimizer = Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LR*0.1)
# scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
# ----------------------------
# 6. Training loop
# ----------------------------

# use only if the file is not executing or taking too long to run
torch.backends.cuda.preferred_linalg_library("magma")


# def spectral_dp_gradient_update(model, sigma=0.1, clip_bound=1, spec_k=None):
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

def spectral_dp_gradient_update(model, sigma, clip_bound, spec_k):
    clip_count = 0
    total_count = 0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        original_shape = grad.shape

        # Handle 1D gradients (e.g., biases) separately
        if grad.dim() == 1:
            norm = torch.norm(grad)
            factor = min(1.0, clip_bound / (norm + 1e-6))
            if factor < 1.0:
                clip_count += 1
            total_count += 1
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
            grad_2d = grad.view(1, -1)

        try:
            U, S, Vh = torch.linalg.svd(grad_2d, full_matrices=False)
            S_clipped = torch.clamp(S, max=clip_bound)
            clip_count += (S > clip_bound).sum().item()
            total_count += S.numel()
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
            if factor < 1.0:
                clip_count += 1
            total_count += 1
            grad_clipped = grad_2d * factor
            noise = torch.randn_like(grad_clipped) * sigma * clip_bound
            noisy_grad_2d = grad_clipped + noise

        param.grad = noisy_grad_2d.reshape(original_shape).to(param.grad.device)
    return clip_count, total_count











# clip_bound= MAX_GRAD_NORM
# spec_k = None  # Set to None for no spectral filtering, or an integer for top-k filtering
best_val = float('inf'); no_improve = 0
for epoch in range(1, NUM_EPOCHS+1):


    #my variables
    clip_bound= MAX_GRAD_NORM
    spec_k = None  # Set as per your requirement, or None for no filtering
    sigma=0.1



    start = time.perf_counter()
    print(f"\nEpoch {epoch:02d} → \n")
    




    # dynamic noise
    sigma = NOISE_START + (NOISE_END-NOISE_START)*min(epoch, NUM_EPOCHS)/NUM_EPOCHS
    



    

    # freeze ts-encoder halfway
    if no_improve == PATIENT//2: # epoch == FREEZE_EPOCH:
        for p in model.ts.parameters():
            p.requires_grad = False
    


    model.train(); clip_count = total_count = 0
    tloss = 0
    for x_cat, x_ts, y in train_loader:
        x_cat, x_ts, y = x_cat.to(device), x_ts.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x_cat, x_ts), y)
        loss.backward()
        c,t=spectral_dp_gradient_update(model, sigma, clip_bound, spec_k)
        clip_count += c
        total_count += t
        optimizer.step()
        tloss += loss.item() * y.size(0)
    tloss /= len(train_loader.dataset)

        # scheduler.step()
        # optimizer.zero_grad()

        # # per-sample clipping
        # grads_sum = {n: torch.zeros_like(p) for n,p in model.named_parameters() if p.requires_grad}
        # # grads_sum = {n: torch.zeros_like(p) for n,p in model.named_parameters()}
        # for i in range(x_cat.size(0)):
        #     loss_i = criterion(model(x_cat[i:i+1], x_ts[i:i+1]), y[i:i+1])
        #     grads = autograd.grad(loss_i, [p for p in model.parameters() if p.requires_grad], retain_graph=False)
        #     # grads = autograd.grad(loss_i, [p for p in model.parameters()], retain_graph=False)
        #     norm = torch.norm(torch.stack([g.norm() for g in grads]))
        #     coeff = (MAX_GRAD_NORM/(norm+1e-6)).clamp(max=1.0)
        #     total_count += 1; clip_count += (coeff<1).item()

        #     param_with_grad_req = [(n,p) for (n,p) in model.named_parameters() if p.requires_grad]

        #     # for (n,p), g in zip(model.named_parameters(), grads):
        #     for (n,p), g in zip(param_with_grad_req, grads):
        #         if p.requires_grad:
        #             grads_sum[n] += g * coeff
        
        # # add noise & step
        # for n,p in model.named_parameters():
        #     if not p.requires_grad: continue
        #     noise = torch.randn_like(p)*(sigma*MAX_GRAD_NORM)
        #     p.grad = (grads_sum[n] + noise)/BATCH_SIZE
        # optimizer.step(); scheduler.step(); optimizer.zero_grad()
    


    # # adaptive C
    if total_count > 0:
        frac = clip_count/total_count
        MAX_GRAD_NORM = max(0.5, min(2.0, MAX_GRAD_NORM*(1.1 if frac>0.25 else 0.9 if frac<0.1 else 1)))
        
    # eval
    model.eval()
    vloss = 0
    with torch.no_grad():
        for xc, xt, y in val_loader:
            xc, xt, y = xc.to(device), xt.to(device), y.to(device)
            vloss += criterion(model(xc, xt), y).item() * y.size(0)
        


        
        # train_mse = sum(criterion(model(xc.to(device), xt.to(device)), y.to(device)).item()*y.size(0)
        #                 for xc,xt,y in train_loader)/dataset_size
        # val_mse   = sum(criterion(model(xc.to(device), xt.to(device)), y.to(device)).item()*y.size(0)
        #                 for xc,xt,y in val_loader)/len(val_loader.dataset)
    vloss /= len(val_loader.dataset)


    end = time.perf_counter()
    elapsed = end - start
    h,m,s=map(int, [elapsed//3600, (elapsed%3600)//60, elapsed%60 ])
    print(f"Train MSE: {tloss:.4f} | Val MSE: {vloss:.4f} | σ={sigma:.3f} | C={MAX_GRAD_NORM:.3f}\nTime: {h}hrs. {m}min. {s}sec.")
    
    LR_PAT= PATIENT//3
    # i also added step lr incase

    # early stop
    if vloss < best_val:
        best_val = vloss; no_improve = 0; torch.save(model.state_dict(), 'best_dp_model_adaptive_largeralpha.pt')
        print(f"Model saved with val loss: {best_val}")
    else:
        no_improve += 1
        if no_improve % LR_PAT == 0:
            scheduler.step()
            print(f"Learning rate adjusted to {scheduler.get_last_lr()[0]:.6f}")
        if no_improve >= PATIENT:
            print(f"Early stopping at {epoch}"); break

# ----------------------------
# # 7. Privacy accounting
# # ----------------------------
epsilon, opt_alpha = compute_dp_sgd_privacy(
    sample_rate=sample_rate, noise_multiplier=sigma,
    epochs=epoch, delta=DELTA, alphas=ALPHAS, verbose=True)
print(f"Final (ε,δ)=({epsilon:.2f},{DELTA}), optimal α={opt_alpha}")








# Epoch 01 →

# Train MSE: 273.5648 | Val MSE: 290.3950 | σ=0.100 | C=1.000
# Time: 0hrs. 2min. 45sec.
# Model saved with val loss: 290.3949509656311
# DP-SGD with
#         sampling rate = 4.88%,
#         noise_multiplier = 0.1,
#         iterated over 21 steps,
# satisfies differential privacy with
#         epsilon = 572,
#         delta = 6.3374685224095e-05.
# The optimal alpha is 1.1.
# The privacy estimate is likely to be improved by expanding the set of alpha orders.
# Final (ε,δ)=(572.23,6.3374685224095e-05), optimal α=1.1