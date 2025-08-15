# models.py

import torch
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer

class TimeSeriesEmbedder(nn.Module):
    def __init__(self, num_features, d_model=128, n_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (batch, context, num_features)
        x = self.input_proj(x)       # → (batch, context, d_model)
        x = self.encoder(x)          # → (batch, context, d_model)
        return x[:, -1, :]           # last‐step pooling

class CombinedRULModel(nn.Module):
    def __init__(self, num_sensor_features, context_length, categories, continuous_dim, cont_mean_std=None):
        super().__init__()
        self.tf = TimeSeriesEmbedder(
            num_features=num_sensor_features,
            d_model=continuous_dim,
            n_heads=8,
            num_layers=2,
            dropout=0.1
        )
        if cont_mean_std is None:
            cont_mean_std = torch.stack([
                torch.zeros(continuous_dim),
                torch.ones(continuous_dim)
            ], dim=1)
        self.tabtf = TabTransformer(
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
        cont = self.tf(x_ts)            # → (batch,128)
        return self.tabtf(x_cat, cont)  # → (batch,1)
