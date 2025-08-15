# models.py

import torch
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer

class TimeSeriesEmbedder(nn.Module):
    """
    Takes as input a batch of raw sensor-windows shaped (batch, context_length, num_features),
    applies a small TransformerEncoder, and returns a single d_model-dim vector per window
    by taking the last time-step embedding (instead of mean pooling).
    """
    def __init__(self, num_features: int, d_model: int = 128, n_heads: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        # 1) Linear projection from num_features → d_model
        self.input_proj = nn.Linear(num_features, d_model)

        # 2) TransformerEncoder (batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3) LayerNorm after the encoder
        # self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, context_length, num_features)
        returns: (batch_size, d_model)
        """
        # 1) Project each time-step’s features into d_model
        x = self.input_proj(x)          # → (batch, context_length, d_model)

        # 2) Feed into standard PyTorch TransformerEncoder
        x = self.encoder(x)             # → (batch, context_length, d_model)

        # 3) Take the last-time-step embedding instead of mean:
        x = x[:, -1, :]                 # → (batch, d_model)

        # 4) Apply LayerNorm
        # x = self.layer_norm(x)          # → (batch, d_model)

        return x


class CombinedRULModel(nn.Module):
    """
    Combines:
      (a) TimeSeriesEmbedder  → raw (batch, 70, 105) → (batch, 128)
      (b) TabTransformer      → (batch, 8 categorical) + (batch, 128 continuous) → (batch, 1) RUL regression
    """
    def __init__(
        self,
        num_sensor_features: int,        # e.g. 105
        context_length: int,             # e.g. 70
        categories: tuple,               # cardinalities of each Spec_i column (length 8)
        continuous_dim: int = 128,       # must match TimeSeriesEmbedder’s output size
        cont_mean_std: torch.Tensor = None
    ):
        super().__init__()

        # 1) TimeSeriesEmbedder
        self.tf = TimeSeriesEmbedder(
            num_features=num_sensor_features,
            d_model=continuous_dim,
            n_heads=8,
            num_layers=2,
            dropout=0.1
        )

        # 2) TabTransformer
        #    If cont_mean_std is None, TabTransformer will assume (mean=0, std=1)
        if cont_mean_std is None:
            cont_mean_std = torch.stack([
                torch.zeros(continuous_dim),
                torch.ones(continuous_dim)
            ], dim=1)  # shape: (continuous_dim, 2)

        self.tabtf = TabTransformer(
            categories=categories,              # tuple of 8 ints
            num_continuous=continuous_dim,      # 128
            dim=continuous_dim,                 # identical to continuous_dim
            dim_out=1,                          # regression → 1 scalar
            depth=6,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
            mlp_hidden_mults=(4, 2),
            mlp_act=nn.ReLU(),
            # continuous_mean_std is commented out per your edited code
            continuous_mean_std=cont_mean_std
        )

    def forward(self, x_cat: torch.Tensor, x_ts: torch.Tensor) -> torch.Tensor:
        """
        x_cat: (batch_size, 8)        # each Spec_i is a LongTensor index
        x_ts:  (batch_size, context_length, num_sensor_features)
        Returns:
          - pred: (batch_size, 1)      # predicted RUL
        """
        # 1) Compute 128-dim continuous embedding from raw window
        cont_embed = self.tf(x_ts)         # → (batch, 128)

        # 2) Feed categorical specs + continuous embedding to TabTransformer
        pred = self.tabtf(x_cat, cont_embed)  # → (batch, 1)
        return pred
