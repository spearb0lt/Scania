# models.py
import torch
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer
import torch.nn.functional as F

class TimeSeriesEmbedder(nn.Module):

    """Takes as input a batch of raw sensor‐windows shaped (batch, context_length, num_features),
    applies a small TransformerEncoder, and returns a single d_model-dim vector per window by mean-pooling.
   """
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







"""
x: (batch_size, context_length, num_features)
x = self.input_proj(x)   # → (batch_size, context_length, d_model)#self.input_proj linearly maps each of the num_features at every time step into a d_model-dimensional vector.
x = self.encoder(x)      # → (batch_size, context_length, d_model) #self.encoder (the nn.TransformerEncoder) processes the entire sequence of length context_length and returns another sequence of the same shape:
|
\
  >  x: (batch_size, context_length, d_model)

for each window of 70 time steps, the encoder gives you 70 separate “token” embeddings, 
each of size d_model.





now 
x has shape (batch_size, context_length, d_model). 
specifically in my combined model—want a single fixed-size vector per window,
not a sequence of 70 vectors. 

TabTransformer expects its “continuous” input to be a 2D tensor of shape (batch_size, continuous_dim).
cannot feed it a 3D tensor of shape (batch_size, context_length, continuous_dim)



x = x.mean(dim=1)   # → (batch_size, d_model)
“pooling” (averaging) across all time steps. The result is a single d_model-length vector 
per training example in the batch. That vector is intended to capture, in a coarse way, 
the entire 70-step window’s information.

TransformerEncoder gives you 70 “time-step embeddings” of size d_model.
Need a single 128-dimensional embedding for each window.
Taking the arithmetic mean along dim=1 is a simple way to collapse the time dimension into one vector.

Other common pooling choices might be:
*Last time step: x = x[:, -1, :]
*Max pooling: x = x.max(dim=1).values
*Learned “classification” token: prepend a dummy token and read its embedding (like BERT’s [CLS]).




if i output x just like that then
model’s forward would output a 3D tensor (batch_size, 70, 128). 
The downstream TabTransformer is not designed to accept a 3D input.
 It expects:
*a categorical tensor of shape (batch_size, num_categorical_features)
*a continuous tensor of shape (batch_size, num_continuous_features)

skip the pooling, we need rework TabTransformer to process 70 separate continuous vectors per example.
 Shape mismatch—TabTransformer’s signature is forward(x_cat, x_cont) where x_cont must be 2D: (batch, cont_dim).
Architectural intent—we want each window summarized by a single embedding. Returning a 70-step sequence defeats that design.

"""


class CombinedRULModel(nn.Module):
    """
    Combines:
      (a) TimeSeriesEmbedder: raw 3D window (batch, 70, 105) → (batch, 128)
      (b) TabTransformer: (batch, 8 categorical) + (batch, 128 continuous) → (batch, 1) RUL prediction
    """
    def __init__(
        self,
        num_sensor_features: int,        # e.g. 105
        categories: tuple,               # cardinalities for each of the 8 Spec_i columns
        continuous_dim: int = 128,       # this must match the TimeSeriesEmbedder’s output size (d_model)
        cont_mean_std: torch.Tensor = None
    ):
        super().__init__()

        # self.layer_norm = nn.LayerNorm(continuous_dim)

        # 1) Create TimeSeriesEmbedder
        self.tf = TimeSeriesEmbedder(
            num_features=num_sensor_features,
            d_model=continuous_dim,
            n_heads=8,
            num_layers=2,
            dropout=0.1
        )

        # 2) TabTransformer for combining categorical specs + continuous embeddings
        #    If cont_mean_std is None, create a dummy (zeros, ones) so TabTransformer assumes standardized input.
        if cont_mean_std is None:
            cont_mean_std = torch.stack([
                torch.zeros(continuous_dim),
                torch.ones(continuous_dim)
            ], dim=1)

        self.tabtf = TabTransformer(
            categories=categories,            # e.g. (num_levels_Spec0, …, num_levels_Spec7)
            num_continuous=continuous_dim,    # 128
            dim=continuous_dim,               # TabTransformer’s internal embedding dimension
            dim_out=1,                        # single-value regression output
            depth=6,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
            mlp_hidden_mults=(4, 2),
            mlp_act=nn.ReLU(),
            continuous_mean_std=cont_mean_std # shape (continuous_dim, 2)
        )

    def forward(self, x_cat: torch.Tensor, x_ts: torch.Tensor) -> torch.Tensor:
        """
        - x_cat: (batch_size, 8)     # each Spec_i is a LongTensor index
        - x_ts:  (batch_size, 70, 105) # raw sensor window
        Returns:
          - pred: (batch_size, 1)     # RUL regression
        """
        # 1) Embed the raw window → (batch, 128)
        cont_embed = self.tf(x_ts)

        # 2) Feed categorical specs + continuous embeddings to TabTransformer
        return self.tabtf(x_cat, cont_embed)  # → (batch, 1)
