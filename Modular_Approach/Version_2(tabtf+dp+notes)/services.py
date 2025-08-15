# services.py

import torch
import torch.nn as nn

def get_criterion() -> nn.Module:
    """
    Returns the MSELoss criterion (since we are doing regression of RUL).
    """
    return nn.MSELoss()

def get_optimizer(model: torch.nn.Module, lr: float = 1e-3) -> torch.optim.Optimizer:
    """
    Given a PyTorch model, returns an Adam optimizer over all its parameters.
    """
    return torch.optim.Adam(model.parameters(), lr=lr)
