# services.py
import torch
import torch.nn as nn

def get_criterion():
    """
    Returns the MSELoss criterion for regression.
    """
    return nn.MSELoss()

def get_optimizer(model: torch.nn.Module, lr: float = 1e-3) -> torch.optim.Optimizer:
    """
    Given a model, returns a torch.optim.Adam optimizer for all parameters.
    """
    return torch.optim.Adam(model.parameters(), lr=lr)
