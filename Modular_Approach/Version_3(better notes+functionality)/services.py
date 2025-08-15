# services.py

import torch.nn as nn
import torch.optim as optim

def get_criterion():
    return nn.MSELoss()

def get_optimizer(model, lr=1e-3):
    return optim.Adam(model.parameters(), lr=lr)
