import torch
from torch import nn



def get_optimal_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_device(model:nn.Module) -> torch.device:
    return next(model.parameters()).device