import torch
from torch import Tensor



def normalize_tensor(input:Tensor, eps=1e-8) -> Tensor:
    min_val = input.min(dim=-1, keepdim=True)[0]
    max_val = input.max(dim=-1, keepdim=True)[0]
    return (input - min_val) / (max_val - min_val + eps)


def standardize_tensor(input: Tensor, eps=1e-8) -> Tensor:
    mean = input.mean(dim=-1, keepdim=True)
    std = input.std(dim=-1, keepdim=True)
    return (input - mean) / (std + eps)