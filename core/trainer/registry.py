from torch import optim, nn

from core.abstract import ABSTRACT_Loss
#from .loss.base_loss import Base_Loss



OPTIMIZER_REGISTRY: dict[str, optim.Optimizer] = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
}


CRITERION_REGISTRY: dict[str, nn.Module] = {
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "huber": nn.HuberLoss,
    "cross_entropy": nn.CrossEntropyLoss,
}


LOSS_REGISTRY: dict[str, ABSTRACT_Loss] = {
    "base_loss": None,
}