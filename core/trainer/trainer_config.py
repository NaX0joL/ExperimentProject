from __future__ import annotations 
from dataclasses import dataclass, field

import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from core.abstract import ABSTRACT_Config
from core.dataset.datamodule import DataModule

from .registry import OPTIMIZER_REGISTRY, CRITERION_REGISTRY
#from .loss_manager import LossManager



@dataclass
class TrainerConfig(ABSTRACT_Config):
    train_epochs: int
    learning_rate: float
    
    use_scheduler: bool
    max_learning_rate: float
    percentage_start: float
    div_factor: float 
    final_div_factor: float
    
    weight_decay: float
    grad_clip_max_norm: float
    
    optimizer_name: str
    criterion_name: str
    
    use_best_model: bool
    
    loss_coefficients: dict[str, float] = field(default_factory=dict)
    
    @classmethod
    def default(cls, **modified_parameter):
        trainer_config = cls(
            train_epochs = 100,
            learning_rate = 1e-4,
            
            use_scheduler = False,
            max_learning_rate = 1e-3,
            percentage_start = 0.3,
            div_factor = 10,
            final_div_factor = 1e2,
            
            weight_decay = 1e-4,
            grad_clip_max_norm = 1.0,
            
            optimizer_name = "adamw",
            criterion_name = "mae",
            
            use_best_model = True,
            
            loss_coefficients = {
                "base_loss": 1,
                "weighted_mse_loss": 0,
            },
        )
        for name, value in modified_parameter.items():
            if hasattr(trainer_config, name):
                setattr(trainer_config, name, value)
        return trainer_config