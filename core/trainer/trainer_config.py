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
    def default(self):
        return self(
            train_epochs = 100,
            learning_rate = 1e-4,
            
            max_learning_rate = 1e-3,
            percentage_start = 0.3,
            div_factor = 25,
            final_div_factor = 1e4,
            
            weight_decay = 1e-4,
            grad_clip_max_norm = 1.0,
            
            optimizer_name = 'adamw',
            criterion_name = 'MAE',
            
            use_best_model = True,
            
            loss_coefficients = {
                "base_loss": 1,
            },
        )