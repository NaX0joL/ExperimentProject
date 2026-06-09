from __future__ import annotations

import os
from dataclasses import dataclass
from functools import wraps

import torch
from torch import optim, nn

from core.trainer.trainer_config import TrainerConfig
from core.trainer.schema import TrainerComponents, TrainerState

from modules.device_resolve import get_model_device



CHECKPOINT_PATH = "core/savefolder/checkpoint"



def check_if_enabled(func:function):
    @wraps(func)
    def wrapper(self:CheckpointManager, *args, **kwargs):
        if not self.trainer_config.use_best_model:
            return
        return func(self, *args, **kwargs)
    return wrapper



class CheckpointManager():
    
    def __init__(self, model:nn.Module, trainer_config:TrainerConfig, trainer_components:TrainerComponents, trainer_state:TrainerState):
        self.model = model
        self.trainer_config = trainer_config
        self.trainer_components = trainer_components
        self.trainer_state = trainer_state

        self.reset()
        
        os.makedirs(CHECKPOINT_PATH, exist_ok=True)
        return
    
    ### public functions
    
    @check_if_enabled
    def reset(self) -> None:
        self.best_validation_loss = float('inf')
        self.best_epoch = -1
        return
    
    @check_if_enabled
    def save(self) -> None:
        last_validation_loss = self.trainer_state.validation_loss[-1]
        
        if last_validation_loss < self.best_validation_loss:
            self.best_validation_loss = last_validation_loss
            self.best_epoch = self.trainer_state.current_epoch
            
            checkpoint = {
                "best_val_loss": self.best_validation_loss,
                "epoch": self.best_validation_loss,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.trainer_components.optimizer.state_dict(),
            }
            save_path = os.path.join(CHECKPOINT_PATH, "checkpoint.pth")
            torch.save(checkpoint, save_path)
            
        return
    
    @check_if_enabled
    def load_best(self) -> None:
        load_path = os.path.join(CHECKPOINT_PATH, "checkpoint.pth")
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path, map_location=get_model_device(self.model))
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            raise FileNotFoundError("trainer error! model checkpoint not found")
        
        return