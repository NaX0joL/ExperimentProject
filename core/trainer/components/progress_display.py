from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from tqdm import tqdm, trange

from torch import nn

from core.trainer.trainer_config import TrainerConfig
from core.trainer.schema import TrainerComponents, TrainerState
from modules.device_resolve import get_model_device



class ProgressDisplay():
    def __init__(self, model:nn.Module, trainer_state:TrainerState, trainer_components:TrainerComponents):
        self.model = model
        self.trainer_state = trainer_state
        self.trainer_components = trainer_components
        
        self.epoch_pbar = None
        return
    
    def start(self):
        print("Model training started!")
        print(f"using device {get_model_device(self.model)}")
        self.epoch_pbar = tqdm(
            total=self.trainer_state.total_epochs,
            desc="progress",
            position=0,
            leave=True,
        )
        return
    
    def epoch_update(self):
        self.epoch_pbar.update(1)
        self.epoch_pbar.set_postfix({
            "epoch": f"{self.trainer_state.current_epoch}/{self.trainer_state.total_epochs}",
            "train_loss": f"{self.trainer_state.train_loss[-1]:.5f}",
            "validation_loss": f"{self.trainer_state.validation_loss[-1]:.5f}",
        })
        return
    
    def batch_update(self):
        return
    
    def finish(self):
        if self.epoch_pbar:
            self.epoch_pbar.close()
        print("Model training finished!")
        return