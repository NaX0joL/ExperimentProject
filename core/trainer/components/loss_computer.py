from __future__ import annotations
from torch import Tensor

from core.abstract import ABSTRACT_Loss
from core.trainer.trainer_config import TrainerConfig
from core.trainer.schema import TrainerComponents



class LossComputer():
    
    def __init__(self, trainer_config:TrainerConfig, trainer_components:TrainerComponents):
        self.trainer_config = trainer_config
        self.trainer_components = trainer_components
        self._init_loss_queue()
        return
    
    ### public functions
    
    def compute(self, x, y) -> Tensor:
        total_loss = 0

        for cls, coeff in self.loss_queue:
            loss = cls.forward(x, y)
            total_loss += loss * coeff
        
        return total_loss
    
    ### private functions
    
    def _init_loss_queue(self) -> None:
        self.loss_queue: list[tuple[ABSTRACT_Loss, float]] = []
        
        for name in LOSS_REGISTRY.keys():
            
            if name in self.trainer_config.loss_coefficients.keys():
                cls = LOSS_REGISTRY[name](self.trainer_components.criterion)
                coeff = self.trainer_config.loss_coefficients[name]
                
                self.loss_queue.append((cls, coeff))
        
        return



class BaseLoss(ABSTRACT_Loss):
    def __init__(self, criterion):
        self.criterion = criterion
        return
    
    def forward(self, x, y):
        loss = self.criterion(x, y)
        return loss



LOSS_REGISTRY = {
    "base_loss": BaseLoss,
}