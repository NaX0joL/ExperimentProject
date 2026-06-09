from dataclasses import dataclass, field

import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from core.schema import ExperimentState
from core.dataset.datamodule import DataModule

from .registry import OPTIMIZER_REGISTRY, CRITERION_REGISTRY
from .trainer_config import TrainerConfig



@dataclass
class TrainerComponents:
    optimizer: optim.Optimizer
    criterion: nn.Module
    scheduler: lr_scheduler.LRScheduler


@dataclass
class TrainerState: 
    total_epochs: int = 0
    current_epoch: int = 0
    train_loss: list = field(default_factory=list) 
    validation_loss: list = field(default_factory=list)



class TrainerComponentsFactory():
    
    def create(self, experiment_state:ExperimentState) -> TrainerComponents:
        optimizer = self._build_optimizer(experiment_state.master_config.trainer_config, experiment_state.model)
        criterion = self._build_criterion(experiment_state.master_config.trainer_config)
        scheduler = self._build_scheduler(experiment_state.master_config.trainer_config, optimizer, experiment_state.datamodule)
        
        return TrainerComponents(
            optimizer = optimizer,
            criterion = criterion,
            scheduler = scheduler,
        )
    
    ### private functions
    
    def _build_optimizer(self, trainer_config:TrainerConfig, model:nn.Module) -> optim.Optimizer:
        return OPTIMIZER_REGISTRY[trainer_config.optimizer_name](
            model.parameters(),
            lr=trainer_config.learning_rate,
            weight_decay=trainer_config.weight_decay,
        )
        
    def _build_criterion(self, trainer_config:TrainerConfig) -> nn.Module:
        return CRITERION_REGISTRY[trainer_config.criterion_name]()
    
    def _build_scheduler(self, trainer_config:TrainerConfig, optimizer:optim.Optimizer, datamodule:DataModule):
        return lr_scheduler.OneCycleLR(
            optimizer = optimizer,
            steps_per_epoch = len(datamodule.train_dataloader),
            epochs = trainer_config.train_epochs,
            pct_start = trainer_config.percentage_start,
            max_lr = trainer_config.max_learning_rate,
            cycle_momentum = False if trainer_config.optimizer_name == "adamw" else True,
            #anneal_strategy = 'cos',
            #div_factor = self.config.div_factor,
            #final_div_factor = self.config.final_div_factor,
        )



class TrainerStateFactory():
    
    @classmethod
    def create(self, experiment_state:ExperimentState) -> TrainerState:
        return TrainerState(
            total_epochs = experiment_state.master_config.trainer_config.train_epochs,
        )