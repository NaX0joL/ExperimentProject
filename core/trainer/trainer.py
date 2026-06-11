from __future__ import annotations
from dataclasses import dataclass, field

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from core.abstract import ABSTRACT_Trainer
from core.schema import ExperimentState

from .registry import OPTIMIZER_REGISTRY, CRITERION_REGISTRY
from .schema import TrainerComponents, TrainerState, TrainerComponentsFactory, TrainerStateFactory
from .components.epoch_processor import EpochProcessor
from .components.loss_computer import LossComputer
from .components.checkpoint_manager import CheckpointManager
from .components.progress_display import ProgressDisplay
from .trainer_config import TrainerConfig



class Trainer(ABSTRACT_Trainer):
    def __init__(self, experiment_state:ExperimentState) -> None:
        self.experiment_state = experiment_state
        
        self._set_model_device()
        self._init_components()
        self._reset_state()

        self.epoch_processor = EpochProcessor(
            model = self.experiment_state.model, 
            trainer_config = self.experiment_state.master_config.trainer_config, 
            trainer_components = self.trainer_components, 
            trainer_state = self.trainer_state,
        )
        self.checkpoint_manager = CheckpointManager(
            model = self.experiment_state.model,
            trainer_config = self.experiment_state.master_config.trainer_config,
            trainer_components = self.trainer_components, 
            trainer_state = self.trainer_state,
        )
        self.progress_display = ProgressDisplay(
            model = experiment_state.model,
            trainer_components = self.trainer_components,
            trainer_state = self.trainer_state,
        )
        return
    
    ### --- public functions ---
    
    def get_model(self):
        return self.experiment_state.model
    
    def fit(self, timer=False) -> None:
        self._reset_state()
        self._relink_state_to_services()
        self.checkpoint_manager.reset()
        self.progress_display.start()
        
        for epoch in range(self.trainer_state.total_epochs):
            self.trainer_state.current_epoch = epoch + 1
            
            self.epoch_processor.process(
                epoch_type="train", 
                dataloader=self.experiment_state.datamodule.train_dataloader
            )
            self.epoch_processor.process(
                epoch_type="validation", 
                dataloader=self.experiment_state.datamodule.test_dataloader
            )
            
            self.checkpoint_manager.save()
            self.progress_display.epoch_update()
        
        self._sync_loss_log()
        self.checkpoint_manager.load_best()
        self.progress_display.finish()
        return
    
    ### --- private functions ---
    
    def _set_model_device(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experiment_state.model.to(device)
        return
    
    def _init_components(self) -> None:
        self.trainer_components = TrainerComponentsFactory().create(self.experiment_state)
        return
    
    def _reset_state(self) -> None:
        self.trainer_state = TrainerStateFactory.create(self.experiment_state)
        return
    
    def _relink_state_to_services(self) -> None:
        self.epoch_processor.trainer_state = self.trainer_state
        self.checkpoint_manager.trainer_state = self.trainer_state
        self.progress_display.trainer_state = self.trainer_state
        return
    
    def _sync_loss_log(self) -> None:
        self.experiment_state.loss_log["train"] = self.trainer_state.train_loss
        self.experiment_state.loss_log["validation"] = self.trainer_state.validation_loss
        return