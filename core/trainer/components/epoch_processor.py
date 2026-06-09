import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from core.schema import ExperimentState
from core.trainer.schema import TrainerComponents, TrainerState
from core.trainer.trainer_config import TrainerConfig

from modules.device_resolve import get_model_device

from .loss_computer import LossComputer



EPOCH_TYPE = [
    "train",
    "validation",
]



class EpochProcessor():
    
    def __init__(self, model:nn.Module, trainer_config:TrainerConfig, trainer_components:TrainerComponents, trainer_state:TrainerState) -> None:
        self.model = model
        self.trainer_config = trainer_config
        self.trainer_components = trainer_components
        self.trainer_state = trainer_state
        
        self.device = get_model_device(model)
        
        self.loss_computer = LossComputer(trainer_config, trainer_components)
        return
    
    ### --- public functions ---
    
    def process(self, epoch_type:str, dataloader:DataLoader) -> None:
        self._check_epoch_type(epoch_type)
        self._change_model_mode(epoch_type)
        
        total_loss = self._get_batches_loss(epoch_type, dataloader)
        average_loss = total_loss / len(dataloader)
        
        self._update_log(epoch_type, average_loss)
        return
    
    ### --- private helper functions ---
    
    def _check_epoch_type(self, epoch_type:str) -> None:
        if not epoch_type in EPOCH_TYPE:
            raise TypeError("trainer error! invalid epoch type")
        return
    
    def _change_model_mode(self, epoch_type:str):
        if epoch_type == "train":
            self.model.train()
        elif epoch_type == "validation":
            self.model.eval()
        return
    
    def _get_batches_loss(self, epoch_type:str, dataloader:DataLoader) -> float:
        total_loss = 0.0
        
        with self._get_gradient_context(epoch_type):
            for batch in dataloader:
                batch_loss = self._get_single_batch_loss(batch, epoch_type)
                total_loss += batch_loss
        
        return total_loss
    
    def _get_gradient_context(self, epoch_type:str):
        if epoch_type == "train":
            gradient_context = torch.enable_grad()
        elif epoch_type == "validation":
            gradient_context = torch.no_grad()
        return gradient_context
    
    def _get_single_batch_loss(self, batch:dict[str, Tensor|list], epoch_type:str) -> float:
        if epoch_type == "train":
            self.trainer_components.optimizer.zero_grad()
        
        input = batch["value"].float().to(self.device)
        label = batch["ground_truth"].float().to(self.device)
        
        output = self.model(input)
        loss = self.loss_computer.compute(output, label)
        
        if epoch_type == "train":
            self._apply_gradients(loss)
            
        return loss.item()
    
    def _apply_gradients(self, loss:Tensor) -> None:
        loss.backward()
        self._clip_gradients()
        self.trainer_components.optimizer.step()
        self.trainer_components.scheduler.step()
        return
    
    def _clip_gradients(self) -> None:
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm = self.trainer_config.grad_clip_max_norm,
        )
        return     
    
    def _update_log(self, epoch_type:str, loss:float) -> None:
        if epoch_type == "train":
            self.trainer_state.train_loss.append(loss)
        if epoch_type == "validation":
            self.trainer_state.validation_loss.append(loss)
        return