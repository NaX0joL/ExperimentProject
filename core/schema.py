from dataclasses import dataclass

from torch import nn

from .master_config import (
    MasterConfig, 
    DatasetConfig, 
    ModelConfig, 
    MetricsConfig,
)

from .dataset.datamodule import DataModule, get_datamodule
from .model.model_control import get_model



@dataclass
class ExperimentState:
    master_config: MasterConfig
    datamodule: DataModule
    model: nn.Module
    loss_log: dict[str, list[float]]
    metrics_result: dict[str, dict]



class ExperimentStateFactory():
    
    def create(self, master_config:MasterConfig) -> ExperimentState:
        datamodule = self._get_datamodule(master_config.dataset_config)
        model = self._get_model(master_config.model_config)
        loss_log = self._init_loss_log()
        metrics_result = self._init_metrics_result(master_config.metrics_config, master_config.dataset_config)
        
        return ExperimentState(
            master_config = master_config,
            datamodule = datamodule,
            model = model,
            loss_log = loss_log,
            metrics_result = metrics_result,
        )
    
    ### private functions
        
    def _get_datamodule(self, dataset_config:DatasetConfig) -> DataModule:
        return get_datamodule(dataset_config)
    
    def _get_model(self, model_config:ModelConfig) -> nn.Module:
        return get_model(model_config)
    
    def _init_loss_log(self) -> dict[str, list[float]]:
        return {"train": [], "validation": []}
    
    def _init_metrics_result(self, metrics_config:MetricsConfig, dataset_config:DatasetConfig) -> dict[str, dict]:
        pass