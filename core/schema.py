from dataclasses import dataclass

from torch import nn

from .master_config import (
    MasterConfig, 
    DatasetConfig, 
    ModelConfig, 
    MetricsConfig,
)

from .dataset.datamodule import DataModule



@dataclass
class ExperimentState:
    master_config: MasterConfig
    datamodule: DataModule
    model: nn.Module
    loss_log: dict[str, list[float]]
    metrics_result: dict[str, dict]



class ExperimentStateFactory():
    
    def create(self, master_config:MasterConfig) -> ExperimentState:
        experiment_state = ExperimentState(
            master_config = master_config,
            datamodule = self._get_datamodule(master_config.dataset_config),
            model = self._get_model(master_config.model_config),
            loss_log = self._init_loss_log(),
            metrics_result = self._init_metrics_result(master_config.metrics_config, master_config.dataset_config)
        )
        return experiment_state
    
    ### private functions
        
    def _get_datamodule(self, dataset_config:DatasetConfig) -> DataModule:
        pass
    
    def _get_model(self, model_config:ModelConfig) -> nn.Module:
        pass
    
    def _init_loss_log(self) -> dict[str, list[float]]:
        return {"train": [], "validation": []}
    
    def _init_metrics_result(self, metrics_config:MetricsConfig, dataset_config:DatasetConfig) -> dict[str, dict]:
        pass