from dataclasses import dataclass

from .model.model_config import ModelConfig
from .dataset.dataset_config import DatasetConfig
from .metrics.metrics_config import MetricsConfig
from .trainer.trainer_config import TrainerConfig



@dataclass
class MasterConfig:
    model_config: ModelConfig
    dataset_config: DatasetConfig
    metrics_config: MetricsConfig
    trainer_config: TrainerConfig
    
    @classmethod
    def default(self):
        master_config = self(
            model_config = ModelConfig.default(),
            dataset_config = DatasetConfig.default(),
            metrics_config = MetricsConfig.default(),
            trainer_config = TrainerConfig.default(),
        )
        return master_config