from core.master_config import MasterConfig
from core.schema import ExperimentStateFactory
from core.dataset.datamodule import get_datamodule, DatasetConfig
from core.model.model_control import get_model, ModelConfig

from .trainer import Trainer
from .trainer_config import TrainerConfig



def unit_test():
    master_config = MasterConfig.default()
    experiment_state = ExperimentStateFactory().create(master_config)
    
    trainer = Trainer(experiment_state)
    trainer.fit()
    
    return



if __name__ == "__main__":
    unit_test()
    print("Done!")