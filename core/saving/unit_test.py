from core.master_config import MasterConfig
from core.schema import ExperimentStateFactory
from core.trainer.trainer import Trainer
from core.saving.saving import SaverLoader



def unit_test():
    master_config = MasterConfig.default()
    master_config.trainer_config.train_epochs = 5
    experiment_state = ExperimentStateFactory().create(master_config)
    
    trainer = Trainer(experiment_state)
    trainer.fit()
    
    saver_loader_service = SaverLoader(experiment_state)
    saver_loader_service.save()
    
    return



if __name__ == "__main__":    
    unit_test()
    print("Done!")