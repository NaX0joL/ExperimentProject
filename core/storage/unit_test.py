from core.master_config import MasterConfig
from core.schema import ExperimentStateFactory
from core.trainer.trainer import Trainer

from .save import SaveService
from .load import LoadService



def unit_test():
    master_config = MasterConfig.default()
    master_config.trainer_config.train_epochs = 5
    experiment_state = ExperimentStateFactory().create(master_config)
    
    trainer = Trainer(experiment_state)
    trainer.fit()
    
    save_service = SaveService(experiment_state)
    save_service.put_in_marker(
        a="a",
        b=True,
        c=2,
    )
    save_path = "tmp/mpkg-for_unit_test"
    save_service.save(save_path)
    
    load_service = LoadService()
    load_path = f"savefolder/mpkg/{save_path}"
    load_service.load(load_path)
    print(load_service.get_marker_content())
    
    return



if __name__ == "__main__":    
    unit_test()
    print("Done!")