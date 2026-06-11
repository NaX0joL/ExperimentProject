from core.dataset.unit_test import unit_test as dataset_unit_test
from core.model.unit_test import unit_test as model_unit_test
from core.trainer.unit_test import unit_test as trainer_unit_test
from core.metrics.unit_test import unit_test as metrics_unit_test
from core.storage.unit_test import unit_test as storage_unit_test

from core.experiment import Experiment
from core.master_config import MasterConfig



def unit_test():
    dataset_unit_test()
    model_unit_test()
    trainer_unit_test()
    metrics_unit_test()
    storage_unit_test()
    
    experiment_unit_test()
    return



def experiment_unit_test():
    exp_id = "main_unit_test"
    rand_seed = 69
    determ = False
    
    master_config = MasterConfig.default()
    master_config.trainer_config.train_epochs = 5
    experiment = Experiment(
        master_config, 
        experiment_id=exp_id,
        random_seed=rand_seed,
        determinism=determ,
    )
    
    experiment.train_model()
    experiment.compute_metrics()
    experiment.save(f"tmp/{exp_id}")
    experiment.load(f"savefolder/mpkg/tmp/{exp_id}")
    print(
        experiment.experiment_id,
        experiment.random_seed,
        experiment.determinism,
        experiment.experiment_state,
    )
    
    return



if __name__ == "__main__":
    import torch
    print("PyTorch Version:", torch.__version__)
    print("PyTorch CUDA Version:", torch.version.cuda)
    print("Is CUDA available?:", torch.cuda.is_available())
    
    unit_test()
    print("Done!")