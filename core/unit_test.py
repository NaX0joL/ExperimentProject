from .master_config import MasterConfig
from .experiment import Experiment



def unit_test():
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
    unit_test()
    print("Done!")