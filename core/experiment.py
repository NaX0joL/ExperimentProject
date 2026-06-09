import os
import random
from datetime import datetime

import numpy as np

import torch
from torch import nn, Tensor

from .master_config import MasterConfig
from .schema import ExperimentStateFactory



class Experiment():
    
    def __init__(self, master_config:MasterConfig, experiment_id:str=None, random_seed:int=42, determinism:bool=False) -> None:
        self.master_config = master_config
        
        self._resolve_experiment_id(experiment_id)
        self._set_random_seed(random_seed)
        self._switch_determinism(determinism)
        
        self._reload_experiment_state()
        return
    
    ### public functions
    
    def save(self) -> None:
        pass
    
    def load(self) -> None:
        pass
    
    def train_model(self) -> nn.Module:
        pass
    
    def get_model_metrics(self) -> None:
        pass
     
    def model_inference(self) -> Tensor:
        pass
    
    ### private functions
    
    def _resolve_experiment_id(self, experiment_id:str) -> None:
        if experiment_id is None:
            current_time = datetime.now().strftime("%Y_%m_%d-%H%M%S")
            experiment_id = f"tmp_exp-{current_time}"
        self.experiment_id = experiment_id
        return
    
    def _set_random_seed(self, random_seed:int) -> None:
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        return
    
    def _switch_determinism(self, determinism:bool) -> None:
        torch.backends.cudnn.deterministic = determinism
        torch.backends.cudnn.benchmark = not determinism
        torch.use_deterministic_algorithms(determinism, warn_only=True)
        return
    
    def _reload_experiment_state(self) -> None:
        self.experiment_state = ExperimentStateFactory().create(self.master_config)
        return