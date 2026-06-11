import os
import random
from pathlib import Path
from datetime import datetime

import numpy as np

import torch
from torch import nn, Tensor

from modules.device_resolve import get_model_device
from core.trainer.trainer import Trainer
from core.metrics.metrics import MetricsCalculator
from core.saving.saving import SaverLoader

from .master_config import MasterConfig
from .schema import ExperimentStateFactory



class Experiment():
    
    def __init__(self, master_config:MasterConfig, experiment_id:str=None, random_seed:int=42, determinism:bool=False) -> None:
        self.master_config = master_config
        self.experiment_id = experiment_id
        self.random_seed = random_seed
        self.determinism = determinism
        
        self._refresh_basic_settings()
        self._reload_experiment_state()
        return
    
    ### public functions
    
    def save(self, path:Path=None) -> None:
        saver = SaverLoader(self.experiment_state)
        saver.put_in_init(
            experiment_id = self.experiment_id,
            random_seed = self.random_seed,
            determinism = self.determinism,
        )
        saver.save(path)
        return
    
    def load(self, path:Path) -> None:
        return
    
    def train_model(self) -> nn.Module:
        trainer = Trainer(self.experiment_state)
        trainer.fit()
        return trainer.get_model()
    
    def compute_metrics(self) -> dict[str, dict]:
        metrics = MetricsCalculator(self.experiment_state)
        metrics.calculate()
        return metrics.get_metrics_result()
     
    def model_inference(self, input:Tensor) -> Tensor:
        model = self.experiment_state.model
        
        model.eval()
        with torch.no_grad():
            input = input.float().to(get_model_device(model))
            output = model(input)
            
        return output
    
    ### private functions
    
    def _refresh_basic_settings(self) -> None:
        self._resolve_experiment_id(self.experiment_id)
        self._set_random_seed(self.random_seed)
        self._switch_determinism(self.determinism)
        return
    
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