import os
import json
import pickle
from pathlib import Path

import torch
from torch import nn

from modules.device_resolve import get_model_device
from core.schema import ExperimentState, ExperimentStateFactory
from core.master_config import MasterConfig
from core.dataset.datamodule import DataModule



class LoadService():
    
    def __init__(self) -> None:
        return
    
    def get_marker_content(self) -> dict[str, any]:
        return self.marker_content
    
    def load(self, path:Path, verbose:bool=True) -> ExperimentState:
        path = self._validate_path(path)
        
        if verbose:
            print(f"loading mpkg from {path}")
        
        self.marker_content = marker_file.load(path)
        self.master_config = config_file.load(path)
        self._reinstate_experiment_state()
        self.experiment_state.model = model_state_file.load(path, self.experiment_state)
        self.experiment_state.metrics_result = metrics_file.load(path, self.experiment_state)
        
        if verbose:
            print(f"mpkg loaded!")
        return self.experiment_state
    
    def _validate_path(self, path:Path) -> Path:
        validation_path = Path(path) / "__mpkg__.py"
        if not os.path.isfile(validation_path):
            raise FileNotFoundError(f"LoadService error! given path is not an mpkg: {path}")
        return Path(path)
    
    def _reinstate_experiment_state(self) -> None:
        self.experiment_state = ExperimentStateFactory().create(self.master_config)
        return



class marker_file:
    
    @staticmethod
    def load(path:Path) -> dict[str, any]:
        marker_path = path / "__mpkg__.py"
        
        marker_content = {}
        with open(marker_path, "r") as file:
            for line in file:
                
                if not line.strip():
                    continue
                
                key, val = line.strip().split("=")
                val = marker_file._process_val(val)
                
                marker_content[key] = val
                
        return marker_content
    
    @staticmethod
    def _process_val(val:str) -> bool:
        val = val.strip("'\"")
        
        if val == "True": 
            val = True
        elif val == "False": 
            val = False
        elif val.isdigit(): 
            val = int(val)
        return val



class config_file:
    
    @staticmethod
    def load(path:Path) -> MasterConfig:
        config_path = path / "config.pkl"
        with open(config_path, "rb") as file:
            config = pickle.load(file)
        return config



class model_state_file:
    
    @staticmethod
    def load(path:Path, experiment_state:ExperimentState) -> nn.Module:
        model_state_file._run_model_once(experiment_state.model, experiment_state.datamodule)
        
        model_state_path = path / "model.pth"
        state = torch.load(model_state_path, weights_only=True)
        experiment_state.model.load_state_dict(state)
        
        return experiment_state.model
    
    @staticmethod
    def _run_model_once(model:nn.Module, datamodule:DataModule) -> None:
        batch = next(iter(datamodule.train_dataloader))
        
        model.eval()
        with torch.no_grad():
            device = get_model_device(model)
            input = batch["value"].float().to(device)
            model(input)

        return



class metrics_file:
    
    @staticmethod
    def load(path:Path, experiment_state:ExperimentState) -> dict[str, dict]:
        metrics_path = path / "metrics.json"
        with open(metrics_path, "r") as file:
            metrics_result = json.load(file)
        return metrics_result