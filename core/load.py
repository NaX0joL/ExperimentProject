import os
import pickle
from pathlib import Path

import torch
from torch import nn, Tensor


from core.schema import ExperimentContext
from core.master_config import MasterConfig
from core.architectures.new_architecture_control import ArchitectureControl
from core.datasets.new_data_module import DataModuleGetter, DataModule
from core.modules.device_resolve import get_optimal_device, get_model_device



class LoadingService():
    def __init__(self, experiment_context:ExperimentContext):
        self.experiment_context = experiment_context
        return
    
    ### public functions
    
    def load(self, path:Path) -> None:
        self._check_path(path)
        
        config = self._load_config_pkl(path)
        data_module = self._get_data_module(config)
        model = self._load_model(path, config)
        
        self._reinstate_context(
            master_config = config,
            data_module = data_module,
            model = model,
        )
        return
    
    ### private helper functions
    
    def _reinstate_context(self, master_config:MasterConfig, data_module:DataModule, model:nn.Module) -> None:
        self.experiment_context.master_config = master_config
        self.experiment_context.data_module = data_module
        self.experiment_context.model = model
        return
    
    # -- path --
    
    def _check_path(self, path:Path) -> None:
        marker_file_path = path / "__mpkg__.py"
        if os.path.isfile(marker_file_path):
            pass
        return
    
    # -- config --
    
    def _load_config_pkl(self, path:Path) -> None:
        config_pkl_path = path / "config.pkl"
        with open(config_pkl_path, "rb") as file:
            config = pickle.load(file)
        return config
    
    # -- data module --
    
    def _get_data_module(self, config:MasterConfig) -> DataModule:
        data_module = DataModuleGetter(
            config.dataset_config
        ).get_data_module()
        return data_module
    
    # -- model --
    
    def _load_model(self, path:Path, config:MasterConfig) -> nn.Module:
        state = self._load_model_state(path)
        model = self._rebuild_model(state, config)
        return model
        
    def _load_model_state(self, path:Path) -> dict[str, Tensor]:
        model_path = path / "model.pth"
        state = torch.load(model_path, weights_only=True)
        return state
    
    def _rebuild_model(self, state:dict[str, Tensor], config:MasterConfig) -> nn.Module:
        model = self._get_model_base(config)
        self._run_model_once(model, config)     # some model form full weights after the first pass
        model.load_state_dict(state)
        return
    
    def _get_model_base(self, config:MasterConfig) -> nn.Module:
        device = get_optimal_device()
        model = ArchitectureControl(
            config.architecture_config
        ).get_model().to(device)
        return model
    
    def _run_model_once(self, model:nn.Module, config:MasterConfig) -> None:
        data_batch = self._fetch_single_data_batch(config)
        
        model.eval()
        with torch.no_grad():
            device = get_model_device(model)
            input = data_batch["value"].float().to(device)
            model(input)
            
        return
    
    def _fetch_single_data_batch(self, config:MasterConfig) -> dict[str, Tensor|list]:
        data_module = self._get_data_module(config)
        first_batch = next(iter(data_module.train_dataloader))
        return first_batch