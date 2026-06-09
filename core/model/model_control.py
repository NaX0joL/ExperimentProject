from dataclasses import dataclass

from torch import nn

from core.abstract import ABSTRACT_ModelControl

from .registry import ARCHITECTURE_REGISTRY
from .model_config import ModelConfig



class ModelControl(ABSTRACT_ModelControl):
    def __init__(self, architecture_config: ModelConfig):
        self.architecture_config = architecture_config
        return

    def get_model(self) -> nn.Module:
        entry = ARCHITECTURE_REGISTRY[self.architecture_config.architecture_name]
        model = entry.model_class(self.architecture_config.architecture_config)
        return model