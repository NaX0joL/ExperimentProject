from dataclasses import dataclass

from core.abstract import ABSTRACT_Config

from .registry import ARCHITECTURE_REGISTRY



@dataclass
class ModelConfig(ABSTRACT_Config):
    architecture_name: str
    architecture_config: ABSTRACT_Config = None

    def __post_init__(self):
        if self.architecture_config is None:
            self.architecture_config = ARCHITECTURE_REGISTRY[self.architecture_name].config_class.default()
        return

    @classmethod
    def default(self):
        name = "proposed_model"

        return self(
            architecture_name = name,
            architecture_config = ARCHITECTURE_REGISTRY[name].config_class.default(),
        )