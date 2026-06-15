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
    def default(cls, **modified_parameter):
        context = {
            "architecture_name": "proposed_model"
        }
        context.update({k: v for k, v in modified_parameter.items() if k in context})
        model_config = cls(
            architecture_name = context["architecture_name"],
            architecture_config = ARCHITECTURE_REGISTRY[context["architecture_name"]].config_class.default(**modified_parameter),
        )
        return model_config