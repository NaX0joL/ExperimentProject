from dataclasses import dataclass

from torch import nn

from core.abstract import ABSTRACT_Config

from .architectures.proposed_model.model import Model as ProposedModel, ModelConfig as ProposedModelConfig
from .architectures.simple_cnn.model import Model as SimpleCNN, ModelConfig as SimpleCNNConfig
from .architectures.traditional_matrix_profile.model import Model as TraditionalMatrixProfile, ModelConfig as TraditionalMatrixProfileConfig



@dataclass
class ArchitectureEntry:
    model_class: type[nn.Module]
    config_class: type[ABSTRACT_Config]



ARCHITECTURE_REGISTRY: dict[str, ArchitectureEntry] = {
    "proposed_model": ArchitectureEntry(ProposedModel, ProposedModelConfig),
    "simple_cnn": ArchitectureEntry(SimpleCNN, SimpleCNNConfig),
    "traditional_matrix_profile": ArchitectureEntry(TraditionalMatrixProfile, TraditionalMatrixProfileConfig),
}