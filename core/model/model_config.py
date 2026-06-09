from dataclasses import dataclass



@dataclass
class ModelConfig:
    @classmethod
    def default():
        return