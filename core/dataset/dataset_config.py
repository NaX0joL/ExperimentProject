from dataclasses import dataclass

from core.abstract import ABSTRACT_RawGetter

from .registry import DATASET_REGISTRY



@dataclass
class DatasetConfig:
    
    batch_size: int
    shuffle: bool
    drop_last: bool
    
    dataset_name: str
    raw_getter_config: ABSTRACT_RawGetter = None
    
    def __post_init__(self) -> None:
        if self.raw_getter_config is None:
            self.raw_getter_config = DATASET_REGISTRY[self.dataset_name].config_class.default()
        return
 
    @classmethod
    def default(cls, **modified_parameter):
        context = {
            "batch_size": 16,
            "shuffle": True,
            "drop_last": False,
            "dataset_name": "UCR_Anomaly_Detection",
        }
        context.update({k: v for k, v in modified_parameter.items() if k in context})
        return cls(
            batch_size = context["batch_size"],
            shuffle = context["shuffle"],
            drop_last = context["drop_last"],
            dataset_name = context["dataset_name"],
            raw_getter_config = DATASET_REGISTRY[context["dataset_name"]].config_class.default(**modified_parameter),
        )