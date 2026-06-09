from dataclasses import dataclass

from core.abstract import ABSTRACT_RawGetter

from .REGISTRY import DATASET_REGISTRY



@dataclass
class DatasetConfig:
    
    batch_size: int
    shuffle: bool
    drop_last: bool
    
    dataset_name: str
    raw_getter_config: ABSTRACT_RawGetter = None
    
    def __post_init__(self) -> None:
        if self.raw_getter_config is None:
            self.raw_getter_config = DATASET_REGISTRY[self.dataset_name].config_class
        return
 
    @classmethod
    def default(self):
        dataset_name = "UCR_Classification"
        return self(
            batch_size = 16,
            shuffle = True,
            drop_last = False,
            dataset_name = dataset_name,
            raw_getter_config = DATASET_REGISTRY[dataset_name].config_class,
        )