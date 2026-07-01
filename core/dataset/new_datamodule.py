from dataclasses import dataclass
import pandas as pd

from torch.utils.data import DataLoader

from .dataset import ExperimentDataset
from .dataset_config import DatasetConfig



@dataclass
class DataModule:
    train_dataloader: DataLoader
    validation_dataloader: DataLoader
    test_dataloader: DataLoader
    
    def unwrap(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        return (
            self.train_dataloader,
            self.validation_dataloader,
            self.test_dataloader,
        )