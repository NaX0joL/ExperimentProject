from dataclasses import dataclass
import pandas as pd

from torch.utils.data import DataLoader

from .registry import DATASET_REGISTRY
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



def get_datamodule(dataset_config:DatasetConfig) -> DataModule:
    raw_data = _get_raw(dataset_config)
    splits = _split_raw(raw_data)
    datasets = _build_dataset(splits)
    data_module = _build_datamodule(
        datasets=datasets,
        batch_size=dataset_config.batch_size, 
        shuffle=dataset_config.shuffle,
        drop_last=dataset_config.drop_last,
    )
    return data_module


def _get_raw(dataset_config:DatasetConfig) -> dict[str, pd.DataFrame]:
    entry = DATASET_REGISTRY[dataset_config.dataset_name]
    raw_getter = entry.raw_getter_class(dataset_config.raw_getter_config)
    raw = raw_getter.get_raw()
    return raw


def _split_raw(raw_data:dict[str, pd.DataFrame]) -> dict[str, dict[str, pd.DataFrame]]:
    splits = {
        split: {
            name: df[(raw_data["split_type"] == split).squeeze()].reset_index(drop=True)
            for name, df in raw_data.items()
        }
        for split in ["train", "validation", "test"]
    }
    return splits


def _build_dataset(split_data: dict[str, dict[str, pd.DataFrame]]) -> dict[str, ExperimentDataset]:
    datasets = {
        split: ExperimentDataset(split_data[split])
        for split in ["train", "validation", "test"]
    }
    return datasets


def _build_datamodule(datasets:dict[str, ExperimentDataset], batch_size:int, shuffle:bool, drop_last:bool):
    data_module = DataModule(
        train_dataloader = _build_dataloader(datasets["train"], batch_size=batch_size, shuffle=shuffle, drop_last=drop_last),
        validation_dataloader = _build_dataloader(datasets["validation"], batch_size=batch_size, shuffle=False, drop_last=drop_last),
        test_dataloader = _build_dataloader(datasets["test"], batch_size=batch_size, shuffle=False, drop_last=drop_last),
    )
    return data_module


def _build_dataloader(dataset:ExperimentDataset, batch_size:int, shuffle:bool, drop_last:bool) -> DataLoader:
        if len(dataset) == 0:
            return None
        
        dataloader = DataLoader(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            drop_last = drop_last,
        )
        return dataloader