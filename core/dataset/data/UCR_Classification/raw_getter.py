from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from core.abstract import ABSTRACT_RawGetter, ABSTRACT_Config



CURR_DIR = Path(__file__).resolve().parent



@dataclass
class RawGetterConfig(ABSTRACT_Config):
    variant_name: str|None
    sub_dataset_name: str|list[str]
    task_type: str = "classification"
    
    @classmethod
    def default(self) -> RawGetterConfig:
        return self(
            variant_name=None,
            sub_dataset_name="FordA",
        )



class RawGetter(ABSTRACT_RawGetter):
    DATASET_NAME = "UCR_Classification"
    TASK_TYPE = "classification"
    ATTRIBUTES = [
        "value", 
        "ground_truth", 
        "codename", 
        "split_type",
    ]
    
    def __init__(self, raw_getter_config:RawGetterConfig) -> None:
        self.raw_getter_config = raw_getter_config
        return
    
    def get_raw(self) -> dict[str, pd.DataFrame]:
        path_resolver = PathResolver(CURR_DIR, self.raw_getter_config.variant_name)
        fetch_dir = path_resolver.resolve_fetch_directory()
        
        loader = RawLoader(fetch_dir, self.ATTRIBUTES)
        raw_data = loader.load(self.raw_getter_config.sub_dataset_name)
        
        normalized_raw_data = RawNormalizer.normalize(raw_data)
        return normalized_raw_data



class PathResolver():
    
    def __init__(self, base_dir:Path, variant_name:str|None) -> None:
        self.base_dir = base_dir
        self.variant_name = variant_name
        return

    def resolve_fetch_directory(self) -> Path:
        if self.variant_name is not None:
            path = self.base_dir / "raw" / "variants" / self.variant_name
        else:
            path = self.base_dir / "raw" / "original"
        return path



class RawLoader():
    
    def __init__(self, fetch_dir:Path, attribute_names:list[str]) -> None:
        self.fetch_dir = fetch_dir
        self.attribute_names = attribute_names
        return

    def load(self, sub_datasets:str|list[str]) -> dict[str, pd.DataFrame]:
        if isinstance(sub_datasets, str):
            raw = self._load_single_sub_dataset(sub_datasets)
        elif isinstance(sub_datasets, list):
            raw = self._load_multiple_sub_datasets(sub_datasets)
        else:
            raise TypeError(f"RawLoader error! invalid sub_datasets type: {type(sub_datasets)}")
        return raw

    def _load_single_sub_dataset(self, dataset_name:str) -> dict[str, pd.DataFrame]:
        dataset_path = self.fetch_dir / dataset_name
        
        attributes = {
            name: pd.read_csv(dataset_path / f"{name}.csv", header=None)
            for name in self.attribute_names
        }
        return attributes

    def _load_multiple_sub_datasets(self, dataset_names:list[str]) -> dict[str, pd.DataFrame]:
        collected_data = {name: [] for name in self.attribute_names}
        
        for name in dataset_names:
            single_sub_dataset = self._load_single_sub_dataset(name)
            for attribute in self.attribute_names:
                collected_data[attribute].append(single_sub_dataset[attribute])
        
        attributes = {
            attr: pd.concat(collected_data[attr], ignore_index=True) 
            for attr in self.attribute_names
        }
        return attributes



class RawNormalizer():
    
    @classmethod
    def normalize(self, raw:dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        raw = self._normalize_label_class(raw)
        raw = self._normalize_numerical_value(raw)
        return raw

    @classmethod
    def _normalize_label_class(self, raw:dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        raw["ground_truth"].iloc[:, 0] = raw["ground_truth"].iloc[:, 0].astype("category").cat.codes
        return raw
    
    @classmethod
    def _normalize_numerical_value(self, raw:dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        
        def _normalize_per_row(attribute:pd.DataFrame) -> pd.DataFrame:
            row_min = attribute.min(axis=1)
            row_max = attribute.max(axis=1)
            row_range = (row_max - row_min).replace(0, 1)
            normed_row = attribute.sub(row_min, axis=0).div(row_range, axis=0)
            return normed_row

        raw = {
            name: _normalize_per_row(attribute) if pd.api.types.is_numeric_dtype(attribute.dtypes) else attribute
            for name, attribute in raw.items()
        }
        return raw