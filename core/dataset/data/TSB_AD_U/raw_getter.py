from __future__ import annotations
from dataclasses import dataclass, fields
import json
from pathlib import Path
import pandas as pd

from core.abstract import ABSTRACT_RawGetter, ABSTRACT_Config



CURR_DIR = Path(__file__).resolve().parent



@dataclass
class RawGetterConfig(ABSTRACT_Config):
    variant_name: str|None
    grouping_file: str
    group: str
    task_type: str = "regression"
    
    @classmethod
    def default(self) -> RawGetterConfig:
        return self(
            variant_name="size-200",
            grouping_file="random_grouping",
            group="random_group_1",
        )



class RawGetter(ABSTRACT_RawGetter):
    DATASET_NAME = "TSB_AD_U"
    TASK_TYPE = "regression"
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
        raw_data = loader.load()
        
        grouper = RawGrouper(fetch_dir, self.raw_getter_config.grouping_file, self.raw_getter_config.group)
        raw_data = grouper.regroup(raw_data)
        
        normalized_raw_data = RawNormalizer.normalize(raw_data)
        return normalized_raw_data



class PathResolver:
    
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



class RawLoader:
    
    def __init__(self, fetch_dir:Path, attribute_names:list[str]) -> None:
        self.fetch_dir = fetch_dir
        self.attribute_names = attribute_names
        return

    def load(self) -> dict[str, pd.DataFrame]:
        attributes = {
            name: pd.read_csv(self.fetch_dir / f"{name}.csv", header=None)
            for name in self.attribute_names
        }
        return attributes



class RawGrouper:
    
    def __init__(self, fetch_dir:Path, grouping_filename:str, group:str) -> None:
        self.fetch_dir = fetch_dir
        self.grouping_filename = grouping_filename
        self.group = group
        return
    
    def regroup(self, raw:dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        grouping = self._get_grouping(self.fetch_dir)
        raw = self._update_split_type_based_on_grouping(raw, grouping)
        return raw
    
    def _get_grouping(self, fetch_dir:Path) -> dict[str, list[str]]:
        grouping_path = fetch_dir / "grouping" / self.grouping_filename
        
        with open(f"{grouping_path}.json", "r") as file:
            grouping = json.load(file)
            
        return grouping
    
    def _update_split_type_based_on_grouping(self, raw:dict[str, pd.DataFrame], grouping:dict[str, list[str]]) -> dict[str, pd.DataFrame]:
        def _label_row(codename: str) -> str:
            return "test" if codename in grouping[self.group] else "train"
        
        raw["split_type"] = raw["codename"][0].apply(_label_row).to_frame().reset_index(drop=True)
        return raw



class RawNormalizer:
    
    @classmethod
    def normalize(self, raw:dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        def _normalize_per_row(attribute: pd.DataFrame) -> pd.DataFrame:
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