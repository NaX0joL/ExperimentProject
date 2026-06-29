from concurrent.futures import ThreadPoolExecutor

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

from ...schema import Loader, DATA_DIR



class TSB_AD_U_Loader(Loader):
    NAME = "TSB_AD_U"
    PATH = DATA_DIR / NAME / "source_data"
    COLUMNS = [
        "series_id",
        "timestep",
        "value",
        "label",
        "split",
        "source",
        "domain",
    ]
    
    DEFAULT_GROUPING_STRATUM = "source_dataset"
    
    @classmethod
    def load_data(cls, parallelized:bool=False) -> pd.DataFrame:
        indexes_and_file_paths = [
            (index, file_path)
            for index, file_path in enumerate(sorted(cls.PATH.iterdir()), start=1)
            if file_path.is_file()
        ]
        
        if parallelized:
            with ThreadPoolExecutor() as executor:
                all_files_rows_dfs = list(executor.map(
                    lambda args: cls._make_single_file_rows(*args), indexes_and_file_paths  
                ))
        else:
            all_files_rows_dfs = [
                cls._make_single_file_rows(*args)
                for args in indexes_and_file_paths
            ]
        return pd.concat(all_files_rows_dfs, ignore_index=True)
    
    @classmethod
    def _make_single_file_rows(cls, index:int, file_path:Path) -> pd.DataFrame:
        #print(file_path.name)
        
        series_id = cls._make_series_id(index=index)
        time_series, labels = cls._load_time_series_and_label(file_path)
        
        n = len(time_series)
        filename = TSB_AD_U_Filename.parse(file_path.name)
        splits = filename.make_splits(size=n)
        
        rows = {
            "series_id": series_id,
            "timestep": range(n),
            "value": time_series,
            "label": labels,
            "split": splits,
            "source": filename.source_dataset,
            "domain": filename.domain,
        }
        return pd.DataFrame(rows, columns=cls.COLUMNS)
    
    @classmethod
    def _make_series_id(cls, index:int) -> str:
        series_id = f"{cls.NAME}_{index}"
        return series_id
    
    @staticmethod
    def _load_time_series_and_label(file_path:Path) -> tuple[list[float], list[int]]:
        content = pd.read_csv(file_path, header=None, skiprows=1)
        feature = content[0].reset_index(drop=True).squeeze().to_list()
        label = content[1].reset_index(drop=True).squeeze().to_list()
        return feature, label



@dataclass
class TSB_AD_U_Filename:
    data_number: str
    source_dataset: str
    domain: str
    train_interval: tuple[int, int]
    first_anomaly_index: int
    
    @classmethod
    def parse(cls, filename:str) -> "TSB_AD_U_Filename":
        format_removed = filename.split(".")[0]
        parts = format_removed.split("_")
        
        instance = cls(
            data_number = parts[0],
            source_dataset = parts[1],
            domain = parts[4],
            train_interval = (0, int(parts[6]) - 1),
            first_anomaly_index = int(parts[8]) - 1,
        )
        return instance
    
    def make_splits(self, size:int) -> np.ndarray:
        splits = np.full(size, "test", dtype=object)
        start, end = self.train_interval
        splits[start:end + 1] = "train"
        return splits
    
    def extract_group(self, stratum:str) -> str:
        if hasattr(self, stratum):
            group = getattr(self, stratum)
        else:
            raise KeyError(f"TSB_AD_U_Filename Error! invalid stratum: {stratum}")
        return group