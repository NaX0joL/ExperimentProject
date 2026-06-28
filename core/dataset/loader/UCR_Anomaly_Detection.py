from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
import re

from ..schema import Loader, DATA_DIR



class UCR_Anomaly_Detection_Loader(Loader):
    NAME = "UCR_Anomaly_Detection"
    PATH = DATA_DIR / NAME
    
    SPACE_DELIMITED_INDEX = {
        204, 205, 206, 207, 208, 225, 226, 242, 243     # for some reason, files with these numbers has multiple spaces delimiter instead of \n
    }
    
    @classmethod
    def get_data(cls, parallelized:bool=False) -> pd.DataFrame:
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
        # print(file_path.name)
        
        series_id = cls._make_series_id(index=index)
        time_series = cls._load_time_series(index=index, file_path=file_path)
        
        n = len(time_series)
        filename = UCR_Anomaly_Detection_Filename.parse(file_path.name)
        labels = filename.make_labels(size=n)
        splits = filename.make_splits(size=n)
        group = filename.extract_group()
        
        rows = {
            "series_id": series_id,
            "timestep": range(n),
            "value": time_series,
            "label": labels,
            "split": splits,
            "group": group,
        }
        return pd.DataFrame(rows, columns=cls.COLUMNS)
    
    @classmethod
    def _make_series_id(cls, index:int) -> str:
        series_id = f"{cls.NAME}_{index}"
        return series_id
    
    @classmethod
    def _load_time_series(cls, index:int, file_path:Path) -> list[float]:
        if index in cls.SPACE_DELIMITED_INDEX:
            time_series = cls._load_space_delimited_time_series(file_path)
        else:
            time_series = cls._load_newline_delimited_time_series(file_path)
        return time_series
    
    @staticmethod
    def _load_space_delimited_time_series(file_path:Path) -> list[float]:
        content = file_path.read_text(encoding="utf-8")
        raw_strings = content.split()
        feature = [float(val) for val in raw_strings]
        return feature
    
    @staticmethod
    def _load_newline_delimited_time_series(file_path:Path) -> list[float]:
        content = pd.read_csv(file_path, header=None, usecols=[0])
        feature = content.iloc[:, 0].to_list()
        return feature



@dataclass
class UCR_Anomaly_Detection_Filename:
    data_number: str
    mnemonic_name: str
    train_interval: tuple[int, int]
    anomaly_interval: tuple[int, int]
    
    @classmethod
    def parse(cls, filename:str) -> "UCR_Anomaly_Detection_Filename":
        format_removed = filename.split(".")[0]
        parts = format_removed.split("_")
        
        instance = cls(
            data_number = parts[0],
            mnemonic_name = parts[3],
            train_interval = (0, int(parts[4]) - 1),
            anomaly_interval = (int(parts[5]) - 1, int(parts[6]) - 1),
        )
        return instance
    
    def make_labels(self, size:int) -> np.ndarray:
        labels = np.zeros(size, dtype=np.int8)
        start, end = self.anomaly_interval
        labels[start:end + 1] = 1
        return labels
    
    def make_splits(self, size:int) -> np.ndarray:
        splits = np.full(size, "test", dtype=object)
        start, end = self.train_interval
        splits[start:end + 1] = "train"
        return splits
    
    def extract_group(self) -> str:
        # manual grouping through keywords
        KEYWORD_GROUP_MAP = {
            # multi-file groups (specific first)
            "gaitHunt":     "gaitHunt",             # before "gait"
            "gait":         "gait",
            "tiltAPB":      "tiltAPB",              # before "tilt"
            "tilt12":       "tilttable",            # tilt12744, tilt12754, tilt12755
            "Tkeep":        "TkeepMARS",            # TkeepFirst/Second/Third/Forth/Fifth
            "ltstdbs":      "ltstdbs",              # ltstdbs30791AI/AS/ES
            "qtdbSel":      "qtdbSel",              # qtdbSel1005V, qtdbSel100MLII
            "s20101":       "s20101",               # s20101m, s20101mML
            "CHARIS":       "CHARIS",               # CHARISfive, CHARISten
            "mit":          "mitlongtermecg",       # mit14046, mit14134, mit14157

            # singletons - included for explicitness
            "sddb":                     "sddb",
            "BIDMC":                    "BIDMC",
            "CIMIS44AirTemperature":    "AirTemperature",
            "ECG":                      "ECG",
            "GP711MarkerLFM5z":         "Marker",
            "InternalBleeding":         "InternalBleeding",
            "Lab2Cmac":                 "Lab2Cmac",
            "MesoplodonDensirostris":   "Mesoplodon",
            "PowerDemand":              "PowerDemand",
            "WalkingAceleration":       "WalkingAceleration",
            "apneaecg":                 "apneaecg",
            "insectEPG":                "insectEPG",
            "park3m":                   "park3m",
            "resperation":              "resperation",
            "sel840mECG":               "sel840mECG",
            "Fantasia":                 "Fantasia",
            "Italianpowerdemand":       "Italianpowerdemand",
            "STAFFIIIDatabase":         "STAFFIIIdb",
            "taichidbS0715Master":      "taichidb",
            "weallwalk":                "weallwalk",
        }
        for keyword, group in KEYWORD_GROUP_MAP.items():
            if keyword in self.mnemonic_name:
                group = group
                break
        
        return group