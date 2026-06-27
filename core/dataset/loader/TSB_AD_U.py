from concurrent.futures import ThreadPoolExecutor

from pathlib import Path
import pandas as pd

from ..schema import Loader, DATA_DIR



class TSB_AD_U_Loader(Loader):
    NAME = "TSB_AD_U"
    PATH = DATA_DIR / NAME
    
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
        series_id = cls._make_series_id(index=index)
        time_series, labels = cls._load_time_series_and_label(file_path)
        
        rows = {
            "series_id": series_id,
            "timestep": range(len(time_series)),
            "value": time_series,
            "label": labels,
            "split": None,
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