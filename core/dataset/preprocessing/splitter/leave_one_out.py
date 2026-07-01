from collections.abc import Generator
import pandas as pd

from ..schema import Splitter



class LeaveOneOut(Splitter):
    
    def __init__(self, data:pd.DataFrame) -> None:
        self.data = data
        return
    
    def split_data(self, n_fold:int=1) -> Generator[tuple[pd.DataFrame, pd.DataFrame]]:
        groups = self._get_unique_groups(self.data)
        
        for index, group in enumerate(groups):
            if index >= n_fold:
                break
            
            train_fold = self.data[self.data["group"] == group].reset_index(drop=True)
            test_fold = self.data[self.data["group"] != group].reset_index(drop=True)
            
            yield train_fold, test_fold
        
        return
    
    def _get_unique_groups(self, data:pd.DataFrame):
        return data["group"].unique()