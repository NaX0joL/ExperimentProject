import pandas as pd



class PreFilter():
    
    def __init__(self, filter_columns:dict=None) -> None:
        self.filter_columns = filter_columns if filter_columns is not None else {}
        return
    
    def apply(self, data:pd.DataFrame) -> pd.DataFrame:
        filtered = self._filter_by_column(data)
        return filtered
    
    def _filter_by_column(self, data:pd.DataFrame) -> pd.DataFrame:
        if not self.filter_columns:
            return data
        
        masks = [
            data[col] == val 
            for col, val in self.filter_columns.items() 
            if col in data.columns
        ]
        
        if not masks:
            return data

        final_mask = pd.concat(masks, axis=1).any(axis=1)
        filtered = data.loc[final_mask]
        return filtered