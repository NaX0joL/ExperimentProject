import pandas as pd
import numpy as np

from ...schema import Transformer



class Normalization(Transformer):
    
    def __init__(self, transformed_columns:list[str]=["value"]) -> None:
        self.transformed_columns = transformed_columns
        return
    
    def transform_data(self, data_segments:pd.DataFrame) -> pd.DataFrame:
        for column in self.transformed_columns:
            data_segments[column] = self._normalize_column(data_segments[column])
        return
    
    # min-max normalization
    def _normalize_column(self, segment_column:pd.Series[list], eps:float=1e-8) -> None:
        matrix = np.array(segment_column.tolist())
        
        row_mins = matrix.min(axis=1, keepdims=True)
        row_maxs = matrix.max(axis=1, keepdims=True)
        
        normed_matrix = (matrix - row_mins) / (row_maxs - row_mins + eps)
        return normed_matrix