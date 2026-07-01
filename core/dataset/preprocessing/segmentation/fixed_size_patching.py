import numpy as np
import pandas as pd

from ..schema import Segmenter



POSITIVE_ANOMALY_LABEL = 1



class FixedSizePatching(Segmenter):
    
    def __init__(self, patch_size:int) -> None:
        self.patch_size = patch_size
        return
    
    def segment_data(self, data:pd.DataFrame) -> pd.DataFrame:
        all_segments = []
        
        for (series_id, split, group), series in data.groupby(["series_id", "split", "group"]):
            segments = self._segment_series(series, series_id, split, group)
            if segments is not None:
                all_segments.append(segments)
                
        if not all_segments:
            raise ValueError(
                "FixedSizePatching error! empty data after segmentation.\n"
                "Segment results were either shorter than patch_size or contained no anomalies."
            )
        
        segmented = pd.concat(all_segments, ignore_index=True)
        return segmented
    
    def _segment_series(self, series:pd.DataFrame, series_id:str, split:str, group:str) -> pd.DataFrame:
        values, labels = self._extract_arrays(series)
        
        if len(values) < self.patch_size:
            return None
        
        anomaly_positions = np.where(labels == POSITIVE_ANOMALY_LABEL)[0]
        if len(anomaly_positions) == 0:
            return None

        patch_indices = self._calculate_valid_patch_indices(anomaly_positions, len(values))
        
        if len(patch_indices) == 0:
            return None
    
        value_windows = self._build_windows(values, patch_indices)
        label_windows = self._build_windows(labels, patch_indices)
        
        segments = {
            "series_id": series_id,
            "value":     list(value_windows),
            "label":     list(label_windows),
            "split":     split,
            "group":     group,
        }
        return pd.DataFrame(segments)
    
    def _extract_arrays(self, series:pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        values = series["value"].to_numpy()
        labels = series["label"].to_numpy()
        return values, labels
    
    def _calculate_valid_patch_indices(self, anomaly_positions:np.ndarray, series_size:int) -> np.ndarray:
        patch_indices = np.unique(anomaly_positions // self.patch_size)
        max_complete_patch_idx = series_size // self.patch_size
        patch_indices = patch_indices[patch_indices < max_complete_patch_idx]
        return patch_indices
    
    def _build_windows(self, array:np.ndarray, patch_indices:np.ndarray) -> np.ndarray:
        windows = np.stack([array[i*self.patch_size : (i+1)*self.patch_size] for i in patch_indices])
        return windows