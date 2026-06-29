import numpy as np
import pandas as pd

from ...schema import Segmenter



POSITIVE_ANOMALY_LABEL = 1



class CutAroundAnomaly(Segmenter):
    
    def __init__(self, window_size:int, stride:int=1) -> None:
        self.window_size = window_size
        self.stride = stride
        return
    
    def segment_data(self, data:pd.DataFrame) -> pd.DataFrame:
        all_segments = []
        
        for (series_id, split, group), series in data.groupby(["series_id", "split", "group"]):
            segments = self._segment_series(series, series_id, split, group)
            if segments is not None:
                all_segments.append(segments)
        
        if not all_segments:
            raise ValueError(
                "CutAroundAnomaly error! empty data after segmentation.\n"
                "Segment results were either too small or containeed no anomalies."
            )
        
        segmented = pd.concat(all_segments, ignore_index=True)
        return segmented
    
    def _segment_series(self, series:pd.DataFrame, series_id:str, split:str, group:str) -> pd.DataFrame:
        values, labels = self._extract_arrays(series)
        
        if len(values) < self.window_size:
            return None
        
        anomaly_positions = np.where(labels == POSITIVE_ANOMALY_LABEL)[0]
        if len(anomaly_positions) == 0:
            return None
        
        valid_starts = self._compute_valid_starts(anomaly_positions, len(values))
        value_windows = self._build_windows(values, valid_starts)
        label_windows = self._build_windows(labels, valid_starts)
        
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
    
    def _compute_valid_starts(self, anomaly_positions:np.ndarray, series_length:int) -> list[int]:
        valid_starts = set()
        
        for pos in anomaly_positions:
            start = max(0, pos - self.window_size + 1)
            end   = min(series_length - self.window_size, pos) + 1
            
            valid_starts.update(range(start, end, self.stride))
        
        valid_starts_list = sorted(valid_starts)
        return valid_starts_list
    
    def _build_windows(self, array:np.ndarray, valid_starts:list[int]) -> np.ndarray:
        windows = np.stack([array[s : s + self.window_size] for s in valid_starts])
        return windows