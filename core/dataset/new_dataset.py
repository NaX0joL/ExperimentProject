import pandas as pd
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset



class ExperimentDataset(Dataset):
    FEATURE = [
        "value",
        "ground_truth",
        "codename",     # add later to the pipeline
    ]
    
    def __init__(self, data:pd.DataFrame) -> None:
        self.value = np.array(data["value"].tolist())
        self.ground_truth = np.array(data["label"].tolist())
        return
        
    def __len__(self) -> int:
        return self.value.shape[0]
    
    def __getitem__(self, index:int) -> tuple[Tensor, Tensor]:
        value = torch.tensor(self.value[index], dtype=torch.float32)
        ground_truth = torch.tensor(self.ground_truth[index], dtype=torch.float32)
        return {
            "value": value,
            "ground_truth": ground_truth,
        }



class ExperimentDatasetFactory:
    
    @staticmethod
    def create(data:pd.DataFrame) -> ExperimentDataset:
        if data is not None:
            dataset = ExperimentDataset(data)
        else:
            dataset = data
        return dataset