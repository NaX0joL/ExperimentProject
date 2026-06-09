import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

import torch
from torch import Tensor
from torch.utils.data import Dataset



class ExperimentDataset(Dataset):
    def __init__(self, raw:dict[str, pd.DataFrame]) -> None:
        self.raw = {
            name: turn_dataframe_into_list_or_tensor(raw)
            for name, raw in raw.items()
        }
        return
    
    def __len__(self) -> int:
        first_item = next(iter(self.raw.values()))
        if isinstance(first_item, list):
            length = len(first_item)
        elif isinstance(first_item, Tensor):
            length = first_item.shape[0]
        else:
            raise TypeError(
                f"Dataset error, invalid raw type! type found:{type(first_item)}"
            )
        return length
    
    def __getitem__(self, index) -> dict[str, list|Tensor]:
        return {
            name: raw[index] for name, raw in self.raw.items()
        }



def turn_dataframe_into_list_or_tensor(df:pd.DataFrame) -> list|Tensor:
    if is_numeric_dtype(df.values.dtype):
        return torch.from_numpy(df.values.copy()).float()

    elif is_string_dtype(df.iloc[:, 0]):
        return df.iloc[:, 0].to_list()
    
    else:
        raise TypeError(
            f"Dataset error, invalid dataframe dtype! dtype found: {df.dtypes.unique()}"
        )