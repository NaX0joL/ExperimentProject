from abc import ABC, abstractmethod
from pathlib import Path



DATA_DIR = Path("data")



class Loader(ABC):
    COLUMNS = ["series_id", "timestep", "value", "label", "split", "group"]
    
    @abstractmethod
    def get_data(*args, **kwargs):
        pass