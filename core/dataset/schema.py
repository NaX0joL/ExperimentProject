from abc import ABC, abstractmethod
from pathlib import Path



DATA_DIR = Path("data")



class Loader(ABC):
    
    COLUMNS = [
        "series_id", 
        "timestep", 
        "value", 
        "label", 
        "split", 
        "group",
    ]
    
    @abstractmethod
    def load_data(*args, **kwargs):
        pass



class Segmenter(ABC):
    
    @abstractmethod
    def segment_data(*args, **kwargs):
        pass



class Transformer(ABC):
    
    @abstractmethod
    def transform_data(*args, **kwargs):
        pass