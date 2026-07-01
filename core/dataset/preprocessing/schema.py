from abc import ABC, abstractmethod
from pathlib import Path



DATA_DIR = Path("data")



class Loader(ABC):
    
    @abstractmethod
    def load_data(*args, **kwargs):
        pass



class Filter(ABC):
    
    @abstractmethod
    def filter_data(*args, **kwargs):
        pass



class Grouper(ABC):
    
    @abstractmethod
    def group_data(*args, **kwargs):
        pass



class Regrouper(ABC):
    
    @abstractmethod
    def regroup_data(*args, **kwargs):
        pass



class Segmenter(ABC):
    
    @abstractmethod
    def segment_data(*args, **kwargs):
        pass



class Transformer(ABC):
    
    @abstractmethod
    def transform_data(*args, **kwargs):
        pass



class Splitter(ABC):
    
    @abstractmethod
    def split_data(*args, **kwargs):
        pass