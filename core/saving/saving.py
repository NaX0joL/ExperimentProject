from pathlib import Path

from core.schema import ExperimentState

from .schema import MPKGState
from .save import SaveService
from .load import LoadService



class SaverLoader():
    
    def __init__(self, experiment_state:ExperimentState) -> None:
        self.experiment_state = experiment_state
        return
    
    def save(self, path:Path=None):
        save_service = SaveService(
            experiment_state = self.experiment_state,
            init_content = getattr(self, "init_content", None)
        )
        save_service.save(path)
        return
    
    def load(self, path:Path):
        mpkg_state = MPKGState.init_empty()
        load_service = LoadService(mpkg_state)
        pass
    
    def put_in_init(self, **kwargs):
        self.init_content = kwargs
        return