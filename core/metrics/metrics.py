import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from core.abstract import ABSTRACT_metric
from core.schema import ExperimentState
from core.dataset.datamodule import DataModule
from modules.tensor_normalization import normalize_tensor
from modules.device_resolve import get_model_device

from .metrics_config import MetricsConfig
from .registry import REGRESSION_METRICS, CLASSIFICATION_METRICS



class MetricsCalculator():
    
    def __init__(self, experiment_state:ExperimentState) -> None:
        self.experiment_state = experiment_state
        
        self.metrics_config = experiment_state.master_config.metrics_config
        self.model = experiment_state.model
        self.datamodule = experiment_state.datamodule
        self.metrics_result = experiment_state.metrics_result
        
        self.device = get_model_device(self.model)
        self.used_dataloader = self._get_used_dataloader(self.datamodule)
        self.metric_calculators = self._get_metric_calculators()
        self.metrics_result = {}
        return
    
    ### public functions
    
    def calculate(self) -> None:
        accumulator = Accumulator(
            calculators = self.metric_calculators,
            tolerance = self.metrics_config.tolerance,
        )
        
        self.model.eval()
        with torch.no_grad():
            for batch in self.used_dataloader:
                batch: dict[str, Tensor|list]
                
                input = batch.get("value").to(self.device)
                label = batch.get("ground_truth").to(self.device)
                output = self.model(input)
                
                accumulator.accumulate(output, label)
        
        self.experiment_state.metrics_result = accumulator.calculate()
        return
    
    def get_metrics_result(self) -> dict[str, dict]:
        return self.metrics_result
    
    ### private functions
    
    def _get_used_dataloader(self, datamodule:DataModule) -> DataLoader:
        return datamodule.test_dataloader
    
    def _get_metric_calculators(self) -> list[ABSTRACT_metric]:
        metric_calculators = []
        
        if self.metrics_config.task_type == "regression":
            for calc in REGRESSION_METRICS.values():
                metric_calculators.append(calc())
                
        elif self.metrics_config.task_type == "classification":
            pass
        
        else:
            raise TypeError(f"MetricsCalculator error! invalid task type: {self.metrics_config.task_type}")
        
        return metric_calculators



class Accumulator(ABSTRACT_metric):
    
    def __init__(self, calculators:list[ABSTRACT_metric], tolerance=0) -> None:
        self.calculators = calculators
        self.tolerance = tolerance
        return
    
    ### public functions
    
    def reset(self) -> None:
        for calc in self.calculators:
            calc.reset()
        return
    
    def accumulate(self, output:Tensor, label:Tensor) -> None:
        self._validate_shape(output, label)
        
        output = normalize_tensor(output)
        label = normalize_tensor(label)
        
        label = self._expand_label(label)
        
        for calc in self.calculators:
            calc.accumulate(output, label)
        
        return
    
    def calculate(self) -> dict[str, dict]:
        result = {}
        for calc in self.calculators:
            result[calc.NAME] = calc.calculate()
        return result
    
    ### private functions
    
    def _validate_shape(self, output:Tensor, label:Tensor) -> None:
        assert output.dim() in (1, 2) and label.dim() in (1, 2), (
            f"Accumulator error! invalid dim, expected 1-d or 2-d tensor, output:{output.shape}, label:{label.shape}"
        )
        
        assert output.shape == label.shape, (
            f"Accumulator error! shape mismatch, output:{output.shape}, label:{label.shape}"
        )
        
        # if self._expected_last_dim is not None:
        #     assert output.shape[-1] == self._expected_last_dim, (
        #         f"Last-dim mismatch — got {output.shape[-1]}, expected {self._expected_last_dim}"
        #     )
        
        return
    
    def _expand_label(self, label:Tensor) -> Tensor:
        if self.tolerance == 0:
            return label
 
        expanded = label.clone()
        bs, seq_len = label.shape
 
        for i in range(bs):
            for pos in torch.where(label[i] == 1)[0]:
                start = max(0, pos.item() - self.tolerance)
                end   = min(seq_len, pos.item() + self.tolerance + 1)
                expanded[i, start:end] = 1
 
        return expanded