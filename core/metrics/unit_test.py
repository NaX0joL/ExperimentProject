from .metrics import MetricsCalculator

from core.master_config import MasterConfig
from core.model.model_control import get_model
from core.dataset.datamodule import get_datamodule



def unit_test():
    master_config = MasterConfig.default()
    
    model = get_model(master_config.model_config)
    datamodule = get_datamodule(master_config.dataset_config)
    
    metrics_calculator = MetricsCalculator(
        metrics_config = master_config.metrics_config,
        model = model,
        datamodule = datamodule,
    )
    result = metrics_calculator.calculate()
    print(result)
    
    return



if __name__ == "__main__":
    unit_test()
    print("Done!")