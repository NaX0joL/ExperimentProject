from .REGISTRY import DATASET_REGISTRY
from .datamodule import get_datamodule
from .dataset_config import DatasetConfig



def unit_test():
    for name in DATASET_REGISTRY.keys():
        dataset_config = DatasetConfig.default()
        dataset_config.dataset_name = name
        dataset_config.raw_getter_config = DATASET_REGISTRY[name].config_class.default()
        
        datamodule = get_datamodule(dataset_config)
    
    return    



if __name__ == "__main__":
    unit_test()
    print("Done!")