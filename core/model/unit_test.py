from .model_control import ModelControl
from .model_config import ModelConfig
from .registry import ARCHITECTURE_REGISTRY



def unit_test():
    model_config = ModelConfig.default()
    
    for name in ARCHITECTURE_REGISTRY.keys():
        model_config.architecture_name = name
        model_config.architecture_config = ARCHITECTURE_REGISTRY[name].config_class.default()
        
        model = ARCHITECTURE_REGISTRY[name].model_class(model_config.architecture_config)
        
    return



if __name__ == "__main__":
    unit_test()
    print("Done!")