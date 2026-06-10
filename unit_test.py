from core.dataset.unit_test import unit_test as dataset_unit_test
from core.model.unit_test import unit_test as model_unit_test
from core.trainer.unit_test import unit_test as trainer_unit_test
from core.metrics.unit_test import unit_test as metrics_unit_test



def unit_test():
    dataset_unit_test()
    model_unit_test()
    trainer_unit_test()
    metrics_unit_test()
    
    return



if __name__ == "__main__":
    import torch
    print("PyTorch Version:", torch.__version__)
    print("PyTorch CUDA Version:", torch.version.cuda)
    print("Is CUDA available?:", torch.cuda.is_available())
    
    unit_test()
    print("Done!")