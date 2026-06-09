from core.dataset.unit_test import unit_test as dataset_unit_test
from core.model.unit_test import unit_test as model_unit_test



def unit_test():
    dataset_unit_test()
    model_unit_test()
    return



if __name__ == "__main__":
    unit_test()
    print("Done!")