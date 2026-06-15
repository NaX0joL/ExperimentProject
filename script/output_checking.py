import os
import sys
from pathlib import Path

print(os.getcwd())
sys.path.append(os.getcwd())

import torch

from core.experiment import Experiment
from core.master_config import MasterConfig



def main() -> None:
    exp = Experiment()
    load_path = Path("savefolder/mpkg/tmp/no_weighted_loss/no_scheduler/elayers_2-nheads_4") / "mpkg-proposed_model-TSB_AD_U-TSB_AD_U-random_group_1"
    exp.load(path=load_path)
    
    pos_outputs = []
    neg_outputs = []
    
    dataloader = exp.experiment_state.datamodule.test_dataloader
    for batch in dataloader:
        input = batch["value"]
        label = batch["ground_truth"]
        output = exp.model_inference(input)
    
        pos_mask = label == 1
        neg_mask = label == 0
        
        pos_outputs.append(output[pos_mask])
        neg_outputs.append(output[neg_mask])
    
    pos_outputs = torch.cat(pos_outputs)
    neg_outputs = torch.cat(neg_outputs)
    
    print("anomaly positions  -> mean: %.4f, std: %.4f, min: %.4f, max: %.4f"
      % (pos_outputs.mean(), pos_outputs.std(), pos_outputs.min(), pos_outputs.max()))
    print("normal   positions -> mean: %.4f, std: %.4f, min: %.4f, max: %.4f"
        % (neg_outputs.mean(), neg_outputs.std(), neg_outputs.min(), neg_outputs.max()))
    
    return



if __name__ == "__main__":
    main()
    print("DONE!")