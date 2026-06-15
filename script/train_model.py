from __future__ import annotations
from typing import Optional
import os
import copy
from pathlib import Path

print(os.getcwd())

from torch import nn

from core.master_config import (
    MasterConfig,
    ModelConfig, 
    DatasetConfig, 
    TrainerConfig,
    MetricsConfig
)
from core.experiment import Experiment
from core.dataset.registry import DATASET_REGISTRY
from core.model.registry import ARCHITECTURE_REGISTRY



class GROUP_PRESET:
    GROUP = {
        "TSB_AD_U": [
            "random_group_1",
            "random_group_2",
            "random_group_3",
            "random_group_4",
            "random_group_5",
            "random_group_6",
            "random_group_7",
            "random_group_8",
            "random_group_9",
            "random_group_10",
        ],
        
        "UCR_Anomaly_Detection": [
            "sddb",
            "BIDMC",
            "AirTemperature",
            "ECG",
            "Marker",
            "InternalBleeding",
            "Lab2Cmac",
            "Mesoplodon",
            "PowerDemand",
            "MARS",
            "WalkingAceleration",
            "apneaecg",
            "gait",
            "insectEPG",
            "ltstdbs",
            "qtdbSel",
            "resperation",
            "s20101",
            "sel840m",
            "tilt",
            "park3m",
            "CHARIS",
            "Fantasia",
            "Italianpowerdemand"
        ]
    }
    
    @classmethod
    def get(cls, dataset_name:str) -> list[str]:
        return cls.GROUP[dataset_name]



class MetricsConfig_PRESET:
    
    @staticmethod
    def get() -> MetricsConfig:
        metrics_config = MetricsConfig.default(
                task_type = "regression",
                tolerance = 2,
        )
        return metrics_config



class TrainerConfig_PRESET:
    
    @staticmethod
    def get() -> TrainerConfig:
        trainer_config = TrainerConfig.default(
            train_epochs = 2,
            learning_rate = 1e-4,
            
            max_learning_rate = 1e-3,
            percentage_start = 0.3,
            div_factor = 25,
            final_div_factor = 1e4,
            
            weight_decay = 1e-4,
            grad_clip_max_norm = 1.0,
            
            optimizer_name = "adamw",
            criterion_name = "mae",
            
            use_best_model = True,
            
            loss_coefficients = {
                "base_loss": 1,
            },
        )
        return trainer_config



class DatasetConfig_PRESET:
    
    class TSB_AD_U:
        
        @staticmethod
        def get() -> DatasetConfig:
            dataset_config = DatasetConfig.default(
                batch_size = 16,
                shuffle = True,
                drop_last = False,
                
                dataset_name = "TSB_AD_U",
            )
            return dataset_config
    
    class UCR_Anomaly_Detection:
        
        @staticmethod
        def get() -> DatasetConfig:
            dataset_config = DatasetConfig.default(
                batch_size = 16,
                shuffle = True,
                drop_last = False,
                
                dataset_name = "UCR_Anomaly_Detection",
            )
            return dataset_config



class ModelConfig_PRESET:
    
    class Proposed_Model:
        
        @staticmethod
        def get() -> ModelConfig:
            model_config = ModelConfig.default(
                architecture_name = "proposed_model",
                
                seq_len = 1000,
                pred_len = 1000,
                patch_len = 50,
                stride = 1,
                
                e_layers_num = 1,
                enc_in_feature = 1,
                d_layers_num = 1,
                dec_in_feature = 1,
                
                n_heads_num = 1,
                n_normal_heads = 0,
                n_mp_attn_heads = 0,
                qk_weight_share = False,
                d_model = 256,
                d_ff = 512,
                
                dropout = 0.5,
                fc_dropout = 0.3,
                head_dropout = 0.1,
                attn_dropout = 0.1,
                
                use_pre_norm = False,
            )
            return model_config
    
    class Others:
        pass



class TrainingScript:
    
    @staticmethod
    def train_on_all_groups(
        model_config: ModelConfig, 
        dataset_config: DatasetConfig,  
        metrics_config: MetricsConfig, 
        trainer_config: TrainerConfig,
        *,
        groups: list[str] = None,
        base_id: str = "",
        save_path: str = "tmp",
        ) -> None:
        
        if groups is None:
            groups = GROUP_PRESET.get(dataset_config.dataset_name)
        
        models_folder_path = Path(f"{save_path}/{dataset_config.dataset_name}")
        os.makedirs(name=models_folder_path, exist_ok=False)
        
        for group in groups:
            current_dataset_config = copy.deepcopy(dataset_config)
            current_dataset_config.raw_getter_config.group = group
        
            master_config = MasterConfig(
                model_config,
                current_dataset_config,
                metrics_config,
                trainer_config,
            )
            
            exp_id = f"mpkg-{base_id}-{dataset_config.dataset_name}-{group}"
            exp = Experiment(master_config, experiment_id=exp_id)
            
            exp.train_model()
            exp.compute_metrics()
            exp.save(f"{save_path}/{exp_id}")
            
        return



if __name__ == "__main__":
    TrainingScript.train_on_all_groups(
        metrics_config = MetricsConfig_PRESET.get(),
        trainer_config = TrainerConfig_PRESET.get(),
        model_config = ModelConfig_PRESET.Proposed_Model.get(),
        dataset_config = DatasetConfig_PRESET.TSB_AD_U.get(),
        
        base_id = "test_exp1",
        save_path = "tmp/test_exp1"
    )
    print("DONE!")