from dataclasses import dataclass

from .schema import (
    Loader,
    Filter,
    Grouper,
    Segmenter,
    Transformer,
    Splitter,
)

from .loader.TSB_AD_U import TSB_AD_U_Loader
from .loader.UCR_Anomaly_Detection import UCR_Anomaly_Detection_Loader

from .filter.pre_filter import PreFilter

from .grouper.group import Grouper

from .segmentation.cut_around_anomaly import CutAroundAnomaly
from .segmentation.fixed_size_patching import FixedSizePatching

from .transform.normalize import Normalization



@dataclass
class PreprocessConfig:
    loader: Loader
    pre_filter: Filter | None
    grouper: Grouper
    regrouper: None
    segmenter: Segmenter
    post_filter: None
    transformer: Transformer | None
    splitter: Splitter
    
    @classmethod
    def default(cls, dataset:str="UCR_Anomaly_Detection") -> "PreprocessConfig":
        if dataset == "TSB_AD_U":
            preprocess_config = None
        
        elif dataset == "UCR_Anomaly_Detection":
            preprocess_config = cls(
                loader = UCR_Anomaly_Detection_Loader(),
                pre_filter = PreFilter(),
                grouper = Grouper(stratum="mnemonic"),
                regrouper = None,
                segmenter = CutAroundAnomaly(window_size=1000),
                post_filter = None,
                transformer = Normalization(transformed_columns=["value"]),
                splitter = None,
            )
            
        else:
            preprocess_config = None
        
        return preprocess_config