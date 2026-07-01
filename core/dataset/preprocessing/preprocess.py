from __future__ import annotations
from dataclasses import dataclass

from .preprocess_config import PreprocessConfig

from .schema import (
    Loader,
    Filter,
    Grouper,
    Regrouper,
    Segmenter,
    Transformer,
    Splitter,
)

from .loader.TSB_AD_U import TSB_AD_U_Loader
from .loader.UCR_Anomaly_Detection import UCR_Anomaly_Detection_Loader

from .filter.pre_filter import PreFilter
from .filter.post_filter import PostFilter

from .grouper.group import DataGrouper
from .grouper.regroup import DataRegrouper

from .segmentation.cut_around_anomaly import CutAroundAnomaly
from .segmentation.fixed_size_patching import FixedSizePatching

from .transform.normalize import Normalization


    
PIPELINE = [
    "loader",
    "pre_filter",
    "grouper",
    "regrouper",
    "segmenter",
    "post_filter",
    "transformer",
    "splitter",
]




class Preprocessor():
    
    def __init__(self, data_pipeline:DataPipeline, preprocess_config:PreprocessConfig=None):
        self.preprocess_config = preprocess_config
        self.data_pipeline = data_pipeline
        return
    
    def preprocess_data(self) -> None:
        processes = self.data_pipeline
        
        data = processes.loader.load_data(parallelized=True)
        if processes.pre_filter is not None:
            data = processes.pre_filter.filter_data(data)
        grouped_data = processes.grouper.group_data(data)
        if processes.regrouper is not None:
            grouped_data = processes.regrouper.regroup_data(grouped_data)
        segmented_data = processes.segmenter.segment_data(grouped_data)
        if processes.post_filter is not None:
            segmented_data = processes.post_filter.filter_data(segmented_data)
        if processes.transformer is not None:
            segmented_data = processes.transformer.transform_data(segmented_data)
        
        # print(segmented_data)
        return segmented_data



@dataclass
class DataPipeline:
    loader: Loader
    pre_filter: Filter | None
    grouper: Grouper
    regrouper: Regrouper | None
    segmenter: Segmenter
    post_filter: Filter | None
    transformer: Transformer | None
    splitter: Splitter
    
    @classmethod
    def default(cls, dataset:str="UCR_Anomaly_Detection") -> "PreprocessConfig":
        if dataset == "TSB_AD_U":
            pipeline = None
        
        elif dataset == "UCR_Anomaly_Detection":
            pipeline = cls(
                loader = UCR_Anomaly_Detection_Loader(),
                pre_filter = PreFilter(),
                grouper = DataGrouper(stratum="mnemonic"),
                regrouper = None,
                segmenter = CutAroundAnomaly(window_size=1000),
                post_filter = None,
                transformer = Normalization(transformed_columns=["value"]),
                splitter = None,
            )
            
        else:
            pipeline = None
        
        return pipeline