from dataclasses import dataclass

from core.abstract import ABSTRACT_RawGetter, ABSTRACT_Config

from .data.TSB_AD_U.raw_getter import RawGetter as TSB_AD_U_RawGetter, RawGetterConfig as TSB_AD_U_Config
from .data.UCR_Anomaly_Detection.raw_getter import RawGetter as UCR_Anomaly_Detection_RawGetter, RawGetterConfig as UCR_Anomaly_Detection_Config
from .data.UCR_Classification.raw_getter import RawGetter as UCR_Classification_RawGetter, RawGetterConfig as UCR_Classification_Config



@dataclass
class DatasetEntry:
    raw_getter_class: type[ABSTRACT_RawGetter]
    config_class: type[ABSTRACT_Config]



DATASET_REGISTRY = {
    "TSB_AD_U": DatasetEntry(TSB_AD_U_RawGetter, TSB_AD_U_Config),
    "UCR_Anomaly_Detection": DatasetEntry(UCR_Anomaly_Detection_RawGetter, UCR_Anomaly_Detection_Config),
    "UCR_Classification": DatasetEntry(UCR_Classification_RawGetter, UCR_Classification_Config),
}