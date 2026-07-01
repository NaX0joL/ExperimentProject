from dataclasses import dataclass

from .loader.TSB_AD_U import TSB_AD_U_Loader
from .loader.UCR_Anomaly_Detection import UCR_Anomaly_Detection_Loader

from .filter.pre_filter import PreFilter

from .grouper import group

from .segmentation.cut_around_anomaly import CutAroundAnomaly
from .segmentation.fixed_size_patching import FixedSizePatching

from .transform.normalize import Normalization



class RegistryEntry:
    object_class: Any
    config_class: Any



LOADER = {
    "TSB_AD_U": TSB_AD_U_Loader,
    "UCR_Anomaly_Detection": UCR_Anomaly_Detection_Loader,
}


PRE_FILTER = {
    "normal": PreFilter,
}


GROUP = {
    "normal": group
}


REGROUP = {
}


SEGMENT = {
    "cut_around_anomaly": CutAroundAnomaly,
    "fixed_size_patching": FixedSizePatching,
}


POST_FILTER = {
}


TRANSFORM = {
    "normalize": Normalization,
}