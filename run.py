import time

#from script.train_model import TrainingScript
from core.dataset.preprocessing.loader.TSB_AD_U import TSB_AD_U_Loader
from core.dataset.preprocessing.loader.UCR_Anomaly_Detection import UCR_Anomaly_Detection_Loader

from core.dataset.preprocessing.filter.pre_filter import PreFilter

from core.dataset.preprocessing.grouper.group import Grouper

from core.dataset.preprocessing.segmentation.cut_around_anomaly import CutAroundAnomaly
from core.dataset.preprocessing.segmentation.fixed_size_patching import FixedSizePatching

from core.dataset.preprocessing.transform.normalize import Normalization

from core.dataset.preprocessing.preprocess import Preprocessor, DataPipeline
from core.dataset.preprocessing.preprocess_config import PreprocessConfig

from core.dataset.data_server import DataServer
from core.dataset.dataset_config import DatasetConfig

from core.experiment import Experiment



def main() -> None:
    # notes for future self:
    #   this training is for exp basis tomorrow, 1layer 4heads noScheduler 300 epochs 
    # TrainingScript.train_proposed_model_on_tsb_ad_u()
    # TrainingScript.train_proposed_model_on_ucr_anomaly_detection()
    
    start_time = time.time()
    
    exp = Experiment(experiment_id="TEST HARI INI!1!1")
    exp.train_model()
    exp.save()
    
    end_time = time.time()
    print(end_time - start_time)
    
    return
    
    # pipeline = DataPipeline.default(dataset="UCR_Anomaly_Detection")
    # preprocessor = Preprocessor(data_pipeline=pipeline)
    # preprocessor.get_data()
    
    dataset_config = DatasetConfig.default()
    data_server = DataServer(dataset_config)
    for datamodule in data_server.serve_datamodule(n_fold=999):
        for batch in datamodule.train_dataloader:
            print(batch)
        break
    
    end_time = time.time()
    print(end_time - start_time)
    
    return
    
    tsb_ad_u_df = TSB_AD_U_Loader.load_data(parallelized=True)
    ucr_anomaly_detection_df = UCR_Anomaly_Detection_Loader.load_data(parallelized=True)
    
    print(tsb_ad_u_df)
    print(ucr_anomaly_detection_df)
    
    tsb_pre_filter = PreFilter(filter_columns={"source": "YAHOO"})
    ucr_pre_filter = PreFilter()
    tsb_ad_u_df = tsb_pre_filter.apply(tsb_ad_u_df)
    ucr_anomaly_detection_df = ucr_pre_filter.apply(ucr_anomaly_detection_df)
    
    print(tsb_ad_u_df)
    print(ucr_anomaly_detection_df)
    
    tsb_grouper = Grouper(stratum="series_id")
    ucr_grouper = Grouper(stratum="mnemonic")
    tsb_ad_u_df = tsb_grouper.group_data(tsb_ad_u_df)
    ucr_anomaly_detection_df = ucr_grouper.group_data(ucr_anomaly_detection_df)
    
    print(tsb_ad_u_df)
    print(ucr_anomaly_detection_df)
    
    ucr_segmenter = CutAroundAnomaly(window_size=1000)
    tsb_segmenter = FixedSizePatching(patch_size=200)
    tsb_ad_u_df = tsb_segmenter.segment_data(tsb_ad_u_df)
    ucr_anomaly_detection_df = ucr_segmenter.segment_data(ucr_anomaly_detection_df)
    
    print(tsb_ad_u_df)
    print(ucr_anomaly_detection_df)
    
    transformer = Normalization(transformed_columns=["value"])
    tsb_ad_u_df = transformer.transform_data(tsb_ad_u_df)
    ucr_anomaly_detection_df = transformer.transform_data(ucr_anomaly_detection_df)
    
    print(tsb_ad_u_df)
    print(ucr_anomaly_detection_df)
    
    end_time = time.time()
    print(end_time - start_time)
    
    return



if __name__ == "__main__":
    main()
    print("DONE!")