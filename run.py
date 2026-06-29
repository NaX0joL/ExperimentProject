import time

#from script.train_model import TrainingScript
from core.dataset.loader.TSB_AD_U import TSB_AD_U_Loader
from core.dataset.loader.UCR_Anomaly_Detection import UCR_Anomaly_Detection_Loader

from core.dataset.preprocessing.segmentation.cut_around_anomaly import CutAroundAnomaly
from core.dataset.preprocessing.transform.normalize import Normalization



def main() -> None:
    # notes for future self:
    #   this training is for exp basis tomorrow, 1layer 4heads noScheduler 300 epochs 
    # TrainingScript.train_proposed_model_on_tsb_ad_u()
    # TrainingScript.train_proposed_model_on_ucr_anomaly_detection()
    
    start_time = time.time()
    
    #tsb_ad_u_df = TSB_AD_U_Loader.get_data(parallelized=True)
    #print(tsb_ad_u_df)
    #print(tsb_ad_u_df["group"].unique())
    
    ucr_anomaly_detection_df = UCR_Anomaly_Detection_Loader.get_data(parallelized=True)
    print(ucr_anomaly_detection_df)
    print(ucr_anomaly_detection_df["group"].unique())
    
    segmenter = CutAroundAnomaly(window_size=1000)
    segmented = segmenter.segment_data(ucr_anomaly_detection_df)
    print(segmented)
    
    transformer = Normalization(transformed_columns=["value"])
    transformed = transformer.transform_data(segmented)
    print(transformed)

    end_time = time.time()
    print(end_time - start_time)
    
    return



if __name__ == "__main__":
    main()
    print("DONE!")