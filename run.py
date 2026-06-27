import time

#from script.train_model import TrainingScript
from core.dataset.loader.TSB_AD_U import TSB_AD_U_Loader
from core.dataset.loader.UCR_Anomaly_Detection import UCR_Anomaly_Detection_Loader



def main() -> None:
    # notes for future self:
    #   this training is for exp basis tomorrow, 1layer 4heads noScheduler 300 epochs 
    # TrainingScript.train_proposed_model_on_tsb_ad_u()
    # TrainingScript.train_proposed_model_on_ucr_anomaly_detection()
    start_time = time.time()
    
    tsb_ad_u_df = TSB_AD_U_Loader.get_data(parallelized=True)
    print(tsb_ad_u_df)
    ucr_anomaly_detection_df = UCR_Anomaly_Detection_Loader.get_data(parallelized=True)
    print(ucr_anomaly_detection_df)

    end_time = time.time()
    print(end_time - start_time)
    
    return



if __name__ == "__main__":
    main()
    print("DONE!")