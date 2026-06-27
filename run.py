#from script.train_model import TrainingScript
from core.dataset.reader import UCR_Anomaly_Detection, TSB_AD_U



def main() -> None:
    # notes for future self:
    #   this training is for exp basis tomorrow, 1layer 4heads noScheduler 300 epochs 
    # TrainingScript.train_proposed_model_on_tsb_ad_u()
    # TrainingScript.train_proposed_model_on_ucr_anomaly_detection()
    
    TSB_AD_U.get_data()
    #UCR_Anomaly_Detection.get_data()
    
    return



if __name__ == "__main__":
    main()
    print("DONE!")