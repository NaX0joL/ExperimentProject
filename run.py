from script.train_model import TrainingScript



def main() -> None:
    # notes for future self:
    #   this training is for exp basis tomorrow, 1layer 4heads noScheduler 300 epochs 
    TrainingScript.train_proposed_model_on_tsb_ad_u()
    TrainingScript.train_proposed_model_on_ucr_anomaly_detection()
    return



if __name__ == "__main__":
    main()
    print("DONE!")