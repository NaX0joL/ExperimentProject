from pathlib import Path

import pandas as pd



DATA_DIR = Path("data")



class DataReader():
    
    def __init__(self):
        pass
    
    def read_data(self, data_name:str):
        pass



class TSB_AD_U:
    NAME = "TSB_AD_U"
    PATH = DATA_DIR / NAME / "source_data"      # remove source data later and make all the data sit under the dataset folder directly
    
    METADATA = [
        "id", "filename", "feature_1", "label"
    ]
    
    @classmethod
    def get_data(cls) -> pd.DataFrame:
        data_rows = []
        cnt = 1
        
        for file_path in sorted(cls.PATH.iterdir()):
            if file_path.is_file():
                print(file_path.name)
                row = cls._make_row(id=cnt, file_path=file_path)
                data_rows.append(row)
                cnt += 1
        
        df = pd.DataFrame(data_rows, columns=cls.METADATA)
        return df
    
    @classmethod
    def _make_row(cls, id:int, file_path:Path) -> dict:
        feature_1, label = cls._load_feature_and_label(file_path)
        
        row = {
            "id": id,
            "filename": file_path.name,
            "feature_1": feature_1,
            "label": label,
        }
        return row
    
    @classmethod
    def _load_feature_and_label(cls, file_path:Path) -> list:
        content = pd.read_csv(file_path, header=None).iloc[1:]
        feature = content[0].reset_index(drop=True).squeeze().to_list()
        label = content[1].reset_index(drop=True).squeeze().to_list()
        return feature, label



class UCR_Anomaly_Detection:
    NAME = "UCR_Anomaly_Detection"
    PATH = DATA_DIR / NAME / "source_data"      # remove source data later and make all the data sit under the dataset folder directly
    
    METADATA = [
        "id", "filename", "feature_1",
    ]
    EXCEPTION_ID = [
        204, 205, 206, 207, 208, 225, 226, 242, 243     # for some reason, these files has multiple spaces delimiter instead of \n
    ]
    
    @classmethod
    def get_data(cls) -> pd.DataFrame:
        data_rows = []
        cnt = 1
        
        for file_path in sorted(cls.PATH.iterdir()):
            if file_path.is_file():
                row = cls._make_row(id=cnt, file_path=file_path)
                data_rows.append(row)
                cnt += 1
        
        df = pd.DataFrame(data_rows, columns=cls.METADATA)
        return df
    
    @classmethod
    def _make_row(cls, id:int, file_path:Path) -> dict:
        if id in cls.EXCEPTION_ID:
            feature_1 = cls._load_feature_exception(file_path)
        else:
            feature_1 = cls._load_feature(file_path)
            
        row = {
            "id": id,
            "filename": file_path.name,
            "feature_1": feature_1,
        }
        return row
    
    @classmethod
    def _load_feature(cls, file_path:Path) -> list:
        content = pd.read_csv(file_path, header=None)
        feature = content.reset_index(drop=True).squeeze().to_list()
        return feature
    
    @classmethod
    def _load_feature_exception(cls, file_path:Path) -> list:
        content = file_path.read_text(encoding="utf-8")
        raw_strings = content.split()
        feature = [float(val) for val in raw_strings]
        return feature