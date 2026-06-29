import pandas as pd



class Grouper():
    
    def __init__(self, stratum:str) -> None:
        self.stratum = stratum
        return
    
    def group_data(self, data:pd.DataFrame) -> pd.DataFrame:
        data["group"] = data[self.stratum]
        return data