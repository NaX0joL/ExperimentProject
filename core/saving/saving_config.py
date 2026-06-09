from dataclasses import dataclass



@dataclass
class SavingConfig:
    dummy: None
    
    @classmethod
    def default(self):
        return self(
            dummy = None
        )