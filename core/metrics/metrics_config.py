from dataclasses import dataclass



@dataclass
class MetricsConfig:
    dummy: None
    
    @classmethod
    def default(self):
        return self(
            dummy = None
        )