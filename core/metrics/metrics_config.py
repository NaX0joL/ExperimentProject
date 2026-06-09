from dataclasses import dataclass



@dataclass
class MetricsConfig:
    @classmethod
    def default(self):
        return