from dataclasses import dataclass



@dataclass
class SavingConfig:
    @classmethod
    def default(self):
        return