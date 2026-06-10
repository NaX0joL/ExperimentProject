from dataclasses import dataclass



@dataclass
class MetricsConfig:
    task_type: str
    tolerance: int
    
    max_accuracy_eps: float
    
    f_score_eps: float
    f_score_beta: float
    f_score_threshold: float
    
    @classmethod
    def default(self):
        return self(
            task_type = "regression",
            tolerance = 2,
            
            max_accuracy_eps = 1e-8,
            
            f_score_eps = 1e-8,
            f_score_beta = 1,
            f_score_threshold = 0.5,
        )