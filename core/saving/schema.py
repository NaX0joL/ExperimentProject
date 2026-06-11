from dataclasses import dataclass

from torch import nn

from core.master_config import MasterConfig



@dataclass
class MPKGState:
    config: MasterConfig
    model: nn.Module
    loss_log: dict[str, list]
    metrics_result: dict[str, dict]
    init_content: dict[str, any]
    
    @classmethod
    def init_empty(self):
        return self(
            config = None,
            model = None,
            loss_log = None,
            metrics_result = None,
            init_content = None,
        )