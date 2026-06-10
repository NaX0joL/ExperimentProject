import torch
from torch import Tensor

from core.abstract import ABSTRACT_metric



class MaxAccuracy(ABSTRACT_metric):
    NAME = "max_accuracy"
    
    def __init__(self, eps=1e-8) -> None:
        self.eps = eps
        self.reset()
        return
    
    def reset(self) -> None:
        self._true = 0
        self._false = 0
        return
    
    def accumulate(self, output:Tensor, label:Tensor) -> None:
        output_max = output.max(dim=-1, keepdim=True)[0]
        output_mask = torch.isclose(output, output_max, rtol=0, atol=self.eps)
 
        label_max = label.max(dim=-1, keepdim=True)[0]
        label_mask = torch.isclose(label, label_max, rtol=0, atol=self.eps)
 
        row_has_match = (output_mask & label_mask).any(dim=1)
 
        self._true  += row_has_match.sum().item()
        self._false += (~row_has_match).sum().item()
        return
    
    def calculate(self) -> dict:
        max_accuracy = self._true / (self._true + self._false + self.eps)
        return {
            "max_accuracy": max_accuracy,
            "true": self._true,
            "false": self._false,
        }



class FScore(ABSTRACT_metric):
    NAME = "f_score"
    
    def __init__(self, eps:float=1e-8, beta:float=1, threshold:float=0.5) -> None:
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        
        self.reset()
        return
    
    def reset(self) -> None:
        self._TP = 0
        self._TN = 0
        self._FP = 0
        self._FN = 0
        return
    
    def accumulate(self, output:Tensor, label:Tensor) -> None:
        output_mask = (output >= self.threshold).int()
        label_mask  = (label  >= self.threshold).int()
 
        self._TP += ((output_mask == 1) & (label_mask == 1)).sum().item()
        self._TN += ((output_mask == 0) & (label_mask == 0)).sum().item()
        self._FP += ((output_mask == 1) & (label_mask == 0)).sum().item()
        self._FN += ((output_mask == 0) & (label_mask == 1)).sum().item()
        return

    def calculate(self) -> dict:
        TP, TN, FP, FN = self._TP, self._TN, self._FP, self._FN
        
        b2 = self.beta * self.beta
        precision = TP / (TP + FP + self.eps)
        recall = TP / (TP + FN + self.eps)
        
        f_score = (1 + b2) * TP / ((1 + b2) * TP + b2 * FN + FP + self.eps)
        return {
            "f_score": f_score,
            "precision": precision,
            "recall": recall,
            "TP": TP, 
            "TN": TN, 
            "FP": FP, 
            "FN": FN,
        }