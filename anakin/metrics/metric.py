from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
import torch


class Metric(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def is_empty(self) -> bool:
        return self.count == 0

    def num_sample(self) -> int:
        return self.count

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def feed(self, preds: Dict, targs: Dict, **kwargs):
        pass

    @abstractmethod
    def get_measures(self, **kwargs) -> Dict:
        pass


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self,):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def update_by_mean(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        return f"{self.avg:.4e}"

