from abc import ABC, abstractmethod
from anakin.criterions.criterion import TensorLoss
from typing import Dict, List

# from numpy.testing._private.utils import measure
from anakin.metrics.metric import AverageMeter, Metric
from anakin.utils.builder import METRIC
from anakin.utils.logger import logger


@METRIC.register_module
class LossesMetric(Metric):
    def __init__(self, **cfg) -> None:
        super().__init__()
        self._losses: Dict[str, AverageMeter] = {}
        self._vis_loss_keys: List[str] = cfg["VIS_LOSS_KEYS"]

    def reset(self):
        self._losses: Dict[str, AverageMeter] = {}
        self.count = 0

    def feed(self, losses: Dict[str, TensorLoss], batch_size: int = 1, **kwargs):
        for k, v in losses.items():
            if v is None:
                continue
            if k in self._losses:
                self._losses[k].update_by_mean(v.item(), batch_size)
            else:
                self._losses[k] = AverageMeter()
                self._losses[k].update_by_mean(v.item(), batch_size)
        self.count += batch_size

    def get_measures(self, **kwargs) -> Dict:
        measure = {}
        for k, v in self._losses.items():
            measure[k] = v.avg
        return measure

    def __str__(self) -> str:
        out = ", ".join(
            [f"final_loss: {self._losses['final_loss']}"]
            + [f"{k}: {v}" for k, v in self._losses.items() if k in self._vis_loss_keys]
        )

        return out
