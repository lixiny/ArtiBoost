import os
import time
from typing import Dict, Optional, TypeVar

from anakin.metrics.evaluator import Evaluator
from anakin.utils.misc import TrainMode
from torch.utils.tensorboard import SummaryWriter

T = TypeVar("T", bound="Summarizer")


class Summarizer:
    def __init__(
        self: T,
        exp_id: str,
        cfg: Dict,
        tensorboard_path: str = "./runs",
        rank: Optional[int] = None,
        time_f: Optional[float] = None,
    ) -> None:
        self.timestamp = time.strftime("%Y_%m%d_%H%M_%S", time.localtime(time_f if time_f else time.time()))
        self.exp_id = exp_id
        self.cfg = cfg
        self.rank = rank
        self._n_iter = 0
        if not self.rank:
            self.tb_writer = SummaryWriter(os.path.join(tensorboard_path, f"{exp_id}_{self.timestamp}"))

    def summarize_evaluator(self: T, evaluator: Evaluator, epoch: int, train_mode: TrainMode):
        if self.rank:
            return

        file_perfixes = {
            TrainMode.TRAIN: "train",
            TrainMode.VAL: "val",
            TrainMode.TEST: "test",
        }
        prefix = file_perfixes[train_mode]
        eval_measures = evaluator.get_measures_all_striped(return_losses=False)
        for k, v in eval_measures.items():
            if isinstance(v, Dict):
                for k_, v_ in v.items():
                    self.tb_writer.add_scalar(f"{k}/{prefix}/{k_}", v_, epoch)
            else:
                self.tb_writer.add_scalar(f"{k}/{prefix}", v, epoch)

    def summarize_losses(self: T, losses: Dict):
        if self.rank:
            return
        self.tb_writer.add_scalar("Loss", losses["final_loss"], self._n_iter)
        self.tb_writer.add_scalars(
            "Losses", {k: v for k, v in losses.items() if k != "final_loss" and v is not None}, self._n_iter
        )
        self._n_iter += 1

    def clear_summarizer(self: T):
        self._n_iter = 0
