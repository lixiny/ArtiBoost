import functools
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import torch
from anakin.datasets.hoquery import Queries
from anakin.metrics.metric import AverageMeter, Metric
from anakin.utils.builder import METRIC
from anakin.utils.logger import logger


class _MeanEPE(Metric):
    def __init__(self, **cfg) -> None:
        """
        Mean End-Point-Error Metric
        These metric track the mean of 2D/3D joints, vertices 2-norm distance
        Args:
            **cfg: VAL_KEY : the key of the tracking value, eg: joints_3d_abs
        """
        super(_MeanEPE, self).__init__()
        self.val_keys_list: List[str] = cfg["VAL_KEYS"]
        self.avg_meters: Dict[str, AverageMeter] = {}
        for key in self.val_keys_list:
            self.avg_meters[key] = AverageMeter()
        self.to_millimeters = cfg.get("MILLIMETERS", False)

        if "arg" in cfg:
            self.filter_unseen_obj_idxs = cfg["arg"].filter_unseen_obj_idxs
        else:
            self.filter_unseen_obj_idxs = []

        self.reset()

    def reset(self):
        for k, meter in self.avg_meters.items():
            meter.reset()

    def feed(self, preds: Dict, targs: Dict, **kwargs):
        for key in self.val_keys_list:
            pred_val = preds[key]

            if "_abs" in key:
                targ_key = key.replace("_abs", "")
                val_ = targs[targ_key]
                root_joint = targs[Queries.ROOT_JOINT]
                val = val_ + root_joint.unsqueeze(1)
            else:
                val = targs[key]
            val = val.to(pred_val.device)

            assert len(pred_val.shape) == 3, logger.error(
                "X pred shape, should as (BATCH, NPOINTS, 2|3)"
            )  # TENSOR (BATCH, NPOINTS, 2|3)

            diff = pred_val - val  # TENSOR (B, N, 2|3)
            if self.to_millimeters:
                diff = diff * 1000.0
            dist_ = torch.norm(diff, p="fro", dim=2)  # TENSOR (B, N)
            dist_batch = torch.mean(dist_, dim=1, keepdim=True)  # TENSOR (B, 1)

            if "corners" in key and len(self.filter_unseen_obj_idxs) > 0:
                obj_idx_mask = torch.ones_like(targs[Queries.OBJ_IDX]).bool()
                for idx in self.filter_unseen_obj_idxs:
                    obj_idx_mask = obj_idx_mask & (targs[Queries.OBJ_IDX] != idx)
                dist_batch = dist_batch[obj_idx_mask]  # [B, 1]

            batch_size = dist_batch.shape[0]
            sum_dist_batch = torch.sum(dist_batch)
            self.avg_meters[key].update(sum_dist_batch.item(), n=batch_size)

    def get_measures(self, **kwargs) -> Dict[str, float]:
        """
        Args:
            **kwargs:

        Returns:
            eg: {joints_3d_abs_mepe : 22.0, }

        """
        measures = {}
        for key in self.val_keys_list:
            avg = (self.avg_meters[key]).avg
            measures[f"{key}_mepe"] = avg

        return measures

    def __str__(self):
        return " | ".join([f"{key}_mepe: {self.avg_meters[key].avg:6.4f}" for key in self.val_keys_list])


@METRIC.register_module
class Mean3DEPE(_MeanEPE):
    pass


@METRIC.register_module
class Mean2DEPE(_MeanEPE):
    def __init__(self, **cfg) -> None:
        super().__init__(**cfg)
        self.to_millimeters = False
