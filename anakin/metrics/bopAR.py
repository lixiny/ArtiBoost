import json
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import torch
from anakin.datasets.hoquery import Queries
from anakin.metrics.metric import AverageMeter, Metric
from anakin.utils.bop_toolkit.bop_misc import get_symmetry_transformations
from anakin.utils.bop_toolkit.bop_pose_error import mssd, re
from anakin.utils.builder import METRIC
from anakin.utils.logger import logger


@METRIC.register_module
class AR(Metric):

    def __init__(self, **cfg) -> None:
        super().__init__()
        self.vsd = VSD(**cfg) if cfg.get("USE_VSD", False) else None
        self.mssd = MSSD(**cfg) if cfg.get("USE_MSSD", False) else None
        self.mspd = MSPD(**cfg) if cfg.get("USE_MSPD", False) else None
        self.reset()

    def reset(self):
        if self.vsd is not None:
            self.vsd.reset()
        if self.mssd is not None:
            self.mssd.reset()
        if self.mspd is not None:
            self.mspd.reset()

    def feed(self, preds: Dict, targs: Dict, **kwargs):
        if self.vsd is not None:
            self.vsd.feed(preds, targs)
        if self.mssd is not None:
            self.mssd.feed(preds, targs)
        if self.mspd is not None:
            self.mspd.feed(preds, targs)

    def get_measures(self, **kwargs) -> Dict[str, float]:
        # TODO
        measures = {}
        if self.vsd is not None:
            raise NotImplementedError()
        if self.mssd is not None:
            measures["MSSD"] = self.mssd.avg
            measures.update(self.mssd.values)
        if self.mspd is not None:
            raise NotImplementedError()
        return measures

    def __str__(self) -> str:
        ar_res = []
        if self.vsd is not None:
            ar_res.append(str(self.vsd))
        if self.mssd is not None:
            ar_res.append(str(self.mssd))
        if self.mspd is not None:
            ar_res.append(str(self.mspd))
        return " | ".join(res for res in ar_res)


class VSD:

    def __init__(self, **cfg) -> None:
        super().__init__()
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class MSSD:

    def __init__(self, **cfg) -> None:
        super().__init__()

        model_info_path = cfg["MODEL_INFO_PATH"]
        self.model_info = json.load(open(model_info_path, "r"))
        self.max_sym_disc_step = cfg.get("MAX_SYM_DISC_STEP", 0.01)
        self.mssd_use_corners = cfg.get("MSSD_USE_CORNERS", False)
        self.center_idx = cfg["DATA_PRESET"]["CENTER_IDX"] if cfg.get("MSSD_USE_CENTER_IDX", False) else None
        self.use_ho3d_ycb = cfg.get("USE_HO3D_YCB", False)
        if self.mssd_use_corners:
            logger.info("MSSD use corners to simplify calculation")
        if self.center_idx is not None:
            logger.info("MSSD w.r.t. hand root")
        if self.use_ho3d_ycb:
            logger.info("MSSD use ho3d ycb cam extr")

        self.model_sym = {}
        for obj_idx in range(1, len(self.model_info) + 1):
            self.model_sym[obj_idx] = get_symmetry_transformations(self.model_info[str(obj_idx)],
                                                                   self.max_sym_disc_step)
        R, t = [], []
        for obj_idx in range(1, len(self.model_info) + 1):
            obj_R, obj_t = [], []
            for transf in self.model_sym[obj_idx]:
                obj_R.append(transf["R"])
                obj_t.append(transf["t"])
            obj_R = np.stack(obj_R)  # [Ki, 3, 3]
            obj_t = np.stack(obj_t)  # [Ki, 3, 1]
            R.append(torch.Tensor(obj_R))
            t.append(torch.Tensor(obj_t) / 1000.0)  # mm to m
        self.R = R  # list(N, (K), 3, 3)
        self.t = t  # list(N, (K), 3, 1)

        self.objs_error = {idx + 1: AverageMeter() for idx in range(len(self.model_info))}

    def reset(self):
        for avg_m in self.objs_error.values():
            avg_m.reset()

    def slow_feed(self, preds: Dict, targs: Dict, **kwargs):
        batch_pred_R = preds["box_rot_rotmat"].detach().cpu().numpy()
        batch_pred_t = preds["boxroot_3d_abs"].detach().cpu().transpose(-1, -2).numpy()
        batch_gt_R = targs["obj_transf"][:, :3, :3].numpy()
        batch_gt_t = targs["obj_transf"][:, :3, 3:].numpy()

        if self.mssd_use_corners:
            batch_pts = targs[Queries.CORNERS_CAN].numpy()
        else:
            batch_pts = targs["obj_verts_can"].numpy()

        for i, obj_idx in enumerate(targs["obj_idx"].tolist()):
            e = mssd(batch_pred_R[i], batch_pred_t[i], batch_gt_R[i], batch_gt_t[i], batch_pts[i],
                     self.model_sym[obj_idx])
            self.objs_error[obj_idx].update(e, n=1)

    def feed(self, preds: Dict, targs: Dict, **kwargs):
        device = preds["box_rot_rotmat"].device
        if self.mssd_use_corners:
            obj_can = targs[Queries.CORNERS_CAN].to(device)
        else:
            obj_can = targs[Queries.OBJ_VERTS_CAN].to(device)
        obj_transf = targs[Queries.OBJ_TRANSF].to(device)
        for obj_idx in range(1, len(self.model_info) + 1):
            sym_R = self.R[obj_idx - 1].to(device)
            sym_t = self.t[obj_idx - 1].to(device)
            mask = targs[Queries.OBJ_IDX] == obj_idx
            if not torch.any(mask):
                continue
            can = obj_can[mask]
            transf = obj_transf[mask]

            if not self.use_ho3d_ycb:
                sym_can = (torch.einsum("kmn,bvn->bkmv", sym_R, can) + sym_t[None, :]).transpose(-2, -1)
            else:
                cam_extr = torch.tensor(
                    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                    dtype=torch.float32,
                    device=device,
                )
                sym_can = (cam_extr @ (torch.einsum("kmn,bnv->bkmv", sym_R, cam_extr @ can.transpose(-2, -1)) + sym_t)
                          ).transpose(-2, -1)

            sym_3d_abs = (torch.einsum("bij,bklj->bkil", transf[:, :3, :3], sym_can) +
                          transf[:, None, :3, 3:]).transpose(-2, -1)

            pred_rot = preds["box_rot_rotmat"][mask]
            pred_tsl = preds["boxroot_3d_abs"][mask]
            if self.mssd_use_corners:
                pred_3d_abs = preds["corners_3d_abs"][mask]
            else:
                pred_3d_abs = (pred_rot @ can.transpose(-2, -1)).transpose(-2, -1) + pred_tsl
            if self.center_idx is None:
                mssd_value = torch.norm(sym_3d_abs - pred_3d_abs.unsqueeze(1), dim=-1).max(-1)[0].min(-1)[0]
            else:
                mssd_value = (torch.norm(
                    (sym_3d_abs - targs[Queries.ROOT_JOINT][mask][:, None, None, :].to(device)) -
                    (pred_3d_abs - preds["joints_3d_abs"][mask][:, [self.center_idx]]).unsqueeze(1),
                    dim=-1,
                ).max(-1)[0].min(-1)[0])
            self.objs_error[obj_idx].update(mssd_value.sum().item(), n=mssd_value.numel())

    @property
    def avg(self) -> float:
        sum_mssd = 0.0
        count_mssd = 0
        for avg_m in self.objs_error.values():
            sum_mssd += avg_m.sum
            count_mssd += avg_m.count
        return sum_mssd / count_mssd * 1000.0  # in millimeter

    @property
    def values(self) -> Dict:
        _values = {}
        for idx, avg_m in self.objs_error.items():
            _values[
                f"{str(idx)}{'.corner' if self.mssd_use_corners else ''}.mssd"] = avg_m.avg * 1000.0  # in millimeter
        return _values

    def __str__(self) -> str:
        return f"mssd: {self.avg:6.4f}"


class MSPD:

    def __init__(self, **cfg) -> None:
        super().__init__()
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()
