import functools
import json
from abc import ABC, abstractmethod
from itertools import compress
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from anakin.datasets.hoquery import Queries, SynthQueries
from anakin.metrics.metric import AverageMeter, Metric
from anakin.utils.bop_toolkit.bop_misc import get_symmetry_transformations
from anakin.utils.builder import METRIC
from anakin.utils.logger import logger


class ValMetric2:
    build_mapping: Mapping[str, Metric] = {}

    @staticmethod
    def build(cata, cfg):
        return ValMetric2.build_mapping[cata](**cfg)

    @staticmethod
    def check(cata):
        return cata in ValMetric2.build_mapping


class IDMappingMeters:
    """Computes and stores the mapping, from sample_identifier to metric value"""

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self):
        self.count = 0
        self.storage = {}

    def update(self, seq_id: Sequence[Tuple[int]], seq_val: np.ndarray, synth_flag: np.ndarray):
        seq_id = np.array(seq_id)
        real_flag = ~synth_flag

        seq_id_synth = seq_id[synth_flag]
        seq_val_synth = seq_val[synth_flag]
        # seq_id_real = seq_id[real_flag]
        # seq_val_real = seq_val[real_flag]

        self.count += len(seq_id_synth)

        for idx, val in zip(seq_id_synth, seq_val_synth):
            self.storage[tuple(idx)] = val


@METRIC.register_module
class ValMetricMean3DEPE2(Metric):

    def __init__(self, **cfg) -> None:
        """
        Mean End-Point-Error Metric
        These metric track the mean of 2D/3D joints, vertices 2-norm distance
        Args:
            **cfg: VAL_KEY : the key of the tracking value, eg: joints_3d_abs
        """
        super(ValMetricMean3DEPE2, self).__init__()
        self.val_keys_list: List[str] = cfg["VAL_KEYS"]
        self.id_mapping_meters: Dict[str, IDMappingMeters] = {}
        for key in self.val_keys_list:
            self.id_mapping_meters[key] = IDMappingMeters()
        self.to_millimeters = cfg.get("MILLIMETERS", False)
        self.reset()

    def reset(self):
        for k, meter in self.id_mapping_meters.items():
            meter.reset()

    def feed(self, preds: Dict, targs: Dict, **kwargs):
        # get id list
        synth_flag = targs[SynthQueries.IS_SYNTH]
        obj_id = targs[SynthQueries.OBJ_ID]
        persp_id = targs[SynthQueries.PERSP_ID]
        grasp_id = targs[SynthQueries.GRASP_ID]
        zipped_id = self.zip_seq_id(obj_id, persp_id, grasp_id)

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
                "X pred shape, should as (BATCH, NPOINTS, 2|3)")  # TENSOR (BATCH, NPOINTS, 2|3)

            diff = pred_val - val  # TENSOR (B, N, 2|3)
            if self.to_millimeters:
                diff = diff * 1000.0
            dist_ = torch.norm(diff, p="fro", dim=2)  # TENSOR (B, N)
            dist_batch = torch.mean(dist_, dim=1, keepdim=False)  # TENSOR (B,)
            dist_batch = dist_batch.detach().cpu().numpy()
            synth_flag_ = synth_flag.detach().cpu().numpy()
            self.id_mapping_meters[key].update(zipped_id, dist_batch, synth_flag_)

    def get_measures(self, **kwargs) -> Dict[str, Dict[Tuple[int], float]]:
        """
        Args:
            **kwargs:

        Returns:
            eg: {joints_3d_abs : {storage}, }

        """
        measures = {}
        for key in self.val_keys_list:
            measures[f"{key}"] = self.id_mapping_meters[key].storage

        return measures

    def get_measures_averaged(self, **kwargs) -> Dict[Tuple[int], float]:
        """Get average measures of all metric in id_mapping_meters

        Returns:
            Dict[Tuple[int], float]: dict of (zipped_id -> average measures)
        """
        meas_average = {}
        storage_list = [self.id_mapping_meters[key].storage for key in self.val_keys_list]

        for key in storage_list[0].keys():  # key = (oid, vid, gid)
            collected_value = [sto[key] for sto in storage_list]
            meas_average[key] = sum(collected_value) / len(collected_value)
        return meas_average

    @staticmethod
    def zip_seq_id(*args):
        return list([tuple(int(x) for x in el) for el in zip(*args)])

    def __str__(self):
        return ""


@METRIC.register_module
class ValMetricAR2(Metric):

    def __init__(self, **cfg) -> None:
        """
        Mean End-Point-Error Metric
        These metric track the mean of 2D/3D joints, vertices 2-norm distance
        Args:
            **cfg: VAL_KEY : the key of the tracking value, eg: joints_3d_abs
        """
        super(ValMetricAR2, self).__init__()
        self.vsd = VSD(**cfg) if cfg.get("USE_VSD", False) else None
        self.mssd = MSSD(**cfg) if cfg.get("USE_MSSD", False) else None
        self.mspd = MSPD(**cfg) if cfg.get("USE_MSPD", False) else None
        # self.id_mapping_meters: Dict[str, IDMappingMeters] = {}
        # for key in self.val_keys_list:
        #     self.id_mapping_meters[key] = IDMappingMeters()
        self.reset()

    def reset(self):
        # for k, meter in self.id_mapping_meters.items():
        #     meter.reset()
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

    def get_measures(self, **kwargs) -> Dict[str, Dict[Tuple[int], float]]:
        """
        Args:
            **kwargs:

        Returns:
            eg: {joints_3d_abs : {storage}, }

        """
        measures = {}
        if self.vsd is not None:
            measures["vsd"] = self.vsd.id_mapping_meters.storage
        if self.mssd is not None:
            measures["mssd"] = self.mssd.id_mapping_meters.storage
        if self.mspd is not None:
            measures["mspd"] = self.mspd.id_mapping_meters.storage

        return measures

    def get_measures_averaged(self, **kwargs) -> Dict[Tuple[int], float]:
        """Get average measures of all metric in id_mapping_meters

        Returns:
            Dict[Tuple[int], float]: dict of (zipped_id -> average measures)
        """
        meas_average = {}
        storage_list = []
        if self.vsd is not None:
            storage_list.append(self.vsd.id_mapping_meters.storage)
        if self.mssd is not None:
            storage_list.append(self.mssd.id_mapping_meters.storage)
        if self.mspd is not None:
            storage_list.append(self.mspd.id_mapping_meters.storage)

        for key in storage_list[0].keys():
            collected_value = [sto[key] for sto in storage_list]
            meas_average[key] = sum(collected_value) / len(collected_value)
        return meas_average

    def __str__(self):
        return ""


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
        self.use_ho3d_ycb = cfg.get("USE_HO3D_YCB", False)
        if self.mssd_use_corners:
            logger.info("MSSD use corners to simplify calculation")
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

        self.id_mapping_meters = IDMappingMeters()

    def reset(self):
        self.id_mapping_meters.reset()

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
            synth_flag = targs[SynthQueries.IS_SYNTH][mask]
            obj_id = targs[SynthQueries.OBJ_ID][mask]
            persp_id = targs[SynthQueries.PERSP_ID][mask]
            grasp_id = targs[SynthQueries.GRASP_ID][mask]
            zipped_id = self.zip_seq_id(obj_id, persp_id, grasp_id)
            can = obj_can[mask]
            transf = obj_transf[mask]

            if not self.use_ho3d_ycb:
                sym_can = (torch.einsum("kmn,bvn->bkmv", sym_R, can) + sym_t[None, :]).transpose(-2, -1)
            else:
                cam_extr = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                                        dtype=torch.float32,
                                        device=device)

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

            mssd_value = torch.norm(sym_3d_abs - pred_3d_abs.unsqueeze(1), dim=-1).max(-1)[0].min(-1)[0]  # [N, ]
            mssd_value = mssd_value * 1000.0  # to mm
            mssd_value = mssd_value.detach().cpu().numpy()
            # mssd_value_list = [float(el) for el in torch.split(mssd_value, 1)]
            # synth_flag_list = [bool(el) for el in torch.split(synth_flag, 1)]
            synth_flag = synth_flag.detach().cpu().numpy()
            self.id_mapping_meters.update(zipped_id, mssd_value, synth_flag)

    @staticmethod
    def zip_seq_id(*args):
        return list([tuple(int(x) for x in el) for el in zip(*args)])

    def __str__(self) -> str:
        return f"mssd: {self.avg:6.4f}"


class MSPD:

    def __init__(self, **cfg) -> None:
        super().__init__()
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()
