from typing import Dict, Tuple

import torch
import torch.nn.functional as torch_f
from anakin.datasets.hoquery import Queries
from anakin.utils.builder import LOSS
from anakin.utils.logger import logger

from .criterion import TensorLoss


@LOSS.register_module
class AlignLoss(TensorLoss):
    def __init__(self, **cfg):
        super().__init__()
        self.lambda_procrustes_align = cfg.get("LAMBDA_PROCRUSTES_ALIGN", 1.0)
        self.lambda_st_align = cfg.get("LAMBDA_ST_ALIGN", 0.0)

        logger.info(f"Construct {type(self).__name__} with lambda: ")
        logger.info(f"  |   LAMBDA_PROCRUSTES_ALIGN : {self.lambda_procrustes_align}")
        logger.info(f"  |   LAMBDA_ST_ALIGN : {self.lambda_st_align}")

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        final_loss, losses = super().__call__(preds, targs, **kwargs)  # TENSOR(0.), {}
        pred_joints_3d_abs = preds["joints_3d_abs"]
        joints_3d = targs[Queries.JOINTS_3D]  # TENSOR(B, NJOINTS, 3)
        root_joint = targs[Queries.ROOT_JOINT]  # TENSOR(B, 3)
        joints_3d_abs = joints_3d + root_joint.unsqueeze(1)
        joints_3d_abs = joints_3d_abs.to(final_loss.device)

        # ============== HAND JOINTS 3D PROCRUSTES ALIGNED MSE LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if self.lambda_procrustes_align:
            pred_joints_3d_abs_procrustes_aligned = self.procrustes_align(joints_3d_abs, pred_joints_3d_abs)
            procrustes_aligned_loss = torch_f.mse_loss(pred_joints_3d_abs_procrustes_aligned, joints_3d_abs)
            final_loss += self.lambda_procrustes_align * procrustes_aligned_loss
        else:
            procrustes_aligned_loss = None
        losses["procrustes_aligned_loss"] = procrustes_aligned_loss
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ============== HAND JOINTS 3D ST ALIGNED MSE LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if self.lambda_st_align:
            raise NotImplementedError()
        else:
            st_aligned_loss = None
        losses["st_aligned_loss"] = st_aligned_loss
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        losses[self.output_key] = final_loss
        return final_loss, losses

    @staticmethod
    def torch_orthogonal_procrustes(A, B):
        u, w, v = torch.svd(B.transpose(1, 2).bmm(A).transpose(1, 2))
        R = u.bmm(v.transpose(1, 2))
        scale = w.sum(dim=1, keepdim=True).unsqueeze(-1)
        return R, scale

    @staticmethod
    def procrustes_align(xyz, pred_xyz):
        """
        xyz: [B, N, 3]
        pred_xyz: [B, N, 3]

        return: [B, N, 3]
        """
        tsl = xyz.mean(1, keepdim=True)
        pred_tsl = pred_xyz.mean(1, keepdim=True)
        xyz_tsl = xyz - tsl
        pred_xyz_tsl = pred_xyz - pred_tsl

        scale = torch.norm(xyz_tsl, dim=(1, 2), keepdim=True) + 1e-8
        xyz_tsl_scale = xyz_tsl / scale
        pred_scale = torch.norm(pred_xyz_tsl, dim=(1, 2), keepdim=True) + 1e-8
        pred_xyz_tsl_scale = pred_xyz_tsl / pred_scale

        R, s = AlignLoss.torch_orthogonal_procrustes(xyz_tsl_scale, pred_xyz_tsl_scale)
        pred_xyz_aligned = torch.bmm(pred_xyz_tsl_scale, R.transpose(1, 2)) * s
        pred_xyz_aligned = pred_xyz_aligned * scale + tsl
        return pred_xyz_aligned
