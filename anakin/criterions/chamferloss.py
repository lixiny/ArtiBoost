from typing import Dict, Tuple

import torch
from anakin.datasets.hoquery import Queries
from anakin.utils.builder import LOSS
from anakin.utils.logger import logger

from .criterion import TensorLoss


@LOSS.register_module
class ChamferLoss(TensorLoss):

    def __init__(self, **cfg):
        super().__init__()
        self.lambda_chamfer = cfg.get("LAMBDA_CHAMFER", 0.0)
        import chamfer_distance as chd  # chd will call torch.cuda, might lead to CUDA_VISIBLE_DEVICES setup fail!

        self.ch_dist = chd.ChamferDistance()

        logger.info(f"Construct {type(self).__name__} with lambda: ")
        logger.info(f"  |   LAMBDA_CHAMFER : {self.lambda_chamfer}")

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        final_loss, losses = super().__call__(preds, targs, **kwargs)  # TENSOR(0.), {}
        if self.lambda_chamfer:
            obj_verts_can = targs[Queries.OBJ_VERTS_CAN].to(final_loss.device)
            pred_boxroot_3d_abs = preds["boxroot_3d_abs"]
            pred_box_rot_rotmat = preds["box_rot_rotmat"]
            pred_obj_verts_3d_abs = (
                torch.matmul(pred_box_rot_rotmat, obj_verts_can.permute(0, 2, 1)).permute(0, 2, 1) +
                pred_boxroot_3d_abs)
            obj_verts_3d = targs[Queries.OBJ_VERTS_3D]
            root_joint = targs[Queries.ROOT_JOINT]
            obj_verts_3d_abs = obj_verts_3d + root_joint.unsqueeze(1)

            # mask invisible samples
            corners_vis_mask = targs[Queries.CORNERS_VIS]
            pred_obj_verts_3d_abs = torch.einsum("bij,b->bij", pred_obj_verts_3d_abs,
                                                 torch.any(corners_vis_mask.to(final_loss.device), axis=1).float())
            obj_verts_3d_abs = torch.einsum("bij,b->bij", obj_verts_3d_abs, torch.any(corners_vis_mask, axis=1).float())

            dist_xy, dist_yx, _, _ = self.ch_dist(pred_obj_verts_3d_abs, obj_verts_3d_abs.to(final_loss.device))

            chamfer_loss = (torch.mean(dist_xy)) + (torch.mean(dist_yx))

            final_loss += self.lambda_chamfer * chamfer_loss
        else:
            chamfer_loss = None
        losses["chamfer_loss"] = chamfer_loss
        losses[self.output_key] = final_loss
        return final_loss, losses
