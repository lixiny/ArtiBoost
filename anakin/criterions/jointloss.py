from typing import Dict, Tuple

import torch
import torch.nn.functional as torch_f
from anakin.datasets.hoquery import Queries
from anakin.utils.builder import LOSS
from anakin.utils.logger import logger
from anakin.utils.misc import CONST

from .criterion import TensorLoss


@LOSS.register_module
class JointsLoss(TensorLoss):

    def __init__(self, **cfg):
        super(JointsLoss, self).__init__()
        self.lambda_joints_3d = cfg.get("LAMBDA_JOINTS_3D", 0.0)
        self.lambda_corners_3d = cfg.get("LAMBDA_CORNERS_3D", 0.0)

        logger.info(f"Construct {type(self).__name__} with lambda: ")
        logger.info(f"  |   LAMBDA_JOINTS_3D : {self.lambda_joints_3d}")
        logger.info(f"  |   LAMBDA_CORNERS_3D : {self.lambda_corners_3d}")

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        final_loss, losses = super().__call__(preds, targs, **kwargs)  # TENSOR(0.), {}
        # ============== HAND JOINTS 3D MSE LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if self.lambda_joints_3d:
            pred_joints_3d_abs = preds["joints_3d_abs"]
            joints_3d = targs[Queries.JOINTS_3D]  # TENSOR(B, NJOINTS, 3)
            root_joint = targs[Queries.ROOT_JOINT]  # TENSOR(B, 3)
            joints_3d_abs = joints_3d + root_joint.unsqueeze(1)

            # mask invisible joints
            joints_vis_mask = targs[Queries.JOINTS_VIS]
            pred_joints_3d_abs = torch.einsum("bij,bi->bij", pred_joints_3d_abs, joints_vis_mask.to(final_loss.device))
            joints_3d_abs = torch.einsum("bij,bi->bij", joints_3d_abs, joints_vis_mask)

            joints_3d_loss = torch_f.mse_loss(pred_joints_3d_abs, joints_3d_abs.to(final_loss.device))
            final_loss += self.lambda_joints_3d * joints_3d_loss
        else:
            joints_3d_loss = None
        losses["joints_3d_loss"] = joints_3d_loss
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ============== OBJ CORNERS 3D MSE LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if self.lambda_corners_3d:
            pred_corners_3d_abs = preds["corners_3d_abs"]
            corners_3d = targs[Queries.CORNERS_3D]  # TENSOR(B, NCORNERS, 3)
            root_joint = targs[Queries.ROOT_JOINT]  # TENSOR(B, 3)
            corners_3d_abs = corners_3d + root_joint.unsqueeze(1)  # TENSOR (B, NCORNERS, 3)

            # mask invisible corners
            corners_vis_mask = targs[Queries.CORNERS_VIS]
            pred_corners_3d_abs = torch.einsum("bij,bi->bij", pred_corners_3d_abs,
                                               corners_vis_mask.to(final_loss.device))
            corners_3d_abs = torch.einsum("bij,bi->bij", corners_3d_abs, corners_vis_mask)

            corners_3d_loss = torch_f.mse_loss(pred_corners_3d_abs, corners_3d_abs.to(final_loss.device))
            final_loss += self.lambda_corners_3d * corners_3d_loss
        else:
            corners_3d_loss = None
        losses["corners_3d_loss"] = corners_3d_loss
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        losses[self.output_key] = final_loss
        return final_loss, losses


@LOSS.register_module
class HandJointsLoss(TensorLoss):

    def __init__(self, **cfg):
        super(JointsLoss, self).__init__()
        self.lambda_joints_3d = cfg.get("LAMBDA_JOINTS_3D", 0.0)

        logger.info(f"Construct {type(self).__name__} with lambda: ")
        logger.info(f"  |   LAMBDA_JOINTS_3D : {self.lambda_joints_3d}")

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        final_loss, losses = super().__call__(preds, targs, **kwargs)  # TENSOR(0.), {}
        # ============== HAND JOINTS 3D MSE LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if self.lambda_joints_3d:
            pred_joints_3d_abs = preds["joints_3d_abs"]
            joints_3d = targs[Queries.JOINTS_3D]  # TENSOR(B, NJOINTS, 3)
            root_joint = targs[Queries.ROOT_JOINT]  # TENSOR(B, 3)
            joints_3d_abs = joints_3d + root_joint.unsqueeze(1)

            # mask invisible joints
            joints_vis_mask = targs[Queries.JOINTS_VIS]
            pred_joints_3d_abs = torch.einsum("bij,bi->bij", pred_joints_3d_abs, joints_vis_mask.to(final_loss.device))
            joints_3d_abs = torch.einsum("bij,bi->bij", joints_3d_abs, joints_vis_mask)

            joints_3d_loss = torch_f.mse_loss(pred_joints_3d_abs, joints_3d_abs.to(final_loss.device))
            final_loss += self.lambda_joints_3d * joints_3d_loss
        else:
            joints_3d_loss = None
        losses["joints_3d_loss"] = joints_3d_loss
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        losses[self.output_key] = final_loss
        return final_loss, losses