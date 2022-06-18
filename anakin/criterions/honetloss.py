from typing import Dict, Tuple

import torch
import torch.nn.functional as torch_f
from anakin.criterions.criterion import TensorLoss
from anakin.datasets.hoquery import Queries
from anakin.utils.builder import LOSS
from anakin.utils.logger import logger


@LOSS.register_module
class ManoLoss(TensorLoss):
    def __init__(self, **cfg):
        """
        Computed terms of MANO weighted loss, which encompasses vertex/joint
        supervision and pose/shape regularization
        """
        super(ManoLoss, self).__init__()
        self.lambda_joints_3d = float(cfg["LAMBDA_JOINTS_3D"])
        self.lambda_hand_verts_3d = float(cfg["LAMBDA_HAND_VERTS_3D"])
        self.lambda_shape_reg = float(cfg["LAMBDA_SHAPE_REG"])
        self.lambda_pose_reg = float(cfg["LAMBDA_POSE_REG"])

        logger.info(f"Construct {type(self).__name__} with lambda: ")
        logger.info(f"  |   LAMBDA_JOINTS_3D : {self.lambda_joints_3d}")
        logger.info(f"  |   LAMBDA_HAND_VERTS_3D : {self.lambda_hand_verts_3d}")
        logger.info(f"  |   LAMBDA_SHAPE_REG : {self.lambda_shape_reg}")
        logger.info(f"  |   LAMBDA_POSE_REG : {self.lambda_pose_reg}")

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        final_loss, losses = super().__call__(preds, targs, **kwargs)  # TENSOR(0.), {}

        # Compute hand shape regularization loss
        if self.lambda_shape_reg:
            pred_shape = preds["mano_shape"]
            shape_reg_loss = torch_f.mse_loss(pred_shape, torch.zeros_like(pred_shape))
            final_loss += self.lambda_shape_reg * shape_reg_loss
        else:
            shape_reg_loss = None
        losses["mano_shape"] = shape_reg_loss

        # Compute hand pose regularization loss
        if self.lambda_pose_reg:
            pred_pose = preds["mano_pca_pose"][:, 3:]  # ignore root rotations at [:, :3]
            pose_reg_loss = torch_f.mse_loss(pred_pose, torch.zeros_like(pred_pose))
            final_loss += self.lambda_pose_reg * pose_reg_loss
        else:
            pose_reg_loss = None
        losses["mano_pca_pose"] = pose_reg_loss

        # Compute hand losses in and camera coordinates
        if self.lambda_joints_3d and Queries.JOINTS_3D in targs:
            pred_joints_3d_abs = preds["joints_3d_abs"]
            joints_3d = targs[Queries.JOINTS_3D].to(pred_joints_3d_abs.device)
            root_joint = targs[Queries.ROOT_JOINT].to(pred_joints_3d_abs.device)
            joints_3d_loss = torch_f.mse_loss(pred_joints_3d_abs, joints_3d + root_joint.unsqueeze(1))
            final_loss += self.lambda_joints_3d * joints_3d_loss
        else:
            joints_3d_loss = None
        losses["joints_3d_loss"] = joints_3d_loss

        if self.lambda_hand_verts_3d and Queries.HAND_VERTS_3D in targs:
            pred_hand_verts_3d_abs = preds["hand_verts_3d_abs"]
            hand_verts_3d = targs[Queries.HAND_VERTS_3D].to(pred_hand_verts_3d_abs.device)
            root_joint = targs[Queries.ROOT_JOINT].to(pred_hand_verts_3d_abs.device)

            hand_verts_3d_loss = torch_f.mse_loss(pred_hand_verts_3d_abs, hand_verts_3d + root_joint.unsqueeze(1))
            final_loss += self.lambda_hand_verts_3d * hand_verts_3d_loss
        else:
            hand_verts_3d_loss = None
        losses["hand_verts_3d_loss"] = hand_verts_3d_loss

        return final_loss, losses


@LOSS.register_module
class ObjLoss(TensorLoss):
    def __init__(self, **cfg):
        super(ObjLoss, self).__init__()
        self.lambda_obj_verts_3d = cfg["LAMBDA_OBJ_VERTS_3D"]
        logger.info(f"Construct {type(self).__name__} with lambda: ")
        logger.info(f"  |   LAMBDA_OBJ_VERTS_3D : {self.lambda_obj_verts_3d}")

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        final_loss, losses = super().__call__(preds, targs, **kwargs)  # TENSOR(0.), {}

        if self.lambda_obj_verts_3d and Queries.OBJ_VERTS_3D in targs:
            pred_obj_verts_3d_abs = preds["obj_verts_3d_abs"]
            obj_verts_3d = targs[Queries.OBJ_VERTS_3D].to(pred_obj_verts_3d_abs.device)
            root_joint = targs[Queries.ROOT_JOINT].to(pred_obj_verts_3d_abs.device)
            obj_verts_3d_loss = torch_f.mse_loss(pred_obj_verts_3d_abs, obj_verts_3d + root_joint.unsqueeze(1))
            final_loss += self.lambda_obj_verts_3d * obj_verts_3d_loss
        else:
            obj_verts_3d_loss = None
        losses["obj_verts_3d_loss"] = obj_verts_3d_loss

        return final_loss, losses

