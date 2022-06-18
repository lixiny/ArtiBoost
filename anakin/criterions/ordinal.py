from anakin.criterions import jointloss
from typing import Dict, List, Tuple, ValuesView

import numpy as np
import torch
import torch.nn.functional as F
from anakin.criterions.criterion import TensorLoss
from anakin.datasets.hoquery import Queries
from anakin.utils.builder import LOSS
from anakin.utils.logger import logger
from anakin.utils.misc import CONST
from itertools import combinations
from itertools import product
import random


def partlevel_ordinal_relation(ppair: torch.Tensor, view_vecs: torch.Tensor):
    """

    Args:
        ppair: TENSOR (B, NPPAIRS, 6)
        view_vecs: TENSOR (B, NVIEWS, 3)

    Returns:
        ppair_ord: TENSOR (B, NPPAIRS, NVIEWS, 1)

    """
    nviews = view_vecs.shape[1]
    npairs = ppair.shape[1]
    ppair = ppair.unsqueeze(2).expand(-1, -1, nviews, -1)  # (B, NPPAIRS, NVIEWS, 6)
    view_vecs = view_vecs.unsqueeze(1).expand(-1, npairs, -1, -1)  # (B, NPPAIRS, NVIEWS, 3)
    ppair_cross = torch.cross(ppair[..., :3], ppair[..., 3:])  # (B, NPPAIRS, NVIEWS, 3)

    ppair_ord = torch.einsum("bijk, bijk->bij", ppair_cross, view_vecs)  # (B, NPPAIRS, NVIEWS)
    return ppair_ord.unsqueeze(-1)  # (B, NPPAIRS, NVIEWS, 1)


def jointlevel_ordinal_relation(jpair: torch.Tensor, view_vecs: torch.Tensor):
    """

    Args:
        jpair: TENSOR (B, NPAIRS, 6)
        view_vecs: TENSOR (B, NVIEWS, 3)

    Returns:
        jpair_ord: TENSOR (B, NPAIRS, NVIEWS, 1)

    """
    nviews = view_vecs.shape[1]
    npairs = jpair.shape[1]
    jpair = jpair.unsqueeze(2).expand(-1, -1, nviews, -1)  # (B, NPAIRS, NVIEWS, 6)
    view_vecs = view_vecs.unsqueeze(1).expand(-1, npairs, -1, -1)  # (B, NPAIRS, NVIEWS, 3)
    jpair_diff = jpair[..., :3] - jpair[..., 3:]  # (B, NPAIRS, NVIEWS, 3)

    jpair_ord = torch.einsum("bijk, bijk->bij", jpair_diff, view_vecs)  # (B, NPAIRS, NVIEWS)
    return jpair_ord.unsqueeze(-1)  # (B, NPAIRS, NVIEWS, 1)


def sample_view_vectors(n_virtual_views=20):

    cam_vec = torch.Tensor([0.0, 0.0, 1.0]).unsqueeze(0)  # TENSOR (1, NVIEWS)
    theta = torch.rand(n_virtual_views) * 2.0 * np.pi  # TENSSOR (NVIEWS, )
    u = torch.rand(n_virtual_views)

    nv_x = torch.sqrt(1.0 - u ** 2) * torch.cos(theta)  # TENSSOR (NVIEWS, )
    nv_y = torch.sqrt(1.0 - u ** 2) * torch.sin(theta)  # TENSSOR (NVIEWS, )
    nv_z = u  # TENSSOR (NVIEWS, )

    nv = torch.cat([nv_x.unsqueeze(1), nv_y.unsqueeze(1), nv_z.unsqueeze(1)], dim=1)  # TENSSOR (NVIEWS, 3)
    nv = torch.cat([cam_vec, nv], dim=0)  # TENSOR (NVIEWS, 3)
    return nv


@LOSS.register_module
class HandOrdLoss(TensorLoss):
    def __init__(self, **cfg):
        super(HandOrdLoss, self).__init__()
        self.lambda_part_lev = float(cfg.get("LAMBDA_PART_LEVEL", 1.0))
        self.lambda_joint_lev = float(cfg.get("LAMBDA_JOINTS_LEVEL", 1.0))
        self.n_virtual_views = int(cfg.get("N_VIRTUAL_VIEWS", 20))
        self.nviews = self.n_virtual_views + 1

        self.njoints = CONST.NUM_JOINTS
        self.nparts = CONST.NUM_JOINTS - 1

        # crate joint pair index
        joints_idx = list(range(self.njoints))
        self.joint_pairs_idx = list(combinations(joints_idx, 2))

        # create part pair index
        parts_idx = list(range(self.nparts))
        self.parts_pairs_idx = list(combinations(parts_idx, 2))

        logger.info(f"Construct {type(self).__name__} with lambda: ")
        logger.info(f"  |  LAMBDA_PART_LEVEL : {self.lambda_part_lev}")
        logger.info(f"  |  LAMBDA_JOINT_LEVEL : {self.lambda_joint_lev}")

    def joints_2_part_pairs(self, joints: torch.Tensor) -> torch.Tensor:
        """

        Args:
            joints: TENSOR (B, NJOINTS, 3)

        Returns:
            ppairs: TENSOR (B, NPAIRS, 6)

        """
        child_idx = list(range(self.njoints))
        parents_idx = CONST.JOINTS_IDX_PARENTS
        parts_ = joints[:, child_idx, :] - joints[:, parents_idx, :]  # (B, NJOINTS, 3)
        parts = parts_[:, 1:, :]  # (B, NPARTS, 3)

        pairs_idx = np.array(self.parts_pairs_idx)  # (NPAIRS, 2)
        pairs_idx1 = pairs_idx[:, 0]
        pairs_idx2 = pairs_idx[:, 1]

        pairs_parts1 = parts[:, pairs_idx1, :]  # (B, NPAIRS, 3)
        pairs_parts2 = parts[:, pairs_idx2, :]  # (B, NPAIRS, 3)

        pparis = torch.cat([pairs_parts1, pairs_parts2], dim=2)  # (B, NPAIRS, 6)
        return pparis

    def joints_2_joint_pairs(self, joints: torch.Tensor) -> torch.Tensor:
        """
        Converts joints3d into joint pairs. The pairing idx are defined in self.joint_pair_idx

        Args:
            joints: TENSOR (B, NJOINTS, 3)

        Returns:
            jpairs: TENSOR (B, NPAIRS, 6)

        """
        pairs_idx = np.array(self.joint_pairs_idx)  # (NPAIRS, 2)
        pairs_idx1 = pairs_idx[:, 0]
        pairs_idx2 = pairs_idx[:, 1]

        pairs_joints1 = joints[:, pairs_idx1, :]  # (B, NPAIRS, 3)
        pairs_joints2 = joints[:, pairs_idx2, :]  # (B, NPAIRS, 3)

        jpairs = torch.cat([pairs_joints1, pairs_joints2], dim=2)  # (B, NPAIRS, 6)
        return jpairs

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:

        batch_size = preds["joints_3d_abs"].shape[0]
        pred_joints_3d_abs = preds["joints_3d_abs"]
        device = pred_joints_3d_abs.device

        joints_3d = targs[Queries.JOINTS_3D].to(device)
        root_joint = targs[Queries.ROOT_JOINT].to(device)
        joints_3d_abs = joints_3d + root_joint.unsqueeze(1)

        # mask invisible joints
        joints_vis_mask = targs[Queries.JOINTS_VIS].to(device)
        pred_joints_3d_abs = torch.einsum("bij,bi->bij", pred_joints_3d_abs, joints_vis_mask)
        joints_3d_abs = torch.einsum("bij,bi->bij", joints_3d_abs, joints_vis_mask)

        final_loss, losses = super().__call__(preds, targs, **kwargs)  # TENSOR(0.), {}
        view_vecs = sample_view_vectors(self.n_virtual_views).to(device)  # TENOSR (NVIEWS, 3)
        view_vecs = view_vecs.unsqueeze(0).expand(batch_size, -1, -1)  # TENOSR (B, NVIEWS, 3)

        # ============== JOINT LEVEL ORDINAL LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        pred_jpairs = self.joints_2_joint_pairs(pred_joints_3d_abs)  # TENSOR (BATCH, NPAIRS, 6)
        jpairs = self.joints_2_joint_pairs(joints_3d_abs)  # TENSOR (BATCH, NPAIRS, 6)

        shuffle_idx = list(range(len(self.joint_pairs_idx)))
        random.shuffle(shuffle_idx)
        shuffle_idx = shuffle_idx[: len(shuffle_idx) // 3]
        pred_jpairs = pred_jpairs[:, shuffle_idx, :]
        jpairs = jpairs[:, shuffle_idx, :]

        jpairs_ord = jointlevel_ordinal_relation(jpairs, view_vecs)  # TENSOR (B, NPAIRS, NVIEWS, 1)
        jpairs_sign = torch.sign(jpairs_ord)

        pred_jpairs_ord = jointlevel_ordinal_relation(pred_jpairs, view_vecs)  # TENSOR (B, NPAIRS, NVIEWS, 1)

        joint_ord_loss_ = F.relu(-1.0 * jpairs_sign * pred_jpairs_ord)  # TENSOR (B, NPAIRS, NVIEWS, 1)
        joint_ord_loss_ = torch.log(1.0 + joint_ord_loss_)
        joint_ord_loss = torch.mean(joint_ord_loss_)  # mean on batch, npairs, nviews

        ### >>> None batch implementation
        # for i, vec_n in enumerate(view_vecs):
        #     vec_n = vec_n.to(final_loss.device)
        #
        #     jt_sign = self.get_joint_depth_sign(jpairs, vec_n)  # TENSER (BATCH, NPAIRS, 1)
        #     joint_loss_ = F.relu(
        #         jt_sign
        #         * torch.einsum(
        #             "bij, bij->bi",
        #             pred_jpairs[:, :, :3] - pred_jpairs[:, :, 3:],  # TENSOR (BATCH, NPAIRS, 3)
        #             vec_n.reshape(1, 1, 3).repeat(batch_size, pred_jpairs.shape[1], 1),  # TENSOR (BATCH, NPAIRS, 3)
        #         ).unsqueeze(2)
        #     )
        #     joint_loss_ = torch.log(1.0 + joint_loss_)
        #     mean_joint_loss_ = torch.mean(joint_loss_)
        #     joint_ord_loss += mean_joint_loss_
        # joint_ord_loss = joint_ord_loss / len(view_vecs)

        final_loss += self.lambda_joint_lev * joint_ord_loss
        losses["joint_ord_loss"] = joint_ord_loss
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ============== PART LEVEL ORDINAL LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        pred_ppairs = self.joints_2_part_pairs(pred_joints_3d_abs)  # TENSOR(BATCH, NPAIRS, 6)
        ppairs = self.joints_2_part_pairs(joints_3d_abs)

        shuffle_idx = list(range(len(self.parts_pairs_idx)))
        random.shuffle(shuffle_idx)
        shuffle_idx = shuffle_idx[: len(shuffle_idx) // 3]
        pred_ppairs = pred_ppairs[:, shuffle_idx, :]
        ppairs = ppairs[:, shuffle_idx, :]

        ppairs_ord = partlevel_ordinal_relation(ppairs, view_vecs)
        ppairs_sign = torch.sign(ppairs_ord)  # G.T. sign

        pred_ppairs_ord = partlevel_ordinal_relation(pred_ppairs, view_vecs)

        part_ord_loss_ = F.relu(-1.0 * ppairs_sign * pred_ppairs_ord)  # TENSOR (B, NPAIRS, NVIEWS, 1)
        part_ord_loss = torch.mean(part_ord_loss_)  # mean on batch, npairs, nviews

        final_loss += self.lambda_part_lev * part_ord_loss
        losses["part_ord_loss"] = part_ord_loss
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        losses[self.output_key] = final_loss
        return final_loss, losses


@LOSS.register_module
class SceneOrdLoss(TensorLoss):
    def __init__(self, **cfg):
        super(SceneOrdLoss, self).__init__()
        self.lambda_scene_lev = float(cfg.get("LAMBDA_SCENE_LEVEL", 1.0))
        self.n_virtual_views = int(cfg.get("N_VIRTUAL_VIEWS", 40))
        self.nviews = self.n_virtual_views + 1

        # crate joint | corners index
        joints_idx = list(range(CONST.NUM_JOINTS))  # [0, 1, ..., 20]
        corners_idx = list(range(CONST.NUM_CORNERS))  # [0, 1, ..., 7]

        # create hand-object points pairs index
        self.ho_pairs_idx = list(product(joints_idx, corners_idx))

        logger.info(f"Construct {type(self).__name__} with lambda: ")
        logger.info(f"  |  LAMBDA_SCENE_LEVEL : {self.lambda_scene_lev}")

    def ho_joints_2_ho_pairs(self, joints: torch.Tensor, corners: torch.Tensor):
        pairs_idx = np.array(self.ho_pairs_idx)  # (NPAIRS, 2)

        pairs_idx1 = pairs_idx[:, 0]
        pairs_idx2 = pairs_idx[:, 1]

        pairs_joints = joints[:, pairs_idx1, :]  # (B, NPAIRS, 3)
        pairs_corners = corners[:, pairs_idx2, :]  # (B, NPAIRS, 3)

        ho_pairs = torch.cat([pairs_joints, pairs_corners], dim=2)  # (B, NPAIRS, 6)
        return ho_pairs

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        batch_size = preds["joints_3d_abs"].shape[0]
        pred_joints_3d_abs = preds["joints_3d_abs"]
        pred_corners_3d_abs = preds["corners_3d_abs"]
        device = pred_joints_3d_abs.device

        joints_3d = targs[Queries.JOINTS_3D].to(device)
        corners_3d = targs[Queries.CORNERS_3D].to(device)
        root_joint = targs[Queries.ROOT_JOINT].to(device)
        joints_3d_abs = joints_3d + root_joint.unsqueeze(1)  # (B, 21, 3)
        corners_3d_abs = corners_3d + root_joint.unsqueeze(1)  # (B, 8, 3)

        # mask invisible joints
        joints_vis_mask = targs[Queries.JOINTS_VIS].to(device)
        pred_joints_3d_abs = torch.einsum("bij,bi->bij", pred_joints_3d_abs, joints_vis_mask)
        joints_3d_abs = torch.einsum("bij,bi->bij", joints_3d_abs, joints_vis_mask)

        # mask invisible corners
        corners_vis_mask = targs[Queries.CORNERS_VIS].to(device)
        pred_corners_3d_abs = torch.einsum("bij,bi->bij", pred_corners_3d_abs, corners_vis_mask)
        corners_3d_abs = torch.einsum("bij,bi->bij", corners_3d_abs, corners_vis_mask)

        final_loss, losses = super().__call__(preds, targs, **kwargs)  # TENSOR(0.), {}
        view_vecs = sample_view_vectors(self.n_virtual_views).to(device)  # TENOSR (NVIEWS, 3)
        view_vecs = view_vecs.unsqueeze(0).expand(batch_size, -1, -1)  # TENOSR (B, NVIEWS, 3)

        pred_ho_pairs = self.ho_joints_2_ho_pairs(pred_joints_3d_abs, pred_corners_3d_abs)
        ho_pairs = self.ho_joints_2_ho_pairs(joints_3d_abs, corners_3d_abs)

        shuffle_idx = list(range(len(self.ho_pairs_idx)))
        random.shuffle(shuffle_idx)
        shuffle_idx = shuffle_idx[: len(shuffle_idx) // 3]
        pred_ho_pairs = pred_ho_pairs[:, shuffle_idx, :]
        ho_pairs = ho_pairs[:, shuffle_idx, :]

        ho_pairs_ord = jointlevel_ordinal_relation(ho_pairs, view_vecs)  # TENSOR (B, NPAIRS, NVIEWS, 1)
        ho_pairs_sign = torch.sign(ho_pairs_ord)

        pred_ho_pairs_ord = jointlevel_ordinal_relation(pred_ho_pairs, view_vecs)  # TENSOR (B, NPAIRS, NVIEWS, 1)

        scene_ord_loss_ = F.relu(-1.0 * ho_pairs_sign * pred_ho_pairs_ord)
        scene_ord_loss_ = torch.log(1.0 + scene_ord_loss_)
        scene_ord_loss = torch.mean(scene_ord_loss_)

        final_loss += self.lambda_scene_lev * scene_ord_loss
        losses["scene_ord_loss"] = scene_ord_loss
        return final_loss, losses
