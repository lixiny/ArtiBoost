import pickle
from typing import Dict

import numpy as np
import torch
from anakin.utils.builder import HEAD
from anakin.utils.logger import logger
from anakin.utils.misc import CONST
from manotorch.manolayer import ManoLayer, MANOOutput
from torch import nn


class ManoAdaptor(torch.nn.Module):
    def __init__(self, mano_layer: ManoLayer, load_path: str = ""):
        super().__init__()
        self.adaptor = torch.nn.Linear(778, 21, bias=False)
        if load_path is not None:
            with open(load_path, "rb") as p_f:
                exp_data = pickle.load(p_f)
                weights = exp_data["adaptor"]
            regressor = torch.from_numpy(weights)
            self.register_buffer("J_regressor", regressor)
        else:
            regressor = mano_layer._buffers["th_J_regressor"]
            tip_reg = regressor.new_zeros(5, regressor.shape[1])
            tip_reg[0, 745] = 1
            tip_reg[1, 317] = 1
            tip_reg[2, 444] = 1
            tip_reg[3, 556] = 1
            tip_reg[4, 673] = 1
            reordered_reg = torch.cat([regressor, tip_reg])[
                [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
            ]
            self.register_buffer("J_regressor", reordered_reg)
        self.adaptor.weight.data = self.J_regressor

    def forward(self, inp):
        fix_idxs = [0, 4, 8, 12, 16, 20]
        for idx in fix_idxs:
            self.adaptor.weight.data[idx] = self.J_regressor[idx]
        return self.adaptor(inp.transpose(2, 1)), self.adaptor.weight - self.J_regressor


@HEAD.register_module
class ManoBranch(nn.Module):
    def __init__(self, **cfg):
        """
        Args:
            mano_root (path): dir containing mano pickle files
            center_idx: Joint idx on which to hand is centered (given joint has position
                [0, 0, 0]
            ncomps: Number of pose principal components that are predicted
        """
        super(ManoBranch, self).__init__()

        self.inp_dim = cfg["INPUT_DIM"]
        self.ncomps = cfg["NCOMPS"]
        self.use_pca = cfg["USE_PCA"]
        self.center_idx = cfg["CENTER_IDX"]
        self.mano_assets_root = cfg["MANO_ASSETS_ROOT"]
        self.use_shape = cfg.get("USE_SHAPE", True)

        self.mano_side = CONST.SIDE

        base_neurons = [self.inp_dim, 512, 512]

        if self.use_pca:
            # Final number of coefficients to predict for pose
            # is sum of PCA components and 3 global axis-angle params
            # for the global rotation
            mano_pose_size = self.ncomps + 3
        else:
            # 15 joints + 1 global rotations, 9 components per joint
            # rotation
            mano_pose_size = 16 * 9
        # Initial base layers of MANO decoder
        base_layers = []
        for inp_neurons, out_neurons in zip(base_neurons[:-1], base_neurons[1:]):
            base_layers.append(nn.Linear(inp_neurons, out_neurons))
            base_layers.append(nn.ReLU())
        self.base_layer = nn.Sequential(*base_layers)

        # Pose layers to predict pose parameters
        self.pose_reg = nn.Linear(base_neurons[-1], mano_pose_size)
        if not self.use_pca:
            # Initialize all nondiagonal items on rotation matrix weights to 0
            self.pose_reg.bias.data.fill_(0)
            weight_mask = self.pose_reg.weight.data.new(np.identity(3)).view(9).repeat(16)
            self.pose_reg.weight.data = torch.abs(
                weight_mask.unsqueeze(1).repeat(1, 256).float() * self.pose_reg.weight.data
            )

        # Shape layers to predict MANO shape parameters
        if self.use_shape:
            self.shape_reg = torch.nn.Sequential(nn.Linear(base_neurons[-1], 10))

        # Mano layer which outputs the hand mesh given the hand pose and shape
        # paramters
        self.mano_layer = ManoLayer(
            ncomps=self.ncomps,
            center_idx=self.center_idx,
            side=self.mano_side,
            mano_assets_root=self.mano_assets_root,
            use_pca=self.use_pca,
            flat_hand_mean=False,
        )
        self.faces = self.mano_layer.th_faces
        logger.info(f"{type(self).__name__} uses center_idx {self.center_idx}")

    def forward(self, featrue: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.base_layer(featrue)  # TENSOR (B, 512)
        pose = self.pose_reg(x)  # TENSOR (B, N_PCA)

        # Get shape
        if self.use_shape:
            shape = self.shape_reg(x)
        else:
            shape = None

        mano_out: MANOOutput = self.mano_layer(pose, shape)

        # Gather results in metric space (vs MANO millimeter outputs)
        # pose: the 18 ncomps (3 global rot + 15 pca hand pose)
        # full_pose: the 48 (16 * 3) full relative axis-angles of all 16 joints rotations (from root to finger)
        mano_branch_results = {
            "hand_verts_3d": mano_out.verts,
            "joints_3d": mano_out.joints,
            "mano_shape": shape,
            "mano_pca_pose": pose,
            "mano_full_pose": mano_out.full_poses,
        }

        return mano_branch_results
