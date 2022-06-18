import os
import pickle
from abc import ABC
from typing import List, Optional

import numpy as np
import torch
from anakin.utils.transform import aa_to_rotmat, rotmat_to_aa
from manotorch.manolayer import ManoLayer, MANOOutput


class GraspEngine(ABC):

    @staticmethod
    def build(dataset_type: str, obj_names: List[str]):
        if dataset_type == "HO3D":
            return HO3DGraspEngine("assets/grasp_engine/ycb_grasp", obj_names)
        elif dataset_type == "DexYCB":
            return DexYCBGraspEngine("assets/grasp_engine/ycb_grasp", obj_names)
        else:
            raise NotImplementedError()

    def __init__(self, grasp_dir: str, obj_names: List[str]):

        self._obj_names = obj_names
        self.obj_grasps = {}

        for obj_name in obj_names:
            grasp_path = os.path.join(grasp_dir, obj_name + ".pkl")
            with open(grasp_path, "rb") as stream:
                grasps = pickle.load(stream)
            self.obj_grasps[obj_name] = grasps

        self.mano_layer: ManoLayer = None

    @property
    def obj_names(self):
        return self._obj_names

    @obj_names.setter
    def obj_names(self, value):
        self._obj_names = value

    def has_obj(self, name: str):
        return name in self.obj_names

    def get_obj_grasp(self, obj_name: str, grasp_idx: int):
        hand_pose, hand_shape, hand_tsl = self.obj_grasps[obj_name][grasp_idx]
        if not hand_shape:
            hand_shape = np.zeros(10)
        if (isinstance(hand_tsl, (int, float)) and hand_tsl == 0) or hand_tsl is None:
            hand_tsl = np.zeros(3)
        return hand_pose, hand_shape, hand_tsl

    def get_mapping_len(self):
        return {n: len(v) for n, v in self.obj_grasps.items()}

    def decode_full_hand(self, blob):
        hand_pose, hand_shape, hand_tsl = blob
        if not hand_shape:
            th_hand_shape = torch.zeros((1, 10)).float()
        else:
            th_hand_shape = torch.from_numpy(hand_shape).float().unsqueeze(0)
        th_hand_pose = torch.from_numpy(hand_pose).float().unsqueeze(0)
        th_hand_tsl = torch.from_numpy(hand_tsl).float().unsqueeze(0)
        manooutput: MANOOutput = self.mano_layer(th_hand_pose, th_hand_shape)

        hand_verts = manooutput.verts.squeeze(0)
        hand_verts = hand_verts + th_hand_tsl
        hand_verts = hand_verts.numpy()
        hand_joints = manooutput.joints.squeeze(0)
        hand_joints = hand_joints + th_hand_tsl
        hand_joints = hand_joints.numpy()
        full_transf = manooutput.transforms_abs.squeeze(0).numpy()
        axis_angle_hand_pose = manooutput.full_poses.squeeze(0).numpy()
        return {
            "hand_verts": hand_verts,
            "hand_joints": hand_joints,
            "root_transf": full_transf[0, ...],
            "h_pose": axis_angle_hand_pose,
            "h_shape": hand_shape,
            "h_tsl": hand_tsl,
        }


class HO3DGraspEngine(GraspEngine):

    def __init__(self, grasp_dir: str, obj_names: List[str]):
        super().__init__(grasp_dir, obj_names)
        self.mano_layer = ManoLayer(
            mano_assets_root="assets/mano_v1_2",
            side="right",
            use_pca=False,
            flat_hand_mean=True,
        )


class DexYCBGraspEngine(GraspEngine):

    def __init__(self, grasp_dir: str, obj_names: List[str]):
        super().__init__(grasp_dir, obj_names)
        self.cam_extr = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float32,
        )

        class InnerManoLayer(ManoLayer):

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.register_buffer("th_cam_extr", torch.Tensor(kwargs["cam_extr"]).unsqueeze(0))  # (1, 3, 3)

            def forward(self, pose_coeffs: torch.Tensor, betas: Optional[torch.Tensor] = None, **kwargs):
                hand_tsl = kwargs["hand_tsl"]  # (B, 3)
                batch_size = pose_coeffs.shape[0]
                glob_pose, rel_pose = pose_coeffs[:, :3], pose_coeffs[:, 3:]
                flip_glob_rotmat = torch.bmm(self.th_cam_extr, aa_to_rotmat(glob_pose))
                flip_glob_pose = rotmat_to_aa(flip_glob_rotmat)  # (B, 3)
                flip_hand_pose = (torch.cat([flip_glob_pose, rel_pose], dim=1),)  # (B, 48)

                gathered_hand_pose = torch.cat([pose_coeffs, flip_hand_pose], dim=0).float()  # (2B, 48)
                if betas is not None:
                    gathered_hand_shape = betas.repeat(2, 1)  # (2B, 10)
                else:
                    gathered_hand_shape = None

                gathered_mano_out: MANOOutput = super().forward(gathered_hand_pose, gathered_hand_shape)
                gathered_joints = gathered_mano_out.joints
                fitted_hand_tsl = torch.bmm(self.th_cam_extr,
                                            (gathered_joints[:batch_size, 0] + hand_tsl).unsqueeze(-1))
                fitted_hand_tsl = fitted_hand_tsl.squeeze(-1) - gathered_joints[batch_size:, 0]  # (B ,3)

                mano_out = MANOOutput(
                    verts=gathered_mano_out.verts[batch_size:] + fitted_hand_tsl - hand_tsl,
                    joints=gathered_mano_out.joints[batch_size:] + fitted_hand_tsl - hand_tsl,
                    center_idx=gathered_mano_out.center_idx[batch_size:],
                    center_joint=gathered_mano_out.center_idx[batch_size:],
                    full_poses=flip_hand_pose,
                    betas=gathered_mano_out.betas[batch_size:],
                    transforms_abs=gathered_mano_out.transforms_abs[batch_size:],
                )

                return mano_out

        self.mano_layer = InnerManoLayer(
            mano_assets_root="assets/mano_v1_2",
            side="right",
            use_pca=False,
            flat_hand_mean=True,
            cam_extr=self.cam_extr,
        )

    def decode_full_hand(self, blob):
        hand_pose, hand_shape, hand_tsl = blob
        if not hand_shape:
            th_hand_shape = torch.zeros((2, 10)).float()
        else:
            th_hand_shape = torch.from_numpy(hand_shape).float().unsqueeze(0).expand(2, -1)
        cam_extr = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)
        pose_flip = np.concatenate([rotmat_to_aa(cam_extr @ aa_to_rotmat(hand_pose[:3])), hand_pose[3:]])

        th_hand_pose = torch.from_numpy(np.concatenate([hand_pose[None], pose_flip[None]])).float()
        mano_out: MANOOutput = self.mano_layer(th_hand_pose, th_hand_shape)
        hand_verts = mano_out.verts.numpy()
        hand_joints = mano_out.joints.numpy()
        full_transf, axis_angle_hand_pose = mano_out.transforms_abs, mano_out.full_poses

        fit_hand_tsl = (cam_extr @ (hand_joints[0][0] + hand_tsl).T).T - hand_joints[1][0]

        hand_verts = hand_verts[1]
        hand_verts = hand_verts + fit_hand_tsl
        hand_joints = hand_joints[1]
        hand_joints = hand_joints + fit_hand_tsl
        full_transf = full_transf[1].numpy()
        axis_angle_hand_pose = axis_angle_hand_pose[1].numpy()
        return {
            "hand_verts": hand_verts,
            "hand_joints": hand_joints,
            "root_transf": full_transf[0, ...],
            "h_pose": axis_angle_hand_pose,
            "h_shape": hand_shape,
            "h_tsl": fit_hand_tsl,
        }
