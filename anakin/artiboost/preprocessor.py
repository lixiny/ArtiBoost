import torch
import torch.nn as nn
import numpy as np
from manotorch.manolayer import ManoLayer, MANOOutput
from anakin.utils.transform import aa_to_rotmat, rotmat_to_aa, th_homogeneous

from .refiner import Refiner
from .scrambler import Scrambler


class PreProcessorPoseGenerator(nn.Module):

    def __init__(self, refiner, scrambler, ge_mano_layer, rf_mano_layer):
        super().__init__()
        self.refiner: Refiner = refiner
        self.scrambler: Scrambler = scrambler
        self.ge_mano_layer: ManoLayer = ge_mano_layer
        self.rf_mano_layer: ManoLayer = rf_mano_layer

    def forward(self, synth_extend):
        obj_id = synth_extend["obj_id"]
        obj_name = synth_extend["obj_name"]

        # * ======= decode raw mano param from grasp engine >>>>>>>
        mano_out: MANOOutput = self.ge_mano_layer(synth_extend["hand_pose"], synth_extend["hand_shape"])
        hand_glob_rotmat = mano_out.transforms_abs[:, 0, :3, :3]  # (B, 3, 3)
        hand_verts = mano_out.verts + synth_extend["hand_tsl"].unsqueeze(1)  # (B, 778, 3)
        joints = mano_out.joints + synth_extend["hand_tsl"].unsqueeze(1)  # (B, 21, 3)
        synth_extend["hand_pose"] = mano_out.full_poses  # replace !

        # * ======= decode view param from view engine >>>>>>>
        persp_rotmat_inv = synth_extend["persp_rotmat"].permute(0, 2, 1)  # (B, 3, 3)
        camera_free_trasnf = synth_extend["camera_free_transf"]  # (B, 4, 4)
        camera_free_rotmat = camera_free_trasnf[:, :3, :3]  # (B, 3, 3)

        #  (B, 3, 3) bmm (B, 3, 1) -> (B, 3)  #  middle point of object center, hand joint center
        op_offset = (torch.bmm(persp_rotmat_inv, joints[:, 9, :].unsqueeze(-1)).squeeze(-1) / 2.0)
        cam_sys_offset = synth_extend["z_offset"] - op_offset  # (B, 3)
        obj_pose = th_homogeneous(torch.cat([persp_rotmat_inv, cam_sys_offset.unsqueeze(2)], dim=2).view(-1, 3, 4))
        obj_pose = torch.bmm(synth_extend["camera_free_transf"], obj_pose)  # (B, 4, 4)

        # * ======= retrieve the new hand pose >>>>>>
        hand_pose = synth_extend["hand_pose"]
        hand_shape = synth_extend["hand_shape"]
        hand_tsl = synth_extend["hand_tsl"]

        new_hand_verts = torch.bmm(persp_rotmat_inv, hand_verts.transpose(2, 1)).transpose(2, 1)
        new_joints = torch.bmm(persp_rotmat_inv, joints.transpose(2, 1)).transpose(2, 1)
        new_hand_glob_rotmat = torch.bmm(persp_rotmat_inv, hand_glob_rotmat)  # (B, 3, 3)
        new_hand_glob_pose = rotmat_to_aa(new_hand_glob_rotmat)  # [B, 3]

        new_hand_pose = torch.cat([new_hand_glob_pose, hand_pose[:, 3:]], dim=1)  # (B, 16x3)

        # see https://github.com/lixiny/manotorch/blob/master/manotorch/manolayer.py for more details
        mano_rotation_center = self.rf_mano_layer.get_rotation_center(hand_shape)  # (B, 3)
        root_rot = aa_to_rotmat(hand_pose[:, :3])  # (B, 3, 3)
        offset_0 = mano_rotation_center - torch.bmm(root_rot, mano_rotation_center.unsqueeze(-1)).squeeze(-1)
        new_root_rot = aa_to_rotmat(new_hand_pose[:, :3])  # (B, 3, 3)
        offset_1 = mano_rotation_center - torch.bmm(new_root_rot, mano_rotation_center.unsqueeze(-1)).squeeze(-1)
        new_hand_tsl = torch.bmm(persp_rotmat_inv, (offset_0 + hand_tsl).unsqueeze(-1)).squeeze(-1) - offset_1

        out_ = self.rf_mano_layer(new_hand_pose, hand_shape)  # inorder to get the hand_transf
        new_hand_transf = out_.transforms_abs  # (B, 16, 4, 4)

        # * ======= scrambler >>>>>>
        scrambler_feed = {
            "hand_pose": new_hand_pose,
            "hand_tsl": new_hand_tsl,
            "joints": new_joints,
            "hand_verts": new_hand_verts,
            "hand_transf": new_hand_transf,
        }
        scrambler_res = self.scrambler(scrambler_feed)

        # * ======= refiner >>>>>>
        refiner_feed = {
            "hand_pose": scrambler_res["hand_pose"],
            "hand_tsl": scrambler_res["hand_tsl"],
            "obj_rot": obj_pose[:, :3, :3],
        }
        refiner_res = self.refiner(refiner_feed, obj_name)

        # * apply camera system offset and camera free transf back on refined hand
        final_hand_verts = refiner_res["hand_verts"] + cam_sys_offset.unsqueeze(1)  # (B, 778, 3)
        final_joints = refiner_res["joints"] + cam_sys_offset.unsqueeze(1)  # (B, 21, 3)

        final_hand_verts = torch.bmm(camera_free_rotmat, final_hand_verts.transpose(2, 1)).transpose(2, 1)
        final_joints = torch.bmm(camera_free_rotmat, final_joints.transpose(2, 1)).transpose(2, 1)

        return {
            "index": synth_extend["index"],
            "obj_id": synth_extend["obj_id"],
            "obj_name": synth_extend["obj_name"],
            "persp_id": synth_extend["persp_id"],
            "grasp_id": synth_extend["grasp_id"],
            "final_obj_pose": obj_pose,
            "final_hand_verts": final_hand_verts,
            "final_joints": final_joints,
        }
