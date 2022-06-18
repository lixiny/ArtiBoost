from typing import Callable, Mapping

import torch
from anakin.utils.transform import aa_to_rotmat, rotmat_to_aa
from manotorch.axislayer import AxisLayer
from torch import nn
from torch.distributions.normal import Normal


def register(reg, key):

    def fn(cls):
        reg[key] = cls
        return cls

    return fn


def axis_angle_op(aa_1, aa_2):
    # aa_1 [B, X, 3],  aa_2 [B, X, 3]
    batch_size = aa_1.shape[0]
    num_joint = aa_1.shape[1]
    rotmat_1 = aa_to_rotmat(aa_1).reshape(-1, 3, 3)  # [B * X, 3, 3]
    rotmat_2 = aa_to_rotmat(aa_2).reshape(-1, 3, 3)  # [B * X, 3, 3]
    comb_rotmat = torch.matmul(rotmat_1, rotmat_2)  # [B * X, 3, 3]
    comb_aa = rotmat_to_aa(comb_rotmat.reshape(batch_size, num_joint, 3, 3))
    return comb_aa


class Scrambler(nn.Module):
    build_mapping: Mapping[str, Callable] = {}

    @staticmethod
    def build(type, *args, **kwargs) -> "Scrambler":
        return Scrambler.build_mapping[type](*args, **kwargs)


@register(reg=Scrambler.build_mapping, key="naive")
class NaiveScrambler(Scrambler):

    def __init__(self, cfg) -> None:
        super(NaiveScrambler, self).__init__()
        self.hand_tsl_dist = Normal(torch.tensor(0.0), torch.tensor(cfg["HAND_TSL_SIGMA"]))

    def forward(self, inp):
        init_hand_pose = inp["hand_pose"]
        init_hand_tsl = inp["hand_tsl"]

        batch_size = init_hand_pose.shape[0]
        rand_hand_tsl = self.hand_tsl_dist.sample((batch_size, 3))

        hand_tsl = init_hand_tsl + rand_hand_tsl

        return {"hand_pose": init_hand_pose, "hand_tsl": hand_tsl}


@register(reg=Scrambler.build_mapping, key="random")
class RandomScrambler(Scrambler):

    def __init__(self, cfg) -> None:
        super(RandomScrambler, self).__init__()
        self.hand_tsl_dist = Normal(torch.tensor(0.0), torch.tensor(cfg["HAND_TSL_SIGMA"]))
        self.hand_pose_dist = Normal(torch.tensor(0.0), torch.tensor(cfg["HAND_POSE_SIGMA"]))

    def forward(self, inp, **kwargs):
        init_hand_pose = inp["hand_pose"]
        init_hand_tsl = inp["hand_tsl"]
        batch_size = init_hand_pose.shape[0]
        device = init_hand_pose.device

        rand_hand_tsl = self.hand_tsl_dist.sample((batch_size, 3)).to(device)
        rand_pose_angle = self.hand_pose_dist.sample((batch_size, 16)).to(device)

        hand_pose = init_hand_pose.reshape(-1, 16, 3)
        pose_axis = hand_pose / torch.clip(torch.norm(hand_pose, dim=-1, keepdim=True), min=1e-7)  # [B, 16, 3]
        pose_angle = torch.norm(hand_pose, dim=-1) + rand_pose_angle  # [B, 16] NOTE: might be negative
        hand_pose = pose_axis * pose_angle.unsqueeze(-1)
        hand_pose = hand_pose.reshape(-1, 48)
        hand_tsl = init_hand_tsl + rand_hand_tsl

        return {"hand_pose": hand_pose, "hand_tsl": hand_tsl}


@register(reg=Scrambler.build_mapping, key="random_2")
class RandomScrambler2(Scrambler):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.hand_tsl_dist = Normal(torch.tensor(0.0), torch.tensor(cfg["HAND_TSL_SIGMA"]))
        self.hand_pose_dist = Normal(torch.tensor(0.0), torch.tensor(cfg["HAND_POSE_SIGMA"]))
        self.coef_1 = 1.1
        self.coef_2 = 0.9
        self.axis_layer = AxisLayer()

    def forward(self, inp, **kwargs):
        init_hand_pose = inp["hand_pose"]
        init_hand_tsl = inp["hand_tsl"]
        init_hand_verts = inp["hand_verts"]
        init_joints = inp["joints"]
        init_hand_transf = inp["hand_transf"]
        assert init_hand_transf is not None

        batch_size = init_hand_pose.shape[0]
        device = init_hand_pose.device

        # figure out axis
        # **** axis order right hand

        #         14-13-12-\
        #                   \
        #    2-- 1 -- 0 -----*
        #   5 -- 4 -- 3 ----/
        #   11 - 10 - 9 ---/
        #    8-- 7 -- 6 --/

        b_axis, u_axis, l_axis = self.axis_layer(init_joints, init_hand_transf)

        rand_hand_tsl = self.hand_tsl_dist.sample((batch_size, 3)).to(device)
        # rand_hand_tsl = 0.0

        hand_pose = init_hand_pose.reshape(-1, 16, 3)

        # perturb <0, 3, 6, 9> -> joiny <1, 4, 7, 10>
        rand_splay_angle = self.hand_pose_dist.sample((batch_size, 4)).to(device)  # [B, 4]
        # rand_splay_angle = torch.zeros((batch_size, 4))
        # print(rand_splay_angle)
        rand_splay_aa = u_axis[:, (0, 3, 6, 9), :] * rand_splay_angle.unsqueeze(2)  # [B, 4, 3]
        splay_modifiy_pose = hand_pose[:, (1, 4, 7, 10), :].detach().clone()  # [B, 4, 3]
        splay_modifiy_pose = axis_angle_op(splay_modifiy_pose, rand_splay_aa)
        hand_pose[:, (1, 4, 7, 10), :] = splay_modifiy_pose

        # perturb <0, 1, 2; 3, 4, 5; 6, 7, 8; 9, 10, 11>
        # joint <1, 2, 3; 4, 5, 6,; 7, 8, 9; 10, 11, 12>
        # rand_bend_angle = torch.zeros((batch_size, 5)).to(mano_device)
        rand_bend_angle = self.hand_pose_dist.sample((batch_size, 5)).to(device)  # [B, 4]
        interlink_vec = torch.tensor([1.0, self.coef_1, self.coef_2]).to(device)  # [3, ]
        interlink_vec = interlink_vec.expand(batch_size, -1)
        # print(rand_bend_angle)

        index_rand_bend_angle = rand_bend_angle[:, 0:1] * interlink_vec  # [B, 3]
        index_bend_axis = l_axis[:, (0, 1, 2), :]  # [B, 3, 3]
        index_rand_bend_aa = index_bend_axis * index_rand_bend_angle.unsqueeze(2)  # [B, 3, 3]
        bend_index_modify_pose = hand_pose[:, (1, 2, 3), :].detach().clone()  # [B, 3, 3]
        bend_index_modify_pose = axis_angle_op(index_rand_bend_aa, bend_index_modify_pose)
        hand_pose[:, (1, 2, 3), :] = bend_index_modify_pose

        middle_rand_bend_angle = rand_bend_angle[:, 1:2] * interlink_vec  # [B, 3]
        middle_bend_axis = l_axis[:, (3, 4, 5), :]  # [B, 3, 3]
        middle_rand_bend_aa = middle_bend_axis * middle_rand_bend_angle.unsqueeze(2)  # [B, 3, 3]
        bend_middle_modify_pose = hand_pose[:, (4, 5, 6), :].detach().clone()  # [B, 3, 3]
        bend_middle_modify_pose = axis_angle_op(middle_rand_bend_aa, bend_middle_modify_pose)
        hand_pose[:, (4, 5, 6), :] = bend_middle_modify_pose

        ring_rand_bend_angle = rand_bend_angle[:, 2:3] * interlink_vec  # [B, 3]
        ring_bend_axis = l_axis[:, (9, 10, 11), :]  # [B, 3, 3]
        ring_rand_bend_aa = ring_bend_axis * ring_rand_bend_angle.unsqueeze(2)  # [B, 3, 3]
        bend_ring_modify_pose = hand_pose[:, (10, 11, 12), :].detach().clone()  # [B, 3, 3]
        bend_ring_modify_pose = axis_angle_op(ring_rand_bend_aa, bend_ring_modify_pose)
        hand_pose[:, (10, 11, 12), :] = bend_ring_modify_pose

        little_rand_bend_angle = rand_bend_angle[:, 3:4] * interlink_vec  # [B, 3]
        little_bend_axis = l_axis[:, (6, 7, 8), :]  # [B, 3, 3]
        little_rand_bend_aa = little_bend_axis * little_rand_bend_angle.unsqueeze(2)  # [B, 3, 3]
        bend_little_modify_pose = hand_pose[:, (7, 8, 9), :].detach().clone()  # [B, 3, 3]
        bend_little_modify_pose = axis_angle_op(little_rand_bend_aa, bend_little_modify_pose)
        hand_pose[:, (7, 8, 9), :] = bend_little_modify_pose

        thumb_rand_bend_angle = rand_bend_angle[:, 4:5] * interlink_vec[:, (0, 2)]  # [B, 3]
        thumb_bend_axis = l_axis[:, (13, 14), :]  # [B, 3, 3]
        thumb_rand_bend_aa = thumb_bend_axis * thumb_rand_bend_angle.unsqueeze(2)  # [B, 3, 3]
        bend_thumb_modify_pose = hand_pose[:, (14, 15), :].detach().clone()  # [B, 3, 3]
        bend_thumb_modify_pose = axis_angle_op(thumb_rand_bend_aa, bend_thumb_modify_pose)
        hand_pose[:, (14, 15), :] = bend_thumb_modify_pose

        # thumb, perturb 12 -> joint 13
        thumb_rand_other_angle = self.hand_pose_dist.sample((batch_size, 2)).to(device)  # [B, 2]
        # print(thumb_rand_other_angle[1, ...])
        other_bend_axis = l_axis[:, (12,), :]  # [B, 1, 3]
        other_splay_axis = u_axis[:, (12,), :]  # [B, 1, 3]
        other_bend_axis_aa = other_bend_axis * thumb_rand_other_angle[:, (0,), None]  # [B, 1, 3]
        other_splay_axis_aa = other_splay_axis * thumb_rand_other_angle[:, (1,), None]  # [B, 1, 3]
        other_modify_pose = hand_pose[:, (13,), :].detach().clone()  # [B, 1, 3]
        other_modify_pose = axis_angle_op(other_splay_axis_aa, axis_angle_op(other_bend_axis_aa, other_modify_pose))
        hand_pose[:, (13,), :] = other_modify_pose

        hand_pose = hand_pose.reshape(-1, 48)
        hand_tsl = init_hand_tsl + rand_hand_tsl

        return {"hand_pose": hand_pose, "hand_tsl": hand_tsl}


@register(reg=Scrambler.build_mapping, key="random_3")
class RandomScrambler3(Scrambler):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.hand_tsl_dist = Normal(torch.tensor(0.0), torch.tensor(cfg["HAND_TSL_SIGMA"]))
        self.hand_pose_dist = Normal(torch.tensor(0.0), torch.tensor(cfg["HAND_POSE_SIGMA"]))
        self.axis_layer = AxisLayer()

    def forward(self, inp, **kwargs):
        init_hand_pose = inp["hand_pose"]
        init_hand_tsl = inp["hand_tsl"]
        init_hand_verts = inp["hand_verts"]
        init_joints = inp["joints"]
        init_hand_transf = inp["hand_transf"]
        assert init_hand_transf is not None

        batch_size = init_hand_pose.shape[0]
        # figure out axis
        # **** axis order right hand

        #         14-13-12-\
        #                   \
        #    2-- 1 -- 0 -----*
        #   5 -- 4 -- 3 ----/
        #   11 - 10 - 9 ---/
        #    8-- 7 -- 6 --/
        b_axis, u_axis, l_axis = self.axis_layer(init_joints, init_hand_transf)

        rand_hand_tsl = self.hand_tsl_dist.sample((batch_size, 3))
        # rand_hand_tsl = 0.0

        hand_pose = init_hand_pose.reshape(-1, 16, 3)

        # perturb <0, 3, 6, 9> -> joiny <1, 4, 7, 10>
        rand_splay_angle = self.hand_pose_dist.sample((batch_size, 4))  # [B, 4]
        # rand_splay_angle = torch.zeros((batch_size, 4))
        # print(rand_splay_angle)
        rand_splay_aa = u_axis[:, (0, 3, 6, 9), :] * rand_splay_angle.unsqueeze(2)  # [B, 4, 3]
        splay_modifiy_pose = hand_pose[:, (1, 4, 7, 10), :].detach().clone()  # [B, 4, 3]
        splay_modifiy_pose = axis_angle_op(splay_modifiy_pose, rand_splay_aa)
        hand_pose[:, (1, 4, 7, 10), :] = splay_modifiy_pose

        # perturb <0, 1, 2; 3, 4, 5; 6, 7, 8; 9, 10, 11>
        # joint <1, 2, 3; 4, 5, 6,; 7, 8, 9; 10, 11, 12>
        # rand_bend_angle = torch.zeros((batch_size, 5)).to(mano_device)
        rand_bend_angle = self.hand_pose_dist.sample((batch_size, 14))  # [B, 14]
        bend_axis = l_axis[:, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14), :]  # [B, 14, 3]
        bend_aa = bend_axis * rand_bend_angle.unsqueeze(2)  # [B, 14, 3]
        bend_modify_pose = (hand_pose[:, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15), :].detach().clone()
                           )  # [B, 14, 3]
        bend_modify_pose = axis_angle_op(bend_aa, bend_modify_pose)
        hand_pose[:, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15), :] = bend_modify_pose

        # thumb, perturb 12 -> joint 13
        thumb_rand_other_angle = self.hand_pose_dist.sample((batch_size, 2))  # [B, 2]
        # print(thumb_rand_other_angle[1, ...])
        other_bend_axis = l_axis[:, (12,), :]  # [B, 1, 3]
        other_splay_axis = u_axis[:, (12,), :]  # [B, 1, 3]
        other_bend_axis_aa = other_bend_axis * thumb_rand_other_angle[:, (0,), None]  # [B, 1, 3]
        other_splay_axis_aa = other_splay_axis * thumb_rand_other_angle[:, (1,), None]  # [B, 1, 3]
        other_modify_pose = hand_pose[:, (13,), :].detach().clone()  # [B, 1, 3]
        other_modify_pose = axis_angle_op(other_splay_axis_aa, axis_angle_op(other_bend_axis_aa, other_modify_pose))
        hand_pose[:, (13,), :] = other_modify_pose

        hand_pose = hand_pose.reshape(-1, 48)
        hand_tsl = init_hand_tsl + rand_hand_tsl

        return {"hand_pose": hand_pose, "hand_tsl": hand_tsl}
