from copy import deepcopy
from typing import Callable, Dict, List, Mapping

import numpy as np
import torch
import trimesh
from anakin.utils.transform import aa_to_rotmat, rotmat_to_aa
from manotorch.manolayer import ManoLayer, MANOOutput
from torch import nn


def register(reg, key):

    def fn(cls):
        reg[key] = cls
        return cls

    return fn


def point2point_signed(x, y, x_normals=None, y_normals=None):
    """
    signed distance between two pointclouds

    Args:
        x: FloatTensor of shape (N, P1, D) representing a batch of point clouds
            with P1 points in each batch element, batch size N and feature
            dimension D.
        y: FloatTensor of shape (N, P2, D) representing a batch of point clouds
            with P2 points in each batch element, batch size N and feature
            dimension D.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).

    Returns:

        - y2x_signed: Torch.Tensor
            the sign distance from y to x
        - y2x_signed: Torch.Tensor
            the sign distance from y to x
        - yidx_near: Torch.tensor
            the indices of x vertices closest to y

    """
    import chamfer_distance as chd

    N, P1, D = x.shape
    P2 = y.shape[1]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")

    ch_dist = chd.ChamferDistance()

    x_near, y_near, xidx_near, yidx_near = ch_dist(x, y)

    xidx_near_expanded = xidx_near.view(N, P1, 1).expand(N, P1, D).to(torch.long)
    x_near = y.gather(1, xidx_near_expanded)

    yidx_near_expanded = yidx_near.view(N, P2, 1).expand(N, P2, D).to(torch.long)
    y_near = x.gather(1, yidx_near_expanded)

    x2y = x - x_near
    y2x = y - y_near

    # * We don't need this
    # if x_normals is not None:
    #     y_nn = x_normals.gather(1, yidx_near_expanded)
    #     in_out = torch.bmm(y_nn.view(-1, 1, 3), y2x.view(-1, 3, 1)).view(N, -1).sign()
    #     y2x_signed = y2x.norm(dim=2) * in_out

    # else:
    #     y2x_signed = y2x.norm(dim=2)

    if y_normals is not None:
        x_nn = y_normals.gather(1, xidx_near_expanded)
        in_out_x = torch.bmm(x_nn.view(-1, 1, 3), x2y.view(-1, 3, 1)).view(N, -1).sign()
        x2y_signed = x2y.norm(dim=2) * in_out_x
    else:
        x2y_signed = x2y.norm(dim=2)

    # return y2x_signed, x2y_signed, yidx_near
    return x2y_signed


def CRot2rotmat(pose):

    reshaped_input = pose.view(-1, 3, 2)

    b1 = nn.functional.normalize(reshaped_input[:, :, 0], dim=1)

    dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
    b2 = nn.functional.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=1)

    return torch.stack([b1, b2, b3], dim=-1)


def parms_decode(pose_crot, trans):
    bs = trans.shape[0]
    pose_rotmat = CRot2rotmat(pose_crot)  # (bsx16, 3, 3)
    pose = rotmat_to_aa(pose_rotmat).view(bs, -1)  # (bs, 48)
    hand_parms = {"th_pose_coeffs": pose, "th_tsl": trans}
    return hand_parms


class Refiner(nn.Module):
    build_mapping: Mapping[str, Callable] = {}

    @staticmethod
    def build(type, *args, **kwargs) -> "Refiner":
        return Refiner.build_mapping[type](*args, **kwargs)

    def __init__(self):
        super().__init__()


@register(reg=Refiner.build_mapping, key="hand_obj")
class HORefiner(nn.Module):

    def __init__(self, cfg):
        super(HORefiner, self).__init__()
        rnet_ckp = cfg["PRETRAINED"]
        n_iter = cfg["ITERS"]
        self.refine_net = _RefineNet(n_iters=n_iter)
        self.refine_net.load_state_dict(torch.load(rnet_ckp, map_location=torch.device("cpu")), strict=False)
        self.refine_net.eval()
        self.resampled_objs = []
        self.obj_idx = {}

    def setup(self, obj_meshes: Dict[str, trimesh.base.Trimesh]):
        for name, m in obj_meshes.items():
            self.obj_idx[name] = len(self.obj_idx)
            self.resampled_objs.append(torch.Tensor(self.resample_obj(m)).float())
        self.resampled_objs = torch.stack(self.resampled_objs)
        self.register_buffer("resampled_objs_buffer", self.resampled_objs)  # let torch manage its device

    @staticmethod
    def resample_obj(obj_mesh: trimesh.base.Trimesh, n_sample_verts: int = 10000):
        mesh = deepcopy(obj_mesh)

        while mesh.vertices.shape[0] < n_sample_verts:
            mesh = mesh.subdivide()

        verts_obj = mesh.vertices
        verts_sample_id = np.random.choice(verts_obj.shape[0], n_sample_verts, replace=False)
        verts_sampled = verts_obj[verts_sample_id]
        return verts_sampled

    def forward(self, inp, obj_name: List[str]):
        # preprare
        hand_pose = inp["hand_pose"]  # (B, 16x3)
        bs = hand_pose.shape[0]

        hand_tsl = inp["hand_tsl"]
        hand_rotmat = aa_to_rotmat(hand_pose.reshape(bs, -1, 3))

        hand_glob_rotmat = hand_rotmat[:, 0, ...]  # [B, 3, 3]
        hand_rel_rotmat = hand_rotmat[:, 1:, ...]  # (B, 15, 3, 3)

        mano_out: MANOOutput = self.refine_net.mano_layer(hand_pose)
        hand_verts = mano_out.verts + hand_tsl.unsqueeze(1)

        obj_rot = inp["obj_rot"]  # [B, 3, 3]
        assert (len(obj_name) == obj_rot.shape[0]), \
            f"object name and rotation matrix do not match, got {len(obj_name)} and {obj_rot.shape[0]}"
        obj_idx = [self.obj_idx[name] for name in obj_name]
        verts_object = torch.transpose(torch.bmm(obj_rot, torch.transpose(self.resampled_objs_buffer[obj_idx], -2, -1)),
                                       -2, -1)

        h2o = point2point_signed(hand_verts, verts_object)
        h2o = h2o.abs()  # don't know why to do this

        fitted_hand_param = self.refine_net(
            h2o_dist=h2o,
            fpose_rhand_rotmat_f=hand_rel_rotmat,
            trans_rhand_f=hand_tsl,
            global_orient_rhand_rotmat_f=hand_glob_rotmat,
            verts_object=verts_object,
        )

        fitted_hand_pose = fitted_hand_param["th_pose_coeffs"]
        fitted_hand_tsl = fitted_hand_param["th_tsl"]
        mano_out: MANOOutput = self.refine_net.mano_layer(pose_coeffs=fitted_hand_pose)
        fitted_hand_verts = mano_out.verts + fitted_hand_tsl.unsqueeze(1)
        fitted_joints = mano_out.joints + fitted_hand_tsl.unsqueeze(1)
        return {
            "hand_verts": fitted_hand_verts,
            "joints": fitted_joints,
            "hand_pose": fitted_hand_pose,
            "hand_tsl": fitted_hand_tsl,
        }


class _RefineNet(nn.Module):

    def __init__(self, in_size=778 + 16 * 6 + 3, h_size=512, n_iters=3):

        super(_RefineNet, self).__init__()

        self.n_iters = n_iters
        self.bn1 = nn.BatchNorm1d(778)
        self.rb1 = ResBlock(in_size, h_size)
        self.rb2 = ResBlock(in_size + h_size, h_size)
        self.rb3 = ResBlock(in_size + h_size, h_size)
        self.out_p = nn.Linear(h_size, 16 * 6)
        self.out_t = nn.Linear(h_size, 3)
        self.dout = nn.Dropout(0.3)
        self.actvf = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()

        self.mano_layer = ManoLayer(
            rot_mode="axisang",
            use_pca=False,
            mano_assets_root="assets/mano_v1_2",
            center_idx=None,
            flat_hand_mean=True,
        )

    def forward(self, h2o_dist, fpose_rhand_rotmat_f, trans_rhand_f, global_orient_rhand_rotmat_f, verts_object,
                **kwargs):

        bs = h2o_dist.shape[0]
        init_pose = fpose_rhand_rotmat_f[..., :2].reshape(bs, -1)
        init_rpose = global_orient_rhand_rotmat_f[..., :2].reshape(bs, -1)
        init_pose = torch.cat([init_rpose, init_pose], dim=1)
        init_trans = trans_rhand_f

        for i in range(self.n_iters):

            if i != 0:
                hand_parms = parms_decode(init_pose, init_trans)
                mano_out: MANOOutput = self.mano_layer(hand_parms["th_pose_coeffs"])
                verts_rhand = mano_out.verts + hand_parms["th_tsl"].unsqueeze(1)
                h2o_dist = point2point_signed(verts_rhand, verts_object)

            h2o_dist = self.bn1(h2o_dist)
            X0 = torch.cat([h2o_dist, init_pose, init_trans], dim=1)
            X = self.rb1(X0)
            X = self.dout(X)
            X = self.rb2(torch.cat([X, X0], dim=1))
            X = self.dout(X)
            X = self.rb3(torch.cat([X, X0], dim=1))
            X = self.dout(X)

            pose = self.out_p(X)
            trans = self.out_t(X)

            init_trans = init_trans + trans
            init_pose = init_pose + pose

        hand_parms = parms_decode(init_pose, init_trans)
        return hand_parms


class ResBlock(nn.Module):

    def __init__(self, Fin, Fout, n_neurons=256):

        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)

        self.fc2 = nn.Linear(n_neurons, Fout)
        self.bn2 = nn.BatchNorm1d(Fout)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        Xout = self.fc1(x)  # n_neurons
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)

        Xout = self.fc2(Xout)
        Xout = self.bn2(Xout)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)
        return Xout