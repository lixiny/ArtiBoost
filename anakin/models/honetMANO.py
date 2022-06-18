import os
import pickle
from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
from manotorch.utils.rodrigues import rodrigues

from anakin.datasets.hoquery import Queries
from anakin.models.mano import ManoAdaptor
from anakin.utils.builder import MODEL, build_backbone, build_head
from anakin.utils.transform import batch_persp_proj2d
from anakin.utils.logger import logger
from anakin.utils.misc import CONST, enable_lower_param
from anakin.utils.netutils import recurse_freeze


@MODEL.register_module
class HoNet(nn.Module):
    @enable_lower_param
    def __init__(self, **cfg):
        super(HoNet, self).__init__()

        if cfg["BACKBONE"]["PRETRAINED"] and cfg["PRETRAINED"]:
            logger.warning(
                f"{type(self).__name__}'s backbone {cfg['BACKBONE']['TYPE']} "
                f"weights will be rewritten by {cfg['PRETRAINED']}"
            )

        self.inp_res = cfg["DATA_PRESET"]["IMAGE_SIZE"]
        self.feature_dim = cfg["HEAD"]["INPUT_DIM"]
        self.center_idx = cfg["DATA_PRESET"]["CENTER_IDX"]
        logger.info(f"{type(self).__name__} uses center_idx {self.center_idx}")

        self.base_net = build_backbone(cfg["BACKBONE"])  # ResNet18
        self.mano_branch = build_head(cfg["HEAD"], default_args=cfg["DATA_PRESET"])
        self.obj_trans_factor = cfg["OBJ_TRANS_FACTOR"]
        self.obj_scale_factor = cfg["OBJ_SCALE_FACTOR"]
        self.mano_transhead = HoNet.TransHead(inp_dim=self.feature_dim, out_dim=3)
        self.obj_transhead = HoNet.TransHead(inp_dim=self.feature_dim, out_dim=6)
        self.proj2d_func = batch_persp_proj2d

        if cfg.get("MANO_FHB_ADAPTOR", False):
            mano_fhb_adaptor_dir = cfg["MANO_FHB_ADAPTOR_DIR"]  # assets/hasson20_assets/mano
            adaptor_path = os.path.join(mano_fhb_adaptor_dir, f"fhb_skel_centeridx{self.center_idx}.pkl")
            with open(adaptor_path, "rb") as p_f:
                exp_data = pickle.load(p_f)
            self.register_buffer("fhb_shape", torch.Tensor(exp_data["shape"]))
            self.adaptor = ManoAdaptor(self.mano_branch.mano_layer, adaptor_path)
            recurse_freeze(self.adaptor)
        else:
            self.adaptor = None

        self.init_weights(pretrained=cfg["PRETRAINED"])

    class TransHead(nn.Module):
        def __init__(self, inp_dim: int, out_dim: int):
            super().__init__()

            if out_dim != 3 and out_dim != 6:
                logger.error(f"Unrecognized TransHead out dim: {out_dim}")
                raise ValueError()

            base_neurons = [inp_dim, int(inp_dim / 2)]
            layers = []
            for (inp_neurons, out_neurons) in zip(base_neurons[:-1], base_neurons[1:]):
                layers.append(nn.Linear(inp_neurons, out_neurons))
                layers.append(nn.ReLU())
            self.final_layer = nn.Linear(out_neurons, out_dim)
            self.decoder = nn.Sequential(*layers)

        def forward(self, inp):
            decoded = self.decoder(inp)
            out = self.final_layer(decoded)
            return out

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # ! remapping STATE_DICT from pretrained model of HASSON[CVPR2020] to our HONet
        need_to_be_remove = []
        need_to_be_insert = {}

        for key in state_dict.keys():
            if "mano_layer_left" in key:
                need_to_be_remove.append(key)

            elif "mano_layer_right" in key:
                need_to_be_remove.append(key)
                new_key = key.replace("mano_layer_right", "mano_layer")
                need_to_be_insert[new_key] = state_dict[key]

            elif "scaletrans_branch_obj" in key:
                need_to_be_remove.append(key)
                new_key = key.replace("scaletrans_branch_obj", "obj_transhead")
                need_to_be_insert[new_key] = state_dict[key]

            elif "scaletrans_branch." in key:
                need_to_be_remove.append(key)
                new_key = key.replace("scaletrans_branch", "mano_transhead")
                need_to_be_insert[new_key] = state_dict[key]

        if len(need_to_be_insert) or len(need_to_be_remove):
            logger.warning("remapping STATE_DICT from pretrained model of HASSON[CVPR2020]")
        state_dict.update(need_to_be_insert)
        for key in need_to_be_remove:
            state_dict.pop(key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    @staticmethod
    def recover_3d_proj(
        objpoints3d: torch.Tensor,
        camintr: torch.Tensor,
        est_scale,
        est_trans,
        input_res,
        off_z=0.4,
    ):
        """
        Given estimated centered points, camera intrinsics and predicted scale and translation
        in pixel world, compute the point coordinates in camera coordinate system
        """
        # Estimate scale and trans between 3D and 2D
        focal = camintr[:, :1, :1]
        batch_size = objpoints3d.shape[0]
        focal = focal.view(batch_size, 1)
        est_scale = est_scale.view(batch_size, 1)
        est_trans = est_trans.view(batch_size, 2)
        # est_scale is homogeneous to object scale change in pixels
        est_Z0 = focal * est_scale + off_z
        cam_centers = camintr[:, :2, 2]
        img_centers = (cam_centers.new(input_res) / 2).view(1, 2).repeat(batch_size, 1)
        est_XY0 = (est_trans + img_centers - cam_centers) * est_Z0 / focal
        est_c3d = torch.cat([est_XY0, est_Z0], -1).unsqueeze(1)  # TENSOR(B, 1, 3)
        recons3d = est_c3d + objpoints3d
        return recons3d, est_c3d

    def recover_mano(self, feature: torch.Tensor, samples: Dict):
        # ============= Get hand joints & verts, centered >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        mano_results = self.mano_branch(feature)

        if self.adaptor:
            """
            HASSON[CVPR2020] MANO Adaptor for FHAB dataset
            """
            verts = mano_results["hand_verts_3d"]
            adapt_joints, _ = self.adaptor(verts)
            adapt_joints = adapt_joints.transpose(1, 2)
            mano_results["joints_3d"] = adapt_joints - adapt_joints[:, self.center_idx].unsqueeze(1)
            mano_results["hand_verts_3d"] = verts - adapt_joints[:, self.center_idx].unsqueeze(1)
        # ============== Recover hand position in camera coordinates >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        scaletrans = self.mano_transhead(feature)  # TENSOR (B, 3)
        trans = scaletrans[:, 1:]  # TENSOR (B, 2)
        scale = scaletrans[:, :1]  # TENSOR (B, 1)

        final_trans = trans.unsqueeze(1) * self.obj_trans_factor
        final_scale = scale.view(scale.shape[0], 1, 1) * self.obj_scale_factor

        height, width = tuple(samples[Queries.IMAGE].shape[2:])
        cam_intr = samples[Queries.CAM_INTR]  # TENSOR(B, 3, 3)

        joints_3d_abs, root_joint = HoNet.recover_3d_proj(
            mano_results["joints_3d"], cam_intr, final_scale, final_trans, input_res=(width, height)
        )
        hand_verts_3d_abs = mano_results["hand_verts_3d"] + root_joint

        joints_2d = self.proj2d_func(joints_3d_abs, cam_intr)
        hand_verts_2d = self.proj2d_func(hand_verts_3d_abs, cam_intr)

        # * @Xinyu: mano_results["recov_joints3d"] = mano_results["joints3d"] + mano_results["hand_center3d"]
        mano_results["joints_2d"] = joints_2d
        mano_results["root_joint"] = root_joint  # ===== To PICR =====
        mano_results["joints_3d_abs"] = joints_3d_abs  # ===== To PICR =====
        mano_results["hand_verts_3d_abs"] = hand_verts_3d_abs  # ===== To PICR =====
        mano_results["hand_verts_2d"] = hand_verts_2d
        mano_results["hand_pred_trans"] = trans
        mano_results["hand_pred_scale"] = scale
        mano_results["hand_trans"] = final_trans
        mano_results["hand_scale"] = final_scale

        return mano_results

    def recover_object(self, feature: torch.Tensor, samples: Dict):
        """
        Compute object vertex and corner positions in camera coordinates by predicting object translation
        and scaling, and recovering 3D positions given known object model
        """
        scaletrans_obj = self.obj_transhead(feature)
        batch_size = scaletrans_obj.shape[0]
        scale = scaletrans_obj[:, :1]
        trans = scaletrans_obj[:, 1:3]
        rotaxisang = scaletrans_obj[:, 3:]

        rotmat = rodrigues(rotaxisang).view(rotaxisang.shape[0], 3, 3)
        obj_verts_can = samples[Queries.OBJ_VERTS_CAN]
        obj_verts_ = rotmat.bmm(obj_verts_can.float().transpose(1, 2)).transpose(1, 2)

        final_trans = trans.unsqueeze(1) * self.obj_trans_factor
        final_scale = scale.view(batch_size, 1, 1) * self.obj_scale_factor
        height, width = tuple(samples[Queries.IMAGE].shape[2:])
        cam_intr = samples[Queries.CAM_INTR]
        obj_verts_3d_abs, obj_center = HoNet.recover_3d_proj(
            obj_verts_, cam_intr, final_scale, final_trans, input_res=(width, height)
        )
        obj_verts_2d = self.proj2d_func(obj_verts_3d_abs, cam_intr)

        # Recover 2D positions given camera intrinsic parameters and object vertex
        # coordinates in camera coordinate reference
        if Queries.CORNERS_3D in samples:
            corners_can = samples[Queries.CORNERS_CAN]
            obj_corners_ = rotmat.bmm(corners_can.float().transpose(1, 2)).transpose(1, 2)
            corners_3d_abs = obj_corners_ + obj_center
            corners_2d = self.proj2d_func(corners_3d_abs, cam_intr)
        else:
            obj_corners_ = None
            corners_3d_abs = None
            corners_2d = None

        #  @Xinyu: obj_results["recov_obj_verts3d"] = \
        #      obj_results["rotaxisang"] @  OBJ_CAN_VERTS + obj_results["obj_center3d"]
        obj_results = {
            "obj_center": obj_center,  # ===== To PICR =====
            "obj_verts_3d_abs": obj_verts_3d_abs,  # ===== To PICR =====
            "corners_3d_abs": corners_3d_abs,
            "obj_pred_scale": scale,
            "obj_pred_trans": trans,
            "obj_rot": rotaxisang,  # ===== To PICR =====
            "obj_scale": final_scale,
            "obj_trans": final_trans,
            "corners_2d": corners_2d,
            "obj_verts_2d": obj_verts_2d,
            # TODO for MSSD
            "box_rot_rotmat": rotmat,
            "boxroot_3d_abs": obj_center,
        }

        return obj_results

    def forward(self, samples: Dict):
        results = {}
        features = self.base_net(image=samples["image"])
        mano_results = self.recover_mano(features["res_layer4_mean"], samples)
        obj_results = self.recover_object(features["res_layer4_mean"], samples)

        # ⬇ make corners_3d root relative
        obj_results["corners_3d"] = obj_results["corners_3d_abs"] - mano_results["root_joint"]
        obj_results["obj_verts_3d"] = obj_results["obj_verts_3d_abs"] - mano_results["root_joint"]
        results = {**mano_results, **obj_results}
        return results

    def init_weights(self, pretrained=""):
        if pretrained == "":
            logger.warning(f"=> Init {type(self).__name__} weights in backbone and head")
            """
            Add init for other modules if has
            ...
            """
        elif os.path.isfile(pretrained):
            # pretrained_state_dict = torch.load(pretrained)
            logger.info(f"=> Loading {type(self).__name__} pretrained model from: {pretrained}")
            # self.load_state_dict(pretrained_state_dict, strict=False)
            checkpoint = torch.load(pretrained)
            if isinstance(checkpoint, OrderedDict):
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict_old = checkpoint["state_dict"]
                state_dict = OrderedDict()
                # delete 'module.' because it is saved from DataParallel module
                for key in state_dict_old.keys():
                    if key.startswith("module."):
                        # state_dict[key[7:]] = state_dict[key]
                        # state_dict.pop(key)
                        state_dict[key[7:]] = state_dict_old[key]  # delete "module." (in nn.parallel and ddp)
                    else:
                        state_dict[key] = state_dict_old[key]
            else:
                logger.error(f"=> No state_dict found in checkpoint file {pretrained}")
                raise RuntimeError()
            self.load_state_dict(state_dict, strict=True)
        else:
            logger.error(f"=> No {type(self).__name__} checkpoints file found in {pretrained}")
            raise FileNotFoundError()
