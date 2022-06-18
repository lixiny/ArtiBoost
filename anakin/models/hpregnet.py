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
from anakin.utils.transform import batch_persp_proj2d, compute_rotation_matrix_from_ortho6d
from anakin.utils.logger import logger
from anakin.utils.misc import CONST, enable_lower_param
from anakin.utils.netutils import recurse_freeze


@MODEL.register_module
class HOPRegNet(nn.Module):

    @enable_lower_param
    def __init__(self, **cfg):
        super(HOPRegNet, self).__init__()

        if cfg["BACKBONE"]["PRETRAINED"] and cfg["PRETRAINED"]:
            logger.warning(f"{type(self).__name__}'s backbone {cfg['BACKBONE']['TYPE']} "
                           f"weights will be rewritten by {cfg['PRETRAINED']}")

        self.inp_res = cfg["DATA_PRESET"]["IMAGE_SIZE"]
        self.feature_dim = cfg["HEAD"]["INPUT_DIM"]
        self.center_idx = cfg["DATA_PRESET"]["CENTER_IDX"]
        logger.info(f"{type(self).__name__} uses center_idx {self.center_idx}")

        self.base_net = build_backbone(cfg["BACKBONE"])  # ResNet18
        self.mano_branch = build_head(cfg["HEAD"], default_args=cfg["DATA_PRESET"])

        self.obj_transfhead = HOPRegNet.TransHead(inp_dim=self.feature_dim, out_dim=9)
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

            if out_dim != 3 and out_dim != 9:
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

        cam_intr = samples[Queries.CAM_INTR]  # TENSOR(B, 3, 3)
        root_joint = samples[Queries.ROOT_JOINT]  # [B, 3]
        joints_3d_abs = mano_results["joints_3d"] + root_joint.unsqueeze(1)
        hand_verts_3d_abs = mano_results["hand_verts_3d"] + root_joint.unsqueeze(1)

        joints_2d = self.proj2d_func(joints_3d_abs, cam_intr)
        hand_verts_2d = self.proj2d_func(hand_verts_3d_abs, cam_intr)

        mano_results["joints_2d"] = joints_2d
        mano_results["root_joint"] = root_joint
        mano_results["joints_3d_abs"] = joints_3d_abs
        mano_results["hand_verts_3d_abs"] = hand_verts_3d_abs
        mano_results["hand_verts_2d"] = hand_verts_2d

        return mano_results

    def recover_object(self, feature: torch.Tensor, samples: Dict):
        """
        Compute object vertex and corner positions in camera coordinates by predicting object translation
        and scaling, and recovering 3D positions given known object model
        """
        transf_obj = self.obj_transfhead(feature)
        batch_size = transf_obj.shape[0]
        tsl_wrt_hand = transf_obj[:, :3]  # [B, 3]
        box_rot_6d = transf_obj[:, 3:]

        rotmat = compute_rotation_matrix_from_ortho6d(box_rot_6d).view(batch_size, 3, 3)

        root_joint = samples[Queries.ROOT_JOINT]  # [B, 3]

        cam_intr = samples[Queries.CAM_INTR]
        obj_center = root_joint + tsl_wrt_hand

        corners_can = samples[Queries.CORNERS_CAN]
        obj_corners_ = rotmat.bmm(corners_can.float().transpose(1, 2)).transpose(1, 2)
        corners_3d_abs = obj_corners_ + obj_center.unsqueeze(1)
        corners_2d = self.proj2d_func(corners_3d_abs, cam_intr)

        obj_results = {
            "obj_center": obj_center,
            "corners_3d_abs": corners_3d_abs,
            "obj_pred_tsl": tsl_wrt_hand,
            "obj_pred_rot": rotmat,
            "corners_2d": corners_2d,
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
        obj_results["corners_3d"] = obj_results["corners_3d_abs"] - mano_results["root_joint"].unsqueeze(1)
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
