import os
from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from anakin.datasets.hoquery import Queries
from anakin.utils.builder import HEAD, MODEL, build_backbone, build_head
from anakin.utils.transform import batch_uvd2xyz
from anakin.utils.logger import logger
from anakin.utils.misc import CONST, enable_lower_param, param_size


def norm_heatmap(norm_type: str, heatmap: torch.Tensor) -> torch.Tensor:
    """
    Args:
        norm_type: str: either in [softmax, sigmoid, divide_sum],
        heatmap: TENSOR (BATCH, C, ...)

    Returns:
        TENSOR (BATCH, C, ...)
    """
    shape = heatmap.shape
    if norm_type == "softmax":
        heatmap = heatmap.reshape(*shape[:2], -1)
        # global soft max
        heatmap = F.softmax(heatmap, 2)
        return heatmap.reshape(*shape)
    elif norm_type == "sigmoid":
        return heatmap.sigmoid()
    elif norm_type == "divide_sum":
        # ! Best not to use this method
        # ! This may get an unstable result and cause the network not to converge.
        heatmap = heatmap.reshape(*shape[:2], -1)
        heatmap = heatmap / heatmap.sum(dim=2, keepdim=True)
        return heatmap.reshape(*shape)
    else:
        raise NotImplementedError


def integral_heatmap3d(heatmap3d: torch.Tensor) -> torch.Tensor:
    """
    Integral 3D heatmap into whd corrdinates. u stand for the prediction in WIDTH dimension
    ref: https://arxiv.org/abs/1711.08229

    Args:
        heatmap3d: TENSOR (BATCH, NCLASSES, DEPTH, HEIGHT, WIDTH) d,v,u

    Returns:
        uvd: TENSOR (BATCH, NCLASSES, 3) RANGE:0~1
    """
    d_accu = torch.sum(heatmap3d, dim=[3, 4])
    v_accu = torch.sum(heatmap3d, dim=[2, 4])
    u_accu = torch.sum(heatmap3d, dim=[2, 3])

    weightd = torch.arange(d_accu.shape[-1], dtype=d_accu.dtype, device=d_accu.device) / d_accu.shape[-1]
    weightv = torch.arange(v_accu.shape[-1], dtype=v_accu.dtype, device=v_accu.device) / v_accu.shape[-1]
    weightu = torch.arange(u_accu.shape[-1], dtype=u_accu.dtype, device=u_accu.device) / u_accu.shape[-1]

    d_ = d_accu.mul(weightd)
    v_ = v_accu.mul(weightv)
    u_ = u_accu.mul(weightu)

    d_ = torch.sum(d_, dim=-1, keepdim=True)
    v_ = torch.sum(v_, dim=-1, keepdim=True)
    u_ = torch.sum(u_, dim=-1, keepdim=True)

    uvd = torch.cat([u_, v_, d_], dim=-1)
    return uvd  # TENSOR (BATCH, NCLASSES, 3)


@HEAD.register_module
class IntegralDeconvHead(nn.Module):

    @enable_lower_param
    def __init__(self, **cfg):
        super().__init__()
        self.inplanes = cfg["INPUT_CHANNEL"]  # depending on your backbone
        self.depth_res = cfg["DEPTH_RESOLUTION"]
        self.height_res = cfg["HEATMAP_SIZE"][1]
        self.width_res = cfg["HEATMAP_SIZE"][0]
        self.deconv_with_bias = cfg["DECONV_WITH_BIAS"]
        self.nclasses = cfg["NCLASSES"]
        self.norm_type = cfg["NORM_TYPE"]

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            cfg["NUM_DECONV_LAYERS"],
            cfg["NUM_DECONV_FILTERS"],
            cfg["NUM_DECONV_KERNELS"],
        )

        self.final_layer = nn.Conv2d(
            in_channels=cfg["NUM_DECONV_FILTERS"][-1],
            out_channels=cfg["NCLASSES"] * self.depth_res,
            kernel_size=cfg["FINAL_CONV_KERNEL"],
            stride=1,
            padding=1 if cfg["FINAL_CONV_KERNEL"] == 3 else 0,
        )
        self.init_weights()

    def init_weights(self):
        logger.info("=> init deconv weights from normal distribution")
        for m in self.deconv_layers.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        logger.info("=> init final conv weights from normal distribution")
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def view_to_bcdhw(self, x: torch.Tensor) -> torch.Tensor:
        """
        view a falttened 2D heatmap to 3D heatmap, sharing the same memory by using view()
        Args:
            x: TENSOR (BATCH, NCLASSES * DEPTH, HEIGHT|ROWS, WIDTH|COLS)

        Returns:
            TENSOR (BATCH, NCLASSES, DEPTH, HEIGHT, WIDTH)
        """
        return x.contiguous().view(
            x.shape[0],  # BATCH,
            self.nclasses,  # NCLASSES
            self.depth_res,  # DEPTH
            self.height_res,  # HEIGHT,
            self.width_res,  # WIDTH
        )

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError()

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), "ERROR: num_deconv_layers is different len(num_deconv_filters)"
        assert num_layers == len(num_kernels), "ERROR: num_deconv_layers is different len(num_deconv_filters)"

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias,
                ))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        x = kwargs["feature"]
        x = self.deconv_layers(x)
        x = self.final_layer(x)

        x = x.reshape((x.shape[0], self.nclasses, -1))  # TENSOR (B, NCLASS, DEPTH x HEIGHT x WIDTH)
        x = norm_heatmap(self.norm_type, x)  # TENSOR (B, NCLASS, DEPTH x HEIGHT x WIDTH)

        confd = torch.max(x, dim=-1).values  # TENSOR (B, NCLASS)
        assert x.dim() == 3, logger.error(f"Unexpected dim, expect x has shape (B, C, DxHxW), got {x.shape}")
        x = x / (x.sum(dim=-1, keepdim=True) + 1e-7)
        x = self.view_to_bcdhw(x)  # TENSOR(BATCH, NCLASSES, DEPTH, HEIGHT, WIDTH)
        x = integral_heatmap3d(x)  # TENSOR (BATCH, NCLASSES, 3)
        return {"kp3d": x, "kp3d_confd": confd}


@MODEL.register_module
class SimpleBaseline(nn.Module):

    @enable_lower_param
    def __init__(self, **cfg):
        super(SimpleBaseline, self).__init__()

        if cfg["BACKBONE"]["PRETRAINED"] and cfg["PRETRAINED"]:
            logger.warning(
                f"{type(self).__name__}'s backbone {cfg['BACKBONE']['TYPE']} weights will be rewritten by {cfg['PRETRAINED']}"
            )

        self.center_idx = cfg["DATA_PRESET"].get("CENTER_IDX", 9)
        self.inp_res = cfg["DATA_PRESET"]["IMAGE_SIZE"]
        self.backbone = build_backbone(cfg["BACKBONE"], default_args=cfg["DATA_PRESET"])  # ResNet50
        self.pose_head = build_head(cfg["HEAD"], default_args=cfg["DATA_PRESET"])  # IntegralDeconvHead
        self.init_weights(pretrained=cfg["PRETRAINED"])
        logger.info(f"{type(self).__name__} uses center_idx {self.center_idx}")
        logger.info(f"{type(self).__name__} has {param_size(self)}M parameters")

    def forward(self, inputs: Dict):
        features = self.backbone(image=inputs["image"])
        pose_results = self.pose_head(feature=features["res_layer4"])
        kp3d_abs = batch_uvd2xyz(
            uvd=pose_results["kp3d"],
            root_joint=inputs[Queries.ROOT_JOINT],
            intr=inputs[Queries.CAM_INTR],
            inp_res=self.inp_res,
        )  # TENSOR (B, NKP, 3)

        # dispatch
        joints_3d_abs = kp3d_abs[:, :CONST.NUM_JOINTS, :]  # (B, 21, 3)
        corners_3d_abs = kp3d_abs[:, CONST.NUM_JOINTS:, :]  # (B, 8, 3)
        root_joint = joints_3d_abs[:, self.center_idx, :]  # (B, 3)
        joints_confd = pose_results["kp3d_confd"][:, :CONST.NUM_JOINTS]  # (B, 21)
        corners_confid = pose_results["kp3d_confd"][:, CONST.NUM_JOINTS:]  # (B, 8)

        # diff = torch.norm(inputs[Queries.ROOT_JOINT] - root_joint, dim=1)
        # logger.debug(diff)

        return {
            # ↓ absolute value feed to criterion
            "joints_3d_abs": joints_3d_abs,
            "corners_3d_abs": corners_3d_abs,
            # ↓ root relative valus feed to evaluator
            "joints_3d": joints_3d_abs - root_joint.unsqueeze(1),
            "corners_3d": corners_3d_abs - root_joint.unsqueeze(1),
            "2d_uvd": pose_results["kp3d"],
        }

    def init_weights(self, pretrained=""):
        if pretrained == "":
            logger.warning(f"=> Init {type(self).__name__} weights in backbone and head")
            """
            Add init for other modules
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
                        state_dict[key[7:]] = state_dict_old[key]  # delete "module." (in nn.parallel)
                    else:
                        state_dict[key] = state_dict_old[key]
            else:
                logger.error(f"=> No state_dict found in checkpoint file {pretrained}")
                raise RuntimeError()
            self.load_state_dict(state_dict, strict=False)
        else:
            logger.error(f"=> No {type(self).__name__} checkpoints file found in {pretrained}")
            raise FileNotFoundError()
