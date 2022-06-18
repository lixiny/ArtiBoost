import json
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as torch_f
from anakin.datasets.hoquery import Queries
from anakin.utils.bop_toolkit.bop_misc import get_symmetry_transformations
from anakin.utils.builder import LOSS
from ..utils.transform import batch_ref_bone_len
from anakin.utils.logger import logger
from anakin.utils.misc import CONST

from .criterion import TensorLoss


@LOSS.register_module
class SymCornerLoss(TensorLoss):
    def __init__(self, **cfg):
        super(SymCornerLoss, self).__init__()
        self.lambda_sym_corners_3d = cfg.get("LAMBDA_SYM_CORNERS_3D", 0.0)

        # region pre-compute sym transf
        model_info_path = cfg["MODEL_INFO_PATH"]
        self.model_info = json.load(open(model_info_path, "r"))
        self.max_sym_disc_step = cfg.get("MAX_SYM_DISC_STEP", 0.01)
        self.use_ho3d_ycb = cfg.get("USE_HO3D_YCB", False)

        self.model_sym = {}
        max_sym_len = 0
        for obj_idx in range(1, len(self.model_info) + 1):
            self.model_sym[obj_idx] = get_symmetry_transformations(self.model_info[str(obj_idx)], self.max_sym_disc_step)
            max_sym_len = max(max_sym_len, len(self.model_sym[obj_idx]))
        R, t = [], []
        for obj_idx in range(1, len(self.model_info) + 1):
            obj_R, obj_t = [], []
            for transf in self.model_sym[obj_idx]:
                obj_R.append(transf["R"])
                obj_t.append(transf["t"])
            while len(obj_R) < max_sym_len:
                obj_R.append(np.eye(3))
                obj_t.append(np.zeros((3, 1)))
            obj_R = np.stack(obj_R)  # [max_sym_len, 3, 3]
            obj_t = np.stack(obj_t)  # [max_sym_len, 3, 1]
            R.append(obj_R)
            t.append(obj_t / 1000.0)  # mm to m
        self.R = torch.Tensor(np.stack(R))  # [N, max_sym_len, 3, 3]
        self.t = torch.Tensor(np.stack(t))  # [N, max_sym_len, 3, 1]
        # endregion

        logger.info(f"Construct {type(self).__name__} with lambda: ")
        logger.info(f"  |   LAMBDA_SYM_CORNERS_3D : {self.lambda_sym_corners_3d}")

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        final_loss, losses = super().__call__(preds, targs, **kwargs)  # TENSOR(0.), {}
        # ============== OBJ CORNERS 3D MSE LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if self.lambda_sym_corners_3d:

            sym_R = self.R.to(final_loss.device)
            sym_t = self.t.to(final_loss.device)
            sym_R = sym_R[(targs[Queries.OBJ_IDX] - 1).tolist()]
            sym_t = sym_t[(targs[Queries.OBJ_IDX] - 1).tolist()]

            corner_can = targs[Queries.CORNERS_CAN].to(final_loss.device)
            obj_transf = targs[Queries.OBJ_TRANSF].to(final_loss.device)

            if not self.use_ho3d_ycb:
                sym_corner_can = (torch.einsum("bkmn,bcn->bkmc", sym_R, corner_can) + sym_t).transpose(
                    -2, -1
                )  # [B, max_sym_len, NCORNERS, 3]
            else:
                cam_extr = torch.tensor(
                    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=torch.float32, device=final_loss.device
                )
                sym_corner_can = (
                    cam_extr @ (torch.einsum("bkmn,bnc->bkmc", sym_R, cam_extr @ corner_can.transpose(-2, -1)) + sym_t)
                ).transpose(-2, -1)

            sym_corner_3d_abs = (
                torch.einsum("bij,bklj->bkil", obj_transf[:, :3, :3], sym_corner_can) + obj_transf[:, None, :3, 3:]
            ).transpose(
                -2, -1
            )  # [B, max_sym_len, NCORNERS, 3]

            pred_corners_3d_abs = preds["corners_3d_abs"]  # [B, NCORNERS, 3]

            # mask invisible corners
            corners_vis_mask = targs[Queries.CORNERS_VIS].to(final_loss.device)
            pred_corners_3d_abs = torch.einsum("bij,bi->bij", pred_corners_3d_abs, corners_vis_mask)
            sym_corners_3d_abs = torch.einsum("bkij,bi->bkij", sym_corner_3d_abs, corners_vis_mask)
            sym_corners_3d_loss = (
                ((sym_corners_3d_abs - pred_corners_3d_abs[:, None, :, :]) ** 2).mean(-1).mean(-1).min(dim=-1)[0].mean()
            )

            final_loss += self.lambda_sym_corners_3d * sym_corners_3d_loss
        else:
            sym_corners_3d_loss = None
        losses["sym_corners_3d_loss"] = sym_corners_3d_loss
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        losses[self.output_key] = final_loss
        return final_loss, losses
