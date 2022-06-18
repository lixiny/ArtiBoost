from typing import Optional, Union, Mapping, TypeVar, Any, Callable
import torch
import numpy as np
import os
from typing import Sequence

from anakin.datasets.hodata import HOdata
from anakin.criterions.criterion import Criterion
from anakin.metrics.evaluator import Evaluator
from anakin.models.arch import Arch

from .submit_epoch_pass import SubmitEpochPass


class HOSubmitEpochPass(SubmitEpochPass):
    def draw_batch(
        self,
        image: torch.Tensor,
        cam_intr: torch.Tensor,
        pred_joints: torch.Tensor,
        fitted_verts: Sequence[np.ndarray],
        pred_obj_rotmat: torch.Tensor,
        pred_obj_tsl: torch.Tensor,
        pred_obj_corners: torch.Tensor,
        hand_faces: np.ndarray,
        dataset: HOdata,
        **kwargs,
    ):
        from anakin.viztools.draw import save_a_image_with_mesh_joints_objects

        # prepare path
        if self.postprocess_draw_path is None:
            save_prefix = "tmp/saveimg2"
        else:
            save_prefix = self.postprocess_draw_path
        os.makedirs(save_prefix, exist_ok=True)

        img_list = image + 0.5
        img_list = img_list.permute(0, 2, 3, 1)
        img_list = img_list.cpu().numpy()
        intr_mat = cam_intr.cpu().numpy()

        batch_size = img_list.shape[0]
        for batch_id in range(batch_size):
            obj_faces = dataset.get_obj_faces(self.sample_counter)
            curr_obj_corner = pred_obj_corners[batch_id].cpu().numpy()
            curr_obj_rotmat = pred_obj_rotmat[batch_id].cpu().numpy()
            curr_obj_tsl = pred_obj_tsl[batch_id].cpu().numpy()

            obj_v_can, _, _ = dataset.get_obj_verts_can(self.sample_counter)
            curr_obj_v = (curr_obj_rotmat @ obj_v_can.T).T + curr_obj_tsl

            curr_j = pred_joints[batch_id].cpu().numpy()
            curr_intr = intr_mat[batch_id]
            curr_j2d = (curr_intr @ curr_j.T).T
            curr_j2d = curr_j2d[:, 0:2] / curr_j2d[:, 2:3]
            curr_obj_corner_2d = (curr_intr @ curr_obj_corner.T).T
            curr_obj_corner_2d = curr_obj_corner_2d[:, 0:2] / curr_obj_corner_2d[:, 2:3]
            save_a_image_with_mesh_joints_objects(
                img_list[batch_id],
                curr_intr,
                fitted_verts[batch_id],
                hand_faces,
                curr_j2d,
                curr_j,
                curr_obj_v,
                obj_faces,
                curr_obj_corner_2d,
                curr_obj_corner,
                os.path.join(save_prefix, f"{self.sample_counter:0>4}.png"),
                renderer=self.renderer,
            )

            self.sample_counter += 1
            # if self.sample_counter > 30:
            #     exit(0)

        return self.sample_counter
