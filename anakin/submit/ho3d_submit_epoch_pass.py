from typing import Optional, Union, Any, List
import torch
import numpy as np

from anakin.criterions.criterion import Criterion
from anakin.metrics.evaluator import Evaluator
from anakin.models.arch import Arch
from anakin.opt import arg
from anakin.utils.etqdm import etqdm
from termcolor import colored

from .submit_epoch_pass import SubmitEpochPass
from .ho_submit_epoch_pass import HOSubmitEpochPass

import json
import shutil
import subprocess


@SubmitEpochPass.reg("ho3d")
class HO3DSubmitEpochPass(HOSubmitEpochPass):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.true_root = arg.true_root

    @staticmethod
    def get_order_idxs():
        reorder_idxs = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
        unorder_idxs = np.argsort(reorder_idxs)
        return reorder_idxs, unorder_idxs

    def dump_json(self, pred_out_path, xyz_pred_list, verts_pred_list, codalab=True):
        """ Save predictions into a json file for official ho3dv2 evaluation. """
        # make sure its only lists
        def roundall(rows):
            return [[round(val, 5) for val in row] for row in rows]

        xyz_pred_list = [roundall(x.tolist()) for x in xyz_pred_list]
        verts_pred_list = [roundall(x.tolist()) for x in verts_pred_list]

        # save to a json
        with open(pred_out_path, "w") as fo:
            json.dump([xyz_pred_list, verts_pred_list], fo)
        print(
            "Dumped %d joints and %d verts predictions to %s" % (len(xyz_pred_list), len(verts_pred_list), pred_out_path)
        )
        if codalab:
            file_name = ".".join(pred_out_path.split("/")[-1].split(".")[:-1])
            if pred_out_path != f"./common/{file_name}.json":
                shutil.copy(pred_out_path, f"./common/{file_name}.json")
            subprocess.call(["zip", "-j", f"./common/{file_name}.zip", f"./common/{file_name}.json"])

    def __call__(
        self,
        epoch_idx: int,
        data_loader: Optional[torch.utils.data.DataLoader],
        arch_model: Union[Arch, torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel],
        criterion: Optional[Criterion],
        evaluator: Optional[Evaluator],
        rank: int,
        dump_path: str,
    ):
        arch_model.eval()
        if evaluator:
            evaluator.reset_all()

        # ? <<<<<<<<<<<<<<<<<<<<<<<<<
        res_joints: List[Any] = []
        res_verts: List[Any] = []
        reorder_idxs, unorder_idxs = self.get_order_idxs()
        # ? >>>>>>>>>>>>>>>>>>>>>>>>>

        bar = etqdm(data_loader, rank=rank)
        self.sample_counter = 0
        for batch_idx, batch in enumerate(bar):
            predict_arch_dict = arch_model(batch)
            predicts = {}
            for key in predict_arch_dict.keys():
                predicts.update(predict_arch_dict[key])

            # ==== criterion >>>>
            if criterion:
                final_loss, losses = criterion.compute_losses(predicts, batch)
            else:
                final_loss, losses = torch.Tensor([0.0]), {}
            # <<<<<<<<<<<<<<<<<<<<

            # fitting
            pred_joints = predicts["joints_3d_abs"].detach()
            pred_obj_rotmat = predicts["box_rot_rotmat"].detach()
            pred_obj_tsl = predicts["boxroot_3d_abs"].detach()
            pred_obj_corners = predicts["corners_3d_abs"].detach()
            if self.true_root:
                pred_joints[:, 0] = batch["root_joint"].to(pred_joints.device)
            # evaluate
            if evaluator:
                evaluator.feed_all(predicts, batch, losses)
            bar.set_description(f"{colored('Submit', 'yellow')} Epoch {epoch_idx} | {str(evaluator)}")

            if self.fit_mesh:
                fitted_verts, fitted_joints = self.mesh_fit(batch, pred_joints)

            # draw image for each sample
            if self.fit_mesh and self.postprocess_draw:
                hand_faces = self.fitting_unit.face
                self.sample_counter = self.draw_batch(
                    batch["image"],
                    batch["cam_intr"],
                    pred_joints,
                    fitted_verts,
                    pred_obj_rotmat,
                    pred_obj_tsl,
                    pred_obj_corners,
                    hand_faces,
                    data_loader.dataset,
                )

            # ? <<<<<<<<<<<<<<<<<<<<<<<<<

            # condition on which joints to use
            if self.fit_mesh and self.fit_mesh_use_fitted_joints:
                # pred_joints = pred_joints[:, unorder_idxs]
                # pred_joints[:, :, 0] = -pred_joints[:, :, 0]
                # pred_joints[:] = -pred_joints
                # TODO: fix this later!
                for jid in range(len(fitted_joints)):
                    item_jid = fitted_joints[jid]
                    item_jid = item_jid[unorder_idxs, :]
                    item_jid[:, 1] = -item_jid[:, 1]
                    item_jid[:, 2] = -item_jid[:, 2]
                    fitted_joints[jid] = item_jid
                res_joints.extend(fitted_joints)
            else:
                pred_joints = predicts["joints_3d_abs"].cpu().detach()[:, unorder_idxs]
                pred_joints[:, :, 0] = -pred_joints[:, :, 0]
                joints = [-val.numpy()[0] for val in pred_joints.split(1)]
                res_joints.extend(joints)

            # condition on whether we have verts or fill 0
            if self.fit_mesh:
                res_verts.extend(fitted_verts)
            else:
                batch_size = len(joints)
                hand_verts = [np.zeros((778, 3))] * batch_size
                res_verts.extend(hand_verts)

        if self.dump:
            self.dump_json(dump_path, res_joints, res_verts, codalab=True)

        # ? >>>>>>>>>>>>>>>>>>>>>>>>>
