import hashlib
import json
import os
import pickle
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch
import trimesh
from PIL import Image
from dex_ycb_toolkit.dex_ycb import DexYCBDataset
from dex_ycb_toolkit.factory import get_dataset
from manotorch.manolayer import MANOOutput, ManoLayer
from scipy.spatial.distance import cdist

from anakin.datasets.hodata import HOdata
from anakin.utils import transform
from anakin.utils.builder import DATASET
from anakin.utils.etqdm import etqdm
from anakin.utils.logger import logger
from anakin.utils.misc import enable_lower_param, CONST
from anakin.utils.transform import batch_ref_bone_len


@DATASET.register_module
class DexYCB(HOdata):

    @enable_lower_param
    def __init__(self, **cfg):
        super().__init__(**cfg)

        self.split_mode = cfg["SPLIT_MODE"]
        self.use_left_hand = cfg["USE_LEFT_HAND"]
        self.filter_invisible_hand = cfg["FILTER_INVISIBLE_HAND"]
        self.dataset = None

        self.dexycb_mano_right = ManoLayer(
            flat_hand_mean=False,
            side="right",
            mano_assets_root="assets/mano_v1_2",
            use_pca=True,
            ncomps=45,
        )
        self.dexycb_mano_left = (ManoLayer(
            flat_hand_mean=False,
            side="left",
            mano_assets_root="assets/mano_v1_2",
            use_pca=True,
            ncomps=45,
        ) if self.use_left_hand else None)

        self.load_dataset()

    def _preload(self):
        self.name = "DexYCB"
        self.root = os.path.join(self.data_root, self.name)
        os.environ["DEX_YCB_DIR"] = self.root

        self.cache_identifier_dict = {
            "filter_thresh": float(self.filter_thresh),
            "data_split": self.data_split,
            "split_mode": self.split_mode,
            "fliter_no_contact": self.filter_no_contact,
            "use_left_hand": self.use_left_hand,
            "filter_invisible_hand": self.filter_invisible_hand,
        }
        self.cache_identifier_raw = json.dumps(self.cache_identifier_dict, sort_keys=True)
        self.cache_identifier = hashlib.md5(self.cache_identifier_raw.encode("ascii")).hexdigest()
        self.cache_path = os.path.join("common", "cache", self.name, "{}.pkl".format(self.cache_identifier))

    def load_dataset(self):
        self._preload()
        cache_folder = os.path.dirname(self.cache_path)
        os.makedirs(cache_folder, exist_ok=True)

        dexycb_name = f"{self.split_mode}_{self.data_split}"
        logger.info(f"DexYCB use split: {dexycb_name}")
        self.dataset: DexYCBDataset = get_dataset(dexycb_name)
        self.raw_size = (640, 480)
        self.load_obj_mesh()

        # region filter sample
        if self.use_left_hand and not self.filter_no_contact and not self.filter_invisible_hand:
            self.sample_idxs = list(range(len(self.dataset)))
        else:
            if self.use_cache and os.path.exists(self.cache_path):
                with open(self.cache_path, "rb") as p_f:
                    self.sample_idxs = pickle.load(p_f)
                logger.info(f"Loaded cache for {self.name}_{self.data_split}_{self.split_mode} from {self.cache_path}")
            else:
                self.sample_idxs = []
                logger.info("filtering samples")
                for i, sample in enumerate(etqdm(self.dataset)):
                    if not self.use_left_hand and sample["mano_side"] == "left":
                        continue
                    if (self.filter_invisible_hand or self.filter_no_contact) and np.all(self.get_joints_2d(i) == -1.0):
                        continue
                    if self.filter_no_contact and (cdist(self.get_obj_verts_transf(i), self.get_joints_3d(i)).min() *
                                                   1000.0 > self.filter_thresh):
                        continue
                    self.sample_idxs.append(i)
                with open(self.cache_path, "wb") as p_f:
                    pickle.dump(self.sample_idxs, p_f)
                logger.info(f"Wrote cache for {self.name}_{self.data_split}_{self.split_mode} to {self.cache_path}")

        # endregion

    def load_obj_mesh(self):
        self.obj_raw_meshes = {}
        for obj_idx, obj_file in self.dataset.obj_file.items():
            obj_mesh = trimesh.load(obj_file, process=False)
            self.obj_raw_meshes[obj_idx] = obj_mesh

    def __len__(self):
        return len(self.sample_idxs)

    def get_sample_idxs(self) -> List[int]:
        return self.sample_idxs

    # @lru_cache(maxsize=None)
    def get_label(self, label_file: str):
        return np.load(label_file)

    def get_cam_intr(self, idx):
        sample = self.dataset[idx]
        return np.array(
            [
                [sample["intrinsics"]["fx"], 0.0, sample["intrinsics"]["ppx"]],
                [0.0, sample["intrinsics"]["fy"], sample["intrinsics"]["ppy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def get_center_scale_wrt_bbox(self, idx):
        if self.require_full_image:
            full_width, full_height = self.raw_size[0], self.raw_size[1]
            center = np.array((full_width / 2, full_height / 2))
            scale = full_width
            return center, scale

        if self.crop_model == "hand_obj":
            joints2d = self.get_joints_2d(idx)  # (21, 2)
            corners_2d = self.get_corners_2d(idx)  # (8, 2)
            all2d = np.concatenate([joints2d, corners_2d], axis=0)  # (29, 2)
            center = HOdata.get_annot_center(all2d)
            scale = HOdata.get_annot_scale(all2d)
            return center, scale
        elif self.crop_model == "hand":
            joints2d = self.get_joints_2d(idx)  # (21, 2)
            center = HOdata.get_annot_center(joints2d)
            scale = HOdata.get_annot_scale(joints2d)
            return center, scale
        else:
            raise NotImplementedError()

    def get_corners_vis(self, idx):
        if self.data_split not in ["train", "trainval"]:
            corners_vis = np.ones(self.ncorners)
        else:
            corners_2d = self.get_corners_2d(idx)
            corners_vis = ((corners_2d[:, 0] >= 0) &
                           (corners_2d[:, 0] < self.raw_size[0])) & ((corners_2d[:, 1] >= 0) &
                                                                     (corners_2d[:, 1] < self.raw_size[1]))
            # TODO use Depth Testing
            # sample = self.dataset[idx]
            # corners_2d_idx = corners_2d[corners_vis].astype(np.int)
            # seg = self.get_label(sample["label_file"])["seg"]
            # grasp_ycb_idx = sample["ycb_ids"][sample["ycb_grasp_ind"]]
            # corner_cls = seg[corners_2d_idx[:, 1], corners_2d_idx[:, 0]]
            # corners_vis[corners_vis] = corners_vis[corners_vis] & (
            #     (corner_cls == 0) | (corner_cls == 255) | (corner_cls == grasp_ycb_idx)
            # )

        return corners_vis.astype(np.float32)

    def get_corners_2d(self, idx):
        corners_3d = self.get_corners_3d(idx)
        cam_intr = self.get_cam_intr(idx)
        return HOdata.persp_project(corners_3d, cam_intr)

    def get_corners_3d(self, idx):
        transf = self.get_obj_transf(idx)
        R, t = transf[:3, :3], transf[:3, [3]]
        corners_can = self.get_corners_can(idx)
        corners = (R @ corners_can.T + t).T
        return corners

    def get_corners_can(self, idx):
        sample = self.dataset[idx]
        grasp_ycb_idx = sample["ycb_ids"][sample["ycb_grasp_ind"]]
        obj_mesh = self.obj_raw_meshes[grasp_ycb_idx]
        # NOTE: verts_can = verts - bbox_center
        _, offset, _ = transform.center_vert_bbox(obj_mesh.vertices, scale=False)  # !! CENTERED HERE
        corners = trimesh.bounds.corners(obj_mesh.bounds)
        corners = corners - offset
        return np.asfarray(corners, dtype=np.float32)

    def get_hand_faces(self, idx):
        sample = self.dataset[idx]
        mano_layer = self.dexycb_mano_left if sample["mano_side"] == "left" else self.dexycb_mano_right
        faces = np.array(mano_layer.th_faces).astype(np.long)
        return faces

    def get_hand_verts_2d(self, idx):
        verts_3d = self.get_hand_verts_3d(idx)
        cam_intr = self.get_cam_intr(idx)
        return HOdata.persp_project(verts_3d, cam_intr)

    def get_hand_verts_3d(self, idx):
        sample = self.dataset[idx]
        label = self.get_label(sample["label_file"])  # keys: seg, pose_y, pose_m, joint_3d, joint_2d
        pose_m = torch.from_numpy(label["pose_m"])
        shape = torch.tensor(sample["mano_betas"]).unsqueeze(0)
        mano_layer = self.dexycb_mano_left if sample["mano_side"] == "left" else self.dexycb_mano_right
        mano_out: MANOOutput = mano_layer(pose_m[:, :48], shape)
        hand_verts = mano_out.verts + pose_m[:, 48:]
        return hand_verts.squeeze(0).numpy().astype(np.float32)

    def get_bone_scale(self, idx):
        joints_3d = self.get_joints_3d(idx)
        bone_len = batch_ref_bone_len(np.expand_dims(joints_3d, axis=0)).squeeze(0)
        return bone_len.astype(np.float32)

    def get_image(self, idx):
        img_path = self.get_image_path(idx)
        img = Image.open(img_path).convert("RGB")
        return img

    def get_image_path(self, idx):
        sample = self.dataset[idx]
        return sample["color_file"]

    def get_joints_2d(self, idx):
        sample = self.dataset[idx]
        label = self.get_label(sample["label_file"])  # keys: seg, pose_y, pose_m, joint_3d, joint_2d
        return label["joint_2d"].squeeze(0)

    def get_joints_3d(self, idx):
        sample = self.dataset[idx]
        label = self.get_label(sample["label_file"])  # keys: seg, pose_y, pose_m, joint_3d, joint_2d
        return label["joint_3d"].squeeze(0)

    def get_obj_faces(self, idx):
        sample = self.dataset[idx]
        grasp_ycb_idx = sample["ycb_ids"][sample["ycb_grasp_ind"]]
        obj_mesh = self.obj_raw_meshes[grasp_ycb_idx]
        faces = np.array(obj_mesh.faces).astype(np.long)
        return faces

    def get_obj_idx(self, idx):
        sample = self.dataset[idx]
        grasp_ycb_idx = sample["ycb_ids"][sample["ycb_grasp_ind"]]
        return grasp_ycb_idx

    def get_obj_transf(self, idx):
        sample = self.dataset[idx]
        label = self.get_label(sample["label_file"])
        transf = label["pose_y"][sample["ycb_grasp_ind"]]
        grasp_ycb_idx = sample["ycb_ids"][sample["ycb_grasp_ind"]]
        obj_mesh = self.obj_raw_meshes[grasp_ycb_idx]
        _, offset, _ = transform.center_vert_bbox(obj_mesh.vertices, scale=False)  # !! CENTERED HERE
        R, t = transf[:3, :3], transf[:, 3:]
        new_t = R @ offset.reshape(3, 1) + t
        new_transf = np.concatenate(
            [np.concatenate([R, new_t], axis=1),
             np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)])
        return new_transf.astype(np.float32)

    # * deprecated
    def _get_raw_obj_transf(self, idx):
        sample = self.dataset[idx]
        label = self.get_label(sample["label_file"])  # keys: seg, pose_y, pose_m, joint_3d, joint_2d
        transf = label["pose_y"][sample["ycb_grasp_ind"]]
        transf = np.concatenate([transf, np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)])
        return transf

    def get_obj_verts_2d(self, idx):
        verts_3d = self.get_obj_verts_transf(idx)
        cam_intr = self.get_cam_intr(idx)
        return HOdata.persp_project(verts_3d, cam_intr)

    def get_obj_verts_can(self, idx):
        sample = self.dataset[idx]
        grasp_ycb_idx = sample["ycb_ids"][sample["ycb_grasp_ind"]]
        obj_mesh = self.obj_raw_meshes[grasp_ycb_idx]
        # NOTE: verts_can = verts - bbox_center
        verts_can, obj_cantrans, obj_canscale = transform.center_vert_bbox(np.asfarray(obj_mesh.vertices,
                                                                                       dtype=np.float32),
                                                                           scale=False)  # !! CENTERED HERE
        return verts_can, obj_cantrans, obj_canscale

    # * deprecated
    def _get_raw_obj_verts(self, idx):
        sample = self.dataset[idx]
        grasp_ycb_idx = sample["ycb_ids"][sample["ycb_grasp_ind"]]

        obj_mesh = trimesh.load(self.dataset.obj_file[grasp_ycb_idx], process=False)
        return np.array(obj_mesh.vertices).astype(np.float32)

    def get_obj_verts_transf(self, idx):
        # * deprecated
        # transf = self._get_raw_obj_transf(idx)
        # R, t = transf[:3, :3], transf[:3, [3]]
        # verts_can = self._get_raw_obj_verts(idx)
        # raw_verts = (R @ verts_can.T + t).T

        transf = self.get_obj_transf(idx)
        R, t = transf[:3, :3], transf[:3, [3]]
        verts_can, _, _ = self.get_obj_verts_can(idx)
        verts = (R @ verts_can.T + t).T

        return verts

    def get_sample_identifier(self, idx):
        res = f"{self.name}__{self.cache_identifier_raw}__{idx}"
        return res

    def obj_load_driver(self) -> Tuple[List[str], List[trimesh.base.Trimesh], List[np.ndarray]]:
        obj_names = []
        obj_meshes = []
        obj_corners_can = []
        for idx, obj_mesh in self.obj_raw_meshes.items():
            obj_name = CONST.YCB_IDX2CLASSES[idx]
            obj_names.append(obj_name)

            # ===== meshes can >>>>>>
            omesh = deepcopy(obj_mesh)
            verts_can, bbox_center, bbox_scale = transform.center_vert_bbox(omesh.vertices, scale=False)
            omesh.vertices = verts_can
            obj_meshes.append(omesh)

            # ===== corners can >>>>>
            corners = trimesh.bounds.corners(obj_mesh.bounds)
            corners_can = (corners - bbox_center) / bbox_scale
            obj_corners_can.append(corners_can)
        return (obj_names, obj_meshes, obj_corners_can)

    def get_sides(self, idx):
        sample = self.dataset[idx]
        return sample["mano_side"]
