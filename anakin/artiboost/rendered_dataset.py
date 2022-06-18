import os
import pickle
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import torch
import torchvision.transforms.functional as tvF
import trimesh
from anakin.datasets.hodata import HOdata
from anakin.datasets.hoquery import Queries, SynthQueries
from anakin.utils import img_augment, transform
from anakin.utils.logger import logger
from anakin.utils.misc import CONST
from PIL import Image, ImageFilter
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform


class RenderedDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        fetch_root: str,
        obj_meshes: List[trimesh.Trimesh],
        obj_corners: List[trimesh.Trimesh],
        cam_intr: np.ndarray,
        cfg_dataset,
        cfg_preset,
        crop_image: Optional[Dict] = None,
    ):
        logger.info("rendered_dataset: initializing...")
        self.fetch_root = fetch_root

        self.obj_meshes_list = obj_meshes
        self.obj_corners_list = obj_corners
        self.obj_map = {v: k for k, v in CONST.YCB_IDX2CLASSES.items()}

        self.cam_intr = cam_intr
        self.image_size = cfg_preset["IMAGE_SIZE"]  # (W, H)
        self.raw_size = cfg_preset["IMAGE_SIZE"]
        self.center_idx = cfg_preset["CENTER_IDX"]
        logger.info(f"{type(self).__name__} uses center_idx {self.center_idx}")

        self.bbox_expand_ratio = (float(cfg_preset["BBOX_EXPAND_RATIO"])
                                  if crop_image is None else float(crop_image["BBOX_EXPAND_RATIO"]))
        self.require_full_image = cfg_preset["FULL_IMAGE"] if crop_image is None else crop_image["FULL_IMAGE"]

        self.crop_model = cfg_preset.get("CROP_MODEL", "hand_obj") if crop_image is None else crop_image["CROP_MODEL"]
        logger.warning(f"Use {self.crop_model} to crop image in {type(self).__name__}")

        if self.require_full_image:
            logger.warning(f"bbox_expand_ratio is set to be 1.0 in FULL_IMAGE mode")
            self.bbox_expand_ratio = 1.0

        self.cfg_dataset = deepcopy(cfg_dataset)
        self.cfg_preset = deepcopy(cfg_preset)

        self.aug_param = cfg_dataset["AUG_PARAM"]
        self.aug = cfg_dataset["AUG"]

        if self.aug:
            self.hue = 0.075
            self.saturation = 0.1
            self.contrast = 0.1
            self.brightness = 0.1
            self.blur_radius = 0.1
            if self.aug_param is not None:
                self.scale_jittering = self.aug_param["SCALE_JIT"]
                self.center_jittering = self.aug_param["CENTER_JIT"]
                self.max_rot = self.aug_param["MAX_ROT"] * np.pi
        else:
            self.hue = 0.0
            self.saturation = 0.0
            self.brightness = 0.0
            self.contrast = 0.0
            self.blur_radius = 0.0
            self.scale_jittering = 0.0
            self.center_jittering = 0.0
            self.max_rot = 0.0

        self.sides = CONST.SIDE
        self.njoints = CONST.NUM_JOINTS
        self.ncorners = CONST.NUM_CORNERS

        self.update()
        logger.info("rendered_dataset: initialize complete!")

        # modify when going into torch.DataLoader
        self.out_queue = None
        self.in_queue = None
        self.id = None

    def update(self, new_fetch_root=None):
        if new_fetch_root is not None:
            self.fetch_root = new_fetch_root
        self.file_list = sorted(os.listdir(self.fetch_root))
        self.n_item = len(os.listdir(self.fetch_root))

    def __len__(self):
        return self.n_item

    def prepare_essential(self, index):
        # === preparation work >>>
        # get path of the index-th dumped object
        fetch_path = os.path.join(self.fetch_root, f"{int(index):0>4}.pkl")
        # load
        with open(fetch_path, "rb") as stream:
            blob = pickle.load(stream)
        objid = blob["obj_id"]
        objname = blob["obj_name"]
        pose = blob["obj_pose"]
        hand_verts = blob["hand_verts"]
        hand_joints = blob["hand_joints"]
        persp_id = blob["persp_id"]
        grasp_id = blob["grasp_id"]

        msg = {"id": self.id, "objname": objname, "pose": pose, "hand_verts": hand_verts}
        self.out_queue.put(msg)

        img = self.in_queue.get()
        img = deepcopy(img)
        img = img[:, :, (2, 1, 0)]
        # <<<

        # === get essential >>>
        hand_joints_2d = (self.cam_intr @ hand_joints.T).T
        hand_joints_2d = hand_joints_2d[:, 0:2] / (hand_joints_2d[:, 2:3] + 1e-8)

        obj_corners_can = np.asarray(self.obj_corners_list[objid])
        obj_corners_3d = (pose[:3, :3] @ obj_corners_can.T).T + pose[:3, 3]
        obj_corners_2d = (self.cam_intr @ obj_corners_3d.T).T
        obj_corners_2d = obj_corners_2d[:, 0:2] / (obj_corners_2d[:, 2:3] + 1e-8)
        essentials = {
            "img": img,
            "hand_joints_3d": hand_joints,
            "hand_joints_2d": hand_joints_2d,
            "obj_corners_can": obj_corners_can,
            "obj_corners_3d": obj_corners_3d,
            "obj_corners_2d": obj_corners_2d,
            "cam_intr": self.cam_intr,
            "hand_side": self.sides,
            "pose": pose,
            "objname": objname,
        }
        id_essentials = {
            "obj_id": objid,
            "persp_id": persp_id,
            "grasp_id": grasp_id,
        }
        # <<<

        return essentials, id_essentials

    def __getitem__(self, index):
        essentials, id_essentials = self.prepare_essential(index)

        sample = {}
        sample[SynthQueries.IS_SYNTH] = True
        sample[SynthQueries.OBJ_ID] = id_essentials["obj_id"]
        sample[SynthQueries.PERSP_ID] = id_essentials["persp_id"]
        sample[SynthQueries.GRASP_ID] = id_essentials["grasp_id"]

        bbox_center, bbox_scale = self.get_center_scale_wrt_bbox(essentials)
        bbox_scale *= self.bbox_expand_ratio

        img = Image.fromarray(essentials["img"]).convert("RGB")
        cam_intr = essentials["cam_intr"]
        joints_3d = essentials["hand_joints_3d"].astype(np.float32)
        joints_2d = essentials["hand_joints_2d"].astype(np.float32)
        corners_3d = essentials["obj_corners_3d"].astype(np.float32)
        corners_2d = essentials["obj_corners_2d"].astype(np.float32)
        corners_can = essentials["obj_corners_can"].astype(np.float32)

        if self.aug:
            # * Randomly jitter center
            # Center is located in square of size 2*center_jitter_factor
            # in center of cropped image
            center_jit = Uniform(low=-1, high=1).sample((2,)).numpy()
            center_offsets = self.center_jittering * bbox_scale * center_jit
            bbox_center = bbox_center + center_offsets.astype(int)

            # Scale jittering
            jittering_ = Normal(0, self.scale_jittering / 3.0).sample().item()  # (-3, 3)
            jittering = jittering_ + 1.0
            jittering = np.clip(jittering, 1 - self.scale_jittering, 1 + self.scale_jittering)
            bbox_scale = bbox_scale * jittering
            rot_rad = Uniform(low=-self.max_rot, high=self.max_rot).sample().item()
        else:
            rot_rad = 0

        rot_mat = np.array([
            [np.cos(rot_rad), -np.sin(rot_rad), 0],
            [np.sin(rot_rad), np.cos(rot_rad), 0],
            [0, 0, 1],
        ]).astype(np.float32)

        affine_transf, post_rot_transf = transform.get_affine_transform(
            center=bbox_center,
            scale=bbox_scale,
            optical_center=[cam_intr[0, 2], cam_intr[1, 2]],  # (cx, cy)
            out_res=self.image_size,  # (H, W)
            rot=rot_rad,
        )

        cam_intr = post_rot_transf.dot(cam_intr)
        sample[Queries.CAM_INTR] = cam_intr.astype(np.float32)
        joints_3d = rot_mat.dot(joints_3d.transpose(1, 0)).transpose()
        root_joint = joints_3d[self.center_idx]
        sample[Queries.ROOT_JOINT] = root_joint
        sample[Queries.JOINTS_3D] = joints_3d - root_joint  # * make it root relative
        joints_2d = transform.transform_coords(joints_2d, affine_transf).astype(np.float32)
        sample[Queries.JOINTS_2D] = joints_2d

        joints_vis = self.get_joints_vis(essentials["hand_joints_2d"])
        # hand invisible in raw image
        if joints_vis.sum() < CONST.NUM_JOINTS * 0.4:  # magic number
            sample[Queries.JOINTS_VIS] = np.full(CONST.NUM_JOINTS, 0.0, dtype=np.float32)
        else:
            joints_vis_aug = (((joints_2d[:, 0] >= 0) & (joints_2d[:, 0] < self.image_size[0])) &
                              ((joints_2d[:, 1] >= 0) & (joints_2d[:, 1] < self.image_size[1]))).astype(np.float32)
            if joints_vis_aug.sum() < CONST.NUM_JOINTS * 0.4:  # magic number
                sample[Queries.JOINTS_VIS] = np.full(CONST.NUM_JOINTS, 0.0, dtype=np.float32)
            else:
                sample[Queries.JOINTS_VIS] = joints_vis_aug

        # region ===== Object queries >>>>>
        corners_3d = rot_mat.dot(corners_3d.transpose(1, 0)).transpose()
        sample[Queries.CORNERS_3D] = corners_3d - root_joint  # * make it root relative
        corners_2d = transform.transform_coords(corners_2d, affine_transf)
        sample[Queries.CORNERS_2D] = np.array(corners_2d)
        sample[Queries.CORNERS_CAN] = corners_can
        sample[Queries.OBJ_IDX] = self.obj_map[essentials["objname"]]

        corners_vis = self.get_corners_vis(essentials["obj_corners_2d"])
        # corners invisible in raw image
        if corners_vis.sum() < CONST.NUM_CORNERS * 0.4:  # magic number
            sample[Queries.CORNERS_VIS] = np.full(CONST.NUM_CORNERS, 0.0, dtype=np.float32)
        else:
            corners_vis_aug = (((corners_2d[:, 0] >= 0) & (corners_2d[:, 0] < self.image_size[0])) &
                               ((corners_2d[:, 1] >= 0) & (corners_2d[:, 1] < self.image_size[1]))).astype(np.float32)
            if corners_vis_aug.sum() < CONST.NUM_CORNERS * 0.4:  # magic number
                sample[Queries.CORNERS_VIS] = np.full(CONST.NUM_CORNERS, 0.0, dtype=np.float32)
            else:
                sample[Queries.CORNERS_VIS] = corners_vis_aug

        base_trasnf = essentials["pose"].astype(np.float32)
        base_rot = base_trasnf[:3, :3]  # (3, 3)
        base_tsl = base_trasnf[:3, 3:]  # (3, 1)
        trans_rot = rot_mat @ base_rot  # (3, 3)
        trans_tsl = rot_mat.dot(base_tsl)  # (3, 1)
        trans_transf = np.concatenate([trans_rot, trans_tsl], axis=1)  # (3, 4)
        trans_transf = np.concatenate([trans_transf, np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0)
        sample[Queries.OBJ_TRANSF] = trans_transf.astype(np.float32)

        if self.aug:
            blur_radius = Uniform(low=0, high=1).sample().item() * self.blur_radius
            img = img.filter(ImageFilter.GaussianBlur(blur_radius))
            B, C, S, H = img_augment.get_color_params(
                brightness=self.brightness,
                saturation=self.saturation,
                hue=self.hue,
                contrast=self.contrast,
            )
            img = img_augment.apply_jitter(img, brightness=B, contrast=C, saturation=S, hue=H)

        img = img_augment.transform_img(img, affine_transf, self.image_size)
        img = img.crop((0, 0, self.image_size[0], self.image_size[1]))
        img = tvF.to_tensor(img).float()
        img = tvF.normalize(img, [0.5, 0.5, 0.5], [1, 1, 1])
        sample[Queries.IMAGE] = img

        return sample

    def get_center_scale_wrt_bbox(self, essentials):
        # for hand
        if self.require_full_image:
            full_width, full_height = self.raw_size[0], self.raw_size[1]  # 256, 256
            center = np.array((full_width / 2, full_height / 2))
            scale = full_width
            return center, scale

        if self.crop_model == "hand":  # Only use hand joints
            joints2d = essentials["hand_joints_2d"]  # (21, 2)
            center = HOdata.get_annot_center(joints2d)
            scale = HOdata.get_annot_scale(joints2d)
            return center, scale
        elif self.crop_model == "root_obj":
            root_joints2d = essentials["hand_joints_2d"][[0]]  # (1, 2)
            corners_2d = essentials["obj_corners_2d"]  # (8, 2)
            all2d = np.concatenate([root_joints2d, corners_2d], axis=0)  # (9, 2)
            center = HOdata.get_annot_center(all2d)
            scale = HOdata.get_annot_scale(all2d)
            return center, scale
        elif self.crop_model == "hand_obj":
            joints2d = essentials["hand_joints_2d"]  # (21, 2)
            corners_2d = essentials["obj_corners_2d"]  # (8, 2)
            all2d = np.concatenate([joints2d, corners_2d], axis=0)  # (29, 2)
            center = HOdata.get_annot_center(all2d)
            scale = HOdata.get_annot_scale(all2d)
            return center, scale
        else:
            raise NotImplementedError()

    def get_joints_vis(self, joints_2d):
        joints_vis = ((joints_2d[:, 0] >= 0) &
                      (joints_2d[:, 0] < self.raw_size[0])) & ((joints_2d[:, 1] >= 0) &
                                                               (joints_2d[:, 1] < self.raw_size[1]))
        return joints_vis.astype(np.float32)

    def get_corners_vis(self, corners_2d):
        corners_vis = ((corners_2d[:, 0] >= 0) &
                       (corners_2d[:, 0] < self.raw_size[0])) & ((corners_2d[:, 1] >= 0) &
                                                                 (corners_2d[:, 1] < self.raw_size[1]))
        return corners_vis.astype(np.float32)
