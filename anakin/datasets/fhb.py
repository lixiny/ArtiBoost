import hashlib
import json
import os
import pickle
import random
from typing import List

import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import cdist
from termcolor import colored

from anakin.datasets import fhbutils
from anakin.datasets.hodata import HOdata
from anakin.utils import transform
from anakin.utils.builder import DATASET
from anakin.utils.logger import logger
from anakin.utils.misc import enable_lower_param
from anakin.utils.transform import aa_to_rotmat


@DATASET.register_module
class FPHB(HOdata):

    @enable_lower_param
    def __init__(self, **cfg):
        super().__init__(**cfg)

        # ======== FHB params >>>>>>>>>>>>>>>>>>>
        self.split_mode = cfg["SPLIT_MODE"]
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ======== FHB default >>>>>>>>>>>>>>>>>>>
        self.reduce_res = True
        self.mode_opts = ["actions", "subjects"]
        self.subjects = ["Subject_1", "Subject_2", "Subject_3", "Subject_4", "Subject_5", "Subject_6"]
        # Get camera info
        self.cam_extr = np.array([
            [0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
            [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
            [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
            [0, 0, 0, 1],
        ])
        self.cam_intr = np.array([[1395.749023, 0, 935.732544], [0, 1395.749268, 540.681030], [0, 0, 1]])
        self.reorder_idx = np.array([0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20])
        # self.idxs = [0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19]
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        self.obj_map = {"juice": 0, "juice_bottle": 0, "liquid_soap": 1, "milk": 2, "salt": 3}
        self.load_dataset()

    def _preload(self):
        # deal with all the naming and path convention
        self.name = "fhbhands"
        self.root = os.path.join(self.data_root, self.name)
        self.root_supp = os.path.join(self.data_root, f"{self.name}_supp")
        self.root_extra_info = os.path.normpath("assets")
        self.info_root = os.path.join(self.root, "Subjects_info")
        self.info_split = os.path.join(self.root, "data_split_action_recognition.txt")
        small_rgb_root = os.path.join(self.root, "Video_files_480")
        if os.path.exists(small_rgb_root) and self.reduce_res:
            self.rgb_root = small_rgb_root
            self.reduce_factor = float(1 / 4)
        else:
            self.rgb_root = os.path.join(self.root, "Video_files")
            self.reduce_factor = float(1)
        self.skeleton_root = os.path.join(self.root, "Hand_pose_annotation_v1")

        self.rgb_template = "color_{:04d}.jpeg"
        # Joints are numbered from tip to base, we want opposite

        self.cache_identifier_dict = {
            "filter_thresh": float(self.filter_thresh),
            "data_split": self.data_split,
            "split_mode": self.split_mode,
            "reduce_factor": float(self.reduce_factor),
            "fliter_no_contact": self.filter_no_contact,
        }
        self.cache_identifier_raw = json.dumps(self.cache_identifier_dict, sort_keys=True)
        self.cache_identifier = hashlib.md5(self.cache_identifier_raw.encode("ascii")).hexdigest()
        self.cache_path = os.path.join("common", "cache", self.name, "{}.pkl".format(self.cache_identifier))

    def load_dataset(self):
        self._preload()
        cache_folder = os.path.dirname(self.cache_path)
        os.makedirs(cache_folder, exist_ok=True)

        all_objects = ["juice", "liquid_soap", "milk", "salt"]
        if os.path.exists(self.cache_path) and self.use_cache:
            with open(self.cache_path, "rb") as p_f:
                annotations = pickle.load(p_f)
            logger.info(f"Loaded cache for {self.name}_{self.data_split}_{self.split_mode} from {self.cache_path}")
        else:
            subjects_infos = {}
            for subject in self.subjects:
                subject_info_path = os.path.join(self.info_root, "{}_info.txt".format(subject))
                subjects_infos[subject] = {}
                with open(subject_info_path, "r") as subject_f:
                    raw_lines = subject_f.readlines()
                    for line in raw_lines[3:]:
                        line = " ".join(line.split())
                        action, action_idx, length = line.strip().split(" ")
                        subjects_infos[subject][(action, action_idx)] = length
                    subject_f.close()
            skel_info = fhbutils.get_skeletons(self.skeleton_root, subjects_infos)

            with open(self.info_split, "r") as annot_f:
                lines_raw = annot_f.readlines()
                annot_f.close()
            train_list, test_list, all_infos = fhbutils.get_action_train_test(lines_raw, subjects_infos)

            # use object is always ture
            self.fhb_objects = fhbutils.load_objects(
                obj_root=os.path.join(self.root_supp, "Object_models"),
                object_names=all_objects,
            )

            obj_infos = fhbutils.load_object_infos(os.path.join(self.root, "Object_6D_pose_annotation_v1_1"))

            if self.split_mode == "actions":
                if self.data_split == "train":
                    sample_list = train_list
                elif self.data_split == "test":
                    sample_list = test_list
                elif self.data_split == "all":
                    sample_list = {**train_list, **test_list}
                else:
                    raise KeyError(f"Split {self.data_split} not valid for {self.name}, should be [train|test|all]")

            elif self.split_mode == "subjects":
                if self.data_split == "train":
                    subjects = ["Subject_1", "Subject_3", "Subject_4"]
                elif self.data_split == "test":
                    subjects = ["Subject_2", "Subject_5", "Subject_6"]
                else:
                    raise KeyError(f"Split {self.data_split} not in [train|test] for split_type subjects")
                self.subjects = subjects
                sample_list = all_infos

            else:
                raise KeyError(f"split_type should be in [action|subjects], got {self.split_mode}")

            if self.split_mode != "subjects":
                self.subjects = [
                    "Subject_1",
                    "Subject_2",
                    "Subject_3",
                    "Subject_4",
                    "Subject_5",
                    "Subject_6",
                ]
            self.split_objects = self.fhb_objects

            image_names = []
            joints2d = []
            joints3d = []
            hand_sides = []
            clips = []
            sample_infos = []
            obj_names = []
            obj_transforms = []

            for subject, action_name, seq_idx, frame_idx in sample_list:
                if subject not in self.subjects:
                    continue

                # * Skip samples without objects
                if subject not in obj_infos or (action_name, seq_idx, frame_idx) not in obj_infos[subject]:
                    continue

                img_path = os.path.join(self.rgb_root, subject, action_name, seq_idx, "color",
                                        self.rgb_template.format(frame_idx))
                skel = skel_info[subject][(action_name, seq_idx)][frame_idx]
                skel = skel[self.reorder_idx]
                skel_hom = np.concatenate([skel, np.ones([skel.shape[0], 1])], 1)
                skel_camcoords = self.cam_extr.dot(skel_hom.transpose()).transpose()[:, :3].astype(np.float32)

                obj, transf = obj_infos[subject][(action_name, seq_idx, frame_idx)]
                if obj not in self.split_objects:
                    continue

                if self.filter_no_contact:
                    verts = self.split_objects[obj]["verts"]
                    transf_verts = fhbutils.transform_obj_verts(verts, transf, self.cam_extr)
                    all_dists = cdist(transf_verts, skel_camcoords)
                    if all_dists.min() > self.filter_thresh:
                        continue

                # collect the results
                clips.append((subject, action_name, seq_idx))
                obj_transforms.append(transf)
                obj_names.append(obj)

                image_names.append(img_path)
                sample_infos.append({
                    "subject": subject,
                    "action_name": action_name,
                    "seq_idx": seq_idx,
                    "frame_idx": frame_idx,
                })
                joints3d.append(skel_camcoords)
                hom_2d = np.array(self.cam_intr).dot(skel_camcoords.transpose()).transpose()
                skel2d = (hom_2d / hom_2d[:, 2:])[:, :2]
                joints2d.append(skel2d.astype(np.float32))
                hand_sides.append("right")

            # assemble annotation
            mano_objs, mano_infos = fhbutils.load_manofits(sample_infos)
            annotations = {
                "cache_identifier_dict": self.cache_identifier_dict,
                "image_names": image_names,
                "joints2d": joints2d,
                "joints3d": joints3d,
                "hand_sides": hand_sides,
                "sample_infos": sample_infos,
                "mano_infos": mano_infos,
                "mano_objs": mano_objs,
                "obj_names": obj_names,
                "obj_transforms": obj_transforms,
                "split_objects": self.split_objects,
            }

            # dump cache
            with open(self.cache_path, "wb") as fid:
                pickle.dump(annotations, fid)
            logger.info(f"Wrote cache for {self.name}_{self.data_split}_{self.split_mode} to {self.cache_path}")

        # register loaded information into object
        self.image_names = annotations["image_names"]
        self.joints2d = annotations["joints2d"]
        self.joints3d = annotations["joints3d"]
        self.hand_sides = annotations["hand_sides"]
        self.sample_infos = annotations["sample_infos"]
        self.mano_objs = annotations["mano_objs"]
        self.mano_infos = annotations["mano_infos"]
        self.obj_names = annotations["obj_names"]
        self.obj_transforms = annotations["obj_transforms"]
        self.split_objects = annotations["split_objects"]

        self.cam_intr[:2] = self.cam_intr[:2] * self.reduce_factor
        self.raw_size = (int(1920 * self.reduce_factor), int(1080 * self.reduce_factor))
        self.sample_idxs = list(range(len(self.image_names)))
        if self.mini_factor_of_dataset != float(1):
            random.Random(1).shuffle(self.sample_idxs)
            self.sample_idxs = self.sample_idxs[:int(self.mini_factor_of_dataset * len(self.sample_idxs))]

        logger.info(f"{self.name} Got {colored(len(self.sample_idxs), 'yellow', attrs=['bold'])}"
                    f"/{len(self.image_names)} samples for data_split {self.data_split}")
        return True

    def __len__(self):
        return len(self.sample_idxs)

    def get_sample_idxs(self) -> List[int]:
        return self.sample_idxs

    def get_image(self, idx):
        img_path = self.image_names[idx]
        img = Image.open(img_path).convert("RGB")
        return img

    def get_image_path(self, idx):
        return self.image_names[idx]

    # ! Warning! Modify the NumPy array outside may cause an unexpected bug!
    def get_hand_verts_3d(self, idx):
        pose, trans, shape = self._fhb_get_hand_info(idx)
        mano_out = self.mano_layer(torch.Tensor(pose).unsqueeze(0), torch.Tensor(shape).unsqueeze(0))
        verts = mano_out.verts[0].numpy() + trans
        return np.array(verts).astype(np.float32)

    def get_hand_verts_2d(self, idx):
        verts = self.get_hand_verts_3d(idx)
        return HOdata.persp_project(verts, self.cam_intr)

    def get_hand_faces(self, idx):
        faces = np.array(self.mano_layer.th_faces).astype(np.long)
        return faces

    def _fhb_get_hand_info(self, idx):
        """
        Get the hand annotation in the raw fhb datasets.
        !!! This Mehthods shouldn't be called outside.
        :param idx:
        :return:
        """
        mano_info = self.mano_infos[idx]
        return mano_info["fullpose"], mano_info["trans"], mano_info["shape"]

    def get_joints_3d(self, idx):
        joints = self.joints3d[idx]
        return joints / 1000

    def get_joints_2d(self, idx):
        joints = self.joints2d[idx] * self.reduce_factor
        return joints

    def get_obj_faces(self, idx):
        obj = self.obj_names[idx]
        objfaces = self.split_objects[obj]["faces"]
        return np.array(objfaces).astype(np.int32)

    def get_obj_transf_wrt_cam(self, idx):
        verts_can, v_0, _ = self.get_obj_verts_can(idx)
        transf = self.obj_transforms[idx]
        transf = self.cam_extr @ transf
        rot = transf[:3, :3]
        tsl = transf[:3, 3] / 1000.0
        tsl_wrt_cam = rot.dot(v_0) + tsl
        tsl_wrt_cam = tsl_wrt_cam[:, np.newaxis]  # (3, 1)
        obj_transf = np.concatenate([rot, tsl_wrt_cam], axis=1)  # (3, 4)
        obj_transf = np.concatenate([obj_transf, np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0)
        return obj_transf.astype(np.float32)

    def get_obj_pose(self, idx):
        return self.get_obj_transf_wrt_cam(idx)

    def get_obj_transf(self, idx):
        return self.get_obj_transf_wrt_cam(idx)

    def get_obj_rot(self, idx):
        return self.get_obj_transf_wrt_cam(idx)[:3, :3]

    def get_obj_tsl(self, idx):
        return self.get_obj_transf_wrt_cam(idx)[:3, 3]

    def get_obj_verts_transf(self, idx):
        obj = self.obj_names[idx]
        transf = self.obj_transforms[idx]
        verts_raw = self.split_objects[obj]["verts"]
        transf_verts = fhbutils.transform_obj_verts(verts_raw, transf, self.cam_extr) / 1000
        return np.array(transf_verts).astype(np.float32)  # return as meter

    def get_obj_verts_2d(self, idx):
        objpoints3d = self.get_obj_verts_transf(idx)
        hom_2d = np.array(self.cam_intr).dot(objpoints3d.transpose()).transpose()
        verts2d = (hom_2d / hom_2d[:, 2:])[:, :2]
        return verts2d.astype(np.float32)

    def get_obj_verts_can(self, idx):
        obj = self.obj_names[idx]
        verts = self.split_objects[obj]["verts"]
        verts_can, bbox_center, bbox_scale = transform.center_vert_bbox(verts, scale=False)
        return verts_can, bbox_center, bbox_scale

    def get_corners_3d(self, idx):
        obj = self.obj_names[idx]
        transf = self.obj_transforms[idx]
        corners_raw = self.split_objects[obj]["corners"]
        transf_corners = fhbutils.transform_obj_verts(corners_raw, transf, self.cam_extr) / 1000
        return np.array(transf_corners).astype(np.float32)

    def get_corners_2d(self, idx):
        obj_corners3d = self.get_corners_3d(idx)
        obj_corners3d = obj_corners3d * 1000
        hom_2d = np.array(self.cam_intr).dot(obj_corners3d.transpose()).transpose()
        corners2d = (hom_2d / hom_2d[:, 2:])[:, :2]
        return corners2d.astype(np.float32)

    def get_obj_idx(self, idx):
        obj = self.obj_names[idx]
        return self.obj_map[obj]

    def get_corners_can(self, idx):
        obj = self.obj_names[idx]
        corners = self.split_objects[obj]["corners"]
        verts = self.split_objects[obj]["verts"]
        _, center_shift, __ = transform.center_vert_bbox(verts, scale=False)
        corners_can = corners - center_shift
        return corners_can.astype(np.float32)

    def get_cam_intr(self, idx):
        camintr = self.cam_intr
        return camintr.astype(np.float32)

    def get_sides(self, idx):
        side = self.hand_sides[idx]
        return side

    def get_meta(self, idx):
        meta = {"objname": self.obj_names[idx]}
        return meta

    def get_center_scale_wrt_bbox(self, idx):
        if self.require_full_image:
            full_width, full_height = self.raw_size[0], self.raw_size[1]  # 480, 270
            center = np.array((full_width / 2, full_height / 2))
            scale = full_width
            return center, scale
        else:
            # for hand
            joints2d = self.get_joints_2d(idx)  # (21, 2)
            # for obj
            obj_corners2d = self.get_corners_2d(idx)  # (8, 2)
            all2d = np.concatenate([joints2d, obj_corners2d], axis=0)  # (29, 2)
            center = HOdata.get_annot_center(all2d)
            scale = HOdata.get_annot_scale(all2d)
            return center, scale

    def get_obj_verts_can_raw(self, idx):
        obj = self.obj_names[idx]
        verts = self.split_objects[obj]["verts"]
        return verts

    def get_sample_identifier(self, idx):
        res = f"{self.name}__{self.cache_identifier_raw}__{idx}"
        return res

    def get_hand_tsl_wrt_cam(self, idx):
        mano_info = self.mano_infos[idx]
        return mano_info["trans"].astype(np.float32)

    def get_hand_shape(self, idx):
        mano_info = self.mano_infos[idx]
        return mano_info["shape"].astype(np.float32)

    def get_hand_pose_wrt_cam(self, idx):
        mano_info = self.mano_infos[idx]
        return mano_info["fullpose"].astype(np.float32)

    def get_hand_axisang_wrt_cam(self, idx):
        mano_info = self.mano_infos[idx]
        return mano_info["fullpose"][0:3].astype(np.float32)

    def get_hand_rot_wrt_cam(self, idx):
        axisang = self.get_hand_axisang_wrt_cam(idx)
        rot = aa_to_rotmat(axisang)
        return rot.astype(np.float32)
