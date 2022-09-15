import hashlib
import json
import os
import pickle
import random
from collections import defaultdict
from copy import deepcopy
from typing import List, Tuple

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
from deprecated.sphinx import deprecated
from termcolor import colored

from anakin.datasets import ho3dutils
from anakin.datasets.hodata import HOdata
from anakin.utils import transform
from anakin.utils.builder import DATASET
from anakin.utils.etqdm import etqdm
from anakin.utils.logger import logger
from anakin.utils.misc import CONST, enable_lower_param
from anakin.utils.transform import aa_to_rotmat, rotmat_to_aa


@DATASET.register_module
class HO3D(HOdata):

    @enable_lower_param
    def __init__(self, **cfg):
        super().__init__(**cfg)

        # ======== HO3D params >>>>>>>>>>>>>>>>>>
        self.split_mode = cfg["SPLIT_MODE"]
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ======== HO3D default >>>>>>>>>>>>>>>>>
        self.raw_size = (640, 480)
        self.reorder_idxs = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])
        # this camera extrinsic has no translation
        # and this is the reason transforms in following code just use rotation part
        self.cam_extr = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]).astype(np.float32)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        self.load_dataset()

    def _preload(self):
        # deal with all the naming and path convention
        self.name = "HO3D"
        self.root = os.path.join(self.data_root, self.name)
        self.root_supp = os.path.join(self.data_root, f"{self.name}_supp")
        self.root_extra_info = os.path.normpath("assets")

        self.cache_identifier_dict = {
            "filter_thresh": float(self.filter_thresh),
            "data_split": self.data_split,
            "split_mode": self.split_mode,
            "fliter_no_contact": self.filter_no_contact,
        }
        self.cache_identifier_raw = json.dumps(self.cache_identifier_dict, sort_keys=True)
        self.cache_identifier = hashlib.md5(self.cache_identifier_raw.encode("ascii")).hexdigest()
        self.cache_path = os.path.join("common", "cache", self.name, "{}.pkl".format(self.cache_identifier))

    def load_dataset(self):
        self._preload()
        self.obj_meshes = ho3dutils.load_objects(os.path.join(self.data_root, "YCB_models_supp"))
        self.obj_trimeshes = ho3dutils.load_objects_trimesh(os.path.join(self.data_root, "YCB_models_process"))

        cache_folder = os.path.dirname(self.cache_path)
        os.makedirs(cache_folder, exist_ok=True)

        if self.split_mode == "v1":
            seqs, subfolder = ho3dutils.get_v1_seqs(self.data_split, self.name)
            logger.info(f"{self.name} {self.data_split} set has sequence {seqs}")
            seq_frames, subfolder = self._load_seq_frames(subfolder, seqs)
        elif self.split_mode == "paper":  # official paper split V2 and V3
            # For details:
            # V2: https://competitions.codalab.org/competitions/22485
            # V3: https://competitions.codalab.org/competitions/33267
            seq_frames, subfolder = self._load_seq_frames()
            logger.info(f"{self.name} {self.data_split} set has frames {len(seq_frames)}")
        elif self.split_mode == "v2":  # TODO delete this before release
            seq_frames, subfolder = ho3dutils.get_v2_frames(self.name,
                                                            self.data_split,
                                                            self.root,
                                                            keep_original_order=(self.data_split == "test"))
            logger.info(f"{self.name} {self.data_split} set has frames {len(seq_frames)}")
        else:
            raise NotImplementedError()

        if os.path.exists(self.cache_path) and self.use_cache:
            with open(self.cache_path, "rb") as p_f:
                annotations = pickle.load(p_f)
            logger.info(f"Loaded cache for {self.name}_{self.data_split}_{self.split_mode} from {self.cache_path}")
        else:
            annot_mapping, seq_idx = self._load_annots(obj_meshes=self.obj_meshes,
                                                       seq_frames=seq_frames,
                                                       subfolder=subfolder)

            annotations = {"seq_idx": seq_idx, "annot_mapping": annot_mapping}

            with open(self.cache_path, "wb") as p_f:
                pickle.dump(annotations, p_f)
            logger.info(f"Wrote cache for {self.name}_{self.data_split}_{self.split_mode} to {self.cache_path}")

        self.seq_idx = annotations["seq_idx"]
        self.annot_mapping = annotations["annot_mapping"]
        self.sample_idxs = list(range(len(self.seq_idx)))
        self.obj_mapping_name2id = {v: k for k, v in CONST.YCB_IDX2CLASSES.items()}
        self.obj_mapping_id2name = CONST.YCB_IDX2CLASSES
        if self.mini_factor_of_dataset != float(1):
            random.Random(1).shuffle(self.sample_idxs)
            self.sample_idxs = self.sample_idxs[:int(self.mini_factor_of_dataset * len(self.sample_idxs))]

        logger.info(f"{self.name} Got {colored(len(self.sample_idxs), 'yellow', attrs=['bold'])}"
                    f"/{len(self.seq_idx)} samples for data_split {self.data_split}")

    def _load_seq_frames(self, subfolder=None, seqs=None, trainval_idx=6000):
        """
        trainval_idx (int): How many frames to include in training split when
                using trainval/val/test split
        """
        if self.split_mode == "paper":
            if self.data_split in ["train", "trainval", "val"]:
                info_path = os.path.join(self.root, "train.txt")
                subfolder = "train"
            elif self.data_split == "test":
                info_path = os.path.join(self.root, "evaluation.txt")
                subfolder = "evaluation"
            else:
                assert False
            with open(info_path, "r") as f:
                lines = f.readlines()
            seq_frames = [line.strip().split("/") for line in lines]
            if self.data_split == "trainval":
                seq_frames = seq_frames[:trainval_idx]
            elif self.data_split == "val":
                seq_frames = seq_frames[trainval_idx:]
        elif self.split_mode == "v1":
            seq_frames = []
            for seq in sorted(seqs):
                seq_folder = os.path.join(self.root, subfolder, seq)
                meta_folder = os.path.join(seq_folder, "meta")
                img_nb = len(os.listdir(meta_folder))
                for img_idx in range(img_nb):
                    seq_frames.append([seq, f"{img_idx:04d}"])
        else:
            assert False
        return seq_frames, subfolder

    def _load_annots(self, obj_meshes={}, seq_frames=[], subfolder="train"):
        seq_idx = []
        annot_mapping = defaultdict(list)
        seq_counts = defaultdict(int)
        for idx_count, (seq, frame_idx) in enumerate(etqdm(seq_frames)):
            seq_folder = os.path.join(self.root, subfolder, seq)
            meta_folder = os.path.join(seq_folder, "meta")
            rgb_folder = os.path.join(seq_folder, "rgb")

            meta_path = os.path.join(meta_folder, f"{frame_idx}.pkl")

            with open(meta_path, "rb") as p_f:
                annot = pickle.load(p_f)
                if annot["handJoints3D"].size == 3:
                    annot["handTrans"] = annot["handJoints3D"]
                    annot["handJoints3D"] = annot["handJoints3D"][np.newaxis, :].repeat(21, 0)
                    annot["handPose"] = np.zeros(48, dtype=np.float32)
                    annot["handBeta"] = np.zeros(10, dtype=np.float32)

            # filter no contact
            if self.filter_no_contact and ho3dutils.min_contact_dis(annot, obj_meshes) > self.filter_thresh:
                continue

            img_path = os.path.join(rgb_folder, f"{frame_idx}.png")
            annot["img"] = img_path
            annot["frame_idx"] = frame_idx

            annot_mapping[seq].append(annot)
            seq_idx.append((seq, seq_counts[seq]))
            seq_counts[seq] += 1

        return annot_mapping, seq_idx

    def __len__(self):
        return len(self.sample_idxs)

    def get_sample_idxs(self) -> List[int]:
        return self.sample_idxs

    def obj_load_driver(self) -> Tuple[List[str], List[trimesh.base.Trimesh], List[np.ndarray]]:
        obj_names = []
        obj_meshes = []
        obj_corners_can = []
        seqs = [seq for seq in self.annot_mapping.keys()]
        for seq in seqs:
            obj_name = self.annot_mapping[seq][0]["objName"]
            if obj_name in obj_names:
                continue
            obj_names.append(obj_name)

            # ===== meshes can >>>>>>
            omesh = deepcopy(self.obj_trimeshes[obj_name])
            verts = self.cam_extr[:3, :3].dot(omesh.vertices.transpose()).transpose()
            verts_can, bbox_center, bbox_scale = transform.center_vert_bbox(verts, scale=False)
            omesh.vertices = verts_can
            obj_meshes.append(omesh)

            # ===== corners can >>>>>
            corners = self.annot_mapping[seq][0]["objCorners3DRest"]
            corners = self.cam_extr[:3, :3].dot(corners.transpose()).transpose()
            corners_can = (corners - bbox_center) / bbox_scale
            obj_corners_can.append(corners_can)
        return (obj_names, obj_meshes, obj_corners_can)

    def get_seq_frame(self, idx):
        seq, img_idx = self.seq_idx[idx]
        annot = self.annot_mapping[seq][img_idx]
        frame_idx = annot["frame_idx"]
        return seq, frame_idx

    def get_image(self, idx):
        img_path = self.get_image_path(idx)
        img = Image.open(img_path).convert("RGB")
        return img

    def get_image_path(self, idx):
        seq, img_idx = self.seq_idx[idx]
        img_path = self.annot_mapping[seq][img_idx]["img"]
        return img_path

    def _ho3d_get_hand_info(self, idx):
        """
        Get the hand annotation in the raw ho3d datasets.
        !!! This Mehthods shoudln't be called outside.
        :param idx:
        :return: raw hand pose, translate and shape coefficients
        """
        seq, img_idx = self.seq_idx[idx]
        annot = self.annot_mapping[seq][img_idx]
        # Retrieve hand info
        handpose = annot["handPose"]
        handtsl = annot["handTrans"]
        handshape = annot["handBeta"]
        return handpose, handtsl, handshape

    def get_hand_verts_3d(self, idx):
        _handpose, _handtsl, _handshape = self._ho3d_get_hand_info(idx)
        mano_out = self.mano_layer(
            torch.from_numpy(_handpose).unsqueeze(0),
            torch.from_numpy(_handshape).unsqueeze(0),
        )
        # important modify!!!!
        handverts = mano_out.verts[0].numpy() + _handtsl
        transf_handverts = self.cam_extr[:3, :3].dot(handverts.transpose()).transpose()
        return transf_handverts.astype(np.float32)

    def get_hand_verts_2d(self, idx):
        verts_3d = self.get_hand_verts_3d(idx)
        cam_intr = self.get_cam_intr(idx)
        return HOdata.persp_project(verts_3d, cam_intr)

    def get_joints_3d(self, idx):
        seq, img_idx = self.seq_idx[idx]
        annot = self.annot_mapping[seq][img_idx]
        joints_3d = annot["handJoints3D"]
        joints_3d = self.cam_extr[:3, :3].dot(joints_3d.transpose()).transpose()
        joints_3d = joints_3d[self.reorder_idxs]
        return joints_3d.astype(np.float32)

    def get_hand_faces(self, idx):
        faces = np.array(self.mano_layer.th_faces).astype(np.long)
        return faces

    def get_joints_2d(self, idx):
        joints_3d = self.get_joints_3d(idx)
        cam_intr = self.get_cam_intr(idx)
        return HOdata.persp_project(joints_3d, cam_intr)

    def get_center_scale_wrt_bbox(self, idx):
        # ============== FULL IMAGE MODE >>>>>>>>>
        if self.require_full_image:
            full_width, full_height = self.raw_size[0], self.raw_size[1]  # 480, 270
            center = np.array((full_width / 2, full_height / 2))
            scale = full_width
            return center, scale

        if self.crop_model == "hand":  # Only use hand joints or hand bbox
            if self.data_split == "train" or (self.data_split == "test" and self.split_mode == "v1"):
                joints2d = self.get_joints_2d(idx)  # (21, 2)
                center = HOdata.get_annot_center(joints2d)
                scale = HOdata.get_annot_scale(joints2d)
                return center, scale
            elif self.data_split == "test":  # No gt joints annot, using handBoundingBox
                seq, img_idx = self.seq_idx[idx]
                annot = self.annot_mapping[seq][img_idx]
                hand_bbox_coord = annot["handBoundingBox"]  # (x0, y0, x1, y1)
                hand_bbox_2d = np.array(
                    [
                        [hand_bbox_coord[0], hand_bbox_coord[1]],
                        [hand_bbox_coord[2], hand_bbox_coord[3]],
                    ],
                    dtype=np.float32,
                )
                center = HOdata.get_annot_center(hand_bbox_2d)
                scale = HOdata.get_annot_scale(hand_bbox_2d)
                scale = scale
                return center, scale
            else:
                raise RuntimeError()

        elif self.crop_model == "root_obj":  # Only use hand root and obj bbox for crop
            root_joints2d = self.get_joints_2d(idx)[[0]]  # (1, 2)
            corners_2d = self.get_corners_2d(idx)  # (8, 2)
            all2d = np.concatenate([root_joints2d, corners_2d], axis=0)  # (9, 2)
            center = HOdata.get_annot_center(all2d)
            scale = HOdata.get_annot_scale(all2d)
            return center, scale

        elif self.crop_model == "hand_obj":
            if self.data_split == "train" or (self.data_split == "test" and self.split_mode == "v1"):
                joints2d = self.get_joints_2d(idx)  # (21, 2)
                corners_2d = self.get_corners_2d(idx)  # (8, 2)
                all2d = np.concatenate([joints2d, corners_2d], axis=0)  # (29, 2)
                center = HOdata.get_annot_center(all2d)
                scale = HOdata.get_annot_scale(all2d)
                return center, scale
            elif self.data_split == "test":  # No gt joints annot, using handBoundingBox
                seq, img_idx = self.seq_idx[idx]
                annot = self.annot_mapping[seq][img_idx]
                hand_bbox_coord = annot["handBoundingBox"]  # (x0, y0, x1, y1)
                hand_bbox_2d = np.array(
                    [
                        [hand_bbox_coord[0], hand_bbox_coord[1]],
                        [hand_bbox_coord[2], hand_bbox_coord[3]],
                    ],
                    dtype=np.float32,
                )
                corners_2d = self.get_corners_2d(idx)  # (8, 2)
                all2d = np.concatenate([hand_bbox_2d, corners_2d], axis=0)  # (29, 2)
                center = HOdata.get_annot_center(all2d)
                scale = HOdata.get_annot_scale(all2d)
                scale = scale
                return center, scale
            else:
                raise RuntimeError()
        else:
            raise NotImplementedError()

    def get_obj_faces(self, idx):
        seq, img_idx = self.seq_idx[idx]
        annot = self.annot_mapping[seq][img_idx]
        obj_name = annot["objName"]
        objfaces = self.obj_meshes[obj_name]["faces"]
        objfaces = np.array(objfaces).astype(np.int32)
        return objfaces

    def get_obj_idx(self, idx):
        seq, img_idx = self.seq_idx[idx]
        annot = self.annot_mapping[seq][img_idx]
        obj_name = annot["objName"]
        return self.obj_mapping_name2id[obj_name]

    def get_meta(self, idx):
        seq, img_idx = self.seq_idx[idx]
        annot = self.annot_mapping[seq][img_idx]
        meta = {"objname": annot["objName"]}
        return meta

    def get_obj_verts_can(self, idx):
        seq, img_idx = self.seq_idx[idx]
        annot = self.annot_mapping[seq][img_idx]
        obj_name = annot["objName"]
        verts = self.obj_meshes[obj_name]["verts"]
        verts = self.cam_extr[:3, :3].dot(verts.transpose()).transpose()

        # NOTE: verts_can = verts - bbox_center
        verts_can, bbox_center, bbox_scale = transform.center_vert_bbox(verts, scale=False)  # !! CENTERED HERE
        return np.asfarray(verts_can, dtype=np.float32), bbox_center, bbox_scale

    def get_obj_verts_can_by_obj_id(self, obj_id):
        obj_name = self.obj_mapping_id2name[obj_id]
        verts = self.obj_meshes[obj_name]["verts"]
        verts = self.cam_extr[:3, :3].dot(verts.transpose()).transpose()
        # NOTE: verts_can = verts - bbox_center
        verts_can, bbox_center, bbox_scale = transform.center_vert_bbox(verts, scale=False)  # !! CENTERED HERE
        return np.asfarray(verts_can, dtype=np.float32), bbox_center, bbox_scale

    def get_obj_faces_by_obj_id(self, obj_id):
        obj_name = self.obj_mapping_id2name[obj_id]
        objfaces = self.obj_meshes[obj_name]["faces"]
        objfaces = np.array(objfaces).astype(np.int32)
        return objfaces

    def get_obj_verts_transf(self, idx):
        seq, img_idx = self.seq_idx[idx]
        annot = self.annot_mapping[seq][img_idx]
        rot = cv2.Rodrigues(annot["objRot"])[0]
        tsl = annot["objTrans"]
        obj_name = annot["objName"]

        # This verts IS NOT EQUAL to the one in get_obj_verts_can,
        # since this verts is not translated to vertices center
        verts = self.obj_meshes[obj_name]["verts"]
        transf_verts = rot.dot(verts.transpose()).transpose() + tsl
        transf_verts = self.cam_extr[:3, :3].dot(transf_verts.transpose()).transpose()
        return np.array(transf_verts).astype(np.float32)

    def get_obj_transf_wrt_cam(self, idx):
        seq, img_idx = self.seq_idx[idx]
        annot = self.annot_mapping[seq][img_idx]
        rot = cv2.Rodrigues(annot["objRot"])[0]
        tsl = annot["objTrans"]

        verts_can, v_0, _ = self.get_obj_verts_can(idx)  # (N, 3), (3, ), 1
        """ HACK
        v_{can} = E * v_{raw} - v_0
        v_{cam} = E * (R * v_{raw} + t)

        => v_{raw} = E^{-1} * (v_{can} + v_0)
        => v_{cam} = E * (R * (E^{-1} * (v_{can} + v_0)) + t)
        =>         = E*R*E^{-1} * v_{can} + E*R*E^{-1} * v_0 + E * t
        """

        ext_rot = self.cam_extr[:3, :3]
        ext_rot_inv = np.linalg.inv(ext_rot)

        rot_wrt_cam = ext_rot @ (rot @ ext_rot_inv)  # (3, 3)
        tsl_wrt_cam = (ext_rot @ (rot @ ext_rot_inv)).dot(v_0) + ext_rot.dot(tsl)  # (3,)
        tsl_wrt_cam = tsl_wrt_cam[:, np.newaxis]  # (3, 1)

        obj_transf = np.concatenate([rot_wrt_cam, tsl_wrt_cam], axis=1)  # (3, 4)
        obj_transf = np.concatenate([obj_transf, np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0)
        return obj_transf.astype(np.float32)

    # for compatibility
    def get_obj_transf(self, idx):
        return self.get_obj_transf_wrt_cam(idx)

    # for compatibility
    def get_obj_pose(self, idx):
        return self.get_obj_transf_wrt_cam(idx)

    def get_obj_rot(self, idx):
        return self.get_obj_transf_wrt_cam(idx)[:3, :3]

    def get_obj_tsl(self, idx):
        return self.get_obj_transf_wrt_cam(idx)[:3, 3]

    def get_corners_3d(self, idx):
        corners = self.get_corners_can(idx)
        obj_transf = self.get_obj_transf_wrt_cam(idx)
        obj_rot = obj_transf[:3, :3]  # (3, 3)
        obj_tsl = obj_transf[:3, 3:]  # (3, 1)
        obj_corners_transf = (obj_rot.dot(corners.transpose()) + obj_tsl).transpose()
        return obj_corners_transf.astype(np.float32)

    @deprecated(reason="X use the Trimesh corners, HO3D annotated corners instead", version="0.1")
    def get_corners_can_(self, idx):
        seq, img_idx = self.seq_idx[idx]
        annot = self.annot_mapping[seq][img_idx]
        obj_name = annot["objName"]
        corners = self.obj_meshes[obj_name]["corners"]
        corners = self.cam_extr[:3, :3].dot(corners.transpose()).transpose()

        _, obj_cantrans, obj_canscale = self.get_obj_verts_can(idx)
        # NOTE: verts_can = verts - bbox_center
        obj_cancorners = (corners - obj_cantrans) / obj_canscale
        return obj_cancorners.astype(np.float32)

    def get_corners_can(self, idx):
        _, obj_cantrans, obj_canscale = self.get_obj_verts_can(idx)
        seq, img_idx = self.seq_idx[idx]
        annot = self.annot_mapping[seq][img_idx]
        corners = annot["objCorners3DRest"]
        corners = self.cam_extr[:3, :3].dot(corners.transpose()).transpose()
        obj_cancorners = (corners - obj_cantrans) / obj_canscale
        return obj_cancorners.astype(np.float32)

    def get_corners_2d(self, idx):
        corners_3d = self.get_corners_3d(idx)
        cam_intr = self.get_cam_intr(idx)
        return HOdata.persp_project(corners_3d, cam_intr)

    def get_obj_verts_2d(self, idx):
        objpoints3d = self.get_obj_verts_transf(idx)
        cam_intr = self.get_cam_intr(idx)
        verts_2d = HOdata.persp_project(objpoints3d, cam_intr)
        return verts_2d

    def get_obj_vis2d(self, idx):
        objvis = np.ones_like(self.get_obj_verts_2d(idx)[:, 0])
        return objvis

    def get_hand_vis2d(self, idx):
        handvis = np.ones_like(self.get_hand_verts_2d(idx)[:, 0])
        return handvis

    def get_sides(self, idx):
        return "right"

    def get_cam_intr(self, idx):
        seq, img_idx = self.seq_idx[idx]
        cam_intr = self.annot_mapping[seq][img_idx]["camMat"]
        return cam_intr

    def get_annot(self, idx):
        seq, img_idx = self.seq_idx[idx]
        annot = self.annot_mapping[seq][img_idx]
        return annot

    def get_sample_identifier(self, idx):
        res = f"{self.name}__{self.cache_identifier_raw}__{idx}"
        return res

    # ? only used in offline eval
    def get_hand_pose_wrt_cam(self, idx):  # pose = root_rot + ...
        seq, img_idx = self.seq_idx[idx]
        annot = self.annot_mapping[seq][img_idx]
        handpose = annot["handPose"]
        # only the first 3 dimension needs to be transformed by cam_extr
        root, remains = handpose[:3], handpose[3:]
        root = rotmat_to_aa(self.cam_extr[:3, :3] @ aa_to_rotmat(root))
        handpose_transformed = np.concatenate((root, remains), axis=0)
        return handpose_transformed.astype(np.float32)

    def get_hand_tsl_wrt_cam(self, idx):
        hand_pose = torch.from_numpy(self.get_hand_pose_wrt_cam(idx)).unsqueeze(0)
        hand_shape = torch.from_numpy(self.get_hand_shape(idx)).unsqueeze(0)

        mano_out = self.mano_layer(hand_pose, hand_shape)
        hand_verts = np.array(mano_out.verts.squeeze(0))
        tsl = self.get_hand_verts_3d(idx) - hand_verts
        return tsl[0]

    # ? only used in offline eval
    def get_hand_axisang_wrt_cam(self, idx):
        rootRot = self.get_hand_rot_wrt_cam(idx)
        root = rotmat_to_aa(rootRot)
        return root.astype(np.float32)

    # ? only used in offline eval
    def get_hand_rot_wrt_cam(self, idx):
        seq, img_idx = self.seq_idx[idx]
        annot = self.annot_mapping[seq][img_idx]
        handpose = annot["handPose"]
        # only the first 3 dimension needs to be transformed by cam_extr
        root = handpose[:3]
        rootRot = self.cam_extr[:3, :3] @ aa_to_rotmat(root)
        return rootRot.astype(np.float32)

    # ? only used in offline eval
    def get_hand_shape(self, idx):
        seq, img_idx = self.seq_idx[idx]
        annot = self.annot_mapping[seq][img_idx]
        handshape = annot["handBeta"]
        return handshape.astype(np.float32)

    def get_hand_pose(self, idx):
        return self.get_hand_pose_wrt_cam(idx)

    def get_hand_tsl(self, idx):
        return self.get_hand_tsl_wrt_cam(idx)


@DATASET.register_module
class HO3DV3(HO3D):

    def _preload(self):
        # deal with all the naming and path convention
        self.name = "HO3D_v3"
        self.root = os.path.join(self.data_root, self.name)
        self.root_extra_info = os.path.normpath("assets")
        assert self.split_mode == "paper", "HO3D_v3 only support paper split."

        self.cache_identifier_dict = {
            "filter_thresh": float(self.filter_thresh),
            "data_split": self.data_split,
            "split_mode": self.split_mode,
            "fliter_no_contact": self.filter_no_contact,
        }
        self.cache_identifier_raw = json.dumps(self.cache_identifier_dict, sort_keys=True)
        self.cache_identifier = hashlib.md5(self.cache_identifier_raw.encode("ascii")).hexdigest()
        self.cache_path = os.path.join("common", "cache", self.name, "{}.pkl".format(self.cache_identifier))

    def _load_annots(self, obj_meshes={}, seq_frames=[], subfolder="train"):
        annot_mapping, seq_idx = super()._load_annots(obj_meshes=obj_meshes, seq_frames=seq_frames, subfolder=subfolder)
        for seq in annot_mapping.values():
            for annot in seq:
                annot["img"] = annot["img"].replace(".png", ".jpg")  # In HO3D V3, image is given in the form of jpg
        return annot_mapping, seq_idx

    def _ho3d_get_hand_info(self, idx):
        handpose, handtsl, handshape = super()._ho3d_get_hand_info(idx)
        return (
            np.asfarray(handpose, dtype=np.float32),
            np.asfarray(handtsl, dtype=np.float32),
            np.asfarray(handshape, dtype=np.float32),
        )
