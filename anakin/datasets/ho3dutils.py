import json
import os
import pickle
import re
import warnings
from collections import defaultdict
from threading import Condition
from typing import Dict

import cv2
import numpy as np
import torch
import trimesh
from anakin.utils.etqdm import etqdm
from anakin.utils.logger import logger
from manotorch.manolayer import MANOOutput, ManoLayer
from scipy.spatial.distance import cdist
from torch import dtype


def load_objects(obj_root):
    object_names = [obj_name for obj_name in sorted(os.listdir(obj_root)) if ".tgz" not in obj_name]
    objects = {}
    for obj_name in object_names:
        obj_path = os.path.join(obj_root, obj_name, "textured_simple_ds.obj")
        mesh = trimesh.load(obj_path, process=False)
        objects[obj_name] = {
            "verts": np.asfarray(mesh.vertices, dtype=np.float32),
            "faces": mesh.faces,
            "corners": np.array(mesh.bounding_box_oriented.vertices),
        }
    return objects


def load_objects_trimesh(obj_root) -> Dict[str, trimesh.base.Trimesh]:
    object_names = [obj_name for obj_name in sorted(os.listdir(obj_root)) if ".tgz" not in obj_name]
    object_meshes = {}
    for obj_name in object_names:
        obj_path = os.path.join(obj_root, obj_name, "ds_textured.obj")
        mesh = trimesh.load(obj_path, process=False)
        object_meshes[obj_name] = mesh
    return object_meshes


def get_v2_frames(name, split, root, trainval_idx=60000, keep_original_order=False):
    v2_train_seqs = {
        "ABF10",
        "ABF11",
        "ABF12",
        "ABF13",
        "ABF14",
        "GPMF10",
        "GPMF11",
        "GPMF12",
        "GPMF13",
        "GPMF14",
        "SB10",
        "SB12",
        "SB14",
        "SM2",
        "SM3",
        "SM4",
        "SM5",
    }
    v2_test_seqs = {
        "SM1",
        "MPM10",
        "MPM11",
        "MPM12",
        "MPM13",
        "MPM14",
        "SB11",
        "SB13",
        "AP10",
        "AP11",
        "AP12",
        "AP13",
        "AP14",
    }
    if name != "HO3D":
        root = root.replace(name, "HO3D")
    if split in ["train", "trainval", "val"]:
        info_path = os.path.join(root, "train.txt")
        subfolder = "train"
    elif split == "test":
        info_path = os.path.join(root, "evaluation.txt")
        subfolder = "evaluation"
    else:
        assert False
    with open(info_path, "r") as f:
        lines = f.readlines()
    txt_seq_frames = [line.strip().split("/") for line in lines]
    if split == "trainval":
        txt_seq_frames = txt_seq_frames[:trainval_idx]
    elif split == "val":
        txt_seq_frames = txt_seq_frames[trainval_idx:]
    seqs = {}
    ordered_seq_frames = []  # return HO3D original order in train.txt/evaluation.txt
    for sf in txt_seq_frames:
        if sf[0] not in v2_train_seqs and sf[0] not in v2_test_seqs:
            continue
        if sf[0] in seqs:
            seqs[sf[0]].append(sf[1])
        else:
            seqs[sf[0]] = [sf[1]]
        ordered_seq_frames.append([sf[0], sf[1]])

    seq_frames = []
    for s in seqs:
        seqs[s].sort()
        for f in range(len(seqs[s])):
            seq_frames.append([s, seqs[s][f]])
    return ordered_seq_frames if keep_original_order else seq_frames, subfolder


def min_contact_dis(annot, obj_meshes):
    cam_extr = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    if not hasattr(min_contact_dis, "mano_layer"):
        min_contact_dis.mano_layer = ManoLayer(
            rot_mode="axisang",
            use_pca=False,
            mano_root="assets/mano_v1_2",
            center_idx=None,
            flat_hand_mean=True,
        )
    rot = cv2.Rodrigues(annot["objRot"])[0]
    trans = annot["objTrans"]
    obj_id = annot["objName"]
    verts = obj_meshes[obj_id]["verts"]
    trans_verts = rot.dot(verts.transpose()).transpose() + trans
    trans_verts = cam_extr[:3, :3].dot(trans_verts.transpose()).transpose()
    obj_verts = np.array(trans_verts).astype(np.float32)

    handpose = annot["handPose"]
    handtrans = annot["handTrans"]
    handshape = annot["handBeta"]

    mano_out: MANOOutput = min_contact_dis.mano_layer(
        torch.Tensor(handpose).unsqueeze(0), torch.Tensor(handshape).unsqueeze(0)
    )
    handverts = mano_out.verts[0].numpy() + handtrans
    trans_handverts = cam_extr[:3, :3].dot(handverts.transpose()).transpose()

    all_dist = cdist(trans_handverts, obj_verts) * 1000

    return all_dist.min()


def get_v1_seqs(split, name, filtered=True):
    if split == "train":
        if not filtered:
            seqs = {"SM5", "MC6", "MC4", "SM3", "SM4", "SS3", "SS2", "SM2", "SS1", "MC5", "MC1"}
        else:
            seqs = {"MC6", "MC4", "MC5", "MC1"}
        subfolder = "train"
    elif split == "test":
        seqs = {"MC2"}
        subfolder = "train"
        logger.info(f"Using seqs {seqs} for evaluation")
    elif split == "all":  # ! deprecated
        seqs = {"MC1", "MC2", "MC4", "MC5", "MC6", "SM2", "SM3", "SM4", "SM5", "SS1", "SS2", "SS3"}
        subfolder = "train"
        version_descriptor = "v1"
        logger.info(f"Using seqs {seqs} for all, version {version_descriptor}")
    # ! Following splits only depend on split name
    elif split == "all_all":
        seqs = {
            "ABF10",
            "ABF11",
            "ABF12",
            "ABF13",
            "ABF14",
            "BB10",
            "BB11",
            "BB12",
            "BB13",
            "BB14",
            "GPMF10",
            "GPMF11",
            "GPMF12",
            "GPMF13",
            "GPMF14",
            "GSF10",
            "GSF11",
            "GSF12",
            "GSF13",
            "GSF14",
            "MC1",
            "MC2",
            "MC4",
            "MC5",
            "MC6",
            "MDF10",
            "MDF11",
            "MDF12",
            "MDF13",
            "MDF14",
            "SB10",
            "SB12",
            "SB14",
            "SM2",
            "SM3",
            "SM4",
            "SM5",
            "SMu1",
            "SMu40",
            "SMu41",
            "SMu42",
            "SS1",
            "SS2",
            "SS3",
            "ShSu10",
            "ShSu12",
            "ShSu13",
            "ShSu14",
            "SiBF10",
            "SiBF11",
            "SiBF12",
            "SiBF13",
            "SiBF14",
        }
        subfolder = "train"
        logger.info(f"Using seqs {seqs} for total_dataset, regardless of version")
    else:
        assert False, "split mode not found!"
    return seqs, subfolder


def get_seq_object(seq):
    mapping = {
        "ABF10": "021_bleach_cleanser",
        "ABF11": "021_bleach_cleanser",
        "ABF12": "021_bleach_cleanser",
        "ABF13": "021_bleach_cleanser",
        "ABF14": "021_bleach_cleanser",
        "BB10": "011_banana",
        "BB11": "011_banana",
        "BB12": "011_banana",
        "BB13": "011_banana",
        "BB14": "011_banana",
        "GPMF10": "010_potted_meat_can",
        "GPMF11": "010_potted_meat_can",
        "GPMF12": "010_potted_meat_can",
        "GPMF13": "010_potted_meat_can",
        "GPMF14": "010_potted_meat_can",
        "GSF10": "037_scissors",
        "GSF11": "037_scissors",
        "GSF12": "037_scissors",
        "GSF13": "037_scissors",
        "GSF14": "037_scissors",
        "MC1": "003_cracker_box",
        "MC2": "003_cracker_box",
        "MC4": "003_cracker_box",
        "MC5": "003_cracker_box",
        "MC6": "003_cracker_box",
        "MDF10": "035_power_drill",
        "MDF11": "035_power_drill",
        "MDF12": "035_power_drill",
        "MDF13": "035_power_drill",
        "MDF14": "035_power_drill",
        "ND2": "035_power_drill",
        "SB10": "021_bleach_cleanser",
        "SB12": "021_bleach_cleanser",
        "SB14": "021_bleach_cleanser",
        "SM2": "006_mustard_bottle",
        "SM3": "006_mustard_bottle",
        "SM4": "006_mustard_bottle",
        "SM5": "006_mustard_bottle",
        "SMu1": "025_mug",
        "SMu40": "025_mug",
        "SMu41": "025_mug",
        "SMu42": "025_mug",
        "SS1": "004_sugar_box",
        "SS2": "004_sugar_box",
        "SS3": "004_sugar_box",
        "ShSu10": "004_sugar_box",
        "ShSu12": "004_sugar_box",
        "ShSu13": "004_sugar_box",
        "ShSu14": "004_sugar_box",
        "SiBF10": "011_banana",
        "SiBF11": "011_banana",
        "SiBF12": "011_banana",
        "SiBF13": "011_banana",
        "SiBF14": "011_banana",
        "SiS1": "004_sugar_box",
        # test
        "SM1": "006_mustard_bottle",
        "MPM10": "010_potted_meat_can",
        "MPM11": "010_potted_meat_can",
        "MPM12": "010_potted_meat_can",
        "MPM13": "010_potted_meat_can",
        "MPM14": "010_potted_meat_can",
        "SB11": "021_bleach_cleanser",
        "SB13": "021_bleach_cleanser",
        "AP10": "019_pitcher_base",
        "AP11": "019_pitcher_base",
        "AP12": "019_pitcher_base",
        "AP13": "019_pitcher_base",
        "AP14": "019_pitcher_base",
    }
    obj_set = set()
    for s in seq:
        obj_set.add(mapping[s])
    return obj_set, mapping
