import os
import pickle
from abc import ABC
from typing import List

import numpy as np
import trimesh

from anakin.utils import transform


class ObjEngine(ABC):

    def __init__(self):
        self.obj_names = []
        self.obj_meshes = []
        self.obj_corners_can = []
        self.obj_trimeshes_mapping = {}
        self.obj_corners_can_mapping = {}

    @staticmethod
    def build(dataset_type: str, query_obj: List[str]):
        if dataset_type == "HO3D":
            return HO3DObjEngine("assets/ho3d_corners.pkl", query_obj)
        elif dataset_type == "DexYCB":
            return DexYCBObjEngine(query_obj)


class HO3DObjEngine(ObjEngine):

    def __init__(self, corner_file: str, query_obj: List[str]):
        super().__init__()
        obj_corners = pickle.load(open(corner_file, "rb"))
        obj_root = os.path.join("./data", "YCB_models_process")
        cam_extr = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ])
        self.obj_names = []
        self.obj_meshes = []
        self.obj_corners_can = []
        for name in query_obj:

            self.obj_names.append(name)

            # ===== meshes can >>>>>>
            obj_path = os.path.join(obj_root, name, "ds_textured.obj")
            omesh = trimesh.load(obj_path, process=False)
            verts = cam_extr[:3, :3].dot(omesh.vertices.transpose()).transpose()
            verts_can, bbox_center, bbox_scale = transform.center_vert_bbox(verts, scale=False)
            omesh.vertices = verts_can
            self.obj_meshes.append(omesh)

            # ===== corners can >>>>>
            # ! In HO3D, ShSu use a different sugar_box model.
            corners = obj_corners[name]
            corners = cam_extr[:3, :3].dot(corners.transpose()).transpose()
            corners_can = (corners - bbox_center) / bbox_scale
            self.obj_corners_can.append(corners_can)
        self.obj_trimeshes_mapping = {name: m for name, m in zip(self.obj_names, self.obj_meshes)}
        self.obj_corners_can_mapping = {name: c for name, c in zip(self.obj_names, self.obj_corners_can)}


class DexYCBObjEngine(ObjEngine):

    def __init__(self, query_obj: List[str]):
        super().__init__()
        obj_root = os.path.join("./data", "DexYCB/models")

        self.obj_names = []
        self.obj_meshes = []
        self.obj_corners_can = []
        for name in query_obj:
            self.obj_names.append(name)

            # ===== meshes can >>>>>>
            obj_path = os.path.join(obj_root, name, "textured_simple.obj")
            omesh = trimesh.load(obj_path, process=False)
            verts_can, bbox_center, bbox_scale = transform.center_vert_bbox(omesh.vertices, scale=False)

            # ===== corners can >>>>>
            corners = trimesh.bounds.corners(omesh.bounds)
            corners_can = corners - bbox_center

            omesh.vertices = verts_can
            self.obj_meshes.append(omesh)
            self.obj_corners_can.append(corners_can)
        self.obj_trimeshes_mapping = {name: m for name, m in zip(self.obj_names, self.obj_meshes)}
        self.obj_corners_can_mapping = {name: c for name, c in zip(self.obj_names, self.obj_corners_can)}
