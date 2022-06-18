import os
from functools import lru_cache
from time import time
from typing import Dict, List, NamedTuple, Optional, Union

import cv2
import numpy as np
import pyrender
import trimesh
from anakin.utils import builder
from anakin.utils.frender_utils import FMesh, FOffscreenRenderer, FScene
from anakin.utils.logger import logger
from anakin.utils.misc import CONST
from PIL.Image import Image, open


def get_mapping(vertices_dup):
    map_idx = []
    vp = 0
    v_before = np.array([np.inf, np.inf, np.inf])
    for v in vertices_dup:
        if np.allclose(v, v_before):
            map_idx.append(vp - 1)
        else:
            map_idx.append(vp)
            vp += 1
        v_before = v
    return map_idx


@lru_cache(maxsize=None)
def get_motion_blur_k(motion_blur):
    bsize = motion_blur
    kernel_motion_blur = np.zeros((bsize, bsize))
    kernel_motion_blur[int((bsize - 1) / 2), :] = np.ones(bsize)
    kernel_motion_blur = kernel_motion_blur / bsize
    return kernel_motion_blur


PointLight = NamedTuple("PointLight", [("color", np.ndarray), ("intensity", float), ("pose", np.ndarray)])
DirectionalLight = NamedTuple("DirectionalLight", [("color", np.ndarray), ("intensity", float), ("pose", np.ndarray)])


class Renderer:
    def __init__(
        self,
        width: int,
        height: int,
        gpu_id: int = 0,
    ) -> None:
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        self.width = width
        self.height = height
        self.r = FOffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0, gpu_id=gpu_id)

    def setup(
        self,
        cam_intr: np.ndarray,
        cam_extr: np.ndarray,
        obj_meshes: Dict[str, trimesh.base.Trimesh],
        hand_meshes: List[trimesh.base.Trimesh],
        backgrounds: Optional[List[Image]] = None,
        lights: Optional[List[Union[PointLight, DirectionalLight]]] = None,
    ):
        # >>>> get node meshes >>>>
        self.obj_meshes = {
            name: FMesh.from_trimesh(m, is_visible=False) for name, m in obj_meshes.items() if name != CONST.DUMMY
        }
        self.hand_meshes = [FMesh.from_trimesh(m, is_visible=False) for m in hand_meshes]
        self.hand_mapping = [get_mapping(m.primitives[0].positions) for m in self.hand_meshes]
        # <<<<<<<<<<<<<<<<<<<<<<<<<

        # >>>>>>> add scene >>>>>>>
        assert cam_intr.shape == (3, 3) and cam_extr.shape == (4, 4), "camera parameter format error"
        self.camera = pyrender.IntrinsicsCamera(cam_intr[0, 0], cam_intr[1, 1], cam_intr[0, 2], cam_intr[1, 2])
        self.scene = FScene(ambient_light=[0.8, 0.8, 0.8], bg_color=[0.5, 0.5, 0.5])
        self.scene.add(self.camera, pose=cam_extr)
        if lights is None:
            logger.warning("renderer does't have lights in scene")
        else:
            for l in lights:
                if isinstance(l, PointLight):
                    light = pyrender.PointLight(color=l.color, intensity=l.intensity)
                elif isinstance(l, DirectionalLight):
                    light = pyrender.DirectionalLight(color=l.color, intensity=l.intensity)
                else:
                    raise NotImplementedError()
                self.scene.add(light, pose=l.pose)
        for k, o_mesh in self.obj_meshes.items():
            self.scene.add(o_mesh, name=k)
        for i, h_mesh in enumerate(self.hand_meshes):
            self.scene.add(h_mesh, name=f"hand_texture_{i}")
        # <<<<<<<<<<<<<<<<<<<<<<<<<

        if backgrounds is None:
            logger.warning("background is not given, use black instead")
        self.backgrounds = [np.asarray(i) for i in backgrounds]

    def __call__(self, obj_name: str, obj_pose: np.ndarray, hand_verts: np.ndarray, motion_blur: int = 0) -> Image:
        hid = np.random.randint(len(self.hand_meshes))
        self.hand_meshes[hid].update_verts(hand_verts[self.hand_mapping[hid]])
        if obj_name != CONST.DUMMY:
            self.scene.show_node(obj_name, pose=obj_pose)
        self.scene.show_node(f"hand_texture_{hid}")
        color, depth = self.r.render(self.scene, flags=pyrender.RenderFlags.NONE)
        if self.backgrounds:
            color = color.copy()

            if motion_blur:
                kernel_motion_blur = get_motion_blur_k(motion_blur)
                color = cv2.filter2D(color, -1, kernel_motion_blur)

            mask = np.stack((depth, depth, depth), axis=-1)
            np.putmask(color, mask == 0, self.get_rand_bg())
        if obj_name != CONST.DUMMY:
            self.scene.hide_node(obj_name)
        self.scene.hide_node(f"hand_texture_{hid}")
        return color[:, :, (2, 1, 0)]

    def get_rand_bg(self):
        bid = np.random.randint(len(self.backgrounds))
        if self.backgrounds[bid].shape[0] - self.height > self.backgrounds[bid].shape[1] - self.width:
            crop_width = np.random.randint(self.width, self.backgrounds[bid].shape[1] + 1)
            crop_height = int(self.height / self.width * crop_width)
        else:
            crop_height = np.random.randint(self.height, self.backgrounds[bid].shape[0] + 1)
            crop_width = int(self.width / self.height * crop_height)
        rand_x = np.random.randint(self.backgrounds[bid].shape[0] - crop_height + 1)
        rand_y = np.random.randint(self.backgrounds[bid].shape[1] - crop_width + 1)
        crop_img = self.backgrounds[bid][rand_x : rand_x + crop_height, rand_y : rand_y + crop_width]
        return cv2.resize(crop_img, dsize=(self.width, self.height))


def load_bg(bgs_path: Union[str, List[str]] = "assets/synth_bg_HO3D"):
    bgs = []
    if type(bgs_path) == str:
        bgs_path_list = [bgs_path]
    else:
        bgs_path_list = bgs_path
    for bgs_path in bgs_path_list:
        for bg_name in sorted(os.listdir(bgs_path)):
            img = open(os.path.join(bgs_path, bg_name)).convert("RGB")
            bgs.append(img)
    return bgs


if __name__ == "__main__":
    from anakin.opt import arg, cfg
    from anakin.artiboost.hand_texture import HTMLHand

    ho_dataset = builder.build_dataset(cfg["DATASET"]["TEST"], preset_cfg=cfg["DATA_PRESET"], opt=arg)

    bgs = []
    bgs_path = "/home/kailin/Lab/hocontact/data/info/synth_bg_HO3D"
    for bg_name in sorted(os.listdir(bgs_path)):
        img = open(os.path.join(bgs_path, bg_name)).convert("RGB")
        bgs.append(img)

    obj_name, meshes, _ = ho_dataset.obj_load_driver()
    obj_meshes = {name: m for name, m in zip(obj_name, meshes)}

    hand_meshes = HTMLHand.get_HTML_mesh()

    cam_intr = np.array([[343.0, 0.0, 128.0], [0.0, 343.0, 128.0], [0.0, 0.0, 1.0]])
    cam_extr = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    lights = [
        PointLight(color=np.array([0.9, 0.9, 0.9]), intensity=5.0, pose=np.eye(4)),
    ]
    renderer = Renderer(
        width=cfg["DATA_PRESET"]["IMAGE_SIZE"][0],
        height=cfg["DATA_PRESET"]["IMAGE_SIZE"][1],
        gpu_id=3,
    )
    renderer.setup(
        cam_intr=cam_intr,
        cam_extr=cam_extr,
        obj_meshes=obj_meshes,
        hand_meshes=hand_meshes,
        backgrounds=bgs,
        lights=lights,
    )
    for i in range(len(ho_dataset)):
        ho_dataset.get_obj_transf_wrt_cam(i)
        ho_dataset.get_hand_verts_3d(i)
        ho_dataset.get_joints_3d(i)
    sst = time()
    for _ in range(1000):
        id = np.random.randint(len(ho_dataset))
        hand_joint_c = np.array(ho_dataset.get_joints_3d(id)[9])
        hand_joint_c[2] = 0
        pose = np.array(ho_dataset.get_obj_transf_wrt_cam(id))
        pose[0][3] -= hand_joint_c[0]
        pose[1][3] -= hand_joint_c[1]
        hand_verts = ho_dataset.get_hand_verts_3d(id) - hand_joint_c
        img = renderer(ho_dataset.get_meta(id)["objname"], obj_pose=pose, hand_verts=hand_verts)
        cv2.imwrite(f"./tmp/render{id}.png", img)
    logger.error(f"{time() - sst}")
