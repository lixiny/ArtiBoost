import atexit
import random
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import trimesh
from anakin.utils.logger import logger
from anakin.utils.renderer import DirectionalLight, PointLight, Renderer
from PIL import Image
from torch.multiprocessing import Queue


def render_worker(
    exit_event,
    gpu_id: int,
    incoming_queue: Queue,
    output_queue_list: Sequence[Queue],
    render_width: int,
    render_height: int,
    cam_intr: np.ndarray,
    cam_extr: np.ndarray,
    obj_meshes: Dict[str, trimesh.Trimesh],
    hand_meshes: List[trimesh.Trimesh],
    bgs: Optional[List[Image.Image]] = None,
    lights: Optional[List[Union[PointLight, DirectionalLight]]] = None,
    proc_id=0,
    random_seed=1,
):
    np.random.seed(random_seed + proc_id)
    random.seed(random_seed + proc_id)

    renderer = Renderer(
        width=render_width,
        height=render_height,
        gpu_id=gpu_id,
    )
    renderer.setup(
        cam_intr=cam_intr,
        cam_extr=cam_extr,
        obj_meshes=obj_meshes,
        hand_meshes=hand_meshes,
        backgrounds=bgs,
        lights=lights,
    )
    while not exit_event.is_set():
        try:
            msg = incoming_queue.get(block=True, timeout=2)
        except:
            continue

        # fetch one
        obj_name = deepcopy(msg["objname"])  # get a local clone
        pose = deepcopy(msg["pose"])
        hand_verts = deepcopy(msg["hand_verts"])
        xid = msg["id"]
        del msg  # make sure to release shared memory

        img = renderer(obj_name=obj_name, obj_pose=pose, hand_verts=hand_verts)
        output_queue_list[xid].put(img)


class RendererProvider:

    def __init__(
        self,
        num_workers: int,
        gpu_render_id: Sequence[int],
        render_size: List[int],
        cam_intr: np.ndarray,
        cam_extr: np.ndarray,
        obj_meshes: Dict[str, trimesh.Trimesh],
        hand_meshes: List[trimesh.Trimesh],
        bgs: Optional[List[Image.Image]] = None,
        lights: Optional[List[Union[PointLight, DirectionalLight]]] = None,
        cfg_renderer=None,
        cfg_datapreset=None,
        arg_extra=None,
        random_seed=1,
    ):
        logger.info("render_provider: initializing...")
        logger.info(f"render_provider: using gpu {gpu_render_id}")
        self.gpu_render_id = list(gpu_render_id)
        self.gpu_render_used = len(gpu_render_id)

        self.exit_event = torch.multiprocessing.Event()
        self.message_queue: torch.multiprocessing.Queue = torch.multiprocessing.Queue()
        self.image_queue_list: List[torch.multiprocessing.Queue] = []
        for _ in range(num_workers):
            self.image_queue_list.append(torch.multiprocessing.Queue())
        self.server_proc_list = []
        for proc_id, gpu_id in zip(range(self.gpu_render_used), gpu_render_id):
            self.server_proc_list.append(
                torch.multiprocessing.Process(
                    target=render_worker,
                    kwargs={
                        "exit_event": self.exit_event,
                        "gpu_id": gpu_id,
                        "incoming_queue": self.message_queue,
                        "output_queue_list": self.image_queue_list,
                        "render_width": render_size[0],
                        "render_height": render_size[1],
                        "cam_intr": cam_intr,
                        "cam_extr": cam_extr,
                        "obj_meshes": obj_meshes,
                        "hand_meshes": hand_meshes,
                        "bgs": bgs,
                        "lights": lights,
                        "proc_id": proc_id,
                        "random_seed": random_seed,
                    },
                ))

        self.running = False

        # register at exit
        def gracefully_exit_fn():
            if self.running:
                self.exit_event.set()
                for proc_id in range(len(self.server_proc_list)):
                    self.server_proc_list[proc_id].join()

        self.gracefully_exit = gracefully_exit_fn

        atexit.register(self.gracefully_exit)

    def __del__(self):
        self.gracefully_exit()

    def begin(self):
        if not self.running:
            self.running = True
            for proc_id in range(self.gpu_render_used):
                self.server_proc_list[proc_id].start()
            logger.info("render_provider: start!")
        else:
            logger.warn("render_provider: render server alreay running!")

    def end(self):
        if self.running:
            self.gracefully_exit()
            logger.info("render_provider: end!")
        else:
            logger.warn("render_provider: render server not running!")

    def get_process_list(self):
        return self.exit_event

    def get_message_queue(self):
        return self.message_queue

    def get_image_queue_list(self):
        return self.image_queue_list
