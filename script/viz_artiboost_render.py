import os
import random
from time import time

import numpy as np
import torch
from anakin.artiboost import ArtiBoostLoader
from anakin.datasets.hodata import ho_collate
from anakin.opt import arg, cfg
from anakin.opt_extra import data_generation_manager_parse
from anakin.utils import builder
from anakin.utils.etqdm import etqdm
from anakin.utils.logger import logger
from anakin.utils.misc import CONST
from anakin.viztools.draw import save_a_image_with_joints_corners
from anakin.viztools.opendr_renderer import OpenDRRenderer
from termcolor import colored


def setup_seed(seed, conv_repeatable=True):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if conv_repeatable:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        logger.warning("Exp result NOT repeatable!")


def main_worker(time_f: float):
    manager_dict_plus = {
        "VAL_FREQ": cfg["TRAIN"]["EVAL_FREQ"],
        "VAL_START_EPOCH": cfg["TRAIN"]["VAL_START_EPOCH"],
        "EPOCH": cfg["TRAIN"]["EPOCH"],
    }
    cfg["MANAGER"].update(manager_dict_plus)

    train_data = builder.build_dataset(cfg["DATASET"]["TRAIN"], preset_cfg=cfg["DATA_PRESET"])
    arg_extra = data_generation_manager_parse()
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=arg.batch_size,
        shuffle=False,
        num_workers=int(arg.workers),
        pin_memory=True,
        drop_last=arg.drop_last,
        collate_fn=ho_collate,
    )

    artiboost_loader = ArtiBoostLoader(
        train_data,
        arg=arg,
        arg_extra=arg_extra,
        cfg=cfg["MANAGER"],
        cfg_dataset=cfg["DATASET"],
        cfg_preset=cfg["DATA_PRESET"],
        time_f=time_f,
        batch_size=arg.batch_size,
        shuffle=True,
        num_workers=int(arg.workers),
        pin_memory=True,
        drop_last=arg.drop_last,
        collate_fn=ho_collate,
        random_seed=cfg["TRAIN"]["MANUAL_SEED"],
    )

    artiboost_loader.prepare()

    viz_dir = "./tmp/viz_artiboost_render"
    os.makedirs(viz_dir, exist_ok=True)

    # * epoch pass >>>
    artiboost_loader = etqdm(artiboost_loader)
    for batch_idx, batch in enumerate(artiboost_loader):
        image = batch["image"]  # (B, C, H, W)
        image = image + 0.5
        image = image.permute(0, 2, 3, 1).cpu().numpy()
        batch_size = image.shape[0]

        batch_is_synth = batch["is_synth"].cpu().numpy()
        batch_cam_intr = batch["cam_intr"].cpu().numpy()

        batch_corners_can = batch["corners_can"].cpu().numpy()
        batch_sample_idx = batch["sample_idx"].cpu().numpy()
        root_joint = batch["root_joint"]
        batch_joints_3d = batch["joints_3d"] + root_joint.unsqueeze(1)
        batch_corners_3d = batch["corners_3d"] + root_joint.unsqueeze(1)
        batch_joints_3d = batch_joints_3d.cpu().numpy()
        batch_corners_3d = batch_corners_3d.cpu().numpy()
        batch_obj_transf = batch["obj_transf"].cpu().numpy()
        batch_obj_idx = batch["obj_idx"].cpu().numpy()

        for bid in range(batch_size):
            obj_id = int(batch_obj_idx[bid])
            obj_verts_can, _, _ = train_data.get_obj_verts_can_by_obj_id(obj_id)
            obj_face = train_data.get_obj_faces_by_obj_id(obj_id)
            corners_can = batch_corners_can[bid]  # (8, 3)
            obj_transf = batch_obj_transf[bid]  # (4, 4)
            obj_rotmat, obj_tsl = obj_transf[:3, :3], obj_transf[:3, 3]
            obj_verts_3d = (obj_rotmat @ obj_verts_can.T).T + obj_tsl
            corners_3d = (obj_rotmat @ corners_can.T).T + obj_tsl

            cam_intr = batch_cam_intr[bid]  # (3, 3)
            joints_3d = batch_joints_3d[bid]  # (21, 3)
            joints_2d = (cam_intr @ joints_3d.T).T
            joints_2d = joints_2d[:, 0:2] / joints_2d[:, 2:3]
            corners_2d = (cam_intr @ corners_3d.T).T
            corners_2d = corners_2d[:, 0:2] / corners_2d[:, 2:3]

            img_name = f"{'synt' if batch_is_synth[bid] else 'real'}"
            img_name += f"_idx{batch_sample_idx[bid]}.png"
            img_name += f"_{CONST.YCB_IDX2CLASSES[obj_id]}.png"

            if batch_is_synth[bid]:
                save_a_image_with_joints_corners(
                    image[bid],
                    cam_intr,
                    joints_2d,
                    corners_2d,
                    joints_3d,
                    corners_3d,
                    os.path.join(viz_dir, img_name),
                    obj_verts_xyz=obj_verts_3d,
                    obj_faces=obj_face,
                )


def main():
    exp_time = time()
    setup_seed(cfg["TRAIN"]["MANUAL_SEED"], cfg["TRAIN"].get("CONV_REPEATABLE", True))
    logger.info("====> Use Data Parallel <====")
    main_worker(exp_time)  # need to pass in renderer process group info


if __name__ == "__main__":
    main()
