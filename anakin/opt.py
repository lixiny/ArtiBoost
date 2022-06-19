from anakin.utils.misc import update_config
import argparse
import logging
from .utils.logger import logger
import os
import torch

parser = argparse.ArgumentParser(description="ANAKIN SKYWALKER")

parser.add_argument("--vis_toc", type=float, default=5)
"----------------------------- Experiment options -----------------------------"
parser.add_argument("--cfg", help="experiment configure file name", type=str, default=None)
parser.add_argument("--exp_id", default="default", type=str, help="Experiment ID")

parser.add_argument("--resume", help="resume training from exp", type=str, default=None)
parser.add_argument("--workers", help="worker number from data loader", type=int, default=20)
parser.add_argument("--batch_size",
                    help="batch size of exp, will replace bs in cfg file if is given",
                    type=int,
                    default=None)

parser.add_argument("--evaluate", help="evaluate the network (ignore training)", action="store_true")
"----------------------------- General options -----------------------------"
parser.add_argument("--gpu_id", type=str, default=None, help="override enviroment var CUDA_VISIBLE_DEVICES")
parser.add_argument("--snapshot", default=50, type=int, help="How often to take a snapshot of the model (0 = never)")
parser.add_argument("--test_freq",
                    type=int,
                    default=5,
                    help="How often to test, 1 for always -1 for never, caution use this option")
parser.add_argument("--gpu_render_port", type=str, default="34567")
"-------------------------------------dataset submit options-------------------------------------"
parser.add_argument("--resume_epoch", help="resume from the given epoch", type=int, default=0)
parser.add_argument("--submit_dataset", type=str, default="hodata")
parser.add_argument("--filter_unseen_obj_idxs", type=int, nargs="+", default=[])
parser.add_argument("--true_root", action="store_true", help="use GT hand root")
parser.add_argument("--true_bone_scale", action="store_true", help="use GT bone length")
parser.add_argument("--submit_dump", action="store_true", help="whether to save json for benchmark")
parser.add_argument("--postprocess_fit_mesh", action="store_true", help="postprocess fit mesh")
parser.add_argument("--postprocess_fit_mesh_ik",
                    type=str,
                    choices=["iknet", "iksolver"],
                    default="iknet",
                    help="process fit mesh ik method")
parser.add_argument("--postprocess_fit_mesh_use_fitted_joints",
                    action="store_true",
                    help="postprocess fit mesh, whether to use fitted joints or model predicted joints")
parser.add_argument("--use_pseudo_hand_root", action="store_true", help="direactly use pseudo hand root for prediction")
parser.add_argument("--postprocess_draw", action="store_true", help="save debug image in postprocess submission")
parser.add_argument("--postprocess_draw_path",
                    type=str,
                    help="save debug image in postprocess submission, specify path",
                    default=None)

arg, custom_arg_string = parser.parse_known_args()
if arg.resume:
    if arg.cfg:
        logger.warning(f"config will be rewritten by {os.path.join(arg.resume, 'dump_cfg.yaml')}")
    cfg = update_config(os.path.join(arg.resume, "dump_cfg.yaml"))
else:
    cfg = update_config(arg.cfg) if arg.cfg else dict()
    cfg["FILE_NAME"] = arg.cfg

if arg.gpu_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu_id

arg.device = "cuda" if torch.cuda.is_available() else "cpu"
if arg.batch_size:
    cfg["TRAIN"]["BATCH_SIZE"] = arg.batch_size
else:
    arg.batch_size = cfg["TRAIN"]["BATCH_SIZE"]
arg.drop_last = cfg["TRAIN"].get("DROP_LAST", True)

arg.gpus = [i for i in range(torch.cuda.device_count())]
