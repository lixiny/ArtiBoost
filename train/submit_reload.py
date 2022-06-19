import os
import random
from time import time

import numpy as np
import torch
from anakin.criterions.criterion import Criterion
from anakin.datasets.hodata import ho_collate
from anakin.metrics.evaluator import Evaluator
from anakin.models.arch import Arch
from anakin.opt import arg, cfg
from anakin.submit import SubmitEpochPass
from anakin.utils import builder
from anakin.utils.logger import logger
from anakin.utils.misc import TrainMode
from anakin.utils.recorder import Recorder


def set_all_seeds(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main_worker(gpu_id, time_f):
    rank = 0  # only one process.
    set_all_seeds(cfg["TRAIN"]["MANUAL_SEED"])

    cfg_name = arg.cfg.split("/")[-1].split(".")[0]
    exp_id = f"submit_{cfg_name}"
    recorder = Recorder(exp_id, cfg, rank=rank, time_f=time_f, eval_only=True)
    logger.info(f"dump args: {arg, cfg['TRAIN']}")

    logger.warning(f"Submit with {arg.submit_dataset} dataset!")
    submit_epoch_pass = SubmitEpochPass.build(arg.submit_dataset, cfg=None)

    test_data = builder.build_dataset(cfg["DATASET"]["TEST"], preset_cfg=cfg["DATA_PRESET"])
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=arg.batch_size,
                                              shuffle=False,
                                              num_workers=int(arg.workers),
                                              drop_last=False,
                                              collate_fn=ho_collate)

    model_list = builder.build_arch_model_list(cfg["ARCH"], preset_cfg=cfg["DATA_PRESET"])
    model = Arch(cfg, model_list=model_list)
    model = torch.nn.DataParallel(model).to(arg.device)

    loss_list = builder.build_criterion_loss_list(cfg["CRITERION"], cfg["DATA_PRESET"], LAMBDAS=cfg["LAMBDAS"])
    criterion = Criterion(cfg, loss_list=loss_list)
    metrics_list = builder.build_evaluator_metric_list(cfg["EVALUATOR"], cfg["DATA_PRESET"], arg=arg)
    evaluator = Evaluator(cfg, metrics_list=metrics_list)

    dump_fname = cfg_name
    dump_fname += "_trueroot" if arg.true_root else ""
    dump_fname += "_truebonescale" if arg.true_bone_scale else ""
    dump_fname += "_pseudoroot" if arg.use_pseudo_hand_root else ""
    dump_fname += f"_{arg.postprocess_fit_mesh_ik}" if arg.postprocess_fit_mesh else ""
    dump_fname += "_fitjoints" if arg.postprocess_fit_mesh and arg.postprocess_fit_mesh_use_fitted_joints else ""
    dump_fname += "_SUBMIT"
    dump_fname += ".json" if not arg.resume_epoch else f"_epoch{arg.resume_epoch}.json"
    dump_path = os.path.join(recorder.dump_path, dump_fname)

    draw_path = os.path.join(recorder.dump_path, "rendered_image")

    with torch.no_grad():
        model.eval()
        submit_epoch_pass(epoch_idx=0,
                          data_loader=test_loader,
                          arch_model=model,
                          criterion=criterion,
                          evaluator=evaluator,
                          rank=rank,
                          dump_path=dump_path,
                          draw_path=draw_path)

    recorder.record_evaluator(evaluator, 0, TrainMode.TEST)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


def main():
    exp_time = time()
    logger.info("====> Use Data Parallel <====")
    main_worker(arg.gpus[0], exp_time)


if __name__ == "__main__":
    main()
