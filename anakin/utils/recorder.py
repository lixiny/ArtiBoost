import os
import pickle
import random
import sys
import time
from pprint import pformat
from typing import Dict, Optional, TypeVar

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import yaml
from PIL.Image import Image
from git import Repo
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from anakin.metrics.evaluator import Evaluator
from anakin.models.arch import Arch
from anakin.utils.io_utils import (load_arch, load_train_param, load_random_state, save_states)
from anakin.utils.logger import logger
from anakin.utils.misc import TrainMode, singleton, RandomState

T = TypeVar("T", bound="Recorder")


@singleton
class Recorder:

    def __init__(
        self: T,
        exp_id: str,
        cfg: Dict,
        root_path: str = "./exp",
        rank: Optional[int] = None,
        time_f: Optional[float] = None,
        eval_only: bool = False,
    ):
        if not eval_only:
            assert exp_id == "default" or self.get_git_commit(), "MUST commit before the experiment!"
        
        self.timestamp = time.strftime("%Y_%m%d_%H%M_%S", time.localtime(time_f if time_f else time.time()))
        self.exp_id = exp_id
        self.cfg = cfg
        self.dump_path = os.path.join(root_path, f"{exp_id}_{self.timestamp}")
        self.rank = rank
        self.eval_only = eval_only
        self._record_init_info()

    def _record_init_info(self: T):
        if not self.rank:
            if not os.path.exists(self.dump_path):
                os.makedirs(self.dump_path)
            assert logger.filehandler is None, "log file path has been set"
            logger.set_log_file(path=self.dump_path, name=f"{self.exp_id}_{self.timestamp}")
            logger.info(f"run command: {' '.join(sys.argv)}")
            if not self.eval_only and self.exp_id != "default":
                logger.info(f"git commit: {self.get_git_commit()}")
            with open(os.path.join(self.dump_path, "dump_cfg.yaml"), "w") as f:
                yaml.dump(self.cfg, f, Dumper=yaml.Dumper, sort_keys=False)
            logger.info(f"dump cfg file to {os.path.join(self.dump_path, 'dump_cfg.yaml')}")
        else:
            logger.remove_log_stream()
            logger.disabled = True

    def record_checkpoints(self: T, model: Arch, optimizer: Optimizer, scheduler: _LRScheduler, epoch: int,
                           snapshot: int):
        if self.rank:
            return
        checkpoints_path = os.path.join(self.dump_path, "checkpoints")
        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)

        # construct RandomState tuple
        random_state = RandomState(
            torch_rng_state=torch.get_rng_state(),
            torch_cuda_rng_state=torch.cuda.get_rng_state(),
            torch_cuda_rng_state_all=torch.cuda.get_rng_state_all(),
            numpy_rng_state=np.random.get_state(),
            random_rng_state=random.getstate(),
        )

        save_states(
            {
                "epoch": epoch + 1,
                "model_list": model.module.model_list if hasattr(model, "module") else model.model_list,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "random_state": random_state,
            },
            is_best=False,
            checkpoint=checkpoints_path,
            snapshot=snapshot,
        )
        # logger.info(f"record checkpoints to {checkpoints_path}")

    def resume_checkpoints(
        self: T,
        model: Arch,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        resume_path: str,
        resume_epoch: Optional[int] = None,
    ):
        """

        Args:
            model:
            optimizer:
            scheduler:
            resume_path:

        Returns:

        """
        resume_path = os.path.join(resume_path, "checkpoints",
                                   f"checkpoint_{resume_epoch}" if resume_epoch else "checkpoint")
        epoch = load_train_param(optimizer, scheduler, os.path.join(resume_path, "train_param.pth.tar"))
        load_random_state(os.path.join(resume_path, "random_state.pkl"))
        load_arch(model, resume_path, map_location=f"cuda:{self.rank}" if self.rank is not None else "cuda")
        return epoch

    def record_evaluator(self: T, evaluator: Evaluator, epoch: int, train_mode: TrainMode):
        if self.rank:
            return

        file_perfixes = {
            TrainMode.TRAIN: "train",
            TrainMode.VAL: "val",
            TrainMode.TEST: "test",
        }
        prefix = file_perfixes[train_mode]
        evaluations_path = os.path.join(self.dump_path, "evaluations")
        if not os.path.exists(evaluations_path):
            os.makedirs(evaluations_path)

        images: Dict[str, Image] = evaluator.dump_images()
        if len(images):
            images_path = os.path.join(evaluations_path, "images", f"{prefix}_epoch_{epoch}_images")
            if not os.path.exists(images_path):
                os.makedirs(images_path)
            for k, img in images.items():
                cv2.imwrite(os.path.join(images_path, f"{k}.png"), img)

        with open(os.path.join(evaluations_path, f"{prefix}_eval.txt"), "a") as f:
            f.write(f"Epoch {epoch} evaluator msg:\n {pformat(evaluator.get_measures_all_striped())}\n\n")

    def record_arch_graph(self: T, model: Arch):
        if self.rank:
            return
        G = model.to_graph()
        nx.draw_kamada_kawai(G, with_labels=True, node_size=1000, width=3, node_color="cyan")
        plt.savefig(os.path.join(self.dump_path, "arch.png"))
        logger.info(f"dump arch image to {os.path.join(self.dump_path, 'arch.png')}")

    @staticmethod
    def get_git_commit() -> Optional[str]:
        # get current git report
        repo = Repo(".")

        modified_files = [item.a_path for item in repo.index.diff(None)]
        staged_files = [item.a_path for item in repo.index.diff("HEAD")]
        untracked_files = repo.untracked_files

        if len(modified_files):
            logger.error(f"modified_files: {' '.join(modified_files)}")
        if len(staged_files):
            logger.error(f"staged_files: {' '.join(staged_files)}")
        if len(untracked_files):
            logger.error(f"untracked_files: {' '.join(untracked_files)}")

        return (repo.head.commit.hexsha
                if not (len(modified_files) or len(staged_files) or len(untracked_files)) else None)

    def record_artiboost_loader(self: T, artiboost_loader, epoch: int):
        self.record_sample_weight(artiboost_loader.sample_weight_map, epoch)
        self.record_sample_occurence(artiboost_loader.occurence_map, epoch)
        self.record_shutdown(artiboost_loader)

    def record_sample_weight(self, weight_map: torch.Tensor, epoch: int, is_train: bool = True):
        sample_weight_path = os.path.join(self.dump_path, "artiboost", "sample_weight")
        if not os.path.exists(sample_weight_path):
            os.makedirs(sample_weight_path)
        weight_map_np = np.array(weight_map)
        prefix = "train" if is_train else "val"
        with open(os.path.join(sample_weight_path, f"{epoch:0>3}_{prefix}.pkl"), "wb") as f:
            pickle.dump(weight_map_np, f)

    def record_sample_occurence(self, occurence_map: torch.Tensor, epoch: int, is_train: bool = True):
        occurence_map_path = os.path.join(self.dump_path, "artiboost", "occurence_map")
        if not os.path.exists(occurence_map_path):
            os.makedirs(occurence_map_path)
        occurence_map_np = np.array(occurence_map)
        with open(os.path.join(occurence_map_path, f"{epoch:0>3}.pkl"), "wb") as f:
            pickle.dump(occurence_map_np, f)

    def record_shutdown(self, data_generation_manager):
        if not data_generation_manager.use_synth:
            with open(os.path.join(self.dump_path, "artiboost", "shutdown"), "w") as f:
                pass

    def resume_artiboost_loader(self, artiboost_loader, resume_epoch: int, resume_path: str):
        epoch = resume_epoch - 1

        # need to overwrite sample weight, occurence
        sample_weight_path = os.path.join(resume_path, "artiboost", "sample_weight")
        prefix = "train"
        with open(os.path.join(sample_weight_path, f"{epoch:0>3}_{prefix}.pkl"), "rb") as f:
            weight_map_np = pickle.load(f)
        weight_map = torch.from_numpy(weight_map_np)

        occurence_map_path = os.path.join(resume_path, "artiboost", "occurence_map")
        with open(os.path.join(occurence_map_path, f"{epoch:0>3}.pkl"), "rb") as f:
            occurence_map_np = pickle.load(f)
        occurence_map = torch.from_numpy(occurence_map_np)

        # assign
        artiboost_loader.sample_weight_map[:] = weight_map
        artiboost_loader.occurence_map[:] = occurence_map

        # check shutwodn
        if os.path.exists(os.path.join(resume_path, "artiboost", "shutdown")):
            artiboost_loader.synth_shutdown()
        logger.info("Resume artiboost loader finished.")