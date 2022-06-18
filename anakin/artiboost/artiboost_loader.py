import atexit
import hashlib
import itertools
import json
import os
import pickle
import random
import shutil
import signal
import sys
import time
from copy import deepcopy
from functools import partial
from time import localtime, strftime
from typing import Mapping

import numpy as np
import torch
from anakin.artiboost.cache_recorder import CacheRecorder
from anakin.artiboost.grasp_engine import GraspEngine
from anakin.artiboost.hand_texture import HTMLHand
from anakin.artiboost.mixed_dataset import MixedDataset
from anakin.artiboost.object_engine import ObjEngine
from anakin.artiboost.ovg_set import OVGSet
from anakin.artiboost.preprocessor import PreProcessorPoseGenerator
from anakin.artiboost.refiner import Refiner
from anakin.artiboost.render_infra import RendererProvider
from anakin.artiboost.rendered_dataset import RenderedDataset
from anakin.artiboost.scrambler import Scrambler
from anakin.artiboost.view_engine import ViewEngine
from anakin.datasets.hodata import HOdata
from anakin.metrics.val_metric import ValMetricAR2, ValMetricMean3DEPE2
from anakin.utils.etqdm import etqdm
from anakin.utils.logger import logger
from anakin.utils.misc import CONST
from anakin.utils.renderer import PointLight, load_bg
from anakin.utils.transform import aa_to_rotmat


def batch_to_device(batch, device):
    for k in batch.keys():
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    return batch


class ArtiBoostLoader:

    def __init__(
        self,
        real_train_set: HOdata,
        arg,
        arg_extra,
        cfg: Mapping,  # cfg["MANAGER"]
        cfg_dataset: Mapping,  # cfg["DATASET"]
        cfg_preset: Mapping,  # cfg["DATA_PRESET"]
        time_f: float,  # timestamp
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        collate_fn=None,
        random_seed=1,
        **kwargs,
    ):
        self.real_train_set = real_train_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collat_fn = collate_fn

        self.synth_factor = cfg["SYNTH_FACTOR"]
        self.real_len = len(self.real_train_set)
        self.synth_len = int(self.synth_factor * self.real_len)
        self.epoch_len = self.real_len + self.synth_len
        self.val_start_epoch = cfg["VAL_START_EPOCH"]
        self.val_freq = cfg["VAL_FREQ"]
        self.n_epochs = cfg["EPOCH"]
        self.filter_back_flag = cfg["FILTER"].get("BACK", True) if "FILTER" in cfg else True

        ## ovg config
        self.ovg_config_len_train = self.synth_len
        self.ovg_config_len_val = cfg.get("VAL_LEN", self.ovg_config_len_train)
        self.ovg_batch_size = arg_extra.ovg_batch_size
        self.ovg_num_workers = arg_extra.ovg_num_workers

        # important flag! controls whether synth will happen
        self.use_synth = self.synth_len > 0
        logger.warning(f"data generation manager: use_synth <- {self.use_synth}")

        timestamp = strftime("%Y_%m%d_%H%M_%S", localtime(time_f))
        self.temp_parent_dir = os.path.join(arg_extra.synth_root, timestamp)
        self.synth_root = os.path.join(arg_extra.synth_root, timestamp, "intermediate")
        # blacklisted_root, manage blacklisted ids
        self.blacklisted_root = os.path.join(arg_extra.synth_root, timestamp, "blacklisted")
        os.makedirs(self.blacklisted_root)
        atexit.register(self.gracefully_quit, None, None)
        # handle ctrl-C and kill -9
        signal.signal(signal.SIGTERM, self.gracefully_quit)
        signal.signal(signal.SIGINT, self.gracefully_quit)

        logger.info("> build obj engine")
        self.use_cv_space = False
        query_obj = cfg["OBJ_ENGINE"]["OBJ"]
        query_obj_dataset = cfg["OBJ_ENGINE"]["OBJ_ORIGIN_DATASET"]
        self.obj_engine = ObjEngine.build(query_obj_dataset, query_obj)
        self.obj_names = self.obj_engine.obj_names
        self.obj_trimeshes_mapping = self.obj_engine.obj_trimeshes_mapping

        logger.info("> build grasp engine")
        query_grasp_dataset = cfg["GRASP_ENGINE"]["GRASP_ORIGIN_DATASET"]
        self.grasp_engine: GraspEngine = GraspEngine.build(query_grasp_dataset, self.obj_names)

        logger.info("> build view engine")
        self.view_engine: ViewEngine = ViewEngine(cfg["VIEW_ENGINE"])

        sample_n_obj = len(self.obj_names)
        sample_n_persp = cfg["VIEW_ENGINE"]["PERSP_U_BINS"] * cfg["VIEW_ENGINE"]["PERSP_THETA_BINS"]
        sample_n_grasp = cfg["GRASP_ENGINE"]["GRASP_NUM"]

        logger.info("> initialize sample weight map and blacklist map")
        self.sample_weight_map = torch.ones((sample_n_obj, sample_n_persp, sample_n_grasp), dtype=torch.float32)
        self.occurence_map = torch.zeros((sample_n_obj, sample_n_persp, sample_n_grasp), dtype=torch.bool)
        self.blacklist_map = self._construct_blacklist_map(sample_n_obj, sample_n_persp, sample_n_grasp,
                                                           self.filter_back_flag)
        logger.info(f"blacklisted ratio: {torch.sum(self.blacklist_map)} / {self.blacklist_map.numel()}")
        self.sample_weight_map[self.blacklist_map] = 0.0

        # * Object-View-Grasp Triplet Set (need a data loader for streaming data to pose generator) >>>>>>
        logger.info("> build OVG dataset")
        self.OVG_dataset = OVGSet(obj_engine=self.obj_engine,
                                  grasp_engine=self.grasp_engine,
                                  view_engine=self.view_engine,
                                  config_len_train=self.ovg_config_len_train,
                                  config_len_val=self.ovg_config_len_val,
                                  n_grasp=sample_n_grasp,
                                  blacklist_map=self.blacklist_map)

        if self.use_synth:
            logger.info("> build ovg loader for sampling batched triplets")

            def _ovg_loader_init_fn(worker_id):
                seed = int(torch.initial_seed()) % CONST.INT_MAX
                np.random.seed(seed)
                random.seed(seed)

            self.OVG_loader = torch.utils.data.DataLoader(
                self.OVG_dataset,
                batch_size=self.ovg_batch_size,
                shuffle=True,
                num_workers=self.ovg_num_workers,
                drop_last=False,
                worker_init_fn=_ovg_loader_init_fn,
            )
        else:
            self.OVG_loader = None

        logger.info("> build scrambler for pose disturbance")
        self.scrambler = Scrambler.build(cfg["SCRAMBLER"]["TYPE"], cfg=cfg["SCRAMBLER"])

        logger.info("> build refiner for pose refinement")
        self.refiner = Refiner.build(cfg["REFINER"]["TYPE"], cfg=cfg["REFINER"])
        self.refiner.setup(self.obj_trimeshes_mapping)

        logger.info("> clone grasp_engine's mano_layer")
        self.ge_mano_layer = deepcopy(self.grasp_engine.mano_layer)

        logger.info("> store handle to refiner's mano layer")
        self.rf_mano_layer = self.refiner.refine_net.mano_layer

        # * pose_generator >>>>>>
        self.pose_generator = PreProcessorPoseGenerator(
            refiner=self.refiner,
            scrambler=self.scrambler,
            ge_mano_layer=self.ge_mano_layer,
            rf_mano_layer=self.rf_mano_layer,
        )
        self.device = arg.device
        self.pose_generator = self.pose_generator.to(self.device)

        # * cache_recorder, manage synth files >>>>>>
        self.cache_recorder = CacheRecorder(self.synth_root)

        # * render provider, muti-process multi-GPU rendering pipeline >>>>>>
        K = cfg["RENDERER"]["CAM_PARAM"]
        render_intr = np.array([[K["FX"], 0.0, K["CX"]], [0.0, K["FY"], K["CY"]], [0.0, 0.0, 1.0]], dtype=np.float32)
        render_size = cfg["RENDERER"]["RENDER_SIZE"]  # [width, height]
        render_hand_meshes = HTMLHand.get_HTML_mesh()
        render_bgs = load_bg(cfg["RENDERER"]["BGS_PATH"])
        render_lights = [PointLight(color=np.array([0.9, 0.9, 0.9]), intensity=5.0, pose=np.eye(4))]
        render_gpu_ids = [int(el) for el in arg_extra.gpu_render_id.split(",")]
        self.render_provider = RendererProvider(num_workers=arg.workers,
                                                gpu_render_id=render_gpu_ids,
                                                render_size=render_size,
                                                cam_intr=render_intr,
                                                cam_extr=CONST.PYRENDER_EXTRINSIC,
                                                obj_meshes=self.obj_trimeshes_mapping,
                                                hand_meshes=render_hand_meshes,
                                                bgs=render_bgs,
                                                lights=render_lights,
                                                cfg_renderer=cfg["RENDERER"],
                                                cfg_datapreset=cfg_preset,
                                                arg_extra=arg_extra,
                                                random_seed=random_seed)
        self.render_provider.begin()
        self.message_queue = self.render_provider.get_message_queue()
        self.image_queue_list = self.render_provider.get_image_queue_list()

        # * create rendered_dataset, to manage the rendered image and its corresponding groundtruth >>>>>>
        self.rendered_dataset = RenderedDataset(
            self.synth_root,
            obj_meshes=self.obj_engine.obj_meshes,
            obj_corners=self.obj_engine.obj_corners_can,
            cam_intr=render_intr,
            cfg_dataset=cfg_dataset["TRAIN"],
            cfg_preset=cfg_preset,
            crop_image=cfg.get("CROP_IMAGE", None),
        )

        def render_synth_train_worker_init_fn(_):
            seed = int(torch.initial_seed()) % CONST.INT_MAX
            np.random.seed(seed)
            random.seed(seed)
            worker_info = torch.utils.data.get_worker_info()
            assert worker_info is not None
            synth_dataset = worker_info.dataset.synth_set
            worker_id = worker_info.id
            synth_dataset.out_queue = self.message_queue
            synth_dataset.in_queue = self.image_queue_list[worker_id]
            synth_dataset.id = worker_id

        self.render_synth_train_worker_init_fn = render_synth_train_worker_init_fn

        self.final_train_set = MixedDataset(self.real_train_set, self.rendered_dataset)
        self.final_train_loader = None

        if "WEIGHT_UPDATE" in cfg:
            self.sample_weight_lower_bound = cfg["WEIGHT_UPDATE"].get("LOWER", 0.1)
            self.sample_weight_upper_bound = cfg["WEIGHT_UPDATE"].get("UPPER", 10.0)
        else:
            self.sample_weight_lower_bound = 0.1
            self.sample_weight_upper_bound = 10.0
        if "DIST_THRESHOLD" in cfg:
            self.dist_lower_threshold = cfg["DIST_THRESHOLD"].get("LOWER", 8.0)
            self.dist_upper_threshold = cfg["DIST_THRESHOLD"].get("UPPER", 16.0)
        else:
            self.dist_lower_threshold = 8.0
            self.dist_upper_threshold = 16.0
        logger.info(f"sample weight lower bound: {self.sample_weight_lower_bound},"
                    f"upper bound: {self.sample_weight_upper_bound}")
        logger.info(f"dist lower threshold: {self.dist_lower_threshold},"
                    f"dist upper threshold: {self.dist_upper_threshold}")

        self.update_method_mapping = {
            "method_1": ArtiBoostLoader.update_method_1,
            "method_2": ArtiBoostLoader.update_method_2,
            "method_3": ArtiBoostLoader.update_method_3,
            "method_4": ArtiBoostLoader.update_method_4,
        }
        self.update_method_key = cfg.get("UPDATE_METHOD", "method_1")

        logger.info("sample weight update method: {}".format(self.update_method_key))

    def get_feed_stream(self):
        return self.final_feed_loader

    def train(self):
        self.final_feed_loader = self.final_train_loader

    def __iter__(self):
        return iter(self.final_feed_loader)

    def __len__(self):
        return len(self.final_feed_loader)

    def prepare(self):
        """Called before training epoch start.
        1. Reset the ArtiBoost validate evaluator, this will clear all the average meters from previous epochs. 
        2. Generate the cache for online rendering. This process consist of three steps, including  weigh-guided poses 
            sampling, pose perturbation, and pose refinement.  ArtiBoost will hold an OVG dataloader to get batches of 
            triplets iD and process the cooresponding poses in a nn.Moudule: PoseGenerator. 
            After each batch of sampled  poses are processed, a cache recorder will dump the generated pose 
            (later are used for rendering) to the synth_root. 
        3. Switch ArtiBoost to training mode. This op will switch the final data loader to feed data.
        """
        self.generate_render_cache(is_train=True)
        self.train()

    def step_eval(self, epoch_idx, evaluator):
        # val_start: bool = epoch_idx + 1 >= self.val_start_epoch
        # at_val_freq: bool = epoch_idx % self.val_freq == self.val_freq - 1
        # if self.use_synth and val_start and at_val_freq:
        eval_res = self.get_evaluator_result(evaluator)
        if eval_res is not None:
            self.sample_reweight(eval_res, epoch_idx)
            logger.info(f"ArtiBoost finishes mining and update after epoch {epoch_idx}")

    def get_evaluator_result(self, evaluator):
        eval_res = []
        for metric in evaluator.metrics_list:
            if isinstance(metric, ValMetricMean3DEPE2):
                eval_res.append(metric.get_measures_averaged())
            elif isinstance(metric, ValMetricAR2):
                eval_res.append(metric.get_measures_averaged())
            else:
                continue

        if len(eval_res) == 0:
            logger.error("No validation metric have been found")
            raise ValueError()

        if not all(set(ev.keys()) == set(eval_res[0].keys()) for ev in eval_res):
            logger.error("some ccv space idx lost!")
            raise ValueError()

        res = {}
        for id in eval_res[0].keys():
            sum_val = 0.0
            for ev in eval_res:
                sum_val += ev[id]
            res[id] = sum_val / len(eval_res)

        return res

    def sample_reweight(self, eval_res, epoch_idx):
        assert eval_res is not None
        update_res = self.update_method_mapping[self.update_method_key](
            self.sample_weight_map,
            eval_res,
            self.sample_weight_lower_bound,
            self.sample_weight_upper_bound,
            dist_lower_threshold=self.dist_lower_threshold,
            dist_upper_threshold=self.dist_upper_threshold,
            epoch_idx=epoch_idx,
            n_epochs=self.n_epochs,
        )
        self.sample_weight_map = update_res["sample_weight_map"]

    def synth_shutdown(self):
        # * remove synth dataset
        self.final_train_set.remove_synth()
        self.use_synth = False
        self.cache_recorder.clear()  # clear existing cache
        self.OVG_dataset.update_len(config_len_train=0, config_len_val=0)  # set length to zero
        self.OVG_loader = None  # clear OVG loader
        self.rendered_dataset.update(None)  # clear rendered_dataset
        logger.warning("shut down synth dataset engine!")

    def generate_render_cache(self, is_train):
        if self.use_synth:
            # update OVG_dataset
            if is_train:
                self.OVG_dataset.train()
            else:
                self.OVG_dataset.val()

            _, self.occurence_map = self.OVG_dataset.update(self.sample_weight_map, self.occurence_map)

            # cache clear (remove content is ramdisk, may take a while)
            self.cache_recorder.clear()

            # * 1. stream triplets ID from self.OVG_loader,
            # * 2. generate poses in pose_generator
            # * 3. then feed to cache_recorder
            logger.info("> start pose generation")
            with torch.no_grad():
                starttime = time.time()
                for ovg_data_batch in self.OVG_loader:
                    ovg_data_batch = batch_to_device(ovg_data_batch, self.device)
                    ################################################################
                    ############# Generator Pose for Rendereing ####################
                    ################################################################
                    generation_res = self.pose_generator(ovg_data_batch)
                    self.cache_recorder(generation_res)
                endtime = time.time()
            logger.info(f"> end pose generation witin {(endtime - starttime):.2f}s")

            # possible reseat render_synth cache path
            # call update to make sure the filelist is up to date
            self.rendered_dataset.update(None)

            # update random dispatcher
            if is_train:
                self.final_train_set.update()

        # create loader if it is not defined
        if self.final_train_loader is None:
            self.final_train_loader = torch.utils.data.DataLoader(
                self.final_train_set,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=int(self.num_workers),
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
                collate_fn=self.collat_fn,
                worker_init_fn=self.render_synth_train_worker_init_fn,
            )

    def gracefully_quit(self, sig_id, frame):
        if os.path.exists(self.blacklisted_root):
            try:
                shutil.rmtree(self.blacklisted_root)
                logger.info(f"clear manager's cache for blacklisted items")
                shutil.rmtree(self.temp_parent_dir)
                logger.info(f"clear manager's all cache")
            except Exception:
                pass

        if sig_id:
            sys.exit()

    def _construct_blacklist_map(self, sample_n_obj, sample_n_persp, sample_n_grasp, filter_back_flag):
        """_summary_

        Args:
            sample_n_obj (_type_): _description_
            sample_n_persp (_type_): _description_
            sample_n_grasp (_type_): _description_
            filter_back_flag (_type_): _description_
            blacklist_root (_type_): _description_

        Returns:
            _type_: _description_
        """
        # construct an identifier
        obj_engine_type = type(self.obj_engine).__name__
        obj_names = sorted(deepcopy(self.obj_engine.obj_names))
        grasp_engine_type = type(self.grasp_engine).__name__
        view_engine_u_bins = self.view_engine.persp_u_bins
        view_engine_theta_bins = self.view_engine.persp_theta_bins
        CCV_identifier_dict = {
            "obj_engine_type": obj_engine_type,
            "sample_n_obj": sample_n_obj,
            "obj_names": obj_names,
            "grasp_engine_type": grasp_engine_type,
            "sample_n_grasp": sample_n_grasp,
            "view_engine_u_bins": view_engine_u_bins,
            "view_engine_theta_bins": view_engine_theta_bins,
            "filter_back_flag": filter_back_flag
        }

        CCV_identifier = json.dumps(CCV_identifier_dict, sort_keys=True)
        CCV_identifier = hashlib.md5(CCV_identifier.encode("ascii")).hexdigest()
        CCV_identifier_path = os.path.join("common", "cache", "CCV_blacklist", f"{CCV_identifier}.pkl")
        CCV_identifier_dir = os.path.dirname(CCV_identifier_path)
        os.makedirs(CCV_identifier_dir, exist_ok=True)

        blacklist_map = torch.zeros((sample_n_obj, sample_n_persp, sample_n_grasp), dtype=torch.bool)
        if not filter_back_flag:
            return blacklist_map

        if os.path.exists(CCV_identifier_path):
            with open(CCV_identifier_path, "rb") as p_f:
                cached = pickle.load(p_f)
            # sentinel check
            if list(cached.shape) != list(blacklist_map.shape):
                logger.warning(f"blacklist map shape mismatch: {cached.shape()} != {blacklist_map.shape()}")
            else:
                logger.info(f"Loaded cached blacklist_map from {CCV_identifier_path}")
                blacklist_map = cached
                return blacklist_map

        n_filtered = 0
        if filter_back_flag:
            oid_list = list(range(sample_n_obj))
            vid_list = list(range(sample_n_persp))
            gid_list = list(range(sample_n_grasp))
            all_comb = list(itertools.product(oid_list, vid_list, gid_list))
            all_comb_bar = etqdm(all_comb)
            logger.info("iter CCV-space to filter out back holes. this only need to be done once")
            for i, (oi, vi, gi) in enumerate(all_comb_bar):
                # 1. from obj_engine get object_name
                # 2. from grasp_engine get hand pose by obj_name
                # 3. from view engine get presp mats
                obj_name = self.obj_engine.obj_names[oi]
                hand_pose, _, _ = self.grasp_engine.get_obj_grasp(obj_name, gi)
                persp_rotmat, camera_free_transf, z_offset = self.view_engine.get_view(vi)

                back_dir = np.array([1.0, 0.2, 0.0])
                back_dir = back_dir / np.linalg.norm(back_dir)
                wrist_pose = hand_pose[:3]
                wrist_rotmat = aa_to_rotmat(wrist_pose)

                back_arrow = persp_rotmat.T @ wrist_rotmat @ back_dir
                ## compute th_sgn
                th_sgn = back_arrow @ np.array([0, 0, 1])
                if th_sgn < -0.8:
                    blacklist_map[oi, vi, gi] |= True
                    n_filtered += 1

                all_comb_bar.set_description(f"filtering back holes: {n_filtered}/{len(all_comb)}"
                                             f"| oid:{oi:>2} [{obj_name}] | vid:{vi:>4} | gid:{gi:>3}")

        with open(CCV_identifier_path, "wb") as p_f:
            pickle.dump(blacklist_map, p_f)
        logger.info(f"Saved blacklist_map to {CCV_identifier_path}")
        return blacklist_map

    # region ***** mining strategies >>>>>
    @staticmethod
    def update_method_1(sample_weight_map, val_res, sample_weight_lower_bound, sample_weight_upper_bound, **kwarg):
        """Precentile mining strategy"""
        # update logic
        val_ovg_id_list = list(val_res.keys())
        val_values_list = list(val_res.values())
        val_max, val_min = max(val_values_list), min(val_values_list)
        val_range = val_max - val_min
        val_confidence = (val_max - np.array(val_values_list)) / (val_range + 1e-8)

        # base_ = 1.0 / (val_confidence + 0.5)
        # pow_ = 1.0 - val_confidence / (1.0 + val_confidence)
        # weight_update = base_ ** pow_
        weight_update = 1.0 / (val_confidence + 0.5)  # simplest
        weight_update = weight_update.tolist()

        for i, ovg in enumerate(val_ovg_id_list):
            sample_weight_map[ovg[0], ovg[1], ovg[2]] *= weight_update[i]
        # constrain sample weight
        sample_weight_map = torch.clamp(sample_weight_map, sample_weight_lower_bound, sample_weight_upper_bound)
        return {"sample_weight_map": sample_weight_map}

    @staticmethod
    def update_method_2(sample_weight_map, val_res, sample_weight_lower_bound, sample_weight_upper_bound, **kwarg):
        """Incrimental mining strategy"""
        # update logic
        val_ovg_id_list = list(val_res.keys())
        val_values_list = list(val_res.values())
        val_max, val_min = max(val_values_list), min(val_values_list)
        val_range = val_max - val_min
        val_confidence = (val_max - np.array(val_values_list)) / (val_range + 1e-8)

        decrease_mask = val_confidence > 0.5

        for i, ovg in enumerate(val_ovg_id_list):
            if decrease_mask[i]:
                sample_weight_map[ovg[0], ovg[1], ovg[2]] -= 0.1
            else:
                sample_weight_map[ovg[0], ovg[1], ovg[2]] += 0.1

        # constrain sample weight
        sample_weight_map = torch.clamp(sample_weight_map, sample_weight_lower_bound, sample_weight_upper_bound)
        return {"sample_weight_map": sample_weight_map}

    @staticmethod
    def update_method_3(sample_weight_map, val_res, sample_weight_lower_bound, sample_weight_upper_bound, **kwarg):
        """Lower bound deactivation strategy, aka early shutdown, found useful in RHD, HO3D"""
        dist_lower_threshold = kwarg["dist_lower_threshold"]
        dist_upper_threshold = kwarg["dist_upper_threshold"]
        val_ovg_id_list = np.array(list(val_res.keys()))
        val_values_list = np.array(list(val_res.values()))

        val_values_lower_mask = val_values_list < dist_lower_threshold
        val_values_upper_mask = val_values_list > dist_upper_threshold

        for i, ovg in enumerate(val_ovg_id_list):
            if val_values_lower_mask[i]:
                sample_weight_map[ovg[0], ovg[1], ovg[2]] = 0.0
            elif val_values_upper_mask[i]:
                sample_weight_map[ovg[0], ovg[1], ovg[2]] = 1.0
            else:
                sample_weight_map[ovg[0], ovg[1], ovg[2]] *= 0.5

        return {
            "sample_weight_map": sample_weight_map,
            "dist_lower_ratio": val_values_lower_mask.sum() / len(val_values_lower_mask),
        }

    @staticmethod
    def update_method_4(sample_weight_map, val_res, sample_weight_lower_bound, sample_weight_upper_bound, **kwarg):
        """Combined method 1 & 3.
        when use update_method 4, both the dist_lower_threshold and
        dist_upper_threshlod should be less than that in
        update_method_3. Since we have already fit the network in the first 75% epoches.
        """
        curr_epoch_idx = kwarg["epoch_idx"]
        n_epochs = kwarg["n_epochs"]

        if float(curr_epoch_idx) / n_epochs < 0.75:  # less than 75% training epoch pass
            # update_method_1
            update_res = ArtiBoostLoader.update_method_1(
                sample_weight_map,
                val_res,
                sample_weight_lower_bound,
                sample_weight_upper_bound,
            )
            update_res["dist_lower_ratio"] = -1.0  # -1 means invalid
        else:
            update_res = ArtiBoostLoader.update_method_3(
                sample_weight_map,
                val_res,
                sample_weight_lower_bound,
                sample_weight_upper_bound,
                **kwarg,
            )
        return update_res

    # endregion