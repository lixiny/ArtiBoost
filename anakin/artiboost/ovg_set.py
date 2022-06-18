import numpy as np
import torch
from anakin.utils.logger import logger

from .grasp_engine import GraspEngine
from .object_engine import ObjEngine
from .view_engine import ViewEngine


def caculate_align_mat(vec):
    vec = vec / np.linalg.norm(vec)
    z_unit_Arr = np.array([0, 0, 1])

    z_mat = np.array([
        [0, -z_unit_Arr[2], z_unit_Arr[1]],
        [z_unit_Arr[2], 0, -z_unit_Arr[0]],
        [-z_unit_Arr[1], z_unit_Arr[0], 0],
    ])

    z_c_vec = np.matmul(z_mat, vec)
    z_c_vec_mat = np.array([
        [0, -z_c_vec[2], z_c_vec[1]],
        [z_c_vec[2], 0, -z_c_vec[0]],
        [-z_c_vec[1], z_c_vec[0], 0],
    ])

    if np.dot(z_unit_Arr, vec) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, vec) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat, z_c_vec_mat) / (1 + np.dot(z_unit_Arr, vec))

    return qTrans_Mat


# this set need to stream (in loader fashion)
# * ObjectType - Viewpoint - GraspPose : OVG
class OVGSet(torch.utils.data.Dataset):

    def __init__(
        self,
        obj_engine: ObjEngine,
        grasp_engine: GraspEngine,
        view_engine: ViewEngine,
        config_len_train: int,
        config_len_val: int,
        n_grasp: int,
        blacklist_map: torch.Tensor,
    ):
        super().__init__()

        self.obj_engine = obj_engine
        self.grasp_engine = grasp_engine
        self.view_engine = view_engine

        self.config_len_train = config_len_train
        self.config_len_val = config_len_val
        self.train_mode = True

        self.object_mesh_mapping = obj_engine.obj_trimeshes_mapping
        self.n_obj = len(self.obj_engine.obj_names)
        self.n_grasp = n_grasp
        self.n_persp_center = self.view_engine.n_persp_center

        # region check if the sampled number is above all the possible outcomes
        self.n_all_choices = self.n_obj * self.n_persp_center * self.n_grasp
        if self.n_all_choices < self.config_len_train:
            logger.warning(f"ovg config_len_train {self.config_len_train} is over all possible combination "
                           "of number {self.n_all_choices}, but not capped")
        if self.n_all_choices < self.config_len_val:
            logger.warning(f"ovg config_len_val {self.config_len_val} is over all possible combination "
                           "of number {self.n_all_choices}, capped")
            self.config_len_val = self.n_all_choices
            logger.info(f"ovg config_len_val is set to {self.config_len_val}")
        # endregion

        self.blacklist_map = blacklist_map

        self.cata_dist = None
        self.sampled_idx_tensor = None
        self.sampled_obj_idx = None
        self.sampled_persp_idx = None
        self.sampled_grasp_idx = None

    def __len__(self):
        if self.train_mode:
            return self.config_len_train
        else:
            return self.config_len_val

    def update_len(self, config_len_train=None, config_len_val=None):
        if config_len_train is not None:
            self.config_len_train = config_len_train
        if config_len_val is not None:
            self.config_len_val = config_len_val

    def train(self):
        self.train_mode = True

    def val(self):
        self.train_mode = False

    def update(self, global_sample_weight_map, global_occurence_map):
        # condition on self.train_modes
        if self.train_mode:
            this_sample_weight_map = global_sample_weight_map.detach().clone()
        else:
            this_sample_weight_map = torch.ones_like(global_sample_weight_map)
            this_sample_weight_map[self.blacklist_map] = 0.0  # mask by blacklist_map

        if self.train_mode:
            cata_dist = torch.distributions.Categorical(this_sample_weight_map.reshape(-1))
            self.sampled_idx_tensor = cata_dist.sample(sample_shape=(self.config_len_train,))
        else:
            self.sampled_idx_tensor = torch.multinomial(input=this_sample_weight_map.reshape(-1),
                                                        num_samples=self.config_len_val,
                                                        replacement=False)

        # sampled content to ovg tuples
        self.sampled_obj_idx, self.sampled_persp_idx, self.sampled_grasp_idx = \
            self.row_col_calc(self.sampled_idx_tensor, self.n_persp_center, self.n_grasp)

        # compute occurence count & occurence map
        this_occurence_count_map = self.compute_occurence_count_map(self.sampled_obj_idx, self.sampled_persp_idx,
                                                                    self.sampled_grasp_idx, self.n_obj,
                                                                    self.n_persp_center, self.n_grasp)
        this_occurence_map = this_occurence_count_map > 0
        global_occurence_map |= this_occurence_map  # OR with global occurence

        # return according to mode
        return this_sample_weight_map, global_occurence_map

    def __getitem__(self, index):
        ori_index = index

        obj_id = self.sampled_obj_idx[index]
        persp_id = self.sampled_persp_idx[index]
        grasp_id = self.sampled_grasp_idx[index]

        obj_name = self.obj_engine.obj_names[obj_id]
        hand_pose, hand_shape, hand_tsl = self.grasp_engine.get_obj_grasp(obj_name, grasp_id)
        persp_rotmat, camera_free_transf, z_offset = self.view_engine.get_view(persp_id)

        synth_extend = {
            "index": ori_index,
            "obj_id": obj_id,
            "obj_name": obj_name,
            "persp_id": persp_id,
            "grasp_id": grasp_id,
            "hand_pose": hand_pose.astype(np.float32),
            "hand_shape": hand_shape.astype(np.float32),
            "hand_tsl": hand_tsl.astype(np.float32),
            "persp_rotmat": persp_rotmat.astype(np.float32),
            "camera_free_transf": camera_free_transf.astype(np.float32),
            "z_offset": z_offset.astype(np.float32),
        }

        return synth_extend

    @staticmethod
    def row_col_calc(tidx, n_row, n_col):
        # bidx = tidx // (n_row * n_col)
        bidx = torch.div(tidx, n_row * n_col, rounding_mode="floor")
        # ridx = (tidx // n_col) % n_row
        ridx = torch.div(tidx, n_col, rounding_mode="floor") % n_row
        cidx = tidx % n_col
        # print(idx)
        # print((bidx * n_row + ridx) * n_col + cidx)
        return (bidx, ridx, cidx)

    @staticmethod
    def compute_occurence_count_map(bidx, ridx, cidx, n_b, n_r, n_c):
        res = torch.zeros((n_b, n_r, n_c), dtype=torch.long).tolist()
        for b, r, c in zip(bidx, ridx, cidx):
            res[b][r][c] += 1
        res = torch.Tensor(res)
        return res
