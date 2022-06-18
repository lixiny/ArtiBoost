import numpy as np
import torch
from torch.distributions.uniform import Uniform


class ViewEngine:

    def __init__(self, cfg):
        self.persp_u_bins = cfg["PERSP_U_BINS"]
        self.persp_theta_bins = cfg["PERSP_THETA_BINS"]
        self.camera_z_range = cfg["CAMERA_Z_RANGE"]
        self.n_persp_center = self.persp_u_bins * self.persp_theta_bins

        camera_z_min, camera_z_max = self.camera_z_range
        self.camera_z_dist = Uniform(camera_z_min, camera_z_max)

    def get_view(self, persp_id):
        persp_rotmat = self.get_perspective_from_id(persp_id)
        camera_free = np.random.rand() * (2 * np.pi)
        camera_free_rotmat = np.array([
            [np.cos(camera_free), -np.sin(camera_free), 0],
            [np.sin(camera_free), np.cos(camera_free), 0],
            [0, 0, 1],
        ])
        camera_free_transf = np.eye(4)
        camera_free_transf[:3, :3] = camera_free_rotmat

        z_offset = float(self.camera_z_dist.sample())
        z_offset = np.array([0.0, 0.0, z_offset], dtype=np.float)

        return persp_rotmat, camera_free_transf, z_offset

    def get_perspective_from_id(self, persp_id):
        # get row, col
        # u_id = persp_id // self.persp_theta_bins ## deprecated
        u_id = torch.div(persp_id, self.persp_theta_bins, rounding_mode='floor')
        theta_id = persp_id % self.persp_theta_bins

        u_unit = 2 / self.persp_u_bins
        theta_unit = (2 * np.pi) / self.persp_theta_bins

        u_center = (-1 + u_unit / 2) + u_id * u_unit  # u \in [-1, 1]
        theta_center = theta_unit / 2 + theta_id * theta_unit  # \theta \in [0, 2\pi)

        # get camera randomness (torch rand to save for manual seed)
        u_offset = float(torch.rand(1) - 0.5) * u_unit
        theta_offset = float(torch.rand(1) - 0.5) * theta_unit

        u = np.clip(u_center + u_offset, -1, 1)
        theta = np.clip(theta_center + theta_offset, 0, 2 * np.pi)

        # get rotation vector
        x = np.sqrt(1 - u * u) * np.cos(theta)
        y = np.sqrt(1 - u * u) * np.sin(theta)

        target_vec = np.array([x, y, u])
        rotmat = self.caculate_align_mat(target_vec)
        return rotmat

    @staticmethod
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
            qTrans_Mat = (np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat, z_c_vec_mat) /
                          (1 + np.dot(z_unit_Arr, vec)))

        return qTrans_Mat
