from typing import Union, Optional, List

import numpy as np
import torch
from pytorch3d.transforms import (axis_angle_to_matrix, axis_angle_to_quaternion, euler_angles_to_matrix,
                                  matrix_to_euler_angles, matrix_to_quaternion, matrix_to_rotation_6d,
                                  quaternion_to_axis_angle, quaternion_to_matrix, rotation_6d_to_matrix)


class Compose:

    def __init__(self, transforms: list):
        """Composes several transforms together. This transform does not
        support torchscript.

        Args:
            transforms (list): (list of transform functions)
        """
        self.transforms = transforms

    def __call__(self, rotation: Union[torch.Tensor, np.ndarray], convention: str = 'xyz', **kwargs):
        convention = convention.lower()
        if not (set(convention) == set('xyz') and len(convention) == 3):
            raise ValueError(f'Invalid convention {convention}.')
        if isinstance(rotation, np.ndarray):
            data_type = 'np'
            rotation = torch.FloatTensor(rotation)
        elif isinstance(rotation, torch.Tensor):
            data_type = 'tensor'
        else:
            raise TypeError('Type of rotation should be torch.Tensor or np.ndarray')
        for t in self.transforms:
            if 'convention' in t.__code__.co_varnames:
                rotation = t(rotation, convention.upper(), **kwargs)
            else:
                rotation = t(rotation, **kwargs)
        if data_type == 'np':
            rotation = rotation.detach().cpu().numpy()
        return rotation


def aa_to_rotmat(axis_angle: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert axis_angle to rotation matrixs.
    Args:
        axis_angle (Union[torch.Tensor, np.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, np.ndarray]: shape would be (..., 3, 3).
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(f'Invalid input axis angles shape f{axis_angle.shape}.')
    t = Compose([axis_angle_to_matrix])
    return t(axis_angle)


def aa_to_quat(axis_angle: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert axis_angle to quaternions.
    Args:
        axis_angle (Union[torch.Tensor, np.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, np.ndarray]: shape would be (..., 4).
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(f'Invalid input axis angles f{axis_angle.shape}.')
    t = Compose([axis_angle_to_quaternion])
    return t(axis_angle)


def ee_to_rotmat(euler_angle: Union[torch.Tensor, np.ndarray], convention='xyz') -> Union[torch.Tensor, np.ndarray]:
    """Convert euler angle to rotation matrixs.

    Args:
        euler_angle (Union[torch.Tensor, np.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.
    Returns:
        Union[torch.Tensor, np.ndarray]: shape would be (..., 3, 3).
    """
    if euler_angle.shape[-1] != 3:
        raise ValueError(f'Invalid input euler angles shape f{euler_angle.shape}.')
    t = Compose([euler_angles_to_matrix])
    return t(euler_angle, convention.upper())


def rotmat_to_ee(matrix: Union[torch.Tensor, np.ndarray], convention: str = 'xyz') -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation matrixs to euler angle.

    Args:
        matrix (Union[torch.Tensor, np.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.
    Returns:
        Union[torch.Tensor, np.ndarray]: shape would be (..., 3).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f'Invalid rotation matrix shape f{matrix.shape}.')
    t = Compose([matrix_to_euler_angles])
    return t(matrix, convention.upper())


def rotmat_to_quat(matrix: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation matrixs to quaternions.

    Args:
        matrix (Union[torch.Tensor, np.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, np.ndarray]: shape would be (..., 4).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f'Invalid rotation matrix  shape f{matrix.shape}.')
    t = Compose([matrix_to_quaternion])
    return t(matrix)


def rotmat_to_rot6d(matrix: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation matrixs to rotation 6d representations.

    Args:
        matrix (Union[torch.Tensor, np.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, np.ndarray]: shape would be (..., 6).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f'Invalid rotation matrix  shape f{matrix.shape}.')
    t = Compose([matrix_to_rotation_6d])
    return t(matrix)


def quat_to_aa(quaternions: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert quaternions to axis angles.

    Args:
        quaternions (Union[torch.Tensor, np.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, np.ndarray]: shape would be (..., 3).
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f'Invalid input quaternions f{quaternions.shape}.')
    t = Compose([quaternion_to_axis_angle])
    return t(quaternions)


def quat_to_rotmat(quaternions: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert quaternions to rotation matrixs.

    Args:
        quaternions (Union[torch.Tensor, np.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, np.ndarray]: shape would be (..., 3, 3).
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f'Invalid input quaternions shape f{quaternions.shape}.')
    t = Compose([quaternion_to_matrix])
    return t(quaternions)


def rot6d_to_rotmat(rotation_6d: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation 6d representations to rotation matrixs.

    Args:
        rotation_6d (Union[torch.Tensor, np.ndarray]): input shape
                should be (..., 6). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, np.ndarray]: shape would be (..., 3, 3).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f'Invalid input rotation_6d f{rotation_6d.shape}.')
    t = Compose([rotation_6d_to_matrix])
    return t(rotation_6d)


def aa_to_ee(axis_angle: Union[torch.Tensor, np.ndarray], convention: str = 'xyz') -> Union[torch.Tensor, np.ndarray]:
    """Convert axis angles to euler angle.

    Args:
        axis_angle (Union[torch.Tensor, np.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.

    Returns:
        Union[torch.Tensor, np.ndarray]: shape would be (..., 3).
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(f'Invalid input axis_angle shape f{axis_angle.shape}.')
    t = Compose([axis_angle_to_matrix, matrix_to_euler_angles])
    return t(axis_angle, convention)


def aa_to_rot6d(axis_angle: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert axis angles to rotation 6d representations.

    Args:
        axis_angle (Union[torch.Tensor, np.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, np.ndarray]: shape would be (..., 6).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(f'Invalid input axis_angle f{axis_angle.shape}.')
    t = Compose([axis_angle_to_matrix, matrix_to_rotation_6d])
    return t(axis_angle)


def ee_to_aa(euler_angle: Union[torch.Tensor, np.ndarray], convention: str = 'xyz') -> Union[torch.Tensor, np.ndarray]:
    """Convert euler angles to axis angles.

    Args:
        euler_angle (Union[torch.Tensor, np.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.

    Returns:
        Union[torch.Tensor, np.ndarray]: shape would be (..., 3).
    """
    if euler_angle.shape[-1] != 3:
        raise ValueError(f'Invalid input euler_angle f{euler_angle.shape}.')
    t = Compose([euler_angles_to_matrix, matrix_to_quaternion, quaternion_to_axis_angle])
    return t(euler_angle, convention)


def ee_to_quat(euler_angle: Union[torch.Tensor, np.ndarray], convention='xyz') -> Union[torch.Tensor, np.ndarray]:
    """Convert euler angles to quaternions.

    Args:
        euler_angle (Union[torch.Tensor, np.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.

    Returns:
        Union[torch.Tensor, np.ndarray]: shape would be (..., 4).
    """
    if euler_angle.shape[-1] != 3:
        raise ValueError(f'Invalid input euler_angle f{euler_angle.shape}.')
    t = Compose([euler_angles_to_matrix, matrix_to_quaternion])
    return t(euler_angle, convention)


def ee_to_rot6d(euler_angle: Union[torch.Tensor, np.ndarray], convention='xyz') -> Union[torch.Tensor, np.ndarray]:
    """Convert euler angles to rotation 6d representation.

    Args:
        euler_angle (Union[torch.Tensor, np.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.

    Returns:
        Union[torch.Tensor, np.ndarray]: shape would be (..., 6).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if euler_angle.shape[-1] != 3:
        raise ValueError(f'Invalid input euler_angle f{euler_angle.shape}.')
    t = Compose([euler_angles_to_matrix, matrix_to_rotation_6d])
    return t(euler_angle, convention)


def rotmat_to_aa(matrix: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation matrixs to axis angles.

    Args:
        matrix (Union[torch.Tensor, np.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.

    Returns:
        Union[torch.Tensor, np.ndarray]: shape would be (..., 3).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f'Invalid rotation matrix  shape f{matrix.shape}.')
    t = Compose([matrix_to_quaternion, quaternion_to_axis_angle])
    return t(matrix)


def quat_to_ee(quaternions: Union[torch.Tensor, np.ndarray],
               convention: str = 'xyz') -> Union[torch.Tensor, np.ndarray]:
    """Convert quaternions to euler angles.

    Args:
        quaternions (Union[torch.Tensor, np.ndarray]): input shape
                should be (..., 4). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.

    Returns:
        Union[torch.Tensor, np.ndarray]: shape would be (..., 3).
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f'Invalid input quaternions f{quaternions.shape}.')
    t = Compose([quaternion_to_matrix, matrix_to_euler_angles])
    return t(quaternions, convention)


def quat_to_rot6d(quaternions: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert quaternions to rotation 6d representations.

    Args:
        quaternions (Union[torch.Tensor, np.ndarray]): input shape
                should be (..., 4). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, np.ndarray]: shape would be (..., 6).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f'Invalid input quaternions f{quaternions.shape}.')
    t = Compose([quaternion_to_matrix, matrix_to_rotation_6d])
    return t(quaternions)


def rot6d_to_aa(rotation_6d: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation 6d representations to axis angles.

    Args:
        rotation_6d (Union[torch.Tensor, np.ndarray]): input shape
                should be (..., 6). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, np.ndarray]: shape would be (..., 3).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f'Invalid input rotation_6d f{rotation_6d.shape}.')
    t = Compose([rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_axis_angle])
    return t(rotation_6d)


def rot6d_to_ee(rotation_6d: Union[torch.Tensor, np.ndarray],
                convention: str = 'xyz') -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation 6d representations to euler angles.

    Args:
        rotation_6d (Union[torch.Tensor, np.ndarray]): input shape
                should be (..., 6). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, np.ndarray]: shape would be (..., 3).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f'Invalid input rotation_6d f{rotation_6d.shape}.')
    t = Compose([rotation_6d_to_matrix, matrix_to_euler_angles])
    return t(rotation_6d, convention)


def rot6d_to_quat(rotation_6d: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation 6d representations to quaternions.

    Args:
        rotation (Union[torch.Tensor, np.ndarray]): input shape
                should be (..., 6). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, np.ndarray]: shape would be (..., 4).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f'Invalid input rotation_6d shape f{rotation_6d.shape}.')
    t = Compose([rotation_6d_to_matrix, matrix_to_quaternion])
    return t(rotation_6d)


def th_homogeneous(tensor):
    batch_size = tensor.shape[0]
    padding = tensor.new([0.0, 0.0, 0.0, 1.0])
    padding.requires_grad = False
    concat_list = [tensor, padding.view(1, 1, 4).repeat(batch_size, 1, 1)]
    cat_res = torch.cat(concat_list, 1)
    return cat_res


def transform_coords(pts, affine_trans, invert=False):
    """
    Args:
        pts(np.ndarray): (point_nb, 2)
    """
    if invert:
        affine_trans = np.linalg.inv(affine_trans)
    hom2d = np.concatenate([pts, np.ones([np.array(pts).shape[0], 1])], 1)
    transformed_rows = affine_trans.dot(hom2d.transpose()).transpose()[:, :2]
    return transformed_rows


def get_affine_transform(center, scale, optical_center, out_res, rot=0):
    rot_mat = np.zeros((3, 3))
    sn, cs = np.sin(rot), np.cos(rot)
    rot_mat[0, :2] = [cs, -sn]
    rot_mat[1, :2] = [sn, cs]
    rot_mat[2, 2] = 1
    # Rotate center to obtain coordinate of center in rotated image
    origin_rot_center = rot_mat.dot(center.tolist() + [1])[:2]
    # Get center for transform with verts rotated around optical axis
    # (through pixel center, smthg like 128, 128 in pixels and 0,0 in 3d world)
    # For this, rotate the center but around center of image (vs 0,0 in pixel space)
    t_mat = np.eye(3)
    t_mat[0, 2] = -optical_center[0]
    t_mat[1, 2] = -optical_center[1]
    t_inv = t_mat.copy()
    t_inv[:2, 2] *= -1
    transformed_center = t_inv.dot(rot_mat).dot(t_mat).dot(center.tolist() + [1])
    post_rot_trans = get_affine_trans_no_rot(origin_rot_center, scale, out_res)
    total_trans = post_rot_trans.dot(rot_mat)
    # check_t = get_affine_transform_bak(center, scale, res, rot)
    # print(total_trans, check_t)
    affinetrans_post_rot = get_affine_trans_no_rot(transformed_center[:2], scale, out_res)
    return (
        total_trans.astype(np.float32),
        affinetrans_post_rot.astype(np.float32),
    )


def get_affine_trans_no_rot(center, scale, res):
    affinet = np.zeros((3, 3))
    scale_ratio = float(res[0]) / float(res[1])
    affinet[0, 0] = float(res[0]) / scale
    affinet[1, 1] = float(res[1]) / scale * scale_ratio
    affinet[0, 2] = res[0] * (-float(center[0]) / scale + 0.5)
    affinet[1, 2] = res[1] * (-float(center[1]) / scale * scale_ratio + 0.5)
    affinet[2, 2] = 1
    return affinet


def batch_xyz2uvd(
    xyz: torch.Tensor,
    root_joint: torch.Tensor,
    intr: torch.Tensor,
    inp_res: Optional[List[int]] = None,
    depth_range=0.4,
    ref_bone_len: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if inp_res is None:
        inp_res = [256, 256]
    inp_res = torch.Tensor(inp_res).to(xyz.device)  # TENSOR (2,)
    batch_size = xyz.shape[0]
    if ref_bone_len is None:
        ref_bone_len = torch.ones((batch_size, 1)).to(xyz.device)  # TENSOR (B, 1)

    # ================== 1. normalize depth : root_relative, scale_invariant >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    z = xyz[:, :, 2]  # TENSOR (B, NKP)
    xy = xyz[:, :, :2]  # TENSOR (B, NKP, 2)
    xy_ = xy / z.unsqueeze(-1).expand_as(xy)  # TENSOR (B, NKP, 2)
    root_joint_z = root_joint[:, -1].unsqueeze(-1)  # TENSOR (B, 1)
    z_ = (z - root_joint_z.expand_as(z)) / ref_bone_len.expand_as(z)  # TENSOR (B, NKP)

    # ================== 2. xy_ -> uv >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    fx = intr[:, 0, 0].unsqueeze(-1)  # TENSOR (B, 1)
    fy = intr[:, 1, 1].unsqueeze(-1)
    cx = intr[:, 0, 2].unsqueeze(-1)
    cy = intr[:, 1, 2].unsqueeze(-1)
    # cat 4 TENSOR (B, 1)
    camparam = torch.cat((fx, fy, cx, cy), dim=1)  # TENSOR (B, 4)
    camparam = camparam.unsqueeze(1).expand(-1, xyz.shape[1], -1)  # TENSOR (B, NKP, 4)
    uv = (xy_ * camparam[:, :, :2]) + camparam[:, :, 2:4]  # TENSOR (B, NKP, 2)

    # ================== 3. normalize uvd to 0~1 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    uv = torch.einsum("bij, j->bij", uv, 1.0 / inp_res)  # TENSOR (B, NKP, 2), [0 ~ 1]
    d = z_ / depth_range + 0.5  # TENSOR (B, NKP), [0 ~ 1]

    return torch.cat((uv, d.unsqueeze(-1)), -1)  # TENSOR (B, NKP, 3)


def batch_uvd2xyz(
    uvd: torch.Tensor,
    root_joint: torch.Tensor,
    intr: torch.Tensor,
    inp_res: Optional[List[int]] = None,
    depth_range: float = 0.4,  # m
    ref_bone_len: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if inp_res is None:
        inp_res = [256, 256]
    inp_res = torch.Tensor(inp_res).to(uvd.device)
    batch_size = uvd.shape[0]
    if ref_bone_len is None:
        ref_bone_len = torch.ones((batch_size, 1)).to(uvd.device)

    # ================== 1. denormalized uvd >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    uv = torch.einsum("bij,j->bij", uvd[:, :, :2], inp_res)  # TENSOR (B, NKP, 2), [0 ~ INP_RES]
    d = (uvd[:, :, 2] - 0.5) * depth_range  # TENSOR (B, NKP), [-0.2 ~ 0.2]

    root_joint_z = root_joint[:, -1].unsqueeze(-1)  # TENSOR (B, 1)
    z = d * ref_bone_len + root_joint_z.expand_as(uvd[:, :, 2])  # TENSOR (B, NKP)

    # ================== 2. uvd->xyz >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # camparam = torch.zeros((batch_size, 4)).float().to(uvd.device)  # TENSOR (B, 4)
    fx = intr[:, 0, 0].unsqueeze(-1)  # TENSOR (B, 1)
    fy = intr[:, 1, 1].unsqueeze(-1)
    cx = intr[:, 0, 2].unsqueeze(-1)
    cy = intr[:, 1, 2].unsqueeze(-1)
    # cat 4 TENSOR (B, 1)
    camparam = torch.cat((fx, fy, cx, cy), dim=1)  # TENSOR (B, 4)
    camparam = camparam.unsqueeze(1).expand(-1, uvd.shape[1], -1)  # TENSOR (B, NKP, 4)
    xy_ = (uv - camparam[:, :, 2:4]) / camparam[:, :, :2]  # TENSOR (B, NKP, 2)
    xy = xy_ * z.unsqueeze(-1).expand_as(uv)  # TENSOR (B, NKP, 2)

    return torch.cat((xy, z.unsqueeze(-1)), -1)  # TENSOR (B, NKP, 3)


def batch_ref_bone_len(joint: Union[np.ndarray, torch.Tensor], ref_bone_link=None) -> Union[np.ndarray, torch.Tensor]:
    if ref_bone_link is None:
        ref_bone_link = (0, 9)

    if not torch.is_tensor(joint) and not isinstance(joint, np.ndarray):
        raise TypeError("joint should be ndarray or torch tensor. Got {}".format(type(joint)))
    if len(joint.shape) != 3 or joint.shape[1] != 21 or joint.shape[2] != 3:
        raise TypeError("joint should have shape (B, njoint, 3), Got {}".format(joint.shape))

    batch_size = joint.shape[0]
    bone = 0
    if torch.is_tensor(joint):
        bone = torch.zeros((batch_size, 1)).to(joint.device)
        for jid, nextjid in zip(ref_bone_link[:-1], ref_bone_link[1:]):
            bone += torch.norm(joint[:, jid, :] - joint[:, nextjid, :], dim=1, keepdim=True)  # (B, 1)
    elif isinstance(joint, np.ndarray):
        bone = np.zeros((batch_size, 1))
        for jid, nextjid in zip(ref_bone_link[:-1], ref_bone_link[1:]):
            bone += np.linalg.norm((joint[:, jid, :] - joint[:, nextjid, :]), ord=2, axis=1, keepdims=True)  # (B, 1)
    return bone


def batch_persp_proj2d(verts: torch.Tensor, camintr: torch.Tensor):
    # Project 3d vertices on image plane
    verts_hom2d = camintr.bmm(verts.transpose(1, 2)).transpose(1, 2)
    proj_verts2d = verts_hom2d[:, :, :2] / verts_hom2d[:, :, 2:]
    return proj_verts2d


def compute_rotation_matrix_from_ortho6d(poses):
    """
    Code from
    https://github.com/papagina/RotationContinuity
    On the Continuity of Rotation Representations in Neural Networks
    Zhou et al. CVPR19
    https://zhouyisjtu.github.io/project_rotation/rotation.html
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, v.new([1e-8]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v/v_mag
    return v


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)

    return out


def center_vert_bbox(vertices, bbox_center=None, bbox_scale=None, scale=False):
    if bbox_center is None:
        bbox_center = (vertices.min(0) + vertices.max(0)) / 2
    vertices = vertices - bbox_center
    if scale:
        if bbox_scale is None:
            bbox_scale = np.linalg.norm(vertices, 2, 1).max()
        vertices = vertices / bbox_scale
    else:
        bbox_scale = 1
    return vertices, bbox_center, bbox_scale
