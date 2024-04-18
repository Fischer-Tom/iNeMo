import BboxTools as bbt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.renderer import look_at_rotation


def load_off(off_file_name, to_torch=False):
    file_handle = open(off_file_name)

    file_list = file_handle.readlines()
    n_points = int(file_list[1].split(" ")[0])
    all_strings = "".join(file_list[2 : 2 + n_points])
    array_ = np.fromstring(all_strings, dtype=np.float32, sep="\n")

    all_strings = "".join(file_list[2 + n_points :])
    array_int = np.fromstring(all_strings, dtype=np.int32, sep="\n")
    file_handle.close()

    array_ = array_.reshape((-1, 3))

    if to_torch:
        return torch.from_numpy(array_), torch.from_numpy(
            array_int.reshape((-1, 4))[:, 1::],
        )
    else:
        return array_, array_int.reshape((-1, 4))[:, 1::]


def batched_index_select(t, dim, inds):
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy)  # b * e * f
    return out


def keypoints_to_pixel_index(keypoints, img_size: tuple):
    line_size = img_size[1]
    return (keypoints[:, :, 0] * line_size + keypoints[:, :, 1]).clamp(
        min=0, max=img_size[0] * img_size[1] - 1
    )


def get_noise_pixel_index(keypoints, max_size, n_samples, obj_mask=None):
    n = keypoints.shape[0]
    # remove the point in keypoints by set probability to 0 otherwise 1 -> mask [n, size] with 0 or 1
    mask = torch.ones((n, max_size), dtype=torch.float32).to(keypoints.device)
    mask = mask.scatter(1, keypoints.type(torch.long), 0.0)
    if obj_mask is not None:
        mask = obj_mask.view(n, -1)
    # generate the sample by the probabilities
    return torch.multinomial(mask, n_samples)


class GlobalLocalConverter(nn.Module):
    def __init__(self, local_size):
        super().__init__()
        self.local_size = local_size
        self.padding = sum(([t - 1 - t // 2, t // 2] for t in local_size[::-1]), [])

    def forward(self, X):
        X = F.pad(X, self.padding)
        X = F.unfold(X, kernel_size=self.local_size)
        return X


def campos_to_R_T(
    campos,
    theta,
    device="cpu",
    at=((0, 0, 0),),
    up=((0, 1, 0),),
    extra_trans=None,
):
    R = look_at_rotation(campos, at=at, device=device, up=up)  # (n, 3, 3)
    R = torch.bmm(R, rotation_theta(theta, device_=device))
    if extra_trans is not None:
        T = (
            -torch.bmm(R.transpose(1, 2), campos.unsqueeze(2))[:, :, 0] + extra_trans
        )  # (1, 3)
    else:
        T = -torch.bmm(R.transpose(1, 2), campos.unsqueeze(2))[:, :, 0]  # (1, 3)
    return R, T


def rotation_theta(theta, device_=None):
    # cos -sin  0
    # sin  cos  0
    # 0    0    1
    if type(theta) == float or type(theta) == np.float64:
        if device_ is None:
            device_ = "cpu"
        theta = torch.ones((1, 1, 1)).to(device_) * theta
    else:
        if device_ is None:
            device_ = theta.device
        theta = theta.view(-1, 1, 1)

    mul_ = (
        torch.Tensor([[1, 0, 0, 0, 1, 0, 0, 0, 0], [0, -1, 0, 1, 0, 0, 0, 0, 0]])
        .view(1, 2, 9)
        .to(device_)
    )
    bia_ = torch.Tensor([0] * 8 + [1]).view(1, 1, 9).to(device_)

    # [n, 1, 2]
    cos_sin = torch.cat((torch.cos(theta), torch.sin(theta)), dim=2).to(device_)

    # [n, 1, 2] @ [1, 2, 9] + [1, 1, 9] => [n, 1, 9] => [n, 3, 3]
    trans = torch.matmul(cos_sin, mul_) + bia_
    trans = trans.view(-1, 3, 3)

    return trans


def pre_process_mesh_pascal(verts):
    verts = torch.cat((verts[..., 0:1], verts[..., 2:3], -verts[..., 1:2]), dim=-1)
    return verts


def set_bary_coords_to_nearest(bary_coords_):
    ori_shape = bary_coords_.shape
    exr = bary_coords_ * (bary_coords_ < 0)
    bary_coords_ = bary_coords_.view(-1, bary_coords_.shape[-1])
    arg_max_idx = bary_coords_.argmax(1)
    return (
        torch.zeros_like(bary_coords_)
        .scatter(1, arg_max_idx.unsqueeze(1), 1.0)
        .view(*ori_shape)
        + exr
    )


def get_cube_proj(C, theta_gd, inter_module):
    projected_map = inter_module(C, theta_gd)
    box_obj = bbt.nonzero(projected_map)

    # Features extraction
    object_height, object_width = box_obj[0], box_obj[1]

    return projected_map, object_height, object_width


def camera_position_to_spherical_angle(camera_pose):
    distance_o = torch.sum(camera_pose**2, dim=1) ** 0.5
    azimuth_o = torch.atan(
        camera_pose[:, 0] / camera_pose[:, 2]
    ) % torch.pi + torch.pi * (camera_pose[:, 0] < 0).type(camera_pose.dtype).to(
        camera_pose.device
    )
    elevation_o = torch.asin(camera_pose[:, 1] / distance_o)
    return distance_o, elevation_o, azimuth_o


def get_transformation_matrix(azimuth, elevation, distance):
    distance[distance == 0] = 0.1
    # camera center
    C = torch.zeros((azimuth.shape[0], 3), device=azimuth.device)
    C[:, 0] = distance * torch.cos(elevation) * torch.sin(azimuth)
    C[:, 1] = -distance * torch.cos(elevation) * torch.cos(azimuth)
    C[:, 2] = distance * torch.sin(elevation)
    # rotate coordinate system by theta is equal to rotating the model by theta
    azimuth = -azimuth
    elevation = -(torch.pi / 2 - elevation)

    zeros = torch.zeros_like(azimuth)
    ones = torch.ones_like(azimuth)
    r1 = torch.stack([torch.cos(azimuth), -torch.sin(azimuth), zeros], dim=-1)
    r2 = torch.stack([torch.sin(azimuth), torch.cos(azimuth), zeros], dim=-1)
    r3 = torch.stack([zeros, zeros, ones], dim=-1)
    Rz = torch.stack([r1, r2, r3], dim=-1)

    r1 = torch.stack([ones, zeros, zeros], dim=-1)
    r2 = torch.stack([zeros, torch.cos(elevation), -torch.sin(elevation)], dim=-1)
    r3 = torch.stack([zeros, torch.sin(elevation), torch.cos(elevation)], dim=-1)
    Rx = torch.stack([r1, r2, r3], dim=-1)
    R_rot = Rx @ Rz

    r4 = -R_rot @ C[..., None]
    c4 = torch.stack([zeros, zeros, zeros, ones], dim=-1)[..., None, :]

    R = torch.cat((R_rot, r4), dim=2)
    R = torch.cat((R, c4), dim=1)
    return R


def cal_rotation_matrix(theta, elev, azim, dis):
    dis[dis <= 1e-10] = 0.5
    return (
        rotation_theta(theta) @ get_transformation_matrix(azim, elev, dis)[:, 0:3, 0:3]
    )
