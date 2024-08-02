import numpy as np
import torch
from einops import rearrange
from pytorch3d.renderer import camera_position_from_spherical_angles
from scipy.linalg import logm


def cal_pose_err(gt, pred):
    if (
        np.any(np.isnan(gt))
        or np.any(np.isnan(pred))
        or np.any(np.isinf(gt))
        or np.any(np.isinf(pred))
    ):
        return np.pi / 2
    l_ = logm(np.dot(np.transpose(pred), gt)) ** 2
    return (l_.sum() ** 0.5) / (2.0**0.5)


def loss_fun(obj_s: torch.Tensor, clu_s: torch.Tensor = None):
    if clu_s is None:
        return torch.ones(1, device=obj_s.device) - torch.mean(obj_s)
    return torch.ones(1, device=obj_s.device) - (
        torch.mean(torch.max(obj_s, clu_s)) - torch.mean(clu_s)
    )


def batch_loss_fun(
    obj_s: torch.Tensor, obj_m: torch.Tensor = None, clu_s: torch.Tensor = None
):
    obj_s = rearrange(obj_s, "b c h w -> b c (h w)")
    if clu_s is None:
        return torch.ones(obj_s.shape[0], device=obj_s.device) - torch.mean(
            obj_s, dim=1
        )
    max_cl = torch.max(clu_s, dim=1, keepdim=True).values
    j_s = torch.maximum(obj_s, max_cl)

    if obj_m is not None:
        obj_m = rearrange(obj_m, "b h w -> 1 b (h w)")
        zeroed = torch.where(obj_m, j_s, torch.zeros_like(j_s))
        j_s = torch.sum(zeroed, dim=2) / (torch.sum(obj_m, dim=2) + 1e-8)
    else:
        j_s = torch.mean(j_s, dim=2)
    return torch.ones((obj_s.shape[0], 1), device=obj_s.device) - (
        j_s - torch.mean(max_cl, dim=2)
    )


@torch.no_grad()
def get_init_pose(
    pre_rendered_poses,
    pre_rendered_maps,
    predicted_map,
    set_distance=5,
    clutter_score=None,
    device="cpu",
):
    obj_scores = torch.sum(pre_rendered_maps * predicted_map[:, None, ...], dim=2)
    loss = batch_loss_fun(obj_scores, clu_s=clutter_score)
    sample_idxs = torch.argmin(loss, dim=1)
    sampled_positions = torch.stack(
        [torch.tensor(pre_rendered_poses[i], device=device) for i in sample_idxs]
    )
    azim = sampled_positions[:, 0]
    elev = sampled_positions[:, 1]
    thetas = sampled_positions[:, 2]
    C = camera_position_from_spherical_angles(
        set_distance,
        elev,
        azim,
        degrees=False,
        device=device,
    )
    return C, thetas


def pre_render(get_samples, objects_texture, render_engine, xverts, xfaces):
    pre_rendered_maps = []

    class_labels = torch.arange(0, len(xverts)).cuda()

    for sample_ in get_samples:
        t = torch.ones(1).cuda() * sample_[2]
        cam_pos = camera_position_from_spherical_angles(
            5.0,
            sample_[1],
            sample_[0],
            degrees=False,
        ).cuda()

        _, projected_map = render_engine.render(
            objects_texture,
            cam_pos,
            t,
            class_labels,
            xverts,
            xfaces,
            objects_texture.device,
        )
        # TODO: Move to CPU here since there were incorrect outputs when stacking on gpu
        pre_rendered_maps.append(projected_map.cpu())
    pre_rendered_maps = torch.stack(pre_rendered_maps, dim=1)
    return pre_rendered_maps.cuda()
