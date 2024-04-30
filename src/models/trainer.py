import itertools
import math
from dataclasses import dataclass
from copy import deepcopy

import numpy as np
import torch
from einops import einsum, rearrange
from tqdm import tqdm
from torch.utils.data import default_collate
import torch.nn.functional as F
import wandb

from src.lib.inference_utils import (cal_pose_err, get_init_pose, loss_fun,
                                     pre_render)
from src.lib.mesh_utils import (cal_rotation_matrix,
                                camera_position_to_spherical_angle,
                                get_cube_proj)
from src.lib.renderer import MeshInterpolateModule


@dataclass
class TrainerCfg:
    pad_index: torch.tensor
    xverts: list[torch.tensor]
    xfaces: list[torch.tensor]
    n_classes: int
    weight_noise: float
    eps: float = 1e-5
    noise_loss_weight: float = 0.1
    n_gpus: int = 1


@dataclass
class ValidationCfg:
    inf_epochs: int = 30
    inf_bs: int = 8
    inf_lr: float = 5e-2
    inf_adam_beta_0: float = 0.4
    inf_adam_beta_1: float = 0.6
    inf_adam_eps: float = 1e-8
    inf_adam_weight_decay: float = 0.0

@dataclass
class ParamCfg:
    pos_reg: float
    kd_reg: float
    spatial_reg:float
    kappa_main: float
    kappa_kd: float
    kappa_pos: float
    freeze_first_n_epochs: int


class BaseTrainer:
    cfg: TrainerCfg
    inf_cfg: ValidationCfg
    current_pad_index: torch.tensor
    n_classes: int = None
    current_task_id: int = 0

    def __init__(self, net, memory, renderer, criterion, train_cfg, param_cfg, inf_cfg):
        self.net = net
        self.old_net = None
        self.mesh_memory = memory
        self.renderer = renderer
        self.criterion = criterion
        self.optimizer = None
        self.cfg = TrainerCfg(**train_cfg)
        self.param_cfg = ParamCfg(**param_cfg)
        self.inf_cfg = ValidationCfg(**inf_cfg)

    def set_optimizer(self, optim):
        self.optimizer = optim

    def set_current_pad_index(self, n_seen_classes):
        self.current_task_id += 1
        self.current_pad_index = torch.tensor(
            [
                vertex
                for vertices in self.cfg.pad_index[:n_seen_classes]
                for vertex in vertices
            ],
            dtype=torch.long,
        )
        self.n_prev_classes = 0 if self.n_classes is None else n_seen_classes - self.n_classes
        self.n_classes = n_seen_classes

    def set_old_net(self):
        if self.param_cfg.kd_reg > 0:
            self.old_net = deepcopy(self.net)
            self.old_net.requires_grad_(False)
            self.old_net.train()

    def _to_device(self, sample, device="cuda"):
        # TODO: Account for multi-gpu
        img, img_label, pose = (
            sample["img"].to(device),
            sample["label"].to(device),
            sample["pose"].to(device),
        )
        idx = (
            self.mesh_memory.cfg.max_n * img_label[:, None]
            + torch.arange(0, self.mesh_memory.cfg.max_n, device=device)[None, :]
        )
        return img, img_label, idx, pose

    def train_epoch(self, loader, epoch=0):
        self.net.train()

        with tqdm(loader, unit="batch") as tepoch:
            for i, sample in enumerate(tepoch):
                img, label, idx, pose = self._to_device(sample)

                loss, loss_pos_reg, loss_kd = self._train_step(img, label, idx, pose)

                tepoch.set_postfix( epoch=epoch,
                                    nemo_loss=loss,
                                    kd_loss=loss_kd,
                                    pos_reg_loss=loss_pos_reg)

    def validate(self, loader, run_pe=False):
        self.net.eval()
        class_preds, class_gds, pose_errors = [
                                                  torch.tensor([], dtype=torch.float32)
                                              ] * 3

        compare_bank = rearrange(
            self.mesh_memory.memory[: self.n_classes], "b c v -> b v c"
        )
        if run_pe:
            pre_rendered_maps, pre_rendered_poses = self._pre_render(compare_bank)
        with tqdm(loader, unit="batch") as tepoch:
            for i, sample in enumerate(tepoch):
                img, label, idx, pose = self._to_device(sample)
                pred_features = self.net(img)
                clutter_score = rearrange(
                    einsum(
                        self.mesh_memory.clutter_bank,
                        pred_features,
                        "c k, b c h w -> b k h w",
                    ),
                    "b k h w -> b k (h w)",
                )
                cls_pred = self._classify(compare_bank, pred_features, clutter_score)
                class_gds = torch.cat((class_gds, label.cpu()), dim=0)
                class_preds = torch.cat((class_preds, cls_pred), dim=0)

                if run_pe:
                    C, theta = get_init_pose(
                        pre_rendered_poses,
                        pre_rendered_maps[cls_pred],
                        pred_features,
                        clutter_score=clutter_score,
                        device=img.device,
                    )
                    pose_error = self._pose_estimation(
                        pred_features, compare_bank, cls_pred, C, theta, pose
                    )
                    pose_errors = torch.cat((pose_errors, pose_error), dim=0)

                    pose_acc_pi6 = torch.mean((pose_errors < torch.pi / 6).float()).item()
                    pose_acc_pi18 = torch.mean((pose_errors < torch.pi / 18).float()).item()
                else:
                    pose_acc_pi6 = 0.0
                    pose_acc_pi18 = 0.0
                accuracy = torch.mean((class_gds == class_preds).float()).item()

                tepoch.set_postfix(
                    mode="val",
                    acc=accuracy,
                    pi6=pose_acc_pi6,
                    pi18=pose_acc_pi18,
                )
        wandb.log({"Validation Accuracy": accuracy, "Validation Pose Error (pi/6)": pose_acc_pi6,
                   "Validation Pose Error (pi/18)": pose_acc_pi18}, step=self.current_task_id)
    @torch.no_grad()
    def fill_background_model(self, replay_memory):
        bank_size = self.mesh_memory.cfg.bank_size
        n_iter = max(0, bank_size // len(replay_memory))
        for _ in range(n_iter):
            for i, sample in enumerate(replay_memory):
                sample = default_collate([sample])
                img, label, idx, pose = self._to_device(sample)
                keypoint, kpvis, mask, projection = self._annotate(pose, label)
                features, _ = self.net(img, kp=keypoint, mask=mask)

                _, _ = self.mesh_memory.forward(features, kpvis, label, updateVertices=False)
    def _train_step(self, img, label, idx, pose):
        keypoint, kpvis, mask, projection = self._annotate(pose, label)
        features, fmaps = self.net(img, kp=keypoint, mask=1-mask)

        similarity, noise_similarity = self.mesh_memory.forward(features, kpvis, label)
        neighborhood_mask = self._remove_neighbors(keypoint, label)
        masked_similarity = similarity - neighborhood_mask

        masked_similarity = rearrange(masked_similarity, "b c v -> (b c) v")
        kpvis = rearrange(kpvis, "b c -> (b c)").type(torch.bool)
        idx = rearrange(idx, "b c -> (b c)")

        nemo_loss = self.criterion(self.param_cfg.kappa_main * masked_similarity[kpvis, :], idx[kpvis])
        noise_loss = torch.mean(noise_similarity) * self.cfg.noise_loss_weight

        loss_pos_reg = torch.tensor(0.0, device=img.device)
        loss_kd = torch.tensor(0.0, device=img.device)

        if self.param_cfg.pos_reg > 0.0 and self.mesh_memory.ETF is not None:
            class_reg, reg_labels = self.mesh_memory.class_contrastive(features, label)
            loss_pos_reg = self.criterion(
                class_reg[kpvis, :],
                reg_labels[kpvis],
            )
        if self.old_net is not None and self.param_cfg.kd_reg >= 0.0:
            # pod net paper sets the weight for the pod spatial loss as 3 And the flat kd loss as 1.
            loss_kd = self._kd_loss(img, keypoint, mask, kpvis, features, fmaps)

        loss = nemo_loss + noise_loss
        combined_loss = loss + self.param_cfg.pos_reg * loss_pos_reg + self.param_cfg.kd_reg * loss_kd
        self.optimizer.zero_grad()
        combined_loss.backward()
        self.optimizer.step()
        return loss.item(), loss_pos_reg.item(), loss_kd.item()

    def pod_spatial_loss(self,old_fmaps,fmaps,normalize=True):
        """
        a, b: list of [bs, c, w, h]
        """
        loss = torch.tensor(0.0).to(fmaps[0].device)
        for i, (a, b) in enumerate(zip(old_fmaps, fmaps)):
            assert a.shape == b.shape, "Shape error"

            # a, b pair could look like to following: bs,16,32,32

            a = torch.pow(a, 2)
            b = torch.pow(b, 2)

            a_h = a.sum(dim=3).view(a.shape[0], -1)  # [bs, c*w]
            b_h = b.sum(dim=3).view(b.shape[0], -1)  # [bs, c*w]
            a_w = a.sum(dim=2).view(a.shape[0], -1)  # [bs, c*h]
            b_w = b.sum(dim=2).view(b.shape[0], -1)  # [bs, c*h]

            a = torch.cat([a_h, a_w], dim=-1)
            b = torch.cat([b_h, b_w], dim=-1)

            if normalize:
                a = F.normalize(a, dim=1, p=2)
                b = F.normalize(b, dim=1, p=2)

            layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
            loss += layer_loss

        return loss / len(fmaps)

    def _annotate(self, pose, label):
        annotations = self.renderer.get_annotations(
            rearrange(self.mesh_memory.memory, "b c v -> b v c"),
            pose,
            label,
            self.cfg.xverts,
            self.cfg.xfaces,
            label.device,
        )
        return annotations

    def _kd_loss(self, img, keypoints, mask, kpvis, new_feats, new_fmaps):
        old_feats, old_fmaps = self.old_net(img, kp=keypoints, mask=1-mask)
        old_similarity = self.mesh_memory.forward_kd(old_feats, self.n_prev_classes)
        new_similarity = self.mesh_memory.forward_kd(new_feats, self.n_prev_classes)

        # breakpoint()

        # Original KD loss
        # soft_targets = F.log_softmax(
        #     self.param_cfg.kd_reg * old_similarity[kpvis, :], dim=-1
        # )
        # soft_probs = F.log_softmax(
        #     self.param_cfg.kd_reg * new_similarity[kpvis, :], dim=-1
        # )
        # kd_loss = F.kl_div(soft_probs, soft_targets, reduction="batchmean", log_target=True)


        # Apply flat loss here 
        flat_loss = F.cosine_embedding_loss(new_similarity[kpvis, :], old_similarity[kpvis, :], torch.ones(kpvis.sum()).to(kpvis.device))

        # Apply this + the spatial kd loss
        spatial_loss = self.param_cfg.spatial_reg * self.pod_spatial_loss(old_fmaps[:-1], new_fmaps[:-1])

        kd_loss = flat_loss + spatial_loss

        return kd_loss
    
    @torch.no_grad()
    def _remove_neighbors(self, keypoints, img_label):
        zeros = torch.zeros(
            keypoints.shape[0],
            self.mesh_memory.cfg.max_n,
            self.mesh_memory.cfg.max_n * self.mesh_memory.memory.shape[0],
            device=keypoints.device,
        )
        distance = torch.sum(
            (torch.unsqueeze(keypoints, dim=1) - torch.unsqueeze(keypoints, dim=2)).pow(
                2,
            ),
            dim=3,
        ).pow(0.5)
        if self.mesh_memory.cfg.num_clutter == 0:
            return (
                (distance <= self.mesh_memory.cfg.distance_thr).type(torch.float32)
                - torch.eye(keypoints.shape[1], device=distance.device).unsqueeze(dim=0)
            ) * self.cfg.eps
        else:
            tem = (distance <= self.mesh_memory.cfg.distance_thr).type(
                torch.float32
            ) - torch.eye(keypoints.shape[1], device=distance.device).unsqueeze(dim=0)

            for i in range(tem.shape[0]):
                zeros[
                    i,
                    :,
                    img_label[i] * tem.shape[1] : (img_label[i] + 1) * tem.shape[1],
                ] = (
                    tem[i] * self.cfg.eps
                )
            zeros[:, :, self.current_pad_index] = self.cfg.eps

            return torch.cat(
                [
                    zeros,
                    -torch.ones(
                        keypoints.shape[0:2]
                        + (
                            self.mesh_memory.cfg.num_clutter
                            * self.mesh_memory.cfg.bank_size,
                        ),
                        dtype=torch.float32,
                        device=zeros.device,
                    )
                    * math.log(self.cfg.weight_noise),
                ],
                dim=2,
            )



    @torch.no_grad()
    def _classify(self, compare_bank, pred_features, clutter_score):
        flat_features = rearrange(pred_features, "b c h w -> b c (h w)")
        score_per_pixel = einsum(compare_bank, flat_features, "n v c, b c p -> b n v p")
        max_cl = torch.max(clutter_score, dim=1, keepdim=True).values
        max_score = torch.max(score_per_pixel, dim=2).values
        max_2 = torch.topk(max_score, 2, dim=1).values
        diff = 1 - (max_2[..., 0:1, :] - max_2[..., 1:2, :])
        score = torch.max((max_score - max_cl - diff), dim=2).values

        return torch.argmax(score, dim=1).flatten().cpu().detach()

    @torch.no_grad()
    def _pre_render(self, compare_bank):
        azum_s = np.linspace(0, np.pi * 2, 12, endpoint=False, dtype=np.float32)
        elev_s = np.linspace(-np.pi / 6, np.pi / 3, 4, dtype=np.float32)
        theta_s = np.linspace(-np.pi / 6, np.pi / 6, 3, dtype=np.float32)
        c_poses = list(itertools.product(azum_s, elev_s, theta_s))
        pre_rendered_maps = pre_render(
            c_poses, compare_bank, self.renderer, self.cfg.xverts, self.cfg.xfaces
        )
        return pre_rendered_maps, c_poses

    def _pose_estimation(
        self, pred_features, compare_bank, cls_pred, C, theta, pose_gd
    ):
        errors = []
        for b, pred in enumerate(pred_features):
            C_i = torch.nn.Parameter(C[b : b + 1], requires_grad=True)
            theta_i = torch.nn.Parameter(theta[b : b + 1], requires_grad=True)

            optim = torch.optim.Adam(
                params=[C_i, theta_i],
                lr=self.inf_cfg.inf_lr,
                betas=(self.inf_cfg.inf_adam_beta_0, self.inf_cfg.inf_adam_beta_1),
            )
            xvert, xface = self.cfg.xverts[cls_pred[b]], self.cfg.xfaces[cls_pred[b]]
            inter_module = MeshInterpolateModule(
                xvert.cuda(),
                xface.cuda(),
                compare_bank[cls_pred[b]],
                self.renderer.rasterizer,
                # post_process=center_crop_fun(map_shape, (render_image_size,) * 2),
            )
            inter_module = inter_module.cuda()

            for epoch in range(self.inf_cfg.inf_epochs):
                loss = self._regress_pose(pred, C_i, theta_i, inter_module)

                optim.zero_grad()
                loss.backward()
                optim.step()
            (
                distance_pred,
                elevation_pred,
                azimuth_pred,
            ) = camera_position_to_spherical_angle(C_i.clone().detach())
            pred_matrix = cal_rotation_matrix(
                theta_i.clone().detach(), elevation_pred, azimuth_pred, distance_pred
            )
            gd_matrix = cal_rotation_matrix(
                pose_gd[:, 3], pose_gd[:, 1], pose_gd[:, 2], pose_gd[:, 0]
            )
            error = cal_pose_err(
                np.array(gd_matrix[0].cpu(), dtype=np.float64),
                np.array(pred_matrix[0].cpu(), dtype=np.float64),
            )
            errors.append(error)
        return torch.tensor(errors)

    def _regress_pose(self, predicted_features, C, theta, render_module):
        projected_map, obj_height, obj_width = get_cube_proj(C, theta, render_module)
        projected_map = projected_map[
            ...,
            obj_height[0] : obj_height[1],
            obj_width[0] : obj_width[1],
        ].squeeze()
        masked_features = predicted_features[
            ...,
            obj_height[0] : obj_height[1],
            obj_width[0] : obj_width[1],
        ]
        object_score = torch.sum(projected_map * masked_features, dim=0)

        clutter_score = einsum(
            self.mesh_memory.clutter_bank, masked_features, "c v, c h w -> v h w"
        )
        clutter_score = torch.max(clutter_score, dim=0).values
        return loss_fun(object_score, clutter_score)
