from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.lib.config_utils import flatten_config
from src.lib.mesh_utils import (
    GlobalLocalConverter,
    batched_index_select,
    get_noise_pixel_index,
    keypoints_to_pixel_index,
)
from src.models.factory import resnetext


@dataclass
class ModelCfg:
    name: str
    extractor_dim: int
    mesh_dim: int
    local_size: int
    pretrain: bool
    downsample_rate: int
    num_clutter: int


class FeatureExtractor(nn.Module):
    cfg: ModelCfg
    l2_norm = lambda self, x, dim: F.normalize(x, p=2, dim=dim)

    def __init__(self, cfg: ModelCfg):
        super().__init__()
        model_cfg = deepcopy(cfg)
        del model_cfg.mesh
        self.cfg = flatten_config(model_cfg)
        self.net = torch.compile(resnetext(self.cfg), mode="reduce-overhead")
        self.out_layer = nn.Linear(
            self.cfg.extractor_dim,
            self.cfg.mesh_dim,
        )
        self.converter = GlobalLocalConverter([self.cfg.local_size] * 2)

    def forward(self, img, kp=None, mask=None):
        if kp is None:
            return self._forward_inference(img)
        return self._forward_train(img, kp, mask)

    @torch.no_grad()
    def _forward_inference(self, img):
        f = self.net(img)
        mesh_f = F.conv2d(
            f,
            self.out_layer.weight.unsqueeze(2).unsqueeze(3),
        )
        mesh_f = mesh_f + self.out_layer.bias.unsqueeze(-1).unsqueeze(-1)
        return self.l2_norm(mesh_f, 1)

    def _forward_train(self, img, keypoint_positions, obj_mask):
        b, _, h, w = img.shape  # n = 1
        f = self.net(img)

        kp = self.converter(f)
        keypoint_idx = keypoints_to_pixel_index(
            keypoints=keypoint_positions,
            img_size=(h // self.cfg.downsample_rate, w // self.cfg.downsample_rate),
        ).type(torch.long)

        obj_mask = obj_mask.view(obj_mask.shape[0], -1)
        keypoint_noise = get_noise_pixel_index(
            keypoint_idx,
            max_size=kp.shape[2],
            n_samples=self.cfg.num_clutter,
            obj_mask=obj_mask,
        )
        keypoint_all = torch.cat((keypoint_idx, keypoint_noise), dim=1)
        kp = torch.transpose(kp, 1, 2)
        kp = batched_index_select(kp, dim=1, inds=keypoint_all)
        kp = self.l2_norm(self.out_layer(kp), 2)
        kp = kp.view(b, -1, self.out_layer.weight.shape[0])

        return kp
