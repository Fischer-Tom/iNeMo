import math
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from omegaconf import DictConfig, ListConfig

from src.lib.config_utils import flatten_config


@dataclass
class MeshCfg:
    distance_thr: float
    num_clutter: int
    bank_size: int
    momentum: float
    mesh_dim: int
    max_n: int
    weight_noise: float
    n_list: ListConfig[int]


class MeshMemory(nn.Module):
    cfg: MeshCfg
    lru: int = 0
    ETF: torch.Tensor
    l2_norm = lambda self, x, dim: F.normalize(x, p=2, dim=dim)

    def __init__(
        self,
        cfg: DictConfig,
        n_classes: int = 0,
    ):
        super().__init__()
        model_cfg = deepcopy(cfg)
        del model_cfg.extractor
        self.cfg = MeshCfg(**flatten_config(model_cfg))

        memory = torch.randn(n_classes, self.cfg.mesh_dim, self.cfg.max_n)
        clutter_bank = torch.randn(
            self.cfg.mesh_dim, self.cfg.num_clutter * self.cfg.bank_size
        )

        # Register buffers
        self.register_buffer("memory", self.l2_norm(memory, 1))
        self.register_buffer("clutter_bank", self.l2_norm(clutter_bank, 0))

    @property
    def n_classes(self):
        return self.memory.shape[0]

    @torch.no_grad()
    def extend_memory(self, n_add=1):
        new_cubes = torch.randn(
            n_add,
            self.cfg.mesh_dim,
            self.cfg.max_n,
            device=self.memory.device,
            dtype=self.memory.dtype,
        )

        new_cubes = self.l2_norm(new_cubes, 1)
        self.memory = torch.cat((self.memory, new_cubes), dim=0)

    @torch.no_grad()
    def initialize_etf(self, K=1):
        if K <= 0:
            return
        memory = torch.randn(self.cfg.mesh_dim, self.cfg.max_n)
        memory = self.l2_norm(memory, 0)

        I = torch.eye(K)
        Ik = torch.ones(K, K)
        in_ = I - (1 / K) * Ik
        QR = torch.rand(self.cfg.mesh_dim, K)
        Q, _ = torch.linalg.qr(QR)
        E = math.sqrt(K / (K - 1)) * torch.matmul(Q, in_)
        E = self.l2_norm(E, 0)
        E = rearrange(E, "c b -> b c")
        class_memories = []
        for class_base in E:
            dist = torch.norm(class_base[:, None] - memory, p=2, dim=0)
            topk = dist.topk(self.cfg.max_n, largest=False)
            neighbors = memory[:, topk.indices]
            class_memories.append(neighbors)
        self.memory = torch.stack(class_memories, dim=0).cuda()
        self.ETF = E.cuda()

    def forward(
        self,
        features,
        visible,
        img_label,
        updateVertices=True,
        updateClutter=True,
    ):
        count_label = torch.bincount(img_label, minlength=self.n_classes)
        label_weight_onehot = self.label_to_onehot(img_label, count_label)

        vertices = features[:, : self.cfg.max_n, :]

        feature_similarity = einsum(vertices, self.memory, "b i v, c v j -> b i c j")

        clutter_similarity = einsum(vertices, self.clutter_bank, "b i v, v n -> b i n")
        similarity = torch.cat(
            (
                feature_similarity.reshape(vertices.shape[0], self.cfg.max_n, -1),
                clutter_similarity.reshape(vertices.shape[0], self.cfg.max_n, -1),
            ),
            dim=-1,
        )

        noise_similarity = torch.matmul(
            features[:, self.cfg.max_n :, :],
            torch.transpose(
                self.memory.permute(0, 2, 1).reshape(-1, self.memory.shape[1]), 0, 1
            ),
        )
        if updateVertices:
            self.updateMemory(
                vertices,
                visible,
                label_weight_onehot,
                count_label,
            )
        if updateClutter:
            self.updateClutter(features[:, self.cfg.max_n : :, :])
        return (
            similarity,
            noise_similarity,
        )

    def forward_kd(self, features, first_n):
        vertices = features[:, 0 : self.cfg.max_n, :]

        similarity = einsum(vertices, self.memory[:first_n], "b i v, c v j -> b i c j")

        return rearrange(similarity, "b i c j -> (b i) (c j)")

    @torch.no_grad()
    def updateMemory(self, vertices, visible, label_weight_onehot, count_label):
        visible_kp = torch.matmul(
            label_weight_onehot.transpose(0, 1),
            (vertices * visible.type(vertices.dtype).view(*visible.shape, 1)).view(
                vertices.shape[0], -1
            ),
        )

        visible_kp = visible_kp.view(
            visible_kp.shape[0], -1, vertices.shape[-1]
        ).permute(0, 2, 1)
        # handle 0 in get, case that no img of one class is in the batch
        tmp = (count_label == 0).nonzero(as_tuple=True)[0]
        for i in tmp:
            # copy memory to get
            visible_kp[i] = self.memory[i]

        self.memory = self.l2_norm(
            self.memory * self.cfg.momentum + visible_kp * (1 - self.cfg.momentum),
            1,
        )

    @torch.no_grad()
    def updateClutter(
        self,
        clutter,
    ):
        self.lru += 1
        self.lru = self.lru % (self.cfg.bank_size // clutter.shape[0])

        self.clutter_bank = self.l2_norm(
            torch.cat(
                [
                    self.clutter_bank[
                        :, : self.lru * self.cfg.num_clutter * clutter.shape[0]
                    ],
                    clutter.reshape(-1, clutter.shape[2]).permute(1, 0),
                    self.clutter_bank[
                        :, (self.lru + 1) * self.cfg.num_clutter * clutter.shape[0] :
                    ],
                ],
                dim=1,
            ),
            dim=0,
        )

    def class_contrastive(self, features, img_label):
        vertices = features[:, 0 : self.cfg.max_n, :]
        similarity = einsum(vertices, self.ETF, "b i v, c v -> b i c")
        similarity = rearrange(similarity, "b i c -> (b i) c")

        labels = repeat(img_label, "b -> b v", v=vertices.shape[1]).reshape(-1)
        return similarity, labels

    def label_to_onehot(self, img_label, count_label):
        ret = torch.zeros(img_label.shape[0], self.n_classes, device=img_label.device)
        ret = ret.scatter_(1, img_label.unsqueeze(1), 1.0)
        for i in range(self.n_classes):
            count = count_label[i]
            if count == 0:
                continue
            ret[:, i] /= count
        return ret
