import torch
import torch.nn as nn
from torchvision.models import resnet50

from src.models.resnet import ResNetExt


def resnetext(cfg):
    if cfg.pretrain:
        net = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
    else:
        net = resnet50(weights=None)
    return ResNetExt(net, cfg=cfg)
