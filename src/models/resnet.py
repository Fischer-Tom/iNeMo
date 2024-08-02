import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class DoubleConv(nn.Module):
    """(conv - bn - relu) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, padding=(1, 1)):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.doubleconv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=padding[0]),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=padding[1]),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, X):
        X = self.doubleconv(X)
        return X


class Up(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.doubleconv = DoubleConv(in_channels, out_channels, mid_channels)

    def forward(self, X1, X2):
        X1 = self.up(X1)
        diffY = torch.tensor([X2.size()[2] - X1.size()[2]])
        diffX = torch.tensor([X2.size()[3] - X1.size()[3]])
        # just incase:
        X1 = F.pad(X1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        X = torch.cat([X2, X1], dim=1)
        X = self.doubleconv(X)
        return X


class ResNetExt(nn.Module):
    def __init__(self, net, cfg: DictConfig):
        super().__init__()

        self.extractor = nn.Sequential()
        self.extractor.add_module("0", net.conv1)
        self.extractor.add_module("1", net.bn1)
        self.extractor.add_module("2", net.relu)
        self.extractor.add_module("3", net.maxpool)
        self.extractor.add_module("4", net.layer1)
        self.extractor.add_module("5", net.layer2)
        self.extractor1 = net.layer3
        self.extractor2 = net.layer4

        self.upsample0 = DoubleConv(2048, 1024)
        self.upsample1 = Up(1024 + 1024, 1024, 512)
        self.upsample2 = Up(512 + 512, 512, cfg.extractor_dim)

    def forward(self, x):
        x1 = self.extractor(x)
        x2 = self.extractor1(x1)
        x3 = self.extractor2(x2)
        features = self.upsample2(self.upsample1(self.upsample0(x3), x2), x1)

        return features

    def bottleneck(self, x):
        return F.adaptive_avg_pool2d(
            self.extractor2(self.extractor1(self.extractor(x))), (1, 1)
        ).flatten(1)
