import torch
from torchvision import transforms as tf


class ToTensor:
    def __init__(self):
        self.trans = tf.ToTensor()

    def __call__(self, sample):
        sample["img"] = self.trans(sample["img"])
        if "iskpvisible" in sample and not type(sample["iskpvisible"]) == torch.Tensor:
            sample["iskpvisible"] = torch.Tensor(sample["iskpvisible"])
        if "kp" in sample and not type(sample["kp"]) == torch.Tensor:
            sample["kp"] = torch.Tensor(sample["kp"])
        return sample


class Normalize:
    def __init__(self):
        self.trans = tf.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __call__(self, sample):
        sample["img"] = self.trans(sample["img"])
        return sample


class ColorJitter:
    def __init__(self):
        self.trans = tf.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.4,
            hue=0,
        )

    def __call__(self, sample):
        sample["img"] = self.trans(sample["img"])

        return sample
