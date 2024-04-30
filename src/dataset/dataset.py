from copy import deepcopy
from dataclasses import dataclass

import torch
import numpy as np
from omegaconf import MISSING, DictConfig, ListConfig
from torch.utils.data import Dataset
from torchvision import transforms as tf
from torchvision.datasets import DTD

from src.dataset.transforms import Normalize, ToTensor


@dataclass
class DatasetCfg:
    name: str
    image_shape: ListConfig[int]
    augment: bool
    paths: DictConfig
    memory_budget: int
    viewport: int
    classes: ListConfig[str]
    etf_init: int
    n_tasks: int
    for_test: bool = MISSING
    root_path: str = MISSING
    seen_classes: ListConfig[str] = MISSING


class IncrementalDataset(Dataset):
    cfg: DatasetCfg
    transforms = tf.Compose([ToTensor(), Normalize()])
    memory: list = []
    file_list: list = []
    label_list: list = []
    exemplars_per_class: int = 0
    files_per_class: list

    def __init__(self, cfg: DatasetCfg, for_test=False):
        self.cfg = deepcopy(cfg)
        self.cfg.for_test = for_test
        self.cfg.root_path = (
            self.cfg.paths.test_path if for_test else self.cfg.paths.train_path
        )
        split = "test" if for_test else "train"
        self._dtd = DTD(self.cfg.paths.dtd_path, split=split)

    def next_task_classes(self):
        start_index = (
            0
            if (self.cfg.seen_classes is None or self.cfg.for_test)
            else len(self.cfg.seen_classes)
        )
        task_categories = self.cfg.classes[
            start_index : start_index + self.cfg.class_increment
        ]
        self.cfg.seen_classes = self.cfg.classes[
            : start_index + self.cfg.class_increment
        ]
        return task_categories

    def setup_task(self):
        raise NotImplementedError("Must implement setup_task")

    def __getitem__(self, item):
        raise NotImplementedError("Must implement __getitem__")

    def __len__(self):
        return self._len_task + self._len_replay

    @property
    def _len_task(self):
        return len(self.file_list)

    @property
    def _len_replay(self):
        return len(self.memory)

    @property
    def seen_classes(self):
        return self.cfg.seen_classes

    def build_replay_memory(self):
        exemplars_per_class = self.cfg.memory_budget // len(self.cfg.seen_classes)
        self._reduce_exemplars(exemplars_per_class)
        self.exemplars_per_class = exemplars_per_class
        # Add exemplars to memory
        self.memory += self._build_exemplars()

    def _reduce_exemplars(self, new_exemplars_per_class):
        if self._len_replay == 0:
            return
        old_classes = len(self.cfg.seen_classes) - len(self.cfg.classes)
        exemplars = []
        start = 0
        for i in range(old_classes):
            n_samples = []
            current_poses_azimuth = torch.tensor(
                [
                    self.memory[j]["pose"][2]
                    for j in torch.arange(start, start + self.exemplars_per_class, step=1)
                ]
            )
            torch.rad2deg_(current_poses_azimuth)
            value_range = (
                float(torch.min(current_poses_azimuth)),
                float(torch.max(current_poses_azimuth)),
            )
            histogram = torch.histogram(
                current_poses_azimuth, bins=10, range=value_range, density=False
            )
            histogram, bin_edges = (histogram.hist, histogram.bin_edges)

            insignificant_bins = torch.cat(
                (torch.tensor([False]), histogram < 1), dim=0
            )

            bin_edges = bin_edges[~insignificant_bins]
            bin_per_sample = (
                (current_poses_azimuth.unsqueeze(1) - bin_edges.unsqueeze(0))
                .cumsum(1)
                .argmax(1)
            )
            # TODO: Consider Distribution of poses when selecting number of exemplars in multinomial
            for j in range(len(bin_edges)):
                considered = torch.where(bin_per_sample == j)[0]
                if len(considered) > 1:
                    choose_2 = torch.ones_like(
                        considered, dtype=torch.float32
                    ).multinomial(
                        num_samples=min(
                            new_exemplars_per_class // (len(bin_edges) - 1), len(considered)
                        )
                    )
                    n_samples += [
                        self.memory[considered[choose_1] + start] for choose_1 in choose_2
                    ]
            if len(n_samples) < new_exemplars_per_class:
                left = new_exemplars_per_class - len(n_samples)
                indices = np.random.random_integers(start, start + self.exemplars_per_class - 1, left)
                n_samples += [self.memory[index] for index in indices]
            exemplars += n_samples
            start += self.exemplars_per_class
        return exemplars

    def _build_exemplars(self):
        exemplars = []
        start = 0
        for i, n_files in enumerate(self.files_per_class):
            n_samples = []
            current_poses_azimuth = torch.tensor(
                [
                    self[j]["pose"][2]
                    for j in torch.arange(start, start + n_files, step=1)
                ]
            )
            torch.rad2deg_(current_poses_azimuth)
            value_range = (
                float(torch.min(current_poses_azimuth)),
                float(torch.max(current_poses_azimuth)),
            )
            histogram = torch.histogram(
                current_poses_azimuth, bins=10, range=value_range, density=False
            )
            histogram, bin_edges = (histogram.hist, histogram.bin_edges)

            insignificant_bins = torch.cat(
                (torch.tensor([False]), histogram < 10), dim=0
            )

            bin_edges = bin_edges[~insignificant_bins]
            bin_per_sample = (
                (current_poses_azimuth.unsqueeze(1) - bin_edges.unsqueeze(0))
                .cumsum(1)
                .argmax(1)
            )

            # TODO: Consider Distribution of poses when selecting number of exemplars in multinomial
            for j in range(len(bin_edges)):
                considered = torch.where(bin_per_sample == j)[0]
                if len(considered) > 1:
                    choose_2 = torch.ones_like(
                        considered, dtype=torch.float32
                    ).multinomial(
                        num_samples=min(
                            self.exemplars_per_class // (len(bin_edges) - 1), len(considered)
                        )
                    )
                    n_samples += [
                        self[considered[choose_1] + start] for choose_1 in choose_2
                    ]
            if len(n_samples) < self.exemplars_per_class:
                left = self.exemplars_per_class - len(n_samples)
                indices = np.random.random_integers(start, start + n_files - 1, left)
                n_samples += [self[index] for index in indices]

            exemplars += n_samples
            start += n_files

        return exemplars