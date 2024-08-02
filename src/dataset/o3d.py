from os.path import join
from os import listdir

import numpy as np
from PIL import Image

from src.dataset.dataset import IncrementalDataset


class ObjectNet3D(IncrementalDataset):
    def setup_task(self):
        task_classes = self.next_task_classes()
        self.label_list = []
        self.file_list = []
        self.files_per_class = []
        for cls in task_classes:
            deeper_dir = join(self.cfg.root_path, self.cfg.paths.image_folder, cls)
            new_files = [join(cls, l) for l in listdir(deeper_dir)]
            self.file_list = self.file_list + new_files
            self.files_per_class.append(len(new_files))
            self.label_list += [self.cfg.classes.index(cls)] * self.files_per_class[-1]

    def __getitem__(self, item):
        if item >= self._len_task:
            return self.memory[item - self._len_task]

        name_img = self.file_list[item]

        img = Image.open(
            join(self.cfg.root_path, self.cfg.paths.image_folder, name_img)
        )
        if img.mode != "RGB":
            img = img.convert("RGB")

        annotation_file = np.load(
            join(
                self.cfg.root_path,
                self.cfg.paths.anno_folder,
                name_img.split(".")[0] + ".npz",
            ),
            allow_pickle=True,
        )

        pose = np.array(
            [
                5,
                annotation_file["elevation"],
                annotation_file["azimuth"],
                annotation_file["theta"],
            ],
            dtype=np.float32,
        )
        label = self.label_list[item]
        sample = {
            "img": img,
            "pose": pose,
            "label": label,
        }
        if self.cfg.augment:
            sample = self.transforms(sample)
        return sample
