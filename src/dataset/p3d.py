from os.path import join

import numpy as np
from PIL import Image

from src.dataset.dataset import IncrementalDataset


class Pascal3DPlus(IncrementalDataset):
    def setup_task(self):
        task_classes = self.next_task_classes()
        self.label_list = []
        self.file_list = []
        self.files_per_class = []
        for cls in task_classes:
            deeper_dir = join(self.cfg.root_path, self.cfg.paths.list_folder, cls)
            new_files = [
                join(cls, l.strip())
                for l in open(
                    join(deeper_dir, "mesh01.txt"),
                ).readlines()
            ]
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
        img = self._add_background(img, annotation_file)
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

    def _add_background(self, img: Image, annotation_file: dict):
        bg_index = np.random.randint(low=0, high=len(self._dtd), size=1)[0]
        bg = self._dtd[bg_index][0]
        bg = bg.resize(img.size)
        img = np.array(img)
        padding = annotation_file["padding"]
        classification_size = self.cfg.image_shape
        object_height = (
            padding[0, 0],
            classification_size[0] - padding[0, 1],
        )
        object_width = (
            padding[1, 0],
            classification_size[1] - padding[1, 1],
        )

        bg_img = np.array(bg)
        bg_img[
            object_height[0] : object_height[1],
            object_width[0] : object_width[1],
            :,
        ] = img[
            object_height[0] : object_height[1],
            object_width[0] : object_width[1],
            :,
        ]
        img = Image.fromarray(bg_img)
        return img
