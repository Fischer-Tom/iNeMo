from copy import deepcopy
from os.path import join

import argparse
import hydra
import torch
from omegaconf import DictConfig

from src.lib.mesh_utils import load_off
from src.lib.renderer import RenderEngine
from src.models.memory import MeshMemory
from src.models.model import FeatureExtractor
from src.models.inemo import iNeMo
from src.models.trainer import BaseTrainer


def setup_training(model_path: str, cfg: DictConfig):
    if cfg.dataset.name == "Pascal3D":
        from src.dataset.p3d import Pascal3DPlus as ds
    elif cfg.dataset.name == "ObjectNet3D":
        from src.dataset.o3d import ObjectNet3D as ds
    else:
        raise NotImplementedError("Dataset not implemented")
    # Setup Mesh Related Variables
    xverts, xfaces = [], []
    for cls in cfg.dataset.classes:
        mesh_path = join(cfg.dataset.paths.mesh_path, cls, "01.off")
        xvert, xface = load_off(mesh_path, to_torch=True)
        xverts.append(xvert)
        xfaces.append(xface)

    all_classes = len(cfg.dataset.classes)

    cfg.model.mesh.max_n = max([vert.shape[0] for vert in xverts])

    pad_indexes = []
    for i in range(all_classes):
        num = cfg.model.mesh.max_n - xverts[i].shape[0]
        pad_index = [
            cfg.model.mesh.max_n * i + xverts[i].shape[0] + j for j in range(num)
        ]
        pad_indexes.append(pad_index)

    # Setup Dataset and Dataloader related Variables
    test_dataset = ds(cfg=cfg.dataset, for_test=True)
    test_dataset.cfg.seen_classes = cfg.dataset.classes
    test_dataset.setup_task()
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.data_loader.test.batch_size,
        shuffle=False,
        num_workers=cfg.data_loader.test.num_workers,
        pin_memory=True,
    )

    net = FeatureExtractor(cfg.model)

    mesh_memory = MeshMemory(cfg.model)

    # Setup training criterion
    criterion = torch.nn.CrossEntropyLoss(reduction="mean").cuda()

    # Setup Rendering for the AnnotationGenerator
    render_engine = RenderEngine(cfg)

    # Static Variables during training
    n_gpus = torch.cuda.device_count()
    trainer_vars = {
        "pad_index": pad_indexes,
        "n_classes": all_classes,
        "xverts": xverts,
        "xfaces": xfaces,
        "weight_noise": cfg.model.weight_noise,
    }

    # Setup iNeMo Model
    model = iNeMo(
        net=net,
        memory=mesh_memory,
        renderer=render_engine,
        criterion=criterion,
        train_cfg=trainer_vars,
        param_cfg=cfg.nemo.train,
        inf_cfg=cfg.nemo.inference,
    )
    model.load_state(model_path)

    if n_gpus > 1:
        model = torch.nn.DataParallel(
            model.cuda(), device_ids=[i for i in range(n_gpus - 1)]
        )
    else:
        model = torch.nn.DataParallel(model.cuda())

    # Setup Trainer
    trainer = BaseTrainer(
        model=model,
        criterion=criterion,
        cfg=cfg,
    )

    return (trainer, test_loader)


@hydra.main(version_base="1.3", config_path="../confs", config_name="main")
def main(cfg: DictConfig) -> None:

    parser = argparse.ArgumentParser(description="Evaluate iNeMo Model")
    parser.add_argument(
        "--ckpt_path",
        default="/home/fischer/remote/iNeMo/outputs/2024-05-19/21-20-37/model_task_3.pt",
        type=str,
    )
    args = parser.parse_args()

    # Setup and run training
    trainer, test_loader = setup_training(args.ckpt_path, cfg)
    trainer.validate(test_loader, run_pe=True)


if __name__ == "__main__":

    main()
