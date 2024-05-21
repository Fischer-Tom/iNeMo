from copy import deepcopy
from os.path import join

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
import wandb
from tqdm import tqdm

from src.lib.mesh_utils import load_off
from src.lib.renderer import RenderEngine
from src.models.memory import MeshMemory
from src.models.model import FeatureExtractor
from src.models.inemo import iNeMo
from src.models.trainer import BaseTrainer


def setup_training(cfg: DictConfig):
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
    dataset = ds(cfg.dataset)
    net = FeatureExtractor(cfg.model)

    mesh_memory = MeshMemory(cfg.model)
    mesh_memory.initialize_etf(cfg.dataset.etf_init)

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

    return (
        trainer,
        dataset,
    )


@hydra.main(version_base="1.3", config_path="../confs", config_name="main")
def main(cfg: DictConfig) -> None:
    cfg.checkpointing.log_dir = (
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    run = wandb.init(
        project=cfg.wandb.project,
        notes=cfg.wandb.notes,
        config=dict(cfg.nemo.train),
        mode=cfg.wandb.mode,
        dir=cfg.checkpointing.log_dir,
    )

    # Setup and run training
    trainer, dataset = setup_training(cfg)
    trainer.train_incremental(dataset, cfg)

    # Finish run
    OmegaConf.save(cfg, join(cfg.checkpointing.log_dir, f"config.yaml"))
    run.finish()


if __name__ == "__main__":
    main()
