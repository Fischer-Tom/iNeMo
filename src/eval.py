import hydra
import torch
from omegaconf import DictConfig
import wandb


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
    n_gpus = torch.cuda.device_count()
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(
            net.cuda(), device_ids=[i for i in range(n_gpus - 1)]
        )
    else:
        net = torch.nn.DataParallel(net.cuda())

    mesh_memory = MeshMemory(cfg.model).to("cuda")
    mesh_memory.initialize_etf(cfg.dataset.etf_init)

    # Setup training criterion
    criterion = torch.nn.CrossEntropyLoss(reduction="mean").cuda()

    # Setup Rendering for the AnnotationGenerator
    render_engine = RenderEngine(cfg)

    # Static Variables during training
    trainer_vars = {
        "pad_index": pad_indexes,
        "n_gpus": n_gpus,
        "n_classes": all_classes,
        "xverts": xverts,
        "xfaces": xfaces,
        "weight_noise": cfg.model.weight_noise,
    }

    # Setup Trainer
    trainer = BaseTrainer(
        net,
        mesh_memory,
        render_engine,
        criterion,
        train_cfg=trainer_vars,
        param_cfg=cfg.nemo.train,
        inf_cfg=cfg.nemo.inference,
    )

    return (
        trainer,
        dataset,
    )


def setup_task(trainer, dataset, cfg: DictConfig):
    # Setup Dataset Objects
    dataset.setup_task()
    if cfg.dataset.name == "Pascal3D":
        from src.dataset.p3d import Pascal3DPlus as ds
    elif cfg.dataset.name == "ObjectNet3D":
        from src.dataset.o3d import ObjectNet3D as ds
    else:
        raise NotImplementedError("Dataset not implemented")
    val_dataset = ds(cfg=cfg.dataset, for_test=True)
    val_dataset.cfg.seen_classes = dataset.cfg.seen_classes
    val_dataset.setup_task()

    # Setup Dataloaders
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.data_loader.train.batch_size,
        shuffle=True,
        num_workers=cfg.data_loader.train.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.data_loader.test.batch_size,
        shuffle=False,
        num_workers=cfg.data_loader.test.num_workers,
        pin_memory=True,
    )

    # Setup Optimizer
    optim = torch.optim.Adam(
        trainer.net.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    # Update trainer for the next task
    trainer.set_optimizer(optim)
    trainer.set_current_pad_index(len(dataset.cfg.seen_classes))

    return train_loader, val_loader, optim


def train(trainer, dataset, cfg):
    for n in range(dataset.cfg.n_tasks):
        train_loader, test_loader, optim = setup_task(trainer, dataset, cfg)
        if n > 0:
            trainer.set_old_net()
        n_epochs = (
            cfg.nemo.incremental.increment_epoch
            if n == 0
            else cfg.nemo.incremental.subsequent_increment_epoch
        )
        for epoch in range(n_epochs):
            trainer.lr_update(epoch, cfg=cfg.optimizer)
            trainer.train_epoch(train_loader)

        # Fill Replay Memory
        dataset.build_replay_memory()

        # Even out Backround Model with Replay Memory
        trainer.fill_background_model(dataset.memory)

        # Validate
        trainer.validate(test_loader, run_pe=False)

        # Save Model and Config
        # TODO: Add Model Saving


@hydra.main(version_base="1.3", config_path="../confs", config_name="main")
def main(cfg: DictConfig) -> None:
    cfg.checkpointing.log_dir = (
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    train_vars = setup_training(cfg)
    train(*train_vars, cfg)


if __name__ == "__main__":
    main()
