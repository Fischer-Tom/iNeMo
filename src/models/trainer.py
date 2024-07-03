import torch
from einops import rearrange
from os.path import join
from tqdm import tqdm
from torch.utils.data import default_collate
import wandb
from torch import tensor as Tensor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from omegaconf import DictConfig
    from torch.nn import Module


class BaseTrainer:
    current_pad_index: Tensor
    n_classes: int = None
    current_task_id: int = 0

    def __init__(
        self,
        model: "Module",
        criterion: "Module",
        cfg: "DictConfig",
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = None
        self.cfg = cfg
        self.n_tasks = (
            1
            + (len(cfg.dataset.classes) - cfg.dataset.init_task)
            // cfg.dataset.task_incr
        )

    def _lr_update(self, epoch: int, cfg: "DictConfig") -> None:
        if (epoch - 1) % cfg.update_lr_epoch_n == 0:
            lr = cfg.lr * cfg.update_lr_
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

    def train_incremental(self, dataset, cfg: "DictConfig") -> None:
        for n in range(self.n_tasks):
            train_loader, test_loader = self._setup_task(dataset, cfg)
            if n > 0:
                self.model.module.set_old_net()
            n_epochs = (
                cfg.nemo.incremental.increment_epoch
                if n == 0
                else cfg.nemo.incremental.subsequent_increment_epoch
            )
            with tqdm(range(n_epochs), unit="batch") as tepochs:
                for epoch in tepochs:
                    self._lr_update(epoch, cfg=cfg.optimizer)
                    loss = self._train_epoch(train_loader)
                    tepochs.set_description(
                        f"Task: {n}, Epoch={epoch}, Loss={loss:.4f}"
                    )

            # Fill Replay Memory
            dataset.build_replay_memory()

            # Even out Backround Model with Replay Memory
            self.model.module.fill_background_model(dataset.memory)

            # Validate
            accuracy, pose_acc_pi6, pose_acc_pi18 = self.validate(
                test_loader, run_pe=False
            )

            print(
                f"Validation Accuracy: {accuracy}, "
                f"Validation Pose Error (pi/6): {pose_acc_pi6}, "
                f"Validation Pose Error (pi/18): {pose_acc_pi18}"
            )
            wandb.log(
                {
                    "Validation Accuracy": accuracy,
                    "Validation Pose Error (pi/6)": pose_acc_pi6,
                    "Validation Pose Error (pi/18)": pose_acc_pi18,
                },
                step=self.current_task_id,
            )

            # Save Model State
            self.model.module.save_state(
                join(self.cfg.checkpointing.log_dir, f"model_task_{n}.pt")
            )

    def _setup_task(self, dataset, cfg: "DictConfig"):
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
        self.optimizer = torch.optim.Adam(
            self.model.module.net.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
        )

        # Update trainer for the next task
        self.model.module.set_current_pad_index(len(dataset.cfg.seen_classes))

        return train_loader, val_loader

    def _train_epoch(self, loader: "DataLoader") -> float:
        self.model.train()
        running_loss = 0.0
        for i, sample in enumerate(loader):
            loss, loss_pos_reg, loss_kd = self.model(sample)

            # Update Network
            combined_loss = loss + loss_pos_reg + loss_kd
            self.optimizer.zero_grad()
            combined_loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
        return running_loss / len(loader)

    def validate(
        self, loader: "DataLoader", run_pe=False
    ) -> tuple[float, float, float]:
        self.model.eval()
        class_preds, class_gds, pose_errors = [Tensor([], dtype=torch.float32)] * 3

        compare_bank = rearrange(
            self.model.module.mesh_memory.memory[: self.n_classes], "b c v -> b v c"
        )
        if run_pe:
            pre_rendered_maps, pre_rendered_poses = self.model.module._pre_render(
                compare_bank
            )
        else:
            pre_rendered_maps, pre_rendered_poses = None, None
        with tqdm(loader, unit="batch") as tepoch:
            for i, sample in enumerate(tepoch):
                cls_pred, label, pose_error = self.model.module.inference(
                    sample,
                    compare_bank,
                    pre_rendered_maps,
                    pre_rendered_poses,
                    run_pe=run_pe,
                )
                class_gds = torch.cat((class_gds, label.cpu()), dim=0)
                class_preds = torch.cat((class_preds, cls_pred.cpu()), dim=0)
                pose_errors = torch.cat((pose_errors, pose_error.cpu()), dim=0)

                pose_acc_pi6 = torch.mean((pose_errors < torch.pi / 6).float()).item()
                pose_acc_pi18 = torch.mean((pose_errors < torch.pi / 18).float()).item()

                accuracy = torch.mean((class_gds == class_preds).float()).item()
                tepoch.set_postfix(
                    mode="val",
                    acc=accuracy,
                    pi6=pose_acc_pi6,
                    pi18=pose_acc_pi18,
                )

        return accuracy, pose_acc_pi6, pose_acc_pi18
