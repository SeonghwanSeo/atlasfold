import gc
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torchmetrics import MeanMetric, MetricCollection

from atlasfold.model.model_multimer import AtlasFoldMultimerConfig
from atlasfold.model.network.diffusion_head import SamplingConfig
from atlasfold.train.monomer.train_module import (
    TrainingModule as MonomerTrainingModule,
    to_dict,
)
from atlasfold.train.multimer import validation_metrics
from atlasfold.train.multimer.model_train import AtlasFoldForTrain
from atlasfold.train.utils.ema import ExponentialMovingAverage


class TrainingModule(MonomerTrainingModule):
    """Lightning module for multimer training and validation."""

    def __init__(self, config: DictConfig):
        pl.LightningModule.__init__(self)
        self.global_config: DictConfig = config
        self.config = config.train
        self.training_config = self.config.training
        self.validation_config = self.config.validation
        self.optimizer_config = self.config.optimizer
        self.loss_config = self.config.loss
        self.compile_config = self.config.compile

        self.save_hyperparameters(to_dict(self.global_config))

        self.train_trunk = self.training_config.train_trunk
        self.train_diffusion_head = self.training_config.train_diffusion_head
        self.train_confidence_head = self.training_config.train_confidence_head
        self.train_pde_head = self.training_config.train_pde_head
        self.train_pae_head = self.training_config.train_pae_head

        model_cfg = OmegaConf.to_object(
            OmegaConf.merge(AtlasFoldMultimerConfig, self.global_config.model)
        )
        self.model: AtlasFoldForTrain = AtlasFoldForTrain(model_cfg)

        self.freeze_submodules()
        self.setup_losses()
        self.setup_metrics()

        if self.compile_config.enabled:
            self.model.compile_train(
                mode=self.compile_config.mode, dynamic=self.compile_config.dynamic
            )

        self.ema: ExponentialMovingAverage = ExponentialMovingAverage(
            self.model,
            decay=self.optimizer_config.ema_decay,
            submodules_to_ignore=self.optimizer_config.ema_ignore_params,
            submodules_to_update=self.optimizer_config.ema_update_params,
        )
        self.stored_params: dict[str, torch.Tensor] | None = None

        rng = np.random.default_rng(seed=42)
        self.recycles_per_step: np.ndarray = rng.integers(
            0, self.training_config.num_recycles + 1, size=1000_000
        )
        self.last_lr_step: int = -1

    def setup_metrics(self):
        val_metrics = {}
        metric_names = [
            "complex/rmsd",
            "complex/lddt",
            "complex/lddt-ca",
            "chain/rmsd",
            "chain/lddt",
            "chain/lddt-ca",
            "interface/lddt",
            "interface/lddt-ca",
        ]
        for prefix in ["top", "avg", "rank"]:
            for name in metric_names:
                val_metrics[f"{prefix}/{name}"] = MeanMetric()
        self.val_metrics = MetricCollection(val_metrics, prefix="val/")

    def validation_step(
        self,
        batch: dict[str, dict[str, torch.Tensor]],
        batch_idx: int,
    ):
        feat, label = batch["feat"], batch["label"]
        val_config = self.validation_config
        seed = batch_idx
        try:
            devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                sample_out: dict[str, torch.Tensor] = self(
                    batch=feat,
                    label=None,
                    num_recycles=val_config.num_recycles,
                    mode="validation",
                )
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("**WARNING**: ran out of memory, skipping batch")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return
            raise e

        x_pred = sample_out["sample_coords"]  # [N, L, 14, 3]
        plddt = sample_out["plddt"]  # [N, L]
        mask = feat["seq_mask"].float()
        avg_plddt = (plddt * mask).sum(-1) / mask.sum(-1).clamp(min=1)
        rank_idx = int(torch.argmax(avg_plddt))

        metrics: dict[str, float] = validation_metrics.compute_validation_metric(
            x_pred, feat, label, rank_idx
        )

        val_metrics: MetricCollection = self.val_metrics
        for k, v in metrics.items():
            if k in val_metrics:
                val_metrics[k].update(v)

    def on_validation_epoch_end(self):
        torch.backends.cudnn.benchmark = True
        if not self.trainer.sanity_checking:
            avg_values = self.val_metrics.compute()
            for k, v in avg_values.items():
                self.log(
                    k,
                    v,
                    prog_bar=(k == "val/rank/complex/lddt-ca"),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
        self.val_metrics.reset()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        label: dict[str, torch.Tensor] | None,
        num_recycles: int,
        mode: str,
    ) -> Any:
        if mode == "train":
            assert label is not None, "Label cannot be None in training mode"
            training_config = self.training_config
            sampling_config = SamplingConfig(
                num_steps=training_config.num_mini_rollout_steps
            )
            return self.model.forward_train(
                batch=batch,
                label=label,
                num_recycles=num_recycles,
                diffusion_batch_size=training_config.diffusion_batch_size,
                train_trunk=self.train_trunk,
                train_diffusion_head=self.train_diffusion_head,
                train_confidence_head=self.train_confidence_head,
                sampling_config=sampling_config,
            )

        val_config = self.validation_config
        sampling_config = SamplingConfig(num_steps=val_config.num_steps)
        return self.model.inference(
            batch,
            num_recycles=num_recycles,
            num_samples=val_config.num_diffusion_samples,
            sampling_config=sampling_config,
            return_representations=False,
        )
