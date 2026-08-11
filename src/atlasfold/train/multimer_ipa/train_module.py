"""Lightning training module for AtlasFold multimer IPA."""

from __future__ import annotations

import gc
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torchmetrics import MeanMetric, MetricCollection

from atlasfold.model.model_multimer_ipa import AtlasFoldMultimerIPAConfig
from atlasfold.train.losses import structure_ipa
from atlasfold.train.monomer_ipa import train_module as monomer_train_module
from atlasfold.train.multimer import train_alignment, validation_metrics
from atlasfold.train.multimer_ipa.model_train import AtlasFoldMultimerIPAForTrain
from atlasfold.train.utils.ema import ExponentialMovingAverage


def to_dict(config: DictConfig) -> dict:
    """Convert a DictConfig to a standard Python dictionary."""
    return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]


class TrainingModuleIPA(monomer_train_module.TrainingModuleIPA):
    """Lightning module for multimer IPA training and validation."""

    def __init__(self, config: DictConfig):
        pl.LightningModule.__init__(self)
        self.global_config: DictConfig = config
        self.config: monomer_train_module.TrainConfig = config.train
        self.optimizer_config: monomer_train_module.OptimizerConfig = (
            self.config.optimizer
        )
        self.loss_config = self.config.loss
        self.compile_config: monomer_train_module.CompileConfig = self.config.compile

        self.save_hyperparameters(to_dict(self.global_config))

        model_cfg = OmegaConf.to_object(
            OmegaConf.merge(AtlasFoldMultimerIPAConfig, self.global_config.model)
        )
        self.model: AtlasFoldMultimerIPAForTrain = AtlasFoldMultimerIPAForTrain(model_cfg)

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
            0, self.config.num_recycles + 1, size=1_000_000
        )
        self.last_lr_step: int = -1

    def setup_losses(self) -> None:
        super().setup_losses()
        self.chain_com_loss_config = self.loss_config.chain_center_of_mass_loss

    def setup_metrics(self) -> None:
        names = [
            "complex/rmsd",
            "complex/lddt",
            "complex/lddt-ca",
            "chain/rmsd",
            "chain/lddt",
            "chain/lddt-ca",
            "interface/lddt",
            "interface/lddt-ca",
        ]
        if self.loss_weights.plddt != 0:
            names.append("confidence/plddt")
        if self.loss_weights.pae != 0:
            names.extend(("confidence/ptm", "confidence/iptm"))
        self.val_metrics = MetricCollection(
            {name: MeanMetric() for name in names}, prefix="val/"
        )

    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int
    ):
        if not isinstance(batch, dict) or "full_label" not in batch:
            return super().transfer_batch_to_device(batch, device, dataloader_idx)
        return {
            "feat": super().transfer_batch_to_device(
                batch["feat"], device, dataloader_idx
            ),
            "label": super().transfer_batch_to_device(
                batch["label"], device, dataloader_idx
            ),
            "loss_mask": super().transfer_batch_to_device(
                batch["loss_mask"], device, dataloader_idx
            ),
            "full_label": super().transfer_batch_to_device(
                batch["full_label"], device, dataloader_idx
            ),
        }

    def _align_labels(self, model_out, batch, label, full_label=None):
        pred = model_out["structure"]["coords"]
        aligned_coordinates, aligned_masks = [], []
        for i in range(pred.shape[0]):
            coords, mask = train_alignment.get_aligned_gt_structure(
                pred[i],
                {k: v[i] for k, v in batch.items()},
                {k: v[i] for k, v in label.items()},
                full_label[i],
                permutation=True,
            )
            aligned_coordinates.append(coords)
            aligned_masks.append(mask)
        return {
            "coordinates": torch.stack(aligned_coordinates),
            "resolved_mask": torch.stack(aligned_masks),
        }

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        num_recycles = int(
            self.recycles_per_step[self.global_step % len(self.recycles_per_step)]
        )
        out = self(batch["feat"], num_recycles, "train")
        aligned_label = self._align_labels(
            out,
            batch["feat"],
            batch["label"],
            batch["full_label"],
        )
        loss, metrics = self.compute_losses(
            out,
            batch["feat"],
            aligned_label,
            batch["loss_mask"],
        )
        for name, value in metrics.items():
            self.log(f"train/{name}", value, prog_bar=name == "loss", sync_dist=False)
        return loss

    def compute_losses(
        self,
        out: dict[str, Any],
        batch: dict[str, torch.Tensor],
        label: dict[str, torch.Tensor],
        loss_mask: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.autocast(batch["aatype"].device.type, enabled=False):
            batch_size = batch["seq_mask"].shape[0]
            zero = out["structure"]["coords"].new_zeros(batch_size)
            structure, structure_metrics, structure_labels = structure_ipa.structure_loss(
                out["structure"],
                batch["aatype_int"],
                label["coordinates"],
                label["resolved_mask"],
                batch["seq_mask"],
                batch["res_idx"],
                batch["asym_id"],
                multimer=True,
                use_clamped_fape=None,
                reduction="none",
            )
            if self.loss_weights.violation != 0:
                violation, violation_metrics = structure_ipa.violation_loss(
                    out["structure"],
                    structure_labels,
                    batch["res_idx"],
                    batch["seq_mask"],
                    batch["asym_id"],
                    reduction="none",
                    **self.violation_loss_config,
                )
                structure_metrics.update(
                    {
                        "violation": violation.mean().detach(),
                        **{
                            f"violation_{name}": value.detach()
                            for name, value in violation_metrics.items()
                        },
                    }
                )
            else:
                violation = zero

            chain_com_weight = float(self.loss_weights.chain_center_of_mass)
            if chain_com_weight != 0.0:
                chain_com = structure_ipa.chain_center_of_mass_loss(
                    out["structure"]["coords"],
                    label["coordinates"],
                    label["resolved_mask"],
                    batch["asym_id"],
                    reduction="none",
                    **self.chain_com_loss_config,
                )
            else:
                chain_com = zero

            if self.loss_weights.distogram != 0:
                dgram = self.distogram_loss(
                    out["distogram"]["logits"].float(),
                    out["distogram"]["boundaries"],
                    label["coordinates"].float(),
                    label["resolved_mask"],
                    batch["pseudo_beta"],
                )
            else:
                dgram = zero

            confidence_mask = loss_mask["confidence"].float()
            pred_pos = out["structure"]["coords"].float()
            if self.loss_weights.plddt != 0:
                plddt = self.plddt_loss(
                    out["confidence"]["plddt"]["logits"].float(),
                    out["confidence"]["plddt"]["bin_centers"],
                    pred_pos,
                    label["coordinates"].float(),
                    label["resolved_mask"],
                )
                plddt = plddt * confidence_mask
            else:
                plddt = zero

            if self.loss_weights.pae != 0:
                pae = self.pae_loss(
                    out["confidence"]["pae"]["logits"].float(),
                    out["confidence"]["pae"]["bin_centers"],
                    pred_pos,
                    label["coordinates"].float(),
                    label["resolved_mask"],
                )
                pae = pae * confidence_mask
            else:
                pae = zero

            if self.loss_weights.experimentally_resolved != 0:
                exp = self.exp_res_loss(
                    out["confidence"]["experimentally_resolved"]["logits"].float(),
                    structure_ipa.normalize_aatype(batch["aatype_int"]),
                    label["resolved_mask"],
                    batch["seq_mask"].bool(),
                )
                exp = exp * confidence_mask
            else:
                exp = zero

            per_example_total = (
                self.loss_weights.structure * structure
                + self.loss_weights.violation * violation
                + chain_com_weight * chain_com
                + self.loss_weights.distogram * dgram
                + self.loss_weights.plddt * plddt
                + self.loss_weights.pae * pae
                + self.loss_weights.experimentally_resolved * exp
            )
            length_scale = torch.sqrt(batch["seq_mask"].float().sum(-1).clamp(min=1))
            total = (per_example_total * length_scale).mean()

        metrics = {
            "loss": total.detach(),
            "unscaled_loss": per_example_total.mean().detach(),
            "structure/loss": structure.mean().detach(),
            "structure/violation": violation.mean().detach(),
            "structure/chain_center_of_mass": chain_com.mean().detach(),
            "distogram/loss": dgram.mean().detach(),
            "confidence/plddt_loss": plddt.mean().detach(),
            "confidence/pae_loss": pae.mean().detach(),
            "confidence/experimentally_resolved_loss": exp.mean().detach(),
        }
        metrics.update({f"structure/{k}": v for k, v in structure_metrics.items()})
        return total, metrics

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        feat, label = batch["feat"], batch["label"]
        try:
            devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(batch_idx)
                out = self(feat, int(self.config.num_recycles), "validation")
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return
        metrics = validation_metrics._compute_single_sample_metrics(
            out["coords"], feat, label
        )
        for name, value in metrics.items():
            if name in self.val_metrics:
                self.val_metrics[name].update(value)
        if self.loss_weights.plddt != 0:
            mask = feat["seq_mask"].bool()
            self.val_metrics["confidence/plddt"].update(out["plddt"][mask].float().mean())
        if self.loss_weights.pae != 0:
            self.val_metrics["confidence/ptm"].update(out["ptm"].float())
            self.val_metrics["confidence/iptm"].update(out["iptm"].float())

    def on_validation_epoch_end(self) -> None:
        torch.backends.cudnn.benchmark = True
        if not self.trainer.sanity_checking:
            for name, value in self.val_metrics.compute().items():
                self.log(
                    name,
                    value,
                    prog_bar=name == "val/complex/lddt",
                    sync_dist=True,
                )
        self.val_metrics.reset()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
