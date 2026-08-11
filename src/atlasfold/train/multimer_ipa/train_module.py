"""Lightning training module for AtlasFold multimer IPA."""

from __future__ import annotations

import gc
from typing import Any

import torch
from torchmetrics import MeanMetric, MetricCollection

from atlasfold.model.model_multimer_ipa import AtlasFoldMultimerIPAConfig
from atlasfold.train.monomer_ipa.train_module import (
    TrainingModuleIPA as MonomerTrainingModuleIPA,
)
from atlasfold.train.multimer import train_alignment, validation_metrics
from atlasfold.train.multimer_ipa.model_train import AtlasFoldMultimerIPAForTrain


class TrainingModuleIPA(MonomerTrainingModuleIPA):
    model_config_class = AtlasFoldMultimerIPAConfig
    model_class = AtlasFoldMultimerIPAForTrain

    @property
    def is_multimer(self) -> bool:
        return True

    def setup_metrics(self) -> None:
        names = (
            "complex/rmsd",
            "complex/lddt",
            "complex/lddt-ca",
            "chain/rmsd",
            "chain/lddt",
            "chain/lddt-ca",
            "interface/lddt",
            "interface/lddt-ca",
            "confidence/plddt",
            "confidence/ptm",
            "confidence/iptm",
        )
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

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        feat, label = batch["feat"], batch["label"]
        try:
            devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(batch_idx)
                out = self(feat, int(self.validation_config.num_recycles), "validation")
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
        mask = feat["seq_mask"].bool()
        self.val_metrics["confidence/plddt"].update(out["plddt"][mask].float().mean())
        self.val_metrics["confidence/ptm"].update(out["ptm"].float())
        self.val_metrics["confidence/iptm"].update(out["iptm"].float())

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:
            for name, value in self.val_metrics.compute().items():
                self.log(
                    name,
                    value,
                    prog_bar=name == "val/complex/lddt",
                    sync_dist=True,
                )
        self.val_metrics.reset()
