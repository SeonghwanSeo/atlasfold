"""Lightning training module for AtlasFold monomer IPA."""

from __future__ import annotations

import dataclasses
import gc
from collections.abc import Mapping
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torchmetrics import MeanMetric, MetricCollection

from atlasfold.model.model_ipa import AtlasFoldIPAConfig
from atlasfold.train import losses
from atlasfold.train.losses import structure_ipa
from atlasfold.train.monomer import validation_metrics
from atlasfold.train.monomer_ipa.model_train import AtlasFoldIPAForTrain
from atlasfold.train.utils.ema import ExponentialMovingAverage
from atlasfold.train.utils.gradient_logging import gradient_norm, parameter_norm
from atlasfold.train.utils.lr_scheduler import AlphaFoldLRScheduler


def to_dict(config: DictConfig) -> dict:
    return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]


@dataclasses.dataclass(kw_only=True)
class TrainingConfig:
    train_trunk: bool = True
    train_structure_module: bool = True
    train_distogram_head: bool = True
    train_confidence_head: bool = True
    num_recycles: int = 3
    unclamped_fape_probability: float = 0.1


class TrainingModuleIPA(pl.LightningModule):
    model_config_class = AtlasFoldIPAConfig
    model_class = AtlasFoldIPAForTrain

    def __init__(self, config: DictConfig):
        super().__init__()
        self.global_config = config
        self.config = config.train
        self.training_config = OmegaConf.to_object(
            OmegaConf.merge(TrainingConfig, self.config.training)
        )
        self.validation_config = self.config.validation
        self.optimizer_config = self.config.optimizer
        self.loss_config = self.config.loss
        self.compile_config = self.config.compile
        self.save_hyperparameters(to_dict(config))

        model_cfg = OmegaConf.to_object(
            OmegaConf.merge(self.model_config_class, config.model)
        )
        self.model = self.model_class(model_cfg)
        kernel_config = self.config.get("kernel", {})
        self.model.set_forward_flags(
            use_cuequiv_kernels=bool(kernel_config.get("cuequivariance", False))
        )
        self.train_trunk = self.training_config.train_trunk
        self.train_structure_module = self.training_config.train_structure_module
        self.train_distogram_head = self.training_config.train_distogram_head
        self.train_confidence_head = self.training_config.train_confidence_head
        if not 0.0 <= self.training_config.unclamped_fape_probability <= 1.0:
            raise ValueError("unclamped_fape_probability must be between 0 and 1.")
        self.freeze_submodules()
        self.setup_losses()
        self.setup_metrics()
        if self.compile_config.enabled:
            self.model.compile_train(
                mode=self.compile_config.mode, dynamic=self.compile_config.dynamic
            )

        self.ema = ExponentialMovingAverage(
            self.model,
            decay=self.optimizer_config.ema_decay,
            submodules_to_ignore=self.optimizer_config.ema_ignore_params,
            submodules_to_update=self.optimizer_config.ema_update_params,
        )
        self.stored_params: dict[str, torch.Tensor] | None = None
        rng = np.random.default_rng(seed=42)
        self.recycles_per_step = rng.integers(
            0, self.training_config.num_recycles + 1, size=1_000_000
        )
        self.last_lr_step = -1

    def freeze_submodules(self) -> None:
        groups = self.model.get_module_groups()
        frozen = ["lm"]
        if not self.train_trunk:
            frozen.append("trunk")
        if not self.train_structure_module:
            frozen.append("structure_module")
        if not self.train_distogram_head:
            frozen.append("distogram_head")
        if not self.train_confidence_head:
            frozen.append("confidence_head")
        self.modules_to_freeze = frozen
        for name in frozen:
            for module in groups[name]:
                if isinstance(module, torch.nn.Parameter):
                    module.requires_grad_(False)
                else:
                    module.requires_grad_(False)

    def train(self, mode: bool = True):
        result = super().train(mode)
        groups = self.model.get_module_groups()
        for name in self.modules_to_freeze:
            for module in groups[name]:
                if isinstance(module, torch.nn.Module):
                    module.eval()
        return result

    def setup_losses(self) -> None:
        self.loss_weights = self.loss_config.weights
        self.violation_loss_config = self.loss_config.violation_loss
        self.chain_com_loss_config = self.loss_config.get(
            "chain_center_of_mass_loss", {}
        )
        self.distogram_loss = losses.distogram.DistogramLoss(
            **self.loss_config.distogram_loss
        )
        confidence = self.loss_config.confidence_loss
        self.plddt_loss = losses.confidence.PLDDTLoss(**confidence.plddt_loss)
        self.pae_loss = losses.confidence.PAELoss(**confidence.pae_loss)
        self.exp_res_loss = losses.confidence.ExperimentallyResolvedPredictionLoss(
            **confidence.experimentally_resolved_loss
        )

    def setup_metrics(self) -> None:
        self.val_metrics = MetricCollection(
            {
                name: MeanMetric()
                for name in (
                    "rmsd",
                    "lddt",
                    "lddt-ca",
                    "plddt",
                    "plddt_mae",
                    "pae_mae",
                    "ptm",
                )
            },
            prefix="val/",
        )

    def forward(self, batch: dict[str, torch.Tensor], num_recycles: int, mode: str):
        if mode == "train":
            return self.model.forward_train(
                batch,
                num_recycles,
                train_trunk=self.train_trunk,
                train_structure_module=self.train_structure_module,
                run_distogram_head=self.train_distogram_head,
                run_confidence_head=self.train_confidence_head,
            )
        return self.model.inference(batch, num_recycles=num_recycles)

    def _align_labels(
        self,
        model_out: dict[str, Any],
        batch: dict[str, torch.Tensor],
        label: dict[str, torch.Tensor],
        full_label: Any = None,
    ) -> dict[str, torch.Tensor]:
        return label

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        num_recycles = int(
            self.recycles_per_step[self.global_step % len(self.recycles_per_step)]
        )
        out = self(batch["feat"], num_recycles, "train")
        aligned = self._align_labels(
            out, batch["feat"], batch["label"], batch.get("full_label")
        )
        loss, metrics = self.compute_losses(
            out, batch["feat"], aligned, batch["loss_mask"]
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
            zero = out["z"].new_zeros(())
            if self.train_structure_module:
                use_clamped_fape = batch.get("use_clamped_fape")
                if use_clamped_fape is None and not self.is_multimer and self.training:
                    use_clamped_fape = (
                        torch.rand((), device=out["z"].device)
                        >= self.training_config.unclamped_fape_probability
                    ).to(out["z"].dtype)
                structure, structure_metrics, structure_labels = (
                    structure_ipa.structure_loss(
                        out["structure"],
                        batch["aatype_int"],
                        label["coordinates"],
                        label["resolved_mask"],
                        batch["seq_mask"],
                        batch["res_idx"],
                        batch.get("asym_id") if self.is_multimer else None,
                        multimer=self.is_multimer,
                        use_clamped_fape=use_clamped_fape,
                    )
                )
                if self.loss_weights.violation != 0:
                    violation, violation_metrics = structure_ipa.violation_loss(
                        out["structure"],
                        structure_labels,
                        batch["res_idx"],
                        batch["seq_mask"],
                        batch.get("asym_id") if self.is_multimer else None,
                        **self.violation_loss_config,
                    )
                    structure_metrics.update(
                        {
                            "violation": violation.detach(),
                            **{
                                f"violation_{name}": value.detach()
                                for name, value in violation_metrics.items()
                            },
                        }
                    )
                else:
                    violation = zero
                chain_com_weight = float(
                    self.loss_weights.get("chain_center_of_mass", 0.0)
                )
                if self.is_multimer and chain_com_weight != 0.0:
                    chain_com = structure_ipa.chain_center_of_mass_loss(
                        out["structure"]["coords"],
                        label["coordinates"],
                        label["resolved_mask"],
                        batch["asym_id"],
                        **self.chain_com_loss_config,
                    )
                else:
                    chain_com = zero
            else:
                structure = violation = chain_com = zero
                structure_metrics = {}
            if self.train_distogram_head:
                dgram = self.distogram_loss(
                    out["distogram"]["logits"].float(),
                    out["distogram"]["boundaries"],
                    label["coordinates"].float(),
                    label["resolved_mask"],
                    batch["pseudo_beta"],
                ).mean()
            else:
                dgram = zero
            if self.train_confidence_head:
                confidence_mask = loss_mask["confidence"].float()
                confidence_denominator = confidence_mask.sum().clamp(min=1.0)
                pred_pos = out["structure"]["coords"].float()
                plddt = self.plddt_loss(
                    out["confidence"]["plddt"]["logits"].float(),
                    out["confidence"]["plddt"]["bin_centers"],
                    pred_pos,
                    label["coordinates"].float(),
                    label["resolved_mask"],
                )
                plddt = (plddt * confidence_mask).sum() / confidence_denominator
                pae = self.pae_loss(
                    out["confidence"]["pae"]["logits"].float(),
                    out["confidence"]["pae"]["bin_centers"],
                    pred_pos,
                    label["coordinates"].float(),
                    label["resolved_mask"],
                )
                pae = (pae * confidence_mask).sum() / confidence_denominator
                exp = self.exp_res_loss(
                    out["confidence"]["experimentally_resolved_logits"].float(),
                    structure_ipa.normalize_aatype(batch["aatype_int"]),
                    label["resolved_mask"],
                    batch["seq_mask"].bool(),
                )
                exp = (exp * confidence_mask).sum() / confidence_denominator
            else:
                plddt = pae = exp = zero
            total = (
                self.loss_weights.structure * structure
                + self.loss_weights.violation * violation
                + float(self.loss_weights.get("chain_center_of_mass", 0.0))
                * chain_com
                + self.loss_weights.distogram * dgram
                + self.loss_weights.plddt * plddt
                + self.loss_weights.pae * pae
                + self.loss_weights.experimentally_resolved * exp
            )
            length_scale = torch.sqrt(
                batch["seq_mask"].float().sum(-1).clamp(min=1)
            ).mean()
            total = total * length_scale
        metrics = {
            "loss": total.detach(),
            "unscaled_loss": (total / length_scale).detach(),
            "structure/loss": structure.detach(),
            "structure/violation": violation.detach(),
            "structure/chain_center_of_mass": chain_com.detach(),
            "distogram/loss": dgram.detach(),
            "confidence/plddt_loss": plddt.detach(),
            "confidence/pae_loss": pae.detach(),
            "confidence/experimentally_resolved_loss": exp.detach(),
        }
        metrics.update({f"structure/{k}": v for k, v in structure_metrics.items()})
        return total, metrics

    @property
    def is_multimer(self) -> bool:
        return False

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
        raw = validation_metrics.compute_validation_metric(
            out["coords"].unsqueeze(0), feat, label
        )
        for name in ("rmsd", "lddt", "lddt-ca"):
            self.val_metrics[name].update(raw[f"avg/{name}"])
        seq_mask = feat["seq_mask"].bool()
        mean_plddt = out["plddt"][seq_mask].float().mean()
        self.val_metrics["plddt"].update(mean_plddt)
        self.val_metrics["plddt_mae"].update(
            torch.abs(mean_plddt - mean_plddt.new_tensor(raw["avg/lddt-ca"]))
        )
        target_pae = self.pae_loss.get_alignment_error(
            out["coords"].unsqueeze(0).float(),
            label["coordinates"].unsqueeze(0).float(),
            label["resolved_mask"].unsqueeze(0),
        )[0]
        pae_mask = self.pae_loss.get_pair_mask(label["resolved_mask"].unsqueeze(0))[0]
        pae_mae = torch.abs(out["pae"].float() - target_pae)
        pae_mae = (pae_mae * pae_mask).sum() / pae_mask.sum().clamp(min=1)
        self.val_metrics["pae_mae"].update(pae_mae)
        self.val_metrics["ptm"].update(out["ptm"].float())

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:
            for name, value in self.val_metrics.compute().items():
                self.log(name, value, prog_bar=name == "val/lddt", sync_dist=True)
        self.val_metrics.reset()

    def configure_optimizers(self):
        cfg = self.optimizer_config
        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=cfg.max_lr,
            betas=(cfg.beta_1, cfg.beta_2),
            eps=cfg.eps,
        )
        if self.last_lr_step != -1:
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", cfg.max_lr)
        scheduler = AlphaFoldLRScheduler(
            optimizer,
            max_lr=cfg.max_lr,
            warmup_steps=cfg.warmup_steps,
            decay_steps=cfg.decay_steps,
            decay_factor=cfg.lr_decay_factor,
            last_epoch=self.last_lr_step,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def on_before_optimizer_step(self, optimizer) -> None:
        if self.trainer.global_step % 10 == 0:
            self.log("monitor/grad_norm/model", gradient_norm(self.model))
            self.log("monitor/param_norm/model", parameter_norm(self.model))

    def on_train_start(self) -> None:
        self.ema.to(self.device)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        if self.ema.device != self.device:
            self.ema.to(self.device)
        self.ema.update(self.model)

    def on_validation_start(self) -> None:
        self.stored_params = {
            k: p.clone().detach()
            for k, p in self.model.state_dict().items()
            if not k.startswith("lm.")
        }
        self.model.load_state_dict(self.ema.params, strict=False)

    def on_validation_end(self) -> None:
        if self.stored_params is not None:
            self.model.load_state_dict(self.stored_params, strict=False)
            self.stored_params = None

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["state_dict"] = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if not k.startswith("model.lm.")
        }
        checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"], strict=False)

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        stripped = {k.removeprefix("model."): v for k, v in state_dict.items()}
        if strict:
            model_keys = set(self.model.state_dict())
            supplied = set(stripped)
            missing = sorted(
                key for key in model_keys - supplied if not key.startswith("lm.")
            )
            unexpected = sorted(supplied - model_keys)
            if missing or unexpected:
                raise RuntimeError(
                    "IPA checkpoint state mismatch: "
                    f"missing={missing}, unexpected={unexpected}"
                )
        return self.model.load_state_dict(stripped, strict=False)
