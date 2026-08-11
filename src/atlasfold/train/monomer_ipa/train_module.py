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
    """Convert a DictConfig to a standard Python dictionary."""
    return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]


@dataclasses.dataclass(kw_only=True)
class TrainConfig:
    """Configuration for training and validation steps."""

    name: str
    out_dir: str
    seed: int
    compile: CompileConfig
    optimizer: OptimizerConfig
    loss: LossConfig
    kernel: KernelConfig
    # multi-phase training
    load_opt_state: bool = True
    # trunk recycling
    num_recycles: int = 4
    mlm_prob: float = 0.15


@dataclasses.dataclass(kw_only=True)
class OptimizerConfig:
    """Optimizer configuration."""

    # optimizer
    opt: str = "adam"
    beta_1: float = 0.9
    beta_2: float = 0.999
    eps: float = 1e-6
    # lr scheduler
    lr_scheduler: str = "alphafold"
    max_lr: float = 1e-3
    warmup_steps: int = 1000
    decay_steps: int = 50000
    lr_decay_factor: float = 0.95
    # ema
    ema_decay: float = 0.999
    ema_ignore_params: tuple[str] | None = ("lm.",)
    ema_update_params: tuple[str] | None = None


@dataclasses.dataclass(kw_only=True)
class LossConfig:
    """Loss configuration."""

    weights: dict[str, float]
    distogram_loss: Any
    violation_loss: Any
    confidence_loss: Any
    chain_center_of_mass_loss: Any = None


@dataclasses.dataclass(kw_only=True)
class CompileConfig:
    enabled: bool = False
    mode: str = "default"
    dynamic: bool = False


@dataclasses.dataclass(kw_only=True)
class KernelConfig:
    cuequivariance: bool = False


class TrainingModuleIPA(pl.LightningModule):
    """Lightning module for monomer IPA training and validation."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.global_config: DictConfig = config
        self.config: TrainConfig = config.train
        self.optimizer_config: OptimizerConfig = self.config.optimizer
        self.loss_config: LossConfig = OmegaConf.to_object(
            OmegaConf.merge(OmegaConf.structured(LossConfig), self.config.loss)
        )
        self.compile_config: CompileConfig = self.config.compile

        if not 0.0 <= self.config.mlm_prob <= 1.0:
            raise ValueError("train.mlm_prob must be between 0 and 1.")

        # Save hyperparameters
        self.save_hyperparameters(to_dict(self.global_config))

        # Initialize model
        model_cfg = OmegaConf.to_object(
            OmegaConf.merge(AtlasFoldIPAConfig, self.global_config.model)
        )
        self.model: AtlasFoldIPAForTrain = AtlasFoldIPAForTrain(model_cfg)

        # Setup losses and validation metrics
        self.setup_losses()
        self.setup_metrics()

        # Compile model
        if self.compile_config.enabled:
            self.model.compile_train(
                mode=self.compile_config.mode, dynamic=self.compile_config.dynamic
            )

        # Setup EMA
        self.ema: ExponentialMovingAverage = ExponentialMovingAverage(
            self.model,
            decay=self.optimizer_config.ema_decay,
            submodules_to_ignore=self.optimizer_config.ema_ignore_params,
            submodules_to_update=self.optimizer_config.ema_update_params,
        )
        self.stored_params: dict[str, torch.Tensor] | None = None

        # Use a shared recycling schedule across all ranks.
        rng = np.random.default_rng(seed=42)
        self.recycles_per_step: np.ndarray = rng.integers(
            0, self.config.num_recycles + 1, size=1_000_000
        )
        self.last_lr_step: int = -1

    def train(self, mode: bool = True) -> TrainingModuleIPA:
        result = super().train(mode)
        # The pretrained LM is the only permanently frozen component.
        self.model.lm.eval()
        return result

    def setup_losses(self) -> None:
        """Setup loss functions for training."""
        loss_config = dataclasses.asdict(self.loss_config)
        self.loss_weights: dict[str, float] = loss_config["weights"]
        self.violation_loss_config = loss_config["violation_loss"]

        # Distogram loss
        self.distogram_loss = losses.distogram.DistogramLoss(
            **loss_config["distogram_loss"]
        )

        # Confidence losses
        confidence = loss_config["confidence_loss"]
        self.plddt_loss = losses.confidence.PLDDTLoss(**confidence["plddt_loss"])
        self.pae_loss = losses.confidence.PAELoss(**confidence["pae_loss"])
        self.exp_res_loss = losses.confidence.ExperimentallyResolvedPredictionLoss(
            **confidence["experimentally_resolved_loss"]
        )

    def setup_metrics(self) -> None:
        """Setup metrics for validation."""
        names = ["rmsd", "lddt", "lddt-ca"]
        if self.loss_weights["plddt"] != 0:
            names.extend(("plddt", "plddt_mae"))
        if self.loss_weights["pae"] != 0:
            names.extend(("pae_mae", "ptm"))
        self.val_metrics = MetricCollection(
            {name: MeanMetric() for name in names},
            prefix="val/",
        )

    def forward(self, batch: dict[str, torch.Tensor], num_recycles: int, mode: str):
        if mode == "train":
            return self.model.forward_train(
                batch, num_recycles, mlm_prob=self.config.mlm_prob
            )
        return self.model.inference(batch, num_recycles=num_recycles)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        # Sample recycling steps from the shared schedule.
        num_recycles = int(
            self.recycles_per_step[self.global_step % len(self.recycles_per_step)]
        )
        # Compute the forward pass and losses.
        out = self(batch["feat"], num_recycles, "train")
        loss, metrics = self.compute_losses(
            out, batch["feat"], batch["label"], batch["loss_mask"]
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
        """Compute the weighted IPA training losses."""
        metrics: dict[str, torch.Tensor] = {}
        with torch.autocast(batch["aatype"].device.type, enabled=False):
            batch_size = batch["seq_mask"].shape[0]
            device = batch["aatype"].device
            zero = torch.zeros(batch_size, device=device, dtype=torch.float32)
            dummy_loss = torch.zeros((), device=device, dtype=torch.float32)
            structure, structure_metrics, structure_labels = structure_ipa.structure_loss(
                out["structure"],
                batch["aatype_int"],
                label["coordinates"],
                label["resolved_mask"],
                batch["seq_mask"],
                batch["res_idx"],
                use_clamped_fape=batch["use_clamped_fape"],
                reduction="none",
            )
            if self.loss_weights["violation"] != 0:
                violation, violation_metrics = structure_ipa.violation_loss(
                    out["structure"],
                    structure_labels,
                    batch["res_idx"],
                    batch["seq_mask"],
                    None,
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
            if self.loss_weights["distogram"] != 0:
                dgram = self.distogram_loss(
                    out["distogram"]["logits"].float(),
                    out["distogram"]["boundaries"],
                    label["coordinates"].float(),
                    label["resolved_mask"],
                    batch["pseudo_beta"],
                )
                metrics["distogram/loss"] = dgram.mean().detach()
            else:
                dgram = zero
                dummy_loss = dummy_loss + (out["distogram"]["logits"] * 0).mean()
            confidence_mask = loss_mask["confidence"].float()
            pred_pos = out["structure"]["coords"].float()
            if self.loss_weights["plddt"] != 0:
                plddt = self.plddt_loss(
                    out["confidence"]["plddt"]["logits"].float(),
                    out["confidence"]["plddt"]["bin_centers"],
                    pred_pos,
                    label["coordinates"].float(),
                    label["resolved_mask"],
                )
                plddt = plddt * confidence_mask
                metrics["confidence/plddt_loss"] = plddt.mean().detach()
            else:
                plddt = zero
                dummy_loss = (
                    dummy_loss + (out["confidence"]["plddt"]["logits"] * 0).mean()
                )
            if self.loss_weights["pae"] != 0:
                pae = self.pae_loss(
                    out["confidence"]["pae"]["logits"].float(),
                    out["confidence"]["pae"]["bin_centers"],
                    pred_pos,
                    label["coordinates"].float(),
                    label["resolved_mask"],
                )
                pae = pae * confidence_mask
                metrics["confidence/pae_loss"] = pae.mean().detach()
            else:
                pae = zero
                dummy_loss = dummy_loss + (out["confidence"]["pae"]["logits"] * 0).mean()
            if self.loss_weights["experimentally_resolved"] != 0:
                exp = self.exp_res_loss(
                    out["confidence"]["experimentally_resolved"]["logits"].float(),
                    structure_ipa.normalize_aatype(batch["aatype_int"]),
                    label["resolved_mask"],
                    batch["seq_mask"].bool(),
                )
                exp = exp * confidence_mask
                metrics["confidence/experimentally_resolved_loss"] = exp.mean().detach()
            else:
                exp = zero
                dummy_loss = (
                    dummy_loss
                    + (out["confidence"]["experimentally_resolved"]["logits"] * 0).mean()
                )
            per_example_total = (
                self.loss_weights["structure"] * structure
                + self.loss_weights["violation"] * violation
                + self.loss_weights["distogram"] * dgram
                + self.loss_weights["plddt"] * plddt
                + self.loss_weights["pae"] * pae
                + self.loss_weights["experimentally_resolved"] * exp
            )
            length_scale = torch.sqrt(batch["seq_mask"].float().sum(-1).clamp(min=1))
            scaled_per_example = per_example_total * length_scale
            total = scaled_per_example.mean() + dummy_loss
        metrics |= {
            "loss": total.detach(),
            "unscaled_loss": per_example_total.mean().detach(),
            "structure/loss": structure.mean().detach(),
        }
        metrics |= {f"structure/{k}": v for k, v in structure_metrics.items()}
        return total, metrics

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        feat, label = batch["feat"], batch["label"]
        # Use a deterministic per-example seed for stochastic LM masking.
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
        raw = validation_metrics.compute_validation_metric(
            out["coords"].unsqueeze(0), feat, label
        )
        for name in ("rmsd", "lddt", "lddt-ca"):
            self.val_metrics[name].update(raw[f"avg/{name}"])
        if self.loss_weights["plddt"] != 0:
            seq_mask = feat["seq_mask"].bool()
            mean_plddt = out["plddt"][seq_mask].float().mean()
            self.val_metrics["plddt"].update(mean_plddt)
            self.val_metrics["plddt_mae"].update(
                torch.abs(mean_plddt - mean_plddt.new_tensor(raw["avg/lddt-ca"]))
            )
        if self.loss_weights["pae"] != 0:
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
        torch.backends.cudnn.benchmark = True
        if not self.trainer.sanity_checking:
            for name, value in self.val_metrics.compute().items():
                self.log(name, value, prog_bar=name == "val/lddt", sync_dist=True)
        self.val_metrics.reset()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_validation_epoch_start(self) -> None:
        torch.backends.cudnn.benchmark = False

    def configure_optimizers(self):
        config = self.optimizer_config
        if config.opt.lower() == "adam":
            # Adam optimizer without weight decay.
            optimizer = torch.optim.Adam(
                [p for p in self.parameters() if p.requires_grad],
                lr=config.max_lr,
                betas=(config.beta_1, config.beta_2),
                eps=config.eps,
            )
        else:
            raise NotImplementedError(f"Optimizer {config.opt} not implemented yet.")

        if self.last_lr_step != -1:
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", config.max_lr)

        if config.lr_scheduler == "alphafold":
            scheduler = AlphaFoldLRScheduler(
                optimizer,
                max_lr=config.max_lr,
                warmup_steps=config.warmup_steps,
                decay_steps=config.decay_steps,
                decay_factor=config.lr_decay_factor,
                last_epoch=self.last_lr_step,
            )
        else:
            raise NotImplementedError(
                f"LR scheduler {config.lr_scheduler} not implemented yet."
            )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def on_before_optimizer_step(self, optimizer) -> None:
        if self.trainer.global_step % 10 == 0:
            self.log_model_state()

    def log_model_state(self) -> None:
        """Log model parameter and gradient norms."""

        def log(name: str, value: float) -> None:
            self.log(f"monitor/{name}", value, sync_dist=False, prog_bar=False)

        model = self.model
        log("grad_norm/model", gradient_norm(model))
        log("param_norm/model", parameter_norm(model))
        log("grad_norm/main_stack", gradient_norm(model.main_stack))
        log("param_norm/main_stack", parameter_norm(model.main_stack))
        log("grad_norm/lm_stack", gradient_norm(model.lm_stack))
        log("param_norm/lm_stack", parameter_norm(model.lm_stack))
        log("grad_norm/structure_module", gradient_norm(model.structure_module))
        log("param_norm/structure_module", parameter_norm(model.structure_module))

    def on_train_start(self) -> None:
        self.ema.to(self.device)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        if self.ema.device != self.device:
            self.ema.to(self.device)
        self.ema.update(self.model)

    def on_validation_start(self) -> None:
        # Replace live parameters with EMA parameters during validation.
        self.stored_params = {
            k: p.clone().detach()
            for k, p in self.model.state_dict().items()
            if not k.startswith("lm.")
        }
        self.model.load_state_dict(self.ema.params, strict=False)

    def on_validation_end(self) -> None:
        # Restore live parameters after validation.
        if self.stored_params is not None:
            self.model.load_state_dict(self.stored_params, strict=False)
            self.stored_params = None

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # Exclude the frozen pretrained LM and save EMA separately.
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
        """Load a model-only or Lightning-prefixed IPA state dictionary."""
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
