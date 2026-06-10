import dataclasses
import gc
from collections.abc import Mapping
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torchmetrics import MeanMetric, MetricCollection

from atlasfold.model import AtlasFoldConfig
from atlasfold.model.network.diffusion_head import SamplingConfig
from atlasfold.train.utils import structure_metrics

from . import losses, validation_metrics
from .model_train import AtlasFoldForTrain
from .utils.ema import ExponentialMovingAverage
from .utils.gradient_logging import gradient_norm, parameter_norm
from .utils.lr_scheduler import AF2LRScheduler


def to_dict(config: DictConfig) -> dict:
    """Convert a DictConfig to a standard Python dictionary."""
    return OmegaConf.to_container(config, resolve=True)


@dataclasses.dataclass(kw_only=True)
class TrainConfig:
    """Configuration for training and validation steps."""

    name: str
    out_dir: str
    seed: int
    compile: "CompileConfig"
    training: "TrainingConfig"
    validation: "ValidationConfig"
    optimizer: "OptimizerConfig"
    loss: "LossConfig"
    kernel: "KernelConfig"


@dataclasses.dataclass(kw_only=True)
class OptimizerConfig:
    """Optimizer configuration."""

    # optimizer
    opt: str = "adamw"
    beta_1: float = 0.9
    beta_2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 1e-4
    # lr scheduler
    lr_scheduler: str = "af2"
    base_lr: float = 1e-3
    num_warmup_steps: int = 1000
    decay_start_step: int = 50000
    lr_decay_factor: float = 0.95
    # ema
    ema_decay: float = 0.999
    ema_ignore_params: tuple[str] | None = ("lm.",)
    ema_update_params: tuple[str] | None = None
    # multi-phase training
    load_opt_state_from_checkpoint: bool = True


@dataclasses.dataclass(kw_only=True)
class TrainingConfig:
    """Training step configuration."""

    # Whether to train each submodules
    train_trunk: bool = True
    train_diffusion_head: bool = True
    train_confidence_head: bool = True
    train_pae_head: bool = False

    # trunk recycling
    num_recycles: int = 3
    # for structure model training
    diffusion_batch_size: int = 48
    # for confidence module training
    num_mini_rollout_steps: int = 20


@dataclasses.dataclass(kw_only=True)
class ValidationConfig:
    """Validation step configuration."""

    num_recycles: int = 3
    num_steps: int = 100
    num_diffusion_samples: int = 3


@dataclasses.dataclass(kw_only=True)
class LossConfig:
    """Loss configuration."""

    weights: dict[str, float]
    distogram_loss: Any
    diffusion_loss: Any
    confidence_loss: Any


@dataclasses.dataclass(kw_only=True)
class CompileConfig:
    enabled: bool = False
    mode: str = "default"
    dynamic: bool = False


@dataclasses.dataclass(kw_only=True)
class KernelConfig:
    cuequivariance: bool = False


class TrainingModule(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.global_config: DictConfig = config
        self.config: TrainConfig = config.train
        self.training_config: TrainingConfig = self.config.training
        self.validation_config: ValidationConfig = self.config.validation
        self.optimizer_config: OptimizerConfig = self.config.optimizer
        self.loss_config: LossConfig = self.config.loss
        self.compile_config: CompileConfig = self.config.compile

        # Save hyperparameters
        self.save_hyperparameters(to_dict(self.global_config))

        # Whether to train structure and confidence modules
        self.train_trunk: bool = self.training_config.train_trunk
        self.train_diffusion_head: bool = self.training_config.train_diffusion_head
        self.train_confidence_head: bool = self.training_config.train_confidence_head
        self.train_pae_head: bool = self.training_config.train_pae_head

        # Initialize model here
        model_cfg = OmegaConf.to_object(
            OmegaConf.merge(AtlasFoldConfig, self.global_config.model)
        )
        self.model: AtlasFoldForTrain = AtlasFoldForTrain(model_cfg)

        # Freeze parts of the model if needed
        self.freeze_submodules()

        # Setup losses and metrics
        self.setup_losses()
        self.setup_metrics()

        # Compile
        if self.compile_config.enabled:
            self.model.compile_train(
                mode=self.compile_config.mode, dynamic=self.compile_config.dynamic
            )

        # EMA state
        self.ema: ExponentialMovingAverage = ExponentialMovingAverage(
            self,
            decay=self.optimizer_config.ema_decay,
            submodules_to_ignore=self.optimizer_config.ema_ignore_params,
            submodules_to_update=self.optimizer_config.ema_update_params,
        )
        # Cache the current model parameters when using EMA params during validation.
        self.stored_params: dict[str, torch.Tensor] | None = None

        # Pre-sample recycling steps for training
        # This ensures all GPUs use the same recycling schedule
        rng = np.random.default_rng(seed=42)
        self.recycles_per_step: np.ndarray = rng.integers(
            0, self.training_config.num_recycles + 1, size=1000_000
        )

    def freeze_submodules(self):
        """Freeze submodules based on the training configuration."""
        modules_to_freeze: list[str] = []
        modules_to_freeze.append("lm")  # Always freeze the language model
        if self.train_trunk is False:
            modules_to_freeze.append("trunk")
            modules_to_freeze.append("distogram_head")
        if self.train_diffusion_head is False:
            modules_to_freeze.append("diffusion_head")
        if self.train_confidence_head is False:
            modules_to_freeze.append("confidence_head")
        if self.train_pae_head is False:
            modules_to_freeze.append("pae_head")
        print(f"Freezing modules: {modules_to_freeze}")
        self.modules_to_freeze: list[str] = modules_to_freeze

        module_groups = self.model.get_module_groups()
        for group in self.modules_to_freeze:
            for m in module_groups[group]:
                m.requires_grad_(False)

    def train(self, mode: bool = True):
        """Override train() to set sub-modules to eval mode if frozen."""
        out = super().train(mode)
        module_groups = self.model.get_module_groups()
        for group in self.modules_to_freeze:
            for m in module_groups[group]:
                if isinstance(m, torch.nn.Module):
                    m.eval()
        return out

    def setup_losses(self):
        """Setup loss functions for training"""
        loss_config = self.loss_config
        self.loss_weights: dict[str, float] = loss_config.weights

        # Distogram loss
        self.distogram_loss = losses.distogram.DistogramLoss(**loss_config.distogram_loss)

        # Diffusion loss
        diffusion_loss_config = loss_config.diffusion_loss
        self.mse_loss = losses.diffusion.MSELoss(**diffusion_loss_config["mse_loss"])
        self.smooth_lddt_loss = losses.diffusion.SmoothLDDTLoss(
            **diffusion_loss_config["smooth_lddt_loss"]
        )

        confidence_loss_config = loss_config.confidence_loss
        # pLDDT loss
        self.plddt_loss = losses.confidence.PLDDTLoss(
            **confidence_loss_config["plddt_loss"]
        )
        # Experimentally resolved loss
        self.exp_res_loss = losses.confidence.ExperimentallyResolvedPredictionLoss(
            **confidence_loss_config["experimentally_resolved_loss"]
        )
        # PAE loss
        self.pae_loss = losses.confidence.PAELoss(**confidence_loss_config["pae_loss"])

    def setup_metrics(self):
        """Setup metrics for validation"""
        val_metrics = {}
        for prefix in ["top", "avg", "rank"]:
            for k in ["rmsd", "lddt", "lddt-ca"]:
                val_metrics[f"{prefix}/{k}"] = MeanMetric()
        self.val_metrics = MetricCollection(val_metrics, prefix="val/")

    def configure_optimizers(self):
        config = self.optimizer_config
        parameters = [p for p in self.parameters() if p.requires_grad]

        if config.opt.lower() == "adam":
            optimizer = torch.optim.Adam(
                parameters,
                betas=(config.beta_1, config.beta_2),
                eps=config.eps,
                lr=config.base_lr,
            )
        elif config.opt.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                parameters,
                betas=(config.beta_1, config.beta_2),
                eps=config.eps,
                weight_decay=config.weight_decay,
                lr=config.base_lr,
            )
        else:
            raise NotImplementedError(f"Optimizer {config.opt} not implemented yet.")

        if config.lr_scheduler == "af2":
            scheduler = AF2LRScheduler(
                optimizer,
                base_lr=config.base_lr,
                num_warmup_steps=config.num_warmup_steps,
                decay_start_step=config.decay_start_step,
                decay_factor=config.lr_decay_factor,
            )
        else:
            raise NotImplementedError(
                f"LR scheduler {config.lr_scheduler} not implemented yet."
            )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        label: dict[str, torch.Tensor] | None,
        num_recycles: int,
        mode: str,
    ):
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
                train_pae_head=self.train_pae_head,
                sampling_config=sampling_config,
            )

        else:
            val_config = self.validation_config
            sampling_config = SamplingConfig(num_steps=val_config.num_steps)
            return self.model.inference(
                batch,
                mode="base",
                num_recycles=num_recycles,
                num_samples=val_config.num_diffusion_samples,
                sampling_config=sampling_config,
                compute_pae=False,
                return_representations=False,
                mlm_prob=0.0,
            )

    def training_step(
        self,
        batch: dict[str, dict[str, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        training_config = self.training_config

        # Sample recycling steps
        # Use shared recycling schedule across all the gpus
        idx = self.global_step % len(self.recycles_per_step)
        num_recycles = int(self.recycles_per_step[idx])

        # Compute the forward pass
        model_out: dict[str, torch.Tensor] = self(
            batch=batch["feat"],
            label=batch["label"],
            num_recycles=num_recycles,
            mode="train",
        )

        with torch.autocast("cuda", dtype=torch.float32):
            loss, metrics = self.compute_losses(
                model_out,
                batch["feat"],
                batch["label"],
                batch["loss_mask"],
            )

        for k, v in metrics.items():
            self.log(f"train/{k}", v, prog_bar=(k == "loss"), sync_dist=False)

        return loss

    def compute_losses(
        self,
        model_out: dict[str, Any],
        batch: dict[str, torch.Tensor],
        label: dict[str, torch.Tensor],
        loss_mask: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute losses of given the model output."""

        loss = torch.zeros(1, device=batch["aatype"].device, dtype=torch.float32)
        loss_weights = self.loss_weights

        metrics: dict[str, torch.Tensor] = {}
        if self.train_trunk:
            distogram_loss, distogram_metrics = self.compute_distogram_loss(
                model_out["distogram"], batch, label
            )
            loss += loss_weights["distogram"] * distogram_loss
            metrics |= {f"distogram/{k}": v for k, v in distogram_metrics.items()}

            # Augmented distogram loss directly from LM features.
            distogram_loss_aug, distogram_aug_metrics = self.compute_distogram_loss(
                model_out["distogram_aug"], batch, label
            )
            loss += loss_weights["distogram"] * distogram_loss_aug
            metrics |= {f"distogram_aug/{k}": v for k, v in distogram_aug_metrics.items()}

        if self.train_diffusion_head:
            diffusion_loss, diffusion_metrics = self.compute_diffusion_loss(
                model_out["diffusion"], batch, label
            )
            loss += loss_weights["diffusion"] * diffusion_loss
            metrics |= {f"diffusion/{k}": v for k, v in diffusion_metrics.items()}

        if self.train_confidence_head:
            confidence_out = model_out["confidence"]
            x_pred = confidence_out["mini_rollout"]["sample_coords"]  # [B, L, 14, 3]

            # Get aligned GT structure (rigid alignment + atom swapping)
            aligned_coords_list = []
            aligned_mask_list = []
            for b_i in range(x_pred.shape[0]):
                x_gt, mask = structure_metrics.get_aligned_gt_structure(
                    x_gt=label["coordinates"][b_i],
                    x_pred=x_pred[b_i],
                    aatype=batch["aatype_int"][b_i],
                    mask=label["resolved_mask"][b_i],
                )  # [L, 14, 3], [L, 14]
                aligned_coords_list.append(x_gt)
                aligned_mask_list.append(mask)
            x_gt = torch.stack(aligned_coords_list, dim=0)
            mask = torch.stack(aligned_mask_list, dim=0)
            aligned_label = {"coordinates": x_gt, "resolved_mask": mask}

            confidence_loss, confidence_metrics = self.compute_confidence_loss(
                confidence_out, batch, aligned_label, loss_mask["confidence"]
            )
            loss += loss_weights["confidence"] * confidence_loss
            metrics |= {f"confidence/{k}": v for k, v in confidence_metrics.items()}

            # Log the rmsd between mini-rollout sample and GT.
            with torch.no_grad():
                rmsd = structure_metrics.compute_rmsd_atom14(x_pred, x_gt, mask)
                lddt_ca = structure_metrics.compute_lddt_ca(x_pred, x_gt, mask)
                metrics |= {"mini_rollout/rmsd": rmsd.mean()}
                metrics |= {"mini_rollout/lddt-ca": lddt_ca.mean()}

        metrics["loss"] = loss.detach()
        return loss, metrics

    def validation_step(
        self,
        batch: dict[str, dict[str, torch.Tensor]],
        batch_idx: int,
    ):
        feat, label = batch["feat"], batch["label"]
        val_config = self.validation_config
        try:
            sample_out: dict[str, torch.Tensor] = self(
                batch=feat,
                label=None,
                num_recycles=val_config.num_recycles,
                mode="validation",
            )
        except RuntimeError as e:  # catch out of memory exceptions
            if "out of memory" in str(e):
                print("**WARNING**: ran out of memory, skipping batch")
                gc.collect()
                torch.cuda.empty_cache()
                return
            else:
                raise e

        x_pred = sample_out["sample_coords"]  # [N, L, 14, 3]

        # Compute diffusion sample rank based on pLDDT.
        if self.train_confidence_head:
            plddt = sample_out["plddt"]  # [N, L]
            mask = feat["seq_mask"]  # [L]
            w = mask.float()  # [L,]
            w_sum = w.sum(-1).clamp(min=1)
            avg_plddt = (plddt * w).sum(-1) / w_sum  # [N]
            rank_idx = int(torch.argmax(avg_plddt))
        else:
            rank_idx = None

        with torch.autocast("cuda", torch.float32):
            metrics: dict[str, float] = validation_metrics.compute_validation_metric(
                x_pred, feat, label, rank_idx
            )

        # Update validation metrics
        val_metrics: MetricCollection = self.val_metrics
        for k, v in metrics.items():
            val_metrics[k].update(v)

    def on_validation_epoch_start(self):
        torch.backends.cudnn.benchmark = False

    def on_validation_epoch_end(self):
        torch.backends.cudnn.benchmark = True
        if not self.trainer.sanity_checking:
            avg_values = self.val_metrics.compute()
            # NOTE: do not filter out NaN values to avoid deadlock in DDP
            self.log_dict(
                avg_values,
                on_step=False,
                on_epoch=True,
                # Already synced in compute(), but keep to avoid warning...
                sync_dist=True,
            )
        self.val_metrics.reset()
        gc.collect()
        torch.cuda.empty_cache()

    # === Loss functions === #
    def compute_distogram_loss(
        self,
        pred: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        label: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss = self.distogram_loss(
            logits=pred["logits"],
            boundaries=pred["boundaries"],
            x_gt=label["coordinates"],
            mask_gt=label["resolved_mask"],
            cbeta_idx=batch["pseudo_beta"],
        )  # [B,]
        loss = loss.mean()
        metrics = {"loss": loss.detach()}
        return loss, metrics

    def compute_diffusion_loss(
        self,
        pred: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        label: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        metrics: dict[str, torch.Tensor] = {}
        L_mse = self.mse_loss.forward(
            x_pred=pred["x_out"],
            x_gt=label["coordinates"],
            mask=label["resolved_mask"],
        )  # [B, N]
        L_mse = (L_mse * pred["loss_weights"]).mean(-1)
        L_mse = L_mse.mean()
        metrics["mse_loss"] = L_mse.detach()

        if self.loss_weights["smooth_lddt"] > 0:
            L_smooth_lddt = self.smooth_lddt_loss.forward(
                x_pred=pred["x_out"],
                x_gt=label["coordinates"],
                mask=label["resolved_mask"],
            )  # [B, N]
            L_smooth_lddt = L_smooth_lddt.mean(-1)
            L_smooth_lddt = L_smooth_lddt.mean()
            metrics["smooth_lddt_loss"] = L_smooth_lddt.detach()
        else:
            L_smooth_lddt = 0.0

        # Mean over diffusion samples
        L_diffusion = L_mse + L_smooth_lddt
        metrics["loss"] = L_diffusion.detach()

        return L_diffusion, metrics

    def compute_confidence_loss(
        self,
        pred: dict[str, Any],
        batch: dict[str, torch.Tensor],
        label: dict[str, torch.Tensor],
        loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        w = loss_mask.float()
        n_valid_samples = w.sum().clamp(1)

        metrics: dict[str, torch.Tensor] = {}
        alpha_pae = self.loss_weights["pae"]

        x_pred = pred["mini_rollout"]["sample_coords"]  # [B, L, 14, 3]
        x_gt = label["coordinates"]  # [B, L, 14, 3]
        resolved_mask = label["resolved_mask"]  # [B, L, 14]
        seq_mask = batch["seq_mask"]  # [B, L, 14]

        L_plddt = self.plddt_loss.forward(
            logits=pred["plddt"]["logits"],
            bin_centers=pred["plddt"]["bin_centers"],
            x_pred=x_pred,
            x_gt=x_gt,
            mask=resolved_mask,
        )  # [B,]
        L_plddt = (L_plddt * w).sum() / n_valid_samples
        metrics["plddt_loss"] = L_plddt.detach()

        L_resolved = self.exp_res_loss(
            logits=pred["experimentally_resolved"]["logits"],
            aatype=batch["aatype_int"],
            resolved_mask=resolved_mask,
            pad_mask=seq_mask,
        )
        L_resolved = (L_resolved * w).sum() / n_valid_samples
        metrics["resolved_loss"] = L_resolved.detach()

        # NOTE: PAE loss return 0.0 when alpha_pae is 0.
        if alpha_pae > 0:
            L_pae = self.pae_loss.forward(
                logits=pred["pae"]["logits"],
                bin_centers=pred["pae"]["bin_centers"],
                x_pred=x_pred,
                x_gt=x_gt,
                mask=resolved_mask,
            )
            L_pae = (L_pae * w).sum() / n_valid_samples
            metrics["pae_loss"] = L_pae.detach()
        else:
            L_pae = 0.0

        L_confidence = L_plddt + L_resolved + alpha_pae * L_pae
        metrics["loss"] = L_confidence.detach()
        return L_confidence, metrics

    # === Training logs === #
    def on_before_optimizer_step(self, optimizer) -> None:
        if self.trainer.global_step % 50 == 0:
            self.log_model_state()

    def log_model_state(self):
        """Log model parameter and gradient norms."""

        def log(name, value) -> None:
            self.log(f"montitor/{name}", value, sync_dist=False, prog_bar=False)

        model = self.model
        log("grad_norm/model", gradient_norm(model))
        log("param_norm/model", parameter_norm(model))
        if self.train_trunk:
            log("grad_norm/main_stack", gradient_norm(model.main_stack))
            log("param_norm/main_stack", parameter_norm(model.main_stack))
            log("grad_norm/lm_stack", gradient_norm(model.lm_stack))
            log("param_norm/lm_stack", parameter_norm(model.lm_stack))
        if self.train_diffusion_head:
            log("grad_norm/diffusion_head", gradient_norm(model.diffusion_head))
            log("param_norm/diffusion_head", parameter_norm(model.diffusion_head))
        if self.train_confidence_head:
            log("grad_norm/confidence_head", gradient_norm(model.confidence_head))
            log("param_norm/confidence_head", parameter_norm(model.confidence_head))

    # === EMA === #
    # TODO: this may not be robust for compiled models, where the state dict keys
    # are different. Consider implementing kep mapping logic in the EMA class.
    def on_train_start(self) -> None:
        self.ema.to(self.device)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):  # type: ignore
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        if self.ema.device != self.device:
            self.ema.to(self.device)
        self.ema.update(self)

    def on_validation_start(self):
        # Cache current model parameters before validation
        model_state_dict = {
            k: p.clone().detach()
            for k, p in self.model.state_dict().items()
            if not k.startswith("lm.")
        }
        self.stored_params = model_state_dict
        # Replace model parameters with EMA parameters
        self.load_state_dict(self.ema.params, strict=False)

    def on_validation_end(self) -> None:
        # Restore original parameters after validation
        if self.stored_params is not None:
            self.load_state_dict(self.stored_params, strict=False)
            self.stored_params = None

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # Remove pretrained model keys
        state_dict = checkpoint["state_dict"]
        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("model.lm.")
        }
        checkpoint["state_dict"] = state_dict

        # Add EMA state dict if EMA is used
        ema_state_dict = self.ema.state_dict()
        checkpoint["ema"] = ema_state_dict

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.config.optimizer.load_opt_state_from_checkpoint is False:
            # When loading optimizer state from checkpoint is disabled,
            # replace the optimizer state in the checkpoint with the initialized state.
            state = checkpoint["optimizer_states"][0]
            init_state = self.configure_optimizers()[0][0].state_dict()
            state["state"] = init_state["state"]
            state["param_groups"][0]["params"] = init_state["param_groups"][0]["params"]
            # checkpoint.pop("lr_schedulers", None)
        self.load_ema_state_dict(checkpoint["ema"]["params"])

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):  # type: ignore[override]
        """Override load_state_dict to handle EMA state dict."""
        # Remove 'model.' prefix from state dict keys if present
        state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
        return self.model.load_state_dict(state_dict, strict=strict)

    def load_ema_state_dict(self, state_dict: Mapping[str, Any]):
        """Load EMA state dict."""
        self.ema.load_state_dict(state_dict)
