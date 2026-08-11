"""Train AtlasFold's monomer IPA regression model."""

from __future__ import annotations

import argparse
import dataclasses
import logging
from collections.abc import Mapping
from pathlib import Path

import lightning.pytorch as pl
import lightning.pytorch.callbacks as pl_callbacks
import torch
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig

from atlasfold.train.config import print_config, save_config, to_dict
from atlasfold.train.monomer_ipa.datamodule import TrainingDataModule
from atlasfold.train.monomer_ipa.train_module import TrainingModuleIPA

try:
    from scripts.train_monomer import parse_config
except ModuleNotFoundError:  # Direct execution: python scripts/train_monomer_ipa.py
    from train_monomer import parse_config


ATLASFOLD_INIT_MODULES = (
    "s_init",
    "z_init",
    "z_rel_pos",
    "lm_layer_weights",
    "layernorm_lm_emb",
    "lm_emb_to_s_lm",
    "proj_lm_attn",
    "lm_attn_to_z_lm",
    "lm_stack",
    "proj_s_lm",
    "proj_z_lm",
    "main_stack",
    "distogram_head",
)


@dataclasses.dataclass(frozen=True)
class AtlasFoldInitReport:
    loaded_keys: tuple[str, ...]
    loaded_numel: int


def _is_atlasfold_init_key(key: str) -> bool:
    return any(
        key == name or key.startswith(f"{name}.") for name in ATLASFOLD_INIT_MODULES
    )


def _sync_ema_with_model(module: TrainingModuleIPA) -> None:
    model_params = dict(module.model.named_parameters())
    module.ema.params = {
        key: model_params[key].detach().clone() for key in module.ema.params
    }
    module.ema.device = next(iter(module.ema.params.values())).device


def _initialize_from_atlasfold_weight(
    module: TrainingModuleIPA,
    weight_path: str | Path,
) -> AtlasFoldInitReport:
    """Initialize shared IPA modules from a raw AtlasFold state dict."""
    source_weights = torch.load(
        weight_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(source_weights, Mapping) or not source_weights:
        raise TypeError("AtlasFold weight file must be a non-empty state dict.")
    if not all(isinstance(key, str) for key in source_weights):
        raise TypeError("Every key in the AtlasFold state dict must be a string.")

    target_params = dict(module.model.named_parameters())
    expected_keys = {key for key in target_params if _is_atlasfold_init_key(key)}

    missing_keys = sorted(expected_keys - source_weights.keys())
    if missing_keys:
        raise RuntimeError(
            f"AtlasFold partial initialization is missing expected key(s): {missing_keys}"
        )

    shape_mismatches = []
    for key in sorted(expected_keys):
        source_value = source_weights[key]
        if not isinstance(source_value, torch.Tensor):
            shape_mismatches.append(
                f"{key}: expected Tensor, got {type(source_value).__name__}"
            )
        elif source_value.shape != target_params[key].shape:
            shape_mismatches.append(
                f"{key}: source {tuple(source_value.shape)} != "
                f"target {tuple(target_params[key].shape)}"
            )
    if shape_mismatches:
        raise RuntimeError(
            "AtlasFold partial initialization has incompatible parameter(s): "
            + "; ".join(shape_mismatches)
        )

    selected_weights = {key: source_weights[key] for key in expected_keys}
    incompatible = module.model.load_state_dict(selected_weights, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected key(s) while loading AtlasFold weights: "
            f"{sorted(incompatible.unexpected_keys)}"
        )

    # EMA was constructed before loading. Synchronize both transferred and
    # newly initialized IPA parameters with the current model.
    _sync_ema_with_model(module)

    loaded_keys = tuple(sorted(selected_weights))
    return AtlasFoldInitReport(
        loaded_keys=loaded_keys,
        loaded_numel=sum(target_params[key].numel() for key in loaded_keys),
    )


def initialize_from_ipa_checkpoint(
    module: TrainingModuleIPA,
    checkpoint_path: str | Path,
) -> int:
    """Load IPA model weights while resetting all training state."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("IPA checkpoint must contain a mapping.")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("IPA checkpoint does not contain a non-empty state_dict.")

    module.load_state_dict(state_dict, strict=True)
    _sync_ema_with_model(module)
    return sum(
        parameter.numel()
        for name, parameter in module.model.named_parameters()
        if not name.startswith("lm.")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an AtlasFold IPA model.")
    parser.add_argument(
        "config",
        type=str,
        help="Path to the yaml configuration file.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name of the experiment for logging purposes.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Output root directory for saving logs and checkpoints.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        help="Number of GPUs to use for training.",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        help="Number of nodes to use for training.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of workers to use for dataloader.",
    )
    parser.add_argument(
        "--init_weight",
        help=(
            "Path to a raw AtlasFold model state dict, such as "
            "weights/atlasfold-stage1.pth. Shared input, LM-adapter, trunk, and "
            "distogram parameters are loaded."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="Path to a checkpoint file to resume training from.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode.",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="+",
        help="Override configuration options using 'key=value' format.",
    )
    return parser.parse_args()


def build_trainer(cfg: DictConfig, debug: bool = False) -> pl.Trainer:
    train_cfg = cfg.train
    pl_trainer_cfg = train_cfg.trainer
    save_dir = Path(train_cfg.out_dir) / train_cfg.name

    callbacks = []
    if train_cfg.wandb.use:
        from lightning.pytorch.loggers import WandbLogger

        wandb_logger = WandbLogger(
            name=train_cfg.name,
            project=train_cfg.wandb.project,
            entity=train_cfg.wandb.entity,
            config=to_dict(cfg),
            save_dir=save_dir,
        )
        loggers = [wandb_logger]

        @rank_zero_only
        def _save_config() -> None:
            config_out = Path(wandb_logger.experiment.dir) / "train_config.yaml"
            save_config(cfg, config_out)
            wandb_logger.experiment.save("train_config.yaml")

        _save_config()
    else:
        loggers = None

    # Learning rate monitor
    lr_monitor = pl_callbacks.LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Model summary
    model_summary = pl_callbacks.ModelSummary(max_depth=2)
    callbacks.append(model_summary)

    # TQDM
    tqdm_refresh_rate = 1 if debug else pl_trainer_cfg.log_every_n_steps
    tqdm_callback = pl_callbacks.TQDMProgressBar(refresh_rate=tqdm_refresh_rate)
    callbacks.append(tqdm_callback)

    if not debug:
        checkpoint_callback = pl_callbacks.ModelCheckpoint(
            monitor="val/lddt",
            save_top_k=-1,
            filename="epoch{epoch:04d}_step{step:08d}",
            mode="max",
            auto_insert_metric_name=False,
        )
        callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        default_root_dir=save_dir,
        logger=loggers,
        callbacks=callbacks,
        accelerator=pl_trainer_cfg.accelerator,
        strategy=pl_trainer_cfg.strategy,
        devices=pl_trainer_cfg.devices,
        num_nodes=pl_trainer_cfg.num_nodes,
        precision=pl_trainer_cfg.precision,
        max_epochs=pl_trainer_cfg.max_epochs,
        limit_train_batches=pl_trainer_cfg.limit_train_batches,
        limit_val_batches=pl_trainer_cfg.limit_val_batches,
        log_every_n_steps=pl_trainer_cfg.log_every_n_steps,
        enable_checkpointing=pl_trainer_cfg.enable_checkpointing,
        accumulate_grad_batches=pl_trainer_cfg.accumulate_grad_batches,
        gradient_clip_val=pl_trainer_cfg.gradient_clip_val,
        gradient_clip_algorithm=pl_trainer_cfg.gradient_clip_algorithm,
        use_distributed_sampler=False,
        benchmark=True,
    )
    return trainer


def train(args: argparse.Namespace) -> None:
    torch.set_float32_matmul_precision("high")
    if args.init_weight is not None and args.resume_from_checkpoint is not None:
        raise ValueError(
            "--init_weight and --resume_from_checkpoint are mutually exclusive."
        )
    cfg = parse_config(args)
    trainer = build_trainer(cfg, args.debug)
    is_global_zero = trainer.is_global_zero

    pl.seed_everything(cfg.train.seed, workers=True, verbose=False)

    model_module = TrainingModuleIPA(cfg)
    if args.init_weight is not None:
        report = _initialize_from_atlasfold_weight(model_module, args.init_weight)
        if is_global_zero:
            logging.info(
                "Initialized %d parameters across %d tensors from AtlasFold "
                "weights at %s.",
                report.loaded_numel,
                len(report.loaded_keys),
                args.init_weight,
            )
    ckpt_path = args.resume_from_checkpoint
    if not cfg.train.load_opt_state:
        if ckpt_path is None:
            raise ValueError(
                "--resume_from_checkpoint is required when train.load_opt_state=false."
            )
        loaded_numel = initialize_from_ipa_checkpoint(model_module, ckpt_path)
        if is_global_zero:
            logging.info(
                "Initialized %d IPA model parameters from %s; optimizer, scheduler, "
                "and global step will start fresh.",
                loaded_numel,
                ckpt_path,
            )
        ckpt_path = None
    if is_global_zero:
        print_config(cfg)

    data_module = TrainingDataModule(cfg.train.data)
    trainer.fit(
        model_module,
        datamodule=data_module,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train(parse_args())
