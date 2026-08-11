"""Train AtlasFold's monomer IPA regression model."""

from __future__ import annotations

import argparse
import dataclasses
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import lightning.pytorch.callbacks as callbacks
import torch

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
    parser = argparse.ArgumentParser(description="Train AtlasFold IPA.")
    parser.add_argument("config")
    parser.add_argument("--experiment_name")
    parser.add_argument("--out_dir")
    parser.add_argument("--num_gpus", type=int)
    parser.add_argument("--num_nodes", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument(
        "--init_weight",
        help=(
            "Path to a raw AtlasFold model state dict, such as "
            "weights/atlasfold-stage1.pth. Shared input, LM-adapter, trunk, and "
            "distogram parameters are loaded."
        ),
    )
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--override", nargs="+")
    return parser.parse_args()


def build_trainer(cfg, debug: bool = False, multimer: bool = False) -> pl.Trainer:
    train_cfg = cfg.train
    save_dir = Path(train_cfg.out_dir) / train_cfg.name
    logger: Any = None
    if train_cfg.wandb.use:
        from lightning.pytorch.loggers import WandbLogger

        logger = WandbLogger(
            name=train_cfg.name,
            project=train_cfg.wandb.project,
            entity=train_cfg.wandb.entity,
            config=to_dict(cfg),
            save_dir=save_dir,
        )
        save_config(cfg, Path(logger.experiment.dir) / "train_config.yaml")
    monitor = "val/complex/lddt" if multimer else "val/lddt"
    cbs = [
        callbacks.LearningRateMonitor(logging_interval="step"),
        callbacks.ModelSummary(max_depth=2),
        callbacks.TQDMProgressBar(
            refresh_rate=1 if debug else train_cfg.trainer.log_every_n_steps
        ),
    ]
    if not debug:
        cbs.append(
            callbacks.ModelCheckpoint(
                monitor=monitor,
                save_top_k=-1,
                filename="epoch{epoch:04d}_step{step:08d}",
                mode="max",
                auto_insert_metric_name=False,
            )
        )
    tc = train_cfg.trainer
    return pl.Trainer(
        default_root_dir=save_dir,
        logger=logger,
        callbacks=cbs,
        accelerator=tc.accelerator,
        strategy=tc.strategy,
        devices=tc.devices,
        num_nodes=tc.num_nodes,
        precision=tc.precision,
        max_epochs=tc.max_epochs,
        limit_train_batches=tc.limit_train_batches,
        limit_val_batches=tc.limit_val_batches,
        log_every_n_steps=tc.log_every_n_steps,
        enable_checkpointing=tc.enable_checkpointing,
        accumulate_grad_batches=tc.accumulate_grad_batches,
        gradient_clip_val=tc.gradient_clip_val,
        gradient_clip_algorithm=tc.gradient_clip_algorithm,
        use_distributed_sampler=False,
        benchmark=True,
    )


def train(args: argparse.Namespace) -> None:
    torch.set_float32_matmul_precision("high")
    if args.init_weight is not None and args.resume_from_checkpoint is not None:
        raise ValueError(
            "--init_weight and --resume_from_checkpoint are mutually exclusive."
        )
    cfg = parse_config(args)
    trainer = build_trainer(cfg, args.debug)
    pl.seed_everything(cfg.train.seed, workers=True, verbose=False)
    module = TrainingModuleIPA(cfg)
    if args.init_weight is not None:
        report = _initialize_from_atlasfold_weight(module, args.init_weight)
        if trainer.is_global_zero:
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
        loaded_numel = initialize_from_ipa_checkpoint(module, ckpt_path)
        if trainer.is_global_zero:
            logging.info(
                "Initialized %d IPA model parameters from %s; optimizer, scheduler, "
                "and global step will start fresh.",
                loaded_numel,
                ckpt_path,
            )
        ckpt_path = None
    if trainer.is_global_zero:
        print_config(cfg)
    trainer.fit(
        module,
        datamodule=TrainingDataModule(
            cfg.train.data,
            unclamped_fape_probability=cfg.train.data.unclamped_fape_probability,
        ),
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train(parse_args())
