"""Train AtlasFold's multimer IPA regression model."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from pathlib import Path

import lightning.pytorch as pl
import lightning.pytorch.callbacks as pl_callbacks
import torch
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig

from atlasfold.train.config import print_config, save_config, to_dict
from atlasfold.train.multimer_ipa.datamodule import TrainingDataModule
from atlasfold.train.multimer_ipa.train_module import TrainingModuleIPA

try:
    from scripts.train_monomer_ipa import initialize_from_ipa_checkpoint
    from scripts.train_multimer import parse_config
except ModuleNotFoundError:  # Direct execution
    from train_monomer_ipa import initialize_from_ipa_checkpoint
    from train_multimer import parse_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an AtlasFold multimer IPA model.")
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
        type=str,
        help=(
            "Path to a monomer IPA checkpoint or raw state dict. All shared model "
            "parameters are loaded and the multimer template module remains newly "
            "initialized."
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

    callbacks.append(pl_callbacks.LearningRateMonitor(logging_interval="step"))
    callbacks.append(pl_callbacks.ModelSummary(max_depth=2))

    tqdm_refresh_rate = 1 if debug else pl_trainer_cfg.log_every_n_steps
    callbacks.append(pl_callbacks.TQDMProgressBar(refresh_rate=tqdm_refresh_rate))

    if not debug:
        checkpoint_callback = pl_callbacks.ModelCheckpoint(
            monitor="val/complex/lddt",
            save_top_k=-1,
            filename="epoch{epoch:04d}_step{step:08d}",
            mode="max",
            auto_insert_metric_name=False,
        )
        callbacks.append(checkpoint_callback)

    return pl.Trainer(
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


def _sync_ema_with_model(module: TrainingModuleIPA) -> None:
    model_params = dict(module.model.named_parameters())
    module.ema.params = {
        key: model_params[key].detach().clone() for key in module.ema.params
    }
    module.ema.device = next(iter(module.ema.params.values())).device


def _initialize_from_monomer_ipa_weight(
    module: TrainingModuleIPA,
    weight_path: str | Path,
) -> int:
    """Initialize shared multimer parameters from monomer IPA weights."""
    source = torch.load(
        weight_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(source, Mapping) or not source:
        raise TypeError("Monomer IPA weight file must contain a non-empty mapping.")

    state_dict = source.get("state_dict", source)
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("Monomer IPA checkpoint has no non-empty state_dict.")

    source_weights = {
        key.removeprefix("model."): value
        for key, value in state_dict.items()
        if isinstance(key, str)
    }
    target_weights = module.model.state_dict()
    expected_keys = {
        key for key in target_weights if not key.startswith(("lm.", "template_module."))
    }
    missing_keys = sorted(expected_keys - source_weights.keys())
    if missing_keys:
        raise RuntimeError(
            f"Monomer IPA initialization is missing shared parameter(s): {missing_keys}"
        )

    selected_weights = {}
    shape_mismatches = []
    for key in sorted(expected_keys):
        value = source_weights[key]
        if not isinstance(value, torch.Tensor):
            shape_mismatches.append(f"{key}: expected Tensor, got {type(value).__name__}")
        elif value.shape != target_weights[key].shape:
            shape_mismatches.append(
                f"{key}: source {tuple(value.shape)} != "
                f"target {tuple(target_weights[key].shape)}"
            )
        else:
            selected_weights[key] = value
    if shape_mismatches:
        raise RuntimeError(
            "Monomer IPA initialization has incompatible parameter(s): "
            + "; ".join(shape_mismatches)
        )

    module.model.load_state_dict(selected_weights, strict=False)
    _sync_ema_with_model(module)
    return sum(target_weights[key].numel() for key in selected_weights)


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
        loaded_numel = _initialize_from_monomer_ipa_weight(model_module, args.init_weight)
        if is_global_zero:
            logging.info(
                "Initialized %d shared multimer parameters from monomer IPA "
                "weights at %s.",
                loaded_numel,
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
