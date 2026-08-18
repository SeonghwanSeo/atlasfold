"""Train AtlasFold's monomer IPA regression model."""

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

from atlasfold.train.config import load_config, print_config, save_config, to_dict
from atlasfold.train.monomer_ipa.datamodule import TrainingDataModule
from atlasfold.train.monomer_ipa.train_module import TrainingModuleIPA


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
)


def _initialize_from_atlasfold_weight(
    module: TrainingModuleIPA,
    weight_path: str | Path,
) -> int:
    """Initialize shared IPA parameters from AtlasFold weights."""
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
    expected_keys = {
        key
        for key in target_params
        if any(
            key == name or key.startswith(f"{name}.")
            for name in ATLASFOLD_INIT_MODULES
        )
    }
    missing_keys = sorted(expected_keys - source_weights.keys())
    if missing_keys:
        raise RuntimeError(
            "AtlasFold partial initialization is missing expected key(s): "
            f"{missing_keys}"
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

    model_params = dict(module.model.named_parameters())
    module.ema.params = {
        key: model_params[key].detach().clone() for key in module.ema.params
    }
    module.ema.device = next(iter(module.ema.params.values())).device
    return sum(target_params[key].numel() for key in selected_weights)


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
            "weights/atlasfold-stage1.pth. Shared input, LM-adapter, and trunk "
            "parameters are loaded."
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


def parse_config(args: argparse.Namespace) -> DictConfig:
    cfg = load_config(args.config, override_args=args.override)

    if args.out_dir is not None:
        cfg.train.out_dir = args.out_dir
    if args.experiment_name is not None:
        cfg.train.name = args.experiment_name
    if args.num_gpus is not None:
        cfg.train.trainer.devices = args.num_gpus
    if args.num_nodes is not None:
        cfg.train.trainer.num_nodes = args.num_nodes
    if args.num_workers is not None:
        cfg.train.data.num_workers = args.num_workers
    if args.compile:
        cfg.train.compile.enabled = True
    if args.wandb:
        cfg.train.wandb.use = True

    apply_global_hparams_overrides(cfg)

    if args.debug:
        print("Debug mode is enabled: Single GPU, 0 workers, no wandb.")
        cfg.train.trainer.devices = 1
        cfg.train.trainer.num_nodes = 1
        cfg.train.trainer.accumulate_grad_batches = 1
        cfg.train.trainer.log_every_n_steps = 1
        cfg.train.trainer.limit_train_batches = 50
        cfg.train.trainer.limit_val_batches = 10
        cfg.train.trainer.enable_checkpointing = True
        cfg.train.data.num_workers = 0
        cfg.train.wandb.use = False

    return cfg


def apply_global_hparams_overrides(cfg: DictConfig) -> None:
    train_cfg = cfg.train
    if train_cfg.trainer.devices == "auto":
        num_gpus: int = torch.cuda.device_count()
    else:
        num_gpus = cfg.train.trainer.devices
    assert isinstance(num_gpus, int), "num_gpus should be an integer or 'auto'."
    assert num_gpus > 0, "No GPUs available for training."
    world_size: int = train_cfg.trainer.num_nodes * num_gpus

    batch_size: int = train_cfg.data.batch_size
    global_batch_size: int = train_cfg.data.global_batch_size
    if global_batch_size % (batch_size * world_size) != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} is not "
            f"divisible by (batch_size {batch_size} * world_size {world_size})"
        )
    accumulate_grad_batches: int = global_batch_size // (batch_size * world_size)
    train_cfg.trainer.accumulate_grad_batches = accumulate_grad_batches
    train_cfg.trainer.limit_train_batches *= accumulate_grad_batches


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
    data_module = TrainingDataModule(cfg.train.data, seed=cfg.train.seed)
    if args.init_weight is not None:
        loaded_numel = _initialize_from_atlasfold_weight(
            model_module, args.init_weight
        )
        if is_global_zero:
            logging.info(
                "Initialized %d parameters from AtlasFold weights at %s.",
                loaded_numel,
                args.init_weight,
            )
    if is_global_zero:
        print_config(cfg)

    if cfg.train.load_opt_state:
        trainer.fit(
            model_module,
            datamodule=data_module,
            ckpt_path=args.resume_from_checkpoint,
        )
    else:
        # Manually load the model state dict without optimizer state dict.
        assert args.resume_from_checkpoint is not None, (
            "resume_from_checkpoint must be specified if load_opt_state is False."
        )
        ckpt = torch.load(args.resume_from_checkpoint, map_location="cpu")
        model_module.load_state_dict(ckpt["state_dict"], strict=False)
        model_module.ema.load_state_dict(ckpt["ema"], strict=False)
        model_module.last_lr_step = ckpt["global_step"]
        trainer.fit_loop.load_state_dict(ckpt["loops"]["fit_loop"])
        trainer.fit(model_module, datamodule=data_module)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train(parse_args())
