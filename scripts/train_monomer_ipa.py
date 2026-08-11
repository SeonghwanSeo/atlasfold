"""Train AtlasFold's monomer IPA regression model."""

from __future__ import annotations

import argparse
import logging
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AtlasFold IPA.")
    parser.add_argument("config")
    parser.add_argument("--experiment_name")
    parser.add_argument("--out_dir")
    parser.add_argument("--num_gpus", type=int)
    parser.add_argument("--num_nodes", type=int)
    parser.add_argument("--num_workers", type=int)
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
    cfg = parse_config(args)
    trainer = build_trainer(cfg, args.debug)
    pl.seed_everything(cfg.train.seed, workers=True, verbose=False)
    module = TrainingModuleIPA(cfg)
    if trainer.is_global_zero:
        print_config(cfg)
    trainer.fit(
        module,
        datamodule=TrainingDataModule(
            cfg.train.data,
            unclamped_fape_probability=cfg.train.data.unclamped_fape_probability,
        ),
        ckpt_path=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train(parse_args())
