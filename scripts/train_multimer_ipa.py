"""Train AtlasFold's multimer IPA regression model."""

from __future__ import annotations

import logging

import lightning.pytorch as pl
import torch

from atlasfold.train.config import print_config
from atlasfold.train.multimer_ipa.datamodule import TrainingDataModule
from atlasfold.train.multimer_ipa.train_module import TrainingModuleIPA

try:
    from scripts.train_monomer_ipa import (
        build_trainer,
        parse_args,
        parse_config,
    )
except ModuleNotFoundError:  # Direct execution
    from train_monomer_ipa import (
        build_trainer,
        parse_args,
        parse_config,
    )


def train(args) -> None:
    torch.set_float32_matmul_precision("high")
    cfg = parse_config(args)
    trainer = build_trainer(cfg, args.debug, multimer=True)
    pl.seed_everything(cfg.train.seed, workers=True, verbose=False)
    module = TrainingModuleIPA(cfg)
    if trainer.is_global_zero:
        print_config(cfg)
    trainer.fit(
        module,
        datamodule=TrainingDataModule(cfg.train.data),
        ckpt_path=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train(parse_args())
