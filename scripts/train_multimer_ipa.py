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
        initialize_from_ipa_checkpoint,
        parse_args,
        parse_config,
    )
except ModuleNotFoundError:  # Direct execution
    from train_monomer_ipa import (
        build_trainer,
        initialize_from_ipa_checkpoint,
        parse_args,
        parse_config,
    )


def train(args) -> None:
    torch.set_float32_matmul_precision("high")
    cfg = parse_config(args)
    trainer = build_trainer(cfg, args.debug, multimer=True)
    pl.seed_everything(cfg.train.seed, workers=True, verbose=False)
    module = TrainingModuleIPA(cfg)
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
        datamodule=TrainingDataModule(cfg.train.data),
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train(parse_args())
