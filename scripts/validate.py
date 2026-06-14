import argparse
import random

import lightning.pytorch as pl
import torch

from atlasfold.train.config import load_config
from atlasfold.train.datamodule import TrainingDataModule
from atlasfold.train.train_module import TrainingModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Co-Folding model.")
    parser.add_argument(
        "config",
        type=str,
        help="Path to the yaml configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a checkpoint file for validation.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=200,
        help="Number of diffusion steps for validation",
    )
    parser.add_argument(
        "--num_recycles", type=int, default=3, help="Number of cycling for validation"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--num_val_entries",
        type=int,
        default=None,
        help="Number of validation samples to use",
    )
    return parser.parse_args()


def validate(args) -> None:
    # To ignore warning
    torch.set_float32_matmul_precision("high")

    cfg = load_config(args.config)

    # Set random seed
    pl.seed_everything(cfg.train.seed, workers=False)

    cfg.train.validation.num_steps = args.num_steps
    cfg.train.validation.num_recycles = args.num_recycles

    if args.debug:
        cfg.train.data.num_workers = 0

    model_module = TrainingModule.load_from_checkpoint(
        args.checkpoint, map_location="cpu", config=cfg, weights_only=True
    )
    data_module = TrainingDataModule(cfg.train.data)

    if args.num_val_entries is not None:
        # Construct the validation dataset
        data_module.setup(stage="validate")

        # Use a fixed random seed to ensure the same subset of validation data
        # is selected across different runs
        rng = random.Random(42)

        num_all_entries = len(data_module._val_ds)
        num_entries = args.num_val_entries
        if num_entries > num_all_entries:
            raise ValueError(
                f"Requested number of validation entries ({num_entries}) exceeds the "
                f"total available ({num_all_entries})."
            )
        # select indices
        selected_indices = rng.sample(range(num_all_entries), num_entries)
        selected_indices.sort()
        data_module._val_ds.metadatas = [
            data_module._val_ds.metadatas[i] for i in selected_indices
        ]

    trainer = pl.Trainer(
        default_root_dir=None,
        logger=False,
        devices=cfg.train.trainer.devices,
        num_nodes=cfg.train.trainer.num_nodes,
        accelerator=cfg.train.trainer.accelerator,
        precision=cfg.train.trainer.precision,
        strategy=cfg.train.trainer.strategy,
        deterministic=True,
        limit_val_batches=5 if args.debug else None,
        enable_checkpointing=False,
    )

    results = trainer.validate(model_module, datamodule=data_module)


if __name__ == "__main__":
    args = parse_args()
    validate(args)
