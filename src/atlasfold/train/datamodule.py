import dataclasses
import gc
import logging

import lightning.pytorch as pl
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .dataset import (
    MultiTrainingDataset,
    TrainingDatasetConfig,
    ValidationDataset,
    ValidationDatasetConfig,
)
from .utils.dl_sampler import DistributedWeightedSampler


class DataModuleConfig:
    # === Common config for data modules === #
    train_batch_size: int = 1
    val_batch_size: int = 1
    num_workers: int = 0
    persistent_workers: bool = True
    pin_memory: bool = True

    # === Training hyperparameters === #
    max_length: int = 256
    max_seq_length: int = 384

    # === Dataset configs === #
    train_datasets: list[TrainingDatasetConfig] = dataclasses.field(default_factory=list)
    val_datasets: list[ValidationDatasetConfig] = dataclasses.field(default_factory=list)


class TrainingDataModule(pl.LightningDataModule):
    _train_ds: MultiTrainingDataset
    _val_ds: ValidationDataset

    def __init__(self, config: DataModuleConfig) -> None:
        super().__init__()
        self.config = config
        self.logger = logging.getLogger("[DataModule]")

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit":
            self._train_ds = self.construct_train_dataset()
            self._val_ds = self.construct_val_dataset()
        elif stage == "validate":
            self._val_ds = self.construct_val_dataset()
        else:
            raise NotImplementedError("Not implemented yet.")
        gc.collect()
        gc.freeze()

    def construct_train_dataset(self) -> MultiTrainingDataset:
        """Construct training dataset."""
        if hasattr(self, "_train_ds"):
            return self._train_ds
        multi_ds = MultiTrainingDataset(
            configs=self.config.train_datasets,
            max_length=self.config.max_length,
            max_seq_length=self.config.max_seq_length,
        )
        # Print dataset info
        for d in multi_ds.datasets:
            self.print_rank_zero(
                f"Constructed training dataset '{d.name}':\n"
                f"  Weights: {d.config.weight}\n"
                f"  Num complexes: {len(d.metadatas)}\n"
                f"  Num samples: {len(d)}"
            )
        return multi_ds

    def construct_val_dataset(self) -> ValidationDataset:
        """Construct validation dataset."""
        if hasattr(self, "_val_ds"):
            return self._val_ds

        if len(self.config.val_datasets) != 1:
            raise NotImplementedError(
                "Currently only single validation dataset is supported."
            )
        ds = ValidationDataset(
            config=self.config.val_datasets[0],
        )
        self.print_rank_zero(
            f"Constructed validation dataset '{ds.name}':\n"
            f"  Num complexes: {len(ds.metadatas)}\n"
        )
        return ds

    def train_dataloader(self):
        dataset = self._train_ds

        sampler = DistributedWeightedSampler(
            weights=dataset.weights,
            rank=self.trainer.global_rank if self.trainer else 0,
            world_size=self.trainer.world_size if self.trainer else 1,
            epoch=self.trainer.current_epoch if self.trainer else 0,
            replacement=True,
        )
        persistent_workers = (
            self.config.persistent_workers and self.config.num_workers > 0
        )
        return DataLoader(
            dataset,
            batch_size=self.config.train_batch_size,
            shuffle=False,
            sampler=sampler,
            drop_last=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        dataset = self._val_ds
        sampler = None
        if self.trainer is not None:
            if self.trainer.world_size > 1:
                sampler = DistributedSampler(
                    dataset,
                    rank=self.trainer.global_rank,
                    num_replicas=self.trainer.world_size,
                    shuffle=False,
                    drop_last=False,
                )

        return DataLoader(
            dataset,
            batch_size=None,  # do not batch validation samples together
            sampler=sampler,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=False,
            persistent_workers=False,
        )

    def print_rank_zero(self, msg: str) -> None:
        if self.trainer is None or self.trainer.global_rank == 0:
            self.logger.info(f"{msg}")
