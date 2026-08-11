"""Multimer IPA data module."""

from atlasfold.train.multimer.datamodule import *  # noqa: F403
from atlasfold.train.multimer.datamodule import TrainingDataModule as BaseDataModule
from atlasfold.train.multimer_ipa.dataset import MultiTrainingDataset


class TrainingDataModule(BaseDataModule):
    def construct_train_dataset(self) -> MultiTrainingDataset:
        if hasattr(self, "_train_ds"):
            return self._train_ds
        multi_ds = MultiTrainingDataset(
            configs=self.config.train_datasets,
            max_length=self.config.max_length,
            max_seq_length=self.config.max_seq_length,
            max_templates=self.config.max_templates,
            max_contiguous_chains=self.config.max_contiguous_chains,
        )
        for dataset in multi_ds.datasets:
            self.print_rank_zero(
                f"Constructed IPA training dataset '{dataset.name}':\n"
                f"  Weights: {dataset.config.weight}\n"
                f"  Num complexes: {len(dataset.metadatas)}\n"
                f"  Num samples: {len(dataset)}"
            )
        return multi_ds
