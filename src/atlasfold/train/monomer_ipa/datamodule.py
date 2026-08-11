"""Monomer IPA data module."""

from atlasfold.train.monomer.datamodule import *  # noqa: F403
from atlasfold.train.monomer.datamodule import TrainingDataModule as BaseDataModule
from atlasfold.train.monomer_ipa.dataset import MultiTrainingDataset


class TrainingDataModule(BaseDataModule):
    def __init__(
        self,
        config,
        unclamped_fape_probability: float = 0.1,
    ) -> None:
        super().__init__(config)
        self.unclamped_fape_probability = float(unclamped_fape_probability)

    def construct_train_dataset(self) -> MultiTrainingDataset:
        if hasattr(self, "_train_ds"):
            return self._train_ds
        multi_ds = MultiTrainingDataset(
            configs=self.config.train_datasets,
            max_length=self.config.max_length,
            max_seq_length=self.config.max_seq_length,
            unclamped_fape_probability=self.unclamped_fape_probability,
        )
        for dataset in multi_ds.datasets:
            self.print_rank_zero(
                f"Constructed IPA training dataset '{dataset.name}':\n"
                f"  Weights: {dataset.config.weight}\n"
                f"  Num complexes: {len(dataset.metadatas)}\n"
                f"  Num samples: {len(dataset)}"
            )
        return multi_ds
