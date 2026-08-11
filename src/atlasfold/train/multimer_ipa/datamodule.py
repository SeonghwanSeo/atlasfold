"""Multimer IPA data module."""

from atlasfold.train.multimer.datamodule import TrainingDataModule as BaseDataModule
from atlasfold.train.multimer_ipa.dataset import MultiTrainingDataset


class TrainingDataModule(BaseDataModule):
    def construct_train_dataset(self) -> MultiTrainingDataset:
        """Construct training dataset."""
        if hasattr(self, "_train_ds"):
            return self._train_ds
        multi_ds = MultiTrainingDataset(
            configs=self.config.train_datasets,
            max_length=self.config.max_length,
            max_seq_length=self.config.max_seq_length,
            max_templates=self.config.max_templates,
            max_contiguous_chains=self.config.max_contiguous_chains,
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
