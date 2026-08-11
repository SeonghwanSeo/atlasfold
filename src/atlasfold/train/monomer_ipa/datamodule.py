"""Monomer IPA data module."""

from omegaconf import DictConfig, OmegaConf

from atlasfold.train.monomer.datamodule import TrainingDataModule as BaseDataModule
from atlasfold.train.monomer_ipa.dataset import MultiTrainingDataset


class TrainingDataModule(BaseDataModule):
    def __init__(self, config: DictConfig) -> None:
        # Remove the IPA-only option before the base structured-config merge.
        base_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        self.unclamped_fape_probability = float(
            base_config.pop("unclamped_fape_probability", 0.1)
        )
        super().__init__(base_config)

    def construct_train_dataset(self) -> MultiTrainingDataset:
        """Construct training dataset."""
        if hasattr(self, "_train_ds"):
            return self._train_ds
        multi_ds = MultiTrainingDataset(
            configs=self.config.train_datasets,
            max_length=self.config.max_length,
            max_seq_length=self.config.max_seq_length,
            unclamped_fape_probability=self.unclamped_fape_probability,
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
