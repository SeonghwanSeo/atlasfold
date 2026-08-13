"""Monomer IPA data module."""

from omegaconf import DictConfig, OmegaConf

from atlasfold.train.monomer.datamodule import TrainingDataModule as BaseDataModule
from atlasfold.train.monomer_ipa.cropper import CropperConfig
from atlasfold.train.monomer_ipa.dataset import MultiTrainingDataset


class TrainingDataModule(BaseDataModule):
    def __init__(self, config: DictConfig) -> None:
        # Remove the IPA-only cropper block before the base structured-config merge.
        base_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        self.cropper_config: CropperConfig = OmegaConf.to_object(
            OmegaConf.merge(
                OmegaConf.structured(CropperConfig),
                base_config.pop("cropper"),
            )
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
            cropper_config=self.cropper_config,
            resample_threshold=self.config.resample_threshold,
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
