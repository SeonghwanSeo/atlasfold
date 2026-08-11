"""Multimer IPA datasets with AFM cropping and permutation metadata."""

import numpy as np
import torch

from atlasfold.train.multimer import dataset as base
from atlasfold.train.multimer import train_alignment
from atlasfold.train.multimer_ipa.cropper import MultimerIPACropper


class RCSBTrainingDataset(base.RCSBTrainingDataset):
    """RCSB training dataset with AFM-style cropping."""

    def __init__(
        self,
        config: base.TrainingDatasetConfig,
        max_length: int = 384,
        max_seq_length: int = 768,
        max_templates: int = 2,
        max_contiguous_chains: int = 6,
    ):
        super().__init__(
            config,
            max_length,
            max_seq_length,
            max_templates,
            max_contiguous_chains,
        )
        self.cropper = MultimerIPACropper(max_contiguous_chains)


class MultimerDistillationDataset(base.MultimerDistillationDataset):
    """Multimer distillation dataset with AFM-style cropping."""

    def __init__(
        self,
        config: base.TrainingDatasetConfig,
        max_length: int = 384,
        max_seq_length: int = 768,
        max_templates: int = 0,
        max_contiguous_chains: int = 6,
    ):
        super().__init__(
            config,
            max_length,
            max_seq_length,
            max_templates,
            max_contiguous_chains,
        )
        self.cropper = MultimerIPACropper(max_contiguous_chains)


class MonomerTrainingDataset(base.MonomerTrainingDataset):
    """Monomer distillation dataset with AFM-style cropping."""

    def __init__(
        self,
        config: base.TrainingDatasetConfig,
        max_length: int = 384,
        max_seq_length: int = 768,
        max_templates: int = 0,
        max_contiguous_chains: int = 6,
    ):
        super().__init__(
            config,
            max_length,
            max_seq_length,
            max_templates,
            max_contiguous_chains,
        )
        self.cropper = MultimerIPACropper(max_contiguous_chains)


class MultiTrainingDataset(torch.utils.data.Dataset):
    """Use IPA datasets and add chain-alignment metadata."""

    def __init__(
        self,
        configs: list[base.TrainingDatasetConfig],
        max_length: int = 384,
        max_seq_length: int = 768,
        max_templates: int = 2,
        max_contiguous_chains: int = 6,
    ) -> None:
        dataset_classes = {
            "rcsb": RCSBTrainingDataset,
            "multimer_distillation": MultimerDistillationDataset,
            "monomer_distillation": MonomerTrainingDataset,
        }
        self.datasets: list[base.TrainingDataset] = []
        for config in configs:
            try:
                dataset_class = dataset_classes[config.type]
            except KeyError as error:
                raise ValueError(
                    f"Unsupported dataset type '{config.type}' "
                    f"for dataset '{config.name}'."
                ) from error
            self.datasets.append(
                dataset_class(
                    config,
                    max_length,
                    max_seq_length,
                    max_templates,
                    max_contiguous_chains,
                )
            )

        dataset_weights = [dataset.get_sampling_weights() for dataset in self.datasets]
        self.weights = np.concatenate(
            [
                weight * config.weight
                for weight, config in zip(dataset_weights, configs, strict=True)
            ]
        )
        self.cumulative_sizes = np.cumsum([len(dataset) for dataset in self.datasets])

    def __len__(self) -> int:
        return int(self.cumulative_sizes[-1])

    def __getitem__(self, index: int) -> dict[str, dict[str, torch.Tensor]]:
        """Return an item and attach metadata used for chain alignment."""
        dataset_index = int(np.searchsorted(self.cumulative_sizes, index, side="right"))
        if dataset_index > 0:
            index -= int(self.cumulative_sizes[dataset_index - 1])
        sample = self.datasets[dataset_index][index]
        full_label = sample["full_label"]
        if "alignment_metadata" not in full_label:
            full_label["alignment_metadata"] = train_alignment.prepare_alignment_metadata(
                full_label
            )
        return sample
