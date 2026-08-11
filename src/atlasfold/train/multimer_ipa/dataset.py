"""Multimer IPA datasets with permutation metadata on every sample."""

from atlasfold.train.multimer import dataset as base
from atlasfold.train.multimer import train_alignment
from atlasfold.train.multimer.dataset import *  # noqa: F403


class MultiTrainingDataset(base.MultiTrainingDataset):
    """Add chain-alignment metadata without changing diffusion datasets."""

    def __getitem__(self, index: int):
        sample = super().__getitem__(index)
        full_label = sample["full_label"]
        if "alignment_metadata" not in full_label:
            full_label["alignment_metadata"] = train_alignment.prepare_alignment_metadata(
                full_label
            )
        return sample
