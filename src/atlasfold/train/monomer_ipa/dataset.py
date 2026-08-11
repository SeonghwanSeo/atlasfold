"""Monomer IPA datasets with AlphaFold 2 residue cropping."""

from __future__ import annotations

import numpy as np
import torch

from atlasfold.common import featurize, metadata
from atlasfold.train.monomer import dataset as base
from atlasfold.train.monomer_ipa.cropper import MonomerIPACropper


class TrainingDataset(base.TrainingDataset):
    def __init__(
        self,
        config: base.TrainingDatasetConfig,
        max_length: int = 256,
        max_seq_length: int = 384,
        unclamped_fape_probability: float = 0.1,
    ):
        super().__init__(config, max_length, max_seq_length)
        self.cropper = MonomerIPACropper(unclamped_fape_probability)

    def sample_item(self, index: int) -> dict[str, dict[str, torch.Tensor]]:
        rng = np.random.default_rng()

        metadata_dict = self.metadatas[index]
        m = metadata.Metadata.from_dict(metadata_dict)
        prot = self.fetch_protein(m.id)

        feat = featurize.featurize(prot.sequence)
        label = self.prepare_labels(prot)
        loss_mask = self.prepare_loss_masks(m)

        fold_input = {k: v for k, v in feat.items() if not k.startswith("lm.")}
        lm_input = {k: v for k, v in feat.items() if k.startswith("lm.")}

        # AF2 samples the FAPE clamp mode and residue crop jointly.
        crop = self.cropper.crop(prot, self.max_length, rng)
        crop_indices = crop.indices
        fold_input = {k: v[crop_indices] for k, v in fold_input.items()}
        label = {k: v[crop_indices] for k, v in label.items()}

        lm_crop_indices = self._expand_crop_indices_for_lm(
            crop_indices, len(prot), self.max_seq_length
        )
        lm_input = {k: v[lm_crop_indices] for k, v in lm_input.items()}

        is_in_crop = np.isin(lm_input["lm.pos_id"], fold_input["res_idx"])
        fold_input["seq_tok_idx"] = np.where(is_in_crop)[0]

        fold_input = {k: torch.from_numpy(v) for k, v in fold_input.items()}
        lm_input = {k: torch.from_numpy(v) for k, v in lm_input.items()}
        label = {k: torch.from_numpy(v) for k, v in label.items()}
        loss_mask = {k: torch.tensor(v) for k, v in loss_mask.items()}

        fold_input = base.pad_input(fold_input, max_length=self.max_length)
        fold_input["use_clamped_fape"] = torch.tensor(
            float(crop.use_clamped_fape), dtype=torch.float32
        )
        label = base.pad_input(label, max_length=self.max_length)
        lm_input = base.pad_input(lm_input, max_length=self.max_seq_length)
        return {
            "feat": {**fold_input, **lm_input},
            "label": label,
            "loss_mask": loss_mask,
        }


class MultiTrainingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        configs: list[base.TrainingDatasetConfig],
        max_length: int = 256,
        max_seq_length: int = 384,
        unclamped_fape_probability: float = 0.1,
    ) -> None:
        self.datasets = [
            TrainingDataset(
                config,
                max_length,
                max_seq_length,
                unclamped_fape_probability,
            )
            for config in configs
        ]
        ds_weights = [ds.get_sampling_weights() for ds in self.datasets]
        self.weights = np.concatenate(
            [w * cfg.weight for w, cfg in zip(ds_weights, configs, strict=True)]
        )
        self.cumulative_sizes = np.cumsum([len(ds) for ds in self.datasets])

    def __len__(self) -> int:
        return int(self.cumulative_sizes[-1])

    def __getitem__(self, index: int) -> dict[str, dict[str, torch.Tensor]]:
        dataset_idx = int(np.searchsorted(self.cumulative_sizes, index, side="right"))
        if dataset_idx > 0:
            index -= int(self.cumulative_sizes[dataset_idx - 1])
        return self.datasets[dataset_idx][index]
