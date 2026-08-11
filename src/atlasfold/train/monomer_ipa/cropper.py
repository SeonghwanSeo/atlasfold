"""AlphaFold 2 residue cropping for monomer IPA training."""

from __future__ import annotations

import dataclasses

import numpy as np

from atlasfold.common import protein
from atlasfold.utils.misc import spawn_rng


@dataclasses.dataclass(frozen=True)
class MonomerCrop:
    indices: np.ndarray
    use_clamped_fape: bool


class MonomerIPACropper:
    """Sample the AF2 FAPE mode and its coupled contiguous residue crop."""

    def __init__(self, unclamped_fape_probability: float = 0.1):
        if not 0.0 <= unclamped_fape_probability <= 1.0:
            raise ValueError("unclamped_fape_probability must be between 0 and 1.")
        self.unclamped_fape_probability = float(unclamped_fape_probability)

    def crop(
        self,
        prot: protein.Protein,
        max_length: int,
        rng: np.random.Generator | None = None,
    ) -> MonomerCrop:
        if max_length <= 0:
            raise ValueError("max_length must be positive.")
        rng = spawn_rng(rng)
        use_clamped_fape = bool(rng.random() >= self.unclamped_fape_probability)

        crop_size = min(len(prot), max_length)
        num_crop_starts = len(prot) - crop_size
        if use_clamped_fape:
            right_anchor = num_crop_starts
        else:
            # AF2 Supplement 1.2.8: x ~ U[0, n], start ~ U[0, n - x].
            x = int(rng.integers(0, num_crop_starts + 1))
            right_anchor = num_crop_starts - x

        crop_start = int(rng.integers(0, right_anchor + 1))
        indices = np.arange(crop_start, crop_start + crop_size, dtype=np.int64)
        return MonomerCrop(indices=indices, use_clamped_fape=use_clamped_fape)
