"""AlphaFold 2 residue cropping for monomer IPA training."""

import dataclasses
from typing import NamedTuple

import numpy as np

from atlasfold.common import protein
from atlasfold.utils.misc import spawn_rng


@dataclasses.dataclass(kw_only=True)
class CropperConfig:
    """Configuration for monomer IPA crop and FAPE-mode sampling."""

    prob_contiguous: float = 1.0
    prob_multi_contiguous: float = 0.0
    unclamped_fape_probability: float = 0.1
    max_num_segments: int = 4
    min_segment_length: int = 64


class MonomerCrop(NamedTuple):
    indices: np.ndarray
    use_clamped_fape: bool


class MonomerIPACropper:
    """Sample the AF2 FAPE mode and a contiguous or multi-segment crop."""

    def __init__(
        self,
        config: CropperConfig | None = None,
    ) -> None:
        config = config or CropperConfig()
        if not 0.0 <= config.prob_contiguous <= 1.0:
            raise ValueError("prob_contiguous must be between 0 and 1.")
        if not 0.0 <= config.prob_multi_contiguous <= 1.0:
            raise ValueError("prob_multi_contiguous must be between 0 and 1.")
        if not np.isclose(
            config.prob_contiguous + config.prob_multi_contiguous,
            1.0,
        ):
            raise ValueError("Crop probabilities must sum to 1.")
        if not 0.0 <= config.unclamped_fape_probability <= 1.0:
            raise ValueError("unclamped_fape_probability must be between 0 and 1.")
        if config.max_num_segments < 2:
            raise ValueError("max_num_segments must be at least 2.")
        if config.min_segment_length <= 0:
            raise ValueError("min_segment_length must be positive.")
        self.prob_contiguous = float(config.prob_contiguous)
        self.prob_multi_contiguous = float(config.prob_multi_contiguous)
        self.unclamped_fape_probability = float(config.unclamped_fape_probability)
        self.max_num_segments = int(config.max_num_segments)
        self.min_segment_length = int(config.min_segment_length)

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

        length = len(prot)

        if length <= max_length:
            # If the protein is shorter than the max length, return all indices.
            indices = np.arange(length, dtype=np.int64)
            return MonomerCrop(indices, use_clamped_fape)

        if self.prob_multi_contiguous > 0.0 and rng.random() < self.prob_multi_contiguous:
            # Multi-segment cropping is only possible
            indices = self._crop_multi_contiguous(length, max_length, rng)
            if indices is not None:
                return MonomerCrop(indices, use_clamped_fape)

        indices = self._crop_contiguous(length, max_length, use_clamped_fape, rng)
        return MonomerCrop(indices, use_clamped_fape)

    def _crop_contiguous(
        self,
        sequence_length: int,
        crop_size: int,
        use_clamped_fape: bool,
        rng: np.random.Generator,
    ) -> np.ndarray:
        num_crop_starts = sequence_length - crop_size
        if use_clamped_fape:
            right_anchor = num_crop_starts
        else:
            # AF2 Supplement 1.2.8: x ~ U[0, n], start ~ U[0, n - x].
            x = int(rng.integers(0, num_crop_starts + 1))
            right_anchor = num_crop_starts - x

        st = int(rng.integers(0, right_anchor + 1))
        return np.arange(st, st + crop_size, dtype=np.int64)

    def _crop_multi_contiguous(
        self,
        sequence_length: int,
        crop_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        """Sample two or more contiguous fragments spread across the sequence."""

        max_segments = min(
            self.max_num_segments,
            sequence_length // self.min_segment_length,
            crop_size // self.min_segment_length,
        )
        if max_segments < 2:
            return None

        num_segments = int(rng.integers(2, max_segments + 1))
        base_bin_size, remainder = divmod(sequence_length, num_segments)
        bin_sizes = [base_bin_size] * num_segments
        bin_sizes[-1] += remainder

        crop_sizes = [self.min_segment_length] * num_segments
        remaining = crop_size - num_segments * self.min_segment_length
        available = [bin_size - self.min_segment_length for bin_size in bin_sizes]
        while remaining > 0:
            candidates = [
                index for index, capacity in enumerate(available) if capacity > 0
            ]
            if not candidates:
                raise RuntimeError("Multi-segment crop cannot use the requested budget.")
            index = int(rng.choice(candidates))
            amount = int(rng.integers(1, min(remaining, available[index]) + 1))
            crop_sizes[index] += amount
            available[index] -= amount
            remaining -= amount

        segments = []
        bin_start = 0
        for bin_size, segment_size in zip(bin_sizes, crop_sizes, strict=True):
            offset = int(rng.integers(0, bin_size - segment_size + 1))
            segment_start = bin_start + offset
            segments.append(
                np.arange(segment_start, segment_start + segment_size, dtype=np.int64)
            )
            bin_start += bin_size

        indices = np.concatenate(segments)
        if len(indices) != crop_size or np.any(np.diff(indices) <= 0):
            raise RuntimeError("Invalid multi-segment crop was generated.")
        return indices
