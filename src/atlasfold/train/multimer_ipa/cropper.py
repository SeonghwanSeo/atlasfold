"""AlphaFold-Multimer cropping for multimer IPA training."""

from __future__ import annotations

import numpy as np

from atlasfold.common import metadata, protein
from atlasfold.train.multimer.cropper import MultimerCropper


class MultimerIPACropper(MultimerCropper):
    """Use the AFM 50:50 contiguous/interface-spatial crop policy."""

    interface_threshold = 10.0

    def __init__(self, max_contiguous_chains: int = 6):
        super().__init__(
            prob_spatial=0.0,
            prob_interface_spatial=0.5,
            prob_contiguous=0.5,
            max_contiguous_chains=max_contiguous_chains,
            # AFM does not discard short per-chain fragments after allocation.
            min_cropped_chain_length=0,
        )

    def _crop_interface_spatial(
        self,
        compl: protein.ProteinMultimer,
        m: metadata.MultimerMetadata,
        max_length: int,
        bias_chain_id: int | tuple[int, int] | None,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        del m, bias_chain_id  # AFM samples the anchor from all observed interfaces.

        def fallback() -> list[np.ndarray]:
            # AFM has no general-spatial third strategy.
            return self._crop_contiguous(compl, max_length, rng)

        ca_coords_list = [chain.coordinates[:, 1, :] for chain in compl.chains]
        ca_mask_list = [np.isfinite(coords).all(axis=-1) for coords in ca_coords_list]
        interface_residue_masks = [np.zeros_like(mask) for mask in ca_mask_list]
        for chain1_id in range(len(compl.chains)):
            for chain2_id in range(chain1_id + 1, len(compl.chains)):
                chain1_coords = ca_coords_list[chain1_id]
                chain2_coords = ca_coords_list[chain2_id]
                chain1_mask = ca_mask_list[chain1_id]
                chain2_mask = ca_mask_list[chain2_id]
                distances = np.linalg.norm(
                    chain1_coords[:, None, :] - chain2_coords[None, :, :], axis=-1
                )
                interface_mask = (
                    (distances < self.interface_threshold)
                    & chain1_mask[:, None]
                    & chain2_mask[None, :]
                )
                interface_residue_masks[chain1_id] |= interface_mask.any(axis=1)
                interface_residue_masks[chain2_id] |= interface_mask.any(axis=0)

        anchor_counts = np.asarray(
            [int(mask.sum()) for mask in interface_residue_masks], dtype=np.int64
        )
        if anchor_counts.sum() == 0:
            return fallback()

        anchor_offset = int(rng.integers(0, int(anchor_counts.sum())))
        chain_id = int(
            np.searchsorted(np.cumsum(anchor_counts), anchor_offset, side="right")
        )
        previous_count = int(anchor_counts[:chain_id].sum())
        residue_id = np.where(interface_residue_masks[chain_id])[0][
            anchor_offset - previous_count
        ]
        anchor_coord = ca_coords_list[chain_id][residue_id]

        return self.get_closest_residues(
            ca_coords_list,
            ca_mask_list,
            anchor_coord,
            max_length,
        )

    @staticmethod
    def get_closest_residues(
        coords_list: list[np.ndarray],
        mask_list: list[np.ndarray],
        anchor_coord: np.ndarray,
        max_length: int,
    ) -> list[np.ndarray]:
        all_coords = np.concatenate(coords_list, axis=0)
        all_mask = np.concatenate(mask_list, axis=0)
        distances = np.full(len(all_coords), np.inf, dtype=np.float32)
        distances[all_mask] = np.linalg.norm(all_coords[all_mask] - anchor_coord, axis=-1)
        # AFM Algorithm 2 adds small unique offsets to make distance ties stable.
        distances += np.arange(len(distances), dtype=np.float32) * 1e-3
        indices = np.argsort(distances)[:max_length]

        indices_list = []
        chain_start = 0
        for mask in mask_list:
            chain_end = chain_start + len(mask)
            chain_indices = (
                indices[(indices >= chain_start) & (indices < chain_end)] - chain_start
            )
            indices_list.append(chain_indices)
            chain_start = chain_end
        return indices_list
