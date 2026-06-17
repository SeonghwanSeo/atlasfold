import numpy as np
from scipy.spatial.distance import cdist

from atlasfold.common import metadata, protein
from atlasfold.utils.misc import spawn_rng


class ProteinCropper:
    def __init__(
        self,
        prob_contiguous: float = 0.2,
        prob_spatial: float = 0.6,
        prob_multi_contiguous: float = 0.2,
        max_num_segments: int = 4,
        min_segment_length: int = 32,
    ):
        self.prob_spatial = prob_spatial
        self.prob_contiguous = prob_contiguous
        self.prob_multi_contiguous = prob_multi_contiguous
        self.max_num_segments = max_num_segments
        self.min_segment_length = min_segment_length

    def crop(
        self,
        prot: protein.Protein,
        max_length: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Crop the input structure

        Parameters
        ----------
        prot : protein.Protein
            The input protein structure to be cropped.
        rng : np.random.Generator
            The random number generator for reproducibility.

        Returns
        -------
        sequence_indices : np.ndarray
            The indices of the residues to be included in the cropped structure.
            NOTE: this is different from the residue index (1-based).
        """
        rng = spawn_rng(rng)
        if len(prot) <= max_length:
            return np.arange(len(prot))

        # Randomly choose cropping strategy
        r = rng.random()
        if r < self.prob_spatial:
            return self._crop_spatial(prot, max_length, rng)
        elif r < self.prob_spatial + self.prob_contiguous:
            return self._crop_contiguous(prot, max_length, rng)
        else:
            return self._crop_multi_contiguous(prot, max_length, rng)

    def _crop_contiguous(
        self,
        prot: protein.Protein,
        max_length: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Crop a contiguous segment of the input structure."""
        length = len(prot)
        start = rng.integers(0, length - max_length + 1)
        return np.arange(start, start + max_length)

    def _crop_spatial(
        self,
        prot: protein.Protein,
        max_length: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Crop a spatial segment of the input structure."""
        length = len(prot)
        # CA coordinates are at index 1 in coordinates [L, 14, 3]
        ca_coords = prot.coordinates[:, 1, :]
        ca_mask = np.isfinite(ca_coords).all(axis=-1)
        valid_indices = np.where(ca_mask)[0]

        if not np.any(ca_mask):
            # Fallback to contiguous if no CA is found
            return self._crop_contiguous(prot, max_length, rng)

        if len(valid_indices) <= max_length:
            # If there are fewer valid CA residues than max_length,
            # return all valid indices
            return valid_indices

        # Pick a random anchor residue that has a CA
        anchor_idx = rng.choice(list(valid_indices))
        anchor_coord = ca_coords[anchor_idx]

        # Compute distances to all CA coordinates
        distances = np.full(length, 1e6, dtype=np.float32)
        valid_distances = np.linalg.norm(ca_coords[ca_mask] - anchor_coord, axis=-1)
        distances[ca_mask] = valid_distances

        # Break ties randomly by shuffling indices first
        indices = np.arange(length)
        rng.shuffle(indices)
        shuffled_distances = distances[indices]

        # Pick the max_length nearest residues
        nearest_shuffled_indices = np.argsort(shuffled_distances)[:max_length]
        nearest_indices = indices[nearest_shuffled_indices]

        return np.sort(nearest_indices)

    def _crop_multi_contiguous(
        self,
        prot: protein.Protein,
        max_length: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        r"""Crop multiple contiguous segments of the input structure.

        Motivation: The training sample from the spatial cropping and contiguous
        cropping has a restricted variance in the input structure. Therefore,
        we introduce a multi-contiguous cropping strategy that allows for model
        to predict the large structure.

        Strategy:
            1. Randomly choose the number of segments (between 2 and max_num_segments).
            2. Uniformly segment the sequence into num_anchors segments.
            3. Randomly distribute max_length into crop_sizes.
            4. Crop contiguous residues from each uniform segment.
        """
        if max_length < self.min_segment_length * 2:
            # We need at least 2 segments to do multi-contiguous cropping
            return self._crop_contiguous(prot, max_length, rng)

        length = len(prot)
        # 1. Randomly choose the number of segments
        max_num_segments = min(self.max_num_segments, length // self.min_segment_length)
        num_segments = rng.integers(2, max_num_segments + 1)

        # 2. Segment the sequence
        segment_length = length // num_segments
        seg_sizes: list[int] = [segment_length] * num_segments
        # Add the remaining residues to the last segment
        seg_sizes[-1] = length - segment_length * (num_segments - 1)

        # 3. Randomized crop sizes
        budget = max_length - num_segments * self.min_segment_length
        assert budget >= 0
        crop_sizes = [self.min_segment_length] * num_segments
        valid_segments = [
            i for i in range(num_segments) if seg_sizes[i] > self.min_segment_length
        ]
        while budget > 0:
            # Randomly pick a segment to add one more residue to
            idx = rng.choice(valid_segments)
            max_len = seg_sizes[idx]
            curr_len = crop_sizes[idx]
            if curr_len < max_len:
                # Add a random number of residues to this segment.
                max_to_add = min(budget, max_len - curr_len)
                num_to_add = rng.integers(1, max_to_add + 1)
                crop_sizes[idx] += int(num_to_add)
                budget -= int(num_to_add)
            if crop_sizes[idx] >= max_len:
                # This segment cannot be added anymore, remove it from valid segments
                valid_segments.remove(idx)
                if not valid_segments:
                    break

        current_idx = 0
        cropped_indices = []
        for i in range(num_segments):
            # Crop a contiguous segment from this segment
            seg_len = seg_sizes[i]
            crop_len = crop_sizes[i]
            start_in_seg = rng.integers(0, seg_len - crop_len + 1)
            start = current_idx + start_in_seg
            cropped_indices.append(np.arange(start, start + crop_len))

            current_idx += seg_len

        return np.sort(np.concatenate(cropped_indices))


class ComplexCropper:
    def __init__(
        self,
        prob_spatial: float = 0.4,
        prob_interface_spatial: float = 0.4,
        prob_contiguous: float = 0.2,
    ):
        self.prob_spatial: float = prob_spatial
        self.prob_interface_spatial: float = prob_interface_spatial
        self.prob_contiguous: float = prob_contiguous

    def crop(
        self,
        compl: protein.ProteinComplex,
        m: metadata.ComplexMetadata,
        max_length: int,
        bias_chain_id: int | tuple[int, int] | None,
        rng: np.random.Generator | None = None,
    ) -> list[np.ndarray]:
        """Crop the input structure

        Parameters
        ----------
        compl : protein.ProteinComplex
            The input structex structure to be cropped.
        rng : np.random.Generator
            The random number generator for reproducibility.

        Returns
        -------
        sequence_indices : list[np.ndarray]
            A list of length num_chains, where each element is either an array of
            residue indices to be included in the cropped structure for that chain.
        """
        rng = spawn_rng(rng)
        if compl.num_residues <= max_length:
            return [np.arange(c.num_residues) for c in compl.chains]

        # Randomly choose cropping strategy
        r = rng.random()
        if r < self.prob_spatial:
            indices_list = self._crop_spatial(compl, max_length, bias_chain_id, rng)
        elif r < self.prob_spatial + self.prob_contiguous:
            indices_list = self._crop_contiguous(compl, max_length, rng)
        else:
            indices_list = self._crop_interface_spatial(
                compl, m, max_length, bias_chain_id, rng
            )
        return [np.sort(indices) for indices in indices_list]

    def _crop_contiguous(
        self,
        compl: protein.ProteinComplex,
        max_length: int,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        """Crop a contiguous segment of the input structure."""
        # Compute the number of tokens and start indices per chain
        asym_ids = compl.asym_ids
        chain_sizes: dict[int, int] = {
            asym_id: c.num_residues
            for asym_id, c in zip(asym_ids, compl.chains, strict=True)
        }
        chain_starts: dict[int, int] = {}
        st = 0
        for asym_id in asym_ids:
            chain_starts[asym_id] = st
            st += chain_sizes[asym_id]

        # Randomly permute the chain order
        selected_asym_ids = rng.permutation(asym_ids)

        # Line 1
        n_added: int = 0
        # Line 2
        n_remaining: int = sum(chain_sizes[asym_id] for asym_id in selected_asym_ids)
        is_selected: dict[int, np.ndarray] = {
            asym_id: np.zeros(chain_sizes[asym_id], dtype=bool)
            for asym_id in selected_asym_ids
        }

        # Line 3-13
        for asym_id in selected_asym_ids:
            if n_added >= max_length:
                break

            n_k = chain_sizes[asym_id]
            # Line 4
            n_remaining -= n_k

            # Sample length of crop for current chain
            # Line 5
            max_crop = min(max_length - n_added, n_k)
            # Line 6
            min_crop = min(n_k, max(0, max_length - n_added - n_remaining))
            # Line 7
            crop_size = int(rng.integers(min_crop, max_crop + 1))
            # Line 8
            n_added += crop_size

            if crop_size == 0:
                continue

            # Line 9
            crop_start = int(rng.integers(0, n_k - crop_size + 1))

            # Line 11
            selected_tokens = np.arange(crop_start, crop_start + crop_size)

            # Line 12
            is_selected[asym_id][selected_tokens] = True

        indices_list = []
        for asym_id in compl.asym_ids:
            indices_list.append(np.where(is_selected[asym_id])[0])

        return indices_list

    def _crop_spatial(
        self,
        compl: protein.ProteinComplex,
        max_length: int,
        bias_chain_id: int | tuple[int, int] | None,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        """Crop a spatial segment of the input structure."""
        # CA coordinates are at index 1 in coordinates [L, 14, 3]
        ca_coords_list = [c.coordinates[:, 1, :] for c in compl.chains]
        ca_mask_list = [np.isfinite(coords).all(-1) for coords in ca_coords_list]
        ca_coords: np.ndarray = np.concatenate(ca_coords_list, axis=0)  # [L, 3]
        ca_mask: np.ndarray = np.concatenate(ca_mask_list, axis=0)  # [L]
        if not np.any(ca_mask):
            # Edge case. return dummy indices and re-sample datapoint.
            return [np.array([0]) for _ in compl.chains]

        if sum(ca_mask) <= max_length:
            # If there are fewer valid CA residues than max_length,
            # return all valid indices
            return [mask.nonzero()[0] for mask in ca_mask_list]

        # Select anchor residue
        anchor_coord = None
        if bias_chain_id is not None:
            # Pick a random anchor residue from the bias chain
            if isinstance(bias_chain_id, tuple):
                # If bias_chain_id is a tuple, randomly choose one of the two chains
                bias_chain_id = rng.choice(bias_chain_id)
            chain_ca_coords = ca_coords_list[bias_chain_id]
            chain_ca_mask = np.isfinite(chain_ca_coords).all(-1)
            if np.any(chain_ca_mask):
                valid_chain_indices = np.where(chain_ca_mask)[0]
                anchor_coord = chain_ca_coords[rng.choice(valid_chain_indices)]
        if anchor_coord is None:
            # Pick a random anchor residue
            valid_indices = np.where(ca_mask)[0]
            anchor_coord = ca_coords[rng.choice(valid_indices)]

        # Get the closest tokens to the anchor residue
        return self.get_closest_residues(
            ca_coords_list, ca_mask_list, anchor_coord, max_length
        )

    def _crop_interface_spatial(
        self,
        compl: protein.ProteinComplex,
        m: metadata.ComplexMetadata,
        max_length: int,
        bias_chain_id: int | tuple[int, int] | None,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:

        def fallback():
            return self._crop_spatial(compl, max_length, bias_chain_id, rng)

        if bias_chain_id is None:
            if len(m.interfaces) == 0:
                return fallback()
            interface = m.interfaces[rng.integers(0, len(m.interfaces))]
            bias_chain_id = interface.chain_ids
        elif isinstance(bias_chain_id, int):
            if len(m.interfaces) == 0:
                return fallback()
            interfaces = [
                iface for iface in m.interfaces if bias_chain_id in iface.chain_ids
            ]
            if len(interfaces) == 0:
                return fallback()
            interface = interfaces[rng.integers(0, len(interfaces))]
            bias_chain_id = interface.chain_ids

        assert isinstance(bias_chain_id, tuple)

        ca_coords_list = [c.coordinates[:, 1, :] for c in compl.chains]
        ca_mask_list = [np.isfinite(coords).all(-1) for coords in ca_coords_list]

        # Select anchor residue
        c1, c2 = bias_chain_id
        chain1_coords = ca_coords_list[c1]
        chain1_mask = ca_mask_list[c1]
        chain2_coords = ca_coords_list[c2]
        chain2_mask = ca_mask_list[c2]

        d = cdist(chain1_coords, chain2_coords)  # [L1, L2]
        interface_mask = (d < 15.0) & chain1_mask[:, None] & chain2_mask[None, :]
        if not np.any(interface_mask):
            # Edge case: no interface residues. Fall back to spatial cropping.
            return fallback()
        anchors1 = interface_mask.any(axis=1)
        anchors2 = interface_mask.any(axis=0)
        if not np.any(anchors1) or not np.any(anchors2):
            # Edge case: no valid anchors. Fall back to spatial cropping.
            return fallback()

        w1, w2 = float(np.sum(anchors1)), float(np.sum(anchors2))
        if rng.random() < (w1 / (w1 + w2)):
            anchor_coord = chain1_coords[rng.choice(np.where(anchors1)[0])]
        else:
            anchor_coord = chain2_coords[rng.choice(np.where(anchors2)[0])]

        # Get the closest tokens to the anchor residue
        return self.get_closest_residues(
            ca_coords_list, ca_mask_list, anchor_coord, max_length
        )

    @staticmethod
    def get_closest_residues(
        coords_list: list[np.ndarray],
        mask_list: list[np.ndarray],
        anchor_coord: np.ndarray,
        max_length: int,
    ) -> list[np.ndarray]:
        all_coords = np.concatenate(coords_list, axis=0)  # [L, 3]
        all_mask = np.concatenate(mask_list, axis=0)  # [L]

        length = all_coords.shape[0]
        # Compute distances to all CA coordinates
        distances = np.full(length, 1e6, dtype=np.float32)
        valid_distances = np.linalg.norm(all_coords[all_mask] - anchor_coord, axis=-1)
        distances[all_mask] = valid_distances

        # Pick the max_length nearest residues
        indices = np.argsort(distances)[:max_length]

        # Convert global indices to per-chain indices
        indices_list = []
        start = 0
        for mask in mask_list:
            chain_length = len(mask)
            chain_indices = (
                indices[(indices >= start) & (indices < start + chain_length)] - start
            )
            indices_list.append(chain_indices)
            start += chain_length

        return indices_list
