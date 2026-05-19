"""Featurization utilities for AtlasFold inference."""

from collections.abc import Iterable, Sequence

import numpy as np
import torch

from atlasfold.common import residue_utils
from atlaslm.alphabet import VOCAB

VOCAB_TO_IDX: dict[str, int] = {tok: i for i, tok in enumerate(VOCAB)}
BOS_IDX: int = VOCAB_TO_IDX["<cls>"]
EOS_IDX: int = VOCAB_TO_IDX["<eos>"]
MASK_IDX: int = VOCAB_TO_IDX["<mask>"]
PAD_IDX: int = VOCAB_TO_IDX["<pad>"]

# NOTE: This default bucket list is designed for uncompiled model inference.
# For compiled inference, we recommend using a smaller set of buckets to
# reduce the number of recompilations.
DEFAULT_BUCKETS = [
    32, 64, 128, 192, 256, 384, 512, 640, 768, 896, 1024,
    1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048,
]  # fmt: skip


def featurize(
    sequence: str,
    residue_index: Sequence[int] | None = None,
    lm_mask: Sequence[bool] | None = None,
) -> dict[str, np.ndarray]:
    """Featurize the input sequence for the model."""
    # Sanitize the input sequence
    sequence = sequence.upper()
    sequence = "".join(
        [aa if aa in residue_utils.restype_orders else "X" for aa in sequence]
    )
    length = len(sequence)

    # === Static features (chain-related) === #
    entity_id = np.ones(length, dtype=np.int64)  # Single entity
    asym_id = np.ones(length, dtype=np.int64)  # Single chain
    sym_id = np.ones(length, dtype=np.int64)  # Single symmetric unit

    # === Residue index === #
    if residue_index is not None:
        if len(residue_index) != len(sequence):
            raise ValueError(
                f"Length of residue_index ({len(residue_index)}) does not match "
                f"length of sequence ({len(sequence)})."
            )
        res_idx = np.array(residue_index, dtype=np.int64)
    else:
        res_idx = np.arange(1, length + 1, dtype=np.int64)

    # === Prepare the input for the language model trunk === #
    input_ids = np.array(
        [BOS_IDX] + [VOCAB_TO_IDX[aa] for aa in sequence] + [EOS_IDX], dtype=np.int64
    )
    if lm_mask is not None:
        if len(lm_mask) != len(sequence):
            raise ValueError(
                f"Length of lm_mask ({len(lm_mask)}) does not match "
                f"length of sequence ({len(sequence)})."
            )
        input_ids[1:-1][lm_mask] = MASK_IDX
    pos_id = np.concatenate(([0], res_idx, [res_idx[-1] + 1]))
    seq_id = np.ones_like(input_ids, dtype=np.int64)

    lm_input = {
        "lm.input_ids": input_ids,  # [L+2]
        "lm.pos_id": pos_id,  # [L+2]
        "lm.seq_id": seq_id,  # [L+2]
    }

    # === Prepare the input for the folding trunk === #
    # Convert amino acid sequence to integer indices
    length = len(sequence)
    aatype = np.array([residue_utils.restype_orders[aa] for aa in sequence])
    aatype_onehot = np.eye(21, dtype=np.float32)[aatype]

    # Create masks
    seq_mask = np.ones(length, dtype=bool)
    atom14_mask = residue_utils.restype_atom14_mask[aatype_onehot]
    atom37_mask = residue_utils.restype_atom37_mask[aatype_onehot]

    # Get the pseudo-beta index (CB for most residues, CA for glycine)
    cbeta_idx = np.full((length,), 4, dtype=np.int64)  # CB: index=4
    cbeta_idx[aatype == residue_utils.restype_orders["G"]] = 1

    folding_input = {
        "entity_id": entity_id,  # [L]
        "asym_id": asym_id,  # [L]
        "sym_id": sym_id,  # [L]
        "aatype": aatype_onehot,  # [L, 21]
        "aatype_int": aatype,  # [L]
        "res_idx": res_idx,  # [L]
        "seq_mask": seq_mask,  # [L]
        "atom14_mask": atom14_mask,  # [L, 14]
        "atom37_mask": atom37_mask,  # [L, 37]
        "pseudo_beta": cbeta_idx,  # [L]
    }
    return {**lm_input, **folding_input}


def featurize_batch(
    sequence_list: list[str],
    residue_index_list: list[Sequence[int] | None] | None = None,
    lm_mask_list: list[Sequence[bool] | None] | None = None,
    buckets: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """Featurize the input sequence for the model."""
    bs = len(sequence_list)
    maxlen = max(len(seq) for seq in sequence_list)

    if buckets is not None:
        buckets = sorted(buckets)
        if buckets[-1] < maxlen:
            # If the longest sequence exceeds the largest bucket,
            # pad to the next multiple of 8.
            maxlen = ((maxlen + 7) // 8) * 8
        else:
            for bucket in buckets:
                if maxlen <= bucket:
                    maxlen = bucket
                    break

    batch = {
        "lm.input_ids": np.full((bs, maxlen + 2), PAD_IDX, dtype=int),
        "lm.pos_id": np.zeros((bs, maxlen + 2), dtype=int),
        "lm.seq_id": np.zeros((bs, maxlen + 2), dtype=int),
        "entity_id": np.zeros((bs, maxlen), dtype=np.int64),
        "asym_id": np.zeros((bs, maxlen), dtype=np.int64),
        "sym_id": np.zeros((bs, maxlen), dtype=np.int64),
        "aatype": np.zeros((bs, maxlen, 21), dtype=np.float32),
        "aatype_int": np.zeros((bs, maxlen), dtype=int),
        "res_idx": np.zeros((bs, maxlen), dtype=int),
        "seq_mask": np.zeros((bs, maxlen), dtype=bool),
        "atom14_mask": np.zeros((bs, maxlen, 14), dtype=bool),
        "atom37_mask": np.zeros((bs, maxlen, 37), dtype=bool),
        "pseudo_beta": np.zeros((bs, maxlen), dtype=int),
    }

    for i in range(bs):
        seq = sequence_list[i]
        length = len(seq)
        residue_index = residue_index_list[i] if residue_index_list else None
        lm_mask = lm_mask_list[i] if lm_mask_list else None
        featurized = featurize(seq, residue_index, lm_mask)
        for k, v in featurized.items():
            if k.startswith("lm."):
                batch[k][i, : length + 2] = v
            else:
                batch[k][i, :length] = v

    return batch


class InputFeaturizer(torch.nn.Module):
    """
    Featurizes input amino acid sequences into batched tensor representations.

    This module acts as the data pipeline entry point for AtlasFold inference,
    processing raw protein sequences (and optional residue indices) into the
    structured tensor dictionaries required by the AtlasLM language model trunk
    and the downstream diffusion module.

    A critical feature of this class is its handling of conformational diversity
    through deterministic stochasticity:

    * Deterministic Mode: If `seeds` is None, sequences are featurized normally
      without any Masked Language Modeling (MLM) masking.
    * Stochastic Mode: If a list of `seeds` is provided, the featurizer pre-computes
      reproducible MLM masking patterns. This introduces controlled uncertainty into
      the language model trunk, which allows the downstream diffusion model to sample
      multiple diverse conformational states for the exact same input sequence.

    I note that the deterministicity is only applied to the trunk featurization,
    while the diffusion module is always **stochastic**.

    Parameters
    ----------
    seeds : list of int or None, optional
        A list of master seeds used to generate reproducible stochastic masks for
        sampling multiple conformations. If None or empty, trunk featurization
        is deterministic.
    prob_mlm : float or None, optional
        The fixed probability of masking a given amino acid token when seeds are
        provided. If None, a random probability between 0 and 0.15 is uniformly
        sampled per seed. Default is 0.1.
    max_length : int, optional
        The maximum sequence length supported for the pre-computed shared masking
        patterns. Default is 20000.
    """

    def __init__(
        self,
        seeds: list[int] | None = None,
        prob_mlm: float | None = 0.1,
        max_length: int = 20000,
        buckets: list[int] | None = None,
    ) -> None:
        super().__init__()
        if seeds is not None and len(seeds) == 0:
            seeds = None
        if seeds is None:
            prob_mlm = 0.0
        if buckets is None:
            buckets = DEFAULT_BUCKETS

        self.seeds: list[int] | None = seeds
        self.prob_mlm: float | None = prob_mlm
        self.max_length: int = max_length
        self.buckets: list[int] = sorted(buckets)

        if seeds is not None and any(seed < 0 for seed in seeds):
            raise ValueError(f"All seeds must be non-negative integers, but got {seeds}.")

        if self.seeds is not None:
            # Sample the shared masking patterns to ensure the reproducibility of
            # the stochastic samples.
            num_seeds = len(self.seeds)
            shared_lm_mask = np.zeros((num_seeds, max_length), dtype=bool)
            for i, master_seed in enumerate(self.seeds):
                rng = np.random.default_rng(master_seed)
                if prob_mlm is None:
                    p_mask = rng.uniform(0, 0.15)
                else:
                    p_mask = prob_mlm
                shared_lm_mask[i] = rng.random(max_length) < p_mask
            self.shared_lm_mask = shared_lm_mask

    @property
    def is_trunk_deterministic(self) -> bool:
        """Whether the trunk is deterministic (i.e., no mlm masking)."""
        return self.seeds is None

    @property
    def is_trunk_stochastic(self) -> bool:
        """Whether the trunk is stochastic (i.e., whether to apply stochastic masking)."""
        return self.seeds is not None

    def iter_batched_inputs(
        self,
        input_list: list[tuple[str, str]],
        max_tokens_per_batch: int = 1024,
    ) -> Iterable[dict]:
        """Featurize the input sequence for the model."""
        for batch in self.iter_batched_sequences(input_list, max_tokens_per_batch):
            feat = featurize_batch(
                batch["sequences"],
                residue_index_list=None,
                lm_mask_list=batch["lm_masks"] if self.is_trunk_stochastic else None,
            )
            feat = {k: torch.from_numpy(v) for k, v in feat.items()}
            batch["feat"] = feat
            yield batch

    def iter_batched_sequences(
        self,
        input_list: list[tuple[str, str]],
        max_tokens_per_batch: int = 1024,
    ) -> Iterable[dict]:
        """Featurize the input sequence for the model."""

        def get_bucketed_length(length: int) -> int:
            """Finds the appropriate bucket length for the folding trunk."""
            if self.buckets and length <= self.buckets[-1]:
                for bucket in self.buckets:
                    if length <= bucket:
                        return bucket
            return ((length + 15) // 16) * 16

        def get_empty_batch() -> dict[str, list]:
            """Returns an empty batch dictionary."""
            return dict(headers=[], seeds=[], sequences=[], lm_masks=[])

        batch = get_empty_batch()
        max_len_in_batch = 0
        seeds = self.seeds if self.seeds is not None else [None]

        for header, seq in input_list:
            seq_len = len(seq)
            for i, seed in enumerate(seeds):
                new_max_len = max(max_len_in_batch, seq_len)
                bucketed_len = get_bucketed_length(new_max_len)

                new_tokens = (len(batch["sequences"]) + 1) * bucketed_len
                if new_tokens > max_tokens_per_batch and len(batch["sequences"]) > 0:
                    yield batch
                    batch = get_empty_batch()
                    max_len_in_batch = seq_len
                else:
                    max_len_in_batch = new_max_len

                batch["headers"].append(header)
                batch["seeds"].append(seed)
                batch["sequences"].append(seq)
                if self.is_trunk_stochastic:
                    batch["lm_masks"].append(self.shared_lm_mask[i][:seq_len])

        if len(batch["sequences"]) > 0:
            yield batch
