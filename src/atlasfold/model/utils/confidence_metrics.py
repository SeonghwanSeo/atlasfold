"""Functions for computing confidence metrics from model outputs."""

import math

import numpy as np
import torch


def compute_plddt(
    logits: torch.Tensor,
    bin_centers: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the predicted lDDT-Calpha from the lDDT logits."""
    length = logits.shape[-2]
    if mask.shape[-1] != length:
        raise ValueError(
            f"Mask shape {mask.shape} does not match logits length {length}."
        )
    if mask.ndim != logits.ndim - 1:
        raise ValueError(
            f"Mask shape {mask.shape} is not compatible with logits shape {logits.shape}."
        )

    probs = torch.softmax(logits, dim=-1)  # [*, L, num_bins]
    plddt = (probs * bin_centers).sum(dim=-1)  # [*, L]
    plddt = plddt * mask.float()  # [*, L]
    return plddt  # [*, L]


def compute_pae(
    logits: torch.Tensor,
    bin_centers: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the predicted aligned error (PAE) from the PAE logits."""
    length = logits.shape[-2]
    if mask.shape[-1] != length:
        raise ValueError(
            f"Mask shape {mask.shape} does not match logits length {length}."
        )
    if mask.ndim != logits.ndim - 2:
        raise ValueError(
            f"Mask shape {mask.shape} is not compatible with logits shape {logits.shape}."
        )

    probs = torch.softmax(logits, dim=-1)  # [*, L, L, num_bins]
    return compute_pae_from_probs(probs, bin_centers, mask)


def compute_pae_from_probs(
    probs: torch.Tensor,
    bin_centers: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute PAE from precomputed PAE-bin probabilities."""
    length = probs.shape[-2]
    if mask.shape[-1] != length:
        raise ValueError(
            f"Mask shape {mask.shape} does not match probabilities length {length}."
        )
    if mask.ndim != probs.ndim - 2:
        raise ValueError(
            f"Mask shape {mask.shape} is not compatible with probabilities shape "
            f"{probs.shape}."
        )

    pae = (probs * bin_centers).sum(dim=-1)  # [*, L, L]
    pair_mask = mask[..., :, None] & mask[..., None, :]
    pae = pae * pair_mask.float()  # [*, L, L]
    return pae  # [*, L, L]


def compute_pde(
    logits: torch.Tensor,
    bin_centers: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the predicted distance error (PDE) from the PDE logits."""
    length = logits.shape[-2]
    if mask.shape[-1] != length:
        raise ValueError(
            f"Mask shape {mask.shape} does not match logits length {length}."
        )
    if mask.ndim != logits.ndim - 2:
        raise ValueError(
            f"Mask shape {mask.shape} is not compatible with logits shape {logits.shape}."
        )

    probs = torch.softmax(logits, dim=-1)  # [*, L, L, num_bins]
    pde = (probs * bin_centers).sum(dim=-1)  # [*, L, L]
    pair_mask = mask[..., :, None] & mask[..., None, :]
    pde = pde * pair_mask.float()  # [*, L, L]
    return pde  # [*, L, L]


def compute_ptm(
    logits: torch.Tensor,  # [*, L, L, num_bins]
    bin_centers: torch.Tensor,  # [num_bins]
    mask: torch.Tensor,  # [*, L]
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute the predicted TM-score from the PAE logits."""
    length = logits.shape[-2]
    if mask.shape[-1] != length:
        raise ValueError(
            f"Mask shape {mask.shape} does not match logits length {length}."
        )
    if mask.ndim != logits.ndim - 2:
        raise ValueError(
            f"Mask shape {mask.shape} is not compatible with logits shape {logits.shape}."
        )

    probs = torch.softmax(logits, dim=-1)  # [*, L, L, num_bins]
    return compute_ptm_from_probs(probs, bin_centers, mask, eps)


def compute_ptm_from_probs(
    probs: torch.Tensor,
    bin_centers: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute pTM from precomputed PAE-bin probabilities."""
    length = probs.shape[-2]
    if mask.shape[-1] != length:
        raise ValueError(
            f"Mask shape {mask.shape} does not match probabilities length {length}."
        )
    if mask.ndim != probs.ndim - 2:
        raise ValueError(
            f"Mask shape {mask.shape} is not compatible with probabilities shape "
            f"{probs.shape}."
        )

    # Compute d_0(num_res) as defined by TM-score, eqn. (5) in Yang & Skolnick
    # "Scoring function for automated assessment of protein structure template
    # quality", 2004: http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
    n = mask.sum(dim=-1, dtype=torch.float32)  # [*]
    clipped_n = torch.clamp(n, min=19)
    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3.0) - 1.8  # [*]

    d0 = d0[..., None, None, None]  # [*, 1, 1, 1]
    tm_per_bin = 1.0 / (1 + (bin_centers**2) / (d0**2))
    ptm_term = torch.sum(probs * tm_per_bin, dim=-1)  # [*, L, L]

    pair_mask = mask[..., :, None] & mask[..., None, :]
    w = pair_mask.float()

    ptm_term = ptm_term * w  # [*, L, L]
    denom = eps + w.sum(-1)  # [*, L]
    per_alignment = torch.sum(ptm_term / denom[..., None], dim=-1)  # [*, L]
    weighted = per_alignment.masked_fill(~mask, 0.0)
    return weighted.max(-1).values  # [*]


def compute_iptm(
    logits: torch.Tensor,
    bin_centers: torch.Tensor,
    asym_id: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute ipTM from PAE logits with shape ``[*, L, L, num_bins]``."""
    probs = torch.softmax(logits, dim=-1)  # [*, L, L, num_bins]
    return compute_iptm_from_probs(probs, bin_centers, asym_id, mask, eps)


def compute_iptm_from_probs(
    probs: torch.Tensor,
    bin_centers: torch.Tensor,
    asym_id: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute ipTM from probabilities and residue metadata sharing shape ``*``.

    ``probs``, ``asym_id``, and ``mask`` must have shapes
    ``[*, L, L, num_bins]``, ``[*, L]``, and ``[*, L]``, respectively. The
    returned score has shape ``[*]``.
    """
    if probs.ndim < 3:
        raise ValueError(
            "Expected PAE probabilities with shape [*, L, L, num_bins], "
            f"got {probs.shape}."
        )
    length, pair_length = probs.shape[-3:-1]
    if pair_length != length:
        raise ValueError(f"PAE probabilities must be square, got {probs.shape}.")
    expected_metadata_shape = probs.shape[:-3] + (length,)
    if asym_id.shape != expected_metadata_shape:
        raise ValueError(
            f"Expected asym_id shape {expected_metadata_shape}, got {asym_id.shape}."
        )
    if mask.shape != expected_metadata_shape:
        raise ValueError(
            f"Expected mask shape {expected_metadata_shape}, got {mask.shape}."
        )

    # Compute d_0(num_res) as defined by TM-score, eqn. (5) in Yang & Skolnick
    # "Scoring function for automated assessment of protein structure template
    # quality", 2004: http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
    n = mask.sum(dim=-1, dtype=torch.float32)  # [*]
    clipped_n = torch.clamp(n, min=19)
    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3.0) - 1.8  # [*]

    d0 = d0[..., None, None, None]  # [*, 1, 1, 1]
    tm_per_bin = 1.0 / (1 + (bin_centers**2) / (d0**2))
    ptm_term = torch.sum(probs * tm_per_bin, dim=-1)  # [*, L, L]

    pair_mask = mask[..., :, None] & mask[..., None, :]
    asym_pair_mask = asym_id[..., :, None] != asym_id[..., None, :]
    pair_mask = pair_mask & asym_pair_mask
    w = pair_mask.float()

    ptm_term = ptm_term * w  # [*, L, L]
    denom = eps + w.sum(-1)  # [*, L]
    per_alignment = torch.sum(ptm_term / denom[..., None], dim=-1)  # [*, L]
    weighted = per_alignment.masked_fill(~mask, 0.0)
    return weighted.max(-1).values  # [*]


def compute_chain_tm_scores_from_probs(
    probs: torch.Tensor,
    bin_centers: torch.Tensor,
    asym_id: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-chain pTM and pairwise ipTM for arbitrary leading dimensions.

    Parameters
    ----------
    probs
        PAE-bin probabilities with shape ``[*, L, L, num_bins]``.
    bin_centers
        PAE bin centers with shape ``[num_bins]``.
    asym_id
        Per-residue chain IDs with shape ``[*, L]``. Padding IDs are ignored.
    mask
        Residue mask with shape ``[*, L]``.

    Returns
    -------
    chain_ptm
        Tensor with shape ``[*, C]`` where ``C`` is the maximum number of chains
        over the leading dimensions.
    interface_iptm
        Symmetric tensor with shape ``[*, C, C]``. Entries without a
        corresponding chain pair are NaN.
    """
    if probs.ndim < 3:
        raise ValueError(
            "Expected PAE probabilities with shape [*, L, L, num_bins], "
            f"got {probs.shape}."
        )
    leading_shape = probs.shape[:-3]
    length, pair_length, num_bins = probs.shape[-3:]
    if pair_length != length:
        raise ValueError(f"PAE probabilities must be square, got {probs.shape}.")
    expected_metadata_shape = leading_shape + (length,)
    if asym_id.shape != expected_metadata_shape:
        raise ValueError(
            f"Expected asym_id shape {expected_metadata_shape}, got {asym_id.shape}."
        )
    if mask.shape != expected_metadata_shape:
        raise ValueError(
            f"Expected mask shape {expected_metadata_shape}, got {mask.shape}."
        )

    num_items = math.prod(leading_shape)
    flat_probs = probs.reshape(num_items, length, length, num_bins)
    flat_asym_id = asym_id.reshape(num_items, length)
    flat_mask = mask.reshape(num_items, length).bool()
    chain_ids_by_item = [
        torch.unique(flat_asym_id[item_idx][flat_mask[item_idx]], sorted=True)
        for item_idx in range(num_items)
    ]
    max_chains = max((len(chain_ids) for chain_ids in chain_ids_by_item), default=0)
    chain_ptm = torch.full(
        (num_items, max_chains),
        torch.nan,
        dtype=probs.dtype,
        device=probs.device,
    )
    interface_iptm = torch.full(
        (num_items, max_chains, max_chains),
        torch.nan,
        dtype=probs.dtype,
        device=probs.device,
    )

    for item_idx, chain_ids in enumerate(chain_ids_by_item):
        chain_indices = [
            torch.nonzero(
                flat_mask[item_idx] & (flat_asym_id[item_idx] == chain_id),
                as_tuple=False,
            ).squeeze(-1)
            for chain_id in chain_ids
        ]

        for chain_idx, residue_indices in enumerate(chain_indices):
            chain_probs = flat_probs[item_idx][residue_indices]
            chain_probs = chain_probs[:, residue_indices]
            chain_mask = torch.ones(
                len(residue_indices),
                dtype=torch.bool,
                device=probs.device,
            )
            chain_ptm[item_idx, chain_idx] = compute_ptm_from_probs(
                chain_probs,
                bin_centers,
                chain_mask,
            )

        for chain_i, residue_indices_i in enumerate(chain_indices):
            for chain_j in range(chain_i + 1, len(chain_indices)):
                residue_indices_j = chain_indices[chain_j]
                pair_indices = torch.cat((residue_indices_i, residue_indices_j))
                pair_probs = flat_probs[item_idx][pair_indices]
                pair_probs = pair_probs[:, pair_indices]
                pair_length = len(pair_indices)
                pair_mask = torch.ones(
                    pair_length,
                    dtype=torch.bool,
                    device=probs.device,
                )
                pair_asym_id = torch.cat(
                    (
                        torch.zeros(
                            len(residue_indices_i),
                            dtype=asym_id.dtype,
                            device=probs.device,
                        ),
                        torch.ones(
                            len(residue_indices_j),
                            dtype=asym_id.dtype,
                            device=probs.device,
                        ),
                    )
                )
                score = compute_iptm_from_probs(
                    pair_probs,
                    bin_centers,
                    pair_asym_id,
                    pair_mask,
                )
                interface_iptm[item_idx, chain_i, chain_j] = score
                interface_iptm[item_idx, chain_j, chain_i] = score

    return (
        chain_ptm.reshape(leading_shape + (max_chains,)),
        interface_iptm.reshape(leading_shape + (max_chains, max_chains)),
    )


def compute_pair_mean(
    values: np.ndarray,
    pair_mask: np.ndarray,
) -> float:
    selected = values[pair_mask]
    if selected.size == 0:
        raise ValueError("Cannot compute pair mean over an empty mask.")
    return float(selected.mean())
