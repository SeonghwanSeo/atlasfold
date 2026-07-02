"""Functions for computing confidence metrics from model outputs."""

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
            f"Mask shape {mask.shape} is not compatible with logits shape "
            f"{logits.shape}."
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
            f"Mask shape {mask.shape} is not compatible with logits shape "
            f"{logits.shape}."
        )

    probs = torch.softmax(logits, dim=-1)  # [*, L, L, num_bins]
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
            f"Mask shape {mask.shape} is not compatible with logits shape "
            f"{logits.shape}."
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
            f"Mask shape {mask.shape} is not compatible with logits shape "
            f"{logits.shape}."
        )

    # Compute d_0(num_res) as defined by TM-score, eqn. (5) in Yang & Skolnick
    # "Scoring function for automated assessment of protein structure template
    # quality", 2004: http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
    n = mask.sum(dim=-1, dtype=torch.float32)  # [*]
    clipped_n = torch.clamp(n, min=19)
    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3.0) - 1.8  # [*]
    probs = torch.softmax(logits, dim=-1)  # [*, L, L, num_bins]

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
    """Compute the interface predicted TM-score from PAE logits."""
    length = logits.shape[-2]
    if mask.shape[-1] != length:
        raise ValueError(
            f"Mask shape {mask.shape} does not match logits length {length}."
        )
    if mask.ndim != logits.ndim - 2:
        raise ValueError(
            f"Mask shape {mask.shape} is not compatible with logits shape "
            f"{logits.shape}."
        )
    if asym_id.shape[-1] != length:
        raise ValueError(
            f"asym_id shape {asym_id.shape} does not match logits length {length}."
        )
    if asym_id.ndim == mask.ndim - 1:
        asym_id = asym_id.unsqueeze(-2)
    elif asym_id.ndim != mask.ndim:
        raise ValueError(
            f"asym_id shape {asym_id.shape} is not compatible with mask shape "
            f"{mask.shape}."
        )

    # Compute d_0(num_res) as defined by TM-score, eqn. (5) in Yang & Skolnick
    # "Scoring function for automated assessment of protein structure template
    # quality", 2004: http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
    n = mask.sum(dim=-1, dtype=torch.float32)  # [*]
    clipped_n = torch.clamp(n, min=19)
    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3.0) - 1.8  # [*]
    probs = torch.softmax(logits, dim=-1)  # [*, L, L, num_bins]

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


def compute_pair_mean(
    values: np.ndarray,
    pair_mask: np.ndarray,
) -> float:
    selected = values[pair_mask]
    if selected.size == 0:
        raise ValueError("Cannot compute pair mean over an empty mask.")
    return float(selected.mean())
