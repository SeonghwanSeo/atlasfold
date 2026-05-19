"""Functions for computing confidence metrics from the model's predicted logits."""

import torch


def compute_plddt(
    logits: torch.Tensor,
    bin_centers: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the predicted lDDT-Calpha from the lDDT logits."""
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
    probs = torch.softmax(logits, dim=-1)  # [*, L, L, num_bins]
    pae = (probs * bin_centers).sum(dim=-1)  # [*, L, L]
    pair_mask = mask[..., :, None] & mask[..., None, :]
    pae = pae * pair_mask.float()  # [*, L, L]
    return pae  # [*, L, L]


def compute_ptm(
    logits: torch.Tensor,
    bin_centers: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute the predicted TM-score from the PAE logits."""
    # Compute d_0(num_res) as defined by TM-score, eqn. (5) in Yang & Skolnick
    # "Scoring function for automated assessment of protein structure template
    # quality", 2004: http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
    n = mask.sum(dim=-1, dtype=torch.float32)
    clipped_n = torch.clamp(n, min=19)
    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3.0) - 1.8
    probs = torch.softmax(logits, dim=-1)  # [*, L, L, num_bins]

    tm_per_bin = 1.0 / (1 + (bin_centers**2) / (d0**2))
    ptm_term = torch.sum(probs * tm_per_bin, dim=-1)  # [*, L, L]
    pair_mask = mask[..., :, None] & mask[..., None, :]
    w = pair_mask.float()

    ptm_term = ptm_term * w  # [*, L, L]
    denom = eps + w.sum(-1)  # [*, L]
    per_alignment = torch.sum(ptm_term / denom[..., None], dim=-1)  # [*, L]
    weighted = per_alignment.masked_fill(~mask, 0.0)
    return weighted.max(-1).values  # [*]
