import functools
import math

import numpy as np
import torch

from atlasfold.common import residue_utils
from atlasfold.utils.geometry.rigid_align import rigid_align
from atlasfold.utils.torch_utils import gather_dim


def cdist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise distances between two sets of points."""
    return torch.norm(x[..., :, None, :] - y[..., None, :, :], dim=-1)


@torch.no_grad()
def compute_rmsd(
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    mask: torch.Tensor,
    align: bool = False,
):
    """Compute RMSD between predicted and ground truth coordinates.

    Parameters
    ----------
    x_pred : torch.Tensor
        Predicted coordinates, shape (*, N, 3)
    x_gt : torch.Tensor
        Ground truth coordinates, shape (*, N, 3)
    mask : torch.Tensor
        Mask for valid atoms, shape (*, N)
    align : bool, optional
        Whether to perform rigid alignment before computing RMSD

    Returns
    -------
    rmsd : torch.Tensor
        RMSD between predicted and ground truth coordinates, shape (*,)
    """
    if align:
        x_pred = rigid_align(x_pred, x_gt, mask)
    sq_diff = ((x_pred - x_gt) ** 2).sum(-1)  # (*, N)
    w = mask.float()
    return torch.sqrt((sq_diff * w).sum(-1) / w.sum(-1).clamp(min=1))


@torch.no_grad()
def compute_lddt(
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    mask: torch.Tensor,
    cutoff: float = 15.0,
):
    """Compute lDDT between predicted and ground truth coordinates.

    Parameters
    ----------
    x_pred : torch.Tensor
        Predicted coordinates, shape (*, N, 3)
    x_gt : torch.Tensor
        Ground truth coordinates, shape (*, N, 3)
    mask : torch.Tensor
        Mask for valid atoms, shape (*, N)
    cutoff : float, optional
        Distance cutoff for lDDT calculation

    Returns
    -------
    lddt : torch.Tensor
        lDDT between predicted and ground truth coordinates, shape (*,)
    """
    pdist_gt = cdist(x_gt, x_gt)  # (*, N, N)
    pdist_pred = cdist(x_pred, x_pred)  # (*, N, N)
    error = torch.abs(pdist_gt - pdist_pred)
    score = torch.zeros_like(error)
    for threshold in [0.5, 1.0, 2.0, 4.0]:
        score += (error < threshold).float()
    score *= 0.25

    # Mask out pairs where either atom is unresolved
    pair_mask = mask[..., :, None] & mask[..., None, :]  # [*, N, N]

    # Exclude self-pairs properly for batched N-dimensional tensors
    N = pair_mask.shape[-1]
    diag_mask = ~torch.eye(N, dtype=torch.bool, device=pair_mask.device)
    pair_mask &= diag_mask

    # Exclude pairs that are too far apart
    pair_mask &= pdist_gt < cutoff

    w = pair_mask.float()
    lddt = (score * w).sum((-1, -2)) / w.sum((-1, -2)).clamp(min=1)
    return lddt


@torch.no_grad()
def compute_lddt_ca(
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    mask: torch.Tensor,
    cutoff: float = 15.0,
):
    """Compute CA-lDDT between predicted and ground truth coordinates.

    Parameters
    ----------
    x_pred : torch.Tensor
        Predicted coordinates, shape (*, L, 14, 3)
    x_gt : torch.Tensor
        Ground truth coordinates, shape (*, L, 14, 3)
    mask : torch.Tensor
        Mask for valid atoms, shape (*, L, 14)
    cutoff : float, optional
        Distance cutoff for lDDT calculation

    Returns
    -------
    lddt-ca : torch.Tensor
        lDDT-Calpha between predicted and ground truth coordinates, shape (*,)
    """
    ca_gt = x_gt[..., 1, :]  # (*, L, 3)
    ca_pred = x_pred[..., 1, :]  # (*, L, 3)
    ca_mask = mask[..., 1]  # (*, L)
    return compute_lddt(ca_pred, ca_gt, ca_mask, cutoff)


@torch.no_grad()
def compute_lddt_fullatom(
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    mask: torch.Tensor,
    cutoff: float = 15.0,
):
    """Compute full-atom lDDT between predicted and ground truth coordinates.

    Parameters
    ----------
    x_pred : torch.Tensor
        Predicted coordinates, shape (*, L, 14, 3)
    x_gt : torch.Tensor
        Ground truth coordinates, shape (*, L, 14, 3)
    mask : torch.Tensor
        Mask for valid atoms, shape (*, L, 14)
    cutoff : float, optional
        Distance cutoff for lDDT calculation

    Returns
    -------
    lddt : torch.Tensor
        lDDT between predicted and ground truth coordinates, shape (*,)
    """
    x_pred_flat = x_pred.flatten(-3, -2)  # (*, L*14, 3)
    x_gt_flat = x_gt.flatten(-3, -2)  # (*, L*14, 3)
    mask_flat = mask.flatten(-2)  # (*, L*14)
    return compute_lddt(x_pred_flat, x_gt_flat, mask_flat, cutoff)


# Compute restype_atom_swap based on restype_ambiguous_atoms
@functools.lru_cache(maxsize=1)
def get_restype_atom_swap(device_type: str) -> torch.Tensor:
    restype_atom_swap = np.tile(np.arange(14), (21, 1))
    for res_name, (group1, group2) in residue_utils.restype_ambiguous_atoms.items():
        restype = residue_utils.restype_orders[residue_utils.restype_3to1[res_name]]
        atom14_order = residue_utils.restype_atom14_order[res_name]
        for atom_name1, atom_name2 in zip(group1, group2, strict=True):
            atom_idx1 = atom14_order[atom_name1]
            atom_idx2 = atom14_order[atom_name2]
            restype_atom_swap[restype, atom_idx1] = atom_idx2
            restype_atom_swap[restype, atom_idx2] = atom_idx1
    return torch.from_numpy(restype_atom_swap).to(device_type)


@functools.lru_cache(maxsize=1)
def get_restype_has_ambiguous_atoms(device_type: str) -> torch.Tensor:
    has_ambiguous_atoms = np.zeros(21, dtype=bool)
    for res_name in residue_utils.restype_ambiguous_atoms.keys():
        restype = residue_utils.restype_orders[residue_utils.restype_3to1[res_name]]
        has_ambiguous_atoms[restype] = True
    return torch.from_numpy(has_ambiguous_atoms).to(device_type)


@torch.no_grad()
def get_aligned_gt_structure(
    x_gt: torch.Tensor,
    x_pred: torch.Tensor,
    aatype: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the aligned ground truth coordinates for the confidence prediction losses.

    Parameters
    ----------
    x_gt : torch.Tensor
        Tensor of shape (*, L, 14, 3) containing ground truth coordinates.
    x_pred : torch.Tensor
        Tensor of shape (*, L, 14, 3) containing predicted coordinates.
    aatype : torch.Tensor
        Tensor of shape (*, L) containing the amino acid type indices.
    mask : torch.Tensor
        Tensor of shape (*, L, 14) containing boolean masks for resolved atoms.

    Returns
    -------
    x_gt_aligned : torch.Tensor
        Tensor of shape (*, L, 14, 3) containing the aligned ground truth coordinates.
    mask_aligned : torch.Tensor
        Tensor of shape (*, L, 14) containing boolean masks for resolved atoms
        in the aligned ground truth structure.
    """
    *batch_dims, L, _, _ = x_pred.shape
    B = math.prod(batch_dims)

    x_gt_b = x_gt.view(B, L, 14, 3)
    x_pred_b = x_pred.view(B, L, 14, 3)
    aatype_b = aatype.view(B, L)
    mask_b = mask.view(B, L, 14)

    # Rigid alignment of GT to predicted structure using backbone atoms
    # Align predicted coordinates to ground truth coordinates using Kabsch algorithm.
    # Get anchor (N CA C) indices for alignment
    atom_index = torch.arange(L * 14)  # [L*14]
    anchor_index = atom_index.view(L, 14)[:, [0, 1, 2]].reshape(-1)  # [L*3]
    x_gt_aligned_b = rigid_align(
        coords=x_gt_b.view(B, L * 14, 3),
        target=x_pred_b.view(B, L * 14, 3),
        mask=mask_b.view(B, L * 14),
        anchor_index=anchor_index,
    ).view(B, L, 14, 3)

    # Get the atom swap indices for each residue type
    restype_has_ambiguous_atoms = get_restype_has_ambiguous_atoms(x_gt.device.type)
    restype_atom_swap = get_restype_atom_swap(x_gt.device.type)

    swap_indices = restype_atom_swap[aatype_b]  # [B, L, 14]
    check_ambiguous = restype_has_ambiguous_atoms[aatype_b]  # [B, L]
    num_resolved_atoms = mask_b.sum(-1)  # [B, L]
    check_ambiguous = check_ambiguous & (num_resolved_atoms >= 3)

    swap_indices = swap_indices[check_ambiguous]  # [N, 14]
    query = x_gt_aligned_b[check_ambiguous]  # [N, 14, 3]
    query_swapped = gather_dim(query, dim=-2, index=swap_indices[..., None])  # [N, 14, 3]
    target = x_pred_b[check_ambiguous]  # [N, 14, 3]

    # Use local_mask to avoid overwriting the global mask_b reference
    local_mask = mask_b[check_ambiguous]  # [N, 14]
    mask_swapped = gather_dim(local_mask, dim=-2, index=swap_indices)  # [N, 14]

    # Compute per-residue RMSD with local-alignment
    err_orig = compute_rmsd(query, target, local_mask, align=True)
    err_swap = compute_rmsd(query_swapped, target, mask_swapped, align=True)

    # Choose the original or swapped GT structure based on which has lower local error
    swap_res = err_swap < err_orig  # [N]
    x_to_insert = torch.where(swap_res[:, None, None], query_swapped, query)  # [N, 14, 3]
    mask_to_insert = torch.where(swap_res[:, None], mask_swapped, local_mask)  # [N, 14]

    # Clone from the batched tensors that were NOT overwritten
    final_x_gt = x_gt_aligned_b.clone()
    final_mask = mask_b.clone()
    final_x_gt[check_ambiguous] = x_to_insert
    final_mask[check_ambiguous] = mask_to_insert

    # Restore the original batch dimensions
    return final_x_gt.view(*batch_dims, L, 14, 3), final_mask.view(*batch_dims, L, 14)
