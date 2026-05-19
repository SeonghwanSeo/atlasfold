"""Model validation metrics for structure prediction tasks."""

import numpy as np
import torch

from atlasfold.utils.structure_metrics import (
    compute_lddt,
    compute_lddt_ca,
    compute_rmsd,
    get_aligned_gt_structure,
)


def compute_validation_metric(
    x_pred: torch.Tensor,
    batch: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
    rank_idx: torch.Tensor | None = None,
) -> dict:
    """Get structure prediction metrics.

    Parameters
    ----------
    x_pred : torch.Tensor
        Predicted coordinates from the model. Shape (B, N, L, 14, 3).
    batch : dict[str, torch.Tensor]
        Input features.
    label : dict[str, torch.Tensor]
        Ground truth labels.
    rank_idx : torch.Tensor | None
        Top-plddt ranking index for each sample, shape (B,).

    Returns
    -------
    metrics: dict
        The computed metrics
    """
    B, N, _, _, _ = x_pred.shape
    aatype = batch["aatype_int"]  # [B, L]
    x_gt = label["coordinates"]  # [B, L, 14, 3]
    mask = label["resolved_mask"]  # [B, L, 14]

    all_metrics = {
        "avg/rmsd": [],
        "avg/lddt": [],
        "avg/lddt-ca": [],
        "top/rmsd": [],
        "top/lddt": [],
        "top/lddt-ca": [],
    }
    if rank_idx is not None:
        all_metrics["rank/rmsd"] = []
        all_metrics["rank/lddt"] = []
        all_metrics["rank/lddt-ca"] = []

    for b_i in range(B):
        rmsds = []
        lddts = []
        lddt_cas = []
        for n_i in range(N):
            x_pred_i = x_pred[b_i, n_i]  # [L, 14, 3]
            x_gt_i = x_gt[b_i]  # [L, 14, 3]
            m_i = mask[b_i]  # [L, 14]
            aa_i = aatype[b_i]  # [L]

            # Align the predicted structure
            x_gt_i, m_i = get_aligned_gt_structure(x_gt_i, x_pred_i, m_i, aa_i)

            # Compute metrics for this prediction
            rmsds.append(compute_rmsd(x_pred_i, x_gt_i, m_i).item())
            lddts.append(compute_lddt(x_pred_i, x_gt_i, m_i).item())
            lddt_cas.append(compute_lddt_ca(x_pred_i, x_gt_i, m_i).item())

        # Aggregate metrics across N predictions
        all_metrics["avg/rmsd"].append(np.mean(rmsds))
        all_metrics["avg/lddt"].append(np.mean(lddts))
        all_metrics["avg/lddt-ca"].append(np.mean(lddt_cas))
        all_metrics["top/rmsd"].append(min(rmsds))
        all_metrics["top/lddt"].append(max(lddts))
        all_metrics["top/lddt-ca"].append(max(lddt_cas))

        if rank_idx is not None:
            rank_i = int(rank_idx[b_i].item())
            all_metrics["rank/rmsd"].append(rmsds[rank_i])
            all_metrics["rank/lddt"].append(lddts[rank_i])
            all_metrics["rank/lddt-ca"].append(lddt_cas[rank_i])

    return all_metrics
