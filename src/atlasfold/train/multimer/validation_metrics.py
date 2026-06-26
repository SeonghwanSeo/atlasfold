"""Validation metrics for protein multimer structure prediction."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import torch

from atlasfold.train.utils.structure_metrics import (
    compute_lddt,
    compute_rmsd,
    compute_rmsd_atom14,
)
from atlasfold.utils.geometry.rigid_align import rigid_align_atom14

CA_IDX = 1


def _iter_chain_indices(
    batch: dict[str, torch.Tensor],
) -> list[dict[str, int | torch.Tensor]]:
    """Return contiguous protein chain residue indices from an unbatched feature dict."""
    seq_mask = batch["seq_mask"].bool()
    asym_id = batch["asym_id"]
    entity_id = batch["entity_id"]

    chains = []
    seen_asym_ids = torch.unique(asym_id[seq_mask], sorted=True)
    for aid in seen_asym_ids.tolist():
        if aid == 0:
            continue
        indices = torch.nonzero(seq_mask & (asym_id == aid), as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        chains.append(
            {
                "asym_id": int(aid),
                "entity_id": int(entity_id[indices[0]].item()),
                "indices": indices,
            }
        )
    return chains


def _generate_chain_permutations(
    chains: list[dict[str, int | torch.Tensor]],
    max_permutations: int = 720,
) -> Iterable[list[int]]:
    """Generate target-chain to source-chain maps for identical chain groups."""
    target_to_source = list(range(len(chains)))

    entity_groups: dict[int, list[int]] = defaultdict(list)
    for i, chain in enumerate(chains):
        entity_groups[int(chain["entity_id"])].append(i)

    swappable_groups: list[list[int]] = []
    for group in entity_groups.values():
        if len(group) <= 1:
            continue
        lengths = {
            int(chains[i]["indices"].numel())  # type: ignore[union-attr]
            for i in group
        }
        if len(lengths) == 1:
            swappable_groups.append(group)

    num_permutations = math.prod(math.factorial(len(g)) for g in swappable_groups)
    if num_permutations > max_permutations:
        yield target_to_source
        return

    if not swappable_groups:
        yield target_to_source
        return

    group_perms = [list(itertools.permutations(group)) for group in swappable_groups]
    for permuted_groups in itertools.product(*group_perms):
        mapping = list(target_to_source)
        for group, perm in zip(swappable_groups, permuted_groups, strict=True):
            for target_i, source_i in zip(group, perm, strict=True):
                mapping[target_i] = source_i
        yield mapping


def _apply_chain_permutation(
    x_gt: torch.Tensor,
    mask: torch.Tensor,
    chains: list[dict[str, int | torch.Tensor]],
    target_to_source: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    x_perm = x_gt.clone()
    mask_perm = mask.clone()
    for target_i, source_i in enumerate(target_to_source):
        if target_i == source_i:
            continue
        target_idx = chains[target_i]["indices"]
        source_idx = chains[source_i]["indices"]
        x_perm[target_idx] = x_gt[source_idx]
        mask_perm[target_idx] = mask[source_idx]
    return x_perm, mask_perm


def _align_on_ca(
    x_gt: torch.Tensor,
    x_pred: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    ca_mask = mask[:, CA_IDX]
    if ca_mask.sum() < 3:
        aligned = x_gt.clone()
        aligned.masked_fill_(~mask[..., None], 0.0)
        return aligned
    return rigid_align_atom14(
        x_gt.float(),
        x_pred.float(),
        mask.bool(),
        align_mode="ca",
        mask_to_zero=True,
    )


@torch.no_grad()
def get_aligned_gt_structure(
    x_pred: torch.Tensor,
    batch: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
    permute_chains: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align protein-only multimer GT to a prediction.

    This intentionally does not run residue-level atom symmetry correction. For the
    current protein-only multimer model, validation is made invariant to rigid-body
    transforms and identical whole-chain permutations, but side-chain atom14 order is
    left as represented in the labels.
    """
    x_gt = label["coordinates"].float()
    mask = label["resolved_mask"].bool()

    best_x, best_mask = x_gt, mask
    if permute_chains:
        chains = _iter_chain_indices(batch)
        best_rmsd = float("inf")
        for target_to_source in _generate_chain_permutations(chains):
            x_perm, mask_perm = _apply_chain_permutation(
                x_gt, mask, chains, target_to_source
            )
            x_aligned = _align_on_ca(x_perm, x_pred, mask_perm)
            ca_mask = mask_perm[:, CA_IDX]
            if ca_mask.any():
                rmsd = compute_rmsd(
                    x_aligned[:, CA_IDX],
                    x_pred[:, CA_IDX],
                    ca_mask,
                    align=False,
                ).item()
            else:
                rmsd = float("inf")
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_x, best_mask = x_perm, mask_perm

    return _align_on_ca(best_x, x_pred, best_mask), best_mask


@torch.no_grad()
def _compute_single_sample_metrics(
    x_pred: torch.Tensor,
    batch: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
) -> dict[str, float]:
    x_gt, mask = get_aligned_gt_structure(x_pred, batch, label)
    seq_mask = batch["seq_mask"].bool()
    ca_mask = mask[:, CA_IDX] & seq_mask

    metrics: dict[str, float] = {}
    metrics["complex/rmsd"] = compute_rmsd_atom14(x_pred, x_gt, mask).item()
    complex_lddt = compute_lddt(
        x_pred[:, CA_IDX],
        x_gt[:, CA_IDX],
        ca_mask,
        cutoff=15.0,
    ).item()
    metrics["complex/lddt"] = complex_lddt
    metrics["complex/lddt-ca"] = complex_lddt

    chains = _iter_chain_indices(batch)
    chain_rmsds = []
    chain_lddts = []
    for chain in chains:
        idx = chain["indices"]
        chain_mask = ca_mask[idx]
        if chain_mask.sum() < 2:
            continue
        chain_rmsds.append(
            compute_rmsd(
                x_pred[idx, CA_IDX],
                x_gt[idx, CA_IDX],
                chain_mask,
                align=True,
            ).item()
        )
        chain_lddts.append(
            compute_lddt(
                x_pred[idx, CA_IDX],
                x_gt[idx, CA_IDX],
                chain_mask,
                cutoff=15.0,
            ).item()
        )
    if chain_rmsds:
        metrics["chain/rmsd"] = float(np.mean(chain_rmsds))
    if chain_lddts:
        chain_lddt = float(np.mean(chain_lddts))
        metrics["chain/lddt"] = chain_lddt
        metrics["chain/lddt-ca"] = chain_lddt

    interface_lddts = []
    pdist_gt = torch.cdist(x_gt[:, CA_IDX], x_gt[:, CA_IDX])
    pdist_pred = torch.cdist(x_pred[:, CA_IDX], x_pred[:, CA_IDX])
    pair_score = torch.zeros_like(pdist_gt)
    for cutoff in (0.5, 1.0, 2.0, 4.0):
        pair_score += (torch.abs(pdist_gt - pdist_pred) < cutoff).float()
    pair_score *= 0.25

    for i, chain_i in enumerate(chains):
        idx_i = chain_i["indices"]
        for chain_j in chains[i + 1 :]:
            idx_j = chain_j["indices"]
            pair_mask = ca_mask[idx_i, None] & ca_mask[idx_j][None, :]
            pair_mask &= pdist_gt[idx_i[:, None], idx_j[None, :]] < 15.0
            if pair_mask.any():
                score = pair_score[idx_i[:, None], idx_j[None, :]]
                interface_lddts.append(score[pair_mask].mean().item())
    if interface_lddts:
        iface_lddt = float(np.mean(interface_lddts))
        metrics["interface/lddt"] = iface_lddt
        metrics["interface/lddt-ca"] = iface_lddt

    return metrics


def compute_validation_metric(
    x_pred: torch.Tensor,
    batch: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
    rank_idx: int | None = None,
) -> dict[str, float]:
    """Compute aggregate validation metrics for multimer diffusion samples."""
    sample_metrics = [
        _compute_single_sample_metrics(x_pred_i, batch, label) for x_pred_i in x_pred
    ]
    metric_names = sorted(set().union(*(m.keys() for m in sample_metrics)))

    all_metrics: dict[str, float] = {}
    for name in metric_names:
        values = [m[name] for m in sample_metrics if name in m]
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        all_metrics[f"avg/{name}"] = float(arr.mean())
        if "rmsd" in name:
            all_metrics[f"top/{name}"] = float(arr.min())
        else:
            all_metrics[f"top/{name}"] = float(arr.max())
        if rank_idx is not None and name in sample_metrics[rank_idx]:
            all_metrics[f"rank/{name}"] = float(sample_metrics[rank_idx][name])

    return all_metrics
