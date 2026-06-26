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
from atlasfold.train.utils.structure_metrics import (
    get_aligned_gt_structure as get_atom_aligned_gt_structure,
)
from atlasfold.utils.geometry.rigid_align import rigid_align_atom14

CA_IDX = 1
CONTACT_CUTOFF = 8.0


def compute_distogram_contact_probability(
    logits: torch.Tensor,
    boundaries: torch.Tensor,
    cutoff: float = CONTACT_CUTOFF,
) -> torch.Tensor:
    """Compute contact probability from distogram logits."""
    probs = torch.softmax(logits.float(), dim=-1)
    if probs.shape[-1] != boundaries.numel() + 1:
        raise ValueError(
            "distogram logits/bin mismatch: "
            f"{probs.shape[-1]} != {boundaries.numel()} + 1"
        )

    if boundaries.numel() == 0:
        contact_bins = torch.ones(1, device=logits.device, dtype=torch.bool)
    elif boundaries.numel() == 1:
        centers = boundaries.new_tensor([boundaries[0] * 0.5, boundaries[0] * 1.5])
        contact_bins = centers < cutoff
    else:
        step = boundaries[1] - boundaries[0]
        first_center = boundaries[0] - 0.5 * step
        middle_centers = 0.5 * (boundaries[:-1] + boundaries[1:])
        last_center = boundaries[-1] + 0.5 * step
        centers = torch.cat(
            [first_center[None], middle_centers, last_center[None]], dim=0
        )
        contact_bins = centers < cutoff

    if not bool(contact_bins.any()):
        contact_bins[0] = True
    return probs[..., contact_bins].sum(dim=-1)


def compute_global_pde(
    pde_score: torch.Tensor,
    prob_contact: torch.Tensor,
    seq_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute contact-probability-weighted global PDE for sample ranking."""
    if pde_score.ndim != 3:
        raise ValueError(f"Expected pde_score shape [N, L, L], got {pde_score.shape}")
    if prob_contact.shape != pde_score.shape[-2:]:
        raise ValueError(
            "prob_contact shape does not match pde_score residues: "
            f"{prob_contact.shape} != {pde_score.shape[-2:]}"
        )

    pair_mask = seq_mask.bool()[:, None] & seq_mask.bool()[None, :]
    prob_contact = prob_contact.float() * pair_mask.float()
    weighted_pde = torch.einsum("sij,ij->s", pde_score.float(), prob_contact)
    return weighted_pde / prob_contact.sum(dim=(-1, -2)).clamp_min(1e-6)


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


def _get_full_chain_info(full_label: dict[str, torch.Tensor]) -> dict[int, dict]:
    asym_id = full_label["asym_id"]
    entity_id = full_label["entity_id"]
    res_idx = full_label["res_idx"]

    chain_info = {}
    for aid in torch.unique(asym_id, sorted=True).tolist():
        if aid == 0:
            continue
        indices = torch.nonzero(asym_id == aid, as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        chain_info[int(aid)] = {
            "entity_id": int(entity_id[indices[0]].item()),
            "indices": indices,
            "res_idx": res_idx[indices],
        }
    return chain_info


def _generate_full_label_chain_mappings(
    crop_to_full_idx: torch.Tensor,
    full_label: dict[str, torch.Tensor],
    max_permutations: int = 720,
) -> Iterable[dict[int, int]]:
    """Generate target-asym to source-asym maps for same-entity full chains."""
    full_asym_id = full_label["asym_id"]
    target_asym_ids = [
        int(aid)
        for aid in torch.unique(full_asym_id[crop_to_full_idx], sorted=True).tolist()
        if aid > 0
    ]
    if not target_asym_ids:
        yield {}
        return

    chain_info = _get_full_chain_info(full_label)
    entity_to_targets: dict[int, list[int]] = defaultdict(list)
    entity_to_sources: dict[int, list[int]] = defaultdict(list)
    for aid in target_asym_ids:
        if aid in chain_info:
            entity_to_targets[int(chain_info[aid]["entity_id"])].append(aid)
    for aid, info in chain_info.items():
        entity_to_sources[int(info["entity_id"])].append(aid)

    swappable_groups: list[tuple[list[int], list[int]]] = []
    for entity, targets in entity_to_targets.items():
        sources = entity_to_sources[entity]
        if len(sources) > 1:
            swappable_groups.append((targets, sources))

    if not swappable_groups:
        yield {}
        return

    num_permutations = math.prod(
        math.factorial(len(sources)) // math.factorial(len(sources) - len(targets))
        for targets, sources in swappable_groups
    )
    if num_permutations > max_permutations:
        yield {}
        return

    group_perms = [
        list(itertools.permutations(sources, len(targets)))
        for targets, sources in swappable_groups
    ]
    for permuted_groups in itertools.product(*group_perms):
        mapping = {}
        for (targets, _), sources_perm in zip(
            swappable_groups, permuted_groups, strict=True
        ):
            for target_asym, source_asym in zip(targets, sources_perm, strict=True):
                mapping[int(target_asym)] = int(source_asym)
        yield mapping


def _map_crop_to_full_indices(
    crop_to_full_idx: torch.Tensor,
    full_label: dict[str, torch.Tensor],
    target_to_source_asym: dict[int, int],
) -> torch.Tensor:
    full_asym_id = full_label["asym_id"]
    full_res_idx = full_label["res_idx"]

    lookup = {
        (int(asym), int(res_idx)): i
        for i, (asym, res_idx) in enumerate(
            zip(full_asym_id.tolist(), full_res_idx.tolist(), strict=True)
        )
    }

    source_indices = []
    for idx in crop_to_full_idx.tolist():
        target_asym = int(full_asym_id[idx].item())
        source_asym = target_to_source_asym.get(target_asym, target_asym)
        res_idx = int(full_res_idx[idx].item())
        source_indices.append(lookup.get((source_asym, res_idx), int(idx)))

    return torch.as_tensor(
        source_indices, device=crop_to_full_idx.device, dtype=torch.long
    )


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
def get_aligned_gt_structure_from_full_label(
    x_pred: torch.Tensor,
    batch: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
    full_label: dict[str, torch.Tensor],
    max_chain_permutations: int = 720,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align full-complex GT to a cropped multimer prediction.

    ``label`` is used only as a fallback. The full label carries uncropped chain
    context, so same-entity chain choices can be resolved even when the crop
    contains a partial or asymmetric subset of chains.
    """
    device = x_pred.device
    seq_mask = batch["seq_mask"].to(device=device, dtype=torch.bool)
    valid_pos = torch.nonzero(seq_mask, as_tuple=False).flatten()
    crop_to_full_idx = full_label["crop_to_full_idx"].to(device=device, dtype=torch.long)
    if crop_to_full_idx.numel() != valid_pos.numel():
        raise ValueError(
            "full_label crop_to_full_idx length does not match cropped seq_mask: "
            f"{crop_to_full_idx.numel()} != {valid_pos.numel()}"
        )

    full_label = {
        k: v.to(device=device)
        for k, v in full_label.items()
        if isinstance(v, torch.Tensor)
    }
    x_pred_crop = x_pred[valid_pos].float()
    full_coords = full_label["coordinates"].float()
    full_mask = full_label["resolved_mask"].bool()
    full_aatype = full_label["aatype_int"].long()

    best_x = None
    best_mask = None
    best_aatype = None
    best_rmsd = float("inf")

    for target_to_source_asym in _generate_full_label_chain_mappings(
        crop_to_full_idx, full_label, max_permutations=max_chain_permutations
    ):
        source_idx = _map_crop_to_full_indices(
            crop_to_full_idx, full_label, target_to_source_asym
        )
        x_candidate = full_coords[source_idx]
        mask_candidate = full_mask[source_idx]
        ca_mask = mask_candidate[:, CA_IDX]
        if ca_mask.sum() < 3:
            continue
        x_aligned = _align_on_ca(x_candidate, x_pred_crop, mask_candidate)
        rmsd = compute_rmsd(
            x_aligned[:, CA_IDX],
            x_pred_crop[:, CA_IDX],
            ca_mask,
            align=False,
        ).item()
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_x = x_candidate
            best_mask = mask_candidate
            best_aatype = full_aatype[source_idx]

    if best_x is None or best_mask is None or best_aatype is None:
        return get_aligned_gt_structure(x_pred, batch, label)

    x_aligned, mask_aligned = get_atom_aligned_gt_structure(
        x_gt=best_x,
        x_pred=x_pred_crop,
        aatype=best_aatype,
        mask=best_mask,
    )

    out_x = torch.zeros_like(x_pred)
    out_mask = torch.zeros(x_pred.shape[:-1], device=device, dtype=torch.bool)
    out_x[valid_pos] = x_aligned
    out_mask[valid_pos] = mask_aligned
    return out_x, out_mask


@torch.no_grad()
def get_aligned_gt_structure(
    x_pred: torch.Tensor,
    batch: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
    permute_chains: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align protein-only multimer GT to a prediction.

    The alignment is invariant to rigid transforms, same-entity whole-chain
    permutations, and residue-level atom14 ambiguity.
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

    if "aatype_int" not in batch:
        return _align_on_ca(best_x, x_pred, best_mask), best_mask

    return get_atom_aligned_gt_structure(
        x_gt=best_x,
        x_pred=x_pred.float(),
        aatype=batch["aatype_int"],
        mask=best_mask,
    )


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
