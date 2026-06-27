"""Training-time alignment for multimer confidence losses."""

from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict

import torch

from atlasfold.train.multimer.validation_metrics import (
    get_aligned_gt_structure as get_validation_aligned_gt_structure,
)
from atlasfold.train.utils.structure_metrics import (
    get_aligned_gt_structure as get_atom_aligned_gt_structure,
)
from atlasfold.utils.geometry.rigid_align import get_rigid_transform

CA_IDX = 1

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ChainInfo:
    asym_id: int
    entity_id: int
    indices: torch.Tensor
    res_idx: torch.Tensor


@torch.no_grad()
def get_aligned_gt_structure(
    x_pred: torch.Tensor,
    batch: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
    full_label: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align GT to a training mini-rollout using the train-time policy."""
    if full_label is None:
        return _align_cropped_label(x_pred, batch, label)
    return get_aligned_gt_structure_from_full_label(x_pred, batch, label, full_label)


@torch.no_grad()
def get_aligned_gt_structure_from_full_label(
    x_pred: torch.Tensor,
    batch: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
    full_label: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align full-complex GT to a cropped training mini-rollout.

    The chain choice follows the KFold/AlphaFold-Multimer train-time heuristic:
    choose one anchor chain, align that source chain to each possible cropped
    anchor, then greedily assign same-entity chains by centroid distance.
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

    try:
        with torch.autocast(device.type, enabled=False):
            target_to_source_asym = _get_greedy_full_label_chain_mapping(
                crop_to_full_idx=crop_to_full_idx,
                full_label=full_label,
                x_pred_crop=x_pred_crop,
            )
    except (RuntimeError, ValueError, KeyError, IndexError) as e:
        logger.warning(
            "Greedy chain alignment failed; falling back to cropped label: %s", e
        )
        return _align_cropped_label(x_pred, batch, label)

    source_idx, source_found = _map_crop_to_full_indices(
        crop_to_full_idx=crop_to_full_idx,
        full_label=full_label,
        target_to_source_asym=target_to_source_asym,
    )
    full_coords = full_label["coordinates"].float()
    full_mask = full_label["resolved_mask"].bool()
    full_aatype = full_label["aatype_int"].long()

    x_candidate = full_coords[source_idx]
    mask_candidate = full_mask[source_idx] & source_found[:, None]
    if mask_candidate[:, CA_IDX].sum() < 3:
        return _align_cropped_label(x_pred, batch, label)

    x_aligned, mask_aligned = get_atom_aligned_gt_structure(
        x_gt=x_candidate,
        x_pred=x_pred_crop,
        aatype=full_aatype[source_idx],
        mask=mask_candidate,
    )

    out_x = torch.zeros_like(x_pred)
    out_mask = torch.zeros(x_pred.shape[:-1], device=device, dtype=torch.bool)
    out_x[valid_pos] = x_aligned
    out_mask[valid_pos] = mask_aligned
    return out_x, out_mask


def _align_cropped_label(
    x_pred: torch.Tensor,
    batch: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    return get_validation_aligned_gt_structure(x_pred, batch, label)


def _get_full_chain_info(full_label: dict[str, torch.Tensor]) -> dict[int, ChainInfo]:
    asym_id = full_label["asym_id"]
    entity_id = full_label["entity_id"]
    res_idx = full_label["res_idx"]

    chain_info: dict[int, ChainInfo] = {}
    for aid in torch.unique(asym_id, sorted=True).tolist():
        if aid == 0:
            continue
        indices = torch.nonzero(asym_id == aid, as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        asym_id_int = int(aid)
        chain_info[asym_id_int] = ChainInfo(
            asym_id=asym_id_int,
            entity_id=int(entity_id[indices[0]].item()),
            indices=indices,
            res_idx=res_idx[indices],
        )
    return chain_info


def _get_crop_positions_by_asym(
    crop_to_full_idx: torch.Tensor,
    full_label: dict[str, torch.Tensor],
) -> dict[int, torch.Tensor]:
    full_asym_id = full_label["asym_id"]
    positions: dict[int, list[int]] = defaultdict(list)
    for crop_pos, full_idx in enumerate(crop_to_full_idx.tolist()):
        asym_id = int(full_asym_id[full_idx].item())
        if asym_id > 0:
            positions[asym_id].append(crop_pos)
    return {
        asym_id: torch.as_tensor(pos, device=crop_to_full_idx.device, dtype=torch.long)
        for asym_id, pos in positions.items()
    }


def _build_full_residue_lookup(
    full_label: dict[str, torch.Tensor],
) -> dict[tuple[int, int], int]:
    full_asym_id = full_label["asym_id"]
    full_res_idx = full_label["res_idx"]
    return {
        (int(asym), int(res_idx)): i
        for i, (asym, res_idx) in enumerate(
            zip(full_asym_id.tolist(), full_res_idx.tolist(), strict=True)
        )
    }


def _map_target_positions_to_source_indices(
    target_positions: torch.Tensor,
    crop_to_full_idx: torch.Tensor,
    full_label: dict[str, torch.Tensor],
    source_asym: int,
    lookup: dict[tuple[int, int], int],
) -> tuple[torch.Tensor, torch.Tensor]:
    full_res_idx = full_label["res_idx"]
    source_indices = []
    found = []
    for target_idx in crop_to_full_idx[target_positions].tolist():
        res_idx = int(full_res_idx[target_idx].item())
        source_idx = lookup.get((source_asym, res_idx))
        source_indices.append(int(target_idx) if source_idx is None else source_idx)
        found.append(source_idx is not None)
    return (
        torch.as_tensor(source_indices, device=crop_to_full_idx.device, dtype=torch.long),
        torch.as_tensor(found, device=crop_to_full_idx.device, dtype=torch.bool),
    )


def _map_crop_to_full_indices(
    crop_to_full_idx: torch.Tensor,
    full_label: dict[str, torch.Tensor],
    target_to_source_asym: dict[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    full_asym_id = full_label["asym_id"]
    full_res_idx = full_label["res_idx"]
    lookup = _build_full_residue_lookup(full_label)

    source_indices = []
    found = []
    for target_idx in crop_to_full_idx.tolist():
        target_asym = int(full_asym_id[target_idx].item())
        source_asym = target_to_source_asym.get(target_asym, target_asym)
        res_idx = int(full_res_idx[target_idx].item())
        source_idx = lookup.get((source_asym, res_idx))
        source_indices.append(int(target_idx) if source_idx is None else source_idx)
        found.append(source_idx is not None)

    return (
        torch.as_tensor(source_indices, device=crop_to_full_idx.device, dtype=torch.long),
        torch.as_tensor(found, device=crop_to_full_idx.device, dtype=torch.bool),
    )


def _build_entity_to_sources(
    chain_info: dict[int, ChainInfo],
) -> dict[int, list[int]]:
    entity_to_sources: dict[int, list[int]] = defaultdict(list)
    for asym_id, info in chain_info.items():
        entity_to_sources[info.entity_id].append(asym_id)
    for sources in entity_to_sources.values():
        sources.sort()
    return entity_to_sources


def _select_anchor_source_asym(
    chain_info: dict[int, ChainInfo],
    entity_to_sources: dict[int, list[int]],
    target_asym_ids: set[int],
    full_mask: torch.Tensor,
) -> int | None:
    candidates = [
        asym_id
        for sources in entity_to_sources.values()
        if len(sources) > 1 and any(aid in target_asym_ids for aid in sources)
        for asym_id in sources
    ]
    if not candidates:
        return None

    def priority(asym_id: int) -> tuple[int, int, int, int]:
        info = chain_info[asym_id]
        n_resolved_ca = int(full_mask[info.indices, CA_IDX].sum().item())
        return (
            -len(entity_to_sources[info.entity_id]),
            n_resolved_ca,
            int(info.indices.numel()),
            asym_id,
        )

    return max(candidates, key=priority)


def _find_greedy_chain_assignment(
    distances: torch.Tensor,
    n_resolved: torch.Tensor,
) -> list[int]:
    """Greedily assign each cropped chain to an unused source chain."""
    distances = distances.clone()
    distances.nan_to_num_(nan=1e9, posinf=1e9, neginf=1e9)
    order = torch.argsort(n_resolved.max(dim=1).values, descending=True).tolist()
    assignment = [-1] * distances.shape[0]
    for target_i in order:
        source_i = int(torch.argmin(distances[target_i]).item())
        assignment[target_i] = source_i
        distances[:, source_i] = float("inf")
    return assignment


def _get_greedy_full_label_chain_mapping(
    crop_to_full_idx: torch.Tensor,
    full_label: dict[str, torch.Tensor],
    x_pred_crop: torch.Tensor,
) -> dict[int, int]:
    target_positions = _get_crop_positions_by_asym(crop_to_full_idx, full_label)
    target_asym_ids = set(target_positions.keys())
    if not target_asym_ids:
        return {}

    chain_info = _get_full_chain_info(full_label)
    full_coords = full_label["coordinates"].float()
    full_mask = full_label["resolved_mask"].bool()
    entity_to_sources = _build_entity_to_sources(chain_info)
    anchor_source = _select_anchor_source_asym(
        chain_info, entity_to_sources, target_asym_ids, full_mask
    )
    if anchor_source is None:
        return {}

    anchor_entity = chain_info[anchor_source].entity_id
    anchor_targets = [
        asym_id
        for asym_id in entity_to_sources[anchor_entity]
        if asym_id in target_asym_ids
    ]
    if not anchor_targets:
        return {}

    lookup = _build_full_residue_lookup(full_label)
    best_cost = float("inf")
    best_mapping: dict[int, int] = {}

    for anchor_target in anchor_targets:
        anchor_pos = target_positions[anchor_target]
        anchor_source_idx, anchor_found = _map_target_positions_to_source_indices(
            anchor_pos, crop_to_full_idx, full_label, anchor_source, lookup
        )
        anchor_mask = anchor_found & full_mask[anchor_source_idx, CA_IDX]
        if anchor_mask.sum() < 3:
            continue

        rt, trans = get_rigid_transform(
            full_coords[anchor_source_idx, CA_IDX].float(),
            x_pred_crop[anchor_pos, CA_IDX].float(),
            anchor_mask,
        )

        total_cost = 0.0
        mapping: dict[int, int] = {}
        for _entity, sources in entity_to_sources.items():
            if len(sources) <= 1:
                continue
            targets = [asym_id for asym_id in sources if asym_id in target_asym_ids]
            if not targets:
                continue

            distances = torch.full(
                (len(targets), len(sources)),
                float("inf"),
                device=x_pred_crop.device,
            )
            n_resolved = torch.zeros_like(distances)
            for target_i, target_asym in enumerate(targets):
                target_pos = target_positions[target_asym]
                x_pred_target = x_pred_crop[target_pos, CA_IDX].float()
                for source_i, source_asym in enumerate(sources):
                    source_idx, source_found = _map_target_positions_to_source_indices(
                        target_pos, crop_to_full_idx, full_label, source_asym, lookup
                    )
                    mask = source_found & full_mask[source_idx, CA_IDX]
                    n_resolved[target_i, source_i] = mask.sum()
                    if not mask.any():
                        continue

                    x_gt_source = full_coords[source_idx, CA_IDX].float() @ rt + trans
                    x_gt_centroid = (x_gt_source * mask[:, None]).sum(dim=0)
                    x_gt_centroid = x_gt_centroid / mask.sum().clamp_min(1)
                    x_pred_centroid = (x_pred_target * mask[:, None]).sum(dim=0)
                    x_pred_centroid = x_pred_centroid / mask.sum().clamp_min(1)
                    distances[target_i, source_i] = torch.norm(
                        x_pred_centroid - x_gt_centroid
                    )

            assignment = _find_greedy_chain_assignment(distances, n_resolved)
            for target_i, source_i in enumerate(assignment):
                if source_i < 0:
                    continue
                target_asym = targets[target_i]
                source_asym = sources[source_i]
                mapping[target_asym] = source_asym
                distance = distances[target_i, source_i]
                total_cost += distance.item() if torch.isfinite(distance) else 1e9

        if total_cost < best_cost:
            best_cost = total_cost
            best_mapping = mapping

    if best_cost == float("inf"):
        return {}
    return {
        target_asym: source_asym
        for target_asym, source_asym in best_mapping.items()
        if target_asym != source_asym
    }
