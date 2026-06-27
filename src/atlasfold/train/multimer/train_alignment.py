"""Training-time alignment for multimer confidence losses."""

from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict
from typing import Any

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

_CPU_METADATA_DTYPES = {
    "crop_to_full_idx": torch.long,
    "asym_id": torch.long,
    "entity_id": torch.long,
    "res_idx": torch.long,
    "resolved_mask": torch.bool,
}


@dataclasses.dataclass(frozen=True)
class ChainInfo:
    asym_id: int
    entity_id: int
    indices: torch.Tensor
    res_idx: torch.Tensor


@dataclasses.dataclass(frozen=True)
class FullLabelMetadata:
    crop_to_full_idx: torch.Tensor
    asym_id: torch.Tensor
    entity_id: torch.Tensor
    res_idx: torch.Tensor
    resolved_mask: torch.Tensor
    target_positions: dict[int, torch.Tensor]
    chain_info: dict[int, ChainInfo]
    entity_to_sources: dict[int, list[int]]
    residue_lookup: dict[tuple[int, int], int]


def prepare_alignment_metadata(full_label: dict[str, Any]) -> FullLabelMetadata:
    """Build CPU metadata used by train-time chain permutation alignment."""
    cpu_tensors = {
        key: _as_cpu_tensor(full_label[key], dtype)
        for key, dtype in _CPU_METADATA_DTYPES.items()
    }
    crop_to_full_idx = cpu_tensors["crop_to_full_idx"]
    asym_id = cpu_tensors["asym_id"]
    entity_id = cpu_tensors["entity_id"]
    res_idx = cpu_tensors["res_idx"]
    resolved_mask = cpu_tensors["resolved_mask"]

    chain_info = _build_chain_info(asym_id, entity_id, res_idx)
    target_positions = _build_crop_positions_by_asym(crop_to_full_idx, asym_id)
    entity_to_sources = _build_entity_to_sources(chain_info)
    residue_lookup = _build_residue_lookup(asym_id, res_idx)

    return FullLabelMetadata(
        crop_to_full_idx=crop_to_full_idx,
        asym_id=asym_id,
        entity_id=entity_id,
        res_idx=res_idx,
        resolved_mask=resolved_mask,
        target_positions=target_positions,
        chain_info=chain_info,
        entity_to_sources=entity_to_sources,
        residue_lookup=residue_lookup,
    )


@torch.no_grad()
def get_aligned_gt_structure(
    x_pred: torch.Tensor,
    batch: dict[str, torch.Tensor],
    label: dict[str, torch.Tensor],
    full_label: dict[str, Any] | None = None,
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
    full_label: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align full-complex GT to a cropped training mini-rollout.

    The chain choice follows the KFold/AlphaFold-Multimer train-time heuristic:
    choose one anchor chain, align that source chain to each possible cropped
    anchor, then greedily assign same-entity chains by centroid distance.
    """
    device = x_pred.device
    seq_mask = batch["seq_mask"].to(device=device, dtype=torch.bool)
    valid_pos = torch.nonzero(seq_mask, as_tuple=False).flatten()
    metadata = _get_alignment_metadata(full_label)
    crop_to_full_idx = metadata.crop_to_full_idx
    if crop_to_full_idx.numel() != valid_pos.numel():
        raise ValueError(
            "full_label crop_to_full_idx length does not match cropped seq_mask: "
            f"{crop_to_full_idx.numel()} != {valid_pos.numel()}"
        )

    x_pred_crop = x_pred[valid_pos].float()
    if not _has_swappable_target(metadata):
        return _align_cropped_label(x_pred, batch, label)

    try:
        with torch.autocast(device.type, enabled=False):
            target_to_source_asym = _get_greedy_full_label_chain_mapping(
                crop_to_full_idx=crop_to_full_idx,
                full_label=full_label,
                x_pred_crop=x_pred_crop,
                metadata=metadata,
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
        metadata=metadata,
    )
    source_idx = source_idx.to(device=device)
    source_found = source_found.to(device=device)

    full_coords = full_label["coordinates"].to(device=device, dtype=torch.float32)
    full_mask = full_label["resolved_mask"].to(device=device, dtype=torch.bool)
    full_aatype = full_label["aatype_int"].to(device=device, dtype=torch.long)

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


def _as_cpu_tensor(tensor: Any, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().to(device="cpu", dtype=dtype)
    return torch.as_tensor(tensor, dtype=dtype, device="cpu")


def _get_alignment_metadata(full_label: dict[str, Any]) -> FullLabelMetadata:
    metadata = full_label.get("alignment_metadata")
    if isinstance(metadata, FullLabelMetadata):
        return metadata
    return prepare_alignment_metadata(full_label)


def _has_swappable_target(metadata: FullLabelMetadata) -> bool:
    target_asym_ids = set(metadata.target_positions.keys())
    return any(
        len(sources) > 1 and any(asym_id in target_asym_ids for asym_id in sources)
        for sources in metadata.entity_to_sources.values()
    )


def _get_full_chain_info(full_label: dict[str, Any]) -> dict[int, ChainInfo]:
    return _get_alignment_metadata(full_label).chain_info


def _build_chain_info(
    asym_id: torch.Tensor,
    entity_id: torch.Tensor,
    res_idx: torch.Tensor,
) -> dict[int, ChainInfo]:
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
    full_label: dict[str, Any],
) -> dict[int, torch.Tensor]:
    crop_to_full_idx = _as_cpu_tensor(crop_to_full_idx, torch.long)
    metadata = _get_alignment_metadata(full_label)
    if torch.equal(crop_to_full_idx, metadata.crop_to_full_idx):
        return metadata.target_positions
    return _build_crop_positions_by_asym(crop_to_full_idx, metadata.asym_id)


def _build_crop_positions_by_asym(
    crop_to_full_idx: torch.Tensor,
    asym_id: torch.Tensor,
) -> dict[int, torch.Tensor]:
    positions: dict[int, list[int]] = defaultdict(list)
    for crop_pos, full_idx in enumerate(crop_to_full_idx.tolist()):
        target_asym_id = int(asym_id[full_idx].item())
        if target_asym_id > 0:
            positions[target_asym_id].append(crop_pos)
    return {
        target_asym_id: torch.as_tensor(pos, dtype=torch.long)
        for target_asym_id, pos in positions.items()
    }


def _build_residue_lookup(
    asym_id: torch.Tensor,
    res_idx: torch.Tensor,
) -> dict[tuple[int, int], int]:
    return {
        (int(asym), int(res)): i
        for i, (asym, res) in enumerate(
            zip(asym_id.tolist(), res_idx.tolist(), strict=True)
        )
    }


def _map_target_positions_to_source_indices(
    target_positions: torch.Tensor,
    crop_to_full_idx: torch.Tensor,
    full_label: dict[str, Any],
    source_asym: int,
    lookup: dict[tuple[int, int], int],
    metadata: FullLabelMetadata | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_positions = _as_cpu_tensor(target_positions, torch.long)
    crop_to_full_idx = _as_cpu_tensor(crop_to_full_idx, torch.long)
    metadata = metadata or _get_alignment_metadata(full_label)
    full_res_idx = metadata.res_idx
    source_indices = []
    found = []
    for target_idx in crop_to_full_idx[target_positions].tolist():
        res_idx = int(full_res_idx[target_idx].item())
        source_idx = lookup.get((source_asym, res_idx))
        source_indices.append(int(target_idx) if source_idx is None else source_idx)
        found.append(source_idx is not None)
    return (
        torch.as_tensor(source_indices, dtype=torch.long),
        torch.as_tensor(found, dtype=torch.bool),
    )


def _map_crop_to_full_indices(
    crop_to_full_idx: torch.Tensor,
    full_label: dict[str, Any],
    target_to_source_asym: dict[int, int],
    metadata: FullLabelMetadata | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    crop_to_full_idx = _as_cpu_tensor(crop_to_full_idx, torch.long)
    metadata = metadata or _get_alignment_metadata(full_label)
    full_asym_id = metadata.asym_id
    full_res_idx = metadata.res_idx
    lookup = metadata.residue_lookup

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
        torch.as_tensor(source_indices, dtype=torch.long),
        torch.as_tensor(found, dtype=torch.bool),
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
    full_ca_mask: torch.Tensor,
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
        n_resolved_ca = int(full_ca_mask[info.indices].sum().item())
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
    full_label: dict[str, Any],
    x_pred_crop: torch.Tensor,
    full_coords: torch.Tensor | None = None,
    full_mask: torch.Tensor | None = None,
    metadata: FullLabelMetadata | None = None,
) -> dict[int, int]:
    crop_to_full_idx = _as_cpu_tensor(crop_to_full_idx, torch.long)
    metadata = metadata or _get_alignment_metadata(full_label)
    target_positions = metadata.target_positions
    target_asym_ids = set(target_positions.keys())
    if not target_asym_ids:
        return {}

    chain_info = metadata.chain_info
    full_ca_mask_cpu = metadata.resolved_mask[:, CA_IDX]
    entity_to_sources = metadata.entity_to_sources
    anchor_source = _select_anchor_source_asym(
        chain_info, entity_to_sources, target_asym_ids, full_ca_mask_cpu
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

    device = x_pred_crop.device
    if full_coords is None:
        full_coords = full_label["coordinates"].to(device=device, dtype=torch.float32)
    else:
        full_coords = full_coords.to(device=device, dtype=torch.float32)
    if full_mask is None:
        full_mask = full_label["resolved_mask"].to(device=device, dtype=torch.bool)
    else:
        full_mask = full_mask.to(device=device, dtype=torch.bool)

    lookup = metadata.residue_lookup
    best_cost = float("inf")
    best_mapping: dict[int, int] = {}

    for anchor_target in anchor_targets:
        anchor_pos_cpu = target_positions[anchor_target]
        anchor_source_idx_cpu, anchor_found_cpu = (
            _map_target_positions_to_source_indices(
                anchor_pos_cpu,
                crop_to_full_idx,
                full_label,
                anchor_source,
                lookup,
                metadata=metadata,
            )
        )
        anchor_pos = anchor_pos_cpu.to(device=device)
        anchor_source_idx = anchor_source_idx_cpu.to(device=device)
        anchor_found = anchor_found_cpu.to(device=device)
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
                target_pos_cpu = target_positions[target_asym]
                target_pos = target_pos_cpu.to(device=device)
                x_pred_target = x_pred_crop[target_pos, CA_IDX].float()
                for source_i, source_asym in enumerate(sources):
                    source_idx_cpu, source_found_cpu = (
                        _map_target_positions_to_source_indices(
                            target_pos_cpu,
                            crop_to_full_idx,
                            full_label,
                            source_asym,
                            lookup,
                            metadata=metadata,
                        )
                    )
                    source_idx = source_idx_cpu.to(device=device)
                    source_found = source_found_cpu.to(device=device)
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
