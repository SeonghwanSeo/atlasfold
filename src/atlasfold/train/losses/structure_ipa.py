"""AlphaFold-style structure losses for the IPA regression models."""

from __future__ import annotations

import functools
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from atlasfold.common import residue_constants as rc

ALA_INDEX = rc.restype_orders["A"]
PRO_INDEX = rc.restype_orders["P"]
CYS_INDEX = rc.restype_orders["C"]
CYS_SG_INDEX = rc.restype_atom14_order["CYS"]["SG"]

Reduction = Literal["mean", "none"]
ViolationLossStyle = Literal["af2", "af_multimer"]


def _reduce_batch(value: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    if reduction == "mean":
        return value.mean()
    if reduction == "none":
        return value
    raise ValueError(f"Unknown reduction: {reduction}")


def normalize_aatype(aatype: torch.Tensor) -> torch.Tensor:
    """Use alanine geometry for X, matching the IPA structure module."""

    aatype = aatype.long().clamp(0, 20)
    return torch.where(aatype == 20, torch.full_like(aatype, ALA_INDEX), aatype)


@functools.lru_cache(maxsize=1)
def _geometry_constants() -> dict[str, torch.Tensor]:
    swap = np.tile(np.arange(14, dtype=np.int64), (21, 1))
    ambiguous = np.zeros((21, 14), dtype=np.float32)
    base_atoms = np.zeros((21, 8, 3), dtype=np.int64)
    group_exists = np.zeros((21, 8), dtype=np.float32)
    chi_mask = np.zeros((21, 4), dtype=np.float32)
    chi_periodic = np.zeros((21, 4), dtype=np.float32)
    radii = np.zeros((21, 14), dtype=np.float32)

    # AlphaFold rigid group base atoms: point on -x axis, origin, xy point.
    for i, letter in enumerate(rc.restypes[:20]):
        name = rc.restype_1to3[letter]
        order = rc.restype_atom14_order[name]
        base_atoms[i, 0] = [order["C"], order["CA"], order["N"]]
        group_exists[i, 0] = 1.0
        base_atoms[i, 3] = [order["CA"], order["C"], order["O"]]
        group_exists[i, 3] = 1.0
        chis = rc.chi_angles_atoms[name]
        for chi_i, atoms in enumerate(chis):
            base_atoms[i, 4 + chi_i] = [order[a] for a in atoms[1:4]]
            group_exists[i, 4 + chi_i] = 1.0
            chi_mask[i, chi_i] = 1.0
        for atom_name, atom_i in order.items():
            radii[i, atom_i] = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8}.get(
                atom_name[0], 1.7
            )

    # Pi-periodic chi angles used by AlphaFold's supervised chi objective.
    for resname, indices in {"ASP": (1,), "GLU": (2,), "PHE": (1,), "TYR": (1,)}.items():
        i = rc.restype_orders[rc.restype_3to1[resname]]
        for chi_i in indices:
            chi_periodic[i, chi_i] = 1.0

    for resname, (left, right) in rc.restype_ambiguous_atoms.items():
        i = rc.restype_orders[rc.restype_3to1[resname]]
        order = rc.restype_atom14_order[resname]
        for a, b in zip(left, right, strict=True):
            ai, bi = order[a], order[b]
            swap[i, ai], swap[i, bi] = bi, ai
            ambiguous[i, ai] = ambiguous[i, bi] = 1.0

    # X is alanine for every training geometry lookup.
    for array in (
        swap,
        ambiguous,
        base_atoms,
        group_exists,
        chi_mask,
        chi_periodic,
        radii,
    ):
        array[20] = array[ALA_INDEX]

    atom_mask = rc.restype_atom14_mask.astype(np.float32)

    return {
        "swap": torch.from_numpy(swap),
        "ambiguous": torch.from_numpy(ambiguous),
        "base_atoms": torch.from_numpy(base_atoms),
        "group_exists": torch.from_numpy(group_exists),
        "chi_mask": torch.from_numpy(chi_mask),
        "chi_periodic": torch.from_numpy(chi_periodic),
        "atom_mask": torch.from_numpy(atom_mask),
        "radii": torch.from_numpy(radii),
    }


def _constant(name: str, like: torch.Tensor) -> torch.Tensor:
    return _geometry_constants()[name].to(device=like.device)


def _gather_atoms(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    # x [B,L,A,...], indices [B,L,...]
    trailing = x.shape[3:]
    flat = indices.reshape(indices.shape[0], indices.shape[1], -1)
    expanded = flat
    for _ in trailing:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand(*flat.shape, *trailing)
    gathered = torch.gather(x, 2, expanded)
    return gathered.reshape(*indices.shape, *trailing)


def _frames_from_points(
    point_neg_x: torch.Tensor, origin: torch.Tensor, point_xy: torch.Tensor
) -> torch.Tensor:
    e0 = F.normalize(origin - point_neg_x, dim=-1, eps=1e-8)
    e1 = point_xy - origin
    e1 = F.normalize(e1 - (e1 * e0).sum(-1, keepdim=True) * e0, dim=-1, eps=1e-8)
    e2 = torch.linalg.cross(e0, e1, dim=-1)
    rotation = torch.stack((e0, e1, e2), dim=-1)
    out = torch.zeros(*origin.shape[:-1], 4, 4, device=origin.device, dtype=torch.float32)
    out[..., :3, :3] = rotation.float()
    out[..., :3, 3] = origin.float()
    out[..., 3, 3] = 1.0
    return out


def _dihedral(points: torch.Tensor) -> torch.Tensor:
    p0, p1, p2, p3 = points.unbind(-2)
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = F.normalize(b1, dim=-1, eps=1e-8)
    v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
    x = (v * w).sum(-1)
    y = (torch.linalg.cross(b1n, v, dim=-1) * w).sum(-1)
    norm = torch.sqrt(x.square() + y.square() + 1e-8)
    return torch.stack((y / norm, x / norm), dim=-1)


def _cdist(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    squared_distance = (x.unsqueeze(-2) - y.unsqueeze(-3)).square().sum(-1)
    return torch.sqrt(squared_distance + eps)


@torch.no_grad()
def make_structure_labels(
    aatype: torch.Tensor, coordinates: torch.Tensor, resolved_mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Derive AlphaFold structure supervision from padded atom14 labels."""

    with torch.autocast(coordinates.device.type, enabled=False):
        aatype = normalize_aatype(aatype)
        coordinates = coordinates.float()
        resolved_mask = resolved_mask.bool()
        atom_exists = _constant("atom_mask", coordinates)[aatype].bool()
        gt_exists = atom_exists & resolved_mask

        swap = _constant("swap", coordinates)[aatype]
        alt_positions = torch.gather(
            coordinates, 2, swap.unsqueeze(-1).expand(*swap.shape, 3)
        )
        alt_exists = torch.gather(gt_exists, 2, swap)
        ambiguous = _constant("ambiguous", coordinates)[aatype].bool()

        base = _constant("base_atoms", coordinates)[aatype]
        base_pos = _gather_atoms(coordinates, base)
        base_mask = _gather_atoms(gt_exists.unsqueeze(-1), base).squeeze(-1)
        frames = _frames_from_points(
            base_pos[..., 0, :], base_pos[..., 1, :], base_pos[..., 2, :]
        )
        # AlphaFold's historical backbone-frame convention mirrors x and z.
        mirror = torch.eye(3, device=coordinates.device)
        mirror[0, 0] = mirror[2, 2] = -1.0
        frames[..., 0, :3, :3] = frames[..., 0, :3, :3] @ mirror
        group_exists = _constant("group_exists", coordinates)[aatype].bool()
        frame_exists = group_exists & base_mask.all(-1)

        alt_frames = frames.clone()
        ambiguous_group = torch.zeros_like(frame_exists)
        for resname in ("ASP", "GLU", "PHE", "TYR"):
            idx = rc.restype_orders[rc.restype_3to1[resname]]
            n_chi = len(rc.chi_angles_atoms[resname])
            sel = aatype == idx
            ambiguous_group[..., 3 + n_chi] |= sel
        rot_x_pi = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=coordinates.device))
        alt_frames[..., :3, :3] = torch.where(
            ambiguous_group[..., None, None],
            frames[..., :3, :3] @ rot_x_pi,
            frames[..., :3, :3],
        )

        chi_atoms = torch.zeros(
            *aatype.shape, 4, 4, dtype=torch.long, device=coordinates.device
        )
        chi_mask = _constant("chi_mask", coordinates)[aatype].bool()
        for i, letter in enumerate(rc.restypes[:20]):
            name = rc.restype_1to3[letter]
            order = rc.restype_atom14_order[name]
            for chi_i, names in enumerate(rc.chi_angles_atoms[name]):
                chi_atoms[aatype == i, chi_i] = torch.tensor(
                    [order[n] for n in names], device=coordinates.device
                )
        chi_pos = _gather_atoms(
            coordinates, chi_atoms.reshape(*aatype.shape, 16)
        ).reshape(*aatype.shape, 4, 4, 3)
        chi_atom_mask = _gather_atoms(
            gt_exists.unsqueeze(-1), chi_atoms.reshape(*aatype.shape, 16)
        ).reshape(*aatype.shape, 4, 4)
        chi_mask = chi_mask & chi_atom_mask.all(-1)
        chi_angles = _dihedral(chi_pos)

        return {
            "aatype": aatype,
            "atom14_gt_positions": coordinates,
            "atom14_gt_exists": gt_exists,
            "atom14_alt_gt_positions": alt_positions,
            "atom14_alt_gt_exists": alt_exists,
            "atom14_atom_is_ambiguous": ambiguous,
            "atom14_atom_exists": atom_exists,
            "rigidgroups_gt_frames": frames,
            "rigidgroups_alt_gt_frames": alt_frames,
            "rigidgroups_gt_exists": frame_exists,
            "backbone_rigid_tensor": frames[..., 0, :, :],
            "backbone_rigid_mask": frame_exists[..., 0],
            "chi_angles_sin_cos": chi_angles,
            "chi_mask": chi_mask,
        }


@torch.no_grad()
def rename_ground_truth(
    labels: dict[str, torch.Tensor], pred_positions: torch.Tensor
) -> dict[str, torch.Tensor]:
    """AlphaFold Algorithm 26 symmetric atom renaming."""

    pred = pred_positions.float()
    gt = labels["atom14_gt_positions"]
    alt = labels["atom14_alt_gt_positions"]
    exists = labels["atom14_gt_exists"]
    alt_exists = labels["atom14_alt_gt_exists"]
    ambiguous = labels["atom14_atom_is_ambiguous"]
    B, L = pred.shape[:2]
    pred_flat = pred.reshape(B, L * 14, 3)
    gt_flat = gt.reshape(B, L * 14, 3)
    alt_flat = alt.reshape(B, L * 14, 3)
    ambiguous_flat = (ambiguous & exists).reshape(B, L * 14)
    non_ambiguous_flat = (~ambiguous & exists).reshape(B, L * 14)
    pair_mask = ambiguous_flat.unsqueeze(-1) & non_ambiguous_flat.unsqueeze(-2)

    pred_d = _cdist(pred_flat, pred_flat)
    gt_d = _cdist(gt_flat, gt_flat)
    original_error = (
        (torch.sqrt((pred_d - gt_d).square() + 1e-10) * pair_mask)
        .reshape(B, L, 14, L * 14)
        .sum((-1, -2))
    )
    del gt_d

    alt_d = _cdist(alt_flat, gt_flat)
    alternate_error = (
        (torch.sqrt((pred_d - alt_d).square() + 1e-10) * pair_mask)
        .reshape(B, L, 14, L * 14)
        .sum((-1, -2))
    )
    use_alt = alternate_error < original_error
    return {
        "alt_naming_is_better": use_alt,
        "renamed_atom14_gt_positions": torch.where(use_alt[..., None, None], alt, gt),
        "renamed_atom14_gt_exists": torch.where(use_alt[..., None], alt_exists, exists),
    }


def _rigid_parts(frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return frames[..., :3, :3].float(), frames[..., :3, 3].float()


def _quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q.float(), dim=-1, eps=1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(*q.shape[:-1], 3, 3)


def trajectory_to_frames(traj: torch.Tensor) -> torch.Tensor:
    if traj.shape[-1] == 7:
        rot = _quat_to_rot(traj[..., :4])
        trans = traj[..., 4:].float()
        out = torch.zeros(*traj.shape[:-1], 4, 4, device=traj.device)
        out[..., :3, :3], out[..., :3, 3], out[..., 3, 3] = rot, trans, 1.0
        return out
    return traj.float()


def compute_fape(
    pred_frames: torch.Tensor,
    target_frames: torch.Tensor,
    frame_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    position_mask: torch.Tensor,
    length_scale: float,
    clamp_distance: float | None,
    pair_mask: torch.Tensor | None = None,
    chunk_size: int | None = 64,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Frame-aligned point error, optionally chunked along the frame axis."""

    pr, pt = _rigid_parts(pred_frames)
    tr, tt = _rigid_parts(target_frames)
    # Broadcast a missing trajectory axis on target tensors.
    while tr.ndim < pr.ndim:
        tr, tt = tr.unsqueeze(0), tt.unsqueeze(0)
        frame_mask = frame_mask.unsqueeze(0)
        target_positions = target_positions.unsqueeze(0)
        position_mask = position_mask.unsqueeze(0)
        if pair_mask is not None:
            pair_mask = pair_mask.unsqueeze(0)
    losses = []
    weights = []
    n_frames = pr.shape[-3]
    if chunk_size is None:
        chunk_size = n_frames
    elif chunk_size <= 0:
        raise ValueError("FAPE chunk_size must be positive or None.")
    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        pr_c, pt_c = pr[..., start:end, :, :], pt[..., start:end, :]
        tr_c, tt_c = tr[..., start:end, :, :], tt[..., start:end, :]
        pred_local = torch.matmul(
            pr_c.transpose(-1, -2).unsqueeze(-3),
            (pred_positions.unsqueeze(-3) - pt_c.unsqueeze(-2)).unsqueeze(-1),
        ).squeeze(-1)
        target_local = torch.matmul(
            tr_c.transpose(-1, -2).unsqueeze(-3),
            (target_positions.unsqueeze(-3) - tt_c.unsqueeze(-2)).unsqueeze(-1),
        ).squeeze(-1)
        error = torch.sqrt((pred_local - target_local).square().sum(-1) + eps)
        if clamp_distance is not None:
            error = error.clamp(max=clamp_distance)
        weight = (
            frame_mask[..., start:end, None].float() * position_mask.unsqueeze(-2).float()
        )
        if pair_mask is not None:
            weight = weight * pair_mask[..., start:end, :].float()
        losses.append((error / length_scale * weight).sum((-1, -2)))
        weights.append(weight.sum((-1, -2)))
    return sum(losses) / (sum(weights) + eps)


def monomer_backbone_fape(
    structure: dict[str, Any],
    labels: dict[str, torch.Tensor],
    use_clamped_fape: torch.Tensor | None = None,
    chunk_size: int | None = 64,
    reduction: Reduction = "mean",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred = trajectory_to_frames(structure["traj"])
    target = labels["backbone_rigid_tensor"]
    mask = labels["backbone_rigid_mask"]
    x_pred = pred[..., :3, 3]
    x_target = target[..., :3, 3]
    value = compute_fape(
        pred, target, mask, x_pred, x_target, mask, 10.0, 10.0, chunk_size=chunk_size
    )
    if use_clamped_fape is not None:
        unclamped = compute_fape(
            pred, target, mask, x_pred, x_target, mask, 10.0, None, chunk_size=chunk_size
        )
        use_clamped_fape = use_clamped_fape.to(value).squeeze()
        while use_clamped_fape.ndim < value.ndim:
            use_clamped_fape = use_clamped_fape.unsqueeze(0)
        value = value * use_clamped_fape + unclamped * (1.0 - use_clamped_fape)
    per_example = value.mean(dim=0)
    return _reduce_batch(per_example, reduction), {"backbone_fape": value[-1].mean()}


def multimer_backbone_fape(
    structure: dict[str, Any],
    labels: dict[str, torch.Tensor],
    asym_id: torch.Tensor,
    chunk_size: int | None = 64,
    reduction: Reduction = "mean",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred = trajectory_to_frames(structure["traj"])
    target = labels["backbone_rigid_tensor"]
    mask = labels["backbone_rigid_mask"]
    x_pred = pred[..., :3, 3]
    x_target = target[..., :3, 3]
    valid_chain = asym_id > 0
    intra_mask = (
        (asym_id[..., :, None] == asym_id[..., None, :])
        & valid_chain[..., :, None]
        & valid_chain[..., None, :]
    )
    inter_mask = (
        (asym_id[..., :, None] != asym_id[..., None, :])
        & valid_chain[..., :, None]
        & valid_chain[..., None, :]
    )
    intra_fape = compute_fape(
        pred, target, mask, x_pred, x_target, mask, 10.0, 10.0, intra_mask, chunk_size
    )
    interface_fape = compute_fape(
        pred, target, mask, x_pred, x_target, mask, 20.0, 30.0, inter_mask, chunk_size
    )
    value = intra_fape + interface_fape
    per_example = value.mean(dim=0)
    return _reduce_batch(per_example, reduction), {
        "intra_chain_fape": intra_fape[-1].mean(),
        "interface_fape": interface_fape[-1].mean(),
    }


def backbone_fape(
    structure: dict[str, Any],
    labels: dict[str, torch.Tensor],
    multimer: bool,
    asym_id: torch.Tensor | None = None,
    use_clamped_fape: torch.Tensor | None = None,
    chunk_size: int | None = 64,
    reduction: Reduction = "mean",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if multimer:
        if asym_id is None:
            raise ValueError("asym_id is required for multimer backbone FAPE.")
        return multimer_backbone_fape(structure, labels, asym_id, chunk_size, reduction)
    else:
        return monomer_backbone_fape(
            structure, labels, use_clamped_fape, chunk_size, reduction
        )


def sidechain_fape(
    structure: dict[str, Any],
    labels: dict[str, torch.Tensor],
    renamed: dict[str, torch.Tensor],
    chunk_size: int | None = 64,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    pred_frames = structure["sidechains"]["frames"][-1]
    use_alt = renamed["alt_naming_is_better"]
    target_frames = torch.where(
        use_alt[..., None, None, None],
        labels["rigidgroups_alt_gt_frames"],
        labels["rigidgroups_gt_frames"],
    )
    per_example = compute_fape(
        pred_frames.flatten(-4, -3),
        target_frames.flatten(-4, -3),
        labels["rigidgroups_gt_exists"].flatten(-2, -1),
        structure["sidechains"]["atom_pos"][-1].flatten(-3, -2),
        renamed["renamed_atom14_gt_positions"].flatten(-3, -2),
        renamed["renamed_atom14_gt_exists"].flatten(-2, -1),
        10.0,
        10.0,
        chunk_size=chunk_size,
    )
    return _reduce_batch(per_example, reduction)


def chain_center_of_mass_loss(
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    target_mask: torch.Tensor,
    asym_id: torch.Tensor,
    distance_scale: float = 20.0,
    tolerance: float = 4.0,
    eps: float = 1e-10,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """AF-Multimer chain center-of-mass loss from paper section 2.5."""

    pred_ca = pred_positions[..., 1, :].float()
    target_ca = target_positions[..., 1, :].float()
    ca_mask = target_mask[..., 1].float() * (asym_id > 0).float()
    chain_membership = F.one_hot(asym_id.long().clamp(min=0)).to(ca_mask.dtype)
    chain_membership = chain_membership * ca_mask.unsqueeze(-1)
    chain_membership = chain_membership.transpose(-1, -2)  # [B, C, L]
    chain_count = chain_membership.sum(-1)
    chain_exists = chain_count > 0

    def centers(positions: torch.Tensor) -> torch.Tensor:
        weighted_sum = torch.einsum("bcl,bld->bcd", chain_membership, positions)
        return weighted_sum / (chain_count.unsqueeze(-1) + eps)

    pred_centers = centers(pred_ca)
    target_centers = centers(target_ca)
    pred_distances = torch.sqrt(
        (pred_centers[..., :, None, :] - pred_centers[..., None, :, :]).square().sum(-1)
        + eps
    )
    target_distances = torch.sqrt(
        (target_centers[..., :, None, :] - target_centers[..., None, :, :])
        .square()
        .sum(-1)
        + eps
    )
    error = torch.clamp(
        (pred_distances - target_distances + tolerance) / distance_scale,
        max=0.0,
    ).square()

    num_chains = chain_exists.shape[-1]
    pair_mask = chain_exists[..., :, None] & chain_exists[..., None, :]
    pair_mask &= ~torch.eye(num_chains, dtype=torch.bool, device=asym_id.device)
    pair_mask = pair_mask.to(error.dtype)
    per_example = (error * pair_mask).sum((-1, -2)) / pair_mask.sum((-1, -2)).clamp(
        min=1.0
    )
    return _reduce_batch(per_example, reduction)


def supervised_chi_loss(
    structure: dict[str, Any],
    labels: dict[str, torch.Tensor],
    seq_mask: torch.Tensor,
    reduction: Reduction = "mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = structure["sidechains"]["angles"][..., 3:, :].float()
    unnorm = structure["sidechains"]["unnormalized_angles"].float()
    true = labels["chi_angles_sin_cos"].unsqueeze(0)
    periodic = (
        _constant("chi_periodic", pred)[labels["aatype"]].unsqueeze(0).unsqueeze(-1)
    )
    shifted = true * (1.0 - 2.0 * periodic)
    sq = torch.minimum((pred - true).square().sum(-1), (pred - shifted).square().sum(-1))
    mask = labels["chi_mask"].float().unsqueeze(0)
    chi = (sq * mask).sum(dim=(0, 2, 3)) / (
        mask[0].sum(dim=(-1, -2)) * pred.shape[0] + 1e-8
    )
    norm_error = torch.abs(torch.sqrt(unnorm.square().sum(-1) + 1e-6) - 1.0)
    norm_mask = seq_mask.float().unsqueeze(0).unsqueeze(-1)
    angle_norm = (norm_error * norm_mask).sum(dim=(0, 2, 3)) / (
        norm_mask[0].sum(dim=(-1, -2)) * unnorm.shape[0] * unnorm.shape[-2] + 1e-8
    )
    total = 0.5 * chi + 0.01 * angle_norm
    return (
        _reduce_batch(total, reduction),
        _reduce_batch(chi, reduction),
        _reduce_batch(angle_norm, reduction),
    )


def _between_bond_loss(
    pos: torch.Tensor,
    mask: torch.Tensor,
    residue_index: torch.Tensor,
    aatype: torch.Tensor,
    asym_id: torch.Tensor | None,
    tolerance_factor: float = 12.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c, ca, next_n, next_ca = pos[:, :-1, 2], pos[:, :-1, 1], pos[:, 1:, 0], pos[:, 1:, 1]
    valid = (
        mask[:, :-1, 2]
        & mask[:, 1:, 0]
        & ((residue_index[:, 1:] - residue_index[:, :-1]) == 1)
    )
    if asym_id is not None:
        valid &= asym_id[:, 1:] == asym_id[:, :-1]
    d = torch.sqrt((c - next_n).square().sum(-1) + 1e-6)
    pro = aatype[:, 1:] == PRO_INDEX
    ideal = torch.where(pro, d.new_tensor(1.341), d.new_tensor(1.329))
    std = torch.where(pro, d.new_tensor(0.016), d.new_tensor(0.014))
    bond = F.relu(torch.abs(d - ideal) - tolerance_factor * std)
    c_ca = F.normalize(ca - c, dim=-1, eps=1e-8)
    c_n = F.normalize(next_n - c, dim=-1, eps=1e-8)
    n_ca = F.normalize(next_ca - next_n, dim=-1, eps=1e-8)
    a1 = F.relu(torch.abs((c_ca * c_n).sum(-1) + 0.4473) - tolerance_factor * 0.0311)
    a2 = F.relu(torch.abs(((-c_n) * n_ca).sum(-1) + 0.5203) - tolerance_factor * 0.0353)

    def mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return (x * m).sum(-1) / (m.sum(-1) + 1e-6)

    return (
        mean(bond, valid),
        mean(a1, valid & mask[:, :-1, 1]),
        mean(a2, valid & mask[:, 1:, 1]),
    )


def violation_loss(
    structure: dict[str, Any],
    labels: dict[str, torch.Tensor],
    residue_index: torch.Tensor,
    seq_mask: torch.Tensor,
    asym_id: torch.Tensor | None,
    bond_angle_weight: float = 0.3,
    clash_overlap_tolerance: float = 1.5,
    violation_tolerance_factor: float = 12.0,
    style: ViolationLossStyle = "af_multimer",
    eps: float = 1e-6,
    reduction: Reduction = "mean",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the AF2 or AF-Multimer structural-violation objective.

    AF2 assigns each between-residue clash penalty to both participating atoms
    and normalizes their sum by the number of atoms. AF-Multimer instead
    averages over atom pairs that actually clash, preventing an initially
    collapsed multimer from producing gradients that scale quadratically with
    its number of atoms.
    """

    if style not in ("af2", "af_multimer"):
        raise ValueError(f"Unknown violation loss style: {style}")

    pos = structure["coords"].float()
    mask = labels["atom14_atom_exists"] & seq_mask.bool().unsqueeze(-1)
    bond, angle1, angle2 = _between_bond_loss(
        pos,
        mask,
        residue_index,
        labels["aatype"],
        asym_id,
        tolerance_factor=violation_tolerance_factor,
    )
    radii = _constant("radii", pos)[labels["aatype"]]
    between_loss_sum = pos.new_zeros(pos.shape[0])
    between_clash_count = pos.new_zeros(pos.shape[0])
    L = pos.shape[1]
    for start in range(0, L, 32):
        end = min(start + 32, L)
        d = torch.sqrt(
            (pos[:, start:end, None, :, None] - pos[:, None, :, None, :]).square().sum(-1)
            + 1e-10
        )
        valid = mask[:, start:end, None, :, None] & mask[:, None, :, None, :]
        ri = residue_index[:, start:end, None, None, None]
        rj = residue_index[:, None, :, None, None]
        global_i = torch.arange(start, end, device=pos.device).view(1, -1, 1, 1, 1)
        global_j = torch.arange(L, device=pos.device).view(1, 1, -1, 1, 1)
        valid &= global_i < global_j
        neighbor = rj == ri + 1
        if asym_id is not None:
            neighbor &= (
                asym_id[:, start:end, None, None, None] == asym_id[:, None, :, None, None]
            )
        atom_i = torch.arange(14, device=pos.device).view(1, 1, 1, 14, 1)
        atom_j = torch.arange(14, device=pos.device).view(1, 1, 1, 1, 14)
        valid &= ~(neighbor & (atom_i == 2) & (atom_j == 0))
        disulfide = (
            (labels["aatype"][:, start:end, None, None, None] == CYS_INDEX)
            & (labels["aatype"][:, None, :, None, None] == CYS_INDEX)
            & (atom_i == CYS_SG_INDEX)
            & (atom_j == CYS_SG_INDEX)
        )
        valid &= ~disulfide
        lower = (
            radii[:, start:end, None, :, None]
            + radii[:, None, :, None, :]
            - clash_overlap_tolerance
        )
        error = F.relu(lower - d) * valid
        is_clash = (error > 0).to(pos.dtype)

        between_loss_sum += error.sum((1, 2, 3, 4))
        between_clash_count += is_clash.sum((1, 2, 3, 4))

    atom14_bounds = rc.make_atom14_dists_bounds(
        overlap_tolerance=clash_overlap_tolerance,
        bond_length_tolerance_factor=violation_tolerance_factor,
    )
    lower = torch.from_numpy(atom14_bounds["lower_bound"]).to(pos.device)[
        labels["aatype"]
    ]
    upper = torch.from_numpy(atom14_bounds["upper_bound"]).to(pos.device)[
        labels["aatype"]
    ]
    d = torch.sqrt((pos[..., :, None, :] - pos[..., None, :, :]).square().sum(-1) + 1e-10)
    pair = (
        mask[..., :, None]
        & mask[..., None, :]
        & ~torch.eye(14, dtype=torch.bool, device=pos.device)
    )
    within_error = (F.relu(lower - d) + F.relu(d - upper)) * pair
    per_atom_within_loss = within_error.sum(-1) + within_error.sum(-2)

    num_atoms = mask.sum((-1, -2)).clamp(min=1).to(pos.dtype)
    if style == "af2":
        # AF2 accumulates every unique pair's error once on each participating
        # atom, then divides the resulting per-atom sum by the atom count.
        clash = 2.0 * between_loss_sum / (num_atoms + eps)
    else:
        clash = between_loss_sum / (between_clash_count + eps)

    # Within-residue distance-bound violations use AF2's atom normalization in
    # both styles.
    within = (per_atom_within_loss * mask).sum((-1, -2)) / num_atoms

    total = bond + bond_angle_weight * (angle1 + angle2) + clash + within
    return _reduce_batch(total, reduction), {
        "bond": bond.mean(),
        "angle_ca_c_n": angle1.mean(),
        "angle_c_n_ca": angle2.mean(),
        "clash": clash.mean(),
        "between_residue_clash": clash.mean(),
        "within": within.mean(),
    }


def structure_loss(
    structure: dict[str, Any],
    aatype: torch.Tensor,
    coordinates: torch.Tensor,
    resolved_mask: torch.Tensor,
    seq_mask: torch.Tensor,
    residue_index: torch.Tensor,
    asym_id: torch.Tensor | None = None,
    multimer: bool = False,
    use_clamped_fape: torch.Tensor | None = None,
    backbone_fape_chunk_size: int | None = 64,
    sidechain_fape_chunk_size: int | None = 64,
    reduction: Reduction = "mean",
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Compute FAPE and torsion losses, excluding structural violations."""

    with torch.autocast(coordinates.device.type, enabled=False):
        labels = make_structure_labels(aatype, coordinates, resolved_mask)
        renamed = rename_ground_truth(labels, structure["coords"])
        bb, metrics = backbone_fape(
            structure,
            labels,
            multimer,
            asym_id,
            use_clamped_fape,
            backbone_fape_chunk_size,
            reduction=reduction,
        )
        side = sidechain_fape(
            structure,
            labels,
            renamed,
            chunk_size=sidechain_fape_chunk_size,
            reduction=reduction,
        )
        fape = 0.5 * bb + 0.5 * side
        chi_total, chi, angle_norm = supervised_chi_loss(
            structure, labels, seq_mask, reduction=reduction
        )
        total = fape + chi_total
        metrics = {key: value.detach() for key, value in metrics.items()}
        metrics.update(
            {
                "fape": fape.mean().detach(),
                "sidechain_fape": side.mean().detach(),
                "chi": chi.mean().detach(),
                "angle_norm": angle_norm.mean().detach(),
            }
        )
        return total, metrics, {**labels, **renamed}
