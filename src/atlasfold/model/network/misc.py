import torch

from atlasfold.common import residue_utils


def relative_position_encoding(
    res_idx: torch.Tensor,
    asym_id: torch.Tensor,
    entity_id: torch.Tensor,
    sym_id: torch.Tensor,
    r_max: int = 32,
    s_max: int = 2,
) -> torch.Tensor:
    """Compute the relative position encoding for the pair representation.
    See Algorithm 5 in the AlphaFold-multimer paper for details.

    Parameters
    ----------
    res_idx : torch.Tensor
        The residue index tensor of shape (*, L).
    asym_id : torch.Tensor
        The asymmetric unit ID tensor of shape (*, L).
    entity_id : torch.Tensor
        The entity ID tensor of shape (*, L).
    sym_id : torch.Tensor
        The symmetric unit ID tensor of shape (*, L).

    Returns
    -------
    torch.Tensor
        The relative position encoding tensor of shape (*, L, L, bins).
        bins = 66 + 6 + 1 = 73.
    """
    # NOTE: we share the same relative
    is_same_chain = torch.eq(asym_id[..., :, None], asym_id[..., None, :])
    rel_pos = res_idx[..., :, None] - res_idx[..., None, :]
    rel_pos = torch.clamp(rel_pos + r_max, min=0, max=2 * r_max)
    rel_pos = torch.where(is_same_chain, rel_pos, 2 * r_max + 1)
    a_rel_pos = torch.nn.functional.one_hot(rel_pos, 2 * r_max + 2)

    is_same_entity = torch.eq(entity_id[:, :, None], entity_id[:, None, :])
    rel_chain = sym_id[:, :, None] - sym_id[:, None, :]
    rel_chain = torch.clip(rel_chain + s_max, min=0, max=2 * s_max)
    rel_chain = torch.where(is_same_entity, rel_chain, 2 * s_max + 1)
    a_rel_chain = torch.nn.functional.one_hot(rel_chain, 2 * s_max + 2)

    rel_position_encoding = torch.cat(
        [a_rel_pos.float(), is_same_entity.float(), a_rel_chain.float()], dim=-1
    )
    return rel_position_encoding  # [B, L, L, 73]


def atom14_to_atom37(atom14: torch.Tensor, restype: torch.Tensor) -> torch.Tensor:
    """Convert atom14 representation to atom37 representation.

    Parameters
    ----------
    atom14 : torch.Tensor
        The atom14 representation of shape (*, 14, dim)
    restype : torch.Tensor
        The residue type indices of shape (*)

    Returns
    -------
    torch.Tensor
        The atom37 representation of shape (*, 37, dim)
    """
    gather_indices = torch.as_tensor(residue_utils._gather_indices, device=atom14.device)
    gather_mask = torch.as_tensor(residue_utils._gather_mask, device=atom14.device)
    atom_map = gather_indices[restype]  # shape: (*, 37)
    atom_mask = gather_mask[restype]  # shape: (*, 37)
    expand_shape = atom_map.shape + (atom14.shape[-1],)
    atom37 = atom14.gather(-2, index=atom_map[..., None].expand(*expand_shape))
    atom37 = atom37.masked_fill(~atom_mask[..., None], 0.0)
    return atom37
