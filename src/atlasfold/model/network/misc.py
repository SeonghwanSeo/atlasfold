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


# === Local Attention Indexing === #
class LocalAttentionIndex:
    def __init__(
        self,
        res_idx: torch.Tensor,
        pad_mask: torch.Tensor,
        window_size: int = 4,
        max_r: int = 6,
    ) -> None:
        """Build indices for window-based local attention from atoms to query windows.

        Parameters
        ----------
        res_idx: int
            The residue indices of shape (*, L).
        pad_mask: torch.Tensor
            The mask tensor of shape (*, L) indicating valid positions
        window_size: int
            The number of query windows (W) per block.
        max_r: int
            The maximum residue distance for local attention
        """
        self.L = res_idx.shape[-1]
        self.device = res_idx.device
        self.Lq = window_size  # = 4
        self.Lk = window_size + 2 * max_r  # = 16

        if self.L % self.Lq != 0:
            raise ValueError(
                f"Length {self.L} must be divisible by window_size {self.Lq}."
            )

        self.W: int = self.L // self.Lq
        num_half_blocks = 2 * self.W
        half_block_size = self.Lq // 2  # = 2
        h = self.Lk // half_block_size

        # Block Logic
        start_offset = -(h // 2) + 1
        block_offsets = torch.arange(h, device=self.device) + start_offset
        window_starts = torch.arange(self.W, device=self.device).unsqueeze(-1) * 2
        block_indices = window_starts + block_offsets  # [W, h]

        # Pad mask for out-of-bounds
        gather_mask = (block_indices < 0) | (block_indices >= num_half_blocks)
        # [W, h] -> [W, Lk]
        gather_mask = gather_mask.repeat_interleave(half_block_size, dim=-1)
        self.gather_mask: torch.Tensor = gather_mask  # [W, Lk]

        # Clamp block indices to valid range
        block_indices = block_indices.clamp(min=0, max=num_half_blocks - 1)

        # Expand block indices to atom indices
        atom_offsets = torch.arange(half_block_size, device=self.device)

        # Broadcasting to construct full [W, Lk] index matrix
        gather_indices = (
            block_indices[..., None] * half_block_size + atom_offsets[None, None, ...]
        )
        gather_indices = gather_indices.view(self.W, self.Lk)
        self.gather_indices: torch.Tensor = gather_indices  # [W, Lk]

        # Compute the attention mask for the local attention
        res_idx_q = res_idx.unflatten(-1, sizes=(self.W, self.Lq))  # [*, W, Lq]
        res_idx_k = res_idx.index_select(-1, gather_indices.view(-1))
        res_idx_k = res_idx_k.unflatten(-1, sizes=(self.W, self.Lk))  # [*, W, Lk]
        attn_mask = torch.abs(res_idx_q[..., :, None] - res_idx_k[..., None, :]) <= max_r

        # Mask out-of-bounds positions in the attention mask
        attn_mask &= ~gather_mask.unsqueeze(-2)

        # Mask for valid key positions
        mask_k = pad_mask.index_select(-1, gather_indices.view(-1))
        attn_mask &= mask_k.unsqueeze(-2)
        self.attn_mask: torch.Tensor = attn_mask  # [*, W, Lq, Lk]

    def __call__(
        self,
        x: torch.Tensor,
        dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Window-based indexing for local attention.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape [*, L, *]
        dim: int, optional
            Dimension along which to unflatten into query/key windows.

        Returns
        -------
        q: torch.Tensor
            Query tensor of shape [*, W, Lq, *]
        k: torch.Tensor
            Key tensor of shape [*, W, Lk, *]
        """
        return self.to_qk(x, dim)

    def to_qk(
        self,
        x: torch.Tensor,
        dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Window-based indexing for local attention.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape [*, L, *]
        dim: int, optional
            Dimension along which to unflatten into query/key windows.

        Returns
        -------
        q: torch.Tensor
            Query tensor of shape [*, W, Lq, *]
        k: torch.Tensor
            Key tensor of shape [*, W, Lk, *]
        """
        dim = dim % x.ndim

        # Get query
        x_q = x.unflatten(dim, sizes=(self.W, self.Lq))

        # Get keys using gather indices
        x_k = x.index_select(dim, self.gather_indices.view(-1))
        x_k = x_k.unflatten(dim, sizes=(self.W, self.Lk))

        # Apply Padding Mask
        mask_shape = [1] * x_k.ndim
        mask_shape[dim : dim + 2] = self.gather_indices.shape  # W, Lk
        mask = self.gather_mask.view(*mask_shape)
        x_k = x_k.masked_fill(mask, 0)
        return x_q, x_k
