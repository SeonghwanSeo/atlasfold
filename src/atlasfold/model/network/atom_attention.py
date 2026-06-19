import einops
import torch
import torch.nn as nn

from atlasfold.model.network.diffusion_transformer import AtomTransformerStack
from atlasfold.model.network.misc import LocalAttentionIndex
from atlasfold.model.network.primitives import LayerNorm, LinearNoBias


class AtomMLP(nn.Module):
    def __init__(self, channel_atom: int = 96, hidden_dim: int = 48) -> None:
        """Initialize the Atom MLP layer.

        Parameters
        ----------
        channel_atom : int
            The atom dimension.
        hidden_dim : int
            The hidden dimension.
        """
        super().__init__()
        self.channel_atom: int = channel_atom
        self.hidden_dim: int = hidden_dim
        self.linear_in = LinearNoBias(channel_atom, hidden_dim, init="default")
        self.mlp = nn.Sequential(
            LinearNoBias(hidden_dim * 14, hidden_dim * 14, init="relu"),
            nn.ReLU(),
            LinearNoBias(hidden_dim * 14, hidden_dim * 14, init="relu"),
        )
        self.linear_out = LinearNoBias(hidden_dim, channel_atom, init="final")

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Input projection
        x = x * mask[..., None]  # Mask out dummy atoms
        x = self.linear_in(x)  # (*, 14, C_h)
        # Flatten the 14 atoms and the hidden dimension together
        x_flat = x.flatten(start_dim=-2)  # (*, 14*C_h)
        # MLP core applied to the flattened intra-residue geometry
        x_flat = self.mlp(x_flat)
        # Unflatten back to distinct atoms and hidden dimension
        x = x_flat.unflatten(-1, (14, self.hidden_dim))  # (*, 14, C_h)
        # Final unactivated projection back to atom channels
        x = self.linear_out(x)  # (*, 14, C_a)
        x = x * mask[..., None]  # Mask out dummy atoms
        return x


class AtomAttentionStack(nn.Module):
    def __init__(
        self,
        channel_atom: int = 96,
        channel_atompair: int = 14,
        num_heads: int = 2,
        num_blocks: int = 2,
    ) -> None:
        """Initialize the Atom Attention Encoder layer.

        Parameters
        ----------
        channel_atom : int
            The atom/token dimension.
        channel_atompair : int
            The atom pair dimension (relative positional encoding dimension).
        num_heads : int
            The number of attention heads.
        num_blocks : int
            The number of local attention blocks.
        max_r : int
            The maximum residue distance for local attention.
        """
        super().__init__()
        self.channel_atom: int = channel_atom
        self.num_heads: int = num_heads
        self.num_blocks: int = num_blocks
        self.linear_pair_bais = LinearNoBias(channel_atompair, (num_blocks * num_heads))
        self.stack = AtomTransformerStack(channel_atom, None, num_heads, num_blocks)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        q: torch.Tensor,
    ) -> torch.Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        batch: dict[str, torch.Tensor]
            The input batch containing 'aatype', 'atom_mask', and 'res_idx'.
        q: torch.Tensor
            The atom representations of shape (B, N, L, 14, C_a)

        Returns
        -------
        q : torch.Tensor
            The updated atom representations of shape (B, N, L, 14, C_a)
        """
        res_idx = batch["res_idx"].unsqueeze(1)  # (B, 1, L)
        asym_id = batch["asym_id"].unsqueeze(1)  # (B, 1, L)
        seq_mask = batch["seq_mask"].unsqueeze(1)  # (B, 1, L)
        rel_pos = batch["atom_rel_pos"].unsqueeze(1)  # (B, 1, W, Lq, 14, Lk, 14, bins)
        atom_mask = batch["atom14_mask"].unsqueeze(1)  # (B, 1, L, 14)

        # HACK: Our code is hard-coded to window_size=4 and max_r=4.
        local_attn_idx = LocalAttentionIndex(
            res_idx, asym_id, seq_mask, window_size=4, max_r=4
        )
        W, Lq, Lk = local_attn_idx.W, local_attn_idx.Lq, local_attn_idx.Lk
        assert rel_pos.shape[2:7] == (W, Lq, 14, Lk, 14), (
            f"Expected rel_pos shape (B, 1, {W}, {Lq}, 14, {Lk}, 14), "
            f"but got {rel_pos.shape}."
        )

        # Convert the atom positional encodings to attention pair bias.
        p = self.linear_pair_bais(rel_pos)
        p = einops.rearrange(
            p,
            "... q a1 k a2 (n h) -> n ... h q a1 k a2",
            a1=14,
            a2=14,
            n=self.num_blocks,
            h=self.num_heads,
        ).contiguous()  # (Nblocks, B, 1, W, Nheads, Lq, 14, Lk, 14)

        q = self.stack(
            q,  # [B, N, L, 14, C_a]
            mask=atom_mask,  # [B, 1, L, 14]
            local_attn_index=local_attn_idx,
            single_cond=None,
            pair_bias=p,  # [Nblocks, B, 1, W, Nheads, Lq, 14, Lk, 14]
        )
        return q


class AtomEncoder(nn.Module):
    def __init__(
        self,
        channel_atom: int = 96,
        channel_atompair: int = 14,
        num_heads: int = 2,
        num_blocks: int = 2,
    ) -> None:
        """Initialize the Atom embedding layer.

        Parameters
        ----------
        channel_atom : int
            The atom/token dimension.
        channel_atompair : int
            The atom pair dimension (relative positional encoding dimension).
        channel_cond : int
            The conditioning dimension.
        """
        super().__init__()
        self.num_heads: int = num_heads
        self.num_blocks: int = num_blocks

        self.embedding_atoms = LinearNoBias(21, 14 * channel_atom)
        self.linear_in = LinearNoBias(3, channel_atom, init="default", precision=32)
        self.mlp = AtomMLP(channel_atom, hidden_dim=channel_atom // 2)
        self.stack = AtomAttentionStack(
            channel_atom, channel_atompair, num_heads, num_blocks
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        r_noisy: torch.Tensor,
    ) -> torch.Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        batch: dict[str, torch.Tensor]
            The input batch containing 'aatype', 'atom_mask', and 'res_idx'.
        r_noisy: torch.Tensor
            The noisy atom coordinates of shape (B, N, L, 14, 3)

        Returns
        -------
        q : torch.Tensor
            The atom representations of shape (B, N, L, 14, C_a)
        """
        # Get the atom representations
        aatype = batch["aatype"]  # (B, L, 21)
        q = self.embedding_atoms(aatype).unflatten(-1, (14, -1))  # (B, L, 14, C_a)
        q = q.unsqueeze(-4)  # (B, 1, L, 14, C_a)

        mask = batch["atom14_mask"].unsqueeze(1).to(q.dtype)  # (B, 1, L, 14)

        # Prepare the input representations
        # Embed the current state: noisy atom coordinates
        q = q + self.linear_in(r_noisy)  # (B, N, L, 14, C_a)
        # Residue-wise MLP on the intra-residue geometry
        q = q + self.mlp(q, mask)  # (B, N, L, 14, C_a)

        # Stack of local attention blocks
        q = self.stack(batch, q)  # (B, N, L, 14, C_a)

        # Mask out dummy atoms
        q = q * mask[..., None]

        return q


class AtomDecoder(nn.Module):
    def __init__(
        self,
        channel_atom: int = 96,
        channel_atompair: int = 14,
        num_heads: int = 2,
        num_blocks: int = 2,
    ) -> None:
        """Initialize the Atom decoding layer."""
        super().__init__()
        self.stack = AtomAttentionStack(
            channel_atom, channel_atompair, num_heads, num_blocks
        )
        self.mlp = AtomMLP(channel_atom, hidden_dim=channel_atom // 2)
        self.linear_out = nn.Sequential(
            LayerNorm(channel_atom),
            LinearNoBias(channel_atom, 3, init="final", precision=32),
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        q: torch.Tensor,
    ) -> torch.Tensor:
        """Maps updated representations back to raw 3D coordinate updates."""
        # Another stack of local attention blocks
        q = self.stack(batch, q)  # (B, N, L, 14, C_a)

        # Final post-processing block of intra-residue updates
        mask = batch["atom14_mask"].unsqueeze(-3).to(q.dtype)  # (B, 1, L, 14)
        q = q + self.mlp(q, mask)
        q = q * mask[..., None]

        # Project representations directly to raw 3D coordinate trajectories
        r_update = self.linear_out(q)  # (B, N, L, 14, 3)
        r_update = r_update * mask[..., None]

        return r_update
