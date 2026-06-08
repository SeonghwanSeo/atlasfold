import einops
import torch
import torch.nn as nn

from atlasfold.model.network.diffusion_transformer import AtomTransformerStack
from atlasfold.model.network.misc import LocalAttentionIndex
from atlasfold.model.network.primitives import LinearNoBias


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
        x_flat = einops.rearrange(x, "... a c -> ... (a c)", a=14)  # (*, 14*C_h)
        # MLP core applied to the flattened intra-residue geometry
        x_flat = self.mlp(x_flat)
        # Unflatten back to distinct atoms and hidden dimension
        x = einops.rearrange(x_flat, "... (a c) -> ... a c", a=14)  # (*, 14, C_h)
        # Final unactivated projection back to atom channels
        x = self.linear_out(x)  # (*, 14, C_a)
        x = x * mask[..., None]  # Mask out dummy atoms
        return x


class AtomAttentionStack(nn.Module):
    def __init__(
        self,
        channel_atom: int = 96,
        num_heads: int = 2,
        num_blocks: int = 2,
        max_r: int = 4,
    ) -> None:
        """Initialize the Atom Attention Encoder layer.

        Parameters
        ----------
        channel_atom : int
            The atom/token dimension.
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
        self.local_stack = AtomTransformerStack(
            channel_atom, channel_atom, num_heads, num_blocks
        )
        self.max_r: int = max_r
        self.rel_pos_dim = 2 * max_r + 2
        self.linear_rel_pos = LinearNoBias(self.rel_pos_dim, (num_blocks * num_heads))

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        q: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        batch: dict[str, torch.Tensor]
            The input batch containing 'aatype', 'atom_mask', and 'res_idx'.
        q: torch.Tensor
            The atom representations of shape (B, N, L, 14, C_a)
        c: torch.Tensor
            The atom conditioning of shape (B, N, L, 14, C_a)

        Returns
        -------
        q : torch.Tensor
            The updated atom representations of shape (B, N, L, 14, C_a)
        """
        res_idx = batch["res_idx"]  # (B, L)
        asym_id = batch["asym_id"]  # (B, L)
        seq_mask = batch["seq_mask"]  # (B, L)
        local_attn_index = LocalAttentionIndex(
            res_idx, asym_id, seq_mask, window_size=4, max_r=self.max_r
        )

        # Compute the relative positional encodings for the local attention
        # NOTE: >max_r, different chain pairs would be masked out in local-attention.
        res_idx_q, res_idx_k = local_attn_index(res_idx, -1, v_pad=int(1e6))
        asym_id_q, asym_id_k = local_attn_index(asym_id, -1, v_pad=-1)

        pad_r = 2 * self.max_r + 1  # NOTE: pad_r position would be masked out
        rel_pos = res_idx_q[..., :, None] - res_idx_k[..., None, :]  # (B, W, Lq, Lk)
        is_same_chain = asym_id_q[..., :, None] == asym_id_k[..., None, :]
        a_rel_pos = torch.clamp(rel_pos + self.max_r, 0, pad_r)
        a_rel_pos = torch.where(is_same_chain, a_rel_pos, pad_r)
        a_rel_pos = torch.nn.functional.one_hot(a_rel_pos, pad_r)

        p = self.linear_rel_pos(a_rel_pos.float())  # (B, W, Lq, Lk, n_blocks, n_heads)
        # Rearrange to (n_blocks, B, W, n_heads,Lq, Lk)
        p = einops.rearrange(p, "b w q k (n h) -> n b w h q k", n=self.num_blocks)
        # Expand to atom level
        p = einops.repeat(p, "... q k -> ... (q a) (k a)", a=14)

        atom_mask = batch["atom14_mask"]  # (B, L, 14)
        q = self.local_stack(
            q,
            mask=atom_mask,
            local_attn_index=local_attn_index,
            single_cond=c,
            pair_bias=p,
        )
        return q


class AtomEncoder(nn.Module):
    def __init__(
        self,
        channel_atom: int = 96,
        channel_cond: int = 768,
        num_heads: int = 2,
        num_blocks: int = 2,
    ) -> None:
        """Initialize the Atom embedding layer.

        Parameters
        ----------
        channel_atom : int
            The atom/token dimension.
        channel_cond : int
            The conditioning dimension.
        """
        super().__init__()
        self.embedding_aa_atoms = LinearNoBias(21, 14 * channel_atom)
        self.linear_in = LinearNoBias(3, channel_atom, init="default", precision=32)
        self.linear_cond = LinearNoBias(channel_cond, channel_atom, init="final")
        self.mlp = AtomMLP(channel_atom, hidden_dim=channel_atom // 2)
        self.stack = AtomAttentionStack(channel_atom, num_heads, num_blocks, max_r=4)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        r_noisy: torch.Tensor,
        cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass.

        Parameters
        ----------
        batch: dict[str, torch.Tensor]
            The input batch containing 'aatype', 'atom_mask', and 'res_idx'.
        r_noisy: torch.Tensor
            The noisy atom coordinates of shape (B, N, L, 14, 3)
        cond : torch.Tensor
            The single conditioning of shape (B, N, L, C_s)

        Returns
        -------
        q : torch.Tensor
            The atom representations of shape (B, N, L, 14, C_a)
        c : torch.Tensor
            The atom conditioning of shape (B, N, L, 14, C_a)
        """
        mask = batch["atom14_mask"].unsqueeze(-3)  # (B, 1, L, 14)
        mask = mask.float()

        # Get the atom representations
        aatype = batch["aatype"]  # (B, L, 21)
        a = self.embedding_aa_atoms(aatype).unflatten(-1, (14, -1))  # (B, L, 14, C_a)
        a = a.unsqueeze(-4)  # (B, 1, L, 14, C_a)

        # Initialize the atom representations and conditioning
        q, c = a, a

        # Prepare the input representations
        # Embed the current state: noisy atom coordinates
        q = q + self.linear_in(r_noisy)  # (B, N, L, 14, C_a)
        # Residue-wise MLP on the intra-residue geometry
        q = q + self.mlp(q, mask)  # (B, N, L, 14, C_a)

        # Prepare the conditioning
        c = c + self.linear_cond(cond).unsqueeze(-2)  # (B, N, L, 14, C_a)

        # Stack of local attention blocks
        q = self.stack(batch, q, c)  # (B, N, L, 14, C_a)

        # Mask out dummy atoms
        q = q * mask[..., None]

        return q, c


class AtomDecoder(nn.Module):
    def __init__(
        self,
        channel_atom: int = 96,
        num_heads: int = 2,
        num_blocks: int = 2,
    ) -> None:
        """Initialize the Atom decoding layer."""
        super().__init__()
        self.stack = AtomAttentionStack(channel_atom, num_heads, num_blocks, max_r=4)
        self.mlp = AtomMLP(channel_atom, hidden_dim=channel_atom // 2)
        self.linear_q_to_r = LinearNoBias(channel_atom, 3, init="final", precision=32)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        q: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """Maps updated representations back to raw 3D coordinate updates."""
        mask = batch["atom14_mask"].unsqueeze(-3).float()  # (B, 1, L, 14)

        # Another stack of local attention blocks
        q = self.stack(batch, q, c)  # (B, N, L, 14, C_a)

        # Final post-processing block of intra-residue updates
        q = q + self.mlp(q, mask)
        q = q * mask[..., None]

        # Project representations directly to raw 3D coordinate trajectories
        r_update = self.linear_q_to_r(q)  # (B, N, L, 14, 3)
        r_update = r_update * mask[..., None]

        return r_update
