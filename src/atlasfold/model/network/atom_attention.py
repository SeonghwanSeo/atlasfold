import einops
import torch
import torch.nn as nn

from atlasfold.model.network.diffusion_transformer import AtomTransformerStack
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection
        x = self.linear_in(x)  # (*, 14, C_h)
        # Flatten the 14 atoms and the hidden dimension together
        x_flat = einops.rearrange(x, "... a c -> ... (a c)", a=14)  # (*, 14*C_h)
        # MLP core applied to the flattened intra-residue geometry
        x_flat = self.mlp(x_flat)
        # Unflatten back to distinct atoms and hidden dimension
        x = einops.rearrange(x_flat, "... (a c) -> ... a c", a=14)  # (*, 14, C_h)
        # Final unactivated projection back to atom channels
        x = self.linear_out(x)  # (*, 14, C_a)
        return x


class AtomEmbedder(nn.Module):
    def __init__(
        self,
        channel_atom: int = 96,
        channel_cond: int = 768,
    ) -> None:
        """Initialize the Atom embedding layer.

        Parameters
        ----------
        channel_atom : int
            The atom/token dimension.
        channel_cond : int
            The conditioning dimension.
        num_heads : int
            The number of attention heads.
        num_blocks : int
            The number of local attention blocks.
        max_r : int
            The maximum residue distance for local attention.
        """
        super().__init__()
        self.embedding_aa_atoms = LinearNoBias(21, 14 * channel_atom)
        self.linear_cond = LinearNoBias(channel_cond, channel_atom, init="final")

        self.linear_r_to_q = LinearNoBias(3, channel_atom, init="default", precision=32)
        self.atom_mlp_q = AtomMLP(channel_atom, hidden_dim=channel_atom // 2)

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
        # Get the initial atom representations
        # NOTE (Seonghwan): Here, I skip the masking for dummy atoms (e.g., CB for Gly)
        # and treat them as learnable bias terms.
        aatype = batch["aatype"]  # (B, L, 21)
        q = self.embedding_aa_atoms(aatype).unflatten(-1, (14, -1))  # (B, L, 14, C_a)

        # Initialize the atom conditioning
        c = q

        # Embed the single conditioning from the trunk
        c = c + self.linear_cond(cond).unsqueeze(-2)  # (B, N, L, 14, C_a)

        # Embed the current state: noisy atom coordinates
        mask = batch["atom14_mask"].unsqueeze(-3)  # (B, 1, L, 14)
        r_noisy = r_noisy.masked_fill(~mask[..., None], 0.0)  # (B, N, L, 14, 3)
        q = q.unsqueeze(-3) + self.linear_r_to_q(r_noisy)  # (B, N, L, 14, C_a)

        # Residue-wise MLP on the intra-residue geometry
        q = q + self.atom_mlp_q(q)  # (B, N, L, 14, C_a)

        return q, c


class AtomAttentionStack(nn.Module):
    def __init__(
        self,
        channel_atom: int = 64,
        num_heads: int = 2,
        num_blocks: int = 2,
        max_r: int = 6,
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
        B, N, L, _, _ = q.shape

        # Compute the relative positional encodings for the local attention
        residx = batch["res_idx"]  # (B, L)
        asym_id = batch["asym_id"]  # (B, L)
        is_same_chain = asym_id[..., :, None] == asym_id[..., None, :]  # (B, L, L)
        rel_pos = residx[..., :, None] - residx[..., None, :]  # (B, L, L)
        rel_pos = torch.clamp(rel_pos + self.max_r, 0, 2 * self.max_r)
        rel_pos = torch.where(is_same_chain, rel_pos, 2 * self.max_r + 1)
        a_rel_pos = torch.nn.functional.one_hot(rel_pos, self.rel_pos_dim)
        p = self.linear_rel_pos(a_rel_pos.float())  # (B, L, L, num_blocks * num_heads)
        # Rearrange to (num_blocks, num_heads, B, L, L)
        p = einops.rearrange(p, "b i j (n h) -> n b h i j", n=self.num_blocks)
        # Expand to atom level
        p = einops.repeat(p, "n b h i j -> n b h (i a) (j a)", a=14)

        # Compute attention mask
        pair_mask = torch.abs(rel_pos) <= self.max_r  # (B, L, L)
        # Expand to atom level
        pair_mask = einops.repeat(pair_mask, "b i j -> b (i 14) (j 14)")
        # Apply atom masks
        atom_mask = batch["atom14_mask"].view(B, L * 14)  # (B, L*14)
        pair_mask &= atom_mask[:, :, None] & atom_mask[:, None, :]

        # Local attention with relative positional bias
        q = self.local_stack(
            q.view(B, N, L * 14, -1),
            mask=pair_mask.view(B, 1, L * 14, L * 14),
            cond=c.view(B, N, L * 14, -1),
            pair_bias=p.view(self.num_blocks, self.num_heads, B, L * 14, L * 14),
        ).view(B, N, L, 14, -1)  # (B, N, L, 14, C_a)
        return q
