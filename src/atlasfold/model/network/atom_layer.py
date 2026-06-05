import einops
import torch
import torch.nn as nn

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


class AtomEncoder(nn.Module):
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
        """
        super().__init__()
        self.embedding_aa_atoms = LinearNoBias(21, 14 * channel_atom)
        self.linear_in = LinearNoBias(3, channel_atom, init="default", precision=32)
        self.linear_cond = LinearNoBias(channel_cond, channel_atom, init="final")
        self.atom_mlp = AtomMLP(channel_atom, hidden_dim=channel_atom // 2)

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
        """
        mask = batch["atom14_mask"].unsqueeze(-3)  # (B, 1, L, 14)
        mask = mask.float()

        # Get the initial atom representations
        aatype = batch["aatype"]  # (B, L, 21)
        q = self.embedding_aa_atoms(aatype).unflatten(-1, (14, -1))  # (B, L, 14, C_a)
        q = q.unsqueeze(-4)  # (B, 1, L, 14, C_a)

        # Embed the current state: noisy atom coordinates
        q = q + self.linear_in(r_noisy)  # (B, N, L, 14, C_a)

        # Embed the conditioning
        q = q + self.linear_cond(cond).unsqueeze(-2)  # (B, N, L, 14, C_a)

        # Residue-wise MLP on the intra-residue geometry
        q = q + self.atom_mlp(q, mask)  # (B, N, L, 14, C_a)

        # Mask out dummy atoms
        q = q * mask[..., None]

        return q


class AtomDecoder(nn.Module):
    def __init__(
        self,
        channel_atom: int = 96,
    ) -> None:
        """Initialize the Atom decoding layer."""
        super().__init__()
        self.atom_mlp = AtomMLP(channel_atom, hidden_dim=channel_atom // 2)
        self.linear_q_to_r = LinearNoBias(channel_atom, 3, init="final", precision=32)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        q: torch.Tensor,
    ) -> torch.Tensor:
        """Maps updated representations back to raw 3D coordinate updates."""
        mask = batch["atom14_mask"].unsqueeze(-3).float()  # (B, 1, L, 14)

        # Final post-processing block of intra-residue updates
        q = q + self.atom_mlp(q, mask)
        q = q * mask[..., None]

        # Project representations directly to raw 3D coordinate trajectories
        r_update = self.linear_q_to_r(q)  # (B, N, L, 14, 3)
        r_update = r_update * mask[..., None]

        return r_update
