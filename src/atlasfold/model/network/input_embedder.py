import torch
import torch.nn as nn

from atlasfold.model.network.misc import relative_position_encoding
from atlasfold.model.network.primitives import LayerNorm, LinearNoBias


class InputEmbedder(nn.Module):
    def __init__(
        self,
        channel_s: int,
        channel_lm: int,
        channel_z: int,
        num_layers_lm: int,
        num_heads_lm: int,
    ) -> None:
        """Initialize the KFoldTrunk module."""
        super().__init__()
        # PLM hidden states [B, L, n_layers, c_lm] -> [B, L, c_plm]
        self.layernorm_lm_emb = nn.Module(
            LayerNorm(channel_lm) for _ in range(num_layers_lm)
        )
        self.gate_lm_emb = nn.Parameter(torch.zeros(num_layers_lm))
        self.proj_lm_emb = nn.Sequential(
            LinearNoBias(channel_lm, channel_s, init="relu"),
            nn.ReLU(),
            LinearNoBias(channel_s, channel_s, init="default"),
        )
        self.embedding_aa = LinearNoBias(21, channel_s)

        # PLM attention maps [B, L, L, n_layers, n_heads] -> [B, L, L, c_z]
        n_attn = num_layers_lm * num_heads_lm
        self.proj_lm_attn = nn.Sequential(
            LayerNorm(n_attn),
            LinearNoBias(n_attn, channel_z, init="relu"),
            nn.ReLU(),
            LinearNoBias(channel_z, channel_z, init="relu"),
        )
        rel_pos_dim = (2 * 32 + 2) + (2 * 2 + 2) + 1  # r_max=32, s_max=2
        self.linear_rel_pos = LinearNoBias(rel_pos_dim, channel_z, init="default")

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        hs: list[torch.Tensor],
        attns: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the input embedder module.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            The input batch.
        hs : list[torch.Tensor]
            List of length n_layers containing tensors of shape (B, L, c_lm) with the
            hidden states from the PLM.
        attns : list[torch.Tensor]
            List of length n_layers containing tensors of shape (B, L, L, n_heads) with
            the attention maps from the PLM.

        Returns
        -------
        s : torch.Tensor
            Tensor of shape (B, L, c_s) containing sequence embeddings.
        z : torch.Tensor
            Tensor of shape (B, L, L, c_z) containing pair embeddings.
        """
        # Initial single representation
        g = self.gate_lm_emb.softmax(dim=0)  # [n_layers,]
        hs = [ln(h) for ln, h in zip(self.layernorm_lm_emb, hs, strict=True)]
        s = torch.einsum("n,blnd->bld", g, torch.stack(hs, dim=-2))  # [B, L, c_lm]
        s = self.proj_lm_emb(s)  # [B, L, c_plm]
        s += self.embedding_aa(batch["aatype"])  # [B, L, c_s]

        # Initial pair representation from attention maps
        z = self.proj_lm_attn(torch.cat(attns, dim=-1))  # [B, L, L, c_z]
        rel_pos = relative_position_encoding(
            batch["residue_index"], batch["asym_id"], batch["entity_id"], batch["sym_id"]
        )  # [B, L, L, rel_pos_dim]
        z += self.linear_rel_pos(rel_pos)  # [B, L, L, c_z]

        return s, z
