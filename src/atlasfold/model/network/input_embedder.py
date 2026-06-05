from functools import partial

import torch

from atlasfold.model.network.primitives import LayerNorm, LinearNoBias, SwiGLU
from atlasfold.utils.torch_utils import add
from atlaslm.model import AtlasLM


class LMInputEmbedder(torch.nn.Module):
    """
    Wrapper module for the AtlasLM language model to extract hidden states
    and attention maps.
    """

    def __init__(self, lm: AtlasLM, channel_s: int, channel_z: int) -> None:
        super().__init__()
        self.lm: AtlasLM = lm
        self.alphabet = lm.alphabet

        self.channel_s: int = channel_s
        self.channel_z: int = channel_z
        self.channel_lm: int = lm.d_model

        # Hidden states [B, L, n_layers, c_lm] -> [B, L, c_s]
        self.layernorm_emb = LayerNorm(lm.d_model)
        self.w_emb = torch.nn.Parameter(torch.zeros(lm.n_layers))
        self.mlp_emb = torch.nn.Sequential(
            LinearNoBias(lm.d_model, channel_s, init="relu"),
            torch.nn.ReLU(),
            LinearNoBias(channel_s, channel_s, init="default"),
        )

        # Attention maps [B, L, L, n_layers, n_heads] -> [B, L, L, c_z]
        self.attn_logit_scale = 1.0 / (lm.n_heads * lm.n_layers) ** 0.5
        self.proj_attn = torch.nn.ModuleList(
            [LinearNoBias(lm.n_heads, channel_z) for _ in range(lm.n_layers)]
        )
        self.mlp_attn = torch.nn.Sequential(
            SwiGLU(channel_z, channel_z),
            LinearNoBias(channel_z, channel_z, init="default"),
        )

        # Restype embedding
        self.embed_aa = LinearNoBias(21, channel_s)

        # Relative positional encoding for pair representation
        rel_pos_dim = (2 * 32 + 2) + (2 * 2 + 2) + 1  # r_max=32, s_max=2
        self.linear_rel_pos = LinearNoBias(rel_pos_dim, channel_z, init="default")

        # MLM masking.
        self.register_buffer(
            "aa_idxs",
            torch.tensor(self.alphabet.aa_idxs, dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        mlm_mask: torch.Tensor,
        train: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the input embedder module.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            The input batch.
        mlm_mask : torch.Tensor
            Boolean mask of shape (B, L+2) indicating which positions are masked for
            language model feature extraction.
        train : bool, optional
            Whether the model is in training mode.


        Returns
        -------
        s : torch.Tensor
            Tensor of shape (B, L, c_s) containing single representations.
        z : torch.Tensor
            Tensor of shape (B, L, L, c_z) containing pair representations.
        """
        _add = partial(add, inplace=not train)

        # Prepare inputs
        input_ids = batch["lm.input_ids"]  # [B, L+2]
        pos_id = batch["lm.pos_id"]  # [B, L+2]
        seq_id = batch["lm.seq_id"]  # [B, L+2]

        # Replace masked positions with the mask token ID
        mlm_mask = mlm_mask & torch.isin(input_ids, self.aa_idxs[None, None, :])
        input_ids = input_ids.masked_fill(mlm_mask, self.alphabet.mask_idx)

        # During training, we may crop the input sequences
        crop_pos_id = batch.get("lm.cropped_pos_id", None)  # [B, L]
        if crop_pos_id is not None:
            b_idx = torch.arange(input_ids.shape[0], device=input_ids.device)
            b_idx_s, b_idx_z = b_idx[:, None], b_idx[:, None, None]
            crop_rows = crop_pos_id[:, :, None]
            crop_cols = crop_pos_id[:, None, :]

        # Embed input tokens
        with torch.no_grad():
            x = self.lm.embed(input_ids)

        # Initialize single and pair representations to accumulate
        # hidden states and attention maps from all layers of the language model
        B, L = batch["aatype"].shape[:2]
        s = torch.zeros((B, L, self.channel_lm), device=x.device, dtype=x.dtype)
        z = torch.zeros((B, L, L, self.channel_z), device=x.device, dtype=x.dtype)

        # Iterate language model layers and extract hidden states and attention maps
        w = self.w_emb.softmax(dim=0)  # [n_layers,]
        for i, block in enumerate(self.lm.transformer.blocks):
            with torch.no_grad():
                x, attn = block(
                    x, seq_id, pos_id, return_attn=True, return_attn_logits=True
                )
                attn = attn.permute(0, 2, 3, 1)  # [B, L+2, L+2, n_heads]
                if crop_pos_id is not None:
                    _x = x[b_idx_s, crop_pos_id, :]
                    _attn = attn[b_idx_z, crop_rows, crop_cols]
                else:
                    _x = x[:, 1:-1, :]
                    _attn = attn[:, 1:-1, 1:-1]
            s = _add(s, w[i] * self.layernorm_emb(_x))
            z = _add(z, self.proj_attn[i](_attn))

        s = self.mlp_emb(s)
        s = _add(s, self.embed_aa(batch["aatype"]))  # [B, L, c_s]

        z = self.mlp_attn(z * self.attn_logit_scale)
        z = _add(z, self.linear_rel_pos(batch["rel_pos"]))  # [B, L, L, c_z]
        return s, z
