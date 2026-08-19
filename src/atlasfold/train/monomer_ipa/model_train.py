"""Training forward pass for the monomer IPA regression model."""

from __future__ import annotations

from typing import Any

import torch

from atlasfold.model.model_ipa import AtlasFold_IPA, get_distogram
from atlasfold.utils.torch_utils import get_context_dtype


class AtlasFoldIPAForTrain(AtlasFold_IPA):
    # ==================================================
    # Forward functions for training step.
    # ==================================================
    def compile_train(self, **kwargs: Any) -> None:
        """Compile the function used in the training step."""
        self._trunk_pass = torch.compile(self._trunk_pass, **kwargs)

    def _trunk_pass(
        self,
        batch: dict[str, torch.Tensor],
        s_prev: torch.Tensor,
        z_prev: torch.Tensor,
        x_prev: torch.Tensor,
        mask: torch.Tensor,
        mlm_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        s = self.s_init(batch["aatype"])
        a, b = self.z_init(batch["aatype"]).chunk(2, dim=-1)
        z = a[..., :, None, :] + b[..., None, :, :]
        dtype = z.dtype

        # Add relative positional encoding to z
        z = z + self.z_rel_pos(batch["rel_pos"])

        # Add recycled embeddings from previous iteration
        s = s + self.recycle_s(s_prev)
        z = z + self.recycle_z(z_prev)
        dgram = get_distogram(x_prev, batch["pseudo_beta"], **self.recycle_pos_cfg)
        z = z + self.recycle_pos(dgram.to(dtype))

        # Add LM embeddings
        s_lm, z_lm = self.run_lm_embedder(batch, mlm_mask)
        s = s + self.proj_s_lm(s_lm)
        z = z + self.proj_z_lm(z_lm)

        # Run the main stack of the model
        s, z = self.main_stack(s, z, mask, self.use_kernel)

        # Run the structure module
        with torch.autocast(device_type=self.device.type, enabled=False):
            _s, _z = s.float(), z.float()
            structure = self.structure_module(_s, _z, batch["aatype_int"], mask)
        return s, z, structure

    def forward_train(
        self,
        batch: dict[str, torch.Tensor],
        num_recycles: int,
        mlm_prob: float = 0.15,
    ) -> dict[str, Any]:
        mask = batch["seq_mask"]
        B, L = mask.shape
        device = mask.device
        dtype = get_context_dtype(device.type)
        s_prev = torch.zeros(B, L, self.channel_s, device=device, dtype=dtype)
        z_prev = torch.zeros(B, L, L, self.channel_z, device=device, dtype=dtype)
        x_prev = torch.zeros(B, L, 14, 3, device=device)

        # Compute relative positional encoding for the batch
        batch["rel_pos"] = self.rel_pos_encoding(batch)

        out: dict[str, Any] = {}
        for recycle_i in range(num_recycles + 1):
            final = recycle_i == num_recycles
            enable_grad = final and self.training
            mlm_mask = self.sample_mlm_mask(batch, mlm_prob, synchronized=False)
            with torch.set_grad_enabled(enable_grad):
                s, z, structure = self._trunk_pass(
                    batch, s_prev, z_prev, x_prev, mask, mlm_mask
                )
            if not final:
                s_prev, z_prev = s.detach(), z.detach()
                x_prev = structure["coords"].detach()

        out["structure"] = structure

        with torch.autocast(device_type=s.device.type, enabled=False):
            s, z = s.float(), z.float()
            out["distogram"] = self.distogram_head(z)
            out["confidence"] = {
                "plddt": self.plddt_head(structure["act"].float()),
                "pae": self.pae_head(z),
                "experimentally_resolved": self.experimentally_resolved_head(s),
            }
        return out

    def run_lm_embedder(
        self,
        batch: dict[str, torch.Tensor],
        mlm_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract frozen-LM features and refine their single/pair projections."""
        # === Prepare LM inputs === #
        input_ids = batch["lm.input_ids"]  # [B, S]
        pos_id = batch["lm.pos_id"]  # [B, S]
        seq_id = batch["lm.seq_id"]  # [B, S]. 1 for valid, 0 for padding

        # Apply MLM mask to input IDs for stochastic feature extraction
        if mlm_mask is not None:
            assert mlm_mask.shape[1] == input_ids.shape[1], (
                f"MLM mask must have the same length dimension as input_ids. "
                f"Got {mlm_mask.shape} and {input_ids.shape}."
            )
            aa_idxs = torch.tensor(self.alphabet.aa_idxs, device=input_ids.device)
            mlm_mask = mlm_mask & torch.isin(input_ids, aa_idxs)
            input_ids = input_ids.masked_fill(mlm_mask, self.alphabet.mask_idx)

        # Prepare crop indices for LM embeddings and attention
        row_s = batch["seq_tok_idx"]
        B, S = row_s.shape
        device = row_s.device
        batch_index = torch.arange(B, device=row_s.device)
        b_i_s = batch_index[:, None]
        b_i_z = batch_index[:, None, None]
        row_z, col_z = row_s[:, :, None], row_s[:, None, :]

        # === Accumulate LM features === #
        lm_emb = torch.zeros((B, S, self.lm.d_model), device=device, dtype=torch.float32)
        lm_attn = torch.zeros(
            (B, S, S, self.channel_z), device=device, dtype=torch.float32
        )
        w_layers = self.lm_layer_weights.softmax(dim=0)  # [n_layers+1,]

        with torch.no_grad():
            x = self.lm.embed(input_ids)
            x_crop = x[b_i_s, row_s]
        lm_emb = lm_emb + w_layers[0] * self.layernorm_lm_emb(x_crop)

        # Run LM blocks and accumulate embeddings and attention
        for i, block in enumerate(self.lm.transformer.blocks):
            with torch.no_grad():
                x, attn = block(x, seq_id, pos_id, return_attn_logits=True)
                # neginf will be set to 0 at the end of this function.
                attn = attn.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                attn = attn.clamp_(-100.0, 100.0).div_(100)
                attn = attn.moveaxis(1, -1)  # [B, S, S, n_heads]
                x_crop = x[b_i_s, row_s]
                attn_crop = attn[b_i_z, row_z, col_z]
                del attn
            lm_emb = lm_emb + w_layers[i + 1] * self.layernorm_lm_emb(x_crop)
            lm_attn = lm_attn + self.proj_lm_attn[i](attn_crop)

        # Project LM embeddings and attention to single/pair representations
        s_lm = self.lm_emb_to_s_lm(lm_emb)
        z_lm = self.lm_attn_to_z_lm(lm_attn)

        # Run LM stack
        mask = batch["seq_mask"]  # [B, L]
        s_lm, z_lm = self.lm_stack(s_lm, z_lm, mask, self.use_kernel)
        return s_lm, z_lm
