"""Training forward pass for the monomer IPA regression model."""

from __future__ import annotations

from typing import Any

import torch

from atlasfold.model.model_ipa import AtlasFold_IPA, get_distogram
from atlasfold.utils.torch_utils import get_context_dtype


class AtlasFoldIPAForTrain(AtlasFold_IPA):
    def compile_train(self, **kwargs: Any) -> None:
        self._trunk_pass = torch.compile(self._trunk_pass, **kwargs)

    def _trunk_pass(
        self,
        batch: dict[str, torch.Tensor],
        s_prev: torch.Tensor,
        z_prev: torch.Tensor,
        x_prev: torch.Tensor,
        mask: torch.Tensor,
        mlm_mask: torch.Tensor,
        train: bool,
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
        s_lm, z_lm = self.run_lm_embedder(batch, mlm_mask, train=train)
        s = s + self.proj_s_lm(s_lm)
        z = z + self.proj_z_lm(z_lm)

        # Run the main stack of the model
        s, z = self.main_stack(s, z, mask, self.use_kernel)
        # Run the structure module
        structure = self.structure_module(s, z, batch["aatype_int"], mask)
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
            mlm_mask = self.sample_mlm_mask(batch, mlm_prob)
            with torch.set_grad_enabled(enable_grad):
                s, z, structure = self._trunk_pass(
                    batch, s_prev, z_prev, x_prev, mask, mlm_mask, train=enable_grad
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
