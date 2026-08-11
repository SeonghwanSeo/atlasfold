"""Training forward pass for the monomer IPA regression model."""

from __future__ import annotations

from typing import Any

import torch

from atlasfold.model.model_ipa import AtlasFold_IPA, get_distogram
from atlasfold.utils.torch_utils import get_context_dtype


class AtlasFoldIPAForTrain(AtlasFold_IPA):
    def get_module_groups(self) -> dict[str, list[torch.nn.Module | torch.nn.Parameter]]:
        return {
            "lm": [self.lm],
            "trunk": [
                self.s_init,
                self.z_init,
                self.z_rel_pos,
                self.recycle_s,
                self.recycle_z,
                self.lm_layer_weights,
                self.layernorm_lm_emb,
                self.lm_emb_to_s_lm,
                self.proj_lm_attn,
                self.lm_attn_to_z_lm,
                self.lm_stack,
                self.proj_s_lm,
                self.proj_z_lm,
                self.main_stack,
                self.recycle_pos,
            ],
            "distogram_head": [self.distogram_head],
            "structure_module": [self.structure_module],
            "confidence_head": [
                self.plddt_head,
                self.pae_head,
                self.experimentally_resolved_head,
            ],
            "plddt_head": [self.plddt_head],
            "pae_head": [self.pae_head],
            "experimentally_resolved_head": [self.experimentally_resolved_head],
        }

    def get_module_group_names(self) -> dict[str, list[str]]:
        return {
            "lm": ["lm"],
            "trunk": [
                "s_init",
                "z_init",
                "z_rel_pos",
                "recycle_s",
                "recycle_z",
                "lm_layer_weights",
                "layernorm_lm_emb",
                "lm_emb_to_s_lm",
                "proj_lm_attn",
                "lm_attn_to_z_lm",
                "lm_stack",
                "proj_s_lm",
                "proj_z_lm",
                "main_stack",
                "recycle_pos",
            ],
            "distogram_head": ["distogram_head"],
            "structure_module": ["structure_module"],
            "confidence_head": [
                "plddt_head",
                "pae_head",
                "experimentally_resolved_head",
            ],
            "plddt_head": ["plddt_head"],
            "pae_head": ["pae_head"],
            "experimentally_resolved_head": ["experimentally_resolved_head"],
        }

    def compile_train(self, **kwargs: Any) -> None:
        self._trunk_pass = torch.compile(self._trunk_pass, **kwargs)
        self.structure_module = torch.compile(self.structure_module, **kwargs)

    def _add_template(
        self,
        batch: dict[str, torch.Tensor],
        z: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        return z

    def _trunk_pass(
        self,
        batch: dict[str, torch.Tensor],
        s_prev: torch.Tensor,
        z_prev: torch.Tensor,
        x_prev: torch.Tensor,
        seq_mask: torch.Tensor,
        mlm_mask: torch.Tensor,
        train: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.s_init(batch["aatype"])
        z_i, z_j = self.z_init(batch["aatype"]).chunk(2, dim=-1)
        z = z_i[..., :, None, :] + z_j[..., None, :, :]
        z = z + self.z_rel_pos(batch["rel_pos"])
        s = s + self.recycle_s(s_prev)
        z = z + self.recycle_z(z_prev)
        dgram = get_distogram(
            x_prev, batch["pseudo_beta"], **self.recycle_pos_cfg
        ).to(z.dtype)
        z = z + self.recycle_pos(dgram)
        s_lm, z_lm = self.run_lm_embedder(batch, mlm_mask, train=train)
        s = s + self.proj_s_lm(s_lm)
        z = z + self.proj_z_lm(z_lm)
        z = self._add_template(batch, z, seq_mask)
        return self.main_stack(s, z, seq_mask, self.use_kernel)

    def forward_train(
        self,
        batch: dict[str, torch.Tensor],
        num_recycles: int,
        train_trunk: bool = True,
        train_structure_module: bool = True,
        run_distogram_head: bool = True,
        run_confidence_head: bool = True,
    ) -> dict[str, Any]:
        seq_mask = batch["seq_mask"]
        B, L = seq_mask.shape
        dtype = get_context_dtype(seq_mask.device.type)
        s_prev = torch.zeros(B, L, self.channel_s, device=seq_mask.device, dtype=dtype)
        z_prev = torch.zeros(B, L, L, self.channel_z, device=seq_mask.device, dtype=dtype)
        x_prev = torch.zeros(B, L, 14, 3, device=seq_mask.device)
        batch["rel_pos"] = self.rel_pos_encoding(batch)

        structure: dict[str, Any] = {}
        for recycle in range(num_recycles + 1):
            final = recycle == num_recycles
            trunk_grad = final and train_trunk and self.training
            structure_grad = final and train_structure_module and self.training
            mlm_mask = self.sample_mlm_mask(batch, 0.15)
            with torch.set_grad_enabled(trunk_grad):
                s, z = self._trunk_pass(
                    batch,
                    s_prev,
                    z_prev,
                    x_prev,
                    seq_mask,
                    mlm_mask,
                    trunk_grad,
                )
            with torch.set_grad_enabled(structure_grad or trunk_grad):
                structure = self.structure_module(
                    s,
                    z,
                    batch["aatype_int"],
                    seq_mask,
                )
            if not final:
                s_prev = s.detach()
                z_prev = z.detach()
                x_prev = structure["coords"].detach()

        s, z = s.float(), z.float()
        out: dict[str, Any] = {
            "s": s,
            "z": z,
            "structure": structure,
        }
        with torch.autocast(device_type=s.device.type, enabled=False):
            if run_distogram_head:
                out["distogram"] = self.distogram_head(z)
            if run_confidence_head:
                plddt = self.plddt_head(structure["act"].float())
                pae = self.pae_head(z)
                experimentally_resolved = self.experimentally_resolved_head(s)
                out["confidence"] = {
                    "plddt": plddt,
                    "pae": {
                        "logits": pae["logits"],
                        "bin_centers": pae["bin_centers"],
                    },
                    "experimentally_resolved_logits": experimentally_resolved[
                        "logits"
                    ],
                }
        return out
