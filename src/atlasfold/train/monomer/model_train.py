"""Training module"""

from typing import Any

import torch

from atlasfold.model import AtlasFold
from atlasfold.model.network.diffusion_head import DiffusionHead, SamplingConfig
from atlasfold.utils.geometry.random_augment import center_random_augmentation_atom14
from atlasfold.utils.torch_utils import expand_dim, get_context_dtype


class AtlasFoldForTrain(AtlasFold):
    def compile_train(self, **kwargs) -> None:
        """Compile the functions used in the training step."""
        self.__forward_trunk = torch.compile(self.__forward_trunk, **kwargs)
        self.__forward_distogram = torch.compile(self.__forward_distogram, **kwargs)
        self.__forward_diffusion = torch.compile(self.__forward_diffusion, **kwargs)
        self.__forward_mini_rollout = torch.compile(self.__forward_mini_rollout, **kwargs)
        self.__forward_confidence = torch.compile(self.__forward_confidence, **kwargs)

    def get_module_groups(self) -> dict[str, list[torch.nn.Module | torch.nn.Parameter]]:
        return {
            "lm": [self.lm],
            "trunk": [
                self.w_lm_emb,
                self.layernorm_lm_emb,
                self.s_init,
                self.embed_aa,
                self.proj_lm_attn,
                self.z_init,
                self.linear_rel_pos,
                self.recycle_s,
                self.recycle_z,
                self.lm_stack,
                self.main_stack,
            ],
            "distogram_head": [self.distogram_head],
            "diffusion_head": [self.diffusion_head],
            "confidence_head": [self.confidence_head],
            "pae_head": [self.confidence_head.pae_head],
        }

    # ==================================================
    # Forward functions for training step.
    # ==================================================
    def forward_train(
        self,
        batch: dict[str, torch.Tensor],
        label: dict[str, torch.Tensor],
        num_recycles: int,
        diffusion_batch_size: int,
        train_trunk: bool,
        train_diffusion_head: bool,
        train_confidence_head: bool,
        train_pae_head: bool,
        sampling_config: SamplingConfig,
    ) -> dict[str, Any]:
        bs = batch["aatype"].shape[0]

        # Return dictionary for training outputs.
        out: dict[str, Any] = {}

        mask = batch["seq_mask"]
        B, L = mask.shape
        device = mask.device
        dtype = get_context_dtype()

        # Compute positional encodings
        self.compute_rel_pos_encoding(batch)

        # Recycling iterations with stochastic masking during training.
        s_prev = torch.zeros(B, L, self.channel_s, device=device, dtype=dtype)
        z_prev = torch.zeros(B, L, L, self.channel_z, device=device, dtype=dtype)
        for i in range(0, num_recycles + 1):
            enable_grad = self.training and i == num_recycles
            with torch.set_grad_enabled(enable_grad):
                if enable_grad and torch.is_autocast_enabled():
                    torch.clear_autocast_cache()
                s, z = self.__forward_trunk(
                    batch, s_prev, z_prev, mask, train=enable_grad
                )
            s_prev, z_prev = s, z
        s, z = s.float(), z.float()

        # Return distogram logits
        if train_trunk:
            distogram_out = self.__forward_distogram(z)
            out["distogram"] = distogram_out

        # Return diffusion head outputs
        if train_diffusion_head:
            # Diffusion conditioning.
            p = torch.rand(bs, device=s.device)
            use_cond = p < 0.8
            _s = use_cond.view(bs, 1, 1) * s
            _z = use_cond.view(bs, 1, 1, 1) * z
            with torch.autocast(s.device.type, dtype=torch.float32, enabled=True):
                diffusion_out = self.__forward_diffusion(
                    batch, label, _s, _z, diffusion_batch_size
                )
            out["diffusion"] = diffusion_out

        # Return confidence head outputs
        if train_confidence_head:
            # Sample the structure for confidence head training.
            sample_coords = self.__forward_mini_rollout(batch, s, z, sampling_config)

            # Select the confidence head conditioning.
            p = torch.rand(bs, device=s.device)
            use_cond = p < 0.8
            _s = s  # No augmentation for confidence head training.
            _z = use_cond.view(bs, 1, 1, 1) * z

            confidence_out = self.__forward_confidence(
                batch, _s, _z, sample_coords, compute_pae=train_pae_head
            )
            confidence_out["mini_rollout"] = {"sample_coords": sample_coords}

            # Remove the sample dimension
            # [B, 1, ...] -> [B, ...]
            for k, v in confidence_out.items():
                for kk, vv in v.items():
                    if kk in ("logits", "sample_coords"):
                        assert vv.shape[1] == 1, (
                            f"Expected sample dimension to be 1, but got {vv.shape}"
                            f"for key {k}/{kk}"
                        )
                        confidence_out[k][kk] = vv.squeeze(1)

            out["confidence"] = confidence_out

        return out

    def __forward_trunk(
        self,
        batch: dict[str, torch.Tensor],
        s_prev: torch.Tensor,
        z_prev: torch.Tensor,
        mask: torch.Tensor,
        train: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # For each recycle step, sample a MLM mask to extract new LM features.
        mlm_mask = self.sample_mlm_mask(batch, 0.15, synchronized=False)
        # Extract LM features
        s, z = self.run_lm_embedder(batch, mlm_mask, train)
        # Recycling
        s = s + self.recycle_s(s_prev)
        z = z + self.recycle_z(z_prev)
        # Run LM module
        s, z = self.lm_stack(s, z, mask, self.use_kernel)
        # Run main trunk
        z = self.main_stack(z, mask, self.use_kernel)
        return s, z

    def sample_mlm_mask(
        self,
        batch: dict[str, torch.Tensor],
        prob: float | torch.Tensor,
        synchronized: bool = True,
    ) -> torch.Tensor:
        """Sample a random MLM mask for the input batch.
        NOTE: synchronized masking is used for inference to ensure that
        the prediction is not changed by the batch size or the order of
        the sequences in the batch.
        """
        input_ids = batch["lm.input_ids"]  # [B, S]
        B, S = input_ids.shape
        shape = (1, S) if synchronized else (B, S)
        return torch.rand(shape, device=input_ids.device) < prob

    def __forward_distogram(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.distogram_head(z)

    def __forward_diffusion(
        self,
        batch: dict[str, torch.Tensor],
        label: dict[str, torch.Tensor],
        s: torch.Tensor,
        z: torch.Tensor,
        diffusion_batch_size: int,
    ) -> dict[str, torch.Tensor]:
        """Perform a single training step for the structure module.
        See Section 5 of EDM paper.
        """
        B, N = s.shape[0], diffusion_batch_size

        model: DiffusionHead = self.diffusion_head

        # Sample noise levels
        nt = torch.randn((B, N), dtype=torch.float32, device=s.device)
        t_hat = model.sigma_data * torch.exp(-1.2 + 1.5 * nt)

        # Repeat label coords with random augmentation for each diffusion training input.
        label_coords = label["coordinates"]  # [B, L, 14, 3]
        resolved_mask = label["resolved_mask"]  # [B, L, 14]
        x_gt = expand_dim(label_coords, N, dim=1)  # [B, N, L, 14, 3]
        mask = expand_dim(resolved_mask, N, dim=1)  # [B, N, L, 14]

        # Random augmentation
        x_gt = center_random_augmentation_atom14(x_gt, mask)

        # Add noise
        noise = torch.randn_like(x_gt)  # [B, N, L, 14, 3]
        x_noisy = x_gt + t_hat.view(B, N, 1, 1, 1) * noise
        x_noisy = x_noisy * mask[..., None]

        # Forward pass through the score model
        _t_hat = t_hat.view(B, N, 1, 1, 1)  # [B, N, 1, 1, 1]
        c_in = model.c_in(_t_hat)
        c_skip = model.c_skip(_t_hat)
        c_out = model.c_out(_t_hat)
        loss_weights = 1 / (model.c_out(t_hat) ** 2 + 1e-10)  # [B, N]

        # DiT Conditioning
        c_noise = model.c_noise(t_hat)
        single_cond = model.single_conditioning(batch, s, c_noise)  # [B, N, L, c_s]
        pair_bias = model.pair_conditioning(batch, z)  # [B, L, L, Nblock, Nhead]

        # Input EDM conditioning
        r_noisy = c_in * x_noisy  # [B, N, L, 14, 3]
        # Forward pass through the score model
        r_update = model.score_model(batch, r_noisy, single_cond, pair_bias)
        # Output EDM conditioning
        x_out = c_skip * x_noisy + c_out * r_update

        return {
            "x_noisy": x_noisy,
            "x_out": x_out,
            "x_gt": x_gt,
            "resolved_mask": resolved_mask,
            "loss_weights": loss_weights,
        }

    def __forward_mini_rollout(
        self,
        batch: dict[str, torch.Tensor],
        s: torch.Tensor,
        z: torch.Tensor,
        sampling_config: SamplingConfig,
    ) -> torch.Tensor:
        """Perform a mini rollout for the confidence head training."""
        with torch.no_grad():
            sample_coords = self.diffusion_head.sample(
                batch,
                s,
                z,
                num_samples=1,
                config=sampling_config,
            )
        return sample_coords  # [B, 1, L, 14, 3]

    def __forward_confidence(
        self,
        batch: dict[str, torch.Tensor],
        s: torch.Tensor,
        z: torch.Tensor,
        sample_coords: torch.Tensor,
        compute_pae: bool,
    ) -> dict[str, dict[str, torch.Tensor]]:
        return self.confidence_head(
            batch, s, z, sample_coords, compute_pae, self.use_kernel
        )
