"""Folding Trunk."""

from typing import Any

import torch

from atlasfold.model import AtlasFold
from atlasfold.model.network.diffusion_head import DiffusionHead, SamplingConfig
from atlasfold.model.utils import confidence_metrics
from atlasfold.utils.geometry.random_augment import center_random_augmentation
from atlasfold.utils.torch_utils import expand_dim


class AtlasFoldForTrain(AtlasFold):
    def compile_submodules(self, **kwargs) -> None:
        """Compile the submodules of the model."""
        self.lm_embedder = torch.compile(self.lm_embedder, **kwargs)
        self.lm_stack = torch.compile(self.lm_stack, **kwargs)
        self.main_stack = torch.compile(self.main_stack, **kwargs)
        self.diffusion_head.do_compile(**kwargs)
        self.confidence_head = torch.compile(self.confidence_head, **kwargs)

    def get_model(self, name: str) -> torch.nn.Module:
        model = getattr(self, name)
        if hasattr(model, "_orig_mod") and not self.training:
            return model._orig_mod
        return model

    def get_module_groups(self) -> dict[str, list[torch.nn.Module]]:
        return {
            "lm": [self.lm_embedder.lm],
            "trunk": [self.lm_embedder, self.recycle_z, self.lm_stack, self.main_stack],
            "distogram": [self.distogram_head],
            "diffusion": [self.diffusion_head],
            "confidence": [self.confidence_head],
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

        # Recycling iterations with stochastic masking during training.
        z_prev = torch.zeros(B, L, L, self.channel_z, device=device)
        mlm_prob = torch.rand(1).item() * 0.2
        for i in range(0, num_recycles + 1):
            enable_grad = self.training and i == num_recycles
            with torch.set_grad_enabled(enable_grad):
                if enable_grad and torch.is_autocast_enabled():
                    torch.clear_autocast_cache()
                # For each recycle step, sample a MLM mask to extract new LM features.
                mlm_mask = self.sample_mlm_mask(batch, mlm_prob, synchronized=False)
                # Extract LM features
                s_lm, z_lm = self.lm_embedder(batch, mlm_mask, enable_grad)
                # Run LM module
                z = self.lm_stack(s_lm, z_lm, mask, self.use_cuequiv_kernels)
                # Recycling
                z = z + self.recycle_z(z_prev)
                # Run main trunk
                z = self.main_stack(z, mask, self.use_cuequiv_kernels)
                z_prev = z
        s, z = s_lm, z

        # Return distogram logits
        if train_trunk:
            distogram_out = self.run_distogram_head(z)
            distogram_aug_out = self.run_distogram_head(z_lm)
            out["distogram"] = distogram_out
            out["distogram_aug"] = distogram_aug_out

        # Return diffusion head outputs
        if train_diffusion_head:
            # Select the diffusion conditioning.
            # trunk : init : zero = 6 : 2 : 2
            p = torch.rand(bs, device=s.device)
            use_trunk = p < 0.6
            use_init = (p >= 0.6) & (p < 0.8)
            use_cond = use_trunk | use_init
            _s = use_cond.view(bs, 1, 1) * s_lm  # s = s_lm
            _z = use_trunk.view(bs, 1, 1, 1) * z + use_init.view(bs, 1, 1, 1) * z_lm
            diffusion_out = self.forward_diffusion(
                batch, label, _s, _z, diffusion_batch_size
            )
            out["diffusion"] = diffusion_out

        # Return confidence head outputs
        if train_confidence_head:
            # Sample the structure for confidence head training.
            assert sampling_config.num_samples == 1, (
                "Only single sample is supported for confidence head training."
            )
            with torch.no_grad():
                sample_coords = self.diffusion_head.sample(batch, s, z, sampling_config)

            # Select the confidence head conditioning.
            # trunk : init : zero = 6 : 2 : 2
            p = torch.rand(bs, device=s.device)
            use_trunk = p < 0.6
            use_init = (p >= 0.6) & (p < 0.8)
            _s = s_lm  # No augmentation for confidence head training.
            _z = use_trunk.view(bs, 1, 1, 1) * z + use_init.view(bs, 1, 1, 1) * z_lm

            compute_pae = train_pae_head
            confidence_out = self.run_confidence_head(
                batch, _s, _z, sample_coords, compute_pae
            )
            confidence_out["mini_rollout"] = {"sample_coords": sample_coords}
            out["confidence"] = confidence_out

        return out

    def forward_diffusion(
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
        with torch.autocast(s.device.type, enabled=False):
            label_coords = label["coords"]  # [B, L, 14, 3]
            resolved_mask = label["resolved_mask"]  # [B, L, 14]
            x_gt = expand_dim(label_coords, N, dim=1)  # [B, N, L, 14, 3]
            mask = expand_dim(resolved_mask, N, dim=1)  # [B, N, L, 14]

            # Random augmentation
            x_gt = center_random_augmentation(x_gt, mask)  # [B, N, L, 14, 3]

            # Add noise
            noise = torch.randn_like(x_gt)  # [B, N, L, 14, 3]
            x_noisy = x_gt + t_hat.view(B, N, 1, 1, 1) * noise

        # Forward pass through the score model
        _t_hat = t_hat.view(B, N, 1, 1, 1)  # [B, N, 1, 1, 1]
        c_in = model.c_in(_t_hat)
        c_skip = model.c_skip(_t_hat)
        c_out = model.c_out(_t_hat)
        loss_weights = 1 / model.c_out(t_hat) ** 2  # [B, N]

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

    # ==================================================
    # Forward functions for validation step.
    # ==================================================
    @torch.inference_mode()
    def sample_validation(
        self,
        batch: dict[str, torch.Tensor],
        num_recycles: int,
        sampling_config: SamplingConfig,
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}

        # For validation, we always use the base trunk without stochastic recycling.
        s, z = self.run_trunk_base(batch, num_recycles, mlm_prob=0.0, train=False)

        # Run distogram head
        distogram_out = self.run_distogram_head(z)
        out["distogram.logits"] = distogram_out["logits"]
        out["distogram.boundaries"] = distogram_out["boundaries"]

        # Run diffusion heads
        sample_coords = self.diffusion_head.sample(
            batch,
            s,
            z,
            sampling_config,
            use_compiled_model=False,
        )
        out["sample_coords"] = sample_coords

        # Run confidence head
        confidence_out = self.run_confidence_head(
            batch, s, z, sample_coords, compute_pae=False
        )
        del s, z

        # Compute confidence metrics
        mask = batch["seq_mask"]
        if "plddt" in confidence_out:
            out["plddt"] = confidence_metrics.compute_plddt(
                **confidence_out["plddt"], mask=mask
            )
        return out

    def run_trunk_base(
        self,
        batch: dict[str, torch.Tensor],
        num_recycles: int,
        mlm_prob: float | None = None,
        train: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run LM module at once before trunk iteration."""
        if num_recycles < 0:
            raise ValueError("num_recycles must be non-negative for base mode.")
        lm_embedder = self.get_model("lm_embedder")
        lm_stack = self.get_model("lm_stack")
        main_stack = self.get_model("main_stack")

        mask = batch["seq_mask"]
        s_lm, z_lm = lm_embedder(batch)
        z_init = lm_stack(s_lm, z_lm, mask, self.use_cuequiv_kernels)

        z_prev = torch.zeros_like(z_lm)  # [B, L, L, c_z]
        for _ in range(0, num_recycles + 1):
            # Recycling embedding
            z = z_init + self.recycle_z(z_prev)
            # Run main trunk
            z = main_stack(z, mask, self.use_cuequiv_kernels)
            z_prev = z
        return s_lm, z

    def run_distogram_head(
        self,
        z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        distogram_head = self.get_model("distogram_head")
        distogram_out = distogram_head(z)
        return distogram_out

    def run_confidence_head(
        self,
        batch: dict[str, torch.Tensor],
        s: torch.Tensor,
        z: torch.Tensor,
        x_pred: torch.Tensor,
        compute_pae: bool,
    ) -> dict[str, dict[str, torch.Tensor]]:
        confidence_head = self.get_model("confidence_head")
        confidence_out = confidence_head(
            batch, s, z, x_pred, compute_pae, self.use_cuequiv_kernels
        )
        return confidence_out
