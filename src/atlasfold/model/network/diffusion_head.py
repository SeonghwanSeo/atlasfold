import dataclasses
import math
from functools import cached_property
from typing import TypeVar

import numpy as np
import torch
import torch.nn as nn

from atlasfold.model.network.atom_attention import AtomAttentionStack, AtomEmbedder
from atlasfold.model.network.diffusion_transformer import (
    DiffusionTransformerStack,
    PairConditioning,
    SingleConditioning,
)
from atlasfold.model.network.primitives import LayerNorm, LinearNoBias
from atlasfold.utils.geometry.random_augment import center_random_augmentation
from atlasfold.utils.torch_utils import expand_dim

SIGMA_DATA = 16.0

_ScalarOrTensor = TypeVar("_ScalarOrTensor", float, torch.Tensor)


@dataclasses.dataclass(kw_only=True)
class SamplingConfig:
    num_samples: int = 1
    num_steps: int = 100
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    rho: float = 7
    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5

    # Optional configurations
    chunk_size: int | None = None  # For memory-efficient inference

    @cached_property
    def sigmas(self) -> list[float]:
        """Get the noise schedule."""
        steps = np.linspace(0, 1, self.num_steps)
        inv_rho = 1 / self.rho
        sigma_max_pow = self.sigma_max**inv_rho
        sigma_min_pow = self.sigma_min**inv_rho
        sigmas = sigma_max_pow + steps * (sigma_min_pow - sigma_max_pow)
        sigmas = (sigmas**self.rho) * SIGMA_DATA
        return sigmas.tolist() + [0.0]

    @cached_property
    def gammas(self) -> list[float]:
        """Get the gamma schedule."""
        return [self.gamma_0 if s > self.gamma_min else 0.0 for s in self.sigmas]


class DiffusionModule(nn.Module):
    def __init__(
        self,
        channel_a: int = 768,
        channel_s: int = 384,
        channel_atom: int = 128,
        num_heads: int = 16,
        num_blocks: int = 24,
        num_atom_enc_heads: int = 4,
        num_atom_enc_blocks: int = 3,
        num_atom_dec_heads: int = 4,
        num_atom_dec_blocks: int = 3,
        blocks_per_ckpt: int | None = None,
    ) -> None:
        """Initialize the diffusion module.

        Parameters
        ----------
        channel_a : int
            The single representation dimension.
        channel_s : int
            The single conditioning dimension.
        channel_atom : int
            The atom representation dimension.
        num_heads : int
            The number of attention heads.
        num_blocks : int
            The number of transformer blocks.
        num_atom_enc_heads : int
            The number of attention heads in the atom encoder.
        num_atom_enc_blocks : int
            The number of blocks in the atom encoder.
        num_atom_dec_heads : int
            The number of attention heads in the atom decoder.
        num_atom_dec_blocks : int
            The number of blocks in the atom decoder.
        blocks_per_ckpt : int | None, optional
            The number of blocks per checkpoint for gradient checkpointing,
            by default None.
        """
        super().__init__()
        self.channel_a: int = channel_a
        self.channel_s: int = channel_s
        self.num_blocks: int = num_blocks
        self.num_heads: int = num_heads

        # Atom attention encoder
        self.atom_embedder = AtomEmbedder(channel_atom, channel_s)
        self.atom_encoder = AtomAttentionStack(
            channel_atom, num_atom_enc_heads, num_atom_enc_blocks
        )

        # Global transformer stack
        self.proj_q_to_a = nn.Sequential(
            LinearNoBias(channel_atom, channel_a, init="default"),
            nn.ReLU(),
        )
        self.proj_cond_to_a = nn.Sequential(
            LayerNorm(channel_s, create_offset=False),
            LinearNoBias(channel_s, channel_a, init="final"),
        )
        self.diffusion_transformer = DiffusionTransformerStack(
            channel_a=channel_a,
            channel_cond=channel_s,
            num_blocks=num_blocks,
            num_heads=num_heads,
            blocks_per_ckpt=blocks_per_ckpt,
        )

        # Atom attention decoder
        self.proj_a_to_q = nn.Sequential(
            LayerNorm(channel_a, create_offset=False),
            LinearNoBias(channel_a, channel_atom, init="default"),
        )
        self.atom_decoder = AtomAttentionStack(
            channel_atom, num_atom_dec_heads, num_atom_dec_blocks
        )

        # Residue to atom projection
        self.proj_r_update = nn.Sequential(
            LayerNorm(channel_atom, create_offset=False),
            LinearNoBias(channel_atom, 3, precision=32, init="final"),
        )

    # === Main forward function for training === #
    def forward(
        self,
        batch: dict[str, torch.Tensor],
        r_noisy: torch.Tensor,
        single_cond: torch.Tensor,
        pair_bias: torch.Tensor,
    ) -> torch.Tensor:
        """Single diffusion step forward pass

        Parameters
        ----------
        batch: dict[str, torch.Tensor]
            The input batch.
        r_noisy: torch.Tensor
            The noisy atom positions, shape [B, N, L, 14, 3],
            where N is number of diffusion samples.
        single_cond: torch.Tensor
            The single conditioning, shape [B, N, L, c_s].
        pair_bias: torch.Tensor
            The pair bias for the diffusion transformer, shape [B, L, L, Nblock, Nhead].

        Returns
        -------
        r_update : torch.Tensor
            The scaled updated atom positions, shape [B, N, L, 14, 3].
        """
        B, N, L, _, _ = r_noisy.shape

        # Atom attention encoder
        q, c = self.atom_embedder(batch, r_noisy, single_cond)  # [B, N, L, 14, c_atom]
        q = self.atom_encoder(batch, q, c)  # [B, N, L, 14, c_atom]

        # Pool atom representations to residue level
        mask = batch["atom14_mask"].view(B, 1, L, 14)  # [B, 1, L, 14]
        num_atoms = mask.sum(dim=-1).clamp(min=1)  # [B, 1, L]
        q_to_a = self.proj_q_to_a(q)  # [B, N, L, 14, c_a]
        a = (q_to_a * mask[..., None]).sum(-2) / num_atoms[..., None]  # [B, N, L, c_a]

        # Add projected single conditioning
        a = a + self.proj_cond_to_a(single_cond)  # [B, N, L, c_a]

        # Global transformer stack
        mask = batch["seq_mask"].view(B, 1, 1, L)  # [B, 1, 1, L]
        a = self.diffusion_transformer(
            a,
            mask=mask,
            single_cond=single_cond,
            pair_bias=pair_bias,
        )  # [B, N, L, c_a]

        # Project back to atom space for decoder
        a_to_q = self.proj_a_to_q(a)  # [B, N, L, c_atom]
        q = q + a_to_q.unsqueeze(-2)  # [*, N, L, 14, c_atom]

        # Atom attention decoder
        q = self.atom_decoder(batch, q, c)  # [B, N, L, 14, c_atom]

        # Project to coordinate updates
        r_update = self.proj_r_update(q)  # [B, N, L, 14, 3]

        return r_update


class DiffusionHead(nn.Module):
    def __init__(
        self,
        channel_a: int = 768,
        channel_s: int = 384,
        channel_z: int = 128,
        channel_atom: int = 128,
        num_heads: int = 16,
        num_blocks: int = 16,
        num_atom_enc_heads: int = 4,
        num_atom_enc_blocks: int = 2,
        num_atom_dec_heads: int = 4,
        num_atom_dec_blocks: int = 2,
        # For training
        blocks_per_ckpt: int | None = None,
    ) -> None:
        """Initialize the diffusion module.

        Parameters
        ----------
        channel_a : int
            The single representation dimension.
        channel_s : int
            The single conditioning dimension.
        channel_z : int
            The pair conditioning dimension.
        channel_atom : int
            The atom representation dimension.
        num_heads : int
            The number of attention heads.
        num_blocks : int
            The number of transformer blocks.
        num_atom_enc_heads : int
            The number of attention heads in the atom encoder.
        num_atom_enc_blocks : int
            The number of blocks in the atom encoder.
        num_atom_dec_heads : int
            The number of attention heads in the atom decoder.
        num_atom_dec_blocks : int
            The number of blocks in the atom decoder.

        # For training
        blocks_per_ckpt : int | None, optional
            The number of blocks per checkpoint for gradient checkpointing,
            by default None.
        """
        super().__init__()
        # Diffision pre-conditioning
        self.sigma_data: float = SIGMA_DATA

        # Diffusion conditioning
        self.single_conditioning = SingleConditioning(channel_s, dim_fourier=256)
        self.pair_conditioning = PairConditioning(channel_z)

        # Diffusion module
        self.score_model = DiffusionModule(
            channel_a=channel_a,
            channel_s=channel_s,
            channel_atom=channel_atom,
            num_heads=num_heads,
            num_blocks=num_blocks,
            num_atom_enc_heads=num_atom_enc_heads,
            num_atom_enc_blocks=num_atom_enc_blocks,
            num_atom_dec_heads=num_atom_dec_heads,
            num_atom_dec_blocks=num_atom_dec_blocks,
            blocks_per_ckpt=blocks_per_ckpt,
        )

        self.is_compiled = False

    def do_compile(self, **kwargs):
        """Compile the trunk module."""
        self.is_compiled = True
        self.score_model: DiffusionModule = torch.compile(self.score_model, **kwargs)

    # ============================================================
    # EDM Pre-conditioning
    # ============================================================
    def c_skip(self, t_hat: _ScalarOrTensor) -> _ScalarOrTensor:
        return (self.sigma_data**2) / (t_hat**2 + self.sigma_data**2)

    def c_out(self, t_hat: _ScalarOrTensor) -> _ScalarOrTensor:
        _sqrt = lambda x: math.sqrt(x) if isinstance(x, float) else torch.sqrt(x)  # noqa
        return t_hat * self.sigma_data / _sqrt(self.sigma_data**2 + t_hat**2)  # type: ignore

    def c_in(self, t_hat: _ScalarOrTensor) -> _ScalarOrTensor:
        _sqrt = lambda x: math.sqrt(x) if isinstance(x, float) else torch.sqrt(x)  # noqa
        return 1 / _sqrt(t_hat**2 + self.sigma_data**2)  # type: ignore

    def c_noise(self, t_hat: _ScalarOrTensor) -> _ScalarOrTensor:
        _log = lambda x: math.log(x) if isinstance(x, float) else torch.log(x)  # noqa
        _clip = lambda x, v: max(x, v) if isinstance(x, float) else x.clamp(v)  # noqa
        return _log(_clip(t_hat / self.sigma_data, 1e-20)) * 0.25  # type: ignore

    # ============================================================
    # For inference
    # ============================================================
    def sample(
        self,
        batch: dict[str, torch.Tensor],
        s: torch.Tensor,
        z: torch.Tensor,
        config: SamplingConfig,
        seed: int | None = None,
        use_compiled_model: bool = True,
    ) -> torch.Tensor:
        """Sample structures via diffusion sampling.
        See Section 3.7: Algorithm 18 of AlphaFold3 paper.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            The input batch.
        s : torch.Tensor
            The single representation, shape [B, L, c_s].
        z : torch.Tensor
            The pair representation, shape [B, L, L, c_z].
        config : SamplingConfig
            The sampling configuration.
        seed : int | None, optional
            The random seed for diffusion sampling, by default None.

        Returns
        -------
        coords : torch.Tensor
            The sampled atom coordinates, shape [B, N, L, 14, 3].
        """
        # Set random seed
        rng = torch.Generator(device=s.device)
        if seed is not None:
            rng.manual_seed(seed)

        # Get noise schedule
        sigmas: list[float] = config.sigmas
        gammas: list[float] = config.gammas

        # sample prior
        mask = batch["atom_mask"].unsqueeze(1)  # (B, L, 14)
        B, _, L, _ = mask.shape
        N = config.num_samples
        device = s.device

        sigma_0 = sigmas[0]
        x = sigma_0 * torch.randn(
            size=(B, N, L, 14, 3), generator=rng, device=device, dtype=torch.float32
        )
        x.masked_fill_(~mask[..., None], 0.0)  # apply atom mask

        # Compute time-independent variables
        pair_bias = self.pair_conditioning(batch, z)
        del z

        def run_step(x_noisy: torch.Tensor, t_hat: float) -> torch.Tensor:
            c_noise = torch.tensor(self.c_noise(t_hat), device=device)
            single_cond = self.single_conditioning(batch, s, c_noise)
            return self.inference_step(
                batch,
                x_noisy,
                t_hat,
                single_cond,
                pair_bias,
                chunk_size=config.chunk_size,
                use_compiled_model=use_compiled_model,
            )

        # Gradually denoise
        for step in range(1, config.num_steps + 1):
            x = center_random_augmentation(x, mask, rng=rng)

            sigma_tm, sigma_t = sigmas[step - 1], sigmas[step]
            gamma = gammas[step]
            t_hat: float = sigma_tm * (1 + gamma)

            # Add noise
            noise_var: float = config.noise_scale**2 * (t_hat**2 - sigma_tm**2)
            eps = math.sqrt(noise_var) * torch.randn_like(x, generator=rng)
            eps.masked_fill_(~mask[..., None], 0.0)  # apply atom mask
            x_noisy = x + eps

            # Denoise
            x_denoised = run_step(x_noisy, t_hat)
            delta = (x_noisy - x_denoised) / t_hat
            dt = sigma_t - t_hat
            x = x_noisy + config.step_scale * dt * delta

        return x

    def inference_step(
        self,
        batch: dict[str, torch.Tensor],
        x_noisy: torch.Tensor,
        t_hat: float,
        single_cond: torch.Tensor,
        pair_bias: torch.Tensor,
        chunk_size: int | None = None,
        use_compiled_model: bool = True,
    ) -> torch.Tensor:
        """Forward pass through the score model.
        See Section 3.7: Diffusion Module, Algorithm 20 of AlphaFold3 paper.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            The input batch.
        x_noisy : torch.Tensor
            Noisy atom coordinates. Shape (B, N, L, 14, 3).
        t_hat : float
            Diffusion noise level (or sigmas of EDM).
        single_cond : torch.Tensor
            The single conditioning, shape [B, N, L, c_s].
        pair_bias : torch.Tensor
            The pair bias for the token transformer, shape [B, Nblock, H, L, L].

        Returns
        -------
        x_out : torch.Tensor
            Denoised atom coordinates. Shape (B, N, L, 14, 3).
        """
        score_model: DiffusionModule = self.score_model
        if self.is_compiled and not use_compiled_model:
            score_model = score_model._orig_mod

        c_in, c_skip, c_out = self.c_in(t_hat), self.c_skip(t_hat), self.c_out(t_hat)

        # Input conditioning
        r_noisy = c_in * x_noisy

        if chunk_size is None:
            r_update = score_model(batch, r_noisy, single_cond, pair_bias)
        else:
            r_update = torch.zeros_like(r_noisy)
            for st in range(0, r_noisy.shape[1], chunk_size):
                end = min(st + chunk_size, r_noisy.shape[1])
                _r_noisy = r_noisy[:, st:end]
                _single_cond = single_cond[:, st:end]
                r_update[:, st:end] = score_model(
                    batch, _r_noisy, _single_cond, pair_bias
                )

        # Output conditioning
        x_out = c_skip * x_noisy + c_out * r_update

        return x_out

    # ============================================================
    # For training
    # ============================================================
    def forward_train(
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

        # Sample noise levels
        nt = torch.randn((B, N), dtype=torch.float32, device=s.device)
        t_hat = self.sigma_data * torch.exp(-1.2 + 1.5 * nt)

        # Repeat label coords with random augmentation for each diffusion sample
        with torch.autocast(s.device.type, enabled=False):
            label_coords = label["coords"]  # [B, L, 14, 3]
            x_gt = expand_dim(label_coords, N, dim=1)  # [B, N, L, 14, 3]
            resolved_mask = label["resolved_mask"]  # [B, L, 14]
            mask = expand_dim(resolved_mask, N, dim=1)  # [B, N, L, 14]
            x_gt = center_random_augmentation(x_gt, mask)  # [B, N, L, 14, 3]

            # Add noise
            noise = torch.randn_like(x_gt)  # [B, N, L, 14, 3]
            x_noisy = x_gt + t_hat.view(B, N, 1, 1, 1) * noise

        # Forward pass through the score model
        x_out = self._forward_train(batch, x_noisy, t_hat, s, z)  # [B, N, L, 14, 3]

        loss_weights = 1 / self.c_out(t_hat) ** 2  # [B, N]

        return {
            "x_noisy": x_noisy,
            "x_out": x_out,
            "x_gt": x_gt,
            "resolved_mask": resolved_mask,
            "loss_weights": loss_weights,
        }

    def _forward_train(
        self,
        batch: dict[str, torch.Tensor],
        x_noisy: torch.Tensor,
        t_hat: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the score model.
        See Section 3.7: Diffusion Module, Algorithm 20 of AlphaFold3 paper.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            The input batch.
        x_noisy : torch.Tensor
            Noisy atom coordinates. Shape (B, N, L, 14, 3).
        t_hat : torch.Tensor
            Diffusion noise level (or sigmas of EDM). Shape (B, N).
        s : torch.Tensor
            Trunk sequence embeddings. Shape (B, L, c_s).
        z : torch.Tensor
            Trunk pairwise embeddings. Shape (B, L, L, c_z).

        Returns
        -------
        x_0_hat : torch.Tensor
            Denoised atom coordinates. Shape (B, N, La, 3).
        """
        t_hat_expanded = t_hat[:, :, None, None, None]  # [B, N, 1, 1, 1]
        c_in = self.c_in(t_hat_expanded)  # [B, N, 1, 1, 1]
        c_skip = self.c_skip(t_hat_expanded)  # [B, N, 1, 1, 1]
        c_out = self.c_out(t_hat_expanded)  # [B, N, 1, 1, 1]

        # DiT Conditioning
        c_noise = self.c_noise(t_hat)  # [B, N, 1, 1, 1]
        single_cond = self.single_conditioning(batch, s, c_noise)  # [B, N, L, c_s]
        pair_bias = self.pair_conditioning(batch, z)  # [B, L, L, Nblock, Nhead]

        # Input EDM conditioning
        r_noisy = c_in * x_noisy  # [B, N, L, 14, 3]
        # Forward pass through the score model
        r_update = self.score_model(batch, r_noisy, single_cond, pair_bias)
        # Output EDM conditioning
        x_out = c_skip * x_noisy + c_out * r_update
        return x_out
