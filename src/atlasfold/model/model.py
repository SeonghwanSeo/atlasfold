"""Folding Trunk."""

import dataclasses

import torch

from atlasfold.model.network.confidence_head import ConfidenceHead
from atlasfold.model.network.diffusion_head import DiffusionHead, SamplingConfig
from atlasfold.model.network.distogram_head import DistogramHead
from atlasfold.model.network.input_embedder import LMInputEmbedder
from atlasfold.model.network.primitives import LayerNorm, LinearNoBias
from atlasfold.model.network.trunk import LMModule, TriangularUpdateTrunk
from atlasfold.model.utils import confidence_metrics
from atlasfold.utils import torch_utils
from atlaslm.model import AtlasLM
from atlaslm.pretrained import load_model


@dataclasses.dataclass(kw_only=True)
class TrunkConfig:
    dropout_z: float = 0.25
    num_lm_blocks: int = 4
    num_blocks: int = 48
    num_heads: int = 12
    blocks_per_ckpt: int | None = None


@dataclasses.dataclass(kw_only=True)
class DiffusionHeadConfig:
    channel_a: int = 768
    channel_atom: int = 96
    num_heads: int = 12
    num_blocks: int = 16
    blocks_per_ckpt: int | None = None


@dataclasses.dataclass(kw_only=True)
class DistogramHeadConfig:
    num_bins: int = 64
    min_dist: float = 2.0
    max_dist: float = 22.0


@dataclasses.dataclass(kw_only=True)
class ConfidenceHeadConfig:
    channel_a: int = 384
    num_heads_attn: int = 12
    dropout_s: float = 0.15
    dropout_z: float = 0.25
    num_blocks: int = 4
    num_pae_blocks: int = 2
    num_bins: int = 39
    min_dist: float = 3.25
    max_dist: float = 50.75
    max_pae_dist: float = 32.0
    num_pae_bins: int = 64
    num_plddt_bins: int = 50


@dataclasses.dataclass(kw_only=True)
class AtlasFoldConfig:
    name: str = "atlasfold-base"
    lm_name: str = "atlaslm-3b"
    lm_path: str | None = None

    channel_s: int = 768
    channel_z: int = 192
    trunk: TrunkConfig = dataclasses.field(default_factory=TrunkConfig)
    distogram_head: DistogramHeadConfig = dataclasses.field(
        default_factory=DistogramHeadConfig
    )
    diffusion_head: DiffusionHeadConfig = dataclasses.field(
        default_factory=DiffusionHeadConfig
    )
    confidence_head: ConfidenceHeadConfig = dataclasses.field(
        default_factory=ConfidenceHeadConfig
    )


class AtlasFold(torch.nn.Module):
    def __init__(self, cfg: AtlasFoldConfig):
        """Initialize the AtlasFold model."""
        super().__init__()
        self.cfg = cfg
        # Trunk dimensions
        self.channel_s: int = cfg.channel_s
        self.channel_z: int = cfg.channel_z

        # Diffusion head dimensions
        self.channel_a: int = cfg.diffusion_head.channel_a
        self.channel_atom: int = cfg.diffusion_head.channel_atom

        # === Language model === #
        lm: AtlasLM = load_model(cfg.lm_name, dtype=torch.bfloat16)
        # Freeze LM parameters
        lm.requires_grad_(False)
        self.lm_embedder = LMInputEmbedder(
            lm,
            channel_s=self.channel_s,
            channel_z=self.channel_z,
        )

        # === Trunk body === #
        self.recycle_z = torch.nn.Sequential(
            LayerNorm(self.channel_z),
            LinearNoBias(self.channel_z, self.channel_z, init="final"),
        )
        self.lm_stack = LMModule(
            channel_s=self.channel_s,
            channel_z=self.channel_z,
            dropout_z=cfg.trunk.dropout_z,
            num_blocks=cfg.trunk.num_lm_blocks,
            blocks_per_ckpt=cfg.trunk.blocks_per_ckpt,
        )
        self.main_stack = TriangularUpdateTrunk(
            channel_z=self.channel_z,
            dropout_z=cfg.trunk.dropout_z,
            num_blocks=cfg.trunk.num_blocks,
            blocks_per_ckpt=cfg.trunk.blocks_per_ckpt,
        )

        # === Distogram head === #
        self.distogram_head = DistogramHead(
            channel_z=self.channel_z,
            **dataclasses.asdict(cfg.distogram_head),
        )

        # === Structure prediction heads === #
        self.diffusion_head = DiffusionHead(
            channel_s=self.channel_s,
            channel_z=self.channel_z,
            **dataclasses.asdict(cfg.diffusion_head),
        )

        # === Confidence prediction head === #
        self.confidence_head = ConfidenceHead(
            channel_s=self.channel_s,
            channel_z=self.channel_z,
            **dataclasses.asdict(cfg.confidence_head),
        )

        # Kernel option
        self.use_cuequiv_kernels: bool = False

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def lm(self) -> AtlasLM:
        return self.lm_embedder.lm

    def set_forward_flags(
        self,
        use_cuequiv_kernels: bool | None = None,
    ) -> None:
        """Set the flags for the forward pass."""
        if use_cuequiv_kernels is not None:
            self.use_cuequiv_kernels = use_cuequiv_kernels

    # ==================================================
    # Inference
    # ==================================================
    @torch.inference_mode()
    def inference(
        self,
        batch: dict[str, torch.Tensor],
        num_recycles: int,
        mode: str,
        sampling_config: SamplingConfig,
        compute_pae: bool = True,
        return_representations: bool = False,
        # Advanced options
        mlm_prob: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Perform the forward pass.

        Parameters
        ----------
        batch: dict[str, torch.Tensor]
            The input batch.
            Required fields:
            - "lm.input_ids": Tensor of shape (B, L+2) containing the input
                token IDs for the language model, including special tokens.
            - "lm.pos_id": Tensor of shape (B, L+2) containing the positional
                IDs for the language model.
            - "lm.seq_id": Tensor of shape (B, L+2) containing the sequence
                IDs for the language model, where padding positions are marked
                with 0.
            - "lm.mlm_mask": Tensor of shape (B, L+2) containing the sequence
                mask for language model feature extraction, where masked positions
                are marked with True.
            - "aatype": Tensor of shape (B, L, 21) containing the one-hot encoded
                amino acid types.
            - "aatype_int": Tensor of shape (B, L) containing the amino acid type.
            - "res_idx": Tensor of shape (B, L) containing the residue indices.
            - "seq_mask": Tensor of shape (B, L) containing boolean mask for valid
                sequence positions (True for valid positions, False for padding).
            - "atom14_mask": Tensor of shape (B, L, 14) containing boolean mask for
                valid atom positions.
            - "atom37_mask": Tensor of shape (B, L, 37) containing boolean mask for
                valid atom positions.
            - "pseudo_beta": Tensor of shape (B, L, 3) containing the pseudo-beta
                coordinates for each residue (C-beta, or C-alpha for glycine).

        num_recycles : int
            The number of recycling steps.
        sampling_config : SamplingConfig
            The configuration for the diffusion sampling process.
        compute_pae : bool
            Whether to compute the predicted aligned error (PAE) and
            predicted TM-score (pTM).
        return_representations : bool
            Whether to return the single and pair representations.

        # Advanced options
        mlm_prob : float | None
            The probability of masking input tokens for the language model.
            If > 0, a random MLM mask will be sampled for each forward pass.

        Returns
        -------
        out: dict[str, torch.Tensor]
            The output dictionary containing the predicted coordinates and optionally
            the distogram logits and representations.
        """
        is_batched = batch["aatype"].dim() == 3
        if not is_batched:
            # Add batch dimension if the input is not already batched
            batch = {k: v.unsqueeze(0) for k, v in batch.items()}

        out: dict[str, torch.Tensor] = {}
        if mode not in ["flash", "base", "full"]:
            raise ValueError(
                f"Invalid mode: {mode}. Must be one of 'flash', 'base', or 'full'."
            )

        # Run trunk
        s, z = self.run_trunk(batch, num_recycles, mode, mlm_prob)
        if return_representations:
            out["trunk.s"] = s
            out["trunk.z"] = z

        # Run distogram head
        distogram_out = self.run_distogram_head(z)
        out["distogram.logits"] = distogram_out["logits"]
        out["distogram.boundaries"] = distogram_out["boundaries"]

        # Run diffusion heads
        sample_coords = self.run_diffusion_head(batch, s, z, sampling_config)
        out["sample_coords"] = sample_coords

        # Run confidence head
        confidence_out = self.run_confidence_head(batch, s, z, sample_coords, compute_pae)
        del s, z

        # Compute confidence metrics
        mask = batch["seq_mask"]
        out["plddt"] = confidence_metrics.compute_plddt(
            **confidence_out["plddt"], mask=mask
        )
        if compute_pae:
            out["pae"] = confidence_metrics.compute_pae(
                **confidence_out["pae"], mask=mask
            )
            out["ptm"] = confidence_metrics.compute_ptm(
                **confidence_out["pae"], mask=mask
            )

        # Remove batch dimension if the input was not originally batched
        if not is_batched:
            for k, v in out.items():
                out[k] = v.squeeze(0)

        return out

    # ==================================================
    # Forward pass components
    # ==================================================
    def run_trunk(
        self,
        batch: dict[str, torch.Tensor],
        num_recycles: int,
        mode: str,
        mlm_prob: float | None = None,
        train: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the trunk."""
        if mode == "flash":
            if num_recycles != -1:
                raise ValueError("num_recycles must be -1 for flash mode.")
            return self.run_trunk_flash(batch, mlm_prob, train)
        elif mode == "base":
            if num_recycles < 0:
                raise ValueError("num_recycles must be non-negative for base mode.")
            return self.run_trunk_base(batch, num_recycles, mlm_prob, train)
        else:  # mode == "full"
            if num_recycles <= 0:
                raise ValueError("num_recycles must be positive for full mode.")
            return self.run_trunk_full(batch, num_recycles, mlm_prob, train)

    def run_trunk_flash(
        self,
        batch: dict[str, torch.Tensor],
        mlm_prob: float | None = None,
        train: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the trunk without triangular updates for fast inference."""
        # Sample a single MLM mask
        mlm_prob = mlm_prob if mlm_prob is not None else 0.0
        mlm_mask = self.sample_mlm_mask(batch, mlm_prob)

        # Extract LM features
        s_lm, z_lm = self.lm_embedder(batch, mlm_mask, train)
        return s_lm, z_lm

    def run_trunk_base(
        self,
        batch: dict[str, torch.Tensor],
        num_recycles: int,
        mlm_prob: float | None = None,
        train: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the trunk with shared LM features."""
        mlm_prob = mlm_prob if mlm_prob is not None else 0.0

        # Sample a single MLM mask for all recycling steps
        mlm_mask = self.sample_mlm_mask(batch, mlm_prob)

        # Extract LM features
        s_lm, z_lm = self.lm_embedder(batch, mlm_mask, train)

        # Run LM stack
        mask = batch["seq_mask"]
        z_init = self.lm_stack(s_lm, z_lm, mask, self.use_cuequiv_kernels)

        # Recycling iteration with shared LM features
        z_prev = torch.zeros_like(z_lm)  # [B, L, L, c_z]
        for i in range(0, num_recycles + 1):
            enable_grad = train and i == num_recycles
            with torch.set_grad_enabled(enable_grad):
                if enable_grad and torch.is_autocast_enabled():
                    torch.clear_autocast_cache()
                # Recycling embedding
                z = z_init + self.recycle_z(z_prev)
                # Run main stack
                z = self.main_stack(z, mask, self.use_cuequiv_kernels)
                z_prev = z
        return s_lm, z

    def run_trunk_full(
        self,
        batch: dict[str, torch.Tensor],
        num_recycles: int,
        mlm_prob: float | None = None,
        train: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the trunk with stochastic recycling."""
        mask = batch["seq_mask"]
        B, L = mask.shape
        device = mask.device
        dtype = torch_utils.get_context_dtype(device.type)

        # Set default MLM probability to 0.15 for full mode if not specified
        mlm_prob = mlm_prob if mlm_prob is not None else 0.15
        if mlm_prob <= 0.0:
            raise ValueError("mlm_prob must be greater than 0 for full mode.")

        # Recycling iteration with stochastic LM features
        z_prev = torch.zeros(B, L, L, self.channel_z, device=device, dtype=dtype)
        for i in range(0, num_recycles + 1):
            enable_grad = train and i == num_recycles
            with torch.set_grad_enabled(enable_grad):
                if enable_grad and torch.is_autocast_enabled():
                    torch.clear_autocast_cache()
                # For each recycle step, sample a MLM mask to extract new LM features.
                mlm_mask = self.sample_mlm_mask(batch, mlm_prob)
                # Extract LM features with different MLM masks for each recycle step
                s_lm, z_lm = self.lm_embedder(batch, mlm_mask, enable_grad)
                # Run LM module
                z = self.lm_stack(s_lm, z_lm, mask, self.use_cuequiv_kernels)
                # Recycling
                z += self.recycle_z(z_prev)
                # Run main trunk
                z = self.main_stack(z, mask, self.use_cuequiv_kernels)
                z_prev = z
        return s_lm, z

    def sample_mlm_mask(
        self,
        batch: dict[str, torch.Tensor],
        prob: float,
        synchronized: bool = True,
    ) -> torch.Tensor:
        """Sample a random MLM mask for the input batch."""
        input_ids = batch["lm.input_ids"]  # [B, L+2]
        B, L = input_ids.shape
        shape = (1, L) if synchronized else (B, L)

        if prob <= 0.0:
            return torch.zeros(shape, dtype=torch.bool, device=input_ids.device)
        else:
            return torch.rand(shape, device=input_ids.device) < prob

    def run_distogram_head(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the distogram head."""
        return self.distogram_head(z)

    def run_diffusion_head(
        self,
        batch: dict[str, torch.Tensor],
        s: torch.Tensor,
        z: torch.Tensor,
        sampling_config: SamplingConfig,
    ) -> torch.Tensor:
        """Run the diffusion head."""
        return self.diffusion_head.sample(batch, s, z, sampling_config)

    def run_confidence_head(
        self,
        batch: dict[str, torch.Tensor],
        s: torch.Tensor,
        z: torch.Tensor,
        sample_coords: torch.Tensor,
        compute_pae: bool = True,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Run the confidence head."""
        return self.confidence_head(
            batch, s, z, sample_coords, compute_pae, self.use_cuequiv_kernels
        )
