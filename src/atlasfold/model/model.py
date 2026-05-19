"""Folding Trunk."""

import dataclasses
from typing import Any

import einops
import torch

from atlasfold.common import confidence_utils
from atlasfold.model.network.confidence_head import ConfidenceHead
from atlasfold.model.network.diffusion_head import DiffusionHead, SamplingConfig
from atlasfold.model.network.distogram_head import DistogramHead
from atlasfold.model.network.input_embedder import InputEmbedder
from atlasfold.model.network.primitives import LayerNorm, LinearNoBias
from atlasfold.model.network.trunk import TriangularUpdateTrunk
from atlaslm.model import AtlasLM, PLMOutput
from atlaslm.pretrained import load_model


@dataclasses.dataclass(kw_only=True)
class TrunkConfig:
    channel_s: int = 768
    channel_z: int = 128
    num_head_attn: int = 24
    num_head_tri_attn: int = 4
    dropout_s: float = 0.15
    dropout_z: float = 0.25
    num_blocks: int = 48
    blocks_per_ckpt: int | None = None


@dataclasses.dataclass(kw_only=True)
class DiffusionHeadConfig:
    channel_a: int = 384
    channel_atom: int = 96
    num_heads: int = 8
    num_blocks: int = 16
    num_atom_enc_heads: int = 3
    num_atom_enc_blocks: int = 2
    num_atom_dec_heads: int = 3
    num_atom_dec_blocks: int = 2
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
    """Configuration for the AtlasFold model.

    Parameters
    ----------
    lm_name : str
        The name of the pretrained language model to use for feature extraction.
    channel_s : int
        The token single embedding size.
    channel_z : int
        The token pairwise embedding size.
    num_heads_attn : int
        The number of attention heads in the attention blocks.
    num_heads_tri_attn : int
        The number of attention heads in the triangular attention blocks.
    num_blocks : int
        The number of blocks in the trunk.
    dropout : float
        The dropout rate for the trunk.
    blocks_per_ckpt : int | None
        The number of blocks to include in each checkpoint for gradient checkpointing.
    """

    name: str = "atlasfold-base"
    lm_name: str = "atlaslm-600m"
    lm_path: str | None = None
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
        self.channel_s: int = cfg.trunk.channel_s
        self.channel_z: int = cfg.trunk.channel_z

        # Diffusion head dimensions
        self.channel_a: int = cfg.diffusion_head.channel_a
        self.channel_atom: int = cfg.diffusion_head.channel_atom

        # === language model === #
        lm: AtlasLM = load_model(cfg.lm_name, cfg.lm_path).bfloat16().eval()
        lm.requires_grad_(False)
        self.lm: AtlasLM = lm

        # === Input projections === #
        self.input_embedder: InputEmbedder = InputEmbedder(
            channel_s=self.channel_s,
            channel_lm=self.lm.d_model,
            channel_z=self.channel_z,
            num_layers_lm=self.lm.n_layers,
            num_heads_lm=self.lm.n_heads,
        )

        # === Trunk body === #
        self.recycle_s = torch.nn.Sequential(
            LayerNorm(self.channel_s),
            LinearNoBias(self.channel_s, self.channel_s, init="final"),
        )
        self.recycle_z = torch.nn.Sequential(
            LayerNorm(self.channel_z),
            LinearNoBias(self.channel_z, self.channel_z, init="final"),
        )
        self.trunk: TriangularUpdateTrunk = TriangularUpdateTrunk(
            **dataclasses.asdict(cfg.trunk)
        )

        # === Distogram head === #
        self.distogram_head: DistogramHead = DistogramHead(
            channel_z=self.channel_z, **dataclasses.asdict(cfg.distogram_head)
        )

        # === Structure prediction heads === #
        self.diffusion_head: DiffusionHead = DiffusionHead(
            channel_s=self.channel_s,
            channel_z=self.channel_z,
            **dataclasses.asdict(cfg.diffusion_head),
        )

        # === Confidence prediction head === #
        self.confidence_head: ConfidenceHead = ConfidenceHead(
            channel_s=self.channel_s,
            channel_z=self.channel_z,
            **dataclasses.asdict(cfg.confidence_head),
        )

        # === Other settings === #
        self.is_compiled = False
        self.use_compiled_models = True  # Only effective when `is_compiled` is True.
        self.use_cuequiv_kernels = False

    @property
    def device(self) -> torch.device:
        """Return the device of the model parameters."""
        return next(self.parameters()).device

    def compile_submodules(self, **kwargs) -> None:
        """Compile the submodules of the model."""
        self.lm = torch.compile(self.lm, **kwargs)
        self.input_embedder = torch.compile(self.input_embedder, **kwargs)
        self.trunk = torch.compile(self.trunk, **kwargs)
        self.diffusion_head = self.diffusion_head.do_compile(**kwargs)
        self.confidence_head = torch.compile(self.confidence_head, **kwargs)

    def set_forward_flags(
        self,
        use_cuequiv_kernels: bool | None = None,
        use_compiled_models: bool | None = None,
    ) -> None:
        """Set the flags for the forward pass."""
        if use_cuequiv_kernels is not None:
            self.use_cuequiv_kernels = use_cuequiv_kernels
        if use_compiled_models is not None:
            self.use_compiled_models = use_compiled_models

    # ==================================================
    # Forward pass
    # ==================================================
    @torch.inference_mode()
    def inference(
        self,
        batch: dict[str, torch.Tensor],
        num_recycles: int,
        sampling_config: SamplingConfig,
        diffusion_seed: int | None = None,
        compute_pae: bool = True,
        return_representations: bool = False,
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
        diffusion_seed : int | None
            The random seed for the diffusion head.
        compute_pae : bool
            Whether to compute the predicted aligned error (PAE) and
            predicted TM-score (pTM).
        return_representations : bool
            Whether to return the single and pair representations.

        Returns
        -------
        out: dict[str, torch.Tensor]
            The output dictionary containing the predicted coordinates and optionally
            the distogram logits and representations.
        """
        out: dict[str, torch.Tensor] = {}

        # Run language model
        lm_out, hs, attns = self.run_lm(batch)
        if return_representations:
            out["lm.embeddings"] = lm_out.embeddings
            out["lm.sequence_logits"] = lm_out.sequence_logits
            out["lm.hidden_states"] = torch.stack(
                lm_out.hidden_states, dim=2
            )  # [B, L, n_layers, c_lm]
            out["lm.attentions"] = torch.stack(
                lm_out.attentions, dim=3
            )  # [B, L, L, n_layers, n_heads]
        del lm_out

        # Prepare folding trunk inputs
        s, z = self.prepare_trunk_inputs(batch, hs, attns)
        del hs, attns

        # Run trunk with recycling
        if num_recycles >= 0:
            s, z = self.run_trunk(batch, s, z, num_recycles)

        if return_representations:
            out["trunk.s"] = s
            out["trunk.z"] = z

        # Run distogram head
        distogram_out = self.distogram_head(z)
        out["distogram.logits"] = distogram_out["logits"]
        out["distogram.boundaries"] = distogram_out["boundaries"]

        # Run diffusion heads
        sample_coords = self.diffusion_head.sample(
            batch,
            s,
            z,
            sampling_config,
            diffusion_seed,
            use_compiled_model=self.use_compiled_models,
        )
        out["sample_coords"] = sample_coords

        # Run confidence head
        confidence_out = self.run_confidence_head(
            batch, s, z, sample_coords, compute_pae=compute_pae
        )
        del s, z

        # Compute confidence metrics
        mask = batch["seq_mask"]
        if "plddt" in confidence_out:
            out["plddt"] = confidence_utils.compute_plddt(
                **confidence_out["plddt"], mask=mask
            )
        if "pae" in confidence_out:
            out["pae"] = confidence_utils.compute_pae(**confidence_out["pae"], mask=mask)
            out["ptm"] = confidence_utils.compute_ptm(**confidence_out["pae"], mask=mask)

        return out

    # ==================================================
    # Forward pass components
    # ==================================================
    def get_model(self, model_name: str) -> torch.nn.Module:
        """Return the compiled or original version of the given model."""
        model = getattr(self, model_name)
        if self.is_compiled and not self.use_compiled_models:
            return model._orig_mod
        return model

    def run_lm(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[PLMOutput, list[torch.Tensor], list[torch.Tensor]]:
        # Prepare the inputs for the language model
        input_ids = batch["lm.input_ids"]  # [B, L+2]
        pos_id = batch["lm.pos_id"]  # [B, L+2]
        seq_id = batch["lm.seq_id"]  # [B, L+2]
        # Mask out the padding positions into PAD
        input_ids = input_ids.masked_fill(seq_id == 0, self.lm.alphabet.pad_idx)

        # Run the language model
        lm = self.get_model("lm")
        lm_out: PLMOutput = lm(
            input_ids,
            seq_id=seq_id,
            pos_id=pos_id,
            return_logits=True,
            return_hidden_states=True,
            return_attentions=True,
        )

        # hidden states: list of [B, L + 2, c_lm]
        # attentions: list of [B, n_heads, L + 2, L + 2]
        hidden_states: list[torch.Tensor] = lm_out.hidden_states
        attentions: list[torch.Tensor] = [
            einops.rearrange(a, "b h l1 l2 -> b l1 l2 h") for a in lm_out.attentions
        ]

        # For training, we use additional field `lm.cropped_pos_id`.
        cropped_pos_id = batch.get("lm.cropped_pos_id", None)  # [B, L_crop]
        if cropped_pos_id is not None:
            # Crop the hidden states and attention maps to the original sequence length L
            b_idx = torch.arange(input_ids.shape[0], device=input_ids.device)

            # [B, L_crop, c_lm] -> [B, L_crop, c_lm] -> [B, L_crop, n_layers, c_lm]
            b_idx_h = b_idx[:, None]
            hs = [h[b_idx_h, cropped_pos_id, :] for h in hidden_states]

            # [B, H, L', L'] -> [B, L, L, H] -> [B, L, L, n_layers, H]
            b_idx_a = b_idx[:, None, None]
            row_idx = cropped_pos_id[:, :, None]
            col_idx = cropped_pos_id[:, None, :]
            attns = [a[b_idx_a, row_idx, col_idx, :] for a in attentions]
        else:
            # Remove the special tokens (BOS and EOS)
            hs = [h[:, 1:-1, :] for h in hidden_states]
            attns = [a[:, 1:-1, 1:-1, :] for a in attentions]
        return lm_out, hs, attns

    def prepare_trunk_inputs(
        self,
        batch: dict[str, torch.Tensor],
        hs: list[torch.Tensor],
        attns: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare the inputs for the trunk.

        Parameters
        ----------
        batch: dict[str, torch.Tensor]
            The input batch.
        hs: list[torch.Tensor]
            The list of hidden states from the language model,
            each of shape (B, L, c_lm).
        attns: list[torch.Tensor]
            The list of attention maps from the language model,
            each of shape (B, L, L, n_heads).

        Returns
        -------
        s: torch.Tensor
            The prepared single representation tensor of shape (B, L, c_s).
        z: torch.Tensor
            The prepared pair representation tensor of shape (B, L, L, c_z).
        """
        input_embedder: InputEmbedder = self.get_model("input_embedder")
        s, z = input_embedder(batch, hs, attns)
        return s, z

    def run_trunk(
        self,
        batch: dict[str, torch.Tensor],
        s_init: torch.Tensor,
        z_init: torch.Tensor,
        num_recycles: int,
        train: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass.

        Parameters
        ----------
        batch: dict[str, torch.Tensor]
            The input batch.
        s_init : torch.Tensor
            The initial single representation tensor of shape (B, L, c_s).
        z_init : torch.Tensor
            The initial pair representation tensor of shape (B, L, L, c_z).
        num_recycles : int
            The number of recycling steps.

        Returns
        -------
        s: torch.Tensor
            The updated tensor of shape (B, L, c_s).
        z: torch.Tensor
            The updated tensor of shape (B, L, L, c_z).
        """
        trunk: TriangularUpdateTrunk = self.get_model("trunk")

        # Main trunk iteration with recycling
        s_prev = torch.zeros_like(s_init)  # [B, L, c_s]
        z_prev = torch.zeros_like(z_init)  # [B, L, L, c_z]
        mask = batch["seq_mask"]  # [B, L]
        for i in range(0, num_recycles + 1):
            enable_grad = train and i == num_recycles
            with torch.set_grad_enabled(enable_grad):
                if enable_grad and torch.is_autocast_enabled():
                    torch.clear_autocast_cache()
                # Recycling
                s = s_init + self.recycle_s(s_prev)
                z = z_init + self.recycle_z(z_prev)
                # Run trunk
                s, z = trunk(s, z, mask, self.use_cuequiv_kernels)
                s_prev, z_prev = s, z
        return s, z

    def run_confidence_head(
        self,
        batch: dict[str, torch.Tensor],
        s: torch.Tensor,
        z: torch.Tensor,
        x_pred: torch.Tensor,
        compute_pae: bool,
    ) -> dict[str, dict[str, torch.Tensor]]:
        confidence_head: ConfidenceHead = self.get_model("confidence_head")
        confidence_out = confidence_head(
            batch, s, z, x_pred, compute_pae, self.use_cuequiv_kernels
        )
        return confidence_out

    # ==================================================
    # Model training
    # ==================================================
    def get_module_groups(self) -> dict[str, list[torch.nn.Module]]:
        return {
            "lm": [self.lm],
            "input_embedder": [self.input_embedder],
            "trunk": [self.recycle_s, self.recycle_z, self.trunk],
            "distogram": [self.distogram_head],
            "diffusion": [self.diffusion_head],
            "confidence": [self.confidence_head],
            "pae_head": [self.confidence_head.pae_head],
        }

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

        # Run language model
        with torch.no_grad():
            _, hs, attns = self.run_lm(batch)

        # Prepare folding trunk inputs
        s_init, z_init = self.prepare_trunk_inputs(batch, hs, attns)
        del hs, attns

        # Run trunk with recycling
        assert num_recycles >= 0, "num_recycles must be non-negative for training"
        s, z = self.run_trunk(batch, s_init, z_init, num_recycles, train=train_trunk)

        # Return distogram logits
        if train_trunk:
            distogram_out = self.distogram_head(z)
            distogram_aug_out = self.distogram_head(z_init)
            out["distogram"] = distogram_out
            out["distogram_aug"] = distogram_aug_out

        # Return diffusion head outputs
        if train_diffusion_head:
            # Select the diffusion conditioning.
            # trunk : init : zero = 6 : 2 : 2
            p = torch.rand(bs, device=s.device)
            use_trunk = p < 0.6
            use_init = (p >= 0.6) & (p < 0.8)
            _s = use_trunk.view(bs, 1, 1) * s + use_init.view(bs, 1, 1) * s_init
            _z = use_trunk.view(bs, 1, 1, 1) * z + use_init.view(bs, 1, 1, 1) * z_init
            diffusion_out = self.diffusion_head.forward_train(
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
                sample_coords = self.diffusion_head.sample(batch, _s, _z, sampling_config)

            # Select the confidence head conditioning.
            # trunk : init : zero = 6 : 2 : 2
            p = torch.rand(bs, device=s.device)
            use_trunk = p < 0.6
            use_init = (p >= 0.6) & (p < 0.8)
            _z = use_trunk.view(bs, 1, 1, 1) * z + use_init.view(bs, 1, 1, 1) * z_init

            confidence_out = self.run_confidence_head(
                batch, _s, _z, sample_coords, compute_pae=train_pae_head
            )
            confidence_out["mini_rollout"] = {"sample_coords": sample_coords}
            out["confidence"] = confidence_out

        return out
