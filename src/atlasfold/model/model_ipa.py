"""AtlasFold monomer with an AlphaFold-style IPA regression decoder."""

from __future__ import annotations

import dataclasses
from functools import partial
from typing import Any

import torch

from atlasfold.model.model import DistogramHeadConfig, TrunkConfig
from atlasfold.model.network import distogram_head, trunk
from atlasfold.model.network.primitives import LayerNorm, Linear, LinearNoBias
from atlasfold.model.network.rel_pos_encoding import RelativePositionEncoding
from atlasfold.model.network.structure_module import StructureModule
from atlasfold.model.utils import confidence_metrics
from atlasfold.utils import torch_utils
from atlaslm.pretrained import get_model, load_model


@dataclasses.dataclass(kw_only=True)
class PositionRecyclingConfig:
    num_bins: int = 15
    min_bin: float = 3.25
    max_bin: float = 20.75


@dataclasses.dataclass(kw_only=True)
class StructureModuleConfig:
    num_layer: int = 8
    num_head: int = 12
    num_scalar_qk: int = 16
    num_scalar_v: int = 16
    num_point_qk: int = 4
    num_point_v: int = 8
    num_layer_in_transition: int = 3
    dropout: float = 0.1
    sidechain_channel: int = 128
    sidechain_num_layer: int = 2
    num_torsion: int = 7
    position_scale: float = 10.0


@dataclasses.dataclass(kw_only=True)
class ConfidenceHeadConfig:
    hidden_channel: int = 128
    num_plddt_bins: int = 50
    num_pae_bins: int = 64
    max_pae_error: float = 31.0


@torch.no_grad()
def get_distogram(
    x: torch.Tensor,
    cbeta_idx: torch.Tensor,
    num_bins: int,
    min_bin: float = 3.25,
    max_bin: float = 20.75,
) -> torch.Tensor:
    """Compute the pseudo-beta distogram used for position recycling."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        cbeta_idx = cbeta_idx.long()
        index = cbeta_idx[..., None, None].expand(*cbeta_idx.shape, 1, 3)
        pseudo_beta = torch.gather(x, dim=-2, index=index).squeeze(-2)
        squared_distance = (
            (pseudo_beta[..., :, None, :] - pseudo_beta[..., None, :, :])
            .square()
            .sum(dim=-1, keepdim=True)
        )

        lower = torch.linspace(
            min_bin,
            max_bin,
            num_bins,
            dtype=torch.float32,
            device=x.device,
        ).square()
        upper = torch.cat((lower[1:], lower.new_tensor([1e8])), dim=0)
        return ((squared_distance > lower) & (squared_distance < upper)).float()


@dataclasses.dataclass(kw_only=True)
class AtlasFoldIPAConfig:
    name: str = "atlasfold-ipa-base"
    lm_name: str = "atlaslm-3b"
    lm_path: str | None = None
    channel_s: int = 384
    channel_s_lm: int = 768
    channel_z: int = 128
    position_recycling: PositionRecyclingConfig = dataclasses.field(
        default_factory=PositionRecyclingConfig
    )
    trunk: TrunkConfig = dataclasses.field(default_factory=TrunkConfig)
    distogram_head: DistogramHeadConfig = dataclasses.field(
        default_factory=DistogramHeadConfig
    )
    structure_module: StructureModuleConfig = dataclasses.field(
        default_factory=StructureModuleConfig
    )
    confidence_head: ConfidenceHeadConfig = dataclasses.field(
        default_factory=ConfidenceHeadConfig
    )


class PredictedLDDTHead(torch.nn.Module):
    def __init__(self, channel_s: int, hidden_channel: int, num_bins: int) -> None:
        super().__init__()
        self.input_layer_norm = LayerNorm(channel_s)
        self.mlp = torch.nn.Sequential(
            Linear(channel_s, hidden_channel, init="relu"),
            torch.nn.ReLU(),
            Linear(hidden_channel, hidden_channel, init="relu"),
            torch.nn.ReLU(),
            Linear(hidden_channel, num_bins, init="final"),
        )
        self.num_bins = num_bins

    def forward(self, s: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.mlp(self.input_layer_norm(s))
        bin_centers = (
            torch.arange(self.num_bins, dtype=torch.float32, device=s.device) + 0.5
        ) / self.num_bins
        return {"logits": logits, "bin_centers": bin_centers}


class ExperimentallyResolvedHead(torch.nn.Module):
    def __init__(self, channel_s: int) -> None:
        super().__init__()
        self.linear = Linear(channel_s, 37, init="final")

    def forward(self, s: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": self.linear(s)}


class PredictedAlignedErrorHead(torch.nn.Module):
    def __init__(self, channel_z: int, num_bins: int, max_error: float) -> None:
        super().__init__()
        if num_bins < 3:
            raise ValueError("PAE requires at least three bins.")
        self.linear = Linear(channel_z, num_bins, init="final")
        self.num_bins = num_bins
        self.max_error = max_error

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.linear(z)
        step = self.max_error / (self.num_bins - 2)
        bin_centers = (
            torch.arange(self.num_bins, dtype=torch.float32, device=z.device) + 0.5
        ) * step
        return {
            "logits": logits,
            "bin_centers": bin_centers,
        }


class AtlasFold_IPA(torch.nn.Module):
    """AtlasFold monomer trunk with an IPA regression structure decoder."""

    def __init__(self, cfg: AtlasFoldIPAConfig, load_lm: bool = True) -> None:
        super().__init__()
        self.cfg: AtlasFoldIPAConfig = cfg
        # Trunk dimensions
        self.channel_s = cfg.channel_s
        self.channel_s_lm = cfg.channel_s_lm
        self.channel_z = cfg.channel_z

        # Relative position encoding
        self.rel_pos_encoding = RelativePositionEncoding(r_max=32, s_max=2)

        # Language model
        if load_lm:
            lm = load_model(cfg.lm_name, path=cfg.lm_path, dtype=torch.bfloat16)
        else:
            lm = get_model(cfg.lm_name)
        self.lm = lm
        self.lm.requires_grad_(False)
        self.alphabet = self.lm.alphabet

        # Representation initialization
        self.s_init = LinearNoBias(21, self.channel_s)
        self.z_init = LinearNoBias(21, 2 * self.channel_z)
        self.z_rel_pos = LinearNoBias(self.rel_pos_encoding.dim, self.channel_z)

        # Recycling embedding
        self.recycle_s = LayerNorm(self.channel_s)
        self.recycle_z = LayerNorm(self.channel_z)
        self.recycle_pos_cfg = {
            "num_bins": cfg.position_recycling.num_bins,
            "min_bin": cfg.position_recycling.min_bin,
            "max_bin": cfg.position_recycling.max_bin,
        }
        self.recycle_pos = Linear(cfg.position_recycling.num_bins, self.channel_z)

        # LM stack
        self.lm_layer_weights = torch.nn.Parameter(torch.zeros(self.lm.n_layers + 1))
        self.layernorm_lm_emb = LayerNorm(self.lm.d_model)
        self.lm_emb_to_s_lm = torch.nn.Sequential(
            LinearNoBias(self.lm.d_model, self.channel_s_lm, init="relu"),
            torch.nn.ReLU(),
            LinearNoBias(self.channel_s_lm, self.channel_s_lm, init="default"),
        )
        self.proj_lm_attn = torch.nn.ModuleList(
            [
                LinearNoBias(self.lm.n_heads, self.channel_z)
                for _ in range(self.lm.n_layers)
            ]
        )
        self.lm_attn_to_z_lm = torch.nn.Sequential(
            LayerNorm(self.channel_z),
            LinearNoBias(self.channel_z, self.channel_z, init="relu"),
            torch.nn.ReLU(),
            LinearNoBias(self.channel_z, self.channel_z, init="default"),
        )
        self.lm_stack = trunk.LMStack(
            channel_s=self.channel_s_lm,
            channel_z=self.channel_z,
            num_heads=cfg.trunk.num_heads,
            num_tri_heads=cfg.trunk.num_tri_heads,
            dropout_z=cfg.trunk.dropout_z,
            num_blocks=cfg.trunk.num_lm_blocks,
            blocks_per_ckpt=cfg.trunk.blocks_per_ckpt,
        )
        self.proj_s_lm = torch.nn.Sequential(
            LayerNorm(self.channel_s_lm),
            LinearNoBias(self.channel_s_lm, self.channel_s),
        )
        self.proj_z_lm = torch.nn.Sequential(
            LayerNorm(self.channel_z),
            LinearNoBias(self.channel_z, self.channel_z),
        )

        # Trunk body
        self.main_stack = trunk.PairformerStack(
            channel_s=self.channel_s,
            channel_z=self.channel_z,
            num_heads=cfg.trunk.num_heads,
            num_tri_heads=cfg.trunk.num_tri_heads,
            dropout_z=cfg.trunk.dropout_z,
            num_blocks=cfg.trunk.num_blocks,
            num_pair_to_single_blocks=cfg.trunk.num_pair_to_single_blocks,
            blocks_per_ckpt=cfg.trunk.blocks_per_ckpt,
        )

        # Prediction heads
        self.distogram_head = distogram_head.DistogramHead(
            channel_z=self.channel_z,
            **dataclasses.asdict(cfg.distogram_head),
        )
        self.structure_module = StructureModule(
            channel_s=self.channel_s,
            channel_z=self.channel_z,
            **dataclasses.asdict(cfg.structure_module),
        )

        confidence_cfg = cfg.confidence_head
        self.plddt_head = PredictedLDDTHead(
            self.channel_s, confidence_cfg.hidden_channel, confidence_cfg.num_plddt_bins
        )
        self.pae_head = PredictedAlignedErrorHead(
            self.channel_z, confidence_cfg.num_pae_bins, confidence_cfg.max_pae_error
        )
        self.experimentally_resolved_head = ExperimentallyResolvedHead(self.channel_s)
        self.use_kernel = True

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def set_forward_flags(
        self,
        use_cuequiv_kernels: bool | None = None,
    ) -> None:
        """Set the flags for the forward pass."""
        if use_cuequiv_kernels is not None:
            self.use_kernel = use_cuequiv_kernels

    @torch.inference_mode()
    def inference(
        self,
        batch: dict[str, torch.Tensor],
        return_representations: bool = False,
        return_structure_representations: bool = False,
        mlm_prob: float = 0.15,
        num_recycles: int = 4,
    ) -> dict[str, Any]:
        if mlm_prob < 0.0:
            raise ValueError("mlm_prob must be non-negative.")
        if num_recycles < 0:
            raise ValueError("num_recycles must be non-negative.")

        is_batched = batch["aatype"].ndim == 3
        if not is_batched:
            batch = {key: value.unsqueeze(0) for key, value in batch.items()}

        out: dict[str, Any] = {}

        # Compute positional encodings
        batch["rel_pos"] = self.rel_pos_encoding(batch)

        # Run trunk and structure module
        s, z, structure = self.run_trunk(batch, num_recycles, mlm_prob)
        s, z = s.float(), z.float()

        out["coords"] = structure["coords"]
        if return_representations:
            out["trunk.s"] = s
            out["trunk.z"] = z
        if return_structure_representations:
            out["structure.s"] = structure["act"]

        # Run output heads and confidence metrics in float32.
        with torch.autocast(device_type=self.device.type, enabled=False):
            distogram = self.distogram_head(z)
            out["distogram.logits"] = distogram["logits"]
            out["distogram.boundaries"] = distogram["boundaries"]

            plddt = self.plddt_head(structure["act"].float())
            pae = self.pae_head(z)
            mask = batch["seq_mask"].bool()
            out["plddt"] = confidence_metrics.compute_plddt(**plddt, mask=mask)
            out["pae"] = confidence_metrics.compute_pae(
                pae["logits"], pae["bin_centers"], mask=mask
            )
            out["ptm"] = confidence_metrics.compute_ptm(
                pae["logits"], pae["bin_centers"], mask=mask
            )

        # Remove batch dimension if the input was not originally batched
        if not is_batched:
            out = {key: value.squeeze(0) for key, value in out.items()}
        return out

    def sample_mlm_mask(
        self,
        batch: dict[str, torch.Tensor],
        prob: float,
        synchronized: bool = True,
    ) -> torch.Tensor:
        """Sample a synchronized random MLM mask for the input batch."""
        input_ids = batch["lm.input_ids"]
        batch_size, sequence_length = input_ids.shape
        shape = (1, sequence_length) if synchronized else (batch_size, sequence_length)
        if prob <= 0.0:
            return torch.zeros(shape, dtype=torch.bool, device=input_ids.device)
        return torch.rand(shape, device=input_ids.device) < prob

    def run_lm_embedder(
        self,
        batch: dict[str, torch.Tensor],
        mlm_mask: torch.Tensor | None = None,
        train: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract LM features and project them to s and z representations."""
        _add = partial(torch_utils.add, inplace=not train)

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

        # === Accumulate LM features === #
        B, S = input_ids.shape
        device = input_ids.device
        lm_emb = torch.zeros((B, S, self.lm.d_model), device=device, dtype=torch.float32)
        lm_attn = torch.zeros(
            (B, S, S, self.channel_z), device=device, dtype=torch.float32
        )
        w_layers = self.lm_layer_weights.softmax(dim=0)  # [n_layers+1,]

        with torch.no_grad():
            x = self.lm.embed(input_ids)
        lm_emb = _add(lm_emb, w_layers[0] * self.layernorm_lm_emb(x))

        for i, block in enumerate(self.lm.transformer.blocks):
            with torch.no_grad():
                x, attn = block(x, seq_id, pos_id, return_attn_logits=True)
                # neginf will be set to 0 at the end of this function.
                attn = attn.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                attn = attn.clamp_(-100.0, 100.0).div_(100)
                attn = attn.moveaxis(1, -1)  # [B, S, S, n_heads]
            lm_emb = _add(lm_emb, w_layers[i + 1] * self.layernorm_lm_emb(x))
            lm_attn = _add(lm_attn, self.proj_lm_attn[i](attn))

        # Extract the s and z representations for the valid sequence positions
        # [B, S, c_s], [B, S, S, c_z] -> [B, L, c_s], [B, L, L, c_z]
        b_i = torch.arange(input_ids.shape[0], device=input_ids.device)
        b_i_s, b_i_z = b_i[:, None], b_i[:, None, None]
        row_s = batch["seq_tok_idx"]  # [B, L]
        row_z, col_z = row_s[:, :, None], row_s[:, None, :]
        lm_emb = lm_emb[b_i_s, row_s]  # [B, L, c_s]
        lm_attn = lm_attn[b_i_z, row_z, col_z]  # [B, L, L, c_z]

        s_lm = self.lm_emb_to_s_lm(lm_emb)
        z_lm = self.lm_attn_to_z_lm(lm_attn)
        del lm_emb, lm_attn, row_s, row_z, col_z, b_i_s, b_i_z

        # Run LM stack
        seq_mask = batch["seq_mask"]  # [B, L]
        s_lm, z_lm = self.lm_stack(s_lm, z_lm, seq_mask, self.use_kernel)
        return s_lm, z_lm

    def run_trunk(
        self,
        batch: dict[str, torch.Tensor],
        num_recycles: int,
        mlm_prob: float = 0.15,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Runs trunk and shared IPA once for every initial/recycling pass."""
        mask = batch["seq_mask"]
        B, L = mask.shape
        device = mask.device
        dtype = torch_utils.get_context_dtype(device.type)

        s_prev = torch.zeros(B, L, self.channel_s, device=device, dtype=dtype)
        z_prev = torch.zeros(B, L, L, self.channel_z, device=device, dtype=dtype)
        x_prev = torch.zeros(B, L, 14, 3, device=device, dtype=torch.float32)

        structure: dict[str, Any] = {}
        for _ in range(num_recycles + 1):
            # Prepare the initial s and z representations
            s = self.s_init(batch["aatype"])
            a, b = self.z_init(batch["aatype"]).chunk(2, dim=-1)
            z = a[..., :, None, :] + b[..., None, :, :]
            z = z + self.z_rel_pos(batch["rel_pos"])

            # Add recycling information from previous iteration
            s = s + self.recycle_s(s_prev)
            z = z + self.recycle_z(z_prev)
            dgram = get_distogram(x_prev, batch["pseudo_beta"], **self.recycle_pos_cfg)
            z = z + self.recycle_pos(dgram.to(dtype))

            # Run LM embedder with stochastic masking
            mlm_mask = self.sample_mlm_mask(batch, mlm_prob)
            s_lm, z_lm = self.run_lm_embedder(batch, mlm_mask)
            s = s + self.proj_s_lm(s_lm)
            z = z + self.proj_z_lm(z_lm)

            # Run the main trunk
            s, z = self.main_stack(s, z, mask, self.use_kernel)

            # Run the structure module
            structure = self.structure_module(s, z, batch["aatype_int"], mask)

            s_prev, z_prev = s, z
            x_prev = structure["coords"]
        return s, z, structure

    @classmethod
    def from_pretrained(
        cls,
        state_dict: dict[str, torch.Tensor],
        config: AtlasFoldIPAConfig | None = None,
        device: str | torch.device = "cuda",
        strict: bool = True,
    ) -> AtlasFold_IPA:
        config = config if config is not None else AtlasFoldIPAConfig()
        device = torch.device(device)
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        with torch.device("meta"):
            model = cls(config, load_lm=False)
            del model.lm
        model = model.to_empty(device=device)
        model.load_state_dict(state_dict, strict=strict)
        model.lm = load_model(
            config.lm_name, path=config.lm_path, device=device, dtype=dtype
        )
        model.lm.requires_grad_(False)
        model.alphabet = model.lm.alphabet
        if dtype == torch.bfloat16:
            model.lm = model.lm.bfloat16()
            model.lm_stack = model.lm_stack.bfloat16()
            model.main_stack = model.main_stack.bfloat16()
        model.requires_grad_(False)
        model.eval()
        return model
