import math
from functools import partial

import einops
import torch
import torch.nn as nn

from atlasfold.model.network.misc import relative_position_encoding
from atlasfold.model.network.primitives import (
    AdaLN,
    LayerNorm,
    Linear,
    LinearNoBias,
    SwiGLU,
    Transition,
)
from atlasfold.utils.checkpointing import checkpoint_blocks
from atlasfold.utils.torch_utils import add

from .attention import SelfAttention


class FourierEmbedding(nn.Module):
    """Fourier embedding layer.
    Section 3.7 Algorithm 22 Fourier Embedding
    """

    def __init__(self, channel: int):
        super().__init__()
        generator = torch.Generator()
        generator.manual_seed(42)
        w = torch.randn(size=(1, channel), generator=generator)
        b = torch.randn(size=(1, channel), generator=generator)
        self.register_buffer("w", w)
        self.register_buffer("b", b)

    def forward(self, t_hat: torch.Tensor) -> torch.Tensor:
        """See Section 3.7 Algorithm 22 of AlphaFold3 paper."""
        return torch.cos((2 * math.pi) * t_hat[..., None] * self.w + self.b)


class ConditionedTransitionBlock(torch.nn.Module):
    """Conditioned Transition Block
    Section 3.7 Algorithm 25 Conditioned Transition Block
    """

    def __init__(self, channel: int, channel_cond: int, expansion_factor: int = 2):
        super().__init__()
        self.adaln = AdaLN(channel, channel_cond)
        model_dim = int(channel * expansion_factor)
        self.swiglu = SwiGLU(channel, model_dim)
        self.linear_g = Linear(channel_cond, channel, init="gating_closed")
        self.linear_out = LinearNoBias(model_dim, channel, init="default")

    def forward(self, a: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """See Section 3.7 Algorithm 25 Conditioned Transition Block"""
        a = self.adaln(a, cond)
        b = self.swiglu(a)
        a = torch.sigmoid(self.linear_g(cond)) * self.linear_out(b)
        return a


class SingleConditioning(nn.Module):
    """Diffusion conditioning layer for single representations.
    See Section 3.7 Algorithm 21 Diffusion Conditioning in the AF3 paper.
    """

    def __init__(self, channel_s: int = 384, dim_fourier: int = 256):
        super().__init__()
        self.layernorm = LayerNorm(channel_s, create_offset=False)
        self.linear = LinearNoBias(channel_s, channel_s, init="default")
        self.embedding_aa = LinearNoBias(21, channel_s)

        self.fourier_embed = FourierEmbedding(dim_fourier)
        self.layernorm_fourier = LayerNorm(dim_fourier, create_offset=False)
        self.linear_fourier = LinearNoBias(dim_fourier, channel_s, init="default")
        self.transitions = nn.ModuleList(
            [Transition(channel_s, expansion_factor=2) for _ in range(2)]
        )

    def forward(
        self, batch: dict[str, torch.Tensor], s: torch.Tensor, c_noise: torch.Tensor
    ) -> torch.Tensor:
        """See Section 3.7 Algorithm 21 Diffusion Conditioning in the AF3 paper.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            The input batch.
        s : torch.Tensor
            Tensor of shape (B, L, c_s) containing single representations.
        c_noise : torch.Tensor
            Tensor of shape (B, N) containing diffusion noise level (or sigma).
            c_noise = 1/4 log(t_hat / sigma_data) (See Algorithm.)

        Returns
        -------
        cond : torch.Tensor
            Tensor of shape (B, N, L, c_s) containing conditioned single embeddings.
        """
        _add = partial(add, inplace=not self.training)

        s = self.linear(self.layernorm(s))  # [B, L, c_s]

        # Embed residue type
        s = _add(s, self.embedding_aa(batch["aatype"]))  # [B, L, c_s]

        # Embed fourier embedding of noise level
        fourier_embed = self.fourier_embed(c_noise)  # [B, N, d_fourier]
        fourier_embed = self.linear_fourier(self.layernorm_fourier(fourier_embed))
        s = s[:, None, :, :] + fourier_embed[:, :, None, :]  # [B, N, L, c_s]

        # Apply transitions
        for transition in self.transitions:
            s = _add(s, transition(s))
        return s


class PairConditioning(nn.Module):
    """Diffusion conditioning layer for pair representations.
    See Section 3.7 Algorithm 21 Diffusion Conditioning in the AF3 paper.
    """

    def __init__(
        self,
        channel_z: int = 128,
        channel_bias: tuple[int, int] = (12, 16),
    ):
        super().__init__()
        self.channel_z: int = channel_z
        self.channel_bias: tuple[int, int] = channel_bias

        rel_pos_dim = (32 * 2 + 2) + (2 * 2 + 2) + 1
        in_channel = channel_z + rel_pos_dim
        self.layernorm = LayerNorm(in_channel, create_offset=False)
        self.linear = LinearNoBias(in_channel, channel_z, init="default")
        self.transitions = nn.ModuleList(
            [Transition(channel_z, expansion_factor=2) for _ in range(2)]
        )

        # To bias
        nblock, nhead = channel_bias
        self.layernorm_bias = LayerNorm(channel_z, create_offset=False)
        self.linear_bias = LinearNoBias(channel_z, nblock * nhead)

    def forward(self, batch: dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        """See Section 3.7 Algorithm 21 Diffusion Conditioning in the AF3 paper.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            The input batch.
        z : torch.Tensor
            Tensor of shape (B, L, L, c_z) containing pair representations.

        Returns
        -------
        pair_bias : torch.Tensor
            Tensor of shape (n_block, B, n_head, L, L) containing attention bias.
        """
        _add = partial(add, inplace=not self.training)

        # Concatenate relative positional encoding
        rel_pos = relative_position_encoding(
            batch["res_idx"], batch["asym_id"], batch["entity_id"], batch["sym_id"]
        )
        z = torch.cat((z, rel_pos.to(z.device)), dim=-1)
        z = self.linear(self.layernorm(z))  # [B, L, L, c_z]

        # Apply transitions
        for transition in self.transitions:
            z = _add(z, transition(z))

        # Convert to attention bias
        pair_bias = self.linear_bias(self.layernorm_bias(z))
        # Reshape to (n_block, B, n_head, L, L)
        pair_bias = einops.rearrange(
            pair_bias,
            "... q k (n h) -> n ... h q k",
            n=self.channel_bias[0],
            h=self.channel_bias[1],
        ).contiguous()
        return pair_bias


class DiffusionTransformerStack(nn.Module):
    """Global Attention Diffusion Transformer Stack."""

    def __init__(
        self,
        channel_a: int,
        channel_cond: int | None,
        num_heads: int,
        num_blocks: int,
        use_pair_bias: bool = True,
        blocks_per_ckpt: int | None = None,
    ):
        """Initialize the diffusion transformer.

        Parameters
        ----------
        channel_a : int
            The single representation dimension.
        channel_cond : int | None
            The single conditioning dimension.
        num_heads : int
            The number of heads.
        num_blocks : int
            The number of blocks.
        blocks_per_ckpt : int | None, optional
            The number of blocks per checkpoint
        """
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                DiffusionTransformerBlock(
                    channel_a, channel_cond, num_heads, use_pair_bias
                )
                for _ in range(num_blocks)
            ]
        )
        self.blocks_per_ckpt: int | None = blocks_per_ckpt

    def forward(
        self,
        a: torch.Tensor,
        mask: torch.Tensor,
        single_cond: torch.Tensor | None,
        pair_bias: torch.Tensor | None,
    ):
        """Global attention diffusion transformer

        Parameters
        ----------
        a : torch.Tensor
            The single representation tensor (*, L, c_a)
        mask : torch.Tensor
            The attention mask tensor (*, L)
        single_cond : torch.Tensor
            The single conditioning tensor (*, L, c_s)
        pair_bias : torch.Tensor | None
            The pair bias tensor (num_blocks, *, num_heads, L, L)
        """
        if pair_bias is not None:
            # Move block dimension to batch dimension for checkpointing
            a = checkpoint_blocks(
                self.blocks,
                args=(a,),
                static_args={"single_cond": single_cond, "mask": mask},
                layer_args={"pair_bias": pair_bias},
                blocks_per_ckpt=self.blocks_per_ckpt,
                use_reentrant=False,
            )[0]
        else:
            a = checkpoint_blocks(
                self.blocks,
                args=(a,),
                static_args={"single_cond": single_cond, "mask": mask, "pair_bias": None},
                blocks_per_ckpt=self.blocks_per_ckpt,
                use_reentrant=False,
            )[0]
        return a


class DiffusionTransformerBlock(nn.Module):
    """Global Attention Diffusion Transformer Block."""

    def __init__(
        self,
        channel_a: int,
        channel_cond: int | None,
        num_heads: int,
        use_pair_bias: bool = True,
    ):
        """Initialize the diffusion transformer block.

        Parameters
        ----------
        channel_a : int
            The single representation dimension.
        channel_cond : int
            The single conditioning dimension.
        num_heads : int
            The number of heads.
        use_pair_bias : bool, optional
            Whether to use pair bias in attention, by default True
        """
        super().__init__()
        self.use_conditioning = channel_cond is not None
        self.attention = SelfAttention(
            channel_a, num_heads, channel_cond, use_pair_bias=use_pair_bias
        )
        if self.use_conditioning:
            assert channel_cond is not None
            self.transition = ConditionedTransitionBlock(channel_a, channel_cond)
        else:
            self.transition = Transition(channel_a, expansion_factor=2)

    def forward(
        self,
        a: torch.Tensor,
        mask: torch.Tensor,
        single_cond: torch.Tensor | None,
        pair_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Global attention diffusion transformer block

        Parameters
        ----------
        a : torch.Tensor
            The single representation tensor (*, L, c_a)
        single_cond : torch.Tensor | None
            The single conditioning tensor (*, L, c_cond)
        mask : torch.Tensor
            The attention mask tensor (*, L)
        pair_bias : torch.Tensor | None
            The pair bias tensor (*, H, L, L)

        Returns
        -------
        a : torch.Tensor
            The output single representation tensor (*, L, c_a)
        """
        a = a + self.attention(a, mask, pair_bias, single_cond)
        if self.use_conditioning:
            a = a + self.transition(a, single_cond)
        else:
            a = a + self.transition(a)
        return a
