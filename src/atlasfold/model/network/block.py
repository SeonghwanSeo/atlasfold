from functools import partial

import einops
import torch

from atlasfold.model.network.attention import SelfAttention
from atlasfold.model.network.primitives import (
    DropoutColumnwise,
    DropoutRowwise,
    LayerNorm,
    Linear,
    LinearNoBias,
    Transition,
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from atlasfold.utils.checkpointing import checkpoint_blocks
from atlasfold.utils.torch_utils import add


# ============================================================
# Triangular update block
# ============================================================
class PairwiseProdDiff(torch.nn.Module):
    """Convert single embeddings to pairwise embeddings.
    Inspired by ESMFold's implementation.
    """

    def __init__(self, channel_s: int, channel_z: int) -> None:
        super().__init__()
        assert channel_z % 2 == 0, "channel_out must be even."
        self.layernorm = LayerNorm(channel_s)
        self.linear_in = Linear(channel_s, channel_z * 2, init="default")
        self.linear_out = LinearNoBias(channel_z, channel_z, init="final")

    def forward(self, s: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        """Compute pairwise embeddings from single embeddings.

        Parameters
        ----------
        s : torch.Tensor
            The single representation (*, L, c_in).

        Returns
        -------
        z: torch.Tensor
            The output tensor (*, L, L, c_out).
        """
        s = self.layernorm(s)  # (*, L, c_in)
        s_i, s_j = self.linear_in(s).chunk(2, dim=-1)  # 2 * (*, L, c_hid)
        s_i = s_i[..., :, None, :]
        s_j = s_j[..., None, :, :]

        # Combine Diff (Asymmetry) and Prod (Correlation)
        z = torch.cat([s_i - s_j, s_i * s_j], dim=-1)  # (*, L, L, c_out)
        z = self.linear_out(z)  # (*, L, L, c_out)
        z = z * pair_mask[..., None]  # Mask out invalid pairs
        return z


class PairStack(torch.nn.Module):
    """Stack multiple triangular update blocks."""

    def __init__(
        self,
        channel_s: int = 384,
        channel_z: int = 128,
        num_heads_attn: int = 12,
        num_heads_tri_attn: int = 4,
        dropout_s: float = 0.15,
        dropout_z: float = 0.25,
        single_to_pair: bool = True,
        pair_to_pair: bool = True,
        pair_to_single: bool = True,
        use_tri_mul: bool = True,
        use_tri_attn: bool = True,
        num_blocks: int = 48,
        blocks_per_ckpt: int | None = None,
    ) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                PairBlock(
                    channel_s,
                    channel_z,
                    num_heads_attn,
                    num_heads_tri_attn,
                    dropout_s,
                    dropout_z,
                    single_to_pair,
                    pair_to_pair,
                    pair_to_single,
                    use_tri_mul,
                    use_tri_attn,
                )
                for _ in range(num_blocks)
            ]
        )
        self.blocks_per_ckpt: int | None = blocks_per_ckpt

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        use_cuequiv_kernels: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass.

        Parameters
        ----------
        s : torch.Tensor
            The single representations of shape (B, L, C_s)
        z : torch.Tensor
            The pair representations of shape (B, L, L, C_z)
        mask : torch.Tensor
            The token mask of shape (B, L)
        use_cuequiv_kernels : bool, optional
            Whether to use cuEQUIV kernels, by default False.

        Returns
        -------
        s : torch.Tensor
            The updated single representations
        z: torch.Tensor
            The updated pair representations
        """
        pair_mask = mask[..., :, None] & mask[..., None, :]
        blocks = [
            partial(
                b,
                mask=mask,
                pair_mask=pair_mask,
                use_cuequiv_kernels=use_cuequiv_kernels,
            )
            for b in self.blocks
        ]
        s, z = checkpoint_blocks(
            blocks,
            (s, z),
            self.blocks_per_ckpt,
            use_reentrant=False,
        )
        return s, z


class PairBlock(torch.nn.Module):
    def __init__(
        self,
        channel_s: int = 384,
        channel_z: int = 128,
        num_heads_attn: int = 32,
        num_heads_tri_attn: int = 4,
        dropout_s: float = 0.15,
        dropout_z: float = 0.25,
        single_to_pair: bool = True,
        pair_to_pair: bool = True,
        pair_to_single: bool = True,
        use_tri_mul: bool = True,
        use_tri_attn: bool = True,
    ) -> None:
        super().__init__()
        self.channel_s: int = channel_s
        self.channel_z: int = channel_z

        # Configurations for the block
        self.single_to_pair = single_to_pair
        self.pair_to_pair = pair_to_pair
        self.pair_to_single = pair_to_single

        # Single to pair
        if self.single_to_pair:
            self.pairwise_prod_diff = PairwiseProdDiff(channel_s, channel_z)

        # Pair to pair
        if self.pair_to_pair:
            self.use_tri_mul: bool = use_tri_mul
            self.use_tri_attn: bool = use_tri_attn
            if use_tri_mul:
                self.tri_mul_out = TriangleMultiplicationOutgoing(channel_z)
                self.tri_mul_in = TriangleMultiplicationIncoming(channel_z)
            if use_tri_attn:
                self.tri_attn_start = TriangleAttentionStartingNode(
                    channel_z, num_heads_tri_attn
                )
                self.tri_attn_end = TriangleAttentionEndingNode(
                    channel_z, num_heads_tri_attn
                )
            self.transition_z = Transition(channel_z, 4)
            self.dropout_rowwise_z = DropoutRowwise(dropout_z)
            self.dropout_columnwise_z = DropoutColumnwise(dropout_z)

        # Pair to single
        if self.pair_to_single:
            self.pair_to_single_bias = torch.nn.Sequential(
                LayerNorm(channel_z),
                LinearNoBias(channel_z, num_heads_attn, init="default"),
            )
            self.attention = SelfAttention(
                channel_a=channel_s,
                channel_cond=None,
                num_heads=num_heads_attn,
                use_pair_bias=True,
            )
            self.transition_s = Transition(channel_s, 4)
            self.dropout_s = torch.nn.Dropout(dropout_s)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        pair_mask: torch.Tensor,
        use_cuequiv_kernels: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass.

        Parameters
        ----------
        s : torch.Tensor
            The single representations of shape (B, L, C_s)
        z : torch.Tensor
            The pair representations of shape (B, L, L, C_z)
        mask : torch.Tensor
            The token mask of shape (B, L)
        pair_mask : torch.Tensor
            The pair mask of shape (B, L, L)

        Returns
        -------
        s : torch.Tensor
            The updated single representations
        z : torch.Tensor
            The updated pair representations
        """
        _add = partial(add, inplace=not self.training)

        # Step 1: single_to_pair
        if self.single_to_pair:
            z = _add(z, self.pairwise_prod_diff(s, pair_mask))

        # Step 2: pair to pair
        if self.pair_to_pair:
            if self.use_tri_mul:
                z = _add(
                    z,
                    self.dropout_rowwise_z(
                        self.tri_mul_out(z, pair_mask, use_kernels=use_cuequiv_kernels)
                    ),
                )
                z = _add(
                    z,
                    self.dropout_rowwise_z(
                        self.tri_mul_in(z, pair_mask, use_kernels=use_cuequiv_kernels)
                    ),
                )
            if self.use_tri_attn:
                z = _add(
                    z,
                    self.dropout_rowwise_z(
                        self.tri_attn_start(z, pair_mask, use_kernels=use_cuequiv_kernels)
                    ),
                )
                z = _add(
                    z,
                    self.dropout_columnwise_z(
                        self.tri_attn_end(z, pair_mask, use_kernels=use_cuequiv_kernels)
                    ),
                )
            z = _add(z, self.transition_z(z))

        # Step 3: pair to single
        if self.pair_to_single:
            pair_bias = einops.rearrange(
                self.pair_to_single_bias(z), "... i j h -> ... h i j"
            )  # [*, H, L, L]
            s = _add(
                s,
                self.dropout_s(self.attention(s, mask, pair_bias=pair_bias)),
            )
            s = _add(s, self.transition_s(s))

        return s, z
