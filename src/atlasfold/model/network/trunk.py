from functools import partial

import torch

from atlasfold.model.network.block import PairBlock
from atlasfold.utils.checkpointing import checkpoint_blocks


class TriangularUpdateTrunk(torch.nn.Module):
    """Stack multiple triangular update blocks."""

    def __init__(
        self,
        channel_s: int = 384,
        channel_z: int = 128,
        num_heads_attn: int = 12,
        num_heads_tri_attn: int = 4,
        dropout_s: float = 0.15,
        dropout_z: float = 0.25,
        num_blocks: int = 48,
        num_single_to_pair_blocks: int = 4,
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
                    single_to_pair=i < num_single_to_pair_blocks,
                )
                for i in range(num_blocks)
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
