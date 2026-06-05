from functools import partial

import torch

from atlasfold.model.network.block import PairBlock
from atlasfold.utils.checkpointing import checkpoint_blocks


class TriangularUpdateTrunk(torch.nn.Module):
    """Main trunk of the AtlasFold model"""

    def __init__(
        self,
        channel_z: int = 192,
        dropout_z: float = 0.25,
        num_blocks: int = 48,
        blocks_per_ckpt: int | None = None,
    ) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                PairBlock(
                    channel_z=channel_z,
                    dropout_z=dropout_z,
                    single_to_pair=False,
                    pair_to_pair=True,
                    pair_to_single=False,
                    use_tri_mul=True,
                    use_tri_attn=False,
                )
                for _ in range(num_blocks)
            ]
        )
        self.blocks_per_ckpt: int | None = blocks_per_ckpt

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        use_cuequiv_kernels: bool = False,
    ) -> torch.Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        z : torch.Tensor
            The pair representations of shape (B, L, L, C_z)
        mask : torch.Tensor
            The token mask of shape (B, L)
        use_cuequiv_kernels : bool, optional
            Whether to use cuEQUIV kernels, by default False.

        Returns
        -------
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
        _, z = checkpoint_blocks(
            blocks,
            (None, z),
            self.blocks_per_ckpt,
            use_reentrant=False,
        )
        return z


class LMModule(torch.nn.Module):
    """LM Module of the AtlasFold model"""

    def __init__(
        self,
        channel_s: int = 768,
        channel_z: int = 192,
        num_heads_attn: int = 12,
        dropout_z: float = 0.25,
        num_blocks: int = 4,
        blocks_per_ckpt: int | None = None,
    ) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                PairBlock(
                    channel_s=channel_s,
                    channel_z=channel_z,
                    num_heads_attn=num_heads_attn,
                    dropout_z=dropout_z,
                    single_to_pair=True,
                    pair_to_pair=True,
                    pair_to_single=(i < num_blocks - 1),
                    use_tri_mul=True,
                    use_tri_attn=False,
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
    ) -> torch.Tensor:
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
        _, z = checkpoint_blocks(
            blocks,
            (s, z),
            self.blocks_per_ckpt,
            use_reentrant=False,
        )
        return z
