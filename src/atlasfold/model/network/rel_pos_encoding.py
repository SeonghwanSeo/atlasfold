import torch


class RelativePositionEncoding(torch.nn.Module):
    def __init__(
        self,
        r_max: int = 32,
        s_max: int | None = None,
    ) -> None:
        """Relative position encoding module for pair representation.

        Parameters
        ----------
        r_max: int
            The maximum residue distance for relative position encoding.
        s_max: int
            The maximum chain distance for relative position encoding.
        """
        super().__init__()
        self.r_max: int = r_max
        self.s_max: int | None = s_max
        if s_max is None:
            self.multimer: bool = False
            self.dim: int = 2 * r_max + 1
        else:
            self.multimer: bool = True
            self.dim: int = 2 * r_max + 2 * s_max + 5

    def forward(
        self,
        feat: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the relative position encoding for the pair representation.

        Parameters
        ----------
        feat: dict[str, torch.Tensor]
            A dictionary containing the following keys:
            - res_idx: torch.Tensor of shape (*, L) containing residue indices.
            - asym_id: torch.Tensor of shape (*, L) containing chain indices (asym_id).
            - entity_id: torch.Tensor of shape (*, L) containing entity indices.
            - sym_id: torch.Tensor of shape (*, L) containing symmetry group indices.

        Returns
        -------
        torch.Tensor
            The relative position encoding tensor of shape (*, L, L, bins).
        """
        r_max, s_max = self.r_max, self.s_max
        if s_max is None:
            res_idx = feat["res_idx"]
            rel_pos = res_idx[..., :, None] - res_idx[..., None, :]
            rel_pos = torch.clamp(rel_pos + r_max, min=0, max=2 * r_max)
            a_rel_pos = torch.nn.functional.one_hot(rel_pos, 2 * r_max + 1)
            rel_position_encoding = a_rel_pos.float()
        else:
            res_idx = feat["res_idx"]
            asym_id = feat["asym_id"]
            entity_id = feat["entity_id"]
            sym_id = feat["sym_id"]

            is_same_chain = torch.eq(asym_id[..., :, None], asym_id[..., None, :])
            rel_pos = res_idx[..., :, None] - res_idx[..., None, :]
            rel_pos = torch.clamp(rel_pos + r_max, min=0, max=2 * r_max)
            rel_pos = torch.where(is_same_chain, rel_pos, 2 * r_max + 1)
            a_rel_pos = torch.nn.functional.one_hot(rel_pos, 2 * r_max + 2).float()

            is_same_entity = torch.eq(entity_id[..., :, None], entity_id[..., None, :])
            rel_chain = sym_id[..., :, None] - sym_id[..., None, :]
            rel_chain = torch.clip(rel_chain + s_max, min=0, max=2 * s_max)
            rel_chain = torch.where(is_same_entity, rel_chain, 2 * s_max + 1)
            a_rel_chain = torch.nn.functional.one_hot(rel_chain, 2 * s_max + 2).float()
            a_entity = is_same_entity.float().unsqueeze(-1)

            rel_position_encoding = torch.cat(
                [a_rel_pos, a_entity.float(), a_rel_chain], dim=-1
            )
        return rel_position_encoding
