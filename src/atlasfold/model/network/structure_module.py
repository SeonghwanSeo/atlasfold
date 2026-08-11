# Copyright 2021 DeepMind Technologies Limited
# Copyright 2021 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AlphaFold-style invariant-point structure modules."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from atlasfold.common import residue_constants as rc

from .primitives import LayerNorm, Linear


def _quat_to_rot(quat: torch.Tensor) -> torch.Tensor:
    quat = quat / torch.clamp(
        torch.linalg.vector_norm(quat, dim=-1, keepdim=True), min=1e-8
    )
    w, x, y, z = quat.unbind(dim=-1)
    two = 2.0
    return torch.stack(
        (
            1.0 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1.0 - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1.0 - two * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quat.shape[:-1], 3, 3)


def _quat_multiply(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = first.unbind(dim=-1)
    bw, bx, by, bz = second.unbind(dim=-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def _rot_to_quat(rot: torch.Tensor) -> torch.Tensor:
    """Converts rotation matrices to scalar-first unit quaternions."""
    r00, r01, r02 = rot[..., 0, 0], rot[..., 0, 1], rot[..., 0, 2]
    r10, r11, r12 = rot[..., 1, 0], rot[..., 1, 1], rot[..., 1, 2]
    r20, r21, r22 = rot[..., 2, 0], rot[..., 2, 1], rot[..., 2, 2]
    q_abs = 0.5 * torch.sqrt(
        torch.clamp(
            torch.stack(
                (
                    1.0 + r00 + r11 + r22,
                    1.0 + r00 - r11 - r22,
                    1.0 - r00 + r11 - r22,
                    1.0 - r00 - r11 + r22,
                ),
                dim=-1,
            ),
            min=0.0,
        )
    )
    qw, qx, qy, qz = q_abs.unbind(dim=-1)
    qx = torch.copysign(qx, r21 - r12)
    qy = torch.copysign(qy, r02 - r20)
    qz = torch.copysign(qz, r10 - r01)
    quat = torch.stack((qw, qx, qy, qz), dim=-1)
    return quat / torch.clamp(
        torch.linalg.vector_norm(quat, dim=-1, keepdim=True), min=1e-8
    )


class Rigid:
    """A rotation-matrix/translation rigid transform."""

    def __init__(
        self,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        quaternion: torch.Tensor | None = None,
    ):
        self.rotation = rotation
        self.translation = translation
        self.quaternion = quaternion

    @property
    def device(self) -> torch.device:
        return self.translation.device

    @classmethod
    def identity(cls, shape: tuple[int, ...], device: torch.device) -> Rigid:
        rotation = torch.eye(3, dtype=torch.float32, device=device)
        rotation = rotation.expand(*shape, 3, 3).clone()
        translation = torch.zeros(*shape, 3, dtype=torch.float32, device=device)
        quaternion = torch.zeros(*shape, 4, dtype=torch.float32, device=device)
        quaternion[..., 0] = 1.0
        return cls(rotation, translation, quaternion)

    @classmethod
    def from_tensor_4x4(cls, tensor: torch.Tensor) -> Rigid:
        return cls(tensor[..., :3, :3], tensor[..., :3, 3])

    def apply(self, points: torch.Tensor) -> torch.Tensor:
        extra_dims = points.ndim - self.translation.ndim
        rotation = self.rotation.reshape(
            *self.rotation.shape[:-2], *([1] * extra_dims), 3, 3
        )
        translation = self.translation.reshape(
            *self.translation.shape[:-1], *([1] * extra_dims), 3
        )
        return torch.matmul(rotation, points.unsqueeze(-1)).squeeze(-1) + translation

    def invert_apply(self, points: torch.Tensor) -> torch.Tensor:
        extra_dims = points.ndim - self.translation.ndim
        rotation = self.rotation.transpose(-1, -2).reshape(
            *self.rotation.shape[:-2], *([1] * extra_dims), 3, 3
        )
        translation = self.translation.reshape(
            *self.translation.shape[:-1], *([1] * extra_dims), 3
        )
        return torch.matmul(rotation, (points - translation).unsqueeze(-1)).squeeze(-1)

    def compose(self, other: Rigid) -> Rigid:
        rotation = torch.matmul(self.rotation, other.rotation)
        translation = self.apply(other.translation)
        quaternion = None
        if self.quaternion is not None and other.quaternion is not None:
            quaternion = _quat_multiply(self.quaternion, other.quaternion)
            quaternion = quaternion / torch.clamp(
                torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True), min=1e-8
            )
        return Rigid(rotation, translation, quaternion)

    def compose_update(self, update: torch.Tensor) -> Rigid:
        one = torch.ones_like(update[..., :1])
        update_quaternion = torch.cat((one, update[..., :3]), dim=-1)
        update_quaternion = update_quaternion / torch.clamp(
            torch.linalg.vector_norm(update_quaternion, dim=-1, keepdim=True), min=1e-8
        )
        update_rotation = _quat_to_rot(update_quaternion)
        update_rigid = Rigid(update_rotation, update[..., 3:], update_quaternion)
        return self.compose(update_rigid)

    def scale_translation(self, scale: float) -> Rigid:
        return Rigid(self.rotation, self.translation * scale, self.quaternion)

    def stop_rotation_gradient(self) -> Rigid:
        quaternion = self.quaternion.detach() if self.quaternion is not None else None
        return Rigid(self.rotation.detach(), self.translation, quaternion)

    def unsqueeze_group(self) -> Rigid:
        """Adds a rigid-group batch axis immediately before xyz dimensions."""

        quaternion = (
            self.quaternion.unsqueeze(-2) if self.quaternion is not None else None
        )
        return Rigid(
            self.rotation.unsqueeze(-3),
            self.translation.unsqueeze(-2),
            quaternion,
        )

    def to_tensor_4x4(self) -> torch.Tensor:
        bottom = torch.zeros(
            *self.translation.shape[:-1], 1, 4, dtype=torch.float32, device=self.device
        )
        bottom[..., 0, 3] = 1.0
        upper = torch.cat((self.rotation, self.translation.unsqueeze(-1)), dim=-1)
        return torch.cat((upper, bottom), dim=-2)

    def to_tensor_7(self) -> torch.Tensor:
        quaternion = (
            self.quaternion
            if self.quaternion is not None
            else _rot_to_quat(self.rotation)
        )
        return torch.cat((quaternion, self.translation), dim=-1)


class _PointProjection(nn.Module):
    """AlphaFold-Multimer point projection"""

    def __init__(
        self,
        channel: int,
        num_head: int,
        num_point: int,
    ):
        super().__init__()
        self.num_head = num_head
        self.num_point = num_point
        self.linear = Linear(channel, num_head * num_point * 3)

    def forward(self, act: torch.Tensor, rigid: Rigid) -> torch.Tensor:
        projected = self.linear(act)
        projected = projected.reshape(*act.shape[:-1], self.num_head, 3 * self.num_point)
        x, y, z = projected.chunk(3, dim=-1)
        local = torch.stack((x, y, z), dim=-1)
        return rigid.apply(local)


class InvariantPointAttention(nn.Module):
    def __init__(
        self,
        channel_s: int,
        channel_z: int,
        num_head: int = 12,
        num_scalar_qk: int = 16,
        num_scalar_v: int = 16,
        num_point_qk: int = 4,
        num_point_v: int = 8,
        epsilon: float = 1e-8,
        inf: float = 1e5,
    ) -> None:
        super().__init__()
        self.num_head = num_head
        self.num_scalar_qk = num_scalar_qk
        self.num_scalar_v = num_scalar_v
        self.num_point_qk = num_point_qk
        self.num_point_v = num_point_v
        self.epsilon = epsilon
        self.inf = inf
        scalar_width = num_head * num_scalar_qk
        value_width = num_head * num_scalar_v

        self.linear_q = Linear(channel_s, scalar_width, bias=False)
        self.linear_k = Linear(channel_s, scalar_width, bias=False)
        self.linear_v = Linear(channel_s, value_width, bias=False)
        self.linear_q_points = _PointProjection(channel_s, num_head, num_point_qk)
        self.linear_k_points = _PointProjection(channel_s, num_head, num_point_qk)
        self.linear_v_points = _PointProjection(channel_s, num_head, num_point_v)

        self.linear_z_bias = Linear(channel_z, num_head)
        self.point_weights = nn.Parameter(torch.full((num_head,), 0.541324854612918))
        output_width = num_head * (num_scalar_v + channel_z + 4 * num_point_v)
        self.linear_output = Linear(output_width, channel_s, init="final")

    def _projections(
        self, act: torch.Tensor, rigid: Rigid
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        q = self.linear_q(act).reshape(*act.shape[:-1], self.num_head, self.num_scalar_qk)
        q_point = self.linear_q_points(act, rigid)
        k = self.linear_k(act).reshape(*act.shape[:-1], self.num_head, self.num_scalar_qk)
        v = self.linear_v(act).reshape(*act.shape[:-1], self.num_head, self.num_scalar_v)
        k_point = self.linear_k_points(act, rigid)
        v_point = self.linear_v_points(act, rigid)
        return q, k, v, q_point, k_point, v_point

    def forward(
        self,
        act: torch.Tensor,
        z: torch.Tensor,
        rigid: Rigid,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        output_dtype = act.dtype
        q, k, v, q_point, k_point, v_point = self._projections(act, rigid)
        z_bias = self.linear_z_bias(z)

        scalar_logits = torch.einsum("...ihc,...jhc->...ijh", q, k)
        point_delta = q_point.unsqueeze(-4) - k_point.unsqueeze(-5)
        point_distance = torch.sum(point_delta.square(), dim=-1)
        point_weight = F.softplus(self.point_weights.float())
        point_weight = point_weight.reshape(
            *([1] * (point_distance.ndim - 2)), self.num_head, 1
        )

        scalar_logits = scalar_logits * math.sqrt(1.0 / self.num_scalar_qk)
        point_logits = -0.5 * torch.sum(
            point_distance
            * (point_weight * math.sqrt(1.0 / (self.num_point_qk * 9.0 / 2.0))),
            dim=-1,
        )
        logits = scalar_logits + z_bias + point_logits

        z_mask = mask.float().unsqueeze(-1) * mask.float().unsqueeze(-2)
        logits = logits + self.inf * (z_mask.unsqueeze(-1) - 1.0)
        logits = logits * math.sqrt(1.0 / 3.0)
        with torch.autocast(device_type=act.device.type, enabled=False):
            attention = torch.softmax(logits.float(), dim=-2)

        scalar_result = torch.einsum("...ijh,...jhc->...ihc", attention, v)
        point_result_global = torch.einsum("...ijh,...jhpq->...ihpq", attention, v_point)
        point_result_local = rigid.invert_apply(point_result_global)
        point_norm = torch.sqrt(
            torch.sum(point_result_local.square(), dim=-1) + self.epsilon
        )
        z_result = torch.einsum("...ijh,...ijc->...ihc", attention, z.float())

        scalar_result = scalar_result.flatten(start_dim=-2)
        point_flat = point_result_local.reshape(*point_result_local.shape[:-3], -1, 3)
        point_x, point_y, point_z = point_flat.unbind(dim=-1)
        features = torch.cat(
            (
                scalar_result,
                point_x,
                point_y,
                point_z,
                point_norm.flatten(start_dim=-2),
                z_result.flatten(start_dim=-2),
            ),
            dim=-1,
        )
        update = self.linear_output(features)
        return update.to(output_dtype)


class StructureTransition(nn.Module):
    def __init__(self, channel: int, depth: int, dropout: float):
        super().__init__()
        layers = []
        for _ in range(depth - 1):
            layers.append(Linear(channel, channel, init="relu"))
            layers.append(nn.ReLU())
        layers.append(Linear(channel, channel, init="final"))
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(channel)

    def forward(self, act: torch.Tensor) -> torch.Tensor:
        act = act + self.mlp(act)
        return self.layer_norm(self.dropout(act))


class TorsionModule(nn.Module):
    def __init__(
        self,
        channel_s: int,
        sidechain_channel: int = 128,
        num_layer: int = 2,
        num_torsion: int = 7,
    ) -> None:
        super().__init__()
        self.proj_input = nn.Sequential(nn.ReLU(), Linear(channel_s, sidechain_channel))
        self.proj_init = nn.Sequential(nn.ReLU(), Linear(channel_s, sidechain_channel))
        self.blocks = nn.ModuleList()
        for _ in range(num_layer):
            self.blocks.append(
                nn.Sequential(
                    nn.ReLU(),
                    Linear(sidechain_channel, sidechain_channel, init="relu"),
                    nn.ReLU(),
                    Linear(sidechain_channel, sidechain_channel, init="final"),
                )
            )
        self.proj_output = nn.Sequential(
            nn.ReLU(), Linear(sidechain_channel, num_torsion * 2)
        )

    def forward(self, s: torch.Tensor, s_init: torch.Tensor) -> torch.Tensor:
        a = self.proj_input(s) + self.proj_init(s_init)
        for block in self.blocks:
            a = a + block(a)
        a = self.proj_output(a).unflatten(-1, (-1, 2))
        return a


def torsion_angles_to_frames(
    backbone: Rigid,
    torsions: torch.Tensor,
    aatype: torch.Tensor,
    default_frames: torch.Tensor,
) -> Rigid:
    aatype = aatype.long()
    defaults = default_frames[aatype]
    default_rigid = Rigid.from_tensor_4x4(defaults)

    backbone_torsion = torch.zeros_like(torsions[..., :1, :])
    backbone_torsion[..., 0, 1] = 1.0
    all_torsions = torch.cat((backbone_torsion, torsions), dim=-2)

    sin_angle, cos_angle = all_torsions.unbind(dim=-1)
    zeros = torch.zeros_like(sin_angle)
    ones = torch.ones_like(sin_angle)
    rotation_x = torch.stack(
        (
            ones,
            zeros,
            zeros,
            zeros,
            cos_angle,
            -sin_angle,
            zeros,
            sin_angle,
            cos_angle,
        ),
        dim=-1,
    ).reshape(*all_torsions.shape[:-1], 3, 3)

    rotated = default_rigid.compose(
        Rigid(rotation_x, torch.zeros_like(default_rigid.translation))
    )

    rotations = [rotated.rotation[..., index, :, :] for index in range(5)]
    translations = [rotated.translation[..., index, :] for index in range(5)]
    previous = Rigid(rotations[4], translations[4])
    for index in range(5, 8):
        current = Rigid(
            rotated.rotation[..., index, :, :], rotated.translation[..., index, :]
        )
        previous = previous.compose(current)
        rotations.append(previous.rotation)
        translations.append(previous.translation)
    relative = Rigid(torch.stack(rotations, dim=-3), torch.stack(translations, dim=-2))
    return backbone.unsqueeze_group().compose(relative)


def frames_to_atom14_positions(
    frames: Rigid,
    aatype: torch.Tensor,
    ref_atom_group: torch.Tensor,
    ref_atom_pos: torch.Tensor,
    ref_atom_mask: torch.Tensor,
) -> torch.Tensor:
    groups = ref_atom_group[aatype]
    gather_rotation = groups[..., None, None].expand(*groups.shape, 3, 3)
    gather_translation = groups[..., None].expand(*groups.shape, 3)
    rotation = torch.gather(frames.rotation, dim=-3, index=gather_rotation)
    translation = torch.gather(frames.translation, dim=-2, index=gather_translation)
    atom_frames = Rigid(rotation, translation)
    coords = atom_frames.apply(ref_atom_pos[aatype])
    return coords * ref_atom_mask[aatype].unsqueeze(-1)


class StructureModule(nn.Module):
    """AlphaFold2 structure module."""

    def __init__(
        self,
        channel_s: int = 384,
        channel_z: int = 128,
        num_layer: int = 8,
        num_head: int = 12,
        num_scalar_qk: int = 16,
        num_scalar_v: int = 16,
        num_point_qk: int = 4,
        num_point_v: int = 8,
        num_layer_in_transition: int = 3,
        dropout: float = 0.1,
        sidechain_channel: int = 128,
        sidechain_num_layer: int = 2,
        num_torsion: int = 7,
        position_scale: float = 10.0,
        epsilon: float = 1e-8,
        inf: float = 1e5,
    ) -> None:
        super().__init__()
        self.num_layer = num_layer
        self.position_scale = position_scale
        self.epsilon = epsilon

        self.layer_norm_s = LayerNorm(channel_s)
        self.layer_norm_z = LayerNorm(channel_z)
        self.linear_s = Linear(channel_s, channel_s)
        self.ipa = InvariantPointAttention(
            channel_s=channel_s,
            channel_z=channel_z,
            num_head=num_head,
            num_scalar_qk=num_scalar_qk,
            num_scalar_v=num_scalar_v,
            num_point_qk=num_point_qk,
            num_point_v=num_point_v,
            epsilon=epsilon,
            inf=inf,
        )
        self.dropout_ipa = nn.Dropout(dropout)
        self.layer_norm_ipa = LayerNorm(channel_s)
        self.transition = StructureTransition(channel_s, num_layer_in_transition, dropout)
        self.backbone_update = Linear(channel_s, 6, init="final")
        self.torsion = TorsionModule(
            channel_s, sidechain_channel, sidechain_num_layer, num_torsion
        )

        self.register_buffer(
            "default_frames",
            torch.tensor(rc.restype_rigid_group_default_frame),
            persistent=False,
        )
        self.register_buffer(
            "ref_atom_group",
            torch.tensor(rc.restype_atom14_to_rigid_group, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "ref_atom_pos",
            torch.tensor(rc.restype_atom14_rigid_group_positions),
            persistent=False,
        )
        self.register_buffer(
            "ref_atom_mask",
            torch.tensor(rc.restype_atom14_mask, dtype=torch.bool),
            persistent=False,
        )

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        aatype: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Run the complete IPA/side-chain module in float32.

        The surrounding LM and Pairformer may use bf16 autocast, but AlphaFold's
        regression structure path is kept outside autocast for stable geometry
        and attention updates.
        """

        with torch.autocast(device_type=s.device.type, enabled=False):
            return self._forward(s.float(), z.float(), aatype, seq_mask)

    def _forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        aatype: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """See Algorithm 20 in the AlphaFold2 SI

        Parameters
        ----------
        s : torch.Tensor
            Single representation of shape [B, L, C_s].
        z : torch.Tensor
            Pair representation of shape [B, L, L, C_z].
        aatype : torch.Tensor
            Amino acid type of shape [B, L].
        mask : torch.Tensor
            Sequence mask of shape [B, L].
        """

        aatype = aatype.long()

        # Line 1
        s_init = self.layer_norm_s(s)
        # Line 2
        z = self.layer_norm_z(z)
        # Line 3
        s = self.linear_s(s_init)
        # Line 4
        T = Rigid.identity(s.shape[:-1], s.device)

        trajectories: list[torch.Tensor] = []
        angles: list[torch.Tensor] = []
        unnormalized_angles: list[torch.Tensor] = []
        sidechain_frames: list[torch.Tensor] = []
        atom_positions: list[torch.Tensor] = []

        # Line 5
        for _ in range(self.num_layer):
            # Line 6
            s = s + self.ipa(s, z, T, mask)
            # Line 7
            s = self.layer_norm_ipa(self.dropout_ipa(s))

            # Line 8-9: Transition
            s = self.transition(s)

            # Line 10: Update backbone
            T = T.compose_update(self.backbone_update(s))

            # Line 11-14: Predict side chain and backbone torsion angles.
            a = self.torsion(s, s_init)

            # Line 24: Compute atom positions from torsion angles and backbone frames.
            a_norm = a / (
                a.square().sum(dim=-1, keepdim=True).clamp(min=self.epsilon).sqrt()
            )
            T_scaled = T.scale_translation(self.position_scale)
            frames = torsion_angles_to_frames(
                T_scaled, a_norm, aatype, self.default_frames
            )
            atom_pos = frames_to_atom14_positions(
                frames, aatype, self.ref_atom_group, self.ref_atom_pos, self.ref_atom_mask
            )
            atom_pos = atom_pos * mask[..., None, None]

            trajectories.append(T_scaled.to_tensor_4x4())
            angles.append(a_norm)
            unnormalized_angles.append(a)
            sidechain_frames.append(frames.to_tensor_4x4())
            atom_positions.append(atom_pos)
            T = T.stop_rotation_gradient()

        structure: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "act": s,
            "traj": torch.stack(trajectories),
            "sidechains": {
                "angles": torch.stack(angles),
                "unnormalized_angles": torch.stack(unnormalized_angles),
                "frames": torch.stack(sidechain_frames),
                "atom_pos": torch.stack(atom_positions),
            },
            "coords": atom_positions[-1],
            "rigids": trajectories[-1],
        }
        return structure
