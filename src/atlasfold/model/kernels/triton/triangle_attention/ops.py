# Copyright 2026 Korea Advanced Institute of Science and Technology (KAIST)
# Adapted for AtlasFold: fixed launches, native tails, and model-specific masking.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public API for fused starting- and ending-node triangle attention."""

import torch

from .._common.layouts import interleave_kv
from ..gated_projection import gated_output_projection_blc
from .kernels import triangle_attn_forward


def precompute(W_ln, B_ln, WQ, WK, WV, W_proj_z, B_proj_z, H, D):
    """Pack projection weights into layouts consumed by forward."""
    WQ_c = WQ.contiguous()  # (C_in, H*D) matmul convention, no fold
    WKV_c = interleave_kv(WK, WV, H, D).contiguous()  # (C_in, 2*H*D), no fold
    # Triton's reshape/split lowering used by the interleaved K/V fast path can
    # issue an illegal access for the Apo-module shape D=16 on Hopper.  Keep the
    # original K and V layouts as well so the kernel can select a genuine
    # two-matmul Triton specialization for D=16.  D>=32 continues to use WKV_c.
    WK_c = WK.contiguous() if D == 16 else None
    WV_c = WV.contiguous() if D == 16 else None
    WZ_c = W_proj_z.t().contiguous()  # (C_in, H) -> (H, C_in) for the bias kernel

    # bias-proj bias as an (H,) fp32 tensor (K-Fold's bias-proj is LinearNoBias).
    if B_proj_z is None:
        BZ = torch.zeros(H, device=WQ.device, dtype=torch.float32)
    else:
        BZ = B_proj_z.float().contiguous()

    return {
        "WQ_c": WQ_c,
        "WKV_c": WKV_c,
        "WK_c": WK_c,
        "WV_c": WV_c,
        "WZ_c": WZ_c,
        "BZ": BZ,
        # x̃ = LN(x) is computed once in `forward`; the kernels and the gate
        # epilogue all consume it, so the LN affine is needed here.
        "W_ln": W_ln,
        "B_ln": B_ln,
    }


def forward(
    X,
    pre,
    *,
    mask=None,
    scale=1.0,
    neg_inf=-1e9,
    eps=1e-5,
    W_proj_g,
    B_proj_g,
    W_proj_o,
    B_proj_o,
):
    """Run triangle attention on batched or unbatched pair features.

    The output rank matches X; the kernels mask native sequence tails.
    Gate/output weights use contiguous (input, output) matmul layouts.
    """
    assert B_proj_g is None and B_proj_o is None, (
        "fused gate path assumes bias-free linear_g / linear_o (K-Fold LinearNoBias)"
    )
    unbatched = X.ndim == 3
    if unbatched:
        X = X.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)
    elif X.ndim != 4:
        raise ValueError(f"X must have shape (N,N,C) or (B,N,N,C), got {X.shape}")
    B, N, N2, C_in = X.shape
    if N != N2:
        raise ValueError(f"triangle attention requires square pair axes, got {X.shape}")
    H = pre["WZ_c"].shape[0]
    D = pre["WQ_c"].shape[1] // H
    if mask is not None and mask.shape != (B, N, N):
        raise ValueError(f"mask must have shape {(B, N, N)}, got {mask.shape}")
    # x̃ = LN(x) computed once and shared by the kernels and the gate.
    X_ln = torch.nn.functional.layer_norm(
        X.float(),
        (C_in,),
        pre["W_ln"].float(),
        pre["B_ln"].float() if pre["B_ln"] is not None else None,
        eps,
    ).to(X.dtype).to(pre["WQ_c"].dtype)

    O_attn = torch.empty(B, N, N, H * D, device=X.device, dtype=X_ln.dtype)
    triangle_attn_forward(
        X_ln,
        pre["WQ_c"],
        pre["WKV_c"],
        pre["WZ_c"],
        pre["BZ"],
        O_attn,
        scale=scale,
        neg_inf=neg_inf,
        mask=mask,
        WK_c=pre["WK_c"],
        WV_c=pre["WV_c"],
    )
    # Tile the projection so C=256 does not require full weight matrices in SMEM.
    output = gated_output_projection_blc(
        X_ln.reshape(B, N * N, C_in),
        O_attn.reshape(B, N * N, H * D).contiguous(),
        W_proj_g,
        W_proj_o,
    ).reshape(B, N, N, C_in)
    return output.squeeze(0) if unbatched else output
