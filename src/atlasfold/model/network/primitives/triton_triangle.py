"""Inference-only adapters for the KFold-derived triangle kernels."""

import torch


@torch.compiler.disable
def triton_triangle(module, x, mask, *, direction=None):
    """Apply a fused triangle operator, caching packed inference weights."""
    if torch.is_grad_enabled():
        raise RuntimeError("The Triton triangle backend is inference-only.")
    if not x.is_cuda:
        raise RuntimeError("The Triton triangle backend requires CUDA.")
    dtype = (
        torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
    )
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            "The Triton triangle backend requires BF16 or FP16 inputs/autocast."
        )
    params = tuple(module.parameters())
    # Inference tensors have no mutation counter: repack those rather than reuse
    # weights that could have changed without invalidating the cache.
    cacheable = not any(torch.is_inference(p) for p in params)
    key = (
        (
            x.device,
            dtype,
            tuple((id(p), p.data_ptr(), p.dtype, p.device, p._version) for p in params),
        )
        if cacheable
        else None
    )
    cached = getattr(module, "_triton_cache", None)
    length, channel = x.shape[-2:]
    with torch.autocast("cuda", enabled=False):
        if cached is None or key is None or cached[0] != key:
            if direction is None:
                from atlasfold.model.kernels.triton.triangle_attention import precompute

                q, k, v = module.linear_qkv.weight.to(dtype).chunk(3, dim=0)
                pre = precompute(
                    module.layernorm.weight,
                    module.layernorm.bias,
                    q.t(),
                    k.t(),
                    v.t(),
                    module.linear_bias.weight.to(dtype).t(),
                    None,
                    module.num_heads,
                    module.channel_hidden,
                )
                pre["gate_weight"] = module.linear_g.weight.to(dtype).t().contiguous()
                pre["output_weight"] = module.linear_out.weight.to(dtype).t().contiguous()
            else:
                from atlasfold.model.kernels.triton.triangle_multiplication import (
                    precompute,
                )

                pre = precompute(
                    module.layernorm_in.weight,
                    module.layernorm_in.bias,
                    module.linear_in.weight.to(dtype),
                    module.linear_g_in.weight.to(dtype),
                    module.layernorm_out.weight,
                    module.layernorm_out.bias,
                    module.linear_out.weight.to(dtype),
                    module.linear_g_out.weight.to(dtype),
                )
            if cacheable:
                module._triton_cache = (key, pre)
        else:
            pre = cached[1]
        # LayerNorm must see the original input: autocast rounds the normalized
        # projection operands, not the unnormalized residual stream.
        flat_x = x.reshape(-1, length, length, channel)
        flat_mask = torch.broadcast_to(mask.bool(), x.shape[:-1]).reshape(
            -1, length, length
        )
        if direction is None:
            from atlasfold.model.kernels.triton.triangle_attention import forward

            output = forward(
                flat_x,
                pre,
                mask=flat_mask,
                scale=module.scale,
                neg_inf=-module.inf,
                eps=module.layernorm.eps,
                W_proj_g=pre["gate_weight"],
                B_proj_g=None,
                W_proj_o=pre["output_weight"],
                B_proj_o=None,
            )
        else:
            from atlasfold.model.kernels.triton.triangle_multiplication import forward

            output = forward(
                flat_x,
                pre,
                direction=direction,
                mask=flat_mask,
                eps=module.layernorm_in.eps,
            )
    return output.reshape(x.shape)
