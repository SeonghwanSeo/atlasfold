"""Kernel backend selection utilities."""

import importlib.util
import warnings
from typing import Literal

KERNEL_BACKENDS = ("auto", "torch", "cuequiv", "triton")


def triton_available(device_type: str) -> bool:
    return device_type == "cuda" and importlib.util.find_spec("triton") is not None


def cuequiv_available(device_type: str) -> bool:
    return (
        device_type == "cuda"
        and importlib.util.find_spec("cuequivariance_torch") is not None
    )


def select_kernel_backend(
    backend: Literal["torch", "cuequiv", "triton", "auto"],
    device_type: str,
) -> str:
    """Resolve auto, torch, cuequiv, or triton to an available kernel backend."""
    if backend not in KERNEL_BACKENDS:
        raise ValueError(
            f"Unknown kernel backend {backend!r}; expected one of {KERNEL_BACKENDS}."
        )
    if backend == "triton":
        if not triton_available(device_type):
            raise RuntimeError("Kernel backend 'triton' requires CUDA and Triton.")
        return "triton"
    if backend == "torch":
        return "torch"
    if backend == "auto" and triton_available(device_type):
        return "triton"
    if backend in ("auto", "cuequiv") and cuequiv_available(device_type):
        return "cuequiv"
    if backend == "cuequiv":
        warnings.warn(
            f"Kernel backend 'cuequiv' is unavailable on {device_type}; using torch.",
            RuntimeWarning,
            stacklevel=2,
        )
    return "torch"
