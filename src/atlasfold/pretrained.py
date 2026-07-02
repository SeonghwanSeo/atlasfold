import copy
from pathlib import Path

import torch

from atlasfold.configs.atlasfold import monomer_config
from atlasfold.configs.atlasfold_multimer import multimer_config
from atlasfold.model import (
    AtlasFold,
    AtlasFold_Multimer,
    AtlasFoldConfig,
    AtlasFoldMultimerConfig,
)

config_dict = {
    "atlasfold": monomer_config,
    "atlasfold-base": monomer_config,
    "atlasfold-3b-base": monomer_config,
    "atlasfold-m": multimer_config,
    "atlasfold-multimer": multimer_config,
    "atlasfold-3b-multimer": multimer_config,
}


def download_model_weights(model_name: str, cache_dir: str | None = None) -> str:
    raise NotImplementedError(
        "AtlasFold HuggingFace checkpoint download is not implemented yet. "
        "Pass a local checkpoint path via `model_path`."
    )


def _load_state_dict(model_path: str | Path) -> dict:
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint)!r}.")
    return {key.removeprefix("model."): value for key, value in checkpoint.items()}


def load_model(
    model_name: str = "atlasfold-3b-base",
    device: torch.device | str | None = None,
    *,
    cache_dir: str | None = None,
    lm_path: str | None = None,
    model_path: str | None = None,
    dtype: torch.dtype | None = None,
) -> AtlasFold | AtlasFold_Multimer:
    """Load a pretrained AtlasFold model by name.

    Parameters
    ----------
    model_name : str
        The name of the pretrained model to load. Options include:
            - "atlasfold-3b-base"
            - "atlasfold-3b-multimer"
    device : str | torch.device, optional
        The device to load the model onto.
    cache_dir : str, optional
        Directory used when downloading model weights in the future.
    lm_path : str, optional
        Path to the local pretrained AtlasLM checkpoint.
    model_path : str, optional
        Path to the local pretrained AtlasFold model checkpoint.
    dtype : torch.dtype, optional
        Model precision. Defaults to bfloat16 on CUDA and float32 on CPU.

    Returns
    -------
    model : AtlasFold | AtlasFold_Multimer
        The loaded AtlasFold model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if dtype is None:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if model_name not in config_dict:
        raise ValueError(
            f"Unknown model name: {model_name}. "
            f"Available models: {list(config_dict.keys())}"
        )

    cfg = copy.deepcopy(config_dict[model_name])
    cfg.lm_path = lm_path

    if model_path is None:
        model_path = download_model_weights(model_name, cache_dir)

    state_dict = _load_state_dict(model_path)
    if isinstance(cfg, AtlasFoldMultimerConfig):
        return AtlasFold_Multimer.from_pretrained(
            state_dict=state_dict,
            config=cfg,
            device=device,
            dtype=dtype,
        )
    if isinstance(cfg, AtlasFoldConfig):
        return AtlasFold.from_pretrained(
            state_dict=state_dict,
            config=cfg,
            device=device,
            dtype=dtype,
        )

    raise TypeError(f"Unsupported AtlasFold config type: {type(cfg)!r}.")
