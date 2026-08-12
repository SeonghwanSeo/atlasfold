import copy
from pathlib import Path

import torch

from atlasfold.configs.atlasfold import monomer_config
from atlasfold.configs.atlasfold_ipa import monomer_ipa_config
from atlasfold.configs.atlasfold_multimer import multimer_config
from atlasfold.configs.atlasfold_multimer_ipa import multimer_ipa_config
from atlasfold.model import (
    AtlasFold,
    AtlasFold_IPA,
    AtlasFold_Multimer,
    AtlasFoldConfig,
    AtlasFoldIPAConfig,
    AtlasFoldMultimer_IPA,
    AtlasFoldMultimerConfig,
    AtlasFoldMultimerIPAConfig,
)
from atlaslm.pretrained import download_model_weights as download_lm_weights

ATLASFOLD_260703 = "atlasfold-260703"
ATLASFOLD_M_260725 = "atlasfold-m-260725"
ATLASFOLD_IPA = "atlasfold-ipa"
ATLASFOLD_MULTIMER_IPA = "atlasfold-multimer-ipa"

SUPPORTED_MODELS = [
    ATLASFOLD_260703,
    ATLASFOLD_M_260725,
    ATLASFOLD_IPA,
    ATLASFOLD_MULTIMER_IPA,
]

MODEL_NAME_MAP = {
    "atlasfold-260703": ATLASFOLD_260703,
    "atlasfold-m-260725": ATLASFOLD_M_260725,
    "atlasfold-ipa": ATLASFOLD_IPA,
    "atlasfold-multimer-ipa": ATLASFOLD_MULTIMER_IPA,
}

MODEL_CONFIGS = {
    ATLASFOLD_260703: monomer_config,
    ATLASFOLD_M_260725: multimer_config,
    ATLASFOLD_IPA: monomer_ipa_config,
    ATLASFOLD_MULTIMER_IPA: multimer_ipa_config,
}

WEIGHT_PATHS = {
    ATLASFOLD_260703: (
        "SeonghwanSeo/atlasfold-260703",
        "weights/atlasfold-260703.pth",
    ),
    ATLASFOLD_M_260725: (
        "SeonghwanSeo/atlasfold-m-260725",
        "weights/atlasfold-m-260725.pth",
    ),
}


def get_model_name(model_name: str) -> str:
    if model_name not in MODEL_NAME_MAP:
        raise ValueError(
            f"Unknown model name: {model_name}. "
            f"Supported models: {', '.join(SUPPORTED_MODELS)}"
        )
    return MODEL_NAME_MAP[model_name]


def download_model_weights(
    model_name: str,
    cache_dir: str | Path | None = None,
) -> str:
    from huggingface_hub import hf_hub_download

    model_name = get_model_name(model_name)
    if model_name not in WEIGHT_PATHS:
        raise ValueError(
            f"No published weights are configured for {model_name!r}; "
            "provide model_path when loading this model."
        )
    repo_id, filename = WEIGHT_PATHS[model_name]
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
    )


def _load_state_dict(model_path: str | Path) -> dict:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint)!r}.")
    return {key.removeprefix("model."): value for key, value in checkpoint.items()}


def load_model(
    model_name: str = ATLASFOLD_260703,
    device: torch.device | str | None = None,
    *,
    cache_dir: str | Path | None = None,
    lm_path: str | Path | None = None,
    model_path: str | Path | None = None,
) -> AtlasFold | AtlasFold_Multimer | AtlasFold_IPA | AtlasFoldMultimer_IPA:
    """Load a pretrained AtlasFold model by name.

    Parameters
    ----------
    model_name : str
        The name of the pretrained model to load. Options include:
            - "atlasfold-260703"
            - "atlasfold-m-260725"
            - "atlasfold-ipa" (requires model_path)
            - "atlasfold-multimer-ipa" (requires model_path)
    device : str | torch.device, optional
        The device to load the model onto.
    cache_dir : str, optional
        Directory used to cache downloaded model weights.
    lm_path : str, optional
        Path to the local pretrained AtlasLM checkpoint.
    model_path : str, optional
        Path to the local pretrained AtlasFold model checkpoint.
    dtype : torch.dtype, optional
        Model precision. Defaults to bfloat16 on CUDA and float32 on CPU.

    Returns
    -------
    model : AtlasFold | AtlasFold_Multimer | AtlasFold_IPA | AtlasFoldMultimer_IPA
        The loaded AtlasFold model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model_name = get_model_name(model_name)
    cfg = copy.deepcopy(MODEL_CONFIGS[model_name])

    if model_path is None:
        model_path = download_model_weights(model_name, cache_dir)
    if lm_path is None:
        lm_path = download_lm_weights(cfg.lm_name, cache_dir)
    cfg.lm_path = str(lm_path)

    state_dict = _load_state_dict(model_path)
    if isinstance(cfg, AtlasFoldMultimerConfig):
        return AtlasFold_Multimer.from_pretrained(
            state_dict=state_dict, config=cfg, device=device
        )
    if isinstance(cfg, AtlasFoldConfig):
        return AtlasFold.from_pretrained(state_dict=state_dict, config=cfg, device=device)
    if isinstance(cfg, AtlasFoldMultimerIPAConfig):
        return AtlasFoldMultimer_IPA.from_pretrained(
            state_dict=state_dict, config=cfg, device=device
        )
    if isinstance(cfg, AtlasFoldIPAConfig):
        return AtlasFold_IPA.from_pretrained(
            state_dict=state_dict, config=cfg, device=device
        )

    raise TypeError(f"Unsupported AtlasFold config type: {type(cfg)!r}.")
