import dataclasses
from pathlib import Path

import torch

from atlaslm.model import AtlasLM


@dataclasses.dataclass(kw_only=True)
class AtlasLMConfig:
    """Configuration for AtlasLM models."""

    d_model: int
    n_heads: int
    n_layers: int


# MODEL LIST
ATLASLM_600M_BASE = "atlaslm-600m-base"
ATLASLM_3B_BASE = "atlaslm-3b-base"
SUPPORTED_MODELS = [
    ATLASLM_600M_BASE,
    ATLASLM_3B_BASE,
]

# User input -> model name mapping.
MODEL_NAME_MAP = {
    "atlaslm-600m": ATLASLM_600M_BASE,
    "atlaslm-600m-base": ATLASLM_600M_BASE,
    "atlaslm-3b": ATLASLM_3B_BASE,
    "atlaslm-3b-base": ATLASLM_3B_BASE,
}

WEIGHT_PATH = {
    ATLASLM_600M_BASE: (
        "SeonghwanSeo/atlaslm-600m-base",
        "weights/atlaslm_600m_base.pth",
    ),
    ATLASLM_3B_BASE: (
        "SeonghwanSeo/atlaslm-3b-base",
        "weights/atlaslm_3b_base.pth",
    ),
}
MODEL_CONFIG_MAP = {
    ATLASLM_600M_BASE: "600m",
    ATLASLM_3B_BASE: "3b",
}
MODEL_CONFIGS = {
    "600m": AtlasLMConfig(d_model=1152, n_heads=18, n_layers=36),
    "3b": AtlasLMConfig(d_model=2304, n_heads=36, n_layers=48),
}
# TODO: Add the links to the pretrained model checkpoints once they are available.


def get_model_name(model_name: str) -> str:
    if model_name not in MODEL_NAME_MAP:
        raise ValueError(
            f"Unknown model name: {model_name}. "
            f"Supported models: {', '.join(SUPPORTED_MODELS)}"
        )
    return MODEL_NAME_MAP[model_name]


def get_model(model_name: str) -> AtlasLM:
    config_name = MODEL_CONFIG_MAP[model_name]
    config = MODEL_CONFIGS[config_name]
    model = AtlasLM(config.d_model, config.n_heads, config.n_layers)
    return model.eval()


def download_model_weights(model_name: str, cache_dir: str | Path | None = None) -> Path:
    from huggingface_hub import snapshot_download

    repo_id, filename = WEIGHT_PATH[model_name]
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=cache_dir,
    )
    return Path(path) / filename


def load_model(
    model_name: str,
    device: str | torch.device = "cpu",
    cache_dir: str | Path | None = None,
) -> AtlasLM:
    """Load a pretrained ESMC model by name.

    Parameters
    ----------
    model_name : str
        The name of the pretrained model to load. Options include:
        - "atlaslm-600m"
        - "atlaslm-3b"
    device : str | torch.device, optional
        The device to load the model onto, by default "cpu".
    cache_dir : str | Path, optional
        Directory to cache the downloaded model weights, by default None.

    Returns
    -------
    model : AtlasLM
        The loaded AtlasLM model.
    """
    # Canonicalize the model name
    model_name = get_model_name(model_name)

    # Initialize the model architecture
    model = get_model(model_name)
    model.eval()

    # Download the model weights
    path = download_model_weights(model_name, cache_dir)

    # Load the model architecture and weights
    state_dict = torch.load(path, map_location=device, weights_only=True)
    # Remove 'model.' prefix if present
    if all(key.startswith("model.") for key in state_dict.keys()):
        state_dict = {key[len("model.") :]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)

    return model.to(device)
