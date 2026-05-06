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


MODEL_CONFIGS = {
    "atlaslm-600m": AtlasLMConfig(d_model=1152, n_heads=18, n_layers=36),
    "atlaslm-3b": AtlasLMConfig(d_model=2304, n_heads=36, n_layers=48),
}

MODEL_NAME_MAP = {
    "atlaslm-600m": "atlaslm-600m",
    "atlaslm_600m": "atlaslm-600m",
    "600m": "atlaslm-600m",
    "atlaslm-3b": "atlaslm-3b",
    "atlaslm_3b": "atlaslm-3b",
    "3b": "atlaslm-3b",
}
SUPPORTED_MODELS = [
    "atlaslm-600m",
    "atlaslm-3b",
]
# TODO: Add the links to the pretrained model checkpoints once they are available.


def get_model_name(model_name: str) -> str:
    """Get the standardized model name for a given input.

    Parameters
    ----------
    model_name : str
        The input model name, which can be in various formats.

    Returns
    -------
    str
        The standardized model name corresponding to the input.
    """
    if model_name not in MODEL_NAME_MAP:
        raise ValueError(
            f"Unknown model name: {model_name}. "
            f"Supported models: {', '.join(SUPPORTED_MODELS)}"
        )
    return MODEL_NAME_MAP[model_name]


def get_model(model_name: str) -> AtlasLM:
    """Load a pretrained AtlasLM model by name.

    Parameters
    ----------
    model_name : str
        The name of the pretrained model to load. Options include:
        - "atlaslm-600m"
        - "atlaslm-3b"

    Returns
    -------
    model: AtlasLM
        The loaded AtlasLM model.
    """
    model_name = get_model_name(model_name)
    config = MODEL_CONFIGS[model_name]
    model = AtlasLM(config.d_model, config.n_heads, config.n_layers)
    return model.eval()


def load_model(
    model_name: str,
    path: str | Path,
    device: str | torch.device = "cpu",
) -> AtlasLM:
    """Load a pretrained ESMC model by name.

    Parameters
    ----------
    model_name : str
        The name of the pretrained model to load. Options include:
        - "atlaslm-600m"
        - "atlaslm-3b"
    path : str | Path
        The path to the pretrained model checkpoint.
    device : str | torch.device, optional
        The device to load the model onto, by default "cpu".

    Returns
    -------
    model : AtlasLM
        The loaded AtlasLM model.
    """
    model = get_model(model_name)
    state_dict = torch.load(path, map_location=device, weights_only=True)

    # Remove 'model.' prefix if present
    if all(key.startswith("model.") for key in state_dict.keys()):
        state_dict = {key[len("model.") :]: value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)
    return model.to(device)
