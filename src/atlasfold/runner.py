import contextlib
import dataclasses

import numpy as np
import torch

from atlasfold.common import featurize, protein
from atlasfold.model import AtlasFold, SamplingConfig


@contextlib.contextmanager
def seed_context(seed: int, device: torch.device):
    if device.type == "cuda":
        with torch.random.fork_rng(device_type="cuda"):
            torch.manual_seed(seed)
            yield
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            yield


@contextlib.contextmanager
def autocast_context(device: torch.device):
    if device.type == "cuda":
        with torch.autocast(device.type, torch.bfloat16, enabled=True):
            yield
    else:
        yield


def default(value, default_value):
    """Return the value if it is not None, otherwise return the default value."""
    return value if value is not None else default_value


@dataclasses.dataclass(kw_only=True)
class ProteinOutput(protein.Protein):
    """A data structure representing a predicted 3D protein structure"""

    name: str
    sequence: str
    coordinates: np.ndarray  # [L, 14, 3]
    b_factors: np.ndarray  # [L,] or [L, 14]
    plddt: np.ndarray  # [L]
    pae: np.ndarray  # [L, L]
    ptm: float
    residue_index: np.ndarray | None = None  # [L], optional residue index

    def __post_init__(self):
        """Validate the input data."""
        super().__post_init__()
        L = len(self.sequence)
        if self.plddt.shape != (L,):
            raise ValueError(
                f"Invalid pLDDT shape: {self.plddt.shape}. "
                f"Expected (L,) where L is the sequence length."
            )
        if self.pae.shape != (L, L):
            raise ValueError(
                f"Invalid PAE shape: {self.pae.shape}. "
                f"Expected (L, L) where L is the sequence length."
            )
        assert self.residue_index is None


class FoldingRunner:
    """A class for running protein folding using the AtlasFold model."""

    def __init__(self, model: AtlasFold):
        self.model: AtlasFold = model
        self.device = self.model.device

    # TODO: add pre-trained model loading from HuggingFace Hub

    def fold(
        self,
        name: str,
        sequence: str,
        num_samples: int = 1,
        *,
        preset: str = "base",
        seed: int = 1,
        num_recycles: int | None = None,
        mlm_prob: float | None = None,
        sampling_config: SamplingConfig | None = None,
    ) -> list[ProteinOutput]:
        feat: dict[str, np.ndarray] = featurize.featurize(sequence)

        # Validate the preset
        if preset not in ["base", "high", "stochastic"]:
            raise ValueError(f"Invalid preset: {preset}")
        # Get the preset settings and override with user-specified values
        settings = self.get_preset_setting(preset)
        settings["num_recycles"] = default(num_recycles, settings["num_recycles"])
        settings["mlm_prob"] = default(mlm_prob, settings["mlm_prob"])
        settings["sampling_config"] = default(
            sampling_config, settings["sampling_config"]
        )

        # Pad the features to the next multiple of 32
        feat = self.pad(feat, 16)

        # Model inference
        out = self.model_run(feat, seed=seed, num_samples=num_samples, **settings)

        length = len(sequence)
        samples = []
        for i in range(num_samples):
            coords = out["sample_coords"][i, :length]  # [L, 14, 3]
            plddt = out["plddt"][i, :length]  # [L]
            b_factor = plddt * 100
            pae = out["pae"][i, :length, :length]  # [L, L]
            ptm = float(out["ptm"][i].item())  # scalar

            sample = ProteinOutput(
                name=name,
                sequence=sequence,
                coordinates=coords,
                b_factors=b_factor,
                plddt=plddt,
                pae=pae,
                ptm=ptm,
            )
            samples.append(sample)
        return samples

    def get_preset_setting(self, preset: str) -> dict:
        num_recycles = 3
        mlm_prob = 0.15
        # TODO: add auto-scaling for num_steps and sigma_max based on sequence length
        sampling_cfg = SamplingConfig(num_steps=100, sigma_max=128)
        stochastic = False
        if preset == "high":
            num_recycles = 6
        elif preset == "stochastic":
            stochastic = True

        return {
            "num_recycles": num_recycles,
            "mlm_prob": mlm_prob,
            "sampling_config": sampling_cfg,
            "stochastic": stochastic,
        }

    def model_run(
        self,
        feat: dict[str, np.ndarray],
        seed: int,
        num_samples: int,
        num_recycles: int,
        mlm_prob: float,
        stochastic: bool,
        sampling_config: SamplingConfig,
    ) -> dict[str, np.ndarray]:
        device = self.device
        feat: dict[str, torch.Tensor] = {
            k: torch.as_tensor(v, device=device) for k, v in feat.items()
        }
        with (
            torch.inference_mode(),
            seed_context(seed, device),
            autocast_context(device),
        ):
            out = self.model.inference(
                feat,
                num_samples=num_samples,
                num_recycles=num_recycles,
                mlm_prob=mlm_prob,
                stochastic=stochastic,
                sampling_config=sampling_config,
            )
        return {k: v.cpu().float().numpy() for k, v in out.items()}

    @staticmethod
    def pad(feat: dict[str, np.ndarray], multiple_of: int = 32) -> dict[str, np.ndarray]:
        """Pad the input features to the specified length."""
        length = feat["aatype_int"].shape[0]
        pad_length = ((length + multiple_of - 1) // multiple_of) * multiple_of
        new_feat: dict[str, np.ndarray] = {}
        pad_len = pad_length - length
        for k, v in feat.items():
            _pad_len = pad_len + 32 if k.startswith("lm.") else pad_len
            if _pad_len == 0:
                new_feat[k] = v
            else:
                pad_width = ((0, _pad_len),) + ((0, 0),) * (v.ndim - 1)
                new_feat[k] = np.pad(v, pad_width, constant_values=0)

        return new_feat
