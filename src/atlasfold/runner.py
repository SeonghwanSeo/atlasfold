import contextlib
import dataclasses

import numpy as np
import torch

from atlasfold.common import featurize, file_io
from atlasfold.model import AtlasFold, SamplingConfig


@contextlib.contextmanager
def seed_context(seed: int, device: torch.device):
    old_cpu_state = torch.get_rng_state()
    is_cuda = device.type == "cuda"
    if is_cuda:
        old_cuda_states = torch.cuda.get_rng_state(device)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        yield
    finally:
        torch.set_rng_state(old_cpu_state)
        if is_cuda:
            torch.cuda.set_rng_state(old_cuda_states, device)


@dataclasses.dataclass
class ProteinOutput:
    """A data structure representing a predicted 3D protein structure"""

    name: str
    sequence: str
    coordinates: np.ndarray  # [L, 14, 3]
    plddt: np.ndarray | None  # [L]
    pae: np.ndarray | None  # [L, L]
    ptm: float | None = None

    def __post_init__(self):
        """Validate the input data."""
        L = len(self.sequence)
        if self.coordinates.shape not in [(L, 14, 3)]:
            raise ValueError(
                f"Invalid coordinates shape: {self.coordinates.shape}. "
                f"Expected (L, 14, 3) where L is the sequence length."
            )
        if self.plddt is not None and self.plddt.shape != (L,):
            raise ValueError(
                f"Invalid pLDDT shape: {self.plddt.shape}. "
                f"Expected (L,) where L is the sequence length."
            )
        if self.pae is not None and self.pae.shape != (L, L):
            raise ValueError(
                f"Invalid PAE shape: {self.pae.shape}. "
                f"Expected (L, L) where L is the sequence length."
            )

    def __len__(self):
        """Return the number of residues in the structure."""
        return len(self.sequence)

    @property
    def num_residues(self) -> int:
        """Return the number of residues in the structure."""
        return len(self.sequence)

    def to_pdb(self) -> str:
        return file_io.to_pdb(self.name, self.sequence, self.coordinates, self.plddt)

    def to_mmcif(self) -> str:
        return file_io.to_mmcif(self.name, self.sequence, self.coordinates, self.plddt)


class FoldingRunner:
    def __init__(self, model: AtlasFold):
        self.model: AtlasFold = model
        self.device = self.model.device

    def fold(
        self,
        name: str,
        sequence: str,
        num_samples: int = 1,
        *,
        preset: str = "full",  # 'flash', 'full', 'stochastic'
        seed: int = 1,
        num_recycles: int | None = None,
        mlm_prob: float | None = None,
        sampling_config: SamplingConfig | None = None,
    ) -> list[ProteinOutput]:
        feat: dict[str, np.ndarray] = featurize.featurize(sequence)

        preset_cfg = self.get_preset(preset)
        mode = preset_cfg["mode"]
        if num_recycles is None:
            num_recycles = preset_cfg["num_recycles"]
        if mlm_prob is None:
            mlm_prob = preset_cfg["mlm_prob"]
        if sampling_config is None:
            sampling_config = preset_cfg["sampling_config"]

        # Pad the features to the next multiple of 32
        feat = self.pad(feat, 32)

        # Model inference
        out = self.model_run(
            feat, seed, mode, num_samples, num_recycles, mlm_prob, sampling_config
        )

        length = len(sequence)
        samples = []
        for i in range(num_samples):
            coords = out["sample_coords"][i, :length]  # [L, 14, 3]
            plddt, pae, ptm = None, None, None
            if "plddt" in out:
                plddt = out["plddt"][i, :length]  # [L]
                plddt = plddt * 100  # Scale to [0, 100]
            if "pae" in out:
                pae = out["pae"][i, :length, :length]  # [L, L]
            if "ptm" in out:
                ptm = float(out["ptm"][i].item())  # scalar

            sample = ProteinOutput(
                name=name,
                sequence=sequence,
                coordinates=coords,
                plddt=plddt,
                pae=pae,
                ptm=ptm,
            )
            samples.append(sample)
        return samples

    def get_preset(self, preset: str) -> dict:
        if preset == "flash":
            mode = "flash"
            num_recycles = -1
            mlm_prob = 0.0
            sampling_cfg = SamplingConfig(num_steps=75, sigma_max=32.0)
        elif preset == "full":
            mode = "full"
            num_recycles = 3
            mlm_prob = 0.15
            sampling_cfg = SamplingConfig(num_steps=100)
        elif preset == "stochastic":
            mode = "base"
            num_recycles = 3
            mlm_prob = 0.15
            sampling_cfg = SamplingConfig(num_steps=100)
        else:
            raise ValueError(f"Invalid preset: {preset}")
        return {
            "mode": mode,
            "num_recycles": num_recycles,
            "mlm_prob": mlm_prob,
            "sampling_config": sampling_cfg,
        }

    def model_run(
        self,
        feat: dict[str, np.ndarray],
        seed: int,
        mode: str,
        num_samples: int,
        num_recycles: int,
        mlm_prob: float,
        sampling_config: SamplingConfig | None,
    ) -> dict[str, np.ndarray]:
        device = self.device
        feat: dict[str, torch.Tensor] = {
            k: torch.as_tensor(v, device=device) for k, v in feat.items()
        }
        is_cuda = device.type == "cuda"
        with (
            torch.inference_mode(),
            seed_context(seed, device),
            (
                torch.autocast(device.type, torch.bfloat16, enabled=True)
                if is_cuda
                else contextlib.nullcontext()
            ),
        ):
            out = self.model.inference(
                feat,
                mode=mode,
                num_samples=num_samples,
                num_recycles=num_recycles,
                mlm_prob=mlm_prob,
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
