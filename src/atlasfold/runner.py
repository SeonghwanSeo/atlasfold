import contextlib
import dataclasses
from collections.abc import Sequence

import numpy as np
import torch
from tqdm import tqdm

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
        *,
        num_samples: int = 1,
        preset: str = "base",
        seed: int = 1,
        num_recycles: int | None = None,
        mlm_prob: float | None = None,
        sampling_config: SamplingConfig | None = None,
        length_buckets: Sequence[int] | None = None,
    ) -> list[ProteinOutput]:
        return self.fold_batch(
            [(name, sequence)],
            num_samples=num_samples,
            preset=preset,
            seed=seed,
            num_recycles=num_recycles,
            mlm_prob=mlm_prob,
            sampling_config=sampling_config,
            length_buckets=length_buckets,
        )[0]

    def fold_batch(
        self,
        inputs: Sequence[tuple[str, str]],
        *,
        num_samples: int = 1,
        preset: str = "base",
        seed: int = 1,
        num_recycles: int | None = None,
        mlm_prob: float | None = None,
        sampling_config: SamplingConfig | None = None,
        length_buckets: Sequence[int] | None = None,
        max_tokens_per_batch: int = 1024,
        disable_tqdm: bool = False,
    ) -> list[list[ProteinOutput]]:
        """Fold a list of sequences using bucketed batched inference.

        By default, residue budgets use ``featurize.DEFAULT_BUCKETS``:
        32, 64, 128, 192, 256, 384, 512, 640, 768, ...
        ``max_tokens_per_batch`` caps ``batch_size * residue_budget`` for each
        model call.

        The returned list has the same order as ``inputs``. Each element is the
        list of ``num_samples`` predictions for the corresponding input.
        """
        inputs = list(inputs)
        if len(inputs) == 0:
            return []
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")
        if max_tokens_per_batch <= 0:
            raise ValueError(
                f"max_tokens_per_batch must be positive, got {max_tokens_per_batch}."
            )

        settings = self._get_run_settings(
            preset=preset,
            num_recycles=num_recycles,
            mlm_prob=mlm_prob,
            sampling_config=sampling_config,
        )

        bucketed_inputs = self._bucket_inputs(inputs, length_buckets)
        outputs: list[list[ProteinOutput] | None] = [None] * len(inputs)

        total_count = len(inputs)
        curr_count = 0
        for bucket_length in (
            pbar := tqdm(
                sorted(bucketed_inputs.keys()),
                desc="Folding",
                disable=disable_tqdm,
                unit="bucket",
            )
        ):
            bucket_items = bucketed_inputs[bucket_length]
            batch_size = max(1, max_tokens_per_batch // bucket_length)

            for start in range(0, len(bucket_items), batch_size):
                chunk = bucket_items[start : start + batch_size]
                sequences = [sequence for _, _, sequence in chunk]
                feat = self._make_batch_features(sequences, bucket_length)

                out = self.model_run(
                    feat,
                    seed=seed,
                    num_samples=num_samples,
                    **settings,
                )

                for batch_idx, (input_idx, name, sequence) in enumerate(chunk):
                    outputs[input_idx] = self._make_protein_outputs(
                        name=name,
                        sequence=sequence,
                        out=out,
                        batch_idx=batch_idx,
                        num_samples=num_samples,
                    )
                curr_count += len(chunk)
                pbar.set_postfix({"progress": f"{curr_count}/{total_count}"})

        completed_outputs = []
        for output in outputs:
            if output is None:
                raise RuntimeError("Some batched folding outputs were not produced.")
            completed_outputs.append(output)
        return completed_outputs

    def _get_run_settings(
        self,
        *,
        preset: str,
        num_recycles: int | None,
        mlm_prob: float | None,
        sampling_config: SamplingConfig | None,
    ) -> dict:
        if preset not in ["base", "high", "stochastic"]:
            raise ValueError(f"Invalid preset: {preset}")

        settings = self.get_preset_setting(preset)
        settings["num_recycles"] = default(num_recycles, settings["num_recycles"])
        settings["mlm_prob"] = default(mlm_prob, settings["mlm_prob"])
        settings["sampling_config"] = default(
            sampling_config, settings["sampling_config"]
        )
        return settings

    def get_preset_setting(self, preset: str) -> dict:
        num_recycles = 4
        mlm_prob = 0.15
        # TODO: add auto-scaling for num_steps and sigma_max based on sequence length
        sampling_cfg = SamplingConfig(num_steps=100, sigma_max=160)
        stochastic = False
        if preset == "high":
            num_recycles = 8
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

    @classmethod
    def _bucket_inputs(
        cls,
        inputs: Sequence[tuple[str, str]],
        length_buckets: Sequence[int] | None,
    ) -> dict[int, list[tuple[int, str, str]]]:
        bucketed_inputs: dict[int, list[tuple[int, str, str]]] = {}
        for input_idx, (name, sequence) in enumerate(inputs):
            if len(sequence) == 0:
                raise ValueError(f"Input {input_idx} ({name!r}) has an empty sequence.")
            bucket_length = cls.get_length_bucket(len(sequence), length_buckets)
            bucketed_inputs.setdefault(bucket_length, []).append(
                (input_idx, name, sequence)
            )
        return bucketed_inputs

    @staticmethod
    def get_length_bucket(
        length: int,
        length_buckets: Sequence[int] | None = None,
    ) -> int:
        """Return the smallest residue budget that can contain ``length``."""
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}.")

        if length_buckets is None:
            for bucket in featurize.DEFAULT_BUCKETS:
                if length <= bucket:
                    return bucket
            return ((length + 127) // 128) * 128

        buckets = sorted(set(length_buckets))
        if len(buckets) == 0:
            raise ValueError("length_buckets must contain at least one bucket.")
        if buckets[0] <= 0:
            raise ValueError("length_buckets must contain only positive integers.")

        for bucket in buckets:
            if length <= bucket:
                return bucket

        raise ValueError(
            f"Sequence length {length} exceeds the largest configured length bucket "
            f"({buckets[-1]})."
        )

    @classmethod
    def _make_batch_features(
        cls,
        sequences: Sequence[str],
        bucket_length: int,
    ) -> dict[str, np.ndarray]:
        feats = [
            cls.pad_to_length(featurize.featurize(sequence), bucket_length)
            for sequence in sequences
        ]
        return {k: np.stack([feat[k] for feat in feats], axis=0) for k in feats[0]}

    @staticmethod
    def _make_protein_outputs(
        *,
        name: str,
        sequence: str,
        out: dict[str, np.ndarray],
        batch_idx: int,
        num_samples: int,
    ) -> list[ProteinOutput]:
        length = len(sequence)
        samples = []
        for sample_idx in range(num_samples):
            coords = out["sample_coords"][batch_idx, sample_idx, :length]
            plddt = out["plddt"][batch_idx, sample_idx, :length]
            b_factor = plddt * 100
            pae = out["pae"][batch_idx, sample_idx, :length, :length]
            ptm = float(out["ptm"][batch_idx, sample_idx].item())

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

    @staticmethod
    def pad_to_length(
        feat: dict[str, np.ndarray],
        length: int,
    ) -> dict[str, np.ndarray]:
        """Pad the input features to an exact residue length."""
        current_length = feat["aatype_int"].shape[0]
        if length < current_length:
            raise ValueError(
                f"Cannot pad features of length {current_length} to shorter "
                f"length {length}."
            )

        new_feat: dict[str, np.ndarray] = {}
        for k, v in feat.items():
            target_length = length + 2 if k.startswith("lm.") else length
            pad_len = target_length - v.shape[0]
            if pad_len < 0:
                raise ValueError(
                    f"Feature {k!r} has length {v.shape[0]}, which exceeds "
                    f"target length {target_length}."
                )
            if pad_len == 0:
                new_feat[k] = v
            else:
                pad_width = ((0, pad_len),) + ((0, 0),) * (v.ndim - 1)
                constant_values = featurize.PAD_IDX if k == "lm.input_ids" else 0
                new_feat[k] = np.pad(
                    v,
                    pad_width,
                    constant_values=constant_values,
                )
        return new_feat

    @staticmethod
    def pad(feat: dict[str, np.ndarray], multiple_of: int = 32) -> dict[str, np.ndarray]:
        """Pad the input features to the specified length."""
        length = feat["aatype_int"].shape[0]
        pad_length = ((length + multiple_of - 1) // multiple_of) * multiple_of
        return FoldingRunner.pad_to_length(feat, pad_length)
