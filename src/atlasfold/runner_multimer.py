from collections.abc import Iterator, Sequence
import dataclasses

import numpy as np
import torch
from tqdm import tqdm

from atlasfold.common import featurize, protein, residue_constants
from atlasfold.model import AtlasFold_Multimer, SamplingConfig
from atlasfold.runner import autocast_context, default, seed_context


def _sanitize_sequence(sequence: str) -> str:
    sequence = "".join(sequence.split()).upper()
    return "".join(
        aa if aa in residue_constants.restype_orders else "X" for aa in sequence
    )


def parse_multimer_sequence(sequence: str) -> list[str]:
    """Parse a colon-separated multimer sequence string.

    A trailing colon is allowed, e.g. ``AAA:BBB:``.
    """
    parts = [part.strip() for part in sequence.split(":")]
    while parts and parts[-1] == "":
        parts.pop()
    if not parts:
        raise ValueError("At least one chain sequence is required.")
    if any(part == "" for part in parts):
        raise ValueError(
            "Empty chain sequence found. Use one colon between chains and only "
            "optional trailing colons."
        )
    return [_sanitize_sequence(part) for part in parts]


def _chain_ids(num_chains: int) -> list[str]:
    chain_ids = []
    for i in range(num_chains):
        chain_id = ""
        n = i
        while True:
            chain_id = chr(ord("A") + (n % 26)) + chain_id
            n //= 26
            if n == 0:
                break
        chain_ids.append(chain_id)
    return chain_ids


class ProteinMultimerOutput(protein.ProteinMultimer):
    """A predicted protein complex with confidence scores."""

    plddt: np.ndarray
    pae: np.ndarray
    ptm: float

    def __init__(
        self,
        *,
        name: str,
        chains: list[protein.Protein],
        plddt: np.ndarray,
        pae: np.ndarray,
        ptm: float,
        entity_ids: list[int] | None = None,
        asym_ids: list[int] | None = None,
        sym_ids: list[int] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            chains=chains,
            entity_ids=entity_ids or [],
            asym_ids=asym_ids or [],
            sym_ids=sym_ids or [],
        )
        num_residues = self.num_residues
        if plddt.shape != (num_residues,):
            raise ValueError(
                f"Invalid pLDDT shape: {plddt.shape}. "
                f"Expected ({num_residues},)."
            )
        if pae.shape != (num_residues, num_residues):
            raise ValueError(
                f"Invalid PAE shape: {pae.shape}. "
                f"Expected ({num_residues}, {num_residues})."
            )
        self.plddt = plddt
        self.pae = pae
        self.ptm = ptm


@dataclasses.dataclass(frozen=True, kw_only=True)
class MultimerFoldBatchOutput:
    """Outputs from one concrete batched multimer model call."""

    bucket_length: int
    batch_index: int
    num_batches: int
    inputs: list[tuple[int, str, list[str]]]
    outputs: list[list[ProteinMultimerOutput]]


class MultimerFoldingRunner:
    """Run AtlasFold-Multimer inference on colon-separated chain sequences."""

    def __init__(self, model: AtlasFold_Multimer):
        self.model: AtlasFold_Multimer = model
        self.device = self.model.device

    def fold(
        self,
        name: str,
        sequence: str | Sequence[str],
        *,
        num_samples: int = 1,
        preset: str = "base",
        seed: int = 1,
        num_recycles: int | None = None,
        mlm_prob: float | None = None,
        sampling_config: SamplingConfig | None = None,
        length_buckets: Sequence[int] | None = None,
    ) -> list[ProteinMultimerOutput]:
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
        inputs: Sequence[tuple[str, str | Sequence[str]]],
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
    ) -> list[list[ProteinMultimerOutput]]:
        inputs = self._normalize_inputs(inputs)
        outputs: list[list[ProteinMultimerOutput] | None] = [None] * len(inputs)

        for batch_output in self.iter_fold_batches(
            [(name, sequences) for _, name, sequences in inputs],
            num_samples=num_samples,
            preset=preset,
            seed=seed,
            num_recycles=num_recycles,
            mlm_prob=mlm_prob,
            sampling_config=sampling_config,
            length_buckets=length_buckets,
            max_tokens_per_batch=max_tokens_per_batch,
            disable_tqdm=disable_tqdm,
        ):
            for (input_idx, _, _), sample_outputs in zip(
                batch_output.inputs, batch_output.outputs, strict=True
            ):
                outputs[input_idx] = sample_outputs

        completed_outputs = []
        for output in outputs:
            if output is None:
                raise RuntimeError("Some batched folding outputs were not produced.")
            completed_outputs.append(output)
        return completed_outputs

    def iter_fold_batches(
        self,
        inputs: Sequence[tuple[str, str | Sequence[str]]],
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
    ) -> Iterator[MultimerFoldBatchOutput]:
        normalized_inputs = self._normalize_inputs(inputs)
        if len(normalized_inputs) == 0:
            return
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
        bucketed_inputs = self._bucket_inputs(normalized_inputs, length_buckets)
        batch_specs = self._make_batch_specs(bucketed_inputs, max_tokens_per_batch)

        total_count = len(normalized_inputs)
        curr_count = 0
        with tqdm(
            batch_specs,
            desc="Folding multimers",
            disable=disable_tqdm,
            unit="batch",
        ) as pbar:
            num_batches = len(batch_specs)
            for batch_idx, (bucket_length, chunk) in enumerate(pbar, start=1):
                pbar.set_postfix(
                    {
                        "bucket": bucket_length,
                        "batch_size": len(chunk),
                        "progress": f"{curr_count}/{total_count}",
                    }
                )
                complexes = [
                    protein.ProteinMultimer.get_empty(name, sequences)
                    for _, name, sequences in chunk
                ]
                feat = self._make_batch_features(complexes, bucket_length)
                out = self.model_run(
                    feat,
                    seed=seed,
                    num_samples=num_samples,
                    **settings,
                )
                batch_outputs = [
                    self._make_multimer_outputs(
                        complex_input=complex_input,
                        out=out,
                        batch_idx=batch_item_idx,
                        num_samples=num_samples,
                    )
                    for batch_item_idx, complex_input in enumerate(complexes)
                ]
                curr_count += len(chunk)
                pbar.set_postfix(
                    {
                        "bucket": bucket_length,
                        "batch_size": len(chunk),
                        "progress": f"{curr_count}/{total_count}",
                    }
                )
                yield MultimerFoldBatchOutput(
                    bucket_length=bucket_length,
                    batch_index=batch_idx,
                    num_batches=num_batches,
                    inputs=chunk,
                    outputs=batch_outputs,
                )

    @staticmethod
    def _normalize_inputs(
        inputs: Sequence[tuple[str, str | Sequence[str]]],
    ) -> list[tuple[int, str, list[str]]]:
        normalized = []
        for input_idx, (name, sequence) in enumerate(inputs):
            if isinstance(sequence, str):
                sequences = parse_multimer_sequence(sequence)
            else:
                sequences = [_sanitize_sequence(seq) for seq in sequence]
            if any(len(seq) == 0 for seq in sequences):
                raise ValueError(f"Input {input_idx} ({name!r}) has an empty chain.")
            normalized.append((input_idx, name, sequences))
        return normalized

    @staticmethod
    def _make_batch_specs(
        bucketed_inputs: dict[int, list[tuple[int, str, list[str]]]],
        max_tokens_per_batch: int,
    ) -> list[tuple[int, list[tuple[int, str, list[str]]]]]:
        batch_specs: list[tuple[int, list[tuple[int, str, list[str]]]]] = []
        for bucket_length in sorted(bucketed_inputs):
            bucket_items = bucketed_inputs[bucket_length]
            batch_size = max(1, max_tokens_per_batch // bucket_length)
            for start in range(0, len(bucket_items), batch_size):
                batch_specs.append(
                    (bucket_length, bucket_items[start : start + batch_size])
                )
        return batch_specs

    def _get_run_settings(
        self,
        *,
        preset: str,
        num_recycles: int | None,
        mlm_prob: float | None,
        sampling_config: SamplingConfig | None,
    ) -> dict:
        if preset not in ["base", "high"]:
            raise ValueError(f"Invalid preset for multimer inference: {preset}")

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
        sampling_cfg = SamplingConfig(num_steps=100, sigma_max=160)
        if preset == "high":
            num_recycles = 8
        return {
            "num_recycles": num_recycles,
            "mlm_prob": mlm_prob,
            "sampling_config": sampling_cfg,
        }

    def model_run(
        self,
        feat: dict[str, np.ndarray],
        seed: int,
        num_samples: int,
        num_recycles: int,
        mlm_prob: float,
        sampling_config: SamplingConfig,
    ) -> dict[str, np.ndarray]:
        device = self.device
        tensor_feat: dict[str, torch.Tensor] = {
            k: torch.as_tensor(v, device=device) for k, v in feat.items()
        }
        with (
            torch.inference_mode(),
            seed_context(seed, device),
            autocast_context(device),
        ):
            out = self.model.inference(
                tensor_feat,
                num_samples=num_samples,
                num_recycles=num_recycles,
                mlm_prob=mlm_prob,
                sampling_config=sampling_config,
            )
        return {k: v.cpu().float().numpy() for k, v in out.items()}

    @classmethod
    def _bucket_inputs(
        cls,
        inputs: Sequence[tuple[int, str, list[str]]],
        length_buckets: Sequence[int] | None,
    ) -> dict[int, list[tuple[int, str, list[str]]]]:
        bucketed_inputs: dict[int, list[tuple[int, str, list[str]]]] = {}
        for input_idx, name, sequences in inputs:
            length = sum(len(seq) for seq in sequences)
            if length == 0:
                raise ValueError(f"Input {input_idx} ({name!r}) has no residues.")
            bucket_length = cls.get_length_bucket(length, length_buckets)
            bucketed_inputs.setdefault(bucket_length, []).append(
                (input_idx, name, sequences)
            )
        return bucketed_inputs

    @staticmethod
    def get_length_bucket(
        length: int,
        length_buckets: Sequence[int] | None = None,
    ) -> int:
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
            f"Complex length {length} exceeds the largest configured length bucket "
            f"({buckets[-1]})."
        )

    @classmethod
    def _make_batch_features(
        cls,
        complexes: Sequence[protein.ProteinMultimer],
        bucket_length: int,
    ) -> dict[str, np.ndarray]:
        feats = [
            featurize.featurize_complex(
                complex_input.sequences,
                complex_input.entity_ids,
                complex_input.asym_ids,
                complex_input.sym_ids,
            )
            for complex_input in complexes
        ]
        lm_length = max(feat["lm.input_ids"].shape[0] for feat in feats)
        feats = [
            cls.pad_to_lengths(feat, residue_length=bucket_length, lm_length=lm_length)
            for feat in feats
        ]
        return {k: np.stack([feat[k] for feat in feats], axis=0) for k in feats[0]}

    @staticmethod
    def _make_multimer_outputs(
        *,
        complex_input: protein.ProteinMultimer,
        out: dict[str, np.ndarray],
        batch_idx: int,
        num_samples: int,
    ) -> list[ProteinMultimerOutput]:
        chain_lengths = [len(chain.sequence) for chain in complex_input.chains]
        total_length = sum(chain_lengths)
        chain_ids = _chain_ids(len(chain_lengths))
        samples = []

        for sample_idx in range(num_samples):
            coords = out["sample_coords"][batch_idx, sample_idx, :total_length]
            plddt = out["plddt"][batch_idx, sample_idx, :total_length]
            pae = out["pae"][batch_idx, sample_idx, :total_length, :total_length]
            ptm = float(out["ptm"][batch_idx, sample_idx].item())

            chains = []
            start = 0
            for chain_id, chain_sequence, chain_length in zip(
                chain_ids,
                complex_input.sequences,
                chain_lengths,
                strict=True,
            ):
                end = start + chain_length
                chains.append(
                    protein.Protein.create(
                        name=chain_id,
                        sequence=chain_sequence,
                        coordinates=coords[start:end],
                        b_factors=plddt[start:end] * 100,
                    )
                )
                start = end

            samples.append(
                ProteinMultimerOutput(
                    name=complex_input.name,
                    chains=chains,
                    entity_ids=complex_input.entity_ids,
                    asym_ids=complex_input.asym_ids,
                    sym_ids=complex_input.sym_ids,
                    plddt=plddt,
                    pae=pae,
                    ptm=ptm,
                )
            )
        return samples

    @staticmethod
    def pad_to_lengths(
        feat: dict[str, np.ndarray],
        *,
        residue_length: int,
        lm_length: int,
    ) -> dict[str, np.ndarray]:
        current_length = feat["aatype_int"].shape[0]
        if residue_length < current_length:
            raise ValueError(
                f"Cannot pad features of length {current_length} to shorter "
                f"length {residue_length}."
            )

        new_feat: dict[str, np.ndarray] = {}
        for k, v in feat.items():
            target_length = lm_length if k.startswith("lm.") else residue_length
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
