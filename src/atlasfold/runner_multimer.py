import dataclasses
import functools
from collections import defaultdict
from collections.abc import Iterator, Sequence

import numpy as np
import torch

from atlasfold.common import featurize, protein, residue_constants
from atlasfold.model import AtlasFold_Multimer, SamplingConfig
from atlasfold.model.utils import confidence_metrics
from atlasfold.runner import SampleKey, autocast_context, seed_context


@dataclasses.dataclass(frozen=True)
class MultimerInput:
    """Input for one multimer folding target."""

    name: str
    sequence: list[str]
    chain_ids: list[str] | None = None

    @property
    def length(self) -> int:
        return sum(len(seq) for seq in self.sequence)

    @classmethod
    def from_sequence(
        cls, name: str, sequence: str, chain_ids: list[str] | None = None
    ) -> "MultimerInput":
        """Create a MultimerInput from a single concatenated sequence."""
        seqs = sequence.split(":")
        return cls(name, seqs, chain_ids)


def _sanitize_sequence(sequence: str) -> str:
    sequence = "".join(sequence.split()).upper()
    return "".join(
        aa if aa in residue_constants.restype_orders else "X" for aa in sequence
    )


@dataclasses.dataclass(kw_only=True)
class ProteinMultimerOutput(protein.ProteinMultimer):
    """A predicted protein complex with confidence scores."""

    name: str
    chains: list[protein.Protein]
    seed: int | None = None
    sample_index: int | None = None
    plddt: np.ndarray
    pae: np.ndarray
    pde: np.ndarray
    ptm: float
    iptm: float
    chain_ptm: list[float] = dataclasses.field(default_factory=list)
    interface_iptm: dict[tuple[int, int], float] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        num_residues = self.num_residues
        if self.plddt.shape != (num_residues,):
            raise ValueError(
                f"Invalid pLDDT shape: {self.plddt.shape}. Expected ({num_residues},)."
            )
        if self.pae.shape != (num_residues, num_residues):
            raise ValueError(
                f"Invalid PAE shape: {self.pae.shape}. "
                f"Expected ({num_residues}, {num_residues})."
            )
        if self.pde.shape != (num_residues, num_residues):
            raise ValueError(
                f"Invalid PDE shape: {self.pde.shape}. "
                f"Expected ({num_residues}, {num_residues})."
            )
        if len(self.chain_ptm) != self.num_chains:
            raise ValueError(
                f"Invalid chain pTM length: {len(self.chain_ptm)}. "
                f"Expected {self.num_chains}."
            )
        missing_interface_iptm = {
            (chain_i, chain_j)
            for chain_i in range(self.num_chains)
            for chain_j in range(chain_i + 1, self.num_chains)
            if (chain_i, chain_j) not in self.interface_iptm
        }
        if missing_interface_iptm:
            raise ValueError(
                f"Missing interface ipTM for chain pairs: {missing_interface_iptm}."
            )

    @property
    def avg_plddt(self) -> float:
        return self.plddt.mean().item() * 100

    @property
    def avg_pde(self) -> float:
        return self.pde.mean().item()

    @property
    def ranking_score(self) -> float:
        return 0.8 * self.iptm + 0.2 * float(self.ptm)

    @functools.cached_property
    def confidence_scores(self) -> dict:
        complex_scores = {
            "avg_plddt": self.avg_plddt,
            "avg_pde": self.avg_pde,
            "ptm": float(self.ptm),
            "iptm": self.iptm,
            "ranking_score": self.ranking_score,
        }

        chain_scores: dict[str, dict] = {}
        chain_names = []
        start = 0
        for chain_idx, chain in enumerate(self.chains):
            end = start + chain.num_residues
            chain_mask = np.zeros(self.num_residues, dtype=bool)
            chain_mask[start:end] = True
            pair_mask = chain_mask[:, None] & chain_mask[None, :]
            chain_scores[chain.name] = {
                "avg_plddt": self.plddt[chain_mask].mean().item() * 100,
                "avg_pde": self.pde[pair_mask].mean().item(),
                "ptm": self.chain_ptm[chain_idx],
            }
            chain_names.append(chain.name)
            start = end

        interface_scores: dict[str, dict] = {}
        for chain_i, chain_name_i in enumerate(chain_names):
            for chain_j, chain_name_j in enumerate(
                chain_names[chain_i + 1 :], chain_i + 1
            ):
                interface_scores[f"{chain_name_i}-{chain_name_j}"] = {
                    "iptm": self.interface_iptm[(chain_i, chain_j)],
                }

        return {
            "complex": complex_scores,
            "chains": chain_scores,
            "interfaces": interface_scores,
        }


@dataclasses.dataclass
class MultimerFoldingOutput:
    """Outputs for one folded target."""

    outputs: dict[SampleKey, ProteinMultimerOutput]
    ranking: list[SampleKey]

    @property
    def name(self) -> str:
        return self.outputs[next(iter(self.outputs))].name

    @property
    def length(self) -> int:
        return self.outputs[next(iter(self.outputs))].num_residues


class MultimerFoldingRunner:
    """Run AtlasFold-Multimer inference on pre-split chain sequences."""

    def __init__(self, model: AtlasFold_Multimer):
        self.model: AtlasFold_Multimer = model
        self.device = self.model.device

    def fold(
        self,
        name: str,
        sequence: str | Sequence[str],
        *,
        num_samples: int = 1,
        seeds: int | Sequence[int] = 1,
        num_recycles: int = 10,
        sampling_config: SamplingConfig | None = None,
        mlm_prob: float = 0.20,
        length_buckets: Sequence[int] | None = None,
    ) -> MultimerFoldingOutput:
        sequences = sequence.split(":") if isinstance(sequence, str) else list(sequence)
        outputs = list(
            self.iter_fold_batch(
                [MultimerInput(name, sequences)],
                num_samples=num_samples,
                seeds=seeds,
                num_recycles=num_recycles,
                mlm_prob=mlm_prob,
                sampling_config=sampling_config,
                length_buckets=length_buckets,
            )
        )
        return outputs[0][0]

    def iter_fold_batch(
        self,
        inputs: Sequence[MultimerInput],
        *,
        num_samples: int = 1,
        seeds: int | Sequence[int] = 1,
        num_recycles: int = 10,
        sampling_config: SamplingConfig | None = None,
        mlm_prob: float = 0.20,
        length_buckets: Sequence[int] | None = None,
        max_tokens_per_batch: int = 1024,
    ) -> Iterator[list[MultimerFoldingOutput]]:
        seeds = [seeds] if isinstance(seeds, int) else list(seeds)

        if len(inputs) == 0:
            raise ValueError("No inputs provided.")
        if len(seeds) == 0:
            raise ValueError("No seeds provided.")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")
        if max_tokens_per_batch <= 0:
            raise ValueError(
                f"max_tokens_per_batch must be positive, got {max_tokens_per_batch}."
            )
        if sampling_config is None:
            sampling_config = SamplingConfig(num_steps=200)

        normalized_inputs = self._normalize_inputs(inputs)
        bucketed_inputs = self._bucket_inputs(normalized_inputs, length_buckets)
        for bucket_length, complexes in self._iter_batch(
            bucketed_inputs, max_tokens_per_batch
        ):
            batch_size = len(complexes)
            batch = self._make_batch_features(complexes, bucket_length)

            model_outputs: list[dict[SampleKey, ProteinMultimerOutput]] = [
                {} for _ in range(batch_size)
            ]
            for seed_value in seeds:
                out = self.model_run(
                    batch,
                    seed=seed_value,
                    num_samples=num_samples,
                    num_recycles=num_recycles,
                    mlm_prob=mlm_prob,
                    sampling_config=sampling_config,
                )

                for batch_i, complex_input in enumerate(complexes):
                    model_outputs[batch_i].update(
                        self._make_outputs(
                            complex_input=complex_input,
                            out=out,
                            batch_idx=batch_i,
                            num_samples=num_samples,
                            seed=seed_value,
                        )
                    )
            batch_outputs = []
            for outputs in model_outputs:
                ranking = sorted(
                    outputs.keys(), key=lambda k: outputs[k].ranking_score, reverse=True
                )
                batch_outputs.append(MultimerFoldingOutput(outputs, ranking))
            yield batch_outputs

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

    @staticmethod
    def _normalize_inputs(
        inputs: Sequence[MultimerInput],
    ) -> list[MultimerInput]:
        normalized: list[MultimerInput] = []
        for item in inputs:
            if not isinstance(item.sequence, list):
                raise TypeError(
                    "MultimerFoldingRunner expects pre-split chain sequences. "
                    "Pass a list[str]."
                )
            sequences = [_sanitize_sequence(seq) for seq in item.sequence]
            if any(len(seq) == 0 for seq in sequences):
                raise ValueError(f"Input ({item.name}) has an empty chain.")
            chain_ids = None
            if item.chain_ids is not None:
                if isinstance(item.chain_ids, str):
                    raise ValueError(
                        f"Input ({item.name}) chain_ids must be a sequence of IDs, "
                        "not a string."
                    )
                chain_ids = [str(chain_id) for chain_id in item.chain_ids]
            normalized.append(MultimerInput(str(item.name), sequences, chain_ids))
        return normalized

    @classmethod
    def _bucket_inputs(
        cls,
        inputs: Sequence[MultimerInput],
        length_buckets: Sequence[int] | None,
    ) -> dict[int, list[MultimerInput]]:
        bucketed_inputs: dict[int, list[MultimerInput]] = defaultdict(list)
        for item in inputs:
            length = sum(len(seq) for seq in item.sequence)
            if length == 0:
                raise ValueError(f"Input {item.name} has no residues.")
            bucket_length = cls._get_length_bucket(length, length_buckets)
            bucketed_inputs[bucket_length].append(item)
        return bucketed_inputs

    def _iter_batch(
        self,
        bucketed_inputs: dict[int, list[MultimerInput]],
        max_tokens_per_batch: int,
    ) -> Iterator[tuple[int, list[protein.ProteinMultimer]]]:
        for bucket_length in sorted(bucketed_inputs):
            bucket_items = bucketed_inputs[bucket_length]
            batch_size = max(1, max_tokens_per_batch // bucket_length)
            for start in range(0, len(bucket_items), batch_size):
                chunk = bucket_items[start : start + batch_size]
                complexes = [
                    protein.ProteinMultimer.get_empty(
                        item.name,
                        item.sequence,
                        chain_ids=item.chain_ids,
                    )
                    for item in chunk
                ]
                yield bucket_length, complexes

    @staticmethod
    def _get_length_bucket(
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
            cls._pad_to_lengths(feat, residue_length=bucket_length, lm_length=lm_length)
            for feat in feats
        ]
        return {k: np.stack([feat[k] for feat in feats], axis=0) for k in feats[0]}

    @staticmethod
    def _make_outputs(
        *,
        complex_input: protein.ProteinMultimer,
        out: dict[str, np.ndarray],
        batch_idx: int,
        num_samples: int,
        seed: int,
    ) -> dict[SampleKey, ProteinMultimerOutput]:
        chain_lengths = [len(chain.sequence) for chain in complex_input.chains]
        total_length = sum(chain_lengths)
        chain_ids = [chain.name for chain in complex_input.chains]
        chain_ranges = []
        start = 0
        for chain_length in chain_lengths:
            end = start + chain_length
            chain_ranges.append((start, end))
            start = end
        samples = {}

        for sample_idx in range(num_samples):
            coords = out["sample_coords"][batch_idx, sample_idx, :total_length]
            plddt = out["plddt"][batch_idx, sample_idx, :total_length]
            pae = out["pae"][batch_idx, sample_idx, :total_length, :total_length]
            pde = out["pde"][batch_idx, sample_idx, :total_length, :total_length]
            ptm = float(out["ptm"][batch_idx, sample_idx].item())
            iptm = float(out["iptm"][batch_idx, sample_idx].item())

            # Create Protein objects for each chain in the complex
            chains: list[protein.Protein] = []
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

            pae_logits = torch.as_tensor(
                out["pae_logits"][batch_idx, sample_idx, :total_length, :total_length]
            )
            pae_bin_centers = torch.as_tensor(out["pae_bin_centers"])

            chain_ptm: list[float] = []
            for start, end in chain_ranges:
                chain_mask = torch.ones(end - start, dtype=torch.bool)
                chain_ptm.append(
                    confidence_metrics.compute_ptm(
                        pae_logits[start:end, start:end],
                        pae_bin_centers,
                        chain_mask,
                    ).item()
                )

            interface_iptm_dict: dict[tuple[int, int], float] = {}
            for chain_i, (start_i, end_i) in enumerate(chain_ranges):
                for chain_j, (start_j, end_j) in enumerate(
                    chain_ranges[chain_i + 1 :], chain_i + 1
                ):
                    residue_idx = list(range(start_i, end_i)) + list(
                        range(start_j, end_j)
                    )
                    interface_logits = pae_logits[residue_idx][:, residue_idx]
                    interface_asym_id = torch.cat(
                        [
                            torch.zeros(end_i - start_i, dtype=torch.long),
                            torch.ones(end_j - start_j, dtype=torch.long),
                        ]
                    )
                    interface_mask = torch.ones(len(residue_idx), dtype=torch.bool)
                    interface_iptm = confidence_metrics.compute_iptm(
                        interface_logits,
                        pae_bin_centers,
                        interface_asym_id,
                        interface_mask,
                    ).item()
                    interface_iptm_dict[(chain_i, chain_j)] = interface_iptm

            samples[(seed, sample_idx)] = ProteinMultimerOutput(
                name=complex_input.name,
                chains=chains,
                seed=seed,
                sample_index=sample_idx,
                plddt=plddt,
                pae=pae,
                pde=pde,
                ptm=ptm,
                iptm=iptm,
                chain_ptm=chain_ptm,
                interface_iptm=interface_iptm_dict,
            )
        return samples

    @staticmethod
    def _pad_to_lengths(
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
                    f"Feature {k} has length {v.shape[0]}, which exceeds "
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
