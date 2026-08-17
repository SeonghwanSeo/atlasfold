import dataclasses
import functools
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence

import numpy as np
import torch

from atlasfold.common import featurize, protein, residue_constants
from atlasfold.model import AtlasFoldMultimer_IPA
from atlasfold.runner import autocast_context, seed_context
from atlasfold.runner_ipa import RecycleCallback, RecycleMetric
from atlasfold.runner_multimer import MultimerInput, MultimerInputLike


def _sanitize_sequence(name: str, sequence: str) -> str:
    sequence = "".join(sequence.split()).upper()
    nonstandard_residues = sorted(
        set(sequence).difference(residue_constants.restype_orders)
    )
    if nonstandard_residues:
        warnings.warn(
            f"{name}: Nonstandard residue(s) "
            f"{', '.join(nonstandard_residues)} replaced with 'X'.",
            stacklevel=2,
        )
    return "".join(
        aa if aa in residue_constants.restype_orders else "X" for aa in sequence
    )


@dataclasses.dataclass(kw_only=True)
class MultimerIPAProteinOutput(protein.ProteinMultimer):
    """A single complex predicted by the multimer IPA model."""

    name: str
    chains: list[protein.Protein]
    seed: int
    plddt: np.ndarray
    pae: np.ndarray
    ptm: float
    iptm: float
    chain_ptm: list[float]
    interface_iptm: dict[tuple[int, int], float]

    def __post_init__(self) -> None:
        super().__post_init__()
        length = self.num_residues
        if self.plddt.shape != (length,):
            raise ValueError(
                f"Invalid pLDDT shape: {self.plddt.shape}. Expected ({length},)."
            )
        if self.pae.shape != (length, length):
            raise ValueError(
                f"Invalid PAE shape: {self.pae.shape}. Expected ({length}, {length})."
            )
        if len(self.chain_ptm) != self.num_chains:
            raise ValueError(
                f"Invalid chain pTM length: {len(self.chain_ptm)}. "
                f"Expected {self.num_chains}."
            )

    @property
    def avg_plddt(self) -> float:
        return float(self.plddt.mean() * 100)

    @property
    def ranking_score(self) -> float:
        return 0.8 * float(self.iptm) + 0.2 * float(self.ptm)

    @functools.cached_property
    def confidence_scores(self) -> dict:
        complex_scores = {
            "avg_plddt": self.avg_plddt,
            "ptm": float(self.ptm),
            "iptm": float(self.iptm),
            "ranking_score": self.ranking_score,
        }
        chain_scores = {}
        chain_names = []
        start = 0
        for chain_index, chain in enumerate(self.chains):
            end = start + chain.num_residues
            chain_scores[chain.name] = {
                "avg_plddt": float(self.plddt[start:end].mean() * 100),
                "ptm": self.chain_ptm[chain_index],
            }
            chain_names.append(chain.name)
            start = end

        interface_scores = {}
        for chain_i, name_i in enumerate(chain_names):
            for chain_j, name_j in enumerate(chain_names[chain_i + 1 :], chain_i + 1):
                interface_scores[f"{name_i}-{name_j}"] = {
                    "iptm": self.interface_iptm[(chain_i, chain_j)]
                }
        return {
            "complex": complex_scores,
            "chains": chain_scores,
            "interfaces": interface_scores,
        }


@dataclasses.dataclass
class MultimerIPAFoldingOutput:
    """All seeded multimer IPA predictions and recycle diagnostics for one target."""

    outputs: dict[int, MultimerIPAProteinOutput]
    ranking: list[int]
    recycle_counts: dict[int, int] = dataclasses.field(default_factory=dict)
    distogram_logits: dict[int, np.ndarray] = dataclasses.field(default_factory=dict)
    distogram_boundaries: np.ndarray | None = None

    @property
    def name(self) -> str:
        return self.outputs[next(iter(self.outputs))].name

    @property
    def length(self) -> int:
        return self.outputs[next(iter(self.outputs))].num_residues

    @property
    def best_seed(self) -> int:
        if not self.ranking:
            raise ValueError("No ranked predictions are available.")
        return self.ranking[0]

    @property
    def best(self) -> MultimerIPAProteinOutput:
        return self.outputs[self.best_seed]


class MultimerIPAFoldingRunner:
    """Run batched multimer IPA inference without diffusion-runner assumptions."""

    def __init__(self, model: AtlasFoldMultimer_IPA):
        self.model = model
        self.device = model.device

    def fold(
        self,
        name: str,
        chains: Sequence[str],
        *,
        seeds: int | Sequence[int] = 1,
        num_recycles: int = 10,
        mlm_prob: float = 0.20,
        recycle_early_stop_tolerance: float = 0.0,
        length_buckets: Sequence[int] | None = None,
        return_distogram: bool = False,
        recycle_callback: RecycleCallback | None = None,
    ) -> MultimerIPAFoldingOutput:
        return next(
            self.fold_iter(
                [MultimerInput(name, chains)],
                seeds=seeds,
                num_recycles=num_recycles,
                mlm_prob=mlm_prob,
                recycle_early_stop_tolerance=recycle_early_stop_tolerance,
                length_buckets=length_buckets,
                return_distogram=return_distogram,
                recycle_callback=recycle_callback,
            )
        )

    def fold_iter(
        self,
        inputs: Sequence[MultimerInputLike],
        *,
        seeds: int | Sequence[int] = 1,
        num_recycles: int = 10,
        mlm_prob: float = 0.20,
        recycle_early_stop_tolerance: float = 0.0,
        length_buckets: Sequence[int] | None = None,
        max_tokens_per_batch: int = 1024,
        return_distogram: bool = False,
        recycle_callback: RecycleCallback | None = None,
    ) -> Iterator[MultimerIPAFoldingOutput]:
        for batch in self.fold_iter_batch(
            inputs,
            seeds=seeds,
            num_recycles=num_recycles,
            mlm_prob=mlm_prob,
            recycle_early_stop_tolerance=recycle_early_stop_tolerance,
            length_buckets=length_buckets,
            max_tokens_per_batch=max_tokens_per_batch,
            return_distogram=return_distogram,
            recycle_callback=recycle_callback,
        ):
            yield from batch

    def fold_iter_batch(
        self,
        inputs: Sequence[MultimerInputLike],
        *,
        seeds: int | Sequence[int] = 1,
        num_recycles: int = 10,
        mlm_prob: float = 0.20,
        recycle_early_stop_tolerance: float = 0.0,
        length_buckets: Sequence[int] | None = None,
        max_tokens_per_batch: int = 1024,
        return_distogram: bool = False,
        recycle_callback: RecycleCallback | None = None,
    ) -> Iterator[list[MultimerIPAFoldingOutput]]:
        seeds = [seeds] if isinstance(seeds, int) else list(seeds)
        inputs = list(inputs)
        if not inputs:
            raise ValueError("No inputs provided.")
        if not seeds:
            raise ValueError("No seeds provided.")
        if max_tokens_per_batch < 0:
            raise ValueError("max_tokens_per_batch must be non-negative.")

        normalized = self._normalize_inputs(inputs)
        bucketed = self._bucket_inputs(normalized, length_buckets)
        warned_batched_early_stop = False
        for bucket_length, complexes in self._iter_batch(bucketed, max_tokens_per_batch):
            if (
                recycle_early_stop_tolerance > 0
                and num_recycles > 0
                and len(complexes) != 1
                and not warned_batched_early_stop
            ):
                warnings.warn(
                    "Recycle early stopping is synchronized across the batch, so "
                    f"all {len(complexes)} targets must satisfy the tolerance "
                    "together. Run targets with batch size 1 for independent early "
                    "stopping.",
                    stacklevel=2,
                )
                warned_batched_early_stop = True
            features = self._make_batch_features(complexes, bucket_length)
            predictions: list[dict[int, MultimerIPAProteinOutput]] = [
                {} for _ in complexes
            ]
            recycle_counts: list[dict[int, int]] = [{} for _ in complexes]
            distograms: list[dict[int, np.ndarray]] = [{} for _ in complexes]
            boundaries = None

            for seed in seeds:

                def model_recycle_callback(
                    recycle_index: int,
                    recycle_output: dict[str, torch.Tensor],
                    *,
                    _seed: int = seed,
                    _complexes: list[protein.ProteinMultimer] = complexes,
                ) -> None:
                    for batch_index, complex_input in enumerate(_complexes):
                        record = self._make_live_recycle_metric(
                            recycle_output,
                            batch_index,
                            recycle_index,
                            complex_input.num_residues,
                        )
                        assert recycle_callback is not None
                        recycle_callback(
                            complex_input.name,
                            _seed,
                            recycle_index,
                            record,
                        )

                out = self.model_run(
                    features,
                    seed=seed,
                    num_recycles=num_recycles,
                    mlm_prob=mlm_prob,
                    recycle_early_stop_tolerance=recycle_early_stop_tolerance,
                    return_distogram=return_distogram,
                    recycle_callback=(
                        model_recycle_callback if recycle_callback is not None else None
                    ),
                )
                if return_distogram:
                    boundaries = out["distogram.boundaries"]
                for batch_index, complex_input in enumerate(complexes):
                    predictions[batch_index][seed] = self._make_output(
                        complex_input, out, batch_index, seed
                    )
                    recycle_counts[batch_index][seed] = int(
                        out["num_recycles"][batch_index]
                    )
                    if return_distogram:
                        length = complex_input.num_residues
                        distograms[batch_index][seed] = out["distogram.logits"][
                            batch_index, :length, :length
                        ]

            yield [
                MultimerIPAFoldingOutput(
                    outputs=outputs,
                    ranking=sorted(
                        outputs,
                        key=lambda seed: outputs[seed].ranking_score,
                        reverse=True,
                    ),
                    recycle_counts=recycle_counts[index],
                    distogram_logits=distograms[index],
                    distogram_boundaries=boundaries,
                )
                for index, outputs in enumerate(predictions)
            ]

    def model_run(
        self,
        features: dict[str, np.ndarray],
        *,
        seed: int,
        num_recycles: int,
        mlm_prob: float,
        recycle_early_stop_tolerance: float,
        return_distogram: bool = False,
        recycle_callback: Callable[[int, dict[str, torch.Tensor]], None] | None = None,
    ) -> dict:
        tensor_features = {
            key: torch.as_tensor(value, device=self.device)
            for key, value in features.items()
        }
        with (
            torch.inference_mode(),
            seed_context(seed, self.device),
            autocast_context(self.device),
        ):
            out = self.model.inference(
                tensor_features,
                num_recycles=num_recycles,
                mlm_prob=mlm_prob,
                recycle_early_stop_tolerance=recycle_early_stop_tolerance,
                recycle_callback=recycle_callback,
            )

        keys = {
            "coords",
            "plddt",
            "pae",
            "ptm",
            "iptm",
            "chain_ptm",
            "interface_iptm",
            "num_recycles",
            "tol",
        }
        if return_distogram:
            keys.update({"distogram.logits", "distogram.boundaries"})
        converted = {}
        for key, value in out.items():
            if key not in keys:
                continue
            value = value.cpu()
            converted[key] = (
                value.float() if value.is_floating_point() else value
            ).numpy()
        return converted

    @staticmethod
    def _make_output(
        complex_input: protein.ProteinMultimer,
        out: dict,
        batch_index: int,
        seed: int,
    ) -> MultimerIPAProteinOutput:
        chain_lengths = [len(chain.sequence) for chain in complex_input.chains]
        total_length = sum(chain_lengths)
        coords = out["coords"][batch_index, :total_length]
        plddt = out["plddt"][batch_index, :total_length]

        chains = []
        start = 0
        for chain, chain_length in zip(complex_input.chains, chain_lengths, strict=True):
            end = start + chain_length
            chains.append(
                protein.Protein.create(
                    name=chain.name,
                    sequence=chain.sequence,
                    coordinates=coords[start:end],
                    b_factors=plddt[start:end] * 100,
                )
            )
            start = end

        num_chains = len(chains)
        interface = out["interface_iptm"][batch_index, :num_chains, :num_chains]
        return MultimerIPAProteinOutput(
            name=complex_input.name,
            chains=chains,
            seed=seed,
            plddt=plddt,
            pae=out["pae"][batch_index, :total_length, :total_length],
            ptm=float(out["ptm"][batch_index]),
            iptm=float(out["iptm"][batch_index]),
            chain_ptm=[
                float(value) for value in out["chain_ptm"][batch_index, :num_chains]
            ],
            interface_iptm={
                (chain_i, chain_j): float(interface[chain_i, chain_j])
                for chain_i in range(num_chains)
                for chain_j in range(chain_i + 1, num_chains)
            },
        )

    @staticmethod
    def _make_live_recycle_metric(
        output: dict[str, torch.Tensor],
        batch_index: int,
        recycle_index: int,
        length: int,
    ) -> RecycleMetric:
        plddt = output["plddt"][batch_index, :length].detach().float()
        tol = float(output["tol"][batch_index].detach().float().cpu().item())
        ptm = float(output["ptm"][batch_index].detach().float().cpu().item())
        return {
            "recycle": recycle_index,
            "tol": None if np.isnan(tol) else tol,
            "plddt": float((plddt.mean() * 100).cpu().item()),
            "ptm": ptm,
        }

    @staticmethod
    def _normalize_inputs(
        inputs: Sequence[MultimerInputLike],
    ) -> list[MultimerInput]:
        normalized = []
        for input_ in inputs:
            item = MultimerInput(*input_)
            chains = [
                _sanitize_sequence(f"{item.name} chain {index}", chain)
                for index, chain in enumerate(item.chains, start=1)
            ]
            if not chains or any(not chain for chain in chains):
                raise ValueError(f"Input ({item.name}) has an empty chain.")
            normalized.append(MultimerInput(item.name, chains))
        return normalized

    @classmethod
    def _bucket_inputs(
        cls,
        inputs: Sequence[MultimerInput],
        length_buckets: Sequence[int] | None,
    ) -> dict[int, list[MultimerInput]]:
        bucketed: dict[int, list[MultimerInput]] = defaultdict(list)
        for item in inputs:
            bucket = cls._get_length_bucket(item.length, length_buckets)
            bucketed[bucket].append(item)
        return bucketed

    @staticmethod
    def _iter_batch(
        bucketed: dict[int, list[MultimerInput]],
        max_tokens_per_batch: int,
    ) -> Iterator[tuple[int, list[protein.ProteinMultimer]]]:
        for bucket_length in sorted(bucketed):
            items = bucketed[bucket_length]
            batch_size = (
                1
                if max_tokens_per_batch == 0
                else max(1, max_tokens_per_batch // bucket_length)
            )
            for start in range(0, len(items), batch_size):
                yield (
                    bucket_length,
                    [
                        protein.ProteinMultimer.get_empty(item.name, list(item.chains))
                        for item in items[start : start + batch_size]
                    ],
                )

    @staticmethod
    def _get_length_bucket(length: int, length_buckets: Sequence[int] | None) -> int:
        if length_buckets is None:
            for bucket in featurize.DEFAULT_BUCKETS:
                if length <= bucket:
                    return bucket
            return ((length + 127) // 128) * 128
        buckets = sorted(set(length_buckets))
        if not buckets or buckets[0] <= 0:
            raise ValueError("length_buckets must contain positive integers.")
        for bucket in buckets:
            if length <= bucket:
                return bucket
        raise ValueError(
            f"Complex length {length} exceeds the largest bucket ({buckets[-1]})."
        )

    @classmethod
    def _make_batch_features(
        cls,
        complexes: Sequence[protein.ProteinMultimer],
        bucket_length: int,
    ) -> dict[str, np.ndarray]:
        features = [
            featurize.featurize_complex(
                complex_input.sequences,
                complex_input.entity_ids,
                complex_input.asym_ids,
                complex_input.sym_ids,
            )
            for complex_input in complexes
        ]
        lm_length = max(feature["lm.input_ids"].shape[0] for feature in features)
        features = [
            cls._pad_to_lengths(feature, bucket_length, lm_length) for feature in features
        ]
        return {
            key: np.stack([feature[key] for feature in features]) for key in features[0]
        }

    @staticmethod
    def _pad_to_lengths(
        features: dict[str, np.ndarray], residue_length: int, lm_length: int
    ) -> dict[str, np.ndarray]:
        padded = {}
        for key, value in features.items():
            target = lm_length if key.startswith("lm.") else residue_length
            pad_length = target - value.shape[0]
            if pad_length < 0:
                raise ValueError(f"Feature {key!r} exceeds target length {target}.")
            if pad_length == 0:
                padded[key] = value
                continue
            pad_width = ((0, pad_length),) + ((0, 0),) * (value.ndim - 1)
            constant = featurize.PAD_IDX if key == "lm.input_ids" else 0
            padded[key] = np.pad(value, pad_width, constant_values=constant)
        return padded
