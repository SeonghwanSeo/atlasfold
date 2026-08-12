import dataclasses
import warnings
from collections.abc import Callable, Iterator, Sequence
from typing import NamedTuple, TypeAlias

import numpy as np
import torch

from atlasfold.common import featurize, protein, residue_constants
from atlasfold.model import AtlasFold_IPA
from atlasfold.runner import autocast_context, seed_context


class IPAFoldingInput(NamedTuple):
    """Input for one monomer IPA folding target."""

    name: str
    sequence: str


IPAFoldingInputLike: TypeAlias = IPAFoldingInput | tuple[str, str]
RecycleMetric = dict[str, float | int | None]
RecycleCallback: TypeAlias = Callable[[str, int, int, RecycleMetric], None]


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
class IPAProteinOutput(protein.Protein):
    """A single structure predicted by the monomer IPA model."""

    name: str
    sequence: str
    seed: int
    coordinates: np.ndarray
    b_factors: np.ndarray
    plddt: np.ndarray
    pae: np.ndarray
    ptm: float
    residue_index: np.ndarray | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        length = len(self.sequence)
        if self.plddt.shape != (length,):
            raise ValueError(
                f"Invalid pLDDT shape: {self.plddt.shape}. Expected ({length},)."
            )
        if self.pae.shape != (length, length):
            raise ValueError(
                f"Invalid PAE shape: {self.pae.shape}. Expected ({length}, {length})."
            )
        if self.residue_index is not None:
            raise ValueError("Custom residue indices are not supported.")

    @property
    def avg_plddt(self) -> float:
        return float(self.plddt.mean() * 100)

    @property
    def ranking_score(self) -> float:
        return self.avg_plddt

    @property
    def confidence_scores(self) -> dict[str, float]:
        return {"avg_plddt": self.avg_plddt, "ptm": float(self.ptm)}


@dataclasses.dataclass
class IPAFoldingOutput:
    """All seeded IPA predictions and recycle diagnostics for one target."""

    outputs: dict[int, IPAProteinOutput]
    ranking: list[int]
    recycle_metrics: dict[int, list[RecycleMetric]]
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
    def best(self) -> IPAProteinOutput:
        return self.outputs[self.best_seed]


class IPAFoldingRunner:
    """Run batched monomer IPA inference without diffusion-runner assumptions."""

    def __init__(self, model: AtlasFold_IPA):
        self.model = model
        self.device = model.device

    def fold(
        self,
        name: str,
        sequence: str,
        *,
        seeds: int | Sequence[int] = 1,
        num_recycles: int = 4,
        mlm_prob: float = 0.15,
        recycle_early_stop_tolerance: float = 0.0,
        length_buckets: Sequence[int] | None = None,
        return_distogram: bool = False,
        recycle_callback: RecycleCallback | None = None,
    ) -> IPAFoldingOutput:
        return next(
            self.fold_iter(
                [IPAFoldingInput(name, sequence)],
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
        inputs: Sequence[IPAFoldingInputLike],
        *,
        seeds: int | Sequence[int] = 1,
        num_recycles: int = 4,
        mlm_prob: float = 0.15,
        recycle_early_stop_tolerance: float = 0.0,
        length_buckets: Sequence[int] | None = None,
        max_tokens_per_batch: int = 1024,
        return_distogram: bool = False,
        recycle_callback: RecycleCallback | None = None,
    ) -> Iterator[IPAFoldingOutput]:
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
        inputs: Sequence[IPAFoldingInputLike],
        *,
        seeds: int | Sequence[int] = 1,
        num_recycles: int = 4,
        mlm_prob: float = 0.15,
        recycle_early_stop_tolerance: float = 0.0,
        length_buckets: Sequence[int] | None = None,
        max_tokens_per_batch: int = 1024,
        return_distogram: bool = False,
        recycle_callback: RecycleCallback | None = None,
    ) -> Iterator[list[IPAFoldingOutput]]:
        seeds = [seeds] if isinstance(seeds, int) else list(seeds)
        inputs = list(inputs)
        if not inputs:
            raise ValueError("No inputs provided.")
        if not seeds:
            raise ValueError("No seeds provided.")
        if max_tokens_per_batch <= 0:
            raise ValueError("max_tokens_per_batch must be positive.")

        normalized = self._normalize_inputs(inputs)
        bucketed = self._bucket_inputs(normalized, length_buckets)
        warned_batched_early_stop = False
        for bucket_length, chunk in self._iter_batch(bucketed, max_tokens_per_batch):
            if (
                recycle_early_stop_tolerance > 0
                and num_recycles > 0
                and len(chunk) != 1
                and not warned_batched_early_stop
            ):
                warnings.warn(
                    "Recycle early stopping is synchronized across the batch, so "
                    f"all {len(chunk)} targets must satisfy the tolerance together. "
                    "Run targets with batch size 1 for independent early stopping.",
                    stacklevel=2,
                )
                warned_batched_early_stop = True
            features = self._make_batch_features(
                [item.sequence for item in chunk], bucket_length
            )
            predictions: list[dict[int, IPAProteinOutput]] = [{} for _ in chunk]
            histories: list[dict[int, list[RecycleMetric]]] = [{} for _ in chunk]
            distograms: list[dict[int, np.ndarray]] = [{} for _ in chunk]
            boundaries = None

            for seed in seeds:
                seed_histories: list[list[RecycleMetric]] = [[] for _ in chunk]

                def model_recycle_callback(
                    recycle_index: int,
                    recycle_output: dict[str, torch.Tensor],
                    *,
                    _seed: int = seed,
                    _chunk: list[IPAFoldingInput] = chunk,
                    _histories: list[list[RecycleMetric]] = seed_histories,
                ) -> None:
                    for batch_index, item in enumerate(_chunk):
                        record = self._make_live_recycle_metric(
                            recycle_output,
                            batch_index,
                            recycle_index,
                            len(item.sequence),
                        )
                        _histories[batch_index].append(record)
                        if recycle_callback is not None:
                            recycle_callback(
                                item.name,
                                _seed,
                                recycle_index,
                                record.copy(),
                            )

                out = self.model_run(
                    features,
                    seed=seed,
                    num_recycles=num_recycles,
                    mlm_prob=mlm_prob,
                    recycle_early_stop_tolerance=recycle_early_stop_tolerance,
                    return_distogram=return_distogram,
                    recycle_callback=model_recycle_callback,
                )
                if return_distogram:
                    boundaries = out["distogram.boundaries"]
                for batch_index, item in enumerate(chunk):
                    predictions[batch_index][seed] = self._make_output(
                        item, out, batch_index, seed
                    )
                    histories[batch_index][seed] = seed_histories[batch_index]
                    if return_distogram:
                        length = len(item.sequence)
                        distograms[batch_index][seed] = out["distogram.logits"][
                            batch_index, :length, :length
                        ]

            yield [
                IPAFoldingOutput(
                    outputs=outputs,
                    ranking=sorted(
                        outputs,
                        key=lambda seed: outputs[seed].ranking_score,
                        reverse=True,
                    ),
                    recycle_metrics=histories[index],
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

        keys = {"coords", "plddt", "pae", "ptm", "num_recycles", "tol"}
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
        item: IPAFoldingInput,
        out: dict,
        batch_index: int,
        seed: int,
    ) -> IPAProteinOutput:
        length = len(item.sequence)
        plddt = out["plddt"][batch_index, :length]
        return IPAProteinOutput(
            name=item.name,
            sequence=item.sequence,
            seed=seed,
            coordinates=out["coords"][batch_index, :length],
            b_factors=plddt * 100,
            plddt=plddt,
            pae=out["pae"][batch_index, :length, :length],
            ptm=float(out["ptm"][batch_index]),
        )

    @staticmethod
    def _make_live_recycle_metric(
        output: dict[str, torch.Tensor],
        batch_index: int,
        recycle_index: int,
        length: int,
    ) -> RecycleMetric:
        plddt = output["plddt"][batch_index, :length].detach().float()
        pae = output["pae"][batch_index, :length, :length].detach().float()
        tol = float(output["tol"][batch_index].detach().float().cpu().item())
        return {
            "recycle": recycle_index,
            "tol": None if np.isnan(tol) else tol,
            "avg_plddt": float((plddt.mean() * 100).cpu().item()),
            "avg_pae": float(pae.mean().cpu().item()),
            "ptm": float(output["ptm"][batch_index].detach().float().cpu().item()),
        }

    @staticmethod
    def _normalize_inputs(
        inputs: Sequence[IPAFoldingInputLike],
    ) -> list[IPAFoldingInput]:
        normalized = []
        for input_ in inputs:
            item = IPAFoldingInput(*input_)
            sequence = _sanitize_sequence(item.name, item.sequence)
            if not sequence:
                raise ValueError(f"Input ({item.name}) has an empty sequence.")
            normalized.append(IPAFoldingInput(item.name, sequence))
        return normalized

    @classmethod
    def _bucket_inputs(
        cls,
        inputs: Sequence[IPAFoldingInput],
        length_buckets: Sequence[int] | None,
    ) -> dict[int, list[IPAFoldingInput]]:
        bucketed: dict[int, list[IPAFoldingInput]] = {}
        for item in inputs:
            bucket = cls._get_length_bucket(len(item.sequence), length_buckets)
            bucketed.setdefault(bucket, []).append(item)
        return bucketed

    @staticmethod
    def _iter_batch(
        bucketed: dict[int, list[IPAFoldingInput]], max_tokens_per_batch: int
    ) -> Iterator[tuple[int, list[IPAFoldingInput]]]:
        for bucket_length in sorted(bucketed):
            items = bucketed[bucket_length]
            batch_size = max(1, max_tokens_per_batch // bucket_length)
            for start in range(0, len(items), batch_size):
                yield bucket_length, items[start : start + batch_size]

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
            f"Sequence length {length} exceeds the largest bucket ({buckets[-1]})."
        )

    @classmethod
    def _make_batch_features(
        cls, sequences: Sequence[str], bucket_length: int
    ) -> dict[str, np.ndarray]:
        features = [
            cls._pad_to_length(featurize.featurize(sequence), bucket_length)
            for sequence in sequences
        ]
        return {
            key: np.stack([feature[key] for feature in features]) for key in features[0]
        }

    @staticmethod
    def _pad_to_length(
        features: dict[str, np.ndarray], length: int
    ) -> dict[str, np.ndarray]:
        padded = {}
        for key, value in features.items():
            target = length + 2 if key.startswith("lm.") else length
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
