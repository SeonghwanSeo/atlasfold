import dataclasses
import io
import logging
import pathlib

import lmdb
import msgpack
import numpy as np
import torch
import torch.nn.functional as F

from atlasfold.common import featurize, metadata, protein
from atlaslm.alphabet import Alphabet

from .utils import cropper


def pad_input(
    data: dict[str, torch.Tensor],
    max_length: int | None = None,
    multiple_of: int | None = None,
) -> dict[str, torch.Tensor]:
    def pad_fn(x: torch.Tensor) -> torch.Tensor:
        if max_length is not None:
            assert max_length >= x.shape[0], (
                f"Input length {x.shape[0]} exceeds max_length {max_length}"
            )
            pad_len = max_length - x.shape[0]
        elif multiple_of is not None:
            pad_len = (multiple_of - x.shape[0] % multiple_of) % multiple_of
        else:
            raise ValueError("Either max_length or multiple_of must be specified.")
        if pad_len > 0:
            pad = [0, 0] * (x.dim() - 1) + [0, pad_len]
            x = F.pad(x, pad)
        return x

    return {k: pad_fn(v) for k, v in data.items()}


@dataclasses.dataclass(slots=True)
class DatasetConfig:
    """Configuration for the training dataset.

    Attributes
    ----------
    name: str
        Name of the dataset, e.g., "pdb", "afdb"
    data_dir: str
        Path to the preprocessed data directory.
    metadata_path: str | str
        Path to the custom metadata file in msgpack format.
    """

    name: str
    data_dir: str | None = None
    metadata_path: str | None = None


@dataclasses.dataclass(slots=True)
class TrainingDatasetConfig(DatasetConfig):
    """Configuration for the training dataset.

    Attributes
    ----------
    is_distillation: bool
        Whether the dataset is for distillation,
        i.e., using predicted structures as labels.
    max_length: int
        Maximum sequence length for the folding model input (default: 256).
    max_seq_length: int
        Maximum sequence length for the language model input (default: 384).
    sampling_strategy: str
        Strategy for sampling training examples.
    """

    is_distillation: bool
    weight: float
    filters: list[dict] = dataclasses.field(default_factory=list)
    sampling_strategy: tuple[str, ...] = ("length",)


@dataclasses.dataclass(slots=True)
class ValidationDatasetConfig(DatasetConfig):
    """Configuration for the validation dataset."""


class LMDBDataset(torch.utils.data.Dataset):
    def __init__(self, config: DatasetConfig):
        super().__init__()
        self.name: str = config.name
        self.lm_alphabet: Alphabet = Alphabet()

        # Set path
        if config.data_dir is None:
            raise ValueError("data_dir is not specified in the config.")
        data_dir = pathlib.Path(config.data_dir)
        self.lmdb_path: str = str(data_dir / "structure.lmdb")
        if config.metadata_path is not None:
            self.metadata_path = config.metadata_path
        else:
            self.metadata_path = str(data_dir / "metadata.msgpack")

        # Load metadatas
        with open(self.metadata_path, "rb") as f:
            self.metadatas: list[dict] = msgpack.unpackb(f.read(), raw=False)

    def __len__(self) -> int:
        return len(self.metadatas)

    @property
    def lmdb_env(self) -> lmdb.Environment:
        if not hasattr(self, "_lmdb_env"):
            self._lmdb_env = lmdb.open(str(self.lmdb_path), readonly=True, lock=False)
        return self._lmdb_env

    def fetch_protein(self, key: str) -> protein.Protein:
        with self.lmdb_env.begin() as txn:
            npz_bytes = txn.get(key.encode())
        if npz_bytes is None:
            raise KeyError(f"Key {key} not found in LMDB database.")
        with io.BytesIO(npz_bytes) as f:
            prot = protein.Protein.load_npz(f)
        # Check if the residue indices are contiguous and start from 1.
        # For training data preparation, we already complete the missing residues
        # in PDB files to 'UNK' with 'NaN' coordinates.
        if prot.residue_index is not None:
            if not np.array_equal(prot.residue_index, np.arange(1, len(prot) + 1)):
                raise NotImplementedError(
                    "Structure with missing residues is not supported yet."
                )
        return prot


class TrainingDataset(LMDBDataset):
    def __init__(
        self,
        config: TrainingDatasetConfig,
        max_length: int = 256,
        max_seq_length: int = 384,
    ):
        super().__init__(config)
        self.config: TrainingDatasetConfig = config
        self.cropper = cropper.ProteinCropper()
        self.max_length: int = max_length  # Folding input length limit
        self.max_seq_length: int = max_seq_length  # LM input length limit

        self.logger = logging.getLogger(f"[Training Dataset:{self.name}]")

        self.sampling_strategy = config.sampling_strategy
        for strategy in config.sampling_strategy:
            if strategy not in ("length", "cluster"):
                raise ValueError(f"Invalid sampling strategy: {strategy}")
        self.filters = config.filters
        for filter_info in config.filters:
            if filter_info["type"] not in ("resolution", "plddt"):
                raise ValueError(f"Invalid filter type: {filter_info['type']}")

    def filter_entries(self):
        """Filter the dataset entries based on the specified criteria in the config."""

        def filter_fn(m: dict) -> bool:
            for filter_info in self.filters:
                if filter_info["type"] == "resolution":
                    r = m["exp"]["resolution"]
                    # NOTE: use NMR structures (resolution=None) for training.
                    if r is not None and r > filter_info["threshold"]:
                        return False
                elif filter_info["type"] == "plddt":
                    plddt = m["pred"]["plddt"]
                    if plddt is None:
                        self.logger.warning(
                            f"PLDDT is missing for entry {m['id']}; treating as 0."
                        )
                        plddt = 0.0
                    if np.isnan(plddt) or plddt < filter_info["threshold"]:
                        return False
            return True

        self.metadatas = [m for m in self.metadatas if filter_fn(m)]

    def get_sampling_weights(self) -> np.ndarray:
        """Get sampling weights."""

        def get_weight(m: dict) -> float:
            w = 1.0
            for strategy in self.sampling_strategy:
                if strategy == "length":
                    length = m["length"]
                    w *= min(max(length, 256), 512)
                elif strategy == "cluster":
                    cluster_size = m["cluster_size"]
                    if cluster_size == 0:
                        self.logger.warning(f"Cluster size is 0 for entry {m['id']}.")
                        cluster_size = 1
                    w *= 1 / cluster_size
                elif strategy == "plddt":
                    plddt = m["pred"]["plddt"]
                    if plddt is None:
                        self.logger.warning(
                            f"PLDDT is missing for entry {m['id']}; treating as 0."
                        )
                        plddt = 0.0
                    w *= min(max(plddt - 30, 0), 40)

            return w

        weights = np.array([get_weight(m) for m in self.metadatas])
        weights /= weights.sum()
        return weights

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, dict[str, torch.Tensor]]:
        # Create a random number generator without a fixed seed.
        rng = np.random.default_rng()

        metadata_dict = self.metadatas[index]
        m = metadata.Metadata.from_dict(metadata_dict)
        prot = self.fetch_protein(m.id)

        feat = self.prepare_folding_input(prot.sequence)
        label = self.prepare_labels(prot)
        loss_mask = self.prepare_loss_masks(m)

        # Crop the input and label if the sequence length exceeds the maximum length.
        crop_indices = self.cropper.crop(prot, self.max_length, rng)
        folding_input = {k: v[crop_indices] for k, v in feat.items()}
        label = {k: v[crop_indices] for k, v in label.items()}

        # Prepare the LM input with expanded crop indices and BOS/EOS tokens.
        lm_input = self.prepare_lm_input(prot.sequence, crop_indices)

        # Pad the input and label to the maximum length.
        folding_input = pad_input(folding_input, max_length=self.max_length)
        label = pad_input(label, max_length=self.max_length)
        lm_input = pad_input(lm_input, max_length=self.max_seq_length)
        return {
            "feat": {**folding_input, **lm_input},
            "label": label,
            "loss_mask": loss_mask,
        }

    def prepare_folding_input(self, sequence: str) -> dict[str, torch.Tensor]:
        """Prepare the folding model input features."""
        # PLM input features will be prepared separately in prepare_lm_input.
        feat = {k: torch.from_numpy(v) for k, v in featurize.featurize(sequence)}
        return {k: v for k, v in feat.items() if not k.startswith("lm.")}

    def prepare_lm_input(
        self,
        sequence: str,
        crop_indices: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        """Get the indices of the sequence tokens to include in the crop,
        expand margins if space allows, and append BOS/EOS tokens.

        Parameters
        ----------
        sequence: str
            The amino acid sequence of the protein.
        crop_indices: np.ndarray
            The indices selected by the structure cropper.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing:
            - 'input_ids': Tokenized sequence including BOS/EOS [L].
            - 'pos_id': Positional indices for the tokens [L].
            - 'is_in_crop': Boolean mask indicating folding crop [L].
            - 'mask': Attention mask for valid tokens [L].
        """
        # NOTE: We simply ensure that there is no missing residue.

        # Expand the crop indices to include neighboring residues
        seq_length = len(sequence)
        seq_crop_indices = self._expand_crop_indices_for_lm(crop_indices, seq_length)

        # Create the input IDs with BOS/EOS tokens
        input_ids = np.array(self.lm_alphabet.encode(sequence, add_special_tokens=True))
        pos_id = np.arange(len(input_ids))
        seq_id = np.ones_like(input_ids, dtype=int)

        # Crop the input IDs, positional IDs, and masks to the expanded crop indices
        lm_input = {
            "lm.input_ids": input_ids[seq_crop_indices],
            "lm.pos_id": pos_id[seq_crop_indices],
            "lm.seq_id": seq_id[seq_crop_indices],
        }

        # Add sequence to aa mapping for the cropped sequence
        is_in_crop = np.isin(crop_indices + 1, lm_input["lm.pos_id"])
        seq_to_aa = np.where(is_in_crop)[0]
        lm_input["lm.seq_to_aa"] = seq_to_aa

        return {k: torch.from_numpy(v) for k, v in lm_input.items()}

    def _expand_crop_indices_for_lm(
        self,
        crop_indices: np.ndarray,
        seqlen: int,
    ) -> np.ndarray:
        assert len(crop_indices) <= seqlen, "Crop indices cannot exceed sequence length"
        if seqlen <= self.max_seq_length - 2:
            # If the full sequence fits within the LM input limit, use the entire sequence
            return np.arange(seqlen + 2)

        # Initialize a boolean mask for the LM input tokens (including BOS and EOS)
        # BOS and EOS will include when the first/last residues are included.
        seq_crop_mask = np.zeros(seqlen + 2, dtype=bool)
        shifted_crops = crop_indices + 1  # Shift by 1 to account for BOS token at index 0
        seq_crop_mask[shifted_crops] = True

        # Determine the segments of contiguous indices in the crop
        breaks = np.where(np.diff(shifted_crops) != 1)[0] + 1
        segments = np.split(shifted_crops, breaks)
        cursors = [[seg[0], seg[-1]] for seg in segments]  # [i, j] pairs for each segment
        active_segments = list(range(len(cursors)))

        budget = self.max_seq_length - len(crop_indices)
        while budget > 0:
            for seg_idx in list(active_segments):
                i, j = cursors[seg_idx]

                # Try to expand to the left
                if i > 0 and (not seq_crop_mask[i - 1]):
                    seq_crop_mask[i - 1] = True
                    i -= 1
                    budget -= 1
                if budget <= 0:
                    break

                # Try to expand to the right
                if j < seqlen + 1 and (not seq_crop_mask[j + 1]):
                    seq_crop_mask[j + 1] = True
                    j += 1
                    budget -= 1

                if budget <= 0:
                    break

                # Check if the segment can still be expanded
                can_expand_left = i > 0 and (not seq_crop_mask[i - 1])
                can_expand_right = j < seqlen + 1 and (not seq_crop_mask[j + 1])
                if not (can_expand_left or can_expand_right):
                    active_segments.remove(seg_idx)
                else:
                    cursors[seg_idx] = [i, j]

        return np.where(seq_crop_mask)[0]

    def prepare_labels(self, prot: protein.Protein) -> dict[str, torch.Tensor]:
        """Prepare the label tensors for training."""
        # Extract the coordinates and the mask for resolved residues.
        coords = torch.from_numpy(prot.coordinates)  # [L, 14, 3]
        resolved_mask = coords.isfinite().all(dim=-1)  # [L, 14]
        return {
            "coordinates": coords,
            "resolved_mask": resolved_mask,
        }

    def prepare_loss_masks(self, m: metadata.Metadata) -> dict[str, torch.Tensor]:
        """Prepare the confidence mask for training."""
        distogram_loss = True
        diffusion_loss = True
        confidence_loss = False

        # Train confidence head only on high-quality X-ray crystal/Cyro-EM structures
        # NMR structures have resolution value of None or 0.0.
        if not self.config.is_distillation:
            assert m.exp is not None, "Experiment metadata is missing"
            resolution = m.exp.resolution
            if resolution is not None and 0.1 <= resolution <= 3.0:
                confidence_loss = True
        return {
            "distogram": torch.tensor(distogram_loss),
            "diffusion": torch.tensor(diffusion_loss),
            "confidence": torch.tensor(confidence_loss),
        }


class MultiTrainingDataset(torch.utils.data.Dataset):
    """Training dataset with AF3-style sampling and cropping."""

    def __init__(
        self,
        configs: list[TrainingDatasetConfig],
        max_length: int = 256,
        max_seq_length: int = 384,
    ) -> None:
        """
        Parameters
        ----------
        configs : list[TrainingDatasetConfig]
            List of dataset configurations.
        max_length : int, optional
            Maximum sequence length for the folding model input (default: 256).
        max_seq_length : int, optional
            Maximum sequence length for the language model input (default: 384).
        """
        self.datasets: list[TrainingDataset] = [
            TrainingDataset(config, max_length, max_seq_length) for config in configs
        ]
        ds_weights: list[np.ndarray] = [ds.get_sampling_weights() for ds in self.datasets]
        self.weights: np.ndarray = np.concatenate(
            [w * cfg.weight for w, cfg in zip(ds_weights, configs, strict=True)]
        )
        self.cumulative_sizes: np.ndarray = np.cumsum([len(ds) for ds in self.datasets])

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int) -> dict[str, dict[str, torch.Tensor]]:
        """Get the folding input for the given index."""
        # Find the dataset index
        dataset_idx = np.searchsorted(self.cumulative_sizes, index, side="right")
        if dataset_idx == 0:
            sample_idx = index
        else:
            sample_idx = index - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


class ValidationDataset(LMDBDataset):
    def __init__(self, config: ValidationDatasetConfig):
        super().__init__(config.metadata_path, config.lmdb_path)
        self.config = config
        self.name = config.name

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, dict[str, torch.Tensor]]:
        metadata_dict = self.metadatas[index]
        m = metadata.Metadata.from_dict(metadata_dict)
        prot = self.fetch_protein(m.id)

        feat = featurize.featurize(prot.sequence)
        feat = {k: torch.from_numpy(v) for k, v in feat.items()}
        label = self.prepare_labels(prot)

        # Pad the input and label to multiple of 32.
        return {
            "feat": pad_input(feat, multiple_of=32),
            "label": pad_input(label, multiple_of=32),
        }

    def prepare_labels(self, prot: protein.Protein) -> dict[str, torch.Tensor]:
        """Prepare the label tensors for training."""
        # Extract the coordinates and the mask for resolved residues.
        coords = torch.from_numpy(prot.coordinates)  # [L, 14, 3]
        resolved_mask = coords.isfinite().all(dim=-1)  # [L, 14]
        return {
            "coordinates": coords,
            "resolved_mask": resolved_mask,
        }
