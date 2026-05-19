import dataclasses
import io
import pathlib

import numpy as np

from atlasfold.common import residue_utils


@dataclasses.dataclass
class Protein:
    """A data structure representing a 3D protein structure"""

    name: str
    sequence: str
    coordinates: np.ndarray  # [L, 14, 3]
    b_factors: np.ndarray  # [L, 14]
    residue_index: np.ndarray | None = None  # [L], optional residue index

    def __post_init__(self):
        """Validate the input data."""
        L = len(self.sequence)
        if self.coordinates.shape not in [(L, 14, 3)]:
            raise ValueError(
                f"Invalid coordinates shape: {self.coordinates.shape}. "
                f"Expected (L, 14, 3) where L is the sequence length."
            )

    def __len__(self):
        """Return the number of residues in the structure."""
        return len(self.sequence)

    @property
    def num_residues(self) -> int:
        """Return the number of residues in the structure."""
        return len(self.sequence)

    @classmethod
    def get_empty_chain(cls, name: str, sequence: str) -> "Protein":
        """Create an empty structure with NaN coordinates."""
        L = len(sequence)
        coordinates = np.full((L, 14, 3), np.nan, dtype=np.float32)
        b_factors = np.full((L, 14), np.nan, dtype=np.float32)
        return cls(name, sequence, coordinates, b_factors)

    @property
    def atom_mask(self) -> np.ndarray:
        """Return a boolean mask indicating which atoms are present in the structure."""
        return np.isfinite(self.coordinates).all(axis=-1)

    def to_pdb(self) -> str:
        raise NotImplementedError("PDB output is not implemented yet.")

    # === Serialization === #
    def save_npz(self, path: str | pathlib.Path):
        # Save the structure to compressed npz format
        name_arr = np.array([self.name], dtype="S")
        seq_arr = np.array([self.sequence], dtype="S")
        # Only save the valid atom coordinates (include unresolved atoms as NaN)
        pad_mask = residue_utils.get_atom14_mask_from_sequence(self.sequence)
        coords_arr = self.coordinates[pad_mask]  # [Natom, 3]
        b_factors_arr = self.b_factors[pad_mask]
        data: dict[str, np.ndarray] = {
            "name": name_arr,
            "sequence": seq_arr,
            "coordinates": coords_arr,
            "b_factors": b_factors_arr,
        }
        if self.residue_index is not None:
            data["residue_index"] = self.residue_index
        np.savez_compressed(path, **data)

    @classmethod
    def load_npz(cls, path: str | pathlib.Path | io.BytesIO) -> "Protein":
        # Load the structure from compressed npz format
        data = np.load(path)
        name = data["name"].item().decode("utf-8")
        sequence = data["sequence"].item().decode("utf-8")
        coordinates = data["coordinates"]  # [Natom, 3]
        b_factors = data["b_factors"]  # [Natom]
        residue_index = data.get("residue_index", None)  # [L], optional

        # Reconstruct the full coordinates array with NaN for missing atoms
        L = len(sequence)
        full_coords = np.full((L, 14, 3), np.nan, dtype=np.float32)
        pad_mask = residue_utils.get_atom14_mask_from_sequence(sequence)
        full_coords[pad_mask] = coordinates
        full_b_factors = np.full((L, 14), np.nan, dtype=np.float32)
        full_b_factors[pad_mask] = b_factors
        data.close()

        return cls(name, sequence, full_coords, full_b_factors, residue_index)
