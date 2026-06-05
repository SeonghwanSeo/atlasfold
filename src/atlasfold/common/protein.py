import dataclasses
import io
import pathlib

import gemmi
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
    def get_empty(cls, name: str, sequence: str) -> "Protein":
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

    def to_mmcif(self) -> str:
        raise NotImplementedError("mmCIF output is not implemented yet.")

    # === Serialization === #
    def save_npz(self, path: str | pathlib.Path):
        np.savez_compressed(path, **self._to_arr_dict(atom14=False))

    @classmethod
    def load_npz(cls, path: str | pathlib.Path | io.BytesIO) -> "Protein":
        # Load the structure from compressed npz format
        with np.load(path) as data:
            return cls._from_arr_dict(data, atom14=False)

    def _to_arr_dict(self, atom14: bool) -> dict:
        name_arr = np.array([self.name], dtype="S")
        seq_arr = np.array([self.sequence], dtype="S")
        if atom14:
            coords_arr = self.coordinates  # [L, 14, 3]
            b_factors_arr = self.b_factors  # [L, 14]
        else:
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
        return data

    @classmethod
    def _from_arr_dict(cls, data: dict, atom14: bool) -> "Protein":
        name = data["name"].item().decode("utf-8")
        sequence = data["sequence"].item().decode("utf-8")
        if atom14:
            coordinates = data["coordinates"]  # [L, 14, 3]
            b_factors = data["b_factors"]  # [L, 14]
        else:
            coords_arr = data["coordinates"]  # [Natom, 3]
            b_factors_arr = data["b_factors"]  # [Natom]
            L = len(sequence)
            full_coords = np.full((L, 14, 3), np.nan, dtype=np.float32)
            pad_mask = residue_utils.get_atom14_mask_from_sequence(sequence)
            full_coords[pad_mask] = coords_arr
            full_b_factors = np.full((L, 14), np.nan, dtype=np.float32)
            full_b_factors[pad_mask] = b_factors_arr
            coordinates = full_coords
            b_factors = full_b_factors
        residue_index = data.get("residue_index", None)
        return cls(name, sequence, coordinates, b_factors, residue_index)

    @classmethod
    def from_pdb(
        cls,
        path: str | pathlib.Path,
        chain_id: int | str | None = None,
        name: str | None = None,
    ) -> "Protein":
        """Load a predicted protein structure from single monomer file."""
        if name is None:
            name = pathlib.Path(path).name.split(".")[0]

        struct: gemmi.Structure = gemmi.read_structure(str(path))
        model: gemmi.Model = struct[0]

        # Get chain
        if chain_id is not None:
            raw_chain = model[chain_id]
        else:
            raw_chain = model[0]

        # Read chain
        aa_list, coords_list, biso_list, res_idx_list = [], [], [], []
        for res in raw_chain:
            aa1 = residue_utils.restype_3to1.get(res.name, "X")
            aa3 = residue_utils.restype_1to3[aa1]
            res_atom14_order = residue_utils.restype_atom14_order[aa3]
            res_coords = np.full((14, 3), np.nan, dtype=np.float32)
            res_b_factors = np.full((14,), np.nan, dtype=np.float32)
            for atom in res:
                atom_name: str = atom.name.upper().strip()
                if atom_name in res_atom14_order:
                    atom_i: int = res_atom14_order[atom_name]
                    res_coords[atom_i, :] = atom.pos.tolist()
                    res_b_factors[atom_i] = atom.b_iso
            aa_list.append(aa1)
            coords_list.append(res_coords)
            biso_list.append(res_b_factors)
            res_idx_list.append(res.seqid.num)

        sequence = "".join(aa_list)
        coordinates = np.stack(coords_list, axis=0)  # [L, 14, 3]
        b_factors = np.stack(biso_list, axis=0)  # [L, 14]
        residue_index = np.array(res_idx_list, dtype=np.int32)  # [L,]
        return cls(name, sequence, coordinates, b_factors, residue_index)


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
        raise NotImplementedError("PDB output is not implemented yet.")

    def to_mmcif(self) -> str:
        raise NotImplementedError("mmCIF output is not implemented yet.")
