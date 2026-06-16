import dataclasses
import pathlib
from collections import defaultdict
from functools import cached_property

import gemmi
import numpy as np
from typing_extensions import Self

from atlasfold.common import file_io, residue_constants


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
        # TODO: support the structure with missing residues
        if self.residue_index is not None:
            if self.residue_index.shape != (L,):
                raise ValueError(
                    f"Invalid residue_index shape: {self.residue_index.shape}. "
                    f"Expected (L,) where L is the sequence length."
                )
            if not np.all(np.diff(self.residue_index) == 1):
                raise ValueError(
                    "residue_index must be a continuous range of integers. "
                    f"Got {self.residue_index}."
                )

    def __len__(self):
        """Return the number of residues in the structure."""
        return len(self.sequence)

    @property
    def num_residues(self) -> int:
        """Return the number of residues in the structure."""
        return len(self.sequence)

    @classmethod
    def get_empty(cls, name: str, sequence: str) -> Self:
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
        return file_io.to_pdb(self.name, self.sequence, self.coordinates, self.b_factors)

    def to_mmcif(self) -> str:
        return file_io.to_mmcif(
            self.name, self.sequence, self.coordinates, self.b_factors
        )

    @classmethod
    def from_pdb(
        cls,
        path: str | pathlib.Path,
        chain_id: int | str | None = None,
        name: str | None = None,
    ) -> Self:
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
            aa1 = residue_constants.restype_3to1.get(res.name, "X")
            aa3 = residue_constants.restype_1to3[aa1]
            res_atom14_order = residue_constants.restype_atom14_order[aa3]
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
class ProteinComplex:
    """A data structure representing a protein complex"""

    name: str
    chains: list[Protein]

    def __len__(self):
        """Return the number of chains in the complex."""
        return self.num_chains

    @property
    def num_chains(self) -> int:
        """Return the number of chains in the complex."""
        return len(self.chains)

    @property
    def sequences(self) -> list[str]:
        """
        Return the list of sequences for each chain in the complex.
        """
        return [c.sequence for c in self.chains]

    @property
    def sequence(self) -> str:
        """
        Return the concatenated sequence of all chains in the complex,
        separated by colons.
        """
        return ":".join(self.sequences)

    @property
    def num_residues(self) -> int:
        """Return the number of residues in the structure."""
        return sum(c.num_residues for c in self.chains)

    # === Chain IDs and Entity IDs === #
    @cached_property
    def asym_ids(self) -> list[int]:
        return list(range(1, self.num_chains + 1))

    @cached_property
    def entity_ids(self) -> list[int]:
        seq_to_eid = {}
        i = 1
        for c in self.chains:
            if c.sequence not in seq_to_eid:
                seq_to_eid[c.sequence] = i
                i += 1
        return [seq_to_eid[c.sequence] for c in self.chains]

    @cached_property
    def sym_ids(self) -> list[int]:
        seq_to_symid = defaultdict(int)
        sym_ids = []
        for c in self.chains:
            seq_to_symid[c.sequence] += 1
            sym_ids.append(seq_to_symid[c.sequence])
        return sym_ids
