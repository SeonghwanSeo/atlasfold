import dataclasses
import io
import pathlib
from collections import defaultdict
from functools import cached_property

import numpy as np

from atlasfold.common.protein import Protein


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

    @cached_property
    def sequence(self) -> str:
        """
        Return the concatenated sequence of all chains in the complex,
        separated by colons.
        """
        return ":".join(c.sequence for c in self.chains)

    @property
    def num_residues(self) -> int:
        """Return the number of residues in the structure."""
        return sum(c.num_residues for c in self.chains)

    # === Chain IDs and Entity IDs === #
    @property
    def asym_ids(self) -> list[int]:
        return list(range(1, self.num_chains + 1))

    @property
    def entity_ids(self) -> list[int]:
        seq_to_eid = {}
        i = 1
        for c in self.chains:
            if c.sequence not in seq_to_eid:
                seq_to_eid[c.sequence] = i
                i += 1
        return [seq_to_eid[c.sequence] for c in self.chains]

    @property
    def sym_ids(self) -> list[int]:
        seq_to_symid = defaultdict(int)
        sym_ids = []
        for c in self.chains:
            seq_to_symid[c.sequence] += 1
            sym_ids.append(seq_to_symid[c.sequence])
        return sym_ids

    # === Output formats === #
    def to_pdb(self) -> str:
        raise NotImplementedError("PDB output is not implemented yet.")

    # === Serialization === #
    def save_npz(self, path: str | pathlib.Path):
        # Save the structure to compressed npz format
        data = {}
        data["name"] = np.array([self.name], dtype="S")
        data["num_chains"] = np.array([self.num_chains], dtype=np.int64)
        for i, c in enumerate(self.chains):
            c_dict = c._to_arr_dict(atom14=False)
            for k, v in c_dict.items():
                data[f"{i}.{k}"] = v
        np.savez_compressed(path, **data)

    @classmethod
    def load_npz(cls, path: str | pathlib.Path | io.BytesIO) -> "ProteinComplex":
        """Load the protein complex structure from compressed npz format."""
        with np.load(path) as data:
            name = data["name"].item().decode("utf-8")
            num_chains = data["num_chains"].item()

            chain_dicts = defaultdict(dict)
            for key in data.files:
                if key in ["name", "num_chains"]:
                    continue
                assert "." in key
                idx_str, field = key.split(".", 1)
                assert idx_str.isdigit(), f"Invalid key format: {key}"
                chain_dicts[int(idx_str)][field] = data[key]

            chains = [
                Protein._from_arr_dict(chain_dicts[i], atom14=False)
                for i in range(num_chains)
            ]

        return cls(name, chains)
