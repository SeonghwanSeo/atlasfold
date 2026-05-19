import numpy as np

# one-letter codes for standard amino acids and nucleic acid bases
amino_acids: tuple[str, ...] = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "UNK"
)  # fmt: skip
restypes = [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
    "X",
]  # fmt: skip
restype_orders = {r: i for i, r in enumerate(restypes)}

restype_3to1: dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "UNK": "X",
}
restype_1to3: dict[str, str] = {v: k for k, v in restype_3to1.items()}

atom_names: tuple[str, ...] = (
    "N",   "CA",  "C",   "CB",  "O",   "CG",  "CG1", "CG2", "OG",  "OG1",
    "SG",  "CD",  "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD",  "CE",
    "CE1", "CE2", "CE3", "NE",  "NE1", "NE2", "OE1", "OE2", "CH2", "NH1",
    "NH2", "OH",  "CZ",  "CZ2", "CZ3", "NZ",  "OXT"
)  # fmt: skip

residue_atoms: dict[str, tuple[str, ...]] = {
    "ALA": ("N", "CA", "C", "O", "CB"),
    "ARG": ("N", "CA", "C", "O", "CB", "CG",  "CD",  "NE",  "CZ",  "NH1", "NH2"),
    "ASN": ("N", "CA", "C", "O", "CB", "CG",  "OD1", "ND2"),
    "ASP": ("N", "CA", "C", "O", "CB", "CG",  "OD1", "OD2"),
    "CYS": ("N", "CA", "C", "O", "CB", "SG"),
    "GLN": ("N", "CA", "C", "O", "CB", "CG",  "CD",  "OE1", "NE2"),
    "GLU": ("N", "CA", "C", "O", "CB", "CG",  "CD",  "OE1", "OE2"),
    "GLY": ("N", "CA", "C", "O"),
    "HIS": ("N", "CA", "C", "O", "CB", "CG",  "ND1", "CD2", "CE1", "NE2"),
    "ILE": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"),
    "LEU": ("N", "CA", "C", "O", "CB", "CG",  "CD1", "CD2"),
    "LYS": ("N", "CA", "C", "O", "CB", "CG",  "CD",  "CE",  "NZ"),
    "MET": ("N", "CA", "C", "O", "CB", "CG",  "SD",  "CE"),
    "PHE": ("N", "CA", "C", "O", "CB", "CG",  "CD1", "CD2", "CE1", "CE2", "CZ"),
    "PRO": ("N", "CA", "C", "O", "CB", "CG",  "CD"),
    "SER": ("N", "CA", "C", "O", "CB", "OG"),
    "THR": ("N", "CA", "C", "O", "CB", "OG1", "CG2"),
    "TRP": ("N", "CA", "C", "O", "CB", "CG",  "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),  # noqa
    "TYR": ("N", "CA", "C", "O", "CB", "CG",  "CD1", "CD2", "CE1", "CE2", "CZ",  "OH"),
    "VAL": ("N", "CA", "C", "O", "CB", "CG1", "CG2"),
    "UNK": ("N", "CA", "C", "O", "CB"),
}  # fmt: skip
restype_atom14_order: dict[str, dict[str, int]] = {
    res: {atom: idx for idx, atom in enumerate(atoms)}
    for res, atoms in residue_atoms.items()
}
num_residue_atoms: dict[str, int] = {
    res: len(atoms) for res, atoms in residue_atoms.items()
}

atom37_order: dict[str, int] = {name: idx for idx, name in enumerate(atom_names)}
restype_atom37_mask: np.ndarray = np.zeros((21, 37), dtype=bool)
restype_atom14_mask: np.ndarray = np.zeros((21, 14), dtype=bool)
for res_name, atoms in residue_atoms.items():
    res_idx = restype_orders[restype_3to1[res_name]]
    restype_atom14_mask[res_idx, : len(atoms)] = True
    for atom_name in atoms:
        atom_idx = atom37_order[atom_name]
        restype_atom37_mask[res_idx, atom_idx] = True


def get_atom14_mask_from_sequence(sequence: str) -> np.ndarray:
    """Get a boolean mask indicating which atoms are present in the structure"""
    res_indices = [restype_orders[aa] for aa in sequence]
    res_indices = np.array(res_indices)
    return restype_atom14_mask[res_indices]


def get_atom37_mask_from_sequence(sequence: str) -> np.ndarray:
    """Get a boolean mask indicating which atoms are present in the structure"""
    res_indices = [restype_orders[aa] for aa in sequence]
    res_indices = np.array(res_indices)
    return restype_atom37_mask[res_indices]


def _get_atom14_to_atom37_mapping() -> tuple[np.ndarray, np.ndarray]:
    gather_indices = np.zeros((21, 37), dtype=np.int64)
    gather_mask = np.zeros((21, 37), dtype=bool)
    for res_name, atoms in residue_atoms.items():
        res_idx = restype_orders[restype_3to1[res_name]]
        for atom14_idx, atom_name in enumerate(atoms):
            if atom_name in atom37_order:
                atom37_idx = atom37_order[atom_name]
                gather_indices[res_idx, atom37_idx] = atom14_idx
                gather_mask[res_idx, atom37_idx] = True
    return gather_indices, gather_mask


_gather_indices, _gather_mask = _get_atom14_to_atom37_mapping()


# Swappable atoms for each residue type
restype_ambiguous_atoms: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "ASP": (("OD1",), ("OD2",)),
    "GLU": (("OE1",), ("OE2",)),
    "PHE": (("CD1", "CE1"), ("CD2", "CE2")),
    "TYR": (("CD1", "CE1"), ("CD2", "CE2")),
    "ARG": (("NH1",), ("NH2",)),
    "LEU": (("CD1",), ("CD2",)),
    "VAL": (("CG1",), ("CG2",)),
}
