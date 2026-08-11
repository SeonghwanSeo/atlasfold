import functools
from typing import NamedTuple

import numpy as np

from atlasfold.common import ccd

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

# Four atoms defining each side-chain chi angle. This is the canonical AF2
# residue-geometry table from alphafold/common/residue_constants.py.
chi_angles_atoms: dict[str, list[list[str]]] = {
    "ALA": [],
    "ARG": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "NE"],
        ["CG", "CD", "NE", "CZ"],
    ],
    "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "CYS": [["N", "CA", "CB", "SG"]],
    "GLN": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLU": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLY": [],
    "HIS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "LYS": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "CE"],
        ["CG", "CD", "CE", "NZ"],
    ],
    "MET": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "SD"],
        ["CB", "CG", "SD", "CE"],
    ],
    "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "PRO": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
    "SER": [["N", "CA", "CB", "OG"]],
    "THR": [["N", "CA", "CB", "OG1"]],
    "TRP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "VAL": [["N", "CA", "CB", "CG1"]],
    "UNK": [],
}

ambiguous_restype_mapping: dict[str, str] = {
    "B": "D",  # Aspartic acid or Asparagine
    "U": "C",  # Selenocysteine (treated as Cysteine)
    "Z": "E",  # Glutamic acid or Glutamine
}


def convert_full_sequence_to_sequence(
    full_sequence: list[str],
    standardize_ambiguous: bool = False,
) -> str:
    """Convert a full sequence (with ambiguous residues) to a standard sequence."""

    def three_to_one(aa3: str) -> str:
        aa3 = str(aa3).upper().strip()
        aa = ccd.CCD_NAME_TO_ONE_LETTER.get(aa3, "X")
        if standardize_ambiguous:
            aa = ambiguous_restype_mapping.get(aa, aa)
        if aa not in restype_orders:
            aa = "X"
        return aa

    return "".join(map(three_to_one, full_sequence))


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


class Bond(NamedTuple):
    atom1_name: str
    atom2_name: str
    length: float
    stddev: float


class BondAngle(NamedTuple):
    atom1_name: str
    atom2_name: str
    atom3_name: str
    angle_rad: float
    stddev: float


@functools.lru_cache(maxsize=1)
def load_stereo_chemical_props() -> tuple[
    dict[str, list[Bond]],
    dict[str, list[Bond]],
    dict[str, list[BondAngle]],
]:
    """Load the pinned OpenStructure properties used by AF2/OpenFold."""
    from importlib import resources

    text = (
        resources.files("atlasfold")
        .joinpath("resources", "stereo_chemical_props.txt")
        .read_text(encoding="utf-8")
    )
    lines = iter(text.splitlines())
    next(lines)
    residue_bonds: dict[str, list[Bond]] = {}
    for line in lines:
        if line.strip() == "-":
            break
        bond, resname, length, stddev = line.split()
        atom1, atom2 = bond.split("-")
        residue_bonds.setdefault(resname, []).append(
            Bond(atom1, atom2, float(length), float(stddev))
        )
    residue_bonds["UNK"] = []

    next(lines)
    next(lines)
    residue_angles: dict[str, list[BondAngle]] = {}
    for line in lines:
        if line.strip() == "-":
            break
        angle, resname, mean, stddev = line.split()
        atom1, atom2, atom3 = angle.split("-")
        residue_angles.setdefault(resname, []).append(
            BondAngle(
                atom1,
                atom2,
                atom3,
                np.deg2rad(float(mean)),
                np.deg2rad(float(stddev)),
            )
        )
    residue_angles["UNK"] = []

    def bond_key(atom1: str, atom2: str) -> tuple[str, str]:
        return tuple(sorted((atom1, atom2)))

    virtual_bonds: dict[str, list[Bond]] = {}
    for resname, angles in residue_angles.items():
        bond_cache = {
            bond_key(bond.atom1_name, bond.atom2_name): bond
            for bond in residue_bonds[resname]
        }
        virtual_bonds[resname] = []
        for angle in angles:
            bond1 = bond_cache[bond_key(angle.atom1_name, angle.atom2_name)]
            bond2 = bond_cache[bond_key(angle.atom2_name, angle.atom3_name)]
            length = np.sqrt(
                bond1.length**2
                + bond2.length**2
                - 2 * bond1.length * bond2.length * np.cos(angle.angle_rad)
            )
            derivative_scale = 0.5 / length
            dl_dangle = (
                2 * bond1.length * bond2.length * np.sin(angle.angle_rad)
            ) * derivative_scale
            dl_dbond1 = (
                2 * bond1.length - 2 * bond2.length * np.cos(angle.angle_rad)
            ) * derivative_scale
            dl_dbond2 = (
                2 * bond2.length - 2 * bond1.length * np.cos(angle.angle_rad)
            ) * derivative_scale
            stddev = np.sqrt(
                (dl_dangle * angle.stddev) ** 2
                + (dl_dbond1 * bond1.stddev) ** 2
                + (dl_dbond2 * bond2.stddev) ** 2
            )
            virtual_bonds[resname].append(
                Bond(angle.atom1_name, angle.atom3_name, float(length), float(stddev))
            )

    return residue_bonds, virtual_bonds, residue_angles


@functools.lru_cache(maxsize=8)
def make_atom14_dists_bounds(
    overlap_tolerance: float = 1.5,
    bond_length_tolerance_factor: float = 12.0,
) -> dict[str, np.ndarray]:
    """Create AF2 atom14 lower/upper distance bounds for violation loss."""

    lower = np.zeros((21, 14, 14), dtype=np.float32)
    upper = np.zeros((21, 14, 14), dtype=np.float32)
    stddev = np.zeros((21, 14, 14), dtype=np.float32)
    residue_bonds, virtual_bonds, _ = load_stereo_chemical_props()
    radii = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8}

    for restype, letter in enumerate(restypes[:20]):
        resname = restype_1to3[letter]
        atoms = residue_atoms[resname]
        for atom1_idx, atom1 in enumerate(atoms):
            for atom2_idx, atom2 in enumerate(atoms):
                if atom1_idx == atom2_idx:
                    continue
                lower[restype, atom1_idx, atom2_idx] = (
                    radii[atom1[0]] + radii[atom2[0]] - overlap_tolerance
                )
                upper[restype, atom1_idx, atom2_idx] = 1e10

        for bond in residue_bonds[resname] + virtual_bonds[resname]:
            atom1_idx = restype_atom14_order[resname][bond.atom1_name]
            atom2_idx = restype_atom14_order[resname][bond.atom2_name]
            bond_lower = bond.length - bond_length_tolerance_factor * bond.stddev
            bond_upper = bond.length + bond_length_tolerance_factor * bond.stddev
            lower[restype, atom1_idx, atom2_idx] = bond_lower
            lower[restype, atom2_idx, atom1_idx] = bond_lower
            upper[restype, atom1_idx, atom2_idx] = bond_upper
            upper[restype, atom2_idx, atom1_idx] = bond_upper
            stddev[restype, atom1_idx, atom2_idx] = bond.stddev
            stddev[restype, atom2_idx, atom1_idx] = bond.stddev

    # IPA consistently treats unknown residues as alanine geometry.
    ala_idx = restype_orders["A"]
    lower[20], upper[20], stddev[20] = lower[ala_idx], upper[ala_idx], stddev[ala_idx]
    return {"lower_bound": lower, "upper_bound": upper, "stddev": stddev}


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

# Reference conformers for each residue
restype_ref_atom_positions: dict[str, dict[str, tuple[float, float, float]]] = {
    "ALA": {
        "N": (-0.525, 1.363, 0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.526, -0.000, -0.000),
        "O": (0.627, 1.062, 0.000),
        "CB": (-0.529, -0.774, -1.205),
    },
    "ARG": {
        "N": (-0.524, 1.362, -0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.525, -0.000, -0.000),
        "O": (0.626, 1.062, 0.000),
        "CB": (-0.524, -0.778, -1.209),
        "CG": (0.616, 1.390, -0.000),
        "CD": (0.564, 1.414, 0.000),
        "NE": (0.539, 1.357, -0.000),
        "CZ": (0.758, 1.093, -0.000),
        "NH1": (0.206, 2.301, 0.000),
        "NH2": (2.078, 0.978, -0.000),
    },
    "ASN": {
        "N": (-0.536, 1.357, 0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.526, -0.000, -0.000),
        "O": (0.625, 1.062, 0.000),
        "CB": (-0.531, -0.787, -1.200),
        "CG": (0.584, 1.399, 0.000),
        "OD1": (0.633, 1.059, 0.000),
        "ND2": (0.593, -1.188, 0.001),
    },
    "ASP": {
        "N": (-0.525, 1.362, -0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.527, 0.000, -0.000),
        "O": (0.626, 1.062, -0.000),
        "CB": (-0.526, -0.778, -1.208),
        "CG": (0.593, 1.398, -0.000),
        "OD1": (0.610, 1.091, 0.000),
        "OD2": (0.592, -1.101, -0.003),
    },
    "CYS": {
        "N": (-0.522, 1.362, -0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.524, 0.000, 0.000),
        "O": (0.625, 1.062, -0.000),
        "CB": (-0.519, -0.773, -1.212),
        "SG": (0.728, 1.653, 0.000),
    },
    "GLN": {
        "N": (-0.526, 1.361, -0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.526, 0.000, 0.000),
        "O": (0.626, 1.062, -0.000),
        "CB": (-0.525, -0.779, -1.207),
        "CG": (0.615, 1.393, 0.000),
        "CD": (0.587, 1.399, -0.000),
        "OE1": (0.634, 1.060, 0.000),
        "NE2": (0.593, -1.189, -0.001),
    },
    "GLU": {
        "N": (-0.528, 1.361, 0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.526, -0.000, -0.000),
        "O": (0.626, 1.062, 0.000),
        "CB": (-0.526, -0.781, -1.207),
        "CG": (0.615, 1.392, 0.000),
        "CD": (0.600, 1.397, 0.000),
        "OE1": (0.607, 1.095, -0.000),
        "OE2": (0.589, -1.104, -0.001),
    },
    "GLY": {
        "N": (-0.572, 1.337, 0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.517, -0.000, -0.000),
        "O": (0.626, 1.062, -0.000),
    },
    "HIS": {
        "N": (-0.527, 1.360, 0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.525, 0.000, 0.000),
        "O": (0.625, 1.063, 0.000),
        "CB": (-0.525, -0.778, -1.208),
        "CG": (0.600, 1.370, -0.000),
        "ND1": (0.744, 1.160, -0.000),
        "CD2": (0.889, -1.021, 0.003),
        "CE1": (2.030, 0.851, 0.002),
        "NE2": (2.145, -0.466, 0.004),
    },
    "ILE": {
        "N": (-0.493, 1.373, -0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.527, -0.000, -0.000),
        "O": (0.627, 1.062, -0.000),
        "CB": (-0.536, -0.793, -1.213),
        "CG1": (0.534, 1.437, -0.000),
        "CG2": (0.540, -0.785, -1.199),
        "CD1": (0.619, 1.391, 0.000),
    },
    "LEU": {
        "N": (-0.520, 1.363, 0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.525, -0.000, -0.000),
        "O": (0.625, 1.063, -0.000),
        "CB": (-0.522, -0.773, -1.214),
        "CG": (0.678, 1.371, 0.000),
        "CD1": (0.530, 1.430, -0.000),
        "CD2": (0.535, -0.774, 1.200),
    },
    "LYS": {
        "N": (-0.526, 1.362, -0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.526, 0.000, 0.000),
        "O": (0.626, 1.062, -0.000),
        "CB": (-0.524, -0.778, -1.208),
        "CG": (0.619, 1.390, 0.000),
        "CD": (0.559, 1.417, 0.000),
        "CE": (0.560, 1.416, 0.000),
        "NZ": (0.554, 1.387, 0.000),
    },
    "MET": {
        "N": (-0.521, 1.364, -0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.525, 0.000, 0.000),
        "O": (0.625, 1.062, -0.000),
        "CB": (-0.523, -0.776, -1.210),
        "CG": (0.613, 1.391, -0.000),
        "SD": (0.703, 1.695, 0.000),
        "CE": (0.320, 1.786, -0.000),
    },
    "PHE": {
        "N": (-0.518, 1.363, 0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.524, 0.000, -0.000),
        "O": (0.626, 1.062, -0.000),
        "CB": (-0.525, -0.776, -1.212),
        "CG": (0.607, 1.377, 0.000),
        "CD1": (0.709, 1.195, -0.000),
        "CD2": (0.706, -1.196, 0.000),
        "CE1": (2.102, 1.198, -0.000),
        "CE2": (2.098, -1.201, -0.000),
        "CZ": (2.794, -0.003, -0.001),
    },
    "PRO": {
        "N": (-0.566, 1.351, -0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.527, -0.000, 0.000),
        "O": (0.621, 1.066, 0.000),
        "CB": (-0.546, -0.611, -1.293),
        "CG": (0.382, 1.445, 0.0),
        "CD": (0.477, 1.424, 0.0),
    },
    "SER": {
        "N": (-0.529, 1.360, -0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.525, -0.000, -0.000),
        "O": (0.626, 1.062, -0.000),
        "CB": (-0.518, -0.777, -1.211),
        "OG": (0.503, 1.325, 0.000),
    },
    "THR": {
        "N": (-0.517, 1.364, 0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.526, 0.000, -0.000),
        "O": (0.626, 1.062, 0.000),
        "CB": (-0.516, -0.793, -1.215),
        "OG1": (0.472, 1.353, 0.000),
        "CG2": (0.550, -0.718, -1.228),
    },
    "TRP": {
        "N": (-0.521, 1.363, 0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.525, -0.000, 0.000),
        "O": (0.627, 1.062, 0.000),
        "CB": (-0.523, -0.776, -1.212),
        "CG": (0.609, 1.370, -0.000),
        "CD1": (0.824, 1.091, 0.000),
        "CD2": (0.854, -1.148, -0.005),
        "NE1": (2.140, 0.690, -0.004),
        "CE2": (2.186, -0.678, -0.007),
        "CE3": (0.622, -2.530, -0.007),
        "CZ2": (3.283, -1.543, -0.011),
        "CZ3": (1.715, -3.389, -0.011),
        "CH2": (3.028, -2.890, -0.013),
    },
    "TYR": {
        "N": (-0.522, 1.362, 0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.524, -0.000, -0.000),
        "O": (0.627, 1.062, -0.000),
        "CB": (-0.522, -0.776, -1.213),
        "CG": (0.607, 1.382, -0.000),
        "CD1": (0.716, 1.195, -0.000),
        "CD2": (0.713, -1.194, -0.001),
        "CE1": (2.107, 1.200, -0.002),
        "CE2": (2.104, -1.201, -0.003),
        "CZ": (2.791, -0.001, -0.003),
        "OH": (4.168, -0.002, -0.005),
    },
    "VAL": {
        "N": (-0.494, 1.373, -0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.527, -0.000, -0.000),
        "O": (0.627, 1.062, -0.000),
        "CB": (-0.533, -0.795, -1.213),
        "CG1": (0.540, 1.429, -0.000),
        "CG2": (0.533, -0.776, 1.203),
    },
    "UNK": {
        "N": (-0.525, 1.363, 0.000),
        "CA": (0.000, 0.000, 0.000),
        "C": (1.526, -0.000, -0.000),
        "O": (0.627, 1.062, 0.000),
        "CB": (-0.529, -0.774, -1.205),
    },
}
restype_atom14_positions: dict[str, np.ndarray] = {}
for res_name, atom_positions in restype_ref_atom_positions.items():
    res_idx = restype_orders[restype_3to1[res_name]]
    atom14_positions = np.zeros((14, 3), dtype=np.float32)
    for atom_name, position in atom_positions.items():
        if atom_name in restype_atom14_order[res_name]:
            atom_idx = restype_atom14_order[res_name][atom_name]
            atom14_positions[atom_idx] = position
    restype_atom14_positions[res_name] = atom14_positions


# Atom-to-rigid-group assignments from AF2's rigid_group_atom_positions table.
# Atoms omitted from a residue entry belong to the backbone group (0).
_rigid_group_by_atom: dict[str, dict[str, int]] = {
    "ALA": {"O": 3},
    "ARG": {"O": 3, "CG": 4, "CD": 5, "NE": 6, "CZ": 7, "NH1": 7, "NH2": 7},
    "ASN": {"O": 3, "CG": 4, "OD1": 5, "ND2": 5},
    "ASP": {"O": 3, "CG": 4, "OD1": 5, "OD2": 5},
    "CYS": {"O": 3, "SG": 4},
    "GLN": {"O": 3, "CG": 4, "CD": 5, "OE1": 6, "NE2": 6},
    "GLU": {"O": 3, "CG": 4, "CD": 5, "OE1": 6, "OE2": 6},
    "GLY": {"O": 3},
    "HIS": {"O": 3, "CG": 4, "ND1": 5, "CD2": 5, "CE1": 5, "NE2": 5},
    "ILE": {"O": 3, "CG1": 4, "CG2": 4, "CD1": 5},
    "LEU": {"O": 3, "CG": 4, "CD1": 5, "CD2": 5},
    "LYS": {"O": 3, "CG": 4, "CD": 5, "CE": 6, "NZ": 7},
    "MET": {"O": 3, "CG": 4, "SD": 5, "CE": 6},
    "PHE": {"O": 3, "CG": 4, "CD1": 5, "CD2": 5, "CE1": 5, "CE2": 5, "CZ": 5},
    "PRO": {"O": 3, "CG": 4, "CD": 5},
    "SER": {"O": 3, "OG": 4},
    "THR": {"O": 3, "OG1": 4, "CG2": 4},
    "TRP": {
        "O": 3,
        "CG": 4,
        "CD1": 5,
        "CD2": 5,
        "NE1": 5,
        "CE2": 5,
        "CE3": 5,
        "CZ2": 5,
        "CZ3": 5,
        "CH2": 5,
    },
    "TYR": {"O": 3, "CG": 4, "CD1": 5, "CD2": 5, "CE1": 5, "CE2": 5, "CZ": 5, "OH": 5},
    "VAL": {"O": 3, "CG1": 4, "CG2": 4},
}

rigid_group_atom_positions: dict[
    str, list[tuple[str, int, tuple[float, float, float]]]
] = {
    resname: [
        (atom_name, _rigid_group_by_atom[resname].get(atom_name, 0), position)
        for atom_name, position in atom_positions.items()
    ]
    for resname, atom_positions in restype_ref_atom_positions.items()
    if resname != "UNK"
}


def _make_rigid_transformation_4x4(
    ex: np.ndarray,
    ey: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Create a rigid transform from two axes and a translation."""

    ex = ex / np.linalg.norm(ex)
    ey = ey - np.dot(ey, ex) * ex
    ey = ey / np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    matrix = np.stack((ex, ey, ez, translation), axis=-1)
    return np.concatenate(
        (matrix, np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)),
        axis=0,
    ).astype(np.float32)


restype_atom14_to_rigid_group = np.zeros((21, 14), dtype=np.int64)
restype_atom14_rigid_group_positions = np.zeros((21, 14, 3), dtype=np.float32)
restype_rigid_group_default_frame = np.zeros((21, 8, 4, 4), dtype=np.float32)


def _make_rigid_group_constants() -> None:
    """Fill the dense AF2 rigid-group lookup arrays."""

    for restype_idx, restype_letter in enumerate(restypes[:20]):
        resname = restype_1to3[restype_letter]
        atom_positions = restype_ref_atom_positions[resname]
        atom_order = restype_atom14_order[resname]

        for atom_name, group_idx, position in rigid_group_atom_positions[resname]:
            atom14_idx = atom_order[atom_name]
            restype_atom14_to_rigid_group[restype_idx, atom14_idx] = group_idx
            restype_atom14_rigid_group_positions[restype_idx, atom14_idx] = position

        restype_rigid_group_default_frame[restype_idx, 0] = np.eye(4, dtype=np.float32)
        restype_rigid_group_default_frame[restype_idx, 1] = np.eye(4, dtype=np.float32)
        restype_rigid_group_default_frame[restype_idx, 2] = (
            _make_rigid_transformation_4x4(
                np.asarray(atom_positions["N"]) - np.asarray(atom_positions["CA"]),
                np.array([1.0, 0.0, 0.0]),
                np.asarray(atom_positions["N"]),
            )
        )
        restype_rigid_group_default_frame[restype_idx, 3] = (
            _make_rigid_transformation_4x4(
                np.asarray(atom_positions["C"]) - np.asarray(atom_positions["CA"]),
                np.asarray(atom_positions["CA"]) - np.asarray(atom_positions["N"]),
                np.asarray(atom_positions["C"]),
            )
        )

        chi_angles = chi_angles_atoms[resname]
        if chi_angles:
            base = [np.asarray(atom_positions[name]) for name in chi_angles[0]]
            restype_rigid_group_default_frame[restype_idx, 4] = (
                _make_rigid_transformation_4x4(
                    base[2] - base[1],
                    base[0] - base[1],
                    base[2],
                )
            )

        for chi_idx in range(1, len(chi_angles)):
            axis_end = np.asarray(atom_positions[chi_angles[chi_idx][2]])
            restype_rigid_group_default_frame[restype_idx, 4 + chi_idx] = (
                _make_rigid_transformation_4x4(
                    axis_end, np.array([-1.0, 0.0, 0.0]), axis_end
                )
            )

    # AtlasFold consistently uses alanine geometry for unknown residues.
    alanine_idx = restype_orders["A"]
    restype_atom14_to_rigid_group[20] = restype_atom14_to_rigid_group[alanine_idx]
    restype_atom14_rigid_group_positions[20] = restype_atom14_rigid_group_positions[
        alanine_idx
    ]
    restype_rigid_group_default_frame[20] = restype_rigid_group_default_frame[alanine_idx]


_make_rigid_group_constants()
