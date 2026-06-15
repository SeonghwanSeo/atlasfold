import copy
import functools

import gemmi
import numpy as np

from atlasfold.common import residue_constants


@functools.lru_cache(21)
def get_residue_template(restype: str) -> gemmi.Residue:
    residue = gemmi.Residue()
    residue.name = restype
    residue.entity_type = gemmi.EntityType.Polymer
    atom_names: tuple[str, ...] = residue_constants.residue_atoms[restype]
    for an in atom_names:
        atom = gemmi.Atom()
        atom.name = an
        atom.element = gemmi.Element(an[0])
        residue.add_atom(atom)
    return residue


def to_gemmi_structure(
    name: str,
    sequence: str,
    coordinates: np.ndarray,
    b_factors: np.ndarray | None = None,
) -> gemmi.Structure:
    length = len(sequence)
    if coordinates.shape != (length, 14, 3):
        raise ValueError(
            f"Invalid coordinates shape: {coordinates.shape}. "
            f"Expected (L, 14, 3) where L is the sequence length."
        )
    if b_factors is not None and b_factors.shape not in [(length, 14), (length,)]:
        raise ValueError(
            f"Invalid b_factors shape: {b_factors.shape}. "
            f"Expected (L, 14) or (L,) where L is the sequence length."
        )

    if b_factors is not None and b_factors.shape == (length,):
        # Broadcast the per-residue B-factors to all atoms in the residue
        b_factors = np.broadcast_to(b_factors[:, np.newaxis], (length, 14))

    full_sequence = [residue_constants.restype_1to3[aa] for aa in sequence]

    # Create a new structure
    struct = gemmi.Structure()
    struct.name = name

    entity = gemmi.Entity("1")
    entity.entity_type = gemmi.EntityType.Polymer
    entity.polymer_type = gemmi.PolymerType.PeptideL
    entity.full_sequence = full_sequence
    entity.subchains = ["A"]
    struct.entities = gemmi.EntityList([entity])

    # Create a model
    try:
        model = gemmi.Model(1)
    except Exception:
        model = gemmi.Model("1")  # Fallback for older Gemmi versions

    chain = gemmi.Chain("A")

    residues: list[gemmi.Residue] = [
        copy.deepcopy(get_residue_template(restype)) for restype in full_sequence
    ]
    for res_i, residue in enumerate(residues):
        # Set residue index
        res_idx = res_i + 1
        residue.entity_id = "1"
        residue.subchain = "A"
        residue.seqid.num = res_idx
        residue.label_seq = res_idx

        # Set atom coordinates and B-factors
        if b_factors is not None:
            biso = b_factors[res_i]  # Shape: (14,)
        for atom_i, atom in enumerate(residue):
            x, y, z = coordinates[res_i, atom_i]
            atom.pos = gemmi.Position(x, y, z)
            atom.b_iso = biso[atom_i] if b_factors is not None else 100.0
    chain.append_residues(residues)
    model.add_chain(chain)

    # Add model to structure
    struct.add_model(model)
    struct.setup_entities()
    # struct.assign_subchains()
    return struct


RAW_PDB_HEADER = """
HEADER
TITLE     ATLASFOLD MONOMER PREDICTION of %s
REMARK   1 REFERENCE 1
REMARK   1  AUTH   SEONGHWAN SEO, HYEONGWOO KIM, WOO YOUN KIM
REMARK   1  TITL   TO BE PUBLISHED
REMARK   1  REF    TO BE PUBLISHED
"""


def to_pdb(
    name: str, sequence: str, coordinates: np.ndarray, b_factors: np.ndarray | None = None
) -> str:
    struct = to_gemmi_structure(name, sequence, coordinates, b_factors)
    header = RAW_PDB_HEADER % name
    struct.raw_remarks = header.strip().splitlines()
    return struct.make_pdb_string()


def to_mmcif(
    name: str, sequence: str, coordinates: np.ndarray, b_factors: np.ndarray | None = None
) -> str:
    cif_block = gemmi.cif.Block(name)
    author_loop = cif_block.init_loop(
        "_citation_author.", ["citation_id", "ordinal", "name"]
    )
    author_loop.add_row(["primary", "1", "Seo, Seonghwan"])
    author_loop.add_row(["primary", "2", "Kim, Hyeongwoo"])
    author_loop.add_row(["primary", "3", "Kim, Woo Youn"])

    software_loop = cif_block.init_loop(
        "_software.",
        ["pdbx_ordinal", "name", "type", "description", "version"],
    )
    software_loop.add_row(
        ["1", "AtlasFold", "model", '"Monomer Prediction Pipeline"', "0.1.0"]
    )

    struct = to_gemmi_structure(name, sequence, coordinates, b_factors)
    struct.update_mmcif_block(cif_block)

    # Round coordinates to 3 decimal places
    atom_site_table = cif_block.find("_atom_site.", ["Cartn_x", "Cartn_y", "Cartn_z"])
    for row in atom_site_table:
        row[0] = f"{float(row[0]):.3f}"
        row[1] = f"{float(row[1]):.3f}"
        row[2] = f"{float(row[2]):.3f}"

    # Round B-factors to 2 decimal places
    b_factor_table = cif_block.find("_atom_site.", ["B_iso_or_equiv"])
    if b_factor_table:
        for row in b_factor_table:
            row[0] = f"{float(row[0]):.2f}"

    return cif_block.as_string()
