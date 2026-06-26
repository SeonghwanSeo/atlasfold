import copy
import dataclasses
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


@dataclasses.dataclass
class ChainInfo:
    chain_id: str  # A B C
    entity_id: int  # 1 2 3
    sequence: str
    coordinates: np.ndarray  # [L, 14, 3]
    b_factors: np.ndarray | None  # [L, 14] or [L,] or None


def to_gemmi_chain(cinfo: ChainInfo) -> gemmi.Chain:
    chain_id = cinfo.chain_id
    entity_id = cinfo.entity_id
    sequence = cinfo.sequence
    coordinates = cinfo.coordinates
    b_factors = cinfo.b_factors
    length = len(cinfo.sequence)
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

    chain = gemmi.Chain(chain_id)

    full_sequence = [residue_constants.restype_1to3[aa] for aa in sequence]
    residues: list[gemmi.Residue] = [
        copy.deepcopy(get_residue_template(restype)) for restype in full_sequence
    ]
    for res_i, residue in enumerate(residues):
        # Set residue index
        res_idx = res_i + 1
        residue.entity_id = str(entity_id)
        residue.subchain = chain_id
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
    return chain


def to_gemmi_structure(name: str, chains: list[ChainInfo]) -> gemmi.Structure:
    """Convert a list of Chains into a gemmi.Structure."""
    struct = gemmi.Structure()
    struct.name = name

    # Set up entities
    entities: dict[int, gemmi.Entity] = {}
    for cinfo in chains:
        if cinfo.entity_id in entities:
            entities[cinfo.entity_id].subchains.append(cinfo.chain_id)
            continue
        full_sequence = [residue_constants.restype_1to3[aa] for aa in cinfo.sequence]
        entity = gemmi.Entity(str(cinfo.entity_id))
        entity.entity_type = gemmi.EntityType.Polymer
        entity.polymer_type = gemmi.PolymerType.PeptideL
        entity.full_sequence = full_sequence
        entities[cinfo.entity_id] = entity

    for cinfo in chains:
        entity = entities[cinfo.entity_id]
        entity.subchains.append(cinfo.chain_id)

    struct.entities = gemmi.EntityList([entities[eid] for eid in sorted(entities.keys())])

    # Create a model
    try:
        model = gemmi.Model(1)
    except Exception:
        model = gemmi.Model("1")  # Fallback for older Gemmi versions

    for cinfo in chains:
        chain = to_gemmi_chain(cinfo)
        model.add_chain(chain)

    # Add model to structure
    struct.add_model(model)
    struct.setup_entities()
    return struct


# ============================================================
# PDB/mmCIF output functions
# ============================================================


RAW_PDB_HEADER = """
HEADER
TITLE     ATLASFOLD MONOMER PREDICTION of %s
REMARK   1 REFERENCE 1
REMARK   1  AUTH   SEONGHWAN SEO, HYEONGWOO KIM, WOO YOUN KIM
REMARK   1  TITL   TO BE PUBLISHED
REMARK   1  REF    TO BE PUBLISHED
"""


def to_pdb(struct: gemmi.Structure) -> str:
    header = RAW_PDB_HEADER % struct.name
    struct.raw_remarks = header.strip().splitlines()
    return struct.make_pdb_string()


def to_mmcif(struct: gemmi.Structure) -> str:
    cif_block = gemmi.cif.Block(struct.name)

    # Add metadata
    cif_block.set_pair("_entry.id", struct.name)

    author_loop: gemmi.cif.Loop = cif_block.init_loop(
        "_citation_author.", ["citation_id", "ordinal", "name"]
    )
    author_loop.add_row(["primary", "1", '"Seo, Seonghwan"'])
    author_loop.add_row(["primary", "2", '"Kim, Hyeongwoo"'])
    author_loop.add_row(["primary", "3", '"Kim, Woo Youn"'])

    software_loop = cif_block.init_loop(
        "_software.",
        ["pdbx_ordinal", "name", "type", "description", "version"],
    )
    software_loop.add_row(
        ["1", "AtlasFold", "model", '"Monomer Prediction Pipeline"', "0.1.0"]
    )

    # Add structure data
    struct.update_mmcif_block(cif_block)

    # Round coordinates and B-factors
    atom_site_table = cif_block.find("_atom_site.", ["Cartn_x", "Cartn_y", "Cartn_z"])
    for row in atom_site_table:
        row[0] = f"{float(row[0]):.3f}"
        row[1] = f"{float(row[1]):.3f}"
        row[2] = f"{float(row[2]):.3f}"
    b_factor_table = cif_block.find("_atom_site.", ["B_iso_or_equiv"])
    if b_factor_table:
        for row in b_factor_table:
            row[0] = f"{float(row[0]):.2f}"

    return cif_block.as_string()
