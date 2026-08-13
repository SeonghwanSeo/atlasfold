"""Preprocess RCSB mmCIF files into multimer complexes."""

import argparse
import collections
import dataclasses
import functools
import hashlib
import json
import logging
import multiprocessing
import os
import pathlib
from datetime import datetime
from typing import Any

import gemmi
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from tqdm import tqdm

from atlasfold.common import metadata, protein
from atlasfold.data import cif_factory
from atlasfold.train.multimer.dataset import MultimerDataPipeline

# Error handling
SUCCESS = 0
FAILED = 1
DATE_FILTERED = 2
RESOLUTION_FILTERED = 3
NO_PROTEIN_FILTERED = 4
CHAIN_COUNT_FILTERED = 5
NO_VALID_CHAINS_FILTERED = 6
TOKEN_COUNT_FILTERED = 7
SEQUENCE_COMPOSITION_FILTERED = 8

MIN_LENGTH: int = 4
MAX_EXTRACTED_CHAINS: int = 20
INTERFACE_CA_DISTANCE: float = 15.0
CONTACT_DISTANCE: float = 5.0
CLASH_DISTANCE: float = 1.7
MAX_CLASH_RATIO: float = 0.3

logger = logging.getLogger(__name__)

GLOBAL_CLUSTER_MAPPING: dict[str, str] = {}


@dataclasses.dataclass(frozen=True)
class DataFilter:
    """RCSB multimer split filters."""

    output_name: str
    date_start: datetime
    date_end: datetime
    max_resolution: float
    min_chains: int
    max_input_chains: int
    max_output_chains: int = MAX_EXTRACTED_CHAINS
    max_residues: int | None = None
    require_cluster_mapping: bool = True


AF3_SPLITS: dict[str, DataFilter] = {
    "train": DataFilter(
        output_name="rcsb_multimer",
        date_start=datetime.min,
        date_end=datetime.fromisoformat("2021-09-30 23:59:59"),
        max_resolution=9.0,
        min_chains=1,
        max_input_chains=300,
        require_cluster_mapping=True,
    ),
    "val": DataFilter(
        output_name="rcsb_multimer_val",
        date_start=datetime.fromisoformat("2021-10-01 00:00:00"),
        date_end=datetime.fromisoformat("2023-01-12 23:59:59"),
        max_resolution=4.5,
        min_chains=2,
        max_input_chains=1000,
        max_output_chains=20,
        max_residues=1536,
        require_cluster_mapping=False,
    ),
}


def get_dataset_name(data_filter: DataFilter, all_assemblies: bool) -> str:
    """Return the output directory name for one structure-processing mode."""
    if not all_assemblies:
        return data_filter.output_name
    return f"{data_filter.output_name}_assembly"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process RCSB mmCIF files.")
    parser.add_argument(
        "--cif_dir",
        type=pathlib.Path,
        required=True,
        help="Path to the `mmCIF/` directory from RCSB.",
    )
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to output directory for processed .npz files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=sorted(AF3_SPLITS),
        help="AF3 split to process. Validation outputs go to rcsb_multimer_val.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=len(os.sched_getaffinity(0)),
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--all_assemblies",
        action="store_true",
        help=(
            "Process every biological assembly as <pdb>-assemblyN instead of "
            "only biological assembly 1. Entries without an assembly definition "
            "use the asymmetric unit as assembly1."
        ),
    )
    args = parser.parse_args()
    return args


# =================================================
# Helper functions for data filtering
# ==================================================
def expand_first_assembly(raw_struct: gemmi.Structure) -> None:
    """Expand the first biological assembly in-place when assembly data exists."""
    if len(raw_struct.assemblies) == 0:
        return
    how = gemmi.HowToNameCopiedChain.AddNumber
    try:
        raw_struct.transform_to_assembly(raw_struct.assemblies[0].name, how=how)
    except Exception as e:
        logger.warning(f"Failed to expand first assembly: {e}")


def expand_assembly(raw_struct: gemmi.Structure, assembly_name: str | None) -> None:
    """Expand one named biological assembly in-place.

    ``assembly_name=None`` deliberately leaves the asymmetric unit unchanged;
    this is the fallback for entries without a biological assembly definition.
    """
    if assembly_name is None:
        return
    how = gemmi.HowToNameCopiedChain.AddNumber
    raw_struct.transform_to_assembly(assembly_name, how=how)


def keep_first_model(raw_struct: gemmi.Structure) -> None:
    """Keep the first coordinate model used by the downstream parser.

    Some NMR entries have subchains that are absent from later models. Gemmi's
    assembly expansion otherwise fails on those later models even though
    ``get_protein_chains`` consumes only model 1.
    """
    while len(raw_struct) > 1:
        del raw_struct[1]


def validate_chain_length(prot: protein.Protein) -> bool:
    """Return True if a protein chain has the minimum sequence length."""
    if len(prot.sequence) < MIN_LENGTH:
        logger.debug(f"{prot.name} marked invalid due to short sequence.")
        return False
    return True


def validate_complex_sequence(chains: list[protein.Protein]) -> bool:
    """Apply the AlphaFold-Multimer 80% composition filter to a whole complex."""
    sequence = "".join(chain.sequence for chain in chains)
    if not sequence:
        return False
    aa_counts = collections.Counter(sequence)
    return max(aa_counts.values()) / len(sequence) <= 0.8


def validate_geometry(prot: protein.Protein) -> bool:
    """Return True if a protein chain has enough resolved CA atoms and no big jumps."""
    ca_coords = prot.coordinates[:, 1, :]
    is_resolved = np.isfinite(ca_coords).all(axis=-1)
    n_resolved = int(np.sum(is_resolved))
    if n_resolved < MIN_LENGTH:
        logger.debug(f"{prot.name} marked invalid due to insufficient resolved CA atoms.")
        return False

    adjacent_resolved = is_resolved[:-1] & is_resolved[1:]
    dists = np.linalg.norm(ca_coords[:-1] - ca_coords[1:], axis=-1)
    if np.any(dists[adjacent_resolved] > 10.0):
        logger.debug(f"{prot.name} marked invalid due to CA trace discontinuity.")
        return False
    return True


def has_protein(raw_struct: gemmi.Structure) -> bool:
    """Return True if the structure has at least one protein entity."""
    for entity in raw_struct.entities:
        if (
            entity.entity_type == gemmi.EntityType.Polymer
            and entity.polymer_type == gemmi.PolymerType.PeptideL
            and len(entity.full_sequence) >= MIN_LENGTH
        ):
            return True
    return False


def detect_interfaces(
    chains: list[protein.Protein],
    chain_metadatas: list[metadata.Metadata],
) -> list[metadata.InterfaceMetadata]:
    """Detect protein-protein interfaces using CA-CA distances."""
    interfaces: list[metadata.InterfaceMetadata] = []
    ca_coords = [c.coordinates[:, 1, :] for c in chains]
    ca_masks = [np.isfinite(coords).all(axis=-1) for coords in ca_coords]

    for i in range(len(chains)):
        if not np.any(ca_masks[i]):
            continue
        for j in range(i + 1, len(chains)):
            if not np.any(ca_masks[j]):
                continue
            d = cdist(ca_coords[i][ca_masks[i]], ca_coords[j][ca_masks[j]])
            if not np.any(d < INTERFACE_CA_DISTANCE):
                continue

            ci = chain_metadatas[i].cluster_id
            cj = chain_metadatas[j].cluster_id
            if ci is None or cj is None:
                cluster_id = None
            else:
                cluster_id = "__".join(sorted((ci, cj)))

            interfaces.append(
                metadata.InterfaceMetadata(
                    chain_ids=(i, j),
                    cluster_id=cluster_id,
                )
            )
    return interfaces


def get_resolved_atom_coordinates(chain: protein.Protein) -> np.ndarray:
    """Return finite atom coordinates for a chain."""
    coords = chain.coordinates.reshape(-1, 3)
    return coords[np.isfinite(coords).all(axis=-1)]


@dataclasses.dataclass
class ClashStatistics:
    """Audit counters collected while applying the all-atom clash filter."""

    contact_chain_pairs: int = 0
    atom_clash_chain_pairs: int = 0
    severe_clash_chain_pairs: int = 0
    chains_removed: int = 0
    max_chain_clash_ratio: float = 0.0


def filter_clashing_chains_with_stats(
    chains: list[protein.Protein],
    chain_metadatas: list[metadata.Metadata],
) -> tuple[list[protein.Protein], list[metadata.Metadata], ClashStatistics]:
    """Remove chains with severe all-atom clashes.

    This mirrors the AF3/KFold criterion: chain pairs in contact are checked for
    atoms closer than 1.7A, and chains with more than 30% clashing atoms are
    removed.
    """
    stats = ClashStatistics()
    invalid_indices: set[int] = set()
    atom_coords = [get_resolved_atom_coordinates(c) for c in chains]
    boxes: list[tuple[np.ndarray, np.ndarray] | None] = []
    trees: list[cKDTree | None] = []
    for coords in atom_coords:
        if len(coords) == 0:
            boxes.append(None)
            trees.append(None)
        else:
            boxes.append((coords.min(axis=0), coords.max(axis=0)))
            trees.append(cKDTree(coords))

    for i in range(len(chains)):
        if i in invalid_indices or trees[i] is None or boxes[i] is None:
            continue
        for j in range(i + 1, len(chains)):
            if j in invalid_indices or trees[j] is None or boxes[j] is None:
                continue

            min_i, max_i = boxes[i]
            min_j, max_j = boxes[j]
            if np.any(min_i - max_j > CONTACT_DISTANCE) or np.any(
                min_j - max_i > CONTACT_DISTANCE
            ):
                continue

            tree_i = trees[i]
            tree_j = trees[j]
            assert tree_i is not None and tree_j is not None
            if tree_i.count_neighbors(tree_j, r=CONTACT_DISTANCE) == 0:
                continue
            stats.contact_chain_pairs += 1
            if tree_i.count_neighbors(tree_j, r=CLASH_DISTANCE) == 0:
                continue
            stats.atom_clash_chain_pairs += 1

            clash_indices_i = tree_i.query_ball_tree(tree_j, r=CLASH_DISTANCE)
            clash_indices_j = tree_j.query_ball_tree(tree_i, r=CLASH_DISTANCE)
            n_clash_atoms_i = sum(1 for neighbors in clash_indices_i if neighbors)
            n_clash_atoms_j = sum(1 for neighbors in clash_indices_j if neighbors)
            n_atoms_i = len(atom_coords[i])
            n_atoms_j = len(atom_coords[j])
            clash_ratio_i = n_clash_atoms_i / n_atoms_i
            clash_ratio_j = n_clash_atoms_j / n_atoms_j
            stats.max_chain_clash_ratio = max(
                stats.max_chain_clash_ratio,
                clash_ratio_i,
                clash_ratio_j,
            )
            is_clash_i = clash_ratio_i > MAX_CLASH_RATIO
            is_clash_j = clash_ratio_j > MAX_CLASH_RATIO
            if is_clash_i or is_clash_j:
                stats.severe_clash_chain_pairs += 1

            if is_clash_i and is_clash_j:
                if clash_ratio_i > clash_ratio_j:
                    remove_i = i
                elif clash_ratio_i < clash_ratio_j:
                    remove_i = j
                elif n_atoms_i > n_atoms_j:
                    remove_i = j
                elif n_atoms_i < n_atoms_j:
                    remove_i = i
                else:
                    asym_i = chain_metadatas[i].asym_id or i
                    asym_j = chain_metadatas[j].asym_id or j
                    remove_i = i if asym_i > asym_j else j
                invalid_indices.add(remove_i)
                logger.debug(
                    f"{chain_metadatas[remove_i].id} marked invalid due to severe "
                    f"clash between chains {chain_metadatas[i].asym_id} and "
                    f"{chain_metadatas[j].asym_id}."
                )
                if remove_i == i:
                    break
            elif is_clash_i:
                invalid_indices.add(i)
                logger.debug(
                    f"{chain_metadatas[i].id} marked invalid due to clash "
                    f"({clash_ratio_i:.2%} atoms clashed with chain "
                    f"{chain_metadatas[j].asym_id})."
                )
                break
            elif is_clash_j:
                invalid_indices.add(j)
                logger.debug(
                    f"{chain_metadatas[j].id} marked invalid due to clash "
                    f"({clash_ratio_j:.2%} atoms clashed with chain "
                    f"{chain_metadatas[i].asym_id})."
                )

    filtered_chains = [c for i, c in enumerate(chains) if i not in invalid_indices]
    filtered_metadatas = [
        m for i, m in enumerate(chain_metadatas) if i not in invalid_indices
    ]
    stats.chains_removed = len(invalid_indices)
    return filtered_chains, filtered_metadatas, stats


def filter_clashing_chains(
    chains: list[protein.Protein],
    chain_metadatas: list[metadata.Metadata],
) -> tuple[list[protein.Protein], list[metadata.Metadata]]:
    """Remove severe clashes while preserving the historical two-value API."""
    filtered_chains, filtered_metadatas, _ = filter_clashing_chains_with_stats(
        chains,
        chain_metadatas,
    )
    return filtered_chains, filtered_metadatas


def reindex_chain_metadata(chain_metadatas: list[metadata.Metadata]) -> None:
    """Reassign contiguous asym/sym ids after filtering or subcomplex extraction."""
    sym_id_by_entity: dict[int, int] = {}
    for asym_id, m in enumerate(chain_metadatas, 1):
        m.asym_id = asym_id
        assert m.entity_id is not None
        sym_id_by_entity[m.entity_id] = sym_id_by_entity.get(m.entity_id, 0) + 1
        m.sym_id = sym_id_by_entity[m.entity_id]


def get_resolved_ca_coordinates(chain: protein.Protein) -> np.ndarray:
    """Return finite CA coordinates for a chain."""
    ca_coords = chain.coordinates[:, 1, :]
    return ca_coords[np.isfinite(ca_coords).all(axis=-1)]


def select_closest_chains(
    name: str,
    chains: list[protein.Protein],
    chain_metadatas: list[metadata.Metadata],
    interfaces: list[metadata.InterfaceMetadata],
    max_chains: int = MAX_EXTRACTED_CHAINS,
) -> tuple[list[protein.Protein], list[metadata.Metadata]]:
    """Select up to max_chains around a sampled interface/contact seed.

    This follows the AF3 SI 2.5.4 subcomplex extraction idea: for assemblies with
    many chains, sample an interface token and keep the chains closest to it.
    """
    if len(chains) <= max_chains:
        return chains, chain_metadatas

    seed = int.from_bytes(hashlib.sha1(name.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)

    ca_coords = [get_resolved_ca_coordinates(c) for c in chains]
    valid_chain_indices = [i for i, coords in enumerate(ca_coords) if len(coords) > 0]
    if len(valid_chain_indices) <= max_chains:
        selected_indices = set(valid_chain_indices)
    else:
        seed_coords = None
        if interfaces:
            iface = interfaces[rng.integers(len(interfaces))]
            chain_i, chain_j = iface.chain_ids
            coords_i = ca_coords[chain_i]
            coords_j = ca_coords[chain_j]
            if len(coords_i) > 0 and len(coords_j) > 0:
                d = cdist(coords_i, coords_j)
                is_contact = d < INTERFACE_CA_DISTANCE
                if np.any(is_contact):
                    if rng.random() < 0.5:
                        contact_indices = np.where(np.any(is_contact, axis=1))[0]
                        seed_coords = coords_i[rng.choice(contact_indices)]
                    else:
                        contact_indices = np.where(np.any(is_contact, axis=0))[0]
                        seed_coords = coords_j[rng.choice(contact_indices)]
                elif rng.random() < 0.5:
                    seed_coords = coords_i[rng.integers(len(coords_i))]
                else:
                    seed_coords = coords_j[rng.integers(len(coords_j))]

        if seed_coords is None:
            chain_i = valid_chain_indices[rng.integers(len(valid_chain_indices))]
            coords_i = ca_coords[chain_i]
            seed_coords = coords_i[rng.integers(len(coords_i))]

        chain_dists = []
        for i in valid_chain_indices:
            d = np.linalg.norm(ca_coords[i] - seed_coords[None, :], axis=-1)
            chain_dists.append((i, float(np.min(d))))
        chain_dists.sort(key=lambda x: x[1])
        selected_indices = {i for i, _ in chain_dists[:max_chains]}

    selected_chains = [c for i, c in enumerate(chains) if i in selected_indices]
    selected_metadatas = [
        m for i, m in enumerate(chain_metadatas) if i in selected_indices
    ]
    return selected_chains, selected_metadatas


STATUS_NAMES = {
    SUCCESS: "success",
    FAILED: "failed",
    DATE_FILTERED: "date_filtered",
    RESOLUTION_FILTERED: "resolution_filtered",
    NO_PROTEIN_FILTERED: "no_protein_filtered",
    CHAIN_COUNT_FILTERED: "chain_count_filtered",
    NO_VALID_CHAINS_FILTERED: "no_valid_chains_filtered",
    TOKEN_COUNT_FILTERED: "token_count_filtered",
    SEQUENCE_COMPOSITION_FILTERED: "sequence_composition_filtered",
}


def _finish_audit(audit: dict[str, Any], flag: int) -> dict[str, Any]:
    audit["flag"] = flag
    audit["status"] = STATUS_NAMES[flag]
    return audit


def parse_cif_block(
    name: str,
    base_pdb_id: str,
    block: gemmi.cif.Block,
    out_dir: pathlib.Path,
    data_filter: DataFilter,
    *,
    assembly_index: int,
    assembly_name: str | None,
    num_assemblies: int,
) -> tuple[int, int, int, dict[str, Any]]:
    """Process one selected assembly from a parsed mmCIF block."""
    global GLOBAL_CLUSTER_MAPPING
    if data_filter.require_cluster_mapping:
        assert GLOBAL_CLUSTER_MAPPING, "Cluster mapping is not initialized in worker."

    audit: dict[str, Any] = {
        "id": name,
        "pdb_id": base_pdb_id,
        "assembly_id": f"assembly{assembly_index}",
        "assembly_name": assembly_name,
        "num_assemblies": num_assemblies,
        "assembly_sampling_weight": 1.0 / num_assemblies,
    }
    npz_path = out_dir / f"{name}.npz"
    json_path = out_dir / f"{name}.json"
    if npz_path.exists() and json_path.exists():
        audit["existing_output"] = True
        return SUCCESS, 1, 0, _finish_audit(audit, SUCCESS)

    exp_record = cif_factory.read_experiment_record(block)
    release_date = datetime.fromisoformat(exp_record.release_date)
    if not data_filter.date_start <= release_date <= data_filter.date_end:
        return DATE_FILTERED, 0, 0, _finish_audit(audit, DATE_FILTERED)

    resolution = exp_record.resolution
    if resolution is not None and resolution > data_filter.max_resolution:
        return RESOLUTION_FILTERED, 0, 0, _finish_audit(audit, RESOLUTION_FILTERED)

    raw_struct: gemmi.Structure = gemmi.make_structure_from_block(block)
    keep_first_model(raw_struct)
    expand_assembly(raw_struct, assembly_name)
    cif_factory.clean_up_gemmi_structure(raw_struct)

    if not has_protein(raw_struct):
        return NO_PROTEIN_FILTERED, 0, 0, _finish_audit(audit, NO_PROTEIN_FILTERED)

    parsed_chains: list[tuple[protein.Protein, metadata.Metadata]] = []
    sym_id_by_entity: dict[int, int] = {}
    missing_cluster_chains = 0
    for asym_id, (chain, ids) in enumerate(
        cif_factory.get_protein_chains(raw_struct),
        1,
    ):
        # Assembly expansion can append digits to copied chain/subchain IDs. Keep
        # the full IDs so copies remain unique within the complex.
        label_id = str(ids["label_id"])
        auth_id = str(ids["auth_id"])
        entity_id = ids["entity_id"]
        chain.name = f"{name}_{label_id}"

        cluster_id = None
        if data_filter.require_cluster_mapping:
            key = f"{base_pdb_id}_{entity_id}".upper()
            cluster_id = GLOBAL_CLUSTER_MAPPING.get(key)
            if cluster_id is None:
                missing_cluster_chains += 1
                logger.warning(
                    f"Cluster ID not found for entity_id {entity_id} in {base_pdb_id}."
                )
                continue

        sym_id_by_entity[entity_id] = sym_id_by_entity.get(entity_id, 0) + 1
        chain_metadata = metadata.Metadata(
            id=f"{name}_{label_id}",
            label_asym_id=label_id,
            auth_asym_id=auth_id,
            entity_id=entity_id,
            asym_id=asym_id,
            sym_id=sym_id_by_entity[entity_id],
            num_residues=len(chain),
            cluster_id=cluster_id,
        )
        parsed_chains.append((chain, chain_metadata))

    audit["parsed_chains"] = len(parsed_chains)
    audit["missing_cluster_chains"] = missing_cluster_chains
    if not (data_filter.min_chains <= len(parsed_chains) <= data_filter.max_input_chains):
        return CHAIN_COUNT_FILTERED, 0, 0, _finish_audit(audit, CHAIN_COUNT_FILTERED)

    if not validate_complex_sequence([chain for chain, _ in parsed_chains]):
        return (
            SEQUENCE_COMPOSITION_FILTERED,
            0,
            0,
            _finish_audit(
                audit,
                SEQUENCE_COMPOSITION_FILTERED,
            ),
        )

    valid_chains: list[tuple[protein.Protein, metadata.Metadata]] = []
    short_sequence_chains = 0
    geometry_invalid_chains = 0
    for chain, chain_metadata in parsed_chains:
        if not validate_chain_length(chain):
            short_sequence_chains += 1
        elif not validate_geometry(chain):
            geometry_invalid_chains += 1
        else:
            valid_chains.append((chain, chain_metadata))
    audit["short_sequence_chains"] = short_sequence_chains
    audit["geometry_invalid_chains"] = geometry_invalid_chains
    audit["chains_before_clash"] = len(valid_chains)
    if not valid_chains:
        return (
            NO_VALID_CHAINS_FILTERED,
            0,
            0,
            _finish_audit(
                audit,
                NO_VALID_CHAINS_FILTERED,
            ),
        )
    if not (data_filter.min_chains <= len(valid_chains) <= data_filter.max_input_chains):
        return CHAIN_COUNT_FILTERED, 0, 0, _finish_audit(audit, CHAIN_COUNT_FILTERED)

    chains = [chain for chain, _ in valid_chains]
    chain_metadatas = [m for _, m in valid_chains]
    reindex_chain_metadata(chain_metadatas)
    chains, chain_metadatas, clash_stats = filter_clashing_chains_with_stats(
        chains,
        chain_metadatas,
    )
    audit.update(dataclasses.asdict(clash_stats))
    audit["chains_after_clash"] = len(chains)
    if not chains:
        return (
            NO_VALID_CHAINS_FILTERED,
            0,
            0,
            _finish_audit(
                audit,
                NO_VALID_CHAINS_FILTERED,
            ),
        )
    if not (data_filter.min_chains <= len(chains) <= data_filter.max_input_chains):
        return CHAIN_COUNT_FILTERED, 0, 0, _finish_audit(audit, CHAIN_COUNT_FILTERED)

    reindex_chain_metadata(chain_metadatas)
    interfaces = detect_interfaces(chains, chain_metadatas)
    audit["interfaces_before_subcomplex"] = len(interfaces)
    chains_before_subcomplex = len(chains)
    chains, chain_metadatas = select_closest_chains(
        name,
        chains,
        chain_metadatas,
        interfaces,
        max_chains=data_filter.max_output_chains,
    )
    audit["chains_removed_by_subcomplex"] = chains_before_subcomplex - len(chains)
    reindex_chain_metadata(chain_metadatas)
    interfaces = detect_interfaces(chains, chain_metadatas)
    audit["output_chains"] = len(chains)
    audit["output_interfaces"] = len(interfaces)
    audit["output_residues"] = sum(len(chain) for chain in chains)
    if not (data_filter.min_chains <= len(chains) <= data_filter.max_output_chains):
        return CHAIN_COUNT_FILTERED, 0, 0, _finish_audit(audit, CHAIN_COUNT_FILTERED)
    if (
        data_filter.max_residues is not None
        and audit["output_residues"] > data_filter.max_residues
    ):
        return TOKEN_COUNT_FILTERED, 0, 0, _finish_audit(audit, TOKEN_COUNT_FILTERED)

    compl = protein.ProteinMultimer(
        name=name,
        chains=chains,
        entity_ids=[m.entity_id for m in chain_metadatas if m.entity_id is not None],
        asym_ids=[m.asym_id for m in chain_metadatas if m.asym_id is not None],
        sym_ids=[m.sym_id for m in chain_metadatas if m.sym_id is not None],
    )
    complex_metadata = metadata.MultimerMetadata(
        id=name,
        chains=chain_metadatas,
        interfaces=interfaces,
        exp=exp_record,
    ).to_dict()
    complex_metadata.update(
        {
            "pdb_id": base_pdb_id,
            "assembly_id": f"assembly{assembly_index}",
            "assembly_name": assembly_name,
            "num_assemblies": num_assemblies,
            "assembly_sampling_weight": 1.0 / num_assemblies,
        }
    )

    MultimerDataPipeline.save(compl, npz_path)
    with open(json_path, "w") as f:
        json.dump(complex_metadata, f)

    return SUCCESS, 1, len(chains), _finish_audit(audit, SUCCESS)


def parse_cif(
    name: str,
    cif_path: pathlib.Path,
    out_dir: pathlib.Path,
    data_filter: DataFilter,
) -> tuple[int, int, int]:
    """Parse one CIF file and save its first biological assembly."""
    block: gemmi.cif.Block = gemmi.cif.read(str(cif_path))[0]
    structure = gemmi.make_structure_from_block(block)
    assembly_name = structure.assemblies[0].name if structure.assemblies else None
    result = parse_cif_block(
        name,
        name,
        block,
        out_dir,
        data_filter,
        assembly_index=1,
        assembly_name=assembly_name,
        num_assemblies=max(len(structure.assemblies), 1),
    )
    return result[:3]


def worker_fn(
    cif_path: pathlib.Path,
    output_dir: pathlib.Path,
    data_filter: DataFilter,
):
    """Worker function to process a single CIF file."""
    pdb_id = cif_path.name.split(".")[0].lower()
    out_dir = output_dir / pdb_id[1:3] / pdb_id
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        return parse_cif(pdb_id, cif_path, out_dir, data_filter)
    except Exception as e:
        logger.error(f"Failed to process ({pdb_id}): {e}")
        return FAILED, 0, 0


def all_assemblies_worker_fn(
    cif_path: pathlib.Path,
    output_dir: pathlib.Path,
    data_filter: DataFilter,
) -> list[tuple[int, int, int, dict[str, Any]]]:
    """Process every biological assembly from one CIF file."""
    pdb_id = cif_path.name.split(".")[0].lower()
    out_dir = output_dir / pdb_id[1:3] / pdb_id
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        block: gemmi.cif.Block = gemmi.cif.read(str(cif_path))[0]
        structure = gemmi.make_structure_from_block(block)
        assembly_names: list[str | None] = [a.name for a in structure.assemblies]
        if not assembly_names:
            assembly_names = [None]

        results = []
        for assembly_index, assembly_name in enumerate(assembly_names, 1):
            name = f"{pdb_id}-assembly{assembly_index}"
            try:
                result = parse_cif_block(
                    name,
                    pdb_id,
                    block,
                    out_dir,
                    data_filter,
                    assembly_index=assembly_index,
                    assembly_name=assembly_name,
                    num_assemblies=len(assembly_names),
                )
            except Exception as e:
                logger.error(f"Failed to process ({name}): {e}")
                audit = _finish_audit(
                    {
                        "id": name,
                        "pdb_id": pdb_id,
                        "assembly_id": f"assembly{assembly_index}",
                        "assembly_name": assembly_name,
                        "num_assemblies": len(assembly_names),
                        "assembly_sampling_weight": 1.0 / len(assembly_names),
                        "error": repr(e),
                    },
                    FAILED,
                )
                result = FAILED, 0, 0, audit
            results.append(result)
        return results
    except Exception as e:
        logger.error(f"Failed to read ({pdb_id}): {e}")
        audit = _finish_audit(
            {
                "id": f"{pdb_id}-assembly1",
                "pdb_id": pdb_id,
                "assembly_id": "assembly1",
                "assembly_name": None,
                "num_assemblies": 1,
                "assembly_sampling_weight": 1.0,
                "error": repr(e),
            },
            FAILED,
        )
        return [(FAILED, 0, 0, audit)]


def worker_init(cluster_path: pathlib.Path | None):
    """Initialize worker process with global entity-to-cluster mapping."""
    global GLOBAL_CLUSTER_MAPPING
    if cluster_path is None:
        GLOBAL_CLUSTER_MAPPING = {}
        return

    cluster_mapping = {}
    with open(cluster_path) as f:
        next(f)  # skip header
        for line in f:
            entity_id, cluster_id, _ = line.strip().split(",", 2)
            cluster_mapping[entity_id.upper()] = cluster_id.upper()
    GLOBAL_CLUSTER_MAPPING = cluster_mapping


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def process_all_assemblies(
    cif_paths: list[pathlib.Path],
    out_dir: pathlib.Path,
    data_filter: DataFilter,
    cluster_mapping_path: pathlib.Path | None,
    num_workers: int,
    data_dir: pathlib.Path,
) -> dict[str, Any]:
    """Run all-assembly processing and write detailed audit statistics."""
    worker_wrapped = functools.partial(
        all_assemblies_worker_fn,
        output_dir=out_dir,
        data_filter=data_filter,
    )
    status_counts: collections.Counter[str] = collections.Counter()
    totals: collections.Counter[str] = collections.Counter()
    audit_path = data_dir / "assembly_processing_audit.jsonl"

    with (
        open(audit_path, "w") as audit_file,
        multiprocessing.Pool(
            num_workers,
            initializer=worker_init,
            initargs=(cluster_mapping_path,),
        ) as pool,
    ):
        pbar = tqdm(
            pool.imap_unordered(worker_wrapped, cif_paths, chunksize=1),
            total=len(cif_paths),
            desc="Processing all RCSB biological assemblies",
        )
        for results in pbar:
            totals["source_entries"] += 1
            if results and results[0][3]["num_assemblies"] > 1:
                totals["source_entries_with_multiple_assemblies"] += 1
            for flag, n_success, n_success_chains, audit in results:
                status_counts[STATUS_NAMES[flag]] += 1
                totals["assemblies_considered"] += 1
                totals["successful_assemblies"] += n_success
                totals["successful_output_chains"] += n_success_chains
                for key in (
                    "parsed_chains",
                    "missing_cluster_chains",
                    "short_sequence_chains",
                    "geometry_invalid_chains",
                    "chains_before_clash",
                    "chains_removed",
                    "chains_removed_by_subcomplex",
                    "output_chains",
                    "output_interfaces",
                    "output_residues",
                    "contact_chain_pairs",
                    "atom_clash_chain_pairs",
                    "severe_clash_chain_pairs",
                ):
                    totals[key] += int(audit.get(key, 0))

                if "chains_before_clash" in audit:
                    totals["assemblies_evaluated_for_clash"] += 1
                if audit.get("atom_clash_chain_pairs", 0) > 0:
                    totals["assemblies_with_any_atom_clash"] += 1
                if audit.get("severe_clash_chain_pairs", 0) > 0:
                    totals["assemblies_with_severe_clash"] += 1
                if audit.get("chains_removed", 0) > 0:
                    totals["assemblies_with_chain_removed_by_clash"] += 1
                if audit.get("chains_removed_by_subcomplex", 0) > 0:
                    totals["assemblies_with_subcomplex_extraction"] += 1
                json.dump(audit, audit_file, separators=(",", ":"))
                audit_file.write("\n")
            pbar.set_postfix(
                {
                    "Assemblies": totals["assemblies_considered"],
                    "Success": totals["successful_assemblies"],
                },
                refresh=False,
            )

    clash_evaluated = totals["assemblies_evaluated_for_clash"]
    parsed_chains = totals["parsed_chains"]
    chains_before_clash = totals["chains_before_clash"]
    summary: dict[str, Any] = {
        "parameters": {
            "split": data_filter.output_name,
            "all_assemblies": True,
            "num_workers": num_workers,
            "max_clash_ratio": MAX_CLASH_RATIO,
            "clash_distance_angstrom": CLASH_DISTANCE,
            "contact_distance_angstrom": CONTACT_DISTANCE,
        },
        "status_counts": dict(sorted(status_counts.items())),
        "counts": dict(sorted(totals.items())),
        "rates": {
            "source_entries_with_multiple_assemblies": _safe_ratio(
                totals["source_entries_with_multiple_assemblies"],
                totals["source_entries"],
            ),
            "successful_assemblies": _safe_ratio(
                totals["successful_assemblies"],
                totals["assemblies_considered"],
            ),
            "assemblies_with_any_atom_clash": _safe_ratio(
                totals["assemblies_with_any_atom_clash"],
                clash_evaluated,
            ),
            "assemblies_with_severe_clash": _safe_ratio(
                totals["assemblies_with_severe_clash"],
                clash_evaluated,
            ),
            "assemblies_with_chain_removed_by_clash": _safe_ratio(
                totals["assemblies_with_chain_removed_by_clash"],
                clash_evaluated,
            ),
            "chains_removed_by_clash": _safe_ratio(
                totals["chains_removed"],
                chains_before_clash,
            ),
            "short_sequence_chains": _safe_ratio(
                totals["short_sequence_chains"],
                parsed_chains,
            ),
            "geometry_invalid_chains": _safe_ratio(
                totals["geometry_invalid_chains"],
                parsed_chains,
            ),
            "assemblies_with_subcomplex_extraction": _safe_ratio(
                totals["assemblies_with_subcomplex_extraction"],
                clash_evaluated,
            ),
        },
        "audit_path": str(audit_path),
    }
    summary_path = data_dir / "assembly_processing_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved per-assembly audit to {audit_path}")
    print(f"Saved aggregate statistics to {summary_path}")
    return summary


def main():
    """Main function to process RCSB mmCIF files."""
    args = parse_args()
    cif_dir: pathlib.Path = args.cif_dir
    data_filter = AF3_SPLITS[args.split]
    dataset_name = get_dataset_name(data_filter, args.all_assemblies)
    data_dir: pathlib.Path = args.data_dir / dataset_name

    cluster_mapping_path: pathlib.Path | None = None
    if data_filter.require_cluster_mapping:
        cluster_mapping_path = data_dir / "rcsb_clusters.csv"
        if args.all_assemblies and not cluster_mapping_path.exists():
            # Entity sequences/clusters do not depend on the chosen biological
            # assembly, so the regular RCSB dataset can provide this shared input.
            cluster_mapping_path = (
                args.data_dir / data_filter.output_name / "rcsb_clusters.csv"
            )
        assert cluster_mapping_path.exists(), (
            f"Cluster mapping file not found: {cluster_mapping_path}"
        )

    print(f"Applying AF3 {args.split} split parameters:")
    print(data_filter)
    print(f"Output dataset: {dataset_name}")

    out_dir: pathlib.Path = data_dir / "npz"
    out_dir.mkdir(parents=True, exist_ok=True)

    worker_wrapped = functools.partial(
        worker_fn,
        output_dir=out_dir,
        data_filter=data_filter,
    )

    cif_paths = sorted(cif_dir.rglob("*.cif.gz"))
    print(f"Found {len(cif_paths)} mmCIF files to process.")

    if args.all_assemblies:
        process_all_assemblies(
            cif_paths,
            out_dir,
            data_filter,
            cluster_mapping_path,
            args.num_workers,
            data_dir,
        )
        return

    flags: list[int] = []
    n_complexes = 0
    n_chains = 0
    with multiprocessing.Pool(
        args.num_workers,
        initializer=worker_init,
        initargs=(cluster_mapping_path,),
    ) as pool:
        pbar = tqdm(
            pool.imap_unordered(worker_wrapped, cif_paths, chunksize=10),
            total=len(cif_paths),
            desc="Processing RCSB mmCIF files",
        )
        for flag, n_success, n_success_chains in pbar:
            n_complexes += n_success
            n_chains += n_success_chains
            flags.append(flag)
            pbar.set_postfix({"Multimeres": n_complexes, "Chains": n_chains})
    print("Processing completed.")

    print("Processing statistics:")
    print(f"  Total files processed: {len(flags)}")
    print(f"  Successfully processed: {flags.count(SUCCESS)}")
    print(f"  Failed to process: {flags.count(FAILED)}")
    print(f"  Date filtered: {flags.count(DATE_FILTERED)}")
    print(f"  Resolution filtered: {flags.count(RESOLUTION_FILTERED)}")
    print(f"  No protein filtered: {flags.count(NO_PROTEIN_FILTERED)}")
    print(f"  Chain count filtered: {flags.count(CHAIN_COUNT_FILTERED)}")
    print(f"  No valid chains filtered: {flags.count(NO_VALID_CHAINS_FILTERED)}")
    print(f"  Token count filtered: {flags.count(TOKEN_COUNT_FILTERED)}")
    print(
        f"  Sequence composition filtered: {flags.count(SEQUENCE_COMPOSITION_FILTERED)}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
