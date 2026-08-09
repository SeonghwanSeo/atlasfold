"""Preprocess RCSB mmCIF files."""

import argparse
import functools
import json
import logging
import multiprocessing
import os
import pathlib

import gemmi
import numpy as np
from tqdm import tqdm

from atlasfold.common import metadata, protein
from atlasfold.data import cif_factory
from atlasfold.train.monomer.dataset import DataPipeline

# Error handling
SUCCESS = 0
FAILED = 1
DATE_FILTERED = 2
RESOLUTION_FILTERED = 3
NO_PROTEIN_FILTERED = 4
MIN_LENGTH = 16

logger = logging.getLogger(__name__)

GLOBAL_CLUSTER_MAPPING: dict[str, str] = {}


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
        "--num_workers",
        type=int,
        default=len(os.sched_getaffinity(0)),
        help="Number of parallel workers.",
    )
    args = parser.parse_args()
    return args


# =================================================
# Helper functions for data filtering
# ==================================================
def validate_sequence(prot: protein.Protein) -> bool:
    # Check any single amino acid accounts for more than 80% of the sequence
    sequence = prot.sequence
    aa_counts = {aa: sequence.count(aa) for aa in set(sequence)}
    if any(count / len(sequence) > 0.8 for count in aa_counts.values()):
        logger.debug(
            f"{prot.name} marked invalid due to "
            f"overrepresentation of a single amino acid."
        )
        return False
    return True


def validate_geometry(prot: protein.Protein) -> bool:
    """Check if a polymer chain is valid."""
    ca_coords = prot.coordinates[:, 1, :]
    is_resolved = np.isfinite(ca_coords).all(axis=-1)
    n_resolved = np.sum(is_resolved)
    if n_resolved < MIN_LENGTH:
        logger.debug(f"{prot.name} marked invalid due to insufficient resolved CA atoms.")
        return False
    left = ca_coords[:-1]
    right = ca_coords[1:]
    dists = np.linalg.norm(left - right, axis=-1)
    if np.any(dists > 10.0):
        logger.debug(f"{prot.name} marked invalid due to CA trace discontinuity.")
        return False
    return True


def parse_cif(
    name: str,
    cif_path: pathlib.Path,
    out_dir: pathlib.Path,
) -> tuple[int, int]:
    """Parse a CIF file and return a gemmi.cif.Document object."""
    global GLOBAL_CLUSTER_MAPPING
    assert GLOBAL_CLUSTER_MAPPING, "Cluster mapping is not initialized in worker."

    # Read CIF file
    doc: gemmi.cif.Document = gemmi.cif.read(str(cif_path))
    block: gemmi.cif.Block = doc[0]

    # Prepare gemmi structure
    raw_struct: gemmi.Structure = gemmi.make_structure_from_block(block)
    raw_struct.assign_subchains()
    raw_struct.setup_entities()
    cif_factory.clean_up_gemmi_structure(raw_struct)

    for entity in raw_struct.entities:
        subchain = entity.subchains[0]
        restypes = [res.name for res in raw_struct[0].get_subchain(subchain)]
        entity.full_sequence = restypes

    chains: list[tuple[protein.Protein, metadata.Metadata]] = []
    for c, ids in cif_factory.get_protein_chains(raw_struct):
        label_id = ids["label_id"]
        auth_id = ids["auth_id"]
        entity_id = ids["entity_id"]
        c.name = f"{name}_{label_id}"

        # Get cluster ID from entity_id
        seq = c.sequence
        if len(seq) < MIN_LENGTH:
            continue
        cluster_id = GLOBAL_CLUSTER_MAPPING.get(seq)
        if cluster_id is None:
            logger.warning(f"Cluster ID not found for sequence {seq} in {name}.")
            continue

        m = metadata.Metadata(
            id=c.name,
            label_asym_id=label_id,
            auth_asym_id=auth_id,
            entity_id=entity_id,
            num_residues=len(c),
            cluster_id=cluster_id,
        )
        chains.append((c, m))

    # Save each chain as a separate NPZ file
    for c, m in chains:
        npz_path = out_dir / f"{c.name}.npz"
        DataPipeline.save(c, npz_path)
        json_path = out_dir / f"{c.name}.json"
        m_dict = m.to_dict()
        with open(json_path, "w") as f:
            json.dump(m_dict, f)
    if not chains:
        logger.warning(f"No valid protein chains found in {name}.")
        return NO_PROTEIN_FILTERED, 0

    return SUCCESS, len(chains)


def worker_fn(
    cif_path: pathlib.Path,
    output_dir: pathlib.Path,
):
    # Output path
    pdb_id = cif_path.name.split(".")[0].lower()
    out_dir = output_dir / pdb_id
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        return parse_cif(pdb_id, cif_path, out_dir)
    except Exception as e:
        logger.error(f"Failed to process ({pdb_id}): {e}")
        return FAILED, 0


def worker_init(cluster_path: pathlib.Path):
    global GLOBAL_CLUSTER_MAPPING
    cluster_mapping = {}
    with open(cluster_path) as f:
        next(f)  # skip header
        for line in f:
            entity_id, cluster_id, seq = line.strip().split(",", 2)
            cluster_mapping[seq] = cluster_id.upper()
    GLOBAL_CLUSTER_MAPPING = cluster_mapping


def main():
    """Main function to process RCSB mmCIF files"""
    args = parse_args()
    cif_dir: pathlib.Path = args.cif_dir
    data_dir: pathlib.Path = args.data_dir / "disordered_pdb_afm/"

    # Read fasta to get pdb ids
    with open(data_dir / "rcsb_sequences.fasta") as f:
        pdb_ids = set()
        for line in f:
            if line.startswith(">"):
                pdb_id = line[1:5].lower()
                pdb_ids.add(pdb_id)

    # Read cluster mapping
    cluster_mapping_path = data_dir / "rcsb_clusters.csv"
    assert cluster_mapping_path.exists(), (
        f"Cluster mapping file not found: {cluster_mapping_path}"
    )
    out_dir: pathlib.Path = data_dir / "npz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare partial function for multiprocessing
    worker_wrapped = functools.partial(worker_fn, output_dir=out_dir)

    cif_paths = list(cif_dir.rglob("*.cif"))
    print(f"Found {len(cif_paths)} mmCIF files to process.")
    cif_paths = [p for p in cif_paths if p.name.split(".")[0].lower() in pdb_ids]
    print(f"Filtered to {len(cif_paths)} mmCIF files based on FASTA")

    flags: list[int] = []
    n_total = 0
    with multiprocessing.Pool(
        args.num_workers,
        initializer=worker_init,
        initargs=(cluster_mapping_path,),
    ) as pool:
        pbar = tqdm(
            pool.imap_unordered(worker_wrapped, cif_paths, chunksize=100),
            total=len(cif_paths),
            desc="Processing RCSB mmCIF files",
        )
        for flag, n_success in pbar:
            n_total += n_success
            flags.append(flag)
            pbar.set_postfix({"Total Chains": n_total})
    print("Processing completed.")

    # Print stats
    print("Processing statistics:")
    print(f"  Total files processed: {len(flags)}")
    print(f"  Successfully processed: {flags.count(SUCCESS)}")
    print(f"  Filtered (no protein): {flags.count(NO_PROTEIN_FILTERED)}")
    print(f"  Failed: {flags.count(FAILED)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
