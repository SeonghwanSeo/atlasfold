"""Construct LMDB database for AF-M homodimer dataset."""

import argparse
import logging
import multiprocessing
import os
import pathlib

import lmdb
import msgpack
import numpy as np
from tqdm import tqdm

from atlasfold.common import metadata, metadata_complex, protein_complex

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Construct synthetic data set.")
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to the preprocessed data directory.",
    )
    parser.add_argument(
        "--size_gb",
        type=int,
        default=128,
        help="LMDB map size in GB.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=len(os.sched_getaffinity(0)),
        help="Number of parallel workers.",
    )
    args = parser.parse_args()
    return args


def process_metadata(npz_path: pathlib.Path) -> tuple[str, dict | None, bool]:
    """Worker function to parse the homodimer structure and extract metadata."""
    entry_id = npz_path.stem
    try:
        compl = protein_complex.ProteinComplex.load_npz(npz_path)

        # Verify homodimer structure
        assert len(compl.chains) == 2, (
            f"Expected 2 chains for homodimer, found {len(compl.chains)}"
        )
    except Exception as e:
        return entry_id, {"error": str(e)}, False

    # Get metadata for each chain
    cms: list[metadata.Metadata] = []
    for asym_id, c in [("A", compl.chains[0]), ("B", compl.chains[1])]:
        cm = metadata.Metadata(
            id=f"{entry_id}_{asym_id}",
            label_asym_id=asym_id,
            entity_id=1,  # Homodimer chains share the same sequence entity
            num_residues=len(c),
        )
        cms.append(cm)

    # Compute the average plddt across both chains
    bfactors = [c.b_factors for c in compl.chains]
    bfactors_ca = [b[:, 1] for b in bfactors]
    plddt = np.nanmean(np.concatenate(bfactors_ca)).item()
    plddt = round(float(plddt), 2)  # Round to 2 decimal places
    pred_record = metadata.PredictionRecord(model="AF-M", plddt=plddt)

    complex_metadata = metadata_complex.ComplexMetadata(
        id=entry_id,
        chains=cms,
        pred=pred_record,
    )
    return entry_id, complex_metadata.to_dict(), True


def main():
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir
    npz_dir: pathlib.Path = data_dir / "npz"

    if not npz_dir.exists():
        raise FileNotFoundError(f"Preprocessed NPZ directory not found at {npz_dir}")

    # Construct LMDB database
    print("Initializing LMDB database...")
    lmdb_path = data_dir / "structure.lmdb"
    env = lmdb.open(
        str(lmdb_path),
        map_size=args.size_gb * 1024 * 1024 * 1024,  # size in Bytes
    )
    subdirs = sorted([d for d in npz_dir.iterdir() if d.is_dir()])
    for subdir in tqdm(subdirs, desc="Constructing LMDB"):
        npz_paths = sorted(list(subdir.glob("*.npz")))
        if not npz_paths:
            continue
        with env.begin(write=True) as txn:
            for npz_path in tqdm(
                npz_paths,
                desc=f"Processing files in {subdir.name}",
                leave=False,
            ):
                with open(npz_path, "rb") as f:
                    value_bytes = f.read()
                entry_id = npz_path.stem
                txn.put(entry_id.encode(), value_bytes)

    # Close the LMDB environment
    env.close()
    print(f"Successfully created LMDB at {lmdb_path}")

    # Extract metadata
    print("Extracting metadata from NPZ files...")
    metadatas: list[dict] = []
    with multiprocessing.Pool(args.num_workers) as pool:
        for subdir in tqdm(subdirs, desc="Extracting Metadata"):
            npz_paths = list(subdir.glob("*.npz"))
            if not npz_paths:
                continue

            shard_iterator = pool.imap_unordered(
                process_metadata, npz_paths, chunksize=50
            )
            for entry_id, meta_dict, success in tqdm(
                shard_iterator,
                total=len(npz_paths),
                desc=f"Processing files in {subdir.name}",
                leave=False,
            ):
                if not success:
                    logger.error(
                        f"Failed to process file {entry_id}: {meta_dict.get('error')}"
                    )
                    continue
                metadatas.append(meta_dict)
    print(f"Total entries processed for metadata extraction: {len(metadatas)}")

    # Save to msgpack file
    manifest_path: pathlib.Path = data_dir / "manifest.msgpack"
    metadatas.sort(key=lambda x: x["id"])  # Ensure manifest is sorted by ID
    with open(manifest_path, "wb") as f:
        msgpack.pack(metadatas, f)
    print(f"Saved manifest (msgpack) to {manifest_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    main()
