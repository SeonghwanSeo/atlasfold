"""Create entry-to-template mapping metadata from template hit NPZ files."""

import argparse
import json
import pathlib

import msgpack
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Create template mapping metadata.")
    parser.add_argument(
        "--metadata_dir",
        type=pathlib.Path,
        required=True,
        help="Path to train_template_metadata directory.",
    )
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to data root or rcsb_multimer directory.",
    )
    args = parser.parse_args()
    return args


def load_entry_mapping(path: pathlib.Path) -> dict:
    """Load one entry mapping NPZ as a JSON/msgpack-serializable dict.

    The source idx_map is 1-based. We expose it as explicit 1-based
    entry_indices/template_indices fields to avoid ambiguity about columns.
    """
    templates = []
    with np.load(path, allow_pickle=True) as data:
        for template_id in data.files:
            info = data[template_id].item()
            idx_map = np.asarray(info["idx_map"], dtype=np.int64)
            if idx_map.ndim != 2 or idx_map.shape[1] != 2:
                raise ValueError(
                    f"{path}:{template_id} has invalid idx_map shape {idx_map.shape}"
                )
            if idx_map.size > 0 and idx_map.min() < 1:
                raise ValueError(
                    f"{path}:{template_id} idx_map is expected to be 1-based."
                )
            templates.append(
                {
                    "template_id": template_id,
                    "index": int(info["index"]),
                    "release_date": str(info["release_date"]),
                    "entry_indices": idx_map[:, 0].astype(int).tolist(),
                    "template_indices": idx_map[:, 1].astype(int).tolist(),
                }
            )

    templates.sort(key=lambda x: (x["index"], x["template_id"]))
    return {
        "entry_id": path.stem,
        "templates": templates,
    }


def main():
    args = parse_args()
    data_dir = args.data_dir / "rcsb_multimer/"
    template_dir = data_dir / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)

    metadata_paths = sorted(args.metadata_dir.glob("*.npz"))
    print(f"Found {len(metadata_paths)} template metadata NPZ files.")

    jsonl_path = template_dir / "template_mapping.json"
    msgpack_path = template_dir / "template_mapping.msgpack"

    packer = msgpack.Packer(use_bin_type=True)
    with open(jsonl_path, "w") as jsonl_f, open(msgpack_path, "wb") as msgpack_f:
        msgpack_f.write(packer.pack_array_header(len(metadata_paths)))
        for path in tqdm(metadata_paths, desc="Creating template mapping"):
            record = load_entry_mapping(path)
            jsonl_f.write(json.dumps(record, separators=(",", ":")) + "\n")
            msgpack_f.write(packer.pack(record))

    print(f"Saved JSONL mapping to {jsonl_path}")
    print(f"Saved msgpack mapping to {msgpack_path}")


if __name__ == "__main__":
    main()
