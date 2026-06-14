# Construct the AlphaFold-Multimer Distillation Dataset

We use the AlphaFold-multimer distillation set from expanded AlphaFold database (AFDB) [[Han et al., 2026](https://www.biorxiv.org/content/10.64898/2026.03.27.714458v1)].

The dataset version we used contains **1.8M predicted homodimer structures**.

## Preprocessing Workflow

After downloading the dataset, you can preprocess it by executing the scripts in this directory sequentially.

> [!NOTE]
> We assume you have already filtered the structure files according to the confidence score cutoff introduced in the original paper.

### 1. Extract Sequences and Structures
Process the raw PDB files (can be `.zst` compressed) and save them as compressed NumPy (`.npz`) files. This script expects the input directory to be organized into subdirectories (shards).

```bash
python a1_process.py \
    --pdb_dir /path/to/raw/pdb_files \
    --data_dir /path/to/output/synthetic_data \
    --num_workers 16
```

### 2. Construct LMDB Database
Merge the processed `.npz` files into a single LMDB file for efficient training access. This step also generates a `manifest.msgpack` file containing metadata (including pLDDT) for all entries.

```bash
python a2_construct_lmdb.py \
    --data_dir /path/to/output/synthetic_data \
    --size_gb 1000
```

## Dataset Structure
After preprocessing, your data directory will look like this:
```
synthetic_data/
├── npz/                   # Intermediate processed files (can be deleted after Step 2)
│   ├── shard_0/
│   │   ├── AF000....npz
│   │   └── ...
│   └── ...
├── structure.lmdb         # Combined structure database
├── manifest.msgpack       # Metadata for all entries
```

> [!TIP]
> The `npz/` directory contains intermediate files and can be safely removed once `structure.lmdb` and `manifest.msgpack` are successfully created to reclaim disk space.

