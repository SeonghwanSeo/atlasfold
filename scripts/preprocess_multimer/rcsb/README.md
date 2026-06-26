# Construct the RCSB Dataset

This directory contains scripts to preprocess the RCSB PDB dataset for training. The workflow includes sequence extraction, clustering, structure processing, and database construction.

## Raw Data Download and Preparation

You can download the raw data files from the RCSB PDB website using the following commands:

```bash
# Create your own raw data directory
mkdir -p /raw_data/RCSB/mmCIF

# mmCIF files for each structure (divided into subdirectories)
rsync -rlpt -v -z --delete --port=33444 rsync.rcsb.org::ftp_data/structures/divided/mmCIF/ /raw_data/RCSB/mmCIF/
```

## Preprocessing Workflow

Execute the scripts sequentially to prepare the dataset.

### 1. Extract Sequences
Extract protein entity sequences from mmCIF files to prepare for clustering. This script filters entries by release date (default: ≤ 2021-09-30).

```bash
python a1_extract_sequences.py \
    --cif_dir /raw_data/RCSB/mmCIF \
    --data_dir /path/to/data/root/ \
    --num_workers 16
```

### 2. Cluster Sequences
Cluster the extracted sequences using MMseqs2 to handle redundancy. This generates `rcsb_clusters.csv`.

```bash
python a2_cluster.py \
    --data_dir /path/to/data/root/ \
    --mmseqs $(which mmseqs)
```

### 3. Process Structures
Perform detailed processing of mmCIF files, including biological assembly expansion, geometry validation, all-atom clash filtering, interface detection, and filtering (resolution ≤ 9.0Å). Assemblies with more than 20 valid protein chains are reduced to the closest 20 chains around a sampled interface/contact seed, following the AF3 SI 2.5.4 subcomplex extraction idea. This step uses the cluster information and saves one complex-level `.npz` and `.json` metadata file per PDB entry.

```bash
python b1_process.py \
    --cif_dir /raw_data/RCSB/mmCIF \
    --data_dir /path/to/data/root/ \
    --num_workers 16
```

### 4. Construct LMDB Database
Merge the processed `.npz` files into a single LMDB database and generate a `manifest.msgpack` containing all metadata and cluster size information.

```bash
python b2_construct_lmdb.py \
    --data_dir /path/to/data/root/ \
    --size_gb 500
```

### 5. Filter for Confidence Head Training
Create a specialized manifest for confidence-head training by filtering complex-level metadata to high-resolution experimental structures (default: 0.1Å to 3.0Å). The script also keeps lightweight AF3/RCSB manifest guards for release date, chain count, and minimum chain length; the expensive bioassembly filters are already applied during Step 3.

```bash
python b3_filter.py \
    --data_dir /path/to/data/root/ \
    --out_prefix manifest_confidence \
    --min_resolution 0.1 \
    --max_resolution 3.0
```

### 6. Process Template Structures
Convert raw template chain `.npz` files into AtlasFold monomer atom14 `.npz` files that can be loaded by `MonomerDataPipeline`.

```bash
python c1_process_templates.py \
    --template_dir /cache/wykim_lab/icl_shwan/templates/templates \
    --data_dir /path/to/data/root/ \
    --num_workers 16
```

### 7. Construct Template LMDB
Pack processed template structures into an LMDB database and write a template manifest.

```bash
python c2_construct_template_lmdb.py \
    --data_dir /path/to/data/root/ \
    --size_gb 256
```

### 8. Create Entry-Template Mapping
Convert per-entry template-hit metadata into JSONL and msgpack mapping files. The source `idx_map` is treated as 1-based and exposed as explicit `entry_indices` and `template_indices` fields.

```bash
python c3_create_template_mapping.py \
    --metadata_dir /cache/wykim_lab/icl_shwan/templates/train_template_metadata \
    --data_dir /path/to/data/root/
```

## Dataset Structure
After preprocessing, your data directory will look like this:
```
rcsb_multimer/
├── rcsb_sequences.fasta    # Extracted sequences
├── rcsb_clusters.csv       # Clustering results
├── npz/                    # Intermediate processed files (can be deleted after Step 4)
│   ├── ab/
│   │   ├── 1abc/
│   │   │   ├── 1abc.npz
│   │   │   └── 1abc.json
│   │   └── ...
│   └── ...
├── structure.lmdb          # Combined structure database
├── manifest.msgpack        # Metadata for all entries
├── manifest_confidence.msgpack # Filtered manifest for confidence training
└── templates/
    ├── npz/                # Processed template atom14 NPZ files
    ├── template.lmdb       # Combined template structure database
    ├── template_manifest.msgpack
    ├── template_mapping.jsonl
    └── template_mapping.msgpack
```

> [!TIP]
> The `npz/` directory can be safely removed once `structure.lmdb` and `manifest.msgpack` are successfully created to reclaim disk space.
