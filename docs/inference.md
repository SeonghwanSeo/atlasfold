# AtlasFold inference

Use the AtlasFold CLI for monomer and multimer predictions from FASTA files.
For model loading, runners, and custom Python workflows, see the [Python API](python_api.md).

## Installation

Install Python 3.10 or later:

```bash
pip install "atlasfold[fold]"
```

To install the optional cuEquivariance triangle kernels for a CUDA 12 PyTorch build, use:

```bash
pip install cuequivariance_torch cuequivariance_ops_cu12 cuequivariance_ops_torch_cu12
```

## Command-line inference

Each FASTA record describes one target.
Use distinct target names in the FASTA headers; each name determines its output subdirectory.
The released CLI accepts sequences only, without templates.

### Monomer

Save one protein sequence per record in `monomers.fasta`:

```fasta
>protein_a
MKTAYIAKQRQISFVKSHFS
```

```bash
atlasfold monomer --input-fasta monomers.fasta --out-dir predictions/monomers
```

### Multimer

Save one complex per record in `multimers.fasta`, separating chain sequences with `:`:

```fasta
>complex_a
MKTAYIAKQRQISFVKSHFS:GGHVDHGKSTTTGHLIYK
```

```bash
atlasfold multimer --input-fasta multimers.fasta --out-dir predictions/multimers
```

Repeated chain sequences represent repeated copies in a homomer.
Both commands write structures, confidence summaries, and completion markers automatically.
See [Command-line options](#command-line-options) for the complete option reference.
Measured GPU memory and runtime across sequence lengths are available in the [performance guide](performance.md).

## Multi-GPU inference

Use `--gpu-ids` with either CLI subcommand to process multiple FASTA targets across GPUs on one machine:

```bash
atlasfold monomer --input-fasta monomers.fasta --out-dir predictions/monomers --gpu-ids 0 1
atlasfold multimer --input-fasta multimers.fasta --out-dir predictions/multimers --gpu-ids 0 1
```

- Without `--gpu-ids`, inference uses one GPU when CUDA is available, otherwise CPU.
- `--gpu-ids` cannot be combined with `--device`.
- GPU IDs are indices within `CUDA_VISIBLE_DEVICES` when it is set; for example, `CUDA_VISIBLE_DEVICES=2,3` maps `--gpu-ids 0 1` to physical GPUs 2 and 3.
- Targets are distributed across GPUs, with each target processed entirely on one GPU.
  `--max-tokens-per-batch` applies per GPU.

## Command-line options

Run `atlasfold monomer --help` or `atlasfold multimer --help` for all options.
Both commands accept the options below.
Defaults are shared unless shown as monomer / multimer.

| Option | Default | Description |
| --- | --- | --- |
| `-i`, `--input-fasta` | Required | Input FASTA: one protein per monomer record, or one complex per multimer record with chains separated by `:`. |
| `-o`, `--out-dir` | Required | Output directory, with one subdirectory per target. |
| `--seeds` | `1` | One or more inference seeds. |
| `--num-samples` | `5` | Diffusion predictions per target/seed. |
| `--num-recycles` | `4` / `10` | Model recycling iterations after the initial trunk pass. |
| `--num-steps` | Automatic / `200` | Diffusion steps; monomer uses the length-dependent [diffusion schedule](#diffusion-configuration). |
| `--mlm-prob` | `0.15` / `0.20` | LM masking probability during recycling; greater than 0 and at most 1. |
| `--kernel` | `auto` | Triangle backend: `auto`, `torch`, `triton`, or `cuequiv`; `auto` prefers Triton, then cuEquivariance, then Torch on CUDA; other devices use Torch. |
| `--device` | CUDA if available, otherwise CPU | Torch device, such as `cuda:0` or `cpu`; excludes `--gpu-ids`. |
| `--gpu-ids` | Not set | Distribute targets across unique non-negative visible CUDA device IDs; excludes `--device`. |
| `--max-tokens-per-batch` | `1024` | Bucketed-residue budget per model call, per GPU; controls batching rather than maximum sequence length. |
| `--length-buckets` | Automatic | One or more positive residue-length buckets; the largest must fit every target, including all chains for multimer inputs. |
| `--model-path` | Published checkpoint | Local folding-model weights matching the monomer or multimer architecture. |
| `--lm-path` | Published AtlasLM checkpoint | Local AtlasLM weights. |
| `--cache-dir` | Hugging Face default | Download cache for AtlasFold and AtlasLM weights. |
| `--format` | `cif` | Structure output format: `cif` (mmCIF) or `pdb`. |
| `--overwrite` | Off | Rerun targets whose output directory contains `done.txt`. |
| `--save-confidence` | Off | Save per-sample pLDDT and PAE arrays as NPZ; multimer also includes PDE. |
| `--save-distogram` | Off | Save distogram logits and boundaries as one NPZ file per seed. |
| `-h`, `--help` | — | Show command help and exit. |

### Sampling

Use `--seeds` for independent runs and `--num-samples` for the number of predictions per seed:

```bash
atlasfold monomer -i monomers.fasta -o predictions/monomers --seeds 1 2 --num-samples 5
atlasfold multimer -i multimers.fasta -o predictions/multimers --seeds 1 2 --num-samples 5
```

Each command produces ten structures per target.
Each target produces one set of `--num-samples` structures for each seed.
Output filenames include both the seed and sample index.

Both commands require `--mlm-prob` to be greater than 0 and at most 1.
LM masking is sampled again for each recycling iteration.

### Diffusion configuration

Monomer inference uses 20 diffusion steps through 512 bucketed residues, 30 steps through 1024, and 100 steps for longer inputs.
Multimer inference uses 200 steps by default.
Set `--num-steps` to override these defaults.

## Batching

Both commands batch targets of similar length automatically.
Use `--max-tokens-per-batch` to set the bucketed-residue budget per GPU, and `--length-buckets` to supply explicit residue-length buckets.
Multimer target length is the sum of all chain lengths.
A target larger than the token budget still runs by itself.
See [Length bucketing and token budget](python_api.md#length-bucketing-and-token-budget) for the batching rules.

## Outputs and confidence

### Output layout

Each FASTA record is written to a separate directory named after the first whitespace-delimited token in its header.
With the default one seed and five samples, the `protein_a` monomer example produces:

```text
predictions/monomers/protein_a/
├── protein_a_seed-1_sample-0_model.cif
├── protein_a_seed-1_sample-0_confidence.json
├── ...
├── protein_a_seed-1_sample-4_model.cif
├── protein_a_seed-1_sample-4_confidence.json
├── protein_a_ranked_model.cif
├── protein_a_ranked_confidence.json
├── protein_a_summary.csv
└── done.txt
```

| File | Contents |
| --- | --- |
| `*_seed-*_sample-*_model.cif` | Predicted structure for one seed and diffusion sample. |
| `*_seed-*_sample-*_confidence.json` | Scalar confidence scores for one sample. |
| `*_ranked_model.cif` | Copy of the highest-ranked sample's structure. |
| `*_ranked_confidence.json` | Scalar confidence scores for the highest-ranked sample. |
| `*_summary.csv` | Seed, sample index, and scalar confidence scores for every sample. |
| `*_seed-*_sample-*_confidence.npz` | Per-sample confidence arrays, written with `--save-confidence`. |
| `*_seed-*_distogram.npz` | Distance-bin logits and boundaries per seed, written with `--save-distogram`. |
| `done.txt` | Completion marker used to skip finished targets. |

Use `--format pdb` to write PDB instead of mmCIF.
A completed target is skipped when `done.txt` exists unless `--overwrite` is supplied.

### Confidence and ranking

Scalar confidence scores are always written to the per-sample confidence JSON files and target summary CSV.
Monomer predictions are ranked by mean pLDDT.
Multimer predictions are ranked by `0.8 * ipTM + 0.2 * pTM + 0.5 * fraction_disordered`.
The fraction disordered is the fraction of residues with window-smoothed relative accessible surface area (RASA) above 0.581.

| Value | Meaning and scale |
| --- | --- |
| pLDDT | Local confidence per residue; raw arrays use 0–1, while `avg_plddt` and structure B-factors use 0–100. |
| PAE | Predicted aligned error between residues, in ångströms. |
| PDE | Predicted distance error between residues, in ångströms; available for multimer predictions. |
| pTM | Confidence in the overall fold, on a 0–1 scale. |
| ipTM | Confidence in interfaces between chains, on a 0–1 scale; available for multimer predictions. |
| Fraction disordered | Estimated disordered fraction, on a 0–1 scale; used in multimer ranking. |
| Multimer ranking score | Combined score from pTM, ipTM, and fraction disordered, on a 0–1.5 scale. |

### Optional arrays

Use `--save-confidence` to additionally save per-residue pLDDT and pairwise PAE arrays as one NPZ file per sample; multimer files also include PDE.
Use `--save-distogram` to save raw distogram logits and boundaries once per seed.
These optional arrays grow quadratically with sequence length for pairwise values.

## Reproducing released benchmarks

The [benchmark release](benchmarks.md) documents the CAMEO22, CASP14, CASP15, and FoldBench sampling and selection protocols.
Large prediction structures and evaluation CSV files are available from the linked Google Drive folder.
