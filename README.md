# AtlasFold

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/atlasfold.svg)](https://badge.fury.io/py/atlasfold)

This repository contains the official code and pre-trained weights for two core models:

* **AtlasLM:** A state-of-the-art protein language model. Trained on a massive metagenomic protein atlas, it outperforms many existing open-source protein language models on unsupervised contact map prediction.
* **AtlasFold:** A protein structure prediction model built on top of AtlasLM. It predicts monomers and multimeric complexes without the need for multiple sequence alignments (MSAs).

All models are free for both academic and commercial use under the **MIT** License.

## Citation

TODO

## Table of Contents

- [Main Models](#main-models)
- [Installation](#installation)
- [AtlasLM protein language model](#atlaslm-protein-language-model)
- [AtlasFold protein structure prediction](#atlasfold-protein-structure-prediction)
- [Training](#training)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Main Models

| Shorthand | Available | Description |
|-----------|-----------| ------------|
| **AtlasLM** | `atlaslm-600m`, `atlaslm-3b` | State-of-the-art protein language models with unsupervised learning for general purpose. Available in `600m` and `3b` sizes. |
| **AtlasFold** | `atlasfold-260703` | Protein structure prediction model. |
| **AtlasFold-M** | `atlasfold-m-260725` | Protein multimeric structure prediction model. |


## Installation

AtlasFold requires Python >= 3.10 and PyTorch.

```bash
# Install from PyPI
pip install atlasfold

# From github
git clone https://github.com/SeonghwanSeo/atlasfold.git
cd atlasfold
pip install .
```

To use AtlasFold, you will also need to install the additional dependencies.

```bash
# Install AtlasFold with optional cuEquivariance kernels
pip install "atlasfold[fold,cuequiv]"
```

## AtlasLM protein language model

**AtlasLM** is the reproduction of [**ESM Cambrian (ESM C)**](https://www.evolutionaryscale.ai/blog/esm-cambrian), the state-of-the-art protein language model developed by EvolutionaryScale.


| Model | Params | Layers | Width | Heads |
| --- | --- | --- | --- | --- |
| `atlaslm-600m` | 575M | 36 | 1152 | 18 |
| `atlaslm-3b` | 3.1B | 48 | 2304 | 36 |


### Quick Start

You can load and use a pretrained AtlasLM model as follows:

```python
import torch

from atlaslm import load_model

# Load AtlasLM model
model = load_model("atlaslm-3b", device="cuda", dtype=torch.bfloat16)

sequences = [
    "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPDWQNYTPGPGIRYPLKF",
    "GILGYTEHQVVSSDFQKAAQQLLQYFQKAGYTPGPGIRYPLKF",
]

# 1. Get Embeddings
# Returns sequence embeddings and optionally hidden states
out = model.embed_sequences(sequences, return_hidden_states=True)
embeddings = out.embeddings  # (batch_size, seq_len, d_model)
hidden_states = out.hidden_states  # List of hidden states per layer

# 2. Get Attention Maps
# Requires return_attentions=True. Note that this increases memory usage.
out = model.embed_sequences(sequences, return_attentions=True)
attentions = (
    out.attentions
)  # List of attention maps: (batch_size, n_heads, seq_len, seq_len)
```

## AtlasFold protein structure prediction

AtlasFold predicts monomer structures directly from a single protein sequence. AtlasFold-Multimer uses a sequence for each chain and predicts the structure and confidence of the complete complex. Neither runner requires an MSA.

> **Note:** Template support for AtlasFold-Multimer is TODO.

### Command-line inference

Pretrained weights are hosted on Hugging Face:

- [`SeonghwanSeo/atlasfold-260703`](https://huggingface.co/SeonghwanSeo/atlasfold-260703)
- [`SeonghwanSeo/atlasfold-m-260725`](https://huggingface.co/SeonghwanSeo/atlasfold-m-260725)

Both the Python API and CLI download AtlasFold and AtlasLM weights automatically.
Use `--cache-dir` to select a cache location, or `--model-path` as an optional
override for a local/custom model parameters.

For monomer inference, provide one sequence per FASTA record:

```text
>protein_a
MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPD
>protein_b
GILGYTEHQVVSSDFQKAAQQLLQYFQKAGY
```

```bash
python run_atlasfold.py monomer \
    --input-fasta monomers.fasta \
    --out-dir predictions/monomers
```

For multimer inference, each FASTA record represents one complex. Separate its chains with `:`:

```text
>complex_a
MKTAYIAKQRQISFVKSHFS:GGHVDHGKSTTTGHLIYK
>complex_b
GILGYTEHQVVSSDFQKAA:QQLLQYFQKAGY
```

```bash
python run_atlasfold.py multimer \
    --input-fasta multimers.fasta \
    --out-dir predictions/multimers
```

When AtlasFold is installed, the same pipelines are also available as CLI subcommands:

```bash
atlasfold monomer --input-fasta monomers.fasta --out-dir predictions/monomers
atlasfold multimer --input-fasta multimers.fasta --out-dir predictions/multimers
```

Both CLI entry points use the first whitespace-delimited token in each FASTA header as the target name. Target names must be unique and cannot be empty, `.`, `..`, or contain path separators. Sequences are uppercased and whitespace is removed; nonstandard residues are replaced with `X` with a warning that identifies the target and, for multimers, the chain.

Common inference options and their model-specific defaults are:

| Option | Monomer default | Multimer default | Description |
| --- | --- | --- | --- |
| `--num-recycles N` | `4` | `10` | Set the number of recycling iterations. |
| `--mlm-prob P` | `0.15` | `0.20` | Set the LM masking probability used during recycling. |
| `--num-samples N` | `5` | `5` | Generate `N` diffusion samples for every seed. |
| `--seed S [S ...]` | `[1]` | `[1]` | Run one or more random seeds. |
| `--num-steps N` | Auto | `200` | Set the number of diffusion steps. |
| `--device DEVICE` | Auto | Auto | Use CUDA when available, otherwise CPU. Accepts Torch device strings such as `cpu`, `cuda`, or `cuda:0`. |
| `--no-kernel` | `off` | `off` | Disable optional cuEquivariance kernels. |
| `--max-tokens-per-batch N` | `1024` | `1024` | Limit bucketed residue tokens in each model call; reduce this after a CUDA out-of-memory error. |
| `--format {cif,pdb}` | `cif` | `cif` | Select the output structure format. |
| `--save-confidence-arrays` | `off` | `off` | Save raw confidence arrays in NumPy NPZ files. |
| `--save-distogram` | `off` | `off` | Return and save raw distogram logits and boundaries. |
| `--overwrite` | `off` | `off` | Recompute targets that already have a `done.txt` marker. |

> **Note:** The automatic monomer diffusion schedule uses `20` steps for L <= 512, `30` steps for L <= 1024, and `100` steps otherwise, where L is the runner's bucketed residue length.

Use `atlasfold monomer --help`, `atlasfold multimer --help`, or the equivalent `python run_atlasfold.py <model> --help` command for the complete model-specific option lists and current defaults.

### Output files

Each target is written to its own output directory. Every generated sample has a structure file and a JSON confidence summary. The highest-ranked sample is also written as `<target>_ranked_model.cif` (or `.pdb`), and all samples are listed in `<target>_summary.csv`. Monomers are ranked by mean pLDDT; multimers are ranked by `0.8 * ipTM + 0.2 * pTM`.

`--save-confidence-arrays` additionally writes one NumPy NPZ file per sample. Monomer files contain `plddt` and `pae`; multimer files contain `plddt`, `pae`, and `pde`. Scalar scores such as pTM and ipTM remain in the JSON and CSV outputs. `--save-distogram` writes `logits` and `boundaries` once per seed. Raw pairwise arrays scale quadratically with sequence length and can require substantial host memory and disk space for long targets.

### Python API

AtlasFold runners support single-target folding, streaming and batched inference, automatic length bucketing, sample ranking, confidence outputs, and optional distogram outputs. See the [inference guide](docs/inference.md) for examples and detailed API documentation.

The following example loads a model and predicts a single target with `fold()`. Fine-tuned model instances can be passed to `get_runner()` in the same way:

```python
from atlasfold.pretrained import get_runner, load_model

# Monomer prediction
monomer_model = load_model("atlasfold", device="cuda")
folding_runner = get_runner(monomer_model)
seq = 'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG'
monomer_out = folding_runner.fold("test", seq, num_samples=5)
print(monomer_out.best.avg_plddt) # 96.42
with open("test_monomer.pdb", "w") as f:
    f.write(monomer_out.best.to_pdb())

# Multimer prediction
multimer_model = load_model("atlasfold-m", device="cuda")
multimer_runner = get_runner(multimer_model)
seq1 = (
    "GSEVQLLESGGGLVQAGDSLRLSCAASGRTFSAYAMGWFRQAPGKEREFVAAISWSGNSTYYAD"
    "SVKGRFTISRDNAKNTVYLQMNSLKPEDTAIYYCAARKPMYRVDISKGQNYDYWGQGTQVTVSS"
)
seq2 = "GAMGPGVDTQIFEDPREFLSHLEEYLRQVGGSEEYWLSQIQNHMNGPAKKWWEFKQGSVKNWVEFKKEFLQYSEG"
multimer_out = multimer_runner.fold(
    "test_m", [seq1, seq2], seeds=[1, 2], num_samples=5
)
print(multimer_out.best.iptm) # 0.956
with open("test_multimer.cif", "w") as f:
    f.write(multimer_out.best.to_mmcif())
```

## Training

To train the monomer or multimer models, see the [training guide](docs/training.md) for staged configurations, memory tuning, distributed batch sizing, and checkpoint handoffs.

---

## Acknowledgements

This project was developed as part of the **K-Fold** initiative supported by the Ministry of Science and ICT (MSIT) of the Republic of Korea. The K-Fold for biomolecular complex prediction is currently under active development with numerous contributors in KAIST and will be released soon!

---

I would like to thank [Dr. Hyeongwoo Kim](https://scholar.google.com/citations?user=YpiY1q8AAAAJ&hl=en&oi=ao) and [Prof. Woo Youn Kim](https://scholar.google.com/citations?user=elJ5KrcAAAAJ&hl=en) for their guidance and support during the development of AtlasFold.

This project is built upon the pioneering works of Google DeepMind, Meta AI, OpenFold Consortium, and EvolutionaryScale in the fields of biomolecular language modeling and structure prediction. I am deeply grateful to the open-source community for advancing the fields of biomolecular language modeling and structure prediction.

**Foundations of AtlasLM:**
- **ESM2**: Lin, Zeming, et al. "Evolutionary-scale prediction of atomic-level protein structure with a language model." Science 379.6637 (2023): 1123-1130.
- **ESM3**: Hayes, Thomas, et al. "Simulating 500 million years of evolution with a language model." Science 387.6736 (2025): 850-858.
- **ESMC**: ESM Team. "ESM Cambrian: Revealing the mysteries of proteins with unsupervised learning." EvolutionaryScale Website, December 4, 2024. https://evolutionaryscale.ai/blog/esm-cambrian."

**Foundations of AtlasFold:**
- **AlphaFold2**: Jumper, John, et al. "Highly accurate protein structure prediction with AlphaFold." nature 596.7873 (2021): 583-589.
- **OpenFold**: Ahdritz, Gustaf, et al. "OpenFold: retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization." Nature methods 21.8 (2024): 1514-1524.
- **ESMFold**: Lin, Zeming, et al. "Evolutionary-scale prediction of atomic-level protein structure with a language model." Science 379.6637 (2023): 1123-1130.
- **AlphaFold3**: Abramson, Josh, et al. "Accurate structure prediction of biomolecular interactions with AlphaFold 3." Nature 630.8016 (2024): 493-500.
- **SimpleFold**: Wang, Yuyang, et al. "Simplefold: Folding proteins is simpler than you think." arXiv preprint arXiv:2509.18480 (2025).
- **ESMFold2**: Candido, Salvatore, et al. "Language Modeling Materializes a World Model of Protein Biology." bioRxiv (2026): 2026-06.

## License

This source code and weights are licensed under the **MIT license**.
