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
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Main Models

| Shorthand | Available | Description |
|-----------|-----------| ------------|
| **AtlasLM** | `atlaslm-600m`, `atlaslm-3b` | State-of-the-art protein language models with unsupervised learning for general purpose. Available in `600m` and `3b` sizes. |
| **AtlasFold** | `atlasfold-0703` | Protein structure prediction model. |
| **AtlasFold-M** | `atlasfold-m-0725` | Protein multimeric structure prediction model. |


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

**AtlasLM** is the reproduction of [**ESM Cambrian (ESM C)**](https://www.evolutionaryscale.ai/blog/esm-cambrian), the state-of-the-art protein language model developed by EvolutionaryScale. ~~While the original model has restrictive licensing, this repository provides pre-trained weights under the **MIT License**, facilitating unrestricted research and commercial application.~~

### Available Models

We provide the following models under the **MIT License**:

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
attentions = out.attentions  # List of attention maps: (batch_size, n_heads, seq_len, seq_len)
```

## AtlasFold protein structure prediction

AtlasFold predicts monomer structures directly from a single protein sequence. AtlasFold-Multimer uses a sequence for each chain and predicts the structure and confidence of the complete complex. Neither runner requires an MSA.

> **Note:** Template support for AtlasFold-Multimer is TODO.

### Command-line inference

The inference scripts currently load model weights from a local checkpoint.

For monomer inference, provide one sequence per FASTA record:

```text
>protein_a
MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPD
>protein_b
GILGYTEHQVVSSDFQKAAQQLLQYFQKAGY
```

```bash
python run_atlasfold.py \
    --model monomer \
    --model-path checkpoints/atlasfold.pt \
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
python run_atlasfold.py \
    --model multimer \
    --model-path checkpoints/atlasfold-multimer.pt \
    --input-fasta multimers.fasta \
    --out-dir predictions/multimers
```

The scripts use the first whitespace-delimited token in each FASTA header as the target name. Target names must be unique and suitable for use in directory and file names. Sequences are uppercased and whitespace is removed; nonstandard residues are replaced with `X` with a warning that identifies the target and, for multimers, the chain.

Common inference options and their model-specific defaults are:

| Option | Monomer default | Multimer default | Description |
| --- | --- | --- | --- |
| `--num-recycles N` | `4` | `10` | Set the number of recycling iterations. |
| `--mlm-prob P` | `0.15` | `0.20` | Set the LM masking probability used during recycling. |
| `--stochastic` | `off` | Not supported | Increase sampling diversity across seeds. |
| `--num-samples N` | `5` | `5` | Generate `N` diffusion samples for every seed. |
| `--seed S [S ...]` | `[1]` | `[1]` | Run one or more random seeds. |
| `--num-steps N` | Auto | `100` | Set the number of diffusion steps. |
| `--device {cpu,cuda}` | Auto | Auto | Use CUDA when available, otherwise CPU. |
| `--no-kernel` | `off` | `off` | Disable optional cuEquivariance kernels. |
| `--max-tokens-per-batch N` | `1024` | `1024` | Limit bucketed residue tokens in each model call; reduce this after a CUDA out-of-memory error. |
| `--format {cif,pdb}` | `cif` | `cif` | Select the output structure format. |
| `--save-confidence-arrays` | `off` | `off` | Save raw confidence arrays in NumPy NPZ files. |
| `--save-distogram` | `off` | `off` | Return and save raw distogram logits and boundaries. |
| `--overwrite` | `off` | `off` | Recompute targets that already have a `done.txt` marker. |

> **Note:** The automatic monomer diffusion schedule uses `20` steps for L <= 512, `30` steps for L <= 1024, and `100` steps otherwise, where L is the runner's bucketed residue length.

Use `python run_atlasfold.py --help` for the complete option list and current defaults.

### Output files

Each target is written to its own output directory. Every generated sample has a structure file and a JSON confidence summary. The highest-ranked sample is also written as `<target>_ranked_model.cif` (or `.pdb`), and all samples are listed in `<target>_summary.csv`. Monomers are ranked by mean pLDDT; multimers are ranked by `0.8 * ipTM + 0.2 * pTM`.

`--save-confidence-arrays` additionally writes one NumPy NPZ file per sample. Monomer files contain `plddt` and `pae`; multimer files contain `plddt`, `pae`, and `pde`. Scalar scores such as pTM and ipTM remain in the JSON and CSV outputs. `--save-distogram` writes `logits` and `boundaries` once per seed. Raw pairwise arrays scale quadratically with sequence length and can require substantial host memory and disk space for long targets.

### Python API

Create a runner from an already loaded model and call `fold()` for one target:

```python
from atlasfold.runner import FoldingRunner
from atlasfold.runner_multimer import MultimerFoldingRunner

folding_runner = FoldingRunner(model)
monomer = folding_runner.fold("protein_a", "MKTAYIAKQRQISFVKSHFS", num_samples=5)
print(monomer.best.avg_plddt)

multimer_runner = MultimerFoldingRunner(multimer_model)
multimer = multimer_runner.fold("complex_a", ["MKTAYIAKQRQISFVKSHFS", "GGHVDHGKSTTTGHLIYK"])
print(multimer.best.iptm)
```

The multimer runner takes pre-split chain sequences and does not interpret colon-delimited strings. See the [folding guide](docs/folding.md) for batched iteration, input and output types, ranking, confidence arrays, and optional distogram output.

---

## Acknowledgements

This project was developed as part of the **K-Fold** initiative supported by the Ministry of Science and ICT (MSIT) of the Republic of Korea. The K-Fold for biomolecular complex prediction is currently under active development with numerous contributors in KAIST and will be released soon!

---

I would like to thank [Dr. Hyeongwoo Kim](https://scholar.google.com/citations?user=YpiY1q8AAAAJ&hl=en&oi=ao) and [Prof. Woo Youn Kim](https://scholar.google.com/citations?user=elJ5KrcAAAAJ&hl=en) for their guidance and support during the development of AtlasFold.

This project is built upon the pioneering works of Google DeepMind, Meta AI, OpenFold Consortium, and EvolutionaryScale in the fields of biomolecular language modeling and structure prediction. I am deeply grateful to the open-source community for advancing the fields of biomolecular language modeling and structure prediction.

**Foundations of AtlasLM:**
- **ESM2**: Lin, Zeming, et al. "Evolutionary-scale prediction of atomic-level protein structure with a language model." Science 379.6637 (2023): 1123-1130.
- **ESM3**: Hayes, Thomas, et al. "Simulating 500 million years of evolution with a language model." Science 387.6736 (2025): 850-858.
- **ESM C**: ESM Team. "ESM Cambrian: Revealing the mysteries of proteins with unsupervised learning." EvolutionaryScale Website, December 4, 2024. https://evolutionaryscale.ai/blog/esm-cambrian."

**Foundations of AtlasFold:**
- **AlphaFold2**: Jumper, John, et al. "Highly accurate protein structure prediction with AlphaFold." nature 596.7873 (2021): 583-589.
- **OpenFold**: Ahdritz, Gustaf, et al. "OpenFold: retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization." Nature methods 21.8 (2024): 1514-1524.
- **ESMFold**: Lin, Zeming, et al. "Evolutionary-scale prediction of atomic-level protein structure with a language model." Science 379.6637 (2023): 1123-1130.
- **AlphaFold3**: Abramson, Josh, et al. "Accurate structure prediction of biomolecular interactions with AlphaFold 3." Nature 630.8016 (2024): 493-500.
- **SimpleFold**: Wang, Yuyang, et al. "Simplefold: Folding proteins is simpler than you think." arXiv preprint arXiv:2509.18480 (2025).

## License

This source code and weights are licensed under the **MIT license**.
