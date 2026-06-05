# AtlasFold

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/atlaslm.svg)](https://badge.fury.io/py/atlaslm)

This repository contains the official code and pre-trained weights for two core models:
* **AtlasLM:** A state-of-the-art protein language model. Trained on a massive metagenomic protein atlas, it outperforms many existing open-source protein language models on unsupervised contact map prediction.
* **AtlasFold:** A protein monomer folding model. Built on the top of AtlasLM, it provides accurate structure predictions without the need for multiple sequence alignments (MSAs).

All models are free for both academic and commercial use under the **MIT** License.

## Citation

TODO

## Table of Contents

- [Main Models](#main-models-)
- [Installation ](#installation-)
- [AtlasLM](#atlaslm-)
- [AtlasFold](#atlasfold-)

## Main Models

| Shorthand | Available | Description |
|-----------|-----------| ------------|
| **AtlasLM** | `atlaslm-600m` `atlaslm-3b` | State-of-the-art protein language models with unsupervised learning for general purpose. Available in `600m` and `3b` sizes. |
| **AtlasFold** | `atlasfold-3b-base` | End-to-end single sequence 3D structure predictor, powered by AtlasLM. |


## Installation

AtlasLM requires Python >= 3.10 and PyTorch.

```bash
# Install from PyPI
pip install atlaslm

# From github
git clone https://github.com/SeonghwanSeo/atlasfold.git
pip install .
```

To use AtlasFold, you will also need to install the additional dependencies.

```bash
# Install AtlasFold with optional cuEquivariance kernels
pip install "atlaslm[fold, cuequiv]"
```

## AtlasLM protein language model.

**AtlasLM** is the reproduction of [**ESM Cambrian (ESM C)**](https://www.evolutionaryscale.ai/blog/esm-cambrian), the state-of-the-art protein language model developed by EvolutionaryScale.
~~While the original model has restrictive licensing, this repository provides pre-trained weights under the **MIT License**, facilitating unrestricted research and commercial application.~~

### Available Models

We provide the following models under the **MIT License**:

| Model | Params | Layers | Width | Heads |
| --- | --- | --- | --- | --- |
| `atlaslm-600m` | 575M | 36 | 1152 | 18 |
| `atlaslm-3b` | 3.1B | 48 | 2304 | 36 |


### Quick Start

You can load and use a pretrained AtlasLM model as follows:

```python
from atlaslm import load_model

# Load AtlasLM model
model = load_model("atlaslm-600m", device='cuda', dtype=torch.bfloat16)

sequences = [
    "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPDWQNYTPGPGIRYPLKF",
    "GILGYTEHQVVSSDFQKAAQQLLQYFQKAGYTPGPGIRYPLKF",
]

# 1. Get Embeddings
# Returns sequence embeddings and optionally hidden states
out = model.embed_sequences(sequences, return_hidden_states=True)
embeddings = out.embeddings      # (batch_size, seq_len, d_model)
hidden_states = out.hidden_states  # List of hidden states per layer

# 2. Get Attention Maps 
# Requires return_attentions=True. Note that this increases memory usage.
out = model.embed_sequences(sequences, return_attentions=True)
attentions = out.attentions  # List of attention maps: (batch_size, n_heads, seq_len, seq_len)
```

## AtlasFold protein structure prediction

TODO.

---

## Acknowledgements

This project was developed as part of the **K-Fold** initiative supported by the Ministry of Science and ICT (MSIT) of the Republic of Korea.
The K-Fold for biomolecular complex prediction is currently under active development with numerous contributors in KAIST and will be released soon!

---

I would like to thank [Dr. Hyeongwoo Kim](https://scholar.google.com/citations?user=YpiY1q8AAAAJ&hl=en&oi=ao) and [Prof. Woo Youn Kim](https://scholar.google.com/citations?user=elJ5KrcAAAAJ&hl=en) for their guidance and support during the development of AtlasFold.

This project is built upon the pioneering works of Google DeepMind, Meta AI, OpenFold Consortium, and EvolutionaryScale in the fields of biomolecular language modeling and structure prediction.
I am deeply grateful to the open-source community for advancing the fields of biomolecular language modeling and structure prediction.

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
