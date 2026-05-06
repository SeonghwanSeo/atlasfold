# AtlasFold

**AtlasFold**: Next-generation protein structure prediction powered by a protein language model trained on an unprecedented atlas of protein sequences.

This repository contains code and pre-trained weights for the **AtlasLM** protein language model and the **AtlasFold** structure prediction model.

AtlasLM outperforms many existing single-sequence protein language models across a range of tasks. AtlasFold harnesses AtlasLM to generate accurate structure predictions directly from the sequence of a protein.

## Main Models

| Shorthand | Description |
|-----------|-------------|
| **AtlasLM** | SOTA general-purpose protein language model. Available in `600m` and `3b` sizes. Can be used to extract sequence features, predict function, and other properties. |
| **AtlasFold** | End-to-end single sequence 3D structure predictor, powered by AtlasLM. |

## Usage

### Installation

AtlasFold requires Python >= 3.10 and PyTorch.

```bash
pip install -e .
```

To install optional dependencies for specific features:
- **Folding:** `pip install -e ".[fold]"`
- **Training:** `pip install -e ".[train]"`
- **CUDA Kernels:** `pip install -e ".[cuequiv]"`

### Quick Start: AtlasLM

You can load and use a pretrained AtlasLM model as follows:

```python
import torch
from atlaslm import load_model

# Load AtlasLM model
# path: path to your model checkpoint
model = load_model("atlaslm-600m", path="atlaslm_600m.pt", device='cuda')
model.eval()

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

### Quick Start: AtlasFold

AtlasFold uses AtlasLM as a feature extractor to predict protein structures.

```python
from atlasfold.model.model import AtlasFold

# Initialize AtlasFold with default configuration
cfg = AtlasFold.Config(lm_name="atlaslm-600m")
model = AtlasFold(cfg)
model.eval().cuda()

# Example sequence
aa = torch.randint(1, 21, (1, 64)).cuda() # (B, L)

# Predict structure (returns dict with distograms, etc.)
# Note: Full PDB inference implementation is in progress
with torch.no_grad():
    results = model(aa, num_recycles=3)
```

## Available Models

We provide the following models under the **MIT License**:

| Model | Params | Layers | Width | Heads |
| --- | --- | --- | --- | --- |
| `atlaslm-600m` | 575M | 36 | 1152 | 18 |
| `atlaslm-3b` | 3.1B | 48 | 2304 | 36 |

## License

This source code is licensed under the MIT license found in the `LICENSE` file in the root directory of this source tree.
