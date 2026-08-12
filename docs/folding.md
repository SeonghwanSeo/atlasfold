# Inference Python API

AtlasFold provides separate runners for monomer and multimer inference. A
runner wraps an already loaded model and handles input normalization, length
bucketing, batching, sampling, ranking, and conversion to structure objects.

## Command-line inference

An installed AtlasFold package provides model-specific CLI subcommands:

```bash
atlasfold monomer --input-fasta monomers.fasta --out-dir predictions/monomers
atlasfold multimer --input-fasta multimers.fasta --out-dir predictions/multimers
```

Use `atlasfold monomer --help` or `atlasfold multimer --help` for model-specific options and defaults. The `--stochastic` option is available only for monomer inference.

Positive values are required for sample counts, diffusion steps, token budgets, and explicit length buckets. Recycling counts may be zero, monomer MLM probability must be between zero and one, and multimer MLM probability must be greater than zero and at most one.

The repository-level `run_atlasfold.py` entry point provides the same subcommands:

```bash
python run_atlasfold.py monomer --input-fasta monomers.fasta --out-dir predictions/monomers
python run_atlasfold.py multimer --input-fasta multimers.fasta --out-dir predictions/multimers
```

## Inputs

Monomer methods accept a `FoldingInput` or an equivalent `(name, sequence)`
tuple:

```python
from atlasfold.runner import FoldingInput

inputs = [
    FoldingInput("protein_a", "MKTAYIAKQRQISFVKSHFS"),
    ("protein_b", "GILGYTEHQVVSSDFQKAA"),
]
```

Multimer methods accept a `MultimerInput` or an equivalent `(name, chains)`
tuple. `chains` must be a sequence containing one sequence string per chain:

```python
from atlasfold.runner_multimer import MultimerInput

inputs = [
    MultimerInput("complex_a", ["MKTAYIAK", "GGHVDHGK"]),
    ("complex_b", ["GILGYTEH", "QQLLQYFQ"]),
]
```

The multimer runner does not split strings on `:`. Colon-delimited chains are a FASTA convention handled by `atlasfold multimer` and `python run_atlasfold.py multimer`; callers of the Python API must pass pre-split chains.

For both runners, sequence whitespace is removed and residues are uppercased.
Nonstandard residues are replaced with `X`, with a warning containing the
target name and, for multimers, the one-based chain number. Empty monomer
sequences and empty multimer chains raise `ValueError`.

## Execution methods

Both runners expose three execution levels:

| Method | Result |
| --- | --- |
| `fold()` | Returns the result for one target. |
| `fold_iter()` | Yields one target result at a time. |
| `fold_iter_batch()` | Yields a list of target results for each model batch. |

`fold_iter()` is the usual choice for custom pipelines because it preserves
streaming without exposing model batch boundaries. Use `fold_iter_batch()`
when downstream work should be scheduled once per model batch.

Length bucketing can change result order. Match results by `result.name`; do not
zip the iterator with the original inputs when input order matters.

Common keyword arguments are:

| Argument | Description |
| --- | --- |
| `num_samples` | Number of diffusion samples generated for each seed. |
| `seeds` | One integer seed or a sequence of seeds. |
| `num_recycles` | Number of recycling iterations. |
| `mlm_prob` | LM masking probability used during recycling. |
| `sampling_config` | Optional `SamplingConfig` overriding diffusion sampling settings. |
| `length_buckets` | Optional explicit residue-length buckets. |
| `max_tokens_per_batch` | Token budget for `fold_iter()` and `fold_iter_batch()`. |
| `return_distogram` | Whether to return raw distogram logits and boundaries. |

The monomer runner additionally accepts `stochastic`, which enables stochastic
LM features during all recycling iterations.

## Monomer inference

Use `fold()` for one sequence:

```python
from atlasfold.runner import FoldingRunner

runner = FoldingRunner(model)
result = runner.fold(
    "protein_a",
    "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPD",
    num_samples=5,
    seeds=[1, 2],
)

best = result.best
print(result.best_key)  # (seed, sample_index)
print(best.avg_plddt, best.ptm)

with open("protein_a.cif", "w") as f:
    f.write(best.to_mmcif())
```

Use `fold_iter()` for multiple sequences:

```python
from atlasfold.runner import FoldingInput, FoldingRunner

runner = FoldingRunner(model)
inputs = [
    FoldingInput("protein_a", "MKTAYIAKQRQISFVKSHFS"),
    ("protein_b", "GILGYTEHQVVSSDFQKAA"),
]

for result in runner.fold_iter(inputs, max_tokens_per_batch=1024):
    print(result.name, result.best.avg_plddt)
```

## Multimer inference

Pass one sequence per chain to `fold()`:

```python
from atlasfold.runner_multimer import MultimerFoldingRunner

runner = MultimerFoldingRunner(multimer_model)
result = runner.fold(
    "complex_a",
    ["MKTAYIAKQRQISFVKSHFS", "GGHVDHGKSTTTGHLIYK"],
    num_samples=5,
    seeds=[1, 2],
)

best = result.best
print(best.iptm, best.ptm)
print(best.confidence_scores["complex"])
```

Multiple complexes can use `MultimerInput` objects or plain tuples:

```python
from atlasfold.runner_multimer import MultimerFoldingRunner, MultimerInput

runner = MultimerFoldingRunner(multimer_model)
inputs = [
    MultimerInput("complex_a", ["MKTAYIAK", "GGHVDHGK"]),
    ("complex_b", ["GILGYTEH", "QQLLQYFQ"]),
]

for result in runner.fold_iter(inputs, max_tokens_per_batch=1024):
    print(result.name, result.best.iptm)
```

## Results and ranking

`FoldingOutput` and `MultimerFoldingOutput` share the same target-level
interface:

| Attribute | Description |
| --- | --- |
| `name` | Target name. |
| `length` | Number of residues in the target. |
| `outputs` | All samples, keyed by `(seed, sample_index)`. |
| `ranking` | Sample keys ordered from best to worst. |
| `best_key` | Key of the highest-ranked sample. |
| `best` | Highest-ranked structure object. |
| `distogram_logits` | Raw distogram logits keyed by seed when requested. |
| `distogram_boundaries` | Shared distogram bin boundaries when requested. |

Monomer samples are `ProteinOutput` objects. They contain coordinates, pLDDT,
PAE, and pTM, and are ranked by mean pLDDT. The `confidence_scores` property
returns scalar `avg_plddt` and `ptm` values.

Multimer samples are `ProteinMultimerOutput` objects. They additionally contain
PDE, ipTM, per-chain pTM, and pairwise interface ipTM values. They are ranked by
`0.8 * ipTM + 0.2 * pTM`. Their `confidence_scores` property groups scores under
`complex`, `chains`, and `interfaces`.

Both sample types provide `to_mmcif()` and `to_pdb()` methods.

## Distogram output

Distogram output is disabled by default because its size scales quadratically
with target length. Enable it only when the raw logits are required:

```python
from atlasfold.runner import FoldingRunner

runner = FoldingRunner(model)
result = runner.fold(
    "protein_a",
    "MKTAYIAKQRQISFVKSHFS",
    seeds=[1, 2],
    return_distogram=True,
)

seed_1_logits = result.distogram_logits[1]
boundaries = result.distogram_boundaries
```

Distogram logits are stored once per seed rather than once per diffusion
sample. PAE, pLDDT, and other confidence values remain available on every
sample through `result.outputs` and `result.best` regardless of
`return_distogram`.
