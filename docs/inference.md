# AtlasFold inference

AtlasFold provides monomer and multimer runners for both diffusion and IPA regression models. All runners support single-target folding, streaming and batched inference, automatic length bucketing, confidence-based ranking, structure serialization, and optional distogram outputs.

## Command-line inference

Use the CLI for FASTA-driven jobs that should write complete output directories, summaries, confidence files, and completion markers automatically:

```bash
atlasfold monomer --input-fasta monomers.fasta --out-dir predictions/monomers
atlasfold multimer --input-fasta multimers.fasta --out-dir predictions/multimers
atlasfold monomer-ipa --input-fasta monomers.fasta --out-dir predictions/monomer-ipa --model-path weights/monomer-ipa.pt
atlasfold multimer-ipa --input-fasta multimers.fasta --out-dir predictions/multimer-ipa --model-path weights/multimer-ipa.pt
```

The published diffusion models download their weights automatically. The IPA variants currently require local weights supplied with `--model-path` and do not accept diffusion-only options such as `--num-samples`, `--num-steps`, or `--stochastic`.

IPA inference supports optional convergence stopping with `--recycle-early-stop-tolerance` and per-recycle console output with `--print-recycle-metrics`. Enabling either option makes the CLI warn and disable batching so convergence and metric state remain target-specific.

See the [README command-line guide](../README.md#command-line-inference) for the common options and output layout, and use `atlasfold <model> --help` for the complete current option list. The repository-level `python run_atlasfold.py <model>` entry point provides the same subcommands.

## Loading a model and runner

`load_model()` downloads published weights when necessary and returns a model on the requested device. `get_runner()` accepts either a loaded pretrained model or a compatible fine-tuned model instance and returns the matching runner.

| Model | Accepted names | Runner |
| --- | --- | --- |
| AtlasFold monomer | `atlasfold`, `atlasfold-260703` | `FoldingRunner` |
| AtlasFold-M multimer | `atlasfold-m`, `atlasfold-m-260725` | `MultimerFoldingRunner` |
| AtlasFold IPA monomer | `atlasfold-ipa` | `IPAFoldingRunner` |
| AtlasFold IPA multimer | `atlasfold-multimer-ipa` | `MultimerIPAFoldingRunner` |

```python
from atlasfold.pretrained import get_runner, load_model

model = load_model("atlasfold", device="cuda")
runner = get_runner(model)
```

If `device` is omitted, AtlasFold uses CUDA when available and otherwise uses CPU. Use `cache_dir` to choose the download cache, `model_path` to load local AtlasFold weights, and `lm_path` to load local AtlasLM weights. A local `model_path` must match the architecture selected by `model_name`.

IPA weights are not currently published, so `load_model("atlasfold-ipa", model_path=...)` and `load_model("atlasfold-multimer-ipa", model_path=...)` require `model_path`.

Fine-tuned models can be passed directly without going through `load_model()` again:

```python
finetuned_model.eval()
runner = get_runner(finetuned_model)
```

## Single-target prediction

### Diffusion monomer

Call `fold()` with a target name and one amino-acid sequence:

```python
from atlasfold.pretrained import get_runner, load_model

runner = get_runner(load_model("atlasfold", device="cuda"))
result = runner.fold(
    "protein_a",
    "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPD",
    num_samples=5,
    seeds=[1, 2],
)

print(result.best_key)  # (seed, sample_index)
print(result.best.avg_plddt, result.best.ptm)
```

### Diffusion multimer

Call `fold()` with one sequence per chain. Chain order is preserved in the output, and repeated sequences represent repeated copies in a homomer.

```python
from atlasfold.pretrained import get_runner, load_model

runner = get_runner(load_model("atlasfold-m", device="cuda"))
result = runner.fold(
    "complex_a",
    ["MKTAYIAKQRQISFVKSHFS", "GGHVDHGKSTTTGHLIYK"],
    num_samples=5,
    seeds=[1, 2],
)

print(result.best.iptm, result.best.ptm)
print(result.best.confidence_scores["complex"])
```

The Python multimer API requires pre-split chain sequences and does not split strings on `:`. Colon-delimited chains are supported only by the FASTA input used by the multimer CLI.

### IPA models

IPA runners return one regression prediction per seed and do not accept diffusion sampling options:

```python
from atlasfold.pretrained import get_runner, load_model

model = load_model(
    "atlasfold-ipa",
    model_path="weights/monomer-ipa.pt",
    device="cuda",
)
runner = get_runner(model)
result = runner.fold(
    "protein_a",
    "MKTAYIAKQRQISFVKSHFS",
    seeds=[1, 2],
    num_recycles=4,
    recycle_early_stop_tolerance=0.0,
)

print(result.best_seed)
print(result.best.avg_plddt, result.recycle_counts)
```

Use `atlasfold-multimer-ipa` with a list of pre-split chain sequences for IPA complex prediction.

## Inputs

Monomer iterators accept runner-specific input objects or equivalent `(name, sequence)` tuples. The diffusion type is `FoldingInput`, while the IPA type is `IPAFoldingInput`:

```python
from atlasfold.runner import FoldingInput

monomer_inputs = [
    FoldingInput("protein_a", "MKTAYIAKQRQISFVKSHFS"),
    ("protein_b", "GILGYTEHQVVSSDFQKAA"),
]
```

Multimer iterators accept runner-specific input objects or equivalent `(name, chains)` tuples. The diffusion type is `MultimerInput`, while the IPA type is `MultimerIPAFoldingInput`:

```python
from atlasfold.runner_multimer import MultimerInput

multimer_inputs = [
    MultimerInput("complex_a", ["MKTAYIAK", "GGHVDHGK"]),
    ("complex_b", ["GILGYTEH", "QQLLQYFQ"]),
]
```

The runners remove sequence whitespace, convert residues to uppercase, and replace nonstandard residues with `X` while emitting a warning. Empty monomer sequences and empty multimer chains raise `ValueError`.

## Batched inference

All four runners expose three execution methods:

| Method | Use |
| --- | --- |
| `fold()` | Return one target result. |
| `fold_iter()` | Stream one target result at a time without exposing model batch boundaries. |
| `fold_iter_batch()` | Yield one list of target results for each model batch. |

`fold_iter()` is the usual choice for processing many targets because outputs can be consumed and saved incrementally:

```python
from atlasfold.pretrained import get_runner, load_model
from atlasfold.runner import FoldingInput

runner = get_runner(load_model("atlasfold", device="cuda"))
inputs = [
    FoldingInput("protein_a", "MKTAYIAKQRQISFVKSHFS"),
    FoldingInput("protein_b", "GILGYTEHQVVSSDFQKAA"),
    FoldingInput("protein_c", "MQDRVKRPMNAFIVWSRDQRRKMALEN"),
]

for result in runner.fold_iter(inputs, max_tokens_per_batch=1024):
    result_path = f"{result.name}.cif"
    with open(result_path, "w") as handle:
        handle.write(result.best.to_mmcif())
```

Use `fold_iter_batch()` when downstream work needs the actual model batches:

```python
for batch_results in runner.fold_iter_batch(inputs, max_tokens_per_batch=1024):
    print([result.name for result in batch_results])
```

### Length bucketing and token budget

Targets are assigned to the smallest length bucket that can contain them and only targets in the same bucket are batched together. With a positive token budget, `max_tokens_per_batch` controls the batch size as `max(1, max_tokens_per_batch // bucket_length)`; it is a batching budget rather than a maximum supported sequence length. IPA runners additionally accept `max_tokens_per_batch=0` to process one target at a time, while diffusion runners require a positive value.

The default buckets cover common lengths from 32 through 2048 residues, and longer inputs use buckets rounded up to a multiple of 128. Custom `length_buckets` are sorted and deduplicated, must contain positive values, and must include a bucket large enough for every input.

Bucketing can change output order. Match outputs by `result.name` instead of zipping an iterator with the original input list.

If CUDA runs out of memory during multi-target inference, reduce `max_tokens_per_batch`. A target larger than the token budget still runs by itself, so for a single large diffusion target also consider reducing `num_samples` or the diffusion `chunk_size`.

IPA recycle early stopping is synchronized across targets in the same model batch: every target must meet the tolerance before the batch stops. Use `max_tokens_per_batch=0` when targets need independent convergence. Leaving `recycle_callback=None` avoids collecting live recycle metrics and enables the plain batched path.

## Inference options

### Diffusion runners

| Argument | Monomer default | Multimer default | Description |
| --- | --- | --- | --- |
| `num_samples` | `5` | `5` | Number of diffusion samples generated for each seed. |
| `seeds` | `[1]` | `[1]` | One integer seed or a sequence of seeds. |
| `num_recycles` | `4` | `10` | Number of recycling iterations. |
| `mlm_prob` | `0.15` | `0.20` | Probability of masking LM input residues during inference. |
| `stochastic` | `False` | Not available | Re-sample monomer LM masking during recycling for additional diversity. |
| `sampling_config` | Length-dependent | 200 diffusion steps | Optional `SamplingConfig` override. |
| `length_buckets` | Automatic | Automatic | Optional explicit residue-length buckets. |
| `max_tokens_per_batch` | `1024` | `1024` | Positive bucketed token budget accepted by iterator methods. |
| `return_distogram` | `False` | `False` | Return raw distogram logits and boundaries. |

### IPA runners

| Argument | Monomer default | Multimer default | Description |
| --- | --- | --- | --- |
| `seeds` | `[1]` | `[1]` | One integer seed or a sequence of seeds; each seed produces one prediction. |
| `num_recycles` | `4` | `10` | Maximum number of recycling iterations. |
| `mlm_prob` | `0.15` | `0.20` | Probability of masking LM input residues during inference. |
| `recycle_early_stop_tolerance` | `0.0` | `0.0` | Convergence threshold; zero disables early stopping. |
| `recycle_callback` | `None` | `None` | Optional callback receiving live metrics after each recycle. |
| `length_buckets` | Automatic | Automatic | Optional explicit residue-length buckets. |
| `max_tokens_per_batch` | `1024` | `1024` | Bucketed iterator token budget; zero disables batching. |
| `return_distogram` | `False` | `False` | Return raw distogram logits and boundaries. |

These are Python runner defaults. The CLI exposes its current model-specific defaults through `--help`.

### Seeds and diffusion samples

Each diffusion target produces `len(seeds) * num_samples` structures. Outputs are keyed by `(seed, sample_index)`, so the two sampling axes remain explicit rather than being flattened into one model number. Each IPA target produces `len(seeds)` structures keyed directly by seed.

Inference runs inside a forked Torch RNG context. Reusing the same model, input, seed, and options reproduces the same random choices without advancing the caller's Torch RNG state.

For diffusion monomers, the default `stochastic=False` samples one LM mask and shares its features across recycling iterations. Setting `stochastic=True` samples LM masking during recycling and requires `mlm_prob > 0`.

Setting `mlm_prob=0` disables LM masking for monomer inference. Multimer inference requires `mlm_prob > 0`.

### Diffusion configuration

When `sampling_config` is omitted, monomer inference selects the number of diffusion steps from the bucketed length: 20 steps through 512 residues, 30 steps through 1024 residues, and 100 steps for longer inputs. Multimer inference uses 200 steps by default.

Pass `SamplingConfig` to override the diffusion schedule. `chunk_size` controls how many diffusion samples pass through the score model at once; lowering it can reduce peak memory when `num_samples` is large.

```python
from atlasfold.model import SamplingConfig

sampling_config = SamplingConfig(num_steps=50, chunk_size=5)
result = runner.fold(
    "protein_a",
    "MKTAYIAKQRQISFVKSHFS",
    num_samples=8,
    sampling_config=sampling_config,
)
```

The remaining `SamplingConfig` fields expose the diffusion noise schedule and solver parameters. Change them only when reproducing an evaluated sampling configuration.

## Results and ranking

All target-level result objects expose `name`, `length`, `outputs`, `ranking`, `best`, `distogram_logits`, and `distogram_boundaries`. Diffusion outputs use `(seed, sample_index)` keys and expose the winning key as `best_key`; IPA outputs use integer seed keys, expose `best_seed`, and record completed recycle counts in `recycle_counts`.

Monomer predictions contain coordinates, per-residue pLDDT, PAE, and pTM. They are ranked by mean pLDDT, exposed on a 0–100 scale as `avg_plddt`.

Multimer predictions additionally contain ipTM, per-chain pTM, and pairwise interface ipTM. Diffusion multimer predictions also contain PDE. Both model families rank complexes by `0.8 * ipTM + 0.2 * pTM`, and `confidence_scores` groups values under `complex`, `chains`, and `interfaces`.

```python
for key in result.ranking:
    prediction = result.outputs[key]
    print(key, prediction.ranking_score)
```

## Saving structures and confidence data

Every prediction supports `to_mmcif()` and `to_pdb()`. These methods return text and do not write to disk themselves.

```python
best = result.best

with open(f"{result.name}.cif", "w") as handle:
    handle.write(best.to_mmcif())

confidence = best.confidence_scores
```

The Python API keeps confidence arrays as NumPy values on each prediction. All predictions expose `plddt` with shape `(L,)` on a 0–1 scale and `pae` with shape `(L, L)` in ångströms; diffusion multimer predictions additionally expose `pde` with shape `(L, L)` in ångströms. `avg_plddt` and structure B-factors use a 0–100 scale, while pTM, ipTM, and the multimer ranking score use a 0–1 scale. Scalar scores are available as properties or through `confidence_scores`.

## Distogram output

Distogram output is disabled by default because its memory use grows quadratically with sequence length. Enable it only when raw distance-bin logits are needed:

```python
result = runner.fold(
    "protein_a",
    "MKTAYIAKQRQISFVKSHFS",
    seeds=[1, 2],
    return_distogram=True,
)

seed_1_logits = result.distogram_logits[1]
boundaries = result.distogram_boundaries
```

Distogram logits are stored once per seed because they come from the folding trunk. Diffusion coordinates and confidence arrays are still stored for every sample.
