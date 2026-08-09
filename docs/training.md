# Training AtlasFold

This guide covers AtlasFold monomer and multimer training.

## Requirements

Install AtlasFold from the repository with the folding and training extras:

```bash
pip install -e ".[fold,train,cuequiv]"
```

The model configurations use `atlaslm-3b-base`. Its weights are downloaded on the first run.

## Training data

The AtlasFold training set will be released in the near future.

## Check a configuration with a debug run

Before starting a full training run, you can check the configuration with a short debug run. This can be useful to adjust memory settings and verify that the data loader is working.

```bash
python scripts/train_monomer.py configs/monomer/train_stage1.yaml --debug
python scripts/train_multimer.py configs/multimer/train_stage1.yaml --debug
```

## Train a monomer model

AtlasFold monomer training uses four progressively longer crop stages:

| Stage | `max_length` | `max_seq_length` | Initialization |
| --- | ---: | ---: | --- |
| 1 | 256 | 512 | Random initialization |
| 2 | 384 | 768 | Initialization from stage 1 |
| 3 | 512 | 1024 | Initialization from stage 2 |
| 4 | 640 | 1280 | Initialization from stage 3, EMA for trunk |

Start stage 1 with:

```bash
python scripts/train_monomer.py configs/monomer/train_stage1.yaml \
    --out_dir ./experiments \
    --num_nodes $NUM_NODES \
    --num_gpus $NUM_GPUS
```

Continue fine-tuning (Stage 2-4):

```bash
python scripts/train_monomer.py configs/monomer/train_stage2.yaml \
    --out_dir ./experiments \
    --num_nodes $NUM_NODES \
    --num_gpus $NUM_GPUS \
    --resume_from_checkpoint /checkpoints/monomer-stage1.ckpt \
    --override train.load_opt_state=false
```

## Train a multimer model

AtlasFold multimer training uses three crop stages:

| Stage | `max_length` | `max_seq_length` | `max_contiguous_chains` | Initialization |
| --- | ---: | ---: | ---: | --- |
| 1 | 384 | 768 | 6 | Initialization from monomer model |
| 2 | 640 | 1280 | 6 | Initialization from stage 1 |
| 3 | 768 | 1536 | 8 | Initialization from stage 2, EMA for trunk |

Start stage 1 with:

```bash
python scripts/train_multimer.py configs/multimer/train_stage1.yaml \
    --out_dir ./experiments \
    --init_weight /weights/atlasfold_260703.pth \
    --num_nodes $NUM_NODES \
    --num_gpus $NUM_GPUS
```

Continue fine-tuning (Stage 2, 3):

```bash
python scripts/train_multimer.py configs/multimer/train_stage2.yaml \
    --out_dir ./experiments \
    --num_nodes $NUM_NODES \
    --num_gpus $NUM_GPUS \
    --resume_from_checkpoint /checkpoints/multimer-stage1.ckpt \
    --override train.load_opt_state=false
```

## Memory tuning

### Activation checkpointing

`blocks_per_ckpt` controls activation checkpointing for the trunk, diffusion head, confidence head, and the multimer template module. `null` disables checkpointing for that stack. A positive integer groups that many consecutive blocks into each checkpointed region; `1` checkpoints every block and minimizes peak activation memory at the cost of additional recomputation and checkpoint overhead. Larger values create fewer checkpointed regions and generally trade more memory for speed.

The setting is independent for each stack. For example, a memory-constrained run can use:

```yaml
model:
  trunk:
    blocks_per_ckpt: 1
  diffusion_head:
    blocks_per_ckpt: 1
  confidence_head:
    blocks_per_ckpt: 1
  # Multimer only:
  template_module:
    blocks_per_ckpt: 1
```

Checkpointing is only applied while the stack is training with gradients enabled. It does not affect inference or a frozen trunk.

### Smooth-lDDT loss chunking

The smooth-lDDT loss constructs pairwise atom-distance tensors for each diffusion sample, so it can be a major memory cost at long crop lengths. `train.loss.diffusion_loss.smooth_lddt_loss.chunk_size` controls how many diffusion samples are evaluated in each checkpointed loss chunk. `null` evaluates all samples together without loss checkpointing; a smaller positive integer lowers peak memory by processing fewer samples at once, but increases backward recomputation and runtime. `1` gives the lowest-memory setting.

```yaml
train:
  loss:
    diffusion_loss:
      smooth_lddt_loss:
        chunk_size: 1
```

This option only matters when `train.loss.weights.smooth_lddt` is greater than zero. Chunking operates over diffusion samples, not residues, so it reduces the sample dimension of the intermediate tensors but does not remove the quadratic dependence on crop length.

#### Pairwise-distance kernel

`train.loss.diffusion_loss.smooth_lddt_loss.use_kernel` enables a Triton kernel equivalent to the following PyTorch implementation:

```python
diff = x[..., :, None, :] - y[..., None, :, :]
distance = torch.sqrt(diff.pow(2).sum(dim=-1) + 1e-8)
```
