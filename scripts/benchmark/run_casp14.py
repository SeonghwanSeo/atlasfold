import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from atlasfold.data.fasta import read_fasta
from atlasfold.model import AtlasFold, AtlasFoldConfig, SamplingConfig
from atlasfold.runner import FoldingRunner

CKPT = "/mnt/parallel_storage/wykim_lab/icl_shwan/train_results/v4/stage-confidence-1/AtlasFold/qchch827/checkpoints/epoch0116_step00117000_lddt0.8292.ckpt"
CKPT_NAME = "117k"
NUM_RECYCLES = 4
NUM_STEPS = 100
SIGMA_MAX = 160

ROOT_DIR = Path("./outputs/casp14/")
EXP_NAME = f"{CKPT_NAME}_recycles{NUM_RECYCLES}_steps{NUM_STEPS}_sigma{SIGMA_MAX}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AtlasFold on CASP14.")
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Random seed(s) to run. Example: --seed 1 2",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of diffusion samples to generate per target and seed.",
    )
    parser.add_argument(
        "--max-tokens-per-batch",
        type=int,
        default=1024,
        help="Maximum bucketed residue tokens per batched model call.",
    )
    return parser.parse_args()


def write_confidence_json(out_path: Path, out) -> None:
    confidence = {
        "avg_plddt": float(out.plddt.mean() * 100) if out.plddt is not None else None,
        "ptm": out.ptm,
    }
    with open(out_path, "w") as f:
        json.dump(confidence, f, indent=2)
        f.write("\n")


def has_all_samples(out_dir: Path, name: str, seed: int, num_samples: int = 5) -> bool:
    return all(
        (out_dir / f"{name}_seed-{seed}_sample-{i}.cif").exists()
        and (out_dir / f"{name}_seed-{seed}_sample-{i}.json").exists()
        for i in range(num_samples)
    )


@torch.inference_mode()
def main():
    args = parse_args()

    # Load model
    device = torch.device("cuda")
    config = AtlasFoldConfig()
    state_dict = torch.load(CKPT, map_location="cuda")
    model_state_dict = {
        k.removeprefix("model."): v for k, v in state_dict["state_dict"].items()
    }
    ema_state_dict = state_dict["ema"]["params"]
    model_state_dict.update(ema_state_dict)
    del state_dict
    model = AtlasFold.from_pretrained(
        state_dict=model_state_dict,
        config=config,
        device=device,
        dtype=torch.bfloat16,
    )

    # Create output directory
    OUT_DIR = ROOT_DIR / EXP_NAME / "predictions"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load sequences
    sequences = read_fasta("./assets/test/casp14_public.fasta")
    sequences.sort(key=lambda x: (len(x[1]), x[0]))
    targets = [(header.split()[0], seq) for header, seq in sequences]

    # Run folding
    runner = FoldingRunner(model)
    sampling_config = SamplingConfig(num_steps=NUM_STEPS, sigma_max=SIGMA_MAX)

    for seed in args.seed:
        pending_targets = [
            (name, seq)
            for name, seq in targets
            if not has_all_samples(OUT_DIR, name, seed, args.num_samples)
        ]
        if len(pending_targets) == 0:
            print(f"Skipping seed {seed}: all samples already exist.")
            continue

        print()
        print(
            f"Folding {len(pending_targets)} CASP14 targets for seed {seed} "
            f"with max_tokens_per_batch={args.max_tokens_per_batch}..."
        )
        start_time = time.time()
        batched_outputs = runner.fold_batch(
            pending_targets,
            num_samples=args.num_samples,
            seed=seed,
            num_recycles=NUM_RECYCLES,
            sampling_config=sampling_config,
            max_tokens_per_batch=args.max_tokens_per_batch,
        )
        elapsed = time.time() - start_time
        print(
            f"Seed {seed}: {elapsed:.2f} seconds total "
            f"({elapsed / len(pending_targets):.2f} seconds/target)"
        )

        for (name, _), outs in tqdm(
            zip(pending_targets, batched_outputs, strict=True),
            total=len(pending_targets),
            desc=f"Writing seed {seed}",
        ):
            for i, out in enumerate(outs):
                out_path = OUT_DIR / f"{name}_seed-{seed}_sample-{i}.cif"
                with open(out_path, "w") as f:
                    f.write(out.to_mmcif())
                confidence_path = OUT_DIR / f"{name}_seed-{seed}_sample-{i}.json"
                write_confidence_json(confidence_path, out)


if __name__ == "__main__":
    main()
