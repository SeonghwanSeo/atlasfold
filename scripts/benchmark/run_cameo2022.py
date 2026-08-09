import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from atlasfold.data.fasta import read_fasta
from atlasfold.model import AtlasFold, AtlasFoldConfig, SamplingConfig
from atlasfold.runner import FoldingRunner

NUM_RECYCLES = 4
NUM_STEPS = 80
SIGMA_MAX = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AtlasFold on CAMEO 2022.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to an AtlasFold training checkpoint.",
    )
    parser.add_argument(
        "--input-fasta",
        type=Path,
        default=Path("./assets/test/cameo2022.fasta"),
        help="Path to the CAMEO 2022 FASTA file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./outputs/cameo2022"),
        help="Root directory for benchmark outputs.",
    )
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
    state_dict = torch.load(args.checkpoint, map_location="cuda", weights_only=True)
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
    experiment_name = (
        f"{args.checkpoint.stem}_recycles{NUM_RECYCLES}_"
        f"steps{NUM_STEPS}_sigma{SIGMA_MAX}"
    )
    out_dir = args.output_root / experiment_name / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load sequences
    sequences = read_fasta(args.input_fasta)
    sequences.sort(key=lambda x: (len(x[1]), x[0]))
    targets = [(header.split()[0], seq) for header, seq in sequences]

    # Run folding
    runner = FoldingRunner(model)
    sampling_config = SamplingConfig(num_steps=NUM_STEPS, sigma_max=SIGMA_MAX)

    for seed in args.seed:
        pending_targets = [
            (name, seq)
            for name, seq in targets
            if not has_all_samples(out_dir, name, seed, args.num_samples)
        ]
        if len(pending_targets) == 0:
            print(f"Skipping seed {seed}: all samples already exist.")
            continue

        print()
        print(
            f"Folding {len(pending_targets)} CAMEO 2022 targets for seed {seed} "
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
                out_path = out_dir / f"{name}_seed-{seed}_sample-{i}.cif"
                with open(out_path, "w") as f:
                    f.write(out.to_mmcif())
                confidence_path = out_dir / f"{name}_seed-{seed}_sample-{i}.json"
                write_confidence_json(confidence_path, out)


if __name__ == "__main__":
    main()
