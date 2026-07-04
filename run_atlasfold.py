import argparse
from pathlib import Path

MODEL_DEFAULTS = {
    "monomer": {
        "num_recycles": 4,
        "mlm_prob": 0.15,
        "num_steps": 20,
    },
    "multimer": {
        "num_recycles": 10,
        "mlm_prob": 0.15,
        "num_steps": 200,
    },
}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run AtlasFold or AtlasFold-Multimer inference."
    )
    parser.add_argument(
        "--model",
        choices=["monomer", "multimer"],
        required=True,
        help="Model pipeline to run.",
    )
    parser.add_argument(
        "-i",
        "--input-fasta",
        type=Path,
        required=True,
        help=(
            "Input FASTA. For multimer, each record is one complex with ':' "
            "separating chains."
        ),
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where predictions will be written.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to an AtlasFold checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Torch device. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--no-kernel",
        action="store_true",
        help="Disable cuequivariance kernels.",
    )
    parser.add_argument(
        "--num-recycles",
        type=int,
        default=None,
        help="Number of recycling iterations. Defaults depend on --model.",
    )
    parser.add_argument(
        "--mlm-prob",
        type=float,
        default=None,
        help="LM masking probability used during recycling.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic LM features during monomer recycling.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of diffusion samples to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=[1],
        help="Random seed(s) for inference. Example: --seed 1 2 3.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Number of diffusion sampling steps. Defaults depend on --model.",
    )
    parser.add_argument(
        "--format",
        choices=["cif", "pdb"],
        default="cif",
        help="Structure file format for sample and ranked outputs.",
    )
    parser.add_argument(
        "--max-tokens-per-batch",
        type=int,
        default=1024,
        help="Maximum bucketed residue tokens per model call.",
    )
    parser.add_argument(
        "--length-buckets",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit residue buckets.",
    )
    parser.add_argument(
        "--use-fasta-chain-ids",
        action="store_true",
        help=(
            "Read optional chain_id=... metadata for monomer inference or "
            "chain_ids=... metadata for multimer inference from FASTA headers."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute targets even when outputs already exist.",
    )
    return parser


def print_run_settings(args: argparse.Namespace) -> None:
    seed = " ".join(str(item) for item in args.seed)
    length_buckets = (
        " ".join(str(item) for item in args.length_buckets)
        if args.length_buckets is not None
        else None
    )

    print("AtlasFold run settings")
    print(f"  model={args.model}")
    print(f"  input_fasta={args.input_fasta}")
    print(f"  out_dir={args.out_dir}")
    print(f"  checkpoint={args.checkpoint}")
    print(f"  device={args.device}")
    print(f"  format={args.format}")
    print(f"  use_fasta_chain_ids={args.use_fasta_chain_ids}")
    print(f"  num_recycles={args.num_recycles}")
    print(f"  mlm_prob={args.mlm_prob}")
    print(f"  num_samples={args.num_samples}")
    print(f"  seed={seed}")
    print(f"  num_steps={args.num_steps}")
    print(f"  max_tokens_per_batch={args.max_tokens_per_batch}")
    print(f"  length_buckets={length_buckets}")
    if args.model == "monomer":
        print(f"  stochastic={args.stochastic}")
    print(f"  overwrite={args.overwrite}")
    print()


def build_monomer_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        model=args.model,
        input_fasta=args.input_fasta,
        out_dir=args.out_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        no_kernel=args.no_kernel,
        num_samples=args.num_samples,
        stochastic=args.stochastic,
        seed=args.seed,
        num_recycles=(
            args.num_recycles
            if args.num_recycles is not None
            else MODEL_DEFAULTS["monomer"]["num_recycles"]
        ),
        mlm_prob=(
            args.mlm_prob
            if args.mlm_prob is not None
            else MODEL_DEFAULTS["monomer"]["mlm_prob"]
        ),
        num_steps=(
            args.num_steps
            if args.num_steps is not None
            else MODEL_DEFAULTS["monomer"]["num_steps"]
        ),
        format=args.format,
        max_tokens_per_batch=args.max_tokens_per_batch,
        length_buckets=args.length_buckets,
        use_fasta_chain_ids=args.use_fasta_chain_ids,
        overwrite=args.overwrite,
    )


def build_multimer_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.stochastic:
        raise ValueError("--stochastic is only supported when --model monomer.")

    return argparse.Namespace(
        model=args.model,
        input_fasta=args.input_fasta,
        out_dir=args.out_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        no_kernel=args.no_kernel,
        num_samples=args.num_samples,
        seed=args.seed,
        num_recycles=(
            args.num_recycles
            if args.num_recycles is not None
            else MODEL_DEFAULTS["multimer"]["num_recycles"]
        ),
        mlm_prob=(
            args.mlm_prob
            if args.mlm_prob is not None
            else MODEL_DEFAULTS["multimer"]["mlm_prob"]
        ),
        num_steps=(
            args.num_steps
            if args.num_steps is not None
            else MODEL_DEFAULTS["multimer"]["num_steps"]
        ),
        format=args.format,
        max_tokens_per_batch=args.max_tokens_per_batch,
        length_buckets=args.length_buckets,
        use_fasta_chain_ids=args.use_fasta_chain_ids,
        overwrite=args.overwrite,
    )


def run(args: argparse.Namespace) -> None:
    if args.model == "monomer":
        from scripts import run_inference

        resolved_args = build_monomer_args(args)
        print_run_settings(resolved_args)
        run_inference.run(resolved_args)
        return

    if args.model == "multimer":
        from scripts import run_inference_multimer

        resolved_args = build_multimer_args(args)
        print_run_settings(resolved_args)
        run_inference_multimer.run(resolved_args)
        return

    raise ValueError(f"Unsupported model: {args.model}")


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
