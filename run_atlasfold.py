import argparse
from pathlib import Path

MODEL_DEFAULTS = {
    "monomer": {
        "num_recycles": 4,
        "mlm_prob": 0.15,
        "num_samples": 5,
        "num_steps": 25,
        "max_tokens_per_batch": 1024,
    },
    "multimer": {
        "num_recycles": 10,
        "mlm_prob": 0.15,
        "num_samples": 5,
        "num_steps": 200,
        "max_tokens_per_batch": 1024,
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
        default=None,
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
        "--sampling-chunk-size",
        type=int,
        default=None,
        help="Optional diffusion sampling chunk size to reduce memory.",
    )
    parser.add_argument(
        "--max-tokens-per-batch",
        type=int,
        default=None,
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
        "--format",
        choices=["cif", "pdb"],
        default="cif",
        help="Structure file format for sample and ranked outputs.",
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


def resolve_default(args: argparse.Namespace, name: str):
    value = getattr(args, name)
    if value is not None:
        return value
    return MODEL_DEFAULTS[args.model][name]


def format_value(value) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def format_setting(
    user_args: argparse.Namespace,
    resolved_args: argparse.Namespace,
    name: str,
    *,
    label: str | None = None,
    source_name: str | None = None,
) -> str:
    label = label or name
    source_name = source_name or name
    source = "default" if getattr(user_args, source_name) is None else "user"
    value = getattr(resolved_args, name)
    return f"{label}={format_value(value)} ({source})"


def print_run_settings(
    user_args: argparse.Namespace,
    resolved_args: argparse.Namespace,
) -> None:
    print("AtlasFold run settings")
    print(f"  model={user_args.model}")
    print(f"  checkpoint={resolved_args.checkpoint}")
    print(f"  out_dir={resolved_args.out_dir}")
    if user_args.model == "monomer":
        print(f"  input_fasta={resolved_args.input_fasta}")
        print(format_setting(user_args, resolved_args, "format", label="  format"))
        print(f"  stochastic={resolved_args.stochastic}")
    else:
        print(f"  input_fasta={resolved_args.input_fasta}")
        print(
            format_setting(
                user_args,
                resolved_args,
                "format",
                label="  format",
                source_name="format",
            )
        )
    print(f"  use_fasta_chain_ids={resolved_args.use_fasta_chain_ids}")
    print(
        format_setting(user_args, resolved_args, "num_recycles", label="  num_recycles")
    )
    print(format_setting(user_args, resolved_args, "mlm_prob", label="  mlm_prob"))
    print(format_setting(user_args, resolved_args, "num_samples", label="  num_samples"))
    print(format_setting(user_args, resolved_args, "seed", label="  seed"))
    print(format_setting(user_args, resolved_args, "num_steps", label="  num_steps"))
    print(
        format_setting(
            user_args,
            resolved_args,
            "max_tokens_per_batch",
            label="  max_tokens_per_batch",
        )
    )
    print(f"  sampling_chunk_size={resolved_args.sampling_chunk_size}")
    print(f"  length_buckets={format_value(resolved_args.length_buckets)}")
    print(f"  overwrite={resolved_args.overwrite}")
    print()


def build_monomer_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        input_fasta=args.input_fasta,
        out_dir=args.out_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        no_kernel=args.no_kernel,
        num_recycles=resolve_default(args, "num_recycles"),
        mlm_prob=resolve_default(args, "mlm_prob"),
        stochastic=args.stochastic,
        num_samples=resolve_default(args, "num_samples"),
        seed=resolve_default(args, "seed"),
        num_steps=resolve_default(args, "num_steps"),
        sampling_chunk_size=args.sampling_chunk_size,
        max_tokens_per_batch=resolve_default(args, "max_tokens_per_batch"),
        length_buckets=args.length_buckets,
        format=resolve_default(args, "format"),
        use_fasta_chain_ids=args.use_fasta_chain_ids,
        overwrite=args.overwrite,
    )


def build_multimer_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.stochastic:
        raise ValueError("--stochastic is only supported when --model monomer.")

    return argparse.Namespace(
        input_fasta=args.input_fasta,
        out_dir=args.out_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        no_kernel=args.no_kernel,
        num_recycles=resolve_default(args, "num_recycles"),
        mlm_prob=resolve_default(args, "mlm_prob"),
        num_samples=resolve_default(args, "num_samples"),
        seed=resolve_default(args, "seed"),
        num_steps=resolve_default(args, "num_steps"),
        sampling_chunk_size=args.sampling_chunk_size,
        max_tokens_per_batch=resolve_default(args, "max_tokens_per_batch"),
        length_buckets=args.length_buckets,
        format=resolve_default(args, "format"),
        use_fasta_chain_ids=args.use_fasta_chain_ids,
        overwrite=args.overwrite,
    )


def run(args: argparse.Namespace) -> None:
    if args.model == "monomer":
        from scripts import run_inference

        resolved_args = build_monomer_args(args)
        print_run_settings(args, resolved_args)
        run_inference.run(resolved_args)
        return

    if args.model == "multimer":
        from scripts import run_inference_multimer

        resolved_args = build_multimer_args(args)
        print_run_settings(args, resolved_args)
        run_inference_multimer.run(resolved_args)
        return

    raise ValueError(f"Unsupported model: {args.model}")


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
