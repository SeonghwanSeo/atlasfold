import argparse
import logging
from pathlib import Path

MODEL_DEFAULTS = {
    "monomer": {
        "num_recycles": 4,
        "mlm_prob": 0.15,
        "num_samples": 5,
        "num_steps": None,  # auto-determined based on sequence length
    },
    "multimer": {
        "num_recycles": 10,
        "mlm_prob": 0.20,
        "num_samples": 5,
        "num_steps": 100,
    },
    "monomer-ipa": {"num_recycles": 4, "mlm_prob": 0.15},
    "multimer-ipa": {"num_recycles": 10, "mlm_prob": 0.20},
}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run AtlasFold or AtlasFold-Multimer inference."
    )
    required = parser.add_argument_group("required arguments")
    inference = parser.add_argument_group("inference options")
    runtime = parser.add_argument_group("runtime and batching options")
    output = parser.add_argument_group("output options")

    required.add_argument(
        "--model",
        choices=["monomer", "multimer", "monomer-ipa", "multimer-ipa"],
        required=True,
        help="Model pipeline to run.",
    )
    required.add_argument(
        "-i",
        "--input-fasta",
        type=Path,
        required=True,
        help=(
            "Input FASTA. For multimer, each record is one complex with ':' "
            "separating chains."
        ),
    )
    required.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where predictions will be written.",
    )

    inference.add_argument(
        "--num-recycles",
        type=int,
        default=None,
        help="Number of recycling iterations. Defaults depend on --model.",
    )
    inference.add_argument(
        "--mlm-prob",
        type=float,
        default=None,
        help="LM masking probability used during recycling.",
    )
    inference.add_argument(
        "--stochastic",
        action="store_true",
        help="Increase sampling diversity across seeds.",
    )
    inference.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of diffusion samples to generate (diffusion models only).",
    )
    inference.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=[1],
        help="Random seed(s) for inference. Example: --seed 1 2 3.",
    )
    inference.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Number of diffusion sampling steps (diffusion models only).",
    )
    inference.add_argument(
        "--recycle-early-stop-tolerance",
        type=float,
        default=None,
        help="Optional convergence threshold for IPA models.",
    )

    runtime.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional local model weights.",
    )
    runtime.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory for AtlasFold and AtlasLM weights.",
    )
    runtime.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Torch device. Defaults to cuda when available, otherwise cpu.",
    )
    runtime.add_argument(
        "--no-kernel",
        action="store_true",
        help="Disable cuequivariance kernels.",
    )
    runtime.add_argument(
        "--max-tokens-per-batch",
        type=int,
        default=1024,
        help="Maximum bucketed residue tokens per model call.",
    )
    runtime.add_argument(
        "--length-buckets",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit residue buckets.",
    )

    output.add_argument(
        "--format",
        choices=["cif", "pdb"],
        default="cif",
        help="Structure file format for sample and ranked outputs.",
    )
    output.add_argument(
        "--save-confidence-arrays",
        action="store_true",
        help="Save raw confidence arrays for each sample as NPZ files.",
    )
    output.add_argument(
        "--save-distogram",
        action="store_true",
        help="Save raw distogram logits and boundaries for each seed as NPZ files.",
    )
    output.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute targets even when outputs already exist.",
    )
    return parser


def log_run_settings(args: argparse.Namespace, logger: logging.Logger) -> None:
    seed = " ".join(str(item) for item in args.seed)
    length_buckets = (
        " ".join(str(item) for item in args.length_buckets)
        if args.length_buckets is not None
        else None
    )

    settings = [
        f"model={args.model}",
        f"model_path={args.model_path}",
        f"cache_dir={args.cache_dir}",
        f"input_fasta={args.input_fasta}",
        f"out_dir={args.out_dir}",
        f"num_recycles={args.num_recycles}",
        f"mlm_prob={args.mlm_prob}",
    ]
    if args.model == "monomer":
        settings.append(f"stochastic={args.stochastic}")
    settings.extend(
        [
            f"num_samples={args.num_samples}",
            f"seed={seed}",
            f"num_steps={args.num_steps}",
            f"device={args.device}",
            f"no_kernel={args.no_kernel}",
            f"max_tokens_per_batch={args.max_tokens_per_batch}",
            f"length_buckets={length_buckets}",
            f"format={args.format}",
            f"save_confidence_arrays={args.save_confidence_arrays}",
            f"save_distogram={args.save_distogram}",
            f"overwrite={args.overwrite}",
        ]
    )
    logger.info("Run settings: %s", ", ".join(settings))


def build_monomer_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        model=args.model,
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        input_fasta=args.input_fasta,
        out_dir=args.out_dir,
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
        stochastic=args.stochastic,
        num_samples=(
            args.num_samples
            if args.num_samples is not None
            else MODEL_DEFAULTS["monomer"]["num_samples"]
        ),
        seed=args.seed,
        num_steps=(
            args.num_steps
            if args.num_steps is not None
            else MODEL_DEFAULTS["monomer"]["num_steps"]
        ),
        device=args.device,
        no_kernel=args.no_kernel,
        max_tokens_per_batch=args.max_tokens_per_batch,
        length_buckets=args.length_buckets,
        format=args.format,
        save_confidence_arrays=args.save_confidence_arrays,
        save_distogram=args.save_distogram,
        overwrite=args.overwrite,
    )


def build_multimer_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.stochastic:
        raise ValueError("--stochastic is only supported when --model monomer.")

    return argparse.Namespace(
        model=args.model,
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        input_fasta=args.input_fasta,
        out_dir=args.out_dir,
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
        num_samples=(
            args.num_samples
            if args.num_samples is not None
            else MODEL_DEFAULTS["multimer"]["num_samples"]
        ),
        seed=args.seed,
        num_steps=(
            args.num_steps
            if args.num_steps is not None
            else MODEL_DEFAULTS["multimer"]["num_steps"]
        ),
        device=args.device,
        no_kernel=args.no_kernel,
        max_tokens_per_batch=args.max_tokens_per_batch,
        length_buckets=args.length_buckets,
        format=args.format,
        save_confidence_arrays=args.save_confidence_arrays,
        save_distogram=args.save_distogram,
        overwrite=args.overwrite,
    )


def build_ipa_args(args: argparse.Namespace, model: str) -> argparse.Namespace:
    if args.model_path is None:
        raise ValueError(f"--model-path is required for --model {model}.")
    if args.stochastic:
        raise ValueError("--stochastic is not supported by IPA regression models.")
    if args.num_samples is not None or args.num_steps is not None:
        raise ValueError("--num-samples and --num-steps apply only to diffusion models.")
    defaults = MODEL_DEFAULTS[model]
    return argparse.Namespace(
        model=model,
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        input_fasta=args.input_fasta,
        out_dir=args.out_dir,
        num_recycles=(
            args.num_recycles
            if args.num_recycles is not None
            else defaults["num_recycles"]
        ),
        recycle_early_stop_tolerance=(
            args.recycle_early_stop_tolerance
            if args.recycle_early_stop_tolerance is not None
            else 0.0
        ),
        mlm_prob=args.mlm_prob if args.mlm_prob is not None else defaults["mlm_prob"],
        seed=args.seed,
        device=args.device,
        no_kernel=args.no_kernel,
        max_tokens_per_batch=args.max_tokens_per_batch,
        length_buckets=args.length_buckets,
        format=args.format,
        save_confidence_arrays=args.save_confidence_arrays,
        save_distogram=args.save_distogram,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    args = create_parser().parse_args()

    import torch

    torch.set_float32_matmul_precision("highest")

    if args.model == "monomer":
        from scripts import run_inference

        resolved_args = build_monomer_args(args)
        run_inference.setup_logging()
        log_run_settings(resolved_args, run_inference.logger)
        run_inference.run(resolved_args)
        exit(0)

    if args.model == "multimer":
        from scripts import run_inference_multimer

        resolved_args = build_multimer_args(args)
        run_inference_multimer.setup_logging()
        log_run_settings(resolved_args, run_inference_multimer.logger)
        run_inference_multimer.run(resolved_args)
        exit(0)

    if args.model == "monomer-ipa":
        from scripts import run_inference_ipa

        run_inference_ipa.run(build_ipa_args(args, "monomer-ipa"))
        exit(0)

    if args.model == "multimer-ipa":
        from scripts import run_inference_multimer_ipa

        run_inference_multimer_ipa.run(build_ipa_args(args, "multimer-ipa"))
        exit(0)

    raise ValueError(f"Unsupported model: {args.model}")
