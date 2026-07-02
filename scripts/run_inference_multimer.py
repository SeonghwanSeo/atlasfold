import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from timeit import default_timer as timer

import torch

from atlasfold.data.fasta import read_fasta
from atlasfold.model import AtlasFold_Multimer, SamplingConfig
from atlasfold.runner_multimer import MultimerFoldingRunner

logger = logging.getLogger("atlasfold.multimer")


def setup_logging() -> None:
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run AtlasFold-Multimer inference on a FASTA file. Each FASTA "
            "record is treated as one complex, with ':' separating chains."
        )
    )
    parser.add_argument(
        "-i",
        "--input-fasta",
        type=Path,
        required=True,
        help=(
            "Input FASTA file. Each FASTA record is treated as one complex, "
            "with ':' separating chains."
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
        help="Path to an AtlasFold-Multimer checkpoint.",
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
        default=10,
        help="Number of recycling iterations.",
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
        default=200,
        help="Number of diffusion sampling steps.",
    )
    parser.add_argument(
        "--max-tokens-per-batch",
        type=int,
        default=1024,
        help=(
            "Maximum bucketed residue tokens per model call. Lower this if "
            "CUDA runs out of memory."
        ),
    )
    parser.add_argument(
        "--length-buckets",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional explicit residue buckets. Defaults to the runner buckets: "
            "32, 64, 128, 192, 256, 384, 512, 640, 768, ..."
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=["cif", "pdb"],
        default="cif",
        help="Structure file format for sample and ranked outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute targets even when outputs already exist.",
    )
    return parser


def safe_target_name(name: str) -> str:
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    if not safe_name:
        raise ValueError(f"Invalid target name: {name!r}")
    return safe_name


def load_inputs(args: argparse.Namespace) -> list[tuple[str, str]]:
    if not args.input_fasta.exists():
        raise FileNotFoundError(f"Input FASTA file does not exist: {args.input_fasta}")

    inputs = [
        (safe_target_name(header.split()[0]), sequence)
        for header, sequence in read_fasta(args.input_fasta)
    ]
    if len(inputs) == 0:
        raise ValueError(f"No sequences found in FASTA file: {args.input_fasta}")

    seen: set[str] = set()
    duplicates: list[str] = []
    for name, _ in inputs:
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError(
            "FASTA target names must be unique after normalization. "
            f"Duplicates: {sorted(set(duplicates))}"
        )

    return inputs


def target_is_complete(
    target_dir: Path,
    output_format: str,
    seeds: list[int],
    num_samples: int,
) -> bool:
    target_name = target_dir.name
    return target_rank_is_complete(
        target_dir,
        target_name,
        output_format,
        seeds,
        num_samples,
    )


def sample_output_paths(
    target_dir: Path,
    target_name: str,
    output_format: str,
    seed: int,
    sample_idx: int,
) -> tuple[Path, Path]:
    sample_name = f"{target_name}_seed-{seed}_sample-{sample_idx}"
    return (
        target_dir / f"{sample_name}.{output_format}",
        target_dir / f"{sample_name}_confidence.json",
    )


def seed_samples_are_complete(
    target_dir: Path,
    target_name: str,
    output_format: str,
    seed: int,
    num_samples: int,
) -> bool:
    return all(
        path.exists()
        for sample_idx in range(num_samples)
        for path in sample_output_paths(
            target_dir,
            target_name,
            output_format,
            seed,
            sample_idx,
        )
    )


def target_samples_are_complete(
    target_dir: Path,
    target_name: str,
    output_format: str,
    seeds: list[int],
    num_samples: int,
) -> bool:
    return all(
        seed_samples_are_complete(
            target_dir,
            target_name,
            output_format,
            seed,
            num_samples,
        )
        for seed in seeds
    )


def target_rank_is_complete(
    target_dir: Path,
    target_name: str,
    output_format: str,
    seeds: list[int],
    num_samples: int,
) -> bool:
    if not target_samples_are_complete(
        target_dir,
        target_name,
        output_format,
        seeds,
        num_samples,
    ):
        return False
    required_outputs = [
        target_dir / f"{target_name}_ranked_model.{output_format}",
        target_dir / f"{target_name}_ranked_confidence.json",
        target_dir / f"{target_name}_summary.csv",
        target_dir / "done.txt",
    ]
    return all(path.exists() for path in required_outputs)


def estimate_num_residues(sequence: str) -> int:
    return sum(len("".join(part.split())) for part in sequence.split(":") if part.strip())


def filter_completed_inputs(
    inputs: list[tuple[str, str]],
    out_dir: Path,
    output_format: str,
    seeds: list[int],
    num_samples: int,
    overwrite: bool,
) -> list[tuple[str, str]]:
    if overwrite:
        return inputs

    filtered = [
        (name, sequence)
        for name, sequence in inputs
        if not target_is_complete(out_dir / name, output_format, seeds, num_samples)
    ]
    num_skipped = len(inputs) - len(filtered)
    if num_skipped > 0:
        logger.info("Skipping %d completed target(s).", num_skipped)
    return filtered


def load_model(args: argparse.Namespace):

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {args.checkpoint}")

    device = torch.device(
        args.device
        if args.device is not None
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    precision = "bf16" if device.type == "cuda" else "fp32"
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32

    logger.info(
        "Loading checkpoint: path=%s, device=%s, precision=%s",
        args.checkpoint,
        device,
        precision,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = {
            k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items()
        }
    else:
        state_dict = {k.removeprefix("model."): v for k, v in checkpoint.items()}

    model = AtlasFold_Multimer.from_pretrained(
        state_dict=state_dict,
        device=device,
        dtype=dtype,
    )
    model.set_forward_flags(use_cuequiv_kernels=not args.no_kernel)
    return model


def build_sampling_config(args: argparse.Namespace):
    return SamplingConfig(
        num_steps=args.num_steps,
        chunk_size=5,
    )


def write_json(path: Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def structure_to_text(sample, output_format: str) -> str:
    if output_format == "cif":
        return sample.to_mmcif()
    if output_format == "pdb":
        return sample.to_pdb()
    raise ValueError(f"Unsupported output format: {output_format}")


def get_ranking_score(confidence: dict) -> float:
    if "complex" in confidence:
        confidence = confidence["complex"]
    if "ranking_score" in confidence:
        return float(confidence["ranking_score"])
    return 0.8 * float(confidence.get("iptm", 0.0)) + 0.2 * float(confidence["ptm"])


def get_complex_confidence(confidence: dict) -> dict:
    if "complex" in confidence:
        return confidence["complex"]
    return confidence


def write_sample_outputs(
    out_dir: Path,
    target_name: str,
    samples,
    *,
    output_format: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    for (seed, sample_idx), sample in sorted(samples.items()):
        sample_text = structure_to_text(sample, output_format)
        model_path, confidence_path = sample_output_paths(
            out_dir,
            target_name,
            output_format,
            seed,
            sample_idx,
        )
        model_path.write_text(sample_text)
        write_json(confidence_path, sample.confidence_scores)


def load_sample_records(
    out_dir: Path,
    target_name: str,
    *,
    output_format: str,
    seeds: list[int],
    num_samples: int,
) -> list[dict]:
    sample_records = []
    for seed in seeds:
        for sample_idx in range(num_samples):
            model_path, confidence_path = sample_output_paths(
                out_dir,
                target_name,
                output_format,
                seed,
                sample_idx,
            )
            confidence = load_json(confidence_path)
            sample_records.append(
                {
                    "seed": seed,
                    "sample": sample_idx,
                    "sample_name": f"seed-{seed}_sample-{sample_idx}",
                    "score": get_ranking_score(confidence),
                    "confidence": confidence,
                    "model_path": model_path,
                }
            )
    return sample_records


def write_ranked_outputs(
    out_dir: Path,
    target_name: str,
    sample_records: list[dict],
    *,
    output_format: str,
) -> None:
    if len(sample_records) == 0:
        raise ValueError(f"No samples to rank for target {target_name!r}.")

    best_record = max(
        sample_records,
        key=lambda record: (record["score"], -record["seed"], -record["sample"]),
    )
    shutil.copyfile(
        best_record["model_path"],
        out_dir / f"{target_name}_ranked_model.{output_format}",
    )

    ranked_confidence = {
        **best_record["confidence"],
        "seed": best_record["seed"],
        "sample": best_record["sample"],
        "ranked_sample": best_record["sample_name"],
        "samples": {
            record["sample_name"]: record["confidence"] for record in sample_records
        },
    }
    write_json(out_dir / f"{target_name}_ranked_confidence.json", ranked_confidence)

    with open(out_dir / f"{target_name}_summary.csv", "w") as f:
        f.write("sample,seed,sample_index,avg_plddt,ptm,iptm,ranking_score\n")
        for record in sample_records:
            confidence = get_complex_confidence(record["confidence"])
            f.write(
                f"{record['sample_name']},{record['seed']},{record['sample']},"
                f"{confidence['avg_plddt']:.3f},{confidence['ptm']:.3f},"
                f"{confidence.get('iptm', 0.0):.3f},"
                f"{confidence.get('ranking_score', record['score']):.3f}\n"
            )


def run(args: argparse.Namespace) -> None:
    setup_logging()
    inputs = load_inputs(args)
    num_residues = [estimate_num_residues(sequence) for _, sequence in inputs]
    logger.info(
        "Loaded %d multimer target(s). Residue range: %d-%d.",
        len(inputs),
        min(num_residues),
        max(num_residues),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    inputs = filter_completed_inputs(
        inputs,
        args.out_dir,
        args.output_format,
        args.seed,
        args.num_samples,
        args.overwrite,
    )
    if len(inputs) == 0:
        logger.info("All targets are complete. Nothing to do.")
        return

    pending_by_seeds: dict[tuple[int, ...], list[tuple[str, str]]] = {}
    for name, sequence in inputs:
        missing_seeds = [
            seed
            for seed in args.seed
            if args.overwrite
            or not seed_samples_are_complete(
                args.out_dir / name,
                name,
                args.output_format,
                seed,
                args.num_samples,
            )
        ]
        if missing_seeds:
            pending_by_seeds.setdefault(tuple(missing_seeds), []).append((name, sequence))
    pending_count = sum(len(seed_inputs) for seed_inputs in pending_by_seeds.values())

    runner = None
    sampling_config = None
    if pending_count > 0:
        model = load_model(args)
        runner = MultimerFoldingRunner(model)
        sampling_config = build_sampling_config(args)

    logger.info(
        "Starting multimer inference: seeds=%s, num_samples=%d, "
        "num_recycles=%d, output_format=%s",
        args.seed,
        args.num_samples,
        args.num_recycles,
        args.output_format,
    )

    start = timer()
    for seed_group, seed_inputs in pending_by_seeds.items():
        if len(seed_inputs) == 0:
            continue
        if runner is None or sampling_config is None:
            raise RuntimeError("Internal error: runner was not initialized.")

        logger.info(
            "Running seeds %s for %d target(s).",
            list(seed_group),
            len(seed_inputs),
        )
        for output in runner.iter_fold(
            seed_inputs,
            num_samples=args.num_samples,
            seeds=list(seed_group),
            num_recycles=args.num_recycles,
            sampling_config=sampling_config,
            length_buckets=args.length_buckets,
            max_tokens_per_batch=args.max_tokens_per_batch,
        ):
            write_sample_outputs(
                args.out_dir / output.name,
                output.name,
                output.outputs,
                output_format=args.output_format,
            )

    for target_name, _ in inputs:
        target_dir = args.out_dir / target_name
        if not target_samples_are_complete(
            target_dir,
            target_name,
            args.output_format,
            args.seed,
            args.num_samples,
        ):
            logger.info(
                "Skipping rank for %s: not all samples are complete.",
                target_name,
            )
            continue

        (target_dir / "done.txt").touch()
        if not args.overwrite and target_rank_is_complete(
            target_dir,
            target_name,
            args.output_format,
            args.seed,
            args.num_samples,
        ):
            logger.info("Ranked outputs for %s already exist.", target_name)
            continue

        sample_records = load_sample_records(
            target_dir,
            target_name,
            output_format=args.output_format,
            seeds=args.seed,
            num_samples=args.num_samples,
        )
        write_ranked_outputs(
            target_dir,
            target_name,
            sample_records,
            output_format=args.output_format,
        )
        ranked_record = max(
            sample_records,
            key=lambda record: (record["score"], -record["seed"], -record["sample"]),
        )
        logger.info(
            "Ranked %s: seed=%d, sample=%d, ptm=%.3f",
            target_name,
            ranked_record["seed"],
            ranked_record["sample"],
            ranked_record["score"],
        )
    elapsed = timer() - start
    logger.info(
        "Finished %d target(s) in %.1fs.",
        len(inputs),
        elapsed,
    )


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("highest")
    main()
