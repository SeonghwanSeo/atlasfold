import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from timeit import default_timer as timer

import torch

from atlasfold.data.fasta import read_fasta
from atlasfold.model import AtlasFold, SamplingConfig
from atlasfold.runner import FoldingRunner

logger = logging.getLogger("atlasfold.monomer")


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
        description="Run AtlasFold batched inference on a FASTA file."
    )
    parser.add_argument(
        "-i",
        "--input-fasta",
        type=Path,
        required=True,
        help="Path to the input FASTA file.",
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
        default=4,
        help="Number of recycling iterations.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic LM features during all recycling iterations.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of diffusion samples to generate per input sequence.",
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
        default=25,
        help="Number of diffusion sampling steps.",
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
        "--format",
        choices=["cif", "pdb"],
        default="cif",
        help="Structure file format for sample and ranked outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute targets even when ranked outputs already exist.",
    )
    return parser


def normalize_target_name(header: str) -> str:
    name = header.split()[0]
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    if not safe_name:
        raise ValueError(f"Invalid FASTA header: {header!r}")
    return safe_name


def load_sequences(input_fasta: Path) -> list[tuple[str, str]]:
    if not input_fasta.exists():
        raise FileNotFoundError(f"Input FASTA file does not exist: {input_fasta}")

    sequences = [
        (normalize_target_name(header), sequence)
        for header, sequence in read_fasta(input_fasta)
    ]
    if len(sequences) == 0:
        raise ValueError(f"No sequences found in FASTA file: {input_fasta}")

    seen: set[str] = set()
    duplicates: list[str] = []
    for name, _ in sequences:
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError(
            "FASTA target names must be unique after normalization. "
            f"Duplicates: {sorted(set(duplicates))}"
        )

    return sorted(sequences, key=lambda item: (len(item[1]), item[0]))


def target_is_complete(
    target_dir: Path,
    format: str,
    seeds: list[int],
    num_samples: int,
) -> bool:
    target_name = target_dir.name
    return target_rank_is_complete(target_dir, target_name, format, seeds, num_samples)


def sample_output_paths(
    target_dir: Path,
    target_name: str,
    format: str,
    seed: int,
    sample_idx: int,
) -> tuple[Path, Path]:
    sample_name = f"{target_name}_seed-{seed}_sample-{sample_idx}"
    return (
        target_dir / f"{sample_name}.{format}",
        target_dir / f"{sample_name}_confidence.json",
    )


def seed_samples_are_complete(
    target_dir: Path,
    target_name: str,
    format: str,
    seed: int,
    num_samples: int,
) -> bool:
    return all(
        path.exists()
        for sample_idx in range(num_samples)
        for path in sample_output_paths(target_dir, target_name, format, seed, sample_idx)
    )


def target_samples_are_complete(
    target_dir: Path,
    target_name: str,
    format: str,
    seeds: list[int],
    num_samples: int,
) -> bool:
    return all(
        seed_samples_are_complete(target_dir, target_name, format, seed, num_samples)
        for seed in seeds
    )


def target_rank_is_complete(
    target_dir: Path,
    target_name: str,
    format: str,
    seeds: list[int],
    num_samples: int,
) -> bool:
    if not target_samples_are_complete(
        target_dir,
        target_name,
        format,
        seeds,
        num_samples,
    ):
        return False
    required_outputs = [
        target_dir / f"{target_name}_ranked_model.{format}",
        target_dir / f"{target_name}_ranked_confidence.json",
        target_dir / f"{target_name}_summary.csv",
        target_dir / "done.txt",
    ]
    return all(path.exists() for path in required_outputs)


def filter_completed_targets(
    sequences: list[tuple[str, str]],
    out_dir: Path,
    format: str,
    seeds: list[int],
    num_samples: int,
    overwrite: bool,
) -> list[tuple[str, str]]:
    if overwrite:
        return sequences

    filtered = [
        (name, sequence)
        for name, sequence in sequences
        if not target_is_complete(out_dir / name, format, seeds, num_samples)
    ]
    num_skipped = len(sequences) - len(filtered)
    if num_skipped > 0:
        logger.info("Skipping %d completed targets.", num_skipped)
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
    state_dict = torch.load(args.checkpoint, map_location="cpu")

    model = AtlasFold.from_pretrained(state_dict=state_dict, device=device, dtype=dtype)
    model.set_forward_flags(use_cuequiv_kernels=not args.no_kernel)
    return model


def build_sampling_config(args: argparse.Namespace):
    return SamplingConfig(
        num_steps=args.num_steps,
        sigma_max=160.0,
        chunk_size=args.sampling_chunk_size,
    )


def write_json(path: Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def structure_to_text(sample, format: str) -> str:
    if format == "cif":
        return sample.to_mmcif()
    if format == "pdb":
        return sample.to_pdb()
    raise ValueError(f"Unsupported output format: {format}")


def write_sample_outputs_for_target(
    target_dir: Path,
    target_name: str,
    samples,
    *,
    format: str,
) -> list[dict]:
    target_dir.mkdir(parents=True, exist_ok=True)
    sample_records = []
    for (seed, sample_idx), sample in sorted(samples.items()):
        sample_text = structure_to_text(sample, format)
        model_path, confidence_path = sample_output_paths(
            target_dir,
            target_name,
            format,
            seed,
            sample_idx,
        )
        model_path.write_text(sample_text)
        confidence = sample.confidence_scores
        write_json(confidence_path, confidence)
        sample_records.append(
            {
                "seed": seed,
                "sample": sample_idx,
                "sample_name": f"seed-{seed}_sample-{sample_idx}",
                "score": sample.ranking_score,
                "confidence": confidence,
                "model_path": model_path,
            }
        )
    return sample_records


def load_sample_records_for_target(
    target_dir: Path,
    target_name: str,
    *,
    format: str,
    seeds: list[int],
    num_samples: int,
) -> list[dict]:
    sample_records = []
    for seed in seeds:
        for sample_idx in range(num_samples):
            model_path, confidence_path = sample_output_paths(
                target_dir,
                target_name,
                format,
                seed,
                sample_idx,
            )
            confidence = load_json(confidence_path)
            sample_records.append(
                {
                    "seed": seed,
                    "sample": sample_idx,
                    "sample_name": f"seed-{seed}_sample-{sample_idx}",
                    "score": float(confidence["avg_plddt"]),
                    "confidence": confidence,
                    "model_path": model_path,
                }
            )
    return sample_records


def write_ranked_outputs_for_target(
    target_dir: Path,
    target_name: str,
    sample_records: list[dict],
    *,
    format: str,
) -> None:
    if len(sample_records) == 0:
        raise ValueError(f"No samples to rank for target {target_name!r}.")

    best_record = max(
        sample_records,
        key=lambda record: (record["score"], -record["seed"], -record["sample"]),
    )
    shutil.copyfile(
        best_record["model_path"],
        target_dir / f"{target_name}_ranked_model.{format}",
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
    write_json(
        target_dir / f"{target_name}_ranked_confidence.json",
        ranked_confidence,
    )
    with open(target_dir / f"{target_name}_summary.csv", "w") as f:
        f.write("sample,seed,sample_index,avg_plddt,ptm\n")
        for record in sample_records:
            confidence = record["confidence"]
            f.write(
                f"{record['sample_name']},{record['seed']},{record['sample']},"
                f"{confidence['avg_plddt']:.3f},{confidence['ptm']:.3f}\n"
            )


def run(args: argparse.Namespace) -> None:
    setup_logging()

    sequences = load_sequences(args.input_fasta)
    logger.info(
        "Loaded %d sequences from %s. Length range: %d-%d.",
        len(sequences),
        args.input_fasta,
        len(sequences[0][1]),
        len(sequences[-1][1]),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sequences = filter_completed_targets(
        sequences,
        args.out_dir,
        args.format,
        args.seed,
        args.num_samples,
        args.overwrite,
    )
    if len(sequences) == 0:
        logger.info("All targets are complete. Nothing to do.")
        return

    pending_by_seeds: dict[tuple[int, ...], list[tuple[str, str]]] = {}
    for name, sequence in sequences:
        missing_seeds = [
            seed
            for seed in args.seed
            if args.overwrite
            or not seed_samples_are_complete(
                args.out_dir / name,
                name,
                args.format,
                seed,
                args.num_samples,
            )
        ]
        if missing_seeds:
            pending_by_seeds.setdefault(tuple(missing_seeds), []).append((name, sequence))
    pending_count = sum(
        len(seed_sequences) for seed_sequences in pending_by_seeds.values()
    )

    runner = None
    sampling_config = None
    if pending_count > 0:
        model = load_model(args)
        runner = FoldingRunner(model)
        sampling_config = build_sampling_config(args)

    logger.info(
        "Starting batched inference: targets=%d, seeds=%s, num_samples=%d, "
        "num_recycles=%d, stochastic=%s, "
        "max_tokens_per_batch=%d, format=%s",
        len(sequences),
        args.seed,
        args.num_samples,
        args.num_recycles,
        args.stochastic,
        args.max_tokens_per_batch,
        args.format,
    )

    start = timer()
    num_written = 0
    try:
        for seed_group, seed_sequences in pending_by_seeds.items():
            if len(seed_sequences) == 0:
                continue
            if runner is None or sampling_config is None:
                raise RuntimeError("Internal error: runner was not initialized.")

            logger.info(
                "Running seeds %s for %d target(s).",
                list(seed_group),
                len(seed_sequences),
            )
            for output in runner.iter_fold(
                seed_sequences,
                num_samples=args.num_samples,
                seeds=list(seed_group),
                num_recycles=args.num_recycles,
                stochastic=args.stochastic,
                sampling_config=sampling_config,
                length_buckets=args.length_buckets,
                max_tokens_per_batch=args.max_tokens_per_batch,
            ):
                target_dir = args.out_dir / output.name
                sample_records = write_sample_outputs_for_target(
                    target_dir,
                    output.name,
                    output.outputs,
                    format=args.format,
                )
                best_record = max(
                    sample_records,
                    key=lambda record: (
                        record["score"],
                        -record["seed"],
                        -record["sample"],
                    ),
                )
                num_written += 1
                logger.info(
                    "Wrote %s seeds=%s: best=%s, plddt=%.3f (%d/%d)",
                    output.name,
                    list(seed_group),
                    best_record["sample"],
                    best_record["score"],
                    num_written,
                    pending_count,
                )

        for name, _ in sequences:
            target_dir = args.out_dir / name
            if not target_samples_are_complete(
                target_dir,
                name,
                args.format,
                args.seed,
                args.num_samples,
            ):
                logger.info("Skipping rank for %s: not all samples are complete.", name)
                continue

            (target_dir / "done.txt").touch()
            if not args.overwrite and target_rank_is_complete(
                target_dir,
                name,
                args.format,
                args.seed,
                args.num_samples,
            ):
                logger.info("Ranked outputs for %s already exist.", name)
                continue

            sample_records = load_sample_records_for_target(
                target_dir,
                name,
                format=args.format,
                seeds=args.seed,
                num_samples=args.num_samples,
            )
            write_ranked_outputs_for_target(
                target_dir,
                name,
                sample_records,
                format=args.format,
            )
            ranked_record = max(
                sample_records,
                key=lambda record: (record["score"], -record["seed"], -record["sample"]),
            )
            logger.info(
                "Ranked %s: seed=%d, sample=%d, plddt=%.3f",
                name,
                ranked_record["seed"],
                ranked_record["sample"],
                ranked_record["score"],
            )
    except RuntimeError as err:
        if "out of memory" in str(err).lower():
            logger.error(
                "CUDA out of memory during inference. Try lowering "
                "--max-tokens-per-batch or --sampling-chunk-size."
            )
        raise

    elapsed = timer() - start
    logger.info(
        "Finished %d targets in %.1fs (%.1fs per target).",
        len(sequences),
        elapsed,
        elapsed / len(sequences),
    )


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("highest")
    main()
