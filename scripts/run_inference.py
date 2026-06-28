import argparse
import json
import logging
import sys
from pathlib import Path
from timeit import default_timer as timer


logger = logging.getLogger("atlasfold.inference")


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
        "--lm-name",
        type=str,
        default="atlaslm-3b",
        help="AtlasLM model name used to construct the AtlasFold config.",
    )
    parser.add_argument(
        "--lm-path",
        type=str,
        default=None,
        help="Optional local path to AtlasLM weights.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--precision",
        choices=["auto", "bf16", "fp32"],
        default="auto",
        help="Inference precision. auto uses bf16 on CUDA and fp32 on CPU.",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Do not use EMA weights from the checkpoint when available.",
    )
    parser.add_argument(
        "--no-kernel",
        action="store_true",
        help="Disable cuequivariance kernels.",
    )
    parser.add_argument(
        "--preset",
        choices=["base", "high", "stochastic"],
        default="base",
        help="FoldingRunner inference preset.",
    )
    parser.add_argument(
        "--num-recycles",
        type=int,
        default=None,
        help="Override the number of recycles from the preset.",
    )
    parser.add_argument(
        "--mlm-prob",
        type=float,
        default=None,
        help="Override the MLM masking probability from the preset.",
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
        default=1,
        help="Random seed for inference.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of diffusion sampling steps.",
    )
    parser.add_argument(
        "--sigma-max",
        type=float,
        default=160.0,
        help="Maximum sigma for diffusion sampling.",
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
        "--rank-by",
        choices=["plddt", "ptm"],
        default="plddt",
        help="Metric used to rank samples for each input sequence.",
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
    from atlasfold.data.fasta import read_fasta

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


def target_is_complete(target_dir: Path, output_format: str) -> bool:
    return (
        (target_dir / "ranking_debug.json").exists()
        and (target_dir / f"ranked_0.{output_format}").exists()
    )


def filter_completed_targets(
    sequences: list[tuple[str, str]],
    out_dir: Path,
    output_format: str,
    overwrite: bool,
) -> list[tuple[str, str]]:
    if overwrite:
        return sequences

    filtered = [
        (name, sequence)
        for name, sequence in sequences
        if not target_is_complete(out_dir / name, output_format)
    ]
    num_skipped = len(sequences) - len(filtered)
    if num_skipped > 0:
        logger.info("Skipping %d completed targets.", num_skipped)
    return filtered


def load_model(args: argparse.Namespace):
    import torch

    from atlasfold.model import AtlasFold, AtlasFoldConfig

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {args.checkpoint}")

    device = torch.device(
        args.device if args.device is not None else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    precision = args.precision
    if precision == "auto":
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
        if not args.no_ema and "ema" in checkpoint and "params" in checkpoint["ema"]:
            state_dict.update(checkpoint["ema"]["params"])
            logger.info("Using EMA weights from checkpoint.")
    else:
        state_dict = {k.removeprefix("model."): v for k, v in checkpoint.items()}

    config = AtlasFoldConfig(lm_name=args.lm_name, lm_path=args.lm_path)
    model = AtlasFold.from_pretrained(
        state_dict=state_dict,
        config=config,
        device=device,
        dtype=dtype,
    )
    model.set_forward_flags(use_cuequiv_kernels=not args.no_kernel)
    return model


def build_sampling_config(args: argparse.Namespace):
    from atlasfold.model import SamplingConfig

    return SamplingConfig(
        num_steps=args.num_steps,
        sigma_max=args.sigma_max,
        chunk_size=args.sampling_chunk_size,
    )


def get_sample_scores(samples, rank_by: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for sample_idx, sample in enumerate(samples):
        sample_name = f"sample_{sample_idx}"
        if rank_by == "plddt":
            scores[sample_name] = float(sample.plddt.mean() * 100)
        elif rank_by == "ptm":
            scores[sample_name] = float(sample.ptm)
        else:
            raise ValueError(f"Unsupported rank_by metric: {rank_by}")
    return scores


def write_json(path: Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def structure_to_text(sample, output_format: str) -> str:
    if output_format == "cif":
        return sample.to_mmcif()
    if output_format == "pdb":
        return sample.to_pdb()
    raise ValueError(f"Unsupported output format: {output_format}")


def write_outputs_for_target(
    target_dir: Path,
    samples,
    *,
    rank_by: str,
    output_format: str,
    seed: int,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    sample_texts: dict[str, str] = {}
    scores = get_sample_scores(samples, rank_by)

    for sample_idx, sample in enumerate(samples):
        sample_name = f"sample_{sample_idx}"
        sample_text = structure_to_text(sample, output_format)
        sample_texts[sample_name] = sample_text
        (target_dir / f"{sample_name}.{output_format}").write_text(sample_text)

        avg_plddt = float(sample.plddt.mean() * 100)
        write_json(
            target_dir / f"confidence_{sample_name}.json",
            {
                "name": sample.name,
                "sample": sample_idx,
                "seed": seed,
                "length": len(sample.sequence),
                "avg_plddt": avg_plddt,
                "ptm": float(sample.ptm),
                "ranking_metric": rank_by,
                "ranking_score": scores[sample_name],
            },
        )

    ranked_order = sorted(
        scores,
        key=lambda sample_name: (-scores[sample_name], int(sample_name.split("_")[1])),
    )
    for rank_idx, sample_name in enumerate(ranked_order):
        (target_dir / f"ranked_{rank_idx}.{output_format}").write_text(
            sample_texts[sample_name]
        )

    metric_label = "plddts" if rank_by == "plddt" else "ptms"
    write_json(
        target_dir / "ranking_debug.json",
        {
            metric_label: scores,
            "order": ranked_order,
        },
    )


def run(args: argparse.Namespace) -> None:
    setup_logging()

    import torch

    from atlasfold.runner import FoldingRunner

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

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
        args.output_format,
        args.overwrite,
    )
    if len(sequences) == 0:
        logger.info("All targets are complete. Nothing to do.")
        return

    model = load_model(args)
    runner = FoldingRunner(model)
    sampling_config = build_sampling_config(args)

    logger.info(
        "Starting batched inference: targets=%d, num_samples=%d, preset=%s, "
        "rank_by=%s, max_tokens_per_batch=%d, output_format=%s",
        len(sequences),
        args.num_samples,
        args.preset,
        args.rank_by,
        args.max_tokens_per_batch,
        args.output_format,
    )

    start = timer()
    num_written = 0
    try:
        for batch_output in runner.iter_fold_batches(
            sequences,
            num_samples=args.num_samples,
            preset=args.preset,
            seed=args.seed,
            num_recycles=args.num_recycles,
            mlm_prob=args.mlm_prob,
            sampling_config=sampling_config,
            length_buckets=args.length_buckets,
            max_tokens_per_batch=args.max_tokens_per_batch,
        ):
            for (_, name, sequence), samples in zip(
                batch_output.inputs, batch_output.outputs, strict=True
            ):
                target_dir = args.out_dir / name
                write_outputs_for_target(
                    target_dir,
                    samples,
                    rank_by=args.rank_by,
                    output_format=args.output_format,
                    seed=args.seed,
                )
                scores = get_sample_scores(samples, args.rank_by)
                best_score = max(scores.values())
                best_sample = max(scores, key=scores.get)
                num_written += 1
                logger.info(
                    "Wrote %s: length=%d, bucket=%d, batch=%d/%d, "
                    "best=%s, %s=%.4f (%d/%d).",
                    name,
                    len(sequence),
                    batch_output.bucket_length,
                    batch_output.batch_index,
                    batch_output.num_batches,
                    best_sample,
                    args.rank_by,
                    best_score,
                    num_written,
                    len(sequences),
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
    main()
